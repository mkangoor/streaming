"""Spark transformations shared by the Lakeflow pipeline and local tests.

Kept free of `dlt` imports so the exact production logic runs in a plain local SparkSession.
All parsing uses try_* functions: serverless runs with ANSI mode, where a bad date in one
record would otherwise fail the whole micro-batch.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

from tax_rag.transform import cleaning

DOCUMENT_JSON_SCHEMA = (
    "document_id STRING, version BIGINT, title STRING, document_type STRING, "
    "jurisdiction STRING, tax_type STRING, tax_year INT, effective_date STRING, "
    "expiration_date STRING, source_url STRING, language STRING, body STRING, "
    "body_format STRING, updated_at STRING, is_deleted BOOLEAN"
)

to_plain_text_udf = F.udf(cleaning.to_plain_text, StringType())
normalize_jurisdiction_udf = F.udf(cleaning.normalize_jurisdiction, StringType())
normalize_document_type_udf = F.udf(cleaning.normalize_document_type, StringType())
normalize_tax_type_udf = F.udf(cleaning.normalize_tax_type, StringType())


def _try_date(col: Column) -> Column:
    return F.try_to_timestamp(col, F.lit("yyyy-MM-dd")).cast("date")


def bronze_from_text(df: DataFrame, payload_col: str, ref_col: Column, mode: str) -> DataFrame:
    """Shape a stream of raw JSON strings (files / kafka) into the bronze schema."""
    payload = F.col(payload_col)
    return df.select(
        F.get_json_object(payload, "$.document_id").alias("document_id"),
        F.expr(f"try_cast(get_json_object({payload_col}, '$.version') AS BIGINT)").alias(
            "document_version"
        ),
        F.try_to_timestamp(F.get_json_object(payload, "$.updated_at")).alias("updated_at"),
        payload.alias("payload"),
        F.lit(mode).alias("source_mode"),
        ref_col.cast("string").alias("source_ref"),
        F.current_timestamp().alias("fetched_at"),
    )


def bronze_from_api(df: DataFrame) -> DataFrame:
    return df.select(
        "document_id",
        "document_version",
        "updated_at",
        "payload",
        F.lit("api").alias("source_mode"),
        "source_ref",
        "fetched_at",
    )


def parse_bronze(bronze: DataFrame) -> DataFrame:
    """Bronze -> typed, normalised silver event columns (plus raw payload for quarantine)."""
    doc = F.from_json("payload", DOCUMENT_JSON_SCHEMA, {"mode": "PERMISSIVE"})
    parsed = bronze.withColumn("doc", doc).withColumn(
        "body_text", to_plain_text_udf(F.col("doc.body"), F.col("doc.body_format"))
    )
    return parsed.select(
        F.coalesce(F.col("doc.document_id"), F.col("document_id")).alias("document_id"),
        F.coalesce(F.col("doc.version"), F.col("document_version")).alias("document_version"),
        F.trim(F.col("doc.title")).alias("title"),
        normalize_document_type_udf(F.col("doc.document_type")).alias("document_type"),
        normalize_jurisdiction_udf(F.col("doc.jurisdiction")).alias("jurisdiction"),
        normalize_tax_type_udf(F.col("doc.tax_type")).alias("tax_type"),
        F.col("doc.tax_year").alias("tax_year"),
        _try_date(F.col("doc.effective_date")).alias("effective_date"),
        _try_date(F.col("doc.expiration_date")).alias("expiration_date"),
        F.col("doc.source_url").alias("source_url"),
        F.coalesce(F.lower(F.col("doc.language")), F.lit("en")).alias("language"),
        F.col("body_text"),
        F.sha2(F.coalesce(F.col("body_text"), F.lit("")), 256).alias("content_hash"),
        F.coalesce(F.col("doc.is_deleted"), F.lit(False)).alias("is_deleted"),
        F.coalesce(F.try_to_timestamp(F.col("doc.updated_at")), F.col("updated_at")).alias(
            "updated_at"
        ),
        F.col("source_mode"),
        F.col("source_ref"),
        F.col("fetched_at"),
        F.current_timestamp().alias("processed_at"),
        F.col("payload"),
    )


def passes_all(rules: dict[str, str]) -> Column:
    """True only when every rule is definitely true (NULL counts as a failure)."""
    expr = " AND ".join(f"COALESCE(({c}), FALSE)" for c in rules.values()) or "TRUE"
    return F.expr(expr)


def failed_rules(rules: dict[str, str]) -> Column:
    # Cast normalises element nullability to match the contract's ARRAY<STRING>.
    return F.array_compact(
        F.array(*[F.when(~F.expr(f"COALESCE(({c}), FALSE)"), F.lit(n)) for n, c in rules.items()])
    ).cast("array<string>")


def valid_events(parsed: DataFrame, event_columns: list[str]) -> DataFrame:
    return parsed.select(*event_columns)


def quarantine(parsed: DataFrame, drop_rules: dict[str, str], columns: list[str]) -> DataFrame:
    return (
        parsed.where(~passes_all(drop_rules))
        .withColumn("failed_rules", failed_rules(drop_rules))
        .select(*columns)
    )
