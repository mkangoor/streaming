"""Lakeflow Declarative Pipeline: tax documents -> bronze -> silver (Unity Catalog).

Source is selected by pipeline configuration `tax_rag.source_mode`:
  api   - custom PySpark streaming source polling the tax REST API (default for prod)
  files - Auto Loader over JSON-lines dropped in the `landing` UC volume (dev/staging fixtures,
          vendor bulk exports)
  kafka - Kafka / Event Hubs topic, for when the vendor can push change events

All three converge on the same bronze schema, so everything downstream is source-agnostic.
Schemas, column comments and expectations come from the data contract.
"""

import dlt
from pyspark.sql import functions as F

from tax_rag import names
from tax_rag.catalog.contract import load_contract
from tax_rag.ingest import datasource
from tax_rag.spark_support import ship_package_by_value
from tax_rag.transform import silver

CONTRACT = load_contract()


def conf(key: str, default: str | None = None) -> str | None:
    value = spark.conf.get(f"tax_rag.{key}", default)
    return value if value not in ("", None) else default


SOURCE_MODE = conf("source_mode", "api")

# UDF / data-source workers must not depend on how the package was installed.
ship_package_by_value()
spark.dataSource.register(datasource.TaxApiDataSource)


def _api_source():
    scope = conf("secret_scope")
    token = dbutils.secrets.get(scope, conf("api_token_key")) if scope else None
    reader = (
        spark.readStream.format("tax_api")
        .option("base_url", conf("api_base_url"))
        .option("start_from", conf("start_from", "2024-01-01T00:00:00Z"))
        .option("lag_seconds", conf("api_lag_seconds", "120"))
        .option("granularity_seconds", conf("api_granularity_seconds", "60"))
        .option("max_partitions", conf("api_max_partitions", "8"))
        .option("page_size", conf("api_page_size", "500"))
    )
    if token:
        # Option name contains "token" so Spark's default redaction regex masks it in plans/UI.
        reader = reader.option("api_token", token)
    return silver.bronze_from_api(reader.load())


def _files_source():
    stream = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "text")
        .option("cloudFiles.includeExistingFiles", "true")
        .load(conf("landing_path"))
    )
    return silver.bronze_from_text(stream, "value", F.col("_metadata.file_path"), "files")


def _kafka_source():
    reader = (
        spark.readStream.format("kafka")
        .option("kafka.bootstrap.servers", conf("kafka_bootstrap_servers"))
        .option("subscribe", conf("kafka_topic"))
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "true")
    )
    jaas_key = conf("kafka_jaas_secret_key")
    if jaas_key:  # e.g. Event Hubs Kafka endpoint: SASL_SSL + PLAIN with a connection string
        reader = (
            reader.option("kafka.security.protocol", "SASL_SSL")
            .option("kafka.sasl.mechanism", "PLAIN")
            .option("kafka.sasl.jaas.config", dbutils.secrets.get(conf("secret_scope"), jaas_key))
        )
    stream = reader.load().withColumn("payload_text", F.col("value").cast("string"))
    ref = F.concat_ws(":", F.col("topic"), F.col("partition"), F.col("offset"))
    return silver.bronze_from_text(stream, "payload_text", ref, "kafka")


SOURCES = {"api": _api_source, "files": _files_source, "kafka": _kafka_source}
if SOURCE_MODE not in SOURCES:
    raise ValueError(f"tax_rag.source_mode must be one of {sorted(SOURCES)}, got {SOURCE_MODE!r}")


# ---------------------------------------------------------------- bronze
@dlt.table(
    name=names.BRONZE,
    comment=CONTRACT.table(names.BRONZE).description,
    schema=CONTRACT.ddl(names.BRONZE),
    cluster_by=["document_id"],
    table_properties={
        "quality": "bronze",
        # Raw history may be older than the API's retention: never wipe it on full refresh.
        "pipelines.reset.allowed": "false",
    },
)
def bronze_tax_documents():
    return SOURCES[SOURCE_MODE]()


# ---------------------------------------------------------------- silver
DROP_RULES = CONTRACT.rules("drop")
SAFE = {name: f"COALESCE(({c}), FALSE)" for name, c in DROP_RULES.items()}
EVENT_COLUMNS = CONTRACT.table(names.SILVER_EVENTS).column_names


def _parsed():
    return silver.parse_bronze(spark.readStream.table(names.BRONZE))


def _with_expectations(fn):
    """Attach contract expectations (skipping empty rule sets)."""
    for decorator, rules in (
        (dlt.expect_all_or_fail, CONTRACT.rules("fail")),
        (dlt.expect_all, CONTRACT.rules("warn")),
        (dlt.expect_all_or_drop, SAFE),
    ):
        if rules:
            fn = decorator(rules)(fn)
    return fn


@dlt.table(
    name=names.SILVER_EVENTS,
    comment=CONTRACT.table(names.SILVER_EVENTS).description,
    schema=CONTRACT.ddl(names.SILVER_EVENTS),
    cluster_by=["document_id"],
    table_properties={"quality": "silver"},
)
@_with_expectations
def silver_tax_document_events():
    return silver.valid_events(_parsed(), EVENT_COLUMNS)


@dlt.table(
    name=names.QUARANTINE,
    comment=CONTRACT.table(names.QUARANTINE).description,
    schema=CONTRACT.ddl(names.QUARANTINE),
    table_properties={"quality": "quarantine"},
)
def silver_tax_document_quarantine():
    columns = CONTRACT.table(names.QUARANTINE).column_names
    return silver.quarantine(_parsed(), DROP_RULES, columns)


# Current state (SCD1) and full history (SCD2) from the validated change log.
_auto_cdc = getattr(dlt, "create_auto_cdc_flow", None) or dlt.apply_changes
CDC = CONTRACT.cdc
SEQUENCE = F.struct(*CDC["sequence_by"])

dlt.create_streaming_table(
    name=names.SILVER_CURRENT,
    comment=CONTRACT.table(names.SILVER_CURRENT).description,
    schema=CONTRACT.ddl(names.SILVER_CURRENT),
    table_properties={"quality": "silver", "delta.enableChangeDataFeed": "true"},
)
_auto_cdc(
    target=names.SILVER_CURRENT,
    source=names.SILVER_EVENTS,
    keys=CDC["keys"],
    sequence_by=SEQUENCE,
    apply_as_deletes=F.expr(CDC["delete_condition"]),
    except_column_list=CDC["except_columns"],
    stored_as_scd_type=1,
)

dlt.create_streaming_table(
    name=names.SILVER_HISTORY,
    comment=CONTRACT.table(names.SILVER_HISTORY).description,
    table_properties={"quality": "silver"},
)
_auto_cdc(
    target=names.SILVER_HISTORY,
    source=names.SILVER_EVENTS,
    keys=CDC["keys"],
    sequence_by=SEQUENCE,
    apply_as_deletes=F.expr(CDC["delete_condition"]),
    except_column_list=CDC["except_columns"],
    stored_as_scd_type=2,
)
