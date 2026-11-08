"""Keep the gold chunk table (vector-search source) in sync with silver document versions.

Why not inside the Lakeflow pipeline: a document update can *shrink* its chunk count, which
needs "delete chunks of older versions" semantics. A streaming table can't express that, and
a materialized view would rewrite the whole table (and re-embed the whole corpus) on refresh.
A foreachBatch MERGE gives precise, idempotent upserts plus deletes, and Change Data Feed
lets the Delta Sync index re-embed only what changed.

Single writer by design (job max_concurrent_runs=1).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, IntegerType, StringType, StructField, StructType

from tax_rag import names as n
from tax_rag.catalog.contract import Contract, load_contract
from tax_rag.names import Names
from tax_rag.transform.chunking import ChunkConfig, build_chunks

log = logging.getLogger(__name__)

CHUNK_STRUCT = ArrayType(
    StructType(
        [
            StructField("chunk_id", StringType(), False),
            StructField("chunk_index", IntegerType(), False),
            StructField("chunk_text", StringType(), False),
            StructField("chunk_hash", StringType(), False),
        ]
    )
)
DOC_COLUMNS = [
    "document_id",
    "document_version",
    "title",
    "document_type",
    "jurisdiction",
    "tax_type",
    "tax_year",
    "effective_date",
    "expiration_date",
    "source_url",
    "language",
    "body_text",
    "is_deleted",
    "updated_at",
]


def latest_per_document(events: DataFrame) -> DataFrame:
    """Highest (version, updated_at) per document within a micro-batch."""
    w = Window.partitionBy("document_id").orderBy(
        F.col("document_version").desc(), F.col("updated_at").desc()
    )
    return (
        events.select(*DOC_COLUMNS)
        .withColumn("_rn", F.row_number().over(w))
        .where("_rn = 1")
        .drop("_rn")
    )


def explode_chunks(docs: DataFrame, config: ChunkConfig, gold_columns: list[str]) -> DataFrame:
    def _chunks(doc_id, title, body, jurisdiction, tax_type, document_type, tax_year):
        return build_chunks(
            doc_id, title, body, jurisdiction, tax_type, document_type, tax_year, config
        )

    chunk_udf = F.udf(_chunks, CHUNK_STRUCT)
    exploded = docs.where(~F.col("is_deleted")).select(
        "*",
        F.explode(
            chunk_udf(
                "document_id",
                "title",
                "body_text",
                "jurisdiction",
                "tax_type",
                "document_type",
                "tax_year",
            )
        ).alias("c"),
    )
    return exploded.select(
        F.col("c.chunk_id").alias("chunk_id"),
        "document_id",
        "document_version",
        F.col("c.chunk_index").alias("chunk_index"),
        F.col("c.chunk_text").alias("chunk_text"),
        F.col("c.chunk_hash").alias("chunk_hash"),
        "title",
        "document_type",
        "jurisdiction",
        "tax_type",
        "tax_year",
        F.date_format("effective_date", "yyyy-MM-dd").alias("effective_date"),
        F.date_format("expiration_date", "yyyy-MM-dd").alias("expiration_date"),
        "source_url",
        "language",
        "updated_at",
        F.current_timestamp().alias("synced_at"),
    ).select(*gold_columns)


def ensure_gold_table(spark: SparkSession, names: Names, contract: Contract) -> None:
    spark.sql(contract.create_table_sql(names, n.GOLD_CHUNKS))


def process_batch(
    spark: SparkSession,
    batch: DataFrame,
    names: Names,
    config: ChunkConfig,
    contract: Contract,
) -> None:
    from delta.tables import DeltaTable

    gold_name = names.table(n.GOLD_CHUNKS)
    gold = DeltaTable.forName(spark, gold_name)

    # Pin the pre-batch state so both MERGEs see the same "current versions".
    pinned = gold.history(1).select("version").first()[0]
    current = spark.sql(
        f"SELECT document_id, max(document_version) AS current_version "
        f"FROM {names.sql(n.GOLD_CHUNKS)} VERSION AS OF {int(pinned)} GROUP BY document_id"
    )
    # Ignore late, older versions (e.g. API window replays): gold only moves forward.
    affected = (
        latest_per_document(batch)
        .join(current, "document_id", "left")
        .where(
            F.col("current_version").isNull()
            | (F.col("document_version") >= F.col("current_version"))
        )
        .drop("current_version")
    )

    # 1) Remove chunks of superseded versions, and all chunks of deleted documents.
    (
        gold.alias("t")
        .merge(
            affected.select("document_id", "document_version", "is_deleted").alias("s"),
            "t.document_id = s.document_id AND (t.document_version < s.document_version "
            "OR (s.is_deleted AND t.document_version <= s.document_version))",
        )
        .whenMatchedDelete()
        .execute()
    )

    # 2) Upsert chunks of the new versions (idempotent on replay of the same batch).
    gold_columns = contract.table(n.GOLD_CHUNKS).column_names
    chunks = explode_chunks(affected, config, gold_columns)
    (
        gold.alias("t")
        .merge(chunks.alias("s"), "t.chunk_id = s.chunk_id")
        .whenMatchedUpdateAll(condition="s.document_version >= t.document_version")
        .whenNotMatchedInsertAll()
        .execute()
    )


def run_chunk_sync(
    spark: SparkSession,
    names: Names,
    *,
    continuous: bool,
    config: ChunkConfig | None = None,
    processing_time: str = "30 seconds",
    on_batch_end: Callable[[int], None] | None = None,
    reset_checkpoint: bool = False,
    checkpoint_root: str | None = None,
) -> None:
    from tax_rag.spark_support import ship_package_by_value

    ship_package_by_value()
    contract = load_contract()
    cfg = config or ChunkConfig()
    ensure_gold_table(spark, names, contract)
    root = checkpoint_root.rstrip("/") if checkpoint_root else f"{names.volume_root}/checkpoints"
    checkpoint = f"{root}/gold_chunk_sync"
    if reset_checkpoint:
        from databricks.sdk.runtime import dbutils

        log.warning("Resetting checkpoint %s (full re-sync; MERGE keeps it idempotent)", checkpoint)
        dbutils.fs.rm(checkpoint, True)

    def _foreach(batch: DataFrame, batch_id: int) -> None:
        process_batch(spark, batch, names, cfg, contract)
        if on_batch_end:
            on_batch_end(batch_id)

    writer = (
        spark.readStream.table(names.table(n.SILVER_EVENTS))
        .writeStream.foreachBatch(_foreach)
        .option("checkpointLocation", checkpoint)
        .queryName("gold_chunk_sync")
    )
    writer = (
        writer.trigger(processingTime=processing_time)
        if continuous
        else writer.trigger(availableNow=True)
    )
    writer.start().awaitTermination()
