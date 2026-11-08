"""Runs the production Spark logic locally: custom streaming source -> bronze -> silver -> chunks.

The Lakeflow decorators can't run outside Databricks, but every transformation they call can.
"""

import pytest

from tax_rag import names as n
from tax_rag.catalog.contract import load_contract
from tax_rag.gold.chunk_sync import explode_chunks, latest_per_document
from tax_rag.ingest.datasource import TaxApiDataSource
from tax_rag.transform import silver
from tax_rag.transform.chunking import ChunkConfig

pytestmark = pytest.mark.spark
CONTRACT = load_contract()


def _struct(spark, table):
    return spark.createDataFrame([], CONTRACT.ddl(table)).schema


@pytest.fixture(scope="module")
def bronze_rows(spark, tmp_path_factory):
    """Stream every fixture through the real custom source against the mock API."""
    from tests.mock_tax_api import MockState, MockTaxApi, load_documents

    spark.dataSource.register(TaxApiDataSource)
    out = tmp_path_factory.mktemp("bronze")
    with MockTaxApi(MockState(load_documents(), token="t")) as api:
        stream = (
            spark.readStream.format("tax_api")
            .option("base_url", api.base_url)
            .option("api_token", "t")
            .option("start_from", "2023-01-01T00:00:00Z")
            .option("lag_seconds", "0")
            .option("granularity_seconds", "3600")
            .option("max_partitions", "3")
            .load()
        )
        query = (
            silver.bronze_from_api(stream)
            .writeStream.format("parquet")
            .option("path", str(out / "data"))
            .option("checkpointLocation", str(out / "cp"))
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination(180)
        assert query.exception() is None
    return spark.read.parquet(str(out / "data"))


def test_stream_delivers_each_fixture_once(bronze_rows):
    rows = bronze_rows.collect()
    assert len(rows) == 11
    assert {r.source_mode for r in rows} == {"api"}
    assert all(r.source_ref.startswith("api:") for r in rows)
    assert bronze_rows.columns == CONTRACT.table(n.BRONZE).column_names


def test_bronze_schema_matches_contract(spark, bronze_rows):
    expected = _struct(spark, n.BRONZE)
    assert [(f.name, f.dataType) for f in bronze_rows.schema] == [
        (f.name, f.dataType) for f in expected
    ]


def test_files_source_shapes_same_bronze_schema(spark):
    from pathlib import Path

    from pyspark.sql import functions as F

    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "tax_documents.jsonl"
    text = spark.read.text(str(fixture))
    bronze = silver.bronze_from_text(text, "value", F.input_file_name(), "files")
    assert bronze.columns == CONTRACT.table(n.BRONZE).column_names
    assert bronze.where("document_id IS NOT NULL").count() == 10


@pytest.fixture(scope="module")
def parsed(bronze_rows):
    return silver.parse_bronze(bronze_rows).cache()


def test_silver_events_and_quarantine_partition_the_input(spark, parsed):
    drop = CONTRACT.rules("drop")
    events = silver.valid_events(
        parsed.where(silver.passes_all(drop)), CONTRACT.table(n.SILVER_EVENTS).column_names
    )
    bad = silver.quarantine(parsed, drop, CONTRACT.table(n.QUARANTINE).column_names)
    assert events.count() + bad.count() == parsed.count() == 11
    [q] = bad.collect()
    assert q.failed_rules == ["valid_document_id"] and "Malformed" in q.payload
    expected = _struct(spark, n.SILVER_EVENTS)
    assert [(f.name, f.dataType) for f in events.schema] == [(f.name, f.dataType) for f in expected]
    expected_q = _struct(spark, n.QUARANTINE)
    assert [(f.name, f.dataType) for f in bad.schema] == [(f.name, f.dataType) for f in expected_q]


def test_silver_normalisation(parsed):
    rows = {(r.document_id, r.document_version): r for r in parsed.collect()}
    v2 = rows[("zz-vat-act-s12", 2)]
    assert v2.jurisdiction == "ZZ" and v2.tax_type == "vat_gst" and v2.document_type == "statute"
    assert str(v2.effective_date) == "2025-01-01" and v2.expiration_date is None
    assert v2.body_text.startswith("## Section 12. Rates of tax") and "<p>" not in v2.body_text
    assert "editor note" not in v2.body_text
    assert rows[("zz-vat-act-s12", 1)].jurisdiction == "ZZ"
    assert rows[("zz-n-provincial-sales-tax", 1)].jurisdiction == "ZZ-N"
    assert rows[("zz-ruling-2025-07", 1)].document_type == "ruling"
    assert rows[("zz-circular-2019-04", 4)].is_deleted


def test_gold_chunks_from_latest_live_versions(parsed):
    drop = CONTRACT.rules("drop")
    events = parsed.where(silver.passes_all(drop))
    latest = latest_per_document(events)
    gold_cols = CONTRACT.table(n.GOLD_CHUNKS).column_names
    chunks = explode_chunks(latest, ChunkConfig(max_chars=300, overlap_chars=50), gold_cols)
    rows = chunks.collect()
    assert chunks.columns == gold_cols
    docs = {r.document_id for r in rows}
    assert "zz-circular-2019-04" not in docs  # deleted -> no chunks
    vat = sorted(
        (r for r in rows if r.document_id == "zz-vat-act-s12"), key=lambda r: r.chunk_index
    )
    assert {r.document_version for r in vat} == {2} and len(
        vat
    ) > 1  # newest version only, multi-chunk
    assert vat[0].chunk_id == "zz-vat-act-s12::0000" and vat[0].effective_date == "2025-01-01"
    assert all(len(r.chunk_text) <= 300 + 200 for r in rows)  # body budget + metadata header
