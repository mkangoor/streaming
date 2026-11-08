from pathlib import Path

import pytest

from tax_rag import names as n
from tax_rag.catalog.contract import load_contract, sql_string
from tax_rag.names import Names

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def contract():
    return load_contract()


def test_contract_covers_every_table(contract):
    expected = {
        n.BRONZE,
        n.SILVER_EVENTS,
        n.QUARANTINE,
        n.SILVER_CURRENT,
        n.SILVER_HISTORY,
        n.GOLD_CHUNKS,
    }
    assert set(contract.tables) == expected
    assert all(c.description for t in contract.tables.values() for c in t.columns)


def test_current_table_matches_cdc_projection(contract):
    """SCD1 target schema must equal the event schema minus except_columns, or Lakeflow fails."""
    events = contract.table(n.SILVER_EVENTS).column_names
    expected = [c for c in events if c not in contract.cdc["except_columns"]]
    assert contract.table(n.SILVER_CURRENT).column_names == expected
    events_types = {c.name: c.type for c in contract.table(n.SILVER_EVENTS).columns}
    for c in contract.table(n.SILVER_CURRENT).columns:
        assert c.type == events_types[c.name], c.name


def test_cdc_columns_exist(contract):
    events = set(contract.table(n.SILVER_EVENTS).column_names)
    assert set(contract.cdc["keys"]) | set(contract.cdc["sequence_by"]) <= events
    assert set(contract.cdc["except_columns"]) <= events


def test_gold_has_vector_search_columns(contract):
    from tax_rag.rag.clients import RETRIEVAL_COLUMNS
    from tax_rag.vector.index import SYNCED_COLUMNS

    gold = set(contract.table(n.GOLD_CHUNKS).column_names)
    assert set(SYNCED_COLUMNS) <= gold
    assert set(RETRIEVAL_COLUMNS) <= set(SYNCED_COLUMNS)
    assert contract.table(n.GOLD_CHUNKS).table_properties["delta.enableChangeDataFeed"] == "true"


def test_ddl_and_escaping(contract):
    ddl = contract.ddl(n.BRONZE)
    assert ddl.startswith("`document_id` STRING COMMENT '")
    assert sql_string("it's \\ fine") == "'it\\'s \\\\ fine'"


def test_rules_split_by_action(contract):
    drop, warn = contract.rules("drop"), contract.rules("warn")
    assert "valid_document_id" in drop and "has_title" in warn
    assert not set(drop) & set(warn)


def test_create_table_sql(contract):
    sql = contract.create_table_sql(Names("tax_dev", "tax_rag"), n.GOLD_CHUNKS)
    assert sql.startswith(
        "CREATE TABLE IF NOT EXISTS `tax_dev`.`tax_rag`.`gold_tax_document_chunks`"
    )
    assert "CLUSTER BY (`document_id`)" in sql and "'delta.enableChangeDataFeed' = 'true'" in sql
    with pytest.raises(ValueError):
        contract.create_table_sql(Names("c", "s"), n.BRONZE)


def test_governance_sql(contract):
    stmts = contract.governance_sql(Names("tax_prod", "tax_rag"))
    assert any(
        s.startswith("ALTER STREAMING TABLE `tax_prod`.`tax_rag`.`bronze_tax_documents` SET TAGS")
        for s in stmts
    )
    assert any(
        s.startswith("COMMENT ON TABLE `tax_prod`.`tax_rag`.`gold_tax_document_chunks`")
        for s in stmts
    )
    assert any("ALTER COLUMN `chunk_id` SET TAGS" in s for s in stmts)
    assert all(
        "'layer' = '" in s
        for s in stmts
        if s.endswith(")") and "SET TAGS" in s and "COLUMN" not in s
    )


def test_data_dictionary_is_up_to_date(contract):
    doc = ROOT / "docs" / "data_dictionary.md"
    assert doc.read_text("utf-8") == contract.to_markdown(), "run: python scripts/gen_docs.py"


def test_names_validation():
    assert Names("c", "s").index == "c.s.gold_tax_document_chunks_index"
    with pytest.raises(ValueError):
        Names("bad.catalog", "s")
