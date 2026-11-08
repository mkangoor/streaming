from datetime import datetime, timedelta, timezone

import pytest

from tax_rag.ingest.api_client import TaxApiClient, TaxApiError
from tax_rag.ingest.datasource import (
    ReaderOptions,
    TaxApiStreamReader,
    in_window,
    quantized_watermark,
    split_window,
    to_row,
)
from tax_rag.timeutil import fmt_ts, parse_ts

UTC = timezone.utc
ALL_TIME = (datetime(2000, 1, 1, tzinfo=UTC), datetime(2100, 1, 1, tzinfo=UTC))


def client(api, **kw):
    return TaxApiClient(api.base_url, "test-token", backoff_factor=0, **kw)


def test_parse_and_format_roundtrip():
    ts = parse_ts("2025-03-01T10:15:00Z")
    assert ts == datetime(2025, 3, 1, 10, 15, tzinfo=UTC)
    assert parse_ts(fmt_ts(ts)) == ts
    assert parse_ts("2025-03-01T10:15:00") == ts.replace(hour=10)  # naive -> UTC
    assert parse_ts("not a date") is None and parse_ts(None) is None


def test_client_paginates_all_documents(mock_api):
    docs = list(client(mock_api, page_size=3).iter_documents(*ALL_TIME))
    assert len(docs) == 11
    pages = [r for r in mock_api.state.requests if r["path"] == "/v1/tax-documents"]
    assert len(pages) == 4 and pages[-1]["page_token"] == "9"


def test_client_window_is_half_open(mock_api):
    lo, hi = parse_ts("2025-01-05T12:00:00Z"), parse_ts("2025-02-14T08:30:00Z")
    ids = [d["document_id"] for d in client(mock_api).iter_documents(lo, hi)]
    assert "zz-income-tax-rates-2025" in ids  # == lower bound: included
    assert "zz-vat-registration-guide" not in ids  # == upper bound: excluded


def test_client_retries_transient_errors(mock_api):
    mock_api.state.fail_first, mock_api.state.rate_limit_first = 2, 1
    assert len(list(client(mock_api, max_retries=5).iter_documents(*ALL_TIME))) == 11


def test_client_gives_up_after_retries(mock_api):
    mock_api.state.fail_first = 10
    with pytest.raises(TaxApiError, match="503"):
        list(client(mock_api, max_retries=1).iter_documents(*ALL_TIME))


def test_client_rejects_bad_token(mock_api):
    with pytest.raises(TaxApiError, match="401"):
        list(TaxApiClient(mock_api.base_url, "wrong", max_retries=0).iter_documents(*ALL_TIME))


def test_reader_options_validation():
    with pytest.raises(ValueError, match="base_url"):
        ReaderOptions.parse({})
    with pytest.raises(ValueError, match="max_partitions"):
        ReaderOptions.parse({"base_url": "http://x", "max_partitions": "0"})
    opts = ReaderOptions.parse({"base_url": "http://x", "start_from": "2025-01-01T00:00:00Z"})
    assert opts.start_from == datetime(2025, 1, 1, tzinfo=UTC) and opts.lag_seconds == 120


def test_quantized_watermark_is_stable_within_granularity():
    a = quantized_watermark(datetime(2025, 1, 1, 0, 11, 5, tzinfo=UTC), 60, 300)
    b = quantized_watermark(datetime(2025, 1, 1, 0, 15, 59, tzinfo=UTC), 60, 300)
    assert a == b == datetime(2025, 1, 1, 0, 10, tzinfo=UTC)


@pytest.mark.parametrize(
    ("hours", "max_parts", "expected"), [(0, 8, 0), (0.5, 8, 1), (5, 8, 5), (1000, 8, 8)]
)
def test_split_window_is_contiguous(hours, max_parts, expected):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    end = start + timedelta(hours=hours)
    parts = split_window(start, end, max_parts, 3600)
    assert len(parts) == expected
    if parts:
        assert parts[0][0] == start and parts[-1][1] == end
        assert all(parts[i][1] == parts[i + 1][0] for i in range(len(parts) - 1))


def test_in_window_keeps_unparseable_records_for_quarantine():
    lo, hi = parse_ts("2025-01-01T00:00:00Z"), parse_ts("2025-02-01T00:00:00Z")
    assert in_window({"updated_at": "2025-01-15T00:00:00Z"}, lo, hi)
    assert not in_window({"updated_at": "2025-02-01T00:00:00Z"}, lo, hi)
    assert in_window({"updated_at": None}, lo, hi)


def test_to_row_shapes_source_schema():
    row = to_row(
        {"document_id": 7, "version": "3", "updated_at": "2025-01-01T00:00:00Z"}, "r", None
    )
    assert row[0] == "7" and row[1] == 3 and row[2] == datetime(2025, 1, 1, tzinfo=UTC)
    assert to_row({"version": "x"}, "r", None)[:3] == (None, None, None)


def test_stream_reader_offsets_partitions_and_read(mock_api):
    reader = TaxApiStreamReader(
        {
            "base_url": mock_api.base_url,
            "api_token": "test-token",
            "start_from": "2023-01-01T00:00:00Z",
            "max_partitions": "4",
            "lag_seconds": "0",
        }
    )
    start, end = reader.initialOffset(), reader.latestOffset()
    assert start["watermark"] < end["watermark"]
    partitions = reader.partitions(start, end)
    assert len(partitions) == 4
    rows = [r for p in partitions for r in reader.read(p)]
    assert len(rows) == 11  # every fixture exactly once across partitions
    assert len({(r[0], r[1]) for r in rows}) == 11
    assert reader.partitions(end, end) == []
