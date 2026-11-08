"""PySpark custom streaming data source that incrementally polls the tax API.

    spark.dataSource.register(TaxApiDataSource)
    spark.readStream.format("tax_api").option("base_url", ...).load()

Offsets are UTC time watermarks: each micro-batch covers [previous_watermark, new_watermark).
Replaying a batch re-requests exactly the same windows, so Structured Streaming's checkpoint
gives exactly-once delivery into bronze *provided the API is consistent for past windows*.

`lag_seconds` keeps the watermark behind "now" so records that land late in the API's own
store are not skipped. `granularity_seconds` quantises the watermark so an idle stream
doesn't hammer the API with empty micro-batches.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from pyspark.sql.datasource import DataSource, DataSourceStreamReader, InputPartition

from tax_rag.ingest.api_client import TaxApiClient
from tax_rag.timeutil import fmt_ts, parse_ts, require_ts, utcnow

SOURCE_SCHEMA = (
    "document_id STRING, document_version BIGINT, updated_at TIMESTAMP, "
    "payload STRING, source_ref STRING, fetched_at TIMESTAMP"
)
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ReaderOptions:
    base_url: str
    api_token: str | None
    start_from: datetime
    lag_seconds: int = 120
    granularity_seconds: int = 60
    max_partitions: int = 8
    min_partition_seconds: int = 3600
    page_size: int = 500
    timeout_seconds: float = 30.0
    max_retries: int = 5

    @classmethod
    def parse(cls, options: dict[str, str]) -> ReaderOptions:
        opts = {k.lower(): v for k, v in options.items()}
        base_url = opts.get("base_url", "").strip()
        if not base_url:
            raise ValueError("tax_api source requires option 'base_url'")

        def _int(name: str, default: int, minimum: int) -> int:
            raw = opts.get(name)
            value = default if raw in (None, "") else int(raw)
            if value < minimum:
                raise ValueError(f"option {name} must be >= {minimum}, got {value}")
            return value

        return cls(
            base_url=base_url,
            api_token=opts.get("api_token") or None,
            start_from=require_ts(opts.get("start_from", "2024-01-01T00:00:00Z"), "start_from"),
            lag_seconds=_int("lag_seconds", cls.lag_seconds, 0),
            granularity_seconds=_int("granularity_seconds", cls.granularity_seconds, 1),
            max_partitions=_int("max_partitions", cls.max_partitions, 1),
            min_partition_seconds=_int("min_partition_seconds", cls.min_partition_seconds, 1),
            page_size=_int("page_size", cls.page_size, 1),
            timeout_seconds=float(opts.get("timeout_seconds") or cls.timeout_seconds),
            max_retries=_int("max_retries", cls.max_retries, 0),
        )


def quantized_watermark(now: datetime, lag_seconds: int, granularity_seconds: int) -> datetime:
    """Floor (now - lag) to a multiple of granularity, so repeated calls are stable."""
    target = (now - timedelta(seconds=lag_seconds)).astimezone(timezone.utc)
    elapsed = math.floor((target - EPOCH).total_seconds())
    return EPOCH + timedelta(seconds=elapsed - elapsed % granularity_seconds)


def split_window(
    start: datetime, end: datetime, max_partitions: int, min_partition_seconds: int
) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into <= max_partitions contiguous, non-overlapping slices.

    Deterministic for a given (start, end), which is what makes batch replay exactly-once.
    """
    if end <= start:
        return []
    total = (end - start).total_seconds()
    count = max(1, min(max_partitions, math.ceil(total / min_partition_seconds)))
    step = total / count
    bounds = [start + timedelta(seconds=step * i) for i in range(count)] + [end]
    return [(bounds[i], bounds[i + 1]) for i in range(count)]


def to_row(doc: dict, window_ref: str, fetched_at: datetime) -> tuple:
    version = doc.get("version")
    try:
        version = int(version) if version is not None else None
    except (TypeError, ValueError):
        version = None
    doc_id = doc.get("document_id")
    return (
        str(doc_id) if doc_id is not None else None,
        version,
        parse_ts(doc.get("updated_at")) if isinstance(doc.get("updated_at"), str) else None,
        json.dumps(doc, sort_keys=True, ensure_ascii=False),
        window_ref,
        fetched_at,
    )


def in_window(doc: dict, start: datetime, end: datetime) -> bool:
    """Guard against APIs that ignore updated_to: keep only [start, end).

    Records with a missing/invalid updated_at are kept so silver can quarantine them
    visibly instead of silently dropping them here.
    """
    raw = doc.get("updated_at")
    ts = parse_ts(raw) if isinstance(raw, str) else None
    return ts is None or start <= ts < end


class TaxApiStreamReader(DataSourceStreamReader):
    def __init__(self, options: dict[str, str]) -> None:
        self.opts = ReaderOptions.parse(options)

    def initialOffset(self) -> dict:
        return {"watermark": fmt_ts(self.opts.start_from)}

    def latestOffset(self) -> dict:
        mark = quantized_watermark(utcnow(), self.opts.lag_seconds, self.opts.granularity_seconds)
        return {"watermark": fmt_ts(max(mark, self.opts.start_from))}

    def partitions(self, start: dict, end: dict) -> Sequence[InputPartition]:
        lo = require_ts(start["watermark"], "start offset")
        hi = require_ts(end["watermark"], "end offset")
        slices = split_window(lo, hi, self.opts.max_partitions, self.opts.min_partition_seconds)
        return [InputPartition({"from": fmt_ts(a), "to": fmt_ts(b)}) for a, b in slices]

    def read(self, partition: InputPartition) -> Iterator[tuple]:
        lo = require_ts(partition.value["from"], "partition.from")
        hi = require_ts(partition.value["to"], "partition.to")
        client = TaxApiClient(
            self.opts.base_url,
            self.opts.api_token,
            page_size=self.opts.page_size,
            timeout_seconds=self.opts.timeout_seconds,
            max_retries=self.opts.max_retries,
        )
        ref = f"api:{partition.value['from']}/{partition.value['to']}"
        fetched_at = utcnow()
        for doc in client.iter_documents(lo, hi):
            if isinstance(doc, dict) and in_window(doc, lo, hi):
                yield to_row(doc, ref, fetched_at)

    def commit(self, end: dict) -> None:  # offsets live in the Spark checkpoint
        return None


class TaxApiDataSource(DataSource):
    @classmethod
    def name(cls) -> str:
        return "tax_api"

    def schema(self) -> str:
        return SOURCE_SCHEMA

    def streamReader(self, schema) -> DataSourceStreamReader:
        return TaxApiStreamReader(self.options)
