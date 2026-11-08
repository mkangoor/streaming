"""Post-deployment health checks. Pure evaluation logic; stats are gathered by the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DataStats:
    bronze_rows: int
    event_rows: int
    quarantine_rows: int
    live_documents: int
    documents_with_chunks: int
    chunk_rows: int
    empty_chunks: int
    hours_since_last_update: float | None
    index_ready: bool
    index_rows: int


@dataclass(frozen=True)
class Thresholds:
    max_quarantine_ratio: float = 0.02
    min_chunk_coverage: float = 0.99
    min_index_coverage: float = 0.95
    max_staleness_hours: float | None = None


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    observed: Any
    expected: str


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_data_health(s: DataStats, t: Thresholds) -> list[CheckResult]:
    quarantine_ratio = _ratio(s.quarantine_rows, s.event_rows + s.quarantine_rows)
    coverage = _ratio(s.documents_with_chunks, s.live_documents)
    index_coverage = _ratio(s.index_rows, s.chunk_rows)
    results = [
        CheckResult("bronze_has_rows", s.bronze_rows > 0, s.bronze_rows, "> 0"),
        CheckResult("silver_has_live_documents", s.live_documents > 0, s.live_documents, "> 0"),
        CheckResult(
            "quarantine_ratio",
            quarantine_ratio <= t.max_quarantine_ratio,
            round(quarantine_ratio, 4),
            f"<= {t.max_quarantine_ratio}",
        ),
        CheckResult(
            "chunk_coverage",
            coverage >= t.min_chunk_coverage,
            round(coverage, 4),
            f">= {t.min_chunk_coverage}",
        ),
        CheckResult("no_empty_chunks", s.empty_chunks == 0, s.empty_chunks, "== 0"),
        CheckResult("index_ready", s.index_ready, s.index_ready, "True"),
        CheckResult(
            "index_coverage",
            index_coverage >= t.min_index_coverage,
            round(index_coverage, 4),
            f">= {t.min_index_coverage}",
        ),
    ]
    if t.max_staleness_hours is not None:
        stale = s.hours_since_last_update
        results.append(
            CheckResult(
                "freshness",
                stale is not None and stale <= t.max_staleness_hours,
                None if stale is None else round(stale, 2),
                f"<= {t.max_staleness_hours}h",
            )
        )
    return results


def validate_agent_response(
    payload: dict[str, Any], *, require_citations: bool, latency_s: float, max_latency_s: float
) -> list[str]:
    from tax_rag.release.gate import response_text

    errors = []
    if not response_text(payload).strip():
        errors.append("empty answer text")
    custom = payload.get("custom_outputs") or {}
    if "citations" not in custom:
        errors.append("custom_outputs.citations missing (response schema changed?)")
    elif require_citations and not custom["citations"] and not custom.get("refused"):
        errors.append("answer has no citations")
    if latency_s > max_latency_s:
        errors.append(f"latency {latency_s:.1f}s > {max_latency_s}s")
    return errors


def format_report(results: list[CheckResult]) -> str:
    width = max(len(r.name) for r in results)
    lines = [
        f"{'PASS' if r.passed else 'FAIL'}  {r.name:<{width}}  observed={r.observed}  expected {r.expected}"
        for r in results
    ]
    return "\n".join(lines)
