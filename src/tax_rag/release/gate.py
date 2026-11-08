"""Pure helpers for the evaluation quality gate (unit-tested; no MLflow import)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GateResult:
    metric: str
    value: float | None
    threshold: float
    passed: bool


def load_eval_set(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Eval set not found: {p}. Curate one from real questions over your corpus before "
            "releasing (see evals/README.md). No eval set, no release."
        )
    records = []
    for lineno, line in enumerate(p.read_text("utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        rec = json.loads(line)
        if "inputs" not in rec or "question" not in rec["inputs"]:
            raise ValueError(f"{p}:{lineno}: each record needs inputs.question")
        records.append(rec)
    if not records:
        raise ValueError(f"{p} contains no eval records")
    return records


def metric_value(metrics: dict[str, Any], scorer: str) -> float | None:
    """Find a scorer's mean in MLflow's metrics dict, tolerating key-format differences
    across MLflow versions (e.g. 'safety/mean', 'safety/value/mean', 'metric/safety/mean')."""
    pattern = re.compile(rf"(^|/){re.escape(scorer)}(/value)?/mean$")
    for key, value in metrics.items():
        if pattern.search(key) and value is not None:
            return float(value)
    return None


def evaluate_gate(
    metrics: dict[str, Any], thresholds: dict[str, float], *, require_all: bool = True
) -> list[GateResult]:
    results = []
    for name, threshold in thresholds.items():
        value = metric_value(metrics, name)
        passed = value is not None and value >= threshold
        if value is None and not require_all:
            passed = True
        results.append(GateResult(name, value, threshold, passed))
    return results


def response_text(response: Any) -> str:
    """Text of a ResponsesAgent response (dict or pydantic)."""
    if hasattr(response, "model_dump"):
        response = response.model_dump()
    texts = []
    for item in (response or {}).get("output", []) or []:
        for part in item.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") in ("output_text", "text"):
                texts.append(part.get("text", ""))
    return "\n".join(t for t in texts if t)


def deterministic_metrics(
    outputs: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, float]:
    """Cheap, judge-free metrics computed from the agent's custom_outputs."""
    if not outputs:
        return {}
    answerable = [
        o
        for o, r in zip(outputs, records, strict=False)
        if not r.get("expectations", {}).get("should_refuse")
    ]
    unanswerable = [
        o
        for o, r in zip(outputs, records, strict=False)
        if r.get("expectations", {}).get("should_refuse")
    ]
    metrics = {
        "grounded_rate/mean": sum(bool(o.get("grounded")) for o in outputs) / len(outputs),
    }
    if answerable:
        metrics["citation_rate/mean"] = sum(bool(o.get("citations")) for o in answerable) / len(
            answerable
        )
    if unanswerable:
        metrics["refusal_rate/mean"] = sum(bool(o.get("refused")) for o in unanswerable) / len(
            unanswerable
        )
    return metrics
