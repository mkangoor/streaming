import json
from pathlib import Path

import pytest

from tax_rag.release.gate import (
    deterministic_metrics,
    evaluate_gate,
    load_eval_set,
    metric_value,
    response_text,
)
from tax_rag.verify.checks import (
    DataStats,
    Thresholds,
    evaluate_data_health,
    format_report,
    validate_agent_response,
)

ROOT = Path(__file__).resolve().parents[2]


def test_metric_value_tolerates_key_formats():
    metrics = {
        "safety/mean": 1.0,
        "metric/correctness/value/mean": 0.75,
        "relevance_to_query/count": 9,
    }
    assert metric_value(metrics, "safety") == 1.0
    assert metric_value(metrics, "correctness") == 0.75
    assert metric_value(metrics, "relevance_to_query") is None


def test_gate_fails_on_missing_or_low_metrics():
    results = {
        g.metric: g
        for g in evaluate_gate(
            {"safety/mean": 1.0, "correctness/mean": 0.5},
            {"safety": 1.0, "correctness": 0.7, "absent": 0.1},
        )
    }
    assert (
        results["safety"].passed
        and not results["correctness"].passed
        and not results["absent"].passed
    )
    assert evaluate_gate({}, {"absent": 0.1}, require_all=False)[0].passed


def test_deterministic_metrics():
    records = [
        {"expectations": {}},
        {"expectations": {}},
        {"expectations": {"should_refuse": True}},
    ]
    outputs = [
        {"citations": [1], "grounded": True},
        {"citations": [], "grounded": False},
        {"refused": True, "grounded": True},
    ]
    m = deterministic_metrics(outputs, records)
    assert m == {
        "grounded_rate/mean": pytest.approx(2 / 3),
        "citation_rate/mean": 0.5,
        "refusal_rate/mean": 1.0,
    }


def test_response_text():
    resp = {"output": [{"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]}
    assert response_text(resp) == "Hello" and response_text({}) == ""


def test_shipped_eval_set_is_valid():
    records = load_eval_set(ROOT / "evals" / "fixture_eval_set.jsonl")
    assert len(records) >= 8
    assert any(r["expectations"].get("should_refuse") for r in records)
    for r in records:
        assert r["expectations"].get("expected_facts") or r["expectations"].get("expected_response")


def test_missing_eval_set_blocks_release(tmp_path):
    with pytest.raises(FileNotFoundError, match="No eval set, no release"):
        load_eval_set(tmp_path / "nope.jsonl")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"inputs": {}}) + "\n")
    with pytest.raises(ValueError, match=r"inputs\.question"):
        load_eval_set(bad)


def healthy(**overrides):
    base = dict(
        bronze_rows=11,
        event_rows=10,
        quarantine_rows=1,
        live_documents=8,
        documents_with_chunks=8,
        chunk_rows=9,
        empty_chunks=0,
        hours_since_last_update=2.0,
        index_ready=True,
        index_rows=9,
    )
    return DataStats(**{**base, **overrides})


def test_data_health_pass_and_failures():
    t = Thresholds(max_quarantine_ratio=0.15)
    assert all(r.passed for r in evaluate_data_health(healthy(), t))
    failed = {
        r.name
        for r in evaluate_data_health(
            healthy(documents_with_chunks=5, index_ready=False, empty_chunks=1), t
        )
        if not r.passed
    }
    assert failed == {"chunk_coverage", "index_ready", "no_empty_chunks"}
    stale = evaluate_data_health(
        healthy(hours_since_last_update=30), Thresholds(0.15, max_staleness_hours=24)
    )
    assert not stale[-1].passed and stale[-1].name == "freshness"
    assert "FAIL" in format_report(evaluate_data_health(healthy(bronze_rows=0), t))


def test_validate_agent_response():
    ok = {
        "output": [{"content": [{"type": "output_text", "text": "18% [1]"}]}],
        "custom_outputs": {"citations": [{"index": 1}]},
    }
    assert validate_agent_response(ok, require_citations=True, latency_s=1, max_latency_s=10) == []
    refused = {
        "output": [{"content": [{"type": "output_text", "text": "not found"}]}],
        "custom_outputs": {"citations": [], "refused": True},
    }
    assert (
        validate_agent_response(refused, require_citations=True, latency_s=1, max_latency_s=10)
        == []
    )
    errors = validate_agent_response(
        {"output": []}, require_citations=True, latency_s=20, max_latency_s=10
    )
    assert len(errors) == 3
