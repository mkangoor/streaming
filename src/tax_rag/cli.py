"""`tax-rag` entry point used by every Databricks job task (python_wheel_task)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence

from tax_rag.names import Names

log = logging.getLogger("tax_rag")


def _names(args: argparse.Namespace) -> Names:
    return Names(args.catalog, args.schema)


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.builder.getOrCreate()


def _task_value(key: str, value) -> None:
    try:
        from databricks.sdk.runtime import dbutils

        dbutils.jobs.taskValues.set(key=key, value=value)
    except Exception as exc:  # local runs have no job context
        log.warning("Could not set task value %s=%s: %s", key, value, exc)


# ------------------------------------------------------------------ commands
def cmd_sync_chunks(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from tax_rag import names as n
    from tax_rag.catalog.contract import load_contract
    from tax_rag.gold.chunk_sync import ensure_gold_table, run_chunk_sync
    from tax_rag.transform.chunking import ChunkConfig
    from tax_rag.vector.index import (
        IndexSpec,
        ensure_endpoint,
        ensure_index,
        trigger_sync,
        wait_until_ready,
    )

    names = _names(args)
    spark = _spark()
    w = WorkspaceClient()
    ensure_gold_table(spark, names, load_contract())
    spec = IndexSpec(
        endpoint_name=args.vector_search_endpoint,
        index_name=names.index,
        source_table=names.table(n.GOLD_CHUNKS),
        embedding_endpoint=args.embedding_endpoint,
        pipeline_type=args.index_pipeline_type,
    )
    ensure_endpoint(w, spec.endpoint_name)
    ensure_index(w, spec)
    triggered = spec.pipeline_type == "TRIGGERED"
    continuous = args.trigger == "continuous"
    last_sync = [0.0]

    def on_batch_end(batch_id: int) -> None:
        # Continuous stream + TRIGGERED index: push syncs, at most every N seconds.
        if continuous and triggered and time.monotonic() - last_sync[0] >= args.min_sync_interval_s:
            try:
                trigger_sync(w, spec.index_name)
                last_sync[0] = time.monotonic()
            except Exception as exc:  # a sync already running is fine
                log.warning("Index sync after batch %s skipped: %s", batch_id, exc)

    run_chunk_sync(
        spark,
        names,
        continuous=continuous,
        config=ChunkConfig(max_chars=args.chunk_max_chars, overlap_chars=args.chunk_overlap_chars),
        on_batch_end=on_batch_end,
        reset_checkpoint=args.reset_checkpoint,
        checkpoint_root=args.checkpoint_root,
    )
    if triggered:
        wait_until_ready(w, spec.index_name, timeout_s=args.index_timeout_s)
        trigger_sync(w, spec.index_name)
        wait_until_ready(w, spec.index_name, timeout_s=args.index_timeout_s)
    return 0


def cmd_apply_governance(args: argparse.Namespace) -> int:
    from tax_rag.catalog.contract import load_contract

    spark = _spark()
    failures = []
    for stmt in load_contract().governance_sql(_names(args)):
        try:
            spark.sql(stmt)
            log.info("OK   %s", stmt.splitlines()[0][:160])
        except Exception as exc:
            failures.append((stmt, exc))
            log.error("FAIL %s -> %s", stmt[:160], exc)
    if failures and args.strict:
        raise SystemExit(f"{len(failures)} governance statement(s) failed")
    return 0


def cmd_register_agent(args: argparse.Namespace) -> int:
    from tax_rag.release.mlflow_ops import register_agent

    version = register_agent(
        _names(args),
        experiment=args.experiment,
        llm_endpoint=args.llm_endpoint,
        embedding_endpoint=args.embedding_endpoint,
        git_sha=args.git_sha,
    )
    _task_value("model_version", version)
    print(f"model_version={version}")
    return 0


def cmd_evaluate_agent(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from tax_rag.release.mlflow_ops import evaluate_agent
    from tax_rag.vector.index import wait_until_ready

    names = _names(args)
    wait_until_ready(WorkspaceClient(), names.index, timeout_s=args.index_timeout_s)
    thresholds = {
        "relevance_to_query": args.min_relevance,
        "retrieval_groundedness": args.min_groundedness,
        "safety": args.min_safety,
        "correctness": args.min_correctness,
        "cites_or_declines": args.min_cites_or_declines,
        "citation_rate": args.min_citation_rate,
    }
    gate = evaluate_agent(
        names,
        int(args.model_version),
        experiment=args.experiment,
        eval_set=args.eval_set,
        thresholds=thresholds,
    )
    failed = [g for g in gate if not g.passed]
    if failed:
        raise SystemExit(
            "Quality gate failed: "
            + ", ".join(f"{g.metric}={g.value}<{g.threshold}" for g in failed)
        )
    return 0


def cmd_deploy_agent(args: argparse.Namespace) -> int:
    from tax_rag.release.mlflow_ops import deploy_agent

    deploy_agent(
        _names(args),
        int(args.model_version),
        endpoint_name=args.endpoint_name,
        scale_to_zero=args.scale_to_zero == "true",
        git_sha=args.git_sha,
    )
    return 0


def cmd_verify_data(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from tax_rag import names as n
    from tax_rag.vector.index import index_status
    from tax_rag.verify.checks import DataStats, Thresholds, evaluate_data_health, format_report

    names = _names(args)
    spark = _spark()

    def scalar(sql: str):
        return spark.sql(sql).first()[0]

    gold, current = names.sql(n.GOLD_CHUNKS), names.sql(n.SILVER_CURRENT)
    ready, index_rows, message = index_status(WorkspaceClient(), names.index)
    stats = DataStats(
        bronze_rows=scalar(f"SELECT count(*) FROM {names.sql(n.BRONZE)}"),
        event_rows=scalar(f"SELECT count(*) FROM {names.sql(n.SILVER_EVENTS)}"),
        quarantine_rows=scalar(f"SELECT count(*) FROM {names.sql(n.QUARANTINE)}"),
        live_documents=scalar(f"SELECT count(*) FROM {current}"),
        documents_with_chunks=scalar(
            f"SELECT count(DISTINCT c.document_id) FROM {current} c "
            f"JOIN {gold} g ON g.document_id = c.document_id AND g.document_version = c.document_version"
        ),
        chunk_rows=scalar(f"SELECT count(*) FROM {gold}"),
        empty_chunks=scalar(
            f"SELECT count(*) FROM {gold} WHERE chunk_text IS NULL OR length(chunk_text) = 0"
        ),
        hours_since_last_update=scalar(
            f"SELECT (unix_timestamp(current_timestamp()) - unix_timestamp(max(updated_at))) / 3600.0 "
            f"FROM {current}"
        ),
        index_ready=ready,
        index_rows=index_rows,
    )
    thresholds = Thresholds(
        max_quarantine_ratio=args.max_quarantine_ratio,
        min_chunk_coverage=args.min_chunk_coverage,
        min_index_coverage=args.min_index_coverage,
        max_staleness_hours=args.max_staleness_hours if args.max_staleness_hours > 0 else None,
    )
    results = evaluate_data_health(stats, thresholds)
    print(format_report(results))
    if message:
        print(f"index status: {message}")
    if not all(r.passed for r in results):
        raise SystemExit("Data health checks failed")
    return 0


def cmd_smoke_test(args: argparse.Namespace) -> int:
    from databricks.sdk import WorkspaceClient

    from tax_rag.verify.checks import validate_agent_response

    w = WorkspaceClient()
    failures = []
    for question in args.question:
        started = time.monotonic()
        payload = w.api_client.do(
            "POST",
            f"/serving-endpoints/{args.endpoint_name}/invocations",
            body={"input": [{"role": "user", "content": question}]},
        )
        latency = time.monotonic() - started
        errors = validate_agent_response(
            payload,
            require_citations=args.require_citations == "true",
            latency_s=latency,
            max_latency_s=args.max_latency_s,
        )
        status = "FAIL" if errors else "PASS"
        print(f"{status} ({latency:.1f}s) {question!r} {errors or ''}")
        failures += errors
    if failures:
        raise SystemExit(f"Smoke test failed: {failures}")
    return 0


# ------------------------------------------------------------------ parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tax-rag")
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, fn, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--catalog", required=True)
        sp.add_argument("--schema", required=True)
        sp.set_defaults(func=fn)
        return sp

    s = add("sync-chunks", cmd_sync_chunks, "silver events -> gold chunks -> vector index")
    s.add_argument("--trigger", choices=["available-now", "continuous"], default="available-now")
    s.add_argument("--vector-search-endpoint", required=True)
    s.add_argument("--embedding-endpoint", required=True)
    s.add_argument(
        "--index-pipeline-type", choices=["TRIGGERED", "CONTINUOUS"], default="TRIGGERED"
    )
    s.add_argument("--chunk-max-chars", type=int, default=2000)
    s.add_argument("--chunk-overlap-chars", type=int, default=200)
    s.add_argument("--min-sync-interval-s", type=int, default=300)
    s.add_argument("--index-timeout-s", type=int, default=3600)
    s.add_argument("--reset-checkpoint", action="store_true")
    s.add_argument("--checkpoint-root", default=None, help="UC volume path for stream checkpoints")

    s = add("apply-governance", cmd_apply_governance, "apply UC comments + tags from the contract")
    s.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    s = add("register-agent", cmd_register_agent, "log + register the agent in Unity Catalog")
    s.add_argument("--experiment", required=True)
    s.add_argument("--llm-endpoint", required=True)
    s.add_argument("--embedding-endpoint", required=True)
    s.add_argument("--git-sha", default=None)

    s = add("evaluate-agent", cmd_evaluate_agent, "MLflow GenAI evaluation + quality gate")
    s.add_argument("--model-version", required=True)
    s.add_argument("--experiment", required=True)
    s.add_argument("--eval-set", required=True)
    s.add_argument("--index-timeout-s", type=int, default=3600)
    s.add_argument("--min-relevance", type=float, default=0.8)
    s.add_argument("--min-groundedness", type=float, default=0.8)
    s.add_argument("--min-safety", type=float, default=1.0)
    s.add_argument("--min-correctness", type=float, default=0.7)
    s.add_argument("--min-cites-or-declines", type=float, default=0.9)
    s.add_argument("--min-citation-rate", type=float, default=0.9)

    s = add("deploy-agent", cmd_deploy_agent, "deploy a gated model version to Model Serving")
    s.add_argument("--model-version", required=True)
    s.add_argument("--endpoint-name", required=True)
    s.add_argument("--scale-to-zero", choices=["true", "false"], default="true")
    s.add_argument("--git-sha", default="unknown")

    s = add("verify-data", cmd_verify_data, "data health checks across bronze/silver/gold/index")
    s.add_argument("--max-quarantine-ratio", type=float, default=0.02)
    s.add_argument("--min-chunk-coverage", type=float, default=0.99)
    s.add_argument("--min-index-coverage", type=float, default=0.95)
    s.add_argument("--max-staleness-hours", type=float, default=0, help="0 disables")

    s = add("smoke-test", cmd_smoke_test, "call the deployed agent endpoint")
    s.add_argument("--endpoint-name", required=True)
    s.add_argument("--question", action="append", required=True)
    s.add_argument("--require-citations", choices=["true", "false"], default="false")
    s.add_argument("--max-latency-s", type=float, default=60.0)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = build_parser().parse_args(argv if argv is not None else sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
