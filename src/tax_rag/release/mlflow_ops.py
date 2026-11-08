"""Register, evaluate and deploy the agent (runs inside Databricks jobs).

Flow per environment ("deploy code", not "deploy model"): log from code -> register in that
environment's UC catalog -> evaluate on that environment's index -> deploy only if the gate
passes -> move the `champion` alias. Rollback = redeploy the previous champion version.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient

from tax_rag.names import Names
from tax_rag.release.gate import (
    GateResult,
    deterministic_metrics,
    evaluate_gate,
    load_eval_set,
)

log = logging.getLogger(__name__)
PIP_REQUIREMENTS = [
    "mlflow[databricks]>=3.1",
    "databricks-sdk>=0.50",
    "pydantic>=2",
]
CHAMPION = "champion"


def agent_config(
    names: Names, llm_endpoint: str, top_k: int = 6, min_score: float = 0.0
) -> dict[str, Any]:
    return {
        "index_name": names.index,
        "llm_endpoint": llm_endpoint,
        "top_k": top_k,
        "min_score": min_score,
        "max_tokens": 2048,
        "query_type": "HYBRID",
    }


def register_agent(
    names: Names,
    *,
    experiment: str,
    llm_endpoint: str,
    embedding_endpoint: str,
    git_sha: str | None = None,
) -> int:
    from mlflow.models.resources import DatabricksServingEndpoint, DatabricksVectorSearchIndex

    import tax_rag

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)
    package_dir = Path(tax_rag.__file__).parent
    config = agent_config(names, llm_endpoint)
    with mlflow.start_run(run_name="register_agent") as run:
        mlflow.set_tags({"git_sha": git_sha or "unknown", "component": "tax_rag_agent"})
        mlflow.log_params({k: v for k, v in config.items()})
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(package_dir / "rag" / "agent.py"),
            model_config=config,
            code_paths=[str(package_dir)],
            pip_requirements=PIP_REQUIREMENTS,
            # Declared resources enable automatic auth passthrough on Model Serving.
            resources=[
                DatabricksServingEndpoint(endpoint_name=llm_endpoint),
                DatabricksServingEndpoint(endpoint_name=embedding_endpoint),
                DatabricksVectorSearchIndex(index_name=names.index),
            ],
            input_example={
                "input": [{"role": "user", "content": "What is the standard VAT rate?"}]
            },
            registered_model_name=names.agent_model,
        )
    version = int(info.registered_model_version)
    log.info("Registered %s version %s (run %s)", names.agent_model, version, run.info.run_id)
    return version


def evaluate_agent(
    names: Names,
    version: int,
    *,
    experiment: str,
    eval_set: str,
    thresholds: dict[str, float],
) -> list[GateResult]:
    from mlflow.genai.scorers import (
        Correctness,
        Guidelines,
        RelevanceToQuery,
        RetrievalGroundedness,
        Safety,
    )

    from tax_rag.release.gate import response_text

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(experiment)
    records = load_eval_set(eval_set)
    model = mlflow.pyfunc.load_model(f"models:/{names.agent_model}/{version}")
    outputs: list[dict[str, Any]] = []

    def predict_fn(question: str, **custom_inputs: Any) -> str:
        payload: dict[str, Any] = {"input": [{"role": "user", "content": question}]}
        if custom_inputs:
            payload["custom_inputs"] = custom_inputs
        response = model.predict(payload)
        resp = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        outputs.append(resp.get("custom_outputs") or {})
        return response_text(resp)

    scorers = [
        RelevanceToQuery(),
        RetrievalGroundedness(),
        Safety(),
        Correctness(),
        Guidelines(
            name="cites_or_declines",
            guidelines=(
                "The response either cites its sources with bracketed markers such as [1], "
                "or clearly states that the provided sources do not contain the answer. "
                "It must not state tax rates, thresholds or deadlines without a citation."
            ),
        ),
    ]
    with mlflow.start_run(run_name=f"evaluate_v{version}"):
        result = mlflow.genai.evaluate(data=records, predict_fn=predict_fn, scorers=scorers)
        metrics = {**(result.metrics or {}), **deterministic_metrics(outputs, records)}
        mlflow.log_metrics(
            {k.replace("/", "_"): float(v) for k, v in metrics.items() if v is not None}
        )
    gate = evaluate_gate(metrics, thresholds)
    for g in gate:
        log.info(
            "gate %-28s value=%s threshold=%s passed=%s", g.metric, g.value, g.threshold, g.passed
        )
    client = MlflowClient(registry_uri="databricks-uc")
    client.set_model_version_tag(
        names.agent_model, str(version), "eval_passed", str(all(g.passed for g in gate)).lower()
    )
    return gate


def deploy_agent(
    names: Names,
    version: int,
    *,
    endpoint_name: str,
    scale_to_zero: bool,
    git_sha: str = "unknown",
    timeout_minutes: int = 45,
) -> None:
    from datetime import timedelta

    from databricks import agents
    from databricks.sdk import WorkspaceClient

    client = MlflowClient(registry_uri="databricks-uc")
    tags = client.get_model_version(names.agent_model, str(version)).tags or {}
    if tags.get("eval_passed") != "true":
        raise RuntimeError(
            f"Refusing to deploy {names.agent_model} v{version}: eval_passed={tags.get('eval_passed')!r}"
        )
    agents.deploy(
        names.agent_model,
        version,
        endpoint_name=endpoint_name,
        scale_to_zero=scale_to_zero,
        tags={"component": "tax_rag_agent", "git_sha": git_sha},
    )
    WorkspaceClient().serving_endpoints.wait_get_serving_endpoint_not_updating(
        endpoint_name, timeout=timedelta(minutes=timeout_minutes)
    )
    client.set_registered_model_alias(names.agent_model, CHAMPION, str(version))
    log.info(
        "Deployed %s v%s to %s; alias %s updated",
        names.agent_model,
        version,
        endpoint_name,
        CHAMPION,
    )
