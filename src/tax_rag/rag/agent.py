"""MLflow ResponsesAgent wrapper (models-from-code). Logged by `tax-rag register-agent`.

Request:
    {"input": [{"role": "user", "content": "..."}],
     "custom_inputs": {"jurisdiction": "US", "tax_year": 2025, "as_of": "2025-06-30"}}
Response custom_outputs carry citations, retrieved chunk ids, refusal and grounding flags.
"""

from __future__ import annotations

import uuid
from datetime import date

import mlflow
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from tax_rag.rag.clients import ServingChatModel, VectorSearchRetriever, build_filters
from tax_rag.rag.engine import TaxRagEngine

DEFAULT_CONFIG = {
    "index_name": "tax_dev.tax_rag.gold_tax_document_chunks_index",
    "llm_endpoint": "databricks-claude-sonnet-4-5",
    "top_k": 6,
    "min_score": 0.0,
    "max_tokens": 2048,
    "query_type": "HYBRID",
}


class TaxRagAgent(ResponsesAgent):
    def __init__(self, engine: TaxRagEngine | None = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> TaxRagEngine:
        # Built lazily: logging the model must not need workspace credentials.
        if self._engine is None:
            cfg = mlflow.models.ModelConfig(development_config=DEFAULT_CONFIG)
            self._engine = TaxRagEngine(
                VectorSearchRetriever(cfg.get("index_name"), query_type=cfg.get("query_type")),
                ServingChatModel(cfg.get("llm_endpoint"), max_tokens=int(cfg.get("max_tokens"))),
                top_k=int(cfg.get("top_k")),
                min_score=float(cfg.get("min_score")),
            )
        return self._engine

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        messages = [m.model_dump() if hasattr(m, "model_dump") else dict(m) for m in request.input]
        custom = dict(request.custom_inputs or {})
        as_of = date.fromisoformat(custom["as_of"]) if custom.get("as_of") else None
        result = self.engine.answer(messages, filters=build_filters(custom), as_of=as_of)
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=result.answer, id=str(uuid.uuid4()))],
            custom_outputs=result.to_dict(),
        )


AGENT = TaxRagAgent()
mlflow.models.set_model(AGENT)
