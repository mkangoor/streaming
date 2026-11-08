"""Databricks-backed Retriever (Mosaic AI Vector Search) and ChatModel (Model Serving)."""

from __future__ import annotations

import json
from typing import Any

from tax_rag.rag.engine import RetrievedChunk, message_text

RETRIEVAL_COLUMNS = [
    "chunk_id",
    "document_id",
    "title",
    "chunk_text",
    "source_url",
    "jurisdiction",
    "tax_type",
    "document_type",
    "tax_year",
    "effective_date",
    "expiration_date",
]


def _opt_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_query_response(payload: dict[str, Any]) -> list[RetrievedChunk]:
    """Map a vector search query response (as_dict) to RetrievedChunk rows."""
    columns = [c["name"] for c in payload.get("manifest", {}).get("columns", [])]
    rows = payload.get("result", {}).get("data_array") or []
    chunks = []
    for row in rows:
        rec = dict(zip(columns, row, strict=False))
        chunks.append(
            RetrievedChunk(
                chunk_id=str(rec.get("chunk_id")),
                document_id=str(rec.get("document_id")),
                title=rec.get("title") or "Untitled",
                text=rec.get("chunk_text") or "",
                source_url=rec.get("source_url"),
                jurisdiction=rec.get("jurisdiction"),
                tax_type=rec.get("tax_type"),
                document_type=rec.get("document_type"),
                tax_year=_opt_int(rec.get("tax_year")),
                effective_date=rec.get("effective_date"),
                expiration_date=rec.get("expiration_date"),
                score=float(rec.get("score") or 0.0),
            )
        )
    return chunks


def build_filters(custom_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Translate request custom_inputs into vector search filters (standard endpoint syntax)."""
    if not custom_inputs:
        return None
    filters: dict[str, Any] = {}
    for key in ("jurisdiction", "tax_type", "document_type"):
        value = custom_inputs.get(key)
        if value:
            filters[key] = value if isinstance(value, list) else [value]
    if custom_inputs.get("tax_year") is not None:
        filters["tax_year"] = int(custom_inputs["tax_year"])
    return filters or None


class VectorSearchRetriever:
    def __init__(self, index_name: str, *, query_type: str = "HYBRID", workspace_client=None):
        self.index_name = index_name
        self.query_type = query_type
        self._w = workspace_client

    @property
    def w(self):
        if self._w is None:
            from databricks.sdk import WorkspaceClient

            self._w = WorkspaceClient()
        return self._w

    def retrieve(self, query: str, k: int, filters: dict[str, Any] | None) -> list[RetrievedChunk]:
        response = self.w.vector_search_indexes.query_index(
            index_name=self.index_name,
            columns=RETRIEVAL_COLUMNS,
            query_text=query,
            num_results=k,
            filters_json=json.dumps(filters) if filters else None,
            query_type=self.query_type,
        )
        return parse_query_response(response.as_dict())


class ServingChatModel:
    """Chat completion against a Databricks Model Serving / Foundation Model API endpoint."""

    def __init__(self, endpoint: str, *, max_tokens: int = 2048, workspace_client=None):
        self.endpoint = endpoint
        self.max_tokens = max_tokens
        self._w = workspace_client

    @property
    def w(self):
        if self._w is None:
            from databricks.sdk import WorkspaceClient

            self._w = WorkspaceClient()
        return self._w

    def complete(self, messages: list[dict[str, str]]) -> str:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = self.w.serving_endpoints.query(
            name=self.endpoint,
            messages=[
                ChatMessage(role=ChatMessageRole(m["role"]), content=m["content"]) for m in messages
            ],
            max_tokens=self.max_tokens,
        )
        if not response.choices:
            raise RuntimeError(f"Endpoint {self.endpoint} returned no choices")
        return message_text(response.choices[0].message.content)
