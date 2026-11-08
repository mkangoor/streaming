"""Idempotent Mosaic AI Vector Search setup: endpoint + Delta Sync index with managed embeddings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.vectorsearch import (
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
    EndpointType,
    PipelineType,
    VectorIndexType,
)

log = logging.getLogger(__name__)

SYNCED_COLUMNS = [
    "chunk_id",
    "document_id",
    "document_version",
    "chunk_text",
    "title",
    "document_type",
    "jurisdiction",
    "tax_type",
    "tax_year",
    "effective_date",
    "expiration_date",
    "source_url",
]


@dataclass(frozen=True)
class IndexSpec:
    endpoint_name: str
    index_name: str
    source_table: str
    embedding_endpoint: str
    pipeline_type: str = "TRIGGERED"
    primary_key: str = "chunk_id"
    embedding_column: str = "chunk_text"


def ensure_endpoint(w: WorkspaceClient, name: str, timeout_s: int = 1800) -> None:
    try:
        w.vector_search_endpoints.get_endpoint(name)
        log.info("Vector search endpoint %s exists", name)
    except NotFound:
        log.info("Creating vector search endpoint %s", name)
        w.vector_search_endpoints.create_endpoint(name=name, endpoint_type=EndpointType.STANDARD)
    _wait(lambda: _endpoint_online(w, name), timeout_s, f"endpoint {name} ONLINE")


def _endpoint_online(w: WorkspaceClient, name: str) -> bool:
    status = w.vector_search_endpoints.get_endpoint(name).endpoint_status
    state = getattr(getattr(status, "state", None), "value", None)
    if state in {"OFFLINE", "RED_STATE"}:
        raise RuntimeError(f"Vector search endpoint {name} is {state}: {status}")
    return state == "ONLINE"


def ensure_index(w: WorkspaceClient, spec: IndexSpec) -> bool:
    """Create the index if missing. Returns True when it was created."""
    try:
        existing = w.vector_search_indexes.get_index(spec.index_name)
        actual = getattr(
            getattr(existing.delta_sync_index_spec, "pipeline_type", None), "value", None
        )
        if actual and actual != spec.pipeline_type:
            log.warning(
                "Index %s is %s but config wants %s; pipeline type can't be changed in place. "
                "Drop and recreate the index to switch.",
                spec.index_name,
                actual,
                spec.pipeline_type,
            )
        return False
    except NotFound:
        log.info("Creating Delta Sync index %s on %s", spec.index_name, spec.source_table)
        w.vector_search_indexes.create_index(
            name=spec.index_name,
            endpoint_name=spec.endpoint_name,
            primary_key=spec.primary_key,
            index_type=VectorIndexType.DELTA_SYNC,
            delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
                source_table=spec.source_table,
                pipeline_type=PipelineType(spec.pipeline_type),
                embedding_source_columns=[
                    EmbeddingSourceColumn(
                        name=spec.embedding_column,
                        embedding_model_endpoint_name=spec.embedding_endpoint,
                    )
                ],
                columns_to_sync=SYNCED_COLUMNS,
            ),
        )
        return True


def index_status(w: WorkspaceClient, index_name: str) -> tuple[bool, int, str]:
    status = w.vector_search_indexes.get_index(index_name).status
    if status is None:
        return False, 0, "no status"
    return bool(status.ready), int(status.indexed_row_count or 0), status.message or ""


def wait_until_ready(w: WorkspaceClient, index_name: str, timeout_s: int = 3600) -> None:
    _wait(lambda: index_status(w, index_name)[0], timeout_s, f"index {index_name} ready")


def trigger_sync(w: WorkspaceClient, index_name: str) -> None:
    log.info("Triggering sync for %s", index_name)
    w.vector_search_indexes.sync_index(index_name)


def _wait(predicate, timeout_s: int, what: str, poll_s: int = 20) -> None:
    deadline = time.monotonic() + timeout_s
    while True:
        if predicate():
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for {what}")
        time.sleep(poll_s)
