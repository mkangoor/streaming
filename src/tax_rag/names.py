"""Single source of truth for Unity Catalog object names.

Every job, the pipeline and the agent derive names from here so a rename is one edit.
"""

from __future__ import annotations

from dataclasses import dataclass

BRONZE = "bronze_tax_documents"
SILVER_EVENTS = "silver_tax_document_events"
QUARANTINE = "silver_tax_document_quarantine"
SILVER_CURRENT = "silver_tax_documents"
SILVER_HISTORY = "silver_tax_documents_history"
GOLD_CHUNKS = "gold_tax_document_chunks"
INDEX_SUFFIX = "_index"
AGENT_MODEL = "tax_rag_agent"


@dataclass(frozen=True)
class Names:
    catalog: str
    schema: str

    def __post_init__(self) -> None:
        for part in (self.catalog, self.schema):
            if not part or "`" in part or "." in part:
                raise ValueError(f"Invalid catalog/schema name: {part!r}")

    def table(self, name: str) -> str:
        """Three-level name for APIs (vector search, MLflow, Spark table readers)."""
        return f"{self.catalog}.{self.schema}.{name}"

    def sql(self, name: str) -> str:
        """Backtick-quoted three-level name for SQL statements."""
        return f"`{self.catalog}`.`{self.schema}`.`{name}`"

    @property
    def index(self) -> str:
        return self.table(GOLD_CHUNKS + INDEX_SUFFIX)

    @property
    def agent_model(self) -> str:
        return self.table(AGENT_MODEL)

    @property
    def volume_root(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}"
