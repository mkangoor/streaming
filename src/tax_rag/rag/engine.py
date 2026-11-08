"""Retrieval-augmented answering, independent of Databricks/MLflow so it is unit-testable.

retrieve -> filter by score / effective date -> refuse if nothing relevant -> prompt the
LLM with numbered sources -> map the [n] markers in the answer back to citations.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol

from tax_rag.rag.prompts import NO_CONTEXT_ANSWER, PROMPT_VERSION, REFUSAL_TOKEN, SYSTEM_PROMPT

try:  # tracing is optional: present on Databricks, absent in plain unit tests
    import mlflow

    def _trace(**kwargs):
        return mlflow.trace(**kwargs)

except ImportError:  # pragma: no cover - exercised only without mlflow

    def _trace(**_kwargs):
        return lambda fn: fn


_CITATION = re.compile(r"\[(\d{1,3})\]")


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    document_id: str
    title: str
    text: str
    source_url: str | None = None
    jurisdiction: str | None = None
    tax_type: str | None = None
    document_type: str | None = None
    tax_year: int | None = None
    effective_date: str | None = None
    expiration_date: str | None = None
    score: float = 0.0

    def in_force_on(self, day: date) -> bool:
        iso = day.isoformat()
        if self.effective_date and self.effective_date > iso:
            return False
        return not (self.expiration_date and self.expiration_date < iso)


@dataclass(frozen=True)
class Citation:
    index: int
    chunk_id: str
    document_id: str
    title: str
    source_url: str | None


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    refused: bool = False
    prompt_version: str = PROMPT_VERSION

    @property
    def grounded(self) -> bool:
        return self.refused or bool(self.citations)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citations": [asdict(c) for c in self.citations],
            "retrieved_chunk_ids": [c.chunk_id for c in self.retrieved],
            "refused": self.refused,
            "grounded": self.grounded,
            "prompt_version": self.prompt_version,
        }


class Retriever(Protocol):
    def retrieve(
        self, query: str, k: int, filters: dict[str, Any] | None
    ) -> list[RetrievedChunk]: ...


class ChatModel(Protocol):
    def complete(self, messages: list[dict[str, str]]) -> str: ...


def message_text(content: Any) -> str:
    """Flatten chat/Responses-style content (str or list of parts) into text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return str(content)


def retrieval_query(messages: Sequence[dict[str, Any]], short_words: int = 5) -> str:
    """Last user turn; short follow-ups ("and for 2024?") get the previous user turn prepended."""
    user_turns = [message_text(m.get("content")) for m in messages if m.get("role") == "user"]
    user_turns = [t.strip() for t in user_turns if t and t.strip()]
    if not user_turns:
        raise ValueError("Request contains no user message")
    question = user_turns[-1]
    if len(question.split()) < short_words and len(user_turns) > 1:
        return f"{user_turns[-2]}\n{question}"
    return question


def format_sources(chunks: Sequence[RetrievedChunk], max_chars: int) -> str:
    blocks, used = [], 0
    for i, c in enumerate(chunks, start=1):
        facets = [c.document_type, c.jurisdiction, c.tax_type]
        if c.tax_year:
            facets.append(f"tax year {c.tax_year}")
        window = f"in force {c.effective_date or '?'} to {c.expiration_date or 'open-ended'}"
        header = f"[{i}] {c.title} ({', '.join(f for f in facets if f)}; {window})"
        url = f"\nURL: {c.source_url}" if c.source_url else ""
        block = f"{header}{url}\n{c.text}"
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block)
    return "\n\n---\n\n".join(blocks)


def extract_citations(answer: str, chunks: Sequence[RetrievedChunk]) -> list[Citation]:
    seen: dict[int, Citation] = {}
    for match in _CITATION.finditer(answer):
        idx = int(match.group(1))
        if 1 <= idx <= len(chunks) and idx not in seen:
            c = chunks[idx - 1]
            seen[idx] = Citation(idx, c.chunk_id, c.document_id, c.title, c.source_url)
    return [seen[i] for i in sorted(seen)]


class TaxRagEngine:
    def __init__(
        self,
        retriever: Retriever,
        llm: ChatModel,
        *,
        top_k: int = 6,
        min_score: float = 0.0,
        max_context_chars: int = 16_000,
        max_history_messages: int = 6,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.max_history_messages = max_history_messages

    @_trace(span_type="RETRIEVER", name="retrieve_tax_sources")
    def retrieve_documents(
        self, query: str, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """MLflow retriever-span shaped output (page_content/metadata), used by eval judges."""
        chunks = self.retriever.retrieve(query, self.top_k, filters)
        return [
            {
                "id": c.chunk_id,
                "page_content": c.text,
                "metadata": {**asdict(c), "doc_uri": c.source_url},
            }
            for c in chunks
        ]

    def _relevant(self, docs: list[dict[str, Any]], as_of: date | None) -> list[RetrievedChunk]:
        chunks = [
            RetrievedChunk(**{k: v for k, v in d["metadata"].items() if k != "doc_uri"})
            for d in docs
        ]
        chunks = [c for c in chunks if c.score >= self.min_score]
        if as_of is not None:
            chunks = [c for c in chunks if c.in_force_on(as_of)]
        return chunks

    def _history(self, messages: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
        prior = [
            {"role": m["role"], "content": message_text(m.get("content"))}
            for m in messages[:-1]
            if m.get("role") in ("user", "assistant")
        ]
        return prior[-self.max_history_messages :] if self.max_history_messages else []

    @_trace(span_type="CHAIN", name="tax_rag_answer")
    def answer(
        self,
        messages: Sequence[dict[str, Any]],
        *,
        filters: dict[str, Any] | None = None,
        as_of: date | None = None,
    ) -> RagAnswer:
        query = retrieval_query(messages)
        chunks = self._relevant(self.retrieve_documents(query, filters), as_of)
        if not chunks:
            return RagAnswer(answer=NO_CONTEXT_ANSWER, refused=True)

        question = message_text(messages[-1].get("content")).strip() or query
        date_note = f"\nAnswer as of: {as_of.isoformat()}" if as_of else ""
        prompt = (
            f"Sources:\n\n{format_sources(chunks, self.max_context_chars)}\n\n"
            f"Question: {question}{date_note}"
        )
        llm_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *self._history(messages),
            {"role": "user", "content": prompt},
        ]
        text = self.llm.complete(llm_messages).strip()
        if text.startswith(REFUSAL_TOKEN):
            detail = text[len(REFUSAL_TOKEN) :].strip()
            answer = f"{NO_CONTEXT_ANSWER} {detail}".strip()
            return RagAnswer(answer=answer, retrieved=chunks, refused=True)
        return RagAnswer(answer=text, citations=extract_citations(text, chunks), retrieved=chunks)
