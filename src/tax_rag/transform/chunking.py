"""Structure-aware chunking for tax documents.

Strategy: split on headings/paragraphs, then sentences for oversized paragraphs, and pack
greedily up to `max_chars` with a sentence-aligned overlap. Each chunk is prefixed with a
short metadata header so the embedding carries jurisdiction / tax type / year context —
without it, "the rate is 18%" chunks from different countries look identical to the
retriever.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?;:])\s+(?=[\"'(\[A-Z0-9§])")


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 2000
    overlap_chars: int = 200
    min_chars: int = 40

    def __post_init__(self) -> None:
        if self.max_chars <= 0:
            raise ValueError("max_chars must be positive")
        if not 0 <= self.overlap_chars < self.max_chars:
            raise ValueError("overlap_chars must be in [0, max_chars)")


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str


def _units(text: str, max_chars: int) -> list[str]:
    """Paragraphs, with oversized ones broken into sentences, and hard-split as a last resort."""
    units: list[str] = []
    for para in (p.strip() for p in _PARAGRAPH.split(text)):
        if not para:
            continue
        if len(para) <= max_chars:
            units.append(para)
            continue
        for sentence in (s.strip() for s in _SENTENCE.split(para)):
            while len(sentence) > max_chars:
                units.append(sentence[:max_chars])
                sentence = sentence[max_chars:]
            if sentence:
                units.append(sentence)
    return units


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Last ~overlap_chars of text, starting at a sentence (else word) boundary."""
    if overlap_chars == 0:
        return ""
    if len(text) <= overlap_chars:
        return text
    tail = text[-overlap_chars:]
    sentence = tail.find(". ")
    if sentence != -1 and sentence + 2 < len(tail):
        return tail[sentence + 2 :].strip()
    space = tail.find(" ")
    return tail[space + 1 :].strip() if space != -1 else tail.strip()


def chunk_text(text: str | None, config: ChunkConfig | None = None) -> list[Chunk]:
    cfg = config or ChunkConfig()
    if not text or not text.strip():
        return []
    chunks: list[str] = []
    current = ""
    for unit in _units(text, cfg.max_chars):
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= cfg.max_chars:
            current = candidate
            continue
        chunks.append(current)
        tail = _overlap_tail(current, cfg.overlap_chars)
        current = (
            f"{tail}\n\n{unit}" if tail and len(tail) + len(unit) + 2 <= cfg.max_chars else unit
        )
    if current:
        chunks.append(current)
    # Fold a tiny trailing chunk into its predecessor when it fits.
    if len(chunks) > 1 and len(chunks[-1]) < cfg.min_chars:
        merged = f"{chunks[-2]}\n\n{chunks[-1]}"
        if len(merged) <= cfg.max_chars:
            chunks[-2:] = [merged]
    return [Chunk(index=i, text=c) for i, c in enumerate(chunks)]


def metadata_header(
    title: str | None,
    jurisdiction: str | None,
    tax_type: str | None,
    document_type: str | None,
    tax_year: int | None,
) -> str:
    facets = [f for f in (jurisdiction, tax_type, document_type) if f]
    if tax_year:
        facets.append(f"tax year {tax_year}")
    header = title.strip() if title else "Untitled document"
    return f"{header}\n[{' | '.join(facets)}]" if facets else header


def build_chunks(
    document_id: str,
    title: str | None,
    body_text: str | None,
    jurisdiction: str | None = None,
    tax_type: str | None = None,
    document_type: str | None = None,
    tax_year: int | None = None,
    config: ChunkConfig | None = None,
) -> list[dict]:
    """Rows for the gold chunk table (one per chunk), deterministic for a given input."""
    header = metadata_header(title, jurisdiction, tax_type, document_type, tax_year)
    rows = []
    for chunk in chunk_text(body_text, config):
        text = f"{header}\n\n{chunk.text}"
        rows.append(
            {
                "chunk_id": f"{document_id}::{chunk.index:04d}",
                "chunk_index": chunk.index,
                "chunk_text": text,
                "chunk_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return rows
