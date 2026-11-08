import pytest

from tax_rag.transform.chunking import ChunkConfig, build_chunks, chunk_text, metadata_header
from tax_rag.transform.cleaning import (
    html_to_text,
    normalize_document_type,
    normalize_jurisdiction,
    normalize_tax_type,
    normalize_whitespace,
    to_plain_text,
)


def test_html_to_text_keeps_structure_and_drops_noise():
    markup = (
        "<h2>Section 12</h2><p>Rate is 18&nbsp;%.</p><script>alert(1)</script>"
        "<!-- note --><ul><li>a</li><li>b &amp; c</li></ul>"
    )
    text = normalize_whitespace(html_to_text(markup))
    assert text.startswith("## Section 12")
    assert "Rate is 18 %." in text and "b & c" in text
    assert "alert" not in text and "note" not in text


def test_to_plain_text_by_format():
    assert to_plain_text("<p>x</p>", "HTML") == "x"
    assert to_plain_text("  a  \n\n\n b ", None) == "a\n\nb"
    assert to_plain_text(None, "html") is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("us", "US"),
        (" us-ca ", "US-CA"),
        ("de_by", "DE-BY"),
        ("USA", None),
        ("", None),
        (None, None),
    ],
)
def test_normalize_jurisdiction(raw, expected):
    assert normalize_jurisdiction(raw) == expected


def test_normalize_types():
    assert normalize_tax_type("VAT") == "vat_gst"
    assert normalize_tax_type("Personal Income") == "income"
    assert normalize_tax_type("lottery") == "other"
    assert normalize_document_type("Revenue Ruling") == "ruling"
    assert normalize_document_type("circular") == "guidance"
    assert normalize_document_type(None) == "other"


def test_chunk_config_validation():
    with pytest.raises(ValueError):
        ChunkConfig(max_chars=100, overlap_chars=100)


def test_chunking_respects_size_and_overlaps():
    paragraphs = [f"Paragraph {i}. " + "The rate applies to supplies. " * 8 for i in range(12)]
    cfg = ChunkConfig(max_chars=600, overlap_chars=80)
    chunks = chunk_text("\n\n".join(paragraphs), cfg)
    assert len(chunks) > 3
    assert all(len(c.text) <= cfg.max_chars for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    # Overlap: each chunk starts with text from the end of the previous one.
    assert chunks[1].text.split("\n\n")[0] in chunks[0].text


def test_chunking_hard_splits_giant_sentences():
    chunks = chunk_text("x" * 2500, ChunkConfig(max_chars=1000, overlap_chars=0))
    assert [len(c.text) for c in chunks] == [1000, 1000, 500]


def test_chunking_empty_inputs():
    assert chunk_text("") == [] and chunk_text(None) == [] and chunk_text("  \n ") == []


def test_build_chunks_is_deterministic_and_carries_metadata():
    args = (
        "doc-1",
        "VAT Act s12",
        "The standard rate is 18 percent.",
        "ZZ",
        "vat_gst",
        "statute",
        2025,
    )
    rows = build_chunks(*args)
    assert rows == build_chunks(*args)
    assert rows[0]["chunk_id"] == "doc-1::0000"
    assert rows[0]["chunk_text"].startswith("VAT Act s12\n[ZZ | vat_gst | statute | tax year 2025]")
    assert len(rows[0]["chunk_hash"]) == 64


def test_metadata_header_without_facets():
    assert metadata_header(None, None, None, None, None) == "Untitled document"
