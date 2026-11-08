from datetime import date

import pytest

from tax_rag.rag.clients import build_filters, parse_query_response
from tax_rag.rag.engine import (
    RetrievedChunk,
    TaxRagEngine,
    extract_citations,
    format_sources,
    message_text,
    retrieval_query,
)
from tax_rag.rag.prompts import NO_CONTEXT_ANSWER, SYSTEM_PROMPT


def chunk(i, **kw):
    base = dict(
        chunk_id=f"d{i}::0000", document_id=f"d{i}", title=f"Doc {i}", text=f"text {i}", score=0.9
    )
    return RetrievedChunk(**{**base, **kw})


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks, self.calls = chunks, []

    def retrieve(self, query, k, filters):
        self.calls.append((query, k, filters))
        return self.chunks[:k]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, messages):
        self.calls.append(messages)
        return self.reply


def user(text):
    return {"role": "user", "content": text}


def test_message_text_variants():
    assert message_text("hi") == "hi"
    assert message_text([{"type": "input_text", "text": "a"}, "b"]) == "a\nb"
    assert message_text(None) == ""


def test_retrieval_query_expands_short_followups():
    msgs = [
        user("What is the VAT registration threshold in Zedland?"),
        {"role": "assistant", "content": "x"},
        user("and the deadline?"),
    ]
    assert (
        retrieval_query(msgs)
        == "What is the VAT registration threshold in Zedland?\nand the deadline?"
    )
    assert retrieval_query([user("What is the standard VAT rate in Zedland today?")]).startswith(
        "What"
    )
    with pytest.raises(ValueError):
        retrieval_query([{"role": "assistant", "content": "x"}])


def test_answer_with_citations_and_history():
    llm = FakeLLM("The standard rate is 18 percent [2]. Food is 7 percent [2][9].")
    engine = TaxRagEngine(FakeRetriever([chunk(1), chunk(2)]), llm, top_k=2)
    result = engine.answer(
        [user("first"), {"role": "assistant", "content": "earlier"}, user("What is the VAT rate?")]
    )
    assert not result.refused and result.grounded
    assert [c.document_id for c in result.citations] == ["d2"]  # [9] is out of range -> ignored
    system, *history, final = llm.calls[0]
    assert system["content"] == SYSTEM_PROMPT
    assert [m["content"] for m in history] == ["first", "earlier"]
    assert "[1] Doc 1" in final["content"] and final["content"].endswith(
        "Question: What is the VAT rate?"
    )
    assert result.to_dict()["retrieved_chunk_ids"] == ["d1::0000", "d2::0000"]


def test_refuses_without_calling_llm_when_nothing_relevant():
    llm = FakeLLM("should not be called")
    engine = TaxRagEngine(FakeRetriever([chunk(1, score=0.1)]), llm, min_score=0.5)
    result = engine.answer([user("What is the VAT rate in France?")])
    assert result.refused and result.answer == NO_CONTEXT_ANSWER and llm.calls == []
    assert result.grounded  # a refusal is a grounded outcome


def test_llm_refusal_token_is_detected():
    engine = TaxRagEngine(FakeRetriever([chunk(1)]), FakeLLM("NOT_IN_SOURCES\nNo French sources."))
    result = engine.answer([user("What is the VAT rate in France?")])
    assert result.refused and result.answer.endswith("No French sources.") and not result.citations


def test_ungrounded_answer_is_flagged():
    engine = TaxRagEngine(FakeRetriever([chunk(1)]), FakeLLM("It is 18 percent."))
    assert not engine.answer([user("What is the VAT rate?")]).grounded


def test_as_of_filters_out_provisions_not_in_force():
    old = chunk(1, effective_date="2023-01-01", expiration_date="2024-12-31")
    new = chunk(2, effective_date="2025-01-01")
    future = chunk(3, effective_date="2027-01-01")
    llm = FakeLLM("20 percent [1].")
    engine = TaxRagEngine(FakeRetriever([old, new, future]), llm)
    result = engine.answer([user("Corporate tax rate?")], as_of=date(2025, 6, 30))
    assert [c.document_id for c in result.retrieved] == ["d2"]
    assert "Answer as of: 2025-06-30" in llm.calls[0][-1]["content"]


def test_format_sources_respects_budget_but_keeps_one():
    chunks = [chunk(i, text="x" * 500) for i in range(5)]
    assert format_sources(chunks, 1200).count("\n[") <= 2
    assert format_sources(chunks[:1], 10).startswith("[1] Doc 0")


def test_extract_citations_dedupes_in_order():
    cites = extract_citations("a [2] b [1] c [2]", [chunk(1), chunk(2)])
    assert [c.index for c in cites] == [1, 2]


def test_parse_query_response():
    payload = {
        "manifest": {
            "columns": [
                {"name": "chunk_id"},
                {"name": "document_id"},
                {"name": "title"},
                {"name": "chunk_text"},
                {"name": "tax_year"},
                {"name": "score"},
            ]
        },
        "result": {"data_array": [["d1::0000", "d1", "Doc", "body", 2025.0, 0.77]]},
    }
    [c] = parse_query_response(payload)
    assert c.chunk_id == "d1::0000" and c.tax_year == 2025 and c.score == pytest.approx(0.77)
    assert parse_query_response({}) == []


def test_build_filters():
    assert build_filters(None) is None
    assert build_filters({"jurisdiction": "ZZ", "tax_year": "2025", "other": 1}) == {
        "jurisdiction": ["ZZ"],
        "tax_year": 2025,
    }
