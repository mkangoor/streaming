"""Pure-Python normalisation used by the silver layer (wrapped as Spark UDFs)."""

from __future__ import annotations

import html
import re

DOCUMENT_TYPES = frozenset(
    {
        "statute",
        "regulation",
        "ruling",
        "guidance",
        "publication",
        "form_instructions",
        "faq",
        "case_law",
        "other",
    }
)
TAX_TYPES = frozenset(
    {
        "income",
        "corporate",
        "payroll",
        "sales_use",
        "vat_gst",
        "property",
        "excise",
        "estate_gift",
        "other",
    }
)

_DOCUMENT_TYPE_ALIASES = {
    "law": "statute",
    "act": "statute",
    "legislation": "statute",
    "reg": "regulation",
    "regulations": "regulation",
    "revenue_ruling": "ruling",
    "private_letter_ruling": "ruling",
    "notice": "guidance",
    "circular": "guidance",
    "instructions": "form_instructions",
    "court_decision": "case_law",
    "case": "case_law",
}
_TAX_TYPE_ALIASES = {
    "personal_income": "income",
    "individual_income": "income",
    "corporate_income": "corporate",
    "corporation": "corporate",
    "employment": "payroll",
    "withholding": "payroll",
    "sales": "sales_use",
    "use": "sales_use",
    "vat": "vat_gst",
    "gst": "vat_gst",
    "value_added": "vat_gst",
    "real_estate": "property",
    "estate": "estate_gift",
    "gift": "estate_gift",
    "inheritance": "estate_gift",
}

_BLOCK_TAGS = re.compile(
    r"<\s*/?\s*(p|div|br|li|ul|ol|tr|table|section|article|h[1-6]|blockquote|pre)\b[^>]*>",
    re.IGNORECASE,
)
_HEADING_OPEN = re.compile(r"<\s*h([1-6])\b[^>]*>", re.IGNORECASE)
_DROP_BLOCKS = re.compile(
    r"<\s*(script|style|noscript)\b.*?<\s*/\s*\1\s*>", re.IGNORECASE | re.DOTALL
)
_ANY_TAG = re.compile(r"<[^>]+>")
_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_SPACES = re.compile("[ \t\f\v\u00a0]+")
_BLANKS = re.compile(r"\n\s*\n+")
_JURISDICTION = re.compile(r"^[A-Z]{2}(-[A-Z0-9]{1,3})?$")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def html_to_text(markup: str) -> str:
    """Strip HTML into readable plain text, keeping paragraph and heading structure."""
    text = _COMMENT.sub(" ", markup)
    text = _DROP_BLOCKS.sub(" ", text)
    text = _HEADING_OPEN.sub(lambda m: "\n\n" + "#" * int(m.group(1)) + " ", text)
    text = _BLOCK_TAGS.sub("\n\n", text)
    text = _ANY_TAG.sub(" ", text)
    return html.unescape(text)


def normalize_whitespace(text: str) -> str:
    lines = [_SPACES.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return _BLANKS.sub("\n\n", "\n".join(lines)).strip()


def to_plain_text(body: str | None, body_format: str | None) -> str | None:
    if body is None:
        return None
    fmt = (body_format or "text").strip().lower()
    text = html_to_text(body) if fmt in {"html", "xhtml"} else body
    return normalize_whitespace(text)


def normalize_jurisdiction(value: str | None) -> str | None:
    """ISO 3166-1 alpha-2, optionally with a subdivision (US, US-CA, DE-BY)."""
    if value is None:
        return None
    code = value.strip().upper().replace("_", "-")
    return code if _JURISDICTION.match(code) else None


def normalize_document_type(value: str | None) -> str:
    if not value:
        return "other"
    slug = _slug(value)
    slug = _DOCUMENT_TYPE_ALIASES.get(slug, slug)
    return slug if slug in DOCUMENT_TYPES else "other"


def normalize_tax_type(value: str | None) -> str:
    if not value:
        return "other"
    slug = _slug(value)
    slug = _TAX_TYPE_ALIASES.get(slug, slug)
    return slug if slug in TAX_TYPES else "other"
