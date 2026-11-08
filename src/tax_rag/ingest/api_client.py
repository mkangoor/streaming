"""HTTP client for the upstream tax-document API.

Contract (see docs/api_contract.md):
    GET {base_url}/v1/tax-documents
        ?updated_from=<ISO, inclusive>&updated_to=<ISO, exclusive>
        &page_size=<int>&page_token=<opaque>
    -> {"items": [ {document...} ], "next_page_token": "<opaque>" | null}

Retries 429/5xx with exponential backoff and honours Retry-After.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from tax_rag.timeutil import fmt_ts

USER_AGENT = "tax-rag-ingest/0.1"
RETRY_STATUSES = (429, 500, 502, 503, 504)


class TaxApiError(RuntimeError):
    """Non-retryable or retries-exhausted API failure."""


class TaxApiClient:
    def __init__(
        self,
        base_url: str,
        token: str | None = None,
        *,
        documents_path: str = "/v1/tax-documents",
        page_size: int = 500,
        timeout_seconds: float = 30.0,
        max_retries: int = 5,
        backoff_factor: float = 1.0,
        max_pages: int = 10_000,
        session: requests.Session | None = None,
    ) -> None:
        if not base_url:
            raise ValueError("base_url is required")
        if page_size <= 0:
            raise ValueError("page_size must be positive")
        self.url = base_url.rstrip("/") + "/" + documents_path.lstrip("/")
        self.page_size = page_size
        self.timeout = timeout_seconds
        self.max_pages = max_pages
        self._session = session or self._build_session(max_retries, backoff_factor)
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    @staticmethod
    def _build_session(max_retries: int, backoff_factor: float) -> requests.Session:
        retry = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=RETRY_STATUSES,
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
            raise_on_status=False,
        )
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def iter_documents(
        self, updated_from: datetime, updated_to: datetime
    ) -> Iterator[dict[str, Any]]:
        """Yield every document whose updated_at is in [updated_from, updated_to)."""
        if updated_to <= updated_from:
            return
        params: dict[str, Any] = {
            "updated_from": fmt_ts(updated_from),
            "updated_to": fmt_ts(updated_to),
            "page_size": self.page_size,
        }
        seen_tokens: set[str] = set()
        for _ in range(self.max_pages):
            body = self._get(params)
            items = body.get("items")
            if not isinstance(items, list):
                raise TaxApiError(f"Malformed response from {self.url}: 'items' is not a list")
            yield from items
            token = body.get("next_page_token")
            if not token:
                return
            if token in seen_tokens:
                raise TaxApiError(f"API returned a repeated page_token {token!r}; aborting")
            seen_tokens.add(token)
            params["page_token"] = token
        raise TaxApiError(f"Exceeded max_pages={self.max_pages} for window {params}")

    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = self._session.get(self.url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:  # connection errors after retries
            raise TaxApiError(f"Request to {self.url} failed: {exc}") from exc
        if response.status_code >= 400:
            raise TaxApiError(
                f"{self.url} returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TaxApiError(f"{self.url} returned non-JSON body") from exc
        if not isinstance(body, dict):
            raise TaxApiError(f"{self.url} returned a JSON {type(body).__name__}, expected object")
        return body
