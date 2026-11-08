"""Contract-faithful mock of the upstream tax-document API (stdlib only).

Used by the integration tests and runnable for local experiments:
    python tests/mock_tax_api.py --port 8765 --token secret
"""

from __future__ import annotations

import argparse
import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

FIXTURES = Path(__file__).parent / "fixtures" / "tax_documents.jsonl"


def _norm(ts: str) -> str:
    """Canonicalise ISO timestamps for lexicographic comparison."""
    from tax_rag.timeutil import fmt_ts, parse_ts

    parsed = parse_ts(ts)
    return fmt_ts(parsed) if parsed else ""


@dataclass
class MockState:
    documents: list[dict]
    token: str | None = None
    fail_first: int = 0  # respond 503 to the first N requests
    rate_limit_first: int = 0  # respond 429 (Retry-After: 0) to the next N requests
    requests: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


def load_documents(path: Path = FIXTURES) -> list[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def make_handler(state: MockState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep test output quiet
            return

        def _send(self, status: int, body: dict, headers: dict | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            url = urlparse(self.path)
            params = {k: v[0] for k, v in parse_qs(url.query).items()}
            with state.lock:
                state.requests.append({"path": url.path, **params})
                if state.fail_first > 0:
                    state.fail_first -= 1
                    return self._send(503, {"error": "unavailable"})
                if state.rate_limit_first > 0:
                    state.rate_limit_first -= 1
                    return self._send(429, {"error": "slow down"}, {"Retry-After": "0"})
            if url.path != "/v1/tax-documents":
                return self._send(404, {"error": "not found"})
            if state.token and self.headers.get("Authorization") != f"Bearer {state.token}":
                return self._send(401, {"error": "unauthorized"})
            lo, hi = _norm(params.get("updated_from", "")), _norm(params.get("updated_to", ""))
            if not lo or not hi:
                return self._send(400, {"error": "updated_from and updated_to are required"})
            page_size = int(params.get("page_size", "100"))
            offset = int(params.get("page_token") or 0)
            matching = sorted(
                (d for d in state.documents if lo <= _norm(d.get("updated_at", "")) < hi),
                key=lambda d: (_norm(d.get("updated_at", "")), str(d.get("document_id"))),
            )
            page = matching[offset : offset + page_size]
            nxt = offset + page_size
            return self._send(
                200, {"items": page, "next_page_token": str(nxt) if nxt < len(matching) else None}
            )

    return Handler


class MockTaxApi:
    def __init__(self, state: MockState, port: int = 0) -> None:
        self.state = state
        self.server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(state))
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __enter__(self) -> MockTaxApi:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()
    api = MockTaxApi(MockState(load_documents(), token=args.token), port=args.port)
    print(f"Mock tax API on {api.base_url}/v1/tax-documents")
    api.server.serve_forever()
