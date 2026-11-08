"""UTC timestamp helpers shared by the API client and the streaming source."""

from __future__ import annotations

from datetime import datetime, timezone


def parse_ts(value: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp into an aware UTC datetime. Naive input is assumed UTC."""
    if value is None or value == "":
        return None
    text = value.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def require_ts(value: str, field: str) -> datetime:
    parsed = parse_ts(value)
    if parsed is None:
        raise ValueError(f"{field} must be an ISO-8601 timestamp, got {value!r}")
    return parsed


def fmt_ts(value: datetime) -> str:
    """Canonical, sortable UTC representation used in offsets and API params."""
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)
