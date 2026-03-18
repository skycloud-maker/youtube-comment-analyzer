"""Date utilities."""

from __future__ import annotations

from datetime import datetime, timezone



def now_utc() -> datetime:
    return datetime.now(timezone.utc)



def safe_parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.replace("Z", "+00:00")
    return datetime.fromisoformat(value)
