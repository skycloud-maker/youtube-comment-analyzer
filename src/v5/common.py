"""Shared helpers for V5 topic-first pipeline."""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd


CONTRACT_VERSION = "v5.2.0"


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [safe_text(item) for item in value if safe_text(item)]
    if isinstance(value, tuple):
        return [safe_text(item) for item in value if safe_text(item)]
    raw = safe_text(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [safe_text(item) for item in parsed if safe_text(item)]
        except Exception:
            pass
    return [safe_text(token) for token in re.split(r"[|,/;\s]+", raw) if safe_text(token)]


def to_json_list(items: list[str]) -> str:
    return json.dumps([safe_text(item) for item in items if safe_text(item)], ensure_ascii=False)


def ensure_required_columns(frame: pd.DataFrame, required: list[str], stage: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"[{stage}] missing required columns: {missing}")


def bool_series(frame: pd.DataFrame, column: str, default: bool) -> pd.Series:
    if column in frame.columns:
        series = frame[column]
        coerced = series.astype("boolean")
        return coerced.fillna(default).astype(bool)
    return pd.Series(default, index=frame.index)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
