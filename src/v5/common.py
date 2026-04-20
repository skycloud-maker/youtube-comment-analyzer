"""Shared helpers for V5 topic-first pipeline."""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

import pandas as pd


CONTRACT_VERSION = "v5.2.0"

VALID_JUDGMENT_AXES = {
    "price_value_fit",
    "installation_space_fit",
    "maintenance_burden",
    "automation_convenience",
    "durability_lifespan",
    "brand_trust",
    "usage_pattern_fit",
    "comparison_substitute_judgment",
    "design_interior_fit",
    "service_after_sales",
    "purchase_strategy_contract",
    "repurchase_recommendation",
}

_LIST_TOKEN_SPLIT_RE = re.compile(r"[|,;/]+")
_WRAPPER_TRIM_RE = re.compile(r"^[\[\(\{'\"]+|[\]\)\}'\"]+$")


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _clean_token(token: Any) -> str:
    cleaned = safe_text(token)
    if not cleaned:
        return ""
    cleaned = _WRAPPER_TRIM_RE.sub("", cleaned).strip()
    return cleaned


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clean_token(item) for item in value if _clean_token(item)]
    if isinstance(value, tuple):
        return [_clean_token(item) for item in value if _clean_token(item)]

    raw = safe_text(value)
    if not raw:
        return []

    # 1) Strict JSON list
    try:
        parsed_json = json.loads(raw)
        if isinstance(parsed_json, list):
            return [_clean_token(item) for item in parsed_json if _clean_token(item)]
    except Exception:
        pass

    # 2) Python-like list string, including single quotes
    try:
        parsed_python = ast.literal_eval(raw)
        if isinstance(parsed_python, (list, tuple)):
            return [_clean_token(item) for item in parsed_python if _clean_token(item)]
    except Exception:
        pass

    # 3) Relaxed bracket/quote fallback
    relaxed = raw.strip()
    relaxed = relaxed.strip("[](){}")
    if not relaxed:
        return []

    primary_tokens = [tok.strip() for tok in _LIST_TOKEN_SPLIT_RE.split(relaxed) if tok.strip()]
    if len(primary_tokens) <= 1 and " " in relaxed and "_" not in relaxed:
        primary_tokens = [tok for tok in relaxed.split() if tok]

    cleaned = [_clean_token(token) for token in primary_tokens if _clean_token(token)]
    return cleaned


def normalize_judgment_axes(
    raw_axes: Any,
    *,
    fallback: str = "usage_pattern_fit",
    logger: logging.Logger | None = None,
    source_label: str = "",
) -> list[str]:
    parsed = parse_list(raw_axes)
    normalized: list[str] = []
    invalid_values: list[str] = []
    for item in parsed:
        axis = safe_text(item).lower()
        if axis in VALID_JUDGMENT_AXES:
            if axis not in normalized:
                normalized.append(axis)
        elif axis:
            invalid_values.append(axis)

    # Some legacy rows store concatenated axis strings without separators.
    extracted: list[str] = []
    if not normalized:
        raw_text = safe_text(raw_axes).lower()
        for axis in sorted(VALID_JUDGMENT_AXES, key=len, reverse=True):
            if axis in raw_text and axis not in extracted:
                extracted.append(axis)
        if extracted:
            normalized.extend(extracted)

    if invalid_values and logger is not None and not extracted:
        logger.warning(
            "[v5][axis-parse] invalid axis values=%s source=%s raw=%s",
            invalid_values,
            source_label,
            safe_text(raw_axes),
        )

    if normalized:
        return normalized
    return [fallback]


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
