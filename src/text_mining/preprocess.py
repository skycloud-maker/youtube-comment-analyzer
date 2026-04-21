"""Repo-specific text preprocessing for dashboard-scoped comment data."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

TEXT_COLUMN_PRIORITY: tuple[str, ...] = ("text_display", "raw_text", "text")
TIME_COLUMN_PRIORITY: tuple[str, ...] = ("작성일시", "published_at", "timestamp")

_URL_PATTERN = re.compile(r"(https?://\S+|\bwww\.\S+)", flags=re.IGNORECASE)
_WHITESPACE_PATTERN = re.compile(r"\s+")


def _first_existing_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def select_scope_source(
    data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, str]:
    """
    Select dashboard scope source by strict priority:
    1) analysis_non_trash
    2) filtered_comments
    """
    analysis_non_trash = data.get("analysis_non_trash", pd.DataFrame())
    if analysis_non_trash is not None and not analysis_non_trash.empty:
        return analysis_non_trash.copy(), "analysis_non_trash"

    filtered_comments = data.get("filtered_comments", pd.DataFrame())
    if filtered_comments is not None and not filtered_comments.empty:
        return filtered_comments.copy(), "filtered_comments"

    return pd.DataFrame(), "empty"


def resolve_text_column(df: pd.DataFrame) -> str | None:
    """Resolve text source column by strict priority."""
    if df is None or df.empty:
        return None
    return _first_existing_column(df, TEXT_COLUMN_PRIORITY)


def resolve_time_column(df: pd.DataFrame) -> str | None:
    """Resolve time source column by strict priority."""
    if df is None or df.empty:
        return None
    return _first_existing_column(df, TIME_COLUMN_PRIORITY)


def normalize_text(text: Any) -> str:
    """Normalize noisy text while preserving meaningful tokens."""
    value = "" if text is None else str(text)
    value = _URL_PATTERN.sub(" ", value)
    value = _WHITESPACE_PATTERN.sub(" ", value).strip()
    return value


def build_base_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build text-mining base frame from dashboard-scoped comment data.

    Output columns:
    - comment_id
    - video_id
    - text_raw
    - text_clean
    - created_at
    - product
    - brand
    - region
    """
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "comment_id",
                "video_id",
                "text_raw",
                "text_clean",
                "created_at",
                "product",
                "brand",
                "region",
            ]
        )

    text_col = resolve_text_column(df)
    if text_col is None:
        raise ValueError(
            "No text column found. Expected one of: "
            + ", ".join(TEXT_COLUMN_PRIORITY)
        )

    time_col = resolve_time_column(df)
    working = df.copy()

    comment_ids = (
        working["comment_id"].astype(str)
        if "comment_id" in working.columns
        else pd.Series([str(idx) for idx in working.index], index=working.index, dtype="string")
    )
    video_ids = (
        working["video_id"].astype(str)
        if "video_id" in working.columns
        else pd.Series(["" for _ in working.index], index=working.index, dtype="string")
    )
    text_raw = working[text_col].fillna("").astype(str)
    created_at = (
        pd.to_datetime(working[time_col], errors="coerce", utc=True)
        if time_col is not None
        else pd.Series(pd.NaT, index=working.index)
    )

    def _pick_col(candidates: tuple[str, ...]) -> pd.Series:
        col = _first_existing_column(working, candidates)
        if col is None:
            return pd.Series(["" for _ in working.index], index=working.index, dtype="string")
        return working[col].fillna("").astype(str)

    out = pd.DataFrame(
        {
            "comment_id": comment_ids,
            "video_id": video_ids,
            "text_raw": text_raw,
            "text_clean": text_raw.map(normalize_text),
            "created_at": created_at,
            "product": _pick_col(("product", "제품")),
            "brand": _pick_col(("brand", "브랜드")),
            "region": _pick_col(("region", "국가")),
        }
    )
    return out
