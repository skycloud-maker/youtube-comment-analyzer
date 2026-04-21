"""Keyword trend builder for TM pipeline."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.text_mining.keyword_extractor import build_comment_keyword_map

_TIME_COLUMN_PRIORITY: tuple[str, ...] = ("created_at", "작성일시", "published_at", "timestamp")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _resolve_time_column(df: pd.DataFrame) -> str | None:
    for col in _TIME_COLUMN_PRIORITY:
        if col in df.columns:
            return col
    return None


def build_keyword_trend(base_df: pd.DataFrame, bucket_freq: str = "W-MON") -> pd.DataFrame:
    """
    Build keyword trend by time bucket.

    Output columns:
    - keyword
    - time_bucket
    - count
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])
    if "comment_id" not in base_df.columns:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])

    time_col = _resolve_time_column(base_df)
    if time_col is None:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])

    comment_scope = base_df[["comment_id", time_col]].copy()
    comment_scope["comment_id"] = comment_scope["comment_id"].map(_safe_text)
    comment_scope[time_col] = pd.to_datetime(comment_scope[time_col], errors="coerce", utc=True)
    comment_scope = comment_scope.dropna(subset=[time_col])
    if comment_scope.empty:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])

    time_series = comment_scope[time_col]
    if getattr(time_series.dt, "tz", None) is not None:
        time_series = time_series.dt.tz_convert("UTC").dt.tz_localize(None)
    comment_scope["time_bucket"] = time_series.dt.to_period(bucket_freq).dt.start_time

    comment_keyword_map = build_comment_keyword_map(base_df)
    if comment_keyword_map.empty:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])

    merged = comment_keyword_map.merge(comment_scope[["comment_id", "time_bucket"]], on="comment_id", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["keyword", "time_bucket", "count"])

    out = (
        merged.groupby(["keyword", "time_bucket"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    out["count"] = out["count"].astype(int)
    out["time_bucket"] = pd.to_datetime(out["time_bucket"], errors="coerce").dt.tz_localize("UTC")
    return out.sort_values(["time_bucket", "count", "keyword"], ascending=[True, False, True]).reset_index(drop=True)
