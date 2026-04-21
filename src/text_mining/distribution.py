"""Keyword distribution builders for product/video scope."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.text_mining.keyword_extractor import build_comment_keyword_map

_BASE_REQUIRED_COLUMNS: tuple[str, ...] = ("comment_id", "video_id", "product")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _validate_base_df(base_df: pd.DataFrame) -> None:
    if base_df is None or base_df.empty:
        raise ValueError("base_df is empty.")
    missing = [col for col in _BASE_REQUIRED_COLUMNS if col not in base_df.columns]
    if missing:
        raise ValueError(f"base_df missing required columns: {missing}")


def _prepare_comment_scope(base_df: pd.DataFrame) -> pd.DataFrame:
    scope = (
        base_df[["comment_id", "video_id", "product"]]
        .copy()
        .drop_duplicates(subset=["comment_id"], keep="first")
    )
    scope["comment_id"] = scope["comment_id"].map(_safe_text)
    scope["video_id"] = scope["video_id"].map(_safe_text)
    scope["product"] = scope["product"].map(_safe_text)
    return scope


def build_product_keyword_distribution(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build product-level keyword distribution.

    Output columns:
    - product
    - keyword
    - count
    """
    _validate_base_df(base_df)
    comment_keyword_map = build_comment_keyword_map(base_df)
    if comment_keyword_map.empty:
        return pd.DataFrame(columns=["product", "keyword", "count"])

    scope = _prepare_comment_scope(base_df)
    merged = comment_keyword_map.merge(scope[["comment_id", "product"]], on="comment_id", how="left")
    merged["keyword"] = merged["keyword"].map(_safe_text)
    grouped = (
        merged.groupby(["product", "keyword"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped["count"] = grouped["count"].astype(int)
    return grouped.sort_values(["count", "product", "keyword"], ascending=[False, True, True]).reset_index(drop=True)


def build_video_keyword_distribution(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build video-level keyword distribution.

    Output columns:
    - video_id
    - keyword
    - count
    """
    _validate_base_df(base_df)
    comment_keyword_map = build_comment_keyword_map(base_df)
    if comment_keyword_map.empty:
        return pd.DataFrame(columns=["video_id", "keyword", "count"])

    scope = _prepare_comment_scope(base_df)
    merged = comment_keyword_map.merge(scope[["comment_id", "video_id"]], on="comment_id", how="left")
    merged["keyword"] = merged["keyword"].map(_safe_text)
    grouped = (
        merged.groupby(["video_id", "keyword"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    grouped["count"] = grouped["count"].astype(int)
    return grouped.sort_values(["count", "video_id", "keyword"], ascending=[False, True, True]).reset_index(drop=True)

