"""Ingest V4 outputs and map to V5 comment context schema."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .common import (
    CONTRACT_VERSION,
    bool_series,
    ensure_required_columns,
    normalize_judgment_axes,
    parse_list,
    safe_text,
)

logger = logging.getLogger(__name__)


INGEST_REQUIRED_COLUMNS = [
    "contract_version",
    "comment_id",
    "video_id",
    "raw_text",
    "language",
    "mixed_flag",
    "hygiene_class",
    "analysis_included",
]


def _infer_language(text: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    if any("\uac00" <= ch <= "\ud7a3" for ch in text):
        return "ko"
    if text:
        return "unknown"
    return "unknown"


def _to_iso_timestamp(value: Any) -> str:
    raw = safe_text(value)
    if not raw:
        return ""
    try:
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            return ""
        return parsed.isoformat()
    except Exception:
        return ""


def validate_ingest_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, INGEST_REQUIRED_COLUMNS, "ingest_context")


def build_v5_comment_context(
    *,
    analysis_non_trash: pd.DataFrame,
    comment_truth: pd.DataFrame,
    video_metadata: pd.DataFrame,
    run_id: str = "",
) -> pd.DataFrame:
    """Build V5 ingest context from V4 outputs."""
    if analysis_non_trash is None or analysis_non_trash.empty:
        empty = pd.DataFrame(columns=INGEST_REQUIRED_COLUMNS)
        validate_ingest_schema(empty)
        return empty

    base = analysis_non_trash.copy()
    ensure_required_columns(base, ["comment_id", "video_id"], "ingest_context_input_analysis")
    analysis_included = bool_series(base, "analysis_included", True)
    hygiene_class = base.get("hygiene_class", pd.Series("strategic", index=base.index)).fillna("strategic").astype(str).str.lower()
    base = base[analysis_included & hygiene_class.ne("trash")].copy()

    truth = comment_truth.copy() if comment_truth is not None else pd.DataFrame()
    if not truth.empty:
        keep_truth = [
            col
            for col in ["comment_id", "sentiment_final", "mixed_flag", "journey_stage", "judgment_axes", "analysis_included", "hygiene_class"]
            if col in truth.columns
        ]
        if keep_truth:
            base = base.drop(columns=[col for col in keep_truth if col != "comment_id"], errors="ignore")
            base = base.merge(truth[keep_truth], on="comment_id", how="left")

    videos = video_metadata.copy() if video_metadata is not None else pd.DataFrame()
    if not videos.empty and "video_id" in videos.columns:
        keep_video = [col for col in ["video_id", "title", "product", "brand", "region", "published_at"] if col in videos.columns]
        base = base.merge(videos[keep_video].drop_duplicates("video_id"), on="video_id", how="left", suffixes=("", "_video"))

    raw_text = base.get("text_display", base.get("cleaned_text", base.get("text", pd.Series("", index=base.index)))).fillna("").astype(str)
    language_fallback = base.get("language", base.get("nlp_language", pd.Series("", index=base.index))).fillna("").astype(str)

    output = pd.DataFrame(
        {
            "contract_version": CONTRACT_VERSION,
            "comment_id": base["comment_id"].astype(str),
            "video_id": base["video_id"].astype(str),
            "raw_text": raw_text,
            "language": [_infer_language(text, fallback) for text, fallback in zip(raw_text.tolist(), language_fallback.tolist())],
            "timestamp": base.apply(
                lambda row: _to_iso_timestamp(
                    row.get("작성일시")
                    or row.get("published_at")
                    or row.get("comment_published_at")
                    or row.get("timestamp")
                ),
                axis=1,
            ),
            "sentiment_seed": base.get("sentiment_final", base.get("sentiment_label", pd.Series("", index=base.index))).fillna("").astype(str),
            "mixed_flag": bool_series(base, "mixed_flag", False),
            "journey_stage_seed": base.get("journey_stage", pd.Series("", index=base.index)).fillna("").astype(str),
            "judgment_axes_seed": base.get(
                "judgment_axes",
                pd.Series([[] for _ in range(len(base))], index=base.index),
            ).map(
                lambda value: normalize_judgment_axes(
                    parse_list(value),
                    logger=logger,
                    source_label="ingest_context",
                )
            ),
            "product_raw": base.get("product", pd.Series("", index=base.index)).fillna("").astype(str),
            "brand_raw": base.get("brand", base.get("brand_mentioned", pd.Series("", index=base.index))).fillna("").astype(str),
            "region": base.get("region", pd.Series("", index=base.index)).fillna("").astype(str),
            "like_count": pd.to_numeric(base.get("like_count", pd.Series(0, index=base.index)), errors="coerce").fillna(0).astype(int),
            "reply_count": pd.to_numeric(base.get("reply_count", pd.Series(0, index=base.index)), errors="coerce").fillna(0).astype(int),
            "hygiene_class": base.get("hygiene_class", pd.Series("strategic", index=base.index)).fillna("strategic").astype(str).str.lower(),
            "analysis_included": bool_series(base, "analysis_included", True),
            "run_id": safe_text(run_id),
        }
    )

    validate_ingest_schema(output)
    logger.info("[v5][ingest] rows=%s", len(output))
    return output
