"""Canonical comment-level truth contract and hygiene enforcement helpers."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd


CANONICAL_COMMENT_TRUTH_COLUMNS = [
    "comment_id",
    "video_id",
    "text",
    "text_display",
    "hygiene_class",
    "analysis_included",
    "sentiment_final",
    "mixed_flag",
    "inquiry_flag",
    "journey_stage",
    "judgment_axes",
    "representative_eligibility",
]

_TRASH_QUALITY_SET = {
    "ad",
    "tagging_only",
    "meaningless",
    "repetitive",
    "generic_engagement",
    "low_context",
}
_ALLOWED_POLARITY = {"positive", "negative", "neutral", "mixed", "trash"}


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and pd.isna(value):
            return default
        return value != 0
    lowered = _coerce_text(value).lower()
    if lowered in {"true", "1", "yes", "y", "t"}:
        return True
    if lowered in {"false", "0", "no", "n", "f", ""}:
        return False
    return default


def _parse_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _has_question_marker(text: str) -> bool:
    lowered = text.lower()
    if "?" in text:
        return True
    inquiry_markers = [
        "어떻게",
        "왜",
        "가능한가",
        "가능할까요",
        "문의",
        "how",
        "why",
        "where",
        "can i",
        "does it",
    ]
    return any(marker in lowered for marker in inquiry_markers)


def _resolve_sentiment_final(row: pd.Series) -> tuple[str, bool]:
    base = _coerce_text(row.get("sentiment_label")).lower()
    nlp_label = _coerce_text(row.get("nlp_label")).lower()
    topic_sentiments = _parse_dict(row.get("nlp_topic_sentiments"))
    topic_values = {str(value).lower() for value in topic_sentiments.values() if str(value).strip()}
    has_topic_conflict = "positive" in topic_values and "negative" in topic_values

    conflict = base in {"positive", "negative"} and nlp_label in {"positive", "negative"} and base != nlp_label

    if nlp_label in _ALLOWED_POLARITY:
        resolved = nlp_label
    elif nlp_label == "undecidable":
        resolved = "neutral"
    elif base in _ALLOWED_POLARITY:
        resolved = base
    else:
        resolved = "neutral"

    mixed_flag = bool(resolved == "mixed" or has_topic_conflict or conflict)
    if mixed_flag and resolved != "trash":
        resolved = "mixed"
    return resolved, mixed_flag


def _resolve_hygiene_class(row: pd.Series, *, inquiry_flag: bool) -> str:
    validity = _coerce_text(row.get("comment_validity")).lower()
    comment_quality = _coerce_text(row.get("comment_quality")).lower()
    nlp_label = _coerce_text(row.get("nlp_label")).lower()
    classification_type = _coerce_text(row.get("classification_type")).lower()

    if validity != "valid":
        return "trash"
    if comment_quality in _TRASH_QUALITY_SET:
        return "trash"
    if nlp_label == "trash":
        return "trash"
    if inquiry_flag:
        return "supporting"
    if classification_type == "informational":
        return "supporting"
    return "strategic"


def build_canonical_comment_truth(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical comment-level truth rows.

    This contract is the single source of truth for downstream analytics consumers.
    """
    if comments_df.empty:
        return pd.DataFrame(columns=CANONICAL_COMMENT_TRUTH_COLUMNS)

    working = comments_df.copy()
    if "comment_id" not in working.columns:
        working["comment_id"] = pd.RangeIndex(start=0, stop=len(working), step=1).astype(str)
    if "video_id" not in working.columns:
        working["video_id"] = pd.NA

    text_series = (
        working.get("cleaned_text", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
        .str.strip()
    )
    text_display_series = (
        working.get("text_display", pd.Series("", index=working.index))
        .fillna(working.get("text_original", pd.Series("", index=working.index)))
        .astype(str)
        .str.strip()
    )
    fallback_text = text_display_series.where(text_display_series.ne(""), text_series)

    sentiment_pairs = working.apply(_resolve_sentiment_final, axis=1)
    sentiment_final = sentiment_pairs.map(lambda item: item[0])
    mixed_flag = sentiment_pairs.map(lambda item: bool(item[1]))

    inquiry_flag = working.apply(
        lambda row: bool(
            _coerce_bool(row.get("nlp_is_inquiry"), default=False)
            or _has_question_marker(_coerce_text(row.get("text_display")))
            or _has_question_marker(_coerce_text(row.get("text_original")))
        ),
        axis=1,
    )

    hygiene_class = working.apply(lambda row: _resolve_hygiene_class(row, inquiry_flag=bool(inquiry_flag.loc[row.name])), axis=1)

    analysis_included = (
        working.get("comment_validity", pd.Series("", index=working.index)).fillna("").astype(str).str.lower().eq("valid")
        & fallback_text.ne("")
        & hygiene_class.ne("trash")
    )

    journey_stage_source = working.get(
        "customer_journey_stage",
        working.get("cej_scene_code", pd.Series(pd.NA, index=working.index)),
    )
    journey_stage = journey_stage_source.where(journey_stage_source.notna(), pd.NA)

    judgment_axes_source = working.get("topic_label", pd.Series(pd.NA, index=working.index))
    judgment_axes = judgment_axes_source.where(judgment_axes_source.notna(), pd.NA)

    representative_eligibility = analysis_included & hygiene_class.eq("strategic")

    truth = pd.DataFrame(
        {
            "comment_id": working["comment_id"],
            "video_id": working["video_id"],
            "text": fallback_text,
            "text_display": text_display_series.where(text_display_series.ne(""), fallback_text),
            "hygiene_class": hygiene_class,
            "analysis_included": analysis_included,
            "sentiment_final": sentiment_final,
            "mixed_flag": mixed_flag,
            "inquiry_flag": inquiry_flag.astype(bool),
            "journey_stage": journey_stage,
            "judgment_axes": judgment_axes,
            "representative_eligibility": representative_eligibility.astype(bool),
        }
    )
    return truth[CANONICAL_COMMENT_TRUTH_COLUMNS].copy()

