"""Select representative comments per topic (evidence samples only)."""

from __future__ import annotations

import logging
import re

import pandas as pd

from .common import clamp, ensure_required_columns, safe_text

logger = logging.getLogger(__name__)


REP_REQUIRED_COLUMNS = [
    "topic_id",
    "representative_comment_ids",
    "alignment_score",
]

_GENERIC_RE = re.compile(r"(?i)\b(?:thanks|thank you|great video|nice video|lol|lmao|first!|subscribe)\b")
_OFFTOPIC_RE = re.compile(r"(?i)\b(?:politics|wife|husband|married|birthday|funny channel)\b")
_TOKEN_RE = re.compile(r"[^A-Za-z0-9\uac00-\ud7a3]+")
_COMPARISON_RE = re.compile(r"(?i)\b(?:compare|comparison|vs|versus|instead|rather than|alternative|switch)\b|비교|대체|대신")

_AXIS_KEYWORDS = {
    "service_after_sales": ["service", "support", "repair", "a/s", "sla", "as"],
    "maintenance_burden": ["filter", "clean", "detergent", "rinse aid", "maintenance", "smell"],
    "automation_convenience": ["button", "ui", "ux", "setup", "easy", "convenient", "workflow"],
    "comparison_substitute_judgment": ["compare", "vs", "versus", "alternative", "instead", "rather than"],
    "price_value_fit": ["price", "cost", "expensive", "cheap", "value"],
}


def _tokenize(text: str) -> set[str]:
    return {token for token in _TOKEN_RE.split(text.lower()) if len(token) >= 2}


def _generic_penalty(signal_text: str) -> float:
    text = safe_text(signal_text)
    if len(text) < 14:
        return 0.40
    lowered = text.lower()
    penalty = 1.0
    if _GENERIC_RE.search(lowered):
        penalty *= 0.55
    if _OFFTOPIC_RE.search(lowered):
        penalty *= 0.45
    if text.count("?") >= 4 and len(re.sub(r"[^A-Za-z0-9\uac00-\ud7a3]", "", text)) < 20:
        penalty *= 0.60
    return penalty


def _axis_keyword_match(signal_text: str, axis: str) -> float:
    text = safe_text(signal_text).lower()
    keywords = _AXIS_KEYWORDS.get(axis, [])
    if not keywords:
        return 1.0
    hit_count = sum(1 for keyword in keywords if keyword in text)
    if hit_count >= 2:
        return 1.0
    if hit_count == 1:
        return 0.75
    return 0.25


def _signature_overlap(signal_text: str, signature: str) -> float:
    text_tokens = _tokenize(signal_text)
    sig_tokens = _tokenize(signature)
    sig_tokens = {
        token
        for token in sig_tokens
        if token
        not in {
            "general",
            "usage",
            "maintenance",
            "exploration",
            "awareness",
            "purchase",
            "negative",
            "neutral",
            "positive",
            "mixed",
        }
    }
    if not sig_tokens:
        return 0.75
    overlap = len(text_tokens & sig_tokens)
    ratio = overlap / max(len(sig_tokens), 1)
    return max(0.2, min(ratio, 1.0))


def _comparison_semantic_score(signal_text: str) -> float:
    text = safe_text(signal_text)
    if not text:
        return 0.0
    if _COMPARISON_RE.search(text):
        return 1.0
    return 0.2


def validate_representative_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, REP_REQUIRED_COLUMNS, "representative_selector")


def select_representatives(topics: pd.DataFrame, signal_units: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(
        topics,
        ["topic_id", "journey_stage", "judgment_axis", "polarity_group", "product_semantic_id", "v5_simple_signature"],
        "representative_selector_input_topics",
    )
    ensure_required_columns(
        signal_units,
        ["topic_id", "comment_id", "signal_type", "weight", "signal_text", "judgment_axis", "polarity", "product_semantic_id"],
        "representative_selector_input_signals",
    )

    rows: list[dict] = []
    for _, topic_row in topics.iterrows():
        topic_id = safe_text(topic_row.get("topic_id"))
        topic_axis = safe_text(topic_row.get("judgment_axis")).lower()
        topic_polarity = safe_text(topic_row.get("polarity_group")).lower()
        topic_product = safe_text(topic_row.get("product_semantic_id"))
        topic_signature = safe_text(topic_row.get("v5_simple_signature"))

        cluster = signal_units[signal_units["topic_id"].astype(str) == topic_id].copy()
        if cluster.empty:
            rows.append(
                {
                    "topic_id": topic_id,
                    "representative_comment_ids": [],
                    "alignment_score": 0.0,
                    "selection_reason": "topic has no signals",
                }
            )
            continue

        cluster["axis_match"] = cluster["judgment_axis"].astype(str).str.lower().eq(topic_axis)
        if topic_polarity == "neutral":
            cluster["polarity_match"] = cluster["polarity"].astype(str).str.lower().isin({"neutral", "mixed"})
        else:
            cluster["polarity_match"] = cluster["polarity"].astype(str).str.lower().eq(topic_polarity)
        cluster["product_match"] = cluster["product_semantic_id"].astype(str).eq(topic_product)
        cluster["generic_penalty"] = cluster["signal_text"].map(_generic_penalty)
        cluster["axis_keyword_ratio"] = cluster["signal_text"].map(lambda text: _axis_keyword_match(text, topic_axis))
        cluster["signature_overlap"] = cluster["signal_text"].map(lambda text: _signature_overlap(text, topic_signature))
        cluster["comparison_semantic_score"] = cluster["signal_text"].map(_comparison_semantic_score)

        aligned = cluster[cluster["axis_match"] & cluster["polarity_match"] & cluster["product_match"]].copy()
        if aligned.empty:
            aligned = cluster[cluster["axis_match"] & cluster["polarity_match"]].copy()
        if aligned.empty:
            aligned = cluster.copy()

        summary = (
            aligned.groupby("comment_id", as_index=False)
            .agg(
                signal_density=("signal_id", "count"),
                avg_weight=("weight", "mean"),
                primary_count=("signal_type", lambda series: int((series.astype(str) == "primary").sum())),
                axis_match_ratio=("axis_match", "mean"),
                polarity_match_ratio=("polarity_match", "mean"),
                product_match_ratio=("product_match", "mean"),
                generic_penalty=("generic_penalty", "mean"),
                axis_keyword_ratio=("axis_keyword_ratio", "mean"),
                signature_overlap=("signature_overlap", "mean"),
                comparison_semantic_score=("comparison_semantic_score", "mean"),
            )
            .copy()
        )

        if topic_axis == "comparison_substitute_judgment":
            summary = summary[
                (summary["axis_keyword_ratio"] >= 0.85)
                & (summary["comparison_semantic_score"] >= 0.6)
                & (summary["signature_overlap"] >= 0.45)
            ].copy()
        else:
            keyword_filtered = summary[summary["axis_keyword_ratio"] >= 0.6].copy()
            if not keyword_filtered.empty:
                summary = keyword_filtered

        if summary.empty:
            rows.append(
                {
                    "topic_id": topic_id,
                    "representative_comment_ids": [],
                    "alignment_score": 0.0,
                    "selection_reason": "no viable representative candidates",
                }
            )
            continue

        max_density = max(int(summary["signal_density"].max()), 1)
        summary["density_norm"] = summary["signal_density"] / max_density
        summary["alignment_component"] = (
            (0.26 * summary["density_norm"])
            + (0.14 * summary["avg_weight"].clip(lower=0.0, upper=1.0))
            + (0.08 * (summary["primary_count"] > 0).astype(float))
            + (0.16 * summary["axis_match_ratio"])
            + (0.10 * summary["polarity_match_ratio"])
            + (0.08 * summary["product_match_ratio"])
            + (0.12 * summary["signature_overlap"])
            + (0.06 * summary["comparison_semantic_score"])
        ) * summary["generic_penalty"] * summary["axis_keyword_ratio"]

        summary = summary.sort_values(
            ["alignment_component", "signal_density", "signature_overlap", "axis_match_ratio", "polarity_match_ratio"],
            ascending=False,
        ).reset_index(drop=True)

        selected = summary.head(3)
        rep_ids = selected["comment_id"].astype(str).tolist()
        alignment_score = float(selected["alignment_component"].mean()) if not selected.empty else 0.0

        # Hard safety floor: if overall topic representative quality is too weak, do not force poor evidence.
        if alignment_score < 0.30:
            rep_ids = []
            selection_reason = "representative quality below hard floor"
        else:
            selection_reason = "axis/polarity/product + signature overlap"

        rows.append(
            {
                "topic_id": topic_id,
                "representative_comment_ids": rep_ids,
                "alignment_score": round(clamp(alignment_score, 0.0, 1.0), 6),
                "selection_reason": selection_reason,
            }
        )

    output = pd.DataFrame(rows)
    validate_representative_schema(output)
    logger.info("[v5][representative_selector] topic_rows=%s", len(output))
    return output

