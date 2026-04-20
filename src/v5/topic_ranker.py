"""Rank topics using fixed V5.2 scoring formulas."""

from __future__ import annotations

import json
import logging
import math
import re

import pandas as pd

from .common import clamp, ensure_required_columns, safe_text

logger = logging.getLogger(__name__)

_COMPARISON_MARKER_RE = re.compile(
    r"(?i)\b(?:compare|comparison|vs|versus|instead|rather than|alternative|switch)\b|비교|대체|대신"
)


TOPIC_RANK_REQUIRED_COLUMNS = [
    "topic_id",
    "frequency_count",
    "coverage_pct",
    "polarity_distribution",
    "negative_or_mixed_weight",
    "importance_signal",
    "rank_score",
    "insufficient_evidence",
]


def validate_topic_rank_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, TOPIC_RANK_REQUIRED_COLUMNS, "topic_ranker")


def rank_topics(topics: pd.DataFrame, signal_units: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(topics, ["topic_id", "signal_count", "comment_count"], "topic_ranker_input_topics")
    ensure_required_columns(signal_units, ["topic_id", "comment_id", "polarity"], "topic_ranker_input_signals")

    if topics.empty:
        empty = pd.DataFrame(columns=TOPIC_RANK_REQUIRED_COLUMNS)
        validate_topic_rank_schema(empty)
        return empty

    scope_unique_comments = max(int(signal_units["comment_id"].astype(str).nunique()), 1)
    rows: list[dict] = []

    for _, topic_row in topics.iterrows():
        topic_id = safe_text(topic_row.get("topic_id"))
        cluster = signal_units[signal_units["topic_id"].astype(str) == topic_id].copy()

        frequency_count = int(pd.to_numeric(topic_row.get("signal_count", len(cluster)), errors="coerce") or len(cluster))
        topic_unique_comment_count = int(
            pd.to_numeric(topic_row.get("comment_count", cluster["comment_id"].astype(str).nunique()), errors="coerce")
            or cluster["comment_id"].astype(str).nunique()
        )
        coverage_pct = (topic_unique_comment_count / scope_unique_comments) * 100.0
        judgment_axis = safe_text(topic_row.get("judgment_axis"))

        polarity_counts = (
            cluster.get("polarity", pd.Series("neutral", index=cluster.index))
            .fillna("neutral")
            .astype(str)
            .str.lower()
            .value_counts()
            .to_dict()
        )
        total_polarity = max(sum(polarity_counts.values()), 1)
        positive_ratio = polarity_counts.get("positive", 0) / total_polarity
        negative_ratio = polarity_counts.get("negative", 0) / total_polarity
        mixed_ratio = polarity_counts.get("mixed", 0) / total_polarity
        neutral_ratio = polarity_counts.get("neutral", 0) / total_polarity

        negative_or_mixed_weight = negative_ratio + (0.7 * mixed_ratio)

        coverage_norm = min(coverage_pct / 100.0, 1.0)
        repetition_norm = min(frequency_count / 50.0, 1.0)
        severity = negative_ratio + (0.7 * mixed_ratio) + (0.2 * neutral_ratio)

        like_sum = float(pd.to_numeric(cluster.get("like_count", pd.Series(0, index=cluster.index)), errors="coerce").fillna(0).sum())
        reply_sum = float(pd.to_numeric(cluster.get("reply_count", pd.Series(0, index=cluster.index)), errors="coerce").fillna(0).sum())
        impact_raw = like_sum + (1.5 * reply_sum)
        impact_norm = min(math.log1p(max(impact_raw, 0.0)) / 6.0, 1.0)

        importance_signal = (0.40 * coverage_norm) + (0.25 * repetition_norm) + (0.25 * severity) + (0.10 * impact_norm)
        insufficient_evidence = (frequency_count < 3) or (topic_unique_comment_count < 2) or (coverage_pct < 1.0)

        # Trust gating for top-rank safety.
        cohesion_ratio = topic_unique_comment_count / max(frequency_count, 1)
        weak_cohesion = cohesion_ratio < 0.56
        signal_density = frequency_count / max(topic_unique_comment_count, 1)
        top_quality_pass = True
        exclusion_reasons: list[str] = []

        penalty = 1.0
        if coverage_pct < 3.0:
            penalty *= 0.80
        if coverage_pct < 1.0:
            penalty *= 0.65
        if insufficient_evidence:
            penalty *= 0.70
            top_quality_pass = False
            exclusion_reasons.append("insufficient_evidence")
        if weak_cohesion:
            penalty *= 0.82
            top_quality_pass = False
            exclusion_reasons.append("weak_cohesion")
        # Penalize narrow topics that are both low-coverage and low-repetition.
        if coverage_pct < 2.0 and frequency_count < 20:
            penalty *= 0.72
            top_quality_pass = False
            exclusion_reasons.append("micro_topic")
        # Penalize over-split clusters with too many signals per comment.
        if signal_density > 3.8:
            penalty *= 0.86
            top_quality_pass = False
            exclusion_reasons.append("over_split")
        # Mild boost for truly repeated topics with broad coverage.
        if coverage_pct >= 6.0 and topic_unique_comment_count >= 25:
            penalty *= 1.06

        marker_ratio = 0.0
        if judgment_axis == "comparison_substitute_judgment":
            text_series = cluster.get("signal_text", pd.Series("", index=cluster.index)).fillna("").astype(str)
            marker_hits = int(text_series.map(lambda txt: bool(_COMPARISON_MARKER_RE.search(txt))).sum())
            marker_ratio = marker_hits / max(len(text_series), 1)

            # Comparison topics must have strong explicit evidence to enter top ranks.
            if marker_ratio < 0.35:
                penalty *= 0.56
                top_quality_pass = False
                exclusion_reasons.append("comparison_marker_weak")
            if coverage_pct < 2.8 or topic_unique_comment_count < 15:
                penalty *= 0.60
                top_quality_pass = False
                exclusion_reasons.append("comparison_support_weak")
            if negative_or_mixed_weight < 0.35:
                penalty *= 0.72
                top_quality_pass = False
                exclusion_reasons.append("comparison_signal_weak")

        rank_score_base = importance_signal * (1.0 + (0.35 * negative_or_mixed_weight)) * 100.0
        rank_score = rank_score_base * penalty

        rows.append(
            {
                "topic_id": topic_id,
                "frequency_count": int(frequency_count),
                "coverage_pct": round(float(coverage_pct), 4),
                "polarity_distribution": json.dumps(
                    {
                        "positive": round(positive_ratio, 4),
                        "negative": round(negative_ratio, 4),
                        "mixed": round(mixed_ratio, 4),
                        "neutral": round(neutral_ratio, 4),
                    },
                    ensure_ascii=False,
                ),
                "negative_or_mixed_weight": round(float(negative_or_mixed_weight), 6),
                "importance_signal": round(float(clamp(importance_signal, 0.0, 1.0)), 6),
                "rank_score": round(float(max(rank_score, 0.0)), 4),
                "insufficient_evidence": bool(insufficient_evidence),
                "trend_score": None,
                "cohesion_ratio": round(float(cohesion_ratio), 6),
                "ranking_penalty": round(float(penalty), 6),
                "signal_density": round(float(signal_density), 6),
                "top_quality_pass": bool(top_quality_pass),
                "top_exclusion_reason": "|".join(sorted(set(exclusion_reasons))),
                "comparison_marker_ratio": round(float(marker_ratio), 6),
            }
        )

    output = pd.DataFrame(rows).sort_values("rank_score", ascending=False).reset_index(drop=True)
    validate_topic_rank_schema(output)
    logger.info("[v5][topic_ranker] ranked_topics=%s", len(output))
    return output
