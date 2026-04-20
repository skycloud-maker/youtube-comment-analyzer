"""Orchestrator for V5 topic-first pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .common import parse_list
from .export_topic_pack import build_export_topic_pack, validate_export_schema
from .ingest_context import build_v5_comment_context, validate_ingest_schema
from .insight_action_generator import generate_insight_actions, validate_insight_action_schema
from .product_normalizer import normalize_products, validate_product_normalized_schema
from .representative_selector import select_representatives, validate_representative_schema
from .signal_splitter import split_signals, validate_signal_schema
from .topic_builder import attach_topic_ids, build_topics, validate_topic_schema
from .topic_ranker import rank_topics, validate_topic_rank_schema

logger = logging.getLogger(__name__)


def _assert_sanity(comment_context: pd.DataFrame, signal_units: pd.DataFrame, topics: pd.DataFrame) -> None:
    if len(topics) <= 0:
        raise ValueError("[v5][sanity] topic count must be > 0")
    if len(signal_units) < len(comment_context):
        raise ValueError(
            f"[v5][sanity] signal count must be >= comment count (signals={len(signal_units)}, comments={len(comment_context)})"
        )


def _enforce_top_quality_gate(
    topic_ranked: pd.DataFrame,
    topics: pd.DataFrame,
    topic_representatives: pd.DataFrame,
) -> pd.DataFrame:
    if topic_ranked.empty:
        return topic_ranked

    merged = (
        topic_ranked.merge(topics[["topic_id", "judgment_axis"]], on="topic_id", how="left")
        .merge(topic_representatives[["topic_id", "alignment_score", "representative_comment_ids"]], on="topic_id", how="left")
        .copy()
    )

    adjusted_scores: list[float] = []
    quality_flags: list[bool] = []
    reasons: list[str] = []
    rep_counts: list[int] = []

    for _, row in merged.iterrows():
        score = float(pd.to_numeric(row.get("rank_score", 0.0), errors="coerce") or 0.0)
        coverage = float(pd.to_numeric(row.get("coverage_pct", 0.0), errors="coerce") or 0.0)
        alignment = float(pd.to_numeric(row.get("alignment_score", 0.0), errors="coerce") or 0.0)
        axis = str(row.get("judgment_axis", "") or "")
        insufficient = bool(row.get("insufficient_evidence", False))
        rep_ids = row.get("representative_comment_ids", [])
        rep_count = len(parse_list(rep_ids))
        rep_counts.append(rep_count)

        pass_flag = bool(row.get("top_quality_pass", True))
        reason_items = [item for item in str(row.get("top_exclusion_reason", "")).split("|") if item]

        if not pass_flag:
            score *= 0.55

        if rep_count == 0:
            score *= 0.30
            pass_flag = False
            reason_items.append("no_representative")

        if alignment < 0.50:
            score *= 0.50
            pass_flag = False
            reason_items.append("low_alignment")

        if axis == "comparison_substitute_judgment":
            marker_ratio = float(pd.to_numeric(row.get("comparison_marker_ratio", 0.0), errors="coerce") or 0.0)
            if alignment < 0.60:
                score *= 0.55
                pass_flag = False
                reason_items.append("comparison_low_alignment")
            if marker_ratio < 0.45:
                score *= 0.60
                pass_flag = False
                reason_items.append("comparison_marker_insufficient")
            if coverage < 3.0:
                score *= 0.65
                pass_flag = False
                reason_items.append("comparison_low_coverage")

        if insufficient:
            score *= 0.70
            pass_flag = False
            reason_items.append("insufficient_evidence")

        adjusted_scores.append(round(max(score, 0.0), 4))
        quality_flags.append(pass_flag)
        reasons.append("|".join(sorted(set(reason_items))))

    merged["rank_score"] = adjusted_scores
    merged["top_quality_pass"] = quality_flags
    merged["top_exclusion_reason"] = reasons
    merged["representative_count"] = rep_counts

    output = merged.drop(columns=["judgment_axis", "alignment_score", "representative_comment_ids"], errors="ignore")
    output = output.sort_values("rank_score", ascending=False).reset_index(drop=True)
    return output


def run_v5_pipeline(input_bundle: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Execute V5 topic-first pipeline with strict schema validation."""
    analysis_non_trash = input_bundle.get("analysis_non_trash", pd.DataFrame())
    comment_truth = input_bundle.get("comment_truth", pd.DataFrame())
    video_metadata = input_bundle.get("video_metadata", input_bundle.get("videos", pd.DataFrame()))
    run_id = str(input_bundle.get("run_id", "") or "")

    logger.info("[v5][pipeline] start")
    comment_context = build_v5_comment_context(
        analysis_non_trash=analysis_non_trash,
        comment_truth=comment_truth,
        video_metadata=video_metadata,
        run_id=run_id,
    )
    validate_ingest_schema(comment_context)
    logger.info("[v5][pipeline] ingest_context rows=%s", len(comment_context))

    comment_context_norm = normalize_products(comment_context)
    validate_product_normalized_schema(comment_context_norm)
    logger.info("[v5][pipeline] product_normalizer rows=%s", len(comment_context_norm))

    signal_units = split_signals(comment_context_norm)
    signal_units = attach_topic_ids(signal_units)
    validate_signal_schema(signal_units)
    logger.info("[v5][pipeline] signal_splitter rows=%s", len(signal_units))

    topics = build_topics(signal_units)
    validate_topic_schema(topics)
    logger.info("[v5][pipeline] topic_builder topics=%s", len(topics))

    topic_ranked = rank_topics(topics, signal_units)
    validate_topic_rank_schema(topic_ranked)
    logger.info("[v5][pipeline] topic_ranker rows=%s", len(topic_ranked))

    topic_representatives = select_representatives(topics, signal_units)
    validate_representative_schema(topic_representatives)
    logger.info("[v5][pipeline] representative_selector rows=%s", len(topic_representatives))

    topic_ranked = _enforce_top_quality_gate(topic_ranked, topics, topic_representatives)
    validate_topic_rank_schema(topic_ranked)
    logger.info(
        "[v5][pipeline] top_quality_pass=%s / %s",
        int(topic_ranked.get("top_quality_pass", pd.Series(dtype=bool)).sum()),
        len(topic_ranked),
    )

    topic_insight_actions = generate_insight_actions(topic_ranked, topics, topic_representatives)
    validate_insight_action_schema(topic_insight_actions)
    logger.info("[v5][pipeline] insight_action_generator rows=%s", len(topic_insight_actions))

    export_topic_rows = build_export_topic_pack(topic_ranked, topics, topic_insight_actions, comment_context_norm)
    validate_export_schema(export_topic_rows)
    logger.info("[v5][pipeline] export_topic_pack rows=%s", len(export_topic_rows))

    _assert_sanity(comment_context, signal_units, topics)
    logger.info("[v5][pipeline] sanity check passed")

    return {
        "v5_comment_context": comment_context,
        "v5_comment_context_norm": comment_context_norm,
        "v5_signal_units": signal_units,
        "v5_topics": topics,
        "v5_topic_ranked": topic_ranked,
        "v5_topic_representatives": topic_representatives,
        "v5_topic_insight_actions": topic_insight_actions,
        "v5_export_topic_rows": export_topic_rows,
    }
