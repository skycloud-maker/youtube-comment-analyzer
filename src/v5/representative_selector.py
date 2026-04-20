"""Select representative comments per topic (evidence samples only)."""

from __future__ import annotations

import logging

import pandas as pd

from .common import clamp, ensure_required_columns, safe_text

logger = logging.getLogger(__name__)


REP_REQUIRED_COLUMNS = [
    "topic_id",
    "representative_comment_ids",
    "alignment_score",
]


def validate_representative_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, REP_REQUIRED_COLUMNS, "representative_selector")


def select_representatives(topics: pd.DataFrame, signal_units: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(topics, ["topic_id", "journey_stage", "judgment_axis", "polarity_group"], "representative_selector_input_topics")
    ensure_required_columns(signal_units, ["topic_id", "comment_id", "signal_type", "weight"], "representative_selector_input_signals")

    rows: list[dict] = []
    for _, topic_row in topics.iterrows():
        topic_id = safe_text(topic_row.get("topic_id"))
        cluster = signal_units[signal_units["topic_id"].astype(str) == topic_id].copy()
        if cluster.empty:
            rows.append(
                {
                    "topic_id": topic_id,
                    "representative_comment_ids": [],
                    "alignment_score": 0.0,
                    "selection_reason": "topic signal 없음",
                }
            )
            continue

        summary = (
            cluster.groupby("comment_id", as_index=False)
            .agg(
                signal_density=("signal_id", "count"),
                avg_weight=("weight", "mean"),
                primary_count=("signal_type", lambda s: int((s.astype(str) == "primary").sum())),
            )
            .copy()
        )
        max_density = max(int(summary["signal_density"].max()), 1)
        summary["density_norm"] = summary["signal_density"] / max_density
        summary["alignment_component"] = (
            (0.7 * summary["density_norm"])
            + (0.2 * summary["avg_weight"].clip(lower=0.0, upper=1.0))
            + (0.1 * (summary["primary_count"] > 0).astype(float))
        )
        summary = summary.sort_values(["alignment_component", "signal_density"], ascending=False).reset_index(drop=True)

        selected = summary.head(3)
        rep_ids = selected["comment_id"].astype(str).tolist()
        alignment_score = float(selected["alignment_component"].mean()) if not selected.empty else 0.0

        rows.append(
            {
                "topic_id": topic_id,
                "representative_comment_ids": rep_ids,
                "alignment_score": round(clamp(alignment_score, 0.0, 1.0), 6),
                "selection_reason": "signal density + alignment 상위 댓글",
            }
        )

    output = pd.DataFrame(rows)
    validate_representative_schema(output)
    logger.info("[v5][representative_selector] topic_rows=%s", len(output))
    return output
