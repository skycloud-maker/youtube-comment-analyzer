"""Build topic-grouped export rows (sheet-ready)."""

from __future__ import annotations

import json
import logging
import re

import pandas as pd

from .common import ensure_required_columns, parse_list, safe_text

logger = logging.getLogger(__name__)


EXPORT_REQUIRED_COLUMNS = [
    "topic_group",
    "topic_rank",
    "polarity_group",
    "topic_display_label",
    "comment_list",
]

POLARITY_GROUP_MAP = {
    "negative": "NEG",
    "positive": "POS",
    "mixed": "MIX",
    "neutral": "NEU",
}

AXIS_LABEL = {
    "price_value_fit": "가격부담",
    "installation_space_fit": "설치불편",
    "maintenance_burden": "관리부담",
    "automation_convenience": "자동화편의",
    "durability_lifespan": "내구이슈",
    "brand_trust": "브랜드신뢰",
    "usage_pattern_fit": "사용적합",
    "comparison_substitute_judgment": "비교판단",
    "design_interior_fit": "디자인적합",
    "service_after_sales": "A/S이슈",
    "purchase_strategy_contract": "구매전략",
    "repurchase_recommendation": "재구매추천",
}


def _clean_sheet_token(value: str) -> str:
    token = re.sub(r"[\\/*?:\[\]]+", "", safe_text(value))
    return token[:31] if len(token) > 31 else token


def _sheet_name(topic_group: str, topic_rank: str, label: str) -> str:
    return _clean_sheet_token(f"{topic_group}_{topic_rank}_{label}")


def validate_export_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, EXPORT_REQUIRED_COLUMNS, "export_topic_pack")


def build_export_topic_pack(
    topic_ranked: pd.DataFrame,
    topics: pd.DataFrame,
    insight_actions: pd.DataFrame,
    comment_context_norm: pd.DataFrame,
) -> pd.DataFrame:
    ensure_required_columns(topic_ranked, ["topic_id", "rank_score"], "export_topic_pack_input_ranked")
    ensure_required_columns(topics, ["topic_id", "judgment_axis", "polarity_group"], "export_topic_pack_input_topics")
    ensure_required_columns(insight_actions, ["topic_id", "supporting_comment_ids"], "export_topic_pack_input_actions")
    ensure_required_columns(comment_context_norm, ["comment_id", "raw_text"], "export_topic_pack_input_comments")

    merged = (
        topic_ranked.merge(topics[["topic_id", "judgment_axis", "polarity_group"]], on="topic_id", how="left")
        .merge(insight_actions[["topic_id", "supporting_comment_ids"]], on="topic_id", how="left")
        .sort_values("rank_score", ascending=False)
        .reset_index(drop=True)
    )

    group_counters: dict[str, int] = {}
    comment_lookup = comment_context_norm.set_index(comment_context_norm["comment_id"].astype(str))
    rows: list[dict] = []

    for _, row in merged.iterrows():
        polarity = safe_text(row.get("polarity_group")).lower()
        topic_group = POLARITY_GROUP_MAP.get(polarity, "NEU")
        group_counters[topic_group] = group_counters.get(topic_group, 0) + 1
        topic_rank = f"T{group_counters[topic_group]}"

        axis = safe_text(row.get("judgment_axis"))
        label = AXIS_LABEL.get(axis, axis or "핵심이슈")

        comment_ids = parse_list(row.get("supporting_comment_ids", []))
        comments_payload: list[dict] = []
        for cid in comment_ids:
            if cid in comment_lookup.index:
                comment_row = comment_lookup.loc[cid]
                comments_payload.append(
                    {
                        "comment_id": cid,
                        "text": safe_text(comment_row.get("raw_text"))[:180],
                    }
                )

        rows.append(
            {
                "topic_id": safe_text(row.get("topic_id")),
                "topic_group": topic_group,
                "topic_rank": topic_rank,
                "polarity_group": polarity or "neutral",
                "topic_display_label": label,
                "comment_list": json.dumps(comments_payload, ensure_ascii=False),
                "sheet_name": _sheet_name(topic_group, topic_rank, label),
            }
        )

    output = pd.DataFrame(rows)
    validate_export_schema(output)
    logger.info("[v5][export_topic_pack] rows=%s", len(output))
    return output
