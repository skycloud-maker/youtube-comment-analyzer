"""Build deterministic topics from signal units."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter

import pandas as pd

from .common import ensure_required_columns, safe_text

logger = logging.getLogger(__name__)


TOPIC_REQUIRED_COLUMNS = [
    "topic_id",
    "topic_key",
    "product_semantic_id",
    "journey_stage",
    "judgment_axis",
    "polarity_group",
    "v5_simple_signature",
    "v5_advanced_signature",
    "signature",
    "signal_count",
    "comment_count",
]

TOKEN_SPLIT_RE = re.compile(r"[^\w\uac00-\ud7a3]+")
STOPWORDS = {
    "그리고",
    "하지만",
    "그런데",
    "정말",
    "너무",
    "진짜",
    "그냥",
    "제품",
    "모델",
    "사용",
    "구매",
}


def _simple_signature(texts: pd.Series, journey_stage: str, judgment_axis: str) -> str:
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        for token in TOKEN_SPLIT_RE.split(text.lower()):
            token = token.strip()
            if len(token) < 2 or token in STOPWORDS:
                continue
            counter[token] += 1
    top_tokens = [token for token, _ in counter.most_common(2)]
    token_part = "_".join(top_tokens) if top_tokens else "general"
    return f"{safe_text(journey_stage)}_{safe_text(judgment_axis)}_{token_part}"


def _topic_key(row: pd.Series) -> str:
    return "|".join(
        [
            safe_text(row.get("journey_stage")),
            safe_text(row.get("judgment_axis")),
            safe_text(row.get("polarity_group")),
            safe_text(row.get("product_semantic_id")),
            safe_text(row.get("v5_simple_signature")),
        ]
    )


def _topic_id(topic_key: str) -> str:
    digest = hashlib.sha1(topic_key.encode("utf-8")).hexdigest()[:12]
    return f"t_{digest}"


def validate_topic_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, TOPIC_REQUIRED_COLUMNS, "topic_builder")


def attach_topic_ids(signal_units: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(
        signal_units,
        ["signal_id", "signal_text", "product_semantic_id", "journey_stage", "judgment_axis", "polarity"],
        "topic_builder_attach_input",
    )
    if signal_units.empty:
        return signal_units.copy()

    working = signal_units.copy()
    working["polarity_group"] = working.get("polarity", pd.Series("neutral", index=working.index)).fillna("neutral").astype(str).str.lower()

    signature_keys = ["product_semantic_id", "journey_stage", "judgment_axis", "polarity_group"]
    signatures: dict[tuple[str, str, str, str], str] = {}
    for group_key, cluster in working.groupby(signature_keys, dropna=False):
        signatures[group_key] = _simple_signature(cluster["signal_text"], safe_text(group_key[1]), safe_text(group_key[2]))

    def _row_key(row: pd.Series) -> tuple[str, str, str, str]:
        return (
            safe_text(row.get("product_semantic_id")),
            safe_text(row.get("journey_stage")),
            safe_text(row.get("judgment_axis")),
            safe_text(row.get("polarity_group")),
        )

    working["v5_simple_signature"] = working.apply(
        lambda row: signatures.get(_row_key(row), "general_general_general"),
        axis=1,
    )
    working["topic_key"] = working.apply(_topic_key, axis=1)
    working["topic_id"] = working["topic_key"].map(_topic_id)
    return working


def build_topics(signal_units: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(
        signal_units,
        ["signal_id", "comment_id", "signal_text", "journey_stage", "judgment_axis", "polarity", "product_semantic_id"],
        "topic_builder_input",
    )
    if signal_units.empty:
        empty = pd.DataFrame(columns=TOPIC_REQUIRED_COLUMNS)
        validate_topic_schema(empty)
        return empty

    working = attach_topic_ids(signal_units)

    group_cols = ["product_semantic_id", "journey_stage", "judgment_axis", "polarity_group", "topic_id", "topic_key", "v5_simple_signature"]
    rows: list[dict] = []
    for group_key, cluster in working.groupby(group_cols, dropna=False):
        product_semantic_id, journey_stage, judgment_axis, polarity_group, topic_id, topic_key, signature = group_key
        rows.append(
            {
                "topic_id": safe_text(topic_id),
                "topic_key": safe_text(topic_key),
                "product_semantic_id": safe_text(product_semantic_id),
                "journey_stage": safe_text(journey_stage),
                "judgment_axis": safe_text(judgment_axis),
                "polarity_group": safe_text(polarity_group),
                "v5_simple_signature": safe_text(signature),
                "v5_advanced_signature": None,
                "signature": safe_text(signature),
                "signal_count": int(len(cluster)),
                "comment_count": int(cluster["comment_id"].astype(str).nunique()),
            }
        )

    output = pd.DataFrame(rows).drop_duplicates(subset=["topic_id"]).reset_index(drop=True)
    validate_topic_schema(output)
    logger.info("[v5][topic_builder] topics=%s", len(output))
    return output
