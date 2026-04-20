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

TOKEN_SPLIT_RE = re.compile(r"[^A-Za-z0-9\uac00-\ud7a3]+")
GENERIC_STOPWORDS = {
    "the",
    "and",
    "but",
    "for",
    "with",
    "that",
    "this",
    "have",
    "has",
    "had",
    "just",
    "very",
    "really",
    "video",
    "dishwasher",
    "washers",
    "washer",
    "machine",
    "using",
    "used",
    "user",
    "users",
    "good",
    "great",
    "nice",
    "thanks",
    "thank",
    "it",
    "to",
    "my",
    "you",
    "your",
    "our",
    "we",
    "they",
    "is",
    "are",
    "was",
    "were",
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "please",
    "help",
    "제품",
    "사용",
    "영상",
    "정말",
    "너무",
    "그냥",
    "완전",
    "진짜",
    "그리고",
    "하지만",
    "그래서",
}
TOKEN_NORMALIZATION = {
    "repairs": "repair",
    "repairing": "repair",
    "service": "service",
    "servicing": "service",
    "cleaning": "clean",
    "cleaned": "clean",
    "filters": "filter",
    "detergents": "detergent",
    "pods": "pod",
    "costs": "cost",
    "prices": "price",
    "expensive": "price",
    "cheaper": "price",
    "delivery": "delivery",
    "install": "installation",
    "installed": "installation",
}


def _normalize_token(token: str) -> str:
    lowered = token.lower().strip()
    if not lowered:
        return ""
    lowered = TOKEN_NORMALIZATION.get(lowered, lowered)
    if len(lowered) < 2:
        return ""
    if lowered in GENERIC_STOPWORDS:
        return ""
    return lowered


def _simple_signature(texts: pd.Series, journey_stage: str, judgment_axis: str) -> str:
    counter: Counter[str] = Counter()
    for text in texts.fillna("").astype(str):
        for raw_token in TOKEN_SPLIT_RE.split(text):
            token = _normalize_token(raw_token)
            if not token:
                continue
            counter[token] += 1

    # Prefer repeated tokens to avoid micro-topic signatures from one-off words.
    min_repeat = 3 if len(texts) >= 20 else 2
    repeated_tokens = [token for token, count in counter.items() if count >= min_repeat]
    ordered_tokens = sorted(repeated_tokens, key=lambda token: (-counter[token], token))
    if not ordered_tokens:
        ordered_tokens = [token for token, _ in counter.most_common(2)]

    token_part = "_".join(ordered_tokens[:2]) if ordered_tokens else "general"
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
