"""Product semantic normalization for V5 topic-first pipeline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from .common import ensure_required_columns, safe_text

logger = logging.getLogger(__name__)


NORMALIZED_REQUIRED_COLUMNS = [
    "comment_id",
    "video_id",
    "product_semantic_id",
    "product_category",
    "normalization_confidence",
    "lineage_source",
]

NAMED_PRODUCT_ALIASES = {
    "tiiun": ["틔운", "티운", "tiiun", "tiiun", "tuiun"],
    "duobo": ["듀오보", "duobo", "듀오 보", "듀오브루"],
}

PRODUCT_LINE_ALIASES = {
    "objet": ["오브제", "objet", "오브제컬렉션"],
    "bespoke": ["비스포크", "bespoke"],
}

PRODUCT_CATEGORY_ALIASES = {
    "refrigerator": ["냉장고", "김치냉장고", "fridge", "refrigerator"],
    "washer": ["세탁기", "washer", "washing machine"],
    "dryer": ["건조기", "dryer"],
    "dishwasher": ["식기세척기", "dishwasher"],
    "plant_care_device": ["식물관리기", "plant care", "plant", "틔운", "티운"],
    "coffee_machine": ["커피머신", "커피 머신", "coffee machine", "커피"],
    "air_conditioner": ["에어컨", "air conditioner", "ac"],
    "vacuum_cleaner": ["청소기", "vacuum"],
}

NAMED_TO_CATEGORY = {
    "tiiun": "plant_care_device",
    "duobo": "coffee_machine",
}


def _contains_any(text: str, aliases: list[str]) -> list[str]:
    lowered = text.lower()
    hits: list[str] = []
    for alias in aliases:
        candidate = alias.lower()
        if candidate and candidate in lowered:
            hits.append(alias)
    return hits


def _normalize_one(row: pd.Series) -> dict[str, Any]:
    raw = " ".join(
        [
            safe_text(row.get("product_raw")),
            safe_text(row.get("raw_text")),
            safe_text(row.get("brand_raw")),
        ]
    ).strip()
    lowered = raw.lower()

    named_product = ""
    named_hits: list[str] = []
    for canonical, aliases in NAMED_PRODUCT_ALIASES.items():
        hits = _contains_any(lowered, aliases)
        if hits:
            named_product = canonical
            named_hits.extend(hits)
            break

    product_line = ""
    line_hits: list[str] = []
    for canonical, aliases in PRODUCT_LINE_ALIASES.items():
        hits = _contains_any(lowered, aliases)
        if hits:
            product_line = canonical
            line_hits.extend(hits)
            break

    product_category = ""
    category_hits: list[str] = []
    if named_product:
        product_category = NAMED_TO_CATEGORY.get(named_product, "")
        category_hits = [named_product]
    if not product_category:
        for canonical, aliases in PRODUCT_CATEGORY_ALIASES.items():
            hits = _contains_any(lowered, aliases)
            if hits:
                product_category = canonical
                category_hits.extend(hits)
                break

    confidence = 0.40
    source = "semantic_fallback"
    if named_product:
        confidence = 0.95
        source = "dictionary"
    elif product_line and product_category:
        confidence = 0.85
        source = "dictionary"
    elif product_category:
        confidence = 0.75
        source = "regex"

    if not product_category:
        product_category = "unknown"

    product_semantic_id = (
        f"cat:{product_category}|line:{product_line or 'na'}|name:{named_product or 'na'}"
    )

    alias_hits = list(dict.fromkeys(named_hits + line_hits + category_hits))
    return {
        "product_semantic_id": product_semantic_id,
        "product_category": product_category,
        "product_line": product_line if product_line else None,
        "named_product": named_product if named_product else None,
        "normalization_confidence": float(confidence),
        "product_alias_hits": alias_hits,
        "brand_normalized": safe_text(row.get("brand_raw")) or None,
        "lineage_source": source,
    }


def validate_product_normalized_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, NORMALIZED_REQUIRED_COLUMNS, "product_normalizer")


def normalize_products(comment_context: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(comment_context, ["comment_id", "video_id", "raw_text"], "product_normalizer_input")
    working = comment_context.copy()
    normalized = working.apply(_normalize_one, axis=1, result_type="expand")
    output = pd.concat([working.reset_index(drop=True), normalized.reset_index(drop=True)], axis=1)
    validate_product_normalized_schema(output)
    logger.info("[v5][product_normalizer] rows=%s unknown_category=%s", len(output), int(output["product_category"].eq("unknown").sum()))
    return output
