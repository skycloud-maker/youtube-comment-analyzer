"""Product semantic normalization for V5 topic-first pipeline."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AliasSpec:
    canonical: str
    aliases: tuple[str, ...]


NAMED_PRODUCT_ALIASES = (
    AliasSpec("tiiun", ("틔운", "티운", "tiiun", "tuiun", "ti-uun")),
    AliasSpec("duobo", ("듀오보", "duobo", "duo-bo")),
)

PRODUCT_LINE_ALIASES = (
    AliasSpec("objet", ("오브제", "objet", "objet collection")),
    AliasSpec("bespoke", ("비스포크", "bespoke")),
)

# Order matters. More specific labels come first.
PRODUCT_CATEGORY_ALIASES = (
    AliasSpec("dishwasher", ("식기세척기", "dishwasher", "dish washer")),
    AliasSpec("washer", ("세탁기", "washing machine", "washer")),
    AliasSpec("dryer", ("건조기", "dryer")),
    AliasSpec("refrigerator", ("냉장고", "김치냉장고", "refrigerator", "fridge")),
    AliasSpec("plant_care_device", ("식물관리기", "plant care", "틔운", "티운", "tiiun")),
    AliasSpec("coffee_machine", ("커피머신", "coffee machine", "듀오보", "duobo")),
    AliasSpec("air_conditioner", ("에어컨", "air conditioner", "ac")),
    AliasSpec("vacuum_cleaner", ("청소기", "vacuum")),
)

NAMED_TO_CATEGORY = {
    "tiiun": "plant_care_device",
    "duobo": "coffee_machine",
}

EXCLUDE_CATEGORY_PATTERNS = {
    "dryer": [
        re.compile(r"(?i)\bhair\s*dryer\b"),
        re.compile(r"(?i)\bhairdry(er)?\b"),
    ]
}


def _normalize_text_for_match(text: str) -> str:
    lowered = safe_text(text).lower()
    lowered = re.sub(r"\s+", " ", lowered)
    return f" {lowered} "


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower().strip())
    if re.fullmatch(r"[a-z0-9\-\s]+", alias.lower().strip()):
        return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
    return re.compile(escaped)


def _match_aliases(text: str, specs: tuple[AliasSpec, ...]) -> tuple[str, list[str]]:
    for spec in specs:
        sorted_aliases = sorted(spec.aliases, key=len, reverse=True)
        hits: list[str] = []
        for alias in sorted_aliases:
            pattern = _alias_pattern(alias)
            if pattern.search(text):
                hits.append(alias)
        if hits:
            return spec.canonical, hits
    return "", []


def _is_excluded_category(text: str, category: str) -> bool:
    patterns = EXCLUDE_CATEGORY_PATTERNS.get(category, [])
    return any(pattern.search(text) for pattern in patterns)


def _normalize_one(row: pd.Series) -> dict[str, Any]:
    raw_text = " ".join(
        [
            safe_text(row.get("product_raw")),
            safe_text(row.get("raw_text")),
            safe_text(row.get("brand_raw")),
        ]
    ).strip()
    text = _normalize_text_for_match(raw_text)

    named_product, named_hits = _match_aliases(text, NAMED_PRODUCT_ALIASES)
    product_line, line_hits = _match_aliases(text, PRODUCT_LINE_ALIASES)

    product_category = ""
    category_hits: list[str] = []
    if named_product:
        product_category = NAMED_TO_CATEGORY.get(named_product, "")
        category_hits = [named_product]

    if not product_category:
        category, hits = _match_aliases(text, PRODUCT_CATEGORY_ALIASES)
        if category and not _is_excluded_category(text, category):
            product_category = category
            category_hits = hits

    confidence = 0.40
    source = "semantic_fallback"
    if named_product:
        confidence = 0.96
        source = "dictionary_named"
    elif product_line and product_category:
        confidence = 0.88
        source = "dictionary_line_category"
    elif product_category:
        confidence = 0.82
        source = "dictionary_category"

    if not product_category:
        product_category = "unknown"

    product_semantic_id = f"cat:{product_category}|line:{product_line or 'na'}|name:{named_product or 'na'}"
    alias_hits = list(dict.fromkeys(named_hits + line_hits + category_hits))
    return {
        "product_semantic_id": product_semantic_id,
        "product_category": product_category,
        "product_line": product_line or None,
        "named_product": named_product or None,
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
    logger.info(
        "[v5][product_normalizer] rows=%s unknown_category=%s dishwasher=%s washer=%s",
        len(output),
        int(output["product_category"].eq("unknown").sum()),
        int(output["product_category"].eq("dishwasher").sum()),
        int(output["product_category"].eq("washer").sum()),
    )
    return output
