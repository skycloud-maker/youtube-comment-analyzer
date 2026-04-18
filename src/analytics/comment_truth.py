"""Canonical comment-level truth contract and taxonomy assignment helpers."""

from __future__ import annotations

import json
import re
from collections import Counter
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

FIXED_JOURNEY_STAGES = (
    "awareness",
    "exploration",
    "purchase",
    "decision",
    "delivery",
    "onboarding",
    "usage",
    "maintenance",
    "replacement",
)

VARIABLE_JUDGMENT_AXES = (
    "price_value_fit",
    "installation_space_fit",
    "maintenance_burden",
    "automation_convenience",
    "durability_lifespan",
    "brand_trust",
    "usage_pattern_fit",
    "comparison_substitute_judgment",
    "design_interior_fit",
    "service_after_sales",
    "purchase_strategy_contract",
    "repurchase_recommendation",
)

_JOURNEY_ALIASES = {
    "awareness": "awareness",
    "aware": "awareness",
    "s1 aware": "awareness",
    "exploration": "exploration",
    "explore": "exploration",
    "s2 explore": "exploration",
    "purchase": "purchase",
    "s4 purchase": "purchase",
    "decision": "decision",
    "decide": "decision",
    "delivery": "delivery",
    "deliver": "delivery",
    "installation": "delivery",
    "install": "delivery",
    "s5 deliver": "delivery",
    "onboarding": "onboarding",
    "on-board": "onboarding",
    "onboard": "onboarding",
    "s6 on-board": "onboarding",
    "usage": "usage",
    "use": "usage",
    "s7 use": "usage",
    "maintenance": "maintenance",
    "maintain": "maintenance",
    "s8 maintain": "maintenance",
    "replacement": "replacement",
    "replace": "replacement",
    "s9 replace": "replacement",
    "\uC778\uC9C0": "awareness",
    "\uD0D0\uC0C9": "exploration",
    "\uAD6C\uB9E4": "purchase",
    "\uACB0\uC815": "decision",
    "\uBC30\uC1A1": "delivery",
    "\uC0AC\uC6A9\uC900\uBE44": "onboarding",
    "\uC0AC\uC6A9": "usage",
    "\uAD00\uB9AC": "maintenance",
    "\uAD50\uCCB4": "replacement",
}

_JOURNEY_MARKERS = {
    "awareness": [
        "\uAD11\uACE0",
        "\uC378\uB124\uC77C",
        "\uCC98\uC74C \uC54C\uAC8C",
        "ad",
        "advertisement",
        "thumbnail",
        "sponsored",
    ],
    "exploration": [
        "\uBE44\uAD50",
        "\uD6C4\uAE30",
        "\uCD94\uCC9C",
        "\uACE0\uBBFC",
        "review",
        "compare",
        "comparison",
        "vs",
        "recommend",
    ],
    "purchase": [
        "\uAD6C\uB9E4",
        "\uC8FC\uBB38",
        "\uACB0\uC81C",
        "\uAD6C\uB3C5",
        "\uB80C\uD0C8",
        "\uD560\uBD80",
        "\uC57D\uC815",
        "purchase",
        "order",
        "checkout",
        "contract",
        "subscription",
    ],
    "decision": [
        "\uACB0\uC815",
        "\uC120\uD0DD",
        "\uD3EC\uAE30",
        "\uC7A5\uBC14\uAD6C\uB2C8",
        "decide",
        "decision",
        "choose",
        "final",
    ],
    "delivery": [
        "\uBC30\uC1A1",
        "\uC124\uCE58",
        "\uBC29\uBB38",
        "\uD30C\uC190",
        "\uC9C0\uC5F0",
        "delivery",
        "shipping",
        "installation",
        "install",
    ],
    "onboarding": [
        "\uCD08\uAE30 \uC124\uC815",
        "\uC124\uC815",
        "\uB4F1\uB85D",
        "\uC5F0\uB3D9",
        "\uB85C\uADF8\uC778",
        "\uC0AC\uC6A9\uBC95",
        "setup",
        "set up",
        "register",
        "pairing",
        "wifi",
        "app",
        "thinq",
        "manual",
    ],
    "usage": [
        "\uC0AC\uC6A9",
        "\uC131\uB2A5",
        "\uC138\uD0C1\uB825",
        "\uC74C\uC2DD\uBB3C",
        "\uC18C\uC74C",
        "\uB0C4\uC0C8",
        "use",
        "usage",
        "performance",
        "noise",
        "odor",
    ],
    "maintenance": [
        "\uAD00\uB9AC",
        "\uCCAD\uC18C",
        "\uD544\uD130",
        "a/s",
        "as",
        "\uC11C\uBE44\uC2A4",
        "\uC218\uB9AC",
        "\uC13C\uD130",
        "repair",
        "service",
        "warranty",
        "maintenance",
    ],
    "replacement": [
        "\uAD50\uCCB4",
        "\uBC14\uAFB8",
        "\uCC98\uBD84",
        "\uC7AC\uAD6C\uB9E4",
        "replace",
        "replacement",
        "trade in",
        "repurchase",
    ],
}

_AXIS_MARKERS = {
    "price_value_fit": [
        "\uAC00\uACA9",
        "\uBE44\uC2F8",
        "\uAC00\uC131\uBE44",
        "\uCD1D\uBE44\uC6A9",
        "\uAC00\uACA9 \uB300\uBE44",
        "price",
        "cost",
        "value",
        "expensive",
        "budget",
    ],
    "installation_space_fit": [
        "\uC124\uCE58",
        "\uACF5\uAC04",
        "\uC790\uB9AC",
        "\uD06C\uAE30",
        "\uAE4A\uC774",
        "\uD3ED",
        "installation",
        "space",
        "fit",
        "clearance",
    ],
    "maintenance_burden": [
        "\uD544\uD130",
        "\uAD50\uCCB4",
        "\uCCAD\uC18C",
        "\uC138\uCC99",
        "\uC18C\uBAA8\uD488",
        "\uB0C4\uC0C8",
        "\uC9C4\uB4DD",
        "maintenance",
        "cleaning",
        "filter",
        "odor",
    ],
    "automation_convenience": [
        "\uC790\uB3D9",
        "\uD3B8\uC758",
        "\uBD88\uD3B8",
        "\uC870\uC791",
        "\uBC84\uD2BC",
        "\uC571",
        "automation",
        "automatic",
        "convenience",
        "usability",
        "setup",
    ],
    "durability_lifespan": [
        "\uACE0\uC7A5",
        "\uBD88\uB7C9",
        "\uB0B4\uAD6C",
        "\uC218\uBA85",
        "\uD30C\uC190",
        "broken",
        "durability",
        "lifespan",
        "failure",
    ],
    "brand_trust": [
        "lg",
        "samsung",
        "\uBE0C\uB79C\uB4DC",
        "\uC2E0\uB8B0",
        "\uBBFF\uC74C",
        "trust",
        "brand",
        "reliability",
    ],
    "usage_pattern_fit": [
        "\uB300\uAC00\uC871",
        "\uC18C\uB7C9",
        "\uC57C\uAC04",
        "\uB9E4\uC77C",
        "\uC8FC\uB9D0",
        "\uD328\uD134",
        "night",
        "daily",
        "family",
        "habit",
        "pattern",
    ],
    "comparison_substitute_judgment": [
        "\uBE44\uAD50",
        "\uB300\uCCB4",
        "vs",
        "comparison",
        "alternative",
        "compared",
        "\uD1B5\uB3CC\uC774",
        "\uB4DC\uB7FC",
    ],
    "design_interior_fit": [
        "\uB514\uC790\uC778",
        "\uC778\uD14C\uB9AC\uC5B4",
        "\uC0C9\uC0C1",
        "\uB9C8\uAC10",
        "design",
        "interior",
        "aesthetic",
        "look",
    ],
    "service_after_sales": [
        "a/s",
        "as",
        "\uC11C\uBE44\uC2A4",
        "\uC13C\uD130",
        "\uAE30\uC0AC",
        "\uC811\uC218",
        "\uC751\uB300",
        "after sales",
        "repair",
        "warranty",
    ],
    "purchase_strategy_contract": [
        "\uAD6C\uB3C5",
        "\uB80C\uD0C8",
        "\uC57D\uC815",
        "\uD560\uBD80",
        "\uD574\uC9C0",
        "\uC704\uC57D\uAE08",
        "\uACC4\uC57D",
        "contract",
        "subscription",
        "rental",
        "installment",
    ],
    "repurchase_recommendation": [
        "\uCD94\uCC9C",
        "\uC7AC\uAD6C\uB9E4",
        "\uC9C0\uC778",
        "\uB2E4\uC2DC \uC0B4",
        "recommend",
        "repurchase",
        "buy again",
        "word of mouth",
    ],
}

_TOPIC_TO_AXIS = {
    "performance": "automation_convenience",
    "convenience/usability": "automation_convenience",
    "reliability/breakdown": "durability_lifespan",
    "delivery/installation": "installation_space_fit",
    "app/connectivity": "automation_convenience",
    "maintenance/cleaning": "maintenance_burden",
    "a/s & support": "service_after_sales",
    "price/value": "price_value_fit",
    "design/build": "design_interior_fit",
}

_CLASSIFICATION_TO_AXIS = {
    "comparison": "comparison_substitute_judgment",
    "preference": "repurchase_recommendation",
    "complaint": "maintenance_burden",
}

_TRASH_QUALITY_SET = {
    "ad",
    "tagging_only",
    "meaningless",
    "repetitive",
    "generic_engagement",
    "low_context",
}
_ALLOWED_POLARITY = {"positive", "negative", "neutral", "mixed", "trash"}
_TOKEN_SPLIT_RE = re.compile(r"[\s,|;/]+")
_INQUIRY_NOISE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"https?://",
        r"\baffiliate\b",
        r"\bsponsor(?:ed|ing)?\b",
        r"\bpromo\s*code\b",
        r"\bdiscount\s*code\b",
        r"\bcoupon\b",
        r"\bcoupang\s*partners?\b",
        r"\bjoint\s*purchase\b",
        r"\binvestment\b",
        r"\bstock\w*\b",
        r"\brestock\w*\b",
        r"\bin\s+stock\b",
        r"\bout\s+of\s+stock\b",
        r"\bsubscribe\s*event\b",
    ]
)
_NON_CUSTOMER_VOICE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"don't\s+even\s+have",
        r"do\s+not\s+even\s+have",
        r"just\s+good\s+content",
        r"watching\s+for\s+fun",
        r"not\s+using\s+this\s+product",
        r"제품\s+없",
        r"사용\s+안\s+해",
        r"시청만",
        r"\bnot\s+a\s+user\b",
        r"\bnot\s+an?\s+owner\b",
    ]
)
_REAL_INQUIRY_MARKERS = (
    "어떻게",
    "어디",
    "왜",
    "가능",
    "문의",
    "비교",
    "추천",
    "설치",
    "고장",
    "수리",
    "환불",
    "교체",
    "어떤",
    "얼마",
    "몇",
    "되나요",
    "인가요",
    "까요",
    "how",
    "why",
    "where",
    "which",
    "can i",
    "should i",
    "does it",
    "is it",
    "repair",
    "install",
    "price",
    "cost",
)
_REAL_DECISION_USAGE_MARKERS = (
    "고장",
    "수리",
    "설치",
    "교체",
    "환불",
    "비용",
    "가격",
    "비교",
    "추천",
    "사용",
    "as",
    "a/s",
    "repair",
    "install",
    "replacement",
    "warranty",
    "price",
    "cost",
)
_NON_INQUIRY_NARRATIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi used to\b",
        r"\bi can still remember\b",
        r"\bi noticed years ago\b",
        r"\bi'd like to share\b",
        r"\bthought i'd share\b",
        r"\bgreat video\b",
        r"\bthank(s| you)\b",
        r"\btip\b",
        r"\bstock\w*\b",
        r"\brestock\w*\b",
        r"\bin\s+stock\b",
        r"\bout\s+of\s+stock\b",
    ]
)


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


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_coerce_text(item) for item in value if _coerce_text(item)]
    if isinstance(value, tuple):
        return [_coerce_text(item) for item in value if _coerce_text(item)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [_coerce_text(item) for item in parsed if _coerce_text(item)]
            except Exception:
                pass
        return [_coerce_text(item) for item in _TOKEN_SPLIT_RE.split(raw) if _coerce_text(item)]
    return []


def _has_question_marker(text: str) -> bool:
    lowered = text.lower()
    if "?" in text:
        return True
    inquiry_markers = [
        "\uC5B4\uB5BB\uAC8C",
        "\uC65C",
        "\uAC00\uB2A5",
        "\uBB38\uC758",
        "how",
        "why",
        "where",
        "can i",
        "does it",
    ]
    return any(marker.lower() in lowered for marker in inquiry_markers)


def _is_noise_like_text(text: str) -> bool:
    lowered = _coerce_text(text).lower()
    if not lowered:
        return False
    return any(pattern.search(lowered) for pattern in _INQUIRY_NOISE_PATTERNS)


def _is_real_inquiry_text(text: str) -> bool:
    value = _coerce_text(text)
    if not value:
        return False
    lowered = value.lower()
    if _is_noise_like_text(value):
        return False
    has_question_mark = "?" in value
    has_inquiry_marker = any(marker in lowered for marker in _REAL_INQUIRY_MARKERS)
    if not (has_question_mark or has_inquiry_marker):
        return False
    if any(pattern.search(lowered) for pattern in _NON_INQUIRY_NARRATIVE_PATTERNS) and not has_question_mark:
        return False
    has_decision_usage_marker = any(marker in lowered for marker in _REAL_DECISION_USAGE_MARKERS)
    if not has_decision_usage_marker and len(value) > 140:
        return False
    # Very long narratives without explicit inquiry intent are weak supporting inquiries.
    if len(value) > 520 and not has_question_mark:
        return False
    return True


def _is_non_customer_voice(text: str) -> bool:
    lowered = _coerce_text(text).lower()
    if not lowered:
        return False
    return any(pattern.search(lowered) for pattern in _NON_CUSTOMER_VOICE_PATTERNS)


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


def _normalize_journey_stage(value: Any) -> str:
    lowered = _coerce_text(value).lower()
    if not lowered:
        return ""
    return _JOURNEY_ALIASES.get(lowered, "")


def _build_semantic_blob(row: pd.Series) -> str:
    parts: list[str] = []
    for key in [
        "text_display",
        "text_original",
        "cleaned_text",
        "title",
        "product",
        "classification_type",
        "topic_label",
    ]:
        value = _coerce_text(row.get(key))
        if value:
            parts.append(value.lower())
    for key in ["nlp_topics", "nlp_core_points", "nlp_context_tags", "nlp_similarity_keys", "nlp_product_mentions"]:
        parts.extend([token.lower() for token in _parse_list(row.get(key)) if token])
    return " ".join(parts)


def _score_markers(blob: str, markers: list[str]) -> int:
    if not blob:
        return 0
    return sum(1 for marker in markers if marker and marker.lower() in blob)


def _resolve_journey_stage(
    row: pd.Series,
    *,
    analysis_included: bool,
    inquiry_flag: bool,
) -> str | Any:
    for key in ["journey_stage", "customer_journey_stage", "cej_scene_code"]:
        normalized = _normalize_journey_stage(row.get(key))
        if normalized:
            return normalized

    if not analysis_included:
        return pd.NA

    blob = _build_semantic_blob(row)
    classification = _coerce_text(row.get("classification_type")).lower()
    sentiment = _coerce_text(row.get("sentiment_final")).lower()
    product_related = _coerce_bool(row.get("product_related"), default=False)

    scores: Counter[str] = Counter()
    for stage, markers in _JOURNEY_MARKERS.items():
        score = _score_markers(blob, markers)
        if score:
            scores[stage] += score

    if classification == "comparison":
        scores["exploration"] += 2
    if classification == "preference":
        scores["decision"] += 2
    if inquiry_flag:
        scores["exploration"] += 1
    if classification == "complaint":
        scores["usage"] += 1
        if any(token in blob for token in ["a/s", "repair", "\uC11C\uBE44\uC2A4", "\uC218\uB9AC"]):
            scores["maintenance"] += 2
    if sentiment == "mixed":
        scores["usage"] += 1

    if scores:
        return max(FIXED_JOURNEY_STAGES, key=lambda stage: scores.get(stage, 0))

    if product_related:
        if classification in {"comparison", "informational"} and inquiry_flag:
            return "exploration"
        if classification == "preference":
            return "decision"
        return "usage"
    return pd.NA


def _resolve_judgment_axes(
    row: pd.Series,
    *,
    journey_stage: str | Any,
    inquiry_flag: bool,
) -> list[str]:
    blob = _build_semantic_blob(row)
    classification = _coerce_text(row.get("classification_type")).lower()
    sentiment = _coerce_text(row.get("sentiment_final")).lower()
    topic_value = _coerce_text(row.get("topic_label")).lower()
    product_related = _coerce_bool(row.get("product_related"), default=False)

    scores: Counter[str] = Counter()
    for axis, markers in _AXIS_MARKERS.items():
        marker_score = _score_markers(blob, markers)
        if marker_score:
            scores[axis] += marker_score

    if topic_value:
        mapped_axis = _TOPIC_TO_AXIS.get(topic_value)
        if mapped_axis:
            scores[mapped_axis] += 1
    if classification in _CLASSIFICATION_TO_AXIS:
        scores[_CLASSIFICATION_TO_AXIS[classification]] += 2
    if inquiry_flag:
        scores["automation_convenience"] += 1
    if sentiment == "mixed":
        scores["usage_pattern_fit"] += 1

    stage_value = _coerce_text(journey_stage).lower()
    if stage_value in {"purchase", "decision"}:
        scores["purchase_strategy_contract"] += 1
        scores["price_value_fit"] += 1
    if stage_value == "delivery":
        scores["installation_space_fit"] += 1
    if stage_value == "onboarding":
        scores["automation_convenience"] += 1
    if stage_value == "maintenance":
        scores["service_after_sales"] += 2
        scores["maintenance_burden"] += 1
    if stage_value == "replacement":
        scores["durability_lifespan"] += 1
        scores["repurchase_recommendation"] += 1

    ranked = sorted(
        [axis for axis in VARIABLE_JUDGMENT_AXES if scores.get(axis, 0) > 0],
        key=lambda axis: (-scores[axis], VARIABLE_JUDGMENT_AXES.index(axis)),
    )
    selected = ranked[:3]

    if not selected and product_related:
        if stage_value == "maintenance":
            selected = ["service_after_sales"]
        elif stage_value == "delivery":
            selected = ["installation_space_fit"]
        elif stage_value in {"purchase", "decision", "exploration"}:
            selected = ["price_value_fit"]
        else:
            selected = ["automation_convenience"]
    return selected


def build_canonical_comment_truth(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build canonical comment-level truth rows with 2-layer taxonomy fields."""
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
    working["sentiment_final"] = sentiment_final

    inquiry_flag = working.apply(
        lambda row: bool(
            _coerce_bool(row.get("nlp_is_inquiry"), default=False)
            or _is_real_inquiry_text(_coerce_text(row.get("text_display")))
            or _is_real_inquiry_text(_coerce_text(row.get("text_original")))
            or _is_real_inquiry_text(_coerce_text(row.get("cleaned_text")))
        ),
        axis=1,
    )

    hygiene_class = working.apply(
        lambda row: _resolve_hygiene_class(row, inquiry_flag=bool(inquiry_flag.loc[row.name])),
        axis=1,
    )

    analysis_included = (
        working.get("comment_validity", pd.Series("", index=working.index)).fillna("").astype(str).str.lower().eq("valid")
        & fallback_text.ne("")
        & hygiene_class.ne("trash")
    )

    journey_stage = working.apply(
        lambda row: _resolve_journey_stage(
            row,
            analysis_included=bool(analysis_included.loc[row.name]),
            inquiry_flag=bool(inquiry_flag.loc[row.name]),
        ),
        axis=1,
    )

    judgment_axes = working.apply(
        lambda row: _resolve_judgment_axes(
            row,
            journey_stage=journey_stage.loc[row.name],
            inquiry_flag=bool(inquiry_flag.loc[row.name]),
        ),
        axis=1,
    )

    non_customer_voice = fallback_text.map(_is_non_customer_voice)
    representative_eligibility = (
        analysis_included
        & hygiene_class.eq("strategic")
        & journey_stage.notna()
        & ~non_customer_voice
    )

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
