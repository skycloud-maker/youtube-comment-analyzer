"""Rule-based signal splitter for multi-label comment understanding."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .common import VALID_JUDGMENT_AXES, ensure_required_columns, normalize_judgment_axes, parse_list, safe_text

logger = logging.getLogger(__name__)


SIGNAL_REQUIRED_COLUMNS = [
    "signal_id",
    "comment_id",
    "video_id",
    "signal_text",
    "signal_type",
    "split_rule",
    "journey_stage",
    "judgment_axis",
    "polarity",
    "weight",
    "product_semantic_id",
]

_MIN_SEGMENT_LEN = 4
_MAX_SIGNAL_PER_COMMENT = 4

CONNECTOR_PATTERNS: list[tuple[str, list[re.Pattern[str]]]] = [
    (
        "contrast",
        [
            re.compile(r"(?i)\b(?:but|however|though|although)\b"),
            re.compile(r"(?:하지만|반면|그런데|다만|그러나)"),
            re.compile(r"(?<=[가-힣])(?:지만)(?=[\s,.;!?]|$)"),
        ],
    ),
    (
        "causal",
        [
            re.compile(r"(?i)\b(?:because|therefore|since)\b"),
            re.compile(r"(?:때문에|그래서|라서|탓에)"),
        ],
    ),
    (
        "conditional",
        [
            re.compile(r"(?i)\b(?:if only|as long as|if)\b"),
            re.compile(r"(?:면 좋겠다|으면 좋겠다)"),
            re.compile(r"(?<=[가-힣])(?:면|으면)(?=[\s,.;!?]|$)"),
        ],
    ),
    (
        "additive",
        [
            re.compile(r"(?i)\b(?:and|also|plus)\b"),
            re.compile(r"(?:그리고|또한|게다가|및)"),
            re.compile(r"(?<=[가-힣])도(?=\s+[가-힣A-Za-z])"),
        ],
    ),
]

BRAND_TOKENS = ["lg", "samsung", "whirlpool", "ge", "dyson", "삼성", "엘지", "미디어"]
PRODUCT_TOKENS = [
    "냉장고",
    "세탁기",
    "건조기",
    "식기세척기",
    "dishwasher",
    "washer",
    "dryer",
    "fridge",
    "refrigerator",
    "틔운",
    "티운",
    "tiiun",
    "듀오보",
    "duobo",
    "오브제",
    "objet",
    "비스포크",
    "bespoke",
]

POSITIVE_TOKENS = [
    "좋",
    "만족",
    "편리",
    "추천",
    "최고",
    "good",
    "great",
    "helpful",
    "amazing",
    "works well",
    "love",
]
NEGATIVE_TOKENS = [
    "비싸",
    "불편",
    "문제",
    "고장",
    "느리",
    "소음",
    "진동",
    "불만",
    "별로",
    "오류",
    "bad",
    "worst",
    "cloudy",
    "slow",
    "broken",
    "hate",
    "issue",
]

JOURNEY_MARKERS = {
    "awareness": ["광고", "브랜드", "처음", "awareness", "brand"],
    "exploration": ["비교", "리뷰", "탐색", "찾아", "explore", "review", "compare"],
    "purchase": ["가격", "구매", "결제", "계약", "buy", "purchase", "price"],
    "decision": ["결정", "선택", "고민", "decide", "decision", "choose"],
    "delivery": ["배송", "설치", "방문", "delivery", "install"],
    "onboarding": ["초기 설정", "연동", "가입", "onboard", "setup", "pairing"],
    "usage": ["사용", "성능", "작동", "usage", "performance", "work"],
    "maintenance": ["관리", "청소", "필터", "수리", "서비스", "a/s", "maintenance", "repair"],
    "replacement": ["교체", "재구매", "바꾸", "replace", "replacement"],
}

AXIS_MARKERS: dict[str, list[tuple[str, float]]] = {
    "price_value_fit": [("가격", 1.0), ("비싸", 1.0), ("가성비", 1.0), ("비용", 0.8), ("price", 1.0), ("value", 0.8), ("expensive", 1.0), ("cost", 0.8)],
    "installation_space_fit": [("설치", 1.0), ("공간", 1.0), ("크기", 0.8), ("자리", 0.8), ("install", 1.0), ("space", 1.0), ("size", 0.8)],
    "maintenance_burden": [("관리", 1.0), ("청소", 1.0), ("필터", 1.0), ("세제", 0.8), ("maintenance", 1.0), ("clean", 1.0), ("detergent", 1.0), ("rinse aid", 0.9)],
    "automation_convenience": [("자동", 1.0), ("편의", 1.0), ("버튼", 0.8), ("ui", 0.8), ("ux", 0.8), ("easy", 1.0), ("convenient", 1.0), ("setting", 0.8)],
    "durability_lifespan": [("고장", 1.0), ("내구", 1.0), ("수명", 1.0), ("durable", 1.0), ("broken", 1.0), ("lifespan", 1.0), ("reliable", 0.8)],
    "brand_trust": [("브랜드", 1.0), ("신뢰", 1.0), ("회사", 0.8), ("brand", 1.0), ("trust", 1.0), ("reputation", 0.8)],
    "usage_pattern_fit": [("사용", 1.0), ("패턴", 1.0), ("가족", 0.7), ("생활", 0.7), ("usage", 1.0), ("daily", 0.8), ("habit", 0.7)],
    "comparison_substitute_judgment": [("비교", 1.0), ("대체", 1.0), ("vs", 1.0), ("compare", 1.0), ("alternative", 1.0), ("instead", 0.9), ("rather than", 0.9)],
    "design_interior_fit": [("디자인", 1.0), ("인테리어", 1.0), ("색상", 0.8), ("외관", 0.8), ("design", 1.0), ("interior", 1.0), ("look", 0.7)],
    "service_after_sales": [("서비스", 1.0), ("a/s", 1.0), ("센터", 0.9), ("service", 1.0), ("after sales", 1.0), ("support", 1.0), ("sla", 1.0), ("repair", 0.9)],
    "purchase_strategy_contract": [("계약", 1.0), ("할인", 0.9), ("구매", 0.8), ("구독", 1.0), ("promotion", 0.8), ("contract", 1.0), ("discount", 0.9)],
    "repurchase_recommendation": [("재구매", 1.0), ("추천", 0.8), ("다시 산다", 1.0), ("repurchase", 1.0), ("recommend", 0.8)],
}

GENERIC_SIGNAL_PATTERNS = [
    re.compile(r"(?i)\b(?:thanks|thank you|great video|nice video|lol|lmao)\b"),
    re.compile(r"^[\W_]+$"),
]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def _contains_marker(text_lower: str, marker: str) -> bool:
    marker_lower = marker.lower().strip()
    if not marker_lower:
        return False
    if re.fullmatch(r"[a-z0-9\s/.-]+", marker_lower):
        return bool(re.search(rf"(?<![a-z0-9]){re.escape(marker_lower)}(?![a-z0-9])", text_lower))
    return marker_lower in text_lower


def _is_meaningful_segment(segment: str) -> bool:
    cleaned = _normalize_space(segment)
    if len(cleaned) < _MIN_SEGMENT_LEN:
        return False
    letters = re.sub(r"[^A-Za-z0-9가-힣]", "", cleaned)
    return len(letters) >= 3


def _split_with_single_pattern(text: str, pattern: re.Pattern[str], rule_name: str) -> tuple[list[str], str]:
    if not pattern.search(text):
        return [text], "fallback_single"

    pieces = [piece.strip(" ,.;!?") for piece in pattern.split(text) if piece and piece.strip(" ,.;!?")]
    if len(pieces) <= 1:
        return [text], "fallback_single"
    if not all(_is_meaningful_segment(piece) for piece in pieces):
        return [text], "fallback_single"
    return pieces, rule_name


def _split_once(text: str) -> tuple[list[str], str]:
    cleaned = _normalize_space(text)
    if not cleaned:
        return [], "fallback_single"

    for rule_name, rule_patterns in CONNECTOR_PATTERNS:
        for pattern in rule_patterns:
            pieces, detected = _split_with_single_pattern(cleaned, pattern, rule_name)
            if len(pieces) > 1:
                return pieces, detected
    return [cleaned], "fallback_single"


def _recursive_split(text: str, depth: int = 0, max_depth: int = 3) -> list[tuple[str, str]]:
    if depth >= max_depth:
        return [(safe_text(text), "fallback_single")]

    pieces, rule = _split_once(text)
    if len(pieces) <= 1:
        return [(safe_text(text), rule)] if safe_text(text) else []

    results: list[tuple[str, str]] = []
    for piece in pieces:
        sub_pieces, sub_rule = _split_once(piece)
        if len(sub_pieces) > 1 and sub_rule != "fallback_single":
            for item in sub_pieces:
                if _is_meaningful_segment(item):
                    results.append((safe_text(item), sub_rule))
        else:
            if _is_meaningful_segment(piece):
                results.append((safe_text(piece), rule))

    if len(results) > _MAX_SIGNAL_PER_COMMENT:
        results = results[:_MAX_SIGNAL_PER_COMMENT]
    return results


def _brand_scope(signal_text: str) -> str:
    lowered = signal_text.lower()
    has_brand = any(token in lowered for token in BRAND_TOKENS)
    has_product = any(token in lowered for token in PRODUCT_TOKENS)
    if has_brand and has_product:
        return "both"
    if has_brand:
        return "brand_only"
    if has_product:
        return "product_only"
    return "product_only"


def _infer_polarity(signal_text: str, seed: str) -> str:
    lowered = signal_text.lower()
    pos = any(token in lowered for token in POSITIVE_TOKENS)
    neg = any(token in lowered for token in NEGATIVE_TOKENS)
    if pos and neg:
        return "mixed"
    if neg:
        return "negative"
    if pos:
        return "positive"
    seed_norm = safe_text(seed).lower()
    if seed_norm in {"positive", "negative", "mixed", "neutral"}:
        return seed_norm
    return "neutral"


def _infer_journey(signal_text: str, seed: str) -> str:
    lowered = signal_text.lower()
    scores: dict[str, float] = {}
    for stage, markers in JOURNEY_MARKERS.items():
        score = 0.0
        for marker in markers:
            if _contains_marker(lowered, marker):
                score += 1.0
        scores[stage] = score
    best_stage, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score >= 1.0:
        return best_stage

    seed_norm = safe_text(seed).lower()
    if seed_norm:
        return seed_norm
    return "usage"


def _infer_axis(signal_text: str, seed_axes: list[str]) -> str:
    lowered = signal_text.lower()
    axis_scores: dict[str, float] = {}
    for axis, markers in AXIS_MARKERS.items():
        score = 0.0
        for marker, weight in markers:
            if _contains_marker(lowered, marker):
                score += weight
        axis_scores[axis] = score

    best_axis, best_score = max(axis_scores.items(), key=lambda item: item[1])
    # Require stronger proof for comparison/substitute to avoid weak drift.
    if best_axis == "comparison_substitute_judgment":
        has_explicit_compare = bool(
            re.search(r"(?i)\b(?:compare|comparison|vs|versus|instead|rather than|alternative|switch)\b|비교|대체|대신", lowered)
        )
        if best_score >= 2.2 or (best_score >= 1.8 and has_explicit_compare):
            return best_axis
    elif best_score >= 1.0:
        return best_axis

    normalized_seed = normalize_judgment_axes(seed_axes, logger=logger, source_label="signal_splitter_seed")
    seed_candidates = [safe_text(item).lower() for item in normalized_seed]
    # Do not blindly fallback to comparison axis without explicit comparison signals.
    if seed_candidates and seed_candidates[0] == "comparison_substitute_judgment":
        has_explicit_compare = bool(
            re.search(r"(?i)\b(?:compare|comparison|vs|versus|instead|rather than|alternative|switch)\b|비교|대체|대신", lowered)
        )
        if not has_explicit_compare:
            seed_candidates = [item for item in seed_candidates if item != "comparison_substitute_judgment"]
    axis = safe_text(seed_candidates[0]).lower() if seed_candidates else "usage_pattern_fit"
    if axis not in VALID_JUDGMENT_AXES:
        logger.warning("[v5][axis-parse] invalid seed axis in splitter: %s", axis)
        return "usage_pattern_fit"
    return axis


def _is_generic_signal(signal_text: str) -> bool:
    text = _normalize_space(signal_text)
    if len(text) <= 10:
        return True
    if text.count("?") >= 4 and len(re.sub(r"[^A-Za-z0-9가-힣]", "", text)) < 15:
        return True
    lowered = text.lower()
    if any(pattern.search(lowered) for pattern in GENERIC_SIGNAL_PATTERNS):
        if not any(_contains_marker(lowered, marker) for markers in AXIS_MARKERS.values() for marker, _ in markers):
            return True
    return False


def _weight(signal_type: str, split_rule: str, signal_text: str) -> float:
    if _is_generic_signal(signal_text):
        return 0.45
    if signal_type == "primary":
        base = 1.0
    elif split_rule in {"contrast", "causal", "conditional", "brand_product"}:
        base = 0.85
    else:
        base = 0.75
    return float(max(min(base, 1.0), 0.3))


def split_comment_to_signals(row: pd.Series) -> list[dict[str, Any]]:
    comment_id = safe_text(row.get("comment_id"))
    video_id = safe_text(row.get("video_id"))
    text = _normalize_space(row.get("raw_text"))
    if not comment_id or not text:
        return []

    seed_axes = normalize_judgment_axes(
        parse_list(row.get("judgment_axes_seed", [])),
        logger=logger,
        source_label=f"comment_id={comment_id}",
    )
    seed_journey = safe_text(row.get("journey_stage_seed"))
    seed_polarity = safe_text(row.get("sentiment_seed"))
    product_semantic_id = safe_text(row.get("product_semantic_id"))

    split_pairs = _recursive_split(text)
    if len(split_pairs) <= 1:
        split_pairs = [(text, "fallback_single")]

    output: list[dict[str, Any]] = []
    for index, (segment, split_rule) in enumerate(split_pairs, start=1):
        clean_segment = _normalize_space(segment)
        if not _is_meaningful_segment(clean_segment):
            continue

        signal_type = "primary" if index == 1 else "secondary"
        brand_scope = _brand_scope(clean_segment)
        effective_rule = split_rule
        if brand_scope in {"brand_only", "product_only", "both"} and split_rule == "fallback_single":
            mixed_scope = _brand_scope(text)
            if mixed_scope == "both":
                effective_rule = "brand_product"

        output.append(
            {
                "signal_id": f"{comment_id}#s{index}",
                "comment_id": comment_id,
                "video_id": video_id,
                "signal_text": clean_segment,
                "signal_type": signal_type,
                "split_rule": effective_rule,
                "journey_stage": _infer_journey(clean_segment, seed_journey),
                "judgment_axis": _infer_axis(clean_segment, seed_axes),
                "polarity": _infer_polarity(clean_segment, seed_polarity),
                "weight": _weight(signal_type, effective_rule, clean_segment),
                "product_semantic_id": product_semantic_id,
                "brand_scope": brand_scope,
                "like_count": int(pd.to_numeric(row.get("like_count", 0), errors="coerce") or 0),
                "reply_count": int(pd.to_numeric(row.get("reply_count", 0), errors="coerce") or 0),
            }
        )

    if not output:
        output.append(
            {
                "signal_id": f"{comment_id}#s1",
                "comment_id": comment_id,
                "video_id": video_id,
                "signal_text": text,
                "signal_type": "primary",
                "split_rule": "fallback_single",
                "journey_stage": _infer_journey(text, seed_journey),
                "judgment_axis": _infer_axis(text, seed_axes),
                "polarity": _infer_polarity(text, seed_polarity),
                "weight": 0.6,
                "product_semantic_id": product_semantic_id,
                "brand_scope": _brand_scope(text),
                "like_count": int(pd.to_numeric(row.get("like_count", 0), errors="coerce") or 0),
                "reply_count": int(pd.to_numeric(row.get("reply_count", 0), errors="coerce") or 0),
            }
        )

    return output[:_MAX_SIGNAL_PER_COMMENT]


def validate_signal_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, SIGNAL_REQUIRED_COLUMNS, "signal_splitter")


def split_signals(comment_context_norm: pd.DataFrame) -> pd.DataFrame:
    ensure_required_columns(
        comment_context_norm,
        ["comment_id", "video_id", "raw_text", "product_semantic_id"],
        "signal_splitter_input",
    )
    rows: list[dict[str, Any]] = []
    for _, row in comment_context_norm.iterrows():
        rows.extend(split_comment_to_signals(row))
    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=SIGNAL_REQUIRED_COLUMNS)
    validate_signal_schema(output)
    logger.info("[v5][signal_splitter] comments=%s signals=%s", len(comment_context_norm), len(output))
    return output
