"""Rule-based signal splitter for multi-label comment understanding."""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .common import ensure_required_columns, parse_list, safe_text

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

RULE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("contrast", re.compile(r"\s*(?:하지만|반면|그런데|다만|지만)\s*")),
    ("causal", re.compile(r"\s*(?:때문에|그래서|라서|탓에)\s*")),
    ("conditional", re.compile(r"\s*(?:이면|면|으면|면 좋겠다|였으면)\s*")),
    ("additive", re.compile(r"\s*(?:그리고|또한|게다가|도)\s*")),
]

BRAND_TOKENS = ["lg", "삼성", "samsung", "엘지", "소니", "dyson", "다이슨"]
PRODUCT_TOKENS = ["모델", "제품", "냉장고", "세탁기", "건조기", "식기세척기", "틔운", "듀오보", "오브제", "비스포크"]

POSITIVE_TOKENS = ["좋", "만족", "편리", "추천", "잘", "훌륭", "최고", "성능 좋", "튼튼"]
NEGATIVE_TOKENS = ["비싸", "불편", "문제", "고장", "느리", "소음", "진동", "불만", "별로", "어렵"]

JOURNEY_MARKERS = {
    "awareness": ["광고", "썸네일", "처음 알", "브랜드 인지"],
    "exploration": ["비교", "리뷰", "추천", "고민", "찾아보"],
    "purchase": ["가격", "구매", "결제", "할부", "계약"],
    "decision": ["결정", "선택", "포기", "장바구니"],
    "delivery": ["배송", "설치", "지연", "방문"],
    "onboarding": ["초기 설정", "연동", "앱", "로그인", "설정"],
    "usage": ["사용", "성능", "소음", "진동", "작동"],
    "maintenance": ["관리", "청소", "필터", "수리", "서비스", "a/s"],
    "replacement": ["교체", "재구매", "바꾸", "업그레이드"],
}

AXIS_MARKERS = {
    "price_value_fit": ["가격", "비싸", "가성비", "비용", "부담"],
    "installation_space_fit": ["설치", "공간", "크기", "자리"],
    "maintenance_burden": ["관리", "청소", "필터", "소모품", "냄새"],
    "automation_convenience": ["자동", "편리", "불편", "버튼", "앱"],
    "durability_lifespan": ["고장", "내구", "수명", "튼튼"],
    "brand_trust": ["브랜드", "믿", "신뢰", "회사"],
    "usage_pattern_fit": ["사용", "패턴", "가족", "생활"],
    "comparison_substitute_judgment": ["비교", "대체", "vs", "추천"],
    "design_interior_fit": ["디자인", "인테리어", "색상", "외관"],
    "service_after_sales": ["서비스", "a/s", "as", "수리", "센터"],
    "purchase_strategy_contract": ["계약", "할부", "구매", "구독"],
    "repurchase_recommendation": ["재구매", "추천", "다시 살", "지인 추천"],
}


def _split_once(text: str) -> tuple[list[str], str]:
    cleaned = safe_text(text)
    if not cleaned:
        return [], "fallback_single"
    for rule_name, pattern in RULE_PATTERNS:
        if pattern.search(cleaned):
            parts = [segment.strip(" ,.") for segment in pattern.split(cleaned) if segment.strip(" ,.")]
            if len(parts) > 1:
                return parts, rule_name
    return [cleaned], "fallback_single"


def _recursive_split(text: str, depth: int = 0, max_depth: int = 2) -> list[tuple[str, str]]:
    if depth >= max_depth:
        return [(safe_text(text), "fallback_single")]
    parts, rule = _split_once(text)
    if len(parts) <= 1:
        return [(parts[0], rule)] if parts else []
    results: list[tuple[str, str]] = []
    for part in parts:
        sub_parts, sub_rule = _split_once(part)
        if len(sub_parts) > 1 and sub_rule != "fallback_single":
            for sub in sub_parts:
                results.append((safe_text(sub), sub_rule))
        else:
            results.append((safe_text(part), rule))
    return [item for item in results if item[0]]


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
    for stage, tokens in JOURNEY_MARKERS.items():
        if any(token in lowered for token in tokens):
            return stage
    seed_norm = safe_text(seed).lower()
    if seed_norm:
        return seed_norm
    return "usage"


def _infer_axis(signal_text: str, seed_axes: list[str]) -> str:
    lowered = signal_text.lower()
    for axis, tokens in AXIS_MARKERS.items():
        if any(token in lowered for token in tokens):
            return axis
    if seed_axes:
        return safe_text(seed_axes[0]) or "usage_pattern_fit"
    return "usage_pattern_fit"


def _weight(signal_type: str, split_rule: str) -> float:
    if signal_type == "primary":
        return 1.0
    if split_rule in {"contrast", "causal", "conditional", "brand_product"}:
        return 0.8
    return 0.7


def split_comment_to_signals(row: pd.Series) -> list[dict[str, Any]]:
    comment_id = safe_text(row.get("comment_id"))
    video_id = safe_text(row.get("video_id"))
    text = safe_text(row.get("raw_text"))
    if not comment_id or not text:
        return []

    seed_axes = parse_list(row.get("judgment_axes_seed", []))
    seed_journey = safe_text(row.get("journey_stage_seed"))
    seed_polarity = safe_text(row.get("sentiment_seed"))
    product_semantic_id = safe_text(row.get("product_semantic_id"))

    split_pairs = _recursive_split(text)
    if len(split_pairs) <= 1:
        split_pairs = [(text, "fallback_single")]

    output: list[dict[str, Any]] = []
    for index, (segment, split_rule) in enumerate(split_pairs, start=1):
        signal_type = "primary" if index == 1 else "secondary"
        brand_scope = _brand_scope(segment)
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
                "signal_text": segment,
                "signal_type": signal_type,
                "split_rule": effective_rule,
                "journey_stage": _infer_journey(segment, seed_journey),
                "judgment_axis": _infer_axis(segment, seed_axes),
                "polarity": _infer_polarity(segment, seed_polarity),
                "weight": _weight(signal_type, effective_rule),
                "product_semantic_id": product_semantic_id,
                "brand_scope": brand_scope,
                "like_count": int(pd.to_numeric(row.get("like_count", 0), errors="coerce") or 0),
                "reply_count": int(pd.to_numeric(row.get("reply_count", 0), errors="coerce") or 0),
            }
        )
    return output


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
