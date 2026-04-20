"""Generate decision-ready topic insight + action outputs."""

from __future__ import annotations

import json
import logging

import pandas as pd

from .common import clamp, ensure_required_columns, parse_list, safe_text

logger = logging.getLogger(__name__)


INSIGHT_REQUIRED_COLUMNS = [
    "topic_id",
    "insight_text",
    "action_text",
    "priority_level",
    "evidence_strength",
    "supporting_comment_ids",
]

AXIS_KO = {
    "price_value_fit": "가격 부담",
    "installation_space_fit": "설치/공간 적합",
    "maintenance_burden": "유지관리 부담",
    "automation_convenience": "자동화 편의",
    "durability_lifespan": "내구/수명",
    "brand_trust": "브랜드 신뢰",
    "usage_pattern_fit": "사용 패턴 적합",
    "comparison_substitute_judgment": "비교/대체 판단",
    "design_interior_fit": "디자인/인테리어",
    "service_after_sales": "서비스/A/S",
    "purchase_strategy_contract": "구매전략/계약",
    "repurchase_recommendation": "재구매/추천",
}

STAGE_KO = {
    "awareness": "인지",
    "exploration": "탐색",
    "purchase": "구매",
    "decision": "결정",
    "delivery": "배송",
    "onboarding": "사용준비",
    "usage": "사용",
    "maintenance": "관리",
    "replacement": "교체",
}

DOMAIN_BY_AXIS = {
    "price_value_fit": "정책",
    "installation_space_fit": "제품",
    "maintenance_burden": "운영",
    "automation_convenience": "UX",
    "durability_lifespan": "제품",
    "brand_trust": "메시지",
    "usage_pattern_fit": "제품",
    "comparison_substitute_judgment": "메시지",
    "design_interior_fit": "제품",
    "service_after_sales": "운영",
    "purchase_strategy_contract": "정책",
    "repurchase_recommendation": "메시지",
}

ACTION_BY_AXIS = {
    "price_value_fit": "가격 정책을 재조정",
    "installation_space_fit": "설치 제약을 줄이는 설계와 안내를 강화",
    "maintenance_burden": "유지관리 절차와 A/S 응답 체계를 단축",
    "automation_convenience": "초기 설정과 제어 흐름을 단순화",
    "durability_lifespan": "내구 이슈를 줄이는 품질 개선 항목을 우선 반영",
    "brand_trust": "신뢰 저하 지점을 해소하는 커뮤니케이션을 정비",
    "usage_pattern_fit": "실사용 패턴에 맞는 기능 우선순위를 재배치",
    "comparison_substitute_judgment": "비교 구간에서 우위가 드러나는 메시지를 명확화",
    "design_interior_fit": "디자인 선택 기준과 옵션 구성을 개선",
    "service_after_sales": "서비스 응답 SLA와 처리 표준을 개선",
    "purchase_strategy_contract": "구매/계약 조건을 사용자 친화적으로 조정",
    "repurchase_recommendation": "재구매/추천 전환을 높이는 운영 시나리오를 강화",
}


def validate_insight_action_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, INSIGHT_REQUIRED_COLUMNS, "insight_action_generator")


def _priority(rank_score: float, negative_or_mixed_weight: float, insufficient: bool) -> str:
    if insufficient:
        return "모니터링"
    if rank_score >= 75.0 and negative_or_mixed_weight >= 0.6:
        return "즉시 대응"
    if rank_score >= 40.0:
        return "단기 개선"
    return "모니터링"


def _polarity_phrase(distribution_json: str) -> str:
    try:
        dist = json.loads(distribution_json or "{}")
    except Exception:
        dist = {}
    neg = float(dist.get("negative", 0.0))
    mixed = float(dist.get("mixed", 0.0))
    pos = float(dist.get("positive", 0.0))
    if neg >= 0.5:
        return "부정 반응이 우세"
    if mixed >= 0.35:
        return "혼합 반응이 높아 기대-현실 격차가 큼"
    if pos >= 0.6:
        return "긍정 반응이 우세"
    return "반응이 분산"


def _product_target(semantic_id: str) -> str:
    text = safe_text(semantic_id)
    named = ""
    if "|name:" in text:
        named = text.split("|name:", 1)[1].split("|", 1)[0]
    category = ""
    if "cat:" in text:
        category = text.split("cat:", 1)[1].split("|", 1)[0]
    target = named if named and named != "na" else category
    return target or "해당 제품군"


def generate_insight_actions(
    topic_ranked: pd.DataFrame,
    topics: pd.DataFrame,
    topic_representatives: pd.DataFrame,
) -> pd.DataFrame:
    ensure_required_columns(topic_ranked, ["topic_id", "frequency_count", "coverage_pct", "polarity_distribution", "negative_or_mixed_weight", "importance_signal", "rank_score", "insufficient_evidence"], "insight_action_input_ranked")
    ensure_required_columns(topics, ["topic_id", "journey_stage", "judgment_axis", "product_semantic_id"], "insight_action_input_topics")
    ensure_required_columns(topic_representatives, ["topic_id", "representative_comment_ids", "alignment_score"], "insight_action_input_representatives")

    merged = topic_ranked.merge(topics, on="topic_id", how="left").merge(topic_representatives, on="topic_id", how="left")
    rows: list[dict] = []

    for _, row in merged.iterrows():
        topic_id = safe_text(row.get("topic_id"))
        journey_stage = safe_text(row.get("journey_stage"))
        judgment_axis = safe_text(row.get("judgment_axis"))
        stage_label = STAGE_KO.get(journey_stage, journey_stage or "미분류")
        axis_label = AXIS_KO.get(judgment_axis, judgment_axis or "핵심 이슈")
        coverage_pct = float(pd.to_numeric(row.get("coverage_pct", 0.0), errors="coerce") or 0.0)
        frequency_count = int(pd.to_numeric(row.get("frequency_count", 0), errors="coerce") or 0)
        rank_score = float(pd.to_numeric(row.get("rank_score", 0.0), errors="coerce") or 0.0)
        importance_signal = float(pd.to_numeric(row.get("importance_signal", 0.0), errors="coerce") or 0.0)
        negative_or_mixed_weight = float(pd.to_numeric(row.get("negative_or_mixed_weight", 0.0), errors="coerce") or 0.0)
        insufficient = bool(row.get("insufficient_evidence", False))

        target = _product_target(safe_text(row.get("product_semantic_id")))
        domain = DOMAIN_BY_AXIS.get(judgment_axis, "제품")
        action_base = ACTION_BY_AXIS.get(judgment_axis, "핵심 문제를 직접 해결하는 실행안을 적용")

        insight_text = (
            f"{stage_label} 단계에서 {axis_label} 관련 반응이 {coverage_pct:.1f}% (빈도 {frequency_count:,}건)로 확인되며, "
            f"{_polarity_phrase(safe_text(row.get('polarity_distribution')))} 상태라 우선 대응 근거가 충분합니다."
        )
        action_text = (
            f"{target} {stage_label} 고객의 {axis_label} 이슈를 줄이기 위해 "
            f"{action_base}을(를) {domain} 영역에서 우선 실행해야 합니다."
        )
        priority_level = _priority(rank_score, negative_or_mixed_weight, insufficient)

        rep_ids_raw = row.get("representative_comment_ids", [])
        rep_ids = parse_list(rep_ids_raw)
        evidence_strength = clamp((0.7 * importance_signal) + (0.3 * float(pd.to_numeric(row.get("alignment_score", 0.0), errors="coerce") or 0.0)), 0.0, 1.0)
        if insufficient:
            evidence_strength = min(evidence_strength, 0.49)

        rows.append(
            {
                "topic_id": topic_id,
                "insight_text": insight_text,
                "action_text": action_text,
                "priority_level": priority_level,
                "evidence_strength": round(float(evidence_strength), 6),
                "supporting_comment_ids": rep_ids,
            }
        )

    output = pd.DataFrame(rows).sort_values("evidence_strength", ascending=False).reset_index(drop=True)
    validate_insight_action_schema(output)
    logger.info("[v5][insight_action_generator] rows=%s", len(output))
    return output
