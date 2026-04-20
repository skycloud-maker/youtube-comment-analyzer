"""Generate decision-ready topic insight + action outputs."""

from __future__ import annotations

import hashlib
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

AXIS_LABEL = {
    "price_value_fit": "가격-가치 불일치",
    "installation_space_fit": "설치/공간 제약",
    "maintenance_burden": "유지관리 부담",
    "automation_convenience": "자동화 편의성",
    "durability_lifespan": "내구성 리스크",
    "brand_trust": "브랜드 신뢰",
    "usage_pattern_fit": "사용패턴 적합성",
    "comparison_substitute_judgment": "비교/대체 판단",
    "design_interior_fit": "디자인 적합성",
    "service_after_sales": "서비스/A/S",
    "purchase_strategy_contract": "구매전략/계약",
    "repurchase_recommendation": "재구매 의향",
}

STAGE_LABEL = {
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
    "price_value_fit": "가격 정책",
    "installation_space_fit": "제품/설치",
    "maintenance_burden": "운영/서비스",
    "automation_convenience": "UX",
    "durability_lifespan": "제품 품질",
    "brand_trust": "메시지/브랜딩",
    "usage_pattern_fit": "제품 기획",
    "comparison_substitute_judgment": "메시지/포지셔닝",
    "design_interior_fit": "제품 디자인",
    "service_after_sales": "운영/서비스",
    "purchase_strategy_contract": "정책/계약",
    "repurchase_recommendation": "CRM/메시지",
}

ACTION_CATALOG = {
    "price_value_fit": [
        "가격 구조와 번들 구성을 재정비해 진입 장벽을 낮춘다",
        "가격 대비 가치가 보이도록 등급별 비교 기준을 명확히 제시한다",
    ],
    "installation_space_fit": [
        "설치 전 공간 적합성 체크와 시각 안내를 기본 프로세스로 넣는다",
        "설치 제약과 호환 조건을 구매 전 단계에서 명확히 고지한다",
    ],
    "maintenance_burden": [
        "청소/관리 절차를 단순화하고 리마인드 체계를 도입해 관리 부담을 줄인다",
        "유지관리 대응 프로세스를 표준화해 처리 시간을 단축한다",
    ],
    "automation_convenience": [
        "초기 설정 단계를 줄여 핵심 기능까지의 진입 시간을 단축한다",
        "핵심 UX 흐름에서 이탈을 유발하는 고마찰 단계를 제거한다",
    ],
    "durability_lifespan": [
        "반복 고장 패턴을 우선 대상으로 신뢰성 개선 과제를 집행한다",
        "고위험 부품 중심의 사전 품질 점검 항목을 강화한다",
    ],
    "brand_trust": [
        "신뢰를 떨어뜨리는 접점을 식별해 근거 기반 메시지로 보완한다",
        "지원 채널과 마케팅 채널의 메시지 일관성을 재정렬한다",
    ],
    "usage_pattern_fit": [
        "실사용 패턴에 맞춰 기본 기능 설정과 우선순위를 재배치한다",
        "대표 사용 시나리오 기반으로 모드 추천 가이드를 제공한다",
    ],
    "comparison_substitute_judgment": [
        "비교 구간에서 핵심 우위 포인트가 한눈에 보이도록 메시지를 재정비한다",
        "대체 선택지와의 차이를 사용 시나리오 기준으로 명확히 제시한다",
    ],
    "design_interior_fit": [
        "요청 빈도가 높은 외관 니즈 중심으로 디자인 옵션 구성을 개선한다",
        "구매 단계에서의 기대 불일치를 줄이도록 비주얼 선택지를 조정한다",
    ],
    "service_after_sales": [
        "미해결 건에 대한 응답 SLA와 에스컬레이션 기준을 강제한다",
        "서비스 운영 프로세스를 정비해 수리 리드타임을 단축한다",
    ],
    "purchase_strategy_contract": [
        "계약 조건을 단순화해 구매 단계의 마찰을 줄인다",
        "프로모션/할인 규칙을 재구성해 구매 판단의 명확성을 높인다",
    ],
    "repurchase_recommendation": [
        "재구매 전환을 높이기 위한 메시지 타이밍과 트리거를 정교화한다",
        "긍정 반응이 실제 재구매로 이어지도록 전환 경로를 설계한다",
    ],
}


def validate_insight_action_schema(frame: pd.DataFrame) -> None:
    ensure_required_columns(frame, INSIGHT_REQUIRED_COLUMNS, "insight_action_generator")


def _priority(rank_score: float, negative_or_mixed_weight: float, insufficient: bool) -> str:
    if insufficient:
        return "monitor"
    if rank_score >= 72.0 and negative_or_mixed_weight >= 0.55:
        return "immediate_fix"
    if rank_score >= 42.0:
        return "short_term_improvement"
    return "monitor"


def _polarity_phrase(distribution_json: str) -> str:
    try:
        dist = json.loads(distribution_json or "{}")
    except Exception:
        dist = {}
    neg = float(dist.get("negative", 0.0))
    mixed = float(dist.get("mixed", 0.0))
    pos = float(dist.get("positive", 0.0))
    if neg >= 0.5:
        return "부정 반응이 우세합니다"
    if mixed >= 0.35:
        return "기대-현실 간 격차를 시사하는 혼합 반응이 큽니다"
    if pos >= 0.6:
        return "긍정 반응이 우세합니다"
    return "반응이 분산되어 추가 확인이 필요합니다"


def _product_target(semantic_id: str) -> str:
    text = safe_text(semantic_id)
    named = ""
    if "|name:" in text:
        named = text.split("|name:", 1)[1].split("|", 1)[0]
    category = ""
    if "cat:" in text:
        category = text.split("cat:", 1)[1].split("|", 1)[0]
    target = named if named and named != "na" else category
    return target or "핵심 제품군"


def _pick_variant(topic_id: str, options: list[str]) -> str:
    if not options:
        return "execute a focused corrective plan"
    digest = hashlib.sha1(topic_id.encode("utf-8")).hexdigest()
    idx = int(digest[:4], 16) % len(options)
    return options[idx]


def _insight_sentence(
    topic_id: str,
    *,
    stage_label: str,
    axis_label: str,
    coverage_pct: float,
    frequency_count: int,
    polarity_phrase: str,
) -> str:
    templates = [
        f"{stage_label} 단계에서 {axis_label} 이슈가 반복적으로 관찰됩니다 (비중 {coverage_pct:.1f}%, 근거 {frequency_count:,}건). {polarity_phrase}",
        f"{axis_label} 관련 반응이 {stage_label} 구간에서 집중되어 나타납니다 (근거 {frequency_count:,}건, 비중 {coverage_pct:.1f}%). {polarity_phrase}",
        f"{stage_label} 여정에서 {axis_label} 이슈가 유의미한 비중으로 확인됩니다 (비중 {coverage_pct:.1f}%, 근거 {frequency_count:,}건). {polarity_phrase}",
    ]
    digest = hashlib.sha1((topic_id + "|insight").encode("utf-8")).hexdigest()
    idx = int(digest[:4], 16) % len(templates)
    return templates[idx]


def _action_sentence(topic_id: str, *, target: str, action_base: str, domain: str) -> str:
    templates = [
        f"{target} 대상에서는 {domain} 영역의 우선 과제로 '{action_base}'를 즉시 실행해야 합니다.",
        f"{target} 이슈를 줄이기 위해 {domain} 측면에서 '{action_base}'를 먼저 집행해야 합니다.",
        f"{target} 고객경험 개선을 위해 {domain} 축의 핵심 조치로 '{action_base}'를 바로 추진해야 합니다.",
    ]
    digest = hashlib.sha1((topic_id + "|action").encode("utf-8")).hexdigest()
    idx = int(digest[:4], 16) % len(templates)
    return templates[idx]


def generate_insight_actions(
    topic_ranked: pd.DataFrame,
    topics: pd.DataFrame,
    topic_representatives: pd.DataFrame,
) -> pd.DataFrame:
    ensure_required_columns(
        topic_ranked,
        [
            "topic_id",
            "frequency_count",
            "coverage_pct",
            "polarity_distribution",
            "negative_or_mixed_weight",
            "importance_signal",
            "rank_score",
            "insufficient_evidence",
        ],
        "insight_action_input_ranked",
    )
    ensure_required_columns(
        topics,
        ["topic_id", "journey_stage", "judgment_axis", "product_semantic_id"],
        "insight_action_input_topics",
    )
    ensure_required_columns(
        topic_representatives,
        ["topic_id", "representative_comment_ids", "alignment_score"],
        "insight_action_input_representatives",
    )

    merged = topic_ranked.merge(topics, on="topic_id", how="left").merge(topic_representatives, on="topic_id", how="left")
    rows: list[dict] = []

    for _, row in merged.iterrows():
        topic_id = safe_text(row.get("topic_id"))
        journey_stage = safe_text(row.get("journey_stage"))
        judgment_axis = safe_text(row.get("judgment_axis"))
        stage_label = STAGE_LABEL.get(journey_stage, journey_stage or "unspecified_stage")
        axis_label = AXIS_LABEL.get(judgment_axis, judgment_axis or "unspecified_issue")
        coverage_pct = float(pd.to_numeric(row.get("coverage_pct", 0.0), errors="coerce") or 0.0)
        frequency_count = int(pd.to_numeric(row.get("frequency_count", 0), errors="coerce") or 0)
        rank_score = float(pd.to_numeric(row.get("rank_score", 0.0), errors="coerce") or 0.0)
        importance_signal = float(pd.to_numeric(row.get("importance_signal", 0.0), errors="coerce") or 0.0)
        negative_or_mixed_weight = float(pd.to_numeric(row.get("negative_or_mixed_weight", 0.0), errors="coerce") or 0.0)
        insufficient = bool(row.get("insufficient_evidence", False))

        target = _product_target(safe_text(row.get("product_semantic_id")))
        domain = DOMAIN_BY_AXIS.get(judgment_axis, "product")
        action_base = _pick_variant(topic_id, ACTION_CATALOG.get(judgment_axis, []))

        polarity_phrase = _polarity_phrase(safe_text(row.get("polarity_distribution")))
        insight_text = _insight_sentence(
            topic_id,
            stage_label=stage_label,
            axis_label=axis_label,
            coverage_pct=coverage_pct,
            frequency_count=frequency_count,
            polarity_phrase=polarity_phrase,
        )
        action_text = _action_sentence(topic_id, target=target, action_base=action_base, domain=domain)
        priority_level = _priority(rank_score, negative_or_mixed_weight, insufficient)

        rep_ids_raw = row.get("representative_comment_ids", [])
        rep_ids = parse_list(rep_ids_raw)
        evidence_strength = clamp(
            (0.7 * importance_signal) + (0.3 * float(pd.to_numeric(row.get("alignment_score", 0.0), errors="coerce") or 0.0)),
            0.0,
            1.0,
        )
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
