"""Strategy-first Streamlit dashboard layer.

This module keeps the existing analytics engine intact and rebuilds only the
presentation layer around strategy issues first, evidence second.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
import streamlit as st

from src import dashboard_app as legacy

_FORCE_WIDE_SCOPE_KEY = "v2_force_wide_scope"
_PENDING_RESET_FILTERS_KEY = "v2_pending_reset_filters"


_JOURNEY_KO = {
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

_AXIS_KO = {
    "price_value_fit": "가격 대비 납득감",
    "installation_space_fit": "설치/공간 적합성",
    "maintenance_burden": "유지관리 부담",
    "automation_convenience": "자동화/편의성",
    "durability_lifespan": "내구성/수명",
    "brand_trust": "브랜드 신뢰",
    "usage_pattern_fit": "사용 패턴 적합성",
    "comparison_substitute_judgment": "비교/대체재 판단",
    "design_interior_fit": "디자인/인테리어 적합성",
    "service_after_sales": "서비스/A/S",
    "purchase_strategy_contract": "구매 전략/약정",
    "repurchase_recommendation": "재구매/추천 의향",
}

_ACTION_DOMAIN_BY_AXIS = {
    "price_value_fit": "business / policy",
    "installation_space_fit": "product",
    "maintenance_burden": "product",
    "automation_convenience": "UX / guide",
    "durability_lifespan": "product",
    "brand_trust": "marketing / message",
    "usage_pattern_fit": "UX / guide",
    "comparison_substitute_judgment": "marketing / message",
    "design_interior_fit": "product",
    "service_after_sales": "operations / service",
    "purchase_strategy_contract": "business / policy",
    "repurchase_recommendation": "marketing / message",
}

_ACTION_LABEL_KO = {
    "immediate_fix": "즉시 조치",
    "monitor": "모니터링",
    "product_improvement": "제품 개선",
    "service_operation": "서비스 운영",
    "marketing_message": "메시지/전달 개선",
    "maintain_strength": "강점 유지",
}

_PRIORITY_LEVELS = ("High", "Medium", "Low")
_IMPACT_LEVELS = ("High", "Medium", "Low")
_PRIORITY_RANK = {"High": 3, "Medium": 2, "Low": 1}
_IMPACT_RANK = {"High": 3, "Medium": 2, "Low": 1}
_LEVEL_KO = {"High": "높음", "Medium": "보통", "Low": "낮음"}
_RECOMMENDED_ACTION_TEXT = {
    "immediate_fix": "즉시 실행 필요",
    "product_improvement": "제품 백로그 우선 반영",
    "service_operation": "서비스 운영 프로세스 보정",
    "marketing_message": "커뮤니케이션 전략 보정",
    "maintain_strength": "강점 유지·확장",
    "monitor": "모니터링 및 근거 추가 수집",
}


@dataclass
class DashboardContext:
    active_mode: str
    active_snapshot_run: str
    bundle: dict[str, Any]
    raw_bundle: dict[str, pd.DataFrame]
    strategy_video: pd.DataFrame
    strategy_product: pd.DataFrame
    top_issues: pd.DataFrame
    top_issues_fallback: pd.DataFrame
    filtered_comments: pd.DataFrame
    filtered_videos: pd.DataFrame
    analysis_non_trash: pd.DataFrame
    analysis_non_trash_full: pd.DataFrame
    representative_comments: pd.DataFrame
    dashboard_options: dict[str, list[str]]
    search_query: str
    force_wide_scope: bool


def _safe_text(value: Any) -> str:
    return legacy._safe_text(value)


def _parse_list_like(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(v) for v in value if _safe_text(v)]
    if isinstance(value, tuple):
        return [_safe_text(v) for v in value if _safe_text(v)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [_safe_text(v) for v in parsed if _safe_text(v)]
            except Exception:
                pass
        return [_safe_text(v) for v in raw.replace("|", ",").split(",") if _safe_text(v)]
    return []


def _journey_label(value: str) -> str:
    key = _safe_text(value).lower()
    return _JOURNEY_KO.get(key, key or "미분류")


def _axis_labels(value: Any) -> list[str]:
    axes = _parse_list_like(value)
    if not axes:
        return ["미분류"]
    labels: list[str] = []
    for axis in axes:
        key = _safe_text(axis)
        labels.append(_AXIS_KO.get(key, key))
    return labels


def _sentiment_label(value: str) -> str:
    lowered = _safe_text(value).lower()
    mapping = {
        "positive": "긍정",
        "negative": "부정",
        "mixed": "혼합",
        "neutral": "중립",
    }
    return mapping.get(lowered, "중립")


def _level_label(value: Any) -> str:
    raw = _safe_text(value)
    return _LEVEL_KO.get(raw, raw or "미분류")


def _action_domains_for_axes(value: Any) -> str:
    axes = _parse_list_like(value)
    if not axes:
        return "operations / service"
    domains: list[str] = []
    for axis in axes:
        domain = _ACTION_DOMAIN_BY_AXIS.get(_safe_text(axis), "operations / service")
        if domain not in domains:
            domains.append(domain)
    return ", ".join(domains[:2])


def _scope_size_for_issue(row: pd.Series, analysis_non_trash: pd.DataFrame) -> int:
    if analysis_non_trash is None or analysis_non_trash.empty:
        return 0
    working = analysis_non_trash.copy()
    level_type = _safe_text(row.get("level_type", "")).lower()
    level_id = _safe_text(row.get("level_id", ""))
    if level_type == "video" and level_id and "video_id" in working.columns:
        working = working[working["video_id"].astype(str).eq(level_id)]
    elif level_type == "product_group" and level_id and "product" in working.columns:
        working = working[working["product"].astype(str).eq(level_id)]
    return int(len(working))


def _compute_issue_ratio(row: pd.Series, analysis_non_trash: pd.DataFrame) -> float:
    denominator = _scope_size_for_issue(row, analysis_non_trash)
    if denominator <= 0:
        return 0.0
    cluster_size = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    if cluster_size <= 0:
        return 0.0
    return float(min(cluster_size / denominator, 1.0))


def _derive_impact_level(row: pd.Series) -> str:
    if bool(row.get("insufficient_evidence", False)):
        return "Low"
    cluster = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    ratio = float(pd.to_numeric(row.get("issue_ratio", 0.0), errors="coerce") or 0.0)
    impact = float(pd.to_numeric(row.get("impact_signal", 0.0), errors="coerce") or 0.0)
    if impact >= 15000 or cluster >= 500 or ratio >= 0.08:
        return "High"
    if impact >= 3000 or cluster >= 120 or ratio >= 0.03:
        return "Medium"
    return "Low"


def _derive_priority_level(row: pd.Series) -> str:
    if bool(row.get("insufficient_evidence", False)):
        return "Low"
    sentiment = _safe_text(row.get("polarity_direction", "neutral")).lower()
    stage = _safe_text(row.get("journey_stage", "")).lower()
    impact_level = _safe_text(row.get("impact_level", "Low"))
    cluster = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    critical_stages = {"maintenance", "usage", "delivery", "onboarding"}

    if sentiment == "negative":
        if impact_level == "High":
            return "High"
        if impact_level == "Medium":
            return "High" if stage in critical_stages else "Medium"
        return "Medium" if stage in critical_stages and cluster >= 30 else "Low"
    if sentiment == "mixed":
        if impact_level == "High":
            return "High"
        if impact_level == "Medium":
            return "Medium"
        return "Low"
    if sentiment == "positive":
        return "Medium" if impact_level == "High" else "Low"
    return "Low"


def _derive_action_label(row: pd.Series) -> str:
    if bool(row.get("insufficient_evidence", False)):
        return "monitor"
    stage = _safe_text(row.get("journey_stage", "")).lower()
    sentiment = _safe_text(row.get("polarity_direction", "neutral")).lower()
    axes = set(_parse_list_like(row.get("judgment_axes", [])))
    cluster = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    ratio = float(pd.to_numeric(row.get("issue_ratio", 0.0), errors="coerce") or 0.0)
    impact = float(pd.to_numeric(row.get("impact_signal", 0.0), errors="coerce") or 0.0)
    high_scale = cluster >= 120 or ratio >= 0.03 or impact >= 5000

    service_axes = {"service_after_sales"}
    product_axes = {
        "maintenance_burden",
        "durability_lifespan",
        "installation_space_fit",
        "design_interior_fit",
        "automation_convenience",
        "usage_pattern_fit",
    }
    marketing_axes = {"brand_trust", "comparison_substitute_judgment"}
    business_axes = {"price_value_fit", "purchase_strategy_contract"}

    if sentiment == "positive":
        return "maintain_strength" if high_scale else "monitor"
    if sentiment == "mixed":
        if stage in {"awareness", "exploration", "purchase", "decision"} and axes.intersection(marketing_axes | business_axes):
            return "marketing_message"
        if stage in {"delivery", "onboarding"} or axes.intersection(service_axes):
            return "service_operation"
        if axes.intersection(product_axes):
            return "product_improvement"
        return "monitor"
    if sentiment == "negative":
        if high_scale and (stage in {"maintenance", "usage", "delivery", "onboarding"} or axes.intersection(service_axes | product_axes)):
            return "immediate_fix"
        if stage in {"delivery", "onboarding"} or axes.intersection(service_axes):
            return "service_operation"
        if axes.intersection(product_axes):
            return "product_improvement"
        if axes.intersection(marketing_axes | business_axes) or stage in {"awareness", "exploration", "purchase", "decision"}:
            return "marketing_message"
        return "monitor"
    return "monitor"


def _derive_action_summary(row: pd.Series) -> str:
    stage_label = _journey_label(_safe_text(row.get("journey_stage", "")))
    axis_labels = _axis_labels(row.get("judgment_axes", []))
    core_problem = axis_labels[0] if axis_labels else "핵심 이슈"
    action_label = _safe_text(row.get("action_label", "monitor"))
    sentiment = _safe_text(row.get("polarity_direction", "neutral")).lower()
    mixed_ratio = float(pd.to_numeric(row.get("mixed_ratio", 0.0), errors="coerce") or 0.0)

    if action_label == "immediate_fix":
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 마찰을 반복 경험하고 있으므로, 즉시 수정·운영 대응 태스크를 우선 배정해야 합니다."
    if action_label == "service_operation":
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 관련 서비스 공백을 겪고 있으므로, 응답 SLA·처리 프로세스 개선을 우선 실행해야 합니다."
    if action_label == "product_improvement":
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 한계를 체감하고 있으므로, 제품 구조·기능 개선 항목을 다음 릴리스 우선순위로 올려야 합니다."
    if action_label == "marketing_message":
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 기대-현실 괴리를 보이므로, 메시지·비교 가이드·기대치 정렬을 우선 보정해야 합니다."
    if action_label == "maintain_strength":
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 강점을 반복 확인하므로, 검증된 강점 메시지와 운영 방식을 확장 유지해야 합니다."
    if sentiment == "mixed" or mixed_ratio > 0.0:
        return f"{stage_label} 단계에서 사용자들이 '{core_problem}'에 대해 기대와 불편을 함께 언급하므로, 추가 근거를 축적하며 조건부 개선안을 병행 검토해야 합니다."
    return f"{stage_label} 단계에서 사용자들이 '{core_problem}' 신호를 보이고 있으므로, 우선 모니터링하며 근거 축적 후 실행 여부를 결정해야 합니다."


def _derive_ranking_score(row: pd.Series) -> float:
    priority_rank = _PRIORITY_RANK.get(_safe_text(row.get("priority_level", "Low")), 1)
    impact_rank = _IMPACT_RANK.get(_safe_text(row.get("impact_level", "Low")), 1)
    cluster = float(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0.0)
    ratio = float(pd.to_numeric(row.get("issue_ratio", 0.0), errors="coerce") or 0.0)
    sentiment = _safe_text(row.get("polarity_direction", "neutral")).lower()
    sentiment_boost = 0.25 if sentiment == "negative" else (0.1 if sentiment == "mixed" else 0.0)
    return (priority_rank * 100.0) + (impact_rank * 25.0) + (cluster * 0.1) + (ratio * 100.0) + sentiment_boost


def _ranking_reason_text(row: pd.Series) -> str:
    ratio_pct = float(pd.to_numeric(row.get("issue_ratio", 0.0), errors="coerce") or 0.0) * 100.0
    cluster_size = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    sentiment = _sentiment_label(_safe_text(row.get("polarity_direction", "neutral")))
    stage = _journey_label(_safe_text(row.get("journey_stage", "")))
    return (
        f"[{stage}] 구간에서 같은 사용자 반응이 반복되어 체감 영향이 커졌고, "
        f"{sentiment} 방향의 마찰이 누적되어 최우선 대응이 필요한 상태입니다. "
        f"(영향 범위 {ratio_pct:.1f}%, 반복 확인 {cluster_size:,}건)"
    )


def _action_domain_ko(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return "운영"
    domain_map = {
        "product": "제품",
        "ux / guide": "UX/가이드",
        "operations / service": "운영/서비스",
        "marketing / message": "마케팅/메시지",
        "business / policy": "사업/정책",
    }
    first = raw.split(",")[0].strip().lower()
    return domain_map.get(first, raw)


def _action_sentence(row: pd.Series) -> str:
    stage_label = _journey_label(_safe_text(row.get("journey_stage", "")))
    axis_labels = _axis_labels(row.get("judgment_axes", []))
    focus_axis = axis_labels[0] if axis_labels else "핵심 이슈"
    action_label = _safe_text(row.get("action_label", "monitor"))

    if action_label == "immediate_fix":
        return f"{stage_label} 단계의 '{focus_axis}' 문제를 즉시 개선해야 한다."
    if action_label == "product_improvement":
        return f"{stage_label} 단계의 '{focus_axis}' 이슈를 제품 개선 우선 과제로 설정해야 한다."
    if action_label == "service_operation":
        return f"{stage_label} 단계의 '{focus_axis}' 문제를 운영/서비스 대응 과제로 즉시 반영해야 한다."
    if action_label == "marketing_message":
        return f"{stage_label} 단계의 '{focus_axis}' 기대-현실 괴리를 메시지 정렬 과제로 우선 대응해야 한다."
    if action_label == "maintain_strength":
        return f"{stage_label} 단계에서 확인된 '{focus_axis}' 강점을 유지·확장하는 실행을 이어가야 한다."
    return f"{stage_label} 단계의 '{focus_axis}' 신호를 모니터링하며 실행 여부를 빠르게 결정해야 한다."


def _decision_confidence_text(row: pd.Series) -> tuple[str, str]:
    insufficient = bool(row.get("insufficient_evidence", False))
    cluster_size = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    if insufficient or cluster_size < 3:
        return "근거 부족", "근거 축적이 더 필요하며 즉시 단정 결론은 보류해야 합니다."
    if cluster_size >= 120:
        return "충분한 근거", "반복 규모가 커서 실행 우선순위를 바로 적용할 수 있습니다."
    return "충분한 근거", "현재 근거 수준에서 실행 우선순위 판단이 가능합니다."


def _get_primary_issue(top_issues: pd.DataFrame) -> pd.Series | None:
    if top_issues.empty:
        return None
    working = top_issues.copy()
    working["_priority_rank"] = working["priority_level"].map(_PRIORITY_RANK).fillna(0)
    working["_impact_rank"] = working["impact_level"].map(_IMPACT_RANK).fillna(0)
    sorted_issues = working.sort_values(
        ["_priority_rank", "_impact_rank", "supporting_cluster_count", "issue_ratio", "priority_signal"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    sorted_issues = sorted_issues.drop(columns=["_priority_rank", "_impact_rank"], errors="ignore")
    return sorted_issues.iloc[0]


def _render_decision_card(issue_row: pd.Series, issue_comments: pd.DataFrame) -> None:
    st.subheader("1) 🔥 주요 1순위 이슈")
    if issue_row is None:
        st.info("현재 필터에서 표시할 전략 이슈가 없습니다.")
        return

    title = _safe_text(issue_row.get("insight_title", "전략 이슈"))
    scope_label = _safe_text(issue_row.get("scope_label", ""))
    level_id = _safe_text(issue_row.get("level_id", ""))
    action_label = _safe_text(issue_row.get("action_label", "monitor"))
    action_label_ko = _safe_text(issue_row.get("action_label_ko", _ACTION_LABEL_KO.get(action_label, action_label)))
    recommended_action = _RECOMMENDED_ACTION_TEXT.get(action_label, "모니터링 및 근거 추가 수집")
    action_domain = _action_domain_ko(issue_row.get("action_domain", "operations / service"))
    priority_text = _level_label(issue_row.get("priority_level", "Low"))
    impact_text = _level_label(issue_row.get("impact_level", "Low"))
    action_sentence = _action_sentence(issue_row)

    st.markdown(f"### {title}")
    st.caption(f"범위: {scope_label} · 식별값: {level_id}")

    st.markdown("#### 👉 지금 해야 할 것")
    st.success(action_sentence)

    st.markdown("#### 📌 왜 중요한가")
    combined_reason = (
        f"{_ranking_reason_text(issue_row)} "
        f"{_safe_text(issue_row.get('insight_summary', '핵심 설명 정보가 없습니다.'))}"
    ).strip()
    st.info(combined_reason)

    stage_label = _journey_label(_safe_text(issue_row.get("journey_stage", "")))
    axis_text = ", ".join(_axis_labels(issue_row.get("judgment_axes", [])))
    ratio_pct = float(pd.to_numeric(issue_row.get("issue_ratio", 0.0), errors="coerce") or 0.0) * 100.0
    cluster_size = int(pd.to_numeric(issue_row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    polarity = _sentiment_label(_safe_text(issue_row.get("polarity_direction", "neutral")))

    st.markdown("#### 📊 핵심 판단 근거")
    st.markdown(
        "\n".join(
            [
                f"- 일시 반응이 아니라 반복적으로 확인되는 핵심 이슈입니다. (반복 확인 {cluster_size:,}건)",
                f"- 제한된 사례가 아니라 사용자 경험 전반에 영향을 주는 사안입니다. (영향 범위 {ratio_pct:.1f}%)",
                f"- 문제는 **{stage_label}** 구간에서 **{polarity}** 방향으로 집중되어 즉시 판단이 필요합니다.",
                f"- 판단 축은 **{axis_text if axis_text else '미분류'}**이며 현재 우선순위/영향도는 **{priority_text}/{impact_text}**입니다.",
            ]
        )
    )

    st.markdown("#### 🧭 실행 방향")
    st.write(f"이 이슈는 **{action_label_ko}** 유형으로, **{action_domain}** 영역에서 **{recommended_action}**을 실행해야 합니다.")

    confidence_title, confidence_detail = _decision_confidence_text(issue_row)
    st.markdown("#### 📍 신뢰도")
    if confidence_title == "근거 부족":
        st.warning(f"{confidence_title}: {confidence_detail}")
    else:
        st.success(f"{confidence_title}: {confidence_detail}")

    if issue_comments.empty:
        st.warning("이 이슈와 직접 연결된 댓글 근거가 현재 필터에서 부족합니다.")


def _render_secondary_issues(top_issues: pd.DataFrame, primary_issue_key: str) -> None:
    st.subheader("2) 🧩 함께 고려할 이슈")
    secondary = top_issues[top_issues["issue_key"].astype(str).ne(_safe_text(primary_issue_key))].copy()
    if secondary.empty:
        st.caption("현재 보조 이슈가 없습니다.")
        return
    secondary = secondary.head(5).copy()
    with st.expander("함께 고려할 이슈 상위 5개", expanded=False):
        for idx, (_, row) in enumerate(secondary.iterrows(), start=1):
            issue_title = _safe_text(row.get("insight_title", "보조 이슈"))
            action_text = _safe_text(row.get("action_label_ko", "모니터링"))
            priority_text = _level_label(row.get("priority_level", "Low"))
            impact_text = _level_label(row.get("impact_level", "Low"))
            stage_text = _journey_label(_safe_text(row.get("journey_stage", "")))
            st.markdown(
                f"**{idx}. {issue_title}**  \n"
                f"- {stage_text} 구간에서 **{action_text}** 방향으로 대응해야 한다. (우선순위 {priority_text}, 영향도 {impact_text})"
            )


def _render_all_issues_table(top_issues: pd.DataFrame) -> None:
    st.subheader("4) 📋 전체 이슈(참고)")
    with st.expander("전체 이슈 보기(참고/탐색)", expanded=False):
        if top_issues.empty:
            st.caption("표시할 이슈가 없습니다.")
            return
        for idx, (_, row) in enumerate(top_issues.head(20).iterrows(), start=1):
            issue_title = _safe_text(row.get("insight_title", "이슈"))
            action_label = _safe_text(row.get("action_label_ko", "모니터링"))
            stage = _journey_label(_safe_text(row.get("journey_stage", "")))
            sentiment = _sentiment_label(_safe_text(row.get("polarity_direction", "neutral")))
            st.markdown(f"**{idx}. {issue_title}**")
            st.caption(f"{stage} 구간 · {sentiment} 반응 · {action_label} 방향")


def _apply_entry_search(top_issues: pd.DataFrame, search_query: str) -> pd.DataFrame:
    if top_issues is None or top_issues.empty:
        return pd.DataFrame() if top_issues is None else top_issues
    query = _safe_text(search_query).strip().lower()
    if not query:
        return top_issues

    terms = [term for term in query.replace(",", " ").split() if term]
    if not terms:
        return top_issues

    def _row_blob(row: pd.Series) -> str:
        parts = [
            _safe_text(row.get("insight_title")),
            _safe_text(row.get("insight_summary")),
            _safe_text(row.get("action_label_ko")),
            _safe_text(row.get("level_id")),
            _journey_label(_safe_text(row.get("journey_stage"))),
            ", ".join(_axis_labels(row.get("judgment_axes"))),
        ]
        return " ".join(parts).lower()

    blobs = top_issues.apply(_row_blob, axis=1)
    mask = pd.Series(True, index=top_issues.index)
    for term in terms:
        mask = mask & blobs.str.contains(term, regex=False)
    return top_issues[mask].reset_index(drop=True)


def _reset_sidebar_filters(options: dict[str, list[str]]) -> None:
    st.session_state["filter_products"] = list(options.get("products", []))
    st.session_state["filter_regions"] = list(options.get("regions", []))
    st.session_state["filter_brands"] = list(options.get("brands", []))
    st.session_state["filter_sentiments"] = list(options.get("sentiments", []))
    st.session_state["filter_cej"] = list(options.get("cej", []))
    st.session_state["filter_keyword_query"] = ""
    st.session_state["filter_keyword_mode"] = "AND"
    st.session_state["filter_keyword_mode_label"] = "AND(모두 포함)"
    st.session_state[_FORCE_WIDE_SCOPE_KEY] = False


def _render_data_controls(active_mode: str, active_snapshot_run: str) -> None:
    with st.sidebar:
        st.markdown("### 결과 모드")
        st.caption(f"현재 모드: {'예시 모드' if active_mode == 'sample' else '실행 결과 모드'}")
        st.caption("결과가 보이지 않으면 `예시 보기`로 화면을 확인하세요.")
        col_sample, col_real = st.columns(2)
        with col_sample:
            if st.button("예시 보기", width="stretch", type="secondary", key="v2_load_sample"):
                legacy._set_data_mode("sample", force_refresh=False)
                st.rerun()
        with col_real:
            if st.button("실행 결과 새로고침", width="stretch", type="primary", key="v2_refresh_real"):
                legacy._set_data_mode("real", force_refresh=True, snapshot_run_id=active_snapshot_run)
                st.rerun()

        if active_mode == "real":
            snapshot_options = legacy._build_run_snapshot_options()
            snapshot_value_to_label = {value: label for label, value in snapshot_options}
            snapshot_values = list(snapshot_value_to_label.keys())
            selected_snapshot = active_snapshot_run if active_snapshot_run in snapshot_values else legacy.RUN_SNAPSHOT_ALL
            selected_index = snapshot_values.index(selected_snapshot) if selected_snapshot in snapshot_values else 0
            selected_snapshot = st.selectbox(
                "실행 결과 스냅샷",
                options=snapshot_values,
                index=selected_index,
                format_func=lambda value: snapshot_value_to_label.get(value, value),
                help="전체 실행 통합 보기 또는 특정 실행만 고정해 불러올 수 있습니다.",
                key="v2_snapshot_select",
            )
            if st.button("선택 스냅샷 불러오기", width="stretch", type="secondary", key="v2_snapshot_apply"):
                legacy._set_data_mode("real", force_refresh=False, snapshot_run_id=selected_snapshot)
                st.rerun()
            if active_snapshot_run != legacy.RUN_SNAPSHOT_ALL:
                st.caption(f"현재 고정 실행 식별값: `{active_snapshot_run}`")
            st.caption("선택한 스냅샷은 로컬에 저장되어 다음 실행 시에도 자동 복원됩니다.")


def _apply_pending_filter_reset_if_needed(options: dict[str, list[str]]) -> None:
    if not bool(st.session_state.get(_PENDING_RESET_FILTERS_KEY, False)):
        return
    _reset_sidebar_filters(options)
    st.session_state[_PENDING_RESET_FILTERS_KEY] = False


def _prepare_strategy_frame(frame: pd.DataFrame, level_type: str) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    working["level_type"] = level_type
    working["priority_signal"] = pd.to_numeric(working.get("priority_signal", 0), errors="coerce").fillna(0.0)
    working["supporting_cluster_count"] = pd.to_numeric(
        working.get("supporting_cluster_count", 0), errors="coerce"
    ).fillna(0).astype(int)
    working["repetition_signal"] = pd.to_numeric(working.get("repetition_signal", 0), errors="coerce").fillna(0.0)
    working["impact_signal"] = pd.to_numeric(working.get("impact_signal", 0), errors="coerce").fillna(0.0)
    working["insufficient_evidence"] = working.get("insufficient_evidence", False).fillna(False).astype(bool)
    working["journey_stage"] = working.get("journey_stage", "").fillna("").astype(str)
    working["judgment_axes_list"] = working.get("judgment_axes", "").apply(_parse_list_like)
    working["supporting_representatives_list"] = working.get("supporting_representatives", "").apply(_parse_list_like)
    working["scope_label"] = "영상" if level_type == "video" else "제품군"
    return working


def _axis_match(series: pd.Series, axis_value: str) -> pd.Series:
    axis_clean = _safe_text(axis_value)
    if not axis_clean:
        return pd.Series(True, index=series.index)
    return series.apply(lambda value: axis_clean in _parse_list_like(value))


def _infer_trend_state(row: pd.Series, analysis_non_trash: pd.DataFrame) -> str:
    if bool(row.get("insufficient_evidence", False)):
        return "근거 부족"
    if analysis_non_trash is None or analysis_non_trash.empty:
        return "근거 부족"
    stage = _safe_text(row.get("journey_stage", "")).lower()
    polarity = _safe_text(row.get("polarity_direction", "")).lower()
    axis_list = _parse_list_like(row.get("judgment_axes", []))
    primary_axis = axis_list[0] if axis_list else ""

    working = analysis_non_trash.copy()
    if "journey_stage" in working.columns and stage:
        working = working[working["journey_stage"].fillna("").astype(str).str.lower().eq(stage)]
    if "sentiment_final" in working.columns and polarity:
        if polarity == "mixed":
            mixed_series = working.get("mixed_flag", False).fillna(False).astype(bool)
            working = working[mixed_series]
        else:
            working = working[working["sentiment_final"].fillna("").astype(str).str.lower().eq(polarity)]
    if primary_axis and "judgment_axes" in working.columns:
        working = working[_axis_match(working["judgment_axes"], primary_axis)]
    if working.empty:
        return "근거 부족"

    date_source = working.get("published_at", working.get(legacy.COL_WRITTEN_AT, pd.Series(dtype=object)))
    dt_series = pd.to_datetime(date_source, errors="coerce")
    dt_series = dt_series.dropna()
    if dt_series.empty:
        return "근거 부족"

    end = dt_series.max()
    recent_start = end - pd.Timedelta(days=14)
    prev_start = recent_start - pd.Timedelta(days=14)
    recent = int(((dt_series > recent_start) & (dt_series <= end)).sum())
    previous = int(((dt_series > prev_start) & (dt_series <= recent_start)).sum())

    if recent + previous < 6:
        return "근거 부족"
    if previous == 0 and recent >= 3:
        return "신규"
    if previous == 0:
        return "근거 부족"
    if recent >= previous * 1.2 and (recent - previous) >= 2:
        return "상승"
    if recent <= previous * 0.8 and (previous - recent) >= 2:
        return "완화"
    return "지속"


def _filter_strategy_frames(
    strategy_video: pd.DataFrame,
    strategy_product: pd.DataFrame,
    filtered_videos: pd.DataFrame,
    filtered_comments: pd.DataFrame,
    selected_filters: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_filtered = strategy_video.copy()
    product_filtered = strategy_product.copy()

    if not video_filtered.empty and "video_id" in filtered_videos.columns:
        allowed_video_ids = set(filtered_videos["video_id"].dropna().astype(str).tolist())
        if allowed_video_ids:
            video_filtered = video_filtered[video_filtered["level_id"].astype(str).isin(allowed_video_ids)]

    products = list(selected_filters.get("products") or [])
    products_active = bool(selected_filters.get("products_active", bool(products)))
    if products_active and not product_filtered.empty:
        product_filtered = product_filtered[product_filtered["level_id"].astype(str).isin(set(products))]
    elif not product_filtered.empty and "product" in filtered_comments.columns:
        allowed_products = set(filtered_comments["product"].dropna().astype(str).tolist())
        if allowed_products:
            product_filtered = product_filtered[product_filtered["level_id"].astype(str).isin(allowed_products)]

    sentiments = list(selected_filters.get("sentiments") or [])
    sentiments_active = bool(selected_filters.get("sentiments_active", bool(sentiments)))
    if sentiments_active:
        allowed_sentiment_labels = set(sentiments)

        def _sentiment_match(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return df
            localized = df.get("polarity_direction", pd.Series("", index=df.index)).fillna("").astype(str).map(
                lambda value: legacy.localize_sentiment(value.lower())
            )
            return df[localized.isin(allowed_sentiment_labels)]

        video_filtered = _sentiment_match(video_filtered)
        product_filtered = _sentiment_match(product_filtered)

    return video_filtered.reset_index(drop=True), product_filtered.reset_index(drop=True)


def _compose_top_issue_table(
    strategy_video: pd.DataFrame,
    strategy_product: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
) -> pd.DataFrame:
    merged = pd.concat([strategy_product, strategy_video], ignore_index=True) if not strategy_product.empty or not strategy_video.empty else pd.DataFrame()
    if merged.empty:
        return merged

    merged = merged.copy()
    merged["journey_stage_label"] = merged["journey_stage"].map(_journey_label)
    merged["judgment_axis_label"] = merged["judgment_axes"].apply(lambda value: ", ".join(_axis_labels(value)))
    merged["sentiment_label"] = merged.get("polarity_direction", "").map(_sentiment_label)
    merged["issue_state"] = merged.apply(lambda row: _infer_trend_state(row, analysis_non_trash), axis=1)
    merged["action_domain"] = merged.get("judgment_axes", "").apply(_action_domains_for_axes)
    merged["issue_ratio"] = merged.apply(lambda row: _compute_issue_ratio(row, analysis_non_trash), axis=1)
    merged["impact_level"] = merged.apply(_derive_impact_level, axis=1)
    merged["priority_level"] = merged.apply(_derive_priority_level, axis=1)
    merged["action_label"] = merged.apply(_derive_action_label, axis=1)
    merged["action_summary"] = merged.apply(_derive_action_summary, axis=1)
    merged["action_label_ko"] = merged["action_label"].map(lambda value: _ACTION_LABEL_KO.get(_safe_text(value), _safe_text(value)))
    merged["issue_key"] = merged.apply(
        lambda row: f"{_safe_text(row.get('scope_label'))}|{_safe_text(row.get('level_id'))}|{_safe_text(row.get('journey_stage'))}|{_safe_text(row.get('judgment_axes'))}|{_safe_text(row.get('polarity_direction'))}",
        axis=1,
    )
    merged["ranking_score"] = merged.apply(_derive_ranking_score, axis=1)
    merged["_priority_rank"] = merged["priority_level"].map(_PRIORITY_RANK).fillna(0)
    merged["_impact_rank"] = merged["impact_level"].map(_IMPACT_RANK).fillna(0)
    merged = merged.sort_values(
        ["_priority_rank", "_impact_rank", "supporting_cluster_count", "issue_ratio", "priority_signal"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    merged = merged.drop(columns=["_priority_rank", "_impact_rank"], errors="ignore")
    return merged


def _filter_comments_for_issue(issue_row: pd.Series, analysis_non_trash: pd.DataFrame) -> pd.DataFrame:
    if analysis_non_trash.empty:
        return pd.DataFrame()
    working = analysis_non_trash.copy()
    level_type = _safe_text(issue_row.get("level_type"))
    level_id = _safe_text(issue_row.get("level_id"))
    stage = _safe_text(issue_row.get("journey_stage")).lower()
    polarity = _safe_text(issue_row.get("polarity_direction")).lower()
    axes = _parse_list_like(issue_row.get("judgment_axes"))
    primary_axis = axes[0] if axes else ""

    if level_type == "video" and level_id and "video_id" in working.columns:
        working = working[working["video_id"].astype(str).eq(level_id)]
    if level_type == "product_group":
        if "product" in working.columns and level_id:
            working = working[working["product"].astype(str).eq(level_id)]
    if stage and "journey_stage" in working.columns:
        working = working[working["journey_stage"].fillna("").astype(str).str.lower().eq(stage)]
    if polarity and "sentiment_final" in working.columns:
        if polarity == "mixed":
            working = working[working.get("mixed_flag", False).fillna(False).astype(bool)]
        else:
            mixed_mask = working.get("mixed_flag", False).fillna(False).astype(bool)
            working = working[working["sentiment_final"].fillna("").astype(str).str.lower().eq(polarity) & ~mixed_mask]
    if primary_axis and "judgment_axes" in working.columns:
        working = working[_axis_match(working["judgment_axes"], primary_axis)]
    return working.reset_index(drop=True)


def _render_evidence_drilldown(
    issue_row: pd.Series,
    issue_comments: pd.DataFrame,
    representative_comments: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
) -> None:
    st.subheader("3) 🔎 이 판단의 근거")
    if issue_row is None:
        st.info("현재 필터에서 연결할 근거가 없습니다.")
        return

    with st.expander("근거 보기", expanded=False):
        st.caption("아래 내용은 현재 결정이 타당한지 확인하기 위한 대표 사례 중심 근거입니다.")
        rep_ids = _parse_list_like(issue_row.get("supporting_representatives", []))
        rep_pool = representative_comments.copy()
        if not rep_pool.empty and "comment_id" in rep_pool.columns and rep_ids:
            rep_pool = rep_pool[rep_pool["comment_id"].astype(str).isin(rep_ids)].copy()
        if rep_pool.empty:
            rep_pool = issue_comments.copy()
        rep_pool = rep_pool.sort_values(
            by=[col for col in ["like_count", legacy.COL_LIKES, "published_at"] if col in rep_pool.columns],
            ascending=False,
            na_position="last",
        ) if not rep_pool.empty else rep_pool

        if rep_pool.empty:
            st.caption("대표 근거 댓글이 없습니다.")
        else:
            st.markdown("#### 대표 근거 3개")
            similar_evidence_lines: list[str] = []
            similar_examples: list[str] = []
            for idx, (_, row) in enumerate(rep_pool.head(3).iterrows(), start=1):
                comment_id = _safe_text(row.get("comment_id", f"row_{idx}"))
                sentiment = _safe_text(row.get("sentiment_final", row.get("sentiment_label", "neutral"))).lower()
                body = _safe_text(row.get("text_display", row.get(legacy.COL_ORIGINAL, row.get("cleaned_text", ""))))
                with st.container(border=True):
                    st.markdown(f"**#{idx} · 감성 `{_sentiment_label(sentiment)}` · 댓글 식별값 `{comment_id}`**")
                    st.write(body)

                    similar_df, similar_meta = legacy._build_canonical_similar_comments(
                        analysis_non_trash=analysis_non_trash,
                        selected=row,
                        working_selected=row,
                        sentiment_name=sentiment,
                        rep_id=comment_id,
                    )
                    similar_count = int(similar_meta.get("cluster_size", 0) or 0)
                    insufficient = bool(similar_meta.get("insufficient_evidence", True))
                    if insufficient:
                        similar_evidence_lines.append(f"- 대표 근거 #{idx}와 같은 반응은 반복 확인이 아직 충분하지 않습니다.")
                    else:
                        similar_evidence_lines.append(
                            f"- 대표 근거 #{idx}와 같은 반응은 구조적으로 반복되는 패턴입니다. (반복 확인 {similar_count:,}건)"
                        )
                        if not similar_df.empty:
                            for _, sample_row in similar_df.head(2).iterrows():
                                sample_text = _safe_text(sample_row.get("text_display", sample_row.get("cleaned_text", "")))
                                if sample_text:
                                    similar_examples.append(sample_text)

            st.markdown("#### 반복 근거 요약")
            if similar_evidence_lines:
                st.markdown("\n".join(similar_evidence_lines))
            else:
                st.caption("표시할 반복 근거가 없습니다.")

            if similar_examples:
                with st.expander("반복 반응 일부 보기", expanded=False):
                    for idx, text in enumerate(similar_examples[:8], start=1):
                        st.markdown(f"{idx}. {text}")

        st.markdown("#### 원천 댓글(최하단)")
        if issue_comments.empty:
            st.caption("표시할 댓글이 없습니다.")
            return
        comment_display = pd.DataFrame()
        comment_display["댓글 식별값"] = issue_comments.get("comment_id", "")
        comment_display["감성"] = issue_comments.get("sentiment_final", issue_comments.get("sentiment_label", ""))
        comment_display["고객 여정"] = issue_comments.get("journey_stage", "").map(_journey_label)
        comment_display["판단 축"] = issue_comments.get("judgment_axes", "").apply(lambda value: ", ".join(_axis_labels(value)))
        comment_display["문의"] = issue_comments.get("inquiry_flag", False).fillna(False).astype(bool)
        comment_display["원문"] = issue_comments.get("text_display", issue_comments.get("cleaned_text", ""))
        comment_display["좋아요"] = pd.to_numeric(issue_comments.get("like_count", issue_comments.get(legacy.COL_LIKES, 0)), errors="coerce").fillna(0).astype(int)
        with st.expander("원천 댓글 보기(최하단)", expanded=False):
            st.dataframe(comment_display.head(120), width="stretch", hide_index=True)
            csv_bytes = comment_display.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "근거 댓글 파일 다운로드",
                data=csv_bytes,
                file_name="strategy_issue_evidence_comments.csv",
                mime="text/csv",
                type="secondary",
                key="v2_download_evidence_comments",
            )


def _load_context() -> DashboardContext | None:
    legacy._initialize_data_session_state()
    active_mode = st.session_state.get(legacy.SESSION_DATA_MODE_KEY, "real")
    active_snapshot_run = _safe_text(st.session_state.get(legacy.SESSION_RUN_SNAPSHOT_KEY, legacy.RUN_SNAPSHOT_ALL)) or legacy.RUN_SNAPSHOT_ALL
    expected_signature = legacy._expected_mode_signature(active_mode, snapshot_run_id=active_snapshot_run)
    if st.session_state.get(legacy.SESSION_DATA_SIGNATURE_KEY) != expected_signature:
        with st.spinner("최신 결과와 동기화 중..."):
            legacy._set_data_mode(active_mode, force_refresh=True, snapshot_run_id=active_snapshot_run)

    _render_data_controls(active_mode, active_snapshot_run)
    force_wide_scope = bool(st.session_state.get(_FORCE_WIDE_SCOPE_KEY, False))

    data = st.session_state.get(legacy.SESSION_DATA_BUNDLE_KEY, legacy._empty_dashboard_bundle())
    comments_raw = data.get("comments", pd.DataFrame()).copy()
    videos_raw = data.get("videos", pd.DataFrame()).copy()
    if active_mode == "real" and (comments_raw.empty or videos_raw.empty):
        st.warning(legacy._build_missing_data_message())
        st.info("좌측 사이드바의 `예시 보기`를 눌러 예시 모드로 화면을 빠르게 확인할 수 있습니다.")
        return None

    comments_scored, videos_df = legacy.ensure_lg_relevance_columns(comments_raw, videos_raw)
    comments_df = legacy.add_localized_columns(comments_scored)
    if comments_df.empty:
        st.warning("표시할 결과가 없습니다. 먼저 내용을 수집해주세요.")
        return None
    videos_df[legacy.COL_COUNTRY] = videos_df.get("region", pd.Series(dtype=str)).map(legacy.localize_region)

    analysis_non_trash_raw = data.get("analysis_non_trash", pd.DataFrame()).copy()
    if analysis_non_trash_raw.empty:
        analysis_comments_raw = data.get("analysis_comments", pd.DataFrame()).copy()
        if not analysis_comments_raw.empty:
            included = analysis_comments_raw.get("analysis_included", pd.Series(True, index=analysis_comments_raw.index)).fillna(False).astype(bool)
            hygiene = analysis_comments_raw.get("hygiene_class", pd.Series("strategic", index=analysis_comments_raw.index)).fillna("").astype(str).str.lower()
            analysis_non_trash_raw = analysis_comments_raw[included & hygiene.ne("trash")].copy()
    if not analysis_non_trash_raw.empty:
        analysis_non_trash_raw = legacy.add_localized_columns(analysis_non_trash_raw)

    dashboard_options = legacy._build_dashboard_options(comments_df, videos_df)
    _apply_pending_filter_reset_if_needed(dashboard_options)
    selected_filters = legacy._render_sidebar_filters(dashboard_options)
    search_query = _safe_text(selected_filters.get("keyword_query", "")).strip()
    bundle = legacy.compute_filtered_bundle(
        comments_df,
        videos_df,
        data.get("quality_summary", pd.DataFrame()),
        selected_filters,
        data.get("representative_bundles", pd.DataFrame()),
        data.get("opinion_units", pd.DataFrame()),
        analysis_non_trash_raw,
    )

    filtered_comments = bundle.get("all_comments", bundle.get("comments", pd.DataFrame())).copy()
    filtered_videos = bundle.get("videos", pd.DataFrame()).copy()
    analysis_non_trash = bundle.get("analysis_non_trash", pd.DataFrame()).copy()
    representative_comments = bundle.get("representative_comments", pd.DataFrame()).copy()

    strategy_video = _prepare_strategy_frame(data.get("strategy_video_insights", pd.DataFrame()).copy(), "video")
    strategy_product = _prepare_strategy_frame(data.get("strategy_product_group_insights", pd.DataFrame()).copy(), "product_group")
    strategy_video_full = strategy_video.copy()
    strategy_product_full = strategy_product.copy()
    strategy_video, strategy_product = _filter_strategy_frames(
        strategy_video,
        strategy_product,
        filtered_videos,
        filtered_comments,
        selected_filters,
    )
    top_issues = _compose_top_issue_table(strategy_video, strategy_product, analysis_non_trash)
    top_issues = _apply_entry_search(top_issues, search_query)

    top_issues_fallback = _compose_top_issue_table(strategy_video_full, strategy_product_full, analysis_non_trash_raw)
    top_issues_fallback = _apply_entry_search(top_issues_fallback, search_query)
    if force_wide_scope and not top_issues_fallback.empty:
        top_issues = top_issues_fallback.copy()

    return DashboardContext(
        active_mode=active_mode,
        active_snapshot_run=active_snapshot_run,
        bundle=bundle,
        raw_bundle=data,
        strategy_video=strategy_video,
        strategy_product=strategy_product,
        top_issues=top_issues,
        top_issues_fallback=top_issues_fallback,
        filtered_comments=filtered_comments,
        filtered_videos=filtered_videos,
        analysis_non_trash=analysis_non_trash,
        analysis_non_trash_full=analysis_non_trash_raw,
        representative_comments=representative_comments,
        dashboard_options=dashboard_options,
        search_query=search_query,
        force_wide_scope=force_wide_scope,
    )


def _render_entry_recovery_panel(
    *,
    has_fallback: bool,
    search_query: str,
    dashboard_options: dict[str, list[str]],
) -> None:
    st.warning("현재 조건에서는 결과가 없습니다.")
    st.caption("조건이 너무 좁거나 검색어가 과도하게 제한적일 수 있습니다. 아래에서 바로 복구할 수 있습니다.")
    if search_query:
        st.caption(f"현재 검색어: `{search_query}`")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("필터 초기화", width="stretch", type="secondary", key="v2_recover_reset_filters"):
            st.session_state[_PENDING_RESET_FILTERS_KEY] = True
            st.rerun()
    with c2:
        if st.button("예시 보기", width="stretch", type="secondary", key="v2_recover_sample_mode"):
            st.session_state[_FORCE_WIDE_SCOPE_KEY] = False
            legacy._set_data_mode("sample", force_refresh=False)
            st.rerun()
    with c3:
        if st.button("최근 주요 이슈 보기", width="stretch", type="primary", key="v2_recover_wide_scope"):
            st.session_state[_FORCE_WIDE_SCOPE_KEY] = True
            st.rerun()
    if not has_fallback:
        st.caption("최근 주요 이슈가 바로 없으면, 넓은 범위로 재시도해 복구 경로를 제공합니다.")


def main() -> None:
    st.set_page_config(page_title="가전 VoC 전략 대시보드 v2", layout="wide")
    legacy.apply_theme()

    ctx = _load_context()
    if ctx is None:
        return

    st.markdown(
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
        '<div style="font-size:28px; font-weight:800; color:#0f172a; letter-spacing:-0.02em;">가전 VoC 전략 대시보드 v2</div>'
        '<span style="display:inline-block; padding:4px 11px; border-radius:999px; background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700;">전략 우선</span>'
        f'<span style="display:inline-block; padding:4px 11px; border-radius:999px; background:{("#dcfce7" if ctx.active_mode == "real" else "#ffedd5")}; color:{("#166534" if ctx.active_mode == "real" else "#9a3412")}; font-size:11px; font-weight:700;">{"실행 결과" if ctx.active_mode == "real" else "예시"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "전략 이슈를 먼저 확인하고, 우선순위와 액션 방향을 정한 뒤, 필요할 때만 대표 댓글/원천 댓글을 근거로 내려가도록 정보 구조를 재설계한 레이어입니다."
    )

    display_issues = ctx.top_issues.copy()
    fallback_applied = False

    if display_issues.empty and not ctx.top_issues_fallback.empty:
        fallback_applied = True
        display_issues = ctx.top_issues_fallback.copy()
        st.info("현재 조건 결과가 없어, 가장 넓은 범위의 최근 주요 이슈를 대신 표시합니다.")

    if display_issues.empty:
        _render_entry_recovery_panel(
            has_fallback=not ctx.top_issues_fallback.empty,
            search_query=ctx.search_query,
            dashboard_options=ctx.dashboard_options,
        )
        return

    if ctx.force_wide_scope or fallback_applied:
        st.info("넓은 범위 주요 이슈 보기 모드입니다. 필요 시 필터 초기화로 일반 모드로 돌아갈 수 있습니다.")

    primary_issue = _get_primary_issue(display_issues)
    if primary_issue is None:
        _render_entry_recovery_panel(
            has_fallback=not ctx.top_issues_fallback.empty,
            search_query=ctx.search_query,
            dashboard_options=ctx.dashboard_options,
        )
        return

    analysis_source = ctx.analysis_non_trash_full if (fallback_applied or ctx.force_wide_scope) else ctx.analysis_non_trash
    issue_comments = _filter_comments_for_issue(primary_issue, analysis_source)

    _render_decision_card(primary_issue, issue_comments)
    if primary_issue is not None:
        _render_secondary_issues(display_issues, _safe_text(primary_issue.get("issue_key", "")))

    st.divider()
    _render_evidence_drilldown(
        primary_issue,
        issue_comments,
        ctx.representative_comments,
        analysis_source,
    )

    st.divider()
    _render_all_issues_table(display_issues)

    with st.expander("레거시 대시보드 안내", expanded=False):
        st.caption("기존 검증/레거시 화면은 `streamlit run streamlit_app_legacy.py`로 그대로 사용할 수 있습니다.")


if __name__ == "__main__":
    main()
