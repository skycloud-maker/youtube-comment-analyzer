"""Integrated strategy page renderer for legacy dashboard (Page 3)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class StrategyPageContext:
    selected_filters: dict[str, Any]
    active_mode: str
    active_snapshot_run: str
    filtered_comments: pd.DataFrame
    all_comments: pd.DataFrame
    filtered_videos: pd.DataFrame
    analysis_non_trash: pd.DataFrame
    representative_comments: pd.DataFrame
    strategy_video_insights: pd.DataFrame
    strategy_product_group_insights: pd.DataFrame
    strategy_video_insight_candidates: pd.DataFrame
    strategy_product_group_insight_candidates: pd.DataFrame
    selected_video_id: str = ""


def build_strategy_context(
    *,
    selected_filters: dict[str, Any],
    active_mode: str,
    active_snapshot_run: str,
    filtered_comments: pd.DataFrame,
    all_comments: pd.DataFrame,
    filtered_videos: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
    representative_comments: pd.DataFrame,
    strategy_video_insights: pd.DataFrame,
    strategy_product_group_insights: pd.DataFrame,
    strategy_video_insight_candidates: pd.DataFrame | None = None,
    strategy_product_group_insight_candidates: pd.DataFrame | None = None,
    selected_video_id: str = "",
) -> StrategyPageContext:
    return StrategyPageContext(
        selected_filters=selected_filters or {},
        active_mode=str(active_mode or "real"),
        active_snapshot_run=str(active_snapshot_run or ""),
        filtered_comments=filtered_comments.copy() if filtered_comments is not None else pd.DataFrame(),
        all_comments=all_comments.copy() if all_comments is not None else pd.DataFrame(),
        filtered_videos=filtered_videos.copy() if filtered_videos is not None else pd.DataFrame(),
        analysis_non_trash=analysis_non_trash.copy() if analysis_non_trash is not None else pd.DataFrame(),
        representative_comments=representative_comments.copy() if representative_comments is not None else pd.DataFrame(),
        strategy_video_insights=strategy_video_insights.copy() if strategy_video_insights is not None else pd.DataFrame(),
        strategy_product_group_insights=strategy_product_group_insights.copy() if strategy_product_group_insights is not None else pd.DataFrame(),
        strategy_video_insight_candidates=strategy_video_insight_candidates.copy() if strategy_video_insight_candidates is not None else pd.DataFrame(),
        strategy_product_group_insight_candidates=strategy_product_group_insight_candidates.copy() if strategy_product_group_insight_candidates is not None else pd.DataFrame(),
        selected_video_id=str(selected_video_id or "").strip(),
    )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _parse_list_like(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(v) for v in value if _safe_text(v)]
    if isinstance(value, tuple):
        return [_safe_text(v) for v in value if _safe_text(v)]
    raw = _safe_text(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [_safe_text(v) for v in parsed if _safe_text(v)]
        except Exception:
            pass
    return [_safe_text(item) for item in raw.replace("|", ",").split(",") if _safe_text(item)]


def _scope_summary_line(values: list[str], *, empty_label: str = "전체") -> str:
    cleaned = [_safe_text(v) for v in values if _safe_text(v)]
    if not cleaned:
        return empty_label
    if len(cleaned) <= 3:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:3])} 외 {len(cleaned)-3}개"


def _issue_product_scope(row: pd.Series) -> str:
    level_type = _safe_text(row.get("level_type")).lower()
    if level_type == "product_group":
        return _safe_text(row.get("level_id"))
    return _safe_text(row.get("product_group"))


def _filtered_primary_candidates(top_issues: pd.DataFrame, selected_filters: dict[str, Any]) -> pd.DataFrame:
    if top_issues.empty:
        return top_issues
    working = top_issues.copy()
    products_active = bool(selected_filters.get("products_active", False))
    selected_products = {_safe_text(v) for v in selected_filters.get("products", []) if _safe_text(v)}
    if products_active and selected_products:
        product_mask = working.apply(lambda row: _issue_product_scope(row) in selected_products, axis=1)
        working = working[product_mask].copy()
    return working.reset_index(drop=True)


def _build_top_issues(context: StrategyPageContext) -> pd.DataFrame:
    from src import dashboard_v2 as strategy_v2

    strategy_video = strategy_v2._prepare_strategy_frame(context.strategy_video_insights.copy(), "video")
    strategy_product = strategy_v2._prepare_strategy_frame(context.strategy_product_group_insights.copy(), "product_group")
    strategy_video, strategy_product = strategy_v2._filter_strategy_frames(
        strategy_video,
        strategy_product,
        context.filtered_videos,
        context.filtered_comments,
        context.selected_filters,
    )
    top_issues = strategy_v2._compose_top_issue_table(strategy_video, strategy_product, context.analysis_non_trash)
    top_issues = _filtered_primary_candidates(top_issues, context.selected_filters)
    return top_issues.reset_index(drop=True)


def _prioritize_by_selected_video(top_issues: pd.DataFrame, selected_video_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    info = {
        "selected_video_id": _safe_text(selected_video_id),
        "selected_video_matched": False,
        "selected_video_match_count": 0,
        "scope_mode": "filtered_scope",
    }
    if top_issues.empty or not info["selected_video_id"]:
        return top_issues, info

    working = top_issues.copy()
    video_mask = (
        working.get("level_type", pd.Series("", index=working.index)).astype(str).str.lower().eq("video")
        & working.get("level_id", pd.Series("", index=working.index)).astype(str).eq(info["selected_video_id"])
    )
    match_count = int(video_mask.sum())
    info["selected_video_match_count"] = match_count
    if match_count <= 0:
        info["scope_mode"] = "filtered_scope_no_video_match"
        return working.reset_index(drop=True), info

    info["selected_video_matched"] = True
    info["scope_mode"] = "selected_video_priority"
    focused = pd.concat([working[video_mask], working[~video_mask]], ignore_index=True)
    return focused.reset_index(drop=True), info


def _pick_primary_issue(top_issues: pd.DataFrame, analysis_non_trash: pd.DataFrame) -> tuple[pd.Series | None, pd.DataFrame, pd.DataFrame]:
    if top_issues.empty:
        return None, pd.DataFrame(), top_issues

    from src import dashboard_v2 as strategy_v2

    working = top_issues.copy()
    valid_indices: list[int] = []
    issue_comments_by_idx: dict[int, pd.DataFrame] = {}
    for idx, row in working.iterrows():
        issue_comments = strategy_v2._filter_comments_for_issue(row, analysis_non_trash)
        issue_comments_by_idx[idx] = issue_comments
        cluster_size = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
        if cluster_size > 0 and not issue_comments.empty:
            valid_indices.append(idx)

    if valid_indices:
        valid = working.iloc[valid_indices].reset_index(drop=True)
        first_original_idx = valid_indices[0]
        primary = working.iloc[first_original_idx]
        return primary, issue_comments_by_idx.get(first_original_idx, pd.DataFrame()), valid

    downgraded = working.copy()
    downgraded.loc[:, "insufficient_evidence"] = True
    first = downgraded.iloc[0]
    return first, pd.DataFrame(), downgraded


def _journey_label(issue: pd.Series) -> str:
    return _safe_text(issue.get("journey_stage_label")) or _safe_text(issue.get("journey_stage")) or "미분류"


def _axis_label(issue: pd.Series) -> str:
    return _safe_text(issue.get("judgment_axis_label")) or ", ".join(_parse_list_like(issue.get("judgment_axes")))


def _action_domain_label(issue: pd.Series) -> str:
    domain = _safe_text(issue.get("action_domain"))
    if not domain:
        return "운영/가이드"
    mapping = {
        "product": "제품 개선",
        "ux_guide": "UX/가이드",
        "operations_service": "운영/서비스",
        "marketing_message": "메시지/커뮤니케이션",
        "business_policy": "정책/사업",
    }
    return mapping.get(domain, domain)


def _format_action_sentence(issue: pd.Series) -> str:
    journey = _journey_label(issue)
    axis = _axis_label(issue) or "핵심 판단 축"
    domain = _action_domain_label(issue)
    summary = _safe_text(issue.get("action_summary")).rstrip(".")
    target = f"{journey} 단계의 '{axis}'"

    domain_actions = {
        "제품 개선": "기능·품질 개선안을 바로 반영",
        "UX/가이드": "설정·안내 흐름을 단순화",
        "운영/서비스": "응답·처리 절차를 단축",
        "메시지/커뮤니케이션": "핵심 메시지를 재정렬",
        "정책/사업": "정책 기준을 조정",
    }
    default_action = domain_actions.get(domain, "실행 항목으로 전환")
    vague_tokens = ("필요", "검토", "고려")

    if summary and not any(token in summary for token in vague_tokens):
        return f"{target} 이슈를 줄이기 위해 {summary}하고, {domain} 영역에서 즉시 실행해야 합니다."
    return f"{target} 이슈를 줄이기 위해 {default_action}하고, {domain} 영역에서 즉시 실행해야 합니다."


def _format_importance_sentence(issue: pd.Series) -> str:
    summary = _safe_text(issue.get("insight_summary"))
    clusters = int(pd.to_numeric(issue.get("supporting_cluster_count", 0), errors="coerce") or 0)
    polarity = _safe_text(issue.get("sentiment_label")) or _safe_text(issue.get("polarity_direction")) or "중립"
    if not summary:
        summary = "동일한 반응이 반복되어 일시적 의견이 아니라 구조적 신호로 해석됩니다."
    risk_tail = (
        "대응이 늦어질수록 불만 확산 위험이 커집니다."
        if str(polarity).startswith("부정")
        else "강점을 유지하지 않으면 체감 가치가 빠르게 약화될 수 있습니다."
    )
    return f"{summary} 현재 반복 근거는 {clusters:,}건이며, {risk_tail}"


def _format_signal_sentence(issue: pd.Series, issue_comments: pd.DataFrame) -> str:
    ratio = float(pd.to_numeric(issue.get("issue_ratio", 0.0), errors="coerce") or 0.0) * 100.0
    clusters = int(pd.to_numeric(issue.get("supporting_cluster_count", 0), errors="coerce") or 0)
    journey = _journey_label(issue)
    polarity = _safe_text(issue.get("sentiment_label")) or _safe_text(issue.get("polarity_direction")) or "중립"
    return (
        f"{journey} 단계에서 {polarity} 반응이 반복되며, 영향 범위는 {ratio:.1f}%, "
        f"반복 근거 {clusters:,}건, 연결 댓글 {len(issue_comments):,}건입니다."
    )


def _priority_visibility_label(issue: pd.Series) -> str:
    priority = _safe_text(issue.get("priority_level")).lower()
    impact = _safe_text(issue.get("impact_level")).lower()
    if priority == "high" or impact == "high":
        return "즉시 대응 필요"
    if priority == "medium" or impact == "medium":
        return "단기 개선 필요"
    return "모니터링 대상"


def _render_scope_summary(context: StrategyPageContext, video_info: dict[str, Any]) -> None:
    selected_filters = context.selected_filters
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.caption("데이터 모드")
            st.write("실데이터" if _safe_text(context.active_mode).lower() == "real" else "샘플")
        with c2:
            st.caption("스냅샷")
            st.write(_safe_text(context.active_snapshot_run) or "전체 run")
        with c3:
            st.caption("영상 수")
            st.write(f"{len(context.filtered_videos):,}")
        with c4:
            st.caption("댓글 수")
            st.write(f"{len(context.filtered_comments):,}")

        st.caption(
            "현재 범위: "
            f"제품({_scope_summary_line(list(selected_filters.get('products', [])))}), "
            f"시장({_scope_summary_line(list(selected_filters.get('regions', [])))}), "
            f"브랜드({_scope_summary_line(list(selected_filters.get('brands', [])))}), "
            f"감성({_scope_summary_line(list(selected_filters.get('sentiments', [])))}), "
            f"고객 여정({_scope_summary_line(list(selected_filters.get('cej', [])))})."
        )

        selected_video_id = _safe_text(video_info.get("selected_video_id"))
        if not selected_video_id:
            st.caption("해석 기준: 현재 필터 전체 범위")
            return
        if bool(video_info.get("selected_video_matched", False)):
            st.success(f"선택 영상 `{selected_video_id}` 문맥을 우선 반영해 이슈를 정렬했습니다.")
        else:
            st.info(f"선택 영상 `{selected_video_id}`와 직접 연결된 이슈가 없어, 필터 전체 범위 기준으로 해석합니다.")


def _render_primary(primary: pd.Series, issue_comments: pd.DataFrame) -> None:
    title = _safe_text(primary.get("insight_title")) or "핵심 전략 이슈"
    action_sentence = _format_action_sentence(primary)
    importance_sentence = _format_importance_sentence(primary)
    signal_sentence = _format_signal_sentence(primary, issue_comments)
    journey = _journey_label(primary)
    axes = _axis_label(primary) or "판단 축 미분류"
    priority = _safe_text(primary.get("priority_level")) or "Low"
    impact = _safe_text(primary.get("impact_level")) or "Low"

    urgency_label = _priority_visibility_label(primary)
    st.markdown("## 🔥 주요 1순위 이슈")
    with st.container(border=True):
        st.markdown(f"### {title}")
        st.markdown(f"**{urgency_label}**")
        st.markdown(f"👉 **지금 해야 할 것**: {action_sentence}")
        st.markdown(f"📌 **왜 중요한가**: {importance_sentence}")
        st.markdown(f"📊 **핵심 판단 근거**: {signal_sentence}")
        st.caption(
            f"실행 범주: {_action_domain_label(primary)} · 고객 여정: {journey} · 판단 축: {axes} · "
            f"우선순위/영향도: {priority}/{impact}"
        )

    clusters = int(pd.to_numeric(primary.get("supporting_cluster_count", 0), errors="coerce") or 0)
    insufficient = bool(primary.get("insufficient_evidence", False)) or clusters < 3 or issue_comments.empty
    if insufficient:
        st.warning("근거가 충분하지 않아 방향성 점검 단계로 해석해야 합니다. 필터를 완화하거나 범위를 넓혀 주세요.")


def _render_secondary(top_issues: pd.DataFrame, primary_issue_key: str) -> None:
    rest = top_issues[top_issues.get("issue_key", pd.Series("", index=top_issues.index)).astype(str).ne(_safe_text(primary_issue_key))].copy()
    if rest.empty:
        return

    with st.expander("🧩 함께 고려할 이슈", expanded=False):
        for _, row in rest.head(5).iterrows():
            title = _safe_text(row.get("insight_title")) or "보조 이슈"
            action = _format_action_sentence(row)
            st.caption(f"- {title}: {action[:110]}{'...' if len(action) > 110 else ''}")


def _render_evidence(primary: pd.Series, issue_comments: pd.DataFrame, representative_comments: pd.DataFrame) -> None:
    with st.expander("🔎 판단 근거 보기", expanded=False):
        rep_ids = _parse_list_like(primary.get("supporting_representatives", []))
        rep_pool = representative_comments.copy()
        if not rep_pool.empty and "comment_id" in rep_pool.columns and rep_ids:
            rep_pool = rep_pool[rep_pool["comment_id"].astype(str).isin(rep_ids)].copy()
        if rep_pool.empty:
            rep_pool = issue_comments.copy()

        if rep_pool.empty:
            st.caption("연결 가능한 대표 근거가 없습니다.")
        else:
            st.caption("대표 근거 1~2개")
            for idx, (_, row) in enumerate(rep_pool.head(2).iterrows(), start=1):
                body = _safe_text(row.get("text_display", row.get("원문 댓글", row.get("cleaned_text", ""))))
                sentiment = _safe_text(row.get("sentiment_final", row.get("sentiment_label", "neutral")))
                st.markdown(f"{idx}. ({sentiment}) {body[:160]}{'...' if len(body) > 160 else ''}")

        clusters = int(pd.to_numeric(primary.get("supporting_cluster_count", 0), errors="coerce") or 0)
        if issue_comments.empty:
            st.caption("현재 범위에서는 반복 근거가 충분하지 않습니다.")
            return

        st.caption(
            f"동일 맥락의 반응이 반복 확인됩니다. (연결 댓글 {len(issue_comments):,}건, 반복 근거 {clusters:,}건)"
        )
        with st.expander("원문 댓글 상세(참고)", expanded=False):
            raw = pd.DataFrame()
            raw["comment_id"] = issue_comments.get("comment_id", "")
            raw["sentiment"] = issue_comments.get("sentiment_final", issue_comments.get("sentiment_label", ""))
            raw["journey_stage"] = issue_comments.get("journey_stage", "")
            raw["judgment_axes"] = issue_comments.get("judgment_axes", "")
            raw["text"] = issue_comments.get("text_display", issue_comments.get("cleaned_text", ""))
            st.dataframe(raw.head(20), width="stretch", hide_index=True)


def render_strategy_page(context: StrategyPageContext) -> None:
    st.markdown("### 전략 인사이트")
    st.caption("현재 필터/선택 범위를 그대로 반영해 전략 해석을 제공합니다.")

    if context.filtered_comments.empty or context.filtered_videos.empty:
        st.warning("현재 범위에서 전략 해석에 필요한 댓글·영상 데이터가 부족합니다. Page 1에서 범위를 넓혀 주세요.")
        return
    if context.analysis_non_trash.empty:
        st.warning("현재 범위에서 전략 해석용 데이터가 부족합니다. 필터를 완화하거나 다른 스냅샷을 확인해 주세요.")
        return
    if context.strategy_video_insights.empty and context.strategy_product_group_insights.empty:
        st.warning("전략 인사이트 데이터가 없어 현재 범위의 해석을 생성할 수 없습니다.")
        return

    top_issues = _build_top_issues(context)
    if top_issues.empty:
        st.info("현재 범위에서 계산 가능한 전략 이슈가 없습니다. 필터를 완화해 다시 확인해 주세요.")
        return

    prioritized_issues, video_info = _prioritize_by_selected_video(top_issues, context.selected_video_id)
    primary, issue_comments, eligible_issues = _pick_primary_issue(prioritized_issues, context.analysis_non_trash)
    if primary is None:
        st.info("현재 범위에서 우선 이슈를 확정하기 어렵습니다. 범위를 넓혀 다시 확인해 주세요.")
        return

    if (
        bool(video_info.get("selected_video_matched", False))
        and _safe_text(primary.get("level_type")).lower() != "video"
        and _safe_text(video_info.get("selected_video_id"))
    ):
        st.info("선택한 영상에서는 충분한 근거가 없어, 현재 필터 범위에서 가장 근거가 많은 이슈를 표시합니다.")

    _render_primary(primary, issue_comments)
    _render_scope_summary(context, video_info)
    _render_secondary(eligible_issues, _safe_text(primary.get("issue_key", "")))
    _render_evidence(primary, issue_comments, context.representative_comments)
