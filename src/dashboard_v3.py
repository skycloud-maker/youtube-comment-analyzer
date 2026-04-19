"""Dashboard v3 (stabilized sidebar flow).

Policy for this phase:
- One sidebar flow only
- Clear separation between analysis-run keyword and quick-search keyword
- No backend/pipeline/schema changes
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src import dashboard_app as legacy
from src import dashboard_v2 as strategy_v2

_ENTRY_SEARCH_KEY = "v3_entry_search_query"
_PENDING_FILTER_RESET_KEY = "v3_pending_filter_reset"
_FORCE_WIDE_SCOPE_KEY = "v3_force_wide_scope"


@dataclass
class V3Context:
    active_mode: str
    active_snapshot_run: str
    selected_filters: dict[str, Any]
    filtered_comments: pd.DataFrame
    filtered_videos: pd.DataFrame
    analysis_non_trash: pd.DataFrame
    representative_comments: pd.DataFrame
    top_issues: pd.DataFrame
    top_issues_fallback: pd.DataFrame
    quick_search_query: str


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


def _clip_text(text: str, max_len: int = 180) -> str:
    raw = _safe_text(text).replace("\n", " ").strip()
    if len(raw) <= max_len:
        return raw
    return f"{raw[:max_len].rstrip()}..."


def _sentiment_ko(value: Any) -> str:
    mapping = {"positive": "긍정", "negative": "부정", "mixed": "혼합", "neutral": "중립"}
    return mapping.get(_safe_text(value).lower(), _safe_text(value) or "중립")


def _extract_comment_text(row: pd.Series) -> str:
    for col in ("text_display", legacy.COL_ORIGINAL, "cleaned_text", "text_original"):
        if col in row:
            text = _safe_text(row.get(col))
            if text:
                return text
    return ""


def _issue_ratio_pct(issue_row: pd.Series) -> float:
    return float(pd.to_numeric(issue_row.get("issue_ratio", 0.0), errors="coerce") or 0.0) * 100.0


def _first_axis_label(issue_row: pd.Series) -> str:
    label = _safe_text(issue_row.get("judgment_axis_label"))
    if label:
        return _safe_text(label.split(",")[0])
    axes = _parse_list_like(issue_row.get("judgment_axes", []))
    return _safe_text(axes[0]) if axes else "핵심 판단축 미분류"


def _build_action_sentence(issue_row: pd.Series) -> str:
    stage = _safe_text(issue_row.get("journey_stage_label")) or _safe_text(issue_row.get("journey_stage")) or "해당 여정"
    focus_axis = _first_axis_label(issue_row)
    action_label = _safe_text(issue_row.get("action_label", "")).lower()
    issue_key = _safe_text(issue_row.get("issue_key"))

    def _variant(total: int) -> int:
        if total <= 1:
            return 0
        seed = sum(ord(ch) for ch in (issue_key or f"{stage}|{focus_axis}|{action_label}"))
        return seed % total

    if action_label == "immediate_fix":
        patterns = [
            f"{focus_axis} 문제는 {stage} 단계에서 즉시 대응이 필요하며, 단기 조치와 운영 보완을 동시에 실행해야 합니다.",
            f"{stage} 구간에서 반복되는 {focus_axis} 리스크를 이번 주 즉시 개선 과제로 고정하고 실행 점검을 병행해야 합니다.",
            f"{focus_axis} 마찰은 {stage} 경험을 직접 훼손하고 있어, 지연 없이 긴급 수정과 운영 대응을 함께 추진해야 합니다.",
        ]
        return patterns[_variant(len(patterns))]
    if action_label == "product_improvement":
        patterns = [
            f"{stage} 구간에서 반복되는 {focus_axis} 불편을 다음 제품·기능 개선 백로그의 최우선 항목으로 반영해야 합니다.",
            f"{focus_axis} 이슈는 {stage} 단계에서 재현성이 높아 다음 릴리스 개선 항목으로 즉시 확정해야 합니다.",
            f"{stage} 경험에서 누적되는 {focus_axis} 문제를 제품 개선 로드맵의 선행 과제로 이동해야 합니다.",
        ]
        return patterns[_variant(len(patterns))]
    if action_label == "service_operation":
        patterns = [
            f"{stage} 단계에서 누적되는 {focus_axis} 이슈를 줄이기 위해 서비스 응대 기준과 처리 흐름을 우선 정비해야 합니다.",
            f"{focus_axis} 관련 문의가 {stage} 구간에서 반복되므로, 처리 SLA와 이관 규칙을 먼저 재정렬해야 합니다.",
            f"{stage} 접점에서의 {focus_axis} 대응 실패를 줄이도록 운영 시나리오와 대응 기준을 즉시 보강해야 합니다.",
        ]
        return patterns[_variant(len(patterns))]
    if action_label == "marketing_message":
        patterns = [
            f"{stage} 구간의 기대-현실 괴리를 줄이도록 {focus_axis} 관련 안내 문구와 비교 메시지를 먼저 수정해야 합니다.",
            f"{focus_axis} 인식 차이가 {stage} 단계에서 커지고 있어, 구매 전 메시지와 가이드 표현을 우선 보정해야 합니다.",
            f"{stage} 여정에서 발생한 {focus_axis} 오해를 줄이기 위해 핵심 커뮤니케이션 문안을 즉시 재작성해야 합니다.",
        ]
        return patterns[_variant(len(patterns))]
    if action_label == "maintain_strength":
        patterns = [
            f"{stage} 단계에서 확인된 {focus_axis} 강점을 유지·확장할 수 있도록 운영 기준과 커뮤니케이션을 일관되게 유지해야 합니다.",
            f"{focus_axis} 강점은 {stage} 구간에서 재확인되므로, 동일 경험을 재현하는 운영 기준을 표준화해야 합니다.",
            f"{stage} 단계에서 유효한 {focus_axis} 강점을 잃지 않도록 확산 가능한 실행안으로 고정해 유지해야 합니다.",
        ]
        return patterns[_variant(len(patterns))]
    return f"{stage} 단계의 {focus_axis} 신호를 우선 관찰 대상으로 지정하고 추가 근거를 빠르게 축적해야 합니다."


def _build_why_important_text(issue_row: pd.Series) -> str:
    stage = _safe_text(issue_row.get("journey_stage_label")) or _safe_text(issue_row.get("journey_stage")) or "해당 여정"
    sentiment = _safe_text(issue_row.get("sentiment_label")) or _sentiment_ko(issue_row.get("polarity_direction"))
    issue_ratio = _issue_ratio_pct(issue_row)
    cluster_size = int(pd.to_numeric(issue_row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    action_label = _safe_text(issue_row.get("action_label", "")).lower()

    if action_label in {"immediate_fix", "service_operation"}:
        risk = "지금 대응하지 않으면 불만 누적으로 서비스 신뢰 하락 위험이 커집니다."
    elif action_label == "product_improvement":
        risk = "개선이 늦어질수록 동일 사용 불편이 반복되어 제품 만족 회복 속도가 떨어질 수 있습니다."
    elif action_label == "marketing_message":
        risk = "기대 대비 체감 차이가 지속되면 구매 전환과 추천 의향이 함께 약화될 수 있습니다."
    elif action_label == "maintain_strength":
        risk = "강점을 관리하지 않으면 경쟁 비교 구간에서 현재 우위가 빠르게 희석될 수 있습니다."
    else:
        risk = "근거가 약한 상태에서 방치하면 이슈가 커진 뒤 대응하게 될 수 있습니다."

    return (
        f"{stage} 구간에서 {sentiment} 반응이 반복되어 사용자 경험 교란이 확인됩니다. "
        f"일시 신호가 아니라 우선 검토해야 할 수준입니다 (영향 범위 {issue_ratio:.1f}%, 반복 근거 {cluster_size:,}건). "
        f"{risk}"
    )


def _build_action_direction(issue_row: pd.Series) -> str:
    action_label_ko = _safe_text(issue_row.get("action_label_ko")) or _safe_text(issue_row.get("action_label")) or "모니터링"
    action_label = _safe_text(issue_row.get("action_label", "")).lower()
    focus_axis = _first_axis_label(issue_row)

    if action_label in {"immediate_fix", "product_improvement"}:
        domain = "제품/기능"
    elif action_label == "service_operation":
        domain = "서비스/운영"
    elif action_label == "marketing_message":
        domain = "메시지/커뮤니케이션"
    elif action_label == "maintain_strength":
        domain = "강점 유지/확장"
    else:
        domain = "관찰/검증"
    return f"권장 조치 유형은 **{action_label_ko}**이며, 이번 라운드는 **{domain}** 영역에서 `{focus_axis}` 중심으로 실행해야 합니다."


def _sanitize_filter_options(options: dict[str, list[str]]) -> dict[str, list[str]]:
    """UI only: noisy product labels like '가전제품, 귀곰' 제거."""
    cleaned = dict(options)
    product_values = list(cleaned.get("products", []))
    available_values = list(cleaned.get("available_products", []))
    cleaned["products"] = [item for item in product_values if "," not in _safe_text(item)]
    cleaned["available_products"] = [item for item in available_values if "," not in _safe_text(item)]
    return cleaned


def _issue_product_scope(issue_row: pd.Series) -> str:
    level_type = _safe_text(issue_row.get("level_type"))
    if level_type == "product_group":
        return _safe_text(issue_row.get("level_id"))
    return _safe_text(issue_row.get("product_group"))


def _enforce_issue_product_consistency(top_issues: pd.DataFrame, selected_filters: dict[str, Any]) -> pd.DataFrame:
    if top_issues is None or top_issues.empty:
        return pd.DataFrame() if top_issues is None else top_issues
    products_active = bool(selected_filters.get("products_active", False))
    selected_products = {_safe_text(v) for v in selected_filters.get("products", []) if _safe_text(v)}
    if not products_active or not selected_products:
        return top_issues

    working = top_issues.copy()
    issue_products = working.apply(_issue_product_scope, axis=1)
    mask = issue_products.isin(selected_products)
    return working[mask].reset_index(drop=True)


def _enforce_priority_ratio_consistency(top_issues: pd.DataFrame) -> pd.DataFrame:
    if top_issues is None or top_issues.empty:
        return pd.DataFrame() if top_issues is None else top_issues
    working = top_issues.copy()
    ratio = pd.to_numeric(working.get("issue_ratio", 0.0), errors="coerce").fillna(0.0)
    high_mask = ratio.le(0) & working.get("priority_level", "").astype(str).eq("High")
    if high_mask.any():
        working.loc[high_mask, "priority_level"] = "Medium"
        working.loc[high_mask, "insufficient_evidence"] = True
    return working


def _matches_issue_context(candidate: pd.Series, issue_row: pd.Series) -> bool:
    stage = _safe_text(issue_row.get("journey_stage")).lower()
    polarity = _safe_text(issue_row.get("polarity_direction")).lower()
    primary_axis = _parse_list_like(issue_row.get("judgment_axes"))
    primary_axis = _safe_text(primary_axis[0]) if primary_axis else ""
    level_type = _safe_text(issue_row.get("level_type"))
    level_id = _safe_text(issue_row.get("level_id"))
    issue_product = _issue_product_scope(issue_row)

    candidate_stage = _safe_text(candidate.get("journey_stage")).lower()
    if stage and candidate_stage and candidate_stage != stage:
        return False

    candidate_axes = {_safe_text(v) for v in _parse_list_like(candidate.get("judgment_axes", [])) if _safe_text(v)}
    if primary_axis and candidate_axes and primary_axis not in candidate_axes:
        return False

    mixed_flag = bool(candidate.get("mixed_flag", False))
    candidate_sentiment = _safe_text(candidate.get("sentiment_final", candidate.get("sentiment_label", ""))).lower()
    if polarity == "mixed":
        if not mixed_flag and candidate_sentiment != "mixed":
            return False
    elif polarity:
        if mixed_flag or (candidate_sentiment and candidate_sentiment != polarity):
            return False

    if level_type == "video" and level_id:
        candidate_video = _safe_text(candidate.get("video_id"))
        if candidate_video and candidate_video != level_id:
            return False

    candidate_product = _safe_text(candidate.get("product"))
    if issue_product and candidate_product and candidate_product != issue_product:
        return False

    return True


def _aligned_comment_pool(comments_df: pd.DataFrame, issue_row: pd.Series) -> pd.DataFrame:
    if comments_df is None or comments_df.empty:
        return pd.DataFrame() if comments_df is None else comments_df
    mask = comments_df.apply(lambda row: _matches_issue_context(row, issue_row), axis=1)
    return comments_df[mask].reset_index(drop=True)


def _apply_pending_filter_reset_if_needed(options: dict[str, list[str]]) -> None:
    if not bool(st.session_state.get(_PENDING_FILTER_RESET_KEY, False)):
        return
    st.session_state["filter_products"] = list(options.get("products", []))
    st.session_state["filter_regions"] = list(options.get("regions", []))
    st.session_state["filter_brands"] = list(options.get("brands", []))
    st.session_state["filter_sentiments"] = list(options.get("sentiments", []))
    st.session_state["filter_cej"] = list(options.get("cej", []))
    st.session_state["filter_keyword_query"] = ""
    st.session_state["filter_keyword_mode"] = "AND"
    st.session_state["filter_keyword_mode_label"] = "AND(모두 포함)"
    st.session_state[_PENDING_FILTER_RESET_KEY] = False


def _render_sidebar_data_mode_section(active_mode: str, active_snapshot_run: str) -> None:
    st.markdown("### 데이터 모드 / 스냅샷")
    st.caption("샘플/실데이터 전환과 run 고정 스냅샷을 관리합니다.")
    col_sample, col_real = st.columns(2)
    with col_sample:
        if st.button("샘플 로드", width="stretch", type="secondary", key="v3_load_sample"):
            legacy._set_data_mode("sample", force_refresh=False)
            st.rerun()
    with col_real:
        if st.button("실데이터 새로고침", width="stretch", type="primary", key="v3_refresh_real"):
            legacy._set_data_mode("real", force_refresh=True, snapshot_run_id=active_snapshot_run)
            st.rerun()

    if active_mode != "real":
        st.caption("현재 샘플 모드입니다. 스냅샷 고정은 실데이터 모드에서만 사용 가능합니다.")
        return

    snapshot_options = legacy._build_run_snapshot_options()
    snapshot_label_by_value = {value: label for label, value in snapshot_options}
    snapshot_values = list(snapshot_label_by_value.keys())
    selected_snapshot = active_snapshot_run if active_snapshot_run in snapshot_values else legacy.RUN_SNAPSHOT_ALL
    selected_index = snapshot_values.index(selected_snapshot) if selected_snapshot in snapshot_values else 0
    selected_snapshot = st.selectbox(
        "실데이터 스냅샷",
        options=snapshot_values,
        index=selected_index,
        format_func=lambda value: snapshot_label_by_value.get(value, value),
        help="특정 run 결과를 고정해 재검토할 수 있습니다.",
        key="v3_snapshot_select",
    )
    if st.button("선택 스냅샷 불러오기", width="stretch", type="secondary", key="v3_apply_snapshot"):
        legacy._set_data_mode("real", force_refresh=False, snapshot_run_id=selected_snapshot)
        st.rerun()
    if active_snapshot_run != legacy.RUN_SNAPSHOT_ALL:
        st.caption(f"현재 고정 run: `{active_snapshot_run}`")


def _render_sidebar_state_section(active_mode: str, active_snapshot_run: str) -> None:
    with st.expander("분석 상태 저장/불러오기", expanded=False):
        st.caption("현재 필터/모드/스냅샷 상태를 저장해 다음 실행에서 바로 복원합니다.")
        save_label = st.text_input(
            "저장 이름",
            value="",
            placeholder="예: 4월_셋째주_세탁기_점검",
            key="v3_ui_state_save_label",
        )
        if st.button("현재 상태 저장", width="stretch", type="secondary", key="v3_ui_state_save"):
            current_mode = _safe_text(st.session_state.get(legacy.SESSION_DATA_MODE_KEY, active_mode)) or "real"
            current_snapshot = _safe_text(st.session_state.get(legacy.SESSION_RUN_SNAPSHOT_KEY, active_snapshot_run)) or legacy.RUN_SNAPSHOT_ALL
            payload = legacy._collect_current_ui_state(current_mode, current_snapshot)
            ok, message = legacy._write_saved_ui_state(save_label or "saved_state", payload)
            if ok:
                st.success(f"저장 완료: {Path(message).name}")
            else:
                st.error(message)

        saved_states = legacy._list_saved_ui_states()
        if not saved_states:
            st.caption("저장된 상태가 없습니다.")
            return

        labels_by_file = {file_name: label for label, file_name in saved_states}
        selected_state = st.selectbox(
            "저장된 상태",
            options=[file_name for _, file_name in saved_states],
            format_func=lambda file_name: labels_by_file.get(file_name, file_name),
            key="v3_ui_state_load_target",
        )
        if st.button("선택 상태 불러오기", width="stretch", type="secondary", key="v3_ui_state_load"):
            payload = legacy._read_saved_ui_state(selected_state)
            if not payload:
                st.error("불러올 상태를 찾지 못했습니다.")
            else:
                legacy._apply_saved_ui_state(payload)
                st.success("저장 상태를 불러왔습니다.")
                st.rerun()


def _render_sidebar_run_analysis_section() -> None:
    st.markdown("### 실제 분석 실행")
    st.caption("새로운 분석을 실행하는 영역입니다. 여기 키워드는 실행용입니다.")

    run_keyword = st.text_input(
        "실행용 검색 키워드",
        value="가전 리뷰",
        placeholder="예: 세탁기 냄새, 건조기 소음",
        key="v3_run_keyword",
    )
    run_keyword_mode = st.selectbox(
        "실행 키워드 결합 규칙",
        ["AND(교집합)", "OR(합집합)"],
        index=0,
        key="v3_run_keyword_mode",
    )

    market_labels = [label for label, _ in legacy.ANALYSIS_MARKET_OPTIONS]
    market_code_by_label = dict(legacy.ANALYSIS_MARKET_OPTIONS)
    default_market_index = next((i for i, (_, code) in enumerate(legacy.ANALYSIS_MARKET_OPTIONS) if code == "KR"), 0)
    selected_market_label = st.selectbox("시장", market_labels, index=default_market_index, key="v3_run_market")
    run_region = market_code_by_label[selected_market_label]

    language_labels = [label for label, _ in legacy.ANALYSIS_LANGUAGE_OPTIONS]
    language_code_by_label = dict(legacy.ANALYSIS_LANGUAGE_OPTIONS)
    default_language_index = next((i for i, (_, code) in enumerate(legacy.ANALYSIS_LANGUAGE_OPTIONS) if code == "ko"), 0)
    selected_language_label = st.selectbox("언어", language_labels, index=default_language_index, key="v3_run_language")
    run_language = language_code_by_label[selected_language_label]

    run_max_videos = st.number_input("수집 영상 수", min_value=5, max_value=200, value=15, step=5, key="v3_run_max_videos")
    run_comments_per_video = st.number_input("영상당 댓글 수(0=전체)", min_value=0, max_value=2000, value=0, step=50, key="v3_run_comments_per_video")
    run_output_prefix = st.text_input("결과 파일 prefix", value="dashboard_live", key="v3_run_output_prefix")

    run_keyword_mode_value = "AND" if run_keyword_mode.startswith("AND") else "OR"
    if run_keyword.strip() and run_region != legacy.ANALYSIS_ALL_MARKET_CODE:
        keyword_preview = legacy._build_search_keyword_variants(
            " ".join(legacy._parse_run_keywords(run_keyword)) if run_keyword_mode_value == "AND" else run_keyword,
            language=run_language,
            region=run_region,
        )
        if keyword_preview:
            st.caption("이번 실행 예상 검색 키워드")
            st.code("\n".join(keyword_preview))

    if st.button("실분석 실행", width="stretch", type="secondary", key="v3_run_analysis"):
        if not run_keyword.strip():
            st.warning("실행용 검색 키워드를 입력해 주세요.")
            return
        with st.spinner("실제 분석을 실행 중입니다..."):
            ok, message = legacy._run_real_analysis_action(
                keyword=run_keyword.strip(),
                max_videos=int(run_max_videos),
                comments_per_video=int(run_comments_per_video),
                language=run_language,
                region=run_region,
                output_prefix=run_output_prefix.strip() or "dashboard_live",
                keyword_match_mode=run_keyword_mode_value,
            )
        if ok:
            st.session_state[legacy.SESSION_LAST_RUN_RESULT_KEY] = message
            legacy._set_data_mode("real", force_refresh=True, snapshot_run_id=legacy._latest_real_run_id())
            st.success("실분석이 완료되었습니다. 최신 실데이터로 갱신합니다.")
            st.rerun()
        else:
            st.error("실분석 실행 중 오류가 발생했습니다.")
            st.code(message)

    if st.session_state.get(legacy.SESSION_LAST_RUN_RESULT_KEY):
        st.caption("최근 실분석 실행 결과")
        st.code(str(st.session_state[legacy.SESSION_LAST_RUN_RESULT_KEY])[:1000])


def _render_sidebar_quick_search_section() -> str:
    st.markdown("### 결과 내 빠른 검색")
    st.caption("이미 로드된 전략 결과를 좁혀보는 검색입니다. 분석 실행과는 완전히 별개입니다.")
    return st.text_input(
        "빠른 검색어",
        value=_safe_text(st.session_state.get(_ENTRY_SEARCH_KEY, "")),
        placeholder="예: 냄새, A/S, 설치, lg",
        key=_ENTRY_SEARCH_KEY,
    ).strip()


def _render_sidebar_filter_section(options: dict[str, list[str]]) -> dict[str, Any]:
    # Keep legacy filter renderer as the single visible filter block.
    return legacy._render_sidebar_filters(options)


def _render_sidebar_controls(active_mode: str, active_snapshot_run: str, options: dict[str, list[str]]) -> tuple[str, dict[str, Any]]:
    with st.sidebar:
        _render_sidebar_data_mode_section(active_mode, active_snapshot_run)
        st.markdown("---")
        _render_sidebar_state_section(active_mode, active_snapshot_run)
        st.markdown("---")
        _render_sidebar_run_analysis_section()
        st.markdown("---")
        quick_search_query = _render_sidebar_quick_search_section()
        st.markdown("---")
        selected_filters = _render_sidebar_filter_section(options)
    return quick_search_query, selected_filters


def _build_analysis_non_trash(data_bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    analysis_non_trash = data_bundle.get("analysis_non_trash", pd.DataFrame()).copy()
    if not analysis_non_trash.empty:
        raw_product = analysis_non_trash.get("product", pd.Series("", index=analysis_non_trash.index)).astype(str)
        localized = legacy.add_localized_columns(analysis_non_trash)
        if "product" in localized.columns:
            localized["product"] = raw_product.values
        return localized

    analysis_comments_raw = data_bundle.get("analysis_comments", pd.DataFrame()).copy()
    if analysis_comments_raw.empty:
        return analysis_non_trash

    included = analysis_comments_raw.get("analysis_included", pd.Series(True, index=analysis_comments_raw.index)).fillna(False).astype(bool)
    hygiene = analysis_comments_raw.get("hygiene_class", pd.Series("strategic", index=analysis_comments_raw.index)).fillna("").astype(str).str.lower()
    analysis_non_trash = analysis_comments_raw[included & hygiene.ne("trash")].copy()
    if analysis_non_trash.empty:
        return analysis_non_trash
    raw_product = analysis_non_trash.get("product", pd.Series("", index=analysis_non_trash.index)).astype(str)
    localized = legacy.add_localized_columns(analysis_non_trash)
    if "product" in localized.columns:
        localized["product"] = raw_product.values
    return localized


def _build_top_issue_tables(
    data_bundle: dict[str, pd.DataFrame],
    selected_filters: dict[str, Any],
    filtered_videos: pd.DataFrame,
    filtered_comments: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
    analysis_non_trash_full: pd.DataFrame,
    quick_search_query: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strategy_video = strategy_v2._prepare_strategy_frame(data_bundle.get("strategy_video_insights", pd.DataFrame()).copy(), "video")
    strategy_product = strategy_v2._prepare_strategy_frame(data_bundle.get("strategy_product_group_insights", pd.DataFrame()).copy(), "product_group")
    strategy_video_full = strategy_video.copy()
    strategy_product_full = strategy_product.copy()

    strategy_video, strategy_product = strategy_v2._filter_strategy_frames(
        strategy_video,
        strategy_product,
        filtered_videos,
        filtered_comments,
        selected_filters,
    )

    top_issues = strategy_v2._compose_top_issue_table(strategy_video, strategy_product, analysis_non_trash)
    top_issues_fallback = strategy_v2._compose_top_issue_table(strategy_video_full, strategy_product_full, analysis_non_trash_full)

    top_issues = _enforce_issue_product_consistency(top_issues, selected_filters)
    top_issues_fallback = _enforce_issue_product_consistency(top_issues_fallback, selected_filters)

    top_issues = _enforce_priority_ratio_consistency(top_issues)
    top_issues_fallback = _enforce_priority_ratio_consistency(top_issues_fallback)

    # Quick search refinement only (no analysis rerun).
    top_issues = strategy_v2._apply_entry_search(top_issues, quick_search_query)
    top_issues_fallback = strategy_v2._apply_entry_search(top_issues_fallback, quick_search_query)
    return top_issues, top_issues_fallback


def _load_context() -> V3Context | None:
    legacy._initialize_data_session_state()
    active_mode = st.session_state.get(legacy.SESSION_DATA_MODE_KEY, "real")
    active_snapshot_run = _safe_text(st.session_state.get(legacy.SESSION_RUN_SNAPSHOT_KEY, legacy.RUN_SNAPSHOT_ALL)) or legacy.RUN_SNAPSHOT_ALL
    expected_signature = legacy._expected_mode_signature(active_mode, snapshot_run_id=active_snapshot_run)
    if st.session_state.get(legacy.SESSION_DATA_SIGNATURE_KEY) != expected_signature:
        with st.spinner("최신 결과와 동기화 중…"):
            legacy._set_data_mode(active_mode, force_refresh=True, snapshot_run_id=active_snapshot_run)

    data_bundle = st.session_state.get(legacy.SESSION_DATA_BUNDLE_KEY, legacy._empty_dashboard_bundle())
    comments_raw = data_bundle.get("comments", pd.DataFrame()).copy()
    videos_raw = data_bundle.get("videos", pd.DataFrame()).copy()

    if active_mode == "real" and (comments_raw.empty or videos_raw.empty):
        empty_options = {
            "products": [],
            "available_products": [],
            "regions": [],
            "brands": [],
            "sentiments": [],
            "cej": [],
            "analysis_scope": ["전체", "문의 포함 댓글만", "문의 제외 댓글만"],
            "relevance_scope": ["전체 수집", "LG 관련성 높은 댓글", "LG 언급 댓글", "LG 비언급 댓글"],
        }
        _render_sidebar_controls(active_mode, active_snapshot_run, empty_options)
        return None

    comments_scored, videos_df = legacy.ensure_lg_relevance_columns(comments_raw, videos_raw)
    comments_df = legacy.add_localized_columns(comments_scored)
    if comments_df.empty:
        return None
    videos_df[legacy.COL_COUNTRY] = videos_df.get("region", pd.Series(dtype=str)).map(legacy.localize_region)
    analysis_non_trash_full = _build_analysis_non_trash(data_bundle)

    options = _sanitize_filter_options(legacy._build_dashboard_options(comments_df, videos_df))
    _apply_pending_filter_reset_if_needed(options)
    quick_search_query, selected_filters = _render_sidebar_controls(active_mode, active_snapshot_run, options)

    bundle = legacy.compute_filtered_bundle(
        comments_df,
        videos_df,
        data_bundle.get("quality_summary", pd.DataFrame()),
        selected_filters,
        data_bundle.get("representative_bundles", pd.DataFrame()),
        data_bundle.get("opinion_units", pd.DataFrame()),
        analysis_non_trash_full,
    )

    filtered_comments = bundle.get("all_comments", bundle.get("comments", pd.DataFrame())).copy()
    filtered_videos = bundle.get("videos", pd.DataFrame()).copy()
    analysis_non_trash = bundle.get("analysis_non_trash", pd.DataFrame()).copy()
    representative_comments = bundle.get("representative_comments", pd.DataFrame()).copy()

    top_issues, top_issues_fallback = _build_top_issue_tables(
        data_bundle=data_bundle,
        selected_filters=selected_filters,
        filtered_videos=filtered_videos,
        filtered_comments=filtered_comments,
        analysis_non_trash=analysis_non_trash,
        analysis_non_trash_full=analysis_non_trash_full,
        quick_search_query=quick_search_query,
    )

    if bool(st.session_state.get(_FORCE_WIDE_SCOPE_KEY, False)) and not top_issues_fallback.empty:
        top_issues = top_issues_fallback.copy()

    return V3Context(
        active_mode=active_mode,
        active_snapshot_run=active_snapshot_run,
        selected_filters=selected_filters,
        filtered_comments=filtered_comments,
        filtered_videos=filtered_videos,
        analysis_non_trash=analysis_non_trash,
        representative_comments=representative_comments,
        top_issues=top_issues,
        top_issues_fallback=top_issues_fallback,
        quick_search_query=quick_search_query,
    )


def _render_recovery_panel(*, has_fallback: bool, quick_search_query: str, selected_filters: dict[str, Any]) -> None:
    st.warning("현재 조건에서는 전략 이슈가 없습니다.")
    if quick_search_query:
        st.caption("빠른 검색 조건으로 결과가 너무 좁혀졌을 수 있습니다.")
        st.caption(f"현재 빠른 검색어: `{quick_search_query}`")
    else:
        active_count = sum(
            [
                bool(selected_filters.get("products_active", False)),
                bool(selected_filters.get("regions_active", False)),
                bool(selected_filters.get("brands_active", False)),
                bool(selected_filters.get("sentiments_active", False)),
                bool(selected_filters.get("cej_active", False)),
                bool(_safe_text(selected_filters.get("keyword_query", ""))),
            ]
        )
        if active_count > 0:
            st.caption("필터 조건이 좁아 전략 이슈가 제외되었을 수 있습니다.")
        else:
            st.caption("현재 범위에서는 전략 신호가 약해 즉시 노출 가능한 이슈가 없습니다.")

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("필터 초기화", width="stretch", type="secondary", key="v3_recover_reset"):
            st.session_state[_PENDING_FILTER_RESET_KEY] = True
            st.session_state[_ENTRY_SEARCH_KEY] = ""
            st.rerun()
    with c2:
        if st.button("예시 보기", width="stretch", type="secondary", key="v3_recover_sample"):
            legacy._set_data_mode("sample", force_refresh=False)
            st.rerun()
    with c3:
        if st.button("최근 주요 이슈 보기", width="stretch", type="primary", key="v3_recover_wide"):
            st.session_state[_FORCE_WIDE_SCOPE_KEY] = True
            st.rerun()

    if has_fallback:
        st.caption("‘최근 주요 이슈 보기’를 누르면 현재 검색 범위에서 넓은 스코프 기준 이슈를 볼 수 있습니다.")
    else:
        st.caption("넓은 스코프에서도 전략 이슈가 부족합니다. 예시 보기로 화면 검증이 가능합니다.")


def _select_primary_issue_with_evidence(
    top_issues: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
) -> tuple[pd.Series | None, pd.DataFrame, pd.DataFrame]:
    if top_issues is None or top_issues.empty:
        return None, pd.DataFrame(), pd.DataFrame() if top_issues is None else top_issues

    working = top_issues.copy().reset_index(drop=True)
    valid_indices: list[int] = []
    issue_comments_by_index: dict[int, pd.DataFrame] = {}

    for idx, row in working.iterrows():
        issue_comments = strategy_v2._filter_comments_for_issue(row, analysis_non_trash)
        issue_comments = _aligned_comment_pool(issue_comments, row)
        issue_comments_by_index[idx] = issue_comments
        cluster_size = int(pd.to_numeric(row.get("supporting_cluster_count", 0), errors="coerce") or 0)
        if cluster_size > 0 and len(issue_comments) > 0:
            valid_indices.append(idx)

    if valid_indices:
        valid_df = working.iloc[valid_indices].reset_index(drop=True)
        primary = valid_df.iloc[0]
        primary_comments = issue_comments_by_index.get(valid_indices[0], pd.DataFrame())
        return primary, primary_comments, valid_df

    downgraded = working.copy()
    downgraded.loc[:, "insufficient_evidence"] = True
    downgraded.loc[:, "priority_level"] = downgraded.get("priority_level", "Low").astype(str).replace({"High": "Medium"})
    first = downgraded.iloc[0]
    return first, pd.DataFrame(), downgraded


def _render_v3_primary_issue(issue_row: pd.Series | None, issue_comments: pd.DataFrame) -> None:
    st.subheader("1) 🔥 주요 1순위 이슈")
    if issue_row is None:
        st.info("현재 조건에서는 표시할 전략 이슈가 없습니다.")
        return

    title = _safe_text(issue_row.get("insight_title")) or "핵심 전략 이슈"
    stage = _safe_text(issue_row.get("journey_stage_label")) or _safe_text(issue_row.get("journey_stage")) or "여정 미분류"
    axes = _safe_text(issue_row.get("judgment_axis_label")) or ", ".join(_parse_list_like(issue_row.get("judgment_axes"))) or "판단축 미분류"
    sentiment = _safe_text(issue_row.get("sentiment_label")) or _sentiment_ko(issue_row.get("polarity_direction"))
    action = _build_action_sentence(issue_row)
    priority = _safe_text(issue_row.get("priority_level")) or "Low"
    impact = _safe_text(issue_row.get("impact_level")) or "Low"
    issue_ratio = _issue_ratio_pct(issue_row)
    cluster_size = int(pd.to_numeric(issue_row.get("supporting_cluster_count", 0), errors="coerce") or 0)
    insufficient = bool(issue_row.get("insufficient_evidence", False))

    st.markdown(f"### {title}")
    st.markdown("#### 👉 지금 해야 할 것")
    st.success(action)

    st.markdown("#### 📌 왜 중요한가")
    st.info(_build_why_important_text(issue_row))

    st.markdown("#### 📊 핵심 판단 근거")
    st.markdown(
        f"- 판단 맥락: **{stage} · {axes}**\n"
        f"- 사용자 반응: **{sentiment}** 방향이며 우선순위/영향도는 **{priority}/{impact}**입니다.\n"
        f"- 반복 신호: 일회성 이슈가 아니라 누적 관리가 필요한 상태입니다 (반복 {cluster_size:,}건, 영향 범위 {issue_ratio:.1f}%)."
    )

    st.markdown("#### 🧭 실행 방향")
    st.write(_build_action_direction(issue_row))

    st.markdown("#### 📍 신뢰도")
    if insufficient or cluster_size < 3:
        st.warning("근거 부족: 추가 반응 축적 후 우선순위 확정이 필요합니다.")
    elif cluster_size >= 120:
        st.success("충분한 근거: 반복 규모가 커서 우선 실행 판단이 가능합니다.")
    else:
        st.success("충분한 근거: 현재 근거 수준에서 실행 판단이 가능합니다.")


def _render_v3_secondary_issues(top_issues: pd.DataFrame, primary_issue_key: str) -> None:
    st.subheader("2) 🧩 함께 고려할 이슈")
    secondary = top_issues[top_issues["issue_key"].astype(str).ne(_safe_text(primary_issue_key))].copy()
    if secondary.empty:
        st.caption("현재 함께 고려할 보조 이슈가 없습니다.")
        return

    with st.expander("보조 이슈 Top 5", expanded=False):
        for idx, (_, row) in enumerate(secondary.head(5).iterrows(), start=1):
            title = _safe_text(row.get("insight_title")) or "보조 이슈"
            action = _safe_text(row.get("action_label_ko")) or _safe_text(row.get("action_label")) or "모니터링"
            stage = _safe_text(row.get("journey_stage_label")) or _safe_text(row.get("journey_stage")) or "미분류"
            st.markdown(f"{idx}. **{title}**")
            st.caption(f"대응 힌트: {stage} 구간 `{action}` 우선")


def _render_v3_evidence(
    issue_row: pd.Series | None,
    issue_comments: pd.DataFrame,
    representative_comments: pd.DataFrame,
) -> None:
    st.subheader("3) 🔎 이 판단의 근거")
    if issue_row is None:
        st.info("현재 조건에서는 연결 가능한 근거가 없습니다.")
        return

    with st.expander("근거 보기", expanded=False):
        total_issue_comments = int(len(issue_comments))
        if total_issue_comments > 0:
            st.caption(
                f"단일 댓글이 아닌 반복 반응을 근거로 판단했습니다 (연결 반응 {total_issue_comments:,}건)."
            )
        else:
            st.caption("연결 반응이 적어 대표 근거 중심으로 제공합니다.")

        st.markdown("#### 대표 근거 댓글")
        rep_pool = representative_comments.copy()
        rep_ids = _parse_list_like(issue_row.get("supporting_representatives", []))
        if not rep_pool.empty and "comment_id" in rep_pool.columns and rep_ids:
            rep_pool = rep_pool[rep_pool["comment_id"].astype(str).isin(rep_ids)].copy()
        rep_pool = _aligned_comment_pool(rep_pool, issue_row)
        if rep_pool.empty:
            rep_pool = issue_comments.copy()

        if rep_pool.empty:
            st.caption("대표 근거 댓글이 없습니다.")
        else:
            sort_cols = [col for col in ("like_count", legacy.COL_LIKES, "published_at") if col in rep_pool.columns]
            if sort_cols:
                rep_pool = rep_pool.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
            for idx, (_, row) in enumerate(rep_pool.head(2).iterrows(), start=1):
                sentiment = _sentiment_ko(row.get("sentiment_final", row.get("sentiment_label", "neutral")))
                text = _clip_text(_extract_comment_text(row), 140)
                st.markdown(f"{idx}. ({sentiment}) {text}")

        st.markdown("#### 반복 근거 요약")
        if total_issue_comments >= 3:
            st.write(
                f"같은 여정/판단축에서 반복 신호가 확인되어 관리 대상 이슈로 볼 수 있습니다 (반복 확인 {total_issue_comments:,}건)."
            )
        elif total_issue_comments > 0:
            st.write("현재는 초기 신호로 보이며 추가 반복 확인이 필요합니다.")
        else:
            st.write("현재 조건에서는 반복 근거 확정이 어렵습니다.")

        st.markdown("#### 원천 댓글 (참고)")
        st.caption("필요 시 세부 검증을 위한 참고 데이터입니다.")
        with st.expander("원천 댓글 상세 보기", expanded=False):
            if issue_comments.empty:
                st.caption("표시할 원천 댓글이 없습니다.")
            else:
                raw_df = pd.DataFrame()
                raw_df["댓글 ID"] = issue_comments.get("comment_id", "")
                raw_df["감성"] = issue_comments.get("sentiment_final", issue_comments.get("sentiment_label", ""))
                raw_df["고객 여정"] = issue_comments.get("journey_stage", "")
                raw_df["판단 축"] = issue_comments.get("judgment_axes", "")
                raw_df["본문"] = issue_comments.apply(_extract_comment_text, axis=1)
                st.dataframe(raw_df.head(50), width="stretch", hide_index=True)


def _render_v3_reference_issues(top_issues: pd.DataFrame) -> None:
    st.subheader("4) 참고 이슈")
    st.caption("필요할 때만 확인하는 참고 목록입니다.")
    with st.expander("전체 이슈 보기", expanded=False):
        if top_issues.empty:
            st.caption("현재 표시할 이슈가 없습니다.")
            return
        ref_df = pd.DataFrame()
        ref_df["이슈"] = top_issues.get("insight_title", "")
        ref_df["권장 조치"] = top_issues.get("action_label_ko", top_issues.get("action_label", ""))
        ref_df["우선순위"] = top_issues.get("priority_level", "")
        ref_df["영향도"] = top_issues.get("impact_level", "")
        ref_df["고객 여정"] = top_issues.get("journey_stage_label", top_issues.get("journey_stage", ""))
        st.dataframe(ref_df.head(25), width="stretch", hide_index=True)


def _render_strategy_shell(ctx: V3Context) -> None:
    top_issues = ctx.top_issues.copy()
    if top_issues.empty:
        _render_recovery_panel(
            has_fallback=not ctx.top_issues_fallback.empty,
            quick_search_query=ctx.quick_search_query,
            selected_filters=ctx.selected_filters,
        )
        st.markdown("---")
        st.subheader("2) 함께 고려할 이슈")
        st.info("현재 조건에서 보조 이슈를 계산할 근거가 충분하지 않습니다.")
        st.subheader("3) 이 판단의 근거")
        st.info("대표 근거와 유사 반응은 이슈가 있을 때 표시됩니다.")
        st.subheader("4) 전체 이슈(참고)")
        with st.expander("전체 이슈 보기", expanded=False):
            st.caption("현재 조건에서는 표시할 이슈가 없습니다.")
        return

    primary_issue, issue_comments, top_issues = _select_primary_issue_with_evidence(top_issues, ctx.analysis_non_trash)
    if primary_issue is None and not ctx.top_issues_fallback.empty:
        primary_issue, issue_comments, top_issues = _select_primary_issue_with_evidence(
            ctx.top_issues_fallback.copy(),
            ctx.analysis_non_trash,
        )
    if primary_issue is None:
        _render_recovery_panel(
            has_fallback=not ctx.top_issues_fallback.empty,
            quick_search_query=ctx.quick_search_query,
            selected_filters=ctx.selected_filters,
        )
        return

    primary_key = _safe_text(primary_issue.get("issue_key", "")) if primary_issue is not None else ""

    _render_v3_primary_issue(primary_issue, issue_comments)
    _render_v3_secondary_issues(top_issues, primary_key)
    _render_v3_evidence(primary_issue, issue_comments, ctx.representative_comments)
    _render_v3_reference_issues(top_issues)


def main() -> None:
    st.set_page_config(page_title="가전 VoC 전략 대시보드 V3", layout="wide")
    legacy.apply_theme()

    ctx = _load_context()

    st.markdown(
        '<div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">'
        '<div style="font-size:28px; font-weight:800; color:#0f172a; letter-spacing:-0.02em;">가전 VoC 전략 대시보드 V3</div>'
        '<span style="display:inline-block; padding:4px 11px; border-radius:999px; background:#dbeafe; color:#1d4ed8; font-size:11px; font-weight:700;">전략 우선</span>'
        '<span style="display:inline-block; padding:4px 11px; border-radius:999px; background:#e2e8f0; color:#334155; font-size:11px; font-weight:700;">V3 검증 중</span>'
        "</div>",
        unsafe_allow_html=True,
    )
    st.caption("레거시 사이드바 진입 UX를 유지하면서, 메인은 전략 이슈 중심으로 보여줍니다.")

    if ctx is None:
        st.warning("현재 실데이터가 연결되지 않았습니다.")
        st.info("좌측에서 샘플 로드 또는 실분석 실행으로 데이터를 연결해 주세요.")
        st.markdown("---")
        st.subheader("1) 주요 1순위 이슈")
        st.info("데이터 연결 후 이 영역에서 즉시 판단 가능한 전략 이슈를 표시합니다.")
        st.subheader("2) 함께 고려할 이슈")
        st.info("보조 이슈 Top5가 이 영역에 표시됩니다.")
        st.subheader("3) 이 판단의 근거")
        with st.expander("근거 보기", expanded=False):
            st.caption("대표 근거/유사 반응/원천 댓글을 확인할 수 있습니다.")
        st.subheader("4) 전체 이슈(참고)")
        with st.expander("전체 이슈 보기", expanded=False):
            st.caption("탐색 참고용 전체 이슈 목록입니다.")
        return

    _render_strategy_shell(ctx)


if __name__ == "__main__":
    main()
