"""Dashboard v3 (stabilized sidebar flow).

Policy for this phase:
- One sidebar flow only
- Clear separation between analysis-run keyword and quick-search keyword
- No backend/pipeline/schema changes
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _sanitize_filter_options(options: dict[str, list[str]]) -> dict[str, list[str]]:
    """UI only: noisy product labels like '가전제품, 귀곰' 제거."""
    cleaned = dict(options)
    product_values = list(cleaned.get("products", []))
    available_values = list(cleaned.get("available_products", []))
    cleaned["products"] = [item for item in product_values if "," not in _safe_text(item)]
    cleaned["available_products"] = [item for item in available_values if "," not in _safe_text(item)]
    return cleaned


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
        return legacy.add_localized_columns(analysis_non_trash)

    analysis_comments_raw = data_bundle.get("analysis_comments", pd.DataFrame()).copy()
    if analysis_comments_raw.empty:
        return analysis_non_trash

    included = analysis_comments_raw.get("analysis_included", pd.Series(True, index=analysis_comments_raw.index)).fillna(False).astype(bool)
    hygiene = analysis_comments_raw.get("hygiene_class", pd.Series("strategic", index=analysis_comments_raw.index)).fillna("").astype(str).str.lower()
    analysis_non_trash = analysis_comments_raw[included & hygiene.ne("trash")].copy()
    if analysis_non_trash.empty:
        return analysis_non_trash
    return legacy.add_localized_columns(analysis_non_trash)


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

    primary_issue = strategy_v2._get_primary_issue(top_issues)
    issue_comments = strategy_v2._filter_comments_for_issue(primary_issue, ctx.analysis_non_trash) if primary_issue is not None else pd.DataFrame()
    primary_key = _safe_text(primary_issue.get("issue_key", "")) if primary_issue is not None else ""

    strategy_v2._render_decision_card(primary_issue, issue_comments)
    strategy_v2._render_secondary_issues(top_issues, primary_key)
    strategy_v2._render_evidence_drilldown(primary_issue, issue_comments, ctx.representative_comments, ctx.analysis_non_trash)
    strategy_v2._render_all_issues_table(top_issues)


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
