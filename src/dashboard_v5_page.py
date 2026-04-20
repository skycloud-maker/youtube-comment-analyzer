"""V5 topic-first experimental page renderer (Page 4)."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.v5 import run_v5_pipeline


@dataclass(frozen=True)
class V5PageContext:
    selected_filters: dict[str, Any]
    active_mode: str
    active_snapshot_run: str
    filtered_comments: pd.DataFrame
    filtered_videos: pd.DataFrame
    analysis_non_trash: pd.DataFrame
    selected_video_id: str = ""


def build_v5_context(
    *,
    selected_filters: dict[str, Any],
    active_mode: str,
    active_snapshot_run: str,
    filtered_comments: pd.DataFrame,
    filtered_videos: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
    selected_video_id: str = "",
) -> V5PageContext:
    return V5PageContext(
        selected_filters=selected_filters or {},
        active_mode=str(active_mode or "real"),
        active_snapshot_run=str(active_snapshot_run or ""),
        filtered_comments=filtered_comments.copy() if filtered_comments is not None else pd.DataFrame(),
        filtered_videos=filtered_videos.copy() if filtered_videos is not None else pd.DataFrame(),
        analysis_non_trash=analysis_non_trash.copy() if analysis_non_trash is not None else pd.DataFrame(),
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
            parsed_json = json.loads(raw)
            if isinstance(parsed_json, list):
                return [_safe_text(v) for v in parsed_json if _safe_text(v)]
        except Exception:
            pass
        try:
            parsed_py = ast.literal_eval(raw)
            if isinstance(parsed_py, list):
                return [_safe_text(v) for v in parsed_py if _safe_text(v)]
        except Exception:
            pass
    return [_safe_text(part) for part in raw.replace("|", ",").split(",") if _safe_text(part)]


def _scope_line(values: list[str], empty_label: str = "전체") -> str:
    cleaned = [_safe_text(v) for v in values if _safe_text(v)]
    if not cleaned:
        return empty_label
    if len(cleaned) <= 3:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:3])} 외 {len(cleaned) - 3}개"


def _build_comment_truth(source: pd.DataFrame) -> pd.DataFrame:
    if source.empty:
        return pd.DataFrame(columns=["comment_id", "sentiment_final", "mixed_flag", "journey_stage", "judgment_axes", "analysis_included", "hygiene_class"])
    truth = pd.DataFrame()
    truth["comment_id"] = source.get("comment_id", pd.Series("", index=source.index)).astype(str)
    truth["sentiment_final"] = source.get("sentiment_final", source.get("sentiment_label", pd.Series("neutral", index=source.index))).fillna("neutral").astype(str)
    truth["mixed_flag"] = source.get("mixed_flag", pd.Series(False, index=source.index)).fillna(False).astype(bool)
    truth["journey_stage"] = source.get("journey_stage", pd.Series("usage", index=source.index)).fillna("usage").astype(str)
    truth["judgment_axes"] = source.get("judgment_axes", pd.Series([[] for _ in range(len(source))], index=source.index))
    truth["analysis_included"] = source.get("analysis_included", pd.Series(True, index=source.index)).fillna(True).astype(bool)
    truth["hygiene_class"] = source.get("hygiene_class", pd.Series("strategic", index=source.index)).fillna("strategic").astype(str)
    return truth


def _build_video_metadata(context: V5PageContext, source: pd.DataFrame) -> pd.DataFrame:
    if not context.filtered_videos.empty:
        return context.filtered_videos.copy()
    if source.empty:
        return pd.DataFrame(columns=["video_id", "title", "product", "region"])
    columns = [col for col in ["video_id", "title", "product", "region"] if col in source.columns]
    if not columns:
        return pd.DataFrame(columns=["video_id", "title", "product", "region"])
    return source[columns].drop_duplicates(subset=["video_id"]).copy() if "video_id" in columns else source[columns].copy()


def _context_signature(context: V5PageContext) -> str:
    source = context.analysis_non_trash if not context.analysis_non_trash.empty else context.filtered_comments
    comment_ids = source.get("comment_id", pd.Series("", index=source.index)).astype(str).head(300).tolist() if not source.empty else []
    payload = {
        "mode": context.active_mode,
        "run": context.active_snapshot_run,
        "selected_video_id": context.selected_video_id,
        "filters": {
            key: sorted([_safe_text(item) for item in context.selected_filters.get(key, []) if _safe_text(item)])
            for key in ["products", "regions", "brands", "sentiments", "cej"]
        },
        "keyword_query": _safe_text(context.selected_filters.get("keyword_query")),
        "keyword_mode": _safe_text(context.selected_filters.get("keyword_mode")),
        "comment_count": int(len(source)),
        "video_count": int(len(context.filtered_videos)),
        "id_head": comment_ids,
    }
    digest = hashlib.sha1(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return digest


def _prepare_v5_outputs(context: V5PageContext) -> tuple[dict[str, pd.DataFrame] | None, str]:
    source = context.analysis_non_trash if not context.analysis_non_trash.empty else context.filtered_comments
    if source.empty:
        return None, "현재 필터 범위에서 V5 분석 대상 댓글이 없습니다. 필터를 완화하거나 데이터를 다시 불러와 주세요."
    if "comment_id" not in source.columns:
        return None, "V5 분석에 필요한 comment_id 컬럼이 없어 결과를 생성할 수 없습니다."

    comment_truth = _build_comment_truth(source)
    video_metadata = _build_video_metadata(context, source)
    run_id = context.active_snapshot_run or _safe_text(source.get("run_id", pd.Series("", index=source.index)).astype(str).iloc[0] if "run_id" in source.columns and not source.empty else "")

    try:
        outputs = run_v5_pipeline(
            {
                "analysis_non_trash": source.copy(),
                "comment_truth": comment_truth,
                "video_metadata": video_metadata,
                "run_id": run_id,
            }
        )
    except Exception as exc:
        return None, f"V5 파이프라인 실행 중 오류가 발생했습니다: {exc}"

    return outputs, ""


def _merge_for_display(outputs: dict[str, pd.DataFrame], source: pd.DataFrame) -> pd.DataFrame:
    ranked = outputs.get("v5_topic_ranked", pd.DataFrame()).copy()
    topics = outputs.get("v5_topics", pd.DataFrame()).copy()
    actions = outputs.get("v5_topic_insight_actions", pd.DataFrame()).copy()
    reps = outputs.get("v5_topic_representatives", pd.DataFrame()).copy()

    if ranked.empty:
        return pd.DataFrame()

    merged = (
        ranked.merge(topics, on="topic_id", how="left")
        .merge(
            actions[["topic_id", "insight_text", "action_text", "priority_level", "evidence_strength"]],
            on="topic_id",
            how="left",
        )
        .merge(
            reps[["topic_id", "representative_comment_ids", "alignment_score"]],
            on="topic_id",
            how="left",
        )
        .sort_values("rank_score", ascending=False)
        .reset_index(drop=True)
    )

    text_col = "text_display" if "text_display" in source.columns else "raw_text"
    if text_col not in source.columns:
        source = source.copy()
        source[text_col] = ""
    comment_lookup = dict(zip(source.get("comment_id", pd.Series("", index=source.index)).astype(str), source[text_col].fillna("").astype(str)))

    def _rep_texts(raw_ids: Any) -> list[str]:
        rep_ids = _parse_list_like(raw_ids)[:3]
        return [comment_lookup.get(cid, "") for cid in rep_ids if comment_lookup.get(cid, "")]

    merged["representative_comments"] = merged["representative_comment_ids"].map(_rep_texts)
    merged["topic_label"] = merged.apply(
        lambda row: f"{_safe_text(row.get('journey_stage'))} · {_safe_text(row.get('judgment_axis'))}",
        axis=1,
    )
    return merged


def _render_scope_summary(context: V5PageContext, outputs: dict[str, pd.DataFrame], merged: pd.DataFrame) -> None:
    source = context.analysis_non_trash if not context.analysis_non_trash.empty else context.filtered_comments
    signal_count = len(outputs.get("v5_signal_units", pd.DataFrame()))
    topic_count = len(merged)

    selected_filters = context.selected_filters
    keyword_text = _safe_text(selected_filters.get("keyword_query")) or "없음"

    st.markdown("### 분석 범위 요약")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("검색어", keyword_text)
    c2.metric("제품", _scope_line(list(selected_filters.get("products", []))))
    c3.metric("영상 수", f"{len(context.filtered_videos):,}")
    c4.metric("댓글 수", f"{len(source):,}")
    c5.metric("Signal 수", f"{signal_count:,}")
    c6.metric("Topic 수", f"{topic_count:,}")

    st.caption(
        f"현재 필터: 시장({_scope_line(list(selected_filters.get('regions', [])))}), "
        f"브랜드({_scope_line(list(selected_filters.get('brands', [])))}), "
        f"감성({_scope_line(list(selected_filters.get('sentiments', [])))}), "
        f"여정({_scope_line(list(selected_filters.get('cej', [])))})."
    )
    if context.selected_video_id:
        st.caption(f"선택 영상 컨텍스트: `{context.selected_video_id}`")


def _render_top_topic_cards(top_topics: pd.DataFrame) -> None:
    st.markdown("### Top 5 Topic (quality-pass)")
    if top_topics.empty:
        st.warning("top_quality_pass=True 조건을 만족하는 Topic이 없습니다. 필터 범위를 넓혀 주세요.")
        return

    for rank, (_, row) in enumerate(top_topics.head(5).iterrows(), start=1):
        with st.container(border=True):
            st.markdown(f"#### {rank}. {_safe_text(row.get('topic_label'))}")
            tag_col1, tag_col2, tag_col3, tag_col4 = st.columns(4)
            tag_col1.caption(f"여정: `{_safe_text(row.get('journey_stage')) or '-'}'")
            tag_col2.caption(f"판단축: `{_safe_text(row.get('judgment_axis')) or '-'}'")
            tag_col3.caption(f"커버리지: `{float(pd.to_numeric(row.get('coverage_pct', 0.0), errors='coerce') or 0.0):.1f}%`")
            tag_col4.caption(f"빈도: `{int(pd.to_numeric(row.get('frequency_count', 0), errors='coerce') or 0):,}`")

            st.markdown(f"**우선순위**: `{_safe_text(row.get('priority_level')) or 'monitor'}`")
            st.markdown(f"**Insight**: {_safe_text(row.get('insight_text'))}")
            st.success(f"Action: {_safe_text(row.get('action_text'))}")

            rep_comments = row.get("representative_comments", []) or []
            if rep_comments:
                st.caption("대표 근거 댓글")
                for comment in rep_comments[:3]:
                    st.markdown(f"- {comment}")
            else:
                st.caption("대표 근거 댓글을 찾지 못했습니다.")


def _render_distribution(merged: pd.DataFrame) -> None:
    st.markdown("### Topic 분포 요약")
    if merged.empty:
        st.info("표시할 Topic 분포가 없습니다.")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        axis_dist = (
            merged.groupby("judgment_axis", as_index=False)["frequency_count"]
            .sum()
            .sort_values("frequency_count", ascending=False)
            .head(8)
        )
        st.caption("판단축 상위 분포")
        st.bar_chart(axis_dist.set_index("judgment_axis"))
    with col_right:
        journey_dist = (
            merged.groupby("journey_stage", as_index=False)["frequency_count"]
            .sum()
            .sort_values("frequency_count", ascending=False)
            .head(8)
        )
        st.caption("여정 단계 상위 분포")
        st.bar_chart(journey_dist.set_index("journey_stage"))


def _render_topic_table_and_drilldown(merged: pd.DataFrame, outputs: dict[str, pd.DataFrame]) -> None:
    st.markdown("### Topic 목록")
    if merged.empty:
        st.info("표시할 Topic 목록이 없습니다.")
        return

    display_cols = [
        "topic_id",
        "journey_stage",
        "judgment_axis",
        "coverage_pct",
        "frequency_count",
        "rank_score",
        "priority_level",
        "top_quality_pass",
        "alignment_score",
        "top_exclusion_reason",
    ]
    display_cols = [col for col in display_cols if col in merged.columns]
    st.dataframe(merged[display_cols], width="stretch", hide_index=True)

    st.markdown("### Topic 상세 드릴다운")
    options = merged["topic_id"].astype(str).tolist()
    selected_topic_id = st.selectbox("상세 Topic 선택", options, index=0, key="v5_topic_detail_select")
    selected = merged[merged["topic_id"].astype(str).eq(selected_topic_id)]
    if selected.empty:
        st.info("선택한 Topic 상세 정보를 찾을 수 없습니다.")
        return

    row = selected.iloc[0]
    st.markdown(f"**Topic**: {_safe_text(row.get('topic_label'))}")
    st.markdown(f"**Insight**: {_safe_text(row.get('insight_text'))}")
    st.success(f"Action: {_safe_text(row.get('action_text'))}")
    st.caption(
        f"근거 강도: {float(pd.to_numeric(row.get('evidence_strength', 0.0), errors='coerce') or 0.0):.3f} | "
        f"정합 점수: {float(pd.to_numeric(row.get('alignment_score', 0.0), errors='coerce') or 0.0):.3f}"
    )

    rep_comments = row.get("representative_comments", []) or []
    if rep_comments:
        st.caption("대표 댓글")
        for comment in rep_comments[:3]:
            st.markdown(f"- {comment}")

    signal_units = outputs.get("v5_signal_units", pd.DataFrame())
    if signal_units.empty:
        return
    related = signal_units[signal_units["topic_id"].astype(str).eq(selected_topic_id)].copy()
    if related.empty:
        return
    st.caption("관련 댓글(보조)")
    related_cols = [col for col in ["comment_id", "signal_text", "signal_type", "polarity", "journey_stage", "judgment_axis"] if col in related.columns]
    with st.expander("관련 댓글 상세 보기"):
        st.dataframe(related[related_cols].head(80), width="stretch", hide_index=True)


def render_v5_page(context: V5PageContext) -> None:
    st.markdown("## 전략 인사이트 (V5 Topic-First)")
    st.caption("실험 페이지: 현재 필터 범위를 그대로 받아 Topic-First 인사이트를 표시합니다.")

    cache_key = _context_signature(context)
    payload_key = "_v5_page_payload"
    cache_key_name = "_v5_page_cache_key"

    payload = None
    if st.session_state.get(cache_key_name) == cache_key:
        payload = st.session_state.get(payload_key)

    if payload is None:
        with st.spinner("V5 Topic-First 결과를 생성 중입니다..."):
            outputs, error = _prepare_v5_outputs(context)
        if error:
            st.warning(error)
            st.info("현재 V5 결과를 생성하지 못해도 기존 V4 페이지는 정상적으로 사용할 수 있습니다.")
            return
        merged = _merge_for_display(outputs, context.analysis_non_trash if not context.analysis_non_trash.empty else context.filtered_comments)
        payload = {"outputs": outputs, "merged": merged}
        st.session_state[cache_key_name] = cache_key
        st.session_state[payload_key] = payload

    outputs = payload.get("outputs", {})
    merged = payload.get("merged", pd.DataFrame())
    if merged is None or merged.empty:
        st.warning("V5 Topic 결과가 비어 있습니다. 현재 필터 범위를 넓혀 다시 확인해 주세요.")
        return

    _render_scope_summary(context, outputs, merged)

    top_topics = merged[merged.get("top_quality_pass", pd.Series(False, index=merged.index)).fillna(False).astype(bool)].copy()
    _render_top_topic_cards(top_topics)

    _render_distribution(merged)
    _render_topic_table_and_drilldown(merged, outputs)

