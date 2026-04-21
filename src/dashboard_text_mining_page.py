"""Text Mining pattern tab renderer (UI-only presentation layer)."""

from __future__ import annotations

from dataclasses import dataclass
import ast
import re
from typing import Any

import pandas as pd
import streamlit as st

from src.text_mining.pattern_builder import build_top_patterns
from src.text_mining.pattern_quality_gate import filter_pattern_quality
from src.text_mining.preprocess import build_base_frame, resolve_text_column, select_scope_source


@dataclass(frozen=True)
class TextMiningPageContext:
    selected_filters: dict[str, Any]
    active_mode: str
    active_snapshot_run: str
    filtered_comments: pd.DataFrame
    filtered_videos: pd.DataFrame
    analysis_non_trash: pd.DataFrame


def build_text_mining_context(
    *,
    selected_filters: dict[str, Any],
    active_mode: str,
    active_snapshot_run: str,
    filtered_comments: pd.DataFrame,
    filtered_videos: pd.DataFrame,
    analysis_non_trash: pd.DataFrame,
) -> TextMiningPageContext:
    return TextMiningPageContext(
        selected_filters=selected_filters or {},
        active_mode=str(active_mode or "real"),
        active_snapshot_run=str(active_snapshot_run or ""),
        filtered_comments=filtered_comments.copy() if filtered_comments is not None else pd.DataFrame(),
        filtered_videos=filtered_videos.copy() if filtered_videos is not None else pd.DataFrame(),
        analysis_non_trash=analysis_non_trash.copy() if analysis_non_trash is not None else pd.DataFrame(),
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


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(v) for v in value if _safe_text(v)]
    raw = _safe_text(value)
    if not raw:
        return []
    if raw.startswith("[") and raw.endswith("]"):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [_safe_text(v) for v in parsed if _safe_text(v)]
        except Exception:
            pass
    return [part.strip() for part in raw.split(",") if part.strip()]


def _scope_line(values: list[str], empty_label: str = "전체") -> str:
    cleaned = [_safe_text(v) for v in values if _safe_text(v)]
    if not cleaned:
        return empty_label
    if len(cleaned) <= 3:
        return ", ".join(cleaned)
    return f"{', '.join(cleaned[:3])} 외 {len(cleaned) - 3}개"


def _extract_comment_text_lookup(source: pd.DataFrame) -> dict[str, str]:
    if source is None or source.empty:
        return {}
    text_col = resolve_text_column(source) or ("text_display" if "text_display" in source.columns else "raw_text")
    if text_col not in source.columns:
        return {}
    comment_ids = source.get("comment_id", pd.Series("", index=source.index)).astype(str)
    texts = source[text_col].fillna("").astype(str)
    return dict(zip(comment_ids, texts))


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", _safe_text(text)).strip()


def _comment_snippet(text: str, *, max_chars: int = 170) -> str:
    value = _normalize_space(text)
    if not value:
        return ""
    sentence_break = re.search(r"[.!?。！？]\s+", value)
    if sentence_break and sentence_break.end() >= 80:
        value = value[: sentence_break.end()].strip()
    if len(value) <= max_chars:
        return value
    return f"{value[:max_chars].rstrip()}..."


def _display_pattern_label(pattern_label: str, keyword_set: list[str]) -> str:
    tokens = {token.lower() for token in keyword_set}
    if {"rinse", "hot", "cycle"}.issubset(tokens):
        return "린스·온수 사이클 체감 성능 이슈"
    if {"pod", "powder", "detergent"}.issubset(tokens):
        return "세제 방식(파우더/팟) 사용성·비용 이슈"
    if {"filter", "rinse", "remove"}.issubset(tokens) or {"filter", "smell"}.issubset(tokens):
        return "필터 관리·세척 냄새 이슈"
    if {"service", "repair"}.issubset(tokens):
        return "A/S·수리 대응 신뢰 이슈"
    compact = [token for token in keyword_set if token][:3]
    if compact:
        return f"반복 이슈: {' · '.join(compact)}"
    return pattern_label or "반복 이슈"


def _build_pattern_insight(pattern_label: str, coverage: float, keyword_set: list[str]) -> str:
    focus_terms = [token for token in keyword_set if token][:2]
    focus_text = ", ".join(focus_terms) if focus_terms else pattern_label
    return (
        f"`{pattern_label}` 축 반응이 현재 범위에서 반복 관찰됩니다 "
        f"(커버리지 {coverage:.3f}%). 핵심 단서 키워드는 `{focus_text}`이며, "
        "운영/제품 판단 시 우선 확인할 신호로 볼 수 있습니다."
    )


def _build_pattern_action(keyword_set: list[str]) -> str:
    tokens = {token.lower() for token in keyword_set}
    if tokens.intersection({"price", "expensive", "cost", "가격", "비용"}):
        return "가격·비용 부담 구간을 우선 점검하고, 대체 옵션 안내를 단기 실행안으로 검토하는 것을 권장합니다."
    if tokens.intersection({"filter", "smell", "remove", "maintenance", "관리"}):
        return "필터 청소/교체 절차를 단순화하고, 관리 주기 안내를 운영 표준으로 정리하는 방향을 우선 검토하세요."
    if tokens.intersection({"rinse", "hot", "cycle", "dry", "세척", "온수"}):
        return "린스·온수·사이클 권장 조합을 재점검하고, 사용자 설정 가이드를 명확히 고정하는 방안을 우선 검토하세요."
    if tokens.intersection({"repair", "service", "failure", "수리", "서비스"}):
        return "재발 빈도가 높은 고장 유형부터 A/S 대응 절차를 점검하고, 처리 우선순위 보강안을 검토하세요."
    return "반복 키워드가 가리키는 원인 구간을 좁혀, 즉시 실행 가능한 개선 항목부터 우선 검토하세요."


def _prepare_tm_pattern_outputs(
    context: TextMiningPageContext,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    source, _ = select_scope_source(
        {
            "analysis_non_trash": context.analysis_non_trash,
            "filtered_comments": context.filtered_comments,
        }
    )
    if source.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "현재 필터 범위에서 텍스트 마이닝에 사용할 댓글이 없습니다."

    base_df = build_base_frame(source)
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "텍스트 정제 결과가 비어 있어 패턴을 생성할 수 없습니다."

    raw_patterns = build_top_patterns(base_df, top_n=10, min_pair_count=5)
    if raw_patterns.empty:
        return base_df, raw_patterns, pd.DataFrame(), pd.DataFrame(), "생성 가능한 패턴이 없어 결과를 표시할 수 없습니다."

    quality = filter_pattern_quality(
        base_df,
        patterns_df=raw_patterns,
        min_coverage=0.5,
        min_rep_keyword_overlap=2,
        max_top_n=5,
    )
    final_patterns = quality.get("final_top_patterns", pd.DataFrame())
    removed_patterns = quality.get("removed_patterns", pd.DataFrame())
    if final_patterns.empty:
        return base_df, raw_patterns, final_patterns, removed_patterns, "품질 게이트를 통과한 패턴이 없어 상단 카드에 표시할 항목이 없습니다."

    text_lookup = _extract_comment_text_lookup(source)
    enriched_rows: list[dict[str, Any]] = []
    for row in final_patterns.itertuples(index=False):
        label = _safe_text(getattr(row, "pattern_label", ""))
        coverage = float(getattr(row, "coverage", 0.0) or 0.0)
        keyword_set = _to_list(getattr(row, "keyword_set", []))
        rep_ids = _to_list(getattr(row, "representative_comment_ids", []))[:3]
        rep_full = [text_lookup.get(comment_id, "") for comment_id in rep_ids if text_lookup.get(comment_id, "")]
        rep_snippets = [_comment_snippet(text) for text in rep_full if _comment_snippet(text)]
        enriched_rows.append(
            {
                "pattern_label": label,
                "display_label": _display_pattern_label(label, keyword_set),
                "coverage": coverage,
                "keyword_set": keyword_set,
                "representative_comment_ids": rep_ids,
                "representative_comment_snippets": rep_snippets[:3],
                "representative_comment_full": rep_full[:3],
                "insight": _build_pattern_insight(label, coverage, keyword_set),
                "action": _build_pattern_action(keyword_set),
            }
        )

    final_enriched = pd.DataFrame(enriched_rows)
    return base_df, raw_patterns, final_enriched, removed_patterns, ""


def render_text_mining_page(context: TextMiningPageContext) -> None:
    base_df, raw_patterns, final_patterns, removed_patterns, warning_message = _prepare_tm_pattern_outputs(context)
    selected_filters = context.selected_filters or {}
    query_text = _safe_text(selected_filters.get("keyword_query")) or "-"
    video_count = (
        int(context.filtered_videos["video_id"].astype(str).nunique())
        if not context.filtered_videos.empty and "video_id" in context.filtered_videos.columns
        else int(len(context.filtered_videos))
    )
    comment_count = int(len(base_df))
    pattern_count = int(len(final_patterns))

    st.markdown("### 전략 인사이트 (Text Mining 패턴)")
    st.caption("패턴 카드는 반복 신호를 빠르게 파악하기 위한 해석 보조 화면입니다. 최종 결론은 상세 근거와 함께 검토해 주세요.")

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("검색어", query_text)
    s2.metric("영상 수", f"{video_count:,}")
    s3.metric("댓글 수", f"{comment_count:,}")
    s4.metric("패턴 수", f"{pattern_count:,}")
    st.caption(
        f"현재 필터 범위: 제품({_scope_line(list(selected_filters.get('products', [])))}), "
        f"시장({_scope_line(list(selected_filters.get('regions', [])))}), "
        f"브랜드({_scope_line(list(selected_filters.get('brands', [])))})."
    )

    if warning_message:
        st.warning(warning_message)
    if final_patterns.empty:
        return

    st.markdown("#### 우선 확인 패턴")
    for rank, row in enumerate(final_patterns.itertuples(index=False), start=1):
        with st.container(border=True):
            st.markdown(f"**{rank}. {row.display_label}**")
            st.caption(f"패턴 키: `{row.pattern_label}`")
            tag_left, tag_right = st.columns([1, 3])
            tag_left.caption(f"Coverage: {float(row.coverage):.3f}%")
            tag_right.caption(f"핵심 키워드: {', '.join(_to_list(row.keyword_set))}")
            st.markdown(f"**해석**: {_safe_text(row.insight)}")
            st.success(f"실행 제안: {_safe_text(row.action)}")

            rep_snippets = (
                row.representative_comment_snippets
                if isinstance(row.representative_comment_snippets, list)
                else _to_list(row.representative_comment_snippets)
            )
            rep_full = (
                row.representative_comment_full
                if isinstance(row.representative_comment_full, list)
                else _to_list(row.representative_comment_full)
            )
            if rep_snippets:
                st.caption("대표 반응 (핵심 발췌)")
                for snippet in rep_snippets[:3]:
                    st.markdown(f"- {snippet}")
            if rep_full:
                with st.expander("원문 근거 보기"):
                    for full_text in rep_full[:3]:
                        st.markdown(f"- {_normalize_space(full_text)}")

    with st.expander("제외된 패턴 보기"):
        if removed_patterns.empty:
            st.caption("제외된 패턴이 없습니다.")
        else:
            show_cols = [col for col in ["pattern_label", "coverage", "removed_reason"] if col in removed_patterns.columns]
            st.dataframe(removed_patterns[show_cols], width="stretch", hide_index=True)

    with st.expander("패턴 생성 결과(참고) 보기"):
        if raw_patterns.empty:
            st.caption("원본 패턴이 없습니다.")
        else:
            st.dataframe(raw_patterns, width="stretch", hide_index=True)
