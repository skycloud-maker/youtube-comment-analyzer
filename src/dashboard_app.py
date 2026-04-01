from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics.keywords import extract_keywords
from src.dashboard_rules import ui_labels
from src.utils.coverage_audit import build_run_coverage_audit, summarize_coverage_audit

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

COL_COUNTRY = "국가"
COL_SENTIMENT = "감성"
COL_CEJ = "고객여정단계"
COL_BRAND = "브랜드"
COL_WRITTEN_AT = "작성일시"
COL_WEEK_START = "주차시작"
COL_WEEK_LABEL = "주차표기"
COL_ORIGINAL = "원문"
COL_TRANSLATION = "번역(참고)"
COL_VIDEO_TITLE = "영상제목"
COL_VIDEO_LINK = "영상 링크"
COL_LIKES = "좋아요 수"
COL_LANGUAGE = "언어"
COL_RAW = "원본 댓글"
COL_REMOVED = "제거 사유"
COL_CONTEXT = "연관 맥락"
COL_CONTEXT_TRANSLATION = "연관 맥락 번역"
COL_SENTIMENT_CODE = "sentiment_code"

REGION_LABELS = {"KR": "한국", "US": "미국"}
SENTIMENT_CODE_TO_LABEL = {"positive": "긍정", "negative": "부정", "neutral": "중립", "excluded": "제외"}
SENTIMENT_LABEL_TO_CODE = {value: key for key, value in SENTIMENT_CODE_TO_LABEL.items()}
ANALYSIS_SCOPE_OPTIONS = ["전체", "문의 포함 댓글만", "문의 제외"]
EXPORT_SHEETS = {
    "댓글원문": lambda frames: frames[0],
    "영상목록": lambda frames: frames[1],
    "주간감성": lambda frames: frames[2],
    "CEJ부정률": lambda frames: frames[3],
    "브랜드언급": lambda frames: frames[4],
    "영상밀도분석": lambda frames: frames[5],
}
RUN_DATASETS = {
    "comments": ["comments.parquet", "comments_raw.parquet"],
    "videos": ["videos_normalized.parquet", "videos_raw.parquet"],
    "weekly_sentiment": ["weekly_sentiment_trend.parquet"],
    "negative_density": ["negative_density.parquet"],
    "representative_bundles": ["representative_bundles.parquet"],
    "opinion_units": ["opinion_units.parquet"],
    "quality_summary": ["quality_summary.parquet"],
    "brand_ratio": ["brand_ratio.parquet"],
    "cej_negative_rate": ["cej_negative_rate.parquet"],
    "video_ranking": ["video_ranking.parquet", "videos_raw.parquet"],
}


def _read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            if path.suffix == ".csv":
                return pd.read_csv(path)
            return pd.read_parquet(path)
        except Exception:
            return pd.DataFrame()
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        try:
            return pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _read_first_available(run_dir: Path, filenames: list[str]) -> pd.DataFrame:
    for filename in filenames:
        frame = _read_frame(run_dir / filename)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _dashboard_run_score(run_dir: Path) -> tuple[int, int, int]:
    if not run_dir.is_dir() or not (run_dir / "run_manifest.json").exists():
        return (-1, -1, -1)
    comments = _read_first_available(run_dir, RUN_DATASETS["comments"])
    videos = _read_first_available(run_dir, RUN_DATASETS["videos"])
    weekly = _read_first_available(run_dir, RUN_DATASETS["weekly_sentiment"])
    reps = _read_first_available(run_dir, RUN_DATASETS["representative_bundles"])
    score = 0
    score += 4 if not comments.empty else 0
    score += 3 if not videos.empty else 0
    score += 2 if not weekly.empty else 0
    score += 2 if not reps.empty else 0
    richness = sum(len(_read_first_available(run_dir, filenames)) for filenames in RUN_DATASETS.values())
    return (score, richness, int(run_dir.stat().st_mtime_ns))


def _pick_dashboard_run_dir(processed_dir: Path) -> Path | None:
    candidates = [p for p in processed_dir.iterdir()] if processed_dir.exists() else []
    valid = [p for p in candidates if p.is_dir() and (p / "run_manifest.json").exists()]
    if not valid:
        return None
    return max(valid, key=_dashboard_run_score)


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    run_dir = _pick_dashboard_run_dir(PROCESSED_DIR)
    if run_dir is None:
        return {key: pd.DataFrame() for key in RUN_DATASETS}
    payload: dict[str, pd.DataFrame] = {}
    for key, filenames in RUN_DATASETS.items():
        frame = _read_first_available(run_dir, filenames)
        if not frame.empty and "run_id" not in frame.columns:
            frame["run_id"] = run_dir.name
        payload[key] = frame
    payload["run_id"] = pd.DataFrame([{"run_id": run_dir.name}])
    return payload


def localize_region(value: str) -> str:
    return REGION_LABELS.get(str(value), str(value))


def render_card(title: str, value: str) -> None:
    st.markdown(
        f'<div style="background:#fff;border:1px solid #e5e7eb;border-radius:16px;padding:16px 18px;">'
        f'<div style="font-size:13px;color:#6b7280;">{title}</div>'
        f'<div style="font-size:28px;font-weight:700;color:#111827;">{value}</div></div>',
        unsafe_allow_html=True,
    )


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_garbled(value: str) -> bool:
    text = _safe_text(value)
    if not text:
        return False
    return "???" in text or "�" in text or any(ord(ch) < 32 and ch not in "\t\n\r" for ch in text)


def _clean_ui_text(value: Any, fallback: str = "") -> str:
    text = _safe_text(value)
    if not text or _looks_garbled(text):
        return fallback
    return text


def _coverage_reason_label(reason_code: str) -> str:
    mapping = {
        "COVERED": "정상 반영",
        "COMMENTS_DISABLED_OR_ZERO": "댓글 0건 또는 댓글 비활성화",
        "COMMENTS_NOT_COLLECTED": "댓글 수집 미완료",
        "COMMENTS_AUDIT_MISSING": "댓글 감사로그 없음",
        "INGEST_API_ERROR": "수집 API 오류",
        "TRIMMED_BY_MAX_VIDEOS": "수집 상한으로 제외",
        "COVERAGE_MISSING_HOT_VIDEO": "설명 없는 누락",
    }
    return mapping.get(str(reason_code or ""), str(reason_code or "-"))


def _latest_run_with_manifest_for_coverage(processed_dir: Path) -> str | None:
    run_dir = _pick_dashboard_run_dir(processed_dir)
    return run_dir.name if run_dir is not None else None


def _load_latest_coverage_summary(processed_dir: Path, raw_dir: Path) -> tuple[str | None, dict[str, int], pd.DataFrame]:
    run_id = _latest_run_with_manifest_for_coverage(processed_dir)
    if run_id is None:
        return None, {"hot_candidates": 0, "covered": 0, "missing": 0, "trimmed": 0, "missing_comments": 0}, pd.DataFrame()
    raw_frame = build_run_coverage_audit(processed_dir, raw_dir, run_id, hot_rank_cutoff=10)
    frame = raw_frame.copy()
    if not frame.empty:
        frame["coverage_status"] = frame["coverage_status"].map(
            {"covered": "정상 반영", "missing_comments": "댓글 미반영", "trimmed": "수집 상한 제외", "missing": "누락"}
        ).fillna(frame["coverage_status"])
        frame["reason_code"] = frame["reason_code"].map(_coverage_reason_label).fillna(frame["reason_code"])
    return run_id, summarize_coverage_audit(raw_frame), frame


def _coverage_scope_label(selected_filters: dict[str, Any]) -> str:
    products = ", ".join(selected_filters.get("products") or ["전체"])
    regions = ", ".join(selected_filters.get("regions") or ["전체"])
    return f"제품 {products} / 국가 {regions}"


def _prepare_coverage_display_frame(coverage_audit: pd.DataFrame, filtered_videos: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if coverage_audit is None or coverage_audit.empty:
        return pd.DataFrame(), {"hot_candidates": 0, "covered": 0, "missing": 0, "trimmed": 0, "missing_comments": 0}
    scoped = coverage_audit.copy()
    if filtered_videos is not None and not filtered_videos.empty and "video_id" in filtered_videos.columns:
        visible_ids = set(filtered_videos["video_id"].astype("string").dropna().tolist())
        filtered_scope = scoped[scoped["video_id"].astype("string").isin(visible_ids)].copy()
        if not filtered_scope.empty:
            scoped = filtered_scope
    display = scoped[["search_rank", "video_id", "coverage_status", "reason_code", "api_comment_count", "title"]].rename(
        columns={
            "search_rank": "검색 순위",
            "coverage_status": "상태",
            "reason_code": "사유",
            "api_comment_count": "API 댓글 수",
            "title": "영상 제목",
        }
    )
    counts = scoped.rename(columns={"상태": "coverage_status"}) if "상태" in scoped.columns else scoped
    counts = counts.copy()
    if "coverage_status" in counts.columns:
        counts["coverage_status"] = counts["coverage_status"].map({
            "정상 반영": "covered",
            "댓글 미반영": "missing_comments",
            "수집 상한 제외": "trimmed",
            "누락": "missing",
        }).fillna(counts["coverage_status"])
    return display, summarize_coverage_audit(counts)


def _collect_similar_comments_from_samples(selected: pd.Series) -> pd.DataFrame:
    payload = selected.get("sample_comments_json", "[]")
    try:
        items = json.loads(payload) if isinstance(payload, str) else list(payload or [])
    except Exception:
        items = []
    rows = []
    for item in items:
        rows.append(
            {
                "comment_id": item.get("comment_id", ""),
                COL_ORIGINAL: item.get("display_text", ""),
                COL_TRANSLATION: item.get("translated_text", ""),
                COL_LIKES: item.get("likes_count", 0),
                "PII": item.get("pii_flag", "N"),
                COL_WRITTEN_AT: item.get("written_at", ""),
            }
        )
    return pd.DataFrame(rows, columns=["comment_id", COL_ORIGINAL, COL_TRANSLATION, COL_LIKES, "PII", COL_WRITTEN_AT])


def _rows_from_representative_bundles(bundle_df: pd.DataFrame, *, sentiment: str, limit: int = 3) -> pd.DataFrame:
    if bundle_df is None or bundle_df.empty or "sentiment_code" not in bundle_df.columns:
        return pd.DataFrame()
    filtered = bundle_df[bundle_df["sentiment_code"].astype(str).eq(sentiment)].head(limit).copy()
    if filtered.empty:
        return pd.DataFrame()
    labels = ui_labels()
    rows = []
    for _, row in filtered.iterrows():
        rows.append(
            {
                COL_ORIGINAL: _safe_text(row.get("top_display_text") or row.get("top_original_text")),
                COL_VIDEO_TITLE: _safe_text(row.get("top_video_title")),
                COL_CEJ: _safe_text(row.get("cej_scene_code", "")),
                COL_BRAND: _safe_text(row.get("brand_mentioned", "")),
                "selection_reason_ui": "대표성 판단 근거를 별도 검토 중" if _looks_garbled(_safe_text(row.get("selection_reason", ""))) else _safe_text(row.get("selection_reason", "")),
                "header_ui": f"Top{int(row.get('rank', 1) or 1)} {_safe_text(row.get('cej_scene_code', '기타'))} 단계 {SENTIMENT_CODE_TO_LABEL.get(sentiment, sentiment)} 코멘트",
                labels.get("representative_related_video", COL_VIDEO_TITLE): _safe_text(row.get("top_video_title")),
            }
        )
    return pd.DataFrame(rows)


def build_weekly_sentiment_window(comments_df: pd.DataFrame, weeks_per_view: int = 8, page: int = 1) -> tuple[pd.DataFrame, int]:
    if comments_df is None or comments_df.empty:
        return pd.DataFrame(columns=[COL_WEEK_LABEL, COL_SENTIMENT_CODE, "count"]), 1
    frame = comments_df.copy()
    if COL_WEEK_START not in frame.columns:
        frame[COL_WEEK_START] = pd.NaT
    if COL_WEEK_LABEL not in frame.columns:
        frame[COL_WEEK_LABEL] = frame[COL_WEEK_START].astype(str)
    frame = frame[frame[COL_WEEK_LABEL].ne("NaT")]
    sentiments = ["positive", "negative"]
    weeks = sorted(frame[COL_WEEK_LABEL].dropna().unique().tolist())
    if not weeks:
        return pd.DataFrame(columns=[COL_WEEK_LABEL, COL_SENTIMENT_CODE, "count"]), 1
    total_pages = max(1, (len(weeks) + weeks_per_view - 1) // weeks_per_view)
    page = max(1, min(page, total_pages))
    start = (page - 1) * weeks_per_view
    page_weeks = weeks[start : start + weeks_per_view]
    grouped = (
        frame[frame[COL_WEEK_LABEL].isin(page_weeks)]
        .groupby([COL_WEEK_LABEL, "sentiment_label"], dropna=False)
        .size()
        .to_dict()
    )
    rows = []
    for week in page_weeks:
        for sentiment in sentiments:
            rows.append({COL_WEEK_LABEL: week, COL_SENTIMENT_CODE: sentiment, "count": int(grouped.get((week, sentiment), 0))})
    return pd.DataFrame(rows), total_pages


def build_raw_download_package(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    cej_df: pd.DataFrame,
    brand_df: pd.DataFrame,
    density_df: pd.DataFrame,
) -> bytes:
    frames = [comments_df, videos_df, weekly_df, cej_df, brand_df, density_df]
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, picker in EXPORT_SHEETS.items():
            picker(frames).to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return output.getvalue()


def compute_filtered_bundle(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    selected_filters: dict[str, Any],
    representative_bundles: pd.DataFrame,
    opinion_units: pd.DataFrame,
) -> dict[str, Any]:
    filtered = comments_df.copy()
    if filtered.empty:
        return {"comments": filtered, "videos": videos_df.copy(), "meaningful_count": 0, "removed_count": 0, "inquiry_count": 0, "all_comments": filtered}
    if selected_filters.get("products") and "product" in filtered.columns:
        filtered = filtered[filtered["product"].isin(selected_filters["products"])]
    if selected_filters.get("regions"):
        if COL_COUNTRY not in filtered.columns and "region" in filtered.columns:
            filtered[COL_COUNTRY] = filtered["region"].map(localize_region)
        filtered = filtered[filtered[COL_COUNTRY].isin(selected_filters["regions"])]
    if selected_filters.get("brands") and COL_BRAND in filtered.columns:
        filtered = filtered[filtered[COL_BRAND].isin(selected_filters["brands"])]
    if selected_filters.get("cej") and COL_CEJ in filtered.columns:
        filtered = filtered[filtered[COL_CEJ].isin(selected_filters["cej"])]
    if selected_filters.get("sentiments") and COL_SENTIMENT in filtered.columns:
        filtered = filtered[filtered[COL_SENTIMENT].isin(selected_filters["sentiments"])]
    keyword_query = str(selected_filters.get("keyword_query") or "").strip().lower()
    if keyword_query:
        source = filtered.get("text_display", pd.Series("", index=filtered.index)).astype(str).str.lower()
        filtered = filtered[source.str.contains(keyword_query, na=False)]

    inquiry_ids: set[str] = set()
    if opinion_units is not None and not opinion_units.empty and "inquiry_flag" in opinion_units.columns:
        inquiry_ids = set(
            opinion_units[opinion_units["inquiry_flag"].fillna(False)]["source_content_id"].astype(str).tolist()
        )
    scope = selected_filters.get("analysis_scope", "전체")
    if scope == "문의 포함 댓글만" and inquiry_ids:
        filtered = filtered[filtered.get("comment_id", pd.Series("", index=filtered.index)).astype(str).isin(inquiry_ids)]
    elif scope == "문의 제외" and inquiry_ids:
        filtered = filtered[~filtered.get("comment_id", pd.Series("", index=filtered.index)).astype(str).isin(inquiry_ids)]

    filtered_videos = videos_df.copy()
    if not filtered_videos.empty and "video_id" in filtered_videos.columns and "video_id" in filtered.columns:
        filtered_videos = filtered_videos[filtered_videos["video_id"].astype(str).isin(filtered["video_id"].astype(str))]

    validity = filtered.get("comment_validity", pd.Series("valid", index=filtered.index))
    meaningful_count = int(validity.eq("valid").sum())
    removed_count = int(validity.eq("excluded").sum())
    inquiry_count = int(filtered.get("comment_id", pd.Series("", index=filtered.index)).astype(str).isin(inquiry_ids).sum())
    return {
        "comments": filtered,
        "videos": filtered_videos,
        "meaningful_count": meaningful_count,
        "removed_count": removed_count,
        "inquiry_count": inquiry_count,
        "all_comments": filtered.copy(),
    }


def _build_filter_options(comments_df: pd.DataFrame) -> dict[str, list[str]]:
    products = sorted(comments_df.get("product", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    regions = sorted(comments_df.get("region", pd.Series(dtype=str)).dropna().map(localize_region).astype(str).unique().tolist())
    brands = sorted(comments_df.get(COL_BRAND, pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
    sentiments = ["긍정", "부정", "중립"]
    return {"products": products, "regions": regions, "brands": brands, "sentiments": sentiments}


def _filter_structured_frame(frame: pd.DataFrame, selected_filters: dict[str, Any], brand_col: str | None = None) -> pd.DataFrame:
    scoped = frame.copy()
    if scoped.empty:
        return scoped
    if selected_filters.get("products") and "product" in scoped.columns:
        scoped = scoped[scoped["product"].isin(selected_filters["products"])]
    if selected_filters.get("regions"):
        if "region" in scoped.columns and COL_COUNTRY not in scoped.columns:
            scoped[COL_COUNTRY] = scoped["region"].map(localize_region)
        if COL_COUNTRY in scoped.columns:
            scoped = scoped[scoped[COL_COUNTRY].isin(selected_filters["regions"])]
    if brand_col and selected_filters.get("brands") and brand_col in scoped.columns:
        scoped = scoped[scoped[brand_col].isin(selected_filters["brands"])]
    return scoped


def _keyword_panel(filtered_comments: pd.DataFrame, sentiment_code: str, title: str) -> None:
    source = filtered_comments.copy()
    if source.empty or "sentiment_label" not in source.columns:
        st.info(f"{title}에 표시할 데이터가 없습니다.")
        return
    source = source[source["sentiment_label"].astype(str).eq(sentiment_code)]
    if source.empty:
        st.info(f"{title}에 표시할 데이터가 없습니다.")
        return
    keywords = extract_keywords(source.get("text_display", pd.Series(dtype=str)).fillna(""), top_n=12)
    st.markdown(f"### {title}")
    if keywords.empty:
        st.info("키워드 추출 결과가 없습니다.")
    else:
        st.dataframe(keywords.rename(columns={"keyword": "키워드", "count": "언급 수"}), use_container_width=True, hide_index=True)


def _render_representative_section(rep_df: pd.DataFrame, selected_filters: dict[str, Any]) -> None:
    labels = ui_labels()
    scoped = _filter_structured_frame(rep_df, selected_filters, brand_col="brand_mentioned")
    st.markdown(f"### {labels.get('representative_section_title', '핵심 대표 코멘트')}")
    st.caption(labels.get('representative_section_caption_ko', '현재 필터 범위에서 반복적으로 나타나는 제품 의견을 대표 코멘트 중심으로 정리합니다.'))
    if scoped.empty:
        st.info(labels.get('representative_empty_info', '현재 조건에서 보여줄 대표 코멘트가 없습니다.'))
        return
    for sentiment_code, title in [("negative", labels.get("representative_negative_label", "부정 대표 코멘트")), ("positive", labels.get("representative_positive_label", "긍정 대표 코멘트"))]:
        rows = scoped[scoped["sentiment_code"].astype(str).eq(sentiment_code)].sort_values(["rank", "bundle_score"], ascending=[True, False]).head(2)
        if rows.empty:
            continue
        st.markdown(f"#### {title}")
        for _, row in rows.iterrows():
            with st.container(border=True):
                stage = _safe_text(row.get("cej_scene_code", "기타"))
                sentiment_label = SENTIMENT_CODE_TO_LABEL.get(sentiment_code, sentiment_code)
                st.markdown(f"**Top{int(row.get('rank', 1) or 1)} {stage} 단계 {sentiment_label} 코멘트**")
                st.caption(f"브랜드 {(_safe_text(row.get('brand_mentioned')) or '미언급')} · 제품 {(_safe_text(row.get('product')) or '미지정')} · 유사 의견 {int(row.get('cluster_size', 0) or 0)}건")
                st.write(f"{labels.get('representative_related_video', '영상제목')}: {_clean_ui_text(row.get('top_source_link'), fallback='링크 정보 없음')}")
                st.write(f"{labels.get('representative_summary', '요약(1~2문장)')}: {_clean_ui_text(row.get('selection_reason'), fallback='대표성 판단 근거를 별도 검토 중')}")
                original = _clean_ui_text(row.get('top_display_text') or row.get('top_original_text'))
                if original:
                    snippet = original.splitlines()[0][:180]
                    st.write(f"{labels.get('representative_evidence', '근거 발췌(1~2개)')}: {snippet}")
                similar = _collect_similar_comments_from_samples(pd.Series(row))
                if not similar.empty:
                    st.write("유사 의견 예시")
                    st.dataframe(similar.head(3), use_container_width=True, hide_index=True)


def _render_inquiry_section(opinion_units: pd.DataFrame, selected_filters: dict[str, Any]) -> None:
    if opinion_units is None or opinion_units.empty or "inquiry_flag" not in opinion_units.columns:
        return
    scoped = _filter_structured_frame(opinion_units, selected_filters, brand_col="brand_mentioned")
    scoped = scoped[scoped["inquiry_flag"].fillna(False)]
    st.markdown("### 반복 문의 코멘트")
    if scoped.empty:
        st.info("현재 조건에서 반복 문의로 분류된 코멘트가 없습니다.")
        return
    summary = (
        scoped.groupby(["aspect_key", "cej_scene_code"], dropna=False)
        .size()
        .reset_index(name="문의 수")
        .sort_values("문의 수", ascending=False)
        .head(10)
        .rename(columns={"aspect_key": "문의 유형", "cej_scene_code": "고객여정단계"})
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)


def _render_video_summary(videos_df: pd.DataFrame, density_df: pd.DataFrame, selected_filters: dict[str, Any]) -> None:
    st.markdown("### 영상 분석 요약")
    scoped_videos = _filter_structured_frame(videos_df, selected_filters)
    scoped_density = _filter_structured_frame(density_df, selected_filters)
    if not scoped_videos.empty:
        summary = scoped_videos[["title", "channel_title", "view_count", "comment_count", "search_rank"]].copy()
        summary = summary.rename(columns={"title": "영상 제목", "channel_title": "채널", "view_count": "조회수", "comment_count": "댓글 수", "search_rank": "검색 순위"})
        st.dataframe(summary.sort_values("검색 순위").head(10), use_container_width=True, hide_index=True)
    else:
        st.info("현재 조건에서 보여줄 영상 요약이 없습니다.")
    st.markdown("### 영상별 부정 댓글 밀도")
    if not scoped_density.empty:
        density = scoped_density.rename(columns={"title": "영상 제목", "negative_comments": "부정 댓글 수", "total_comments": "전체 댓글 수", "negative_density": "부정 밀도"})
        st.dataframe(density[["영상 제목", "부정 댓글 수", "전체 댓글 수", "부정 밀도"]], use_container_width=True, hide_index=True)
    else:
        st.info("현재 조건에서 보여줄 밀도 정보가 없습니다.")


def main() -> None:
    st.set_page_config(page_title="YouTube Appliance VoC Dashboard", layout="wide")
    data = load_dashboard_data()
    comments_df = data.get("comments", pd.DataFrame()).copy()
    videos_df = data.get("videos", pd.DataFrame()).copy()
    weekly_sentiment_df = data.get("weekly_sentiment", pd.DataFrame()).copy()
    negative_density_df = data.get("negative_density", pd.DataFrame()).copy()
    representative_bundles_df = data.get("representative_bundles", pd.DataFrame()).copy()
    opinion_units_df = data.get("opinion_units", pd.DataFrame()).copy()
    brand_ratio_df = data.get("brand_ratio", pd.DataFrame()).copy()
    cej_negative_df = data.get("cej_negative_rate", pd.DataFrame()).copy()
    run_id_frame = data.get("run_id", pd.DataFrame())
    active_run_id = _safe_text(run_id_frame.iloc[0]["run_id"]) if not run_id_frame.empty else "-"

    if COL_COUNTRY not in comments_df.columns and "region" in comments_df.columns:
        comments_df[COL_COUNTRY] = comments_df["region"].map(localize_region)
    if COL_BRAND not in comments_df.columns:
        comments_df[COL_BRAND] = comments_df.get("brand_mentioned", "")
    if COL_CEJ not in comments_df.columns:
        comments_df[COL_CEJ] = comments_df.get("cej_scene_code", "기타")
    if COL_SENTIMENT not in comments_df.columns and "sentiment_label" in comments_df.columns:
        comments_df[COL_SENTIMENT] = comments_df["sentiment_label"].map(SENTIMENT_CODE_TO_LABEL).fillna(comments_df["sentiment_label"])

    if not weekly_sentiment_df.empty:
        weekly_sentiment_df = weekly_sentiment_df.copy()
        if COL_WEEK_START not in weekly_sentiment_df.columns and "week_start" in weekly_sentiment_df.columns:
            weekly_sentiment_df[COL_WEEK_START] = weekly_sentiment_df["week_start"]
        if COL_WEEK_LABEL not in weekly_sentiment_df.columns and "week_label" in weekly_sentiment_df.columns:
            weekly_sentiment_df[COL_WEEK_LABEL] = weekly_sentiment_df["week_label"]

    st.title("가전 VoC 대시보드")
    st.caption(f"현재 필터 기준으로 제품 반응, 문의, 대표 코멘트, 커버리지 상태를 빠르게 점검합니다. 기준 run: {active_run_id}")

    options = _build_filter_options(comments_df)
    with st.sidebar:
        st.markdown("### 탐색 필터")
        selected_products = st.multiselect("제품", options["products"], default=options["products"][:4])
        selected_regions = st.multiselect("국가", options["regions"], default=options["regions"][:2])
        selected_brands = st.multiselect("브랜드", options["brands"], default=[])
        selected_sentiments = st.multiselect("감성 라벨", options["sentiments"], default=[])
        analysis_scope = st.radio("분석 유형", ANALYSIS_SCOPE_OPTIONS, index=0)
        keyword_query = st.text_input("댓글 내 키워드 검색", value="")

    selected_filters = {
        "products": selected_products,
        "regions": selected_regions,
        "brands": selected_brands,
        "cej": [],
        "sentiments": selected_sentiments,
        "keyword_query": keyword_query,
        "analysis_scope": analysis_scope,
    }

    bundle = compute_filtered_bundle(comments_df, videos_df, pd.DataFrame(), selected_filters, representative_bundles_df, opinion_units_df)
    filtered_comments = bundle["comments"]
    filtered_videos = bundle["videos"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        render_card("분석 대상 댓글", f"{len(filtered_comments):,}")
    with kpi2:
        render_card("의미 있는 댓글", f"{bundle['meaningful_count']:,}")
    with kpi3:
        render_card("문의 댓글", f"{bundle['inquiry_count']:,}")
    with kpi4:
        render_card("제외 댓글", f"{bundle['removed_count']:,}")

    coverage_run_id, _, coverage_audit = _load_latest_coverage_summary(PROCESSED_DIR, RAW_DIR)
    coverage_display, coverage_summary = _prepare_coverage_display_frame(coverage_audit, filtered_videos)
    with st.container(border=True):
        st.markdown("### 수집 커버리지 점검")
        if coverage_run_id:
            st.caption(f"최근 수집 run `{coverage_run_id}` 기준입니다. 현재 화면 필터는 {_coverage_scope_label(selected_filters)} 입니다.")
        cov1, cov2, cov3, cov4 = st.columns(4)
        with cov1:
            render_card("핫 후보 수", f"{coverage_summary['hot_candidates']:,}")
        with cov2:
            render_card("정상 반영", f"{coverage_summary['covered']:,}")
        with cov3:
            render_card("댓글 미반영", f"{coverage_summary['missing_comments']:,}")
        with cov4:
            render_card("설명 없는 누락", f"{coverage_summary['missing']:,}")
        if not coverage_display.empty:
            with st.expander("커버리지 상세 보기"):
                st.dataframe(coverage_display, use_container_width=True, hide_index=True)

    tab_comments, tab_videos = st.tabs(["댓글 VoC 대시보드", "영상 분석 요약"])

    with tab_comments:
        weekly_source = weekly_sentiment_df if not weekly_sentiment_df.empty else filtered_comments
        weekly_window, _ = build_weekly_sentiment_window(weekly_source, weeks_per_view=8, page=1)
        if not weekly_window.empty:
            chart = alt.Chart(weekly_window).mark_bar().encode(
                x=alt.X(f"{COL_WEEK_LABEL}:N", title="주차"),
                y=alt.Y("count:Q", title="댓글 수"),
                color=alt.Color(f"{COL_SENTIMENT_CODE}:N", title="감성"),
                xOffset=alt.XOffset(f"{COL_SENTIMENT_CODE}:N"),
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("현재 조건에서 주차별 감성 차트를 그릴 데이터가 없습니다.")

        kw_left, kw_right = st.columns(2)
        with kw_left:
            _keyword_panel(filtered_comments, "negative", "부정 키워드")
        with kw_right:
            _keyword_panel(filtered_comments, "positive", "긍정 키워드")

        _render_representative_section(representative_bundles_df, selected_filters)
        _render_inquiry_section(opinion_units_df, selected_filters)

        if not brand_ratio_df.empty:
            st.markdown("### 브랜드 언급 비중")
            st.dataframe(_filter_structured_frame(brand_ratio_df, selected_filters).rename(columns={"brand_label": "브랜드", "count": "언급 수", "ratio": "비중"}), use_container_width=True, hide_index=True)
        if not cej_negative_df.empty:
            st.markdown("### CEJ 단계별 부정률")
            st.dataframe(_filter_structured_frame(cej_negative_df, selected_filters).rename(columns={"topic_label": "주제", "comments": "댓글 수", "negative_comments": "부정 댓글 수", "negative_rate": "부정률"}), use_container_width=True, hide_index=True)

    with tab_videos:
        _render_video_summary(videos_df, negative_density_df, selected_filters)

    weekly_export, _ = build_weekly_sentiment_window(weekly_source, weeks_per_view=8, page=1)
    st.download_button(
        "전체 Raw Data 다운로드",
        data=build_raw_download_package(filtered_comments, filtered_videos, weekly_export, cej_negative_df, brand_ratio_df, negative_density_df),
        file_name="voc_dashboard_raw_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
