"""Streamlit dashboard for appliance VoC exploration."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "data" / "mplcache").resolve()))

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics.keywords import build_keyword_counter, filter_business_keywords, normalize_keyword
from src.utils.translation import translate_to_korean

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "dashboard_bundle.pkl"
CACHE_META = CACHE_DIR / "dashboard_bundle_meta.json"
REGION_LABELS = {"KR": "한국", "US": "미국"}
SENTIMENT_LABELS = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
CEJ_ORDER = ["인지", "구매", "배송", "설치", "사용준비", "사용", "관리교체", "기타"]
PRODUCT_ORDER = ["세탁기", "냉장고", "건조기", "식기세척기"]
FONT_CANDIDATES = [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")]


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    latest_signature = _latest_signature()
    if CACHE_FILE.exists() and CACHE_META.exists():
        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        if meta.get("signature") == latest_signature:
            return pd.read_pickle(CACHE_FILE)

    buckets: dict[str, list[pd.DataFrame]] = {
        "comments": [],
        "videos": [],
        "brand_ratio": [],
        "cej_negative_rate": [],
        "negative_density": [],
        "new_issue_keywords": [],
        "persistent_issue_keywords": [],
        "quality_summary": [],
    }
    run_dirs = sorted(
        (path for path in PROCESSED_DIR.iterdir() if path.is_dir()) if PROCESSED_DIR.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        config_json = manifest.get("config_json", {})
        context = {
            "run_id": manifest.get("run_id"),
            "product": config_json.get("product") or config_json.get("keyword"),
            "region": config_json.get("region"),
        }
        for key, filename in [
            ("comments", "comments.parquet"),
            ("videos", "videos_normalized.parquet"),
            ("brand_ratio", "brand_ratio.parquet"),
            ("cej_negative_rate", "cej_negative_rate.parquet"),
            ("negative_density", "negative_density.parquet"),
            ("new_issue_keywords", "new_issue_keywords.parquet"),
            ("persistent_issue_keywords", "persistent_issue_keywords.parquet"),
            ("quality_summary", "quality_summary.parquet"),
        ]:
            frame = _read_frame(run_dir / filename)
            if frame.empty:
                continue
            for column, value in context.items():
                if column not in frame.columns:
                    frame[column] = value
            buckets[key].append(frame)
    data = {key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame() for key, frames in buckets.items()}
    pd.to_pickle(data, CACHE_FILE)
    CACHE_META.write_text(json.dumps({"signature": latest_signature}, ensure_ascii=False), encoding="utf-8")
    return data


def compute_filtered_bundle(comments_df: pd.DataFrame, videos_df: pd.DataFrame, quality_df: pd.DataFrame, filters: dict[str, Any]) -> dict[str, Any]:
    filtered_comments = comments_df.copy()
    products = filters.get("products") or []
    regions = filters.get("regions") or []
    sentiments = filters.get("sentiments") or []
    cej = filters.get("cej") or []
    keyword_query = str(filters.get("keyword_query") or "").strip()

    if products:
        filtered_comments = filtered_comments[filtered_comments["product"].isin(products)]
    if regions:
        filtered_comments = filtered_comments[filtered_comments["국가"].isin(regions)]
    if sentiments:
        filtered_comments = filtered_comments[filtered_comments["감성"].isin(sentiments)]
    if cej:
        filtered_comments = filtered_comments[filtered_comments["고객여정단계"].isin(cej)]
    if keyword_query:
        filtered_comments = filtered_comments[
            filtered_comments["cleaned_text"].fillna("").str.contains(keyword_query, case=False, na=False)
            | filtered_comments["text_display"].fillna("").str.contains(keyword_query, case=False, na=False)
        ]

    filtered_videos = videos_df.copy()
    if "국가" not in filtered_videos.columns and "region" in filtered_videos.columns:
        filtered_videos["국가"] = filtered_videos["region"].map(localize_region)
    if products:
        filtered_videos = filtered_videos[filtered_videos["product"].isin(products)]
    if regions and "국가" in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos["국가"].isin(regions)]

    removed_count = int(quality_df[quality_df["comment_quality"] != "meaningful"]["count"].sum()) if not quality_df.empty else 0
    return {
        "comments": filtered_comments,
        "videos": filtered_videos,
        "removed_count": removed_count,
    }


@st.cache_data(show_spinner=False)
def build_keyword_summary(texts: tuple[str, ...], top_n: int = 6, version: str = "v2") -> pd.DataFrame:
    counter = build_keyword_counter(texts)
    rows = [{"keyword": keyword, "count": count} for keyword, count in filter_business_keywords(counter, top_n * 4)]
    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return keyword_df
    keyword_df["keyword"] = keyword_df["keyword"].map(normalize_keyword)
    keyword_df = keyword_df[keyword_df["keyword"] != ""]
    keyword_df = keyword_df.groupby("keyword", as_index=False)["count"].sum().sort_values("count", ascending=False).head(top_n)
    return keyword_df


@st.cache_data(show_spinner=False)
def build_wordcloud_image(frequencies: tuple[tuple[str, int], ...], sentiment_code: str) -> bytes | None:
    if not frequencies:
        return None
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    font_path = next((str(path) for path in FONT_CANDIDATES if path.exists()), None)
    wc = WordCloud(
        width=900,
        height=420,
        background_color="white",
        colormap="Greens" if sentiment_code == "positive" else "Oranges",
        font_path=font_path,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(dict(frequencies))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output.getvalue()


def build_weekly_sentiment_window(comments_df: pd.DataFrame, weeks_per_view: int, page: int) -> tuple[pd.DataFrame, int]:
    columns = ["주차시작", "주차표기", "감성", "비율", "댓글수"]
    if comments_df.empty:
        return pd.DataFrame(columns=columns), 1
    dated = comments_df.dropna(subset=["주차시작"]).copy()
    if dated.empty:
        return pd.DataFrame(columns=columns), 1
    all_weeks = sorted(dated["주차시작"].dropna().unique())
    total_pages = max(1, (len(all_weeks) + weeks_per_view - 1) // weeks_per_view)
    safe_page = min(max(page, 1), total_pages)
    end_index = len(all_weeks) - (safe_page - 1) * weeks_per_view
    start_index = max(0, end_index - weeks_per_view)
    selected_weeks = sorted(all_weeks[start_index:end_index])
    if not selected_weeks:
        return pd.DataFrame(columns=columns), total_pages

    window = dated[dated["주차시작"].isin(selected_weeks)].copy()
    totals = window.groupby("주차시작", dropna=False).size().rename("총댓글수")
    grouped = window.groupby(["주차시작", "sentiment_label"], dropna=False).size().reset_index(name="댓글수").merge(totals, on="주차시작", how="left")
    grouped["비율"] = (grouped["댓글수"] / grouped["총댓글수"]).fillna(0).round(4)
    base = pd.MultiIndex.from_product([selected_weeks, ["positive", "negative"]], names=["주차시작", "sentiment_label"]).to_frame(index=False)
    merged = base.merge(grouped[["주차시작", "sentiment_label", "댓글수", "비율"]], on=["주차시작", "sentiment_label"], how="left")
    merged[["댓글수", "비율"]] = merged[["댓글수", "비율"]].fillna(0)
    merged["감성"] = merged["sentiment_label"].map(localize_sentiment)
    merged["주차표기"] = pd.to_datetime(merged["주차시작"]).dt.strftime("%Y-%m-%d")
    return merged[columns], total_pages


def build_comment_showcase(frame: pd.DataFrame, show_translation: bool, limit: int = 120) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.rename(columns={
        "text_display": "원문 댓글",
        "text_original": "원본 댓글",
        "title": "영상 제목",
        "video_url": "영상 링크",
        "like_count": "좋아요 수",
        "language_detected": "언어",
        "removed_reason": "제거 사유",
    }).copy()
    working["한국어 번역"] = working["원문 댓글"].fillna("")
    if show_translation:
        english_mask = working.get("언어", pd.Series("", index=working.index)).fillna("").eq("en")
        if english_mask.any():
            working.loc[english_mask, "한국어 번역"] = working.loc[english_mask, "원문 댓글"].fillna("").map(translate_to_korean)
    working = working.sort_values(["sentiment_score", "좋아요 수"], ascending=[True, False], na_position="last")
    return working.head(limit)



def _latest_signature() -> float:
    if not PROCESSED_DIR.exists():
        return 0.0
    mtimes = [path.stat().st_mtime for path in PROCESSED_DIR.rglob("*.parquet")]
    return max(mtimes) if mtimes else 0.0



def _read_frame(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()



def apply_theme() -> None:
    st.markdown(
        """
        <style>
        .stApp {background: #f6f8f5; color: #123b2f;}
        section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6ece8;}
        .block-container {padding-top: 1.15rem; padding-bottom: 2rem; max-width: 1550px;}
        .voc-card {background: #ffffff; border: 1px solid #e5ece7; border-radius: 18px; padding: clamp(12px, 1.3vw, 18px); box-shadow: 0 10px 30px rgba(16, 64, 48, 0.05); min-height: clamp(88px, 9vw, 112px);}
        .voc-card h4 {margin: 0; font-size: clamp(11px, 0.9vw, 14px); color: #6d8177; font-weight: 600; line-height: 1.35;}
        .voc-card p {margin: 8px 0 0; font-size: clamp(18px, 1.8vw, 30px); font-weight: 700; color: #0d4b3a; line-height: 1.1; word-break: keep-all;}
        .voc-panel {background: #ffffff; border: 1px solid #e5ece7; border-radius: 24px; padding: clamp(16px, 1.4vw, 22px); box-shadow: 0 10px 30px rgba(16, 64, 48, 0.05);}
        .voc-chip {display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e8f5ef; color: #0d7b57; font-size: 12px; margin-right: 8px;}
        @media (max-width: 1200px) { .block-container {max-width: 100%;} }
        @media (max-width: 980px) { .voc-card p {font-size: clamp(16px, 3vw, 22px);} }
        </style>
        """,
        unsafe_allow_html=True,
    )



def localize_region(value: str) -> str:
    return REGION_LABELS.get(str(value), str(value))



def localize_sentiment(value: str) -> str:
    return SENTIMENT_LABELS.get(str(value), str(value))



def render_card(title: str, value: str) -> None:
    st.markdown(f'<div class="voc-card"><h4>{title}</h4><p>{value}</p></div>', unsafe_allow_html=True)



def add_localized_columns(comments_df: pd.DataFrame) -> pd.DataFrame:
    working = comments_df.copy()
    working["국가"] = working.get("region", pd.Series(dtype=str)).map(localize_region)
    working["감성"] = working.get("sentiment_label", pd.Series(dtype=str)).map(localize_sentiment)
    working["고객여정단계"] = working.get("topic_label", "기타").fillna("기타")
    working["브랜드"] = working.get("brand_label", "미언급").fillna("미언급")
    working["작성일시"] = pd.to_datetime(working.get("published_at"), errors="coerce", utc=True).dt.tz_localize(None)
    working["주차시작"] = working["작성일시"].dt.to_period("W-SUN").dt.start_time
    working["주차표기"] = working["주차시작"].dt.strftime("%Y-%m-%d")
    return working



def filter_issue_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "keyword" not in frame.columns:
        return frame
    working = frame.copy()
    working["keyword"] = working["keyword"].map(normalize_keyword)
    return working[working["keyword"] != ""]



def to_excel_bytes(dataframes: dict[str, pd.DataFrame]) -> bytes:
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in dataframes.items():
            frame.copy().to_excel(writer, sheet_name=name[:31], index=False)
    return output.getvalue()



def build_raw_download_package(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame, weekly_df: pd.DataFrame, cej_df: pd.DataFrame, brand_df: pd.DataFrame, density_df: pd.DataFrame) -> bytes:
    return to_excel_bytes({
        "댓글원문": filtered_comments,
        "영상목록": filtered_videos,
        "주간감성": weekly_df,
        "CEJ부정률": cej_df,
        "브랜드언급": brand_df,
        "영상당부정밀도": density_df,
    })



def render_keyword_panel(comments_df: pd.DataFrame, sentiment_code: str, title: str) -> None:
    texts = tuple(comments_df.loc[comments_df["sentiment_label"] == sentiment_code, "cleaned_text"].fillna("").astype(str).tolist())
    keyword_df = build_keyword_summary(texts, top_n=6)
    st.markdown(f"### {title}")
    if keyword_df.empty:
        st.info("표시할 키워드가 없습니다.")
        return
    color_scheme = ["#B8E0B2", "#95D29A", "#72BF7A", "#52AE61", "#34994D", "#1F8741"] if sentiment_code == "positive" else ["#FFC489", "#FFAB59", "#FF9138", "#F26A00", "#E85A00", "#D24D00"]
    chart = alt.Chart(keyword_df).mark_arc(innerRadius=62, outerRadius=108).encode(
        theta=alt.Theta("count:Q"),
        color=alt.Color("keyword:N", legend=alt.Legend(title="키워드", orient="bottom"), scale=alt.Scale(range=color_scheme)),
        tooltip=[alt.Tooltip("keyword:N", title="키워드"), alt.Tooltip("count:Q", title="언급 수")],
    ).properties(height=250)
    st.altair_chart(chart, use_container_width=True)
    image = build_wordcloud_image(tuple(keyword_df.itertuples(index=False, name=None)), sentiment_code)
    if image:
        st.image(image, use_container_width=True)



def render_issue_tables(data: dict[str, pd.DataFrame], selected_products: list[str], selected_regions: list[str]) -> None:
    new_df = filter_issue_frame(data["new_issue_keywords"].copy())
    persistent_df = filter_issue_frame(data["persistent_issue_keywords"].copy())
    for frame in [new_df, persistent_df]:
        if not frame.empty:
            frame["국가"] = frame["region"].map(localize_region)
            if selected_products:
                frame.drop(frame[~frame["product"].isin(selected_products)].index, inplace=True)
            if selected_regions:
                frame.drop(frame[~frame["국가"].isin(selected_regions)].index, inplace=True)
    st.markdown("<span class='voc-chip'>신규 이슈 키워드</span><span class='voc-chip'>지속 이슈 키워드</span>", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        if new_df.empty:
            st.info("신규 이슈 키워드가 없습니다.")
        else:
            st.dataframe(new_df.rename(columns={"keyword": "키워드", "count": "언급 수", "latest_week": "최근 주차"})[["product", "국가", "키워드", "언급 수", "최근 주차"]].head(12), use_container_width=True, hide_index=True)
    with right:
        if persistent_df.empty:
            st.info("지속 이슈 키워드가 없습니다.")
        else:
            st.dataframe(persistent_df.rename(columns={"keyword": "키워드", "count": "언급 수", "weeks_active": "지속 주차 수", "latest_week": "최근 주차"})[["product", "국가", "키워드", "언급 수", "지속 주차 수", "최근 주차"]].head(12), use_container_width=True, hide_index=True)



def main() -> None:
    st.set_page_config(page_title="가전 VoC Dashboard", layout="wide")
    apply_theme()
    data = load_dashboard_data()
    comments_df = add_localized_columns(data["comments"])
    if comments_df.empty:
        st.warning("표시할 분석 결과가 없습니다. 먼저 데이터를 수집해주세요.")
        return

    videos_df = data["videos"].copy()
    videos_df["국가"] = videos_df.get("region", pd.Series(dtype=str)).map(localize_region)
    products = [value for value in PRODUCT_ORDER if value in comments_df["product"].astype(str).unique().tolist()]
    regions = [value for value in ["한국", "미국"] if value in comments_df["국가"].astype(str).unique().tolist()]
    sentiments = [value for value in ["긍정", "부정", "중립"] if value in comments_df["감성"].astype(str).unique().tolist()]
    cej_labels = [value for value in CEJ_ORDER if value in comments_df["고객여정단계"].astype(str).unique().tolist()]

    st.markdown("## 가전 VoC Dashboard")
    st.caption("주간 감성 변화, 키워드 신호, 대표 코멘트를 제품·국가별로 빠르게 확인합니다.")

    with st.sidebar:
        st.markdown("### 탐색 필터")
        selected_products = st.multiselect("제품군", products, default=products)
        selected_regions = st.multiselect("국가", regions, default=regions)
        selected_sentiments = st.multiselect("감성", sentiments, default=sentiments)
        selected_cej = st.multiselect("고객경험여정 단계", cej_labels, default=cej_labels)
        keyword_query = st.text_input("댓글 내 키워드 검색", "")
        weeks_per_view = st.slider("차트 한 화면 주차 수", min_value=4, max_value=24, value=12, step=2)
        show_translation = st.toggle("영문 댓글 한국어 번역 보기", value=True)

    bundle = compute_filtered_bundle(
        comments_df,
        videos_df,
        data["quality_summary"],
        {
            "products": selected_products,
            "regions": selected_regions,
            "sentiments": selected_sentiments,
            "cej": selected_cej,
            "keyword_query": keyword_query,
        },
    )
    filtered_comments = bundle["comments"]
    filtered_videos = bundle["videos"]
    all_filtered_weeks = sorted(filtered_comments["주차시작"].dropna().unique())
    total_pages = max(1, (len(all_filtered_weeks) + weeks_per_view - 1) // weeks_per_view) if all_filtered_weeks else 1
    with st.sidebar:
        page = st.slider("주차 페이지", min_value=1, max_value=total_pages, value=1)

    weekly_window, total_pages = build_weekly_sentiment_window(filtered_comments, weeks_per_view, page)

    cej_df = data["cej_negative_rate"].copy()
    if not cej_df.empty:
        cej_df["국가"] = cej_df["region"].map(localize_region)
        if selected_products:
            cej_df = cej_df[cej_df["product"].isin(selected_products)]
        if selected_regions:
            cej_df = cej_df[cej_df["국가"].isin(selected_regions)]

    brand_df = data["brand_ratio"].copy()
    if not brand_df.empty:
        brand_df["국가"] = brand_df["region"].map(localize_region)
        if selected_products:
            brand_df = brand_df[brand_df["product"].isin(selected_products)]
        if selected_regions:
            brand_df = brand_df[brand_df["국가"].isin(selected_regions)]

    density_df = data["negative_density"].copy()
    if not density_df.empty:
        density_df["국가"] = density_df["region"].map(localize_region)
        if selected_products:
            density_df = density_df[density_df["product"].isin(selected_products)]
        if selected_regions:
            density_df = density_df[density_df["국가"].isin(selected_regions)]

    positive_ratio = 0.0 if filtered_comments.empty else (filtered_comments["sentiment_label"].eq("positive").mean() * 100)
    negative_ratio = 0.0 if filtered_comments.empty else (filtered_comments["sentiment_label"].eq("negative").mean() * 100)
    top_brand = filtered_comments[filtered_comments["브랜드"] != "미언급"]["브랜드"].mode().iloc[0] if not filtered_comments.empty and not filtered_comments[filtered_comments["브랜드"] != "미언급"].empty else "-"

    raw_comments_export = filtered_comments.rename(columns={
        "text_display": "원문 댓글",
        "text_original": "원본 댓글",
        "title": "영상 제목",
        "video_url": "영상 링크",
        "like_count": "좋아요 수",
        "language_detected": "언어",
        "removed_reason": "제거 사유",
    })

    _, top_action = st.columns([0.72, 0.28])
    with top_action:
        st.download_button(
            "전체 Raw Data 엑셀 다운로드",
            data=build_raw_download_package(raw_comments_export, filtered_videos, weekly_window, cej_df, brand_df, density_df),
            file_name="voc_dashboard_raw_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: render_card("분석 대상 댓글", f"{len(filtered_comments):,}")
    with c2: render_card("관련 영상", f"{filtered_comments['video_id'].nunique():,}")
    with c3: render_card("긍정 응답률", f"{positive_ratio:.1f}%")
    with c4: render_card("부정 응답률", f"{negative_ratio:.1f}%")
    with c5: render_card("제거 댓글 수", f"{bundle['removed_count']:,}")
    with c6: render_card("최다 언급 브랜드", top_brand)

    st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
    st.markdown("### 주차별 긍정/부정 응답률")
    if not weekly_window.empty:
        order = weekly_window["주차표기"].drop_duplicates().tolist()
        chart = alt.Chart(weekly_window).mark_bar(size=14, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("주차표기:N", title="주차 시작일", sort=order, axis=alt.Axis(labelAngle=-45, labelLimit=120)),
            y=alt.Y("비율:Q", title="응답률", axis=alt.Axis(format="%")),
            xOffset=alt.XOffset("감성:N", sort=["긍정", "부정"]),
            color=alt.Color("감성:N", scale=alt.Scale(domain=["긍정", "부정"], range=["#4D79F6", "#FFAA3B"]), legend=alt.Legend(title=None, orient="top-right")),
            tooltip=[alt.Tooltip("주차표기:N", title="주차 시작일"), alt.Tooltip("감성:N", title="감성"), alt.Tooltip("댓글수:Q", title="댓글 수"), alt.Tooltip("비율:Q", title="응답률", format=".1%")],
        ).properties(height=340, width=max(720, len(order) * 86))
        st.altair_chart(chart, use_container_width=False)
        st.caption(f"시간순으로 정렬된 최근 {len(order)}개 주차를 표시합니다. 전체 {total_pages}페이지 중 {min(page, total_pages)}페이지입니다.")
    else:
        st.info("해당 조건에서 주간 차트를 그릴 데이터가 없습니다.")
    st.markdown("</div>", unsafe_allow_html=True)

    left, mid, right = st.columns([1.05, 1.05, 1.2])
    with left:
        st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
        render_keyword_panel(filtered_comments, "negative", "부정 키워드 비중")
        st.markdown("</div>", unsafe_allow_html=True)
    with mid:
        st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
        render_keyword_panel(filtered_comments, "positive", "긍정 키워드 비중")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
        st.markdown("### CEJ 단계별 부정률")
        if not cej_df.empty:
            cej_display = cej_df.rename(columns={"topic_label": "고객여정단계", "comments": "댓글 수", "negative_comments": "부정 댓글 수", "negative_rate": "부정률"}).copy()
            cej_display["고객여정단계"] = pd.Categorical(cej_display["고객여정단계"], categories=CEJ_ORDER, ordered=True)
            cej_display = cej_display.sort_values(["product", "국가", "고객여정단계"])
            st.dataframe(cej_display[["product", "국가", "고객여정단계", "댓글 수", "부정 댓글 수", "부정률"]], use_container_width=True, hide_index=True)
        else:
            st.info("표시할 CEJ 데이터가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    bottom_left, bottom_right = st.columns([1.15, 1.0])
    with bottom_left:
        st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
        st.markdown("### 개선 포인트 신호")
        render_issue_tables(data, selected_products, selected_regions)
        st.markdown("</div>", unsafe_allow_html=True)
    with bottom_right:
        st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
        st.markdown("### 브랜드 언급 비율")
        if not brand_df.empty:
            st.dataframe(brand_df.rename(columns={"brand_label": "브랜드", "count": "언급 댓글 수", "ratio": "언급 비율"})[["product", "국가", "브랜드", "언급 댓글 수", "언급 비율"]], use_container_width=True, hide_index=True)
        else:
            st.info("표시할 브랜드 데이터가 없습니다.")
        st.markdown("### 영상당 부정 댓글 밀도")
        if not density_df.empty:
            st.dataframe(density_df.rename(columns={"negative_comments": "부정 댓글 수", "total_comments": "분석 댓글 수", "negative_density": "부정 밀도", "title": "영상 제목", "video_url": "영상 링크"})[["product", "국가", "영상 제목", "부정 댓글 수", "분석 댓글 수", "부정 밀도", "영상 링크"]].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("표시할 영상 밀도 데이터가 없습니다.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='voc-panel'>", unsafe_allow_html=True)
    st.markdown("### 대표 VoC 코멘트")
    comment_showcase = build_comment_showcase(filtered_comments, show_translation, limit=150)
    if comment_showcase.empty:
        st.info("현재 필터 조건에서 보여줄 대표 VoC 코멘트가 없습니다.")
    else:
        st.dataframe(
            comment_showcase[[column for column in ["product", "국가", "감성", "고객여정단계", "브랜드", "언어", "원문 댓글", "한국어 번역", "영상 제목", "좋아요 수"] if column in comment_showcase.columns]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "원문 댓글": st.column_config.TextColumn("원문 댓글", width="large", help="셀 위에 마우스를 올리면 긴 문장을 더 쉽게 확인할 수 있습니다."),
                "한국어 번역": st.column_config.TextColumn("한국어 번역", width="large"),
                "영상 제목": st.column_config.TextColumn("관련 영상", width="medium"),
            },
        )
        selected_index = st.selectbox("상세 확인할 대표 코멘트", options=comment_showcase.index.tolist(), format_func=lambda idx: f"{comment_showcase.loc[idx, '감성']} | {comment_showcase.loc[idx, '브랜드']} | {str(comment_showcase.loc[idx, '원문 댓글'])[:50]}")
        selected = comment_showcase.loc[selected_index]
        with st.container(border=True):
            st.markdown(f"**제품군**: {selected.get('product', '-')}")
            st.markdown(f"**국가/감성/단계**: {selected.get('국가', '-')} / {selected.get('감성', '-')} / {selected.get('고객여정단계', '-')}")
            st.markdown(f"**관련 영상**: [{selected.get('영상 제목', '-')}]({selected.get('영상 링크', '#')})")
            st.markdown("**원문 댓글**")
            st.write(selected.get("원문 댓글", ""))
            st.markdown("**한국어 번역**")
            st.write(selected.get("한국어 번역", ""))
    st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
