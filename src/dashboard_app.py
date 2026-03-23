"""Streamlit dashboard for appliance VoC exploration."""

from __future__ import annotations

from io import BytesIO
import re
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "data" / "mplcache").resolve()))

import koreanize_matplotlib

import altair as alt
import pandas as pd
import streamlit as st

from src.analytics.comment_analysis import analyze_comment_with_context
from src.analytics.keywords import build_keyword_counter, filter_business_keywords, normalize_keyword, tokenize_text
from src.utils.translation import translate_to_korean

BASE_DIR = Path(__file__).resolve().parents[1]
#회사용
import matplotlib.font_manager as fm  # 상단 import에 없다면 추가

def _get_font_file() -> Path | None:
    """Cloud/Local 모두에서 한글 폰트 파일을 찾아 반환"""
    candidates = [
        BASE_DIR / "fonts" / "NanumGothic.ttf",   # ✅ Cloud(레포에 넣은 폰트)
        Path("C:/Windows/Fonts/malgun.ttf"),      # 로컬(윈도우) 보조
        Path("C:/Windows/Fonts/NanumGothic.ttf"), # 로컬(윈도우) 보조
    ]
    return next((p for p in candidates if p.exists()), None)

def _get_font_prop():
    """Matplotlib용 FontProperties"""
    font_file = _get_font_file()
    return fm.FontProperties(fname=str(font_file)) if font_file else None

#

PROCESSED_DIR = BASE_DIR / "data" / "processed"
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "dashboard_bundle.pkl"
CACHE_META = CACHE_DIR / "dashboard_bundle_meta.json"
REGION_LABELS = {"KR": "한국", "US": "미국"}
SENTIMENT_LABELS = {"positive": "\uae0d\uc815", "negative": "\ubd80\uc815", "neutral": "\uc911\ub9bd", "excluded": "\uc81c\uc678"}
CEJ_ORDER = ["인지", "구매", "배송", "설치", "사용준비", "사용", "관리교체", "기타"]
PRODUCT_ORDER = ["세탁기", "냉장고", "건조기", "식기세척기"]


#집노트북용 FONT_CANDIDATES = [Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/NanumGothic.ttf")]
#회사용
FONT_CANDIDATES = [
    BASE_DIR / "fonts" / "NanumGothic.ttf",          # ✅ Streamlit Cloud용(레포 안 폰트)
    Path("C:/Windows/Fonts/malgun.ttf"),             # 로컬(윈도우)용
    Path("C:/Windows/Fonts/NanumGothic.ttf"),        # 로컬(윈도우)용
]


COL_COUNTRY = "국가"
COL_SENTIMENT = "감성"
COL_CEJ = "고객여정단계"
COL_BRAND = "브랜드"
COL_WRITTEN_AT = "작성일시"
COL_WEEK_START = "주차시작"
COL_WEEK_LABEL = "주차표기"
COL_ORIGINAL = "원문 댓글"
COL_TRANSLATION = "한국어 번역"
COL_VIDEO_TITLE = "영상 제목"
COL_VIDEO_LINK = "영상 링크"
COL_LIKES = "좋아요 수"
COL_LANGUAGE = "언어"
COL_RAW = "원본 댓글"
COL_REMOVED = "제거 사유"
COL_CONTEXT = "\uc5f0\uad00 \ub9e5\ub77d"
COL_CONTEXT_TRANSLATION = "\uc5f0\uad00 \ub9e5\ub77d \ubc88\uc5ed"
COL_SENTIMENT_CODE = "sentiment_code"


@st.cache_data(show_spinner=False)
#def load_dashboard_data() -> dict[str, pd.DataFrame]:
#    latest_signature = _latest_signature()
#    if CACHE_FILE.exists() and CACHE_META.exists():
#        meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
#        if meta.get("signature") == latest_signature:
#            return pd.read_pickle(CACHE_FILE)
#회사에서 공유용으로 수정
#삭제 @st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    @st.cache_resource
    def get_dashboard_data_resource():
        return load_dashboard_data()
        
    
    # ✅ 캐시가 있으면 일단 즉시 사용 (초기 로딩/health-check 안정화)
    if CACHE_FILE.exists():
        try:
            return pd.read_pickle(CACHE_FILE)
        except Exception:
            pass  # 캐시가 깨졌으면 아래에서 재생성

    # ✅ 캐시가 없을 때만 무거운 시그니처 계산
    latest_signature = _latest_signature()

    if CACHE_FILE.exists() and CACHE_META.exists():
        try:
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("signature") == latest_signature:
                return pd.read_pickle(CACHE_FILE)
        except Exception:
            pass


    #


    if CACHE_FILE.exists() and (not PROCESSED_DIR.exists() or not any(PROCESSED_DIR.iterdir())):
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
        "representative_comments": [],
        "analysis_comments": [],
    }
    run_dirs = sorted(
        (path for path in PROCESSED_DIR.iterdir() if path.is_dir()) if PROCESSED_DIR.exists() else [],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    #회사 공유용으로 추가
    run_dirs = run_dirs[:3]   # ✅ 최근 5개만 읽기 (원하면 3~10으로 조절)

    #
    
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
            ("representative_comments", "representative_comments.parquet"),
            ("analysis_comments", "analysis_comments.parquet"),
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


@st.cache_data(show_spinner=False)
def build_keyword_summary(texts: tuple[str, ...], top_n: int = 6, version: str = "v5") -> pd.DataFrame:
    counter = build_keyword_counter(texts)
    rows = [{"keyword": keyword, "count": count} for keyword, count in filter_business_keywords(counter, top_n * 6)]
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

#
    
# 집 노트북용
#    font_path = next((str(path) for path in FONT_CANDIDATES if path.exists()), None)
#    wc = WordCloud(
#        width=1000,
#        height=420,
#        background_color="white",
#        colormap="Greens" if sentiment_code == "positive" else "Oranges",
#        font_path=font_path,
#        prefer_horizontal=0.9,
#        max_words=60,
#        collocations=False,
#    ).generate_from_frequencies(dict(frequencies))


    # 회사용
    font_file = next((path for path in FONT_CANDIDATES if path.exists()), None)
    if font_file is None:
        raise RuntimeError("WordCloud용 한글 폰트를 찾지 못했습니다. fonts/NanumGothic.ttf를 레포에 추가하세요.")
    wc = WordCloud(
        width=1000,
        height=420,
        background_color="white",
        colormap="Greens" if sentiment_code == "positive" else "Oranges",
        font_path=str(font_file),   # ✅ 무조건 실제 파일 경로가 들어가게 됨
        prefer_horizontal=0.9,
        max_words=60,
        collocations=False,
    ).generate_from_frequencies(dict(frequencies))

#
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def build_donut_chart_image(rows: tuple[tuple[str, int, str], ...], sentiment_code: str, center_label: str) -> bytes | None:
    if not rows:
        return None
    import matplotlib.pyplot as plt

    labels = [row[0] for row in rows]
    counts = [row[1] for row in rows]
    shares = [row[2] for row in rows]
    colors = [
        "#B8E0B2", "#95D29A", "#72BF7A", "#52AE61", "#34994D", "#1F8741"
    ] if sentiment_code == "positive" else [
        "#FFC489", "#FFAB59", "#FF9138", "#F26A00", "#E85A00", "#D24D00"
    ]
    colors = colors[:len(labels)]
    font_path = next((str(candidate) for candidate in FONT_CANDIDATES if candidate.exists()), None)

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    display_labels = [f"{label}\n{share}" for label, share in zip(labels, shares)]
    wedges, texts = ax.pie(
        counts,
        labels=display_labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.42, "edgecolor": "white", "linewidth": 1.2},
        labeldistance=0.78,
        textprops={"fontsize": 10, "color": "#24453b", "ha": "center", "va": "center",
                   #집노트북용 "fontfamily": "Malgun Gothic" if font_path else None,
                   #회사용
                   "fontproperties": _get_font_prop(),
                   },
    )

    centre_circle = plt.Circle((0, 0), 0.38, fc="white")
    ax.add_artist(centre_circle)
    ax.text(0, 0.02, center_label, ha="center", va="center", fontsize=16, fontweight="bold", color="#24453b",
            #집노트북용 fontfamily="Malgun Gothic" if font_path else None
            #회사용
            fontproperties=_get_font_prop()
            ,
           )
    ax.set_aspect("equal")
    plt.margins(0.06)

    output = BytesIO()
    fig.savefig(output, format="png", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return output.getvalue()


@st.cache_data(show_spinner=False)
def build_comment_showcase(frame: pd.DataFrame, sentiment: str, limit: int = 30) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    if "comment_validity" in working.columns:
        valid_comments = working[working["comment_validity"] == "valid"].copy()
        if not valid_comments.empty:
            working = valid_comments
    elif "analysis_included" in working.columns:
        included_comments = working[working["analysis_included"] == True].copy()
        if not included_comments.empty:
            working = included_comments
    if working.empty:
        return pd.DataFrame()

    # Preselect a manageable candidate pool before expensive re-analysis so the
    # representative section stays responsive even with tens of thousands of comments.
    candidate_pool = working.copy()
    if sentiment and "sentiment_label" in candidate_pool.columns:
        sentiment_candidates = candidate_pool[candidate_pool["sentiment_label"] == sentiment].copy()
        if not sentiment_candidates.empty:
            candidate_pool = sentiment_candidates
    likes_seed = pd.to_numeric(candidate_pool.get("like_count", pd.Series(0, index=candidate_pool.index)), errors="coerce").fillna(0)
    length_seed = pd.to_numeric(candidate_pool.get("text_length", pd.Series(0, index=candidate_pool.index)), errors="coerce").fillna(0)
    score_seed = pd.to_numeric(candidate_pool.get("sentiment_score", pd.Series(0.0, index=candidate_pool.index)), errors="coerce").fillna(0.0).abs()
    candidate_pool["candidate_seed"] = (likes_seed * 1.8) + (length_seed * 0.05) + (score_seed * 80)
    working = candidate_pool.sort_values(["candidate_seed", "like_count"], ascending=[False, False]).head(max(limit * 20, 120)).copy()

    source_original = working.get("text_original", pd.Series("", index=working.index)).fillna("")
    source_display = working.get("text_display", source_original).fillna(source_original)
    source_language = working.get("language_detected", pd.Series("", index=working.index)).fillna("")
    source_comment_id = working.get("comment_id", pd.Series("", index=working.index)).fillna("")
    original_lookup = pd.Series(source_display.values, index=source_comment_id.astype(str)).to_dict()
    language_lookup = pd.Series(source_language.values, index=source_comment_id.astype(str)).to_dict()

    working = working.rename(columns={
        "text_display": COL_ORIGINAL,
        "text_original": COL_RAW,
        "title": COL_VIDEO_TITLE,
        "video_url": COL_VIDEO_LINK,
        "like_count": COL_LIKES,
        "language_detected": COL_LANGUAGE,
        "removed_reason": COL_REMOVED,
    }).copy()
    working[COL_ORIGINAL] = working.get(COL_ORIGINAL, pd.Series("", index=working.index)).fillna("")
    parent_ids = working.get("parent_comment_id", pd.Series(pd.NA, index=working.index)).fillna("").astype(str)
    working[COL_CONTEXT] = parent_ids.map(lambda value: original_lookup.get(value, "") if value else "")
    working[COL_CONTEXT_TRANSLATION] = working[COL_CONTEXT]

    working[COL_TRANSLATION] = working.get(COL_ORIGINAL, pd.Series("", index=working.index)).fillna("")
    language_series = working.get(COL_LANGUAGE, pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    needs_translation = (~language_series.str.startswith("ko")) & working[COL_ORIGINAL].str.strip().ne("")
    if needs_translation.any():
        working.loc[needs_translation, COL_TRANSLATION] = working.loc[needs_translation, COL_ORIGINAL].map(translate_to_korean)

    context_language = parent_ids.map(lambda value: language_lookup.get(value, "") if value else "").fillna("").astype(str).str.lower()
    needs_context_translation = working[COL_CONTEXT].str.strip().ne("") & (~context_language.str.startswith("ko"))
    if needs_context_translation.any():
        working.loc[needs_context_translation, COL_CONTEXT_TRANSLATION] = working.loc[needs_context_translation, COL_CONTEXT].map(translate_to_korean)

    def _reanalyze(row: pd.Series) -> pd.Series:
        result = analyze_comment_with_context(
            _safe_text(row.get(COL_ORIGINAL, "")),
            validity=_safe_text(row.get("comment_validity", "valid")) or "valid",
            exclusion_reason=_safe_text(row.get("exclusion_reason", "")) or None,
            parent_comment=_safe_text(row.get(COL_CONTEXT, "")) or None,
            context_comments=[_safe_text(row.get(COL_CONTEXT, ""))] if _safe_text(row.get(COL_CONTEXT, "")) else None,
            is_reply=bool(_safe_text(row.get("parent_comment_id", ""))),
        )
        return pd.Series({
            "display_sentiment": result.sentiment_label or "neutral",
            "display_signal_strength": result.signal_strength,
            "display_classification_type": result.classification_type,
            "display_confidence_level": result.confidence_level,
            "display_context_used": result.context_used,
            "display_context_required": result.context_required_for_display,
            "display_product_target": result.product_target or "",
            "display_reason": result.reason,
            "display_needs_review": result.needs_review,
        })

    analysis_df = working.apply(_reanalyze, axis=1)
    working = pd.concat([working.reset_index(drop=True), analysis_df.reset_index(drop=True)], axis=1)

    if sentiment:
        working = working[working["display_sentiment"] == sentiment].copy()
    if working.empty:
        return pd.DataFrame()

    like_score = pd.to_numeric(working.get(COL_LIKES, pd.Series(0, index=working.index)), errors="coerce").fillna(0)
    text_score = pd.to_numeric(working.get("text_length", pd.Series(0, index=working.index)), errors="coerce").fillna(0)
    sentiment_score = pd.to_numeric(working.get("sentiment_score", pd.Series(0.0, index=working.index)), errors="coerce").fillna(0.0)
    rep_score = pd.to_numeric(working.get("representative_score", pd.Series(0, index=working.index)), errors="coerce").fillna(0)
    if sentiment == "negative":
        working["showcase_score"] = (-sentiment_score * 100) + (like_score * 1.5) + (text_score * 0.04) + (rep_score * 2)
    elif sentiment == "positive":
        working["showcase_score"] = (sentiment_score * 100) + (like_score * 1.5) + (text_score * 0.04) + (rep_score * 2)
    else:
        working["showcase_score"] = sentiment_score.abs() * 100 + (like_score * 1.5) + (text_score * 0.04) + (rep_score * 2)
    if "comment_id" in working.columns:
        working = working.drop_duplicates(subset=["comment_id"], keep="first")
    else:
        working = working.drop_duplicates(subset=[COL_ORIGINAL], keep="first")
    working = working.sort_values(["showcase_score", COL_LIKES], ascending=[False, False], na_position="last").head(limit).copy()
    return working


#def _latest_signature() -> float:
#    if not PROCESSED_DIR.exists():
#        return 0.0
#    mtimes = [path.stat().st_mtime for path in PROCESSED_DIR.rglob("*.parquet")]
#    return max(mtimes) if mtimes else 0.0
#회사에서 공유용으로 수정
def _latest_signature() -> float:
    if not PROCESSED_DIR.exists():
        return 0.0

    # ✅ 너무 많은 파일을 다 훑지 않도록 상한을 둬서 startup을 안정화
    mt = 0.0
    count = 0
    try:
        for path in PROCESSED_DIR.rglob("*.parquet"):
            mt = max(mt, path.stat().st_mtime)
            count += 1
            if count >= 200:   # ✅ 상한(원하면 200~500 사이로)
                break
    except Exception:
        pass
    return mt

#


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
        header[data-testid="stHeader"] {background: rgba(246,248,245,0.92); backdrop-filter: blur(6px);}
        section[data-testid="stSidebar"] {background: #ffffff; border-right: 1px solid #e6ece8;}
        .block-container {padding-top: 5.2rem; padding-bottom: 2rem; max-width: 1240px;}
        .voc-card {background: #ffffff; border: 1px solid #e5ece7; border-radius: 18px; padding: clamp(12px, 1.3vw, 18px); box-shadow: 0 10px 30px rgba(16, 64, 48, 0.05); min-height: clamp(88px, 9vw, 112px);}
        .voc-card h4 {margin: 0; font-size: clamp(11px, 0.9vw, 14px); color: #6d8177; font-weight: 600; line-height: 1.35;}
        .voc-card p {margin: 8px 0 0; font-size: clamp(18px, 1.8vw, 30px); font-weight: 700; color: #0d4b3a; line-height: 1.1; word-break: keep-all;}
        .voc-panel {background: #ffffff; border: 1px solid #e5ece7; border-radius: 24px; padding: clamp(16px, 1.4vw, 22px); box-shadow: 0 10px 30px rgba(16, 64, 48, 0.05); overflow: hidden;}
        .voc-chip {display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e8f5ef; color: #0d7b57; font-size: 12px; margin-right: 8px;}
        .voc-legend {display: flex; flex-wrap: wrap; gap: 8px 10px; margin-top: 8px; margin-bottom: 6px;}
        .voc-legend-compact {display: flex; flex-direction: column; gap: 10px; padding: 10px 0 4px;}
        .voc-donut-wrap {display: flex; align-items: center; gap: 20px;}
        .voc-donut-chart {flex: 0 0 min(64%, 520px); min-width: 260px;}
        .voc-donut-side {flex: 1 1 260px; min-width: 220px;}
        .voc-legend-item {display: inline-flex; align-items: center; gap: 8px; padding: 6px 10px; border: 1px solid #e5ece7; border-radius: 999px; background: #fbfcfb; font-size: 13px; color: #48665b; line-height: 1.2;}
        .voc-legend-swatch {width: 10px; height: 10px; border-radius: 999px; display: inline-block; flex: 0 0 auto;}
        .voc-comment-card {background: #f9fbfa; border: 1px solid #e5ece7; border-radius: 16px; padding: 14px 16px; margin-bottom: 12px;}
        .voc-comment-meta {font-size: 12px; color: #6d8177; margin-bottom: 8px;}
        .voc-comment-text {font-size: 14px; line-height: 1.6; color: #123b2f; white-space: pre-wrap;}
        .voc-reason {margin: 12px 0 18px; padding: 12px 14px; border-radius: 14px; background: #eef7f3; border-left: 5px solid #0d7b57; font-size: 15px; font-weight: 700; color: #0f4a39; line-height: 1.45;}
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



def summarize_comment_reason(selected: pd.Series, sentiment_name: str) -> str:
    topic = str(selected.get(COL_CEJ, "\uae30\ud0c0") or "\uae30\ud0c0")
    text = " ".join([
        str(selected.get(COL_TRANSLATION, "") or ""),
        str(selected.get(COL_ORIGINAL, "") or ""),
    ])
    lowered = text.lower()

    if any(token in lowered for token in ["smart", "ai", "digital button", "digital control", "스마트", "ai 기능", "디지털 버튼"]):
        topic = "\uc0ac\uc6a9"

    signals = []
    signal_map = [
        (["스마트", "smart", "ai", "digital"], "\uc4f8\ub370\uc5c6\ub294 \uae30\ub2a5"),
        (["소음", "noise"], "\uc18c\uc74c"),
        (["고장", "repair", "replace", "수리"], "\uace0\uc7a5\u00b7\uc218\ub9ac"),
        (["성능", "wash", "clean", "세탁"], "\uc138\ud0c1 \uc131\ub2a5"),
        (["비싸", "price", "deal"], "\uac00\uaca9 \ubd80\ub2f4"),
        (["delivery", "shipping", "배송"], "\ubc30\uc1a1"),
        (["customer support", "service", "as", "고객지원", "서비스"], "\uace0\uac1d\uc9c0\uc6d0"),
    ]
    for aliases, label in signal_map:
        if any(alias in lowered for alias in aliases):
            signals.append(label)
    if not signals:
        for token in tokenize_text(text):
            if token not in signals:
                signals.append(token)
            if len(signals) >= 2:
                break
    signal_text = ", ".join(signals[:2]) if signals else "\ud575\uc2ec \uc0ac\uc6a9 \uacbd\ud5d8"

    if sentiment_name == "negative":
        return f"\ubd80\uc815 \ud3ec\uc778\ud2b8: {topic} \ub2e8\uacc4\uc5d0\uc11c {signal_text}\uc5d0 \ub300\ud55c \ubd88\ub9cc\uc774 \ud575\uc2ec\uc785\ub2c8\ub2e4."
    if sentiment_name == "positive":
        return f"\uae0d\uc815 \ud3ec\uc778\ud2b8: {topic} \ub2e8\uacc4\uc5d0\uc11c {signal_text}\uc5d0 \ub300\ud55c \ub9cc\uc871\uac10\uc774 \ud575\uc2ec\uc785\ub2c8\ub2e4."
    return f"\uc911\ub9bd \ud3ec\uc778\ud2b8: {topic} \ub2e8\uacc4\uc5d0\uc11c {signal_text} \uad00\ub828 \uc758\uacac\uc774 \ud568\uaed8 \uc5b8\uae09\ub429\ub2c8\ub2e4."




def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    cleaned = str(value).strip()
    if cleaned.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return cleaned


def _looks_garbled(text_value: str) -> bool:
    text_value = _safe_text(text_value)
    if not text_value:
        return False
    lowered = text_value.lower()
    if lowered in {"nan", "none", "null"}:
        return True
    if "???" in text_value or "??" in text_value:
        return True
    if text_value.count("?") >= 3 and not any(ch.isalpha() for ch in text_value):
        return True
    return False


def _is_long_text(text_value: str, threshold: int = 260) -> bool:
    return len(_safe_text(text_value)) > threshold

def _looks_korean_text(text_value: str) -> bool:
    value = _safe_text(text_value)
    if not value:
        return False
    hangul_count = sum(1 for ch in value if 0xAC00 <= ord(ch) <= 0xD7A3)
    ascii_alpha = sum(1 for ch in value if ch.isascii() and ch.isalpha())
    return hangul_count > max(3, ascii_alpha)


def _resolve_card_translation(original_text: str, current_translation: str, is_korean: bool) -> str:
    if is_korean:
        return ""

    translation = _safe_text(current_translation)
    if translation and translation != original_text and _looks_korean_text(translation) and not _looks_garbled(translation):
        return translation

    translated_fallback = _safe_text(translate_to_korean(original_text))
    if translated_fallback and translated_fallback != original_text and _looks_korean_text(translated_fallback) and not _looks_garbled(translated_fallback):
        return translated_fallback

    return "한국어 번역을 준비 중입니다."


def _find_evidence_span(original_text: str, phrases: list[str]) -> str:
    source = _safe_text(original_text)
    lowered = source.lower()
    for phrase in phrases:
        idx = lowered.find(phrase.lower())
        if idx != -1:
            return source[idx: idx + len(phrase)]
    return ""


def _extract_voc_items(selected: pd.Series, sentiment_name: str) -> list[dict[str, str]]:
    original = _safe_text(selected.get(COL_ORIGINAL, ""))
    lowered = original.lower()
    sentiment_label = "부정" if sentiment_name == "negative" else "긍정" if sentiment_name == "positive" else "중립"

    candidates = [
        {
            "topic": "청소 편의",
            "phrases": ["easier to clean", "easy to clean"],
            "insight": "예전 제품이 더 청소하기 쉬웠다는 비교가 드러나며 관리 편의에 대한 선호가 읽힙니다.",
        },
        {
            "topic": "내구성",
            "phrases": ["lasted forever", "30+ years old", "never had any issues", "never had a problem"],
            "insight": "오랜 사용 기간과 문제 없음이 언급되어 내구성에 대한 강한 인식이 드러납니다.",
        },
        {
            "topic": "설계 단순성",
            "phrases": ["newer isn't always better", "newer isn't always better", "overdesign", "more skus", "unreliable products", "smart crap", "digital buttons", "cpu", "reorder milk", "starve"],
            "insight": "과한 기능보다 단순하고 본질적인 설계를 선호하거나, 과설계에 대한 문제의식이 나타납니다.",
        },
        {
            "topic": "핵심 성능",
            "phrases": ["cleaning performance", "washes clothes", "cooling", "temperature", "not cooling", "warmer than", "melted"],
            "insight": "제품의 기본 성능이나 온도 유지, 세척 성능에 대한 평가가 포함됩니다.",
        },
        {
            "topic": "수리/서비스",
            "phrases": ["service", "warranty", "tech", "compressor", "board", "repair", "called for service"],
            "insight": "수리 대응, 보증, 서비스 경험에 대한 고객 불만 또는 평가가 포함됩니다.",
        },
        {
            "topic": "가격/비용",
            "phrases": ["price increased", "cost", "expensive", "$"],
            "insight": "가격 부담이나 비용 대비 가치 인식이 드러납니다.",
        },
    ]

    items: list[dict[str, str]] = []
    for candidate in candidates:
        evidence = _find_evidence_span(original, candidate["phrases"])
        if evidence:
            items.append({
                "topic": candidate["topic"],
                "sentiment": sentiment_label,
                "insight": candidate["insight"],
                "evidence_span": evidence,
            })

    if not items:
        items.append({
            "topic": "기타 사용 경험",
            "sentiment": sentiment_label,
            "insight": "제품 관련 경험은 언급되지만, 추가 해석이 필요한 약한 신호입니다.",
            "evidence_span": _find_evidence_span(original, [original[:30]]) if original else "",
        })

    deduped = []
    seen = set()
    for item in items:
        if item["topic"] in seen:
            continue
        seen.add(item["topic"])
        deduped.append(item)
    return deduped[:4]


def _summarize_for_dashboard(selected: pd.Series, sentiment_name: str) -> str:
    original = _safe_text(selected.get(COL_ORIGINAL, ""))
    translation = _safe_text(selected.get(COL_TRANSLATION, ""))
    if not translation or translation == original or _looks_garbled(translation) or not _looks_korean_text(translation):
        translation = _resolve_card_translation(original, translation, False)

    lowered_original = original.lower()
    if "old refrigerators were easier to clean" in lowered_original or "sometimes newer isn't always better" in lowered_original:
        return "예전 냉장고가 더 단순하고 오래 갔다는 경험을 바탕으로, 최신 제품이 꼭 더 낫지는 않다는 의견입니다."
    if any(token in lowered_original for token in ["sku", "unreliable", "overdesign", "overdesigned"]):
        return "회사들이 제품을 지나치게 복잡하게 설계하면서 신뢰성이 떨어진다는 불만과, 단순하고 튼튼한 설계를 선호한다는 인식이 함께 드러납니다."
    if any(token in lowered_original for token in ["cpu", "reorder milk", "starve"]):
        return "스마트 기능을 풍자적으로 언급하며, 냉장고에 과한 자동화 기능을 기대하거나 그런 기능 중심 설계를 비판하는 의견입니다."
    if any(token in lowered_original for token in ["compressor", "warranty", "service", "called for service", "not cooling", "warmer than"]):
        return "냉각 성능 저하와 반복되는 수리·보증 경험을 설명하며, 제품 신뢰성과 서비스 대응에 대한 불만을 담고 있습니다."

    if not translation or translation == "한국어 번역을 준비 중입니다.":
        return "요약 가능한 댓글 내용이 없습니다."

    sentences = [part.strip() for part in re.split(r'(?<=[.!?])\s+', translation) if part.strip()]
    if not sentences:
        compact = translation[:110].rstrip()
        return compact + ("..." if len(translation) > 110 else "")
    summary = " ".join(sentences[:2]).strip()
    if len(summary) > 120:
        summary = summary[:117].rstrip() + "..."
    return summary


def _decision_badge(decision: str) -> tuple[str, str]:
    mapping = {
        "strong_candidate": ("대표 적합", "의사결정자에게 바로 보여줘도 이해가 쉬운 댓글입니다."),
        "review_needed": ("검토 필요", "의미는 있지만 맥락이나 해석 보강이 필요한 댓글입니다."),
        "not_suitable": ("대표 부적합", "대표 코멘트로 쓰기에는 맥락 의존성이나 설명력이 부족합니다."),
    }
    return mapping.get(decision, ("검토 필요", "대표성 판단 정보가 부족해 추가 검토가 필요합니다."))


def render_card(title: str, value: str) -> None:
    st.markdown(f'<div class="voc-card"><h4>{title}</h4><p>{value}</p></div>', unsafe_allow_html=True)



def format_selection(values: list[str], empty_label: str = "전체") -> str:
    if not values:
        return empty_label
    if len(values) <= 3:
        return " · ".join(values)
    return f"{' · '.join(values[:3])} 외 {len(values) - 3}개"


def pick_top_signal(frame: pd.DataFrame, label_col: str, value_col: str, fallback: str = "-") -> str:
    if frame.empty or label_col not in frame.columns or value_col not in frame.columns:
        return fallback
    ordered = frame.sort_values(value_col, ascending=False)
    if ordered.empty:
        return fallback
    top = ordered.iloc[0]
    label = str(top.get(label_col, fallback))
    value = top.get(value_col, 0)
    try:
        return f"{label} ({int(value):,})"
    except Exception:
        return label



def pick_top_cej_signal(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "-"
    ordered = frame.sort_values(["negative_rate", "negative_comments"], ascending=[False, False])
    if ordered.empty:
        return "-"
    top = ordered.iloc[0]
    topic = str(top.get("topic_label", "기타"))
    rate = float(top.get("negative_rate", 0.0)) * 100
    return f"{topic} ({rate:.1f}%)"


def add_localized_columns(comments_df: pd.DataFrame) -> pd.DataFrame:
    working = comments_df.copy()
    region_series = working["region"] if "region" in working.columns else pd.Series("", index=working.index, dtype="object")
    sentiment_series = working["sentiment_label"] if "sentiment_label" in working.columns else pd.Series("", index=working.index, dtype="object")
    topic_series = working["topic_label"] if "topic_label" in working.columns else pd.Series("\uae30\ud0c0", index=working.index, dtype="object")
    brand_series = working["brand_label"] if "brand_label" in working.columns else pd.Series("\ubbf8\uc5b8\uae09", index=working.index, dtype="object")
    published_series = working["published_at"] if "published_at" in working.columns else pd.Series(pd.NaT, index=working.index)

    validity_series = working["comment_validity"] if "comment_validity" in working.columns else pd.Series("valid", index=working.index, dtype="object")
    sentiment_bucket = sentiment_series.where(validity_series.eq("valid"), "excluded")
    working[COL_SENTIMENT_CODE] = sentiment_bucket.fillna("neutral")
    working[COL_COUNTRY] = region_series.map(localize_region)
    working[COL_SENTIMENT] = working[COL_SENTIMENT_CODE].map(localize_sentiment)
    working[COL_CEJ] = topic_series.fillna("\uae30\ud0c0")
    working[COL_BRAND] = brand_series.fillna("\ubbf8\uc5b8\uae09")
    working[COL_WRITTEN_AT] = pd.to_datetime(published_series, errors="coerce", utc=True).dt.tz_localize(None)
    working[COL_WEEK_START] = working[COL_WRITTEN_AT].dt.to_period("W-SUN").dt.start_time
    working[COL_WEEK_LABEL] = working[COL_WEEK_START].dt.strftime("%Y-%m-%d")
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



def compute_filtered_bundle(comments_df: pd.DataFrame, videos_df: pd.DataFrame, quality_df: pd.DataFrame, filters: dict[str, Any]) -> dict[str, Any]:
    base_comments = comments_df.copy()
    products = filters.get("products") or []
    regions = filters.get("regions") or []
    sentiments = filters.get("sentiments") or []
    cej = filters.get("cej") or []
    keyword_query = str(filters.get("keyword_query") or "").strip()

    if products:
        base_comments = base_comments[base_comments["product"].isin(products)]
    if regions:
        base_comments = base_comments[base_comments[COL_COUNTRY].isin(regions)]
    if cej:
        base_comments = base_comments[base_comments[COL_CEJ].isin(cej)]
    if keyword_query:
        base_comments = base_comments[
            base_comments["cleaned_text"].fillna("").str.contains(keyword_query, case=False, na=False)
            | base_comments["text_display"].fillna("").str.contains(keyword_query, case=False, na=False)
        ]

    filtered_comments = base_comments.copy()
    if sentiments:
        filtered_comments = filtered_comments[filtered_comments[COL_SENTIMENT].isin(sentiments)]

    representative_comments = base_comments.copy()
    valid_series = representative_comments.get("comment_validity", pd.Series("valid", index=representative_comments.index))
    representative_comments = representative_comments[valid_series.eq("valid")].copy()

    filtered_videos = videos_df.copy()
    if COL_COUNTRY not in filtered_videos.columns and "region" in filtered_videos.columns:
        filtered_videos[COL_COUNTRY] = filtered_videos["region"].map(localize_region)
    if products:
        filtered_videos = filtered_videos[filtered_videos["product"].isin(products)]
    if regions and COL_COUNTRY in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos[COL_COUNTRY].isin(regions)]

    quality_filtered = quality_df.copy()
    if not quality_filtered.empty:
        if products and "product" in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered["product"].isin(products)]
        if regions and "region" in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered["region"].map(localize_region).isin(regions)]
    if "comment_validity" in filtered_comments.columns:
        removed_count = int((filtered_comments["comment_validity"] == "excluded").sum())
        meaningful_count = int((filtered_comments["comment_validity"] == "valid").sum())
    else:
        removed_count = 0
        meaningful_count = len(filtered_comments)
    return {
        "comments": filtered_comments,
        "representative_comments": representative_comments,
        "videos": filtered_videos,
        "removed_count": removed_count,
        "meaningful_count": meaningful_count,
    }


def build_weekly_sentiment_window(comments_df: pd.DataFrame, weeks_per_view: int, page: int) -> tuple[pd.DataFrame, int]:
    columns = [COL_WEEK_START, COL_WEEK_LABEL, "\uc8fc\ucc28\ucd95\uc57d", COL_SENTIMENT_CODE, COL_SENTIMENT, "\ube44\uc728", "\ub313\uae00\uc218"]
    if comments_df.empty:
        return pd.DataFrame(columns=columns), 1
    dated = comments_df.dropna(subset=[COL_WEEK_START]).copy()
    if dated.empty:
        return pd.DataFrame(columns=columns), 1
    all_weeks = sorted(dated[COL_WEEK_START].dropna().unique())
    total_pages = max(1, (len(all_weeks) + weeks_per_view - 1) // weeks_per_view)
    safe_page = min(max(page, 1), total_pages)
    end_index = len(all_weeks) - (safe_page - 1) * weeks_per_view
    start_index = max(0, end_index - weeks_per_view)
    selected_weeks = sorted(all_weeks[start_index:end_index])
    if not selected_weeks:
        return pd.DataFrame(columns=columns), total_pages

    window = dated[dated[COL_WEEK_START].isin(selected_weeks)].copy()
    totals = window.groupby(COL_WEEK_START, dropna=False).size().rename("\ucd1d\ub313\uae00\uc218")
    grouped = (
        window.groupby([COL_WEEK_START, "sentiment_label"], dropna=False)
        .size()
        .reset_index(name="\ub313\uae00\uc218")
        .merge(totals, on=COL_WEEK_START, how="left")
    )
    grouped["\ube44\uc728"] = (grouped["\ub313\uae00\uc218"] / grouped["\ucd1d\ub313\uae00\uc218"]).fillna(0).round(4)
    base = pd.MultiIndex.from_product(
        [selected_weeks, ["positive", "negative"]],
        names=[COL_WEEK_START, "sentiment_label"],
    ).to_frame(index=False)
    merged = base.merge(
        grouped[[COL_WEEK_START, "sentiment_label", "\ub313\uae00\uc218", "\ube44\uc728"]],
        on=[COL_WEEK_START, "sentiment_label"],
        how="left",
    )
    merged[["\ub313\uae00\uc218", "\ube44\uc728"]] = merged[["\ub313\uae00\uc218", "\ube44\uc728"]].fillna(0)
    merged[COL_SENTIMENT_CODE] = merged["sentiment_label"]
    merged[COL_SENTIMENT] = merged[COL_SENTIMENT_CODE].map(localize_sentiment)
    merged[COL_WEEK_LABEL] = pd.to_datetime(merged[COL_WEEK_START]).dt.strftime("%Y-%m-%d")
    merged["\uc8fc\ucc28\ucd95\uc57d"] = pd.to_datetime(merged[COL_WEEK_START]).dt.strftime("%m-%d")
    return merged[columns], total_pages


def build_keyword_pack(comments_df: pd.DataFrame, sentiment_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = tuple(comments_df.loc[comments_df["sentiment_label"] == sentiment_code, "cleaned_text"].fillna("").astype(str).tolist())
    return build_keyword_summary(texts, top_n=6, version="v6"), build_keyword_summary(texts, top_n=30, version="v6")




def render_keyword_panel(comments_df: pd.DataFrame, sentiment_code: str, title: str) -> None:
    donut_df, full_df = build_keyword_pack(comments_df, sentiment_code)
    st.markdown(f"### {title}")
    if donut_df.empty:
        st.info("\ud45c\uc2dc\ud560 \ud0a4\uc6cc\ub4dc \ub370\uc774\ud130\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        return

    total = max(int(donut_df["count"].sum()), 1)
    donut_df = donut_df.copy()
    donut_df["share_text"] = donut_df["count"].map(lambda v: f"{(v / total) * 100:.1f}%")
    donut_rows = tuple((str(row.keyword), int(row.count), str(row.share_text)) for row in donut_df.itertuples(index=False))
    donut_image = build_donut_chart_image(
        donut_rows,
        sentiment_code=sentiment_code,
        center_label="\uae0d\uc815" if sentiment_code == "positive" else "\ubd80\uc815",
    )

    left, right = st.columns([0.72, 0.28])
    with left:
        if donut_image:
            st.image(donut_image, use_container_width=True)
    with right:
        st.markdown('<div class="voc-legend-compact">', unsafe_allow_html=True)
        palette = [
            "#B8E0B2", "#95D29A", "#72BF7A", "#52AE61", "#34994D", "#1F8741"
        ] if sentiment_code == "positive" else [
            "#FFC489", "#FFAB59", "#FF9138", "#F26A00", "#E85A00", "#D24D00"
        ]
        for idx, row in enumerate(donut_df.itertuples(index=False)):
            color = palette[idx % len(palette)]
            st.markdown(
                f'<div class="voc-legend-item"><span class="voc-legend-swatch" style="background:{color}"></span>{row.keyword} {row.share_text}</div>',
                unsafe_allow_html=True,
            )
        st.markdown('</div>', unsafe_allow_html=True)

    full_rows = tuple((str(row.keyword), int(row.count)) for row in full_df.itertuples(index=False))
    wordcloud = build_wordcloud_image(full_rows, sentiment_code=sentiment_code)
    if wordcloud:
        st.image(wordcloud, use_container_width=True)

    with st.expander("\uc804\uccb4 \ud0a4\uc6cc\ub4dc \ubaa9\ub85d \ubcf4\uae30"):
        st.dataframe(full_df.rename(columns={"keyword": "\ud0a4\uc6cc\ub4dc", "count": "\uc5b8\uae09 \uc218"}), use_container_width=True, hide_index=True)


def render_issue_tables(data: dict[str, pd.DataFrame], selected_products: list[str], selected_regions: list[str]) -> None:
    new_df = data.get("new_issue_keywords", pd.DataFrame()).copy()
    persistent_df = data.get("persistent_issue_keywords", pd.DataFrame()).copy()

    for frame in (new_df, persistent_df):
        if not frame.empty:
            frame[COL_COUNTRY] = frame.get("region", pd.Series("", index=frame.index)).map(localize_region)
            if selected_products and "product" in frame.columns:
                frame.drop(frame[~frame["product"].isin(selected_products)].index, inplace=True)
            if selected_regions:
                frame.drop(frame[~frame[COL_COUNTRY].isin(selected_regions)].index, inplace=True)

    st.caption("\ucd5c\uadfc\uc5d0 \uae09\uc99d\ud55c \uc774\uc288\uc640 \uc5ec\ub7ec \uc8fc\ucc28\uc5d0 \uac78\uccd0 \ubc18\ubcf5\ub418\ub294 \uc774\uc288\ub97c \ub098\ub220\uc11c \ubd05\ub2c8\ub2e4.")
    tab_new, tab_persistent = st.tabs(["\uc2e0\uaddc \uc774\uc288 \ud0a4\uc6cc\ub4dc", "\uc9c0\uc18d \uc774\uc288 \ud0a4\uc6cc\ub4dc"])
    with tab_new:
        if new_df.empty:
            st.info("\uc2e0\uaddc \uc774\uc288 \ud0a4\uc6cc\ub4dc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        else:
            new_display = new_df.rename(columns={"product": "\uc81c\ud488\uad70", "keyword": "\ud0a4\uc6cc\ub4dc", "count": "\uc5b8\uae09 \uc218", "latest_week": "\ucd5c\uadfc \uc8fc\ucc28"})
            st.dataframe(new_display[["\uc81c\ud488\uad70", COL_COUNTRY, "\ud0a4\uc6cc\ub4dc", "\uc5b8\uae09 \uc218", "\ucd5c\uadfc \uc8fc\ucc28"]].head(15), use_container_width=True, hide_index=True)
    with tab_persistent:
        if persistent_df.empty:
            st.info("\uc9c0\uc18d \uc774\uc288 \ud0a4\uc6cc\ub4dc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        else:
            persistent_display = persistent_df.rename(columns={"product": "\uc81c\ud488\uad70", "keyword": "\ud0a4\uc6cc\ub4dc", "count": "\uc5b8\uae09 \uc218", "weeks_active": "\uc9c0\uc18d \uc8fc\ucc28 \uc218", "latest_week": "\ucd5c\uadfc \uc8fc\ucc28"})
            st.dataframe(persistent_display[["\uc81c\ud488\uad70", COL_COUNTRY, "\ud0a4\uc6cc\ub4dc", "\uc5b8\uae09 \uc218", "\uc9c0\uc18d \uc8fc\ucc28 \uc218", "\ucd5c\uadfc \uc8fc\ucc28"]].head(15), use_container_width=True, hide_index=True)


def render_representative_comments(filtered_comments: pd.DataFrame) -> None:
    st.subheader("실제 고객 코멘트 샘플")
    st.caption("감성 강도, 좋아요 수, 문장 길이를 함께 반영해 대표성이 높은 코멘트를 우선 보여줍니다.")
    st.caption("Representative comments are prioritized based on sentiment strength, number of likes, and content richness (comment length).")
    negative_showcase = build_comment_showcase(filtered_comments, sentiment="negative", limit=8)
    positive_showcase = build_comment_showcase(filtered_comments, sentiment="positive", limit=8)

    if negative_showcase.empty and positive_showcase.empty:
        fallback_showcase = build_comment_showcase(filtered_comments, sentiment="", limit=6)
        st.warning("현재 조건에서 강한 긍정/부정 대표 코멘트가 부족해, 의미 있는 유효 댓글을 우선 보여줍니다.")
        render_comment_table(fallback_showcase, key_prefix="neutral")
        return

    st.write(f"부정 대표 코멘트 {len(negative_showcase)}건")
    render_comment_table(negative_showcase, key_prefix="negative")

    st.write(f"긍정 대표 코멘트 {len(positive_showcase)}건")
    render_comment_table(positive_showcase, key_prefix="positive")


def render_comment_table(comment_showcase: pd.DataFrame, key_prefix: str) -> None:
    if key_prefix == "negative":
        sentiment_name = "negative"
    elif key_prefix == "positive":
        sentiment_name = "positive"
    else:
        sentiment_name = "neutral"

    if comment_showcase.empty:
        st.info("현재 조건에서 표시할 대표 코멘트가 없습니다.")
        return

    top_rows = comment_showcase.reset_index(drop=True)
    for idx, selected in top_rows.iterrows():
        with st.container(border=True):
            st.markdown(f"#### 대표 코멘트 {idx + 1}")
            likes_value = pd.to_numeric(selected.get(COL_LIKES, 0), errors="coerce")
            likes = 0 if pd.isna(likes_value) else int(likes_value)
            st.caption(
                f"제품군: {_safe_text(selected.get('product', '-')) or '-'} · {COL_COUNTRY}: {_safe_text(selected.get(COL_COUNTRY, '-')) or '-'} · "
                f"{COL_CEJ}: {_safe_text(selected.get(COL_CEJ, '-')) or '-'} · {COL_BRAND}: {_safe_text(selected.get(COL_BRAND, '-')) or '-'} · 좋아요 수: {likes:,}"
            )

            video_title = _safe_text(selected.get(COL_VIDEO_TITLE, '-')) or '-'
            video_link = _safe_text(selected.get(COL_VIDEO_LINK, '#')) or '#'
            st.markdown(f"관련 영상: [{video_title}]({video_link})")

            working_selected = selected.copy()
            original_text = _safe_text(working_selected.get(COL_ORIGINAL, ""))
            raw_translation = _safe_text(working_selected.get(COL_TRANSLATION, ""))
            language = _safe_text(working_selected.get(COL_LANGUAGE, "")).lower()
            is_korean = language.startswith("ko") or _looks_korean_text(original_text)
            translation = _resolve_card_translation(original_text, raw_translation, is_korean)
            working_selected[COL_TRANSLATION] = translation

            display_reason = _safe_text(working_selected.get("display_reason", ""))
            if not display_reason or _looks_garbled(display_reason):
                display_reason = summarize_comment_reason(working_selected, sentiment_name)
            st.markdown(f'<div class="voc-reason">{display_reason}</div>', unsafe_allow_html=True)

            decision_label, decision_help = _decision_badge(_safe_text(working_selected.get("representative_decision", "review_needed")) or "review_needed")
            rep_score = pd.to_numeric(working_selected.get("representative_score", 0), errors="coerce")
            rep_score_text = "-" if pd.isna(rep_score) or rep_score <= 0 else f"{int(rep_score)}/30"
            st.caption(f"{decision_label} · 대표성 점수 {rep_score_text}")
            rep_reason = _safe_text(working_selected.get("representative_reason", ""))
            if not rep_reason or _looks_garbled(rep_reason):
                rep_reason = decision_help
            st.write(rep_reason)

            st.markdown("**요약**")
            st.write(_summarize_for_dashboard(working_selected, sentiment_name))

            voc_items = _extract_voc_items(working_selected, sentiment_name)
            if voc_items:
                st.markdown("**추출된 VoC 항목**")
                for item in voc_items:
                    evidence = _safe_text(item.get("evidence_span", "")) or "근거 문구를 다시 확인 중입니다."
                    st.markdown(f"- `{item['sentiment']}` · **{item['topic']}**: {item['insight']}")
                    st.caption(f"근거: {evidence}")

            context_text = _safe_text(working_selected.get(COL_CONTEXT, ""))
            context_translation = _safe_text(working_selected.get(COL_CONTEXT_TRANSLATION, context_text)) or context_text
            if context_text and (not _looks_korean_text(context_text)) and (not context_translation or context_translation == context_text or _looks_garbled(context_translation)):
                translated_context = _safe_text(translate_to_korean(context_text))
                if translated_context and translated_context != context_text and _looks_korean_text(translated_context):
                    context_translation = translated_context
            if context_text and bool(working_selected.get("display_context_required", False)):
                st.markdown("**연관 맥락**")
                if context_translation and context_translation != context_text and _looks_korean_text(context_translation):
                    st.write(context_translation)
                    with st.expander("연관 원문 보기"):
                        st.write(context_text)
                else:
                    st.write(context_text)

            if is_korean:
                if _is_long_text(original_text):
                    with st.expander("원문 댓글 더보기"):
                        st.markdown(f"**{COL_ORIGINAL}**")
                        st.write(original_text)
                else:
                    st.markdown(f"**{COL_ORIGINAL}**")
                    st.write(original_text)
            else:
                with st.expander("번역/원문 더보기"):
                    st.markdown(f"**{COL_TRANSLATION}**")
                    st.write(translation)
                    st.markdown(f"**{COL_ORIGINAL}**")
                    st.write(original_text)


def build_video_summary(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame) -> pd.DataFrame:
    videos = filtered_videos.copy()
    if videos.empty:
        return pd.DataFrame()

    comments = filtered_comments.copy()
    if comments.empty:
        videos["\ubd84\uc11d \ub313\uae00 \uc218"] = 0
        videos["\uc720\ud6a8 \ub313\uae00 \uc218"] = 0
        videos["\uc81c\uc678 \ub313\uae00 \uc218"] = 0
        videos["\uae0d\uc815 \ub313\uae00 \uc218"] = 0
        videos["\ubd80\uc815 \ub313\uae00 \uc218"] = 0
        videos["\uc911\ub9bd \ub313\uae00 \uc218"] = 0
        videos["\ubd80\uc815 \ube44\uc728"] = 0.0
        return videos

    comment_stats = comments.groupby("video_id", dropna=False).agg(
        **{
            "\ubd84\uc11d \ub313\uae00 \uc218": ("video_id", "size"),
            "\uc720\ud6a8 \ub313\uae00 \uc218": ("comment_validity", lambda s: int((s == "valid").sum())),
            "\uc81c\uc678 \ub313\uae00 \uc218": ("comment_validity", lambda s: int((s == "excluded").sum())),
            "\uae0d\uc815 \ub313\uae00 \uc218": ("sentiment_label", lambda s: int((s == "positive").sum())),
            "\ubd80\uc815 \ub313\uae00 \uc218": ("sentiment_label", lambda s: int((s == "negative").sum())),
            "\uc911\ub9bd \ub313\uae00 \uc218": ("sentiment_label", lambda s: int((s == "neutral").sum())),
        }
    ).reset_index()
    comment_stats["\ubd80\uc815 \ube44\uc728"] = comment_stats.apply(
        lambda row: (row["\ubd80\uc815 \ub313\uae00 \uc218"] / row["\uc720\ud6a8 \ub313\uae00 \uc218"]) if row["\uc720\ud6a8 \ub313\uae00 \uc218"] else 0.0,
        axis=1,
    )

    negative_reason = (
        comments[comments["sentiment_label"] == "negative"]
        .sort_values(["video_id", "like_count"], ascending=[True, False])
        .groupby("video_id", as_index=False)["classification_reason"]
        .first()
        .rename(columns={"classification_reason": "\uc8fc\uc694 \ubd80\uc815 \ud3ec\uc778\ud2b8"})
    )

    summary = videos.merge(comment_stats, on="video_id", how="left").merge(negative_reason, on="video_id", how="left")
    for col in ["\ubd84\uc11d \ub313\uae00 \uc218", "\uc720\ud6a8 \ub313\uae00 \uc218", "\uc81c\uc678 \ub313\uae00 \uc218", "\uae0d\uc815 \ub313\uae00 \uc218", "\ubd80\uc815 \ub313\uae00 \uc218", "\uc911\ub9bd \ub313\uae00 \uc218", "\ubd80\uc815 \ube44\uc728"]:
        if col in summary.columns:
            summary[col] = summary[col].fillna(0)
    summary[COL_COUNTRY] = summary.get(COL_COUNTRY, summary.get("region", pd.Series(dtype=str)).map(localize_region))
    summary[COL_VIDEO_LINK] = summary.get("video_url", "")
    summary[COL_VIDEO_TITLE] = summary.get("title", "")
    summary["\ubc1c\ud589\uc77c"] = pd.to_datetime(summary.get("published_at"), errors="coerce").dt.strftime("%Y-%m-%d")
    return summary.sort_values(["\ubd80\uc815 \ub313\uae00 \uc218", "\ubd84\uc11d \ub313\uae00 \uc218", "view_count"], ascending=[False, False, False])


def render_video_summary_page(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame, density_df: pd.DataFrame) -> None:
    st.markdown("### 영상 분석 요약")
    st.caption("댓글 분석에 활용된 영상들을 영상 단위로 요약해 보고, 선택한 영상의 상세 분석으로 들어갈 수 있습니다.")
    summary = build_video_summary(filtered_comments, filtered_videos)
    if summary.empty:
        st.info("표시할 영상 요약 데이터가 없습니다.")
        return

    k1, k2, k3 = st.columns(3)
    with k1:
        render_card("분석 영상 수", f"{len(summary):,}")
    with k2:
        render_card("총 분석 댓글", f"{int(summary['분석 댓글 수'].sum()):,}")
    with k3:
        top_video = summary.iloc[0][COL_VIDEO_TITLE] if not summary.empty else "-"
        render_card("최상위 이슈 영상", str(top_video)[:28] or "-")

    summary_display = summary.copy()
    summary_display["View Analysis"] = "상세 분석 보기"
    display_cols = [
        "product", COL_COUNTRY, COL_VIDEO_TITLE, "발행일", "view_count", "분석 댓글 수", "유효 댓글 수",
        "긍정 댓글 수", "부정 댓글 수", "중립 댓글 수", "제외 댓글 수", "부정 비율", "주요 부정 포인트", "View Analysis",
    ]
    existing_cols = [col for col in display_cols if col in summary_display.columns]
    st.dataframe(summary_display[existing_cols], use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("#### 영상당 부정 댓글 밀도")
        if not density_df.empty:
            density_display = pd.DataFrame({
                "product": density_df.get("product", pd.Series("", index=density_df.index)),
                COL_COUNTRY: density_df.get(COL_COUNTRY, density_df.get("region", pd.Series("", index=density_df.index)).map(localize_region)),
                COL_VIDEO_TITLE: density_df.get("title", density_df.get(COL_VIDEO_TITLE, pd.Series("", index=density_df.index))),
                "부정 댓글 수": density_df.get("negative_comments", pd.Series(0, index=density_df.index)),
                "분석 댓글 수": density_df.get("total_comments", pd.Series(0, index=density_df.index)),
                "부정 밀도": density_df.get("negative_density", pd.Series(0.0, index=density_df.index)),
                COL_VIDEO_LINK: density_df.get("video_url", density_df.get(COL_VIDEO_LINK, pd.Series("", index=density_df.index))),
            })
            st.dataframe(density_display.head(20), use_container_width=True, hide_index=True)
        else:
            st.info("표시할 영상 밀도 데이터가 없습니다.")


    options = summary[["video_id", COL_VIDEO_TITLE, "product", COL_COUNTRY]].drop_duplicates().to_dict("records")
    selected = st.selectbox(
        "상세 분석할 영상 선택",
        options=options,
        format_func=lambda row: f"[{row.get('product', '-')}/{row.get(COL_COUNTRY, '-')}] {row.get(COL_VIDEO_TITLE, '-')}",
        key="video_analysis_selector",
    )
    if selected:
        selected_row = summary[summary["video_id"] == selected["video_id"]].iloc[0]
        render_video_detail_page(filtered_comments, selected_row)


def render_video_detail_page(filtered_comments: pd.DataFrame, selected_video: pd.Series) -> None:
    video_id = selected_video.get("video_id")
    video_comments = filtered_comments[filtered_comments["video_id"] == video_id].copy()

    st.markdown("---")
    st.markdown("### 선택 영상 상세 분석")
    st.markdown(f"#### {selected_video.get(COL_VIDEO_TITLE, '-')}")
    st.caption(
        f"제품군: {selected_video.get('product', '-')} ? {COL_COUNTRY}: {selected_video.get(COL_COUNTRY, '-')} ? "
        f"발행일: {selected_video.get('발행일', '-')} ? 조회수: {int(selected_video.get('view_count', 0) or 0):,}"
    )
    st.markdown(f"영상 링크: [{selected_video.get(COL_VIDEO_LINK, '-')}]({selected_video.get(COL_VIDEO_LINK, '#')})")

    if video_comments.empty:
        st.info("이 영상에 연결된 분석 댓글이 없습니다.")
        return

    validity = video_comments["comment_validity"] if "comment_validity" in video_comments.columns else pd.Series("valid", index=video_comments.index)
    valid_comments = video_comments[validity.eq("valid")].copy()
    excluded_comments = video_comments[validity.eq("excluded")].copy()

    positive_count = int((valid_comments["sentiment_label"] == "positive").sum()) if not valid_comments.empty else 0
    negative_count = int((valid_comments["sentiment_label"] == "negative").sum()) if not valid_comments.empty else 0
    neutral_count = int((valid_comments["sentiment_label"] == "neutral").sum()) if not valid_comments.empty else 0
    excluded_count = len(excluded_comments)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_card("긍정 댓글 수", f"{positive_count:,}")
    with c2:
        render_card("부정 댓글 수", f"{negative_count:,}")
    with c3:
        render_card("중립 댓글 수", f"{neutral_count:,}")
    with c4:
        render_card("제외 댓글 수", f"{excluded_count:,}")

    dist_rows = pd.DataFrame([
        {"감성코드": "positive", "감성": localize_sentiment("positive"), "댓글 수": positive_count},
        {"감성코드": "negative", "감성": localize_sentiment("negative"), "댓글 수": negative_count},
        {"감성코드": "neutral", "감성": localize_sentiment("neutral"), "댓글 수": neutral_count},
        {"감성코드": "excluded", "감성": localize_sentiment("excluded"), "댓글 수": excluded_count},
    ])
    total = max(1, int(dist_rows["댓글 수"].sum()))
    dist_rows["비율"] = dist_rows["댓글 수"] / total
    chart = alt.Chart(dist_rows).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("감성:N", sort=[localize_sentiment("positive"), localize_sentiment("negative"), localize_sentiment("neutral"), localize_sentiment("excluded")]),
        y=alt.Y("댓글 수:Q", title="댓글 수"),
        color=alt.Color("감성코드:N", scale=alt.Scale(domain=["positive", "negative", "neutral", "excluded"], range=["#4D79F6", "#FFAA3B", "#94A3B8", "#D1D5DB"]), legend=None),
        tooltip=[alt.Tooltip("감성:N", title="감성"), alt.Tooltip("댓글 수:Q", title="댓글 수"), alt.Tooltip("비율:Q", title="비율", format=".1%")],
    ).properties(height=220)
    st.altair_chart(chart, use_container_width=True)

    with st.container(border=True):
        st.markdown("#### 대표 코멘트")
        neg = build_comment_showcase(video_comments, sentiment="negative", limit=3)
        pos = build_comment_showcase(video_comments, sentiment="positive", limit=3)
        st.markdown(f"##### 부정 대표 코멘트 {len(neg)}건")
        render_comment_table(neg, key_prefix="negative")
        st.markdown(f"##### 긍정 대표 코멘트 {len(pos)}건")
        render_comment_table(pos, key_prefix="positive")

    with st.container(border=True):
        st.markdown("#### 전체 댓글 목록")
        display = video_comments.copy()
        display[COL_ORIGINAL] = display.get("text_display", display.get(COL_ORIGINAL, "")).fillna("")
        if COL_TRANSLATION not in display.columns:
            display[COL_TRANSLATION] = display[COL_ORIGINAL]
        lang = display.get("language_detected", pd.Series("", index=display.index)).fillna("").astype(str).str.lower()
        needs_translation = (~lang.str.startswith("ko")) & display[COL_ORIGINAL].str.strip().ne("")
        if needs_translation.any():
            display.loc[needs_translation, COL_TRANSLATION] = display.loc[needs_translation, COL_ORIGINAL].map(translate_to_korean)
        display[COL_CONTEXT] = display.get(COL_CONTEXT, display.get("parent_comment_text", pd.Series("", index=display.index))).fillna("")
        display["분류 이유"] = display.get("classification_reason", pd.Series("", index=display.index)).fillna("")
        display["맥락 필요"] = display.get("context_required_for_display", pd.Series(False, index=display.index)).fillna(False)
        display["분류 유형"] = display.get("classification_type", pd.Series("", index=display.index)).fillna("")
        display["신뢰도"] = display.get("confidence_level", pd.Series("", index=display.index)).fillna("")
        display["제품 타깃"] = display.get("product_target", pd.Series("", index=display.index)).fillna("")
        display[COL_SENTIMENT] = display.get(COL_SENTIMENT, display.get("sentiment_label", pd.Series("", index=display.index))).fillna("")
        cols = [COL_SENTIMENT, "제품 타깃", "분류 유형", "신뢰도", COL_ORIGINAL, COL_TRANSLATION, COL_CONTEXT, "분류 이유", "맥락 필요", COL_LIKES]
        existing = [c for c in cols if c in display.columns]
        st.dataframe(display[existing], use_container_width=True, hide_index=True)


def main() -> None:
    st.set_page_config(page_title="가전 VoC Dashboard", layout="wide")
    apply_theme()

    # 회사 부팅 최적화
    
    # ✅ 1) 첫 실행은 가볍게 통과(health-check 안정화)
    if "booted_once" not in st.session_state:
        st.session_state["booted_once"] = True
        st.title("가전 VoC Dashboard")
        st.caption("초기화 중입니다… (첫 로딩 안정화)")
        st.info("잠시 후 자동으로 데이터 로딩을 시작합니다.")
        st.experimental_rerun()   # ✅ 여기서 바로 재실행

    #

    
    # 기존코드 data = load_dashboard_data()
    #회사 부팅 최적화용 추가
    with st.spinner("데이터 로딩 중…"):
    data = get_dashboard_data_resource()
    #
    comments_df = add_localized_columns(data["comments"])
    if comments_df.empty:
        st.warning("표시할 분석 결과가 없습니다. 먼저 데이터를 수집해주세요.")
        return

    videos_df = data["videos"].copy()
    videos_df[COL_COUNTRY] = videos_df.get("region", pd.Series(dtype=str)).map(localize_region)
    products = [value for value in PRODUCT_ORDER if value in comments_df["product"].astype(str).unique().tolist()]
    regions = [value for value in [localize_region("KR"), localize_region("US")] if value in comments_df[COL_COUNTRY].astype(str).unique().tolist()]
    sentiments = [value for value in [localize_sentiment("positive"), localize_sentiment("negative"), localize_sentiment("neutral"), localize_sentiment("excluded")] if value in comments_df[COL_SENTIMENT].astype(str).unique().tolist()]
    cej_labels = [value for value in CEJ_ORDER if value in comments_df[COL_CEJ].astype(str).unique().tolist()]

    st.markdown("## 가전 VoC Dashboard")
    st.caption("주간 감성 변화, 핵심 키워드, CEJ 단계별 문제, 대표 코멘트를 제품·국가별로 빠르게 확인합니다.")
    st.caption("Build: 2026-03-22 23:25 / representative-header-render-fix-2")

    with st.sidebar:
        st.markdown("### 탐색 필터")
        selected_products = st.multiselect("제품군", products, default=products)
        selected_regions = st.multiselect("국가", regions, default=regions)
        selected_sentiments = st.multiselect("감성", sentiments, default=sentiments)
        selected_cej = st.multiselect("고객경험여정 단계", cej_labels, default=cej_labels)
        keyword_query = st.text_input("댓글 내 키워드 검색", "")
        weeks_per_view = st.slider("차트 한 화면 주차 수", min_value=4, max_value=24, value=12, step=2)

    bundle = compute_filtered_bundle(
        comments_df,
        videos_df,
        data["quality_summary"],
        {"products": selected_products, "regions": selected_regions, "sentiments": selected_sentiments, "cej": selected_cej, "keyword_query": keyword_query},
    )
    filtered_comments = bundle["comments"]
    filtered_videos = bundle["videos"]
    all_filtered_weeks = sorted(filtered_comments[COL_WEEK_START].dropna().unique())
    total_pages = max(1, (len(all_filtered_weeks) + weeks_per_view - 1) // weeks_per_view) if all_filtered_weeks else 1
    with st.sidebar:
        page = st.slider("주차 페이지", min_value=1, max_value=total_pages, value=1)

    weekly_window, total_pages = build_weekly_sentiment_window(filtered_comments, weeks_per_view, page)

    cej_df = data["cej_negative_rate"].copy()
    if not cej_df.empty:
        cej_df[COL_COUNTRY] = cej_df["region"].map(localize_region)
        if selected_products:
            cej_df = cej_df[cej_df["product"].isin(selected_products)]
        if selected_regions:
            cej_df = cej_df[cej_df[COL_COUNTRY].isin(selected_regions)]

    brand_df = data["brand_ratio"].copy()
    if not brand_df.empty:
        brand_df[COL_COUNTRY] = brand_df["region"].map(localize_region)
        if selected_products:
            brand_df = brand_df[brand_df["product"].isin(selected_products)]
        if selected_regions:
            brand_df = brand_df[brand_df[COL_COUNTRY].isin(selected_regions)]

    density_df = data["negative_density"].copy()
    if not density_df.empty:
        density_df[COL_COUNTRY] = density_df["region"].map(localize_region)
        if selected_products:
            density_df = density_df[density_df["product"].isin(selected_products)]
        if selected_regions:
            density_df = density_df[density_df[COL_COUNTRY].isin(selected_regions)]

    validity_series = filtered_comments["comment_validity"] if "comment_validity" in filtered_comments.columns else pd.Series("valid", index=filtered_comments.index)
    valid_base = filtered_comments[validity_series.eq("valid")].copy()
    positive_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("positive").mean() * 100)
    negative_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("negative").mean() * 100)
    neutral_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("neutral").mean() * 100)

    raw_comments_export = filtered_comments.rename(columns={
        "text_display": COL_ORIGINAL,
        "text_original": COL_RAW,
        "title": COL_VIDEO_TITLE,
        "video_url": COL_VIDEO_LINK,
        "like_count": COL_LIKES,
        "language_detected": COL_LANGUAGE,
        "removed_reason": COL_REMOVED,
    })

    top_spacer, top_action = st.columns([0.55, 0.45])
    del top_spacer
    with top_action:
        st.download_button(
            "전체 Raw Data 엑셀 다운로드",
            data=build_raw_download_package(raw_comments_export, filtered_videos, weekly_window, cej_df, brand_df, density_df),
            file_name="voc_dashboard_raw_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    tab_comments, tab_videos = st.tabs(["댓글 VoC 대시보드", "영상 분석 요약"])

    with tab_comments:
        row1_col1, row1_col2, row1_col3 = st.columns(3)
        with row1_col1:
            render_card("분석 대상 댓글", f"{len(filtered_comments):,}")
        with row1_col2:
            render_card("유효 댓글 수", f"{bundle['meaningful_count']:,}")
        with row1_col3:
            render_card("제외 댓글 수", f"{bundle['removed_count']:,}")

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            render_card("긍정 응답률", f"{positive_ratio:.1f}%")
        with row2_col2:
            render_card("부정 응답률", f"{negative_ratio:.1f}%")
        with row2_col3:
            render_card("중립 응답률", f"{neutral_ratio:.1f}%")

        with st.container(border=True):
            st.markdown("### 주차별 긍정/부정 응답률")
            if not weekly_window.empty and float(weekly_window["댓글수"].sum()) > 0:
                week_order = weekly_window[[COL_WEEK_LABEL, "주차축약"]].drop_duplicates().sort_values(COL_WEEK_LABEL)
                axis_order = week_order["주차축약"].tolist()
                bar_size = max(8, min(16, int(180 / max(len(axis_order), 1))))
                chart = alt.Chart(weekly_window).mark_bar(size=bar_size, cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                    x=alt.X(
                        "주차축약:N",
                        title="주차 시작일",
                        sort=axis_order,
                        axis=alt.Axis(labelAngle=-35, labelLimit=56, labelPadding=10, titlePadding=16),
                    ),
                    y=alt.Y("비율:Q", title="응답률", axis=alt.Axis(format="%")),
                    xOffset=alt.XOffset(f"{COL_SENTIMENT_CODE}:N", sort=["positive", "negative"]),
                    color=alt.Color(
                        f"{COL_SENTIMENT_CODE}:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#4D79F6", "#FFAA3B"]),
                        legend=alt.Legend(
                            title=None,
                            orient="top",
                            direction="horizontal",
                            symbolType="square",
                            labelExpr="datum.label === 'positive' ? '\uae0d\uc815' : datum.label === 'negative' ? '\ubd80\uc815' : datum.label",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{COL_WEEK_LABEL}:N", title="주차 시작일"),
                        alt.Tooltip(f"{COL_SENTIMENT}:N", title=COL_SENTIMENT),
                        alt.Tooltip("댓글수:Q", title="댓글 수"),
                        alt.Tooltip("비율:Q", title="응답률", format=".1%"),
                    ],
                ).properties(height=320)
                st.altair_chart(chart, use_container_width=True)
                st.caption(f"시간순으로 정렬된 최근 {len(axis_order)}개 주차를 표시합니다. 전체 {total_pages}페이지 중 {min(page, total_pages)}페이지입니다.")
            else:
                st.info("해당 조건에서 주간 차트를 그릴 데이터가 없습니다.")

        with st.container(border=True):
            render_keyword_panel(filtered_comments, "negative", "부정 키워드 비중")

        with st.container(border=True):
            render_keyword_panel(filtered_comments, "positive", "긍정 키워드 비중")

        with st.container(border=True):
            st.markdown("### CEJ 단계별 부정률")
            if not cej_df.empty:
                cej_display = cej_df.rename(columns={
                    "topic_label": COL_CEJ,
                    "comments": "댓글 수",
                    "negative_comments": "부정 댓글 수",
                    "negative_rate": "부정률",
                }).copy()
                cej_display[COL_CEJ] = pd.Categorical(cej_display[COL_CEJ], categories=CEJ_ORDER, ordered=True)
                cej_display = cej_display.sort_values(["product", COL_COUNTRY, COL_CEJ])
                st.dataframe(cej_display[["product", COL_COUNTRY, COL_CEJ, "댓글 수", "부정 댓글 수", "부정률"]], use_container_width=True, hide_index=True)
            else:
                st.info("표시할 CEJ 데이터가 없습니다.")

        with st.container(border=True):
            st.markdown("### 개선 포인트 신호")
            render_issue_tables(data, selected_products, selected_regions)

        with st.container(border=True):
            st.markdown("### 브랜드 언급 비중")
            if not brand_df.empty:
                brand_display = brand_df.rename(columns={
                    "brand_label": COL_BRAND,
                    "count": "언급 댓글 수",
                    "ratio": "언급 비율",
                })
                st.dataframe(brand_display[["product", COL_COUNTRY, COL_BRAND, "언급 댓글 수", "언급 비율"]], use_container_width=True, hide_index=True)
            else:
                st.info("표시할 브랜드 데이터가 없습니다.")


        with st.container(border=True):
            render_representative_comments(bundle.get("representative_comments", filtered_comments))

    with tab_videos:
        render_video_summary_page(filtered_comments, filtered_videos, density_df)


if __name__ == "__main__":
    main()





