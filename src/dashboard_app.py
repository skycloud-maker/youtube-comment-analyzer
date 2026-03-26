"""Streamlit dashboard for appliance VoC exploration."""

from __future__ import annotations

from io import BytesIO
import codecs
import re
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "data" / "mplcache").resolve()))

try:
    import koreanize_matplotlib  # type: ignore
except Exception:
    koreanize_matplotlib = None

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
CLOUD_CACHE_FILE = CACHE_DIR / "dashboard_bundle_cloud.pkl"
CLOUD_CACHE_META = CACHE_DIR / "dashboard_bundle_cloud_meta.json"
CACHE_VERSION = "2026-03-27-0215"
REGION_LABELS = {"KR": "한국", "US": "미국"}
SENTIMENT_LABELS = {"positive": "\uae0d\uc815", "negative": "\ubd80\uc815", "neutral": "\uc911\ub9bd", "excluded": "\uc81c\uc678"}
CEJ_ORDER = ["\uc778\uc9c0", "\ud0d0\uc0c9", "\uacb0\uc815", "\uad6c\ub9e4", "\ubc30\uc1a1", "\uc0ac\uc6a9\uc900\ube44", "\uc0ac\uc6a9", "\uad00\ub9ac", "\uad50\uccb4", "\uae30\ud0c0"]
PRIMARY_PRODUCT_FILTERS = ["\ub0c9\uc7a5\uace0", "\uc138\ud0c1\uae30", "\uac74\uc870\uae30", "\uc2dd\uae30\uc138\ucc99\uae30", "\uccad\uc18c\uae30", "\uc624\ube10"]
EXTRA_PRODUCT_FILTERS = ["\uc815\uc218\uae30", "\uc778\ub355\uc158", "\uc640\uc778\uc140\ub7ec", "\ucef5\uc138\ucc99\uae30", "\uc2e4\ub0b4\uc2dd\ubb3c\uc7ac\ubc30\uae30(\ud2d4\uc6b4)"]
PRODUCT_ORDER = PRIMARY_PRODUCT_FILTERS + EXTRA_PRODUCT_FILTERS
BRAND_FILTER_OPTIONS = ["LG", "Samsung", "GE", "Whirlpool"]
FONT_CANDIDATES = [
    BASE_DIR / "fonts" / "NanumGothic.ttf",
    Path("C:/Windows/Fonts/malgun.ttf"),
    Path("C:/Windows/Fonts/NanumGothic.ttf"),
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
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    latest_signature = _latest_signature()
    if CACHE_FILE.exists() and CACHE_META.exists():
        try:
            meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
            if meta.get("signature") == latest_signature and meta.get("version") == CACHE_VERSION:
                return pd.read_pickle(CACHE_FILE)
        except Exception:
            pass

    if CACHE_FILE.exists():
        try:
            return pd.read_pickle(CACHE_FILE)
        except Exception:
            pass

    if CLOUD_CACHE_FILE.exists():
        try:
            return pd.read_pickle(CLOUD_CACHE_FILE)
        except Exception:
            pass

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
        "representative_bundles": [],
        "opinion_units": [],
        "analysis_comments": [],
        "monitoring_summary": [],
        "reporting_summary": [],
    }

    for run_dir in sorted(PROCESSED_DIR.iterdir()) if PROCESSED_DIR.exists() else []:
        if not run_dir.is_dir():
            continue
        context = {"run_id": run_dir.name}
        for key, filename in [
            ("comments", "comments.parquet"),
            ("videos", "videos.parquet"),
            ("brand_ratio", "brand_ratio.parquet"),
            ("cej_negative_rate", "cej_negative_rate.parquet"),
            ("negative_density", "negative_density.parquet"),
            ("new_issue_keywords", "new_issue_keywords.parquet"),
            ("persistent_issue_keywords", "persistent_issue_keywords.parquet"),
            ("quality_summary", "quality_summary.parquet"),
            ("representative_comments", "representative_comments.parquet"),
            ("representative_bundles", "representative_bundles.parquet"),
            ("opinion_units", "opinion_units.parquet"),
            ("analysis_comments", "analysis_comments.parquet"),
            ("monitoring_summary", "monitoring_summary.parquet"),
            ("reporting_summary", "reporting_summary.parquet"),
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
    CACHE_META.write_text(json.dumps({"signature": latest_signature, "version": CACHE_VERSION}, ensure_ascii=False), encoding="utf-8")
    return data


@st.cache_resource
def get_dashboard_data_resource():
    return load_dashboard_data()


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
        return None
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
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {background:#ffffff; border:1px solid #e5ece7; border-radius:18px; padding:10px 12px; box-shadow:0 6px 18px rgba(16,64,48,0.04);}
        section[data-testid="stSidebar"] [data-baseweb="tag"] {background:#eef7f3 !important; border:1px solid #d7ebe1 !important; color:#0f4a39 !important; border-radius:999px !important;}
        section[data-testid="stSidebar"] [data-testid="stExpander"] {border:1px solid #e5ece7; border-radius:14px; overflow:hidden;}
        section[data-testid="stSidebar"] .stTextInput > div > div {border-radius:14px;}
        section[data-testid="stSidebar"] .stSlider {padding-top:6px;}
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


def region_codes_from_labels(values: list[str]) -> list[str]:
    selected = set(values or [])
    return [code for code, label in REGION_LABELS.items() if label in selected or code in selected]


BRAND_CANONICAL_MAP = {
    "lg": "LG",
    "l.g": "LG",
    "\uc5d8\uc9c0": "LG",
    "samsung": "Samsung",
    "\uc0bc\uc131": "Samsung",
    "ge": "GE",
    "g.e": "GE",
    "\uc9c0\uc774": "GE",
    "general electric": "GE",
    "whirlpool": "Whirlpool",
    "\uc6d4\ud480": "Whirlpool",
}

CEJ_CANONICAL_MAP = {
    "aware": "\uc778\uc9c0",
    "explore": "\ud0d0\uc0c9",
    "decide": "\uacb0\uc815",
    "purchase": "\uad6c\ub9e4",
    "deliver": "\ubc30\uc1a1",
    "installation": "\ubc30\uc1a1",
    "install": "\ubc30\uc1a1",
    "on-board": "\uc0ac\uc6a9\uc900\ube44",
    "onboard": "\uc0ac\uc6a9\uc900\ube44",
    "use": "\uc0ac\uc6a9",
    "maintain": "\uad00\ub9ac",
    "replace": "\uad50\uccb4",
    "other": "\uae30\ud0c0",
    "\uc124\uce58": "\ubc30\uc1a1",
    "\uad00\ub9ac\uad50\uccb4": "\uad00\ub9ac",
}


def canonicalize_brand(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return "\ubbf8\uc5b8\uae09"
    lowered = raw.lower()
    if lowered in BRAND_CANONICAL_MAP:
        return BRAND_CANONICAL_MAP[lowered]
    return raw


def canonicalize_cej(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return "\uae30\ud0c0"
    lowered = raw.lower()
    if lowered in CEJ_CANONICAL_MAP:
        return CEJ_CANONICAL_MAP[lowered]
    return CEJ_CANONICAL_MAP.get(raw, raw if raw in CEJ_ORDER else "\uae30\ud0c0")


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


def _summarize_original_for_preview(text_value: str, limit: int = 220) -> str:
    value = _safe_text(text_value)
    if len(value) <= limit:
        return value
    compact = re.sub(r"\s+", " ", value).strip()
    return compact[: limit - 3].rstrip() + "..."


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


def _score_signal_detail(original_text: str, sentiment_name: str, signal_strength: str) -> str:
    text_value = _safe_text(original_text).lower()
    strong_markers = ["고장", "문제", "불량", "늦", "안됨", "worst", "broken", "awful", "terrible", "refund", "warranty", "service", "best", "recommend", "great", "love"]
    explicit_hits = sum(1 for token in strong_markers if token in text_value)
    strength_label = _safe_text(signal_strength).lower()
    if strength_label == "strong" or explicit_hits >= 2:
        return "감성 강도는 문제·만족을 단정적으로 표현하는 어휘와 명확한 평가 문장이 반복되어 강한 신호로 계산했습니다."
    if strength_label == "weak":
        return "감성 강도는 직접적인 평가 표현은 있으나 단정성은 상대적으로 약해 보조 지표로 반영했습니다."
    return "감성 강도는 평가 표현의 명확성과 문제·만족의 직접 언급 여부를 기준으로 산정했습니다."


def _score_richness_detail(original_text: str) -> str:
    text_value = _safe_text(original_text).lower()
    cause_markers = ["때문", "그래서", "because", "so that", "blocked", "causing"]
    time_markers = ["개월", "년", "주", "month", "year", "every", "again"]
    comparison_markers = ["보다", "old", "newer", "than", "better", "worse"]
    condition_markers = ["when", "if", "모드", "설정", "while", "after"]
    score = sum(token in text_value for token in cause_markers) + sum(token in text_value for token in time_markers) + sum(token in text_value for token in comparison_markers) + sum(token in text_value for token in condition_markers)
    if score >= 3:
        return "내용 구체성은 원인-결과, 기간·횟수, 비교나 사용 상황 같은 재현 가능한 맥락이 함께 있어 높게 보았습니다."
    if score >= 1:
        return "내용 구체성은 문제 상황 설명과 일부 맥락 정보가 포함되어 있어 중간 이상으로 반영했습니다."
    return "내용 구체성은 길이보다 실제 상황 설명과 재현 가능한 맥락 포함 여부를 중심으로 평가했습니다."


def _recent_trend_detail(similar_comments: pd.DataFrame) -> str:
    if similar_comments is None or similar_comments.empty or "작성일시" not in similar_comments.columns:
        return "최근 작성 흐름은 확보된 데이터 범위에서 안정적으로 유지되었습니다."
    dates = pd.to_datetime(similar_comments["작성일시"], errors="coerce").dropna()
    if dates.empty:
        return "최근 작성 흐름은 확보된 데이터 범위에서 안정적으로 유지되었습니다."
    latest = dates.max()
    recent = int((dates >= latest - pd.Timedelta(days=30)).sum())
    previous = int(((dates < latest - pd.Timedelta(days=30)) & (dates >= latest - pd.Timedelta(days=60))).sum())
    if recent > previous:
        return f"또한 최근 1개월 내 유사 댓글이 {recent}건으로 직전 1개월의 {previous}건보다 늘어 증가 추세도 확인됐습니다."
    if recent < previous:
        return f"최근 1개월 내 유사 댓글은 {recent}건으로 직전 1개월의 {previous}건보다 줄었지만, 누적 언급량은 여전히 충분했습니다."
    return f"최근 1개월 내 유사 댓글은 {recent}건으로 직전 1개월과 유사한 수준이어서 지속적으로 관찰되는 이슈로 판단했습니다."


def _build_selection_narrative(selected: pd.Series, source_comments: pd.DataFrame | None, sentiment_name: str) -> str:
    cluster_size = int(pd.to_numeric(selected.get("cluster_size", 1), errors="coerce") or 1)
    likes = int(pd.to_numeric(selected.get(COL_LIKES, 0), errors="coerce") or 0)
    original_text = _safe_text(selected.get(COL_RAW, selected.get(COL_ORIGINAL, "")))
    summary_text = _summarize_for_dashboard(selected, sentiment_name)
    similar_comments = _collect_similar_comments_for_download(source_comments if source_comments is not None else pd.DataFrame(), selected, sentiment_name)
    if similar_comments.empty:
        similar_comments = _collect_similar_comments_from_samples(selected)
    trend_text = _recent_trend_detail(similar_comments)
    strength_text = _score_signal_detail(original_text, sentiment_name, _safe_text(selected.get("signal_strength", "")))
    richness_text = _score_richness_detail(original_text)
    likes_text = f"좋아요는 {likes:,}개로 보조 가중치를 주되, 대표 선정은 어디까지나 감성 강도와 내용 구체성을 우선했습니다." if likes > 0 else "좋아요 수는 높지 않더라도 유사 언급량과 내용 구체성이 충분해 대표 사례로 유지했습니다."
    sentiment_label = "부정" if sentiment_name == "negative" else "긍정" if sentiment_name == "positive" else "중립"
    return f"{summary_text} 이 댓글은 {sentiment_label} 의견이 분명하고 같은 이슈의 유사 댓글이 {cluster_size}개 확인되어 대표 사례로 선정했습니다. {trend_text} {strength_text} {richness_text} {likes_text}"


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
    working[COL_CEJ] = topic_series.map(canonicalize_cej)
    working[COL_BRAND] = brand_series.map(canonicalize_brand)
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



def compute_filtered_bundle(comments_df: pd.DataFrame, videos_df: pd.DataFrame, quality_df: pd.DataFrame, filters: dict[str, Any], representative_bundles_df: pd.DataFrame | None = None, opinion_units_df: pd.DataFrame | None = None) -> dict[str, Any]:
    base_comments = comments_df.copy()
    if COL_COUNTRY not in base_comments.columns and "region" in base_comments.columns:
        base_comments[COL_COUNTRY] = base_comments["region"].map(localize_region)
    if COL_BRAND not in base_comments.columns:
        if "brand_mentioned" in base_comments.columns:
            base_comments[COL_BRAND] = base_comments["brand_mentioned"].map(canonicalize_brand)
        elif "brand" in base_comments.columns:
            base_comments[COL_BRAND] = base_comments["brand"].map(canonicalize_brand)
    if COL_CEJ not in base_comments.columns:
        if "customer_journey_stage" in base_comments.columns:
            base_comments[COL_CEJ] = base_comments["customer_journey_stage"].map(canonicalize_cej)
        elif "cej_scene_code" in base_comments.columns:
            base_comments[COL_CEJ] = base_comments["cej_scene_code"].map(canonicalize_cej)
    if COL_SENTIMENT not in base_comments.columns:
        if "sentiment_label" in base_comments.columns:
            base_comments[COL_SENTIMENT] = base_comments["sentiment_label"].map(localize_sentiment)
        elif "sentiment" in base_comments.columns:
            base_comments[COL_SENTIMENT] = base_comments["sentiment"].astype(str).str.lower().map(localize_sentiment)
    if "cleaned_text" not in base_comments.columns:
        base_comments["cleaned_text"] = base_comments.get("text_display", base_comments.get(COL_ORIGINAL, pd.Series("", index=base_comments.index))).fillna("")
    if "text_display" not in base_comments.columns:
        base_comments["text_display"] = base_comments.get(COL_ORIGINAL, pd.Series("", index=base_comments.index)).fillna("")

    products = filters.get("products") or []
    regions = filters.get("regions") or []
    sentiments = filters.get("sentiments") or []
    cej = filters.get("cej") or []
    brands = filters.get("brands") or []
    keyword_query = str(filters.get("keyword_query") or "").strip()

    if products:
        base_comments = base_comments[base_comments["product"].isin(products)]
    if regions and COL_COUNTRY in base_comments.columns:
        base_comments = base_comments[base_comments[COL_COUNTRY].isin(regions)]
    if cej and COL_CEJ in base_comments.columns:
        base_comments = base_comments[base_comments[COL_CEJ].isin(cej)]
    if brands and COL_BRAND in base_comments.columns:
        base_comments = base_comments[base_comments[COL_BRAND].isin(brands)]
    if keyword_query:
        base_comments = base_comments[
            base_comments["cleaned_text"].fillna("").str.contains(keyword_query, case=False, na=False)
            | base_comments["text_display"].fillna("").str.contains(keyword_query, case=False, na=False)
        ]

    filtered_comments = base_comments.copy()
    if sentiments and COL_SENTIMENT in filtered_comments.columns:
        filtered_comments = filtered_comments[filtered_comments[COL_SENTIMENT].isin(sentiments)]

    representative_comments = base_comments.copy()
    valid_series = representative_comments.get("comment_validity", pd.Series("valid", index=representative_comments.index))
    representative_comments = representative_comments[valid_series.eq("valid")].copy()

    representative_bundles = (representative_bundles_df.copy() if representative_bundles_df is not None else pd.DataFrame())
    if not representative_bundles.empty:
        if products and "product" in representative_bundles.columns:
            representative_bundles = representative_bundles[representative_bundles["product"].isin(products)]
        if regions and "region" in representative_bundles.columns:
            representative_bundles = representative_bundles[representative_bundles["region"].isin(region_codes_from_labels(regions))]
        if brands and "brand_mentioned" in representative_bundles.columns:
            representative_bundles["brand_mentioned"] = representative_bundles["brand_mentioned"].map(canonicalize_brand)
            representative_bundles = representative_bundles[representative_bundles["brand_mentioned"].isin(brands)]

    opinion_units = (opinion_units_df.copy() if opinion_units_df is not None else pd.DataFrame())
    if not opinion_units.empty:
        if products and "product" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["product"].isin(products)]
        if regions and "region" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["region"].isin(region_codes_from_labels(regions))]
        if cej and "cej_scene_code" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["cej_scene_code"].map(canonicalize_cej).isin(cej)]
        if sentiments and "sentiment" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["sentiment"].astype(str).str.lower().map(localize_sentiment).isin(sentiments)]
        if brands and "brand_mentioned" in opinion_units.columns:
            opinion_units["brand_mentioned"] = opinion_units["brand_mentioned"].map(canonicalize_brand)
            opinion_units = opinion_units[opinion_units["brand_mentioned"].isin(brands)]

    filtered_videos = videos_df.copy()
    if COL_COUNTRY not in filtered_videos.columns and "region" in filtered_videos.columns:
        filtered_videos[COL_COUNTRY] = filtered_videos["region"].map(localize_region)
    if products:
        filtered_videos = filtered_videos[filtered_videos["product"].isin(products)]
    if regions and COL_COUNTRY in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos[COL_COUNTRY].isin(regions)]
    if brands and COL_BRAND in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos[COL_BRAND].isin(brands)]

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
        "representative_bundles": representative_bundles,
        "opinion_units": opinion_units,
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


def _normalize_pill_state(key: str, options: list[str], defaults: list[str]) -> list[str]:
    valid_defaults = [item for item in defaults if item in options]
    if not valid_defaults:
        valid_defaults = list(options)
    current = st.session_state.get(key, valid_defaults)
    if not isinstance(current, list):
        current = list(valid_defaults)
    normalized = [item for item in current if item in options]
    if not normalized:
        normalized = list(valid_defaults)
    if st.session_state.get(key) != normalized:
        st.session_state[key] = normalized
    return normalized


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


def _representative_cluster_key(row: pd.Series) -> str:
    cluster_id = row.get("cluster_id")
    try:
        if pd.notna(cluster_id):
            return f"cluster::{cluster_id}"
    except Exception:
        pass
    return "|".join([
        _safe_text(row.get("sentiment_label", "")),
        _safe_text(row.get("product", "")),
        _safe_text(row.get(COL_CEJ, row.get("topic_label", ""))),
        _safe_text(row.get("topic_label", "")) or _safe_text(row.get("classification_type", "")) or "general",
    ])


def _build_bundle_showcase(filtered_comments: pd.DataFrame, sentiment: str, limit: int = 5) -> pd.DataFrame:
    candidate_pool = build_comment_showcase(filtered_comments, sentiment=sentiment, limit=120)
    if candidate_pool.empty:
        return candidate_pool

    base_pool = filtered_comments.copy()
    validity = base_pool.get("comment_validity", pd.Series("valid", index=base_pool.index))
    base_pool = base_pool[validity.eq("valid")].copy()
    if sentiment:
        base_pool = base_pool[base_pool.get("sentiment_label", pd.Series("", index=base_pool.index)).eq(sentiment)].copy()
    if base_pool.empty:
        base_pool = candidate_pool.copy()

    candidate_pool = candidate_pool.copy()
    candidate_pool["_cluster_key"] = candidate_pool.apply(_representative_cluster_key, axis=1)
    base_pool["_cluster_key"] = base_pool.apply(_representative_cluster_key, axis=1)

    picked = []
    seen = set()
    for _, row in candidate_pool.iterrows():
        cluster_key = row["_cluster_key"]
        if cluster_key in seen:
            continue
        seen.add(cluster_key)
        cluster_rows = base_pool[base_pool["_cluster_key"] == cluster_key].copy()
        cluster_size = max(1, len(cluster_rows))
        like_col = COL_LIKES if COL_LIKES in cluster_rows.columns else ("like_count" if "like_count" in cluster_rows.columns else None)
        if like_col and "text_length" in cluster_rows.columns:
            sample_rows = cluster_rows.sort_values([like_col, "text_length"], ascending=[False, False], na_position="last").head(5).copy()
        elif like_col:
            sample_rows = cluster_rows.sort_values([like_col], ascending=[False], na_position="last").head(5).copy()
        else:
            sample_rows = cluster_rows.head(5).copy()
        sample_comments = []
        for _, sample in sample_rows.iterrows():
            sample_original = _safe_text(sample.get(COL_ORIGINAL, sample.get("text_display", sample.get("display_text", ""))))
            sample_language = _safe_text(sample.get(COL_LANGUAGE, sample.get("language_detected", ""))).lower()
            sample_is_korean = sample_language.startswith("ko") or _looks_korean_text(sample_original)
            sample_translation = _resolve_card_translation(sample_original, _safe_text(sample.get(COL_TRANSLATION, sample.get("translated_text", ""))), sample_is_korean)
            sample_comments.append({
                "display_text": sample_original if sample_is_korean else sample_translation,
                "original_text": sample_original,
                "translated_text": "" if sample_is_korean else sample_translation,
                "likes_count": int(pd.to_numeric(sample.get(COL_LIKES, sample.get("like_count", sample.get("likes_count", 0))), errors="coerce") or 0),
                "pii_flag": _safe_text(sample.get("pii_flag", "N")) or "N",
            })
        bundle_row = row.copy()
        bundle_row["cluster_size"] = cluster_size
        bundle_row["sample_comments_json"] = json.dumps(sample_comments, ensure_ascii=False)
        likes_value = int(pd.to_numeric(bundle_row.get(COL_LIKES, bundle_row.get("like_count", bundle_row.get("top_likes_count", 0))), errors="coerce") or 0)
        likes_text = f", 좋아요 {likes_value:,}개" if likes_value > 0 else ""
        bundle_row["selection_reason_ui"] = f"유사 의견 {cluster_size}개가 같은 이슈를 반복 언급했고, 감성 강도와 내용 구체성{likes_text}을 기준으로 대표 코멘트로 선정했습니다."
        picked.append(bundle_row)
        if len(picked) >= limit:
            break

    if not picked:
        return pd.DataFrame(columns=candidate_pool.columns)
    return pd.DataFrame(picked)


def _rows_from_representative_bundles(bundle_df: pd.DataFrame, sentiment: str, limit: int = 5) -> pd.DataFrame:
    if bundle_df is None or bundle_df.empty:
        return pd.DataFrame()
    working = bundle_df.copy()
    if sentiment and "sentiment_code" in working.columns:
        working = working[working["sentiment_code"].eq(sentiment)].copy()
    if working.empty:
        return pd.DataFrame()
    if "rank" in working.columns:
        working = working.sort_values(["rank", "bundle_score"], ascending=[True, False], na_position="last")
    else:
        working = working.sort_values(["bundle_score", "cluster_size"], ascending=[False, False], na_position="last")

    rows: list[pd.Series] = []
    for _, row in working.head(limit).iterrows():
        converted = pd.Series(dtype=object)
        converted["topic_label"] = _safe_text(row.get("aspect_key", "")) or "대표 코멘트"
        converted["cluster_size"] = int(pd.to_numeric(row.get("cluster_size", 1), errors="coerce") or 1)
        converted[COL_LIKES] = int(pd.to_numeric(row.get("top_likes_count", 0), errors="coerce") or 0)
        converted["product"] = _safe_text(row.get("product", row.get("product_category", "-")))
        scene_raw = _safe_text(row.get("cej_scene_code", row.get(COL_CEJ, "기타")))
        scene_map = {
            "S1 Aware": "인지", "S2 Explore": "탐색", "S3 Decide": "결정", "S4 Purchase": "구매",
            "S5 Deliver": "배송", "S6 On-board": "사용준비", "S7 Use": "사용", "S8 Maintain": "관리",
            "S9 Replace": "교체", "S10 Other": "기타",
        }
        converted[COL_CEJ] = scene_map.get(scene_raw, canonicalize_cej(scene_raw))
        converted[COL_BRAND] = canonicalize_brand(_safe_text(row.get("brand_mentioned", "")))
        converted[COL_VIDEO_TITLE] = _safe_text(row.get(COL_VIDEO_TITLE, row.get("top_video_title", "관련 영상"))) or "관련 영상"
        converted[COL_VIDEO_LINK] = _safe_text(row.get("top_source_link", row.get(COL_VIDEO_LINK, "#"))) or "#"
        converted[COL_ORIGINAL] = _safe_text(row.get("top_display_text", row.get("top_original_text", "")))
        converted[COL_RAW] = _safe_text(row.get("top_original_text", converted[COL_ORIGINAL]))
        converted[COL_TRANSLATION] = _safe_text(row.get("top_translation_text", ""))
        converted[COL_LANGUAGE] = "ko" if _looks_korean_text(converted[COL_ORIGINAL]) else "en"
        selection_reason_raw = _safe_text(row.get("selection_reason", ""))
        if not selection_reason_raw or _looks_garbled(selection_reason_raw):
            selection_reason_raw = f"유사 의견 {converted['cluster_size']}개가 같은 이슈를 반복 언급했고, 감성 강도와 내용 구체성을 기준으로 대표 코멘트로 선정했습니다."
        converted["selection_reason_ui"] = selection_reason_raw
        converted["sample_comments_json"] = _safe_text(row.get("sample_comments_json", "[]")) or "[]"
        converted["pii_flag"] = _safe_text(row.get("top_pii_flag", "N")) or "N"
        converted["pii_types"] = _safe_text(row.get("top_pii_types", ""))
        converted["pii_raw_text"] = _safe_text(row.get("pii_raw_text", ""))
        converted["voc_items_json"] = _safe_text(row.get("voc_items_json", "[]"))
        converted["source_content_id"] = _safe_text(row.get("top_source_content_id", ""))
        converted["opinion_id"] = _safe_text(row.get("top_opinion_id", ""))
        rows.append(converted)
    return pd.DataFrame(rows)



def _enrich_card_row_from_source(selected: pd.Series, source_comments: pd.DataFrame | None) -> pd.Series:
    working = selected.copy()
    if source_comments is None or source_comments.empty:
        return working

    source = source_comments.copy()
    source_row = pd.DataFrame()
    opinion_id = _safe_text(working.get("opinion_id", ""))
    source_content_id = _safe_text(working.get("source_content_id", ""))
    video_link = _safe_text(working.get(COL_VIDEO_LINK, ""))

    if opinion_id and "opinion_id" in source.columns:
        source_row = source[source["opinion_id"].astype(str).eq(opinion_id)].head(1).copy()
    if source_row.empty and source_content_id:
        if "source_content_id" in source.columns:
            source_row = source[source["source_content_id"].astype(str).eq(source_content_id)].head(1).copy()
        elif "comment_id" in source.columns:
            source_row = source[source["comment_id"].astype(str).eq(source_content_id)].head(1).copy()
    if source_row.empty and video_link:
        if COL_VIDEO_LINK in source.columns:
            source_row = source[source[COL_VIDEO_LINK].astype(str).eq(video_link)].head(1).copy()
        elif "video_url" in source.columns:
            source_row = source[source["video_url"].astype(str).eq(video_link)].head(1).copy()

    sample_original = ""
    sample_translation = ""
    try:
        sample_items = json.loads(_safe_text(working.get("sample_comments_json", "[]")) or "[]")
        if sample_items:
            sample_original = _safe_text(sample_items[0].get("original_text", sample_items[0].get("display_text", "")))
            sample_translation = _safe_text(sample_items[0].get("translated_text", ""))
    except Exception:
        pass

    if not source_row.empty:
        row = source_row.iloc[0]
        if not _safe_text(working.get(COL_ORIGINAL, "")):
            working[COL_ORIGINAL] = _safe_text(row.get(COL_ORIGINAL, row.get("original_text", row.get("text_display", row.get("opinion_text", sample_original)))))
        if not _safe_text(working.get(COL_RAW, "")):
            working[COL_RAW] = _safe_text(row.get(COL_RAW, row.get("text_original", row.get(COL_ORIGINAL, working.get(COL_ORIGINAL, sample_original)))))
        if not _safe_text(working.get(COL_TRANSLATION, "")):
            working[COL_TRANSLATION] = _safe_text(row.get(COL_TRANSLATION, row.get("translated_text", sample_translation)))
        if not _safe_text(working.get(COL_VIDEO_TITLE, "")) or _safe_text(working.get(COL_VIDEO_TITLE, "")) == "?? ??":
            working[COL_VIDEO_TITLE] = _safe_text(row.get(COL_VIDEO_TITLE, row.get("title", "?? ??"))) or "?? ??"
        if not _safe_text(working.get(COL_VIDEO_LINK, "")) or _safe_text(working.get(COL_VIDEO_LINK, "")) == "#":
            working[COL_VIDEO_LINK] = _safe_text(row.get(COL_VIDEO_LINK, row.get("video_url", "#"))) or "#"
        if not _safe_text(working.get(COL_LANGUAGE, "")):
            working[COL_LANGUAGE] = _safe_text(row.get(COL_LANGUAGE, row.get("language_detected", "")))
        if int(pd.to_numeric(working.get(COL_LIKES, 0), errors="coerce") or 0) == 0:
            working[COL_LIKES] = int(pd.to_numeric(row.get(COL_LIKES, row.get("like_count", row.get("likes_count", 0))), errors="coerce") or 0)

    if not _safe_text(working.get(COL_ORIGINAL, "")) and sample_original:
        working[COL_ORIGINAL] = sample_original
    if not _safe_text(working.get(COL_RAW, "")):
        working[COL_RAW] = _safe_text(working.get(COL_ORIGINAL, sample_original))
    if not _safe_text(working.get(COL_TRANSLATION, "")) and sample_translation:
        working[COL_TRANSLATION] = sample_translation
    if not _safe_text(working.get(COL_VIDEO_TITLE, "")):
        working[COL_VIDEO_TITLE] = "?? ??"
    if not _safe_text(working.get(COL_VIDEO_LINK, "")):
        working[COL_VIDEO_LINK] = "#"

    return working



def _compute_trust_index(positive_count: int, negative_count: int) -> float:
    total = max(positive_count + negative_count, 1)
    return round(((positive_count - negative_count) / total) * 50 + 50, 1)


def build_brand_trust_frame(filtered_comments: pd.DataFrame) -> pd.DataFrame:
    if filtered_comments is None or filtered_comments.empty:
        return pd.DataFrame()
    frame = filtered_comments.copy()
    if COL_COUNTRY not in frame.columns and "region" in frame.columns:
        frame[COL_COUNTRY] = frame["region"].map(localize_region)
    if COL_BRAND not in frame.columns:
        if "brand_mentioned" in frame.columns:
            frame[COL_BRAND] = frame["brand_mentioned"].map(canonicalize_brand)
        elif "brand" in frame.columns:
            frame[COL_BRAND] = frame["brand"].map(canonicalize_brand)
    if COL_SENTIMENT not in frame.columns:
        if "sentiment_label" in frame.columns:
            frame[COL_SENTIMENT] = frame["sentiment_label"].map(localize_sentiment)
        elif "sentiment" in frame.columns:
            frame[COL_SENTIMENT] = frame["sentiment"].astype(str).str.lower().map(localize_sentiment)
    if COL_BRAND not in frame.columns or COL_COUNTRY not in frame.columns or "product" not in frame.columns:
        return pd.DataFrame()

    grouped = frame.groupby([COL_BRAND, "product", COL_COUNTRY], dropna=False)
    rows = []
    total_counts = frame.groupby(["product", COL_COUNTRY], dropna=False).size().to_dict()
    for (brand, product, country), group in grouped:
        brand = canonicalize_brand(_safe_text(brand)) or "미상"
        sentiments = group.get(COL_SENTIMENT, pd.Series("", index=group.index)).fillna("")
        pos = int((sentiments == localize_sentiment("positive")).sum())
        neg = int((sentiments == localize_sentiment("negative")).sum())
        cnt = len(group)
        denom = max(int(total_counts.get((product, country), 0)), 1)
        rows.append({
            COL_BRAND: brand,
            "product": product,
            COL_COUNTRY: country,
            "언급 댓글 수": cnt,
            "긍정 댓글 수": pos,
            "부정 댓글 수": neg,
            "언급 비율": cnt / denom,
            "고객 신뢰도 지수": _compute_trust_index(pos, neg),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values([COL_BRAND, "언급 댓글 수", "고객 신뢰도 지수"], ascending=[True, False, False], na_position="last")

def build_cej_trust_frame(filtered_comments: pd.DataFrame) -> pd.DataFrame:
    if filtered_comments is None or filtered_comments.empty:
        return pd.DataFrame()
    frame = filtered_comments.copy()
    if COL_COUNTRY not in frame.columns and "region" in frame.columns:
        frame[COL_COUNTRY] = frame["region"].map(localize_region)
    if COL_BRAND not in frame.columns:
        if "brand_mentioned" in frame.columns:
            frame[COL_BRAND] = frame["brand_mentioned"].map(canonicalize_brand)
        elif "brand" in frame.columns:
            frame[COL_BRAND] = frame["brand"].map(canonicalize_brand)
    if COL_CEJ not in frame.columns:
        if "customer_journey_stage" in frame.columns:
            frame[COL_CEJ] = frame["customer_journey_stage"].map(canonicalize_cej)
        elif "cej_scene_code" in frame.columns:
            frame[COL_CEJ] = frame["cej_scene_code"].map(canonicalize_cej)
    if COL_SENTIMENT not in frame.columns:
        if "sentiment_label" in frame.columns:
            frame[COL_SENTIMENT] = frame["sentiment_label"].map(localize_sentiment)
        elif "sentiment" in frame.columns:
            frame[COL_SENTIMENT] = frame["sentiment"].astype(str).str.lower().map(localize_sentiment)
    if COL_CEJ not in frame.columns or COL_COUNTRY not in frame.columns or "product" not in frame.columns:
        return pd.DataFrame()

    grouped = frame.groupby([COL_CEJ, "product", COL_COUNTRY, COL_BRAND], dropna=False)
    rows = []
    for (cej, product, country, brand), group in grouped:
        sentiments = group.get(COL_SENTIMENT, pd.Series("", index=group.index)).fillna("")
        pos = int((sentiments == localize_sentiment("positive")).sum())
        neg = int((sentiments == localize_sentiment("negative")).sum())
        rows.append({
            COL_CEJ: canonicalize_cej(_safe_text(cej)),
            "product": product,
            COL_COUNTRY: country,
            COL_BRAND: canonicalize_brand(_safe_text(brand)) or "미상",
            "댓글 수": len(group),
            "긍정 댓글 수": pos,
            "부정 댓글 수": neg,
            "고객 신뢰도 지수": _compute_trust_index(pos, neg),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result[COL_CEJ] = pd.Categorical(result[COL_CEJ], categories=CEJ_ORDER, ordered=True)
    return result.sort_values([COL_CEJ, "product", COL_COUNTRY, COL_BRAND], na_position="last")


def render_representative_comments(filtered_comments: pd.DataFrame, representative_bundles: pd.DataFrame | None = None, opinion_units: pd.DataFrame | None = None) -> None:
    st.subheader("실제 고객 코멘트 샘플")
    st.caption("감성 강도, 좋아요 수, 문장 길이를 함께 반영하되, 유사 이슈는 묶음으로 정리해 대표성을 보여줍니다.")

    source_for_cards = filtered_comments if filtered_comments is not None and not filtered_comments.empty else (opinion_units if opinion_units is not None else pd.DataFrame())
    negative_showcase = _rows_from_representative_bundles(representative_bundles, sentiment="negative", limit=5) if representative_bundles is not None and not representative_bundles.empty else pd.DataFrame()
    positive_showcase = _rows_from_representative_bundles(representative_bundles, sentiment="positive", limit=5) if representative_bundles is not None and not representative_bundles.empty else pd.DataFrame()

    if negative_showcase.empty:
        negative_showcase = _build_bundle_showcase(filtered_comments, sentiment="negative", limit=5)
    if positive_showcase.empty:
        positive_showcase = _build_bundle_showcase(filtered_comments, sentiment="positive", limit=5)

    if negative_showcase.empty and positive_showcase.empty:
        fallback_showcase = _build_bundle_showcase(filtered_comments, sentiment="", limit=5)
        st.warning("현재 조건에서 강한 긍정/부정 대표 코멘트가 부족해, 의미 있는 유효 댓글을 우선 보여줍니다.")
        render_comment_table(fallback_showcase, key_prefix="neutral", source_comments=source_for_cards)
        return

    st.write(f"부정 대표 코멘트 {len(negative_showcase)}건")
    render_comment_table(negative_showcase, key_prefix="negative", source_comments=source_for_cards)

    st.write(f"긍정 대표 코멘트 {len(positive_showcase)}건")
    render_comment_table(positive_showcase, key_prefix="positive", source_comments=source_for_cards)


def _collect_similar_comments_for_download(source_comments: pd.DataFrame, selected: pd.Series, sentiment_name: str) -> pd.DataFrame:
    if source_comments is None or source_comments.empty:
        return pd.DataFrame()
    working = source_comments.copy()

    if sentiment_name in {"negative", "positive", "neutral"}:
        if "sentiment_code" in working.columns:
            working = working[working["sentiment_code"].astype(str).eq(sentiment_name)].copy()
        else:
            sentiment_series = working.get("sentiment_label", pd.Series("", index=working.index))
            working = working[sentiment_series.astype(str).eq(sentiment_name)].copy()

    topic_value = _safe_text(selected.get("topic_label", "")) or _safe_text(selected.get("aspect_key", ""))
    if topic_value:
        if "aspect_key" in working.columns:
            working = working[working["aspect_key"].astype(str).eq(topic_value)].copy()
        elif "topic_label" in working.columns:
            working = working[working["topic_label"].astype(str).eq(topic_value)].copy()

    product_value = _safe_text(selected.get("product", ""))
    if product_value and "product" in working.columns:
        working = working[working["product"].astype(str).eq(product_value)].copy()

    cej_value = _safe_text(selected.get(COL_CEJ, ""))
    if cej_value:
        scene_map = {
            "인지": "S1 Aware",
            "탐색": "S2 Explore",
            "결정": "S3 Decide",
            "구매": "S4 Purchase",
            "배송": "S5 Deliver",
            "사용준비": "S6 On-board",
            "사용": "S7 Use",
            "관리": "S8 Maintain",
            "교체": "S9 Replace",
            "기타": "S10 Other",
        }
        if "cej_scene_code" in working.columns:
            working = working[working["cej_scene_code"].astype(str).eq(scene_map.get(cej_value, cej_value))].copy()
        elif COL_CEJ in working.columns:
            working = working[working[COL_CEJ].astype(str).eq(cej_value)].copy()

    brand_value = _safe_text(selected.get(COL_BRAND, ""))
    if brand_value:
        if "brand_mentioned" in working.columns:
            working = working[working["brand_mentioned"].map(canonicalize_brand).astype(str).eq(brand_value)].copy()
        elif COL_BRAND in working.columns:
            working = working[working[COL_BRAND].astype(str).eq(brand_value)].copy()

    if working.empty:
        return working

    like_col = COL_LIKES if COL_LIKES in working.columns else ("likes_count" if "likes_count" in working.columns else "like_count")
    if like_col not in working.columns:
        like_col = None
    sort_cols = [c for c in [like_col, "text_length", COL_WRITTEN_AT, "published_at"] if c and c in working.columns]
    sort_working = working.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last") if sort_cols else working.copy()

    display_rows = []
    seen = set()
    for _, sample in sort_working.iterrows():
        sample_original = _safe_text(sample.get(COL_ORIGINAL, sample.get("original_text", sample.get("display_text", sample.get("text_display", sample.get("opinion_text", ""))))))
        unique_id = _safe_text(sample.get("comment_id", sample.get("opinion_id", ""))) or sample_original
        if unique_id in seen or not sample_original:
            continue
        seen.add(unique_id)
        sample_language = _safe_text(sample.get(COL_LANGUAGE, sample.get("language_detected", ""))).lower()
        sample_is_korean = sample_language.startswith("ko") or _looks_korean_text(sample_original)
        sample_translation = _resolve_card_translation(sample_original, _safe_text(sample.get(COL_TRANSLATION, sample.get("translated_text", ""))), sample_is_korean)
        written_at = _safe_text(sample.get(COL_WRITTEN_AT, sample.get("published_at", "")))
        display_rows.append({
            "comment_id": unique_id,
            "원문": sample_original,
            "번역(참고)": "" if sample_is_korean else sample_translation,
            "좋아요 수": int(pd.to_numeric(sample.get(COL_LIKES, sample.get("likes_count", sample.get("like_count", 0))), errors="coerce") or 0),
            "PII": "있음" if _safe_text(sample.get("pii_flag", "N")) == "Y" else "없음",
            "작성일시": written_at,
        })
    return pd.DataFrame(display_rows)


def _collect_similar_comments_from_samples(selected: pd.Series) -> pd.DataFrame:
    try:
        sample_items = json.loads(_safe_text(selected.get("sample_comments_json", "[]")) or "[]")
    except Exception:
        sample_items = []

    rows = []
    seen = set()

    for idx, item in enumerate(sample_items):
        original_text = _safe_text(item.get("original_text", item.get("display_text", "")))
        translated_text = _safe_text(item.get("translated_text", ""))
        unique_id = _safe_text(item.get("opinion_id", item.get("comment_id", ""))) or original_text or f"sample_{idx}"

        if not original_text or unique_id in seen:
            continue
        seen.add(unique_id)

        rows.append({
            "comment_id": unique_id,
            "원문": original_text,
            "번역(참고)": translated_text,
            "좋아요 수": int(pd.to_numeric(item.get("likes_count", 0), errors="coerce") or 0),
            "PII": "있음" if _safe_text(item.get("pii_flag", "N")) == "Y" else "없음",
            "작성일시": _safe_text(item.get("written_at", "")),
        })

    return pd.DataFrame(rows)

def _localized_video_title(title: str) -> str:
    clean = _safe_text(title)
    if not clean:
        return "영상 제목 미상"
    if _looks_korean_text(clean):
        return clean
    translated = _safe_text(translate_to_korean(clean))
    if translated and not _looks_garbled(translated) and translated != clean:
        return translated
    return clean


def render_comment_table(comment_showcase: pd.DataFrame, key_prefix: str, source_comments: pd.DataFrame | None = None) -> None:
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
            aspect_label = _safe_text(selected.get("topic_label", "")) or _safe_text(selected.get("aspect_key", "")) or f"대표 코멘트 {idx + 1}"
            st.markdown(f"#### {idx + 1}. {aspect_label}")
            likes_value = pd.to_numeric(selected.get(COL_LIKES, 0), errors="coerce")
            likes = 0 if pd.isna(likes_value) else int(likes_value)
            cluster_size = int(pd.to_numeric(selected.get("cluster_size", 1), errors="coerce") or 1)
            st.caption(
                f"유사 의견 {cluster_size}개 · 제품군: {_safe_text(selected.get('product', '-')) or '-'} · {COL_CEJ}: {_safe_text(selected.get(COL_CEJ, '-')) or '-'} · {COL_BRAND}: {_safe_text(selected.get(COL_BRAND, '-')) or '-'} · 좋아요 수: {likes:,}"
            )

            video_title_raw = _safe_text(selected.get(COL_VIDEO_TITLE, '-')) or '-'
            video_title = _localized_video_title(video_title_raw)
            video_link = _safe_text(selected.get(COL_VIDEO_LINK, '#')) or '#'
            st.markdown(f"[{video_title}]({video_link})")
            if video_title != video_title_raw and video_title_raw not in {'', '-'}:
                st.caption(f"원제: {video_title_raw}")

            working_selected = _enrich_card_row_from_source(selected.copy(), source_comments)
            original_text = _safe_text(working_selected.get(COL_ORIGINAL, ""))
            raw_translation = _safe_text(working_selected.get(COL_TRANSLATION, ""))
            language = _safe_text(working_selected.get(COL_LANGUAGE, "")).lower()
            is_korean = language.startswith("ko") or _looks_korean_text(original_text)
            translation = _resolve_card_translation(original_text, raw_translation, is_korean)
            working_selected[COL_TRANSLATION] = translation

            inquiry_flag = bool(working_selected.get("inquiry_flag", False))
            inquiry_types = _safe_text(working_selected.get("inquiry_type_json", ""))
            if inquiry_flag:
                st.caption(f"문의 flag 적용 · 유형: {inquiry_types or '확인 필요'}")

            st.markdown("**선정 이유 및 AI 해석**")
            selection_reason = _build_selection_narrative(working_selected, source_comments, sentiment_name)
            st.markdown(f'<div class="voc-reason">{selection_reason}</div>', unsafe_allow_html=True)

            voc_items = _extract_voc_items(working_selected, sentiment_name)
            if voc_items:
                st.markdown("**추출 키워드/VoC(참고)**")
                for item in voc_items:
                    evidence = _safe_text(item.get("evidence_span", "")) or "근거 문구 확인 필요"
                    st.markdown(f"- {item['sentiment']} ? **{item['topic']}**: {item['insight']} (근거: {evidence})")

            primary_label = "원문(Original)" if is_korean else "번역(참고)"
            primary_text = original_text if is_korean else (translation or "한국어 번역을 준비 중입니다.")
            st.markdown(f"**{primary_label}**")
            st.write(_summarize_original_for_preview(primary_text))

            if _is_long_text(primary_text):
                with st.expander("원문 전체 보기" if is_korean else "번역 전체 보기"):
                    st.write(primary_text)

            if not is_korean:
                with st.expander("원문(Original) 보기"):
                    st.write(original_text)

            similar_comments = _collect_similar_comments_for_download(
                source_comments if source_comments is not None else pd.DataFrame(),
                selected,
                sentiment_name,
            )
            if similar_comments.empty:
                similar_comments = _collect_similar_comments_from_samples(selected)
            if not similar_comments.empty:
                with st.expander("유사 댓글 샘플 보기"):
                    preview_rows = similar_comments.head(5)

                    for _, sample in preview_rows.iterrows():
                        st.markdown(f"- {_safe_text(sample.get('원문', ''))}")

                        sample_translation = _safe_text(sample.get('번역(참고)', ''))
                        if sample_translation:
                            st.caption(f"번역(참고): {sample_translation}")

                        meta_bits = [
                            f"좋아요 {_safe_text(sample.get('좋아요 수', 0))}",
                            f"PII {_safe_text(sample.get('PII', '없음'))}",
                        ]
                        written_at = _safe_text(sample.get('작성일시', ''))
                        if written_at:
                            meta_bits.append(str(written_at))

                        st.caption(" · ".join(meta_bits))

                    csv_bytes = similar_comments.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="유사 댓글 CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"similar_comments_{key_prefix}_{idx + 1}.csv",
                        mime="text/csv",
                        key=f"similar_download_{key_prefix}_{idx}",
                        use_container_width=True,
                    )


def build_video_summary(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame) -> pd.DataFrame:
    videos = filtered_videos.copy()
    comments = filtered_comments.copy()
    if videos.empty and not comments.empty:
        rebuild_cols = ["product", "region", "title", "video_url", "published_at", "view_count", "like_count", "comment_count"]
        available = [c for c in rebuild_cols if c in comments.columns]
        if available:
            videos = comments.groupby(["video_id"], dropna=False).agg({c: "first" for c in available}).reset_index()
    if videos.empty:
        return pd.DataFrame()

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
    st.caption("댓글 분석에 활용된 영상을 영상 단위로 요약하고, 필요할 때만 상세 분석을 확인할 수 있습니다.")
    summary = build_video_summary(filtered_comments, filtered_videos)
    st.caption(f"영상 요약 계산 결과: 댓글 {len(filtered_comments):,}건 / 영상 {len(filtered_videos):,}건 / 요약 {len(summary):,}건")
    if summary.empty:
        st.info("표시할 영상 요약 데이터가 없습니다.")
        return

    k1, k2, k3 = st.columns(3)
    with k1:
        render_card("분석 영상 수", f"{len(summary):,}")
    with k2:
        render_card("총 분석 댓글", f"{int(summary['분석 댓글 수'].sum()):,}")
    with k3:
        top_video = _localized_video_title(_safe_text(summary.iloc[0].get(COL_VIDEO_TITLE, '-'))) if not summary.empty else '-'
        render_card("최상위 이슈 영상", str(top_video)[:28] or "-")

    summary_display = summary.copy()
    if COL_VIDEO_TITLE in summary_display.columns:
        summary_display[COL_VIDEO_TITLE] = summary_display[COL_VIDEO_TITLE].map(_localized_video_title)
    summary_display["View Analysis"] = "상세 분석 보기"
    display_cols = [
        "product", COL_COUNTRY, COL_VIDEO_TITLE, "발행일", "view_count", "분석 댓글 수", "유효 댓글 수",
        "긍정 댓글 수", "부정 댓글 수", "중립 댓글 수", "제외 댓글 수", "부정 비율", "주요 부정 포인트", "View Analysis",
    ]
    existing_cols = [col for col in display_cols if col in summary_display.columns]
    st.dataframe(summary_display[existing_cols].head(100), use_container_width=True, hide_index=True)
    if len(summary_display) > 100:
        st.caption(f"전체 {len(summary_display):,}개 영상 중 상위 100개만 먼저 표시합니다.")

    with st.container(border=True):
        st.markdown("#### 영상당 부정 댓글 밀도")
        if not density_df.empty:
            density_display = pd.DataFrame({
                "product": density_df.get("product", pd.Series("", index=density_df.index)),
                COL_COUNTRY: density_df.get(COL_COUNTRY, density_df.get("region", pd.Series("", index=density_df.index)).map(localize_region)),
                COL_VIDEO_TITLE: density_df.get("title", density_df.get(COL_VIDEO_TITLE, pd.Series("", index=density_df.index))).map(_localized_video_title),
                "부정 댓글 수": density_df.get("negative_comments", pd.Series(0, index=density_df.index)),
                "분석 댓글 수": density_df.get("total_comments", pd.Series(0, index=density_df.index)),
                "부정 밀도": density_df.get("negative_density", pd.Series(0.0, index=density_df.index)),
                COL_VIDEO_LINK: density_df.get("video_url", density_df.get(COL_VIDEO_LINK, pd.Series("", index=density_df.index))),
            })
            st.dataframe(density_display.head(100), use_container_width=True, hide_index=True)
            if len(density_display) > 100:
                st.caption(f"전체 {len(density_display):,}개 중 상위 100개만 먼저 표시합니다.")
        else:
            st.info("표시할 영상 밀도 데이터가 없습니다.")

    options = summary[["video_id", COL_VIDEO_TITLE, "product", COL_COUNTRY]].drop_duplicates().to_dict("records")
    select_options = [{"video_id": "", COL_VIDEO_TITLE: "선택 안 함", "product": "-", COL_COUNTRY: "-"}] + options
    selected = st.selectbox(
        "상세 분석할 영상 선택",
        options=select_options,
        format_func=lambda row: "상세 분석할 영상을 선택하세요" if not row.get("video_id") else f"[{row.get('product', '-')}/{row.get(COL_COUNTRY, '-')}] {_localized_video_title(_safe_text(row.get(COL_VIDEO_TITLE, '-')))}",
        key="video_analysis_selector",
    )
    if selected and selected.get("video_id"):
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
        neg = _build_bundle_showcase(video_comments, sentiment="negative", limit=5)
        pos = _build_bundle_showcase(video_comments, sentiment="positive", limit=5)
        st.markdown(f"##### 부정 대표 코멘트 {len(neg)}건")
        render_comment_table(neg, key_prefix="negative", source_comments=video_comments)
        st.markdown(f"##### 긍정 대표 코멘트 {len(pos)}건")
        render_comment_table(pos, key_prefix="positive", source_comments=video_comments)

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

def get_mode() -> str:
    """URL 파라미터로 모드 선택: ?mode=lite / ?mode=full"""
    try:
        mode = st.query_params.get("mode", "full")  # Streamlit query params 
    except Exception:
        mode = "full"
    return str(mode).lower()


def render_comments_lite(comments_df: pd.DataFrame) -> None:
    """댓글은 보여주되, 화면 렌더링/정렬 부담을 줄인 샘플/페이지네이션 뷰"""
    st.subheader("댓글 (Lite: 샘플/페이지네이션)")

    if comments_df.empty:
        st.info("댓글 데이터가 없습니다.")
        return

    # 한 번에 그리는 양을 줄여서(피크 감소) 꺼짐/healthz 문제 완화
    page_size = st.selectbox("페이지 크기", [50, 100, 200, 500], index=1)
    max_rows = st.selectbox("최대 로딩 행 수(경량)", [1000, 3000, 5000, 10000], index=1)

    view_df = comments_df

    # 정렬 컬럼이 있으면 최신/좋아요 기준으로 상위만
    sort_cols = [c for c in ["작성일시", "좋아요 수"] if c in view_df.columns]
    if len(view_df) > max_rows:
        if sort_cols:
            view_df = view_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        view_df = view_df.head(max_rows).copy()
        st.caption(f"전체 {len(comments_df):,}건 중 상위 {len(view_df):,}건만 경량 로딩했습니다.")

    # 보여줄 컬럼만 최소로 (있으면 보여주고 없으면 자동 제외)
    prefer_cols = [
        "국가", "감성", "고객여정단계", "브랜드",
        "작성일시", "좋아요 수",
        "영상 제목", "영상 링크",
        "원문 댓글", "한국어 번역",
        "제거 사유",
    ]
    show_cols = [c for c in prefer_cols if c in view_df.columns]

    total = len(view_df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("페이지", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    st.caption(f"{start+1:,}~{end:,} / {total:,} (page {page}/{total_pages})")
    st.dataframe(view_df.iloc[start:end][show_cols] if show_cols else view_df.iloc[start:end],
                 use_container_width=True, hide_index=True)
def load_dashboard_data_lite_comments() -> dict[str, pd.DataFrame]:
    """
    Lite(댓글 포함):
    - 기존 load_dashboard_data()를 그대로 재사용 (파일명/컬럼명 불일치로 비는 문제 방지)
    - 댓글은 표시하되, 너무 큰 경우 상위 N건만 잡아서 렌더/메모리 피크를 낮춤
    """
    data = load_dashboard_data()  # ✅ 이미 검증된 full 로더 재사용

    comments_df = data.get("comments", pd.DataFrame())
    if not comments_df.empty:
        # ✅ Lite에서는 너무 많이 잡지 않게 상한(필요시 3000~10000 조절)
        comments_df = comments_df.head(1000).copy()
    data["comments"] = comments_df

    return data


@st.cache_resource
def get_dashboard_data_resource_lite_comments():
    return load_dashboard_data_lite_comments()
def main() -> None:
    st.set_page_config(page_title="가전 VoC Dashboard", layout="wide")
    apply_theme()

    
    # 기존코드 data = load_dashboard_data()
    #회사 부팅 최적화용 추가
    mode = get_mode()  # ✅ 먼저 모드 결정 (?mode=lite / ?mode=full)

    with st.spinner("데이터 로딩 중…"):
        # ✅ Lite면 Lite용 경량 로더(댓글 포함)로 로딩
        if mode == "lite":
            data = get_dashboard_data_resource_lite_comments()
        else:
            data = get_dashboard_data_resource()

    comments_df = add_localized_columns(data.get("comments", pd.DataFrame()))
    if comments_df.empty:
        st.warning("표시할 분석 결과가 없습니다. 먼저 데이터를 수집해주세요.")
        return

    # =====================================================
    # ✅ Lite / Full 분기 (기존 full은 그대로 유지)
    # =====================================================
    if mode == "lite":
        st.markdown("## 가전 VoC Dashboard (Lite)")
        st.caption("댓글은 포함하되, 샘플/페이지네이션 방식으로 빠르게 표시합니다.")
        render_comments_lite(comments_df)

        st.info("전체 분석(차트/키워드/대표코멘트/영상분석)은 mode=full 로 접속하세요.")
        st.markdown("- 전체 모드: `?mode=full`")
        st.markdown("- Lite 모드: `?mode=lite`")
        return

    # =====================================================
    # ✅ 여기부터는 기존 Full 모드 (당신 기존 코드 그대로)
    # =====================================================

    videos_df = data["videos"].copy()
    videos_df[COL_COUNTRY] = videos_df.get("region", pd.Series(dtype=str)).map(localize_region)
    products = PRODUCT_ORDER
    comment_products = comments_df.get("product", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    video_products = videos_df.get("product", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()
    available_products = [item for item in PRODUCT_ORDER if item in comment_products or item in video_products]
    if not available_products:
        available_products = [item for item in PRODUCT_ORDER if item in comment_products]
    if not available_products:
        available_products = PRIMARY_PRODUCT_FILTERS[:4]
    regions = [localize_region("KR"), localize_region("US")]
    sentiments = [localize_sentiment("positive"), localize_sentiment("negative"), localize_sentiment("neutral"), localize_sentiment("excluded")]
    cej_labels = CEJ_ORDER
    brands = BRAND_FILTER_OPTIONS

    st.markdown("## 가전 VoC Dashboard")
    st.caption("주간 감성 변화, 핵심 키워드, CEJ 단계별 문제, 대표 코멘트를 제품·국가별로 빠르게 확인합니다.")
    st.caption("Build: 2026-03-27 08:15 / representative-video-restore-pass-2")
    default_products = available_products
    default_regions = regions
    default_brands = [item for item in brands if item in comments_df.get(COL_BRAND, pd.Series(dtype=str)).dropna().astype(str).unique().tolist()] or brands
    default_sentiments = sentiments
    default_cej = [item for item in cej_labels if item in comments_df.get(COL_CEJ, pd.Series(dtype=str)).dropna().astype(str).unique().tolist()] or cej_labels

    filter_product_defaults = _normalize_pill_state("filter_products", products, default_products)
    filter_region_defaults = _normalize_pill_state("filter_regions", regions, default_regions)
    filter_brand_defaults = _normalize_pill_state("filter_brands", brands, default_brands)
    filter_sentiment_defaults = _normalize_pill_state("filter_sentiments", sentiments, default_sentiments)
    filter_cej_defaults = _normalize_pill_state("filter_cej", cej_labels, default_cej)

with st.sidebar:
    st.markdown("### 탐색 필터")

    with st.container(border=True):
        st.caption("제품 범위")
        st.caption(f"현재 분석 데이터가 있는 제품군: {', '.join(available_products)}")
        selected_products = st.pills(
            "제품군",
            products,
            selection_mode="multi",
            key="filter_products",
            label_visibility="collapsed",
            width="content",
        ) or filter_product_defaults

    with st.container(border=True):
        st.caption("시장")
        selected_regions = st.pills(
            "국가",
            regions,
            selection_mode="multi",
            key="filter_regions",
            label_visibility="collapsed",
            width="content",
        ) or filter_region_defaults

    with st.container(border=True):
        st.caption("브랜드")
        selected_brands = st.pills(
            "브랜드",
            brands,
            selection_mode="multi",
            key="filter_brands",
            label_visibility="collapsed",
            width="content",
        ) or filter_brand_defaults

    with st.container(border=True):
        st.caption("감성 라벨")
        selected_sentiments = st.pills(
            "감성",
            sentiments,
            selection_mode="multi",
            key="filter_sentiments",
            label_visibility="collapsed",
            width="content",
        ) or filter_sentiment_defaults

    with st.container(border=True):
        st.caption("고객경험여정")
        selected_cej = st.pills(
            "고객경험여정 단계",
            cej_labels,
            selection_mode="multi",
            key="filter_cej",
            label_visibility="collapsed",
            width="content",
        ) or filter_cej_defaults

    with st.container(border=True):
        st.caption("고급 옵션")
        keyword_query = st.text_input(
            "댓글 내 키워드 검색",
            "",
            placeholder="예: 소음, 냉각, warranty",
        )
    weeks_per_view = st.slider(
        "차트 한 화면 주차 수",
        min_value=4,
        max_value=24,
        value=12,
        step=2,
    )

    
bundle = compute_filtered_bundle(
    comments_df,
    videos_df,
    data["quality_summary"],
    {"products": selected_products, "regions": selected_regions, "sentiments": selected_sentiments, "cej": selected_cej, "brands": selected_brands, "keyword_query": keyword_query},
     data.get("representative_bundles", pd.DataFrame()),
     data.get("opinion_units", pd.DataFrame()),
 )

filtered_comments = bundle["comments"]
filtered_videos = bundle["videos"]
if filtered_comments.empty and not keyword_query:
      fallback_filters = {"products": available_products, "regions": regions, "sentiments": sentiments, "cej": cej_labels, "brands": brands, "keyword_query": ""}
      bundle = compute_filtered_bundle(
          comments_df,
          videos_df,
          data["quality_summary"],
          fallback_filters,
          data.get("representative_bundles", pd.DataFrame()),
          data.get("opinion_units", pd.DataFrame()),
      )
      filtered_comments = bundle["comments"]
      filtered_videos = bundle["videos"]
all_filtered_weeks = sorted(filtered_comments[COL_WEEK_START].dropna().unique())
total_pages = max(1, (len(all_filtered_weeks) + weeks_per_view - 1) // weeks_per_view) if all_filtered_weeks else 1


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
        if selected_brands and COL_BRAND in brand_df.columns:
            brand_df[COL_BRAND] = brand_df[COL_BRAND].map(canonicalize_brand)
            brand_df = brand_df[brand_df[COL_BRAND].isin(selected_brands)]

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

    monitoring_df = data.get("monitoring_summary", pd.DataFrame()).copy()
    reporting_df = data.get("reporting_summary", pd.DataFrame()).copy()
    if not monitoring_df.empty:
        if selected_products and "product" in monitoring_df.columns:
            monitoring_df = monitoring_df[monitoring_df["product"].isin(selected_products)]
        if selected_regions and "region" in monitoring_df.columns:
            region_codes = region_codes_from_labels(selected_regions)
            monitoring_df = monitoring_df[monitoring_df["region"].isin(region_codes)]
    if not reporting_df.empty:
        if selected_products and "product" in reporting_df.columns:
            reporting_df = reporting_df[reporting_df["product"].isin(selected_products)]
        if selected_regions and "region" in reporting_df.columns:
            region_codes = region_codes_from_labels(selected_regions)
            reporting_df = reporting_df[reporting_df["region"].isin(region_codes)]
    if not filtered_comments.empty:
        if "comment_id" in filtered_comments.columns:
            comment_ids = filtered_comments["comment_id"].astype("string").fillna("")
            direct_source_count = float(comment_ids.replace("", pd.NA).dropna().nunique())
        else:
            direct_source_count = float(len(filtered_comments))
    else:
        direct_source_count = 0.0
    direct_opinion_count = float(len(filtered_comments))
    direct_neutral_count = float(filtered_comments.get("sentiment_label", pd.Series("", index=filtered_comments.index)).eq("neutral").sum()) if not filtered_comments.empty else 0.0
    direct_exclude_count = float(filtered_comments.get("comment_validity", pd.Series("valid", index=filtered_comments.index)).eq("excluded").sum()) if not filtered_comments.empty else 0.0
    direct_valid = filtered_comments[filtered_comments.get("comment_validity", pd.Series("valid", index=filtered_comments.index)).eq("valid")].copy() if not filtered_comments.empty else pd.DataFrame()
    direct_positive_count = float(direct_valid.get("sentiment_label", pd.Series("", index=direct_valid.index)).eq("positive").sum()) if not direct_valid.empty else 0.0
    direct_negative_count = float(direct_valid.get("sentiment_label", pd.Series("", index=direct_valid.index)).eq("negative").sum()) if not direct_valid.empty else 0.0
    direct_kpi_count = direct_positive_count + direct_negative_count

    monitoring_row = pd.Series({
        "source_count": direct_source_count,
        "opinion_count": direct_opinion_count,
        "neutral_opinion_count": direct_neutral_count,
        "exclude_opinion_count": direct_exclude_count,
        "split_source_count": max(0.0, direct_opinion_count - direct_source_count),
    })
    source_count_val = direct_source_count
    opinion_count_val = direct_opinion_count
    split_sources = max(0.0, direct_opinion_count - direct_source_count)
    monitoring_row["split_rate"] = (split_sources / source_count_val) if source_count_val else 0.0
    monitoring_row["neutral_rate"] = (direct_neutral_count / opinion_count_val) if opinion_count_val else 0.0
    monitoring_row["exclude_rate"] = (direct_exclude_count / opinion_count_val) if opinion_count_val else 0.0

    reporting_row = pd.Series({
        "kpi_opinion_count": direct_kpi_count,
        "positive_count": direct_positive_count,
        "negative_count": direct_negative_count,
    })
    kpi_count_val = direct_kpi_count
    reporting_row["positive_share"] = (direct_positive_count / kpi_count_val) if kpi_count_val else 0.0
    reporting_row["negative_share"] = (direct_negative_count / kpi_count_val) if kpi_count_val else 0.0

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
            render_card("\uae0d\uc815 \ube44\uc728", f"{positive_ratio:.1f}%")
        with row2_col2:
            render_card("\ubd80\uc815 \ube44\uc728", f"{negative_ratio:.1f}%")
        with row2_col3:
            render_card("\uc911\ub9bd \ube44\uc728", f"{neutral_ratio:.1f}%")

        row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns(5)
        with row3_col1:
            render_card("\uc6d0\ucc9c \ucf58\ud150\uce20 \uc218", f"{int(monitoring_row.get('source_count', 0) or 0):,}")
        with row3_col2:
            render_card("\uc758\uacac \ub2e8\uc704 \uc218", f"{int(monitoring_row.get('opinion_count', 0) or 0):,}")
        with row3_col3:
            render_card("\ubd84\ub9ac \ube44\uc728", f"{float(monitoring_row.get('split_rate', 0) or 0):.1%}")
        with row3_col4:
            render_card("\uc911\ub9bd \ube44\uc728", f"{float(monitoring_row.get('neutral_rate', 0) or 0):.1%}")
        with row3_col5:
            render_card("\uc81c\uc678 \ube44\uc728", f"{float(monitoring_row.get('exclude_rate', 0) or 0):.1%}")

        row4_col1, row4_col2, row4_col3 = st.columns(3)
        with row4_col1:
            render_card("KPI \uc758\uacac \uc218", f"{int(reporting_row.get('kpi_opinion_count', 0) or 0):,}")
        with row4_col2:
            render_card("KPI \uae0d\uc815 \ube44\uc728", f"{float(reporting_row.get('positive_share', 0) or 0):.1%}")
        with row4_col3:
            render_card("KPI \ubd80\uc815 \ube44\uc728", f"{float(reporting_row.get('negative_share', 0) or 0):.1%}")

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
            st.markdown("### CEJ 단계별 고객 신뢰도")
            cej_trust_df = build_cej_trust_frame(filtered_comments)
            if not cej_trust_df.empty:
                st.caption("각 고객경험여정 단계에서 제품·국가·브랜드별 긍정/부정 반응을 바탕으로 고객 신뢰도 지수를 계산했습니다.")
                st.dataframe(
                    cej_trust_df[[COL_CEJ, "product", COL_COUNTRY, COL_BRAND, "댓글 수", "긍정 댓글 수", "부정 댓글 수", "고객 신뢰도 지수"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("현재 조건에서 표시할 CEJ 신뢰도 데이터가 없습니다.")

        with st.container(border=True):
            st.markdown("### 브랜드 언급 비중")
            brand_trust_df = build_brand_trust_frame(filtered_comments)
            if not brand_trust_df.empty:
                st.caption("브랜드가 언급된 댓글의 양과 긍정/부정 분포를 함께 보며 고객 신뢰도 지수를 참고할 수 있습니다.")
                st.dataframe(
                    brand_trust_df[[COL_BRAND, "product", COL_COUNTRY, "언급 댓글 수", "긍정 댓글 수", "부정 댓글 수", "언급 비율", "고객 신뢰도 지수"]],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("현재 조건에서 표시할 브랜드 신뢰도 데이터가 없습니다.")

        with st.container(border=True):
            render_representative_comments(bundle.get("representative_comments", filtered_comments), bundle.get("representative_bundles", pd.DataFrame()), bundle.get("opinion_units", pd.DataFrame()))

    with tab_videos:
        render_video_summary_page(filtered_comments, filtered_videos, density_df)


if __name__ == "__main__":
    main()





