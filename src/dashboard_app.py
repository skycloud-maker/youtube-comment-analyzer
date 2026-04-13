"""Streamlit dashboard for appliance VoC exploration."""

from __future__ import annotations

from io import BytesIO
import codecs
import hashlib
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path(__file__).resolve().parents[1] / "data" / "mplcache").resolve()))

try:
    import koreanize_matplotlib  # type: ignore
except Exception:
    koreanize_matplotlib = None

import altair as alt
import matplotlib.font_manager as fm
import pandas as pd
import streamlit as st

from src.analytics.keywords import BUSINESS_KEEPWORDS, build_keyword_counter, filter_business_keywords, normalize_keyword, tokenize_text
from src.config import load_yaml_settings_optional
from src.utils.translation import translate_text, translate_to_korean

BASE_DIR = Path(__file__).resolve().parents[1]
APP_SETTINGS = load_yaml_settings_optional()
DASHBOARD_SETTINGS = APP_SETTINGS.dashboard
VIDEO_SELECTION_SETTINGS = APP_SETTINGS.collection.video_selection


def _resolve_repo_path(path: Path) -> Path:
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def _get_font_file() -> Path | None:
    return next((path for path in FONT_CANDIDATES if path.exists()), None)


def _get_font_prop() -> fm.FontProperties | None:
    font_file = _get_font_file()
    return fm.FontProperties(fname=str(font_file)) if font_file else None


PROCESSED_DIR = _resolve_repo_path(APP_SETTINGS.storage.processed_dir)
CACHE_DIR = BASE_DIR / "data" / "dashboard_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "dashboard_bundle.pkl"
CACHE_META = CACHE_DIR / "dashboard_bundle_meta.json"
CLOUD_CACHE_FILE = CACHE_DIR / "dashboard_bundle_cloud.pkl"
CLOUD_CACHE_META = CACHE_DIR / "dashboard_bundle_cloud_meta.json"
CACHE_SCHEMA_VERSION = DASHBOARD_SETTINGS.cache_schema_version
DASHBOARD_BUILD_LABEL = DASHBOARD_SETTINGS.build_label
REGION_LABELS = {"KR": "\uD55C\uAD6D", "US": "\uBBF8\uAD6D"}
REGION_OTHER_LABEL = "그 외"
SENTIMENT_LABELS = {
    "positive": "\uAE0D\uC815",
    "negative": "\uBD80\uC815",
    "neutral": "\uC911\uB9BD",
    "excluded": "\uC81C\uC678",
}
CEJ_ORDER = [
    "\uC778\uC9C0",
    "\uD0D0\uC0C9",
    "\uACB0\uC815",
    "\uAD6C\uB9E4",
    "\uBC30\uC1A1",
    "\uC0AC\uC6A9\uC900\uBE44",
    "\uC0AC\uC6A9",
    "\uAD00\uB9AC",
    "\uAD50\uCCB4",
    "\uAE30\uD0C0",
]
PRIMARY_PRODUCT_FILTERS = list(DASHBOARD_SETTINGS.primary_product_filters)
EXTRA_PRODUCT_FILTERS = list(DASHBOARD_SETTINGS.extra_product_filters)
PRODUCT_ORDER = PRIMARY_PRODUCT_FILTERS + EXTRA_PRODUCT_FILTERS
BRAND_FILTER_OPTIONS = list(DASHBOARD_SETTINGS.brand_filter_options)
REGION_CODES = list(DASHBOARD_SETTINGS.supported_regions)
FONT_CANDIDATES = tuple(_resolve_repo_path(path) for path in DASHBOARD_SETTINGS.mpl_font_candidates)
LITE_COMMENTS_LIMIT = DASHBOARD_SETTINGS.lite_comments_limit
WEEKLY_CHART_MIN = DASHBOARD_SETTINGS.weekly_chart_page_min
WEEKLY_CHART_MAX = DASHBOARD_SETTINGS.weekly_chart_page_max
WEEKLY_CHART_DEFAULT = DASHBOARD_SETTINGS.weekly_chart_page_default
WEEKLY_CHART_STEP = DASHBOARD_SETTINGS.weekly_chart_page_step
LG_TARGET_TERMS = [str(item).lower() for item in VIDEO_SELECTION_SETTINGS.target_brand_terms if str(item).strip()]
LG_COMPETITOR_TERMS = [str(item).lower() for item in VIDEO_SELECTION_SETTINGS.competitor_brand_terms if str(item).strip()]
LG_PRODUCT_TERMS = [str(item).lower() for item in VIDEO_SELECTION_SETTINGS.product_terms if str(item).strip()]
LG_PREFERRED_CHANNEL_TERMS = [str(item).lower() for item in VIDEO_SELECTION_SETTINGS.preferred_channel_keywords if str(item).strip()]
LG_BLOCKED_CHANNEL_TERMS = [str(item).lower() for item in VIDEO_SELECTION_SETTINGS.blocked_channel_keywords if str(item).strip()]
DASHBOARD_DATASETS: tuple[tuple[str, str], ...] = (
    ("comments", "comments.parquet"),
    ("videos", "videos_normalized.parquet"),
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
    ("nlp_topic_insights", "nlp_topic_insights.parquet"),
    ("nlp_keyword_insights", "nlp_keyword_insights.parquet"),
    ("nlp_inquiry_summary", "nlp_inquiry_summary.parquet"),
    ("nlp_product_mentions", "nlp_product_mentions.parquet"),
    ("nlp_sentiment_distribution", "nlp_sentiment_distribution.parquet"),
    ("monitoring_summary", "monitoring_summary.parquet"),
    ("reporting_summary", "reporting_summary.parquet"),
)

COL_COUNTRY = "\uAD6D\uAC00"
COL_SENTIMENT = "\uAC10\uC131"
COL_CEJ = "\uACE0\uAC1D\uC5EC\uC815\uB2E8\uACC4"
COL_BRAND = "\uBE0C\uB79C\uB4DC"
COL_WRITTEN_AT = "\uC791\uC131\uC77C\uC2DC"
COL_WEEK_START = "\uC8FC\uCC28\uC2DC\uC791"
COL_WEEK_LABEL = "\uC8FC\uCC28\uD45C\uAE30"
COL_ORIGINAL = "\uC6D0\uBB38 \uB313\uAE00"
COL_TRANSLATION = "\uD55C\uAD6D\uC5B4 \uBC88\uC5ED"
COL_VIDEO_TITLE = "\uC601\uC0C1 \uC81C\uBAA9"
COL_VIDEO_LINK = "\uC601\uC0C1 \uB9C1\uD06C"
COL_LIKES = "\uC88B\uC544\uC694 \uC218"
COL_LANGUAGE = "\uC5B8\uC5B4"
COL_RAW = "\uC6D0\uBCF8 \uB313\uAE00"
COL_REMOVED = "\uC81C\uAC70 \uC0AC\uC720"
COL_CONTEXT = "\uC5F0\uAD00 \uB9E5\uB77D"
COL_CONTEXT_TRANSLATION = "\uC5F0\uAD00 \uB9E5\uB77D \uBC88\uC5ED"
COL_SENTIMENT_CODE = "sentiment_code"
SESSION_DATA_BUNDLE_KEY = "dashboard_data_bundle"
SESSION_DATA_MODE_KEY = "dashboard_data_mode"
SESSION_DATA_SIGNATURE_KEY = "dashboard_data_signature"
SESSION_LAST_RUN_RESULT_KEY = "dashboard_last_run_result"
ANALYSIS_ALL_MARKET_CODE = "ALL"
ANALYSIS_MARKET_OPTIONS: tuple[tuple[str, str], ...] = (
    ("전체(20개국)", ANALYSIS_ALL_MARKET_CODE),
    ("한국", "KR"),
    ("미국", "US"),
    ("캐나다", "CA"),
    ("영국", "GB"),
    ("프랑스", "FR"),
    ("이탈리아", "IT"),
    ("독일", "DE"),
    ("사우디아라비아", "SA"),
    ("이집트", "EG"),
    ("인도", "IN"),
    ("인도네시아", "ID"),
    ("러시아", "RU"),
    ("베트남", "VN"),
    ("대만", "TW"),
    ("태국", "TH"),
    ("호주", "AU"),
    ("브라질", "BR"),
    ("콜롬비아", "CO"),
    ("멕시코", "MX"),
    ("페루", "PE"),
)
ANALYSIS_LANGUAGE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("한국어", "ko"),
    ("영어", "en"),
    ("기타 언어(언어 제한 없음)", "all"),
)
REGION_PRIMARY_LANGUAGE_HINTS: dict[str, tuple[str, ...]] = {
    "KR": ("ko",),
    "US": ("en",),
    "CA": ("en", "fr"),
    "GB": ("en",),
    "FR": ("fr",),
    "IT": ("it",),
    "DE": ("de",),
    "SA": ("ar",),
    "EG": ("ar",),
    "IN": ("hi", "en"),
    "ID": ("id",),
    "RU": ("ru",),
    "VN": ("vi",),
    "TW": ("zh-TW",),
    "TH": ("th",),
    "AU": ("en",),
    "BR": ("pt",),
    "CO": ("es",),
    "MX": ("es",),
    "PE": ("es",),
}
LANGUAGE_NO_FILTER_MARKERS = {"all", "any", "other", "*"}
MAX_QUERY_VARIANTS = 5

def _build_empty_buckets() -> dict[str, list[pd.DataFrame]]:
    return {key: [] for key, _ in DASHBOARD_DATASETS}


def _iter_dashboard_run_dirs() -> list[Path]:
    if not PROCESSED_DIR.exists():
        return []
    return [path for path in sorted(PROCESSED_DIR.iterdir()) if path.is_dir()]


def _iter_dashboard_source_files() -> list[Path]:
    files: list[Path] = []
    for run_dir in _iter_dashboard_run_dirs():
        for _, filename in DASHBOARD_DATASETS:
            parquet_path = run_dir / filename
            csv_path = parquet_path.with_suffix(".csv")
            if parquet_path.exists():
                files.append(parquet_path)
            elif csv_path.exists():
                files.append(csv_path)
    return files


def _latest_signature() -> str:
    digest = hashlib.sha256()
    source_files = _iter_dashboard_source_files()
    if not source_files:
        digest.update(b"empty")
        return digest.hexdigest()

    for path in source_files:
        stat = path.stat()
        digest.update(path.relative_to(PROCESSED_DIR).as_posix().encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def _read_cache_bundle(path: Path, expected_signature: str | None = None) -> dict[str, pd.DataFrame] | None:
    meta_path = CACHE_META if path == CACHE_FILE else CLOUD_CACHE_META
    if expected_signature is not None:
        if not meta_path.exists():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if meta.get("signature") != expected_signature or meta.get("version") != CACHE_SCHEMA_VERSION:
            return None

    if not path.exists():
        return None
    try:
        return pd.read_pickle(path)
    except Exception:
        return None


def _write_cache_bundle(data: dict[str, pd.DataFrame], signature: str) -> None:
    pd.to_pickle(data, CACHE_FILE)
    CACHE_META.write_text(
        json.dumps({"signature": signature, "version": CACHE_SCHEMA_VERSION}, ensure_ascii=False),
        encoding="utf-8",
    )


@st.cache_data(show_spinner=False)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    latest_signature = _latest_signature()

    # Only trust caches that match the latest source signature.
    # Falling back to unsigned local cache can keep stale metrics after new runs.
    for cache_path in (CACHE_FILE, CLOUD_CACHE_FILE):
        cached = _read_cache_bundle(cache_path, latest_signature)
        if cached is not None:
            if cache_path == CLOUD_CACHE_FILE:
                _write_cache_bundle(cached, latest_signature)
            return cached

    buckets = _build_empty_buckets()
    for run_dir in _iter_dashboard_run_dirs():
        context = {"run_id": run_dir.name}
        for key, filename in DASHBOARD_DATASETS:
            frame = _read_frame(run_dir / filename)
            if frame.empty:
                continue
            for column, value in context.items():
                if column not in frame.columns:
                    frame[column] = value
            buckets[key].append(frame)

    data = {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in buckets.items()
    }
    _write_cache_bundle(data, latest_signature)
    return data


@st.cache_resource
def get_dashboard_data_resource():
    return load_dashboard_data()


REQUIRED_RESULT_FILES = ("videos_normalized.parquet", "comments.parquet")
SAMPLE_MAX_RUNS = 6
SAMPLE_MAX_ROWS = {
    "comments": 3000,
    "analysis_comments": 3000,
    "opinion_units": 4500,
    "videos": 400,
}


def _empty_dashboard_bundle() -> dict[str, pd.DataFrame]:
    return {key: pd.DataFrame() for key, _ in DASHBOARD_DATASETS}


def _run_dir_has_required_outputs(run_dir: Path) -> bool:
    return all((run_dir / filename).exists() or (run_dir / Path(filename).with_suffix(".csv")).exists() for filename in REQUIRED_RESULT_FILES)


def _real_ready_run_dirs() -> list[Path]:
    return [run_dir for run_dir in _iter_dashboard_run_dirs() if _run_dir_has_required_outputs(run_dir)]


def _build_missing_data_message() -> str:
    run_dirs = _iter_dashboard_run_dirs()
    required_text = ", ".join(REQUIRED_RESULT_FILES)
    if not run_dirs:
        return f"`{PROCESSED_DIR}` 아래에 run 폴더가 없습니다. 먼저 분석을 실행해 `{required_text}` 파일을 생성해 주세요."

    missing_rows: list[str] = []
    for run_dir in run_dirs[-5:]:
        missing = [name for name in REQUIRED_RESULT_FILES if not ((run_dir / name).exists() or (run_dir / Path(name).with_suffix('.csv')).exists())]
        if missing:
            missing_rows.append(f"- {run_dir.name}: {', '.join(missing)} 누락")
    if missing_rows:
        return "결과 파일이 불완전합니다.\n" + "\n".join(missing_rows)
    return f"결과 파일을 찾지 못했습니다. `{required_text}` 파일 존재 여부를 확인해 주세요."


def _build_synthetic_sample_bundle() -> dict[str, pd.DataFrame]:
    comments = pd.DataFrame(
        [
            {
                "run_id": "sample_run_001",
                "comment_id": "c1",
                "video_id": "v1",
                "product": "세탁기",
                "region": "KR",
                "title": "세탁기 비교 리뷰",
                "channel_title": "샘플 리뷰 채널",
                "video_url": "https://www.youtube.com/watch?v=sample_v1",
                "text_display": "세탁 소음이 커서 밤에는 사용이 어렵습니다.",
                "cleaned_text": "세탁 소음이 커서 밤에는 사용이 어렵습니다.",
                "like_count": 34,
                "comment_validity": "valid",
                "sentiment_label": "negative",
                "signal_strength": "strong",
                "classification_type": "complaint",
                "confidence_level": "high",
                "classification_reason": "소음 불만이 명확함",
                "brand_label": "LG",
                "nlp_is_inquiry": False,
                "is_lg_relevant_video": True,
                "video_lg_relevance_score": 0.83,
            },
            {
                "run_id": "sample_run_001",
                "comment_id": "c2",
                "video_id": "v1",
                "product": "세탁기",
                "region": "KR",
                "title": "세탁기 비교 리뷰",
                "channel_title": "샘플 리뷰 채널",
                "video_url": "https://www.youtube.com/watch?v=sample_v1",
                "text_display": "이 모델 에너지 효율은 어떤가요?",
                "cleaned_text": "이 모델 에너지 효율은 어떤가요?",
                "like_count": 12,
                "comment_validity": "valid",
                "sentiment_label": "neutral",
                "signal_strength": "weak",
                "classification_type": "informational",
                "confidence_level": "medium",
                "classification_reason": "질문형 댓글",
                "brand_label": "미언급",
                "nlp_is_inquiry": True,
                "is_lg_relevant_video": True,
                "video_lg_relevance_score": 0.83,
            },
            {
                "run_id": "sample_run_001",
                "comment_id": "c3",
                "video_id": "v2",
                "product": "냉장고",
                "region": "US",
                "title": "Best refrigerator review",
                "channel_title": "Sample Appliance Lab",
                "video_url": "https://www.youtube.com/watch?v=sample_v2",
                "text_display": "Cooling performance is great and design is clean.",
                "cleaned_text": "Cooling performance is great and design is clean.",
                "like_count": 22,
                "comment_validity": "valid",
                "sentiment_label": "positive",
                "signal_strength": "strong",
                "classification_type": "preference",
                "confidence_level": "high",
                "classification_reason": "성능 만족",
                "brand_label": "LG",
                "nlp_is_inquiry": False,
                "is_lg_relevant_video": False,
                "video_lg_relevance_score": 0.42,
            },
        ]
    )
    videos = pd.DataFrame(
        [
            {
                "run_id": "sample_run_001",
                "video_id": "v1",
                "product": "세탁기",
                "region": "KR",
                "title": "세탁기 비교 리뷰",
                "channel_title": "샘플 리뷰 채널",
                "video_url": "https://www.youtube.com/watch?v=sample_v1",
                "comment_count": 120,
                "is_lg_relevant_video": True,
                "video_lg_relevance_score": 0.83,
            },
            {
                "run_id": "sample_run_001",
                "video_id": "v2",
                "product": "냉장고",
                "region": "US",
                "title": "Best refrigerator review",
                "channel_title": "Sample Appliance Lab",
                "video_url": "https://www.youtube.com/watch?v=sample_v2",
                "comment_count": 98,
                "is_lg_relevant_video": False,
                "video_lg_relevance_score": 0.42,
            },
        ]
    )
    opinion_units = pd.DataFrame(
        [
            {"run_id": "sample_run_001", "source_content_id": "c1", "inquiry_flag": False, "product": "세탁기", "region": "KR", "brand_mentioned": "LG", "cej_scene_code": "사용", "sentiment": "negative", "opinion_text": "세탁 소음이 큼"},
            {"run_id": "sample_run_001", "source_content_id": "c2", "inquiry_flag": True, "product": "세탁기", "region": "KR", "brand_mentioned": "미언급", "cej_scene_code": "탐색", "sentiment": "neutral", "opinion_text": "에너지 효율 질문"},
            {"run_id": "sample_run_001", "source_content_id": "c3", "inquiry_flag": False, "product": "냉장고", "region": "US", "brand_mentioned": "LG", "cej_scene_code": "사용", "sentiment": "positive", "opinion_text": "냉각 성능 만족"},
        ]
    )
    quality_summary = pd.DataFrame(
        [
            {"comment_validity": "valid", "count": 3},
            {"comment_validity": "excluded", "count": 0},
        ]
    )
    nlp_inquiry_summary = comments[comments["nlp_is_inquiry"] == True][["comment_id", "cleaned_text"]].rename(columns={"cleaned_text": "text"})

    bundle = _empty_dashboard_bundle()
    bundle["comments"] = comments
    bundle["analysis_comments"] = comments.copy()
    bundle["videos"] = videos
    bundle["quality_summary"] = quality_summary
    bundle["opinion_units"] = opinion_units
    bundle["nlp_inquiry_summary"] = nlp_inquiry_summary
    return bundle


@st.cache_data(show_spinner=False)
def load_sample_dashboard_data() -> dict[str, pd.DataFrame]:
    run_dirs = list(reversed(_real_ready_run_dirs()))[:SAMPLE_MAX_RUNS]
    if not run_dirs:
        return _build_synthetic_sample_bundle()

    buckets = _build_empty_buckets()
    for run_dir in run_dirs:
        context = {"run_id": run_dir.name}
        for key, filename in DASHBOARD_DATASETS:
            frame = _read_frame(run_dir / filename)
            if frame.empty:
                continue
            for column, value in context.items():
                if column not in frame.columns:
                    frame[column] = value
            row_cap = SAMPLE_MAX_ROWS.get(key)
            if row_cap is not None:
                frame = frame.head(row_cap)
            buckets[key].append(frame)

    data = {
        key: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for key, frames in buckets.items()
    }
    if data.get("comments", pd.DataFrame()).empty or data.get("videos", pd.DataFrame()).empty:
        return _build_synthetic_sample_bundle()
    return data


def _clear_dashboard_caches() -> None:
    load_dashboard_data.clear()
    get_dashboard_data_resource.clear()
    load_sample_dashboard_data.clear()


def _expected_mode_signature(mode: str) -> str:
    source_sig = _latest_signature()
    return f"{mode}:{source_sig}"


def _load_bundle_for_mode(mode: str, *, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    if force_refresh:
        _clear_dashboard_caches()
    if mode == "sample":
        return load_sample_dashboard_data()
    return get_dashboard_data_resource()


def _initialize_data_session_state() -> None:
    requested_mode = ""
    try:
        requested_mode = str(st.query_params.get("data_mode", st.query_params.get("mode", ""))).strip().lower()
    except Exception:
        requested_mode = ""
    default_mode = "real" if _real_ready_run_dirs() else "sample"
    if os.getenv("PYTEST_CURRENT_TEST"):
        default_mode = "sample"
    if requested_mode in {"sample", "lite"}:
        default_mode = "sample"
    elif requested_mode == "real":
        default_mode = "real"

    if SESSION_DATA_MODE_KEY not in st.session_state:
        st.session_state[SESSION_DATA_MODE_KEY] = default_mode
    if SESSION_DATA_BUNDLE_KEY not in st.session_state:
        mode = st.session_state[SESSION_DATA_MODE_KEY]
        bundle = _load_bundle_for_mode(mode, force_refresh=False)
        st.session_state[SESSION_DATA_BUNDLE_KEY] = bundle
        st.session_state[SESSION_DATA_SIGNATURE_KEY] = _expected_mode_signature(mode)


def _set_data_mode(mode: str, *, force_refresh: bool = False) -> None:
    bundle = _load_bundle_for_mode(mode, force_refresh=force_refresh)
    st.session_state[SESSION_DATA_MODE_KEY] = mode
    st.session_state[SESSION_DATA_BUNDLE_KEY] = bundle
    st.session_state[SESSION_DATA_SIGNATURE_KEY] = _expected_mode_signature(mode)


def _analysis_target_regions(region: str) -> list[str]:
    selected = str(region or "").strip().upper()
    all_regions = [code for _, code in ANALYSIS_MARKET_OPTIONS if code != ANALYSIS_ALL_MARKET_CODE]
    if selected == ANALYSIS_ALL_MARKET_CODE:
        return all_regions
    if selected in all_regions:
        return [selected]
    return ["KR"]


def _run_real_analysis_action(keyword: str, max_videos: int, comments_per_video: int, language: str, region: str, output_prefix: str) -> tuple[bool, str]:
    target_regions = _analysis_target_regions(region)
    run_logs: list[str] = []

    for target_region in target_regions:
        keyword_variants = _build_search_keyword_variants(keyword, language=language, region=target_region)
        primary_keyword = keyword_variants[0] if keyword_variants else keyword
        region_prefix = output_prefix if len(target_regions) == 1 else f"{output_prefix}_{target_region.lower()}"
        command = [
            sys.executable,
            "-m",
            "src.main",
            "run",
            "--keyword",
            primary_keyword,
            "--max-videos",
            str(max_videos),
            "--comments-per-video",
            str(comments_per_video),
            "--region",
            target_region,
            "--output-prefix",
            region_prefix,
        ]
        if len(keyword_variants) > 1:
            command.extend(["--keywords", *keyword_variants[1:]])
        if language and language.lower() not in LANGUAGE_NO_FILTER_MARKERS:
            command.extend(["--language", language])
        completed = subprocess.run(command, cwd=str(BASE_DIR), capture_output=True, text=True)
        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "").strip()
            return False, f"[{target_region}] {error_text[-1200:]}"
        stdout_tail = (completed.stdout or "").strip()[-200:]
        run_logs.append(f"[{target_region}] 완료" + (f" | {stdout_tail}" if stdout_tail else ""))

    if len(target_regions) > 1:
        return True, f"전체 국가 실행 완료 ({len(target_regions)}개)\n" + "\n".join(run_logs)
    return True, (run_logs[0] if run_logs else "실행 완료")


def _normalize_query_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _normalize_language_code(code: str) -> str:
    raw = str(code or "").strip()
    lowered = raw.lower()
    if lowered in {"zh_tw", "zh-tw", "zh_hant"}:
        return "zh-TW"
    if lowered in {"pt_br", "pt-br"}:
        return "pt"
    if lowered in {"en-us", "en-gb"}:
        return "en"
    if lowered in {"ko-kr"}:
        return "ko"
    if lowered in {"ar-sa"}:
        return "ar"
    return raw


def _build_search_keyword_variants(keyword: str, *, language: str, region: str) -> list[str]:
    seed = _normalize_query_text(keyword)
    if not seed:
        return []

    target_languages: list[str] = []
    chosen_language = str(language or "").strip().lower()
    region_code = str(region or "").strip().upper()

    if chosen_language and chosen_language not in LANGUAGE_NO_FILTER_MARKERS:
        target_languages.append(_normalize_language_code(chosen_language))
    else:
        target_languages.extend(_normalize_language_code(code) for code in REGION_PRIMARY_LANGUAGE_HINTS.get(region_code, ()))
        target_languages.extend(["ko", "en"])

    deduped_targets: list[str] = []
    seen_targets: set[str] = set()
    for code in target_languages:
        normalized = _normalize_language_code(code).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen_targets:
            continue
        seen_targets.add(key)
        deduped_targets.append(normalized)

    variants: list[str] = [seed]
    seen_variants: set[str] = {seed.casefold()}
    for target in deduped_targets:
        translated = _normalize_query_text(translate_text(seed, target))
        if not translated:
            continue
        key = translated.casefold()
        if key in seen_variants:
            continue
        seen_variants.add(key)
        variants.append(translated)
        if len(variants) >= MAX_QUERY_VARIANTS:
            break
    return variants


@st.cache_data(show_spinner=False)
def build_keyword_summary(
    texts: tuple[str, ...],
    top_n: int = 6,
    version: str = "v5",
    blocked_keywords: tuple[str, ...] = (),
) -> pd.DataFrame:
    counter = build_keyword_counter(texts)
    rows = [{"keyword": keyword, "count": count} for keyword, count in filter_business_keywords(counter, top_n * 6)]
    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return keyword_df
    keyword_df["keyword"] = keyword_df["keyword"].map(normalize_keyword)
    keyword_df = keyword_df[keyword_df["keyword"] != ""]
    blocked = {item for item in blocked_keywords if item}
    if blocked:
        keyword_df = keyword_df[~keyword_df["keyword"].isin(blocked)]
    keyword_df = keyword_df.groupby("keyword", as_index=False)["count"].sum().sort_values("count", ascending=False).head(top_n)
    return keyword_df


def _build_dynamic_keyword_excludes(comments_df: pd.DataFrame) -> tuple[str, ...]:
    if comments_df.empty:
        return ()

    blocked: set[str] = set()
    for col in ["channel_title", "author_display_name"]:
        if col not in comments_df.columns:
            continue
        for text in comments_df[col].dropna().astype(str).head(500).tolist():
            for raw_token in re.findall(r"[0-9A-Za-z가-힣]+", text):
                token = normalize_keyword(raw_token)
                if not token:
                    continue
                if token in BUSINESS_KEEPWORDS:
                    continue
                blocked.add(token)
    return tuple(sorted(blocked))


@st.cache_data(show_spinner=False)
def build_wordcloud_image(frequencies: tuple[tuple[str, int], ...], sentiment_code: str) -> bytes | None:
    if not frequencies:
        return None
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    font_file = _get_font_file()
    if font_file is None:
        return None

    wc = WordCloud(
        width=1000,
        height=420,
        background_color="#ffffff",
        colormap="Blues" if sentiment_code == "positive" else "Oranges",
        font_path=str(font_file),
        prefer_horizontal=0.9,
        max_words=60,
        collocations=False,
    ).generate_from_frequencies(dict(frequencies))

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
        "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e40af"
    ] if sentiment_code == "positive" else [
        "#fdba74", "#fb923c", "#f97316", "#ea580c", "#c2410c", "#9a3412"
    ]
    colors = colors[:len(labels)]
    font_prop = _get_font_prop()

    fig, ax = plt.subplots(figsize=(8.8, 4.8), dpi=180)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    display_labels = [f"{label}\n{share}" for label, share in zip(labels, shares)]
    ax.pie(
        counts,
        labels=display_labels,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.38, "edgecolor": "white", "linewidth": 2},
        labeldistance=0.78,
        textprops={
            "fontsize": 10,
            "color": "#334155",
            "ha": "center",
            "va": "center",
            "fontproperties": font_prop,
        },
    )

    centre_circle = plt.Circle((0, 0), 0.42, fc="white")
    ax.add_artist(centre_circle)
    ax.text(
        0,
        0.02,
        center_label,
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#1e293b",
        fontproperties=font_prop,
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

    # Use pipeline outputs as the only comment-level analysis source.
    # Dashboard must not re-run a separate NLP/rule path.
    working["display_sentiment"] = (
        working.get("sentiment_label", pd.Series("neutral", index=working.index))
        .fillna("neutral")
        .astype(str)
        .str.lower()
    )
    working["display_signal_strength"] = working.get("signal_strength", pd.Series("", index=working.index)).fillna("").astype(str)
    working["display_classification_type"] = working.get("classification_type", pd.Series("", index=working.index)).fillna("").astype(str)
    working["display_confidence_level"] = working.get("confidence_level", pd.Series("", index=working.index)).fillna("").astype(str)
    working["display_context_used"] = working.get("context_used", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    working["display_context_required"] = working.get("context_required_for_display", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    working["display_product_target"] = working.get("product_target", pd.Series("", index=working.index)).fillna("").astype(str)
    working["display_reason"] = (
        working.get("classification_reason", pd.Series("", index=working.index))
        .fillna("")
        .astype(str)
    )
    working["display_needs_review"] = working.get("needs_review", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    if "insight_type" not in working.columns:
        working["insight_type"] = ""

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
        /* ===== Foundation ===== */
        :root {
            --bg-main: #f4f6f9;
            --bg-card: #ffffff;
            --bg-sidebar: #1a2332;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --accent-primary: #3b82f6;
            --accent-positive: #10b981;
            --accent-negative: #f97316;
            --accent-neutral: #6366f1;
            --border: #e2e8f0;
            --shadow-sm: 0 1px 3px rgba(0,0,0,0.04);
            --shadow-md: 0 4px 16px rgba(0,0,0,0.06);
            --shadow-lg: 0 8px 30px rgba(0,0,0,0.08);
            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 20px;
        }

        /* ===== App Shell ===== */
        .stApp {background: var(--bg-main); color: var(--text-primary); font-family: 'Pretendard', 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;}
        header[data-testid="stHeader"] {background: rgba(244,246,249,0.85); backdrop-filter: blur(12px); border-bottom: 1px solid var(--border);}
        .block-container {padding-top: 4.5rem; padding-bottom: 2rem; max-width: 1320px;}

        /* ===== Sidebar ===== */
        section[data-testid="stSidebar"] {background: var(--bg-sidebar); border-right: none;}
        section[data-testid="stSidebar"] * {color: #cbd5e1 !important;}
        section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {color: #f1f5f9 !important;}
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: var(--radius-md); padding: 10px 12px;}
        section[data-testid="stSidebar"] [data-baseweb="tag"] {background: rgba(59,130,246,0.15) !important; border: 1px solid rgba(59,130,246,0.3) !important; color: #93c5fd !important; border-radius: 999px !important;}
        section[data-testid="stSidebar"] [data-testid="stExpander"] {border: 1px solid rgba(255,255,255,0.1); border-radius: var(--radius-sm); overflow: hidden;}
        section[data-testid="stSidebar"] .stTextInput > div > div {border-radius: var(--radius-sm); background: rgba(255,255,255,0.06); border-color: rgba(255,255,255,0.15);}
        section[data-testid="stSidebar"] .stSlider {padding-top: 6px;}
        section[data-testid="stSidebar"] label {color: #94a3b8 !important; font-weight: 500;}

        /* ===== KPI Cards ===== */
        .voc-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: clamp(14px, 1.4vw, 20px);
            box-shadow: var(--shadow-sm);
            min-height: clamp(90px, 9vw, 110px);
            transition: box-shadow 0.2s, transform 0.2s;
            position: relative;
            overflow: hidden;
        }
        .voc-card:hover {box-shadow: var(--shadow-md); transform: translateY(-1px);}
        .voc-card h4 {margin: 0; font-size: 12px; color: var(--text-secondary); font-weight: 600; letter-spacing: 0.02em; text-transform: uppercase;}
        .voc-card p {margin: 6px 0 0; font-size: clamp(22px, 2vw, 32px); font-weight: 800; color: var(--text-primary); line-height: 1.15; word-break: keep-all;}
        .voc-card .voc-card-delta {font-size: 13px; font-weight: 600; margin-top: 4px; display: inline-flex; align-items: center; gap: 3px;}
        .voc-card .voc-card-delta.positive {color: var(--accent-positive);}
        .voc-card .voc-card-delta.negative {color: var(--accent-negative);}
        .voc-card .voc-card-sub {font-size: 11px; color: var(--text-muted); margin-top: 2px;}
        .voc-card::after {content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px; border-radius: 0 0 var(--radius-md) var(--radius-md);}
        .voc-card.accent-blue::after {background: var(--accent-primary);}
        .voc-card.accent-green::after {background: var(--accent-positive);}
        .voc-card.accent-orange::after {background: var(--accent-negative);}
        .voc-card.accent-purple::after {background: var(--accent-neutral);}

        /* ===== Panels / Containers ===== */
        .voc-panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: var(--radius-lg);
            padding: clamp(18px, 1.6vw, 26px);
            box-shadow: var(--shadow-sm);
            overflow: hidden;
        }

        /* ===== Section Headers ===== */
        .stApp h3 {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.01em;
            padding-bottom: 4px;
            border-bottom: 2px solid var(--accent-primary);
            display: inline-block;
        }

        /* ===== Chips & Tags ===== */
        .voc-chip {display: inline-block; padding: 4px 12px; border-radius: 999px; background: #eff6ff; color: var(--accent-primary); font-size: 12px; font-weight: 600; margin-right: 6px; border: 1px solid #dbeafe;}

        /* ===== Legend ===== */
        .voc-legend {display: flex; flex-wrap: wrap; gap: 8px 10px; margin-top: 8px; margin-bottom: 6px;}
        .voc-legend-compact {display: flex; flex-direction: column; gap: 10px; padding: 10px 0 4px;}
        .voc-donut-wrap {display: flex; align-items: center; gap: 20px;}
        .voc-donut-chart {flex: 0 0 min(64%, 520px); min-width: 260px;}
        .voc-donut-side {flex: 1 1 260px; min-width: 220px;}
        .voc-legend-item {display: inline-flex; align-items: center; gap: 8px; padding: 5px 12px; border: 1px solid var(--border); border-radius: 999px; background: #fafbfc; font-size: 13px; color: var(--text-secondary); line-height: 1.2;}
        .voc-legend-swatch {width: 10px; height: 10px; border-radius: 999px; display: inline-block; flex: 0 0 auto;}

        /* ===== Comment Cards ===== */
        .voc-comment-card {background: #fafbfc; border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px 18px; margin-bottom: 10px;}
        .voc-comment-meta {font-size: 12px; color: var(--text-muted); margin-bottom: 8px; display: flex; gap: 12px; flex-wrap: wrap;}
        .voc-comment-text {font-size: 14px; line-height: 1.7; color: var(--text-primary); white-space: pre-wrap;}

        /* ===== Reason Block ===== */
        .voc-reason {
            margin: 12px 0 18px;
            padding: 14px 16px;
            border-radius: var(--radius-sm);
            background: linear-gradient(135deg, #eff6ff 0%, #f0f9ff 100%);
            border-left: 4px solid var(--accent-primary);
            font-size: 14px;
            font-weight: 600;
            color: #1e40af;
            line-height: 1.55;
        }

        /* ===== NLP Insight Badge ===== */
        .nlp-badge {display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.03em;}
        .nlp-badge.positive {background: #d1fae5; color: #065f46;}
        .nlp-badge.negative {background: #ffedd5; color: #9a3412;}
        .nlp-badge.neutral {background: #e0e7ff; color: #3730a3;}
        .nlp-badge.trash {background: #f1f5f9; color: #475569;}
        .insight-badge {display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.02em; background: #faf5ff; color: #7c3aed; border: 1px solid #ede9fe;}
        .rep-ai-card {margin-top: 10px; background: linear-gradient(180deg, #071329 0%, #030b1b 100%); border: 1px solid #1e3a5f; border-radius: 14px; padding: 14px 14px 12px; color: #e2e8f0;}
        .rep-ai-head {display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px;}
        .rep-ai-pill {display: inline-flex; align-items: center; border: 1px solid #24466d; border-radius: 10px; padding: 4px 10px; font-size: 13px; font-weight: 800; letter-spacing: 0.01em;}
        .rep-ai-pill.positive {color: #60a5fa; background: rgba(37,99,235,0.12);}
        .rep-ai-pill.negative {color: #fb7185; background: rgba(190,24,93,0.12);}
        .rep-ai-pill.neutral {color: #c4b5fd; background: rgba(124,58,237,0.14);}
        .rep-ai-pill.trash {color: #94a3b8; background: rgba(100,116,139,0.15);}
        .rep-ai-score {font-size: 30px; font-weight: 900; line-height: 1; color: #f8fafc;}
        .rep-ai-progress-wrap {height: 8px; background: rgba(148,163,184,0.35); border-radius: 999px; margin: 4px 0 12px;}
        .rep-ai-progress-fill {height: 100%; border-radius: 999px; background: linear-gradient(90deg, #60a5fa, #3b82f6);}
        .rep-ai-flags {display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;}
        .rep-ai-flag {font-size: 12px; font-weight: 700; color: #93c5fd; background: rgba(30,64,175,0.25); border: 1px solid rgba(96,165,250,0.35); border-radius: 8px; padding: 4px 8px;}
        .rep-ai-sec {margin-top: 10px;}
        .rep-ai-label {font-size: 12px; color: #93c5fd; font-weight: 800; letter-spacing: 0.02em; margin-bottom: 4px;}
        .rep-ai-box {background: rgba(15,23,42,0.68); border: 1px solid rgba(148,163,184,0.25); border-radius: 10px; padding: 8px 10px; font-size: 13px; line-height: 1.5; color: #f1f5f9;}
        .rep-ai-topic {display: flex; align-items: center; justify-content: space-between; gap: 8px; background: rgba(15,23,42,0.68); border: 1px solid rgba(148,163,184,0.25); border-radius: 10px; padding: 8px 10px;}
        .rep-ai-topic-title {font-size: 14px; font-weight: 700; color: #e2e8f0;}
        .rep-ai-topic-sent {font-size: 12px; font-weight: 700; color: #93c5fd;}
        .rep-ai-kws {display: flex; gap: 6px; flex-wrap: wrap;}
        .rep-ai-kw {font-size: 12px; font-weight: 700; color: #cbd5e1; background: rgba(30,41,59,0.85); border: 1px solid rgba(148,163,184,0.35); border-radius: 999px; padding: 3px 8px;}
        .rep-ai-insight {margin-top: 10px; border-left: 3px solid #38bdf8; background: rgba(14,116,144,0.12); border-radius: 8px; padding: 8px 10px; font-size: 13px; color: #e0f2fe;}
        .rep-ai-action {margin-top: 8px; border-left: 3px solid #22c55e; background: rgba(22,101,52,0.16); border-radius: 8px; padding: 8px 10px; font-size: 13px; color: #dcfce7;}

        /* ===== Data Tables ===== */
        [data-testid="stDataFrame"] {border-radius: var(--radius-sm); overflow: hidden;}
        [data-testid="stDataFrame"] table {font-size: 13px;}

        /* ===== Expanders ===== */
        [data-testid="stExpander"] {border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; overflow: hidden;}

        /* ===== Tabs ===== */
        [data-testid="stTabs"] [data-baseweb="tab-list"] {gap: 0; border-bottom: 2px solid var(--border);}
        [data-testid="stTabs"] [data-baseweb="tab"] {font-weight: 600; font-size: 14px; padding: 10px 20px; border-bottom: 3px solid transparent; color: var(--text-secondary);}
        [data-testid="stTabs"] [aria-selected="true"] {border-bottom-color: var(--accent-primary) !important; color: var(--accent-primary) !important;}

        /* ===== Containers with border ===== */
        [data-testid="stVerticalBlockBorderWrapper"] {border-color: var(--border) !important; border-radius: var(--radius-md) !important; box-shadow: var(--shadow-sm);}

        /* ===== Download Button ===== */
        .stDownloadButton > button {background: var(--accent-primary) !important; color: #fff !important; border: none !important; border-radius: var(--radius-sm) !important; font-weight: 600 !important; padding: 8px 20px !important; box-shadow: 0 2px 8px rgba(59,130,246,0.25) !important; transition: all 0.2s;}
        .stDownloadButton > button:hover {background: #2563eb !important; box-shadow: 0 4px 12px rgba(59,130,246,0.35) !important; transform: translateY(-1px);}

        /* ===== Metrics Row ===== */
        .voc-metrics-row {display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 16px;}
        .voc-metrics-row > * {flex: 1; min-width: 160px;}

        /* ===== Responsive ===== */
        @media (max-width: 1200px) { .block-container {max-width: 100%;} }
        @media (max-width: 980px) { .voc-card p {font-size: clamp(18px, 3vw, 24px);} }
        @media (max-width: 768px) { .voc-donut-wrap {flex-direction: column;} .voc-donut-chart {flex: 1; min-width: unset;} }
        </style>
        """,
        unsafe_allow_html=True,
    )



PRODUCT_TARGET_KR_MAP = {
    "refrigerator": "냉장고",
    "washer": "세탁기",
    "dryer": "건조기",
    "dishwasher": "식기세척기",
    "drum_washer": "드럼세탁기",
    "top_load_washer": "통돌이세탁기",
    "air_conditioner": "에어컨",
    "microwave": "전자레인지",
    "vacuum": "청소기",
    "styler": "스타일러",
}


PRODUCT_LABEL_ALIASES = {
    "냉장고": "냉장고",
    "세탁기": "세탁기",
    "건조기": "건조기",
    "식기세척기": "식기세척기",
    "식세기": "식기세척기",
    "청소기": "청소기",
    "오븐": "오븐",
    "정수기": "정수기",
    "인덕션": "인덕션",
    "와인셀러": "와인셀러",
    "컵세척기": "컵세척기",
    "실내식물재배기": "실내식물재배기(틔운)",
    "틔운": "실내식물재배기(틔운)",
}


def canonicalize_product_label(value: Any) -> str:
    raw = _safe_text(value)
    if not raw:
        return ""

    normalized = re.sub(r"\s+", " ", raw).strip()
    normalized = re.sub(r"(추천|리뷰|후기|비교|영상)\s*$", "", normalized).strip()
    if not normalized:
        return ""
    if normalized in PRODUCT_ORDER:
        return normalized

    for marker, canonical in PRODUCT_LABEL_ALIASES.items():
        if marker in normalized:
            return canonical
    return normalized


def localize_product_target(value: str) -> str:
    """Map English product_target to Korean product sub-category."""
    raw = _safe_text(value)
    if not raw:
        return ""
    return PRODUCT_TARGET_KR_MAP.get(raw.lower().strip(), raw)


def localize_region(value: str) -> str:
    return REGION_LABELS.get(str(value), str(value))



def localize_sentiment(value: str) -> str:
    return SENTIMENT_LABELS.get(str(value), str(value))


def region_codes_from_labels(values: list[str]) -> list[str]:
    selected = set(values or [])
    codes = [code for code, label in REGION_LABELS.items() if label in selected or code in selected]
    if REGION_OTHER_LABEL in selected:
        codes.append(REGION_OTHER_LABEL)
    return codes


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
    source_text = (translation if translation and translation != "한국어 번역을 준비 중입니다." else original).strip()
    if not source_text:
        return "제품 관련 경험을 언급했지만 핵심 맥락이 부족해 추가 확인이 필요한 댓글입니다."

    lowered = source_text.lower()
    products = [token for token in ["드럼", "통돌이", "세탁기", "건조기", "냉장고", "식기세척기"] if token in source_text]
    product_subject = "·".join(products[:2]) if products else "해당 제품"

    scenarios = [
        (["구독", "가입", "권유", "매니저", "할부"], "구독 가입/권유 과정"),
        (["고장", "수리", "as", "서비스", "출장", "보증"], "A/S·수리 대응"),
        (["배송", "설치", "기사"], "배송·설치 과정"),
        (["소음", "진동", "탈수", "세척", "건조", "성능", "냉각"], "핵심 성능 체감"),
        (["불편", "무겁", "허리", "꺼내", "조작"], "사용 편의"),
        (["가격", "비싸", "비용", "전기세"], "가격·비용 부담"),
        (["옷감", "보풀", "손상", "엉킴", "늘어"], "옷감 관리 경험"),
    ]
    scenario = next((label for markers, label in scenarios if any(marker in lowered for marker in markers)), "사용 경험")

    inquiry_flag = _truthy(selected.get("nlp_is_inquiry")) or _truthy(selected.get("inquiry_flag")) or ("?" in source_text)
    keywords = _extract_card_keywords(selected, sentiment_name, limit=3)
    focus = ", ".join(keywords[:2]) if keywords else scenario

    if inquiry_flag:
        return f"{product_subject} 관련 {scenario}에서 '{focus}' 정보를 확인하려는 문의성 댓글입니다."
    if sentiment_name == "negative":
        return f"{product_subject} 관련 {scenario}에서 '{focus}' 이슈로 불편·불만을 제기한 댓글입니다."
    if sentiment_name == "positive":
        return f"{product_subject} 관련 {scenario}에서 '{focus}' 강점에 대한 만족·추천 의사를 드러낸 댓글입니다."
    return f"{product_subject} 관련 {scenario}에서 '{focus}'에 대한 평가와 경험을 함께 전달한 댓글입니다."


def _decision_badge(decision: str) -> tuple[str, str]:
    mapping = {
        "strong_candidate": ("대표 적합", "의사결정자에게 바로 보여줘도 이해가 쉬운 댓글입니다."),
        "review_needed": ("검토 필요", "의미는 있지만 맥락이나 해석 보강이 필요한 댓글입니다."),
        "not_suitable": ("대표 부적합", "대표 코멘트로 쓰기에는 맥락 의존성이나 설명력이 부족합니다."),
    }
    return mapping.get(decision, ("검토 필요", "대표성 판단 정보가 부족해 추가 검토가 필요합니다."))


def render_card(
    title: str,
    value: str,
    delta: str | None = None,
    delta_direction: str | None = None,
    subtitle: str | None = None,
    accent: str = "",
) -> None:
    """Render a KPI metric card.

    Args:
        title: Card header label.
        value: Main metric value (large text).
        delta: Optional delta text (e.g. "+2.5%").
        delta_direction: "positive" or "negative" for arrow direction/color.
        subtitle: Optional small text below value (e.g. "전주 대비").
        accent: Card accent color class ("blue", "green", "orange", "purple").
    """
    accent_cls = f" accent-{accent}" if accent else ""
    delta_html = ""
    if delta:
        direction = delta_direction or ("positive" if delta.startswith("+") else "negative")
        arrow = "&#9650;" if direction == "positive" else "&#9660;"
        delta_html = f'<div class="voc-card-delta {direction}">{arrow} {delta}</div>'
    sub_html = f'<div class="voc-card-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="voc-card{accent_cls}"><h4>{title}</h4><p>{value}</p>{delta_html}{sub_html}</div>',
        unsafe_allow_html=True,
    )



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
    if "product" in working.columns:
        working["product"] = working["product"].map(canonicalize_product_label)
    # Enrich product with sub-category from product_target when available
    if "product_target" in working.columns:
        sub_product = working["product_target"].map(localize_product_target).map(canonicalize_product_label)
        has_sub = sub_product.fillna("").ne("")
        working.loc[has_sub, "product"] = sub_product[has_sub]

    working[COL_WRITTEN_AT] = pd.to_datetime(published_series, errors="coerce", utc=True).dt.tz_localize(None)
    working[COL_WEEK_START] = working[COL_WRITTEN_AT].dt.to_period("W-SUN").dt.start_time
    working[COL_WEEK_LABEL] = working[COL_WEEK_START].dt.strftime("%Y-%m-%d")
    return working


def _contains_terms(text: str, terms: list[str]) -> bool:
    lowered = str(text or "").lower()
    for raw in terms:
        token = str(raw or "").strip().lower()
        if token and token in lowered:
            return True
    return False


def _annotate_videos_lg_relevance(videos_df: pd.DataFrame) -> pd.DataFrame:
    working = videos_df.copy()
    if working.empty:
        return working
    if "is_lg_relevant_video" in working.columns and "video_lg_relevance_score" in working.columns:
        return working
    if not VIDEO_SELECTION_SETTINGS.enabled:
        working["is_lg_relevant_video"] = True
        working["video_lg_relevance_score"] = 0.0
        return working

    title = working.get("title", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    description = working.get("description", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    channel = working.get("channel_title", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    keyword = working.get("keyword", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    product = working.get("product", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
    tags_blob = working.get("tags", pd.Series([[] for _ in range(len(working))], index=working.index)).map(
        lambda value: " ".join(str(item) for item in (value if isinstance(value, list) else [value]))
    ).fillna("").astype(str).str.lower()
    blob = (title + " " + description + " " + channel + " " + keyword + " " + product + " " + tags_blob).str.strip()

    target_hit = blob.map(lambda text: _contains_terms(text, LG_TARGET_TERMS))
    product_hit = blob.map(lambda text: _contains_terms(text, LG_PRODUCT_TERMS))
    competitor_hit = blob.map(lambda text: _contains_terms(text, LG_COMPETITOR_TERMS))
    preferred_hit = channel.map(lambda text: _contains_terms(text, LG_PREFERRED_CHANNEL_TERMS))
    blocked_hit = channel.map(lambda text: _contains_terms(text, LG_BLOCKED_CHANNEL_TERMS))

    score = (
        target_hit.astype(float) * 0.45
        + product_hit.astype(float) * 0.35
        + preferred_hit.astype(float) * 0.30
    )
    if VIDEO_SELECTION_SETTINGS.exclude_competitor_brands:
        score = score - ((competitor_hit & ~target_hit).astype(float) * 0.40)
    score = score.clip(lower=0.0, upper=1.0)

    is_relevant = score >= 0.45
    if VIDEO_SELECTION_SETTINGS.require_product_term:
        is_relevant = is_relevant & (product_hit | preferred_hit)
    if VIDEO_SELECTION_SETTINGS.require_target_brand:
        is_relevant = is_relevant & (target_hit | preferred_hit)
    is_relevant = is_relevant & (~blocked_hit)

    working["is_lg_relevant_video"] = is_relevant
    working["video_lg_relevance_score"] = score.round(3)
    return working


def ensure_lg_relevance_columns(comments_df: pd.DataFrame, videos_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    videos = _annotate_videos_lg_relevance(videos_df)
    comments = comments_df.copy()

    if comments.empty:
        return comments, videos

    if "video_id" in comments.columns and not videos.empty and "video_id" in videos.columns:
        context_cols = [
            col
            for col in ["video_id", "is_lg_relevant_video", "video_lg_relevance_score"]
            if col in videos.columns
        ]
        merge_cols = ["video_id"] + [col for col in context_cols if col != "video_id" and col not in comments.columns]
        if len(merge_cols) > 1:
            comments = comments.merge(
                videos[merge_cols].drop_duplicates(subset=["video_id"]),
                on="video_id",
                how="left",
            )

    if "video_lg_relevance_score" not in comments.columns:
        comments["video_lg_relevance_score"] = 0.0
    score_series = pd.to_numeric(comments["video_lg_relevance_score"], errors="coerce").fillna(0.0)
    video_related_series = comments.get("is_lg_relevant_video", pd.Series(False, index=comments.index)).fillna(False).astype(bool)

    product_related_series = comments.get("product_related", pd.Series(False, index=comments.index)).fillna(False).astype(bool)
    target_series = comments.get("product_target", pd.Series("", index=comments.index)).fillna("").astype(str).str.strip().ne("")
    classification_series = comments.get("classification_type", pd.Series("", index=comments.index)).fillna("").astype(str).str.lower()
    inquiry_series = comments.get("nlp_is_inquiry", pd.Series(False, index=comments.index)).fillna(False).astype(bool)

    computed_score = (score_series * 0.65).clip(0.0, 1.0)
    computed_score = computed_score + (product_related_series.astype(float) * 0.25)
    computed_score = computed_score + (target_series.astype(float) * 0.08)
    computed_score = computed_score + (classification_series.isin({"complaint", "comparison", "preference"}).astype(float) * 0.05)
    computed_score = computed_score + (inquiry_series.astype(float) * 0.03)
    computed_score = computed_score.clip(0.0, 1.0)

    if "lg_relevance_score" not in comments.columns:
        comments["lg_relevance_score"] = computed_score.round(3)
    if "lg_relevant_comment" not in comments.columns:
        comments["lg_relevant_comment"] = video_related_series & (computed_score >= 0.5)

    return comments, videos


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

    products = list(filters.get("products") or [])
    regions = list(filters.get("regions") or [])
    sentiments = list(filters.get("sentiments") or [])
    cej = list(filters.get("cej") or [])
    brands = list(filters.get("brands") or [])
    products_active = bool(filters.get("products_active", bool(products)))
    regions_active = bool(filters.get("regions_active", bool(regions)))
    sentiments_active = bool(filters.get("sentiments_active", bool(sentiments)))
    cej_active = bool(filters.get("cej_active", bool(cej)))
    brands_active = bool(filters.get("brands_active", bool(brands)))
    keyword_query = str(filters.get("keyword_query") or "").strip()
    analysis_scope = str(filters.get("analysis_scope") or "전체").strip() or "전체"
    relevance_scope = str(filters.get("relevance_scope") or "전체 수집").strip() or "전체 수집"

    opinion_units = (opinion_units_df.copy() if opinion_units_df is not None else pd.DataFrame())
    if not opinion_units.empty:
        if products_active and "product" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["product"].isin(products)]
        if regions_active and "region" in opinion_units.columns:
            region_codes = region_codes_from_labels(regions)
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                opinion_units = opinion_units[
                    opinion_units["region"].isin(explicit_codes) | ~opinion_units["region"].isin(REGION_CODES)
                ]
            else:
                opinion_units = opinion_units[opinion_units["region"].isin(region_codes)]
        if cej_active and "cej_scene_code" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["cej_scene_code"].map(canonicalize_cej).isin(cej)]
        if sentiments_active and "sentiment" in opinion_units.columns:
            opinion_units = opinion_units[opinion_units["sentiment"].astype(str).str.lower().map(localize_sentiment).isin(sentiments)]
        if brands_active and "brand_mentioned" in opinion_units.columns:
            opinion_units["brand_mentioned"] = opinion_units["brand_mentioned"].map(canonicalize_brand)
            opinion_units = opinion_units[opinion_units["brand_mentioned"].isin(brands)]

    inquiry_source_ids: set[str] = set()
    if not opinion_units.empty and "source_content_id" in opinion_units.columns and "inquiry_flag" in opinion_units.columns:
        inquiry_source_ids = set(opinion_units.loc[opinion_units["inquiry_flag"].fillna(False), "source_content_id"].astype(str))

    if products_active:
        base_comments = base_comments[base_comments["product"].isin(products)]
    if regions_active and COL_COUNTRY in base_comments.columns:
        known_labels = set(REGION_LABELS.values())
        if REGION_OTHER_LABEL in regions:
            base_comments = base_comments[
                base_comments[COL_COUNTRY].isin(regions) | ~base_comments[COL_COUNTRY].isin(known_labels)
            ]
        else:
            base_comments = base_comments[base_comments[COL_COUNTRY].isin(regions)]
    if cej_active and COL_CEJ in base_comments.columns:
        base_comments = base_comments[base_comments[COL_CEJ].isin(cej)]
    if brands_active and COL_BRAND in base_comments.columns:
        base_comments = base_comments[base_comments[COL_BRAND].isin(brands)]
    if keyword_query:
        base_comments = base_comments[
            base_comments["cleaned_text"].fillna("").str.contains(keyword_query, case=False, na=False)
            | base_comments["text_display"].fillna("").str.contains(keyword_query, case=False, na=False)
        ]

    if analysis_scope != "전체" and "comment_id" in base_comments.columns:
        comment_ids = base_comments["comment_id"].astype(str)
        if analysis_scope == "문의 포함 댓글만":
            base_comments = base_comments[comment_ids.isin(inquiry_source_ids)]
        elif analysis_scope == "문의 제외":
            base_comments = base_comments[~comment_ids.isin(inquiry_source_ids)]

    pre_lg_comment_count = len(base_comments)
    if relevance_scope == "LG 관련만":
        if "lg_relevant_comment" in base_comments.columns:
            lg_mask = base_comments["lg_relevant_comment"].fillna(False).astype(bool)
            base_comments = base_comments[lg_mask]
        elif "is_lg_relevant_video" in base_comments.columns:
            lg_video_mask = base_comments["is_lg_relevant_video"].fillna(False).astype(bool)
            base_comments = base_comments[lg_video_mask]
        elif "video_lg_relevance_score" in base_comments.columns:
            score_mask = pd.to_numeric(base_comments["video_lg_relevance_score"], errors="coerce").fillna(0.0) >= 0.45
            base_comments = base_comments[score_mask]
    post_lg_comment_count = len(base_comments)

    all_comments = base_comments.copy()
    
    filtered_comments = base_comments.copy()
    if sentiments_active and COL_SENTIMENT in filtered_comments.columns:
        filtered_comments = filtered_comments[filtered_comments[COL_SENTIMENT].isin(sentiments)]

    representative_comments = base_comments.copy()
    valid_series = representative_comments.get("comment_validity", pd.Series("valid", index=representative_comments.index))
    representative_comments = representative_comments[valid_series.eq("valid")].copy()

    representative_bundles = (representative_bundles_df.copy() if representative_bundles_df is not None else pd.DataFrame())
    if not representative_bundles.empty:
        if products_active and "product" in representative_bundles.columns:
            representative_bundles = representative_bundles[representative_bundles["product"].isin(products)]
        if regions_active and "region" in representative_bundles.columns:
            region_codes = region_codes_from_labels(regions)
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                representative_bundles = representative_bundles[
                    representative_bundles["region"].isin(explicit_codes) | ~representative_bundles["region"].isin(REGION_CODES)
                ]
            else:
                representative_bundles = representative_bundles[representative_bundles["region"].isin(region_codes)]
        if brands_active and "brand_mentioned" in representative_bundles.columns:
            representative_bundles["brand_mentioned"] = representative_bundles["brand_mentioned"].map(canonicalize_brand)
            representative_bundles = representative_bundles[representative_bundles["brand_mentioned"].isin(brands)]
        if analysis_scope != "전체" and "top_source_content_id" in representative_bundles.columns:
            top_ids = representative_bundles["top_source_content_id"].astype(str)
            if analysis_scope == "문의 포함 댓글만":
                representative_bundles = representative_bundles[top_ids.isin(inquiry_source_ids)]
            elif analysis_scope == "문의 제외":
                representative_bundles = representative_bundles[~top_ids.isin(inquiry_source_ids)]

    filtered_videos = videos_df.copy()
    if COL_COUNTRY not in filtered_videos.columns and "region" in filtered_videos.columns:
        filtered_videos[COL_COUNTRY] = filtered_videos["region"].map(localize_region)
    if products_active and "product" in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos["product"].isin(products)]
    if regions_active and COL_COUNTRY in filtered_videos.columns:
        known_labels = set(REGION_LABELS.values())
        if REGION_OTHER_LABEL in regions:
            filtered_videos = filtered_videos[
                filtered_videos[COL_COUNTRY].isin(regions) | ~filtered_videos[COL_COUNTRY].isin(known_labels)
            ]
        else:
            filtered_videos = filtered_videos[filtered_videos[COL_COUNTRY].isin(regions)]
    if brands_active and COL_BRAND in filtered_videos.columns:
        filtered_videos = filtered_videos[filtered_videos[COL_BRAND].isin(brands)]
    if relevance_scope == "LG 관련만":
        if "is_lg_relevant_video" in filtered_videos.columns:
            filtered_videos = filtered_videos[filtered_videos["is_lg_relevant_video"].fillna(False).astype(bool)]
        elif "video_lg_relevance_score" in filtered_videos.columns:
            filtered_videos = filtered_videos[
                pd.to_numeric(filtered_videos["video_lg_relevance_score"], errors="coerce").fillna(0.0) >= 0.45
            ]

    quality_filtered = quality_df.copy()
    if not quality_filtered.empty:
        if products_active and "product" in quality_filtered.columns:
            quality_filtered = quality_filtered[quality_filtered["product"].isin(products)]
        if regions_active and "region" in quality_filtered.columns:
            region_codes = region_codes_from_labels(regions)
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                quality_filtered = quality_filtered[
                    quality_filtered["region"].isin(explicit_codes) | ~quality_filtered["region"].isin(REGION_CODES)
                ]
            else:
                quality_filtered = quality_filtered[quality_filtered["region"].isin(region_codes)]
    if "comment_validity" in filtered_comments.columns:
        removed_count = int((filtered_comments["comment_validity"] == "excluded").sum())
        meaningful_count = int((filtered_comments["comment_validity"] == "valid").sum())
    else:
        removed_count = 0
        meaningful_count = len(filtered_comments)
    return {
        "comments": filtered_comments,
        "all_comments": all_comments,   # ✅ 추가
        "representative_comments": representative_comments,
        "videos": filtered_videos,
        "representative_bundles": representative_bundles,
        "opinion_units": opinion_units,
        "removed_count": removed_count,
        "meaningful_count": meaningful_count,
        "inquiry_count": int(opinion_units.get("inquiry_flag", pd.Series(dtype=bool)).fillna(False).sum()) if not opinion_units.empty else 0,
        "pre_lg_comment_count": int(pre_lg_comment_count),
        "post_lg_comment_count": int(post_lg_comment_count),
        "relevance_scope": relevance_scope,
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
    """Normalize pill selection state. Returns the default when no session state exists yet.
    If user explicitly cleared the selection (empty list in session_state), respect that."""
    valid_defaults = [item for item in defaults if item in options]
    if not valid_defaults:
        valid_defaults = list(options)
    if key not in st.session_state:
        # First load — use defaults (= all selected)
        st.session_state[key] = list(valid_defaults)
        return list(valid_defaults)
    current = st.session_state[key]
    if not isinstance(current, list):
        current = list(valid_defaults)
    normalized = [item for item in current if item in options]
    # Respect empty selection — user may have clicked "전체 해제"
    if st.session_state.get(key) != normalized:
        st.session_state[key] = normalized
    return normalized


def build_keyword_pack(comments_df: pd.DataFrame, sentiment_code: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    texts = tuple(comments_df.loc[comments_df["sentiment_label"] == sentiment_code, "cleaned_text"].fillna("").astype(str).tolist())
    blocked_keywords = _build_dynamic_keyword_excludes(comments_df)
    return (
        build_keyword_summary(texts, top_n=6, version="v7", blocked_keywords=blocked_keywords),
        build_keyword_summary(texts, top_n=30, version="v7", blocked_keywords=blocked_keywords),
    )




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
            st.image(donut_image, width="stretch")
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
        st.image(wordcloud, width="stretch")

    with st.expander("\uc804\uccb4 \ud0a4\uc6cc\ub4dc \ubaa9\ub85d \ubcf4\uae30"):
        st.dataframe(full_df.rename(columns={"keyword": "\ud0a4\uc6cc\ub4dc", "count": "\uc5b8\uae09 \uc218"}), width="stretch", hide_index=True)


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
            st.dataframe(new_display[["\uc81c\ud488\uad70", COL_COUNTRY, "\ud0a4\uc6cc\ub4dc", "\uc5b8\uae09 \uc218", "\ucd5c\uadfc \uc8fc\ucc28"]].head(15), width="stretch", hide_index=True)
    with tab_persistent:
        if persistent_df.empty:
            st.info("\uc9c0\uc18d \uc774\uc288 \ud0a4\uc6cc\ub4dc\uac00 \uc5c6\uc2b5\ub2c8\ub2e4.")
        else:
            persistent_display = persistent_df.rename(columns={"product": "\uc81c\ud488\uad70", "keyword": "\ud0a4\uc6cc\ub4dc", "count": "\uc5b8\uae09 \uc218", "weeks_active": "\uc9c0\uc18d \uc8fc\ucc28 \uc218", "latest_week": "\ucd5c\uadfc \uc8fc\ucc28"})
            st.dataframe(persistent_display[["\uc81c\ud488\uad70", COL_COUNTRY, "\ud0a4\uc6cc\ub4dc", "\uc5b8\uae09 \uc218", "\uc9c0\uc18d \uc8fc\ucc28 \uc218", "\ucd5c\uadfc \uc8fc\ucc28"]].head(15), width="stretch", hide_index=True)


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
        # Deduplicate index to prevent InvalidIndexError when building DataFrame
        if bundle_row.index.duplicated().any():
            bundle_row = bundle_row[~bundle_row.index.duplicated(keep="first")]
        bundle_row["cluster_size"] = cluster_size
        bundle_row["sample_comments_json"] = json.dumps(sample_comments, ensure_ascii=False)
        likes_value = int(pd.to_numeric(bundle_row.get(COL_LIKES, bundle_row.get("like_count", bundle_row.get("top_likes_count", 0))), errors="coerce") or 0)
        likes_text = f", 좋아요 {likes_value:,}개" if likes_value > 0 else ""
        bundle_row["selection_reason_ui"] = f"유사 의견 {cluster_size}개가 같은 이슈를 반복 언급했고, 감성 강도와 내용 구체성{likes_text}을 기준으로 대표 코멘트로 선정했습니다."
        picked.append(bundle_row)
        if len(picked) >= limit:
            break

    if not picked:
        return pd.DataFrame()
    records = [s.to_dict() for s in picked]
    return pd.DataFrame(records)


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
        converted["nlp_summary"] = _safe_text(row.get("top_summary_text", ""))
        converted["nlp_sentiment_reason"] = _safe_text(row.get("selection_reason", ""))
        converted["nlp_confidence"] = float(pd.to_numeric(row.get("bundle_score", 0), errors="coerce") or 0)
        selection_reason_raw = _safe_text(row.get("selection_reason", ""))
        if not selection_reason_raw or _looks_garbled(selection_reason_raw):
            selection_reason_raw = f"유사 의견 {converted['cluster_size']}개가 같은 이슈를 반복 언급했고, 감성 강도와 내용 구체성을 기준으로 대표 코멘트로 선정했습니다."
        converted["selection_reason_ui"] = selection_reason_raw
        converted["representative_reason"] = selection_reason_raw
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
        if not _safe_text(working.get(COL_VIDEO_TITLE, "")) or _safe_text(working.get(COL_VIDEO_TITLE, "")) == "관련 영상":
            working[COL_VIDEO_TITLE] = _safe_text(row.get(COL_VIDEO_TITLE, row.get("title", "관련 영상"))) or "관련 영상"
        if not _safe_text(working.get(COL_VIDEO_LINK, "")) or _safe_text(working.get(COL_VIDEO_LINK, "")) == "#":
            working[COL_VIDEO_LINK] = _safe_text(row.get(COL_VIDEO_LINK, row.get("video_url", "#"))) or "#"
        if not _safe_text(working.get(COL_LANGUAGE, "")):
            working[COL_LANGUAGE] = _safe_text(row.get(COL_LANGUAGE, row.get("language_detected", "")))
        if int(pd.to_numeric(working.get(COL_LIKES, 0), errors="coerce") or 0) == 0:
            working[COL_LIKES] = int(pd.to_numeric(row.get(COL_LIKES, row.get("like_count", row.get("likes_count", 0))), errors="coerce") or 0)
        passthrough_fields = [
            "nlp_label",
            "nlp_confidence",
            "nlp_sentiment_reason",
            "nlp_summary",
            "nlp_keywords",
            "nlp_is_inquiry",
            "nlp_is_quantitative",
            "classification_reason",
            "display_reason",
            "signal_strength",
            "confidence_level",
            "classification_type",
            "product_target",
            "topic_label",
            COL_CEJ,
            COL_BRAND,
            "representative_reason",
        ]
        for field_name in passthrough_fields:
            if field_name in row.index and not _safe_text(working.get(field_name, "")):
                working[field_name] = row.get(field_name)

    if not _safe_text(working.get(COL_ORIGINAL, "")) and sample_original:
        working[COL_ORIGINAL] = sample_original
    if not _safe_text(working.get(COL_RAW, "")):
        working[COL_RAW] = _safe_text(working.get(COL_ORIGINAL, sample_original))
    if not _safe_text(working.get(COL_TRANSLATION, "")) and sample_translation:
        working[COL_TRANSLATION] = sample_translation
    if not _safe_text(working.get(COL_VIDEO_TITLE, "")):
        working[COL_VIDEO_TITLE] = "관련 영상"
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


def render_representative_comments(filtered_comments: pd.DataFrame, representative_bundles: pd.DataFrame | None = None, opinion_units: pd.DataFrame | None = None, key_suffix: str = "") -> None:
    st.subheader("핵심 대표 코멘트")
    st.caption("현재 필터 범위에서 반복적으로 나타나는 제품 의견을 대표 이슈 중심으로 압축해 보여줍니다. 먼저 핵심 코멘트를 읽고, 필요할 때만 유사 댓글과 상세 정보를 펼쳐보세요.")

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
        render_comment_table(fallback_showcase, key_prefix=f"neutral{key_suffix}", source_comments=source_for_cards)
        return

    st.write(f"부정 핵심 대표 이슈 {len(negative_showcase):,}건")
    render_comment_table(negative_showcase, key_prefix=f"negative{key_suffix}", source_comments=source_for_cards)

    st.write(f"긍정 핵심 대표 이슈 {len(positive_showcase):,}건")
    render_comment_table(positive_showcase, key_prefix=f"positive{key_suffix}", source_comments=source_for_cards)


def _collect_similar_comments_for_download(
    source_comments: pd.DataFrame,
    selected: pd.Series,
    sentiment_name: str,
    exclude_id: str | None = None,
) -> pd.DataFrame:
    if source_comments is None or source_comments.empty:
        return pd.DataFrame()
    working = source_comments.copy()
    # ✅ 대표로 뽑힌 코멘트는 유사 댓글 샘플에서 제외
    exclude_id = _safe_text(exclude_id)
    if exclude_id:
        if "comment_id" in working.columns:
            working = working[working["comment_id"].astype(str) != exclude_id].copy()
        if "source_content_id" in working.columns:
            working = working[working["source_content_id"].astype(str) != exclude_id].copy()
        if "opinion_id" in working.columns:
            working = working[working["opinion_id"].astype(str) != exclude_id].copy()
    # ✅ 텍스트가 완전히 동일한 경우도 제외(보조 안전장치)
    selected_text = _safe_text(selected.get(COL_ORIGINAL, selected.get("text_display", "")))
    selected_raw = _safe_text(selected.get(COL_RAW, selected.get("text_original", "")))
    if selected_text:
        for col in [COL_ORIGINAL, "text_display", "original_text", "display_text"]:
            if col in working.columns:
                working = working[working[col].astype(str) != selected_text].copy()
    if selected_raw:
        for col in [COL_RAW, "text_original", "raw_text"]:
            if col in working.columns:
                working = working[working[col].astype(str) != selected_raw].copy()


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

def _build_similar_comments(
    pool: pd.DataFrame,
    selected: pd.Series,
    working_selected: pd.Series,
    sentiment_name: str,
    rep_id: str,
    video_id: str,
) -> pd.DataFrame:
    """Build a DataFrame of similar comments for a representative comment card."""
    if pool.empty:
        return pd.DataFrame()
    working_pool = pool.copy()

    video_link_for_filter = _safe_text(working_selected.get(COL_VIDEO_LINK, selected.get(COL_VIDEO_LINK, "")))
    if video_id and "video_id" in working_pool.columns:
        working_pool = working_pool[working_pool["video_id"].astype(str) == video_id].copy()
    elif video_link_for_filter:
        if COL_VIDEO_LINK in working_pool.columns:
            working_pool = working_pool[working_pool[COL_VIDEO_LINK].astype(str) == video_link_for_filter].copy()
        elif "video_url" in working_pool.columns:
            working_pool = working_pool[working_pool["video_url"].astype(str) == video_link_for_filter].copy()

    if "comment_validity" in working_pool.columns:
        working_pool = working_pool[working_pool["comment_validity"].astype(str) == "valid"].copy()
    if "sentiment_label" in working_pool.columns:
        working_pool = working_pool[working_pool["sentiment_label"].astype(str) == sentiment_name].copy()
    elif "sentiment_code" in working_pool.columns:
        working_pool = working_pool[working_pool["sentiment_code"].astype(str) == sentiment_name].copy()

    if rep_id:
        for key in ["comment_id", "source_content_id", "opinion_id"]:
            if key in working_pool.columns:
                working_pool = working_pool[working_pool[key].astype(str) != rep_id].copy()

    # Narrow similar-comment pool by topic/product/brand/cej when available to avoid
    # showing an unrelated broad pool for every representative card.
    narrowed = working_pool.copy()
    topic_value = _safe_text(selected.get("topic_label", selected.get("aspect_key", "")))
    if topic_value:
        if "topic_label" in narrowed.columns:
            narrowed = narrowed[narrowed["topic_label"].astype(str) == topic_value].copy()
        elif "aspect_key" in narrowed.columns:
            narrowed = narrowed[narrowed["aspect_key"].astype(str) == topic_value].copy()

    product_value = _safe_text(selected.get("product", working_selected.get("product", "")))
    if product_value and "product" in narrowed.columns:
        narrowed = narrowed[narrowed["product"].astype(str) == product_value].copy()

    cej_value = _safe_text(selected.get(COL_CEJ, working_selected.get(COL_CEJ, "")))
    if cej_value:
        if COL_CEJ in narrowed.columns:
            narrowed = narrowed[narrowed[COL_CEJ].astype(str) == cej_value].copy()
        elif "cej_scene_code" in narrowed.columns:
            narrowed = narrowed[narrowed["cej_scene_code"].astype(str) == cej_value].copy()

    brand_value = _safe_text(selected.get(COL_BRAND, working_selected.get(COL_BRAND, "")))
    if brand_value:
        if COL_BRAND in narrowed.columns:
            narrowed = narrowed[narrowed[COL_BRAND].astype(str) == brand_value].copy()
        elif "brand_mentioned" in narrowed.columns:
            narrowed = narrowed[narrowed["brand_mentioned"].map(canonicalize_brand).astype(str) == brand_value].copy()

    if len(narrowed) >= 3:
        working_pool = narrowed

    def _coerce_scalar(value):
        if isinstance(value, pd.Series):
            return value.iloc[0] if not value.empty else ""
        if isinstance(value, (list, tuple)):
            return value[0] if value else ""
        return value

    records = []
    for _, row in working_pool.iterrows():
        text = _safe_text(_coerce_scalar(row.get(COL_ORIGINAL, row.get("text_display", row.get("original_text", "")))))
        if not text.strip():
            continue
        cid = ""
        for key in ["comment_id", "source_content_id", "opinion_id"]:
            if key in working_pool.columns:
                candidate = _safe_text(_coerce_scalar(row.get(key, "")))
                if candidate:
                    cid = candidate
                    break
        records.append({
            "comment_id": cid,
            "원문": text,
            "좋아요 수": int(pd.to_numeric(_coerce_scalar(row.get(COL_LIKES, row.get("like_count", 0))), errors="coerce") or 0),
            "작성일시": _safe_text(_coerce_scalar(row.get(COL_WRITTEN_AT, row.get("published_at", "")))),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.drop_duplicates(subset=["comment_id", "원문"]).copy()
    return result


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


def _resolve_sentiment_for_card(working_selected: pd.Series, fallback_sentiment: str) -> str:
    sentiment = _safe_text(
        working_selected.get(
            "nlp_label",
            working_selected.get("display_sentiment", working_selected.get("sentiment_label", fallback_sentiment)),
        )
    ).lower()
    if sentiment not in {"positive", "negative", "neutral", "trash"}:
        return fallback_sentiment if fallback_sentiment in {"positive", "negative", "neutral", "trash"} else "neutral"

    # Lightweight consistency guard: if label and text polarity clearly conflict,
    # avoid showing an obviously wrong label in the card header.
    text_blob = " ".join(
        [
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            _safe_text(working_selected.get("text_display", "")),
            _safe_text(working_selected.get("cleaned_text", "")),
        ]
    ).lower()
    negative_markers = ["고장", "불량", "문제", "불편", "별로", "최악", "안됨", "안돼", "수리", "as", "환불", "불만"]
    positive_markers = ["만족", "좋아요", "좋음", "추천", "최고", "잘됨", "편함", "가성비", "great", "good", "best", "love"]
    neg_hits = sum(1 for marker in negative_markers if marker in text_blob)
    pos_hits = sum(1 for marker in positive_markers if marker in text_blob)

    if sentiment == "positive" and neg_hits >= 2 and pos_hits == 0:
        return "negative"
    if sentiment == "negative" and pos_hits >= 2 and neg_hits == 0:
        return "positive"
    if abs(neg_hits - pos_hits) <= 1 and (neg_hits >= 1 or pos_hits >= 1):
        return "neutral"
    return sentiment


def _confidence_ratio_for_card(working_selected: pd.Series, sentiment_label: str) -> float:
    raw = working_selected.get("nlp_confidence", working_selected.get("confidence_level", ""))
    if raw is None:
        raw = ""

    if isinstance(raw, (int, float)):
        num = abs(float(raw))
        if num > 1:
            num = num / 100 if num <= 100 else 1.0
        if 0 <= num <= 1:
            return max(0.35, min(0.99, num))

    text_val = _safe_text(raw).strip().lower()
    if text_val.endswith("%"):
        try:
            return max(0.35, min(0.99, float(text_val.replace("%", "")) / 100))
        except Exception:
            pass

    if text_val in {"very_high", "high", "높음"}:
        return 0.90
    if text_val in {"medium", "중간", "보통"}:
        return 0.76
    if text_val in {"low", "낮음"}:
        return 0.62

    # Dynamic fallback scoring to avoid fixed-looking percentages across cards.
    base = 0.58
    confidence_level = _safe_text(working_selected.get("confidence_level", "")).lower()
    if confidence_level in {"very_high", "high", "높음"}:
        base += 0.18
    elif confidence_level in {"medium", "중간", "보통"}:
        base += 0.12
    elif confidence_level in {"low", "낮음"}:
        base += 0.06
    else:
        base += 0.09

    signal_strength = _safe_text(working_selected.get("signal_strength", working_selected.get("display_signal_strength", ""))).lower()
    if signal_strength == "strong":
        base += 0.08
    elif signal_strength == "medium":
        base += 0.04
    elif signal_strength == "weak":
        base += 0.01

    if _safe_text(working_selected.get("nlp_sentiment_reason", "")):
        base += 0.05
    if _safe_text(working_selected.get("nlp_summary", "")):
        base += 0.04
    if _safe_text(working_selected.get("classification_reason", "")):
        base += 0.03

    likes = float(pd.to_numeric(working_selected.get(COL_LIKES, working_selected.get("like_count", 0)), errors="coerce") or 0.0)
    base += min(0.05, likes / 300.0)

    text_length = len(_safe_text(working_selected.get(COL_ORIGINAL, working_selected.get("text_display", ""))))
    if text_length >= 70:
        base += 0.03
    elif text_length >= 30:
        base += 0.015

    if sentiment_label == "negative":
        base += 0.01
    elif sentiment_label == "positive":
        base += 0.005

    return max(0.52, min(0.95, base))


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text_val = _safe_text(value).strip().lower()
    return text_val in {"1", "true", "t", "y", "yes", "예", "네"}


def _infer_inquiry_flag(working_selected: pd.Series, display_text: str) -> bool:
    # Layer 1(analysis pipeline)에서 확정한 inquiry 플래그만 사용한다.
    for key in ["nlp_is_inquiry", "inquiry_flag", "is_inquiry", "question_flag"]:
        if key in working_selected.index and _truthy(working_selected.get(key)):
            return True
    # UI fallback heuristic: keeps false negatives low when upstream flag is missing.
    source = " ".join(
        [
            _safe_text(display_text),
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
        ]
    )
    lowered = source.lower()
    if "?" in source:
        return True
    inquiry_markers = [
        "어떻게", "어디", "왜", "무엇", "뭔가요", "가능한가", "가능할까요", "인가요", "될까요",
        "문의", "질문", "how", "where", "why", "what", "can i", "does it", "is it",
    ]
    return any(marker in lowered for marker in inquiry_markers)


def _infer_quantitative_flag(working_selected: pd.Series, display_text: str) -> bool:
    # Layer 1(analysis pipeline)에서 전달된 수치성 플래그만 표시한다.
    for key in ["nlp_is_quantitative", "quantitative_flag", "is_quantitative", "has_numeric", "numeric_flag"]:
        if key in working_selected.index and _truthy(working_selected.get(key)):
            return True
    source = " ".join(
        [
            _safe_text(display_text),
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
        ]
    )
    lowered = source.lower()
    has_number = bool(re.search(r"\d", lowered))
    if not has_number:
        return False
    quantitative_units = ["년", "개월", "주", "일", "시간", "분", "초", "번", "회", "만원", "원", "%", "kg", "l", "리터", "도", "w", "kwh"]
    quantitative_questions = ["몇", "얼마", "몇번", "몇 회", "how many", "how much", "비용", "가격", "용량", "전기세"]
    return any(unit in lowered for unit in quantitative_units) or any(marker in lowered for marker in quantitative_questions)


def _extract_card_keywords(working_selected: pd.Series, sentiment_name: str, limit: int = 5) -> list[str]:
    raw_keywords = working_selected.get("nlp_keywords", [])
    if isinstance(raw_keywords, str):
        try:
            raw_keywords = json.loads(raw_keywords)
        except Exception:
            raw_keywords = [part.strip() for part in raw_keywords.split(",") if part.strip()]
    if isinstance(raw_keywords, list):
        normalized = [normalize_keyword(item) for item in raw_keywords]
        cleaned = [item for item in normalized if item]
        deduped = list(dict.fromkeys(cleaned))
        if deduped:
            return deduped[:limit]

    text_candidates = [
        _safe_text(working_selected.get("cleaned_text", "")),
        _safe_text(working_selected.get(COL_ORIGINAL, "")),
        _safe_text(working_selected.get("text_display", "")),
    ]
    text_candidates = [value for value in text_candidates if value]
    if not text_candidates:
        return []

    counter = build_keyword_counter(tuple(text_candidates))
    ranked = [keyword for keyword, _ in filter_business_keywords(counter, top_n=limit * 3)]
    if not ranked:
        return []

    normalized_ranked = [normalize_keyword(keyword) for keyword in ranked]
    cleaned_ranked = [item for item in normalized_ranked if item]
    deduped_ranked = list(dict.fromkeys(cleaned_ranked))
    return deduped_ranked[:limit]


def _build_video_context_insight(working_selected: pd.Series, sentiment_name: str, aspect_label: str, keywords: list[str]) -> str:
    channel = _safe_text(working_selected.get("channel_title", ""))
    insight_type = _safe_text(working_selected.get("insight_type", ""))
    voc_items = _extract_voc_items(working_selected, sentiment_name)
    voc_insight = _safe_text(voc_items[0].get("insight", "")) if voc_items else ""
    keyword_hint = ", ".join(keywords[:3]) if keywords else "핵심 항목"
    cej = _safe_text(working_selected.get(COL_CEJ, "")).strip() or "사용"
    title = _safe_text(working_selected.get(COL_VIDEO_TITLE, "")).strip()
    context_prefix = f"[{title}] " if title else ""

    if voc_insight and not ("약한 신호" in voc_insight and sentiment_name in {"positive", "negative"}):
        base = voc_insight
    elif sentiment_name == "negative":
        if any(marker in keyword_hint for marker in ["수리", "AS", "서비스"]):
            base = f"{cej} 단계에서 서비스/수리 기대치와 실제 대응 간의 간극이 드러났고, {keyword_hint} 관련 불만이 반복됩니다."
        elif any(marker in keyword_hint for marker in ["소음", "진동", "성능", "세탁"]):
            base = f"{cej} 단계에서 체감 성능 이슈가 누적되며, {keyword_hint} 항목이 구매 후 만족도를 직접 깎는 신호가 보입니다."
        else:
            base = f"{cej} 단계에서 '{aspect_label}' 관련 불만이 반복되어, {keyword_hint} 이슈를 우선 점검할 필요가 있습니다."
    elif sentiment_name == "positive":
        if any(marker in keyword_hint for marker in ["편의", "세탁", "성능", "추천"]):
            base = f"{cej} 단계에서 실사용 만족 포인트가 확인되며, {keyword_hint} 항목이 추천·전환 동인으로 작동합니다."
        else:
            base = f"{cej} 단계에서 '{aspect_label}' 강점이 반복 언급되어, {keyword_hint} 포인트가 긍정 경험을 견인합니다."
    else:
        base = f"{cej} 단계에서 '{aspect_label}' 관련 언급이 누적되고 있어, {keyword_hint}의 방향성(불만/만족) 추세 확인이 필요합니다."

    if channel:
        return f"{context_prefix}[{channel}] 채널 맥락: {base}"
    if insight_type and insight_type != "일반_의견":
        return f"{context_prefix}[{insight_type}] {base}"
    return f"{context_prefix}{base}"


def _build_resolution_point(sentiment_name: str, keywords: list[str], inquiry_flag: bool) -> str:
    focus = ", ".join(keywords[:3]) if keywords else "핵심 이슈"
    if sentiment_name == "negative":
        if inquiry_flag:
            return f"{focus} 관련 FAQ/고정댓글/챗봇 답변을 먼저 정교화하고, 반복 불만은 AS 프로세스 및 제품 개선 백로그로 즉시 연결하세요."
        if any(token in focus for token in ["수리", "AS", "서비스"]):
            return f"{focus} 건은 접수-진단-방문-완료 리드타임을 분해해 병목을 먼저 제거하고, 재발 유형은 원인코드로 묶어 주간 재발률을 관리하세요."
        if any(token in focus for token in ["소음", "진동", "세탁", "성능"]):
            return f"{focus} 이슈는 사용 조건(모드/용량/설치 상태)별 재현 테스트를 우선 수행하고, 확인된 조건부터 제품·가이드·콘텐츠 개선 액션으로 연결하세요."
        return f"{focus}의 원인 분류(제품/설치/서비스)를 먼저 나눈 뒤, 상위 케이스부터 담당 조직별 개선 액션을 주 단위로 추적하세요."
    if sentiment_name == "positive":
        if any(token in focus for token in ["편의", "추천", "만족"]):
            return f"{focus} 강점을 구매 전환 구간(썸네일/설명란/고정댓글/비교표)에 일관되게 배치해 전환 포인트로 확장하세요."
        return f"{focus} 강점을 영상 설명란·썸네일·후속 콘텐츠에 일관되게 재노출해 전환 포인트로 확장하세요."
    return f"{focus} 관련 반응을 주차별로 모니터링하고, 긍정/부정 전환 신호가 커지면 즉시 대응 플로우를 붙이세요."


ASPECT_LABEL_KO = {
    "A/S & Support": "A/S·고객지원",
    "Performance": "성능",
    "Convenience/Usability": "사용 편의",
    "Cost": "가격/비용",
    "Delivery/Install": "배송/설치",
    "Durability": "내구성",
    "Product Experience": "제품 경험",
    "General": "제품 경험",
}


def _localize_aspect_label(label: str) -> str:
    clean = _safe_text(label).strip()
    if not clean:
        return "제품 경험"
    return ASPECT_LABEL_KO.get(clean, clean)


def _issue_strength_for_card(
    working_selected: pd.Series,
    sentiment_label: str,
    inquiry_flag: bool,
    keywords: list[str],
) -> tuple[str, float]:
    score = 0.28
    if sentiment_label == "negative":
        score += 0.22
    elif sentiment_label == "positive":
        score += 0.16
    else:
        score += 0.1

    signal_strength = _safe_text(working_selected.get("signal_strength", working_selected.get("display_signal_strength", ""))).lower()
    if signal_strength == "strong":
        score += 0.16
    elif signal_strength == "medium":
        score += 0.09
    elif signal_strength == "weak":
        score += 0.03

    if inquiry_flag:
        score += 0.06
    if _truthy(working_selected.get("nlp_is_quantitative")) or _truthy(working_selected.get("numeric_flag")):
        score += 0.04

    cluster_size = float(pd.to_numeric(working_selected.get("cluster_size", 1), errors="coerce") or 1.0)
    score += min(0.14, cluster_size / 180.0)

    likes = float(pd.to_numeric(working_selected.get(COL_LIKES, working_selected.get("like_count", 0)), errors="coerce") or 0.0)
    score += min(0.08, likes / 300.0)

    if keywords:
        score += 0.05
    if len(keywords) >= 3:
        score += 0.03

    text_blob = " ".join(
        [
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            _safe_text(working_selected.get("cleaned_text", "")),
        ]
    ).lower()
    severe_markers = ["고장", "수리", "환불", "불량", "최악", "소음", "진동", "누수", "문제"]
    if any(marker in text_blob for marker in severe_markers):
        score += 0.07

    score = max(0.05, min(0.99, score))
    if score >= 0.72:
        return "높음", score
    if score >= 0.52:
        return "중간", score
    return "보통", score


def _refine_aspect_label(aspect_label: str, working_selected: pd.Series, keywords: list[str], display_text: str) -> str:
    label = _safe_text(aspect_label).strip()
    if not label:
        label = "General"
    if label.lower() not in {"general", "기타", "other", "대표 코멘트"}:
        return label

    text_blob = " ".join(
        [
            _safe_text(display_text),
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            " ".join(keywords),
        ]
    ).lower()

    mapping = [
        (["as", "수리", "서비스", "출장", "보증", "연락"], "A/S & Support"),
        (["소음", "진동", "탈수", "세척", "성능", "냉각", "건조"], "Performance"),
        (["불편", "편함", "무겁", "허리", "꺼내", "조작", "사용"], "Convenience/Usability"),
        (["가격", "비싸", "비용", "전기세"], "Cost"),
        (["배송", "설치", "기사"], "Delivery/Install"),
        (["고장", "내구", "수명", "망가"], "Durability"),
    ]
    for markers, mapped_label in mapping:
        if any(marker in text_blob for marker in markers):
            return mapped_label
    return "Product Experience"


def _render_representative_nlp_panel(working_selected: pd.Series, sentiment_name: str, aspect_label: str, display_text: str) -> None:
    sentiment_value = _resolve_sentiment_for_card(working_selected, sentiment_name)
    sentiment_map = {"positive": "긍정", "negative": "부정", "neutral": "중립", "trash": "스팸"}
    sentiment_label = sentiment_map.get(sentiment_value, sentiment_value)
    ratio = _confidence_ratio_for_card(working_selected, sentiment_value)
    score_pct = int(round(ratio * 100))
    score_text = f"{score_pct}%"
    keywords = _extract_card_keywords(working_selected, sentiment_value, limit=5)
    inquiry_flag = _infer_inquiry_flag(working_selected, display_text)
    quantitative_flag = _infer_quantitative_flag(working_selected, display_text)
    issue_strength_label, issue_strength_score = _issue_strength_for_card(
        working_selected=working_selected,
        sentiment_label=sentiment_value,
        inquiry_flag=inquiry_flag,
        keywords=keywords,
    )
    localized_aspect_label = _localize_aspect_label(aspect_label)

    reason_candidates = [
        _safe_text(working_selected.get("nlp_sentiment_reason", "")),
        _safe_text(working_selected.get("display_reason", "")),
        _safe_text(working_selected.get("classification_reason", "")),
        _safe_text(working_selected.get("representative_reason", "")),
    ]
    bad_reason_tokens = {"감성 판단 근거를 구조화하기 위한 정보가 부족해, 원문 맥락 중심으로 확인이 필요합니다."}
    reason = next(
        (
            item
            for item in reason_candidates
            if item and not _looks_garbled(item) and item not in bad_reason_tokens and len(item) >= 12
        ),
        "",
    )
    if not reason:
        evidence = []
        raw_text = " ".join(
            [
                _safe_text(working_selected.get(COL_ORIGINAL, "")),
                _safe_text(working_selected.get("cleaned_text", "")),
                _safe_text(display_text),
            ]
        ).lower()
        evidence_map = [
            (["고장", "수리", "as", "서비스"], "A/S·수리 관련 직접 경험"),
            (["소음", "진동", "탈수", "세척", "성능"], "성능 관련 체감 표현"),
            (["불편", "힘들", "무겁", "허리"], "사용 편의성 불만/평가"),
            (["좋아요", "만족", "추천", "최고"], "긍정 평가 어휘"),
            (["비싸", "가격", "비용", "전기세"], "비용 관련 언급"),
        ]
        for markers, label in evidence_map:
            if any(marker in raw_text for marker in markers):
                evidence.append(label)
        reason_focus = ", ".join(evidence[:2]) if evidence else "원문의 핵심 평가 문장"
        reason = f"{reason_focus}이(가) 확인되어 '{sentiment_label}' 의견으로 판정했습니다."

    summary = _safe_text(working_selected.get("nlp_summary", ""))
    low_quality_summary_markers = {"요약 가능한 댓글 내용이 없습니다.", "한국어 번역을 준비 중입니다."}
    if (
        not summary
        or _looks_garbled(summary)
        or summary in low_quality_summary_markers
        or len(summary) < 12
        or (summary.endswith("...") and _safe_text(display_text).startswith(summary[:-3]))
    ):
        summary = _summarize_for_dashboard(working_selected, sentiment_value)
    if not summary:
        summary = "핵심 의견: 댓글 원문에서 제품 사용 경험에 대한 평가가 확인됩니다."

    context_insight = _build_video_context_insight(working_selected, sentiment_value, aspect_label, keywords)
    action_point = _build_resolution_point(sentiment_value, keywords, inquiry_flag)

    fill_color = {"negative": "#ef4444", "positive": "#3b82f6", "neutral": "#8b5cf6", "trash": "#64748b"}.get(sentiment_value, "#3b82f6")
    keywords_html = "".join(f'<span class="rep-ai-kw">{html.escape(keyword)}</span>' for keyword in keywords)
    if not keywords_html:
        keywords_html = '<span class="rep-ai-kw">키워드 없음</span>'

    st.markdown(
        f"""
        <div class="rep-ai-card">
          <div class="rep-ai-head">
            <div class="rep-ai-pill {sentiment_value}">{sentiment_label} ({sentiment_value})</div>
            <div class="rep-ai-score">{score_text}</div>
          </div>
          <div class="rep-ai-progress-wrap">
            <div class="rep-ai-progress-fill" style="width:{ratio * 100:.1f}%; background:{fill_color};"></div>
          </div>
          <div class="rep-ai-flags">
            <span class="rep-ai-flag">분류신뢰도: {score_pct}%</span>
            <span class="rep-ai-flag">이슈강도: {issue_strength_label} ({int(round(issue_strength_score * 100))}%)</span>
            <span class="rep-ai-flag">문의: {str(inquiry_flag).lower()}</span>
            <span class="rep-ai-flag">수치적질문: {str(quantitative_flag).lower()}</span>
          </div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">판단 근거</div>
            <div class="rep-ai-box">{html.escape(reason)}</div>
          </div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">요약</div>
            <div class="rep-ai-box">{html.escape(summary)}</div>
          </div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">토픽 · 감성</div>
            <div class="rep-ai-topic">
              <span class="rep-ai-topic-title">{html.escape(localized_aspect_label)}</span>
              <span class="rep-ai-topic-sent">{sentiment_label}</span>
            </div>
          </div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">키워드</div>
            <div class="rep-ai-kws">{keywords_html}</div>
          </div>
          <div class="rep-ai-insight"><strong>영상 맥락 인사이트</strong><br>{html.escape(context_insight)}</div>
          <div class="rep-ai-action"><strong>해결 포인트</strong><br>{html.escape(action_point)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comment_table(comment_showcase: pd.DataFrame, key_prefix: str, source_comments: pd.DataFrame | None = None) -> None:
    if key_prefix.startswith("negative"):
        sentiment_name = "negative"
    elif key_prefix.startswith("positive"):
        sentiment_name = "positive"
    else:
        sentiment_name = "neutral"

    if comment_showcase.empty:
        st.info("현재 조건에서 표시할 대표 코멘트가 없습니다.")
        return

    top_rows = comment_showcase.reset_index(drop=True)
    for idx, selected in top_rows.iterrows():
        with st.container(border=True):
            # --- Data extraction ---
            raw_aspect_label = _safe_text(selected.get("topic_label", "")) or _safe_text(selected.get("aspect_key", "")) or f"이슈 {idx + 1}"
            likes_value = pd.to_numeric(selected.get(COL_LIKES, 0), errors="coerce")
            likes = 0 if pd.isna(likes_value) else int(likes_value)
            cluster_size = int(pd.to_numeric(selected.get("cluster_size", 1), errors="coerce") or 1)

            working_selected = _enrich_card_row_from_source(selected.copy(), source_comments)
            original_text = _safe_text(working_selected.get(COL_ORIGINAL, ""))
            language = _safe_text(working_selected.get(COL_LANGUAGE, "")).lower()
            is_korean = language.startswith("ko") or _looks_korean_text(original_text)
            raw_translation = _safe_text(working_selected.get(COL_TRANSLATION, ""))
            translation = _resolve_card_translation(original_text, raw_translation, is_korean)
            display_text = original_text if is_korean else (translation or original_text)
            rough_keywords = _extract_card_keywords(working_selected, sentiment_name, limit=5)
            aspect_label = _refine_aspect_label(raw_aspect_label, working_selected, rough_keywords, display_text)
            aspect_label_display = _localize_aspect_label(aspect_label)

            insight_type_val = _safe_text(selected.get("insight_type", ""))
            product_val = _safe_text(selected.get("product", ""))
            cej_val = _safe_text(selected.get(COL_CEJ, ""))
            brand_val = _safe_text(selected.get(COL_BRAND, ""))

            # ============================================================
            # HEADER: 번호 + 토픽명 + 메타 배지
            # ============================================================
            badge_html_parts = [f'<strong style="font-size:16px;">{idx + 1}. {aspect_label_display}</strong>']
            if product_val:
                badge_html_parts.append(f'<span class="voc-chip">{product_val}</span>')
            if cej_val and cej_val != "기타":
                badge_html_parts.append(f'<span class="voc-chip">{cej_val}</span>')
            if brand_val and brand_val != "미언급":
                badge_html_parts.append(f'<span class="voc-chip">{brand_val}</span>')
            if insight_type_val and insight_type_val != "일반_의견":
                badge_html_parts.append(f'<span class="insight-badge">{insight_type_val}</span>')
            badge_html_parts.append(f'<span style="color:#94a3b8; font-size:12px;">👍 {likes:,} · 유사 {cluster_size:,}건</span>')

            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:8px;">{"".join(badge_html_parts)}</div>',
                unsafe_allow_html=True,
            )

            _render_representative_nlp_panel(
                working_selected=working_selected,
                sentiment_name=sentiment_name,
                aspect_label=aspect_label,
                display_text=display_text,
            )

            # ============================================================
            # 관련 영상 + 날짜
            # ============================================================
            video_title_raw = _safe_text(selected.get(COL_VIDEO_TITLE, '')) or ''
            video_title = _localized_video_title(video_title_raw) if video_title_raw else ''
            video_link = _safe_text(selected.get(COL_VIDEO_LINK, '')) or ''
            written_at = _safe_text(working_selected.get(COL_WRITTEN_AT, working_selected.get("published_at", "")))
            meta_parts = []
            if video_title and video_link:
                meta_parts.append(f'<a href="{video_link}" target="_blank">{video_title[:60]}{"…" if len(video_title) > 60 else ""}</a>')
            if written_at:
                meta_parts.append(str(written_at)[:10])
            if meta_parts:
                st.caption(" · ".join(meta_parts), unsafe_allow_html=True)

            # ============================================================
            # 원문 보기 (expander)
            # ============================================================
            with st.expander("원문 보기"):
                st.write(display_text)
                if not is_korean and original_text:
                    st.markdown("**원문(Original)**")
                    st.write(original_text)

            # ============================================================
            # 유사 댓글
            # ============================================================
            rep_id = _safe_text(selected.get("comment_id", selected.get("source_content_id", selected.get("opinion_id", ""))))
            video_id = _safe_text(selected.get("video_id", "")) or _safe_text(working_selected.get("video_id", ""))

            pool = source_comments if source_comments is not None else pd.DataFrame()
            similar_comments = _build_similar_comments(pool, selected, working_selected, sentiment_name, rep_id, video_id)

            if not similar_comments.empty:
                with st.expander(f"유사 댓글 {cluster_size:,}건"):
                    st.caption(f"현재 필터에서 매칭된 유사 댓글: {len(similar_comments):,}건")
                    for _, sample in similar_comments.head(5).iterrows():
                        st.markdown(f"- {_safe_text(sample.get('원문', ''))}")
                    if len(similar_comments) > 5:
                        st.caption(f"외 {len(similar_comments) - 5:,}건 더...")
                    csv_bytes = similar_comments.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"similar_{key_prefix}_{idx+1}.csv",
                        mime="text/csv",
                        key=f"similar_download_{key_prefix}_{idx}",
                    )


def build_video_summary(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame) -> pd.DataFrame:
    videos = filtered_videos.copy()
    comments = filtered_comments.copy()

    # ✅ [1번 수정] video_id 기준으로 영상 1행으로 정리 + product는 합쳐서 표시
    if not comments.empty and "video_id" in comments.columns:
        # 이 영상에 연결된 댓글의 product를 모아서 "A, B, C" 형태로 만듦
        if "product" in comments.columns:
            product_map = (
                comments.dropna(subset=["video_id"])
                .groupby("video_id")["product"]
                .apply(lambda s: ", ".join(sorted({str(x).strip() for x in s.dropna() if str(x).strip()})))
            )
        else:
            product_map = pd.Series(dtype=str)

        # videos가 product 단위로 중복되어 있을 수 있으므로 video_id 1행으로 압축
        if "video_id" in videos.columns and not videos.empty:
            keep_cols = [
                "video_id",
                "region",
                "channel_title",
                "title",
                "video_url",
                "published_at",
                "view_count",
                "like_count",
                "comment_count",
                "is_lg_relevant_video",
                "video_lg_relevance_score",
            ]
            keep_cols = [c for c in keep_cols if c in videos.columns]
            videos = videos.sort_values(["video_id"]).drop_duplicates(subset=["video_id"], keep="first")
            # product 컬럼은 댓글 기반으로 재작성(있으면 덮어씀)
            if len(product_map) > 0:
                videos["product"] = videos["video_id"].map(product_map).fillna(videos.get("product", ""))
    
    if videos.empty and not comments.empty:
        rebuild_cols = [
            "product",
            "region",
            "channel_title",
            "title",
            "video_url",
            "published_at",
            "view_count",
            "like_count",
            "comment_count",
            "is_lg_relevant_video",
            "video_lg_relevance_score",
        ]
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

    if "sentiment_label" not in comments.columns:
        comments["sentiment_label"] = "neutral"
    if "comment_validity" not in comments.columns:
        comments["comment_validity"] = "valid"
    if "like_count" not in comments.columns:
        comments["like_count"] = 0

    comment_stats = comments.groupby("video_id", dropna=False).agg(
        분석_전체_댓글_수=("video_id", "size"),
        긍정_댓글_수=("sentiment_label", lambda s: int((s == "positive").sum())),
        부정_댓글_수=("sentiment_label", lambda s: int((s == "negative").sum())),
        중립_댓글_수=("sentiment_label", lambda s: int((s == "neutral").sum())),
        제외_댓글_수=("comment_validity", lambda s: int((s == "excluded").sum())),
    ).reset_index()

    negative_reason = pd.DataFrame(columns=["video_id", "\uc8fc\uc694 \ubd80\uc815 \ud3ec\uc778\ud2b8"])
    if "classification_reason" in comments.columns:
        neg_comments = comments[comments["sentiment_label"] == "negative"]
        if not neg_comments.empty:
            negative_reason = (
                neg_comments.sort_values(["video_id", "like_count"], ascending=[True, False])
                .groupby("video_id", as_index=False)["classification_reason"]
                .first()
                .rename(columns={"classification_reason": "\uc8fc\uc694 \ubd80\uc815 \ud3ec\uc778\ud2b8"})
            )

    summary = videos.merge(comment_stats, on="video_id", how="left").merge(negative_reason, on="video_id", how="left")

    
    # ✅ [2번 수정] 주요 부정 포인트 깨짐 방지 (_safe_text 적용)
    if "주요 부정 포인트" in summary.columns:
        summary["주요 부정 포인트"] = summary["주요 부정 포인트"].map(_safe_text)

    
    for col in ["분석_전체_댓글_수", "긍정_댓글_수", "부정_댓글_수", "중립_댓글_수", "제외_댓글_수"]:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors="coerce").fillna(0)
    if COL_COUNTRY in summary.columns:
        country_series = summary[COL_COUNTRY]
    elif "region" in summary.columns:
        country_series = summary["region"].map(localize_region)
    else:
        country_series = pd.Series(pd.NA, index=summary.index, dtype="object")
    summary[COL_COUNTRY] = country_series
    summary[COL_VIDEO_LINK] = summary.get("video_url", "")
    summary[COL_VIDEO_TITLE] = summary.get("title", "")
    published_source = summary["published_at"] if "published_at" in summary.columns else pd.Series(pd.NA, index=summary.index, dtype="object")
    summary["발행일"] = pd.to_datetime(published_source, errors="coerce").dt.strftime("%Y-%m-%d")
    # ✅ 새 컬럼명 기준으로 정렬 (없으면 있는 컬럼만으로 정렬)
    sort_cols = [c for c in ["부정_댓글_수", "분석_전체_댓글_수", "view_count"] if c in summary.columns]
    if not sort_cols:
        sort_cols = ["view_count"] if "view_count" in summary.columns else ["video_id"]

    return summary.sort_values(
        sort_cols,
        ascending=[False] * len(sort_cols),
        na_position="last"
    )



def render_video_summary_page(all_comments: pd.DataFrame, filtered_videos: pd.DataFrame) -> None:
    st.markdown("### 영상 분석 요약")
    st.caption("상단은 영상별 전체 댓글 분포(긍/부/중립/제외), 하단은 분석 대상(valid) 댓글 기준으로 밀도/상세 분석을 제공합니다.")

    # 1) 전체 분포용(긍/부/중립/제외 포함)
    all_video_comments = all_comments.copy()

    # 2) 분석 대상(valid)만 (여기서 중립까지 포함)
    validity_series = all_video_comments.get("comment_validity", pd.Series("valid", index=all_video_comments.index))
    analysis_comments = all_video_comments[validity_series.astype(str) == "valid"].copy()

    summary = build_video_summary(all_video_comments, filtered_videos)

    st.caption(
        f"영상 요약 계산 결과: 댓글 {len(all_video_comments):,}건 / "
        f"영상 {len(filtered_videos):,}건 / 요약 {len(summary):,}건"
    )
    if summary.empty:
        st.info("표시할 영상 요약 데이터가 없습니다.")
        return

    # ✅ 상단 카드(전체 기준)
    k1, k2, k3 = st.columns(3)
    with k1:
        render_card("분석 영상 수", f"{len(summary):,}")
    with k2:
        # 전체 댓글 기준 합계
        total_all = int(pd.to_numeric(summary.get("분석_전체_댓글_수", 0), errors="coerce").fillna(0).sum())
        render_card("전체 댓글(합)", f"{total_all:,}")
    with k3:
        top_video = _localized_video_title(_safe_text(summary.iloc[0].get(COL_VIDEO_TITLE, "-")))
        render_card("최상위 이슈 영상", str(top_video)[:28] or "-")

    if "is_lg_relevant_video" in filtered_videos.columns:
        relevant_videos = int(filtered_videos["is_lg_relevant_video"].fillna(False).astype(bool).sum())
        st.caption(f"LG 관련 영상: {relevant_videos:,} / 전체 영상: {len(filtered_videos):,}")

    if "channel_title" in filtered_videos.columns:
        channel_overview = (
            filtered_videos.assign(channel_title=filtered_videos["channel_title"].fillna("").astype(str))
            .query("channel_title != ''")
            .groupby("channel_title", dropna=False)
            .agg(
                영상수=("video_id", "nunique"),
                댓글합=("comment_count", "sum"),
            )
            .reset_index()
            .sort_values(["영상수", "댓글합"], ascending=[False, False])
            .head(15)
        )
        if not channel_overview.empty:
            with st.expander("채널 포함 현황 보기"):
                st.dataframe(channel_overview, width="stretch", hide_index=True)

    # ✅ 상단 리스트(전체 분포)
    summary_display = summary.copy()
    if COL_VIDEO_TITLE in summary_display.columns:
        summary_display[COL_VIDEO_TITLE] = summary_display[COL_VIDEO_TITLE].map(_localized_video_title)

    # 보기 좋은 컬럼만 노출
    show_cols = [
        "product", COL_COUNTRY, "channel_title", COL_VIDEO_TITLE, "발행일", "view_count",
        "분석_전체_댓글_수", "긍정_댓글_수", "부정_댓글_수", "중립_댓글_수", "제외_댓글_수",
        "is_lg_relevant_video", "video_lg_relevance_score",
        "주요 부정 포인트",
    ]
    show_cols = [c for c in show_cols if c in summary_display.columns]
    st.dataframe(summary_display[show_cols].head(100), width="stretch", hide_index=True)

    # ✅ 3) 부정 밀도(분석 대상 valid 기준) 계산 + 영상 메타 결합
    density_display = pd.DataFrame()
    edited = pd.DataFrame()

    with st.container(border=True):
        st.markdown("#### 영상당 부정 댓글 밀도 (분석 대상 valid 기준)")

        if analysis_comments.empty or "video_id" not in analysis_comments.columns:
            st.info("분석 대상(valid) 댓글이 없어 밀도 계산이 불가합니다.")
        else:
            density_core = (
                analysis_comments.groupby("video_id", dropna=False)
                .agg(
                    분석_댓글_수=("video_id", "size"),
                    긍정_댓글_수=("sentiment_label", lambda s: int((s == "positive").sum())),
                    부정_댓글_수=("sentiment_label", lambda s: int((s == "negative").sum())),
                    중립_댓글_수=("sentiment_label", lambda s: int((s == "neutral").sum())),
                )
                .reset_index()
            )
            density_core["부정_밀도"] = density_core["부정_댓글_수"] / density_core["분석_댓글_수"]

            # 영상 메타(제목/링크/국가/제품) 붙이기
            meta = filtered_videos.copy()
            if "video_id" in meta.columns:
                meta = meta.drop_duplicates(subset=["video_id"], keep="first")
            else:
                meta["video_id"] = ""

            meta[COL_COUNTRY] = meta.get(COL_COUNTRY, meta.get("region", pd.Series("", index=meta.index)).map(localize_region))
            meta[COL_VIDEO_TITLE] = meta.get("title", meta.get(COL_VIDEO_TITLE, pd.Series("", index=meta.index))).map(_localized_video_title)
            meta[COL_VIDEO_LINK] = meta.get("video_url", meta.get(COL_VIDEO_LINK, pd.Series("", index=meta.index)))

            if "product" not in meta.columns:
                meta["product"] = ""
            density_display = density_core.merge(
                meta[["video_id", "product", COL_COUNTRY, COL_VIDEO_TITLE, COL_VIDEO_LINK]],
                on="video_id",
                how="left",
            )

            table_df = density_display.sort_values(["부정_밀도", "부정_댓글_수"], ascending=[False, False]).head(100).copy()
            table_df.insert(0, "선택", False)

            edited = st.data_editor(
                table_df,
                width="stretch",
                hide_index=True,
                num_rows="fixed",
                column_config={
                    "선택": st.column_config.CheckboxColumn("선택", help="체크하면 아래에 해당 영상 상세 분석이 표시됩니다."),
                    COL_VIDEO_TITLE: st.column_config.TextColumn("영상 제목", width="large"),
                },
                key="density_table_editor",
            )

    # ✅ 4) 선택된 영상 → 상세 분석(분석 대상 valid 기준과 정합)
    if density_display.empty:
        st.info("부정 밀도 데이터가 없어 영상 상세 분석 선택을 표시하지 않습니다.")
        return

    checked = edited[edited["선택"] == True] if (not edited.empty and "선택" in edited.columns) else pd.DataFrame()
    if checked.empty:
        st.caption("👆 위 표에서 ‘선택’을 체크하면 아래에 상세 분석이 표시됩니다.")
        return

    selected_video_id = checked.iloc[0]["video_id"]
    matched_rows = summary[summary["video_id"] == selected_video_id]
    if matched_rows.empty:
        st.warning("선택한 영상의 요약 데이터를 찾을 수 없습니다.")
        return
    selected_row = matched_rows.iloc[0]
    render_video_detail_page(analysis_comments, selected_row)  # ✅ 상세는 valid 풀로만


def render_video_detail_page(filtered_comments: pd.DataFrame, selected_video: pd.Series) -> None:
    video_id = selected_video.get("video_id")
    video_comments = filtered_comments[filtered_comments["video_id"] == video_id].copy()

    st.markdown("---")
    st.markdown("### 선택 영상 상세 분석")
    st.markdown(f"#### {selected_video.get(COL_VIDEO_TITLE, '-')}")
    st.caption(
        f"제품군: {selected_video.get('product', '-')} · {COL_COUNTRY}: {selected_video.get(COL_COUNTRY, '-')} · "
        f"발행일: {selected_video.get('발행일', '-')} · 조회수: {int(selected_video.get('view_count', 0) or 0):,}"
    )
    st.markdown(f"영상 링크: [{selected_video.get(COL_VIDEO_LINK, '-')}]({selected_video.get(COL_VIDEO_LINK, '#')})")

    if video_comments.empty:
        st.info("이 영상에 연결된 분석 댓글이 없습니다.")
        return

    validity = video_comments["comment_validity"] if "comment_validity" in video_comments.columns else pd.Series("valid", index=video_comments.index)
    valid_comments = video_comments[validity.eq("valid")].copy()
    excluded_comments = video_comments[validity.eq("excluded")].copy()
    sentiment_col = valid_comments.get("sentiment_label", pd.Series("neutral", index=valid_comments.index))

    positive_count = int((sentiment_col == "positive").sum()) if not valid_comments.empty else 0
    negative_count = int((sentiment_col == "negative").sum()) if not valid_comments.empty else 0
    neutral_count = int((sentiment_col == "neutral").sum()) if not valid_comments.empty else 0
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
    st.altair_chart(chart, width="stretch")

    with st.container(border=True):
        render_representative_comments(
            filtered_comments=video_comments,
            representative_bundles=None,
            opinion_units=None,
            key_suffix=f"_vid_{video_id}",
        )

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
        display["인사이트 유형"] = display.get("insight_type", pd.Series("", index=display.index)).fillna("")
        display[COL_SENTIMENT] = display.get(COL_SENTIMENT, display.get("sentiment_label", pd.Series("", index=display.index))).fillna("")
        cols = [COL_SENTIMENT, "제품 타깃", "인사이트 유형", "분류 유형", "신뢰도", COL_ORIGINAL, COL_TRANSLATION, COL_CONTEXT, "분류 이유", "맥락 필요", COL_LIKES]
        existing = [c for c in cols if c in display.columns]
        st.dataframe(display[existing], width="stretch", hide_index=True)

def get_mode() -> str:
    """Dashboard mode selector. Full mode only."""
    return "full"


def is_streamlit_cloud_runtime() -> bool:
    sharing_mode = os.getenv("STREAMLIT_SHARING_MODE", "").strip().lower() == "true"
    mount_src = str(BASE_DIR).replace("\\", "/").startswith("/mount/src/")
    return sharing_mode or mount_src


def should_autoload_full_data() -> bool:
    """Cloud health-check 안정성을 위해 기본은 수동 로딩."""
    if not is_streamlit_cloud_runtime():
        return True
    try:
        autoload = str(st.query_params.get("autoload", "0")).strip().lower()
    except Exception:
        autoload = "0"
    return autoload in {"1", "true", "yes", "on"}


def _build_dashboard_options(comments_df: pd.DataFrame, videos_df: pd.DataFrame) -> dict[str, list[str]]:
    comment_products_raw = comments_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist()
    video_products_raw = videos_df.get("product", pd.Series(dtype=str)).dropna().astype(str).tolist()
    comment_products = [canonicalize_product_label(item) for item in comment_products_raw]
    video_products = [canonicalize_product_label(item) for item in video_products_raw]
    all_data_products = sorted({item for item in (comment_products + video_products) if item})
    available_products = [item for item in PRODUCT_ORDER if item in all_data_products]
    extra_from_data = [p for p in all_data_products if p and p not in PRODUCT_ORDER]
    available_products.extend(extra_from_data)
    if not available_products:
        available_products = PRIMARY_PRODUCT_FILTERS[: min(4, len(PRIMARY_PRODUCT_FILTERS))]

    full_product_list = PRODUCT_ORDER + [p for p in extra_from_data if p not in PRODUCT_ORDER]

    # Build brand options: configured list + any brands in data not already listed
    brand_options = BRAND_FILTER_OPTIONS[:]
    data_brands = set()
    if COL_BRAND in comments_df.columns:
        data_brands = set(comments_df[COL_BRAND].dropna().astype(str).unique())
    elif "brand_label" in comments_df.columns:
        data_brands = {canonicalize_brand(b) for b in comments_df["brand_label"].dropna().unique()}
    for b in sorted(data_brands):
        if b and b not in brand_options:
            brand_options.append(b)

    return {
        "products": full_product_list,
        "available_products": available_products,
        "regions": [localize_region(code) for code in REGION_CODES] + [REGION_OTHER_LABEL],
        "sentiments": [
            localize_sentiment("positive"),
            localize_sentiment("negative"),
            localize_sentiment("neutral"),
            localize_sentiment("excluded"),
        ],
        "cej": CEJ_ORDER[:],
        "brands": brand_options,
        "analysis_scope": ["전체", "문의 포함 댓글만", "문의 제외"],
        "relevance_scope": ["전체 수집", "LG 관련만"],
    }


def _render_filter_toggle(label: str, pill_key: str, all_options: list[str]) -> None:
    """Render 전체 선택 / 전체 해제 toggle buttons for a pill filter."""
    col_all, col_clear = st.columns(2)
    with col_all:
        if st.button("전체 선택", key=f"{pill_key}_all", width="stretch", type="secondary"):
            st.session_state[pill_key] = list(all_options)
            st.rerun()
    with col_clear:
        if st.button("전체 해제", key=f"{pill_key}_clear", width="stretch", type="secondary"):
            st.session_state[pill_key] = []
            st.rerun()


def _render_sidebar_filters(options: dict[str, list[str]]) -> dict[str, Any]:
    filter_product_defaults = _normalize_pill_state("filter_products", options["products"], options["available_products"])
    filter_region_defaults = _normalize_pill_state("filter_regions", options["regions"], options["regions"])
    filter_brand_defaults = _normalize_pill_state("filter_brands", options["brands"], options["brands"])
    filter_sentiment_defaults = _normalize_pill_state("filter_sentiments", options["sentiments"], options["sentiments"])
    filter_cej_defaults = _normalize_pill_state("filter_cej", options["cej"], options["cej"])

    with st.sidebar:
        st.markdown("### 탐색 필터")
        if st.button("필터 초기화 (전체 데이터 보기)", width="stretch", type="secondary"):
            st.session_state["filter_products"] = list(options["products"])
            st.session_state["filter_regions"] = list(options["regions"])
            st.session_state["filter_brands"] = list(options["brands"])
            st.session_state["filter_sentiments"] = list(options["sentiments"])
            st.session_state["filter_cej"] = list(options["cej"])
            st.rerun()

        with st.container(border=True):
            st.caption("제품 범위")
            st.caption(f"데이터 보유 제품군: {', '.join(options['available_products'])}")
            _render_filter_toggle("제품", "filter_products", options["products"])
            selected_products = st.pills(
                "제품군",
                options["products"],
                selection_mode="multi",
                key="filter_products",
                label_visibility="collapsed",
                width="content",
            )

        with st.container(border=True):
            st.caption("시장")
            _render_filter_toggle("시장", "filter_regions", options["regions"])
            selected_regions = st.pills(
                "국가",
                options["regions"],
                selection_mode="multi",
                key="filter_regions",
                label_visibility="collapsed",
                width="content",
            )

        with st.container(border=True):
            st.caption("브랜드")
            _render_filter_toggle("브랜드", "filter_brands", options["brands"])
            selected_brands = st.pills(
                "브랜드",
                options["brands"],
                selection_mode="multi",
                key="filter_brands",
                label_visibility="collapsed",
                width="content",
            )

        with st.container(border=True):
            st.caption("감성 라벨")
            _render_filter_toggle("감성", "filter_sentiments", options["sentiments"])
            selected_sentiments = st.pills(
                "감성",
                options["sentiments"],
                selection_mode="multi",
                key="filter_sentiments",
                label_visibility="collapsed",
                width="content",
            )

        with st.container(border=True):
            st.caption("고객경험여정")
            _render_filter_toggle("CEJ", "filter_cej", options["cej"])
            selected_cej = st.pills(
                "고객경험여정 단계",
                options["cej"],
                selection_mode="multi",
                key="filter_cej",
                label_visibility="collapsed",
                width="content",
            )

        with st.container(border=True):
            st.caption("분석 유형")
            analysis_scope = st.radio(
                "분석 유형",
                options["analysis_scope"],
                index=0,
                label_visibility="collapsed",
            )

        with st.container(border=True):
            st.caption("LG 관련성 범위")
            relevance_scope = st.radio(
                "LG 관련성 범위",
                options["relevance_scope"],
                index=0,
                label_visibility="collapsed",
            )

        with st.container(border=True):
            st.caption("고급 옵션")
            keyword_query = st.text_input(
                "댓글 내 키워드 검색",
                "",
                placeholder="예: 소음, 냉각, warranty",
            )

    def _normalize_selection(selected: list[str], all_opts: list[str]) -> list[str]:
        return [item for item in selected if item in all_opts]

    def _is_filter_active(selected: list[str], all_opts: list[str]) -> bool:
        if not all_opts:
            return False
        return set(selected) != set(all_opts)

    selected_products = _normalize_selection(list(selected_products or []), options["products"])
    selected_regions = _normalize_selection(list(selected_regions or []), options["regions"])
    selected_brands = _normalize_selection(list(selected_brands or []), options["brands"])
    selected_sentiments = _normalize_selection(list(selected_sentiments or []), options["sentiments"])
    selected_cej = _normalize_selection(list(selected_cej or []), options["cej"])

    filters = {
        "products": selected_products,
        "regions": selected_regions,
        "sentiments": selected_sentiments,
        "cej": selected_cej,
        "brands": selected_brands,
        "products_active": _is_filter_active(selected_products, options["products"]),
        "regions_active": _is_filter_active(selected_regions, options["regions"]),
        "sentiments_active": _is_filter_active(selected_sentiments, options["sentiments"]),
        "cej_active": _is_filter_active(selected_cej, options["cej"]),
        "brands_active": _is_filter_active(selected_brands, options["brands"]),
        "keyword_query": keyword_query,
        "analysis_scope": analysis_scope,
        "relevance_scope": relevance_scope,
    }
    return filters


def render_comments_lite(comments_df: pd.DataFrame) -> None:
    """Render a paged comment table with a bounded working set for lite mode."""
    st.subheader("\uB313\uAE00 (Lite: \uC0D8\uD50C/\uD398\uC774\uC9C0\uB124\uC774\uC158)")

    if comments_df.empty:
        st.info("\uB313\uAE00 \uB370\uC774\uD130\uAC00 \uC5C6\uC2B5\uB2C8\uB2E4.")
        return

    page_size = st.selectbox("\uD398\uC774\uC9C0 \uD06C\uAE30", [50, 100, 200, 500], index=1)
    max_rows = st.selectbox("\uCD5C\uB300 \uB85C\uB529 \uD589 \uC218(\uACBD\uB7C9)", [1000, 3000, 5000, 10000], index=1)

    view_df = comments_df
    sort_cols = [c for c in [COL_WRITTEN_AT, COL_LIKES] if c in view_df.columns]
    if len(view_df) > max_rows:
        if sort_cols:
            view_df = view_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        view_df = view_df.head(max_rows).copy()
        st.caption(f"\uC804\uCCB4 {len(comments_df):,}\uAC74 \uC911 \uC0C1\uC704 {len(view_df):,}\uAC74\uB9CC \uACBD\uB7C9 \uB85C\uB529\uD588\uC2B5\uB2C8\uB2E4.")

    prefer_cols = [
        COL_COUNTRY,
        COL_SENTIMENT,
        COL_CEJ,
        COL_BRAND,
        COL_WRITTEN_AT,
        COL_LIKES,
        COL_VIDEO_TITLE,
        COL_VIDEO_LINK,
        COL_ORIGINAL,
        COL_TRANSLATION,
        COL_REMOVED,
    ]
    show_cols = [c for c in prefer_cols if c in view_df.columns]

    total = len(view_df)
    total_pages = max(1, (total + page_size - 1) // page_size)
    page = st.number_input("\uD398\uC774\uC9C0", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = min(start + page_size, total)

    st.caption(f"{start + 1:,}~{end:,} / {total:,} (page {page}/{total_pages})")
    st.dataframe(
        view_df.iloc[start:end][show_cols] if show_cols else view_df.iloc[start:end],
        width="stretch",
        hide_index=True,
    )


def load_dashboard_data_lite_comments() -> dict[str, pd.DataFrame]:
    """Reuse the full loader and trim the comment payload for lite mode."""
    data = load_dashboard_data()
    comments_df = data.get("comments", pd.DataFrame())
    if not comments_df.empty:
        data["comments"] = comments_df.head(LITE_COMMENTS_LIMIT).copy()
    return data


@st.cache_resource
def get_dashboard_data_resource_lite_comments():
    return load_dashboard_data_lite_comments()


def render_nlp_insights(data: dict[str, pd.DataFrame], filtered_comments: pd.DataFrame) -> None:
    """Render NLP Analyzer insight panels on the dashboard."""
    topic_insights = data.get("nlp_topic_insights", pd.DataFrame())
    keyword_insights = data.get("nlp_keyword_insights", pd.DataFrame())
    product_mentions = data.get("nlp_product_mentions", pd.DataFrame())
    sentiment_dist = data.get("nlp_sentiment_distribution", pd.DataFrame())

    has_any = not topic_insights.empty or not keyword_insights.empty or not product_mentions.empty
    if not has_any:
        if "nlp_label" not in filtered_comments.columns or filtered_comments["nlp_label"].isna().all():
            return

    st.markdown("### NLP 심층 분석 인사이트")
    st.caption("nlp_analyzer(LLM 기반)로 추출한 토픽, 키워드, 제품 언급, 문의 패턴을 요약합니다.")

    # NLP sentiment distribution - KPI row
    if not sentiment_dist.empty:
        accent_map = {"positive": "green", "negative": "orange", "neutral": "purple", "trash": "", "undecidable": ""}
        label_map = {"positive": "긍정", "negative": "부정", "neutral": "중립", "trash": "스팸", "undecidable": "판단불가"}
        dist_rows = list(sentiment_dist.head(5).iterrows())
        cols = st.columns(len(dist_rows))
        for i, (_, row) in enumerate(dist_rows):
            with cols[i]:
                label_kr = label_map.get(row["label"], row["label"])
                accent = accent_map.get(row["label"], "blue")
                render_card(
                    f"NLP {label_kr}",
                    f'{int(row["count"]):,}',
                    delta=f"{row['share']:.0%}",
                    delta_direction="positive" if row["label"] == "positive" else ("negative" if row["label"] == "negative" else None),
                    accent=accent,
                )

    # Topic + Keyword side by side
    col_left, col_right = st.columns(2)

    with col_left:
        if not topic_insights.empty:
            with st.container(border=True):
                st.markdown("#### 토픽별 감성 분포")
                st.caption("LLM이 식별한 주요 토픽과 해당 토픽 내 긍정/부정/중립 분포입니다.")
                display_topics = topic_insights.head(10).copy()
                display_topics = display_topics.rename(columns={
                    "topic": "토픽", "positive_count": "긍정", "negative_count": "부정",
                    "neutral_count": "중립", "total": "합계", "dominant_sentiment": "주요 감성",
                    "negative_rate": "부정 비율",
                })
                if "토픽" in display_topics.columns:
                    display_topics["토픽"] = display_topics["토픽"].map(lambda x: _localize_aspect_label(_safe_text(x)))
                sentiment_kr = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
                if "주요 감성" in display_topics.columns:
                    display_topics["주요 감성"] = display_topics["주요 감성"].map(lambda x: sentiment_kr.get(x, x))
                st.dataframe(display_topics, width="stretch", hide_index=True)

    with col_right:
        if not keyword_insights.empty:
            with st.container(border=True):
                st.markdown("#### NLP 키워드 분석")
                st.caption("LLM이 추출한 핵심 키워드의 빈도 및 감성 분포입니다.")
                display_kw = keyword_insights.head(10).copy()
                display_kw = display_kw.rename(columns={
                    "keyword": "키워드", "count": "빈도", "positive_count": "긍정",
                    "negative_count": "부정", "neutral_count": "중립", "dominant_sentiment": "주요 감성",
                })
                sentiment_kr = {"positive": "긍정", "negative": "부정", "neutral": "중립"}
                if "주요 감성" in display_kw.columns:
                    display_kw["주요 감성"] = display_kw["주요 감성"].map(lambda x: sentiment_kr.get(x, x))
                st.dataframe(display_kw, width="stretch", hide_index=True)

    # Product mentions + Negative summaries side by side
    col_prod, col_neg = st.columns(2)

    with col_prod:
        if not product_mentions.empty:
            with st.container(border=True):
                st.markdown("#### 제품 언급 분석")
                st.caption("LLM이 감지한 제품 언급과 해당 감성 분포입니다.")
                display_prod = product_mentions.head(8).copy()
                display_prod = display_prod.rename(columns={
                    "product": "제품", "mention_count": "언급 수", "positive_count": "긍정",
                    "negative_count": "부정", "neutral_count": "중립", "negative_rate": "부정 비율",
                })
                st.dataframe(display_prod, width="stretch", hide_index=True)

    with col_neg:
        if "nlp_summary" in filtered_comments.columns:
            summaries = filtered_comments[filtered_comments["nlp_summary"].notna() & filtered_comments["nlp_summary"].ne("")].copy()
            if not summaries.empty and "nlp_label" in summaries.columns:
                neg_summaries = summaries[summaries["nlp_label"] == "negative"].sort_values(
                    "nlp_confidence", ascending=False
                ).head(5)
                if not neg_summaries.empty:
                    with st.container(border=True):
                        st.markdown("#### 주요 부정 의견 요약")
                        for _, row in neg_summaries.iterrows():
                            conf = row.get("nlp_confidence", 0)
                            topics_list = row.get("nlp_topics", [])
                            topics_str = ", ".join(topics_list) if isinstance(topics_list, list) and topics_list else ""
                            badge_html = f'<span class="nlp-badge negative">{conf:.0%}</span>' if conf else ""
                            topic_html = f' <span style="color:#64748b; font-size:12px;">({topics_str})</span>' if topics_str else ""
                            st.markdown(
                                f'{badge_html} {row["nlp_summary"]}{topic_html}',
                                unsafe_allow_html=True,
                            )


def main() -> None:
    st.set_page_config(page_title="가전 VoC Dashboard", layout="wide")
    apply_theme()
    _initialize_data_session_state()
    active_mode = st.session_state.get(SESSION_DATA_MODE_KEY, "real")
    expected_signature = _expected_mode_signature(active_mode)
    if st.session_state.get(SESSION_DATA_SIGNATURE_KEY) != expected_signature:
        with st.spinner("최신 결과와 동기화 중…"):
            _set_data_mode(active_mode, force_refresh=True)

    with st.sidebar:
        st.markdown("### 데이터 모드")
        st.caption(f"현재 모드: {'샘플 모드' if active_mode == 'sample' else '실데이터 모드'}")
        col_sample, col_real = st.columns(2)
        with col_sample:
            if st.button("샘플 로드", width="stretch", type="secondary"):
                _set_data_mode("sample", force_refresh=False)
                st.rerun()
        with col_real:
            if st.button("실데이터 새로고침", width="stretch", type="primary"):
                _set_data_mode("real", force_refresh=True)
                st.rerun()

        with st.expander("실제 분석 실행", expanded=False):
            run_keyword = st.text_input("검색 키워드", value="가전 리뷰")
            market_labels = [label for label, _ in ANALYSIS_MARKET_OPTIONS]
            market_code_by_label = dict(ANALYSIS_MARKET_OPTIONS)
            default_market_index = next((i for i, (_, code) in enumerate(ANALYSIS_MARKET_OPTIONS) if code == "KR"), 0)
            selected_market_label = st.selectbox("시장", market_labels, index=default_market_index)
            run_region = market_code_by_label[selected_market_label]

            language_labels = [label for label, _ in ANALYSIS_LANGUAGE_OPTIONS]
            language_code_by_label = dict(ANALYSIS_LANGUAGE_OPTIONS)
            default_language_index = next((i for i, (_, code) in enumerate(ANALYSIS_LANGUAGE_OPTIONS) if code == "ko"), 0)
            selected_language_label = st.selectbox("언어", language_labels, index=default_language_index)
            run_language = language_code_by_label[selected_language_label]
            run_max_videos = st.number_input("수집 영상 수", min_value=5, max_value=200, value=40, step=5)
            run_comments_per_video = st.number_input("영상당 댓글 수(0=전체)", min_value=0, max_value=2000, value=0, step=50)
            run_output_prefix = st.text_input("결과 파일 prefix", value="dashboard_live")
            target_regions_preview = _analysis_target_regions(run_region)
            if run_keyword.strip():
                if run_region == ANALYSIS_ALL_MARKET_CODE:
                    st.caption(f"전체 시장 실행: {len(target_regions_preview)}개 국가를 순차 실행합니다.")
                    preview_lines: list[str] = []
                    for region_code in target_regions_preview[:5]:
                        variants = _build_search_keyword_variants(
                            run_keyword,
                            language=run_language,
                            region=region_code,
                        )
                        if variants:
                            preview_lines.append(f"[{region_code}] " + " | ".join(variants))
                    if preview_lines:
                        st.caption("예상 검색 키워드 예시(상위 5개 국가)")
                        st.code("\n".join(preview_lines))
                else:
                    keyword_preview = _build_search_keyword_variants(
                        run_keyword,
                        language=run_language,
                        region=run_region,
                    )
                    if keyword_preview:
                        st.caption("이번 실행 예상 검색 키워드")
                        st.code("\n".join(keyword_preview))
            if st.button("실분석 실행", width="stretch", type="secondary"):
                if not run_keyword.strip():
                    st.warning("검색 키워드를 입력해 주세요.")
                else:
                    with st.spinner("실제 분석을 실행 중입니다..."):
                        ok, message = _run_real_analysis_action(
                            keyword=run_keyword.strip(),
                            max_videos=int(run_max_videos),
                            comments_per_video=int(run_comments_per_video),
                            language=run_language,
                            region=run_region,
                            output_prefix=run_output_prefix.strip() or "dashboard_live",
                        )
                    if ok:
                        st.session_state[SESSION_LAST_RUN_RESULT_KEY] = message
                        _set_data_mode("real", force_refresh=True)
                        st.success("실분석이 완료되어 실데이터 모드로 갱신했습니다.")
                        st.rerun()
                    else:
                        st.error("실분석 실행 중 오류가 발생했습니다.")
                        st.code(message)

        if st.session_state.get(SESSION_LAST_RUN_RESULT_KEY):
            st.caption("최근 실분석 실행 결과")
            st.code(str(st.session_state[SESSION_LAST_RUN_RESULT_KEY])[:1200])

    data = st.session_state.get(SESSION_DATA_BUNDLE_KEY, _empty_dashboard_bundle())

    comments_raw = data.get("comments", pd.DataFrame()).copy()
    videos_raw = data.get("videos", pd.DataFrame()).copy()
    if active_mode == "real" and (comments_raw.empty or videos_raw.empty):
        st.warning(_build_missing_data_message())
        st.info("좌측 사이드바의 `샘플 로드`를 눌러 샘플 모드로 빠르게 UI를 검증할 수 있습니다.")
        return
    comments_scored, videos_df = ensure_lg_relevance_columns(comments_raw, videos_raw)
    comments_df = add_localized_columns(comments_scored)
    if comments_df.empty:
        st.warning("표시할 분석 결과가 없습니다. 먼저 데이터를 수집해주세요.")
        return

    videos_df[COL_COUNTRY] = videos_df.get("region", pd.Series(dtype=str)).map(localize_region)
    dashboard_options = _build_dashboard_options(comments_df, videos_df)

    st.markdown(
        '<div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">'
        '<div style="font-size:28px; font-weight:800; color:#1e293b; letter-spacing:-0.02em;">가전 VoC Dashboard</div>'
        f'<span style="display:inline-block; padding:3px 10px; border-radius:999px; background:#eff6ff; color:#3b82f6; font-size:11px; font-weight:700; letter-spacing:0.03em;">{DASHBOARD_BUILD_LABEL}</span>'
        f'<span style="display:inline-block; padding:3px 10px; border-radius:999px; background:{("#dcfce7" if active_mode == "real" else "#ffedd5")}; color:{("#166534" if active_mode == "real" else "#9a3412")}; font-size:11px; font-weight:700; letter-spacing:0.03em;">{"REAL" if active_mode == "real" else "SAMPLE"}</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.caption("주간 감성 변화, 핵심 키워드, CEJ 단계별 문제, 대표 코멘트를 제품·국가별로 빠르게 확인합니다.")

    selected_filters = _render_sidebar_filters(dashboard_options)

    bundle = compute_filtered_bundle(
        comments_df,
        videos_df,
        data.get("quality_summary", pd.DataFrame()),
        selected_filters,
        data.get("representative_bundles", pd.DataFrame()),
        data.get("opinion_units", pd.DataFrame()),
    )

    filtered_comments = bundle["comments"]
    all_comments = bundle.get("all_comments", filtered_comments)
    filtered_videos = bundle["videos"]
    products_active = bool(selected_filters.get("products_active", bool(selected_filters.get("products"))))
    regions_active = bool(selected_filters.get("regions_active", bool(selected_filters.get("regions"))))
    brands_active = bool(selected_filters.get("brands_active", bool(selected_filters.get("brands"))))

    # Default weekly window for download (chart controls override inline later)
    weekly_window, _ = build_weekly_sentiment_window(filtered_comments, WEEKLY_CHART_DEFAULT, 1)

    cej_df = data.get("cej_negative_rate", pd.DataFrame()).copy()
    if not cej_df.empty:
        cej_df[COL_COUNTRY] = cej_df["region"].map(localize_region)
        if products_active:
            cej_df = cej_df[cej_df["product"].isin(selected_filters["products"])]
        if regions_active:
            known_labels = set(REGION_LABELS.values())
            if REGION_OTHER_LABEL in selected_filters["regions"]:
                cej_df = cej_df[
                    cej_df[COL_COUNTRY].isin(selected_filters["regions"]) | ~cej_df[COL_COUNTRY].isin(known_labels)
                ]
            else:
                cej_df = cej_df[cej_df[COL_COUNTRY].isin(selected_filters["regions"])]

    brand_df = data.get("brand_ratio", pd.DataFrame()).copy()
    if not brand_df.empty:
        brand_df[COL_COUNTRY] = brand_df["region"].map(localize_region)
        if products_active:
            brand_df = brand_df[brand_df["product"].isin(selected_filters["products"])]
        if regions_active:
            known_labels = set(REGION_LABELS.values())
            if REGION_OTHER_LABEL in selected_filters["regions"]:
                brand_df = brand_df[
                    brand_df[COL_COUNTRY].isin(selected_filters["regions"]) | ~brand_df[COL_COUNTRY].isin(known_labels)
                ]
            else:
                brand_df = brand_df[brand_df[COL_COUNTRY].isin(selected_filters["regions"])]
        if brands_active and COL_BRAND in brand_df.columns:
            brand_df[COL_BRAND] = brand_df[COL_BRAND].map(canonicalize_brand)
            brand_df = brand_df[brand_df[COL_BRAND].isin(selected_filters["brands"])]

    density_df = data.get("negative_density", pd.DataFrame()).copy()
    if not density_df.empty:
        if products_active and "product" in density_df.columns:
            density_df = density_df[density_df["product"].isin(selected_filters["products"])]
        if regions_active and "region" in density_df.columns:
            region_codes = region_codes_from_labels(selected_filters["regions"])
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                density_df = density_df[
                    density_df["region"].isin(explicit_codes) | ~density_df["region"].isin(REGION_CODES)
                ]
            else:
                density_df = density_df[density_df["region"].isin(region_codes)]

    if filtered_comments.empty:
        st.warning("현재 선택한 필터에 맞는 댓글이 없습니다. 필터를 넓히거나 문의/감성 범위를 다시 선택해 주세요.")
        st.caption("선택 결과를 다른 데이터로 대체하지 않고, 실제 0건 상태를 그대로 보여줍니다.")
        st.dataframe(pd.DataFrame([{
            "제품": format_selection(selected_filters["products"], empty_label="선택 없음"),
            "국가": format_selection(selected_filters["regions"], empty_label="선택 없음"),
            "브랜드": format_selection(selected_filters["brands"], empty_label="선택 없음"),
            "감성": format_selection(selected_filters["sentiments"], empty_label="선택 없음"),
            "CEJ": format_selection(selected_filters["cej"], empty_label="선택 없음"),
            "분석 유형": selected_filters.get("analysis_scope", "전체"),
            "LG 관련성 범위": selected_filters.get("relevance_scope", "전체 수집"),
            "키워드": selected_filters["keyword_query"] or "-",
        }]), hide_index=True, width="stretch")
        return

    validity_series = filtered_comments["comment_validity"] if "comment_validity" in filtered_comments.columns else pd.Series("valid", index=filtered_comments.index)
    valid_base = filtered_comments[validity_series.eq("valid")].copy()
    positive_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("positive").mean() * 100)
    negative_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("negative").mean() * 100)
    neutral_ratio = 0.0 if valid_base.empty else (valid_base["sentiment_label"].eq("neutral").mean() * 100)

    monitoring_df = data.get("monitoring_summary", pd.DataFrame()).copy()
    reporting_df = data.get("reporting_summary", pd.DataFrame()).copy()
    if not monitoring_df.empty:
        if products_active and "product" in monitoring_df.columns:
            monitoring_df = monitoring_df[monitoring_df["product"].isin(selected_filters["products"])]
        if regions_active and "region" in monitoring_df.columns:
            region_codes = region_codes_from_labels(selected_filters["regions"])
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                monitoring_df = monitoring_df[
                    monitoring_df["region"].isin(explicit_codes) | ~monitoring_df["region"].isin(REGION_CODES)
                ]
            else:
                monitoring_df = monitoring_df[monitoring_df["region"].isin(region_codes)]
    if not reporting_df.empty:
        if products_active and "product" in reporting_df.columns:
            reporting_df = reporting_df[reporting_df["product"].isin(selected_filters["products"])]
        if regions_active and "region" in reporting_df.columns:
            region_codes = region_codes_from_labels(selected_filters["regions"])
            if REGION_OTHER_LABEL in region_codes:
                explicit_codes = [code for code in region_codes if code != REGION_OTHER_LABEL]
                reporting_df = reporting_df[
                    reporting_df["region"].isin(explicit_codes) | ~reporting_df["region"].isin(REGION_CODES)
                ]
            else:
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
        st.caption("현재 필터 결과를 그대로 내려받습니다. 화면에서 보이는 핵심 데이터와 동일한 범위를 엑셀로 확인할 수 있습니다.")
        st.download_button(
            "전체 Raw Data 엑셀 다운로드",
            data=build_raw_download_package(raw_comments_export, filtered_videos, weekly_window, cej_df, brand_df, density_df),
            file_name="voc_dashboard_raw_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            width="stretch",
        )


    tab_comments, tab_videos = st.tabs(["댓글 VoC 대시보드", "영상 분석 요약"])

    with tab_comments:
        st.caption("먼저 핵심 요약을 확인한 뒤, 아래 차트와 대표 코멘트에서 어떤 이슈가 실제로 반복되는지 내려가며 읽을 수 있습니다.")
        row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
        with row1_col1:
            render_card("분석 대상 댓글", f"{len(filtered_comments):,}", accent="blue")
        with row1_col2:
            render_card("유효 댓글 수", f"{bundle['meaningful_count']:,}", accent="blue",
                        subtitle=f"전체의 {bundle['meaningful_count'] / max(len(filtered_comments), 1) * 100:.0f}%")
        with row1_col3:
            render_card("문의 댓글 수", f"{bundle.get('inquiry_count', 0):,}", accent="purple")
        with row1_col4:
            render_card("제외 댓글 수", f"{bundle['removed_count']:,}", accent="orange")
        if bundle.get("relevance_scope") == "LG 관련만":
            pre_count = int(bundle.get("pre_lg_comment_count", len(filtered_comments)))
            post_count = int(bundle.get("post_lg_comment_count", len(filtered_comments)))
            ratio = (post_count / pre_count * 100.0) if pre_count else 0.0
            st.caption(f"전체 수집: {pre_count:,}건 → LG 관련: {post_count:,}건 ({ratio:.1f}%)")

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            render_card("긍정 비율", f"{positive_ratio:.1f}%", accent="green")
        with row2_col2:
            render_card("부정 비율", f"{negative_ratio:.1f}%", accent="orange")
        with row2_col3:
            render_card("중립 비율", f"{neutral_ratio:.1f}%", accent="purple")

        with st.expander("분석 운영 지표 보기"):
            st.caption("아래 지표는 데이터 가공 상태와 KPI 계산 범위를 점검하기 위한 운영용 보조 지표입니다.")
            row3_col1, row3_col2, row3_col3, row3_col4, row3_col5 = st.columns(5)
            with row3_col1:
                render_card("원천 콘텐츠 수", f"{int(monitoring_row.get('source_count', 0) or 0):,}", accent="blue")
            with row3_col2:
                render_card("의견 단위 수", f"{int(monitoring_row.get('opinion_count', 0) or 0):,}", accent="blue")
            with row3_col3:
                render_card("분리 비율", f"{float(monitoring_row.get('split_rate', 0) or 0):.1%}", accent="purple")
            with row3_col4:
                render_card("중립 비율", f"{float(monitoring_row.get('neutral_rate', 0) or 0):.1%}", accent="purple")
            with row3_col5:
                render_card("제외 비율", f"{float(monitoring_row.get('exclude_rate', 0) or 0):.1%}", accent="orange")

            row4_col1, row4_col2, row4_col3 = st.columns(3)
            with row4_col1:
                render_card("KPI 의견 수", f"{int(reporting_row.get('kpi_opinion_count', 0) or 0):,}", accent="blue")
            with row4_col2:
                render_card("KPI 긍정 비율", f"{float(reporting_row.get('positive_share', 0) or 0):.1%}", accent="green")
            with row4_col3:
                render_card("KPI 부정 비율", f"{float(reporting_row.get('negative_share', 0) or 0):.1%}", accent="orange")


        with st.container(border=True):
            st.markdown("### 주차별 긍정/부정 응답률")
            chart_ctrl_col1, chart_ctrl_col2, chart_ctrl_spacer = st.columns([1, 1, 3])
            with chart_ctrl_col1:
                weeks_per_view = st.slider(
                    "표시 주차 수",
                    min_value=WEEKLY_CHART_MIN,
                    max_value=WEEKLY_CHART_MAX,
                    value=WEEKLY_CHART_DEFAULT,
                    step=WEEKLY_CHART_STEP,
                )
            with chart_ctrl_col2:
                page = st.number_input("주차 페이지", min_value=1, value=1, step=1)
            del chart_ctrl_spacer
            weekly_window, total_pages = build_weekly_sentiment_window(filtered_comments, weeks_per_view, page)
            page = min(max(int(page), 1), int(total_pages))
            if not weekly_window.empty and float(weekly_window["댓글수"].sum()) > 0:
                week_order = weekly_window[[COL_WEEK_LABEL, "주차축약"]].drop_duplicates().sort_values(COL_WEEK_LABEL)
                axis_order = week_order["주차축약"].tolist()
                bar_size = max(8, min(16, int(180 / max(len(axis_order), 1))))
                chart = alt.Chart(weekly_window).mark_bar(size=bar_size, cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
                    x=alt.X(
                        "주차축약:N",
                        title="주차 시작일",
                        sort=axis_order,
                        axis=alt.Axis(
                            labelAngle=-35, labelLimit=56, labelPadding=10, titlePadding=16,
                            labelColor="#64748b", titleColor="#475569", gridColor="#f1f5f9",
                        ),
                    ),
                    y=alt.Y(
                        "비율:Q", title="응답률",
                        axis=alt.Axis(format="%", labelColor="#64748b", titleColor="#475569", gridColor="#f1f5f9"),
                    ),
                    xOffset=alt.XOffset(f"{COL_SENTIMENT_CODE}:N", sort=["positive", "negative"]),
                    color=alt.Color(
                        f"{COL_SENTIMENT_CODE}:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#3b82f6", "#f97316"]),
                        legend=alt.Legend(
                            title=None,
                            orient="top",
                            direction="horizontal",
                            symbolType="square",
                            labelExpr="datum.label === 'positive' ? '긍정' : datum.label === 'negative' ? '부정' : datum.label",
                            labelColor="#475569",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(f"{COL_WEEK_LABEL}:N", title="주차 시작일"),
                        alt.Tooltip(f"{COL_SENTIMENT}:N", title=COL_SENTIMENT),
                        alt.Tooltip("댓글수:Q", title="댓글 수"),
                        alt.Tooltip("비율:Q", title="응답률", format=".1%"),
                    ],
                ).properties(height=340).configure_view(strokeWidth=0)
                st.altair_chart(chart, width="stretch")
                st.caption(f"시간순으로 정렬된 최근 {len(axis_order)}개 주차를 표시합니다. 전체 {total_pages}페이지 중 {min(page, total_pages)}페이지입니다.")
            else:
                st.info("해당 조건에서 주간 차트를 그릴 데이터가 없습니다.")

        with st.container(border=True):
            render_keyword_panel(filtered_comments, "negative", "부정 키워드 비중")

        with st.container(border=True):
            render_keyword_panel(filtered_comments, "positive", "긍정 키워드 비중")

        with st.container(border=True):
            render_nlp_insights(data, filtered_comments)

        # Insight type distribution (video_context 기반)
        if "insight_type" in filtered_comments.columns:
            insight_counts = filtered_comments["insight_type"].dropna().value_counts()
            insight_counts = insight_counts[insight_counts.index != "일반_의견"]
            if not insight_counts.empty:
                with st.container(border=True):
                    st.markdown("### 인사이트 유형 분포")
                    st.caption("영상 맥락(프로모션/리뷰/비교 등)과 댓글 내용을 결합하여 비즈니스 인사이트 유형을 자동 분류합니다.")
                    insight_df = insight_counts.reset_index()
                    insight_df.columns = ["인사이트 유형", "건수"]
                    chart = alt.Chart(insight_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#7c3aed").encode(
                        x=alt.X("건수:Q", title="건수"),
                        y=alt.Y("인사이트 유형:N", sort="-x", title=""),
                        tooltip=["인사이트 유형", "건수"],
                    ).properties(height=max(120, len(insight_counts) * 32))
                    st.altair_chart(chart, width="stretch")

        with st.container(border=True):
            st.markdown("### CEJ 단계별 고객 신뢰도")
            cej_trust_df = build_cej_trust_frame(filtered_comments)
            if not cej_trust_df.empty:
                st.caption("각 고객경험여정 단계에서 제품·국가·브랜드별 긍정/부정 반응을 바탕으로 고객 신뢰도 지수를 계산했습니다.")
                st.caption("※ 고객 신뢰도 지수 = ((긍정 댓글 수 − 부정 댓글 수) ÷ (긍정 댓글 수 + 부정 댓글 수)) × 50 + 50")
                st.dataframe(
                    cej_trust_df[[COL_CEJ, "product", COL_COUNTRY, COL_BRAND, "댓글 수", "긍정 댓글 수", "부정 댓글 수", "고객 신뢰도 지수"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("현재 조건에서 표시할 CEJ 신뢰도 데이터가 없습니다.")

        with st.container(border=True):
            st.markdown("### 브랜드 언급 비중")
            brand_trust_df = build_brand_trust_frame(filtered_comments)
            if not brand_trust_df.empty:
                st.caption("브랜드가 언급된 댓글의 양과 긍정/부정 분포를 함께 보며 고객 신뢰도 지수를 참고할 수 있습니다.")
                st.caption("※ 고객 신뢰도 지수 = ((긍정 댓글 수 − 부정 댓글 수) ÷ (긍정 댓글 수 + 부정 댓글 수)) × 50 + 50")
                st.dataframe(
                    brand_trust_df[[COL_BRAND, "product", COL_COUNTRY, "언급 댓글 수", "긍정 댓글 수", "부정 댓글 수", "언급 비율", "고객 신뢰도 지수"]],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("브랜드 신뢰 점수를 계산할 수 있는 데이터가 아직 충분하지 않습니다.")

        inquiry_units = bundle.get("opinion_units", pd.DataFrame())
        if not inquiry_units.empty and "inquiry_flag" in inquiry_units.columns:
            inquiry_units = inquiry_units[inquiry_units["inquiry_flag"].fillna(False)].copy()
        if not inquiry_units.empty:
            with st.container(border=True):
                st.markdown("### 반복 문의 코멘트")
                display_inquiry = inquiry_units.rename(columns={
                    "product": "제품",
                    "region": "국가",
                    "aspect_key": "고객경험여정",
                    "sentiment_code": "감성",
                    "opinion_text": "의견 문장",
                    "classification_type": "분류 유형",
                })
                keep_cols = [col for col in ["제품", "국가", "고객경험여정", "감성", "의견 문장", "분류 유형"] if col in display_inquiry.columns]
                st.dataframe(display_inquiry[keep_cols].head(20), width="stretch", hide_index=True)
                st.caption("문의 여부는 감성 라벨과 별도로 유지합니다. 반복 문의는 불만과 별개의 운영 개선 과제로 추적하세요.")

        with st.container(border=True):
            render_representative_comments(
                bundle.get("representative_comments", filtered_comments),
                bundle.get("representative_bundles", pd.DataFrame()),
                bundle.get("opinion_units", pd.DataFrame()),
            )

    with tab_videos:
        render_video_summary_page(all_comments, filtered_videos)


if __name__ == "__main__":
    main()

