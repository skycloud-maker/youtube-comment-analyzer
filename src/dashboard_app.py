"""Streamlit dashboard for appliance VoC exploration."""

from __future__ import annotations

from io import BytesIO, StringIO
import codecs
from collections import Counter
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
SNAPSHOT_PREF_FILE = CACHE_DIR / "last_snapshot_pref.json"
DEFAULT_SAMPLE_STATE_FILE = CACHE_DIR / "default_sample_state.json"
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
    ("analysis_non_trash", "analysis_non_trash.parquet"),
    ("analysis_strategic", "analysis_strategic.parquet"),
    ("nlp_topic_insights", "nlp_topic_insights.parquet"),
    ("nlp_keyword_insights", "nlp_keyword_insights.parquet"),
    ("nlp_inquiry_summary", "nlp_inquiry_summary.parquet"),
    ("nlp_product_mentions", "nlp_product_mentions.parquet"),
    ("nlp_sentiment_distribution", "nlp_sentiment_distribution.parquet"),
    ("strategy_video_insights", "strategy_video_insights.parquet"),
    ("strategy_video_insight_candidates", "strategy_video_insight_candidates.parquet"),
    ("strategy_product_group_insights", "strategy_product_group_insights.parquet"),
    ("strategy_product_group_insight_candidates", "strategy_product_group_insight_candidates.parquet"),
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
SESSION_RUN_SNAPSHOT_KEY = "dashboard_run_snapshot"
SESSION_SELECTED_VIDEO_ID_KEY = "dashboard_selected_video_id"
RUN_SNAPSHOT_ALL = "__ALL_RUNS__"
STATE_SNAPSHOT_DIR = CACHE_DIR / "saved_states"
STATE_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
UI_STATE_FILE_VERSION = "dashboard_ui_state.v1"
UI_STATE_FILE_VERSION_EMBEDDED = "dashboard_ui_state.v1+embedded"
UI_STATE_EMBEDDED_SCHEMA = "bundle-split-v1"
UI_STATE_EMBEDDED_DEFAULT_LIMIT = 50000
UI_STATE_EMBEDDED_ROW_LIMITS: dict[str, int] = {
    "comments": 50000,
    "analysis_comments": 50000,
    "analysis_non_trash": 50000,
    "opinion_units": 80000,
    "videos": 8000,
}
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
    "analysis_non_trash": 3000,
    "opinion_units": 4500,
    "videos": 400,
}


def _empty_dashboard_bundle() -> dict[str, pd.DataFrame]:
    return {key: pd.DataFrame() for key, _ in DASHBOARD_DATASETS}


def _run_dir_has_required_outputs(run_dir: Path) -> bool:
    return all((run_dir / filename).exists() or (run_dir / Path(filename).with_suffix(".csv")).exists() for filename in REQUIRED_RESULT_FILES)


def _real_ready_run_dirs() -> list[Path]:
    return [run_dir for run_dir in _iter_dashboard_run_dirs() if _run_dir_has_required_outputs(run_dir)]


def _build_run_snapshot_options(max_items: int = 120) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = [("전체 run (통합)", RUN_SNAPSHOT_ALL)]
    run_dirs = sorted(_real_ready_run_dirs(), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    for run_dir in run_dirs[:max_items]:
        try:
            ts = pd.to_datetime(run_dir.stat().st_mtime, unit="s").strftime("%Y-%m-%d %H:%M")
        except Exception:
            ts = "-"
        options.append((f"{run_dir.name} · {ts}", run_dir.name))
    return options


def _latest_real_run_id() -> str:
    run_dirs = sorted(_real_ready_run_dirs(), key=lambda path: path.stat().st_mtime_ns, reverse=True)
    if not run_dirs:
        return RUN_SNAPSHOT_ALL
    return run_dirs[0].name


def _read_snapshot_preference() -> str:
    try:
        if not SNAPSHOT_PREF_FILE.exists():
            return ""
        payload = json.loads(SNAPSHOT_PREF_FILE.read_text(encoding="utf-8"))
        return _safe_text(payload.get("snapshot_run_id", "")).strip()
    except Exception:
        return ""


def _write_snapshot_preference(snapshot_run_id: str) -> None:
    run_id = _safe_text(snapshot_run_id).strip()
    payload = {
        "snapshot_run_id": run_id or RUN_SNAPSHOT_ALL,
        "updated_at": pd.Timestamp.utcnow().isoformat(),
    }
    try:
        SNAPSHOT_PREF_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        # Preference persistence must never break dashboard rendering.
        return


def _parse_keyword_terms(raw_query: str) -> list[str]:
    raw = _safe_text(raw_query).strip()
    if not raw:
        return []
    parts = [segment.strip() for segment in raw.split(",")]
    terms: list[str] = []
    seen: set[str] = set()
    for item in parts:
        if not item:
            continue
        key = item.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(item)
    return terms


def _build_keyword_mask(frame: pd.DataFrame, terms: list[str], mode: str) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    if not terms:
        return pd.Series(True, index=frame.index)
    merged_text = (
        frame.get("cleaned_text", pd.Series("", index=frame.index)).fillna("").astype(str)
        + " "
        + frame.get("text_display", pd.Series("", index=frame.index)).fillna("").astype(str)
    ).str.lower()
    term_masks = [merged_text.str.contains(re.escape(term.lower()), regex=True, na=False) for term in terms]
    if not term_masks:
        return pd.Series(True, index=frame.index)
    if str(mode or "AND").upper() == "OR":
        combined = term_masks[0].copy()
        for mask in term_masks[1:]:
            combined = combined | mask
        return combined
    combined = term_masks[0].copy()
    for mask in term_masks[1:]:
        combined = combined & mask
    return combined


def _safe_snapshot_filename(label: str) -> str:
    normalized = re.sub(r"[^\w\-가-힣]+", "_", _safe_text(label)).strip("_")
    if not normalized:
        normalized = "saved_state"
    return normalized[:80]


def _list_saved_ui_states() -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in sorted(STATE_SNAPSHOT_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            label = _safe_text(payload.get("label", path.stem)) or path.stem
            created_at = _safe_text(payload.get("saved_at", ""))
            suffix = ""
            if created_at:
                try:
                    suffix = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M")
                except Exception:
                    suffix = created_at[:16]
            display = f"{label} · {suffix}" if suffix else label
            items.append((display, path.name))
        except Exception:
            continue
    return items


def _validate_ui_state_payload(payload: Any) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "상태 파일 형식이 올바르지 않습니다. (ui_state 객체 누락)"

    mode = _safe_text(payload.get("mode", "")).lower()
    if mode not in {"sample", "real"}:
        return False, "상태 파일의 데이터 모드가 올바르지 않습니다."

    snapshot_run = _safe_text(payload.get("snapshot_run", RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
    if not snapshot_run:
        return False, "상태 파일의 스냅샷 정보가 올바르지 않습니다."

    filters = payload.get("filters")
    if not isinstance(filters, dict):
        return False, "상태 파일의 필터 정보가 누락되었습니다."

    required_keys = [
        "filter_products",
        "filter_regions",
        "filter_brands",
        "filter_sentiments",
        "filter_cej",
        "filter_keyword_query",
        "filter_keyword_mode",
        "filter_analysis_scope",
        "filter_relevance_scope",
    ]
    missing_keys = [key for key in required_keys if key not in filters]
    if missing_keys:
        return False, f"상태 파일의 필수 필드가 누락되었습니다: {', '.join(missing_keys)}"

    for key in ["filter_products", "filter_regions", "filter_brands", "filter_sentiments", "filter_cej"]:
        if not isinstance(filters.get(key), list):
            return False, f"상태 파일의 `{key}` 형식이 올바르지 않습니다."

    keyword_mode = _safe_text(filters.get("filter_keyword_mode", "AND")).upper()
    if keyword_mode not in {"AND", "OR"}:
        return False, "상태 파일의 키워드 결합 규칙이 올바르지 않습니다."

    return True, ""


def _bundle_row_limit(dataset_key: str) -> int:
    return int(UI_STATE_EMBEDDED_ROW_LIMITS.get(dataset_key, UI_STATE_EMBEDDED_DEFAULT_LIMIT))


def _serialize_state_bundle(bundle: dict[str, pd.DataFrame]) -> tuple[dict[str, Any], dict[str, int]]:
    datasets: dict[str, Any] = {}
    exported_rows: dict[str, int] = {}
    for key, _ in DASHBOARD_DATASETS:
        frame = bundle.get(key, pd.DataFrame())
        if frame is None or frame.empty:
            continue
        limit = _bundle_row_limit(key)
        sliced = frame.head(limit).copy()
        try:
            split_obj = json.loads(sliced.to_json(orient="split", date_format="iso", force_ascii=False))
        except Exception:
            continue
        datasets[key] = split_obj
        exported_rows[key] = int(len(sliced))
    return datasets, exported_rows


def _deserialize_state_bundle(doc: Any) -> tuple[dict[str, pd.DataFrame], str]:
    if not isinstance(doc, dict):
        return {}, "임베디드 데이터 형식이 올바르지 않습니다."
    schema = _safe_text(doc.get("schema", "")).strip()
    if schema and schema != UI_STATE_EMBEDDED_SCHEMA:
        return {}, f"지원하지 않는 임베디드 데이터 스키마입니다: {schema}"
    datasets_obj = doc.get("datasets")
    if not isinstance(datasets_obj, dict):
        return {}, "임베디드 데이터셋 구조가 올바르지 않습니다."

    bundle = _empty_dashboard_bundle()
    for key, _ in DASHBOARD_DATASETS:
        obj = datasets_obj.get(key)
        if not isinstance(obj, dict):
            continue
        try:
            frame = pd.read_json(StringIO(json.dumps(obj, ensure_ascii=False)), orient="split")
            bundle[key] = frame
        except Exception:
            return {}, f"임베디드 데이터셋 복원 실패: {key}"
    return bundle, ""


def _has_usable_embedded_bundle(bundle: dict[str, pd.DataFrame] | None) -> bool:
    if not bundle:
        return False
    comments = bundle.get("comments", pd.DataFrame())
    videos = bundle.get("videos", pd.DataFrame())
    return not comments.empty and not videos.empty


def _build_ui_state_document(
    label: str,
    payload: dict[str, Any],
    *,
    embedded_bundle: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    mode = _safe_text(payload.get("mode", "real")).lower()
    snapshot_run = _safe_text(payload.get("snapshot_run", RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
    doc: dict[str, Any] = {
        "version": UI_STATE_FILE_VERSION,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
        "label": _safe_text(label) or "saved_state",
        "data_mode": mode if mode in {"sample", "real"} else "real",
        "snapshot_run_id": snapshot_run,
        "ui_state": payload,
    }
    if embedded_bundle:
        datasets_obj, exported_rows = _serialize_state_bundle(embedded_bundle)
        if datasets_obj:
            doc["version"] = UI_STATE_FILE_VERSION_EMBEDDED
            doc["embedded_data"] = {
                "schema": UI_STATE_EMBEDDED_SCHEMA,
                "datasets": datasets_obj,
                "row_counts": exported_rows,
            }
    return doc


def _extract_ui_state_document(doc: Any) -> tuple[bool, dict[str, Any], dict[str, pd.DataFrame] | None, str]:
    if not isinstance(doc, dict):
        return False, {}, None, "상태 파일이 JSON 객체 형식이 아닙니다."

    version = _safe_text(doc.get("version", "")).strip()
    allowed_versions = {UI_STATE_FILE_VERSION, UI_STATE_FILE_VERSION_EMBEDDED}
    if version and version not in allowed_versions:
        return False, {}, None, f"지원하지 않는 상태 파일 버전입니다: {version}"

    payload = doc.get("ui_state")
    if not isinstance(payload, dict):
        # Backward compatibility for old internal save format.
        payload = doc.get("payload")
    if not isinstance(payload, dict):
        return False, {}, None, "상태 파일에 `ui_state`가 없습니다."

    ok, message = _validate_ui_state_payload(payload)
    if not ok:
        return False, {}, None, message

    embedded_bundle: dict[str, pd.DataFrame] | None = None
    if isinstance(doc.get("embedded_data"), dict):
        embedded_bundle, embedded_error = _deserialize_state_bundle(doc.get("embedded_data"))
        if embedded_error:
            return False, {}, None, embedded_error
    return True, payload, embedded_bundle, ""


def _extract_ui_state_payload(doc: Any) -> tuple[bool, dict[str, Any], str]:
    ok, payload, _, message = _extract_ui_state_document(doc)
    return ok, payload, message


def _resolve_default_sample_state_file() -> Path:
    env_path = _safe_text(os.getenv("VOC_DEFAULT_SAMPLE_STATE_FILE", ""))
    if env_path:
        candidate = Path(env_path).expanduser()
        return candidate if candidate.is_absolute() else (BASE_DIR / candidate).resolve()
    return DEFAULT_SAMPLE_STATE_FILE


def _read_ui_state_document_from_path(path: Path) -> tuple[bool, dict[str, Any], dict[str, pd.DataFrame] | None, str]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return False, {}, None, "기본 샘플 상태 파일이 없습니다."
    except UnicodeDecodeError:
        return False, {}, None, "기본 샘플 상태 파일 인코딩이 올바르지 않습니다."
    except json.JSONDecodeError:
        return False, {}, None, "기본 샘플 상태 파일이 유효한 JSON 형식이 아닙니다."
    except Exception as exc:
        return False, {}, None, f"기본 샘플 상태 파일을 읽지 못했습니다: {exc}"
    return _extract_ui_state_document(doc)


def _apply_default_sample_state() -> tuple[bool, str]:
    default_path = _resolve_default_sample_state_file()
    if not default_path.exists():
        return False, ""

    ok, payload, embedded_bundle, message = _read_ui_state_document_from_path(default_path)
    if not ok:
        return False, f"기본 샘플 상태 파일 적용 실패: {message}"

    sample_payload = dict(payload)
    sample_payload["mode"] = "sample"
    sample_payload["snapshot_run"] = RUN_SNAPSHOT_ALL

    can_restore, restore_message = _validate_restore_data_availability(
        sample_payload,
        embedded_bundle=embedded_bundle,
    )
    if not can_restore:
        return False, f"기본 샘플 상태 파일 적용 실패: {restore_message}"

    _apply_saved_ui_state(sample_payload, embedded_bundle=embedded_bundle)
    return True, f"샘플 기본 상태 적용: {default_path.name}"


def _read_uploaded_ui_state(uploaded_file: Any) -> tuple[bool, dict[str, Any], dict[str, pd.DataFrame] | None, str]:
    if uploaded_file is None:
        return False, {}, None, "업로드된 파일이 없습니다."
    try:
        raw = uploaded_file.getvalue()
        text = raw.decode("utf-8-sig")
        doc = json.loads(text)
    except UnicodeDecodeError:
        return False, {}, None, "상태 파일 인코딩이 올바르지 않습니다. UTF-8 JSON 파일만 지원합니다."
    except json.JSONDecodeError:
        return False, {}, None, "유효한 JSON 파일이 아닙니다."
    except Exception as exc:
        return False, {}, None, f"상태 파일을 읽지 못했습니다: {exc}"
    return _extract_ui_state_document(doc)


def _validate_restore_data_availability(
    payload: dict[str, Any],
    *,
    embedded_bundle: dict[str, pd.DataFrame] | None = None,
) -> tuple[bool, str]:
    mode = _safe_text(payload.get("mode", "real")).lower()
    snapshot_run = _safe_text(payload.get("snapshot_run", RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL

    if mode == "sample":
        # Sample bundle is always available via synthetic fallback.
        return True, ""

    if _has_usable_embedded_bundle(embedded_bundle):
        return True, ""

    ready_runs = _real_ready_run_dirs()
    if not ready_runs:
        return (
            False,
            "이 상태 파일은 화면 설정만 포함하며, 현재 환경에 필요한 실데이터가 없습니다.\n"
            "먼저 분석 실행 또는 샘플 로드를 해주세요.\n"
            "현재 파일만으로는 원본 댓글/영상 데이터를 복원할 수 없습니다.",
        )

    if snapshot_run != RUN_SNAPSHOT_ALL:
        ready_run_ids = {run_dir.name for run_dir in ready_runs}
        if snapshot_run not in ready_run_ids:
            return (
                False,
                f"요청한 스냅샷(run_id=`{snapshot_run}`)이 현재 환경에 없습니다.\n"
                "먼저 해당 run 결과를 다시 생성하거나, 다른 저장 상태를 선택해 주세요.\n"
                "현재 파일만으로는 원본 댓글/영상 데이터를 복원할 수 없습니다.",
            )

    return True, ""


def _write_saved_ui_state(label: str, payload: dict[str, Any]) -> tuple[bool, str]:
    filename = _safe_snapshot_filename(label)
    target = STATE_SNAPSHOT_DIR / f"{filename}.json"
    doc = _build_ui_state_document(label or filename, payload)
    try:
        target.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        return False, f"저장 실패: {exc}"
    return True, str(target)


def _read_saved_ui_state(file_name: str) -> dict[str, Any]:
    file_name = _safe_text(file_name)
    if not file_name:
        return {}
    target = (STATE_SNAPSHOT_DIR / file_name).resolve()
    if not target.exists():
        return {}
    try:
        doc = json.loads(target.read_text(encoding="utf-8"))
        ok, payload, _ = _extract_ui_state_payload(doc)
        return payload if ok else {}
    except Exception:
        return {}


def _collect_current_ui_state(active_mode: str, active_snapshot_run: str) -> dict[str, Any]:
    return {
        "mode": _safe_text(active_mode) or "real",
        "snapshot_run": _safe_text(active_snapshot_run) or RUN_SNAPSHOT_ALL,
        "filters": {
            "filter_products": list(st.session_state.get("filter_products", [])),
            "filter_regions": list(st.session_state.get("filter_regions", [])),
            "filter_brands": list(st.session_state.get("filter_brands", [])),
            "filter_sentiments": list(st.session_state.get("filter_sentiments", [])),
            "filter_cej": list(st.session_state.get("filter_cej", [])),
            "filter_keyword_query": _safe_text(st.session_state.get("filter_keyword_query", "")),
            "filter_keyword_mode": _safe_text(st.session_state.get("filter_keyword_mode", "AND")) or "AND",
            "filter_analysis_scope": _safe_text(st.session_state.get("filter_analysis_scope", "전체")) or "전체",
            "filter_relevance_scope": _safe_text(st.session_state.get("filter_relevance_scope", "전체 수집")) or "전체 수집",
        },
    }


def _apply_saved_ui_state(payload: dict[str, Any], *, embedded_bundle: dict[str, pd.DataFrame] | None = None) -> None:
    if not isinstance(payload, dict):
        return
    filter_payload = payload.get("filters", {}) if isinstance(payload.get("filters"), dict) else {}
    for key in [
        "filter_products",
        "filter_regions",
        "filter_brands",
        "filter_sentiments",
        "filter_cej",
        "filter_keyword_query",
        "filter_keyword_mode",
        "filter_analysis_scope",
        "filter_relevance_scope",
    ]:
        if key in filter_payload:
            st.session_state[key] = filter_payload[key]
    restored_mode = _safe_text(filter_payload.get("filter_keyword_mode", st.session_state.get("filter_keyword_mode", "AND"))).upper()
    st.session_state["filter_keyword_mode_label"] = "OR(하나 이상)" if restored_mode == "OR" else "AND(모두 포함)"

    mode = _safe_text(payload.get("mode", "real")).lower()
    snapshot_run = _safe_text(payload.get("snapshot_run", RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
    if mode == "sample":
        _set_data_mode("sample", force_refresh=False, snapshot_run_id=RUN_SNAPSHOT_ALL)
    elif _has_usable_embedded_bundle(embedded_bundle) and not _real_ready_run_dirs():
        st.session_state[SESSION_DATA_MODE_KEY] = "real"
        st.session_state[SESSION_RUN_SNAPSHOT_KEY] = RUN_SNAPSHOT_ALL
        st.session_state[SESSION_DATA_BUNDLE_KEY] = embedded_bundle
        st.session_state[SESSION_DATA_SIGNATURE_KEY] = _expected_mode_signature("real", snapshot_run_id=RUN_SNAPSHOT_ALL)
        try:
            st.query_params["data_mode"] = "real"
            if "run_snapshot" in st.query_params:
                del st.query_params["run_snapshot"]
        except Exception:
            pass
    else:
        _set_data_mode("real", force_refresh=False, snapshot_run_id=snapshot_run)


def _filter_bundle_by_snapshot(bundle: dict[str, pd.DataFrame], snapshot_run_id: str) -> dict[str, pd.DataFrame]:
    selected = _safe_text(snapshot_run_id).strip()
    if not selected or selected == RUN_SNAPSHOT_ALL:
        return bundle

    filtered: dict[str, pd.DataFrame] = {}
    for key, frame in bundle.items():
        if frame is None or frame.empty:
            filtered[key] = frame
            continue
        if "run_id" not in frame.columns:
            filtered[key] = frame
            continue
        scoped = frame[frame["run_id"].astype(str).eq(selected)].copy()
        filtered[key] = scoped
    return filtered


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
    bundle["analysis_non_trash"] = comments.copy()
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


def _expected_mode_signature(mode: str, snapshot_run_id: str = RUN_SNAPSHOT_ALL) -> str:
    source_sig = _latest_signature()
    if mode == "real":
        return f"{mode}:{snapshot_run_id}:{source_sig}"
    return f"{mode}:{source_sig}"


def _load_bundle_for_mode(mode: str, *, force_refresh: bool = False) -> dict[str, pd.DataFrame]:
    if force_refresh:
        _clear_dashboard_caches()
    if mode == "sample":
        return load_sample_dashboard_data()
    return get_dashboard_data_resource()


def _initialize_data_session_state() -> None:
    requested_mode = ""
    requested_snapshot = ""
    try:
        requested_mode = str(st.query_params.get("data_mode", st.query_params.get("mode", ""))).strip().lower()
        requested_snapshot = str(st.query_params.get("run_snapshot", "")).strip()
    except Exception:
        requested_mode = ""
        requested_snapshot = ""
    default_mode = "real" if (_real_ready_run_dirs() or CLOUD_CACHE_FILE.exists()) else "sample"
    if os.getenv("PYTEST_CURRENT_TEST"):
        default_mode = "sample"
    if requested_mode in {"sample", "lite"}:
        default_mode = "sample"
    elif requested_mode == "real":
        default_mode = "real"

    if SESSION_DATA_MODE_KEY not in st.session_state:
        st.session_state[SESSION_DATA_MODE_KEY] = default_mode
    if SESSION_RUN_SNAPSHOT_KEY not in st.session_state:
        preferred_snapshot = requested_snapshot or _read_snapshot_preference()
        st.session_state[SESSION_RUN_SNAPSHOT_KEY] = preferred_snapshot or RUN_SNAPSHOT_ALL
    if st.session_state[SESSION_DATA_MODE_KEY] == "real":
        valid_snapshots = {value for _, value in _build_run_snapshot_options()}
        current_snapshot = _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
        if current_snapshot not in valid_snapshots:
            st.session_state[SESSION_RUN_SNAPSHOT_KEY] = _latest_real_run_id()
    if SESSION_DATA_BUNDLE_KEY not in st.session_state:
        mode = st.session_state[SESSION_DATA_MODE_KEY]
        snapshot_run_id = _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
        bundle = _load_bundle_for_mode(mode, force_refresh=False)
        if mode == "real":
            bundle = _filter_bundle_by_snapshot(bundle, snapshot_run_id)
        st.session_state[SESSION_DATA_BUNDLE_KEY] = bundle
        st.session_state[SESSION_DATA_SIGNATURE_KEY] = _expected_mode_signature(mode, snapshot_run_id=snapshot_run_id)


def _set_data_mode(mode: str, *, force_refresh: bool = False, snapshot_run_id: str | None = None) -> None:
    selected_snapshot = _safe_text(snapshot_run_id) or _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
    if mode != "real":
        selected_snapshot = RUN_SNAPSHOT_ALL
    else:
        valid_snapshots = {value for _, value in _build_run_snapshot_options()}
        if selected_snapshot not in valid_snapshots:
            selected_snapshot = _latest_real_run_id()
    bundle = _load_bundle_for_mode(mode, force_refresh=force_refresh)
    if mode == "real":
        bundle = _filter_bundle_by_snapshot(bundle, selected_snapshot)
    st.session_state[SESSION_DATA_MODE_KEY] = mode
    st.session_state[SESSION_DATA_BUNDLE_KEY] = bundle
    st.session_state[SESSION_RUN_SNAPSHOT_KEY] = selected_snapshot
    st.session_state[SESSION_DATA_SIGNATURE_KEY] = _expected_mode_signature(mode, snapshot_run_id=selected_snapshot)
    if mode == "real":
        _write_snapshot_preference(selected_snapshot)
    try:
        st.query_params["data_mode"] = mode
        if mode == "real" and selected_snapshot != RUN_SNAPSHOT_ALL:
            st.query_params["run_snapshot"] = selected_snapshot
        elif "run_snapshot" in st.query_params:
            del st.query_params["run_snapshot"]
    except Exception:
        # Query params sync is best-effort only.
        pass


def _analysis_target_regions(region: str) -> list[str]:
    selected = str(region or "").strip().upper()
    all_regions = [code for _, code in ANALYSIS_MARKET_OPTIONS if code != ANALYSIS_ALL_MARKET_CODE]
    if selected == ANALYSIS_ALL_MARKET_CODE:
        return all_regions
    if selected in all_regions:
        return [selected]
    return ["KR"]


def _parse_run_keywords(value: str) -> list[str]:
    parsed = _parse_keyword_terms(value)
    return parsed if parsed else ([_safe_text(value).strip()] if _safe_text(value).strip() else [])


def _run_real_analysis_action(
    keyword: str,
    max_videos: int,
    comments_per_video: int,
    language: str,
    region: str,
    output_prefix: str,
    keyword_match_mode: str = "OR",
) -> tuple[bool, str]:
    target_regions = _analysis_target_regions(region)
    run_logs: list[str] = []
    seeds = _parse_run_keywords(keyword)
    if not seeds:
        return False, "검색 키워드가 비어 있습니다."
    match_mode = _safe_text(keyword_match_mode).upper() or "OR"

    for target_region in target_regions:
        keyword_variants: list[str] = []
        seen: set[str] = set()
        seed_inputs = [" ".join(seeds)] if match_mode == "AND" and len(seeds) > 1 else seeds
        for seed in seed_inputs:
            for variant in _build_search_keyword_variants(seed, language=language, region=target_region):
                key = variant.casefold()
                if key in seen:
                    continue
                seen.add(key)
                keyword_variants.append(variant)
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
        if match_mode != "AND" and len(keyword_variants) > 1:
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
    hard_block_keywords = {
        "커서", "밤에", "지금", "요즘", "이번", "그냥", "진짜", "정말", "약간",
        "사람들", "이유", "이게", "그게", "저게", "못쓰겠고",
        "라고", "보니", "있고요", "좀더", "점점", "기본", "혹시", "대단한",
        "만들었네", "영상보고", "먹을수있고", "사고싶은데", "하지", "추측이긴", "파는",
    }
    low_decision_weight_keywords = {
        "드럼", "통돌이", "세탁기", "건조기", "냉장고", "식기세척기", "틔운", "스타일러", "에어컨", "공기청정기",
        "가전", "가전제품", "제품",
        "사람들", "이유", "지금", "요즘", "이번", "그냥", "진짜", "밤", "밤에", "커서", "때문", "이게", "그게", "못쓰겠고",
        "영상", "리뷰", "기능", "효과", "라고", "보니", "있고요", "좀더", "점점", "직접", "혹시",
    }
    counter = build_keyword_counter(texts)
    rows = [{"keyword": keyword, "count": count} for keyword, count in filter_business_keywords(counter, top_n * 6)]
    keyword_df = pd.DataFrame(rows)
    if keyword_df.empty:
        return keyword_df
    keyword_df["keyword"] = keyword_df["keyword"].map(normalize_keyword)
    keyword_df = keyword_df[keyword_df["keyword"] != ""]
    keyword_df = keyword_df[~keyword_df["keyword"].isin(hard_block_keywords)]
    blocked = {item for item in blocked_keywords if item}
    if blocked:
        keyword_df = keyword_df[~keyword_df["keyword"].isin(blocked)]
    keyword_df = keyword_df.groupby("keyword", as_index=False)["count"].sum()
    keyword_df["_weight"] = 1.0
    keyword_df.loc[keyword_df["keyword"].isin(low_decision_weight_keywords), "_weight"] = 0.12
    keyword_df["_rank_score"] = keyword_df["count"] * keyword_df["_weight"]
    keyword_df = keyword_df.sort_values(["_rank_score", "count"], ascending=[False, False]).head(top_n).drop(columns=["_weight", "_rank_score"])
    return keyword_df


def _build_dynamic_keyword_excludes(comments_df: pd.DataFrame) -> tuple[str, ...]:
    if comments_df.empty:
        return ()

    blocked: set[str] = {
        "커서", "밤에", "지금", "요즘", "이번", "그냥", "진짜", "정말", "약간", "사람들", "이유",
        "있고", "있음", "없는", "없음", "없이", "없어서", "때문", "처럼", "이게", "그게", "저게",
        "왜", "왜요", "왜죠", "어떻게", "뭐지", "뭔가", "못쓰겠고",
        "라고", "보니", "있고요", "좀더", "점점", "기본", "혹시", "대단한",
        "만들었네", "영상보고", "먹을수있고", "사고싶은데", "하지", "추측이긴", "파는",
    }
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
    if "hygiene_class" in working.columns:
        non_trash = working[working["hygiene_class"].fillna("").astype(str).str.lower().ne("trash")].copy()
        if not non_trash.empty:
            working = non_trash
    if "representative_eligibility" in working.columns:
        eligible_mask = working["representative_eligibility"].fillna(False).astype(bool)
        if eligible_mask.any():
            working = working[eligible_mask].copy()
    if working.empty:
        return pd.DataFrame()

    working["_canonical_sentiment"] = (
        working.get("sentiment_final", working.get("sentiment_label", pd.Series("neutral", index=working.index)))
        .fillna(working.get("sentiment_label", pd.Series("neutral", index=working.index)))
        .astype(str)
        .str.lower()
    )

    # Preselect a manageable candidate pool before expensive re-analysis so the
    # representative section stays responsive even with tens of thousands of comments.
    candidate_pool = working.copy()
    if sentiment and "_canonical_sentiment" in candidate_pool.columns:
        sentiment_candidates = candidate_pool[candidate_pool["_canonical_sentiment"] == sentiment].copy()
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
        working.get("_canonical_sentiment", pd.Series("neutral", index=working.index))
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
    working = working.drop(columns=["_canonical_sentiment"], errors="ignore")
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
        .rep-top-card {height: 100%; background: #f8fafc; border: 1px solid #dbeafe; border-radius: 10px; padding: 10px 12px; margin-bottom: 6px;}
        .rep-top-title {font-size: 14px; font-weight: 800; color: #1e293b; margin-bottom: 4px; line-height: 1.3;}
        .rep-top-meta {font-size: 11px; color: #2563eb; margin-bottom: 6px; font-weight: 700;}
        .rep-top-text {font-size: 12px; color: #334155; line-height: 1.5;}

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
    "가전": "가전제품",
    "가전제품": "가전제품",
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
    # 검색 키워드/채널명 같은 비제품 토큰이 제품 필터로 오염되지 않도록
    # 인식 가능한 제품 표기만 유지하고 나머지는 제거한다.
    return ""


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
    "awareness": "\uc778\uc9c0",
    "aware": "\uc778\uc9c0",
    "exploration": "\ud0d0\uc0c9",
    "explore": "\ud0d0\uc0c9",
    "decision": "\uacb0\uc815",
    "decide": "\uacb0\uc815",
    "purchase": "\uad6c\ub9e4",
    "delivery": "\ubc30\uc1a1",
    "deliver": "\ubc30\uc1a1",
    "installation": "\ubc30\uc1a1",
    "install": "\ubc30\uc1a1",
    "onboarding": "\uc0ac\uc6a9\uc900\ube44",
    "on-board": "\uc0ac\uc6a9\uc900\ube44",
    "onboard": "\uc0ac\uc6a9\uc900\ube44",
    "usage": "\uc0ac\uc6a9",
    "use": "\uc0ac\uc6a9",
    "maintenance": "\uad00\ub9ac",
    "maintain": "\uad00\ub9ac",
    "replacement": "\uad50\uccb4",
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


def _nlp_enrichment_health(comments_df: pd.DataFrame) -> tuple[str, float, str]:
    if comments_df.empty:
        return "unknown", 0.0, "댓글 데이터가 없어 NLP 상태를 확인하지 못했습니다."
    n = len(comments_df)
    score = pd.Series(False, index=comments_df.index)
    if "nlp_label" in comments_df.columns:
        score = score | comments_df["nlp_label"].notna()
    if "nlp_summary" in comments_df.columns:
        score = score | comments_df["nlp_summary"].fillna("").astype(str).str.len().ge(8)
    if "nlp_core_points" in comments_df.columns:
        score = score | comments_df["nlp_core_points"].fillna("").astype(str).str.contains(r"[가-힣A-Za-z0-9]{2,}", regex=True)
    if "nlp_sentiment_reason" in comments_df.columns:
        score = score | comments_df["nlp_sentiment_reason"].fillna("").astype(str).str.len().ge(10)

    coverage = float(score.sum()) / max(1, n)
    if coverage >= 0.5:
        return "ok", coverage, f"NLP 분석 필드가 댓글의 {coverage * 100:.1f}%에서 확인됩니다."
    if coverage >= 0.15:
        return "partial", coverage, f"NLP 분석 필드 커버리지가 {coverage * 100:.1f}%로 부분 적용 상태입니다."
    return "fallback", coverage, f"NLP 분석 필드 커버리지가 {coverage * 100:.1f}%로 낮아 원문 기반 해석 비중이 높습니다."


def _parse_list_like(value: Any) -> list[str]:
    if isinstance(value, list):
        values = value
    elif isinstance(value, tuple):
        values = list(value)
    elif hasattr(value, "tolist"):
        try:
            converted = value.tolist()
            if isinstance(converted, list):
                values = converted
            elif isinstance(converted, tuple):
                values = list(converted)
            elif converted is None:
                return []
            else:
                values = [converted]
        except Exception:
            return []
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parsed: Any = None
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
            except Exception:
                parsed = None
        if isinstance(parsed, list):
            values = parsed
        else:
            values = [part.strip() for part in re.split(r"[,\|;/]", raw) if part.strip()]
    else:
        return []
    output: list[str] = []
    for item in values:
        text = _safe_text(item)
        if text:
            output.append(text)
    return output


def _parse_dict_like(value: Any) -> dict[str, float]:
    if isinstance(value, dict):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
            raw = loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}
    else:
        return {}
    parsed: dict[str, float] = {}
    for key, raw_val in raw.items():
        k = _safe_text(key)
        if not k:
            continue
        try:
            parsed[k] = float(raw_val)
        except Exception:
            continue
    return parsed


def _clean_point_tokens(tokens: list[str], limit: int = 5) -> list[str]:
    noise = {
        "해당", "관련", "제품", "가전", "이슈", "문제", "정도", "부분", "그냥", "진짜", "너무",
        "있다", "없다", "이게", "그게", "저게", "이번", "한번", "저번",
    }
    alias = {
        "통돌": "통돌이",
        "통돌이는": "통돌이",
        "통돌이로": "통돌이",
        "통돌이랑": "통돌이",
        "a/s": "A/S",
        "as": "A/S",
        "티비": "TV",
        "tv": "TV",
    }
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in tokens:
        token = normalize_keyword(raw)
        if not token:
            continue
        token = alias.get(token, token)
        token = re.sub(r"(으로|로|은|는|이|가|을|를|와|과|랑|도|만)$", "", token)
        if not token or token in noise or token.lower() in noise:
            continue
        if len(token) < 2:
            continue
        if token in seen:
            continue
        seen.add(token)
        cleaned.append(token)
        if len(cleaned) >= limit:
            break
    return cleaned


def _extract_row_nlp_metadata(row: pd.Series) -> tuple[list[str], list[str], list[str], list[str], dict[str, float], str]:
    core_points = _clean_point_tokens(_parse_list_like(row.get("nlp_core_points", [])), limit=6)
    context_tags = _parse_list_like(row.get("nlp_context_tags", []))[:6]
    similarity_keys = _parse_list_like(row.get("nlp_similarity_keys", []))[:10]
    confidence_factors = _parse_list_like(row.get("nlp_confidence_factors", []))[:8]
    confidence_breakdown = _parse_dict_like(row.get("nlp_confidence_breakdown", {}))
    insight_summary = _safe_text(row.get("nlp_insight_summary", ""))
    return core_points, context_tags, similarity_keys, confidence_factors, confidence_breakdown, insight_summary


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
    similar_comments, similar_meta = _build_canonical_similar_comments(
        source_comments if source_comments is not None else pd.DataFrame(),
        selected,
        selected,
        sentiment_name,
        _safe_text(selected.get("comment_id", selected.get("source_content_id", selected.get("opinion_id", "")))),
    )
    if similar_meta.get("cluster_size", 0):
        cluster_size = int(similar_meta["cluster_size"])
    trend_text = _recent_trend_detail(similar_comments)
    strength_text = _score_signal_detail(original_text, sentiment_name, _safe_text(selected.get("signal_strength", "")))
    richness_text = _score_richness_detail(original_text)
    likes_text = f"좋아요는 {likes:,}개로 보조 가중치를 주되, 대표 선정은 어디까지나 감성 강도와 내용 구체성을 우선했습니다." if likes > 0 else "좋아요 수는 높지 않더라도 유사 언급량과 내용 구체성이 충분해 대표 사례로 유지했습니다."
    sentiment_label = "부정" if sentiment_name == "negative" else "긍정" if sentiment_name == "positive" else "중립"
    return f"{summary_text} 이 댓글은 {sentiment_label} 의견이 분명하고 같은 이슈의 유사 댓글이 {cluster_size}개 확인되어 대표 사례로 선정했습니다. {trend_text} {strength_text} {richness_text} {likes_text}"


def _negative_issue_context(selected: pd.Series, source_comments: pd.DataFrame | None) -> dict[str, str]:
    if source_comments is None or source_comments.empty:
        return {}
    working = source_comments.copy()
    if "comment_validity" in working.columns:
        working = working[working["comment_validity"].astype(str).eq("valid")].copy()
    if working.empty:
        return {}

    topic_value = _safe_text(selected.get("topic_label", selected.get("aspect_key", "")))
    if topic_value:
        if "topic_label" in working.columns:
            working = working[working["topic_label"].astype(str).eq(topic_value)].copy()
        elif "aspect_key" in working.columns:
            working = working[working["aspect_key"].astype(str).eq(topic_value)].copy()
    product_value = _safe_text(selected.get("product", ""))
    if product_value and "product" in working.columns:
        working = working[working["product"].astype(str).eq(product_value)].copy()
    if working.empty:
        return {}

    sentiment_series = (
        working.get("sentiment_final", working.get("display_sentiment", working.get("sentiment_label", pd.Series("", index=working.index))))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    negative_rows = working[sentiment_series.eq("negative")].copy()
    if negative_rows.empty:
        return {}

    total_scope = max(len(working), 1)
    share = len(negative_rows) / total_scope
    recurrence = "지속 이슈" if len(negative_rows) >= 4 else "일회성 신호"

    trend = "추세 데이터 부족"
    date_series = pd.to_datetime(
        negative_rows.get(COL_WRITTEN_AT, negative_rows.get("published_at", pd.Series(pd.NaT, index=negative_rows.index))),
        errors="coerce",
        utc=True,
    ).dt.tz_localize(None)
    dates = date_series.dropna()
    if not dates.empty:
        latest = dates.max()
        recent = int((dates >= latest - pd.Timedelta(days=14)).sum())
        previous = int(((dates < latest - pd.Timedelta(days=14)) & (dates >= latest - pd.Timedelta(days=28))).sum())
        if recent >= previous + 2:
            trend = "최근 증가"
        elif previous >= recent + 2:
            trend = "최근 감소"
        else:
            trend = "최근 안정"

    return {
        "share_text": f"비중 {share * 100:.1f}%",
        "recurrence_text": recurrence,
        "trend_text": trend,
    }


def _build_representative_selection_reason(
    selected: pd.Series,
    sentiment_name: str,
    source_comments: pd.DataFrame | None,
    similar_count: int | None = None,
) -> str:
    cluster_size = int(pd.to_numeric(selected.get("cluster_size", 1), errors="coerce") or 1)
    matched_similar = int(similar_count) if similar_count is not None else cluster_size
    confidence = int(round(_confidence_ratio_for_card(selected, sentiment_name) * 100))
    base_parts = [f"감성 강도 상위 {confidence}%", f"유사 맥락 {matched_similar}건"]

    if sentiment_name == "negative":
        ctx = _negative_issue_context(selected, source_comments)
        if ctx:
            base_parts.extend([ctx["share_text"], ctx["recurrence_text"], ctx["trend_text"]])
    return " · ".join(base_parts)


def _build_similarity_basis_labels(working_selected: pd.Series, sentiment_name: str) -> list[str]:
    core_points, context_tags, similarity_keys, _, _, _ = _extract_row_nlp_metadata(working_selected)
    context_label_map = {
        "구독/계약": "구독/약관 맥락",
        "A/S·수리": "A/S·수리 대응",
        "배송·설치": "배송·설치 경험",
        "가격·비용": "가격/가치 인식",
        "성능·품질": "핵심 성능 체감",
        "사용성·편의": "사용성·편의",
        "비교 맥락": "비교 맥락",
        "구매 전환": "구매 전환 상황",
        "추천 의도": "추천 의도",
        "문의/해결요청": "문의/해결요청",
    }
    labels_from_metadata: list[str] = []
    for tag in context_tags:
        mapped = context_label_map.get(_safe_text(tag), _safe_text(tag))
        if mapped:
            labels_from_metadata.append(mapped)
    for key in similarity_keys:
        clean = _safe_text(key)
        if clean.startswith("ctx:"):
            mapped = context_label_map.get(clean[4:], clean[4:])
            if mapped:
                labels_from_metadata.append(mapped)
        elif clean.startswith("pt:") and len(labels_from_metadata) < 3:
            point = _safe_text(clean[3:])
            if point:
                labels_from_metadata.append(f"{point} 포인트")
    if not labels_from_metadata and core_points:
        labels_from_metadata.extend([f"{point} 포인트" for point in core_points[:2]])
    if labels_from_metadata:
        if sentiment_name == "negative":
            labels_from_metadata.insert(0, "불만 핵심 포인트 유사")
        elif sentiment_name == "positive":
            labels_from_metadata.insert(0, "강점 발현 상황 유사")
        return list(dict.fromkeys(labels_from_metadata))[:3]

    source_text = " ".join(
        [
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get("cleaned_text", "")),
            _safe_text(working_selected.get("nlp_summary", "")),
        ]
    ).lower()
    mappings = [
        (["구독", "가입", "권유", "약관", "해지", "위약금"], "구독/약관 맥락"),
        (["수리", "as", "서비스", "기사", "출장", "보증"], "A/S·수리 대응"),
        (["배송", "설치"], "배송·설치 경험"),
        (["소음", "진동", "세척", "건조", "냉각", "성능"], "핵심 성능 체감"),
        (["가격", "비용", "비싸", "가성비"], "가격/가치 인식"),
        (["추천", "재구매", "추천해", "추천합니다"], "추천 의도"),
        (["비교", "대비", "이전", "기존", "옛날"], "비교 맥락"),
    ]
    labels: list[str] = []
    for markers, label in mappings:
        if any(marker in source_text for marker in markers):
            labels.append(label)
    if not labels:
        labels.append("핵심 포인트 유사")
    if sentiment_name == "negative" and "불만 유형 유사" not in labels:
        labels.insert(0, "불만 유형 유사")
    if sentiment_name == "positive" and "강점 발현 상황 유사" not in labels:
        labels.insert(0, "강점 발현 상황 유사")
    return list(dict.fromkeys(labels))[:3]


def _render_representative_preview_strip(
    showcase: pd.DataFrame,
    sentiment_name: str,
    source_comments: pd.DataFrame | None,
) -> None:
    if showcase is None or showcase.empty:
        return
    top3 = showcase.head(3).reset_index(drop=True)
    st.caption("빠른 판단용 Top 3")
    cols = st.columns(len(top3))
    for idx, (_, row) in enumerate(top3.iterrows()):
        with cols[idx]:
            topic = _localize_aspect_label(_safe_text(row.get("topic_label", row.get("aspect_key", ""))) or f"이슈 {idx + 1}")
            sentiment_label = "부정" if sentiment_name == "negative" else "긍정" if sentiment_name == "positive" else "중립"
            preview_text = _safe_text(row.get("summary_1line", ""))
            if not preview_text:
                preview_text = _summarize_for_dashboard(row, sentiment_name)
            preview_text = _summarize_original_for_preview(preview_text, limit=92) or "요약 정보가 부족합니다."
            st.markdown(
                f"""
                <div class="rep-top-card">
                  <div class="rep-top-title">{idx + 1}. {html.escape(topic)}</div>
                  <div class="rep-top-meta">{sentiment_label}</div>
                  <div class="rep-top-text">{html.escape(preview_text)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


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
        return "원문 정보가 부족해 요약을 만들지 못했습니다."

    products = [token for token in ["드럼", "통돌이", "세탁기", "건조기", "냉장고", "식기세척기"] if token in source_text]
    product_subject = "·".join(products[:2]) if products else "제품"

    inquiry_flag = _truthy(selected.get("nlp_is_inquiry")) or _truthy(selected.get("inquiry_flag")) or ("?" in source_text)
    keywords = _extract_card_keywords(selected, sentiment_name, limit=3)
    focus = ", ".join(keywords[:2]) if keywords else "핵심 쟁점"
    fact = _extract_comment_fact_summary(source_text, sentiment_name, focus)
    if not fact:
        fact = _summarize_original_for_preview(source_text, limit=120)
    if inquiry_flag:
        return f"{product_subject} 문의 요지: {fact}"
    if sentiment_name == "negative":
        return f"{product_subject} 불편·불만 요지: {fact}"
    if sentiment_name == "positive":
        return f"{product_subject} 강점 요지: {fact}"
    return f"{product_subject} 관찰 요지: {fact}"


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
    def _excel_safe_value(value: Any) -> Any:
        """Convert timezone-aware datetime values to timezone-naive for Excel."""
        if isinstance(value, pd.Timestamp):
            if value.tzinfo is not None:
                return value.tz_localize(None)
            return value
        if hasattr(value, "tzinfo") and getattr(value, "tzinfo", None) is not None and hasattr(value, "replace"):
            try:
                return value.replace(tzinfo=None)
            except Exception:
                return value
        return value

    def _prepare_excel_frame(frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        for col in prepared.columns:
            series = prepared[col]
            if pd.api.types.is_datetime64tz_dtype(series.dtype):
                prepared[col] = series.dt.tz_localize(None)
                continue
            if pd.api.types.is_object_dtype(series.dtype):
                prepared[col] = series.map(_excel_safe_value)
        return prepared

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for name, frame in dataframes.items():
            _prepare_excel_frame(frame).to_excel(writer, sheet_name=name[:31], index=False)
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



def compute_filtered_bundle(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    filters: dict[str, Any],
    representative_bundles_df: pd.DataFrame | None = None,
    opinion_units_df: pd.DataFrame | None = None,
    analysis_non_trash_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
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
    keyword_terms = list(filters.get("keyword_terms") or _parse_keyword_terms(keyword_query))
    keyword_mode = str(filters.get("keyword_mode") or "AND").strip().upper()
    analysis_scope = str(filters.get("analysis_scope") or "전체").strip() or "전체"
    relevance_scope = str(filters.get("relevance_scope") or "전체 수집").strip() or "전체 수집"

    opinion_units = (opinion_units_df.copy() if opinion_units_df is not None else pd.DataFrame())
    analysis_non_trash = (analysis_non_trash_df.copy() if analysis_non_trash_df is not None else pd.DataFrame())
    if not analysis_non_trash.empty:
        if COL_COUNTRY not in analysis_non_trash.columns and "region" in analysis_non_trash.columns:
            analysis_non_trash[COL_COUNTRY] = analysis_non_trash["region"].map(localize_region)
        if COL_BRAND not in analysis_non_trash.columns:
            if "brand_mentioned" in analysis_non_trash.columns:
                analysis_non_trash[COL_BRAND] = analysis_non_trash["brand_mentioned"].map(canonicalize_brand)
            elif "brand" in analysis_non_trash.columns:
                analysis_non_trash[COL_BRAND] = analysis_non_trash["brand"].map(canonicalize_brand)
        if COL_CEJ not in analysis_non_trash.columns:
            if "customer_journey_stage" in analysis_non_trash.columns:
                analysis_non_trash[COL_CEJ] = analysis_non_trash["customer_journey_stage"].map(canonicalize_cej)
            elif "cej_scene_code" in analysis_non_trash.columns:
                analysis_non_trash[COL_CEJ] = analysis_non_trash["cej_scene_code"].map(canonicalize_cej)
        analysis_included_series = analysis_non_trash.get("analysis_included", pd.Series(True, index=analysis_non_trash.index)).fillna(False).astype(bool)
        hygiene_series = analysis_non_trash.get("hygiene_class", pd.Series("strategic", index=analysis_non_trash.index)).fillna("").astype(str).str.lower()
        analysis_non_trash = analysis_non_trash[analysis_included_series & hygiene_series.ne("trash")].copy()
        if products_active and "product" in analysis_non_trash.columns:
            analysis_non_trash = analysis_non_trash[analysis_non_trash["product"].isin(products)]
        if regions_active and COL_COUNTRY in analysis_non_trash.columns:
            known_labels = set(REGION_LABELS.values())
            if REGION_OTHER_LABEL in regions:
                analysis_non_trash = analysis_non_trash[
                    analysis_non_trash[COL_COUNTRY].isin(regions) | ~analysis_non_trash[COL_COUNTRY].isin(known_labels)
                ]
            else:
                analysis_non_trash = analysis_non_trash[analysis_non_trash[COL_COUNTRY].isin(regions)]
        if cej_active and COL_CEJ in analysis_non_trash.columns:
            analysis_non_trash = analysis_non_trash[analysis_non_trash[COL_CEJ].isin(cej)]
        if brands_active and COL_BRAND in analysis_non_trash.columns:
            analysis_non_trash = analysis_non_trash[analysis_non_trash[COL_BRAND].isin(brands)]
        if keyword_terms:
            analysis_keyword_mask = _build_keyword_mask(analysis_non_trash, keyword_terms, keyword_mode)
            analysis_non_trash = analysis_non_trash[analysis_keyword_mask]
        if sentiments_active:
            sentiment_series = analysis_non_trash.get("sentiment_final", analysis_non_trash.get("sentiment_label", pd.Series("", index=analysis_non_trash.index)))
            localized_sentiment = sentiment_series.fillna("").astype(str).str.lower().map(localize_sentiment)
            analysis_non_trash = analysis_non_trash[localized_sentiment.isin(sentiments)]
        if relevance_scope == "LG 관련만":
            if "lg_relevant_comment" in analysis_non_trash.columns:
                analysis_non_trash = analysis_non_trash[analysis_non_trash["lg_relevant_comment"].fillna(False).astype(bool)]
            elif "is_lg_relevant_video" in analysis_non_trash.columns:
                analysis_non_trash = analysis_non_trash[analysis_non_trash["is_lg_relevant_video"].fillna(False).astype(bool)]
            elif "video_lg_relevance_score" in analysis_non_trash.columns:
                analysis_non_trash = analysis_non_trash[
                    pd.to_numeric(analysis_non_trash["video_lg_relevance_score"], errors="coerce").fillna(0.0) >= 0.45
                ]
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
    if analysis_scope != "전체" and not analysis_non_trash.empty and "comment_id" in analysis_non_trash.columns:
        analysis_ids = analysis_non_trash["comment_id"].astype(str)
        if analysis_scope == "문의 포함 댓글만":
            analysis_non_trash = analysis_non_trash[analysis_ids.isin(inquiry_source_ids)]
        elif analysis_scope == "문의 제외":
            analysis_non_trash = analysis_non_trash[~analysis_ids.isin(inquiry_source_ids)]

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
    if keyword_terms:
        keyword_mask = _build_keyword_mask(base_comments, keyword_terms, keyword_mode)
        base_comments = base_comments[keyword_mask]

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
    analysis_included = representative_comments.get("analysis_included", pd.Series(True, index=representative_comments.index)).fillna(False).astype(bool)
    representative_comments = representative_comments[analysis_included].copy()
    hygiene_series = representative_comments.get("hygiene_class", pd.Series("strategic", index=representative_comments.index)).fillna("").astype(str).str.lower()
    representative_comments = representative_comments[hygiene_series.ne("trash")].copy()
    if "representative_eligibility" in representative_comments.columns:
        eligibility_series = representative_comments["representative_eligibility"].fillna(False).astype(bool)
        if eligibility_series.any():
            representative_comments = representative_comments[eligibility_series].copy()
    else:
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
        "analysis_non_trash": analysis_non_trash,
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
    nlp_noise_terms = {
        "영상", "리뷰", "제품", "효과", "기능", "사람들", "사람", "이유", "기본", "혹시",
        "정말", "진짜", "그냥", "라고", "보니", "있고요", "좀더", "점점", "직접", "대단한",
        "만들었네", "영상보고", "먹을수있고", "사고싶은데", "하지", "추측이긴", "파는",
    }
    sentiment_df = comments_df.loc[comments_df["sentiment_label"] == sentiment_code].copy()
    raw_texts = sentiment_df["cleaned_text"].fillna("").astype(str).tolist()

    # Boost with LLM-extracted keywords so meaningful issue tokens surface above narration noise.
    # Keep this light and filtered to avoid amplifying weak generic terms.
    for _, row in sentiment_df.iterrows():
        nlp_tokens: list[str] = []
        for raw_kw in _parse_list_like(row.get("nlp_keywords", [])):
            token = normalize_keyword(raw_kw)
            if not token:
                continue
            if token in nlp_noise_terms:
                continue
            if token in {"문제", "이슈", "상태", "부분", "경험", "느낌", "후기"}:
                continue
            nlp_tokens.append(token)
        if not nlp_tokens:
            continue
        dedup_tokens = list(dict.fromkeys(nlp_tokens))[:6]
        raw_texts.append(" ".join(dedup_tokens * 3))

    texts = tuple(raw_texts)
    blocked_keywords = _build_dynamic_keyword_excludes(comments_df)
    return (
        build_keyword_summary(texts, top_n=6, version="v8", blocked_keywords=blocked_keywords),
        build_keyword_summary(texts, top_n=30, version="v8", blocked_keywords=blocked_keywords),
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
        _safe_text(row.get("sentiment_final", row.get("display_sentiment", row.get("sentiment_label", "")))),
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
        canonical_sentiment = (
            base_pool.get("sentiment_final", base_pool.get("sentiment_label", pd.Series("", index=base_pool.index)))
            .fillna(base_pool.get("sentiment_label", pd.Series("", index=base_pool.index)))
            .astype(str)
            .str.lower()
        )
        base_pool = base_pool[canonical_sentiment.eq(sentiment)].copy()
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


def render_representative_comments(
    filtered_comments: pd.DataFrame,
    representative_bundles: pd.DataFrame | None = None,
    opinion_units: pd.DataFrame | None = None,
    key_suffix: str = "",
    similar_source_pool: pd.DataFrame | None = None,
) -> None:
    st.subheader("핵심 대표 코멘트")
    st.caption("판단은 대표 코멘트에서 시작합니다. 먼저 Top 3를 빠르게 보고, 필요할 때만 상세와 유사 댓글을 펼쳐 확인하세요.")

    # Representative cards must be selected from comment-level canonical rows only.
    # representative_bundles/opinion_units can remain as optional evidence artifacts.
    source_for_cards = filtered_comments if filtered_comments is not None else pd.DataFrame()
    # Similar comments MUST come from one canonical source pool only (analysis_non_trash).
    canonical_similar_pool = _canonicalize_similar_source_pool(
        similar_source_pool if similar_source_pool is not None else pd.DataFrame()
    )
    negative_showcase = _build_bundle_showcase(source_for_cards, sentiment="negative", limit=5)
    positive_showcase = _build_bundle_showcase(source_for_cards, sentiment="positive", limit=5)

    if negative_showcase.empty and positive_showcase.empty:
        fallback_showcase = _build_bundle_showcase(source_for_cards, sentiment="", limit=5)
        st.warning("현재 조건에서 강한 긍정/부정 대표 코멘트가 부족해, 의미 있는 유효 댓글을 우선 보여줍니다.")
        render_comment_table(
            fallback_showcase,
            key_prefix=f"neutral{key_suffix}",
            source_comments=canonical_similar_pool,
        )
        return

    export_rows: list[dict[str, Any]] = []
    for sentiment_name, showcase in [("negative", negative_showcase), ("positive", positive_showcase)]:
        for _, row in showcase.reset_index(drop=True).iterrows():
            working_selected = _enrich_card_row_from_source(row.copy(), source_for_cards)
            rep_id = _safe_text(row.get("comment_id", row.get("source_content_id", row.get("opinion_id", ""))))
            video_id = _safe_text(row.get("video_id", "")) or _safe_text(working_selected.get("video_id", ""))
            similar_comments, similar_meta = _build_canonical_similar_comments(
                canonical_similar_pool,
                row,
                working_selected,
                sentiment_name,
                rep_id,
            )
            summary_1line = _safe_text(row.get("summary_1line", "")) or _summarize_for_dashboard(working_selected, sentiment_name)
            export_rows.append(
                {
                    "감성": localize_sentiment(sentiment_name),
                    "토픽": _localize_aspect_label(_safe_text(row.get("topic_label", row.get("aspect_key", "")))),
                    "제품": _safe_text(row.get("product", working_selected.get("product", ""))),
                    "여정": _safe_text(row.get(COL_CEJ, working_selected.get(COL_CEJ, ""))),
                    "브랜드": _safe_text(row.get(COL_BRAND, working_selected.get(COL_BRAND, ""))),
                    "요약": summary_1line,
                    "왜 대표 코멘트인가": _build_representative_selection_reason(
                        working_selected, sentiment_name, source_for_cards, similar_count=int(similar_meta.get("cluster_size", len(similar_comments)))
                    ),
                    "유사 댓글 수(필터 기준)": int(similar_meta.get("cluster_size", len(similar_comments))),
                    "원문 댓글": _safe_text(working_selected.get(COL_ORIGINAL, working_selected.get("text_display", ""))),
                }
            )
    if export_rows:
        rep_export_df = pd.DataFrame(export_rows)
        st.download_button(
            "대표 코멘트 엑셀 다운로드",
            data=to_excel_bytes({"대표코멘트": rep_export_df}),
            file_name="representative_comments.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"download_representative_{key_suffix or 'main'}",
        )

    st.write(f"부정 핵심 대표 이슈 {len(negative_showcase):,}건")
    _render_representative_preview_strip(negative_showcase, "negative", source_for_cards)
    with st.expander("부정 Top 3 상세 보기", expanded=True):
        render_comment_table(
            negative_showcase.head(3),
            key_prefix=f"negative{key_suffix}",
            source_comments=canonical_similar_pool,
        )
    if len(negative_showcase) > 3:
        with st.expander(f"부정 추가 이슈 {min(len(negative_showcase) - 3, 2):,}건 더 보기", expanded=False):
            render_comment_table(
                negative_showcase.iloc[3:5],
                key_prefix=f"negative_more{key_suffix}",
                source_comments=canonical_similar_pool,
            )

    st.write(f"긍정 핵심 대표 이슈 {len(positive_showcase):,}건")
    _render_representative_preview_strip(positive_showcase, "positive", source_for_cards)
    with st.expander("긍정 Top 3 상세 보기", expanded=False):
        render_comment_table(
            positive_showcase.head(3),
            key_prefix=f"positive{key_suffix}",
            source_comments=canonical_similar_pool,
        )
    if len(positive_showcase) > 3:
        with st.expander(f"긍정 추가 이슈 {min(len(positive_showcase) - 3, 2):,}건 더 보기", expanded=False):
            render_comment_table(
                positive_showcase.iloc[3:5],
                key_prefix=f"positive_more{key_suffix}",
                source_comments=canonical_similar_pool,
            )


def _infer_claim_direction(text_value: str) -> int:
    text = _safe_text(text_value).lower()
    if not text:
        return 0
    negative_markers = [
        "비추", "불만", "문제", "고장", "안됨", "최악", "불편", "비싸", "돈낭비", "냄새",
        "not work", "doesn't", "broken", "worst", "bad", "refund",
    ]
    positive_markers = [
        "만족", "추천", "좋아요", "좋다", "잘됨", "괜찮", "문제없", "불만없", "최고",
        "works", "great", "excellent", "love", "recommend", "satisfied",
    ]
    neg = sum(token in text for token in negative_markers)
    pos = sum(token in text for token in positive_markers)
    if neg >= pos + 1:
        return -1
    if pos >= neg + 1:
        return 1
    return 0


def _normalize_topic_for_similarity(value: Any) -> str:
    text = _safe_text(value).strip()
    if not text:
        return ""
    if re.match(r"^(이슈\s*\d+|issue\s*\d+)$", text.lower()):
        return "제품 경험"
    return _safe_text(_localize_aspect_label(text)).strip().lower()


def _normalize_sentiment_direction(value: Any) -> str:
    sentiment = _safe_text(value).strip().lower()
    if sentiment in {"positive", "negative", "neutral", "mixed"}:
        return sentiment
    localized_map = {
        localize_sentiment("positive").lower(): "positive",
        localize_sentiment("negative").lower(): "negative",
        localize_sentiment("neutral").lower(): "neutral",
        "혼합": "mixed",
        "mixed": "mixed",
    }
    if sentiment in localized_map:
        return localized_map[sentiment]
    return "neutral"


def _journey_value_from_row(row: pd.Series) -> str:
    for key in ["journey_stage", COL_CEJ, "customer_journey_stage", "cej_scene_code"]:
        if key in row.index:
            value = canonicalize_cej(_safe_text(row.get(key, "")))
            if value:
                return value
    return ""


def _coerce_comment_id_from_row(row: pd.Series) -> str:
    for key in ["comment_id", "source_content_id", "opinion_id"]:
        value = _safe_text(row.get(key, ""))
        if value:
            return value
    return ""


def _intent_tokens_from_row(row: pd.Series) -> set[str]:
    tokens: set[str] = set()
    tokens.update(_clean_point_tokens(_parse_list_like(row.get("nlp_similarity_keys", [])), limit=10))
    tokens.update(_clean_point_tokens(_parse_list_like(row.get("nlp_core_points", [])), limit=10))
    tokens.update(_clean_point_tokens(_parse_list_like(row.get("nlp_context_tags", [])), limit=6))
    classification = _safe_text(row.get("classification_type", "")).strip().lower()
    if classification:
        tokens.add(classification)
    topic_norm = _normalize_topic_for_similarity(row.get("topic_label", row.get("aspect_key", "")))
    if topic_norm:
        tokens.add(topic_norm)
    return {token for token in tokens if token}


def _cohesion_terms_from_row(row: pd.Series, limit: int = 14) -> set[str]:
    terms: set[str] = set()
    terms.update(_intent_tokens_from_row(row))
    text_blob = " ".join(
        [
            _safe_text(row.get(COL_ORIGINAL, "")),
            _safe_text(row.get("cleaned_text", "")),
            _safe_text(row.get("text_display", "")),
        ]
    ).lower()
    stopwords = {
        "this", "that", "with", "from", "have", "there", "were", "been", "very", "really", "just",
        "그리고", "그냥", "진짜", "너무", "정말", "제품", "사용", "관련", "문제", "해당",
    }
    for token in re.findall(r"[a-zA-Z가-힣]{2,}", text_blob):
        clean = _safe_text(token).strip().lower()
        if not clean or clean in stopwords:
            continue
        terms.add(clean)
        if len(terms) >= limit:
            break
    return {term for term in terms if term}


def _cohesion_score_for_candidate(candidate: pd.Series, seed_terms: set[str]) -> int:
    candidate_terms = _cohesion_terms_from_row(candidate, limit=20)
    if not candidate_terms or not seed_terms:
        return 0
    overlap = len(candidate_terms.intersection(seed_terms))
    if overlap:
        return overlap
    # Fallback to intent token overlap if lexical tokens miss.
    return len(_intent_tokens_from_row(candidate).intersection(seed_terms))


def _canonicalize_similar_source_pool(pool: pd.DataFrame) -> pd.DataFrame:
    if pool is None or pool.empty:
        return pd.DataFrame()
    working = pool.copy()
    included_series = working.get("analysis_included", pd.Series(True, index=working.index)).fillna(False).astype(bool)
    hygiene_series = working.get("hygiene_class", pd.Series("strategic", index=working.index)).fillna("").astype(str).str.lower()
    if "comment_validity" in working.columns:
        valid_series = working["comment_validity"].fillna("").astype(str).str.lower().eq("valid")
    else:
        valid_series = pd.Series(True, index=working.index)
    return working[included_series & hygiene_series.ne("trash") & valid_series].copy()


def _build_canonical_similar_comments(
    analysis_non_trash: pd.DataFrame,
    selected: pd.Series,
    working_selected: pd.Series,
    sentiment_name: str,
    rep_id: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Single canonical similar-comment builder used by card/expander/CSV."""
    source_pool = _canonicalize_similar_source_pool(analysis_non_trash)
    if source_pool.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "empty_source_pool"}

    working = source_pool.copy()
    if rep_id and "comment_id" in working.columns:
        working = working[working["comment_id"].astype(str) != rep_id].copy()
    selected_text = _safe_text(working_selected.get(COL_ORIGINAL, selected.get(COL_ORIGINAL, "")))
    if selected_text and COL_ORIGINAL in working.columns:
        working = working[working[COL_ORIGINAL].astype(str) != selected_text].copy()
    if working.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "no_candidates_after_exclusion"}

    selected_journey = _journey_value_from_row(working_selected)
    if not selected_journey:
        selected_journey = _journey_value_from_row(selected)
    if selected_journey:
        journey_mask = working.apply(lambda row: _journey_value_from_row(row) == selected_journey, axis=1)
        working = working[journey_mask].copy()
    if working.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "journey_mismatch"}

    selected_sentiment = _normalize_sentiment_direction(
        working_selected.get(
            "sentiment_final",
            working_selected.get(
                "sentiment_label",
                selected.get(
                    "sentiment_final",
                    selected.get("sentiment_label", sentiment_name),
                ),
            ),
        )
    )
    if selected_sentiment == "neutral" and sentiment_name in {"positive", "negative", "mixed"}:
        selected_sentiment = sentiment_name
    sentiment_source = working.get("sentiment_final", pd.Series("", index=working.index))
    sentiment_text = sentiment_source.fillna("").astype(str).str.strip()
    if sentiment_text.eq("").all():
        sentiment_source = working.get("sentiment_label", pd.Series("", index=working.index))
    sentiment_norm = sentiment_source.fillna("").astype(str).map(_normalize_sentiment_direction)
    if selected_sentiment == "mixed":
        sentiment_mask = (
            working.get("mixed_flag", pd.Series(False, index=working.index)).fillna(False).astype(bool)
            | sentiment_norm.eq("mixed")
        )
    else:
        sentiment_mask = sentiment_norm.eq(selected_sentiment)
        sentiment_mask = sentiment_mask & ~working.get("mixed_flag", pd.Series(False, index=working.index)).fillna(False).astype(bool)
    working = working[sentiment_mask].copy()
    if working.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "sentiment_mismatch"}

    selected_intents = _intent_tokens_from_row(working_selected)
    if not selected_intents:
        selected_intents = _intent_tokens_from_row(selected)
    selected_topic = _normalize_topic_for_similarity(working_selected.get("topic_label", working_selected.get("aspect_key", selected.get("topic_label", selected.get("aspect_key", "")))))
    selected_class = _safe_text(working_selected.get("classification_type", selected.get("classification_type", ""))).strip().lower()

    def _intent_match(candidate: pd.Series) -> bool:
        candidate_tokens = _intent_tokens_from_row(candidate)
        overlap = len(selected_intents.intersection(candidate_tokens)) if selected_intents else 0
        candidate_topic = _normalize_topic_for_similarity(candidate.get("topic_label", candidate.get("aspect_key", "")))
        candidate_class = _safe_text(candidate.get("classification_type", "")).strip().lower()
        if selected_intents and overlap >= 1:
            return True
        if selected_topic and selected_class and candidate_topic == selected_topic and candidate_class == selected_class:
            return True
        if selected_topic and candidate_topic == selected_topic and overlap >= 1:
            return True
        return False

    working = working[working.apply(_intent_match, axis=1)].copy()
    if working.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "intent_mismatch"}

    cohesion_warning = False
    soft_cap = 3200
    hard_cap = 2400
    if len(working) > soft_cap:
        seed_terms = _cohesion_terms_from_row(working_selected)
        if not seed_terms:
            seed_terms = _cohesion_terms_from_row(selected)
        if seed_terms:
            working["_cohesion_score"] = working.apply(lambda row: _cohesion_score_for_candidate(row, seed_terms), axis=1)
            strict = working[working["_cohesion_score"] >= 2].copy()
            if len(strict) >= 3:
                working = strict
            else:
                loose = working[working["_cohesion_score"] >= 1].copy()
                if len(loose) >= 3:
                    working = loose
        if len(working) > hard_cap:
            cohesion_warning = True
            if "_cohesion_score" not in working.columns:
                working["_cohesion_score"] = 1
            like_source = pd.to_numeric(
                working.get(COL_LIKES, working.get("likes_count", working.get("like_count", pd.Series(0, index=working.index)))),
                errors="coerce",
            ).fillna(0.0)
            working["_like_score"] = like_source
            working = (
                working.sort_values(["_cohesion_score", "_like_score"], ascending=[False, False], na_position="last")
                .head(hard_cap)
                .copy()
            )

    display_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _, row in working.iterrows():
        row_text = _safe_text(row.get(COL_ORIGINAL, row.get("text_display", row.get("original_text", ""))))
        if not row_text:
            continue
        comment_id = _coerce_comment_id_from_row(row)
        unique_key = (comment_id, row_text)
        if unique_key in seen:
            continue
        seen.add(unique_key)
        display_rows.append(
            {
                "comment_id": comment_id,
                "원문": row_text,
                "좋아요 수": int(pd.to_numeric(row.get(COL_LIKES, row.get("likes_count", row.get("like_count", 0))), errors="coerce") or 0),
                "작성일시": _safe_text(row.get(COL_WRITTEN_AT, row.get("published_at", ""))),
            }
        )

    result = pd.DataFrame(display_rows)
    if result.empty:
        return pd.DataFrame(), {"cluster_size": 0, "insufficient_evidence": True, "reason": "no_display_rows"}

    result = result.sort_values(["좋아요 수", "작성일시"], ascending=[False, False], na_position="last").reset_index(drop=True)
    cluster_size = len(result)
    if cluster_size < 3:
        return pd.DataFrame(), {
            "cluster_size": 0,
            "candidate_count": cluster_size,
            "insufficient_evidence": True,
            "reason": "cluster_lt_3",
        }
    return result, {
        "cluster_size": cluster_size,
        "candidate_count": cluster_size,
        "insufficient_evidence": False,
        "cohesion_warning": cohesion_warning,
        "reason": "cohesion_guard_applied" if cohesion_warning else "ok",
    }


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
    allowed = {"positive", "negative", "neutral", "mixed", "trash"}
    for key in ("sentiment_final", "display_sentiment", "sentiment_label", "nlp_label"):
        sentiment = _safe_text(working_selected.get(key, "")).lower()
        if sentiment in allowed:
            return sentiment
    fallback = _safe_text(fallback_sentiment).lower()
    return fallback if fallback in allowed else "neutral"


def _confidence_ratio_for_card(working_selected: pd.Series, sentiment_label: str) -> float:
    raw = working_selected.get("nlp_confidence", working_selected.get("confidence_level", ""))
    intensity_raw = working_selected.get("nlp_sentiment_intensity", "")
    intensity = pd.to_numeric(intensity_raw, errors="coerce")
    intensity_val = float(intensity) if pd.notna(intensity) else 0.0
    intensity_val = max(0.0, min(1.0, intensity_val))
    if raw is None:
        raw = ""

    if isinstance(raw, (int, float)):
        num = abs(float(raw))
        if num > 1:
            num = num / 100 if num <= 100 else 1.0
        if 0 <= num <= 1:
            if sentiment_label in {"positive", "negative"}:
                num = max(num, 0.56 + (0.10 * intensity_val))
            return max(0.35, min(0.99, num))

    text_val = _safe_text(raw).strip().lower()
    if text_val.endswith("%"):
        try:
            num = float(text_val.replace("%", "")) / 100
            if sentiment_label in {"positive", "negative"}:
                num = max(num, 0.56 + (0.10 * intensity_val))
            return max(0.35, min(0.99, num))
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
    card_keyword_noise = {
        "이게", "그게", "저게", "문제나", "문제냐", "한달", "이번", "그냥", "진짜", "관련", "해당", "정도",
        "남기는댓글", "왜그럴까요", "필요하면습득하고", "정보를전달해주면",
        "커서", "밤에", "사람들", "이유", "요즘", "지금", "때문", "있고", "없고", "없어서", "아예", "못쓰겠고",
    }
    card_keyword_alias = {
        "통돌": "통돌이",
        "통돌이는": "통돌이",
        "통돌이로": "통돌이",
        "통돌이랑": "통돌이",
        "as": "A/S",
        "a/s": "A/S",
        "티비": "TV",
        "tv": "TV",
    }

    def _card_keyword(raw: Any) -> str:
        token = normalize_keyword(raw)
        if not token:
            return ""
        token = re.sub(r"(으로|로|은|는|이|가|을|를|와|과|랑|도|만)$", "", token)
        token = card_keyword_alias.get(token, token)
        if token.lower() in card_keyword_noise or token in card_keyword_noise:
            return ""
        if len(token) < 2:
            return ""
        if token.isdigit():
            return ""
        return token

    core_points, _, _, _, _, _ = _extract_row_nlp_metadata(working_selected)
    if core_points:
        return core_points[:limit]

    raw_keywords = working_selected.get("nlp_keywords", [])
    if isinstance(raw_keywords, str):
        try:
            raw_keywords = json.loads(raw_keywords)
        except Exception:
            raw_keywords = [part.strip() for part in raw_keywords.split(",") if part.strip()]
    if isinstance(raw_keywords, list):
        normalized = [_card_keyword(item) for item in raw_keywords]
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

    normalized_ranked = [_card_keyword(keyword) for keyword in ranked]
    cleaned_ranked = [item for item in normalized_ranked if item]
    deduped_ranked = list(dict.fromkeys(cleaned_ranked))
    return deduped_ranked[:limit]


def _split_comment_clauses(text_value: str) -> list[str]:
    compact = re.sub(r"\s+", " ", _safe_text(text_value))
    compact = re.sub(r"https?://\S+", "", compact).strip()
    if not compact:
        return []
    raw_parts = re.split(r"[.!?\n]|(?:\s{2,})", compact)
    clause_noise_markers = (
        "쿠팡 파트너스",
        "일정 수수료",
        "본 링크",
        "보러가기",
        "더보기 클릭",
        "구매링크",
        "sponsored",
        "affiliate",
    )
    clauses: list[str] = []
    for raw in raw_parts:
        clause = re.sub(r"\s+", " ", _safe_text(raw)).strip(" -:;,'\"")
        if len(clause) < 4:
            continue
        if not re.search(r"[A-Za-z0-9가-힣]{2,}", clause):
            continue
        if clause in {"ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ"}:
            continue
        lowered = clause.lower()
        if clause.startswith("▼"):
            continue
        if any(marker.lower() in lowered for marker in clause_noise_markers):
            continue
        if clause.count(":") >= 3 and any(token in lowered for token in ["kg", "l", "링크", "보러가기"]):
            continue
        clauses.append(clause)
    if clauses:
        return clauses
    return [compact]


def _combine_unique_text_parts(parts: list[str]) -> str:
    unique_parts: list[str] = []
    for raw in parts:
        text = _safe_text(raw).strip()
        if not text:
            continue
        duplicated = False
        for existing in unique_parts:
            if text == existing or text in existing or existing in text:
                duplicated = True
                break
        if duplicated:
            continue
        unique_parts.append(text)
    return " ".join(unique_parts).strip()


def _pick_salient_clauses(text_value: str, limit: int = 2) -> list[str]:
    clauses = _split_comment_clauses(text_value)
    if not clauses:
        return []
    issue_markers = [
        "고장", "불량", "소음", "진동", "세척", "건조", "세탁", "냉각", "누수", "환불", "수리", "서비스", "가격", "비용",
        "비싸", "권유", "해지", "위약금", "불편", "무겁", "허리", "추천", "만족", "최고", "좋",
    ]
    question_markers = ["왜", "어떻게", "언제", "몇", "얼마", "가능", "되나요", "인가요", "?"]
    scored: list[tuple[float, str]] = []
    for clause in clauses:
        lowered = clause.lower()
        if any(marker in lowered for marker in ["쿠팡 파트너스", "일정 수수료", "본 링크", "보러가기", "sponsored", "affiliate"]):
            continue
        score = min(len(clause), 120) / 120
        if any(marker in lowered for marker in issue_markers):
            score += 1.6
        if re.search(r"\d", clause):
            score += 0.35
        if any(marker in lowered for marker in question_markers):
            score += 0.5
        scored.append((score, clause))
    ranked = [text for _, text in sorted(scored, key=lambda item: item[0], reverse=True)]
    deduped: list[str] = []
    seen = set()
    for clause in ranked:
        key = clause[:32]
        if key in seen:
            continue
        seen.add(key)
        deduped.append(clause)
        if len(deduped) >= limit:
            break
    return deduped


def _extract_comment_fact_summary(text_value: str, sentiment_name: str, fallback_focus: str) -> str:
    text = _safe_text(text_value).strip()
    if not text:
        return ""
    compact = re.sub(r"\s+", " ", re.sub(r"https?://\S+", "", text)).strip()
    clauses = _pick_salient_clauses(compact, limit=2)
    if not clauses:
        cleaned_compact = re.sub(r"(쿠팡 파트너스.*|일정 수수료.*|본 링크.*)", "", compact, flags=re.IGNORECASE).strip()
        fallback_clause = _summarize_original_for_preview(cleaned_compact or compact, limit=90)
        if fallback_clause and not re.search(r"[A-Za-z0-9가-힣]{2,}", fallback_clause):
            fallback_clause = ""
        clauses = [fallback_clause] if fallback_clause else ["핵심 근거 문장을 자동 추출하지 못해 원문 확인이 필요합니다"]
    primary = clauses[0][:90].rstrip()
    secondary = clauses[1][:90].rstrip() if len(clauses) > 1 else ""

    lowered = compact.lower()
    stage_map = [
        (["구독", "가입", "권유", "해지", "위약금", "할부"], "구매/계약"),
        (["수리", "as", "a/s", "서비스", "기사", "보증", "연락"], "A/S"),
        (["배송", "설치"], "배송/설치"),
        (["사용", "써보", "세탁", "건조", "소음", "진동", "세척"], "사용"),
    ]
    stage = next((name for markers, name in stage_map if any(marker in lowered for marker in markers)), "사용")

    issue_map = [
        (["소음", "진동"], "소음·진동"),
        (["세척", "세탁력"], "세척 성능"),
        (["건조"], "건조 성능"),
        (["고장", "불량", "누수"], "품질/고장"),
        (["가격", "비용", "비싸", "전기세"], "가격/비용"),
        (["권유", "해지", "위약금"], "계약/안내"),
        (["불편", "무겁", "허리"], "사용 편의"),
    ]
    issue = next((name for markers, name in issue_map if any(marker in lowered for marker in markers)), fallback_focus or "핵심 쟁점")
    if primary.startswith("핵심 근거 문장을 자동 추출하지 못해"):
        return f"{stage} 맥락에서 {issue} 관련 근거 문장을 자동 추출하지 못했습니다. 원문 확인이 필요합니다."
    inquiry_like = "?" in compact or any(marker in lowered for marker in ["왜", "어떻게", "언제", "몇", "얼마", "되나요", "인가요"])
    if inquiry_like:
        summary = f"{stage} 맥락에서 {issue}를 확인하려는 질문입니다: {primary}"
        if secondary:
            summary += f" / {secondary}"
        return summary
    if sentiment_name == "negative":
        summary = f"{stage} 맥락에서 {issue} 불만이 핵심입니다: {primary}"
        if secondary:
            summary += f" / {secondary}"
        return summary
    if sentiment_name == "positive":
        summary = f"{stage} 맥락에서 {issue} 강점을 강조합니다: {primary}"
        if secondary:
            summary += f" / {secondary}"
        return summary
    summary = f"{stage} 맥락에서 {issue}에 대한 관찰 의견입니다: {primary}"
    if secondary:
        summary += f" / {secondary}"
    return summary


def _collect_video_context_signals(working_selected: pd.Series, source_comments: pd.DataFrame | None) -> dict[str, Any]:
    if source_comments is None or source_comments.empty:
        return {
            "count": 1,
            "negative_share": 0.0,
            "positive_share": 0.0,
            "top_points": [],
            "top_tags": [],
            "evidence_clauses": [],
            "nlp_coverage": 0.0,
        }

    pool = source_comments.copy()
    if "comment_validity" in pool.columns:
        pool = pool[pool["comment_validity"].astype(str).eq("valid")].copy()
    if pool.empty:
        return {
            "count": 1,
            "negative_share": 0.0,
            "positive_share": 0.0,
            "top_points": [],
            "top_tags": [],
            "evidence_clauses": [],
            "nlp_coverage": 0.0,
        }

    video_id = _safe_text(working_selected.get("video_id", ""))
    video_link = _safe_text(working_selected.get(COL_VIDEO_LINK, working_selected.get("video_url", "")))
    if video_id and "video_id" in pool.columns:
        subset = pool[pool["video_id"].astype(str).eq(video_id)].copy()
    elif video_link:
        link_col = COL_VIDEO_LINK if COL_VIDEO_LINK in pool.columns else ("video_url" if "video_url" in pool.columns else "")
        subset = pool[pool[link_col].astype(str).eq(video_link)].copy() if link_col else pool.head(0).copy()
    else:
        subset = pool.head(0).copy()
    if subset.empty:
        subset = pool.copy()

    sentiment = subset.get("sentiment_label", pd.Series("", index=subset.index)).astype(str)
    total = max(1, len(subset))
    negative_share = float((sentiment == "negative").sum()) / total
    positive_share = float((sentiment == "positive").sum()) / total

    point_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    if "nlp_core_points" in subset.columns:
        for items in subset["nlp_core_points"].tolist():
            point_counter.update(_clean_point_tokens(_parse_list_like(items), limit=6))
    if "nlp_context_tags" in subset.columns:
        for items in subset["nlp_context_tags"].tolist():
            tag_counter.update([_safe_text(item) for item in _parse_list_like(items) if _safe_text(item)])
    evidence_clauses: list[str] = []
    if "like_count" in subset.columns:
        ranked_subset = subset.sort_values("like_count", ascending=False, na_position="last")
    else:
        ranked_subset = subset
    for _, row in ranked_subset.head(15).iterrows():
        raw_text = _combine_unique_text_parts(
            [
                _safe_text(row.get(COL_ORIGINAL, "")),
                _safe_text(row.get("cleaned_text", "")),
                _safe_text(row.get("text_display", "")),
            ]
        )
        if not raw_text:
            continue
        clauses = _pick_salient_clauses(raw_text, limit=1)
        if not clauses:
            continue
        clause = clauses[0]
        if clause not in evidence_clauses:
            evidence_clauses.append(clause)
        if len(evidence_clauses) >= 3:
            break
    nlp_enriched_rows = 0
    for _, row in subset.head(300).iterrows():
        if _parse_list_like(row.get("nlp_core_points", [])) or _safe_text(row.get("nlp_summary", "")) or _safe_text(row.get("nlp_sentiment_reason", "")):
            nlp_enriched_rows += 1
    nlp_coverage = nlp_enriched_rows / max(1, min(len(subset), 300))

    return {
        "count": len(subset),
        "negative_share": negative_share,
        "positive_share": positive_share,
        "top_points": [token for token, _ in point_counter.most_common(4)],
        "top_tags": [token for token, _ in tag_counter.most_common(3)],
        "evidence_clauses": evidence_clauses,
        "nlp_coverage": nlp_coverage,
    }


def _extract_video_theme_tokens(working_selected: pd.Series) -> list[str]:
    text_sources = [
        _safe_text(working_selected.get(COL_VIDEO_TITLE, "")),
        _safe_text(working_selected.get("title", "")),
        _safe_text(working_selected.get("description", "")),
    ]
    text_sources = [item for item in text_sources if item]
    if not text_sources:
        return []
    counter = build_keyword_counter(tuple(text_sources))
    raw_tokens = [token for token, _ in filter_business_keywords(counter, top_n=8)]
    removed = {"영상", "리뷰", "가전", "제품", "채널", "추천", "비교"}
    return [token for token in _clean_point_tokens(raw_tokens, limit=4) if token not in removed][:3]


def _negative_priority_bucket(
    negative_context: dict[str, str] | None,
    cluster_size: int,
    inquiry_flag: bool,
) -> tuple[str, str]:
    share = 0.0
    trend = _safe_text((negative_context or {}).get("trend_text", ""))
    share_text = _safe_text((negative_context or {}).get("share_text", ""))
    share_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", share_text)
    if share_match:
        try:
            share = float(share_match.group(1))
        except Exception:
            share = 0.0

    if trend == "최근 증가" or share >= 3.0 or cluster_size >= 20:
        return "즉시 관리", "반복 규모/최근 추세가 높아 즉시 대응 우선순위로 분류"
    if inquiry_flag or share >= 1.5 or cluster_size >= 8:
        return "우선 관찰", "반복 신호가 유의미해 관찰-개선 병행이 필요한 구간으로 판단"
    return "모니터링", "현재는 단발성 가능성이 있어 모니터링 중심으로 관리 권장"


def _build_video_context_insight(
    working_selected: pd.Series,
    sentiment_name: str,
    aspect_label: str,
    keywords: list[str],
    source_comments: pd.DataFrame | None = None,
    negative_context: dict[str, str] | None = None,
) -> str:
    core_points, context_tags, _, _, _, insight_summary = _extract_row_nlp_metadata(working_selected)
    cej = _safe_text(working_selected.get(COL_CEJ, "")).strip() or "사용"
    localized_aspect = _localize_aspect_label(aspect_label)
    theme_tokens = _extract_video_theme_tokens(working_selected)
    point_tokens = core_points or _clean_point_tokens(keywords, limit=4)
    point_focus = ", ".join(point_tokens[:2]) if point_tokens else localized_aspect
    lowered_focus = f"{point_focus} {' '.join(context_tags)}".lower()
    if any(token in lowered_focus for token in ["구독", "가입", "해지", "할부", "위약금", "권유", "판매"]):
        family_label = "구독/판매 프로세스"
    elif any(token in lowered_focus for token in ["a/s", "as", "수리", "서비스", "기사", "보증"]):
        family_label = "A/S·수리 대응"
    elif any(token in lowered_focus for token in ["배송", "설치"]):
        family_label = "배송·설치 경험"
    elif any(token in lowered_focus for token in ["가격", "비용", "비싸", "전기세"]):
        family_label = "가격·비용 인식"
    elif any(token in lowered_focus for token in ["소음", "진동", "세척", "건조", "성능", "품질"]):
        family_label = "핵심 성능 체감"
    else:
        family_label = "핵심 포인트"

    source_text = _combine_unique_text_parts(
        [
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            _safe_text(working_selected.get("cleaned_text", "")),
        ]
    )
    representative_clause = _pick_salient_clauses(source_text, limit=1)
    evidence = _summarize_original_for_preview(representative_clause[0], limit=72) if representative_clause else ""
    cluster_size = int(pd.to_numeric(working_selected.get("cluster_size", 1), errors="coerce") or 1)
    video_signals = _collect_video_context_signals(working_selected, source_comments)
    signal_points = video_signals.get("top_points", []) or []
    signal_tags = video_signals.get("top_tags", []) or []
    signal_clauses = video_signals.get("evidence_clauses", []) or []
    tag_hint = ", ".join(context_tags[:2]) if context_tags else "핵심 포인트"
    signal_point_text = ", ".join(signal_points[:2]) if signal_points else point_focus
    signal_tag_text = ", ".join(signal_tags[:2]) if signal_tags else tag_hint
    neg_share = float(video_signals.get("negative_share", 0.0) or 0.0)
    pos_share = float(video_signals.get("positive_share", 0.0) or 0.0)
    context_count = int(video_signals.get("count", 1) or 1)
    nlp_coverage = float(video_signals.get("nlp_coverage", 0.0) or 0.0)
    supporting_raw = next((item for item in signal_clauses if item and item != evidence), signal_clauses[0] if signal_clauses else "")
    supporting_clause = _summarize_original_for_preview(supporting_raw, limit=72) if supporting_raw else ""

    context_bits = [f"{cej} 단계", localized_aspect]
    if theme_tokens:
        context_bits.append(f"영상 주제({', '.join(theme_tokens[:2])})")
    context_text = " · ".join(context_bits)

    if sentiment_name == "negative":
        priority, _ = _negative_priority_bucket(
            negative_context=negative_context,
            cluster_size=cluster_size,
            inquiry_flag=_truthy(working_selected.get("nlp_is_inquiry")),
        )
        base = insight_summary if insight_summary and not _looks_garbled(insight_summary) else f"실사용 맥락에서 '{signal_point_text}' 문제가 반복됩니다"
        evidence_text = f"대표 댓글은 '{evidence}'를 직접 지적합니다." if evidence else "대표 댓글에서 문제를 직접 지목하는 문장이 확인됩니다."
        if supporting_clause:
            evidence_text += f" 같은 영상에서도 '{supporting_clause}' 유형이 반복됩니다."
        return (
            f"{context_text} · {family_label}에서 {base}. {evidence_text} "
            f"해당 영상 관련 댓글 {context_count:,}건 중 부정 {neg_share * 100:.1f}%로, 현재는 '{priority}' 우선순위로 보는 것이 타당합니다."
        )

    if sentiment_name == "positive":
        if "비교 맥락" in context_tags:
            strength_hypothesis = "비교 상황에서 체감 우위가 실제 선택 근거로 작동"
        elif "추천 의도" in context_tags:
            strength_hypothesis = "만족 경험이 추천 의도로 확장"
        elif "구매 전환" in context_tags:
            strength_hypothesis = "구매 직전 의사결정에서 전환을 밀어주는 신호"
        else:
            strength_hypothesis = "반복 노출 시 전환/재구매로 이어질 수 있는 강점"
        base = insight_summary if insight_summary and not _looks_garbled(insight_summary) else f"'{signal_point_text}' 강점이 반복 언급됩니다"
        evidence_text = f"대표 댓글은 '{evidence}'를 강점 근거로 보여줍니다." if evidence else "대표 댓글에서 강점 근거 문장이 확인됩니다."
        if supporting_clause:
            evidence_text += f" 같은 영상에서도 '{supporting_clause}' 반응이 이어집니다."
        return (
            f"{context_text} · {family_label}에서 {base}. {evidence_text} "
            f"긍정 비중 {pos_share * 100:.1f}%로, '{strength_hypothesis}' 가설을 뒷받침합니다."
        )

    base = insight_summary if insight_summary and not _looks_garbled(insight_summary) else f"{tag_hint} 중심으로 '{signal_point_text}' 의견이 혼재합니다"
    evidence_text = f"대표 댓글 핵심 문장: '{evidence}'." if evidence else "대표 댓글 핵심 문장을 더 확인할 필요가 있습니다."
    if supporting_clause:
        evidence_text += f" 주변 반응은 '{supporting_clause}' 유형으로 나타납니다."
    return (
        f"{context_text}에서 {base}. {evidence_text} "
        f"(부정 {neg_share * 100:.1f}% / 긍정 {pos_share * 100:.1f}%) 단정 결론보다 추가 관찰이 필요한 상태입니다."
    )


def _build_insight_layer(
    working_selected: pd.Series,
    sentiment_name: str,
    aspect_label: str,
    keywords: list[str],
    source_comments: pd.DataFrame | None = None,
    similar_comments: pd.DataFrame | None = None,
    insight_registry: dict[str, set[str]] | None = None,
) -> dict[str, Any]:
    def _infer_usage_context(text: str) -> str:
        lowered_text = _safe_text(text).lower()
        context_rules = [
            (["야간", "밤", "저녁"], "야간 사용 상황"),
            (["6개월", "몇개월", "장기간", "오래"], "장기간 사용 이후 상황"),
            (["필터", "교체"], "필터 교체 이후 사용 상황"),
            (["냄새", "악취", "odor", "smell"], "냄새 체감 상황"),
            (["다용도실", "부엌", "주방"], "설치 위치 제약 상황"),
            (["a/s", "as", "수리", "서비스", "기사", "센터"], "A/S 대응 상황"),
            (["손으로", "직접", "떼어", "붙어"], "수동 처리 개입 상황"),
            (["소음", "진동", "탈수"], "소음·진동 체감 상황"),
            (["폐기", "버리", "junk"], "제품 폐기 판단 상황"),
            (["해지", "위약금", "약정", "할부", "구독"], "구매/계약 부담 상황"),
        ]
        for markers, label in context_rules:
            if any(marker in lowered_text for marker in markers):
                return label
        return "실사용 상황"

    def _extract_similar_comment_evidence(frame: pd.DataFrame | None, limit: int = 2) -> list[str]:
        if frame is None or frame.empty or "원문" not in frame.columns:
            return []
        evidences: list[str] = []
        for text_value in frame["원문"].head(20).tolist():
            raw = _safe_text(text_value)
            if not raw:
                continue
            clauses = _pick_salient_clauses(raw, limit=1)
            if not clauses:
                continue
            snippet = _summarize_original_for_preview(clauses[0], limit=72)
            if snippet and snippet not in evidences:
                evidences.append(snippet)
            if len(evidences) >= limit:
                break
        return evidences

    def _detect_domain_signal(corpus_text: str, tags: list[str], usage_value: str, sentiment_value: str) -> str:
        corpus = _safe_text(corpus_text).lower()
        signal_rules = [
            (["냄새", "악취", "odor", "smell"], "odor_control"),
            (["손으로", "직접", "붙어", "떼어", "manual", "cleaning", "scrub"], "manual_intervention"),
            (["a/s", "as", "수리", "서비스", "기사", "센터", "부품"], "service_reliability"),
            (["가격", "비용", "할부", "약정", "위약금", "price", "cost", "money", "expensive", "scam"], "price_value_gap"),
            (["소음", "진동", "탈수", "시끄"], "noise_load"),
            (["추천", "재구매", "지인", "소개", "recommend"], "recommendability"),
            (["내구", "고장", "수명", "durab", "fail"], "durability"),
        ]
        for markers, signal_name in signal_rules:
            if any(marker in corpus for marker in markers):
                return signal_name
        if "A/S·수리" in tags:
            return "service_reliability"
        if "구매/계약 부담" in usage_value:
            return "price_value_gap"
        return "general_experience"

    def _compose_insight_text(
        expectation_sentence: str,
        actual_sentence: str,
        perception_sentence: str,
        context_sentence: str,
        sentiment_value: str,
        archetype: str,
        domain_signal: str,
    ) -> str:
        pattern_key = f"{sentiment_value}:{archetype}:{domain_signal}"
        if "negative:purchase_vs_use_conflict" in pattern_key or domain_signal == "price_value_gap":
            # 비교/대조 구조
            return f"{actual_sentence} {expectation_sentence} {perception_sentence} {context_sentence}".strip()
        if domain_signal in {"odor_control", "manual_intervention"}:
            # 경험 강조 구조
            return f"{actual_sentence} {perception_sentence} {expectation_sentence} {context_sentence}".strip()
        if sentiment_value == "positive" and domain_signal in {"recommendability", "durability"}:
            # 인식 변화 강조 구조
            return f"{perception_sentence} {expectation_sentence} {actual_sentence} {context_sentence}".strip()
        if archetype in {"responsibility_shift", "cluster_signal_gap"}:
            # 인식→원인 구조
            return f"{perception_sentence} {actual_sentence} {expectation_sentence} {context_sentence}".strip()
        # 기대 중심 기본 구조
        return f"{expectation_sentence} {actual_sentence} {perception_sentence} {context_sentence}".strip()

    def _derive_expectation(
        sentiment_value: str,
        product_value: str,
        focus_value: str,
        usage_value: str,
        archetype: str,
        tags: list[str],
        lowered_text: str,
        representative_text: str,
        similar_texts: list[str],
    ) -> tuple[str, str]:
        corpus = " ".join(
            [
                _safe_text(representative_text),
                " ".join(_safe_text(item) for item in similar_texts),
            ]
        ).lower()
        has_smell_signal = any(token in corpus for token in ["냄새", "악취", "odor", "smell"])
        has_manual_work_signal = any(token in corpus for token in ["손으로", "직접", "붙어", "떼어", "닦", "세척", "manual", "cleaning", "scrub"])
        has_noise_signal = any(token in corpus for token in ["소음", "진동", "탈수", "시끄"])
        has_cost_signal = any(token in corpus for token in ["가격", "비용", "할부", "구독", "약정", "위약금", "price", "cost", "money", "expensive", "scam", "waste of money"])
        has_service_signal = any(token in corpus for token in ["a/s", "as", "수리", "서비스", "기사", "센터"])
        has_recommend_signal = any(token in corpus for token in ["추천", "재구매", "지인", "소개"])

        if sentiment_value == "negative":
            if archetype == "purchase_vs_use_conflict":
                return (
                    f"사용자는 {product_value}를 선택할 때 안내받은 비용·계약 조건이 사용 단계에서도 그대로 유지되길 기대합니다.",
                    "비용·계약 조건 일관성",
                )
            if has_smell_signal:
                return (
                    f"사용자는 {product_value}를 실내에서 써도 냄새 확산 없이 생활 동선이 유지되길 기대합니다.",
                    "실내 사용 위생 안정",
                )
            if has_manual_work_signal:
                return (
                    f"사용자는 {product_value}가 처리 과정을 자동으로 끝내고 추가 손작업을 줄여주길 기대합니다.",
                    "자동 처리 완결성",
                )
            if has_noise_signal:
                return (
                    f"사용자는 {product_value}를 일상 시간대에 사용해도 소음·진동 부담이 크지 않길 기대합니다.",
                    "생활 소음 허용 범위",
                )
            if has_cost_signal:
                return (
                    f"사용자는 {product_value} 비용이 체감 성능과 균형을 이루길 기대합니다.",
                    "비용 대비 성능 납득감",
                )
            if has_service_signal or "A/S·수리" in tags:
                return (
                    f"사용자는 문제가 생기면 {product_value}에서 원인과 해결 일정이 한 번에 정리되길 기대합니다.",
                    "명확한 1회 해결",
                )
            if usage_value in {"필터 교체 이후 사용 상황", "냄새 체감 상황", "설치 위치 제약 상황"}:
                return (
                    f"사용자는 {product_value}를 일상 공간에서 추가 조치 없이 사용해도 '{focus_value}' 불편이 통제되길 기대합니다.",
                    f"{focus_value} 자동 통제",
                )
            return (
                f"사용자는 {product_value} 사용에서 '{focus_value}'가 별도 우회 없이 안정적으로 유지되길 기대합니다.",
                f"{focus_value} 안정 유지",
            )

        if sentiment_value == "positive":
            if archetype == "purchase_expectation_alignment":
                return (
                    f"사용자는 {product_value} 구매 전에 비교했던 장점이 실제 생활에서도 같은 수준으로 체감되길 기대합니다.",
                    "구매 전 기대 체감 유지",
                )
            if has_recommend_signal:
                return (
                    f"사용자는 {product_value} 장점이 일시적 만족이 아니라 타인에게 추천 가능한 수준으로 재현되길 기대합니다.",
                    "추천 가능한 만족 재현",
                )
            if has_smell_signal:
                return (
                    f"사용자는 {product_value} 사용 시 냄새 관리가 예상한 수준으로 안정적으로 유지되길 기대합니다.",
                    "냄새 관리 안정성",
                )
            if has_manual_work_signal:
                return (
                    f"사용자는 {product_value} 사용 과정에서 불필요한 손작업 없이 편의성이 유지되길 기대합니다.",
                    "편의성 유지",
                )
            if has_noise_signal:
                return (
                    f"사용자는 {product_value} 성능이 유지되면서도 생활 소음 부담이 낮길 기대합니다.",
                    "성능·소음 균형",
                )
            return (
                f"사용자는 {product_value}에서 '{focus_value}'가 영상/설명에서 본 수준으로 재현되길 기대합니다.",
                f"{focus_value} 체감 재현",
            )

        return (
            f"사용자는 {product_value} 사용에서 '{focus_value}' 체감 편차가 크지 않길 기대합니다.",
            f"{focus_value} 체감 일관성",
        )

    def _derive_perception(
        sentiment_value: str,
        archetype: str,
        lowered_text: str,
        usage_value: str,
        expectation_short: str,
        representative_text: str,
    ) -> str:
        rep_lower = _safe_text(representative_text).lower()
        lowered_corpus = f"{lowered_text} {rep_lower}"
        if sentiment_value == "negative":
            if any(token in lowered_corpus for token in ["폐기", "버리", "junk"]):
                return "사용자는 이 경험을 단순 불편이 아니라 제품을 계속 쓰기 어려운 리스크로 인식합니다."
            if any(token in lowered_corpus for token in ["해지", "위약금", "권유", "약정"]):
                return "사용자는 제품 성능 이전에 판매·계약 안내 자체의 신뢰가 낮아졌다고 인식합니다."
            if any(token in lowered_corpus for token in ["돈낭비", "비추", "실망", "worst"]):
                return "사용자는 지불한 비용만큼의 성능 설득력이 부족하다고 인식합니다."
            if any(token in lowered_corpus for token in ["냄새", "악취", "odor", "smell"]):
                return "사용자는 위생 관리 부담이 제품 밖으로 전가됐다고 인식합니다."
            if any(token in lowered_corpus for token in ["손으로", "직접", "떼어", "붙어"]):
                return "사용자는 자동화 제품인데도 수동 노동이 남는다고 인식합니다."
            if archetype == "responsibility_shift":
                return "사용자는 문제 원인보다 책임이 자신에게 돌아온다고 인식해 대응 신뢰를 낮춥니다."
            return f"사용자는 {usage_value}에서 '{expectation_short}' 기대가 충족되지 않아 제품과 지원 체계 신뢰를 낮춥니다."

        if sentiment_value == "positive":
            if any(token in lowered_corpus for token in ["추천", "재구매", "recommend"]):
                return "사용자는 이 경험을 타인에게 추천 가능한 확신으로 인식합니다."
            if any(token in lowered_corpus for token in ["조용", "편하", "만족", "최고", "great", "excellent"]):
                return "사용자는 핵심 성능이 일상 기준을 충족해 선택이 타당했다고 인식합니다."
            if any(token in lowered_corpus for token in ["냄새", "악취", "odor", "smell"]):
                return "사용자는 위생/냄새 관리에서 체감한 안정성이 재구매 신뢰로 이어진다고 인식합니다."
            if any(token in lowered_corpus for token in ["손으로", "직접", "편하", "간편"]):
                return "사용자는 사용 절차가 기대보다 단순해 일상 루틴에 무리 없이 정착된다고 인식합니다."
            return f"사용자는 '{expectation_short}' 기대가 실제로 확인돼 선택 신뢰가 강화됐다고 인식합니다."

        return "사용자는 현재 신호를 추가 확인이 필요한 혼합 반응으로 인식합니다."

    def _extract_perception_evidence(text: str) -> str:
        cues = [
            "비추", "돈낭비", "실망", "폐기", "버리", "해지", "위약금", "권유",
            "추천", "만족", "최고", "좋", "great", "recommend",
        ]
        clauses = _pick_salient_clauses(text, limit=4)
        lowered_clauses = [(clause, clause.lower()) for clause in clauses if _safe_text(clause)]
        for cue in cues:
            for clause, lower_clause in lowered_clauses:
                if cue in lower_clause:
                    return _summarize_original_for_preview(clause, limit=72)
        if clauses:
            return _summarize_original_for_preview(clauses[0], limit=72)
        return ""

    core_points, context_tags, _, _, _, _ = _extract_row_nlp_metadata(working_selected)
    focus_tokens = core_points or _clean_point_tokens(keywords, limit=4)
    focus_point = _safe_text(focus_tokens[0]) if focus_tokens else _localize_aspect_label(aspect_label)
    product = canonicalize_product_label(_safe_text(working_selected.get("product", ""))) or "해당 제품"
    source_text = _combine_unique_text_parts(
        [
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            _safe_text(working_selected.get("cleaned_text", "")),
        ]
    )
    lowered = source_text.lower()
    representative_clause = _pick_salient_clauses(source_text, limit=1)
    representative_evidence = _summarize_original_for_preview(representative_clause[0], limit=72) if representative_clause else ""
    theme_tokens = _extract_video_theme_tokens(working_selected)
    theme_hint = ", ".join(theme_tokens[:2]) if theme_tokens else _localize_aspect_label(aspect_label)
    similar_count = len(similar_comments) if similar_comments is not None else int(pd.to_numeric(working_selected.get("cluster_size", 1), errors="coerce") or 1)
    video_signals = _collect_video_context_signals(working_selected, source_comments)
    usage_context = _infer_usage_context(source_text)
    similar_evidences = _extract_similar_comment_evidence(similar_comments, limit=2)
    video_evidence_raw = next((item for item in (video_signals.get("evidence_clauses", []) or []) if _safe_text(item)), "")
    video_evidence = _summarize_original_for_preview(video_evidence_raw, limit=72) if video_evidence_raw else ""

    has_purchase_conflict = any(
        marker in lowered for marker in ["구독", "가입", "해지", "약정", "위약금", "권유", "가격", "비용", "할부", "결제"]
    ) or any(tag in context_tags for tag in ["구독/계약", "가격·비용", "구매 전환"])
    has_responsibility_shift = any(
        marker in lowered for marker in ["사용자", "사용법", "잘못", "탓", "관리 못", "네가", "본인 책임"]
    )
    has_cluster_signal = similar_count >= 8

    candidate_archetypes: list[str] = []
    if sentiment_name == "negative":
        if has_purchase_conflict:
            candidate_archetypes.append("purchase_vs_use_conflict")
        if has_responsibility_shift:
            candidate_archetypes.append("responsibility_shift")
        if has_cluster_signal:
            candidate_archetypes.append("cluster_signal_gap")
        candidate_archetypes.append("expectation_gap")
    elif sentiment_name == "positive":
        if has_purchase_conflict:
            candidate_archetypes.append("purchase_expectation_alignment")
        candidate_archetypes.append("expectation_gap_resolved")
    else:
        candidate_archetypes.append("mixed_signal")

    product_key = product or "unknown"
    used_archetypes = insight_registry.setdefault(product_key, set()) if insight_registry is not None else set()
    selected_archetype = next((arc for arc in candidate_archetypes if arc not in used_archetypes), candidate_archetypes[0])
    if insight_registry is not None:
        used_archetypes.add(selected_archetype)

    expectation_sentence, expectation_short = _derive_expectation(
        sentiment_value=sentiment_name,
        product_value=product,
        focus_value=focus_point,
        usage_value=usage_context,
        archetype=selected_archetype,
        tags=context_tags,
        lowered_text=lowered,
        representative_text=source_text,
        similar_texts=similar_evidences,
    )
    domain_signal = _detect_domain_signal(
        corpus_text=" ".join([source_text] + similar_evidences),
        tags=context_tags,
        usage_value=usage_context,
        sentiment_value=sentiment_name,
    )
    if sentiment_name == "negative":
        actual_sentence = (
            f"실제 사용 장면에서는 대표 코멘트의 '{representative_evidence or focus_point}'처럼 {usage_context}에서 예상과 다른 마찰이 직접 확인됩니다."
        )
    elif sentiment_name == "positive":
        actual_sentence = (
            f"실제 사용 장면에서는 대표 코멘트의 '{representative_evidence or focus_point}'처럼 {usage_context}에서 기대했던 장점이 그대로 확인됩니다."
        )
    else:
        actual_sentence = (
            f"실제 사용 장면에서는 대표 코멘트의 '{representative_evidence or focus_point}'처럼 {usage_context}에서 상반된 신호가 함께 관찰됩니다."
        )
    perception_sentence = _derive_perception(
        sentiment_value=sentiment_name,
        archetype=selected_archetype,
        lowered_text=lowered,
        usage_value=usage_context,
        expectation_short=expectation_short,
        representative_text=source_text,
    )
    if theme_tokens:
        if sentiment_name == "positive":
            context_sentence = f"영상에서 강조된 {theme_hint} 맥락이 댓글 경험과 같은 방향으로 이어져 긍정 해석 근거를 강화합니다."
        elif sentiment_name == "negative":
            context_sentence = f"영상에서 강조된 {theme_hint} 맥락과 댓글 경험이 어긋나며 부정 해석 근거를 강화합니다."
        else:
            context_sentence = f"영상에서 강조된 {theme_hint} 맥락과 댓글 경험이 일부 일치·일부 불일치합니다."
    else:
        context_sentence = ""

    insight = _compose_insight_text(
        expectation_sentence=expectation_sentence,
        actual_sentence=actual_sentence,
        perception_sentence=perception_sentence,
        context_sentence=context_sentence,
        sentiment_value=sentiment_name,
        archetype=selected_archetype,
        domain_signal=domain_signal,
    )

    supporting_signals: list[str] = []
    if representative_evidence:
        supporting_signals.append(
            f"대표 코멘트의 '{representative_evidence}' 표현이 사용자 기대('{expectation_short}')와 실제 경험의 차이를 직접 보여줍니다."
        )
    if similar_evidences:
        supporting_signals.append(
            f"유사 댓글의 '{similar_evidences[0]}' 사례가 같은 사용 장면에서 동일한 판단 방향을 확인시켜 줍니다."
        )
    if len(similar_evidences) >= 2:
        supporting_signals.append(
            f"추가 유사 댓글 '{similar_evidences[1]}'에서도 같은 문제 인식(또는 같은 강점 인식)이 반복됩니다."
        )
    perception_evidence = _extract_perception_evidence(source_text)
    if perception_evidence and perception_evidence not in " ".join(supporting_signals):
        supporting_signals.append(
            f"원문의 '{perception_evidence}' 문장이 사용자의 최종 인식 변화(불신/만족/추천)를 뒷받침합니다."
        )
    if video_evidence:
        supporting_signals.append(
            f"같은 영상 맥락의 '{video_evidence}' 사례가 대표 코멘트와 유사한 기대-경험 간격(또는 일치)을 보여줍니다."
        )

    deduped_signals: list[str] = []
    seen_signals: set[str] = set()
    for signal_item in supporting_signals:
        cleaned_signal = _safe_text(signal_item)
        if not cleaned_signal:
            continue
        if cleaned_signal in seen_signals:
            continue
        deduped_signals.append(cleaned_signal)
        seen_signals.add(cleaned_signal)
    supporting_signals = deduped_signals
    if len(supporting_signals) < 2:
        supporting_signals.append(
            f"대표 코멘트는 '{focus_point}'에 대한 사용자 기대와 실제 체감을 동시에 담고 있어 Insight 해석의 직접 근거로 사용됩니다."
        )

    return {
        "insight": insight,
        "supporting_signals": supporting_signals[:4],
    }


def _build_resolution_point_legacy_template(
    working_selected: pd.Series | str,
    sentiment_name: str | list[str],
    keywords: list[str] | None = None,
    inquiry_flag: bool = False,
    negative_context: dict[str, str] | None = None,
) -> str:
    """Legacy template-based generator kept for before/after validation comparisons."""
    if isinstance(working_selected, pd.Series):
        row = working_selected
        sentiment_value = _safe_text(sentiment_name).lower()
        keyword_values = keywords or []
    else:
        row = pd.Series(dtype=object)
        sentiment_value = _safe_text(working_selected).lower()
        if isinstance(sentiment_name, list):
            keyword_values = sentiment_name
        else:
            keyword_values = keywords or []

    core_points, context_tags, _, _, _, _ = _extract_row_nlp_metadata(row)
    focus_tokens = core_points or _clean_point_tokens(keyword_values, limit=4)
    focus = ", ".join(focus_tokens[:2]) if focus_tokens else "핵심 이슈"
    source_text = _combine_unique_text_parts(
        [
            _safe_text(row.get(COL_ORIGINAL, "")),
            _safe_text(row.get(COL_TRANSLATION, "")),
            _safe_text(row.get("cleaned_text", "")),
        ]
    )
    evidence_clause = _summarize_original_for_preview((_pick_salient_clauses(source_text, limit=1) or [focus])[0], limit=68)
    cluster_size = int(pd.to_numeric(row.get("cluster_size", 1), errors="coerce") or 1)
    product_hint = canonicalize_product_label(_safe_text(row.get("product", ""))) or "해당 제품"
    focus_hint = _safe_text(focus_tokens[0]) if focus_tokens else "핵심 불편"
    primary_tag = _safe_text(context_tags[0]) if context_tags else ""
    if not primary_tag:
        hint_blob = " ".join(
            [
                source_text.lower(),
                " ".join([_safe_text(token).lower() for token in focus_tokens]),
            ]
        )
        inferred_rules = [
            (["a/s", "as", "repair", "service", "support", "고장", "수리", "기사", "센터"], "A/S·수리"),
            (["delivery", "install", "shipping", "배송", "설치", "기사방문"], "배송·설치"),
            (["subscription", "contract", "price", "cost", "fee", "구독", "계약", "약정", "요금", "가격", "해지"], "구독/계약"),
            (["performance", "quality", "noise", "smell", "leak", "세척", "성능", "품질", "소음", "냄새", "누수"], "성능·품질"),
            (["usability", "ui", "button", "manual", "불편", "사용", "조작", "청소", "무거"], "사용성·편의"),
        ]
        for tokens, inferred_tag in inferred_rules:
            if any(token in hint_blob for token in tokens):
                primary_tag = inferred_tag
                break
        if not primary_tag:
            keyword_blob = " ".join([_safe_text(token).lower() for token in keyword_values])
            if any(token in keyword_blob for token in ["구독", "가입", "약정", "해지", "위약금"]):
                primary_tag = "구독/계약"
    inquiry_suffix = " 문의 성격이 강하므로 응답 문구를 고객 언어로 먼저 정리하세요." if inquiry_flag else ""
    detail_blob = " ".join([source_text.lower(), _safe_text(focus_hint).lower(), _safe_text(evidence_clause).lower()])

    if sentiment_value == "negative":
        priority, _ = _negative_priority_bucket(
            negative_context=negative_context,
            cluster_size=cluster_size,
            inquiry_flag=inquiry_flag,
        )
        if any(token in detail_blob for token in ["냄새", "smell", "odor"]):
            as_action = f"{product_hint} 사용 중 냄새 민원이 반복되므로 밀폐 구조·배기 경로·필터 교체주기 안내를 함께 수정해 체감 냄새를 낮추세요."
        elif any(token in detail_blob for token in ["필터", "filter", "소모품"]):
            as_action = f"{product_hint}의 필터/소모품 교체 타이밍에서 혼선이 생기므로 교체 기준·증상별 안내·구매 동선을 한 화면으로 통합하세요."
        elif any(token in detail_blob for token in ["고객지원", "support", "service", "응대", "센터"]):
            as_action = f"{product_hint} 지원 문의가 반복되는 구간을 접수→답변→완료 흐름으로 재설계하고, 첫 답변에서 해결 경로를 명확히 제시하세요."
        else:
            as_action = f"{product_hint}에서 반복된 '{focus_hint}' 불편은 수리 체감 공백이 핵심이므로 고장유형별 1차 진단 기준과 방문 전 부품확정 절차를 표준화하세요."
        negative_actions = {
            "구독/계약": f"'{evidence_clause}'에서 드러난 가입·해지 혼선을 줄이기 위해 {product_hint} 월납/약정/해지비용(위약금) 안내를 한 흐름으로 재작성하고, 과도 권유 멘트는 즉시 교정하세요.",
            "A/S·수리": as_action,
            "배송·설치": f"'{evidence_clause}' 사례를 기준으로 일정 지연과 현장 안내 누락을 분리해 점검하고, 설치 직후 7일 케어 안내를 보강하세요.",
            "가격·비용": f"'{evidence_clause}'처럼 비용 인식이 틀어지는 지점을 찾아 총비용(월납·할부·해지비용) 안내 문구와 결제 직전 고지를 함께 손보세요.",
            "성능·품질": f"'{evidence_clause}' 상황을 재현 테스트로 등록하고, 냄새·소음·세척력처럼 직접 불편을 만드는 항목부터 제품/가이드 개선안을 우선 적용하세요.",
            "사용성·편의": f"'{evidence_clause}'에서 확인된 사용 마찰을 동작 순서 기준으로 쪼개 버튼/안내/청소 동선을 재설계하세요.",
        }
        default_action = f"{product_hint} 사용에서 드러난 '{focus_hint}' 문제를 기준으로 제품·정책·운영 중 책임 영역을 고정하고, 고객이 체감하는 개선 여부를 다음 배치에서 확인하세요."
        return f"[{priority}] {negative_actions.get(primary_tag, default_action)}{inquiry_suffix}"

    if sentiment_value == "positive":
        if "비교 맥락" in context_tags:
            return f"[강점 확장] '{evidence_clause}' 비교 우위를 구매 직전 접점(상세 설명/고정 댓글/짧은 영상)에 같은 표현으로 반복 노출해 전환 근거로 쓰세요."
        if "추천 의도" in context_tags:
            return f"[강점 확장] '{evidence_clause}'처럼 추천 의도가 확인된 강점은 추천 후기 묶음과 함께 노출해 지인 추천 흐름을 강화하세요."
        if "구매 전환" in context_tags:
            return f"[강점 확장] '{evidence_clause}' 구매 결심 포인트를 결제 직전 Q&A와 비교표 상단에 배치해 이탈 구간을 줄이세요."
        return f"[강점 확장] '{evidence_clause}'에서 드러난 {focus} 강점을 재구매·추천 상황에 맞춰 메시지 우선순위로 반영하세요."

    return f"[관찰] '{evidence_clause}' 반응은 혼합 신호이므로 성급한 단정 대신 동일 맥락 재등장 여부를 확인한 뒤 대응 단계를 결정하세요."


def _extract_similar_comment_clauses(frame: pd.DataFrame | None, limit: int = 3) -> list[str]:
    if frame is None or frame.empty or "원문" not in frame.columns:
        return []
    clauses: list[str] = []
    for raw_text in frame["원문"].head(30).tolist():
        text_value = _safe_text(raw_text)
        if not text_value:
            continue
        picked = _pick_salient_clauses(text_value, limit=1)
        if not picked:
            continue
        snippet = _summarize_original_for_preview(picked[0], limit=78)
        if snippet and snippet not in clauses:
            clauses.append(snippet)
        if len(clauses) >= limit:
            break
    return clauses


def _infer_resolution_problem_signal(
    representative_text: str,
    similar_clauses: list[str],
    context_tags: list[str],
    focus_points: list[str],
) -> str:
    corpus = " ".join(
        [
            _safe_text(representative_text),
            " ".join(similar_clauses),
            " ".join(context_tags),
            " ".join(focus_points),
        ]
    ).lower()
    signal_rules = [
        (["냄새", "악취", "odor", "smell"], "odor_leakage"),
        (["필터", "소모품", "교체"], "consumable_burden"),
        (["손으로", "직접", "붙어", "떼어", "scrub", "manual"], "manual_intervention"),
        (["고장", "불량", "멈춤", "누수", "파손", "fail", "broken"], "reliability_failure"),
        (["a/s", "as", "수리", "기사", "센터", "지연", "출장"], "service_response_gap"),
        (["배송", "설치", "방문", "일정", "공간"], "installation_delivery_gap"),
        (["가격", "비용", "할부", "약정", "해지", "위약금", "구독", "price", "cost", "money", "expensive"], "pricing_contract_gap"),
        (["안내", "설명", "매뉴얼", "설정", "사용법", "초기", "세팅"], "onboarding_guide_gap"),
        (["광고", "과장", "홍보", "영상처럼", "기대와", "다름"], "message_expectation_gap"),
        (["추천", "재구매", "지인", "소개", "만족", "좋"], "advocacy_signal"),
    ]
    for tokens, signal_name in signal_rules:
        if any(token in corpus for token in tokens):
            return signal_name
    return "general_experience"


def _infer_resolution_action_domain(
    journey_stage: str,
    sentiment_value: str,
    mixed_flag: bool,
    signal_name: str,
) -> str:
    domain_scores: Counter[str] = Counter(
        {
            "product": 0,
            "UX / guide": 0,
            "operations / service": 0,
            "marketing / message": 0,
            "business / policy": 0,
        }
    )

    journey_to_domain = {
        "인지": [("marketing / message", 2)],
        "탐색": [("marketing / message", 2), ("business / policy", 1)],
        "구매": [("business / policy", 2), ("marketing / message", 1)],
        "결정": [("business / policy", 2), ("marketing / message", 1)],
        "배송": [("operations / service", 3)],
        "사용준비": [("UX / guide", 3), ("operations / service", 1)],
        "사용": [("product", 2), ("UX / guide", 1)],
        "관리": [("operations / service", 2), ("product", 1)],
        "교체": [("business / policy", 2), ("product", 1)],
    }
    for domain, weight in journey_to_domain.get(journey_stage, [("operations / service", 1)]):
        domain_scores[domain] += weight

    signal_to_domain = {
        "odor_leakage": ("product", 3),
        "consumable_burden": ("UX / guide", 2),
        "manual_intervention": ("product", 2),
        "reliability_failure": ("product", 3),
        "service_response_gap": ("operations / service", 3),
        "installation_delivery_gap": ("operations / service", 3),
        "pricing_contract_gap": ("business / policy", 3),
        "onboarding_guide_gap": ("UX / guide", 3),
        "message_expectation_gap": ("marketing / message", 3),
        "advocacy_signal": ("marketing / message", 2),
    }
    if signal_name in signal_to_domain:
        signal_domain, signal_weight = signal_to_domain[signal_name]
        domain_scores[signal_domain] += signal_weight

    if sentiment_value == "positive":
        domain_scores["marketing / message"] += 1
    if sentiment_value == "negative":
        domain_scores["operations / service"] += 1
    if mixed_flag:
        domain_scores["UX / guide"] += 1

    priority_order = ["product", "operations / service", "business / policy", "UX / guide", "marketing / message"]
    return sorted(priority_order, key=lambda item: (-domain_scores[item], priority_order.index(item)))[0]


def _resolve_journey_stage_for_resolution(
    row: pd.Series,
    context_tags: list[str],
    signal_name: str,
) -> tuple[str, bool]:
    direct_stage = _journey_value_from_row(row)
    if direct_stage and direct_stage != "기타":
        return direct_stage, False

    classification = _safe_text(row.get("classification_type", "")).strip().lower()
    context_blob = " ".join([_safe_text(tag).lower() for tag in context_tags])
    text_blob = " ".join(
        [
            _safe_text(row.get(COL_ORIGINAL, "")),
            _safe_text(row.get(COL_TRANSLATION, "")),
            _safe_text(row.get("cleaned_text", "")),
        ]
    ).lower()

    if any(token in context_blob for token in ["배송", "설치"]) or any(token in text_blob for token in ["배송", "설치", "방문"]):
        return "배송", True
    if any(token in context_blob for token in ["a/s", "수리", "고객지원", "문의"]) or signal_name in {"service_response_gap"}:
        return "관리", True
    if any(token in context_blob for token in ["구독", "계약", "구매 전환"]) or signal_name in {"pricing_contract_gap"}:
        return "구매", True
    if any(token in context_blob for token in ["비교", "탐색"]) or classification in {"comparison", "informational"}:
        return "탐색", True
    if any(token in context_blob for token in ["추천"]) or classification == "preference":
        return "결정", True
    if signal_name in {"manual_intervention", "odor_leakage", "reliability_failure", "consumable_burden", "onboarding_guide_gap"}:
        return "사용", True
    return "", False


def _build_resolution_evidence_bundle(
    row: pd.Series,
    sentiment_value: str,
    keywords: list[str] | None,
    inquiry_flag: bool,
    similar_comments: pd.DataFrame | None,
    source_comments: pd.DataFrame | None,
) -> dict[str, Any]:
    core_points, context_tags, _, _, _, _ = _extract_row_nlp_metadata(row)
    focus_points = core_points or _clean_point_tokens(keywords or [], limit=5)
    representative_text = _combine_unique_text_parts(
        [
            _safe_text(row.get(COL_ORIGINAL, "")),
            _safe_text(row.get(COL_TRANSLATION, "")),
            _safe_text(row.get("cleaned_text", "")),
            _safe_text(row.get("text_display", "")),
        ]
    )
    representative_clause = _pick_salient_clauses(representative_text, limit=1)
    representative_evidence = _summarize_original_for_preview(representative_clause[0], limit=82) if representative_clause else ""
    similar_clauses = _extract_similar_comment_clauses(similar_comments, limit=3)
    journey_stage_direct = _journey_value_from_row(row)
    mixed_flag = bool(_truthy(row.get("mixed_flag", False)) or sentiment_value == "mixed")
    similar_count = len(similar_comments) if similar_comments is not None else 0
    product_label = canonicalize_product_label(_safe_text(row.get("product", ""))) or "해당 제품"

    video_signals = _collect_video_context_signals(row, source_comments)
    context_evidence = ""
    for clause in video_signals.get("evidence_clauses", []):
        snippet = _summarize_original_for_preview(_safe_text(clause), limit=82)
        if snippet and snippet != representative_evidence:
            context_evidence = snippet
            break

    signal_name = _infer_resolution_problem_signal(
        representative_text=representative_text,
        similar_clauses=similar_clauses,
        context_tags=context_tags,
        focus_points=focus_points,
    )
    journey_stage, journey_inferred = _resolve_journey_stage_for_resolution(
        row=row,
        context_tags=context_tags,
        signal_name=signal_name,
    )
    if not journey_stage and journey_stage_direct:
        journey_stage = journey_stage_direct
        journey_inferred = False
    action_domain = _infer_resolution_action_domain(
        journey_stage=journey_stage,
        sentiment_value=sentiment_value,
        mixed_flag=mixed_flag,
        signal_name=signal_name,
    )
    insufficient_reasons: list[str] = []
    if not journey_stage or journey_stage == "기타":
        insufficient_reasons.append("journey_stage_missing")
    elif journey_inferred:
        # Phase 6.4 policy: inferred journey is limited fallback, not full-evidence mode.
        insufficient_reasons.append("journey_stage_inferred_fallback")
    if not representative_evidence:
        insufficient_reasons.append("representative_evidence_missing")
    if similar_count < 3:
        insufficient_reasons.append("similar_signal_lt_3")
    insufficient_evidence = bool(insufficient_reasons)

    return {
        "journey_stage": journey_stage,
        "sentiment": sentiment_value,
        "mixed_flag": mixed_flag,
        "inquiry_flag": inquiry_flag,
        "product_label": product_label,
        "focus_points": focus_points,
        "context_tags": context_tags,
        "representative_evidence": representative_evidence,
        "similar_evidence": similar_clauses,
        "similar_count": similar_count,
        "journey_inferred": journey_inferred,
        "context_evidence": context_evidence,
        "signal_name": signal_name,
        "action_domain": action_domain,
        "insufficient_evidence": insufficient_evidence,
        "insufficient_reasons": insufficient_reasons,
    }


def _compose_resolution_action_text(bundle: dict[str, Any]) -> str:
    def _pick_action_variant(options: str | list[str], signal: str) -> str:
        if isinstance(options, str):
            return options
        candidates = [item for item in options if _safe_text(item)]
        if not candidates:
            return ""
        seed = "|".join(
            [
                signal,
                _safe_text(bundle.get("journey_stage", "")),
                _safe_text(bundle.get("representative_evidence", "")),
                str(int(bundle.get("similar_count", 0) or 0)),
            ]
        )
        digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % len(candidates)
        return candidates[idx]

    product_label = _safe_text(bundle.get("product_label", "해당 제품")) or "해당 제품"
    sentiment_value = _safe_text(bundle.get("sentiment", "neutral")).lower()
    journey_stage = _safe_text(bundle.get("journey_stage", ""))
    mixed_flag = bool(bundle.get("mixed_flag", False))
    signal_name = _safe_text(bundle.get("signal_name", "general_experience"))
    action_domain = _safe_text(bundle.get("action_domain", "operations / service"))
    representative_evidence = _safe_text(bundle.get("representative_evidence", ""))
    context_evidence = _safe_text(bundle.get("context_evidence", ""))
    similar_evidence = bundle.get("similar_evidence", [])
    if not isinstance(similar_evidence, list):
        similar_evidence = []
    similar_evidence = [_safe_text(item) for item in similar_evidence if _safe_text(item)]
    similar_count = int(bundle.get("similar_count", 0) or 0)
    focus_points = bundle.get("focus_points", [])
    if not isinstance(focus_points, list):
        focus_points = []
    focus_text = ", ".join([_safe_text(token) for token in focus_points[:2] if _safe_text(token)]) or "핵심 사용 이슈"

    if bundle.get("insufficient_evidence", False):
        evidence_hint = representative_evidence or focus_text
        return (
            f"[근거 부족] '{evidence_hint}' 신호는 확인됐지만 고객 경험 여정 단계가 비어 있어 실행 우선순위를 확정할 수 없습니다. "
            f"여정 단계 라벨을 먼저 보완한 뒤 {action_domain} 관점에서 재평가하세요."
        )

    stage_phrase = f"{journey_stage} 단계"
    repeated_phrase = f"유사 댓글 {similar_count:,}건에서 같은 패턴이 반복" if similar_count >= 3 else "반복 신호는 제한적"
    supporting_phrase = f"대표 근거는 '{representative_evidence}'" if representative_evidence else f"대표 신호는 '{focus_text}'"
    similar_phrase = f"유사 근거는 '{similar_evidence[0]}'" if similar_evidence else ""
    context_phrase = f"영상 맥락에서는 '{context_evidence}'가 함께 관찰" if context_evidence else ""
    connector = " / ".join([piece for piece in [supporting_phrase, similar_phrase, context_phrase] if piece])

    negative_actions: dict[str, str | list[str]] = {
        "odor_leakage": [
            f"{stage_phrase}에서 냄새 차단 실패가 반복되므로 밀폐 구조·배기 경로·필터 수명 조건을 실사용 환경 기준으로 재설계하세요.",
            f"냄새 민원이 {stage_phrase}에서 누적되므로 {product_label}의 배기 구조와 필터 교체 기준을 분리 진단해 즉시 수정하세요.",
        ],
        "consumable_burden": [
            f"{stage_phrase}에서 소모품/교체 부담이 마찰이므로 교체 조건 안내와 구매 동선을 하나의 유지관리 시나리오로 통합하세요.",
            f"{product_label} 유지관리 부담이 {stage_phrase}에서 커지므로 소모품 교체 시점·비용·구매 경로를 단일 가이드로 정리하세요.",
        ],
        "manual_intervention": [
            f"{stage_phrase}에서 자동 처리 기대가 수동 개입으로 깨지므로 잔여물 부착 방지 설계와 자동 세정 가이드를 동시에 보강하세요.",
            f"수동 개입이 {stage_phrase} 핵심 불만으로 확인되므로 자동 처리 실패 구간을 줄이는 설계 보정과 사용 절차 단축을 병행하세요.",
        ],
        "reliability_failure": [
            f"{stage_phrase}에서 기능 고장 신호가 뚜렷하므로 고장 유형별 재현 테스트와 부품 신뢰성 개선 항목을 제품 백로그 우선순위로 상향하세요.",
            f"{product_label} 고장 패턴이 {stage_phrase}에서 반복되므로 재현 가능한 증상부터 부품/펌웨어 안정성 개선 과제로 즉시 분리하세요.",
        ],
        "service_response_gap": [
            f"{stage_phrase}에서 A/S 응답 공백이 확인되므로 접수→진단→해결 리드타임 기준을 분리 관리하고 지연 원인별 대응 SLA를 고정하세요.",
            f"A/S 지연이 {stage_phrase} 핵심 마찰로 확인되어 접수-현장방문-완료 구간별 SLA를 분리하고 누락 케이스를 주간 점검하세요.",
            f"{stage_phrase} 고객경험 저하 원인이 대응 지연으로 수렴하므로 {product_label} A/S를 유형별 우선순위 큐로 운영해 병목을 줄이세요.",
        ],
        "installation_delivery_gap": [
            f"{stage_phrase}에서 설치·배송 경험 손실이 반복되므로 일정 안내, 현장 체크리스트, 설치 후 검수 절차를 한 흐름으로 재정의하세요.",
            f"설치/배송 오류가 {stage_phrase}에서 재발하므로 방문 전 사전안내와 설치 후 검수 항목을 고객 확인 단계까지 고정하세요.",
        ],
        "pricing_contract_gap": [
            f"{stage_phrase}에서 가격/약정 인식 충돌이 발생하므로 총비용·해지비용·약정 조건을 결제 직전 동일 화면에서 선명하게 고지하세요.",
            f"{stage_phrase} 의사결정에서 약정/해지 비용 혼선이 반복되어 결제 전 총비용과 계약 조건을 한 문맥으로 재배치해야 합니다.",
        ],
        "onboarding_guide_gap": [
            f"{stage_phrase}에서 초기 사용 가이드 부족이 드러나므로 첫 사용 7일 가이드를 실제 실패 장면 중심으로 재작성하고 단계별 안내를 단축하세요.",
            f"초기 세팅 실패가 {stage_phrase} 불만으로 반복되므로 첫 7일 온보딩 안내를 핵심 오류 장면 중심으로 재구성하세요.",
        ],
        "message_expectation_gap": [
            f"{stage_phrase}에서 메시지 기대와 실제 체감 간극이 커서 광고/상세 설명의 약속 범위를 실사용 조건 중심으로 재정렬해야 합니다.",
            f"콘텐츠 약속과 실사용 체감이 {stage_phrase}에서 어긋나므로 {product_label} 메시지를 실제 사용조건 중심으로 다시 정의하세요.",
        ],
    }
    positive_actions: dict[str, str | list[str]] = {
        "advocacy_signal": [
            f"{stage_phrase}에서 추천/재구매 신호가 확인되므로 해당 강점을 후기 맥락과 함께 노출해 전환 근거를 강화하세요.",
            f"{stage_phrase}에서 추천 의도가 반복되므로 {product_label} 강점을 비교 근거와 묶어 재구매/추천 동선에 배치하세요.",
        ],
        "reliability_failure": [
            f"{stage_phrase}에서 내구 강점이 긍정 신호로 작동하므로 장기간 사용 근거를 메시지·FAQ·비교표에 일관되게 연결하세요.",
            f"내구 만족 신호가 {stage_phrase}에서 누적되어 장기 사용 사례를 핵심 메시지로 고정해 신뢰 인식을 확장하세요.",
        ],
        "consumable_burden": [
            f"{stage_phrase}에서 유지관리 편의가 강점으로 작동하므로 교체/세척이 쉬운 이유를 사용 시나리오 중심으로 강조하세요.",
            f"{product_label} 관리 편의 강점이 {stage_phrase}에서 확인되므로 세척·교체 절차의 단순성을 구체 사례와 함께 노출하세요.",
        ],
    }
    mixed_actions: dict[str, str | list[str]] = {
        "pricing_contract_gap": [
            f"{stage_phrase}에서 가치 만족과 비용 불신이 동시에 나타나므로 가격 체계 투명화와 강점 메시지 노출을 분리해 재설계하세요.",
            f"가격 납득감과 성능 만족이 충돌하는 {stage_phrase} 구간이므로 비용 고지와 강점 커뮤니케이션을 분리 운영하세요.",
        ],
        "service_response_gap": [
            f"{stage_phrase}에서 성능 만족과 대응 불만이 함께 존재하므로 제품 강점 유지와 서비스 리드타임 개선을 병렬 과제로 운영하세요.",
            f"{stage_phrase}에서 만족과 불만이 동시에 관찰되어 기능 강점은 유지하되 A/S 응답 구간만 별도 개선 트랙으로 분리하세요.",
        ],
        "manual_intervention": [
            f"{stage_phrase}에서 결과 만족과 수동 불편이 공존하므로 자동화 실패 구간을 줄이는 제품 보완과 사용 가이드를 함께 개편하세요.",
            f"사용 결과 만족에도 수동 개입 피로가 남는 {stage_phrase} 구간이라 자동화 품질 보정과 가이드 간소화를 동시 추진하세요.",
        ],
    }

    if mixed_flag:
        action_core = _pick_action_variant(mixed_actions.get(
            signal_name,
            f"{stage_phrase}에서 기대 충족과 불편 신호가 동시에 관찰되어 강점 유지와 마찰 제거를 분리한 이중 실행 계획이 필요합니다.",
        ), signal_name)
    elif sentiment_value == "positive":
        action_core = _pick_action_variant(positive_actions.get(
            signal_name,
            f"{stage_phrase}에서 확인된 '{focus_text}' 강점을 고객 의사결정 접점에 재사용해 추천·재구매 전환으로 연결하세요.",
        ), signal_name)
    elif sentiment_value == "negative":
        action_core = _pick_action_variant(negative_actions.get(
            signal_name,
            f"{stage_phrase}에서 드러난 '{focus_text}' 마찰을 {action_domain} 영역의 우선 개선 항목으로 등록해 다음 배치에서 재검증하세요.",
        ), signal_name)
    else:
        action_core = f"{stage_phrase}에서 신호가 혼재해 즉시 단정은 어렵습니다. '{focus_text}' 관련 증거를 추가 확보한 뒤 {action_domain} 실행안을 확정하세요."

    if sentiment_value == "positive":
        lead_options = [
            "강화 제안: ",
            f"{stage_phrase} 확장 포인트: ",
            f"{product_label} 강점 활용: ",
            "고객 만족 레버리지: ",
            "전략적 강점: ",
        ]
    else:
        lead_options = [
            "실행 우선순위: ",
            f"{stage_phrase} 개선 제안: ",
            f"{action_domain} 실행안: ",
            f"{product_label} 대응 제안: ",
            "즉시 점검 포인트: ",
        ]
    lead_text = _pick_action_variant(lead_options, f"lead::{signal_name}")
    composed_action = f"{lead_text}{action_core}"

    if connector:
        return f"{composed_action} ({repeated_phrase}; {connector})"
    return f"{composed_action} ({repeated_phrase})"


def _build_resolution_point(
    working_selected: pd.Series | str,
    sentiment_name: str | list[str],
    keywords: list[str] | None = None,
    inquiry_flag: bool = False,
    negative_context: dict[str, str] | None = None,
    similar_comments: pd.DataFrame | None = None,
    source_comments: pd.DataFrame | None = None,
) -> dict[str, Any] | str:
    """Evidence-bundle based resolution generator with mandatory journey-stage handling."""
    if not isinstance(working_selected, pd.Series):
        # Keep old non-Series signature for legacy tests/callers.
        return _build_resolution_point_legacy_template(
            working_selected=working_selected,
            sentiment_name=sentiment_name,
            keywords=keywords,
            inquiry_flag=inquiry_flag,
            negative_context=negative_context,
        )

    sentiment_value = _safe_text(sentiment_name).lower()
    if sentiment_value not in {"positive", "negative", "neutral", "mixed"}:
        sentiment_value = _normalize_sentiment_direction(
            working_selected.get("sentiment_final", working_selected.get("sentiment_label", "neutral"))
        )
    bundle = _build_resolution_evidence_bundle(
        row=working_selected,
        sentiment_value=sentiment_value,
        keywords=keywords,
        inquiry_flag=inquiry_flag,
        similar_comments=similar_comments,
        source_comments=source_comments,
    )
    action_text = _compose_resolution_action_text(bundle)
    return {
        "action_text": action_text,
        "action_domain": bundle.get("action_domain", "operations / service"),
        "evidence_trace": {
            "representative_evidence": bundle.get("representative_evidence", ""),
            "similar_evidence": bundle.get("similar_evidence", []),
            "similar_count": int(bundle.get("similar_count", 0) or 0),
            "journey_stage": bundle.get("journey_stage", ""),
            "journey_inferred": bool(bundle.get("journey_inferred", False)),
            "polarity": bundle.get("sentiment", sentiment_value),
            "mixed_flag": bool(bundle.get("mixed_flag", False)),
            "signal_name": bundle.get("signal_name", "general_experience"),
            "context_evidence": bundle.get("context_evidence", ""),
            "focus_points": bundle.get("focus_points", []),
            "context_tags": bundle.get("context_tags", []),
            "insufficient_reasons": bundle.get("insufficient_reasons", []),
        },
        "insufficient_evidence": bool(bundle.get("insufficient_evidence", False)),
    }


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
    if re.match(r"^(이슈\s*\d+|issue\s*\d+)$", label.lower()):
        label = "General"

    core_points, context_tags, _, _, _, _ = _extract_row_nlp_metadata(working_selected)
    if context_tags:
        context_map = {
            "구독/계약": "Cost",
            "A/S·수리": "A/S & Support",
            "배송·설치": "Delivery/Install",
            "가격·비용": "Cost",
            "성능·품질": "Performance",
            "사용성·편의": "Convenience/Usability",
            "비교 맥락": "Product Experience",
            "구매 전환": "Cost",
            "추천 의도": "Product Experience",
            "문의/해결요청": "A/S & Support",
        }
        for tag in context_tags:
            mapped = context_map.get(_safe_text(tag))
            if mapped:
                return mapped

    if label.lower() not in {"general", "기타", "other", "대표 코멘트"}:
        return label

    text_blob = " ".join(
        [
            _safe_text(display_text),
            _safe_text(working_selected.get(COL_ORIGINAL, "")),
            _safe_text(working_selected.get(COL_TRANSLATION, "")),
            " ".join(core_points or keywords),
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


def _reason_conflicts_with_sentiment(reason_text: str, sentiment_value: str) -> bool:
    reason = _safe_text(reason_text).lower()
    if not reason:
        return False
    negative_markers = ["불만", "문제", "고장", "불편", "환불", "불량", "리스크", "개선 필요", "저하"]
    positive_markers = ["만족", "추천", "강점", "우수", "좋", "최고", "전환", "기여"]
    neg_hits = sum(1 for marker in negative_markers if marker in reason)
    pos_hits = sum(1 for marker in positive_markers if marker in reason)
    if sentiment_value == "positive" and neg_hits >= pos_hits + 1:
        return True
    if sentiment_value == "negative" and pos_hits >= neg_hits + 1:
        return True
    return False


def _format_confidence_factor_labels(factors: list[str], breakdown: dict[str, float]) -> str:
    if not factors:
        return "기준 정보 부족"
    label_map = {
        "topic_alignment": "토픽 정합성",
        "keyword_signal": "키워드 신호",
        "point_extraction": "핵심 포인트 추출",
        "context_signal": "맥락 태그 신호",
        "summary_quality": "요약 품질",
        "reason_quality": "판단 근거 품질",
        "negative_marker": "부정 표현 강도",
        "positive_marker": "긍정 표현 강도",
        "inquiry_penalty": "문의 감점",
        "inquiry_signal": "문의 신호",
        "base_model_signal": "모델 기본 신호",
    }
    labels: list[str] = []
    for factor in factors[:4]:
        key = _safe_text(factor)
        if not key:
            continue
        human = label_map.get(key, key)
        if key in breakdown:
            labels.append(f"{human}({breakdown[key]:+0.2f})")
        else:
            labels.append(human)
    return ", ".join(labels) if labels else "기준 정보 부족"


def _render_representative_nlp_panel(
    working_selected: pd.Series,
    sentiment_name: str,
    aspect_label: str,
    display_text: str,
    source_comments: pd.DataFrame | None = None,
    negative_context: dict[str, str] | None = None,
    similar_comments: pd.DataFrame | None = None,
    insight_registry: dict[str, set[str]] | None = None,
) -> None:
    sentiment_value = _resolve_sentiment_for_card(working_selected, sentiment_name)
    sentiment_map = {"positive": "긍정", "negative": "부정", "neutral": "중립", "trash": "스팸"}
    sentiment_label = sentiment_map.get(sentiment_value, sentiment_value)
    ratio = _confidence_ratio_for_card(working_selected, sentiment_value)
    score_pct = int(round(ratio * 100))
    score_text = f"{score_pct}%"
    keywords = _extract_card_keywords(working_selected, sentiment_value, limit=5)
    core_points, context_tags, _, confidence_factors, confidence_breakdown, insight_summary = _extract_row_nlp_metadata(working_selected)
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
    bad_reason_tokens = {
        "감성 판단 근거를 구조화하기 위한 정보가 부족해, 원문 맥락 중심으로 확인이 필요합니다.",
        "이 댓글은 해당 제품 관련 핵심 성능 체감 이슈를 중심으로 분류되었습니다.",
    }
    generic_reason_markers = (
        "실제 불만이 명확하게 드러나",
        "만족감이나 추천 의사가 명확하여",
        "비즈니스 관점에서는",
        "중립적 관찰에 가깝",
        "분류했습니다",
    )
    reason = next(
        (
            item
            for item in reason_candidates
            if item
            and not _looks_garbled(item)
            and item not in bad_reason_tokens
            and len(item) >= 12
            and not any(marker in item for marker in generic_reason_markers)
            and not _reason_conflicts_with_sentiment(item, sentiment_value)
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
    factor_text = _format_confidence_factor_labels(confidence_factors, confidence_breakdown)
    reason = f"{reason} (판단 기준: {factor_text})"

    summary = _safe_text(working_selected.get("nlp_summary", ""))
    low_quality_summary_markers = {"요약 가능한 댓글 내용이 없습니다.", "한국어 번역을 준비 중입니다."}
    generic_summary_markers = (
        "해당 제품 관련",
        "중심으로",
        "강점·만족을 강조합니다",
        "불편·불만을 제기합니다",
        "비즈니스 관점",
    )
    if (
        not summary
        or _looks_garbled(summary)
        or summary in low_quality_summary_markers
        or any(marker in summary for marker in generic_summary_markers)
        or len(summary) < 12
        or (summary.endswith("...") and _safe_text(display_text).startswith(summary[:-3]))
    ):
        summary = _summarize_for_dashboard(working_selected, sentiment_value)
    summary_is_low_quality = (
        not summary
        or _looks_garbled(summary)
        or summary in low_quality_summary_markers
        or any(marker in summary for marker in generic_summary_markers)
        or len(summary) < 12
        or (summary.endswith("...") and _safe_text(display_text).startswith(summary[:-3]))
    )
    if (
        summary_is_low_quality
        and insight_summary
        and len(insight_summary) >= 12
        and not _looks_garbled(insight_summary)
        and not any(marker in insight_summary for marker in generic_summary_markers)
    ):
        summary = insight_summary
    if not summary:
        summary = "핵심 의견: 댓글 원문에서 제품 사용 경험에 대한 평가가 확인됩니다."

    # --- 댓글 맥락: 영상 배경 + context_tags ---
    _vt_raw = working_selected.get("title") or working_selected.get("video_title") or ""
    video_title = _safe_text(_vt_raw if not hasattr(_vt_raw, "tolist") else "")
    raw_context_tags = working_selected.get("nlp_context_tags", [])
    if isinstance(raw_context_tags, str):
        try:
            import ast as _ast
            raw_context_tags = _ast.literal_eval(raw_context_tags)
        except Exception:
            raw_context_tags = [t.strip() for t in raw_context_tags.strip("[]").split(",") if t.strip()]
    elif hasattr(raw_context_tags, "tolist"):
        raw_context_tags = raw_context_tags.tolist()
    if not isinstance(raw_context_tags, list):
        raw_context_tags = []
    context_tag_texts = [_safe_text(t) for t in raw_context_tags if _safe_text(t)][:3]
    context_tag_str = " · ".join(context_tag_texts) if context_tag_texts else aspect_label_display
    if video_title:
        comment_context_text = f"영상 '{video_title[:40]}{'…' if len(video_title) > 40 else ''}' 에서 작성된 댓글입니다. 연관 맥락: {context_tag_str}"
    else:
        comment_context_text = f"연관 맥락: {context_tag_str}"

    # --- 사용자 니즈 / 만족 포인트: nlp_summary 기반 ---
    user_needs_label = "만족 포인트" if sentiment_value == "positive" else "사용자 니즈"
    user_needs_text = summary  # nlp_summary (이미 정제됨)

    # --- 대응 방향 / 발전 방향: action_point 핵심만 ---
    action_payload = _build_resolution_point(
        working_selected=working_selected,
        sentiment_name=sentiment_value,
        keywords=core_points or keywords,
        inquiry_flag=inquiry_flag,
        negative_context=negative_context,
        similar_comments=similar_comments,
        source_comments=source_comments,
    )
    if isinstance(action_payload, dict):
        action_point_full = _safe_text(action_payload.get("action_text", ""))
    else:
        action_point_full = _safe_text(action_payload)
    # 증거 체인 제거: 핵심 권고문만 남김
    action_point = re.sub(r"\s*\([^)]*(?:유사 댓글|반복 신호|대표 근거|유사 근거|영상 맥락)[^)]*\)\s*$", "", action_point_full).strip()
    if not action_point:
        action_point = action_point_full
    response_direction_label = "발전 방향" if sentiment_value == "positive" else "대응 방향"

    # --- 보조 신호 (간소화) ---
    insight_layer = _build_insight_layer(
        working_selected=working_selected,
        sentiment_name=sentiment_value,
        aspect_label=aspect_label,
        keywords=core_points or keywords,
        source_comments=source_comments,
        similar_comments=similar_comments,
        insight_registry=insight_registry,
    )
    supporting_signal_items = insight_layer.get("supporting_signals", [])
    if not isinstance(supporting_signal_items, list):
        supporting_signal_items = []
    supporting_signal_items = [_safe_text(item) for item in supporting_signal_items if _safe_text(item)][:3]
    supporting_signals_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in supporting_signal_items
    ) or "<li>추가 근거 신호를 확보하지 못했습니다.</li>"

    fill_color = {"negative": "#ef4444", "positive": "#3b82f6", "neutral": "#8b5cf6", "trash": "#64748b"}.get(sentiment_value, "#3b82f6")
    keywords_html = "".join(f'<span class="rep-ai-kw">{html.escape(keyword)}</span>' for keyword in keywords)
    if not keywords_html:
        keywords_html = '<span class="rep-ai-kw">키워드 없음</span>'

    extra_negative_flags = ""
    if sentiment_value == "negative" and negative_context:
        if negative_context.get("share_text"):
            extra_negative_flags += f'<span class="rep-ai-flag">{html.escape(negative_context["share_text"])}</span>'
        if negative_context.get("recurrence_text"):
            extra_negative_flags += f'<span class="rep-ai-flag">{html.escape(negative_context["recurrence_text"])}</span>'
        if negative_context.get("trend_text"):
            extra_negative_flags += f'<span class="rep-ai-flag">{html.escape(negative_context["trend_text"])}</span>'

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
            {extra_negative_flags}
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
          <div class="rep-ai-sec">
            <div class="rep-ai-label">댓글 맥락</div>
            <div class="rep-ai-box">{html.escape(comment_context_text)}</div>
          </div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">{html.escape(user_needs_label)}</div>
            <div class="rep-ai-box">{html.escape(user_needs_text)}</div>
          </div>
          <div class="rep-ai-action"><strong>{html.escape(response_direction_label)}</strong><br>{html.escape(action_point)}</div>
          <div class="rep-ai-sec">
            <div class="rep-ai-label">보조 신호</div>
            <div class="rep-ai-box">
              <ul style="margin:0; padding-left:18px;">{supporting_signals_html}</ul>
            </div>
          </div>
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
    insight_registry: dict[str, set[str]] = {}
    for idx, selected in top_rows.iterrows():
        with st.container(border=True):
            # --- Data extraction ---
            raw_aspect_label = _safe_text(selected.get("topic_label", "")) or _safe_text(selected.get("aspect_key", "")) or f"이슈 {idx + 1}"
            likes_value = pd.to_numeric(selected.get(COL_LIKES, 0), errors="coerce")
            likes = 0 if pd.isna(likes_value) else int(likes_value)

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
            rep_id = _safe_text(selected.get("comment_id", selected.get("source_content_id", selected.get("opinion_id", ""))))
            pool = source_comments if source_comments is not None else pd.DataFrame()
            similar_comments, similar_meta = _build_canonical_similar_comments(pool, selected, working_selected, sentiment_name, rep_id)
            similar_count = int(similar_meta.get("cluster_size", len(similar_comments)))
            insufficient_evidence = bool(similar_meta.get("insufficient_evidence", similar_count < 3))
            if similar_count != len(similar_comments):
                raise RuntimeError(
                    f"similar-count mismatch detected for representative {rep_id}: "
                    f"meta={similar_count}, frame={len(similar_comments)}"
                )

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
            if similar_count >= 3:
                badge_html_parts.append(f'<span style="color:#94a3b8; font-size:12px;">👍 {likes:,} · 유사 {similar_count:,}건</span>')
            else:
                badge_html_parts.append(f'<span style="color:#94a3b8; font-size:12px;">👍 {likes:,} · 유사 근거 부족</span>')

            st.markdown(
                f'<div style="display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin-bottom:8px;">{"".join(badge_html_parts)}</div>',
                unsafe_allow_html=True,
            )

            selection_reason = _build_representative_selection_reason(
                working_selected,
                sentiment_name,
                source_comments,
                similar_count=similar_count,
            )
            st.markdown(f"**왜 대표 코멘트인가**: {selection_reason}")
            st.markdown("**원문 댓글**")
            st.write(display_text)
            if not is_korean and original_text and original_text != display_text:
                st.caption("원문(Original)")
                st.write(original_text)

            negative_context = _negative_issue_context(working_selected, source_comments) if sentiment_name == "negative" else {}
            _render_representative_nlp_panel(
                working_selected=working_selected,
                sentiment_name=sentiment_name,
                aspect_label=aspect_label,
                display_text=display_text,
                source_comments=source_comments,
                negative_context=negative_context,
                similar_comments=similar_comments,
                insight_registry=insight_registry,
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
            # 유사 댓글
            # ============================================================
            if similar_count >= 3 and not insufficient_evidence:
                with st.expander(f"유사 댓글 {similar_count:,}건"):
                    st.caption(f"현재 필터에서 매칭된 유사 댓글: {similar_count:,}건")
                    basis_labels = _build_similarity_basis_labels(working_selected, sentiment_name)
                    st.caption("유사 기준: " + " / ".join(basis_labels))
                    for _, sample in similar_comments.head(5).iterrows():
                        st.markdown(f"- {_safe_text(sample.get('원문', ''))}")
                    if similar_count > 5:
                        st.caption(f"외 {similar_count - 5:,}건 더...")
                    csv_bytes = similar_comments.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
                    if similar_count != len(similar_comments):
                        raise RuntimeError(
                            f"similar-count mismatch detected for representative {rep_id} before CSV export: "
                            f"count={similar_count}, csv_rows={len(similar_comments)}"
                        )
                    st.download_button(
                        label="CSV 다운로드",
                        data=csv_bytes,
                        file_name=f"similar_{key_prefix}_{idx+1}.csv",
                        mime="text/csv",
                        key=f"similar_download_{key_prefix}_{idx}",
                    )
            else:
                st.caption("유사 댓글은 최소 3건 이상의 동일 주장 방향·문제 인식이 확인될 때만 표시됩니다.")


def _extract_timestamp_seconds(text_value: str) -> int | None:
    text = _safe_text(text_value)
    if not text:
        return None
    hms = re.search(r"\b(\d{1,2}):([0-5]\d):([0-5]\d)\b", text)
    if hms:
        h, m, s = [int(hms.group(i)) for i in range(1, 4)]
        return h * 3600 + m * 60 + s
    ms = re.search(r"\b(\d{1,2}):([0-5]\d)\b", text)
    if ms:
        m, s = [int(ms.group(i)) for i in range(1, 3)]
        return m * 60 + s
    return None


def _format_timestamp_label(seconds: int | None) -> str:
    if seconds is None:
        return "시점 미확인"
    if pd.isna(seconds):
        return "시점 미확인"
    total = max(0, int(float(seconds)))
    hour = total // 3600
    minute = (total % 3600) // 60
    second = total % 60
    if hour > 0:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


def _append_timestamp_to_video_url(video_url: str, seconds: int | None) -> str:
    url = _safe_text(video_url)
    if not url or seconds is None:
        return url
    if "t=" in url:
        return url
    return f"{url}{'&' if '?' in url else '?'}t={seconds}s"


def _summarize_lg_video_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    if _truthy(row.get("lg_target_brand_hit")):
        reasons.append("LG/브랜드 직접 언급")
    if _truthy(row.get("lg_product_hit")):
        reasons.append("제품 키워드 일치")
    if _truthy(row.get("lg_preferred_channel_hit")):
        reasons.append("우선 채널 매칭")
    if reasons:
        return ", ".join(reasons)
    score = float(pd.to_numeric(row.get("video_lg_relevance_score", 0.0), errors="coerce") or 0.0)
    if score >= 0.45:
        return f"관련성 점수 기반({score:.2f})"
    return "관련 근거 부족"


def _summarize_lg_comment_reason(row: pd.Series) -> str:
    reasons: list[str] = []
    brand = canonicalize_brand(_safe_text(row.get(COL_BRAND, row.get("brand_label", row.get("brand_mentioned", "")))))
    if brand == "LG":
        reasons.append("LG 브랜드 언급")
    target = _safe_text(row.get("product_target", ""))
    if target:
        reasons.append("제품 타깃 매칭")
    cls = _safe_text(row.get("classification_type", "")).lower()
    if cls in {"complaint", "comparison", "preference"}:
        reasons.append("의견 유형 신호")
    relevance_score = float(pd.to_numeric(row.get("lg_relevance_score", 0.0), errors="coerce") or 0.0)
    if not reasons and relevance_score >= 0.45:
        reasons.append(f"관련성 점수 기반({relevance_score:.2f})")
    return ", ".join(reasons) if reasons else "관련 근거 부족"


def _compact_classification_reason(row: pd.Series) -> str:
    core_points = _clean_point_tokens(_parse_list_like(row.get("nlp_core_points", [])), limit=3)
    if core_points:
        return f"핵심포인트: {', '.join(core_points[:2])}"
    contexts = [item for item in _parse_list_like(row.get("nlp_context_tags", [])) if _safe_text(item)]
    if contexts:
        return f"맥락: {', '.join(contexts[:2])}"
    reason = _safe_text(row.get("classification_reason", ""))
    if not reason:
        return "-"
    clauses = _pick_salient_clauses(reason, limit=1)
    return clauses[0] if clauses else _summarize_original_for_preview(reason, limit=50)


def _row_summary_text(row: pd.Series, limit: int = 86) -> str:
    generic_markers = [
        "실제 불만이 명확하게 드러나",
        "만족감이나 추천 의사가 명확하여",
        "비즈니스 관점에서는",
        "부정으로 분류했습니다",
        "긍정으로 분류했습니다",
        "중립으로 분류했습니다",
        "정보 공유 또는 중립적 관찰",
        "쿠팡 파트너스",
        "일정액의 수수료",
        "가격은 구매일자별로",
        "본 포스팅",
        "파트너스 활동의 일환",
        "affiliate",
        "sponsored",
        "link in description",
    ]
    insight_summary = _safe_text(row.get("nlp_insight_summary", ""))
    if insight_summary and not _looks_garbled(insight_summary) and not any(marker in insight_summary for marker in generic_markers):
        return _summarize_original_for_preview(insight_summary, limit=limit)

    core_points = _clean_point_tokens(_parse_list_like(row.get("nlp_core_points", [])), limit=3)
    context_tags = _parse_list_like(row.get("nlp_context_tags", []))
    if core_points:
        context_hint = _safe_text(context_tags[0]) if context_tags else "핵심 맥락"
        phrase = f"{context_hint}: {', '.join(core_points[:2])}"
        return _summarize_original_for_preview(phrase, limit=limit)

    for col in ["nlp_summary", "classification_reason", "opinion_text", "cleaned_text", COL_ORIGINAL, "text_display"]:
        value = _safe_text(row.get(col, ""))
        if value and not _looks_garbled(value):
            if any(marker in value for marker in generic_markers):
                continue
            return _summarize_original_for_preview(value, limit=limit)
    return ""


def _compose_video_mention_point(video_comments: pd.DataFrame) -> tuple[str, str, int | None]:
    if video_comments.empty:
        return "표시 가능한 언급이 없습니다.", "근거 부족", None

    working = video_comments.copy()
    if "comment_validity" in working.columns:
        working = working[working["comment_validity"].astype(str).eq("valid")].copy()
    if working.empty:
        return "유효 댓글이 부족해 언급 포인트를 계산하지 못했습니다.", "근거 부족", None
    if "sentiment_label" not in working.columns:
        working["sentiment_label"] = "neutral"
    if "like_count" not in working.columns:
        working["like_count"] = 0

    ranked = working.sort_values(["like_count"], ascending=[False], na_position="last")
    neg = ranked[ranked["sentiment_label"].astype(str).eq("negative")]
    pos = ranked[ranked["sentiment_label"].astype(str).eq("positive")]
    top_any = ranked.iloc[0] if not ranked.empty else pd.Series(dtype=object)

    neg_text = _row_summary_text(neg.iloc[0]) if not neg.empty else ""
    pos_text = _row_summary_text(pos.iloc[0]) if not pos.empty else ""
    any_text = _row_summary_text(top_any) if not top_any.empty else ""

    point_counter: Counter[str] = Counter()
    context_counter: Counter[str] = Counter()
    if "nlp_core_points" in working.columns:
        for items in working["nlp_core_points"].tolist():
            point_counter.update(_clean_point_tokens(_parse_list_like(items), limit=5))
    if "nlp_context_tags" in working.columns:
        for items in working["nlp_context_tags"].tolist():
            context_counter.update([_safe_text(tag) for tag in _parse_list_like(items) if _safe_text(tag)])

    major_points = [token for token, _ in point_counter.most_common(3) if token]
    major_contexts = [token for token, _ in context_counter.most_common(2) if token]

    mention_parts: list[str] = []
    if neg_text:
        mention_parts.append(f"부정: {neg_text}")
    if pos_text:
        mention_parts.append(f"긍정: {pos_text}")
    if not mention_parts and any_text:
        mention_parts.append(f"혼합: {any_text}")
    if major_points:
        mention_parts.append(f"핵심포인트: {', '.join(major_points[:2])}")
    mention_summary = " | ".join(mention_parts) if mention_parts else "핵심 언급을 특정하기 어려워 원문 확인이 필요합니다."

    if major_contexts:
        labels_text = ", ".join(major_contexts[:2])
    elif major_points:
        labels_text = ", ".join([f"{token} 포인트" for token in major_points[:2]])
    else:
        text_series = working.get("cleaned_text", working.get("opinion_text", pd.Series("", index=working.index))).fillna("").astype(str)
        counter = build_keyword_counter(tuple(text_series.tolist()))
        low_signal_labels = {"번째", "이거", "이게", "그거", "그게", "진짜", "정말", "너무"}
        labels = [
            k
            for k, _ in filter_business_keywords(counter, top_n=8)
            if k and len(k) >= 2 and not re.search(r"\d", k) and k not in low_signal_labels
        ]
        labels_text = ", ".join(labels[:3]) if labels else "핵심 포인트 유사"

    ts_candidate = ""
    for row in [neg.iloc[0] if not neg.empty else pd.Series(dtype=object), pos.iloc[0] if not pos.empty else pd.Series(dtype=object), top_any]:
        if row.empty:
            continue
        ts_candidate = " ".join(
            [
                _safe_text(row.get(COL_ORIGINAL, "")),
                _safe_text(row.get("cleaned_text", "")),
                _safe_text(row.get("opinion_text", "")),
            ]
        )
        sec = _extract_timestamp_seconds(ts_candidate)
        if sec is not None:
            return mention_summary, labels_text, sec
    return mention_summary, labels_text, None


def build_video_summary(filtered_comments: pd.DataFrame, filtered_videos: pd.DataFrame) -> pd.DataFrame:
    comments = filtered_comments.copy() if filtered_comments is not None else pd.DataFrame()
    videos = filtered_videos.copy() if filtered_videos is not None else pd.DataFrame()
    if comments.empty and videos.empty:
        return pd.DataFrame()
    if "video_id" not in comments.columns and "video_id" not in videos.columns:
        return pd.DataFrame()

    if "video_id" in videos.columns and not videos.empty:
        videos = videos.sort_values(["video_id"]).drop_duplicates(subset=["video_id"], keep="first")
    elif videos.empty and "video_id" in comments.columns:
        rebuild_cols = [c for c in ["video_id", "product", "region", "channel_title", "title", "video_url", "published_at", "view_count"] if c in comments.columns]
        videos = comments[rebuild_cols].drop_duplicates(subset=["video_id"], keep="first").copy()

    if comments.empty:
        for col in ["분석_전체_댓글_수", "긍정_댓글_수", "부정_댓글_수", "중립_댓글_수", "제외_댓글_수"]:
            videos[col] = 0
        videos["부정_밀도"] = 0.0
        videos["긍정_밀도"] = 0.0
        videos["주요 언급 포인트"] = "댓글 없음"
        videos["유사 기준"] = "근거 부족"
        videos[COL_VIDEO_LINK] = videos.get("video_url", "")
        videos[COL_VIDEO_TITLE] = videos.get("title", "").map(_localized_video_title)
        return videos

    if "sentiment_label" not in comments.columns:
        comments["sentiment_label"] = "neutral"
    if "comment_validity" not in comments.columns:
        comments["comment_validity"] = "valid"

    comment_stats = comments.groupby("video_id", dropna=False).agg(
        분석_전체_댓글_수=("video_id", "size"),
        긍정_댓글_수=("sentiment_label", lambda s: int((s == "positive").sum())),
        부정_댓글_수=("sentiment_label", lambda s: int((s == "negative").sum())),
        중립_댓글_수=("sentiment_label", lambda s: int((s == "neutral").sum())),
        제외_댓글_수=("comment_validity", lambda s: int((s == "excluded").sum())),
    ).reset_index()

    mention_rows: list[dict[str, Any]] = []
    for vid, group in comments.groupby("video_id", dropna=False):
        mention, label_text, seconds = _compose_video_mention_point(group)
        like_col = COL_LIKES if COL_LIKES in group.columns else ("like_count" if "like_count" in group.columns else None)
        likes_raw = group.get(like_col, pd.Series(0, index=group.index)) if like_col else pd.Series(0, index=group.index)
        likes_series = pd.to_numeric(likes_raw, errors="coerce").fillna(0.0)
        sentiment_series = group.get("sentiment_label", pd.Series("neutral", index=group.index)).fillna("neutral").astype(str)
        issue_strength_raw = group.get("nlp_issue_strength_score", pd.Series(0.0, index=group.index))
        issue_strength_series = pd.to_numeric(issue_strength_raw, errors="coerce").fillna(0.0)

        neg_mask = sentiment_series.eq("negative")
        pos_mask = sentiment_series.eq("positive")
        neg_count = int(neg_mask.sum())
        pos_count = int(pos_mask.sum())
        neg_like_signal = float(likes_series[neg_mask].sum()) if neg_count else 0.0
        pos_like_signal = float(likes_series[pos_mask].sum()) if pos_count else 0.0
        neg_strength_signal = float(issue_strength_series[neg_mask].mean()) if neg_count else 0.0
        pos_strength_signal = float(issue_strength_series[pos_mask].mean()) if pos_count else 0.0
        neg_impact = neg_count + min(20.0, neg_like_signal / 20.0) + (neg_strength_signal * 2.0)
        pos_impact = pos_count + min(20.0, pos_like_signal / 20.0) + (pos_strength_signal * 2.0)
        mention_rows.append(
            {
                "video_id": vid,
                "주요 언급 포인트": mention,
                "유사 기준": label_text,
                "_timestamp_seconds": seconds,
                "_부정_임팩트": round(neg_impact, 3),
                "_긍정_임팩트": round(pos_impact, 3),
            }
        )
    mention_df = pd.DataFrame(mention_rows)

    summary = videos.merge(comment_stats, on="video_id", how="left").merge(mention_df, on="video_id", how="left")
    for col in ["분석_전체_댓글_수", "긍정_댓글_수", "부정_댓글_수", "중립_댓글_수", "제외_댓글_수"]:
        summary[col] = pd.to_numeric(summary.get(col, 0), errors="coerce").fillna(0).astype(int)

    valid_count = (summary["긍정_댓글_수"] + summary["부정_댓글_수"] + summary["중립_댓글_수"]).replace(0, 1)
    summary["부정_밀도"] = summary["부정_댓글_수"] / valid_count
    summary["긍정_밀도"] = summary["긍정_댓글_수"] / valid_count
    def _reaction_type_from_row(row: pd.Series) -> str:
        neg_impact = float(pd.to_numeric(row.get("_부정_임팩트", 0.0), errors="coerce") or 0.0)
        pos_impact = float(pd.to_numeric(row.get("_긍정_임팩트", 0.0), errors="coerce") or 0.0)
        neg_count = int(pd.to_numeric(row.get("부정_댓글_수", 0), errors="coerce") or 0)
        pos_count = int(pd.to_numeric(row.get("긍정_댓글_수", 0), errors="coerce") or 0)
        if neg_impact >= pos_impact + 2.0 and neg_count >= max(2, pos_count):
            return "부정 우세"
        if pos_impact >= neg_impact + 2.0 and pos_count >= max(2, neg_count):
            return "긍정 우세"
        if row["부정_밀도"] >= row["긍정_밀도"] + 0.12:
            return "부정 우세"
        if row["긍정_밀도"] >= row["부정_밀도"] + 0.12:
            return "긍정 우세"
        return "혼합"

    summary["반응 유형"] = summary.apply(_reaction_type_from_row, axis=1)
    if COL_COUNTRY in summary.columns:
        summary[COL_COUNTRY] = summary[COL_COUNTRY]
    elif "region" in summary.columns:
        summary[COL_COUNTRY] = summary["region"].map(localize_region)
    else:
        summary[COL_COUNTRY] = ""
    summary[COL_VIDEO_TITLE] = summary.get("title", summary.get(COL_VIDEO_TITLE, "")).map(_localized_video_title)
    summary[COL_VIDEO_LINK] = summary.get("video_url", summary.get(COL_VIDEO_LINK, ""))
    summary["_영상 바로가기"] = summary.apply(
        lambda row: _append_timestamp_to_video_url(_safe_text(row.get(COL_VIDEO_LINK, "")), row.get("_timestamp_seconds")),
        axis=1,
    )
    summary["LG 관련 여부"] = summary.get("is_lg_relevant_video", pd.Series(False, index=summary.index)).fillna(False).map(lambda v: "예" if bool(v) else "아니오")
    summary["LG 관련 근거"] = summary.apply(_summarize_lg_video_reason, axis=1)
    summary["LG 언급 시점"] = summary.get("_timestamp_seconds", pd.Series([None] * len(summary), index=summary.index)).map(_format_timestamp_label)
    published_source = summary["published_at"] if "published_at" in summary.columns else pd.Series(pd.NaT, index=summary.index)
    summary["발행일"] = pd.to_datetime(published_source, errors="coerce").dt.strftime("%Y-%m-%d")
    summary["주요 언급 포인트"] = summary.get("주요 언급 포인트", pd.Series("", index=summary.index)).fillna("").astype(str)
    summary["유사 기준"] = summary.get("유사 기준", pd.Series("근거 부족", index=summary.index)).fillna("근거 부족").astype(str)

    sort_specs: list[tuple[str, bool]] = [
        ("부정_밀도", False),
        ("분석_전체_댓글_수", False),
        ("view_count", False),
    ]
    sort_cols = [col for col, _ in sort_specs if col in summary.columns]
    ascending = [asc for col, asc in sort_specs if col in summary.columns]
    if sort_cols:
        return summary.sort_values(sort_cols, ascending=ascending, na_position="last")
    return summary


def render_video_summary_page(
    all_comments: pd.DataFrame,
    filtered_videos: pd.DataFrame,
    analysis_non_trash: pd.DataFrame | None = None,
) -> None:
    st.markdown("### 영상 분석 요약")
    st.caption("이 화면은 결론이 아니라 탐색 관문입니다. 한 행에서 ‘무슨 언급이 있었고(정성), 반응 밀도가 어떠한지(정량)’를 함께 보고 다음 확인 영상을 고를 수 있습니다.")

    all_video_comments = all_comments.copy()
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

    k1, k2, k3 = st.columns(3)
    with k1:
        render_card("탐색 대상 영상", f"{len(summary):,}", accent="blue")
    with k2:
        total_all = int(pd.to_numeric(summary.get("분석_전체_댓글_수", 0), errors="coerce").fillna(0).sum())
        render_card("전체 댓글(합)", f"{total_all:,}", accent="purple")
    with k3:
        hot_videos = int((summary["반응 유형"] == "부정 우세").sum()) if "반응 유형" in summary.columns else 0
        render_card("부정 우세 영상", f"{hot_videos:,}", accent="orange")

    with st.container(border=True):
        st.markdown("#### 영상 탐색 테이블")
        st.caption("한 행이 한 영상입니다. ‘주요 언급 포인트’로 맥락을 먼저 보고, 필요하면 바로 영상 링크/상세 분석으로 이동하세요.")
        table_df = summary.copy().head(120)
        table_df.insert(0, "선택", False)
        table_df["_vid"] = table_df.get("video_id", pd.Series("", index=table_df.index)).astype(str)
        table_df["바로가기"] = table_df["_영상 바로가기"]
        table_df["부정_밀도"] = pd.to_numeric(table_df.get("부정_밀도", 0), errors="coerce").fillna(0.0) * 100.0
        table_df["긍정_밀도"] = pd.to_numeric(table_df.get("긍정_밀도", 0), errors="coerce").fillna(0.0) * 100.0

        show_cols = [
            "선택",
            "_vid",
            "product",
            COL_COUNTRY,
            "channel_title",
            COL_VIDEO_TITLE,
            "발행일",
            "분석_전체_댓글_수",
            "부정_밀도",
            "긍정_밀도",
            "반응 유형",
            "LG 관련 여부",
            "LG 관련 근거",
            "LG 언급 시점",
            "주요 언급 포인트",
            "유사 기준",
            "바로가기",
        ]
        show_cols = [c for c in show_cols if c in table_df.columns]

        edited = st.data_editor(
            table_df[show_cols],
            width="stretch",
            hide_index=True,
            num_rows="fixed",
            column_config={
                "선택": st.column_config.CheckboxColumn("선택", help="체크하면 아래에서 해당 영상 상세 분석이 열립니다."),
                "_vid": None,
                COL_VIDEO_TITLE: st.column_config.TextColumn("영상 제목", width="large"),
                "LG 관련 근거": st.column_config.TextColumn("LG 관련 근거", width="medium"),
                "주요 언급 포인트": st.column_config.TextColumn("주요 언급 포인트", width="large"),
                "유사 기준": st.column_config.TextColumn("유사 기준", width="medium"),
                "부정_밀도": st.column_config.NumberColumn("부정 밀도", format="%.1f%%"),
                "긍정_밀도": st.column_config.NumberColumn("긍정 밀도", format="%.1f%%"),
                "바로가기": st.column_config.LinkColumn("영상 링크", display_text="열기"),
            },
            key="video_summary_editor",
        )

    checked = edited[edited["선택"] == True] if (not edited.empty and "선택" in edited.columns) else pd.DataFrame()
    if checked.empty:
        st.caption("👆 위 테이블에서 ‘선택’을 체크하면 아래에 해당 영상의 상세 분석이 표시됩니다.")
        return

    selected_video_id = _safe_text(checked.iloc[0].get("_vid", ""))
    if not selected_video_id:
        st.warning("선택된 영상 ID를 찾지 못했습니다. 다른 행을 선택해 주세요.")
        return
    st.session_state[SESSION_SELECTED_VIDEO_ID_KEY] = selected_video_id

    matched_rows = summary[summary["video_id"].astype(str).eq(selected_video_id)]
    if matched_rows.empty:
        st.warning("선택한 영상의 요약 데이터를 찾을 수 없습니다.")
        return
    selected_row = matched_rows.iloc[0]
    render_video_detail_page(
        analysis_comments,
        selected_row,
        analysis_non_trash=analysis_non_trash,
    )


def render_video_detail_page(
    filtered_comments: pd.DataFrame,
    selected_video: pd.Series,
    analysis_non_trash: pd.DataFrame | None = None,
) -> None:
    video_id = selected_video.get("video_id")
    video_comments = filtered_comments[filtered_comments["video_id"] == video_id].copy()

    st.markdown("---")
    st.markdown("### 선택 영상 상세 분석")
    st.markdown(f"#### {selected_video.get(COL_VIDEO_TITLE, '-')}")
    st.caption(
        f"제품군: {selected_video.get('product', '-')} · {COL_COUNTRY}: {selected_video.get(COL_COUNTRY, '-')} · "
        f"발행일: {selected_video.get('발행일', '-')} · 조회수: {int(selected_video.get('view_count', 0) or 0):,}"
    )
    jump_link = _safe_text(selected_video.get("_영상 바로가기", selected_video.get(COL_VIDEO_LINK, "#"))) or "#"
    st.markdown(f"영상 링크: [{jump_link}]({jump_link})")

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
        # Similar comments must always come from canonical analysis_non_trash source pool.
        canonical_pool = analysis_non_trash.copy() if analysis_non_trash is not None else pd.DataFrame()
        if not canonical_pool.empty and "video_id" in canonical_pool.columns:
            canonical_pool = canonical_pool[canonical_pool["video_id"] == video_id].copy()
        video_similar_pool = _canonicalize_similar_source_pool(canonical_pool)
        render_representative_comments(
            filtered_comments=video_comments,
            representative_bundles=None,
            opinion_units=None,
            key_suffix=f"_vid_{video_id}",
            similar_source_pool=video_similar_pool,
        )

    with st.container(border=True):
        title_col, action_col = st.columns([0.72, 0.28])
        with title_col:
            st.markdown("#### 전체 댓글 목록")
        display = video_comments.copy()
        display[COL_ORIGINAL] = display.get("text_display", display.get(COL_ORIGINAL, "")).fillna("")
        if COL_TRANSLATION not in display.columns:
            display[COL_TRANSLATION] = display[COL_ORIGINAL]
        lang = display.get("language_detected", pd.Series("", index=display.index)).fillna("").astype(str).str.lower()
        needs_translation = (~lang.str.startswith("ko")) & display[COL_ORIGINAL].str.strip().ne("")
        if needs_translation.any():
            display.loc[needs_translation, COL_TRANSLATION] = display.loc[needs_translation, COL_ORIGINAL].map(translate_to_korean)
        display["LG 관련 여부"] = display.get("lg_relevant_comment", pd.Series(False, index=display.index)).fillna(False).map(lambda v: "예" if bool(v) else "아니오")
        display["LG 관련 근거"] = display.apply(_summarize_lg_comment_reason, axis=1)
        display["문의 여부"] = display.get("nlp_is_inquiry", pd.Series(False, index=display.index)).fillna(False).map(lambda v: "예" if bool(v) else "아니오")
        display["분류 유형"] = (
            display.get("classification_type", pd.Series("", index=display.index))
            .fillna("")
            .astype(str)
            .replace(
                {
                    "complaint": "불만",
                    "praise": "칭찬",
                    "question": "문의",
                    "comparison": "비교",
                    "preference": "선호",
                    "informational": "정보공유",
                    "mixed": "혼합",
                }
            )
        )
        display["신뢰 수준"] = (
            display.get("confidence_level", pd.Series("", index=display.index))
            .fillna("")
            .astype(str)
            .replace({"high": "높음", "medium": "중간", "low": "낮음"})
        )
        display["제품 타깃"] = display.get("product_target", pd.Series("", index=display.index)).fillna("").map(localize_product_target)
        display["인사이트 유형"] = display.get("insight_type", pd.Series("", index=display.index)).fillna("")
        display["분류 이유"] = display.apply(_compact_classification_reason, axis=1)
        display[COL_SENTIMENT] = display.get(COL_SENTIMENT, display.get("sentiment_label", pd.Series("", index=display.index))).fillna("")
        cols = [
            COL_SENTIMENT,
            "LG 관련 여부",
            "LG 관련 근거",
            "문의 여부",
            "제품 타깃",
            "인사이트 유형",
            "분류 유형",
            "신뢰 수준",
            COL_ORIGINAL,
            COL_TRANSLATION,
            "분류 이유",
            COL_LIKES,
        ]
        existing = [c for c in cols if c in display.columns]
        export_df = display[existing].copy()
        with action_col:
            export_bytes = to_excel_bytes({"전체댓글": export_df})
            st.download_button(
                "전체 댓글 엑셀 다운로드",
                data=export_bytes,
                file_name=f"video_comments_{_safe_text(video_id) or 'selected'}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                width="stretch",
                key=f"download_video_comments_{_safe_text(video_id) or 'selected'}",
            )
        st.dataframe(export_df, width="stretch", hide_index=True)

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
    # 제품 범위는 실제 제품군만 노출한다.
    # 검색 키워드/일반 카테고리(예: 가전제품, 귀곰)가 제품 필터로 섞이지 않도록 차단한다.
    disallowed_product_tokens = {"가전제품", "가전제품, 귀곰", "귀곰"}
    allowed_products = [item for item in PRODUCT_ORDER if item not in disallowed_product_tokens]
    all_data_products = [item for item in all_data_products if item in allowed_products]
    available_products = [item for item in allowed_products if item in all_data_products]
    if not available_products:
        available_products = PRIMARY_PRODUCT_FILTERS[: min(4, len(PRIMARY_PRODUCT_FILTERS))]

    full_product_list = [item for item in allowed_products if item in available_products]

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
            st.session_state["filter_keyword_query"] = ""
            st.session_state["filter_keyword_mode"] = "AND"
            st.session_state["filter_keyword_mode_label"] = "AND(모두 포함)"
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
                index=options["analysis_scope"].index(_safe_text(st.session_state.get("filter_analysis_scope", "전체")) if _safe_text(st.session_state.get("filter_analysis_scope", "전체")) in options["analysis_scope"] else "전체"),
                label_visibility="collapsed",
                key="filter_analysis_scope",
            )

        with st.container(border=True):
            st.caption("LG 관련성 범위")
            relevance_scope = st.radio(
                "LG 관련성 범위",
                options["relevance_scope"],
                index=options["relevance_scope"].index(_safe_text(st.session_state.get("filter_relevance_scope", "전체 수집")) if _safe_text(st.session_state.get("filter_relevance_scope", "전체 수집")) in options["relevance_scope"] else "전체 수집"),
                label_visibility="collapsed",
                key="filter_relevance_scope",
            )

        with st.container(border=True):
            st.caption("고급 옵션")
            keyword_query = st.text_input(
                "댓글 내 키워드 검색",
                value=_safe_text(st.session_state.get("filter_keyword_query", "")),
                placeholder="예: 소음, 냉각, warranty",
                key="filter_keyword_query",
            )
            keyword_mode_label = st.radio(
                "키워드 결합 방식",
                ["AND(모두 포함)", "OR(하나 이상)"],
                index=0 if _safe_text(st.session_state.get("filter_keyword_mode", "AND")).upper() != "OR" else 1,
                horizontal=True,
                key="filter_keyword_mode_label",
                help="쉼표(,)로 여러 키워드를 입력하면 선택한 방식으로 적용됩니다.",
            )
            st.session_state["filter_keyword_mode"] = "OR" if keyword_mode_label.startswith("OR") else "AND"

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
        "keyword_terms": _parse_keyword_terms(keyword_query),
        "keyword_mode": _safe_text(st.session_state.get("filter_keyword_mode", "AND")) or "AND",
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
    active_snapshot_run = _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, RUN_SNAPSHOT_ALL)) or RUN_SNAPSHOT_ALL
    expected_signature = _expected_mode_signature(active_mode, snapshot_run_id=active_snapshot_run)
    if st.session_state.get(SESSION_DATA_SIGNATURE_KEY) != expected_signature:
        with st.spinner("최신 결과와 동기화 중…"):
            _set_data_mode(active_mode, force_refresh=True, snapshot_run_id=active_snapshot_run)

    with st.sidebar:
        st.markdown("### 데이터 모드")
        st.caption(f"현재 모드: {'샘플 모드' if active_mode == 'sample' else '실데이터 모드'}")
        sample_load_notice = _safe_text(st.session_state.pop("sample_load_notice", ""))
        if sample_load_notice:
            st.info(sample_load_notice)
        col_sample, col_real = st.columns(2)
        with col_sample:
            if st.button("샘플 로드", width="stretch", type="secondary"):
                _set_data_mode("sample", force_refresh=False)
                default_applied, notice = _apply_default_sample_state()
                if default_applied:
                    st.session_state["sample_load_notice"] = notice
                elif notice:
                    st.session_state["sample_load_notice"] = notice
                st.rerun()
        with col_real:
            if st.button("실데이터 새로고침", width="stretch", type="primary"):
                _set_data_mode("real", force_refresh=True, snapshot_run_id=active_snapshot_run)
                st.rerun()

        if active_mode == "real":
            snapshot_options = _build_run_snapshot_options()
            snapshot_value_to_label = {value: label for label, value in snapshot_options}
            snapshot_values = list(snapshot_value_to_label.keys())
            selected_snapshot = active_snapshot_run if active_snapshot_run in snapshot_values else RUN_SNAPSHOT_ALL
            selected_index = snapshot_values.index(selected_snapshot) if selected_snapshot in snapshot_values else 0
            selected_snapshot = st.selectbox(
                "실데이터 스냅샷",
                options=snapshot_values,
                index=selected_index,
                format_func=lambda value: snapshot_value_to_label.get(value, value),
                help="전체 run 통합 보기 또는 특정 분석 run만 고정해 불러올 수 있습니다.",
            )
            if selected_snapshot != active_snapshot_run:
                st.caption(f"선택 대기 중: {snapshot_value_to_label.get(selected_snapshot, selected_snapshot)}")
            if st.button("선택 스냅샷 불러오기", width="stretch", type="secondary"):
                _set_data_mode("real", force_refresh=False, snapshot_run_id=selected_snapshot)
                st.rerun()
            if active_snapshot_run != RUN_SNAPSHOT_ALL:
                st.caption(f"현재 고정 run: `{active_snapshot_run}`")
            st.caption("선택한 스냅샷은 로컬에 저장되어 다음 실행 시에도 자동 복원됩니다.")

        with st.expander("분석 상태 저장/불러오기", expanded=False):
            st.caption("현재 필터/모드/스냅샷 상태를 저장해 다음 실행에서도 같은 화면을 바로 재현합니다.")
            save_label = st.text_input("저장 이름", value="", placeholder="예: 4월_셋째주_세탁기_이슈점검", key="ui_state_save_label")
            if st.button("현재 상태 저장", width="stretch", type="secondary"):
                current_mode = _safe_text(st.session_state.get(SESSION_DATA_MODE_KEY, active_mode)) or "real"
                current_snapshot = _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, active_snapshot_run)) or RUN_SNAPSHOT_ALL
                payload = _collect_current_ui_state(current_mode, current_snapshot)
                ok, message = _write_saved_ui_state(save_label or "saved_state", payload)
                if ok:
                    st.success(f"저장 완료: {Path(message).name}")
                else:
                    st.error(message)

            saved_states = _list_saved_ui_states()
            if saved_states:
                state_label_by_file = {file_name: label for label, file_name in saved_states}
                selected_state = st.selectbox(
                    "저장된 상태",
                    options=[file_name for _, file_name in saved_states],
                    format_func=lambda file_name: state_label_by_file.get(file_name, file_name),
                    key="ui_state_load_target",
                )
                if st.button("선택 상태 불러오기", width="stretch", type="secondary"):
                    payload = _read_saved_ui_state(selected_state)
                    if not payload:
                        st.error("불러올 상태를 찾지 못했습니다.")
                    else:
                        can_restore, restore_message = _validate_restore_data_availability(payload, embedded_bundle=None)
                        if not can_restore:
                            st.warning(restore_message)
                        else:
                            _apply_saved_ui_state(payload, embedded_bundle=None)
                            st.success("저장 상태를 불러왔습니다.")
                            st.rerun()
            else:
                st.caption("저장된 상태가 없습니다.")

            st.markdown("---")
            st.caption("현재 상태를 로컬 JSON 파일로 내보내거나, 저장해 둔 상태 파일을 업로드해 복원할 수 있습니다.")
            export_mode = _safe_text(st.session_state.get(SESSION_DATA_MODE_KEY, active_mode)) or "real"
            export_snapshot = _safe_text(st.session_state.get(SESSION_RUN_SNAPSHOT_KEY, active_snapshot_run)) or RUN_SNAPSHOT_ALL
            export_payload = _collect_current_ui_state(export_mode, export_snapshot)
            export_bundle = st.session_state.get(SESSION_DATA_BUNDLE_KEY, _empty_dashboard_bundle())
            export_doc = _build_ui_state_document(
                save_label or "saved_state",
                export_payload,
                embedded_bundle=export_bundle,
            )
            export_json = json.dumps(export_doc, ensure_ascii=False, indent=2)
            export_suffix = pd.Timestamp.utcnow().strftime("%Y%m%d_%H%M%S")
            export_file_name = f"{_safe_snapshot_filename(save_label or 'saved_state')}_{export_suffix}.json"
            st.download_button(
                "파일로 다운로드",
                data=export_json,
                file_name=export_file_name,
                mime="application/json",
                width="stretch",
                help="현재 대시보드 상태를 JSON 파일로 저장합니다.",
            )

            uploaded_state_file = st.file_uploader(
                "상태 파일 업로드(JSON)",
                type=["json"],
                key="ui_state_upload_file",
                help="다운로드한 상태 JSON 파일을 업로드하면 같은 대시보드 상태를 복원합니다.",
            )
            if uploaded_state_file is not None:
                st.caption(f"업로드 파일: {uploaded_state_file.name}")
                if st.button("업로드 상태 적용", width="stretch", type="secondary"):
                    ok, payload, embedded_bundle, message = _read_uploaded_ui_state(uploaded_state_file)
                    if not ok:
                        st.error(message)
                    else:
                        can_restore, restore_message = _validate_restore_data_availability(
                            payload,
                            embedded_bundle=embedded_bundle,
                        )
                        if not can_restore:
                            st.warning(restore_message)
                        else:
                            _apply_saved_ui_state(payload, embedded_bundle=embedded_bundle)
                            st.success("업로드한 상태를 불러왔습니다.")
                            st.rerun()

        with st.expander("실제 분석 실행", expanded=False):
            run_keyword = st.text_input("검색 키워드", value="가전 리뷰")
            run_keyword_mode = st.selectbox(
                "키워드 결합 규칙",
                ["AND(교집합)", "OR(합집합)"],
                index=0,
                help="쉼표(,)로 여러 키워드를 입력했을 때 AND는 한 질의로 묶어 검색하고, OR은 키워드별로 확장 검색 후 합집합으로 수집합니다.",
            )
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
            run_keyword_mode_value = "AND" if run_keyword_mode.startswith("AND") else "OR"
            if run_keyword.strip():
                if run_region == ANALYSIS_ALL_MARKET_CODE:
                    st.caption(f"전체 시장 실행: {len(target_regions_preview)}개 국가를 순차 실행합니다.")
                    preview_lines: list[str] = []
                    for region_code in target_regions_preview[:5]:
                        variants = _build_search_keyword_variants(
                            " ".join(_parse_run_keywords(run_keyword)) if run_keyword_mode_value == "AND" else run_keyword,
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
                        " ".join(_parse_run_keywords(run_keyword)) if run_keyword_mode_value == "AND" else run_keyword,
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
                            keyword_match_mode=run_keyword_mode_value,
                        )
                    if ok:
                        st.session_state[SESSION_LAST_RUN_RESULT_KEY] = message
                        _set_data_mode("real", force_refresh=True, snapshot_run_id=_latest_real_run_id())
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
    analysis_non_trash_raw = data.get("analysis_non_trash", pd.DataFrame()).copy()
    if analysis_non_trash_raw.empty:
        analysis_comments_raw = data.get("analysis_comments", pd.DataFrame()).copy()
        if not analysis_comments_raw.empty:
            analysis_included_series = analysis_comments_raw.get("analysis_included", pd.Series(True, index=analysis_comments_raw.index)).fillna(False).astype(bool)
            hygiene_series = analysis_comments_raw.get("hygiene_class", pd.Series("strategic", index=analysis_comments_raw.index)).fillna("").astype(str).str.lower()
            analysis_non_trash_raw = analysis_comments_raw[analysis_included_series & hygiene_series.ne("trash")].copy()
    if not analysis_non_trash_raw.empty:
        analysis_non_trash_raw = add_localized_columns(analysis_non_trash_raw)
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
    st.caption("시작 가이드: 1) 필터 설정 → 2) 영상 확인 → 3) 전략 인사이트 확인")
    nlp_health, _, nlp_health_text = _nlp_enrichment_health(comments_df)
    if nlp_health == "fallback":
        st.warning(f"NLP 상태: Fallback 모드. {nlp_health_text}")
    elif nlp_health == "partial":
        st.info(f"NLP 상태: 부분 적용. {nlp_health_text}")
    else:
        st.caption(f"NLP 상태: 정상. {nlp_health_text}")

    selected_filters = _render_sidebar_filters(dashboard_options)

    bundle = compute_filtered_bundle(
        comments_df,
        videos_df,
        data.get("quality_summary", pd.DataFrame()),
        selected_filters,
        data.get("representative_bundles", pd.DataFrame()),
        data.get("opinion_units", pd.DataFrame()),
        analysis_non_trash_raw,
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
            "키워드 결합": selected_filters.get("keyword_mode", "AND"),
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


    from src import dashboard_strategy_page as strategy_page
    from src import dashboard_text_mining_page as tm_page
    from src import dashboard_v5_page as v5_page

    tab_comments, tab_videos, tab_strategy, tab_text_mining, tab_strategy_v5 = st.tabs(
        ["댓글 VoC 대시보드", "영상 분석 요약", "전략 인사이트", "전략 인사이트 (Text Mining 패턴)", "전략 인사이트 (V5 Topic-First 실험)"]
    )

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
                similar_source_pool=bundle.get("analysis_non_trash", pd.DataFrame()),
            )

        with st.container(border=True):
            st.markdown("### 키워드 컨텍스트 (보조)")
            st.caption("아래 도넛/워드클라우드는 대표 코멘트 해석을 돕는 참고 정보입니다. 빈도 자체를 우선순위로 해석하지는 않습니다.")
            render_keyword_panel(filtered_comments, "negative", "부정 키워드 비중")

        with st.container(border=True):
            render_keyword_panel(filtered_comments, "positive", "긍정 키워드 비중")

    with tab_videos:
        render_video_summary_page(
            all_comments,
            filtered_videos,
            analysis_non_trash=bundle.get("analysis_non_trash", pd.DataFrame()),
        )

    with tab_strategy:
        strategy_context = strategy_page.build_strategy_context(
            selected_filters=selected_filters,
            active_mode=active_mode,
            active_snapshot_run=active_snapshot_run,
            filtered_comments=filtered_comments,
            all_comments=all_comments,
            filtered_videos=filtered_videos,
            analysis_non_trash=bundle.get("analysis_non_trash", pd.DataFrame()),
            representative_comments=bundle.get("representative_comments", pd.DataFrame()),
            strategy_video_insights=data.get("strategy_video_insights", pd.DataFrame()),
            strategy_product_group_insights=data.get("strategy_product_group_insights", pd.DataFrame()),
            strategy_video_insight_candidates=data.get("strategy_video_insight_candidates", pd.DataFrame()),
            strategy_product_group_insight_candidates=data.get("strategy_product_group_insight_candidates", pd.DataFrame()),
            selected_video_id=_safe_text(st.session_state.get(SESSION_SELECTED_VIDEO_ID_KEY, "")),
        )
        strategy_page.render_strategy_page(strategy_context)

    with tab_text_mining:
        tm_context = tm_page.build_text_mining_context(
            selected_filters=selected_filters,
            active_mode=active_mode,
            active_snapshot_run=active_snapshot_run,
            filtered_comments=filtered_comments,
            filtered_videos=filtered_videos,
            analysis_non_trash=bundle.get("analysis_non_trash", pd.DataFrame()),
        )
        tm_page.render_text_mining_page(tm_context)

    with tab_strategy_v5:
        v5_context = v5_page.build_v5_context(
            selected_filters=selected_filters,
            active_mode=active_mode,
            active_snapshot_run=active_snapshot_run,
            filtered_comments=filtered_comments,
            filtered_videos=filtered_videos,
            analysis_non_trash=bundle.get("analysis_non_trash", pd.DataFrame()),
            selected_video_id=_safe_text(st.session_state.get(SESSION_SELECTED_VIDEO_ID_KEY, "")),
        )
        v5_page.render_v5_page(v5_context)


if __name__ == "__main__":
    main()

