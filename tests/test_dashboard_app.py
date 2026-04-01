import json
import shutil
from pathlib import Path

import pandas as pd

from src.config import load_yaml_settings_optional
from src.dashboard_app import (
    COL_SENTIMENT_CODE,
    COL_WEEK_LABEL,
    COL_WEEK_START,
    _collect_similar_comments_from_samples,
    _coverage_reason_label,
    _latest_run_with_manifest_for_coverage,
    build_weekly_sentiment_window,
)
from src.dashboard_rules import ui_labels


def test_load_yaml_settings_optional_returns_defaults_for_missing_file():
    temp_dir = Path(r"C:\codex\test_artifacts\dashboard_app")
    if temp_dir.exists():
        shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)

    settings = load_yaml_settings_optional(temp_dir / "missing.yaml")
    assert settings.dashboard.brand_filter_options == ["LG", "Samsung", "GE", "Whirlpool"]
    assert settings.dashboard.weekly_chart_page_default == 12


def test_build_weekly_sentiment_window_fills_missing_positive_negative_pairs():
    comments_df = pd.DataFrame(
        [
            {COL_WEEK_START: "2026-03-02", "sentiment_label": "positive"},
            {COL_WEEK_START: "2026-03-02", "sentiment_label": "positive"},
            {COL_WEEK_START: "2026-03-09", "sentiment_label": "negative"},
        ]
    )

    window, total_pages = build_weekly_sentiment_window(comments_df, weeks_per_view=8, page=1)
    count_column = window.columns[-1]

    assert total_pages == 1
    assert set(window[COL_SENTIMENT_CODE]) == {"positive", "negative"}
    assert set(window[COL_WEEK_LABEL]) == {"2026-03-02", "2026-03-09"}
    assert len(window) == 4
    assert window.loc[(window[COL_WEEK_LABEL] == "2026-03-09") & (window[COL_SENTIMENT_CODE] == "positive"), count_column].iloc[0] == 0


def test_dashboard_rules_use_representative_issue_labels():
    labels = ui_labels()
    assert labels["representative_section_title"] == "핵심 대표 코멘트"
    assert labels["representative_related_video"] == "영상제목"
    assert labels["representative_summary"] == "요약(1~2문장)"


def test_collect_similar_comments_from_samples_returns_clean_columns():
    selected = pd.Series(
        {
            "sample_comments_json": '[{"comment_id":"c1","display_text":"세탁기 소음이 큽니다","translated_text":"","likes_count":3,"pii_flag":"N","written_at":"2026-03-01"}]'
        }
    )
    frame = _collect_similar_comments_from_samples(selected)
    assert list(frame.columns) == ["comment_id", "원문", "번역(참고)", "좋아요 수", "PII", "작성일시"]
    assert frame.iloc[0]["원문"] == "세탁기 소음이 큽니다"


def test_coverage_reason_label_uses_korean_messages():
    assert _coverage_reason_label("COVERED") == "정상 반영"
    assert _coverage_reason_label("COMMENTS_NOT_COLLECTED") == "댓글 수집 미완료"
    assert _coverage_reason_label("TRIMMED_BY_MAX_VIDEOS") == "수집 상한으로 제외"


def test_latest_run_with_manifest_for_coverage_prefers_run_with_comments():
    root = Path(r"C:\codex\test_artifacts\coverage_picker")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    (root / "run_a").mkdir(parents=True, exist_ok=True)
    (root / "run_b").mkdir(parents=True, exist_ok=True)
    (root / "run_a" / "run_manifest.json").write_text(json.dumps({"run_id": "run_a"}), encoding="utf-8")
    (root / "run_b" / "run_manifest.json").write_text(json.dumps({"run_id": "run_b"}), encoding="utf-8")
    (root / "run_a" / "videos_raw.parquet").write_bytes(b"a")
    (root / "run_b" / "videos_raw.parquet").write_bytes(b"b")
    (root / "run_b" / "comments_raw.parquet").write_bytes(b"c")
    assert _latest_run_with_manifest_for_coverage(root) == "run_b"
