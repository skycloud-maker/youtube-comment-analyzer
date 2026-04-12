import shutil
from pathlib import Path

import pandas as pd

from src.config import load_yaml_settings_optional
from src.dashboard_app import (
    COL_SENTIMENT_CODE,
    COL_WEEK_LABEL,
    COL_WEEK_START,
    build_keyword_summary,
    build_weekly_sentiment_window,
    canonicalize_product_label,
)


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


def test_build_keyword_summary_excludes_blocked_keywords():
    texts = ("세척 성능 좋음 소음 있음", "세척 성능 만족", "소음 불만")
    result = build_keyword_summary(texts, top_n=10, version="test", blocked_keywords=("소음",))
    assert "소음" not in set(result["keyword"].tolist())


def test_canonicalize_product_label_removes_recommendation_suffix():
    assert canonicalize_product_label("건조기 추천") == "건조기"
    assert canonicalize_product_label("식세기 리뷰") == "식기세척기"
