from pathlib import Path

import pandas as pd

from src.collectors.video_search import VideoSearchCollector
from src.config import VideoSelectionSettings
from src.dashboard_app import (
    REQUIRED_RESULT_FILES,
    ANALYSIS_ALL_MARKET_CODE,
    _analysis_target_regions,
    _build_missing_data_message,
    _build_search_keyword_variants,
    _build_synthetic_sample_bundle,
    build_comment_showcase,
    build_video_summary,
    load_sample_dashboard_data,
)


class _DummyClient:
    pass


def _collector(policy: VideoSelectionSettings) -> VideoSearchCollector:
    return VideoSearchCollector(_DummyClient(), Path("C:/codex/_tmp/search_cache_arch_tests"), selection_policy=policy)


def test_video_selection_policy_does_not_drop_unmatched_videos() -> None:
    policy = VideoSelectionSettings(
        enabled=True,
        require_product_term=True,
        require_target_brand=False,
        exclude_competitor_brands=True,
        target_brand_terms=["lg"],
        competitor_brand_terms=["samsung"],
        product_terms=["washer"],
    )
    collector = _collector(policy)
    videos = [
        {
            "video_id": "lg_1",
            "title": "LG washer review",
            "description": "",
            "channel_title": "Appliance Lab",
            "tags": [],
            "search_rank": 1,
            "comment_count": 120,
            "view_count": 10000,
        },
        {
            "video_id": "cmp_1",
            "title": "Samsung washer review",
            "description": "",
            "channel_title": "Appliance Lab",
            "tags": [],
            "search_rank": 2,
            "comment_count": 130,
            "view_count": 12000,
        },
    ]

    annotated = collector._apply_video_selection_policy(videos)

    assert {row["video_id"] for row in annotated} == {"lg_1", "cmp_1"}
    relevant_map = {row["video_id"]: row["is_lg_relevant_video"] for row in annotated}
    assert relevant_map["lg_1"] is True
    assert relevant_map["cmp_1"] is False


def test_synthetic_sample_bundle_is_always_available() -> None:
    bundle = _build_synthetic_sample_bundle()
    assert not bundle["comments"].empty
    assert not bundle["videos"].empty


def test_sample_loader_falls_back_to_synthetic_when_no_runs(monkeypatch) -> None:
    monkeypatch.setattr("src.dashboard_app._real_ready_run_dirs", lambda: [])
    bundle = load_sample_dashboard_data.__wrapped__()
    assert not bundle["comments"].empty
    assert not bundle["videos"].empty


def test_build_comment_showcase_uses_pipeline_columns_without_reanalysis() -> None:
    frame = pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "text_display": "Noise is too loud.",
                "cleaned_text": "Noise is too loud.",
                "sentiment_label": "negative",
                "signal_strength": "strong",
                "classification_type": "complaint",
                "confidence_level": "high",
                "classification_reason": "explicit complaint",
                "product_target": "washer",
                "context_used": False,
                "context_required_for_display": False,
                "needs_review": False,
                "comment_validity": "valid",
                "like_count": 12,
                "text_length": 19,
            }
        ]
    )
    showcase = build_comment_showcase(frame, sentiment="negative", limit=5)
    assert len(showcase) == 1
    row = showcase.iloc[0]
    assert row["display_sentiment"] == "negative"
    assert row["display_reason"] == "explicit complaint"
    assert row["display_classification_type"] == "complaint"


def test_missing_data_message_mentions_required_files(monkeypatch) -> None:
    monkeypatch.setattr("src.dashboard_app._iter_dashboard_run_dirs", lambda: [])
    message = _build_missing_data_message()
    for filename in REQUIRED_RESULT_FILES:
        assert filename in message


def test_build_video_summary_handles_missing_published_at_column() -> None:
    comments = pd.DataFrame(
        [
            {"video_id": "v1", "sentiment_label": "negative", "comment_validity": "valid", "like_count": 3},
            {"video_id": "v1", "sentiment_label": "neutral", "comment_validity": "valid", "like_count": 1},
        ]
    )
    videos = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "title": "sample",
                "video_url": "https://example.com/v1",
                "region": "KR",
                # intentionally no published_at
            }
        ]
    )

    summary = build_video_summary(comments, videos)
    assert len(summary) == 1
    assert "video_id" in summary.columns


def test_search_keyword_variants_expand_using_region_language(monkeypatch) -> None:
    def _fake_translate(text: str, target_language: str, source_language: str = "auto") -> str:
        mapping = {
            ("가전제품", "en"): "home appliances",
            ("가전제품", "fr"): "appareils electromenagers",
        }
        return mapping.get((text, target_language), text)

    monkeypatch.setattr("src.dashboard_app.translate_text", _fake_translate)
    variants = _build_search_keyword_variants("가전제품", language="all", region="FR")
    assert variants[0] == "가전제품"
    assert "appareils electromenagers" in variants
    assert "home appliances" in variants


def test_analysis_target_regions_all_expands_to_market_list() -> None:
    regions = _analysis_target_regions(ANALYSIS_ALL_MARKET_CODE)
    assert len(regions) == 20
    assert "KR" in regions
    assert "US" in regions
    assert ANALYSIS_ALL_MARKET_CODE not in regions
