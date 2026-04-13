from pathlib import Path

from src.collectors.video_search import VideoSearchCollector
from src.config import VideoSelectionSettings


class _DummyClient:
    pass


def _collector(policy: VideoSelectionSettings) -> VideoSearchCollector:
    return VideoSearchCollector(_DummyClient(), Path("C:/codex/_tmp/search_cache_tests"), selection_policy=policy)


def test_video_selection_excludes_competitor_brand_when_target_missing() -> None:
    policy = VideoSelectionSettings(
        enabled=True,
        require_product_term=True,
        require_target_brand=False,
        exclude_competitor_brands=True,
        target_brand_terms=["lg", "엘지"],
        competitor_brand_terms=["samsung", "삼성"],
        product_terms=["세탁기", "washer"],
    )
    collector = _collector(policy)
    videos = [
        {
            "video_id": "a",
            "title": "LG 세탁기 실사용 후기",
            "description": "",
            "channel_title": "리뷰채널",
            "tags": [],
            "search_rank": 2,
            "comment_count": 120,
            "view_count": 10000,
        },
        {
            "video_id": "b",
            "title": "Samsung washer review",
            "description": "",
            "channel_title": "테크리뷰",
            "tags": [],
            "search_rank": 1,
            "comment_count": 400,
            "view_count": 20000,
        },
    ]
    selected = collector._apply_video_selection_policy(videos)
    selected_ids = [item["video_id"] for item in selected]
    assert "a" in selected_ids
    assert "b" in selected_ids
    flags = {item["video_id"]: item["is_lg_relevant_video"] for item in selected}
    assert flags["a"] is True
    assert flags["b"] is False


def test_video_selection_keeps_preferred_channel_even_without_target_brand() -> None:
    policy = VideoSelectionSettings(
        enabled=True,
        require_product_term=True,
        require_target_brand=True,
        exclude_competitor_brands=True,
        target_brand_terms=["lg", "엘지"],
        competitor_brand_terms=["samsung", "삼성"],
        product_terms=["세탁기", "washer"],
        preferred_channel_keywords=["귀곰"],
    )
    collector = _collector(policy)
    videos = [
        {
            "video_id": "preferred",
            "title": "세탁기 사용 후기",
            "description": "한달 사용기",
            "channel_title": "귀곰 리뷰",
            "tags": [],
            "search_rank": 5,
            "comment_count": 80,
            "view_count": 5000,
        },
        {
            "video_id": "generic",
            "title": "세탁기 사용 후기",
            "description": "한달 사용기",
            "channel_title": "일반 채널",
            "tags": [],
            "search_rank": 3,
            "comment_count": 90,
            "view_count": 9000,
        },
    ]
    selected = collector._apply_video_selection_policy(videos)
    selected_ids = [item["video_id"] for item in selected]
    assert "preferred" in selected_ids
    assert "generic" in selected_ids
    flags = {item["video_id"]: item["is_lg_relevant_video"] for item in selected}
    assert flags["preferred"] is True
    assert flags["generic"] is False
