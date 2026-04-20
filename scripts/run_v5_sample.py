"""Minimal runnable sample for V5 topic-first pipeline."""

from __future__ import annotations

import logging

import pandas as pd

from src.v5 import run_v5_pipeline


def build_sample_bundle() -> dict[str, pd.DataFrame]:
    analysis_non_trash = pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "video_id": "v1",
                "text_display": "가격은 비싸지만 성능은 좋다",
                "sentiment_label": "mixed",
                "like_count": 11,
                "reply_count": 2,
                "product": "LG 틔운",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c2",
                "video_id": "v1",
                "text_display": "A/S가 느려서 다시는 안 산다",
                "sentiment_label": "negative",
                "like_count": 7,
                "reply_count": 1,
                "product": "오브제 냉장고",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c3",
                "video_id": "v1",
                "text_display": "설치 공간이 좁으면 문이 안 열린다",
                "sentiment_label": "negative",
                "like_count": 5,
                "reply_count": 0,
                "product": "비스포크 냉장고",
                "brand": "Samsung",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c4",
                "video_id": "v2",
                "text_display": "초기 설정은 어렵고 앱 연동도 느리다",
                "sentiment_label": "negative",
                "like_count": 9,
                "reply_count": 2,
                "product": "듀오보",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c5",
                "video_id": "v2",
                "text_display": "디자인은 예쁘고 성능도 안정적이다",
                "sentiment_label": "positive",
                "like_count": 13,
                "reply_count": 3,
                "product": "오브제",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c6",
                "video_id": "v2",
                "text_display": "가격만 조금 낮으면 바로 구매한다",
                "sentiment_label": "mixed",
                "like_count": 4,
                "reply_count": 1,
                "product": "틔운",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
            {
                "comment_id": "c7",
                "video_id": "v3",
                "text_display": "광고 링크 클릭하세요",
                "sentiment_label": "neutral",
                "like_count": 0,
                "reply_count": 0,
                "product": "",
                "brand": "",
                "region": "KR",
                "analysis_included": False,
                "hygiene_class": "trash",
            },
            {
                "comment_id": "c8",
                "video_id": "v3",
                "text_display": "세탁기는 좋은데 건조기는 소음이 크다",
                "sentiment_label": "mixed",
                "like_count": 8,
                "reply_count": 1,
                "product": "세탁기 건조기",
                "brand": "LG",
                "region": "KR",
                "analysis_included": True,
                "hygiene_class": "strategic",
            },
        ]
    )

    comment_truth = pd.DataFrame(
        [
            {"comment_id": "c1", "sentiment_final": "mixed", "mixed_flag": True, "journey_stage": "purchase", "judgment_axes": ["price_value_fit", "durability_lifespan"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c2", "sentiment_final": "negative", "mixed_flag": False, "journey_stage": "maintenance", "judgment_axes": ["service_after_sales"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c3", "sentiment_final": "negative", "mixed_flag": False, "journey_stage": "delivery", "judgment_axes": ["installation_space_fit"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c4", "sentiment_final": "negative", "mixed_flag": False, "journey_stage": "onboarding", "judgment_axes": ["automation_convenience"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c5", "sentiment_final": "positive", "mixed_flag": False, "journey_stage": "usage", "judgment_axes": ["design_interior_fit", "durability_lifespan"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c6", "sentiment_final": "mixed", "mixed_flag": True, "journey_stage": "purchase", "judgment_axes": ["price_value_fit"], "analysis_included": True, "hygiene_class": "strategic"},
            {"comment_id": "c8", "sentiment_final": "mixed", "mixed_flag": True, "journey_stage": "usage", "judgment_axes": ["durability_lifespan", "maintenance_burden"], "analysis_included": True, "hygiene_class": "strategic"},
        ]
    )

    video_metadata = pd.DataFrame(
        [
            {"video_id": "v1", "title": "틔운 구매 후기", "product": "틔운", "brand": "LG", "region": "KR"},
            {"video_id": "v2", "title": "듀오보 사용기", "product": "듀오보", "brand": "LG", "region": "KR"},
            {"video_id": "v3", "title": "세탁기 비교", "product": "세탁기", "brand": "LG", "region": "KR"},
        ]
    )
    return {
        "analysis_non_trash": analysis_non_trash,
        "comment_truth": comment_truth,
        "video_metadata": video_metadata,
        "run_id": "v5_sample_run",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    outputs = run_v5_pipeline(build_sample_bundle())
    print("\n=== v5_topic_ranked ===")
    print(outputs["v5_topic_ranked"].head(10).to_string(index=False))
    print("\n=== v5_topic_insight_actions ===")
    print(outputs["v5_topic_insight_actions"].head(10).to_string(index=False))
