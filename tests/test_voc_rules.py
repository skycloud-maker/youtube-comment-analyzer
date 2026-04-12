import pandas as pd

from src.analytics.comment_analysis import analyze_comment_with_context
from src.analytics.opinion_units import build_representative_bundles
from src.pipelines.analytics_pipeline import AnalyticsPipeline


def test_generic_video_praise_is_not_promoted_to_product_positive():
    result = analyze_comment_with_context(
        "\uC601\uC0C1 \uB108\uBB34 \uC88B\uB124\uC694",
        validity="valid",
        parent_comment="",
        context_comments=["\uAC74\uC870\uAE30 \uCD94\uCC9C \uC601\uC0C1", "dryer"],
        is_reply=False,
    )

    assert result.product_related is False
    assert result.sentiment_label == "neutral"


def test_parent_agreement_can_still_inherit_product_context():
    result = analyze_comment_with_context(
        "\uC800\uB3C4 \uB3D9\uC758\uD574\uC694",
        validity="valid",
        parent_comment="\uC774 \uAC74\uC870\uAE30 \uC18C\uC74C\uC774 \uB108\uBB34 \uC2EC\uD574\uC694",
        context_comments=["\uAC74\uC870\uAE30 \uC0AC\uC6A9 \uD6C4\uAE30", "dryer"],
        is_reply=True,
    )

    assert result.product_related is True
    assert result.context_used is True


def test_lg_relevance_uses_video_context_and_product_relation():
    result = analyze_comment_with_context(
        "세탁기 소음이 너무 심해요",
        validity="valid",
        parent_comment="",
        context_comments=["세탁기 리뷰", "LG"],
        is_reply=False,
        video_context={
            "title": "LG 세탁기 후기",
            "channel": "가전리뷰",
            "video_lg_relevance_score": 0.82,
            "is_lg_relevant_video": True,
        },
    )
    assert result.product_related is True
    assert result.lg_relevance_score >= 0.5
    assert result.lg_relevant_comment is True


def test_representative_comments_deduplicate_same_issue_text():
    comments = pd.DataFrame([
        {
            "comment_id": "1",
            "text_display": "\uAC74\uC870\uAE30 \uC18C\uC74C\uC774 \uB108\uBB34 \uCEE4\uC694",
            "cleaned_text": "\uAC74\uC870\uAE30 \uC18C\uC74C\uC774 \uB108\uBB34 \uCEE4\uC694",
            "sentiment_label": "negative",
            "product_related": True,
            "classification_type": "complaint",
            "representative_decision": "strong_candidate",
            "representative_score": 28,
            "like_count": 10,
            "sentiment_score": -0.8,
            "cluster_id": "noise",
            "product": "dryer",
            "topic_label": "Noise",
        },
        {
            "comment_id": "2",
            "text_display": "\uAC74\uC870\uAE30 \uC18C\uC74C\uC774 \uB108\uBB34 \uCEE4\uC694!!!",
            "cleaned_text": "\uAC74\uC870\uAE30 \uC18C\uC74C\uC774 \uB108\uBB34 \uCEE4\uC694",
            "sentiment_label": "negative",
            "product_related": True,
            "classification_type": "complaint",
            "representative_decision": "strong_candidate",
            "representative_score": 27,
            "like_count": 8,
            "sentiment_score": -0.7,
            "cluster_id": "noise",
            "product": "dryer",
            "topic_label": "Noise",
        },
    ])

    representative = AnalyticsPipeline._representative_comments(comments)
    assert len(representative) == 1


def test_representative_bundles_skip_inquiry_only_rows():
    opinion_units = pd.DataFrame([
        {
            "sentiment_code": "negative",
            "product_related": True,
            "duplicate_opinion_flag": False,
            "inquiry_flag": True,
            "classification_type": "informational",
            "opinion_text": "\uC774 \uAC74\uC870\uAE30 \uC124\uCE58\uB294 \uC5B4\uB5BB\uAC8C \uD558\uB098\uC694?",
            "opinion_score": 10.0,
            "likes_count": 1,
            "product_category": "Dryer",
            "product_subtype": "HeatPump",
            "brand_mentioned": "LG",
            "cej_scene_code": "S6 On-board",
            "aspect_key": "Install",
            "evidence_spans_json": "[]",
            "sentiment": "Negative",
            "classification_reason": "\uBB38\uC758",
            "opinion_id": "x1",
            "source_content_id": "c1",
            "source_link": "",
            "pii_flag": "N",
            "pii_types": "",
            "display_text": "\uC774 \uAC74\uC870\uAE30 \uC124\uCE58\uB294 \uC5B4\uB5BB\uAC8C \uD558\uB098\uC694?",
            "original_text": "\uC774 \uAC74\uC870\uAE30 \uC124\uCE58\uB294 \uC5B4\uB5BB\uAC8C \uD558\uB098\uC694?",
            "masked_text": "",
            "translated_text": "",
            "summary_text": "",
            "pii_raw_text": "",
            "region": "KR",
            "product": "dryer",
        }
    ])

    bundles = build_representative_bundles(opinion_units)
    assert bundles.empty
