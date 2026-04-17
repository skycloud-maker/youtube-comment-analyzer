import pandas as pd

from src.analytics.comment_truth import CANONICAL_COMMENT_TRUTH_COLUMNS, build_canonical_comment_truth
from src.analytics.comment_analysis import CommentAnalysisResult
from src.pipelines.analytics_pipeline import AnalyticsPipeline


def test_build_canonical_comment_truth_contract_and_mixed():
    comments = pd.DataFrame(
        [
            {
                "comment_id": "c1",
                "video_id": "v1",
                "text_display": "냄새가 심한데 왜 이럴까요?",
                "cleaned_text": "냄새가 심한데 왜 이럴까요",
                "comment_validity": "valid",
                "comment_quality": "valid",
                "sentiment_label": "negative",
                "nlp_label": "positive",
                "nlp_topic_sentiments": {"냄새": "negative", "디자인": "positive"},
                "nlp_is_inquiry": True,
                "classification_type": "complaint",
            },
            {
                "comment_id": "c2",
                "video_id": "v1",
                "text_display": "쿠팡 링크 클릭하세요",
                "cleaned_text": "쿠팡 링크 클릭하세요",
                "comment_validity": "excluded",
                "comment_quality": "ad",
                "sentiment_label": "neutral",
                "nlp_label": "trash",
                "nlp_is_inquiry": False,
                "classification_type": "informational",
            },
        ]
    )

    truth = build_canonical_comment_truth(comments)
    assert set(CANONICAL_COMMENT_TRUTH_COLUMNS).issubset(truth.columns)

    c1 = truth.loc[truth["comment_id"] == "c1"].iloc[0]
    assert c1["sentiment_final"] == "mixed"
    assert bool(c1["mixed_flag"]) is True
    assert c1["hygiene_class"] == "supporting"
    assert bool(c1["analysis_included"]) is True

    c2 = truth.loc[truth["comment_id"] == "c2"].iloc[0]
    assert c2["hygiene_class"] == "trash"
    assert bool(c2["analysis_included"]) is False


def test_analytics_pipeline_enforces_hygiene_for_downstream(monkeypatch, tmp_path):
    def _fake_analysis(*args, **kwargs):
        text = kwargs.get("text") or (args[0] if args else "")
        if "광고" in str(text):
            return CommentAnalysisResult(
                product_related=False,
                product_target=None,
                sentiment_label="neutral",
                sentiment_score=0.0,
                signal_strength="weak",
                classification_type="informational",
                confidence_level="low",
                context_used=False,
                context_required_for_display=False,
                reason="noise",
                needs_review=False,
                nlp_label="trash",
                nlp_is_inquiry=False,
            )
        return CommentAnalysisResult(
            product_related=True,
            product_target="washer",
            sentiment_label="negative",
            sentiment_score=-0.7,
            signal_strength="strong",
            classification_type="complaint",
            confidence_level="high",
            context_used=False,
            context_required_for_display=False,
            reason="valid complaint",
            needs_review=False,
            nlp_label="negative",
            nlp_is_inquiry=False,
        )

    monkeypatch.setattr("src.pipelines.analytics_pipeline.analyze_comment_with_context", _fake_analysis)

    videos_df = pd.DataFrame(
        [
            {
                "video_id": "v1",
                "title": "세탁기 리뷰",
                "channel_title": "채널",
                "product": "세탁기",
                "region": "KR",
                "video_url": "https://youtube.com/watch?v=v1",
                "published_at": "2026-04-10T00:00:00Z",
                "view_count": 10,
                "comment_count": 2,
            }
        ]
    )
    comments_df = pd.DataFrame(
        [
            {
                "comment_id": "c-valid",
                "video_id": "v1",
                "text_original": "세탁기 소음이 너무 큽니다",
                "text_display": "세탁기 소음이 너무 큽니다",
                "cleaned_text": "세탁기 소음이 너무 큽니다",
                "text_length": 14,
                "comment_validity": "valid",
                "comment_quality": "valid",
                "analysis_included": True,
                "is_reply": False,
                "like_count": 3,
                "published_at": "2026-04-10T01:00:00Z",
            },
            {
                "comment_id": "c-trash",
                "video_id": "v1",
                "text_original": "광고 링크 클릭",
                "text_display": "광고 링크 클릭",
                "cleaned_text": "광고 링크 클릭",
                "text_length": 8,
                "comment_validity": "valid",
                "comment_quality": "valid",
                "analysis_included": True,
                "is_reply": False,
                "like_count": 0,
                "published_at": "2026-04-10T02:00:00Z",
            },
        ]
    )

    pipeline = AnalyticsPipeline(
        processed_dir=tmp_path,
        risk_keywords=[],
        top_n_keywords=10,
        topic_count=3,
        nlp_analyzer_config={"enabled": False},
    )
    outputs = pipeline.run("run-1", videos_df, comments_df)

    truth = outputs["comment_truth"]
    assert {"hygiene_class", "analysis_included", "sentiment_final", "mixed_flag"}.issubset(truth.columns)
    assert truth.loc[truth["comment_id"] == "c-trash", "hygiene_class"].iloc[0] == "trash"
    assert bool(truth.loc[truth["comment_id"] == "c-trash", "analysis_included"].iloc[0]) is False

    analysis_comments = outputs["analysis_comments"]
    assert "c-trash" not in set(analysis_comments["comment_id"].tolist())
    assert set(analysis_comments["sentiment_label"].dropna().astype(str).str.lower().tolist()) <= {"positive", "negative", "neutral", "mixed", "trash"}
