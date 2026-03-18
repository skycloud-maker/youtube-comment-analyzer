import pandas as pd

from src.analytics.trends import build_weekly_keyword_trend, build_weekly_sentiment_trend



def test_weekly_sentiment_trend_builds_ratio():
    comments_df = pd.DataFrame(
        [
            {"published_at": "2026-03-02T12:00:00Z", "product": "건조기", "region": "KR", "sentiment_label": "positive", "cleaned_text": "만족 추천"},
            {"published_at": "2026-03-03T12:00:00Z", "product": "건조기", "region": "KR", "sentiment_label": "negative", "cleaned_text": "소음 문제"},
        ]
    )
    trend = build_weekly_sentiment_trend(comments_df)
    assert set(trend["sentiment_label"]) == {"positive", "negative"}
    assert trend["ratio"].sum() == 1.0



def test_weekly_keyword_trend_extracts_keywords_by_sentiment():
    comments_df = pd.DataFrame(
        [
            {"published_at": "2026-03-02T12:00:00Z", "product": "건조기", "region": "KR", "sentiment_label": "negative", "cleaned_text": "소음 문제"},
            {"published_at": "2026-03-03T12:00:00Z", "product": "건조기", "region": "KR", "sentiment_label": "negative", "cleaned_text": "소음 심함"},
        ]
    )
    trend = build_weekly_keyword_trend(comments_df, top_n_per_group=5)
    assert "소음" in trend["keyword"].tolist()
