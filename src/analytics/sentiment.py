"""Practical Korean/English sentiment scoring."""

from __future__ import annotations

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

POSITIVE_TERMS = {
    "좋다": 1.1,
    "만족": 1.3,
    "추천": 1.0,
    "최고": 1.4,
    "빠르": 0.8,
    "깔끔": 0.9,
    "예쁘": 0.8,
    "감사": 0.6,
    "편리": 1.1,
    "잘샀": 1.3,
    "튼튼": 1.1,
    "조용": 1.1,
    "성능좋": 1.2,
    "가성비": 1.0,
}
NEGATIVE_TERMS = {
    "별로": 1.0,
    "불량": 1.6,
    "최악": 1.7,
    "느리": 1.0,
    "불편": 1.3,
    "문제": 1.1,
    "소음": 1.2,
    "발열": 1.1,
    "실망": 1.5,
    "나쁘": 1.2,
    "고장": 1.6,
    "하자": 1.5,
    "비싸": 0.9,
    "짜증": 1.4,
    "누수": 1.5,
    "환불": 1.2,
    "반품": 1.2,
}
NEGATION_TERMS = {"안", "못", "않", "없", "never", "no", "not"}
INTENSIFIERS = {"너무", "진짜", "정말", "엄청", "완전", "very", "really", "super"}
ENGLISH_HINT = re.compile(r"[A-Za-z]")
KOREAN_HINT = re.compile(r"[가-힣]")



def _weighted_hits(text: str, lexicon: dict[str, float]) -> float:
    return sum(weight for token, weight in lexicon.items() if token in text)



def score_sentiment(text: str) -> tuple[str, float]:
    lowered = (text or "").lower().strip()
    if not lowered:
        return "neutral", 0.0

    english_score = analyzer.polarity_scores(lowered)["compound"] if ENGLISH_HINT.search(lowered) else 0.0
    positive = _weighted_hits(lowered, POSITIVE_TERMS)
    negative = _weighted_hits(lowered, NEGATIVE_TERMS)

    if any(term in lowered for term in INTENSIFIERS):
        positive *= 1.15 if positive else 1.0
        negative *= 1.15 if negative else 1.0
        english_score *= 1.1

    for token, weight in POSITIVE_TERMS.items():
        if token in lowered and any(neg + token in lowered or f"{neg} {token}" in lowered for neg in NEGATION_TERMS):
            positive -= weight
            negative += weight * 0.9
    for token, weight in NEGATIVE_TERMS.items():
        if token in lowered and any(neg + token in lowered or f"{neg} {token}" in lowered for neg in NEGATION_TERMS):
            negative -= weight
            positive += weight * 0.6

    if KOREAN_HINT.search(lowered):
        lexicon_total = positive + negative
        lexicon_score = (positive - negative) / lexicon_total if lexicon_total else 0.0
        final_score = round((lexicon_score * 0.75) + (english_score * 0.25), 3) if ENGLISH_HINT.search(lowered) else round(lexicon_score, 3)
    else:
        final_score = round(english_score, 3)

    if final_score >= 0.35:
        return "positive", final_score
    if final_score <= -0.35:
        return "negative", final_score
    return "neutral", final_score
