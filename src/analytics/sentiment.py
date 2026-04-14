"""Practical Korean/English sentiment scoring.

Supports two modes:
- Legacy: VADER + custom lexicon (score_sentiment)
- NLP Analyzer: LLM-based analysis via nlp_analyzer (analyze_with_nlp)
"""

from __future__ import annotations

import logging
import re
from typing import Any

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from nlp_analyzer import analyze_comment as nlp_analyze_comment, analyze_batch as nlp_analyze_batch, AnalysisResult, is_valid
    from nlp_analyzer.router import LLMRouter
    from nlp_analyzer.providers.claude import ClaudeProvider
    from nlp_analyzer.providers.openai import OpenAIProvider
    NLP_ANALYZER_AVAILABLE = True
except ImportError:
    NLP_ANALYZER_AVAILABLE = False

logger = logging.getLogger(__name__)
analyzer = SentimentIntensityAnalyzer()

POSITIVE_TERMS = {
    "\uc88b\ub2e4": 1.1,
    "\ub9cc\uc871": 1.3,
    "\ucd94\ucc9c": 1.0,
    "\ucd5c\uace0": 1.4,
    "\ube60\ub974": 0.8,
    "\uae54\ub054": 0.9,
    "\uc608\uc058": 0.8,
    "\uac10\uc0ac": 0.6,
    "\ud3b8\ub9ac": 1.1,
    "\uc798\uc0c0": 1.3,
    "\ud2bc\ud2bc": 1.1,
    "\uc870\uc6a9": 1.1,
    "\uc131\ub2a5\uc88b": 1.2,
    "\uac00\uc131\ube44": 1.0,
    "\ubbff\uc74c": 0.8,
    "worth it": 1.0,
    "love": 1.1,
    "great": 0.9,
}
NEGATIVE_TERMS = {
    "\ubcc4\ub85c": 1.0,
    "\ubd88\ub7c9": 1.6,
    "\ucd5c\uc545": 1.7,
    "\ub290\ub9ac": 1.0,
    "\ubd88\ud3b8": 1.3,
    "\ubb38\uc81c": 1.1,
    "\uc18c\uc74c": 1.2,
    "\ubc1c\uc5f4": 1.1,
    "\uc2e4\ub9dd": 1.5,
    "\ub098\uc058": 1.2,
    "\uace0\uc7a5": 1.6,
    "\ud558\uc790": 1.5,
    "\ube44\uc2f8": 0.9,
    "\uc9dc\uc99d": 1.4,
    "\ub204\uc218": 1.5,
    "\ud658\ubd88": 1.2,
    "\ubc18\ud488": 1.2,
    "\ubd88\ud544\uc694": 1.1,
    "\uc4f8\ub370\uc5c6": 1.3,
    "\ubcf5\uc7a1": 1.0,
    "\uc6d0\ud558\uc9c0": 1.1,
    "hate": 1.5,
    "awful": 1.6,
    "waste": 1.2,
    "worthless": 1.3,
}
NEGATION_TERMS = {"\uc548", "\ubabb", "\uc54a", "\uc5c6", "never", "no", "not"}
INTENSIFIERS = {"\ub108\ubb34", "\uc9c4\uc9dc", "\uc815\ub9d0", "\uc5c4\uccad", "\uc644\uc804", "very", "really", "super"}
ENGLISH_HINT = re.compile(r"[A-Za-z]")
KOREAN_HINT = re.compile(r"[\uac00-\ud7a3]")
QUESTION_COMPLAINT_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"why\s+cant",
        r"why\s+can't",
        r"\uc65c.+\ubabb",
        r"\uc65c.+\uc5c6",
        r"can't they just",
        r"\ud544\uc694\s*\uc5c6",
        r"\uc6d0\ud558\uc9c0\s*\uc54a",
        r"\ube7c\uace0",
    ]
]
POSITIVE_OVERRIDE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\uac15\ucd94",
        r"\uc801\uadf9 \ucd94\ucc9c",
        r"very good",
        r"works great",
        r"love it",
    ]
]


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

    if any(pattern.search(lowered) for pattern in QUESTION_COMPLAINT_PATTERNS):
        negative += 1.1
        if ENGLISH_HINT.search(lowered):
            english_score = min(english_score, -0.25)

    if any(pattern.search(lowered) for pattern in POSITIVE_OVERRIDE_PATTERNS):
        positive += 1.0

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
        final_score = round((lexicon_score * 0.8) + (english_score * 0.2), 3) if ENGLISH_HINT.search(lowered) else round(lexicon_score, 3)
    else:
        final_score = round(((positive - negative) / (positive + negative)) if (positive + negative) else english_score, 3)

    if final_score >= 0.32:
        return "positive", final_score
    if final_score <= -0.22:
        return "negative", final_score
    return "neutral", final_score


# ---------------------------------------------------------------------------
# nlp_analyzer integration
# ---------------------------------------------------------------------------

def _build_nlp_router(
    provider: str = "claude",
    model: str = "claude-sonnet-4-20250514",
    fallback_provider: str = "openai",
    fallback_model: str = "gpt-4o-mini",
) -> Any:
    """Build an LLMRouter with configured primary/secondary providers."""
    if not NLP_ANALYZER_AVAILABLE:
        return None
    primary = ClaudeProvider(model=model) if provider == "claude" else OpenAIProvider(model=model)
    secondary = OpenAIProvider(model=fallback_model) if fallback_provider == "openai" else ClaudeProvider(model=fallback_model)
    return LLMRouter(primary=primary, secondary=secondary)


def analyze_with_nlp(
    comment_id: str,
    text: str,
    provider: Any | None = None,
) -> dict[str, Any] | None:
    """Analyze a single comment using nlp_analyzer. Returns dict of fields or None on failure."""
    if not NLP_ANALYZER_AVAILABLE:
        logger.warning("nlp_analyzer not installed, falling back to legacy sentiment")
        return None
    try:
        result: AnalysisResult = nlp_analyze_comment(comment_id, text, provider=provider)
        if result.error:
            logger.warning("nlp_analyzer error for %s: %s", comment_id, result.error)
            return None
        return _nlp_result_to_dict(result)
    except Exception as exc:
        logger.warning("nlp_analyzer exception for %s: %s", comment_id, exc)
        return None


def analyze_batch_with_nlp(
    comments: list[dict[str, str]],
    provider: Any | None = None,
) -> list[dict[str, Any] | None]:
    """Analyze multiple comments using nlp_analyzer batch mode."""
    if not NLP_ANALYZER_AVAILABLE:
        return [None] * len(comments)
    try:
        results = nlp_analyze_batch(comments, provider=provider)
        return [_nlp_result_to_dict(r) if not r.error else None for r in results]
    except Exception as exc:
        logger.warning("nlp_analyzer batch exception: %s", exc)
        return [None] * len(comments)


def _nlp_result_to_dict(result: Any) -> dict[str, Any]:
    """Convert AnalysisResult to a flat dict for pipeline consumption."""
    confidence_to_score = {
        "positive": result.confidence,
        "negative": -result.confidence,
        "neutral": 0.0,
        "trash": 0.0,
        "undecidable": 0.0,
    }
    return {
        "nlp_label": result.label,
        "nlp_confidence": result.confidence,
        "nlp_sentiment_score": confidence_to_score.get(result.label, 0.0),
        "nlp_sentiment_reason": result.sentiment_reason,
        "nlp_topics": result.topics,
        "nlp_topic_sentiments": result.topic_sentiments,
        "nlp_is_inquiry": result.is_inquiry,
        "nlp_is_rhetorical": result.is_rhetorical,
        "nlp_summary": result.summary,
        "nlp_keywords": result.keywords,
        "nlp_product_mentions": result.product_mentions,
        "nlp_language": result.language,
        "nlp_provider": result.llm_provider,
        "nlp_model": result.model_name,
        "nlp_core_points": getattr(result, "core_points", []),
        "nlp_context_tags": getattr(result, "context_tags", []),
        "nlp_similarity_keys": getattr(result, "similarity_keys", []),
        "nlp_insight_summary": getattr(result, "insight_summary", None),
        "nlp_confidence_factors": getattr(result, "confidence_factors", []),
        "nlp_confidence_breakdown": getattr(result, "confidence_breakdown", {}),
        "nlp_sentiment_intensity": getattr(result, "sentiment_intensity", 0.0),
    }
