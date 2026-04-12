"""Context-aware product feedback analysis helpers.

Supports two analysis modes:
- Legacy: rule-based with VADER sentiment (default fallback)
- NLP Analyzer: LLM-based deep analysis via nlp_analyzer
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import re
from typing import Any

from src.analytics.sentiment import score_sentiment, analyze_with_nlp, NLP_ANALYZER_AVAILABLE

logger = logging.getLogger(__name__)

AGREEMENT_MARKERS = [
    "i agree", "agree with", "exactly", "same", "true that", "right", "100%",
    "\ub3d9\uc758", "\uc800\ub3c4 \uadf8\ub807\uac8c \uc0dd\uac01", "\ub9de\uc544\uc694", "\ub9de\ub294 \ub9d0",
]
SHORT_DEPENDENT_MARKERS = [
    "i agree", "same", "exactly", "\ub3d9\uc758", "\ub9de\uc544\uc694", "\uadf8\ub807\uac8c \uc0dd\uac01\ud574\uc694",
]
COMPARISON_MARKERS = [
    "better", "worse", "than", "prefer", "compared to", "versus", "vs",
    "\ub354 \ub0ab", "\ub354 \uc88b", "\ube44\uad50", "\ubcf4\ub2e4", "\uc120\ud638",
]
PREFERENCE_MARKERS = [
    "prefer", "better", "would choose", "i like", "personally",
    "\uc120\ud638", "\ub354 \uc88b", "\uac1c\uc778\uc801\uc73c\ub85c", "\uc800\ub294",
]
STRONG_NEGATIVE_MARKERS = [
    "bad", "terrible", "awful", "hate", "broken", "broke", "doesn't work", "doesnt work",
    "problem", "issue", "issues", "frustrating", "disappointed", "waste", "useless",
    "poor", "worst", "failed", "failure", "annoying", "inconvenient", "complaint",
    "\uace0\uc7a5", "\ubd88\ub9cc", "\ubb38\uc81c", "\ub2f5\ub2f5", "\ubcc4\ub85c", "\uc2e4\ub9dd", "\ucd5c\uc545", "\ub098\uc058",
    "\uc548 \ub41c\ub2e4", "\uc548\ub3fc", "\ubd88\ud3b8", "\uc218\ub9ac", "\uc5d0\ub7ec",
]
STRONG_POSITIVE_MARKERS = [
    "good", "great", "excellent", "love", "recommend", "satisfied", "works great", "very good",
    "best", "amazing", "perfect", "happy with",
    "\ub9cc\uc871", "\ucd94\ucc9c", "\uc88b\uc544\uc694", "\ub9e4\uc6b0 \uc88b", "\ucd5c\uace0", "\uc798 \ub41c\ub2e4", "\uc88b\ub124\uc694",
]
INFORMATIONAL_MARKERS = [
    "how to", "because", "means", "in case", "for example", "usually", "diagnostic", "repair", "maintenance",
    "\uc124\uba85", "\uc815\ubcf4", "\uc6d0\uc778", "\ubc29\ubc95", "\uc608\ub97c \ub4e4\uba74", "\ubcf4\ud1b5",
]
EMOTIONAL_ONLY_MARKERS = [
    "nightmares", "scary", "funny", "hilarious", "lol", "lmao", "haha",
    "\ubb34\uc11c\uc6e0", "\uc6c3\uae30\ub124", "\u3154\u3154", "\u314b\u314b", "\ud6c4\ub35c\ub35c",
]
CHANNEL_FEEDBACK_MARKERS = [
    "video", "videos", "channel", "creator", "youtuber", "youtube", "editing", "edit", "thumbnail", "host", "reviewer",
    "\uC601\uC0C1", "\uCC44\uB110", "\uC720\uD29C\uBE0C", "\uD3B8\uC9D1", "\uC36C\uB124\uC77C", "\uC9C4\uD589", "\uC9C4\uD589\uC790", "\uB9AC\uBDF0\uC5B4", "\uD06C\uB9AC\uC5D0\uC774\uD130", "\uC124\uBA85 \uC798", "\uC601\uC0C1\uBBF8",
]
GENERIC_POSITIVE_ONLY_MARKERS = [
    "good", "great", "nice", "awesome", "amazing", "best", "love it", "thanks", "thank you",
    "\uC88B\uC544\uC694", "\uC88B\uB124\uC694", "\uC88B\uB124", "\uCD5C\uACE0", "\uAC10\uC0AC", "\uAC10\uC0AC\uD569\uB2C8\uB2E4", "\uAD7F", "\uB300\uBC15",
]
PRODUCT_MARKERS = [
    "washer", "washing machine", "dryer", "refrigerator", "fridge", "dishwasher",
    "drum", "top load", "top-load", "smart", "ai", "digital", "noise", "repair", "clean", "wash",
    "service", "support", "delivery", "shipping", "price", "design", "performance", "maintenance",
    "\uc138\ud0c1\uae30", "\uac74\uc870\uae30", "\ub0c9\uc7a5\uace0", "\uc2dd\uae30\uc138\ucc99\uae30", "\ub4dc\ub7fc", "\ud1b5\ub3cc\uc774",
    "\uc2a4\ub9c8\ud2b8", "\uc18c\uc74c", "\uc138\ucc99", "\uc218\ub9ac", "\ubc30\uc1a1", "\uc11c\ube44\uc2a4", "\uac00\uaca9", "\ub514\uc790\uc778",
]
TARGET_PATTERNS = {
    "drum_washer": ["drum washer", "drum washers", "drum", "\ub4dc\ub7fc", "\ub4dc\ub7fc \uc138\ud0c1\uae30"],
    "top_load_washer": ["top load", "top-load", "top loader", "\ud1b5\ub3cc\uc774", "\ud1b5\ub3cc\uc774 \uc138\ud0c1\uae30"],
    "washer": ["washer", "washing machine", "\uc138\ud0c1\uae30"],
    "dryer": ["dryer", "\uac74\uc870\uae30"],
    "refrigerator": ["refrigerator", "fridge", "\ub0c9\uc7a5\uace0"],
    "dishwasher": ["dishwasher", "\uc2dd\uae30\uc138\ucc99\uae30"],
}
TARGET_LABELS = {
    "drum_washer": "\ub4dc\ub7fc \uc138\ud0c1\uae30",
    "top_load_washer": "\ud1b5\ub3cc\uc774 \uc138\ud0c1\uae30",
    "washer": "\uc138\ud0c1\uae30",
    "dryer": "\uac74\uc870\uae30",
    "refrigerator": "\ub0c9\uc7a5\uace0",
    "dishwasher": "\uc2dd\uae30\uc138\ucc99\uae30",
}


@dataclass
class CommentAnalysisResult:
    product_related: bool
    product_target: str | None
    sentiment_label: str | None
    sentiment_score: float | None
    signal_strength: str
    classification_type: str
    confidence_level: str
    context_used: bool
    context_required_for_display: bool
    reason: str
    needs_review: bool
    insight_type: str = "일반_의견"
    # nlp_analyzer enrichment fields
    nlp_label: str | None = None
    nlp_confidence: float | None = None
    nlp_sentiment_reason: str | None = None
    nlp_topics: list[str] = field(default_factory=list)
    nlp_topic_sentiments: dict[str, str] = field(default_factory=dict)
    nlp_is_inquiry: bool = False
    nlp_is_rhetorical: bool = False
    nlp_summary: str | None = None
    nlp_keywords: list[str] = field(default_factory=list)
    nlp_product_mentions: list[str] = field(default_factory=list)
    nlp_language: str | None = None
    nlp_provider: str | None = None
    nlp_model: str | None = None


def _contains_any(text: str, markers: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").lower().split())


def _is_channel_feedback(text: str) -> bool:
    lowered = _normalize_text(text)
    return _contains_any(lowered, CHANNEL_FEEDBACK_MARKERS)


def _is_generic_positive_only(text: str) -> bool:
    lowered = _normalize_text(text)
    compact = re.sub(r"[^\w\s\uac00-\ud7a3]", " ", lowered)
    compact = " ".join(compact.split())
    return compact in GENERIC_POSITIVE_ONLY_MARKERS


def _detect_target(text: str, parent_comment: str, context_comments: list[str] | None) -> str | None:
    blob = " ".join([text or "", parent_comment or "", " ".join(context_comments or [])]).lower()
    for label, markers in TARGET_PATTERNS.items():
        if any(marker in blob for marker in markers):
            return label
    return None


def _is_product_related(text: str, parent_comment: str, context_comments: list[str] | None, is_agreement: bool) -> tuple[bool, bool]:
    source = text or ""
    parent = parent_comment or ""
    context_blob = " ".join(context_comments or [])
    own_related = _contains_any(source, PRODUCT_MARKERS)
    parent_related = _contains_any(parent, PRODUCT_MARKERS)
    context_related = _contains_any(context_blob, PRODUCT_MARKERS)

    if own_related:
        return True, False
    if _is_channel_feedback(source) or _contains_any(source, EMOTIONAL_ONLY_MARKERS):
        return False, False
    if _is_generic_positive_only(source):
        return False, False
    if is_agreement and (parent_related or context_related):
        return True, True
    if parent_related and (_is_comparison(source, parent) or _is_preference(source)):
        return True, True
    return False, False


def _is_strong_negative(text: str) -> bool:
    lowered = (text or "").lower()
    negation_safe = [
        "no issues", "never had any issues", "without issues",
        "no problem", "never had a problem", "not bad",
        "\ubb38\uc81c\uac00 \uc5c6", "\uace0\uc7a5 \uc5c6\uc774", "\ubd88\ub9cc\uc740 \uc5c6",
    ]
    if any(token in lowered for token in negation_safe):
        trimmed = lowered
        for token in negation_safe:
            trimmed = trimmed.replace(token, "")
        if not _contains_any(trimmed, STRONG_NEGATIVE_MARKERS):
            return False
    return _contains_any(lowered, STRONG_NEGATIVE_MARKERS)


def _is_strong_positive(text: str) -> bool:
    lowered = (text or "").lower()
    if any(token in lowered for token in ["never had any issues", "lasted forever", "\ubb38\uc81c\uac00 \uc5c6", "\uace0\uc7a5 \uc5c6\uc774", "\ubd88\ub9cc\uc740 \uc5c6"]):
        return True
    return _contains_any(lowered, STRONG_POSITIVE_MARKERS)


def _is_comparison(text: str, parent_comment: str) -> bool:
    return _contains_any(f"{text} {parent_comment}", COMPARISON_MARKERS)


def _is_preference(text: str) -> bool:
    return _contains_any(text, PREFERENCE_MARKERS)


def _build_reason(sentiment: str, classification_type: str, signal_strength: str, context_used: bool, target: str | None) -> str:
    target_label = TARGET_LABELS.get(target or "", "\ud574\ub2f9 \uc81c\ud488")
    prefix = "\ub9e5\ub77d\uc744 \ud568\uaed8 \ubd24\uc744 \ub54c, " if context_used else ""
    if sentiment == "negative":
        return f"{prefix}{target_label}\uc5d0 \ub300\ud55c \uc2e4\uc81c \ubd88\ub9cc\uc774 \uba85\ud655\ud558\uac8c \ub4dc\ub7ec\ub098 \ubd80\uc815\uc73c\ub85c \ubd84\ub958\ud588\uc2b5\ub2c8\ub2e4. \ube44\uc988\ub2c8\uc2a4 \uad00\uc810\uc5d0\uc11c\ub294 \uac1c\uc120 \ud544\uc694 \uc2e0\ud638\uac00 {signal_strength}\ud55c \ud3b8\uc785\ub2c8\ub2e4."
    if sentiment == "positive":
        return f"{prefix}{target_label}\uc5d0 \ub300\ud55c \ub9cc\uc871\uac10\uc774\ub098 \ucd94\ucc9c \uc758\uc0ac\uac00 \uba85\ud655\ud558\uc5ec \uae0d\uc815\uc73c\ub85c \ubd84\ub958\ud588\uc2b5\ub2c8\ub2e4. \ube44\uc988\ub2c8\uc2a4 \uad00\uc810\uc5d0\uc11c\ub294 \uac15\uc810 \uc2e0\ud638\uac00 {signal_strength}\ud55c \ud3b8\uc785\ub2c8\ub2e4."
    if classification_type == "comparison":
        return f"{prefix}\uc774 \ub313\uae00\uc740 {target_label} \uad00\ub828 \ube44\uad50\ub098 \uc120\ud638 \ud45c\ud604\uc774\uc9c0\ub9cc, \uba85\ud655\ud55c \ubd88\ub9cc\uc740 \uc544\ub2c8\ub77c\uace0 \ud310\ub2e8\ud574 \uc911\ub9bd\uc73c\ub85c \ubd84\ub958\ud588\uc2b5\ub2c8\ub2e4. \ube44\uc988\ub2c8\uc2a4 \uad00\uc810\uc5d0\uc11c\ub294 {signal_strength}\ud55c \ucc38\uace0 \uc2e0\ud638\uc785\ub2c8\ub2e4."
    if classification_type == "preference":
        return f"{prefix}\uc774 \ub313\uae00\uc740 {target_label} \uc5d0 \ub300\ud55c \uac1c\uc778 \uc120\ud638 \ud45c\ud604\uc73c\ub85c, \ubd88\ub9cc\uc774\ub098 \uc2e4\ub9dd\uc774 \uba85\ud655\ud558\uc9c0 \uc54a\uc544 \uc911\ub9bd\uc73c\ub85c \ubd84\ub958\ud588\uc2b5\ub2c8\ub2e4."
    return f"{prefix}\uc774 \ub313\uae00\uc740 {target_label} \uad00\ub828 \uc815\ubcf4 \uacf5\uc720 \ub610\ub294 \uc911\ub9bd\uc801 \uad00\ucc30\uc5d0 \uac00\uae5d\uae30 \ub54c\ubb38\uc5d0 \uc911\ub9bd\uc73c\ub85c \ubd84\ub958\ud588\uc2b5\ub2c8\ub2e4."


def _analyze_core(text: str, parent_comment: str, context_comments: list[str] | None, is_reply: bool) -> tuple[bool, str | None, str, str, str, bool, bool, bool, float]:
    base_sentiment, base_score = score_sentiment(text or "")
    is_agreement = _contains_any(text, AGREEMENT_MARKERS)
    is_short_dependent = len((text or "").strip()) <= 60 and _contains_any(text, SHORT_DEPENDENT_MARKERS)
    product_related, inherited_context = _is_product_related(text, parent_comment, context_comments, is_agreement)
    target = _detect_target(text, parent_comment, context_comments)
    comparison = _is_comparison(text, parent_comment)
    preference = _is_preference(text)
    direct_negative = _is_strong_negative(text)
    direct_positive = _is_strong_positive(text)
    context_used = False
    context_required = False
    needs_review = False

    if not product_related:
        return False, target, "neutral", "weak", "informational", inherited_context, inherited_context, is_reply and not bool(parent_comment), 0.0

    if comparison:
        if direct_negative and not preference:
            return True, target, "negative", "strong", "comparison", bool(parent_comment), bool(parent_comment), False, float(base_score or -0.3)
        return True, target, "neutral", "weak", "comparison", bool(parent_comment), bool(parent_comment), False, 0.0

    if preference and not direct_negative and not direct_positive:
        return True, target, "neutral", "weak", "preference", False, False, False, 0.0

    if parent_comment and (is_agreement or is_short_dependent):
        parent_related, _ = _is_product_related(parent_comment, "", context_comments, False)
        if parent_related:
            p_neg = _is_strong_negative(parent_comment)
            p_pos = _is_strong_positive(parent_comment)
            if p_neg:
                return True, target, "negative", "strong", "complaint", True, True, False, float(score_sentiment(parent_comment)[1] or -0.4)
            if p_pos:
                return True, target, "positive", "strong", "preference", True, True, False, float(score_sentiment(parent_comment)[1] or 0.4)
        context_used = True
        context_required = True
        needs_review = True
        return True, target, "neutral", "weak", "informational", context_used, context_required, needs_review, 0.0

    if direct_negative:
        return True, target, "negative", "strong", "complaint", False, False, False, float(base_score or -0.4)

    if direct_positive:
        return True, target, "positive", "strong", "preference", False, False, False, float(base_score or 0.4)

    if _contains_any(text, INFORMATIONAL_MARKERS):
        return True, target, "neutral", "weak", "informational", False, False, False, 0.0

    if base_sentiment == "negative":
        return True, target, "neutral", "weak", "informational", False, False, True, float(base_score or 0.0)
    if base_sentiment == "positive":
        return True, target, "neutral", "weak", "informational", False, False, True, float(base_score or 0.0)
    return True, target, "neutral", "weak", "informational", False, False, False, float(base_score or 0.0)


def analyze_comment_with_context(
    text: str,
    *,
    validity: str,
    exclusion_reason: str | None = None,
    parent_comment: str | None = None,
    context_comments: list[str] | None = None,
    is_reply: bool = False,
    comment_id: str | None = None,
    nlp_provider: Any | None = None,
    use_nlp: bool = True,
    video_context: dict[str, str] | None = None,
) -> CommentAnalysisResult:
    if validity != "valid":
        reason = f"\uc81c\uc678 \ub313\uae00\ub85c \ud310\ub2e8\ud588\uc2b5\ub2c8\ub2e4. \uc0ac\uc720: {exclusion_reason or 'low quality'}"
        return CommentAnalysisResult(False, None, None, None, "weak", "informational", "low", False, False, reason, False)

    # Resolve video context
    vc = video_context or {}
    video_type = vc.get("type") or classify_video_type(vc.get("title", ""), vc.get("channel", ""))

    source_text = (text or "").strip()
    parent_text = (parent_comment or "").strip()
    product_related, target, sentiment, signal_strength, classification_type, context_used, context_required, needs_review, score = _analyze_core(
        source_text,
        parent_text,
        context_comments,
        is_reply,
    )

    if sentiment == "negative":
        confidence = "high" if signal_strength == "strong" and classification_type == "complaint" else "medium"
    elif sentiment == "positive":
        confidence = "high" if signal_strength == "strong" else "medium"
    else:
        confidence = "low" if needs_review or context_required else "medium"

    reason = _build_reason(sentiment, classification_type, signal_strength, context_used, target)

    # nlp_analyzer enrichment
    nlp_fields: dict[str, Any] = {}
    if use_nlp and NLP_ANALYZER_AVAILABLE and source_text:
        cid = comment_id or "unknown"
        nlp_result = analyze_with_nlp(cid, source_text, provider=nlp_provider)
        if nlp_result is not None:
            nlp_fields = nlp_result
            # Override sentiment with nlp_analyzer when it has high confidence
            nlp_label = nlp_result.get("nlp_label", "")
            nlp_conf = nlp_result.get("nlp_confidence", 0.0)
            if nlp_label in ("positive", "negative", "neutral") and nlp_conf >= 0.6:
                sentiment = nlp_label
                score = nlp_result.get("nlp_sentiment_score", score)
                reason = nlp_result.get("nlp_sentiment_reason", reason)
                confidence = "high" if nlp_conf >= 0.8 else "medium"
            elif nlp_label == "trash":
                sentiment = "neutral"
                classification_type = "informational"
                signal_strength = "weak"
                reason = nlp_result.get("nlp_sentiment_reason", "스팸/무의미 댓글로 판단")
            # Enrich product detection from nlp_analyzer
            nlp_products = nlp_result.get("nlp_product_mentions", [])
            if nlp_products and not product_related:
                product_related = True
                if not target:
                    target = _detect_target_from_nlp(nlp_products)

    # Compute insight_type from video context + analysis result
    is_inquiry = nlp_fields.get("nlp_is_inquiry", False)
    nlp_topics_list = nlp_fields.get("nlp_topics", [])
    insight_type = _infer_insight_type(video_type, classification_type, sentiment, is_inquiry, nlp_topics_list, source_text)

    return CommentAnalysisResult(
        product_related=product_related,
        product_target=target,
        sentiment_label=sentiment,
        sentiment_score=score,
        signal_strength=signal_strength,
        classification_type=classification_type,
        confidence_level=confidence,
        context_used=context_used,
        context_required_for_display=context_required,
        reason=reason,
        needs_review=needs_review,
        insight_type=insight_type,
        nlp_label=nlp_fields.get("nlp_label"),
        nlp_confidence=nlp_fields.get("nlp_confidence"),
        nlp_sentiment_reason=nlp_fields.get("nlp_sentiment_reason"),
        nlp_topics=nlp_fields.get("nlp_topics", []),
        nlp_topic_sentiments=nlp_fields.get("nlp_topic_sentiments", {}),
        nlp_is_inquiry=nlp_fields.get("nlp_is_inquiry", False),
        nlp_is_rhetorical=nlp_fields.get("nlp_is_rhetorical", False),
        nlp_summary=nlp_fields.get("nlp_summary"),
        nlp_keywords=nlp_fields.get("nlp_keywords", []),
        nlp_product_mentions=nlp_fields.get("nlp_product_mentions", []),
        nlp_language=nlp_fields.get("nlp_language"),
        nlp_provider=nlp_fields.get("nlp_provider"),
        nlp_model=nlp_fields.get("nlp_model"),
    )


VIDEO_TYPE_PATTERNS: dict[str, list[str]] = {
    "promotional": ["할인", "프로모션", "세일", "특가", "쿠팡", "최저가", "이벤트", "sale", "deal", "discount", "coupon"],
    "review": ["리뷰", "후기", "사용기", "솔직", "장단점", "review", "honest", "pros and cons"],
    "comparison": ["비교", "vs", "대결", "차이", "comparison", "versus", "which is better"],
    "unboxing": ["언박싱", "개봉기", "unboxing", "first look"],
    "tutorial": ["설치", "사용법", "설명", "how to", "tutorial", "guide", "방법"],
    "official": ["공식", "official", "광고", "제조사"],
}

INSIGHT_TYPE_MAP: dict[tuple[str, str], str] = {
    # (video_type, classification_type) → insight_type
    ("promotional", "complaint"): "상품구성_피드백",
    ("promotional", "preference"): "구매전환_증거",
    ("promotional", "informational"): "구매_문의",
    ("review", "complaint"): "품질_불만",
    ("review", "preference"): "만족_포인트",
    ("comparison", "complaint"): "경쟁사_비교",
    ("comparison", "preference"): "경쟁사_비교",
    ("comparison", "comparison"): "경쟁사_비교",
    ("tutorial", "complaint"): "사용성_불만",
    ("tutorial", "informational"): "사용_팁",
    ("official", "complaint"): "품질_불만",
    ("official", "preference"): "브랜드_신뢰",
}


def classify_video_type(title: str, channel: str = "") -> str:
    """Infer video type from title/channel text."""
    blob = f"{title or ''} {channel or ''}".lower()
    for vtype, patterns in VIDEO_TYPE_PATTERNS.items():
        if any(p in blob for p in patterns):
            return vtype
    return "general"


_PURCHASE_KEYWORDS = ["구매", "샀", "샀어", "주문", "결제", "bought", "purchased", "ordered"]


def _infer_insight_type(
    video_type: str,
    classification_type: str,
    sentiment: str,
    is_inquiry: bool,
    nlp_topics: list[str] | None = None,
    text: str = "",
) -> str:
    """Determine business insight type from video context + comment analysis."""
    lowered = (text or "").lower()

    if is_inquiry:
        if video_type == "promotional":
            return "구매_문의"
        return "문의_피드백"

    # Purchase confirmation detection (text-level)
    if any(kw in lowered for kw in _PURCHASE_KEYWORDS):
        if video_type == "promotional":
            return "구매전환_증거"

    mapped = INSIGHT_TYPE_MAP.get((video_type, classification_type))
    if mapped:
        return mapped

    # Fallback based on sentiment + content
    topics_blob = " ".join(nlp_topics or []).lower()
    if sentiment == "negative":
        if any(kw in topics_blob for kw in ["가격", "비용", "price", "cost"]):
            return "가격_민감"
        if any(kw in topics_blob for kw in ["수리", "서비스", "as", "repair", "service"]):
            return "서비스_불만"
        return "품질_불만"
    if sentiment == "positive":
        if video_type == "promotional":
            return "구매전환_증거"
        return "만족_포인트"
    return "일반_의견"


def _detect_target_from_nlp(product_mentions: list[str]) -> str | None:
    """Map nlp_analyzer product_mentions to existing target labels."""
    product_map = {
        "냉장고": "refrigerator", "세탁기": "washer", "건조기": "dryer",
        "식기세척기": "dishwasher", "드럼": "drum_washer", "통돌이": "top_load_washer",
        "refrigerator": "refrigerator", "fridge": "refrigerator",
        "washer": "washer", "dryer": "dryer", "dishwasher": "dishwasher",
        "drum": "drum_washer",
    }
    for mention in product_mentions:
        lowered = mention.lower().strip()
        for key, target in product_map.items():
            if key in lowered:
                return target
    return None
