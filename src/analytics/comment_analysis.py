"""Context-aware product feedback analysis helpers."""

from __future__ import annotations

from dataclasses import dataclass

from src.analytics.sentiment import score_sentiment

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


def _contains_any(text: str, markers: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in markers)


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
    if context_related and (_is_strong_negative(source) or _is_strong_positive(source) or _is_comparison(source, parent) or _is_preference(source)):
        return True, True
    if is_agreement and (parent_related or context_related):
        return True, True
    if _contains_any(source, EMOTIONAL_ONLY_MARKERS) and not own_related:
        return False, False
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
) -> CommentAnalysisResult:
    if validity != "valid":
        reason = f"\uc81c\uc678 \ub313\uae00\ub85c \ud310\ub2e8\ud588\uc2b5\ub2c8\ub2e4. \uc0ac\uc720: {exclusion_reason or 'low quality'}"
        return CommentAnalysisResult(False, None, None, None, "weak", "informational", "low", False, False, reason, False)

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
    )
