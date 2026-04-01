"""Text cleaning helpers for preprocessing and validity filtering."""

from __future__ import annotations

import html
import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MULTISPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
REPEATED_TOKEN_PATTERN = re.compile(r"\b(\w{2,})\b(?:\s+\1\b){2,}", re.IGNORECASE)
LATIN_WORD_PATTERN = re.compile(r"[A-Za-z]")
HANGUL_PATTERN = re.compile(r"[\uac00-\ud7a3]")
ONLY_PUNCT_PATTERN = re.compile(r"^[\W_]+$")

_GENERIC_WORDS = {
    "first", "nice", "great", "good", "lol", "wow", "omg", "thanks", "thank you",
    "cool", "awesome", "amazing", "best", "love it"
}
_GENERIC_KO = {
    "\uc88b\uc544\uc694", "\ucd5c\uace0", "\uac10\uc0ac", "\uac10\uc0ac\ud569\ub2c8\ub2e4", "\uc640", "\ub300\ubc15", "\uad7f", "\u314b\u314b", "\u314e\u314e", "\uc624"
}
_AD_MARKERS = {
    "promo code", "discount link", "dm me", "telegram", "whatsapp", "http://", "https://", "www.",
    "\ub9c1\ud06c", "\ucfe0\ud321", "\ud30c\ud2b8\ub108\uc2a4", "\ub124\uc774\ubc84\uc1fc\ud551", "\uc1fc\ud551\ucee4\ub125\ud2b8",
    "\uc81c\ud734", "\uad11\uace0", "\uc218\uc218\ub8cc", "\uace0\uc815\ub313\uae00", "\ub9c1\ud06c\ub85c \uad6c\uc785",
}

_MEANINGLESS_MARKERS = {"lol", "\u314b\u314b", "\u314e\u314e", "wow", "omg"}


def clean_text(text: str, *, remove_urls: bool = True, preserve_emoji: bool = True, normalize_repeats: bool = True) -> str:
    value = html.unescape(text or "")
    value = value.replace("\n", " ").replace("\r", " ")
    if remove_urls:
        value = URL_PATTERN.sub(" ", value)
    if normalize_repeats:
        value = REPEATED_CHAR_PATTERN.sub(lambda match: match.group(1) * 2, value)
        value = REPEATED_TOKEN_PATTERN.sub(lambda match: match.group(1), value)
    if not preserve_emoji:
        value = re.sub(r"[^\w\s\uac00-\ud7a3]", " ", value)
    value = MULTISPACE_PATTERN.sub(" ", value).strip()
    return value


def _normalize_for_rules(text: str) -> str:
    lowered = clean_text(text or "", preserve_emoji=True).lower().strip()
    return MULTISPACE_PATTERN.sub(" ", lowered)


def classify_comment_quality(text: str) -> str:
    raw = html.unescape(text or "").strip()
    normalized = _normalize_for_rules(raw)
    if not normalized:
        return "meaningless"

    if any(marker in normalized for marker in _AD_MARKERS):
        return "ad"
    tokens = raw.split()
    if tokens and len(tokens) <= 3 and all(token.startswith("@") for token in tokens):
        return "tagging_only"
    if ONLY_PUNCT_PATTERN.fullmatch(raw):
        return "meaningless"
    if REPEATED_TOKEN_PATTERN.search(raw):
        return "repetitive"

    stripped = re.sub(r"[^\w\uac00-\ud7a3]", "", normalized)
    if len(stripped) <= 1:
        return "meaningless"

    normalized_no_punct = re.sub(r"[^\w\s\uac00-\ud7a3]", "", normalized).strip()
    if normalized_no_punct in _GENERIC_WORDS or normalized_no_punct in _GENERIC_KO:
        return "generic_engagement"
    if normalized_no_punct in _MEANINGLESS_MARKERS:
        return "meaningless"

    word_count = len(normalized.split())
    if word_count == 1 and len(stripped) <= 4:
        return "low_context"

    return "valid"


def detect_language_hint(text: str) -> str:
    value = text or ""
    if HANGUL_PATTERN.search(value):
        return "ko"
    if LATIN_WORD_PATTERN.search(value):
        return "en"
    return "unknown"
