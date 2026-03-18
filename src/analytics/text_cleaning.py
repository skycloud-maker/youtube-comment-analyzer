"""Text cleaning helpers for Korean-friendly preprocessing."""

from __future__ import annotations

import html
import re

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MULTISPACE_PATTERN = re.compile(r"\s+")
REPEATED_CHAR_PATTERN = re.compile(r"(.)\1{3,}")
REPEATED_TOKEN_PATTERN = re.compile(r"\b(\w{2,})\b(?:\s+\1\b){2,}", re.IGNORECASE)
NON_TEXT_PATTERN = re.compile(r"[^\w\sㄱ-ㅎ가-힣ㅠㅜ]+")
BOILERPLATE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^first!?$",
        r"^1등!?$",
        r"구독\s*좋아요",
        r"알림\s*설정",
        r"카톡",
        r"오픈채팅",
        r"비트코인",
        r"재테크",
    ]
]
LOW_SIGNAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"^(좋아요|잘보고갑니다|감사합니다|최고예요|최고입니다|굿|good|great)[!~. ]*$",
        r"^(맞아요|인정|공감)[!~. ]*$",
        r"^(정말|진짜|너무|완전)?\s*(좋아요|최고예요|최고입니다|감사합니다)[!~. ]*$",
    ]
]



def clean_text(text: str, *, remove_urls: bool = True, preserve_emoji: bool = True, normalize_repeats: bool = True) -> str:
    value = html.unescape(text or "")
    value = value.replace("\n", " ").replace("\r", " ")
    if remove_urls:
        value = URL_PATTERN.sub(" ", value)
    if normalize_repeats:
        value = REPEATED_CHAR_PATTERN.sub(lambda match: match.group(1) * 2, value)
    value = REPEATED_TOKEN_PATTERN.sub(lambda match: match.group(1), value)
    value = MULTISPACE_PATTERN.sub(" ", value).strip()
    for pattern in BOILERPLATE_PATTERNS:
        if pattern.search(value):
            return ""
    if not preserve_emoji:
        value = re.sub(r"[^\w\sㄱ-ㅎ가-힣]", " ", value)
        value = MULTISPACE_PATTERN.sub(" ", value).strip()
    return value



def classify_comment_quality(text: str) -> str:
    value = clean_text(text, preserve_emoji=True)
    if not value:
        return "spam"
    condensed = NON_TEXT_PATTERN.sub("", value)
    if not condensed:
        return "noise"
    if len(condensed) <= 2:
        return "noise"
    if any(pattern.search(value) for pattern in LOW_SIGNAL_PATTERNS):
        return "low_signal"
    if len(value.split()) == 1 and len(condensed) <= 4:
        return "low_signal"
    if sum(1 for char in value if char in "!?~ㅋㅎㅠㅜ") >= max(len(value) // 2, 3):
        return "noise"
    return "meaningful"



def detect_language_hint(text: str) -> str:
    if re.search(r"[가-힣]", text):
        return "ko"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "unknown"
