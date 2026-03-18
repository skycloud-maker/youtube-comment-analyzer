"""Lightweight text translation helpers."""

from __future__ import annotations

from functools import lru_cache

from deep_translator import GoogleTranslator


@lru_cache(maxsize=2048)
def translate_to_korean(text: str) -> str:
    value = (text or "").strip()
    if not value:
        return ""
    try:
        return GoogleTranslator(source="auto", target="ko").translate(value)
    except Exception:
        return value
