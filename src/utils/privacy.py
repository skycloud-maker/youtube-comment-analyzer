"""PII and external-link helpers for dashboard-safe rendering."""

from __future__ import annotations

import re
from typing import Any

PHONE_RE = re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\d{2,4}[\s-]?){2,4}\d{3,4}\b")
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
ACCOUNT_RE = re.compile(r"@[A-Za-z0-9_\.]{3,}")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def extract_external_links(text: str) -> list[str]:
    value = _safe_text(text)
    if not value:
        return []
    return list(dict.fromkeys(match.group(0) for match in URL_RE.finditer(value)))


def detect_pii(text: str) -> tuple[str, str, str]:
    value = _safe_text(text)
    if not value:
        return "N", "", ""
    pii_types: list[str] = []
    pii_values: list[str] = []

    for match in PHONE_RE.finditer(value):
        token = match.group(0)
        digits = re.sub(r"\D", "", token)
        if len(digits) >= 9:
            pii_types.append("phone")
            pii_values.append(token)
    for match in EMAIL_RE.finditer(value):
        pii_types.append("email")
        pii_values.append(match.group(0))
    for match in ACCOUNT_RE.finditer(value):
        pii_types.append("account_id")
        pii_values.append(match.group(0))

    uniq_types = list(dict.fromkeys(pii_types))
    uniq_values = list(dict.fromkeys(pii_values))
    return ("Y" if uniq_values else "N", ", ".join(uniq_types), ", ".join(uniq_values))


def mask_pii_text(text: str, pii_raw_text: str) -> str:
    value = _safe_text(text)
    pii_raw = _safe_text(pii_raw_text)
    if not value or not pii_raw:
        return value
    masked = value
    for token in [part.strip() for part in pii_raw.split(",") if part.strip()]:
        if "@" in token and "." in token:
            name, _, domain = token.partition("@")
            repl = (name[:1] + "***@" + domain) if name else "***@" + domain
        else:
            digits = re.sub(r"\D", "", token)
            if len(digits) >= 9:
                if "-" in token:
                    parts = token.split("-")
                    if len(parts) >= 3:
                        repl = f"{parts[0]}-****-{parts[-1]}"
                    else:
                        repl = digits[:3] + "****" + digits[-4:]
                else:
                    repl = digits[:3] + "****" + digits[-4:]
            elif len(token) >= 3:
                repl = token[0] + "*" * max(1, len(token) - 2) + token[-1]
            else:
                repl = "[PII]"
        masked = masked.replace(token, repl)
    return masked
