"""Keyword and phrase extraction for TM pipeline."""

from __future__ import annotations

from collections import Counter
import re
from typing import Any

import pandas as pd

_REQUIRED_BASE_COLUMNS: tuple[str, ...] = ("comment_id", "text_clean")
_MIN_PHRASE_COUNT = 3

_TOKEN_PATTERN = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)+|[0-9]{2,}[A-Za-z0-9_-]*|[\uac00-\ud7a3]{2,}"
)
_REPEAT_NOISE_PATTERN = re.compile(r"(.)\1{3,}")

# TM.2.5 stopword refinement
_STOPWORDS_KO = {
    "\uadf8\ub9ac\uace0",
    "\uadf8\ub7ec\ub098",
    "\ud558\uc9c0\ub9cc",
    "\uadf8\ub7f0\ub370",
    "\ub610\ud55c",
    "\uc815\ub9d0",
    "\uc9c4\uc9dc",
    "\ub108\ubb34",
    "\uc57d\uac04",
    "\uadf8\ub0e5",
    "\uc774\uac70",
    "\uc800\uac70",
    "\uc774\uac74",
    "\uc800\uac74",
    "\uadf8\uac70",
    "\uc5ec\uae30",
    "\uac70\uae30",
    "\uc5d0\uc11c",
    "\uc73c\ub85c",
    "\uc5d0\uac8c",
    "\uc774\ub791",
    "\uac19\uc740",
    "\ub300\ud55c",
    "\ud558\ub294",
    "\ud558\uace0",
    "\ud574\uc11c",
    "\ud558\uba74",
    "\uc785\ub2c8\ub2e4",
    "\uc788\uc5b4\uc694",
    "\uc788\uc2b5\ub2c8\ub2e4",
    "\uc5c6\uc5b4\uc694",
    "\uc5c6\uc2b5\ub2c8\ub2e4",
    "\uc88b\uc544\uc694",
    "\uac10\uc0ac",
    "\uac10\uc0ac\ud569\ub2c8\ub2e4",
    "\uc601\uc0c1",
    "\ub9ac\ubdf0",
    "\uc88b\ub124",
    "\ub098\uc058\ub2e4",
    "\ub9cc\ub4e4\ub124",
}

_STOPWORDS_EN = {
    "the",
    "a",
    "an",
    "and",
    "as",
    "or",
    "but",
    "if",
    "to",
    "of",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "it",
    "its",
    "it's",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "we",
    "they",
    "me",
    "my",
    "your",
    "our",
    "their",
    "do",
    "does",
    "did",
    "doing",
    "have",
    "has",
    "had",
    "having",
    "can",
    "could",
    "should",
    "would",
    "will",
    "just",
    "very",
    "really",
    "also",
    "than",
    "then",
    "there",
    "here",
    "why",
    "what",
    "when",
    "where",
    "which",
    "who",
    "how",
    "all",
    "one",
    "ones",
    "them",
    "theyre",
    "you're",
    "dont",
    "don't",
    "cant",
    "can't",
    "wont",
    "won't",
    "im",
    "i'm",
    "ive",
    "i've",
    "youre",
    "you've",
    "well",
    "like",
    "use",
    "used",
    "using",
    "get",
    "got",
    "make",
    "made",
    "put",
    "some",
    "know",
    "need",
    "much",
    "many",
    "more",
    "most",
    "less",
    "lot",
    "lots",
    "thing",
    "things",
    "stuff",
    "good",
    "great",
    "nice",
    "bad",
    "hand",
    "people",
    "time",
    "times",
    "now",
    "only",
    "before",
    "after",
    "because",
    "even",
    "never",
    "always",
    "still",
    "again",
    "back",
    "first",
    "last",
    "next",
    "pre",
    "off",
    "someone",
    "anyone",
    "everyone",
    "about",
    "into",
    "out",
    "up",
    "down",
    "so",
    "too",
    "not",
    "no",
    "yes",
    "thanks",
    "thank",
    "video",
    "review",
}

_NOISE_TOKENS = {
    "\u314b\u314b",
    "\u314b\u314b\u314b",
    "\u314b\u314b\u314b\u314b",
    "\u314e\u314e",
    "\u314e\u314e\u314e",
    "\u3160\u3160",
    "\u315c\u315c",
    "lol",
    "lmao",
    "rofl",
    "xd",
}

_TOKEN_NORMALIZATION_MAP = {
    "dishwashers": "dishwasher",
    "dishes": "dish",
    "pods": "pod",
    "detergents": "detergent",
    "filters": "filter",
    "machines": "machine",
    "washing": "wash",
    "washes": "wash",
    "washed": "wash",
    "rinsing": "rinse",
    "rinses": "rinse",
    "rinsed": "rinse",
    "cleaning": "clean",
    "cleans": "clean",
    "cleaned": "clean",
    "prices": "price",
    "costs": "cost",
    "issues": "issue",
    "problems": "problem",
}

_PRODUCT_TOKENS = {
    "dishwasher",
    "dish",
    "machine",
    "filter",
    "detergent",
    "pod",
    "powder",
    "liquid",
    "water",
    "cycle",
    "laundry",
    "price",
    "cost",
    "performance",
    "quality",
    "cleaning",
    "noise",
    "service",
    "repair",
    "delivery",
    "installation",
    "design",
    "maintenance",
    "\uac00\uaca9",
    "\uc131\ub2a5",
    "\ud488\uc9c8",
    "\uc138\ucc99",
    "\uac74\uc870",
    "\uc18c\uc74c",
    "\uc124\uce58",
    "\ubc30\uc1a1",
    "\uc218\ub9ac",
    "\uc11c\ube44\uc2a4",
    "\ud544\ud130",
    "\uc138\uc81c",
    "\ub0c4\uc0c8",
    "\ub514\uc790\uc778",
}

_ACTION_TOKENS = {
    "wash",
    "rinse",
    "clean",
    "dry",
    "install",
    "repair",
    "replace",
    "maintain",
    "remove",
    "\uc138\ucc99",
    "\uad00\ub9ac",
    "\uccad\uc18c",
    "\uad50\uccb4",
    "\uc124\uce58",
    "\uc218\ub9ac",
    "\uc0ac\uc6a9",
    "\uad6c\ub9e4",
}

_ISSUE_TOKENS = {
    "issue",
    "problem",
    "noise",
    "dirty",
    "smell",
    "error",
    "leak",
    "broken",
    "expensive",
    "burden",
    "delay",
    "failure",
    "hot",
    "\ubb38\uc81c",
    "\uace0\uc7a5",
    "\uc18c\uc74c",
    "\ubd88\ub9cc",
    "\ubd80\ub2f4",
    "\uc9c0\uc5f0",
    "\uc624\ub958",
    "\ub204\uc218",
}

_DOMAIN_WORDS = _PRODUCT_TOKENS | _ACTION_TOKENS | _ISSUE_TOKENS

_WEAK_DESCRIPTOR_TOKENS = {
    "clean",
    "dirty",
    "hot",
    "great",
    "good",
    "nice",
    "bad",
    "\uc88b\ub2e4",
    "\ub098\uc058\ub2e4",
}

_ALLOWED_PRODUCT_COMPOUNDS = {
    ("dishwasher", "detergent"),
    ("powder", "detergent"),
    ("liquid", "detergent"),
    ("laundry", "detergent"),
}
_ORDERED_ISSUE_TOKENS = {"hot", "dirty", "expensive", "broken"}


def _validate_base_df(base_df: pd.DataFrame) -> None:
    if base_df is None or base_df.empty:
        raise ValueError("base_df is empty.")
    missing = [col for col in _REQUIRED_BASE_COLUMNS if col not in base_df.columns]
    if missing:
        raise ValueError(f"base_df missing required columns: {missing}")


def _normalize_token_form(token: str) -> str:
    normalized = token.strip("._-").lower()
    if not normalized:
        return ""
    mapped = _TOKEN_NORMALIZATION_MAP.get(normalized, normalized)
    if mapped.endswith("s") and len(mapped) >= 4:
        singular = mapped[:-1]
        if singular in _DOMAIN_WORDS:
            mapped = singular
    return mapped


def _is_domain_token(token: str) -> bool:
    return token in _DOMAIN_WORDS


def _is_meaningless_token(token: str) -> bool:
    if not token:
        return True
    if token in _NOISE_TOKENS:
        return True
    if token.startswith(("http", "www")):
        return True
    if token in _STOPWORDS_KO or token in _STOPWORDS_EN:
        return True
    if len(token) < 2:
        return True
    if token.isdigit():
        return True
    if _REPEAT_NOISE_PATTERN.search(token):
        return True
    if not _is_domain_token(token):
        return True
    return False


def _tokenize_text(text: Any) -> list[str]:
    value = "" if text is None else str(text)
    tokens: list[str] = []
    for matched in _TOKEN_PATTERN.findall(value):
        token = _normalize_token_form(matched)
        if not token or _is_meaningless_token(token):
            continue
        tokens.append(token)
    return tokens


def build_comment_keyword_map(base_df: pd.DataFrame) -> pd.DataFrame:
    _validate_base_df(base_df)
    rows: list[dict[str, str]] = []
    for row in base_df[["comment_id", "text_clean"]].itertuples(index=False):
        comment_id = str(row.comment_id)
        unique_tokens = list(dict.fromkeys(_tokenize_text(row.text_clean)))
        for token in unique_tokens:
            rows.append({"comment_id": comment_id, "keyword": token})
    if not rows:
        return pd.DataFrame(columns=["comment_id", "keyword"])
    out = pd.DataFrame(rows)
    return out.sort_values(["comment_id", "keyword"]).reset_index(drop=True)


def extract_keywords(base_df: pd.DataFrame) -> pd.DataFrame:
    _validate_base_df(base_df)
    token_count: Counter[str] = Counter()
    comment_coverage: Counter[str] = Counter()
    total_comments = int(len(base_df))

    for text in base_df["text_clean"].tolist():
        tokens = _tokenize_text(text)
        if not tokens:
            continue
        token_count.update(tokens)
        comment_coverage.update(set(tokens))

    if not token_count:
        return pd.DataFrame(columns=["keyword", "count", "coverage_pct"])

    rows = [
        {
            "keyword": keyword,
            "count": int(count),
            "coverage_pct": round((comment_coverage[keyword] / max(total_comments, 1)) * 100, 3),
        }
        for keyword, count in token_count.items()
    ]
    out = pd.DataFrame(rows)
    return out.sort_values(["count", "coverage_pct", "keyword"], ascending=[False, False, True]).reset_index(drop=True)


def _is_weak_phrase(window: list[str]) -> bool:
    if not window:
        return True
    if len(set(window)) < len(window):
        return True
    if len(window) == 2 and tuple(window) in _ALLOWED_PRODUCT_COMPOUNDS:
        return False
    if all(token in _WEAK_DESCRIPTOR_TOKENS for token in window):
        return True
    has_product = any(token in _PRODUCT_TOKENS for token in window)
    has_action = any(token in _ACTION_TOKENS for token in window)
    has_issue = any(token in _ISSUE_TOKENS for token in window)
    if not has_product or not (has_action or has_issue):
        return True
    if len(window) == 2:
        first, second = window
        if first in _PRODUCT_TOKENS and second in _PRODUCT_TOKENS:
            return (first, second) not in _ALLOWED_PRODUCT_COMPOUNDS
        if second == "water" and first in {"wash", "clean", "rinse"}:
            return True
        if first in _PRODUCT_TOKENS and second in _ORDERED_ISSUE_TOKENS:
            return True
        if first in _PRODUCT_TOKENS and second in _ACTION_TOKENS:
            return True
    if len(window) == 3 and not has_action:
        return True
    return False


def _iter_phrases(tokens: list[str]) -> list[str]:
    phrases: list[str] = []
    token_len = len(tokens)
    if token_len < 2:
        return phrases
    n = 2
    for idx in range(0, token_len - n + 1):
        window = tokens[idx : idx + n]
        if _is_weak_phrase(window):
            continue
        phrases.append(" ".join(window))
    return phrases


def extract_phrases(base_df: pd.DataFrame) -> pd.DataFrame:
    _validate_base_df(base_df)
    phrase_count: Counter[str] = Counter()
    for text in base_df["text_clean"].tolist():
        tokens = _tokenize_text(text)
        if not tokens:
            continue
        phrase_count.update(_iter_phrases(tokens))

    if not phrase_count:
        return pd.DataFrame(columns=["phrase", "count"])

    rows = [
        {"phrase": phrase, "count": int(count)}
        for phrase, count in phrase_count.items()
        if count >= _MIN_PHRASE_COUNT
    ]
    if not rows:
        return pd.DataFrame(columns=["phrase", "count"])
    out = pd.DataFrame(rows)
    return out.sort_values(["count", "phrase"], ascending=[False, True]).reset_index(drop=True)
