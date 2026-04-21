"""Keyword co-occurrence builder for TM pipeline."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import math
import re
from typing import Any

import pandas as pd

from src.text_mining.keyword_extractor import build_comment_keyword_map

_WEAK_SENTIMENT_KEYWORDS = {
    "good",
    "great",
    "nice",
    "bad",
    "awesome",
    "terrible",
    "okay",
    "ok",
    "best",
    "worst",
    "\uc88b\ub2e4",
    "\ub098\uc058\ub2e4",
}
_TOKEN_VALID_PATTERN = re.compile(r"^[a-z0-9_'\-\uac00-\ud7a3]{2,}$")

_GENERIC_TOKENS = {"water", "clean", "wash", "dish"}

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
_CAUSE_TOKENS = {
    "price",
    "cost",
    "expensive",
    "delay",
    "installation",
    "space",
    "capacity",
    "detergent",
    "filter",
    "\uac00\uaca9",
    "\ubd80\ub2f4",
    "\uc9c0\uc5f0",
    "\uc124\uce58",
    "\uacf5\uac04",
}
_SYMPTOM_TOKENS = {
    "problem",
    "issue",
    "dirty",
    "noise",
    "smell",
    "error",
    "leak",
    "broken",
    "failure",
    "hot",
    "\ubb38\uc81c",
    "\uace0\uc7a5",
    "\uc18c\uc74c",
    "\ub204\uc218",
    "\uc624\ub958",
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_valid_keyword(token: str) -> bool:
    if not token:
        return False
    if token in _WEAK_SENTIMENT_KEYWORDS:
        return False
    if not _TOKEN_VALID_PATTERN.match(token):
        return False
    return True


def _is_trivial_pair(keyword_1: str, keyword_2: str) -> bool:
    if keyword_1 == keyword_2:
        return True
    if not _is_valid_keyword(keyword_1) or not _is_valid_keyword(keyword_2):
        return True
    return False


def _token_generic_penalty(token: str) -> float:
    if token in _GENERIC_TOKENS:
        return 0.62
    return 1.0


def _token_roles(token: str) -> set[str]:
    roles: set[str] = set()
    if token in _PRODUCT_TOKENS:
        roles.add("product")
    if token in _ACTION_TOKENS:
        roles.add("action")
    if token in _ISSUE_TOKENS:
        roles.add("issue")
    if token in _CAUSE_TOKENS:
        roles.add("cause")
    if token in _SYMPTOM_TOKENS:
        roles.add("symptom")
    return roles


def _relation_weight(keyword_1: str, keyword_2: str) -> float:
    roles_1 = _token_roles(keyword_1)
    roles_2 = _token_roles(keyword_2)
    generic_pair = keyword_1 in _GENERIC_TOKENS and keyword_2 in _GENERIC_TOKENS
    if generic_pair:
        return 0.55
    if ("product" in roles_1 and "issue" in roles_2) or ("issue" in roles_1 and "product" in roles_2):
        return 1.35
    if ("action" in roles_1 and "issue" in roles_2) or ("issue" in roles_1 and "action" in roles_2):
        return 1.28
    if ("cause" in roles_1 and "symptom" in roles_2) or ("symptom" in roles_1 and "cause" in roles_2):
        return 1.22
    if "product" in roles_1 and "product" in roles_2:
        return 0.78
    if "action" in roles_1 and "action" in roles_2:
        return 0.82
    if "issue" in roles_1 and "issue" in roles_2:
        return 0.95
    return 1.0


def _build_pair_counts(comment_keyword_map: pd.DataFrame) -> tuple[Counter[tuple[str, str]], Counter[str], int]:
    pair_counter: Counter[tuple[str, str]] = Counter()
    keyword_comment_counter: Counter[str] = Counter()
    total_comment_count = 0

    grouped = comment_keyword_map.groupby("comment_id", sort=False)["keyword"]
    for _, keyword_series in grouped:
        unique_keywords = sorted(
            {
                _safe_text(keyword).lower()
                for keyword in keyword_series.tolist()
                if _safe_text(keyword)
            }
        )
        unique_keywords = [token for token in unique_keywords if _is_valid_keyword(token)]
        if not unique_keywords:
            continue
        total_comment_count += 1
        keyword_comment_counter.update(unique_keywords)
        if len(unique_keywords) < 2:
            continue
        for keyword_1, keyword_2 in combinations(unique_keywords, 2):
            if _is_trivial_pair(keyword_1, keyword_2):
                continue
            pair_counter[(keyword_1, keyword_2)] += 1

    return pair_counter, keyword_comment_counter, total_comment_count


def build_keyword_cooccurrence(base_df: pd.DataFrame, min_pair_count: int = 3) -> pd.DataFrame:
    """
    Build keyword co-occurrence pairs on same-comment basis.

    Output columns:
    - keyword_1
    - keyword_2
    - pair_count
    - pmi
    - lift
    - rank_score
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame(columns=["keyword_1", "keyword_2", "pair_count", "pmi", "lift", "rank_score"])

    comment_keyword_map = build_comment_keyword_map(base_df)
    if comment_keyword_map.empty:
        return pd.DataFrame(columns=["keyword_1", "keyword_2", "pair_count", "pmi", "lift", "rank_score"])

    required_cols = {"comment_id", "keyword"}
    if not required_cols.issubset(comment_keyword_map.columns):
        return pd.DataFrame(columns=["keyword_1", "keyword_2", "pair_count", "pmi", "lift", "rank_score"])

    pair_counter, keyword_comment_counter, total_comment_count = _build_pair_counts(comment_keyword_map)
    if not pair_counter or total_comment_count <= 0:
        return pd.DataFrame(columns=["keyword_1", "keyword_2", "pair_count", "pmi", "lift", "rank_score"])

    rows: list[dict[str, float | int | str]] = []
    for (keyword_1, keyword_2), count in pair_counter.items():
        pair_count = int(count)
        if pair_count < int(min_pair_count):
            continue

        keyword_1_count = int(keyword_comment_counter.get(keyword_1, 0))
        keyword_2_count = int(keyword_comment_counter.get(keyword_2, 0))
        if keyword_1_count <= 0 or keyword_2_count <= 0:
            continue

        pair_prob = pair_count / total_comment_count
        p1 = keyword_1_count / total_comment_count
        p2 = keyword_2_count / total_comment_count
        if pair_prob <= 0 or p1 <= 0 or p2 <= 0:
            continue

        pmi_raw = math.log2(pair_prob / (p1 * p2))
        lift = (pair_count * total_comment_count) / max(keyword_1_count * keyword_2_count, 1)
        generic_penalty = _token_generic_penalty(keyword_1) * _token_generic_penalty(keyword_2)
        relation_weight = _relation_weight(keyword_1, keyword_2)
        rank_score = (
            (max(pmi_raw, 0.0) + 0.2 * math.log1p(max(lift, 1.0)))
            * math.log1p(pair_count)
            * generic_penalty
            * relation_weight
        )

        rows.append(
            {
                "keyword_1": keyword_1,
                "keyword_2": keyword_2,
                "pair_count": pair_count,
                "pmi": round(float(pmi_raw), 6),
                "lift": round(float(lift), 6),
                "rank_score": round(float(rank_score), 6),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["keyword_1", "keyword_2", "pair_count", "pmi", "lift", "rank_score"])

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["rank_score", "pair_count", "keyword_1", "keyword_2"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

