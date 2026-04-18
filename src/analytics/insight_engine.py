"""Insight engine that aggregates nlp_analyzer results into actionable insights.

Consumes per-comment NLP analysis fields (nlp_topics, nlp_topic_sentiments,
nlp_keywords, nlp_summary, etc.) and produces aggregate insight DataFrames
for the dashboard layer.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_INQUIRY_NOISE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"https?://",
        r"\baffiliate\b",
        r"\bsponsor(?:ed|ing)?\b",
        r"\bpromo\s*code\b",
        r"\bdiscount\s*code\b",
        r"\bcoupon\b",
        r"\bcoupang\s*partners?\b",
        r"\bjoint\s*purchase\b",
        r"\binvestment\b",
        r"\bstock\w*\b",
        r"\brestock\w*\b",
        r"\bin\s+stock\b",
        r"\bout\s+of\s+stock\b",
        r"\bsubscribe\s*event\b",
    ]
)
_REAL_INQUIRY_MARKERS = (
    "어떻게",
    "어디",
    "왜",
    "가능",
    "문의",
    "비교",
    "추천",
    "설치",
    "고장",
    "수리",
    "환불",
    "교체",
    "어떤",
    "얼마",
    "몇",
    "되나요",
    "인가요",
    "까요",
    "how",
    "why",
    "where",
    "which",
    "can i",
    "should i",
    "does it",
    "is it",
    "repair",
    "install",
    "price",
    "cost",
)
_REAL_DECISION_USAGE_MARKERS = (
    "고장",
    "수리",
    "설치",
    "교체",
    "환불",
    "비용",
    "가격",
    "비교",
    "추천",
    "사용",
    "as",
    "a/s",
    "repair",
    "install",
    "replacement",
    "warranty",
    "price",
    "cost",
)
_NON_INQUIRY_NARRATIVE_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bi used to\b",
        r"\bi can still remember\b",
        r"\bi noticed years ago\b",
        r"\bi'd like to share\b",
        r"\bthought i'd share\b",
        r"\bgreat video\b",
        r"\bthank(s| you)\b",
        r"\btip\b",
        r"\bstock\w*\b",
        r"\brestock\w*\b",
        r"\bin\s+stock\b",
        r"\bout\s+of\s+stock\b",
    ]
)


def build_topic_insight_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nlp_topics × nlp_topic_sentiments into a topic-level insight table.

    Returns DataFrame with columns:
        topic, positive_count, negative_count, neutral_count, total,
        dominant_sentiment, negative_rate
    """
    if analysis_df.empty or "nlp_topics" not in analysis_df.columns:
        return pd.DataFrame(columns=["topic", "positive_count", "negative_count", "neutral_count", "total", "dominant_sentiment", "negative_rate"])

    rows: list[dict[str, Any]] = []
    for _, row in analysis_df.iterrows():
        topics = _parse_list(row.get("nlp_topics"))
        topic_sentiments = _parse_dict(row.get("nlp_topic_sentiments"))
        for topic in topics:
            sentiment = topic_sentiments.get(topic, "neutral")
            rows.append({"topic": topic, "sentiment": sentiment})

    if not rows:
        return pd.DataFrame(columns=["topic", "positive_count", "negative_count", "neutral_count", "total", "dominant_sentiment", "negative_rate"])

    detail = pd.DataFrame(rows)
    pivot = detail.groupby("topic")["sentiment"].value_counts().unstack(fill_value=0).reset_index()
    for col in ["positive", "negative", "neutral"]:
        if col not in pivot.columns:
            pivot[col] = 0
    pivot = pivot.rename(columns={"positive": "positive_count", "negative": "negative_count", "neutral": "neutral_count"})
    pivot["total"] = pivot["positive_count"] + pivot["negative_count"] + pivot["neutral_count"]
    pivot["dominant_sentiment"] = pivot[["positive_count", "negative_count", "neutral_count"]].idxmax(axis=1).str.replace("_count", "")
    pivot["negative_rate"] = (pivot["negative_count"] / pivot["total"]).round(4)
    return pivot.sort_values("total", ascending=False).reset_index(drop=True)


def build_keyword_insight_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nlp_keywords across all comments into keyword frequency + sentiment breakdown.

    Returns DataFrame with columns:
        keyword, count, positive_count, negative_count, neutral_count, dominant_sentiment
    """
    if analysis_df.empty or "nlp_keywords" not in analysis_df.columns:
        return pd.DataFrame(columns=["keyword", "count", "positive_count", "negative_count", "neutral_count", "dominant_sentiment"])

    keyword_sentiments: dict[str, Counter] = {}
    for _, row in analysis_df.iterrows():
        keywords = _parse_list(row.get("nlp_keywords"))
        label = row.get("nlp_label") or row.get("sentiment_label") or "neutral"
        for kw in keywords:
            if kw not in keyword_sentiments:
                keyword_sentiments[kw] = Counter()
            keyword_sentiments[kw][label] += 1

    if not keyword_sentiments:
        return pd.DataFrame(columns=["keyword", "count", "positive_count", "negative_count", "neutral_count", "dominant_sentiment"])

    rows = []
    for kw, counts in keyword_sentiments.items():
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0) + counts.get("trash", 0) + counts.get("undecidable", 0)
        total = pos + neg + neu
        dominant = max(["positive", "negative", "neutral"], key=lambda s: {"positive": pos, "negative": neg, "neutral": neu}[s])
        rows.append({
            "keyword": kw,
            "count": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "dominant_sentiment": dominant,
        })

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def build_inquiry_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Identify comments flagged as inquiries by nlp_analyzer.

    Returns DataFrame with columns:
        comment_id, text, nlp_summary, nlp_topics, is_rhetorical, product_mentions
    """
    if analysis_df.empty or "nlp_is_inquiry" not in analysis_df.columns:
        return pd.DataFrame(columns=["comment_id", "text", "nlp_summary", "nlp_topics", "is_rhetorical", "product_mentions"])

    inquiry_mask = analysis_df["nlp_is_inquiry"].map(_is_truthy)
    if "inquiry_flag" in analysis_df.columns:
        inquiry_mask = inquiry_mask | analysis_df["inquiry_flag"].map(_is_truthy)
    inquiries = analysis_df[inquiry_mask].copy()
    if inquiries.empty:
        return pd.DataFrame(columns=["comment_id", "text", "nlp_summary", "nlp_topics", "is_rhetorical", "product_mentions"])

    text_col = "cleaned_text" if "cleaned_text" in inquiries.columns else "text_display"
    text_series = inquiries.get(text_col, "").fillna("").astype(str)
    inquiries = inquiries[text_series.map(_is_real_customer_inquiry)].copy()
    if inquiries.empty:
        return pd.DataFrame(columns=["comment_id", "text", "nlp_summary", "nlp_topics", "is_rhetorical", "product_mentions"])

    result = inquiries[["comment_id"]].copy()
    result["text"] = inquiries.get(text_col, "").fillna("")
    result["nlp_summary"] = inquiries.get("nlp_summary", "").fillna("")
    result["nlp_topics"] = inquiries["nlp_topics"].apply(_serialize_list)
    result["is_rhetorical"] = inquiries.get("nlp_is_rhetorical", False).map(_is_truthy)
    result["product_mentions"] = inquiries.get("nlp_product_mentions", "").apply(_serialize_list)
    return result.reset_index(drop=True)


def build_product_mention_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate nlp_product_mentions into product mention frequency table.

    Returns DataFrame with columns:
        product, mention_count, positive_count, negative_count, neutral_count, negative_rate
    """
    if analysis_df.empty or "nlp_product_mentions" not in analysis_df.columns:
        return pd.DataFrame(columns=["product", "mention_count", "positive_count", "negative_count", "neutral_count", "negative_rate"])

    product_sentiments: dict[str, Counter] = {}
    for _, row in analysis_df.iterrows():
        products = _parse_list(row.get("nlp_product_mentions"))
        label = row.get("nlp_label") or row.get("sentiment_label") or "neutral"
        for prod in products:
            if prod not in product_sentiments:
                product_sentiments[prod] = Counter()
            product_sentiments[prod][label] += 1

    if not product_sentiments:
        return pd.DataFrame(columns=["product", "mention_count", "positive_count", "negative_count", "neutral_count", "negative_rate"])

    rows = []
    for prod, counts in product_sentiments.items():
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0) + counts.get("trash", 0) + counts.get("undecidable", 0)
        total = pos + neg + neu
        rows.append({
            "product": prod,
            "mention_count": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "negative_rate": round(neg / total, 4) if total else 0.0,
        })

    return pd.DataFrame(rows).sort_values("mention_count", ascending=False).reset_index(drop=True)


def build_nlp_sentiment_summary(analysis_df: pd.DataFrame) -> pd.DataFrame:
    """Summary of nlp_analyzer sentiment distribution (nlp_label counts).

    Returns DataFrame with columns: label, count, share
    """
    if analysis_df.empty or "nlp_label" not in analysis_df.columns:
        return pd.DataFrame(columns=["label", "count", "share"])

    valid = analysis_df[analysis_df["nlp_label"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=["label", "count", "share"])

    counts = valid["nlp_label"].value_counts().reset_index()
    counts.columns = ["label", "count"]
    counts["share"] = (counts["count"] / counts["count"].sum()).round(4)
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_list(value: Any) -> list[str]:
    """Safely parse a value that may be a list, JSON string, or empty."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, TypeError):
            return []
    return []


def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return False
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y", "t"}
    return False


def _is_real_customer_inquiry(text: str) -> bool:
    value = str(text or "").strip()
    if not value:
        return False
    lowered = value.lower()
    if any(pattern.search(lowered) for pattern in _INQUIRY_NOISE_PATTERNS):
        return False
    has_question_mark = "?" in value
    has_inquiry_marker = any(marker in lowered for marker in _REAL_INQUIRY_MARKERS)
    if not (has_question_mark or has_inquiry_marker):
        return False
    if any(pattern.search(lowered) for pattern in _NON_INQUIRY_NARRATIVE_PATTERNS) and not has_question_mark:
        return False
    has_decision_usage_marker = any(marker in lowered for marker in _REAL_DECISION_USAGE_MARKERS)
    if not has_decision_usage_marker and len(value) > 140:
        return False
    if len(value) > 520 and not has_question_mark:
        return False
    return True


def _parse_dict(value: Any) -> dict[str, str]:
    """Safely parse a value that may be a dict, JSON string, or empty."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def _serialize_list(value: Any) -> str:
    """Convert list to comma-separated string for display."""
    items = _parse_list(value)
    return ", ".join(str(item) for item in items)
