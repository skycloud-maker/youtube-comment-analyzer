from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.comment_analysis import analyze_comment_with_context

CONTRAST_SPLIT_RE = re.compile(r"\b(?:but|however|though|although|while|yet|except)\b|\ud558\uc9c0\ub9cc|\uadf8\ub7f0\ub370|\ub2e4\ub9cc|\ubc18\uba74|\uadfc\ub370|\uadf8\ub7ec\ub098", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\\s+|(?<=\u3002)|\\n+")
MODEL_RE = re.compile(r"\b[A-Z0-9]{4,}(?:[-_][A-Z0-9]+)*\b")
INQUIRY_PATTERNS = {
    "HOW_TO": ["how to", "how do i", "\uc0ac\uc6a9\ubc95", "\uc5b4\ub5bb\uac8c", "\uc124\uc815", "\ubc29\ubc95"],
    "OFFICIAL_SERVICE": ["official service", "service center", "\uc11c\ube44\uc2a4\uc13c\ud130", "\uacf5\uc2dd \uc11c\ube44\uc2a4", "as \uc13c\ud130", "a/s"],
    "RISK_CONCERN": ["safe", "risk", "\uac71\uc815", "\uc704\ud5d8", "\uad1c\ucc2e\uc744\uae4c\uc694", "\uc2e4\uc218"],
    "WARRANTY": ["warranty", "\ubcf4\uc99d", "\ubb34\uc0c1", "\uc720\uc0c1"],
    "COST": ["cost", "price", "\ube44\uc6a9", "\uac00\uaca9", "\uc5bc\ub9c8"],
    "ELIGIBILITY": ["eligible", "\uac00\ub2a5", "\ub300\uc0c1", "\ud574\ub2f9", "\uad50\ud658 \uac00\ub2a5"],
}
SCENE_RULES = [
    ("S8 Maintain", ["repair", "service", "technician", "tech", "warranty", "a/s", "as", "\uc218\ub9ac", "\uc11c\ube44\uc2a4", "\uae30\uc0ac", "\uc13c\ud130", "\ubcf4\uc99d", "\uc810\uac80"]),
    ("S5 Deliver", ["delivery", "install", "installation", "shipping", "\ubc30\uc1a1", "\uc124\uce58", "\ud30c\uc190", "\ubc29\ubb38", "\uc9c0\uc5f0"]),
    ("S6 On-board", ["setup", "set up", "register", "registration", "app", "thinq", "wifi", "\uc5f0\ub3d9", "\ub4f1\ub85d", "\ub85c\uadf8\uc778", "\ucd08\uae30 \uc124\uc815", "\uc571"]),
    ("S4 Purchase", ["bought", "purchase", "purchased", "order", "ordered", "\uad6c\ub9e4", "\uc8fc\ubb38", "\uacb0\uc81c"]),
    ("S9 Replace", ["replace", "replaced", "replacement", "\uad50\uccb4", "\ubc14\uafe8", "\uc0c8\ub85c \uc0c0"]),
    ("S2 Explore", ["compare", "comparison", "recommend", "review", "difference", "\ucd94\ucc9c", "\ube44\uad50", "\ud6c4\uae30", "\ucc28\uc774"]),
    ("S1 Aware", ["ad", "advertisement", "thumbnail", "\uad11\uace0", "\uc378\ub124\uc77c"]),
]
ASPECT_RULES = [
    ("Noise/Vibration", ["\uc18c\uc74c", "\uc2dc\ub044", "\uc870\uc6a9", "\uc6c5\uc6c5", "\uc9c4\ub3d9", "\ub5a8\ub9bc", "\uc18c\ub9ac", "\ucff5\ucff5", "\ub4dc\ub974\ub975", "noise", "vibration", "buzz"]),
    ("Performance", ["\uc138\ud0c1\ub825", "\ub54c", "\uc624\uc5fc", "\ud5f9\uad7c", "\uac74\uc870", "\ub9d0\ub9bc", "\uc138\ucc99", "\uc138\ucc99\ub825", "\uc794\uc5ec\ubb3c", "\ud761\uc785", "\ud761\uc785\ub825", "\ub0c9\uac01", "\uc628\ub3c4", "\uc2e0\uc120", "performance", "cleaning", "cooling", "temperature"]),
    ("Convenience/Usability", ["\ud3b8\ud574", "\ubd88\ud3b8", "\uc870\uc791", "\ubc84\ud2bc", "\uba54\ub274", "ui", "\uac04\ud3b8", "\ubcf5\uc7a1", "\uc124\uc815", "convenient", "inconvenient", "easy to use"]),
    ("Reliability/Breakdown", ["\uace0\uc7a5", "\ubd88\ub7c9", "\uba39\ud1b5", "\uc548 \ucf1c", "\uc791\ub3d9 \uc548", "\uc5d0\ub7ec", "\uc5d0\ub7ec\ucf54\ub4dc", "\uc218\uba85", "\ub0b4\uad6c", "broken", "broke", "doesn't work", "error", "failed"]),
    ("Delivery/Installation", ["\ubc30\uc1a1", "\uc124\uce58", "\uc124\uce58\uae30\uc0ac", "\uc9c0\uc5f0", "\ud30c\uc190", "\uc2dc\uac04\uc57d\uc18d", "delivery", "installation", "shipping"]),
    ("App/Connectivity", ["\uc571", "thinq", "\uc640\uc774\ud30c\uc774", "\uc5f0\ub3d9", "\uc5f0\uacb0", "\ub4f1\ub85d", "\ub85c\uadf8\uc778", "\uc5c5\ub370\uc774\ud2b8", "app", "wifi", "connect", "login"]),
    ("Maintenance/Cleaning", ["\uad00\ub9ac", "\uccad\uc18c", "\uc138\ucc99", "\ud544\ud130", "\ud544\ud130 \uad50\uccb4", "\ub9ac\uc14b", "\uba3c\uc9c0\ud1b5", "\ubb3c\ud1b5", "\ud0c8\ucde8", "clean", "cleaning", "maintenance", "filter"]),
    ("A/S & Support", ["as", "a/s", "\uc11c\ube44\uc2a4\uc13c\ud130", "\uc13c\ud130", "\uc218\ub9ac", "\uc811\uc218", "\uc0c1\ub2f4", "\uc751\ub300", "\ucc98\ub9ac\uc9c0\uc5f0", "\ubd88\uce5c\uc808", "\ubcf4\uc99d", "\uc218\ub9ac\ube44", "service", "support", "warranty", "repair"]),
    ("Price/Value", ["\uac00\uaca9", "\ube44\uc2f8", "\uac00\uc131\ube44", "\ud61c\ud0dd", "\ud560\uc778", "\ube44\uc6a9", "\uad6c\ub3c5\ub8cc", "price", "expensive", "value", "cost"]),
    ("Design/Build", ["\ub514\uc790\uc778", "\ub9c8\uac10", "build", "design", "overdesign", "overdesigned", "sku", "simple", "\ub2e8\uc21c", "\uacfc\uc124\uacc4"]),
    ("Safety", ["\uc548\uc804", "\uc704\ud5d8", "safe", "safety", "hazard"]),
    ("Energy/Consumption", ["\uc804\uae30\uc138", "\uc804\ub825", "\uc18c\ube44\uc804\ub825", "\ud6a8\uc728", "\uc5d0\ub108\uc9c0", "energy", "power"]),
    ("Smell/Odor", ["\ub0c4\uc0c8", "\uc545\ucde8", "\ucff0\ucff0", "odor", "smell"]),
    ("Leakage/Water", ["\ub204\uc218", "\ubb3c\uc0d8", "\uc0c8\uc694", "\ubc30\uc218", "\uc5ed\ub958", "leak", "water"]),
    ("Fit/Space", ["\uacf5\uac04", "\ud06c\uae30", "\uc790\ub9ac", "fit", "space"]),
    ("Documentation/Instructions", ["\uc124\uba85\uc11c", "\ub9e4\ub274\uc5bc", "\uac00\uc774\ub4dc", "manual", "instruction"]),
]
PRODUCT_MAP = {
    "\ub0c9\uc7a5\uace0": "Refrigerator",
    "\uc138\ud0c1\uae30": "Washer",
    "\uac74\uc870\uae30": "Dryer",
    "\uc2dd\uae30\uc138\ucc99\uae30": "Dishwasher",
    "\uccad\uc18c\uae30": "Vacuum",
    "\uc624\ube10": "Oven",
    "\uc815\uc218\uae30": "Water Purifier",
    "\uc778\ub355\uc158": "Induction",
    "\uc640\uc778\uc140\ub7ec": "Wine Cellar",
    "\ucef5\uc138\ucc99\uae30": "Cup Washer",
    "\uc2e4\ub0b4\uc2dd\ubb3c\uc7ac\ubc30\uae30(\ud2d4\uc6b4)": "Indoor Plant Grower (Tiiun)",
}


SUBTYPE_RULES = {
    "Refrigerator": {
        "???": ["???", "side-by-side", "side by side"],
        "???????": ["???", "???", "top freezer", "bottom freezer", "top-fridge-bottom-freezer"],
        "?????": ["?????", "kimchi fridge"],
        "???": ["???", "built-in", "built in"],
        "??(???)": ["?????", "mini fridge", "personal fridge"],
        "????": ["????", "wine cellar", "wine fridge"],
        "????": ["showcase"],
    },
    "Washer": {
        "??": ["front loader", "front-load", "drum washer", "??"],
        "???": ["top loader", "top-load", "???"],
        "????": ["washtower", "wash tower", "????"],
        "??": ["combo", "all-in-one", "all in one", "??"],
        "????": ["mini washer", "????"],
    },
    "Dryer": {
        "??": ["combo", "all-in-one", "all in one", "washer dryer combo", "??"],
        "????": ["heat pump", "????"],
        "???": ["condenser", "???"],
        "??": ["vented", "??"],
    },
    "Dishwasher": {
        "6??": ["6 place", "6-place", "6??"],
        "12~14??": ["12 place", "14 place", "12-14", "12~14", "12??", "14??"],
        "???": ["built-in", "built in", "???"],
        "?????": ["freestanding", "?????"],
    },
    "Vacuum": {
        "??": ["robot", "robot vacuum", "?????"],
        "??": ["stick", "cordless", "??"],
        "???(W&D)": ["wet&dry", "wet and dry", "???", "w&d"],
        "??": ["handheld", "??"],
    },
    "Cup Washer": {
        "??(????)": ["countertop", "compact", "????", "??"],
        "???": ["built-in", "built in", "???"],
        "???(??????)": ["commercial", "cafe", "office", "???", "??", "???"],
    },
    "Indoor Plant Grower (Tiiun)": {
        "??(???)": ["compact", "personal", "??", "???"],
        "????": ["stand", "stand-type", "????"],
    },
}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "<na>"} else text


def _split_opinion_text(text: str) -> list[str]:
    source = _safe_text(text)
    if not source:
        return []
    segments = [source]
    if CONTRAST_SPLIT_RE.search(source):
        segments = [part.strip(" ,.;:-\n\t") for part in re.split(CONTRAST_SPLIT_RE, source) if _safe_text(part)] or [source]
    clauses: list[str] = []
    for segment in segments:
        subparts = [part.strip(" ,.;:-\n\t") for part in SENTENCE_SPLIT_RE.split(segment) if _safe_text(part)]
        clauses.extend(subparts or [segment])
    seen: set[str] = set()
    deduped: list[str] = []
    for clause in clauses:
        normalized = clause.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped or [source]


def _derive_scene(text: str, fallback_scene: str = "") -> str:
    lowered = _safe_text(text).lower()
    for scene, markers in SCENE_RULES:
        if any(marker in lowered for marker in markers):
            return scene
    if fallback_scene:
        return fallback_scene
    return "S7 Use" if lowered else "S10 Other"


def _derive_aspect(text: str) -> tuple[str, str]:
    lowered = _safe_text(text).lower()
    best_key = "General"
    best_hits = 0
    best_marker = ""
    for aspect, markers in ASPECT_RULES:
        hits = sum(1 for marker in markers if marker in lowered)
        if hits > best_hits:
            best_hits = hits
            best_key = aspect
            best_marker = next((marker for marker in markers if marker in lowered), "")
    return best_key, best_marker


def _extract_inquiry(text: str) -> tuple[bool, list[str]]:
    lowered = _safe_text(text).lower()
    flags = [label for label, markers in INQUIRY_PATTERNS.items() if any(marker in lowered for marker in markers)]
    return bool(flags), flags or (["OTHER"] if "?" in lowered else [])


def _extract_model_tokens(source_blob: str) -> list[str]:
    candidates = []
    for match in MODEL_RE.finditer(_safe_text(source_blob).upper()):
        token = match.group(0)
        if any(ch.isdigit() for ch in token) and len(token) >= 5:
            candidates.append(token)
    return list(dict.fromkeys(candidates))


def _derive_product_subtype(category: str, source_blob: str) -> tuple[str, list[str]]:
    blob = _safe_text(source_blob).lower()
    for subtype, markers in SUBTYPE_RULES.get(category, {}).items():
        hits = [marker for marker in markers if marker.lower() in blob]
        if hits:
            return subtype, hits
    return "Unknown", []


def _extract_evidence_spans(clause_text: str, aspect_marker: str = "") -> list[str]:
    clause = _safe_text(clause_text)
    if not clause:
        return []
    spans: list[str] = [clause]
    lowered = clause.lower()
    if aspect_marker:
        marker = aspect_marker.lower()
        idx = lowered.find(marker)
        if idx >= 0:
            spans.append(clause[idx:idx + len(aspect_marker)])
    return list(dict.fromkeys(span for span in spans if _safe_text(span)))


def _product_fields(row: pd.Series, clause_text: str) -> dict[str, Any]:
    product_raw = _safe_text(row.get("product", ""))
    title_text = _safe_text(row.get("title", ""))
    source_blob = " ".join([clause_text, product_raw, title_text])
    category = PRODUCT_MAP.get(product_raw, product_raw or "Other")
    subtype, subtype_hits = _derive_product_subtype(category, source_blob)
    model_tokens = _extract_model_tokens(source_blob)
    model_mentioned = _safe_text(row.get("model_mentioned", "")) or (model_tokens[0] if model_tokens else "")
    evidence = []
    for token in [product_raw, title_text, *subtype_hits, *model_tokens[:1]]:
        token = _safe_text(token)
        if token and token not in evidence and token.lower() in source_blob.lower():
            evidence.append(token)
    return {
        "product_category": category or "Other",
        "product_subtype": _safe_text(row.get("product_subtype", "")) or subtype or "Unknown",
        "brand_mentioned": _safe_text(row.get("brand_label", "")),
        "model_mentioned": model_mentioned,
        "product_evidence_spans": evidence,
    }


def _strength_score(sentiment_code: str, signal_strength: str, text: str) -> float:
    lowered = _safe_text(text).lower()
    score = 8.0 if sentiment_code in {"positive", "negative"} and signal_strength == "strong" else 5.0 if sentiment_code in {"positive", "negative"} else 0.0
    score += 2.0 * sum(1 for token in ["\uc9c4\uc9dc", "\uc644\uc804", "\ub108\ubb34", "\ucd5c\uace0", "\ucd5c\uc545", "\uc808\ub300", "\ubb34\uc870\uac74", "always", "never", "worst", "best"] if token in lowered)
    return max(0.0, min(score, 10.0))


def _richness_score(text: str) -> float:
    lowered = _safe_text(text).lower()
    length = len(lowered)
    len_bucket = 0 if length < 20 else 1 if length < 60 else 2 if length < 140 else 3
    score = float(len_bucket)
    score += 2.0 * sum(1 for token in ["because", "so ", "\ub54c\ubb38", "\ud574\uc11c", "\uadf8\ub798\uc11c"] if token in lowered)
    score += 2.0 * sum(1 for token in ["when", "if ", "\ud560 \ub54c", "\ubaa8\ub4dc", "\uc124\uc815"] if token in lowered)
    score += 2.0 * sum(1 for token in ["month", "week", "year", "3\uac1c\uc6d4", "2\uc8fc", "\ud55c\ub2ec", "\ub9e4\ubc88"] if token in lowered)
    score += 1.0 * sum(1 for token in ["than", "compared", "\ubcf4\ub2e4", "\uc774\uc804", "\uc804 \ubaa8\ub378"] if token in lowered)
    return max(0.0, min(score, 10.0))


def _likes_score(likes_count: Any) -> float:
    try:
        likes = float(likes_count)
    except Exception:
        likes = 0.0
    return max(0.0, min(math.log10(1.0 + max(likes, 0.0)) * 4.0, 10.0))


def _quality_penalty(row: pd.Series) -> float:
    penalty = 0.0
    if bool(row.get("spam_like", False)):
        penalty += 7.0
    return penalty


def _compute_opinion_score(row: pd.Series, sentiment_code: str, clause_text: str, signal_strength: str) -> float:
    if sentiment_code not in {"positive", "negative"}:
        return 0.0
    strength = _strength_score(sentiment_code, signal_strength, clause_text)
    richness = _richness_score(clause_text)
    likes = _likes_score(row.get("like_count", 0))
    penalty = _quality_penalty(row)
    return (5.0 * strength) + (2.0 * richness) + (1.5 * likes) - penalty


def _masked_text(text: str, pii_raw_text: str) -> str:
    masked = _safe_text(text)
    pii_raw = _safe_text(pii_raw_text)
    if not masked or not pii_raw:
        return masked
    for chunk in [part.strip() for part in pii_raw.split(",") if part.strip()]:
        if len(chunk) >= 2:
            masked = masked.replace(chunk, "[PII]")
    return masked


def build_opinion_units(comments_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if comments_df.empty:
        return pd.DataFrame()

    for _, row in comments_df.iterrows():
        source_id = _safe_text(row.get("comment_id", ""))
        source_text = _safe_text(row.get("text_display", row.get("cleaned_text", "")))
        validity = _safe_text(row.get("comment_validity", "valid")) or "valid"
        parent_comment = _safe_text(row.get("parent_comment_text", ""))
        context_comments = [_safe_text(row.get("title", "")), _safe_text(row.get("product", ""))]
        clauses = _split_opinion_text(source_text) or [source_text]
        opinion_count = len([clause for clause in clauses if _safe_text(clause)]) or 1
        fallback_scene = _safe_text(row.get("topic_label", ""))

        for idx, clause in enumerate(clauses, start=1):
            clause_text = _safe_text(clause)
            if not clause_text:
                continue
            if len(clauses) == 1:
                sentiment_code = _safe_text(row.get("sentiment_label", "neutral")).lower() or "neutral"
                signal_strength = _safe_text(row.get("signal_strength", "weak")) or "weak"
                classification_type = _safe_text(row.get("classification_type", "informational")) or "informational"
                confidence_level = _safe_text(row.get("confidence_level", "low")) or "low"
                reason = _safe_text(row.get("classification_reason", ""))
                product_related = bool(row.get("product_related", False))
            else:
                result = analyze_comment_with_context(
                    clause_text,
                    validity=validity,
                    exclusion_reason=_safe_text(row.get("exclusion_reason", "")) or None,
                    parent_comment=parent_comment,
                    context_comments=context_comments,
                    is_reply=bool(row.get("is_reply", False)),
                )
                sentiment_code = _safe_text(result.sentiment_label or "neutral").lower() or "neutral"
                signal_strength = _safe_text(result.signal_strength or "weak") or "weak"
                classification_type = _safe_text(result.classification_type or "informational") or "informational"
                confidence_level = _safe_text(result.confidence_level or "low") or "low"
                reason = _safe_text(result.reason)
                product_related = bool(result.product_related)

            if validity != "valid":
                sentiment_code = "excluded"
                signal_strength = "weak"
                classification_type = "informational"
                confidence_level = "low"

            scene = _derive_scene(clause_text, fallback_scene=_safe_text(fallback_scene))
            aspect_key, aspect_marker = _derive_aspect(clause_text)
            inquiry_flag, inquiry_types = _extract_inquiry(clause_text)
            product_fields = _product_fields(row, clause_text)
            evidence_spans = _extract_evidence_spans(clause_text, aspect_marker=aspect_marker)
            pii_flag = "Y" if _safe_text(row.get("pii_flag", "N")).upper() == "Y" else "N"
            pii_raw = _safe_text(row.get("pii_raw_text", ""))
            opinion_score = _compute_opinion_score(row, sentiment_code, clause_text, signal_strength)

            rows.append({
                "opinion_id": f"{source_id}::u{idx}",
                "source_content_id": source_id,
                "source_type": "YT_COMMENT",
                "source_count": 1,
                "opinion_count": opinion_count,
                "duplicate_opinion_flag": bool(row.get("spam_like", False)),
                "product_related": product_related,
                "opinion_text": clause_text,
                "cej_scene_code": scene,
                "sentiment_code": sentiment_code,
                "sentiment": sentiment_code.title(),
                "signal_strength": signal_strength,
                "classification_type": classification_type,
                "confidence_level": confidence_level,
                "aspect_key": aspect_key,
                "evidence_spans_json": json.dumps(evidence_spans, ensure_ascii=False),
                "inquiry_flag": inquiry_flag,
                "inquiry_type_json": json.dumps(inquiry_types, ensure_ascii=False),
                "product_category": product_fields["product_category"],
                "product_subtype": product_fields["product_subtype"],
                "brand_mentioned": product_fields["brand_mentioned"],
                "model_mentioned": product_fields["model_mentioned"],
                "product_evidence_spans_json": json.dumps(product_fields["product_evidence_spans"], ensure_ascii=False),
                "likes_count": pd.to_numeric(row.get("like_count", 0), errors="coerce"),
                "source_link": _safe_text(row.get("video_url", "")),
                "region": _safe_text(row.get("region", "")),
                "product": _safe_text(row.get("product", "")),
                "title": _safe_text(row.get("title", "")),
                "pii_flag": pii_flag,
                "pii_types": _safe_text(row.get("pii_types", "")),
                "pii_raw_text": pii_raw,
                "display_text": _masked_text(clause_text, pii_raw) if pii_flag == "Y" else clause_text,
                "original_text": clause_text,
                "masked_text": _masked_text(clause_text, pii_raw) if pii_flag == "Y" else "",
                "translated_text": _safe_text(row.get("translated_text", row.get("\ud55c\uad6d\uc5b4 \ubc88\uc5ed", ""))),
                "summary_text": _safe_text(row.get("summary", row.get("short_summary_ko", ""))),
                "classification_reason": reason,
                "opinion_score": opinion_score,
            })
    return pd.DataFrame(rows)


def _bundle_selection_reason(top_row: pd.Series, cluster_size: int) -> str:
    sentiment = _safe_text(top_row.get("sentiment_code", "neutral"))
    sentiment_label = "??" if sentiment == "negative" else "??" if sentiment == "positive" else "??"
    strength = _strength_score(_safe_text(top_row.get("sentiment_code", "")), _safe_text(top_row.get("signal_strength", "weak")), _safe_text(top_row.get("opinion_text", "")))
    richness = _richness_score(_safe_text(top_row.get("opinion_text", "")))
    likes = top_row.get("likes_count")
    likes_text = ""
    try:
        if pd.notna(likes) and float(likes) > 0:
            likes_text = f", ???={int(float(likes))}"
    except Exception:
        likes_text = ""
    return f"?? {sentiment_label} ??(????={strength:.0f}/10), ?? ?? ??(????={richness:.0f}/10){likes_text}, ?? ?? {cluster_size}?? ???? ?? ??? ??????."


def build_representative_bundles(opinion_units_df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    if opinion_units_df.empty:
        return pd.DataFrame()
    working = opinion_units_df[opinion_units_df["sentiment_code"].isin(["positive", "negative"])].copy()
    if working.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["sentiment_code", "product_category", "product_subtype", "brand_mentioned", "cej_scene_code", "aspect_key"]
    for _, group in working.groupby(group_cols, dropna=False):
        ordered = group.sort_values(["opinion_score", "likes_count"], ascending=[False, False], na_position="last")
        top_row = ordered.iloc[0]
        cluster_size = len(group)
        likes_sum = pd.to_numeric(group.get("likes_count", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
        size_bonus = max(0.0, min(math.log10(1.0 + cluster_size) * 5.0, 10.0))
        bundle_score = float(top_row.get("opinion_score", 0.0) or 0.0) + size_bonus + (0.5 * max(0.0, min(math.log10(1.0 + likes_sum) * 4.0, 10.0)))
        sample_comments = [
            {
                "opinion_id": _safe_text(sample.get("opinion_id", "")),
                "display_text": _safe_text(sample.get("display_text", "")),
                "translated_text": _safe_text(sample.get("translated_text", "")),
                "pii_flag": _safe_text(sample.get("pii_flag", "N")) or "N",
                "likes_count": None if pd.isna(sample.get("likes_count", pd.NA)) else int(float(sample.get("likes_count", 0) or 0)),
            }
            for _, sample in ordered.head(5).iterrows()
        ]
        evidence_spans = json.loads(_safe_text(top_row.get("evidence_spans_json", "[]")) or "[]")
        voc_items = [{
            "topic": _safe_text(top_row.get("aspect_key", "General")) or "General",
            "sentiment": _safe_text(top_row.get("sentiment", "Neutral")) or "Neutral",
            "insight_message": _safe_text(top_row.get("classification_reason", "")) or "\ub300\ud45c \uc758\uacac\uc758 \ud575\uc2ec \ud3ec\uc778\ud2b8\uc785\ub2c8\ub2e4.",
            "evidence_span": evidence_spans[0] if evidence_spans else _safe_text(top_row.get("opinion_text", "")),
        }]
        rows.append({
            "sentiment_code": _safe_text(top_row.get("sentiment_code", "")),
            "sentiment": _safe_text(top_row.get("sentiment", "")),
            "aspect_key": _safe_text(top_row.get("aspect_key", "General")) or "General",
            "cluster_size": cluster_size,
            "cej_scene_code": _safe_text(top_row.get("cej_scene_code", "S10 Other")) or "S10 Other",
            "product_category": _safe_text(top_row.get("product_category", "Other")) or "Other",
            "product_subtype": _safe_text(top_row.get("product_subtype", "Unknown")) or "Unknown",
            "brand_mentioned": _safe_text(top_row.get("brand_mentioned", "")),
            "model_mentioned": _safe_text(top_row.get("model_mentioned", "")),
            "bundle_score": bundle_score,
            "selection_reason": _bundle_selection_reason(top_row, cluster_size),
            "top_opinion_id": _safe_text(top_row.get("opinion_id", "")),
            "top_source_content_id": _safe_text(top_row.get("source_content_id", "")),
            "top_source_link": _safe_text(top_row.get("source_link", "")),
            "top_likes_count": None if pd.isna(top_row.get("likes_count", pd.NA)) else int(float(top_row.get("likes_count", 0) or 0)),
            "top_pii_flag": _safe_text(top_row.get("pii_flag", "N")) or "N",
            "top_pii_types": _safe_text(top_row.get("pii_types", "")),
            "top_display_text": _safe_text(top_row.get("display_text", "")),
            "top_original_text": _safe_text(top_row.get("original_text", "")),
            "top_masked_text": _safe_text(top_row.get("masked_text", "")),
            "top_translation_text": _safe_text(top_row.get("translated_text", "")),
            "top_summary_text": _safe_text(top_row.get("summary_text", "")),
            "pii_raw_text": _safe_text(top_row.get("pii_raw_text", "")),
            "sample_comments_json": json.dumps(sample_comments, ensure_ascii=False),
            "voc_items_json": json.dumps(voc_items, ensure_ascii=False),
            "region": _safe_text(top_row.get("region", "")),
            "product": _safe_text(top_row.get("product", "")),
        })
    bundle_df = pd.DataFrame(rows)
    if bundle_df.empty:
        return bundle_df
    ranked_frames = []
    for sentiment_code in ["negative", "positive"]:
        subset = bundle_df[bundle_df["sentiment_code"] == sentiment_code].sort_values(["bundle_score", "cluster_size"], ascending=[False, False]).head(top_n).copy()
        if subset.empty:
            continue
        subset["rank"] = range(1, len(subset) + 1)
        ranked_frames.append(subset)
    return pd.concat(ranked_frames, ignore_index=True) if ranked_frames else pd.DataFrame(columns=bundle_df.columns.tolist() + ["rank"])



def build_source_records(opinion_units_df: pd.DataFrame) -> list[dict[str, Any]]:
    if opinion_units_df.empty:
        return []
    records: list[dict[str, Any]] = []
    for source_id, group in opinion_units_df.groupby("source_content_id", dropna=False):
        first = group.iloc[0]
        opinion_units: list[dict[str, Any]] = []
        for _, row in group.iterrows():
            opinion_units.append({
                "opinion_id": _safe_text(row.get("opinion_id", "")),
                "opinion_text": _safe_text(row.get("opinion_text", "")),
                "cej_scene_code": _safe_text(row.get("cej_scene_code", "S10 Other")) or "S10 Other",
                "sentiment": _safe_text(row.get("sentiment", "Neutral")) or "Neutral",
                "inquiry_flag": bool(row.get("inquiry_flag", False)),
                "inquiry_type": json.loads(_safe_text(row.get("inquiry_type_json", "[]")) or "[]"),
                "aspect_key": _safe_text(row.get("aspect_key", "")),
                "evidence_spans": json.loads(_safe_text(row.get("evidence_spans_json", "[]")) or "[]"),
                "product": {
                    "product_category": _safe_text(row.get("product_category", "Other")) or "Other",
                    "product_subtype": _safe_text(row.get("product_subtype", "Unknown")) or "Unknown",
                    "brand_mentioned": _safe_text(row.get("brand_mentioned", "")),
                    "model_mentioned": _safe_text(row.get("model_mentioned", "")),
                    "product_evidence_spans": json.loads(_safe_text(row.get("product_evidence_spans_json", "[]")) or "[]"),
                },
            })
        records.append({
            "source_content_id": _safe_text(source_id),
            "source_type": _safe_text(first.get("source_type", "YT_COMMENT")) or "YT_COMMENT",
            "source_count": 1,
            "pii_flag": _safe_text(first.get("pii_flag", "N")) or "N",
            "pii_types": _safe_text(first.get("pii_types", "")),
            "pii_raw_text": _safe_text(first.get("pii_raw_text", "")),
            "context_attachment": None,
            "opinion_units": opinion_units,
            "opinion_count": len(opinion_units),
            "duplicate_opinion_flag": bool(first.get("duplicate_opinion_flag", False)),
        })
    return records


def build_representative_section_payload(bundle_df: pd.DataFrame) -> dict[str, Any]:
    payload = {
        "positive_top5_bundles": [],
        "negative_top5_bundles": [],
        "section_notes": [
            "??(Original)? AI ??(??)? ?? ??? ???? ????? ????.",
            "PII ?? ??? ???? ???? ?? ??? ?????, ???? ????? ??? ????.",
            "?? ???? ??? ???? ???? ?? aspect ??? ?? ?? ?? ???? ????.",
        ],
    }
    if bundle_df.empty:
        return payload
    for sentiment_code, key in [("positive", "positive_top5_bundles"), ("negative", "negative_top5_bundles")]:
        subset = bundle_df[bundle_df["sentiment_code"] == sentiment_code].sort_values(["rank", "bundle_score"], ascending=[True, False])
        for _, row in subset.iterrows():
            sample_comments = json.loads(_safe_text(row.get("sample_comments_json", "[]")) or "[]")
            payload[key].append({
                "rank": int(row.get("rank", 0) or 0),
                "sentiment": sentiment_code.upper(),
                "aspect_key": _safe_text(row.get("aspect_key", "General")) or "General",
                "cluster_size": int(row.get("cluster_size", 0) or 0),
                "cej_scene_code": _safe_text(row.get("cej_scene_code", "S10 Other")) or "S10 Other",
                "product": {
                    "product_category": _safe_text(row.get("product_category", "Other")) or "Other",
                    "product_subtype": _safe_text(row.get("product_subtype", "Unknown")) or "Unknown",
                    "brand_mentioned": _safe_text(row.get("brand_mentioned", "")),
                    "model_mentioned": _safe_text(row.get("model_mentioned", "")),
                },
                "top_comment": {
                    "source_content_id": _safe_text(row.get("top_source_content_id", "")),
                    "source_link": _safe_text(row.get("top_source_link", "")),
                    "likes_count": None if pd.isna(row.get("top_likes_count", pd.NA)) else int(float(row.get("top_likes_count", 0) or 0)),
                    "pii_flag": _safe_text(row.get("top_pii_flag", "N")) or "N",
                    "pii_types": _safe_text(row.get("top_pii_types", "")),
                    "display_text": _safe_text(row.get("top_display_text", "")),
                    "original_text": _safe_text(row.get("top_original_text", "")),
                    "masked_text": _safe_text(row.get("top_masked_text", "")),
                    "translation": {"translated_text": _safe_text(row.get("top_translation_text", "")), "note": "??"},
                },
                "selection_reason": _safe_text(row.get("selection_reason", "")),
                "pii_review_panel": {
                    "expand_label": "???? ??(?????)",
                    "pii_raw_text": _safe_text(row.get("pii_raw_text", "")),
                },
                "drilldown": {
                    "expand_label": "?? ??? ??",
                    "sample_comments": sample_comments[:5],
                },
            })
    return payload
