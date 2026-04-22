"""Pattern quality gate filtering for TM.6.5 outputs."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd

from src.text_mining.keyword_extractor import build_comment_keyword_map
from src.text_mining.pattern_builder import build_top_patterns

_ABSTRACT_TOKENS: set[str] = {"issue", "problem"}
_HUB_TOKENS: set[str] = {"dishwasher", "water", "dish", "wash", "clean"}

_PATTERN_COLUMNS: list[str] = [
    "pattern_label",
    "coverage",
    "keyword_set",
    "representative_comment_ids",
]

_REMOVED_COLUMNS: list[str] = [
    "pattern_label",
    "coverage",
    "keyword_set",
    "representative_comment_ids",
    "removed_reason",
]


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _to_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(v) for v in value if _safe_text(v)]
    if value is None:
        return []
    raw = _safe_text(value)
    if not raw:
        return []
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [_safe_text(v) for v in parsed if _safe_text(v)]
    except Exception:
        pass
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_keyword_set(value: Any) -> list[str]:
    tokens = [_safe_text(v).lower() for v in _to_list(value)]
    dedup: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if not token or token in seen:
            continue
        seen.add(token)
        dedup.append(token)
    return dedup


def _normalize_comment_ids(value: Any) -> list[str]:
    dedup: list[str] = []
    seen: set[str] = set()
    for token in _to_list(value):
        cid = _safe_text(token)
        if not cid or cid in seen:
            continue
        seen.add(cid)
        dedup.append(cid)
    return dedup


def _comment_keyword_lookup(base_df: pd.DataFrame) -> dict[str, set[str]]:
    comment_keyword_map = build_comment_keyword_map(base_df)
    lookup: dict[str, set[str]] = {}
    if comment_keyword_map.empty:
        return lookup
    grouped = comment_keyword_map.groupby("comment_id", sort=False)["keyword"]
    for comment_id, series in grouped:
        cid = _safe_text(comment_id)
        keywords = {_safe_text(v).lower() for v in series.tolist() if _safe_text(v)}
        if cid and keywords:
            lookup[cid] = keywords
    return lookup


def _keyword_jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = max(len(a.union(b)), 1)
    return len(a.intersection(b)) / union


def _core_keywords(tokens: set[str]) -> set[str]:
    return {v for v in tokens if v not in _ABSTRACT_TOKENS and v not in _HUB_TOKENS}


def _label_anchor(pattern_label: str, keyword_set: list[str]) -> str:
    parts = [p.strip().lower() for p in str(pattern_label).split("/") if p.strip()]
    for token in parts:
        if token and token not in _ABSTRACT_TOKENS:
            return token
    for token in keyword_set:
        normalized = _safe_text(token).lower()
        if normalized and normalized not in _ABSTRACT_TOKENS:
            return normalized
    return ""


def _remove_with_reason(
    removed: list[dict[str, Any]],
    *,
    pattern_label: str,
    coverage: float,
    keyword_set: list[str],
    representative_comment_ids: list[str],
    reason: str,
) -> None:
    removed.append(
        {
            "pattern_label": pattern_label,
            "coverage": float(coverage),
            "keyword_set": keyword_set,
            "representative_comment_ids": representative_comment_ids,
            "removed_reason": reason,
        }
    )


def _adaptive_min_coverage(total_comments: int, requested: float) -> float:
    if total_comments >= 30000:
        floor = 0.35
    elif total_comments >= 10000:
        floor = 0.30
    elif total_comments >= 5000:
        floor = 0.25
    elif total_comments >= 2000:
        floor = 0.20
    elif total_comments >= 1000:
        floor = 0.15
    elif total_comments >= 300:
        floor = 0.10
    else:
        floor = 0.05
    return min(float(requested), floor)


def _has_meaningful_keywords(keyword_set: list[str]) -> bool:
    normalized = {_safe_text(token).lower() for token in keyword_set if _safe_text(token)}
    non_abstract = normalized.difference(_ABSTRACT_TOKENS)
    return len(non_abstract) >= 2


def _build_relaxed_survivors(
    patterns_df: pd.DataFrame,
    *,
    comment_lookup: dict[str, set[str]],
    min_coverage: float,
    min_rep_keyword_overlap: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in patterns_df.itertuples(index=False):
        pattern_label = _safe_text(getattr(row, "pattern_label", ""))
        coverage = float(getattr(row, "coverage", 0.0) or 0.0)
        keyword_set = _normalize_keyword_set(getattr(row, "keyword_set", []))
        rep_ids = _normalize_comment_ids(getattr(row, "representative_comment_ids", []))
        if coverage < float(min_coverage):
            continue
        if any(token in _ABSTRACT_TOKENS for token in keyword_set) and not _has_meaningful_keywords(keyword_set):
            continue

        keyword_lookup = set(keyword_set)
        valid_rep_ids: list[str] = []
        for cid in rep_ids:
            comment_keywords = comment_lookup.get(cid, set())
            overlap = len(keyword_lookup.intersection(comment_keywords))
            if overlap >= int(min_rep_keyword_overlap):
                valid_rep_ids.append(cid)
        if not valid_rep_ids:
            continue
        rows.append(
            {
                "pattern_label": pattern_label,
                "coverage": coverage,
                "keyword_set": keyword_set,
                "representative_comment_ids": valid_rep_ids,
            }
        )
    return pd.DataFrame(rows, columns=_PATTERN_COLUMNS)


def _append_unique_patterns(base: pd.DataFrame, extra: pd.DataFrame, *, max_top_n: int) -> pd.DataFrame:
    if base is None or base.empty:
        if extra is None or extra.empty:
            return pd.DataFrame(columns=_PATTERN_COLUMNS)
        return extra.head(max(int(max_top_n), 1)).reset_index(drop=True)
    if extra is None or extra.empty:
        return base.head(max(int(max_top_n), 1)).reset_index(drop=True)

    out = base.copy().reset_index(drop=True)
    existing_labels = set(out["pattern_label"].astype(str).tolist())
    for row in extra.itertuples(index=False):
        if len(out) >= max(int(max_top_n), 1):
            break
        pattern_label = _safe_text(row.pattern_label)
        if pattern_label in existing_labels:
            continue
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "pattern_label": pattern_label,
                            "coverage": float(row.coverage),
                            "keyword_set": list(row.keyword_set),
                            "representative_comment_ids": list(row.representative_comment_ids),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        existing_labels.add(pattern_label)
    out = out.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True)
    return out.head(max(int(max_top_n), 1))


def _select_diverse_top_patterns(
    dedup_df: pd.DataFrame,
    *,
    max_top_n: int,
    removed: list[dict[str, Any]],
    similarity_threshold: float = 0.35,
) -> pd.DataFrame:
    if dedup_df.empty:
        return pd.DataFrame(columns=_PATTERN_COLUMNS)

    candidates = dedup_df.copy().reset_index(drop=True)
    selected_rows: list[dict[str, Any]] = []
    selected_keyword_sets: list[set[str]] = []
    selected_core_sets: list[set[str]] = []
    used_anchors: set[str] = set()
    max_coverage = float(candidates["coverage"].max() or 1.0)

    for row in candidates.itertuples(index=False):
        if len(selected_rows) >= max(int(max_top_n), 1):
            break

        pattern_label = _safe_text(row.pattern_label)
        coverage = float(row.coverage)
        keyword_set = list(row.keyword_set)
        rep_ids = list(row.representative_comment_ids)
        anchor = _label_anchor(pattern_label, keyword_set)

        current_keywords = set(keyword_set)
        current_core = _core_keywords(current_keywords)
        max_similarity = 0.0
        for kset, cset in zip(selected_keyword_sets, selected_core_sets):
            keyword_sim = _keyword_jaccard(current_keywords, kset)
            core_sim = _keyword_jaccard(current_core, cset)
            max_similarity = max(max_similarity, keyword_sim, core_sim)

        if max_similarity >= similarity_threshold:
            _remove_with_reason(
                removed,
                pattern_label=pattern_label,
                coverage=coverage,
                keyword_set=keyword_set,
                representative_comment_ids=rep_ids,
                reason=f"top_diversity_overlap(sim>={similarity_threshold})",
            )
            continue

        if anchor and anchor in used_anchors:
            _remove_with_reason(
                removed,
                pattern_label=pattern_label,
                coverage=coverage,
                keyword_set=keyword_set,
                representative_comment_ids=rep_ids,
                reason=f"seed_axis_repetition({anchor})",
            )
            continue

        selected_rows.append(
            {
                "pattern_label": pattern_label,
                "coverage": coverage,
                "keyword_set": keyword_set,
                "representative_comment_ids": rep_ids,
                "_priority_score": (coverage / max(max_coverage, 1e-9)) - (0.55 * max_similarity),
            }
        )
        selected_keyword_sets.append(current_keywords)
        selected_core_sets.append(current_core)
        if anchor:
            used_anchors.add(anchor)

    if not selected_rows:
        return pd.DataFrame(columns=_PATTERN_COLUMNS)

    out = pd.DataFrame(selected_rows)
    out = out.sort_values(["_priority_score", "coverage", "pattern_label"], ascending=[False, False, True])
    out = out.drop(columns=["_priority_score"]).reset_index(drop=True)
    return out.head(max(int(max_top_n), 1))


def filter_pattern_quality(
    base_df: pd.DataFrame,
    *,
    patterns_df: pd.DataFrame | None = None,
    min_coverage: float = 0.5,
    min_rep_keyword_overlap: int = 2,
    max_top_n: int = 5,
    candidate_top_n: int = 20,
    min_pair_count: int = 5,
    dedup_jaccard_threshold: float = 0.40,
) -> dict[str, pd.DataFrame]:
    """
    Filter pattern outputs by quality gate rules.

    Rules:
    - coverage threshold
    - abstract token filter
    - representative comment keyword overlap validation
    - stricter semantic deduplication
    - flexible top-N (no forced 5)
    """
    total_comments = int(base_df["comment_id"].astype(str).nunique()) if base_df is not None and not base_df.empty and "comment_id" in base_df.columns else int(len(base_df) if base_df is not None else 0)
    if base_df is None or base_df.empty:
        empty_patterns = pd.DataFrame(columns=_PATTERN_COLUMNS)
        empty_removed = pd.DataFrame(columns=_REMOVED_COLUMNS)
        return {
            "input_patterns": empty_patterns,
            "filtered_patterns": empty_patterns,
            "removed_patterns": empty_removed,
            "final_top_patterns": empty_patterns,
        }

    if patterns_df is None:
        patterns_df = build_top_patterns(
            base_df,
            top_n=max(max_top_n, candidate_top_n),
            min_pair_count=min_pair_count,
        )
    if patterns_df is None or patterns_df.empty:
        relaxed_pair_count = max(1, min(int(min_pair_count), 3))
        patterns_df = build_top_patterns(
            base_df,
            top_n=max(max_top_n, candidate_top_n, 8),
            min_pair_count=relaxed_pair_count,
        )
    if patterns_df is None or patterns_df.empty:
        empty_patterns = pd.DataFrame(columns=_PATTERN_COLUMNS)
        empty_removed = pd.DataFrame(columns=_REMOVED_COLUMNS)
        return {
            "input_patterns": empty_patterns,
            "filtered_patterns": empty_patterns,
            "removed_patterns": empty_removed,
            "final_top_patterns": empty_patterns,
        }

    comment_lookup = _comment_keyword_lookup(base_df)
    effective_min_coverage = _adaptive_min_coverage(total_comments, float(min_coverage))
    effective_min_rep_overlap = int(min_rep_keyword_overlap)
    if total_comments <= 2500:
        effective_min_rep_overlap = min(effective_min_rep_overlap, 1)
    removed: list[dict[str, Any]] = []
    survivors: list[dict[str, Any]] = []

    for row in patterns_df.itertuples(index=False):
        pattern_label = _safe_text(getattr(row, "pattern_label", ""))
        coverage = float(getattr(row, "coverage", 0.0) or 0.0)
        keyword_set = _normalize_keyword_set(getattr(row, "keyword_set", []))
        rep_ids = _normalize_comment_ids(getattr(row, "representative_comment_ids", []))

        remove_reasons: list[str] = []
        if coverage < float(effective_min_coverage):
            remove_reasons.append(f"coverage_below_threshold(<{effective_min_coverage}%)")
        if any(token in _ABSTRACT_TOKENS for token in keyword_set) and not _has_meaningful_keywords(keyword_set):
            remove_reasons.append("contains_abstract_token(issue/problem)")

        valid_rep_ids: list[str] = []
        keyword_lookup = set(keyword_set)
        for cid in rep_ids:
            comment_keywords = comment_lookup.get(cid, set())
            overlap = len(keyword_lookup.intersection(comment_keywords))
            if overlap >= int(effective_min_rep_overlap):
                valid_rep_ids.append(cid)

        if not valid_rep_ids:
            remove_reasons.append(f"representative_alignment_fail(overlap<{effective_min_rep_overlap})")

        if remove_reasons:
            _remove_with_reason(
                removed,
                pattern_label=pattern_label,
                coverage=coverage,
                keyword_set=keyword_set,
                representative_comment_ids=rep_ids,
                reason="; ".join(remove_reasons),
            )
            continue

        survivors.append(
            {
                "pattern_label": pattern_label,
                "coverage": coverage,
                "keyword_set": keyword_set,
                "representative_comment_ids": valid_rep_ids,
            }
        )

    relaxed_pool = _build_relaxed_survivors(
        patterns_df,
        comment_lookup=comment_lookup,
        min_coverage=max(0.05, effective_min_coverage * 0.5),
        min_rep_keyword_overlap=1,
    )

    if not survivors:
        if relaxed_pool.empty:
            empty_patterns = pd.DataFrame(columns=_PATTERN_COLUMNS)
            removed_df = pd.DataFrame(removed, columns=_REMOVED_COLUMNS)
            return {
                "input_patterns": patterns_df[_PATTERN_COLUMNS].copy(),
                "filtered_patterns": empty_patterns,
                "removed_patterns": removed_df,
                "final_top_patterns": empty_patterns,
            }
        survivors_df = relaxed_pool.copy()
    else:
        survivors_df = pd.DataFrame(survivors, columns=_PATTERN_COLUMNS)

    survivors_df = survivors_df.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True)

    deduped: list[dict[str, Any]] = []
    for row in survivors_df.itertuples(index=False):
        current_keywords = set(row.keyword_set)
        current_core = _core_keywords(current_keywords)
        drop_reason = ""
        for kept in deduped:
            kept_keywords = set(kept["keyword_set"])
            kept_core = _core_keywords(kept_keywords)
            keyword_jaccard = _keyword_jaccard(current_keywords, kept_keywords)
            core_jaccard = _keyword_jaccard(current_core, kept_core)
            if keyword_jaccard >= dedup_jaccard_threshold or core_jaccard >= dedup_jaccard_threshold:
                drop_reason = f"semantic_duplicate_of:{kept['pattern_label']}"
                break
        if drop_reason:
            _remove_with_reason(
                removed,
                pattern_label=row.pattern_label,
                coverage=float(row.coverage),
                keyword_set=list(row.keyword_set),
                representative_comment_ids=list(row.representative_comment_ids),
                reason=drop_reason,
            )
            continue
        deduped.append(
            {
                "pattern_label": row.pattern_label,
                "coverage": float(row.coverage),
                "keyword_set": list(row.keyword_set),
                "representative_comment_ids": list(row.representative_comment_ids),
            }
        )

    dedup_df = pd.DataFrame(deduped, columns=_PATTERN_COLUMNS)
    dedup_df = dedup_df.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True)
    final_df = _select_diverse_top_patterns(
        dedup_df,
        max_top_n=max_top_n,
        removed=removed,
    ).reset_index(drop=True)

    min_required = 2 if total_comments > 300 else 1
    if len(final_df) < min_required:
        relaxed_diverse = _select_diverse_top_patterns(
            relaxed_pool.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True),
            max_top_n=max_top_n,
            removed=[],
            similarity_threshold=0.50,
        ).reset_index(drop=True)
        final_df = _append_unique_patterns(final_df, relaxed_diverse, max_top_n=max_top_n)

    if len(final_df) < min_required:
        provisional_patterns = build_top_patterns(
            base_df,
            top_n=max(max_top_n, 6),
            min_pair_count=max(1, min(int(min_pair_count), 2)),
        )
        provisional_relaxed = _build_relaxed_survivors(
            provisional_patterns,
            comment_lookup=comment_lookup,
            min_coverage=max(0.05, effective_min_coverage * 0.4),
            min_rep_keyword_overlap=1,
        )
        provisional_final = _select_diverse_top_patterns(
            provisional_relaxed.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True),
            max_top_n=max_top_n,
            removed=[],
            similarity_threshold=0.55,
        ).reset_index(drop=True)
        final_df = _append_unique_patterns(final_df, provisional_final, max_top_n=max_top_n)

    removed_df = pd.DataFrame(removed, columns=_REMOVED_COLUMNS)
    return {
        "input_patterns": patterns_df[_PATTERN_COLUMNS].copy(),
        "filtered_patterns": dedup_df,
        "removed_patterns": removed_df,
        "final_top_patterns": final_df,
    }
