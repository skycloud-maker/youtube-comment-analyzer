"""Pattern aggregation from keyword relations (descriptive only)."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.text_mining.cooccurrence import build_keyword_cooccurrence
from src.text_mining.keyword_extractor import build_comment_keyword_map

_OUTPUT_COLUMNS: list[str] = [
    "pattern_label",
    "coverage",
    "keyword_set",
    "representative_comment_ids",
]

# hub tokens are allowed in patterns but cannot connect unrelated clusters.
_HUB_TOKENS: set[str] = {"dishwasher", "water", "dish", "wash", "clean"}
_ABSTRACT_TOKENS: set[str] = {"price", "cost", "expensive", "issue", "problem"}


@dataclass
class _PatternCandidate:
    seed_pair: tuple[str, str]
    keyword_set: list[str]
    matched_comment_ids: list[str]
    coverage: float
    strength: float


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _empty_output() -> pd.DataFrame:
    return pd.DataFrame(columns=_OUTPUT_COLUMNS)


def _build_keyword_frequency(comment_keyword_map: pd.DataFrame) -> Counter[str]:
    counter: Counter[str] = Counter()
    for keyword in comment_keyword_map["keyword"].tolist():
        token = _safe_text(keyword).lower()
        if token:
            counter[token] += 1
    return counter


def _build_comment_keywords(comment_keyword_map: pd.DataFrame) -> dict[str, set[str]]:
    grouped = comment_keyword_map.groupby("comment_id", sort=False)["keyword"]
    out: dict[str, set[str]] = {}
    for comment_id, series in grouped:
        cid = _safe_text(comment_id)
        tokens = {_safe_text(v).lower() for v in series.tolist() if _safe_text(v)}
        if cid and tokens:
            out[cid] = tokens
    return out


def _build_pair_maps(pair_df: pd.DataFrame) -> tuple[dict[tuple[str, str], float], dict[str, list[tuple[str, float]]]]:
    score_map: dict[tuple[str, str], float] = {}
    neighbors: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for row in pair_df.itertuples(index=False):
        k1 = _safe_text(getattr(row, "keyword_1", "")).lower()
        k2 = _safe_text(getattr(row, "keyword_2", "")).lower()
        score = float(getattr(row, "rank_score", 0.0) or 0.0)
        pair_count = int(getattr(row, "pair_count", 0) or 0)
        if not k1 or not k2 or k1 == k2:
            continue
        if pair_count <= 0 or score <= 0:
            continue
        key = tuple(sorted((k1, k2)))
        score_map[key] = max(score_map.get(key, 0.0), score)
        neighbors[k1].append((k2, score))
        neighbors[k2].append((k1, score))
    return score_map, neighbors


def _pair_score(score_map: dict[tuple[str, str], float], a: str, b: str) -> float:
    if not a or not b or a == b:
        return 0.0
    return score_map.get(tuple(sorted((a, b))), 0.0)


def _select_seed_pairs(
    pair_df: pd.DataFrame,
    *,
    min_pair_count: int,
    seed_quantile: float,
    max_seeds: int,
) -> list[tuple[str, str, float]]:
    if pair_df.empty:
        return []
    ranked = pair_df.copy()
    ranked = ranked[ranked["pair_count"] >= min_pair_count]
    if ranked.empty:
        return []
    min_seed_score = float(ranked["rank_score"].quantile(seed_quantile))
    ranked = ranked[ranked["rank_score"] >= min_seed_score]
    if ranked.empty:
        return []
    ranked = ranked.sort_values(["rank_score", "pair_count"], ascending=[False, False])

    seeds: list[tuple[str, str, float]] = []
    used: set[tuple[str, str]] = set()
    for row in ranked.itertuples(index=False):
        k1 = _safe_text(getattr(row, "keyword_1", "")).lower()
        k2 = _safe_text(getattr(row, "keyword_2", "")).lower()
        score = float(getattr(row, "rank_score", 0.0) or 0.0)
        if not k1 or not k2 or k1 == k2:
            continue
        if k1 in _HUB_TOKENS and k2 in _HUB_TOKENS:
            continue
        key = tuple(sorted((k1, k2)))
        if key in used:
            continue
        used.add(key)
        seeds.append((k1, k2, score))
        if len(seeds) >= max_seeds:
            break
    return seeds


def _expand_seed_keywords(
    seed_a: str,
    seed_b: str,
    seed_score: float,
    *,
    score_map: dict[tuple[str, str], float],
    neighbors: dict[str, list[tuple[str, float]]],
    keyword_freq: Counter[str],
    expand_score_threshold: float,
    max_keywords: int = 5,
) -> list[str]:
    keyword_set = [seed_a, seed_b]
    candidates: dict[str, float] = {}
    seed_hub_count = int(seed_a in _HUB_TOKENS) + int(seed_b in _HUB_TOKENS)

    neighbor_tokens = neighbors.get(seed_a, []) + neighbors.get(seed_b, [])
    for token, _ in neighbor_tokens:
        if token in keyword_set:
            continue
        if seed_hub_count == 0 and token in _HUB_TOKENS:
            continue
        score_a = _pair_score(score_map, seed_a, token)
        score_b = _pair_score(score_map, seed_b, token)
        best = max(score_a, score_b)
        if best < expand_score_threshold:
            continue
        # stronger when token is related to both seed terms.
        bridge_bonus = 1.12 if score_a > 0 and score_b > 0 else 1.0
        # discourage hub growth that merges unrelated clusters.
        hub_penalty = 0.58 if token in _HUB_TOKENS else 1.0
        freq_weight = 1.0 + min(keyword_freq.get(token, 0) / 3000.0, 0.35)
        candidates[token] = max(candidates.get(token, 0.0), best * bridge_bonus * hub_penalty * freq_weight)

    ranked_candidates = sorted(candidates.items(), key=lambda x: (-x[1], x[0]))
    for token, score in ranked_candidates:
        if len(keyword_set) >= max_keywords:
            break
        # avoid very weak seed-relative additions.
        if score < (seed_score * 0.42):
            continue
        keyword_set.append(token)
    return keyword_set


def _match_comment_ids_for_seed(
    comment_keywords: dict[str, set[str]],
    *,
    seed_pair: tuple[str, str],
    keyword_set: list[str],
) -> list[tuple[str, int]]:
    a, b = seed_pair
    keyword_lookup = set(keyword_set)
    contextual_keywords = keyword_lookup.difference(_ABSTRACT_TOKENS)
    out: list[tuple[str, int]] = []
    threshold = 2
    for comment_id, tokens in comment_keywords.items():
        if a not in tokens or b not in tokens:
            continue
        if contextual_keywords and not tokens.intersection(contextual_keywords):
            continue
        overlap = len(tokens.intersection(keyword_lookup))
        if overlap < threshold:
            continue
        out.append((comment_id, overlap))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out


def _is_overlapping_pattern(
    existing: list[_PatternCandidate],
    candidate: _PatternCandidate,
    *,
    keyword_overlap_limit: float = 0.40,
    comment_overlap_limit: float = 0.45,
) -> bool:
    cand_keywords = set(candidate.keyword_set)
    cand_comments = set(candidate.matched_comment_ids)
    for item in existing:
        ex_keywords = set(item.keyword_set)
        ex_comments = set(item.matched_comment_ids)
        keyword_union = max(len(cand_keywords.union(ex_keywords)), 1)
        comment_union = max(len(cand_comments.union(ex_comments)), 1)
        keyword_jaccard = len(cand_keywords.intersection(ex_keywords)) / keyword_union
        comment_jaccard = len(cand_comments.intersection(ex_comments)) / comment_union
        if keyword_jaccard >= keyword_overlap_limit or comment_jaccard >= comment_overlap_limit:
            return True
        ex_non_hub = tuple(sorted([v for v in ex_keywords if v not in _HUB_TOKENS])[:2])
        cand_non_hub = tuple(sorted([v for v in cand_keywords if v not in _HUB_TOKENS])[:2])
        if ex_non_hub and ex_non_hub == cand_non_hub:
            return True
    return False


def _finalize_label(
    seed_pair: tuple[str, str],
    keyword_set: list[str],
    keyword_freq: Counter[str],
) -> str:
    if not keyword_set and not seed_pair:
        return "pattern"
    tokens = list(dict.fromkeys([seed_pair[0], seed_pair[1]] + keyword_set))
    # make label interpretable while anchoring with seed pair.
    ranked = sorted(
        tokens,
        key=lambda token: (
            token in _HUB_TOKENS,
            -keyword_freq.get(token, 0),
            token,
        ),
    )
    return " / ".join(ranked[:3])


def _build_candidates(
    *,
    pair_df: pd.DataFrame,
    comment_keywords: dict[str, set[str]],
    keyword_freq: Counter[str],
    top_n: int,
    min_pair_count: int,
    total_comments: int,
) -> list[_PatternCandidate]:
    score_map, neighbors = _build_pair_maps(pair_df)
    if not score_map:
        return []

    all_candidates: list[_PatternCandidate] = []
    seed_quantiles = (0.92, 0.86, 0.80, 0.74, 0.68, 0.60)
    expand_quantiles = (0.88, 0.80, 0.72, 0.64)
    min_support = max(6, int(total_comments * 0.00012))
    min_coverage = 0.02
    max_coverage = 34.0

    for seed_q in seed_quantiles:
        seeds = _select_seed_pairs(
            pair_df,
            min_pair_count=min_pair_count,
            seed_quantile=seed_q,
            max_seeds=120,
        )
        if not seeds:
            continue
        for expand_q in expand_quantiles:
            expand_threshold = float(pair_df["rank_score"].quantile(expand_q))
            for seed_a, seed_b, seed_score in seeds:
                keyword_set = _expand_seed_keywords(
                    seed_a,
                    seed_b,
                    seed_score,
                    score_map=score_map,
                    neighbors=neighbors,
                    keyword_freq=keyword_freq,
                    expand_score_threshold=expand_threshold,
                    max_keywords=5,
                )
                matched = _match_comment_ids_for_seed(
                    comment_keywords,
                    seed_pair=(seed_a, seed_b),
                    keyword_set=keyword_set,
                )
                if len(matched) < min_support:
                    continue
                coverage = (len(matched) / max(total_comments, 1)) * 100
                if coverage < min_coverage or coverage > max_coverage:
                    continue
                candidate = _PatternCandidate(
                    seed_pair=(seed_a, seed_b),
                    keyword_set=keyword_set,
                    matched_comment_ids=[cid for cid, _ in matched],
                    coverage=round(coverage, 3),
                    strength=round(seed_score, 6),
                )
                all_candidates.append(candidate)

        if len(all_candidates) >= top_n * 3:
            break
    if not all_candidates:
        return []

    # stronger candidates first; suppress overlaps to enforce separation.
    all_candidates.sort(
        key=lambda c: (-c.coverage, -c.strength, c.seed_pair[0], c.seed_pair[1])
    )
    selected: list[_PatternCandidate] = []
    used_label_keys: set[tuple[str, ...]] = set()
    used_labels: set[str] = set()
    anchor_counts: dict[str, int] = defaultdict(int)

    def _anchor_token(keyword_set: list[str]) -> str:
        non_hub = [token for token in keyword_set if token not in _HUB_TOKENS]
        if non_hub:
            ranked = sorted(non_hub, key=lambda t: (-keyword_freq.get(t, 0), t))
            return ranked[0]
        ranked = sorted(keyword_set, key=lambda t: (-keyword_freq.get(t, 0), t))
        return ranked[0] if ranked else ""

    def _try_add(candidate: _PatternCandidate, *, enforce_anchor_cap: bool) -> bool:
        if _is_overlapping_pattern(selected, candidate):
            return False
        label_key = tuple(
            sorted([token for token in candidate.keyword_set if token not in _HUB_TOKENS])[:3]
        )
        if label_key and label_key in used_label_keys:
            return False
        preview_label = _finalize_label(candidate.seed_pair, candidate.keyword_set, keyword_freq)
        if preview_label in used_labels:
            return False
        anchor = _anchor_token(candidate.keyword_set)
        if enforce_anchor_cap and anchor and anchor_counts.get(anchor, 0) >= 1:
            return False
        selected.append(candidate)
        if label_key:
            used_label_keys.add(label_key)
        used_labels.add(preview_label)
        if anchor:
            anchor_counts[anchor] += 1
        return True

    # pass 1: enforce anchor diversity for clearer separation.
    for candidate in all_candidates:
        _try_add(candidate, enforce_anchor_cap=True)
        if len(selected) >= top_n:
            break

    # pass 2: fill remaining slots without anchor cap.
    if len(selected) < top_n:
        for candidate in all_candidates:
            _try_add(candidate, enforce_anchor_cap=False)
            if len(selected) >= top_n:
                break
    return selected


def build_top_patterns(
    base_df: pd.DataFrame,
    *,
    top_n: int = 5,
    min_pair_count: int = 5,
) -> pd.DataFrame:
    """
    Build top descriptive patterns.

    Output columns:
    - pattern_label
    - coverage
    - keyword_set
    - representative_comment_ids
    """
    if base_df is None or base_df.empty:
        return _empty_output()
    if "comment_id" not in base_df.columns:
        return _empty_output()

    comment_keyword_map = build_comment_keyword_map(base_df)
    if comment_keyword_map.empty:
        return _empty_output()

    pair_df = build_keyword_cooccurrence(base_df, min_pair_count=min_pair_count)
    if pair_df.empty:
        return _empty_output()

    comment_keywords = _build_comment_keywords(comment_keyword_map)
    keyword_freq = _build_keyword_frequency(comment_keyword_map)
    total_comments = max(int(base_df["comment_id"].astype(str).nunique()), 1)

    selected = _build_candidates(
        pair_df=pair_df,
        comment_keywords=comment_keywords,
        keyword_freq=keyword_freq,
        top_n=max(int(top_n), 1),
        min_pair_count=min_pair_count,
        total_comments=total_comments,
    )
    if not selected:
        return _empty_output()

    rows: list[dict[str, Any]] = []
    for item in selected:
        rows.append(
            {
                "pattern_label": _finalize_label(item.seed_pair, item.keyword_set, keyword_freq),
                "coverage": item.coverage,
                "keyword_set": item.keyword_set,
                "representative_comment_ids": item.matched_comment_ids[:3],
            }
        )

    out = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    out = out.sort_values(["coverage", "pattern_label"], ascending=[False, True]).reset_index(drop=True)
    return out.head(max(int(top_n), 1))
