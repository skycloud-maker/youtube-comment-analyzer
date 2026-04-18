"""Aggregate-level strategy insight builder.

This module intentionally keeps strategy conclusions at aggregate levels
(video/product-group) and never from a single comment.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

import pandas as pd


REQUIRED_STRATEGY_COLUMNS = [
    "level_type",
    "level_id",
    "video_id",
    "product_group",
    "journey_stage",
    "judgment_axes",
    "insight_title",
    "insight_summary",
    "priority_signal",
    "evidence_summary",
    "supporting_representatives",
    "supporting_cluster_count",
    "insufficient_evidence",
    "polarity_direction",
    "mixed_ratio",
    "repetition_signal",
    "impact_signal",
]

_AXIS_KO = {
    "price_value_fit": "가격/가치 적합성",
    "installation_space_fit": "설치/공간 적합성",
    "maintenance_burden": "유지관리 부담",
    "automation_convenience": "자동화/편의성",
    "durability_lifespan": "내구성/수명",
    "brand_trust": "브랜드 신뢰",
    "usage_pattern_fit": "사용 패턴 적합성",
    "comparison_substitute_judgment": "비교/대체재 판단",
    "design_interior_fit": "디자인/인테리어 적합성",
    "service_after_sales": "서비스/A·S",
    "purchase_strategy_contract": "구매전략/계약",
    "repurchase_recommendation": "재구매/추천 의향",
}

_STAGE_KO = {
    "awareness": "인지",
    "exploration": "탐색",
    "purchase": "구매",
    "decision": "결정",
    "delivery": "배송",
    "onboarding": "사용준비",
    "usage": "사용",
    "maintenance": "관리",
    "replacement": "교체",
}

_TOKEN_RE = re.compile(r"[^\w\s가-힣]+")
_SPLIT_RE = re.compile(r"[\s,|;/]+")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, tuple):
        return [_safe_text(item) for item in value if _safe_text(item)]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        if raw.startswith("[") and raw.endswith("]"):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [_safe_text(item) for item in parsed if _safe_text(item)]
            except Exception:
                pass
        return [_safe_text(item) for item in _SPLIT_RE.split(raw) if _safe_text(item)]
    return []


def _serialize_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return json.dumps(items, ensure_ascii=False)


def _normalize_axes(value: Any) -> list[str]:
    axes = [item for item in _parse_list(value) if item]
    deduped: list[str] = []
    for axis in axes:
        if axis not in deduped:
            deduped.append(axis)
    return deduped[:3]


def _primary_axis(value: Any) -> str:
    axes = _normalize_axes(value)
    return axes[0] if axes else "unknown_axis"


def _axis_label(axis: str) -> str:
    return _AXIS_KO.get(axis, axis.replace("_", " "))


def _stage_label(stage: str) -> str:
    return _STAGE_KO.get(stage, stage or "미분류")


def _normalize_polarity(value: Any) -> str:
    lowered = _safe_text(value).lower()
    if lowered in {"positive", "negative", "mixed", "neutral"}:
        return lowered
    return "neutral"


def _impact_value(frame: pd.DataFrame) -> float:
    like_series = pd.to_numeric(frame.get("like_count", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    reply_series = pd.to_numeric(frame.get("reply_count", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    # Replies amplify spread; likes reflect resonance.
    return float((like_series + (reply_series * 1.5)).sum())


def _problem_signature(row: pd.Series) -> str:
    candidates: list[str] = []
    for field in ["nlp_similarity_keys", "nlp_core_points", "nlp_context_tags"]:
        candidates.extend(_normalize_axes(_parse_list(row.get(field, []))))

    if not candidates:
        raw = " ".join(
            [
                _safe_text(row.get("cleaned_text", "")),
                _safe_text(row.get("text_display", "")),
                _safe_text(row.get("classification_type", "")),
            ]
        ).lower()
        cleaned = _TOKEN_RE.sub(" ", raw)
        tokens = [token for token in _SPLIT_RE.split(cleaned) if len(token) >= 2]
        candidates = tokens[:3]

    if not candidates:
        return "general_signal"
    return "|".join(candidates[:3])


def _prepare_base_frame(analysis_non_trash: pd.DataFrame) -> pd.DataFrame:
    if analysis_non_trash.empty:
        return pd.DataFrame()
    base = analysis_non_trash.copy()
    base["journey_stage"] = base.get("journey_stage", pd.Series("", index=base.index)).map(_safe_text)
    base["judgment_axes"] = base.get("judgment_axes", pd.Series([[] for _ in range(len(base))], index=base.index)).map(_normalize_axes)
    base["primary_axis"] = base["judgment_axes"].map(_primary_axis)
    base["polarity_direction"] = base.get("sentiment_final", base.get("sentiment_label", pd.Series("neutral", index=base.index))).map(_normalize_polarity)
    base["mixed_flag"] = base.get("mixed_flag", pd.Series(False, index=base.index)).fillna(False).astype(bool)
    base["problem_signature"] = base.apply(_problem_signature, axis=1)
    base["analysis_included"] = base.get("analysis_included", pd.Series(True, index=base.index)).fillna(False).astype(bool)
    base["hygiene_class"] = base.get("hygiene_class", pd.Series("strategic", index=base.index)).fillna("").astype(str).str.lower()
    base = base[base["analysis_included"] & base["hygiene_class"].ne("trash")].copy()
    return base


def _prepare_representative_index(representative_comments: pd.DataFrame) -> pd.DataFrame:
    if representative_comments.empty:
        return pd.DataFrame(columns=["comment_id", "video_id", "product", "journey_stage", "judgment_axes", "polarity_direction"])
    reps = representative_comments.copy()
    reps["comment_id"] = reps.get("comment_id", pd.Series("", index=reps.index)).map(_safe_text)
    reps = reps[reps["comment_id"].ne("")].copy()
    reps["journey_stage"] = reps.get("journey_stage", pd.Series("", index=reps.index)).map(_safe_text)
    reps["judgment_axes"] = reps.get("judgment_axes", pd.Series([[] for _ in range(len(reps))], index=reps.index)).map(_normalize_axes)
    reps["polarity_direction"] = reps.get("sentiment_final", reps.get("sentiment_label", pd.Series("neutral", index=reps.index))).map(_normalize_polarity)
    return reps


def _priority_signal(
    *,
    cluster_size: int,
    cluster_ratio: float,
    impact_value: float,
    polarity_direction: str,
    mixed_ratio: float,
) -> float:
    polarity_weight = {"negative": 1.25, "mixed": 1.1, "positive": 0.9, "neutral": 0.6}.get(polarity_direction, 0.7)
    repetition_component = min(cluster_size, 30) / 30.0
    impact_component = min(math.log1p(max(impact_value, 0.0)), 6.0) / 6.0
    mixed_component = min(max(mixed_ratio, 0.0), 1.0) * 0.2
    raw = ((cluster_ratio * 0.55) + (repetition_component * 0.30) + (impact_component * 0.15) + mixed_component) * polarity_weight
    return round(min(raw * 100.0, 100.0), 1)


def _insight_title(stage: str, axis: str, polarity: str) -> str:
    stage_label = _stage_label(stage)
    axis_label = _axis_label(axis)
    polarity_label = {"negative": "마찰", "positive": "강점", "mixed": "혼합신호", "neutral": "관찰신호"}.get(polarity, "신호")
    return f"{stage_label} 단계 · {axis_label} {polarity_label}"


def _insight_summary(
    *,
    stage: str,
    axis: str,
    polarity: str,
    cluster_size: int,
    level_total: int,
    distinct_signatures: int,
) -> str:
    stage_label = _stage_label(stage)
    axis_label = _axis_label(axis)
    share = (cluster_size / max(level_total, 1)) * 100.0
    if polarity == "negative":
        return (
            f"{stage_label} 단계에서 '{axis_label}' 마찰이 반복되어 고객 경험 신호가 집중됩니다. "
            f"동일 판단 구조가 {cluster_size:,}건(비중 {share:.1f}%) 확인되어 우선 관리 대상입니다."
        )
    if polarity == "positive":
        return (
            f"{stage_label} 단계에서 '{axis_label}' 강점 신호가 누적됩니다. "
            f"반복 근거 {cluster_size:,}건(비중 {share:.1f}%)으로 확산 가능한 강점 가설입니다."
        )
    if polarity == "mixed":
        return (
            f"{stage_label} 단계에서 '{axis_label}'에 대해 기대와 불만이 동시에 나타납니다. "
            f"혼합 신호 {cluster_size:,}건(비중 {share:.1f}%)으로 세부 원인 분리가 필요한 구간입니다."
        )
    return (
        f"{stage_label} 단계 '{axis_label}' 신호는 관찰 근거 {cluster_size:,}건(비중 {share:.1f}%)입니다. "
        f"현재는 반복 구조 {distinct_signatures:,}개로 추세 관찰이 우선입니다."
    )


def _aggregate_level_insights(
    *,
    level_type: str,
    level_id: str,
    level_df: pd.DataFrame,
    representative_index: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if level_df.empty:
        empty = pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS)
        return empty.copy(), empty

    grouped = level_df.groupby(["journey_stage", "primary_axis", "polarity_direction"], dropna=False)
    level_total = int(len(level_df))

    rows: list[dict[str, Any]] = []
    for (journey_stage, primary_axis, polarity_direction), cluster in grouped:
        stage = _safe_text(journey_stage)
        axis = _safe_text(primary_axis)
        polarity = _normalize_polarity(polarity_direction)
        cluster_size = int(len(cluster))
        cluster_ratio = cluster_size / max(level_total, 1)
        mixed_ratio = float(cluster["mixed_flag"].mean()) if not cluster.empty else 0.0
        impact = _impact_value(cluster)
        distinct_signatures = int(cluster["problem_signature"].nunique()) if "problem_signature" in cluster.columns else 1
        priority = _priority_signal(
            cluster_size=cluster_size,
            cluster_ratio=cluster_ratio,
            impact_value=impact,
            polarity_direction=polarity,
            mixed_ratio=mixed_ratio,
        )

        rep_candidates = representative_index.copy()
        if level_type == "video":
            rep_candidates = rep_candidates[rep_candidates.get("video_id", pd.Series("", index=rep_candidates.index)).map(_safe_text) == level_id]
        else:
            rep_candidates = rep_candidates[rep_candidates.get("product", pd.Series("", index=rep_candidates.index)).map(_safe_text) == level_id]
        if not rep_candidates.empty:
            rep_candidates = rep_candidates[
                rep_candidates.get("journey_stage", pd.Series("", index=rep_candidates.index)).map(_safe_text).eq(stage)
                & rep_candidates.get("judgment_axes", pd.Series([[] for _ in range(len(rep_candidates))], index=rep_candidates.index)).map(lambda axes: axis in _normalize_axes(axes))
                & rep_candidates.get("polarity_direction", pd.Series("neutral", index=rep_candidates.index)).map(_normalize_polarity).eq(polarity)
            ]
        rep_ids = rep_candidates.get("comment_id", pd.Series(dtype=object)).map(_safe_text).tolist()[:5]

        insufficient_reasons: list[str] = []
        if cluster_size < 3:
            insufficient_reasons.append("cluster_lt_3")
        if not stage:
            insufficient_reasons.append("journey_stage_missing")
        if axis in {"", "unknown_axis"}:
            insufficient_reasons.append("judgment_axis_missing")
        if not rep_ids:
            insufficient_reasons.append("representative_link_missing")
        insufficient = bool(insufficient_reasons)

        evidence_summary = (
            f"동일 신호 {cluster_size:,}건 / 레벨 내 비중 {(cluster_ratio * 100.0):.1f}% / "
            f"문제 시그니처 {distinct_signatures:,}개 / 영향지표 {impact:.1f}"
        )

        row = {
            "level_type": level_type,
            "level_id": level_id,
            "video_id": level_id if level_type == "video" else "",
            "product_group": level_id if level_type == "product_group" else _safe_text(cluster.get("product", pd.Series("", index=cluster.index)).mode().iloc[0] if "product" in cluster.columns and not cluster["product"].mode().empty else ""),
            "journey_stage": stage,
            "judgment_axes": _serialize_list([axis] if axis else []),
            "insight_title": _insight_title(stage, axis, polarity),
            "insight_summary": _insight_summary(
                stage=stage,
                axis=axis,
                polarity=polarity,
                cluster_size=cluster_size,
                level_total=level_total,
                distinct_signatures=distinct_signatures,
            ),
            "priority_signal": priority,
            "evidence_summary": evidence_summary,
            "supporting_representatives": _serialize_list(rep_ids),
            "supporting_cluster_count": cluster_size,
            "insufficient_evidence": insufficient,
            "polarity_direction": polarity,
            "mixed_ratio": round(mixed_ratio, 4),
            "repetition_signal": cluster_size,
            "impact_signal": round(impact, 2),
            "insufficient_reasons": _serialize_list(insufficient_reasons),
        }
        rows.append(row)

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        empty = pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS)
        return empty.copy(), empty

    candidates = candidates.sort_values(
        ["insufficient_evidence", "priority_signal", "supporting_cluster_count"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    insights = candidates[candidates["insufficient_evidence"] == False].copy()  # noqa: E712
    return insights[REQUIRED_STRATEGY_COLUMNS].copy(), candidates[REQUIRED_STRATEGY_COLUMNS + ["insufficient_reasons"]].copy()


def build_strategy_insight_aggregates(
    analysis_non_trash: pd.DataFrame,
    representative_comments: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build canonical strategy insights at video/product-group aggregate levels."""
    base = _prepare_base_frame(analysis_non_trash)
    reps = _prepare_representative_index(representative_comments)

    if base.empty:
        empty = pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS)
        return {
            "strategy_video_insights": empty.copy(),
            "strategy_video_insight_candidates": empty.copy(),
            "strategy_product_group_insights": empty.copy(),
            "strategy_product_group_insight_candidates": empty.copy(),
        }

    video_insights_frames: list[pd.DataFrame] = []
    video_candidate_frames: list[pd.DataFrame] = []
    for level_id, group in base.groupby(base.get("video_id", pd.Series("", index=base.index)).map(_safe_text), dropna=False):
        if not level_id:
            continue
        insights, candidates = _aggregate_level_insights(
            level_type="video",
            level_id=level_id,
            level_df=group,
            representative_index=reps,
        )
        if not insights.empty:
            video_insights_frames.append(insights)
        if not candidates.empty:
            video_candidate_frames.append(candidates)

    product_insights_frames: list[pd.DataFrame] = []
    product_candidate_frames: list[pd.DataFrame] = []
    product_series = base.get("product", pd.Series("", index=base.index)).map(_safe_text)
    for level_id, group in base.groupby(product_series, dropna=False):
        if not level_id:
            continue
        insights, candidates = _aggregate_level_insights(
            level_type="product_group",
            level_id=level_id,
            level_df=group,
            representative_index=reps,
        )
        if not insights.empty:
            product_insights_frames.append(insights)
        if not candidates.empty:
            product_candidate_frames.append(candidates)

    video_insights = pd.concat(video_insights_frames, ignore_index=True) if video_insights_frames else pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS)
    video_candidates = pd.concat(video_candidate_frames, ignore_index=True) if video_candidate_frames else pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS + ["insufficient_reasons"])
    product_insights = pd.concat(product_insights_frames, ignore_index=True) if product_insights_frames else pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS)
    product_candidates = pd.concat(product_candidate_frames, ignore_index=True) if product_candidate_frames else pd.DataFrame(columns=REQUIRED_STRATEGY_COLUMNS + ["insufficient_reasons"])

    return {
        "strategy_video_insights": video_insights,
        "strategy_video_insight_candidates": video_candidates,
        "strategy_product_group_insights": product_insights,
        "strategy_product_group_insight_candidates": product_candidates,
    }
