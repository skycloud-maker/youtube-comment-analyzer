from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from src.dashboard_app import _build_canonical_similar_comments, _build_resolution_point


NOISE_PATTERNS = [
    r"https?://",
    r"bit\.ly",
    r"affiliate",
    r"sponsored",
    r"promo code",
    r"discount code",
    r"joint purchase",
    r"investment",
    r"stock",
    r"subscribe event",
]

MEANINGLESS_TOKEN_PATTERNS = [
    r"^\d+$",
    r"^\d+pcs$",
    r"^[khyu]+$",
]

GENERIC_ACTION_PATTERNS = [
    "organize the criteria",
    "strengthen guidance",
    "improve wording",
    "improve process",
    "track as priority",
]

NON_CUSTOMER_PATTERNS = [
    r"don't even have",
    r"just good content",
    r"watching for fun",
    r"not using this product",
]

SENTIMENT_DIR = {"negative": "negative", "positive": "positive", "neutral": "neutral", "mixed": "mixed"}


@dataclass
class SectionResult:
    name: str
    status: str
    summary: str
    details: dict[str, Any]


def parse_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
    return [v.strip() for v in re.split(r"[,;/|]", text) if v.strip()]


def pick_text(row: pd.Series) -> str:
    for key in ("text_display", "cleaned_text", "text_original", "text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def is_noise(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in NOISE_PATTERNS)


def load_df(run_dir: Path, name: str) -> pd.DataFrame:
    path = run_dir / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def sample_rows(df: pd.DataFrame, cols: list[str], n: int) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in df.head(n).iterrows():
        item: dict[str, Any] = {}
        for col in cols:
            item[col] = row.get(col)
        out.append(item)
    return out


def section_data_hygiene(
    comment_truth: pd.DataFrame,
    representatives: pd.DataFrame,
    nlp_keyword: pd.DataFrame,
    nlp_inquiry: pd.DataFrame,
    strategy_video: pd.DataFrame,
) -> SectionResult:
    rep_noise = representatives[representatives.apply(lambda r: is_noise(pick_text(r)), axis=1)] if not representatives.empty else pd.DataFrame()

    inquiry_noise = pd.DataFrame()
    if not nlp_inquiry.empty and "text" in nlp_inquiry.columns:
        inquiry_noise = nlp_inquiry[nlp_inquiry["text"].astype(str).str.lower().map(is_noise)]

    strategy_noise_refs: list[dict[str, Any]] = []
    if not strategy_video.empty and "supporting_representatives" in strategy_video.columns:
        hygiene_map = {}
        if not comment_truth.empty and "comment_id" in comment_truth.columns and "hygiene_class" in comment_truth.columns:
            hygiene_map = {
                str(row["comment_id"]): str(row["hygiene_class"]).lower()
                for _, row in comment_truth[["comment_id", "hygiene_class"]].iterrows()
            }
        for _, row in strategy_video.iterrows():
            for cid in parse_list(row.get("supporting_representatives")):
                if hygiene_map.get(cid) == "trash":
                    strategy_noise_refs.append({"level_id": row.get("level_id"), "comment_id": cid})

    keyword_noise: list[dict[str, Any]] = []
    if not nlp_keyword.empty and "keyword" in nlp_keyword.columns:
        for _, row in nlp_keyword.iterrows():
            token = str(row.get("keyword", "")).strip().lower()
            if any(re.search(pat, token) for pat in MEANINGLESS_TOKEN_PATTERNS):
                keyword_noise.append({"keyword": token, "count": row.get("count", 0)})

    leak_count = len(rep_noise) + len(inquiry_noise) + len(strategy_noise_refs)
    status = "PASS" if leak_count == 0 else "PARTIAL"
    return SectionResult(
        name="Data Hygiene Validation",
        status=status,
        summary="Noise leakage check for representative/keyword/inquiry/strategy paths",
        details={
            "representative_noise_count": int(len(rep_noise)),
            "inquiry_noise_count": int(len(inquiry_noise)),
            "strategy_noise_reference_count": int(len(strategy_noise_refs)),
            "noise_examples": sample_rows(
                pd.concat([rep_noise.assign(source="representative"), inquiry_noise.assign(source="inquiry")], ignore_index=True),
                ["source", "comment_id", "text_display", "text"],
                10,
            ),
            "keyword_noise_examples": keyword_noise[:10],
        },
    )


def section_representative_quality(representatives: pd.DataFrame) -> SectionResult:
    reps = representatives.copy()
    reps["text"] = reps.apply(pick_text, axis=1)
    reps["is_noise"] = reps["text"].map(is_noise)
    reps["is_short"] = reps["text"].str.len().fillna(0) < 35
    reps["has_journey"] = reps["journey_stage"].astype(str).str.len() > 0 if "journey_stage" in reps.columns else False
    reps["has_axes"] = reps["judgment_axes"].map(lambda v: len(parse_list(v)) > 0) if "judgment_axes" in reps.columns else False
    reps["non_customer_voice"] = reps["text"].str.lower().map(lambda t: any(re.search(p, t) for p in NON_CUSTOMER_PATTERNS))

    good = reps[~reps["is_noise"] & ~reps["is_short"] & reps["has_journey"] & reps["has_axes"] & ~reps["non_customer_voice"]]
    problematic = reps[
        reps["is_noise"] | reps["is_short"] | ~reps["has_journey"] | ~reps["has_axes"] | reps["non_customer_voice"]
    ]
    duplicate_count = int(reps["text"].duplicated().sum())

    status = "PASS" if len(problematic) == 0 and duplicate_count == 0 else "PARTIAL"
    return SectionResult(
        name="Representative Comment Quality",
        status=status,
        summary="Check representativeness of customer voice and experience/decision signal",
        details={
            "good_examples": sample_rows(good, ["comment_id", "sentiment_final", "journey_stage", "judgment_axes", "text"], 5),
            "problematic_examples": sample_rows(problematic, ["comment_id", "sentiment_final", "journey_stage", "judgment_axes", "text"], 5),
            "duplicate_text_count": duplicate_count,
            "non_customer_voice_count": int(reps["non_customer_voice"].sum()),
        },
    )


def section_similar_integrity(analysis_non_trash: pd.DataFrame, representatives: pd.DataFrame) -> SectionResult:
    reps = representatives.copy()
    reps["text"] = reps.apply(pick_text, axis=1)

    valid_clusters: list[dict[str, Any]] = []
    invalid_clusters: list[dict[str, Any]] = []
    count_checks: list[dict[str, Any]] = []
    broad_clusters: list[dict[str, Any]] = []

    for _, rep in reps.iterrows():
        sentiment = str(rep.get("sentiment_final", rep.get("sentiment_label", "neutral")))
        similar_df, meta = _build_canonical_similar_comments(
            analysis_non_trash, rep, rep, sentiment, str(rep.get("comment_id", ""))
        )
        cluster_size = int(meta.get("cluster_size", len(similar_df)))
        csv_rows = len(pd.read_csv(StringIO(similar_df.to_csv(index=False)))) if not similar_df.empty else 0
        count_checks.append(
            {
                "comment_id": rep.get("comment_id"),
                "card_count": cluster_size,
                "expander_count": len(similar_df),
                "csv_count": csv_rows,
                "match": cluster_size == len(similar_df) == csv_rows,
                "insufficient_evidence": bool(meta.get("insufficient_evidence", cluster_size < 3)),
                "reason": meta.get("reason"),
            }
        )

        if cluster_size < 3 or bool(meta.get("insufficient_evidence", False)):
            if len(invalid_clusters) < 3:
                invalid_clusters.append(
                    {
                        "comment_id": rep.get("comment_id"),
                        "cluster_size": cluster_size,
                        "insufficient_evidence": True,
                        "reason": meta.get("reason"),
                    }
                )
            continue

        sim_eval = similar_df.merge(
            analysis_non_trash[["comment_id", "journey_stage", "sentiment_final", "judgment_axes"]],
            on="comment_id",
            how="left",
        )

        rep_journey = str(rep.get("journey_stage", ""))
        rep_dir = SENTIMENT_DIR.get(str(rep.get("sentiment_final", "")).lower(), str(rep.get("sentiment_final", "")))
        rep_axes = set(parse_list(rep.get("judgment_axes", [])))

        same_journey = sim_eval["journey_stage"].astype(str).eq(rep_journey).all()
        sim_dirs = sim_eval["sentiment_final"].astype(str).str.lower().map(lambda x: SENTIMENT_DIR.get(x, x)).unique().tolist()
        same_sentiment_dir = len(set(sim_dirs + [rep_dir])) == 1

        same_intent = True
        if rep_axes:
            for axes in sim_eval["judgment_axes"].tolist():
                if not rep_axes.intersection(set(parse_list(axes))):
                    same_intent = False
                    break

        over_broad = cluster_size >= max(int(len(analysis_non_trash) * 0.08), 800)
        row = {
            "comment_id": rep.get("comment_id"),
            "cluster_size": cluster_size,
            "same_journey": bool(same_journey),
            "same_sentiment_direction": bool(same_sentiment_dir),
            "same_intent_overlap": bool(same_intent),
            "over_broad_cluster": bool(over_broad),
            "reason": meta.get("reason"),
        }
        if over_broad:
            broad_clusters.append(row)
        if same_journey and same_sentiment_dir and same_intent and not over_broad and len(valid_clusters) < 3:
            valid_clusters.append(row)
        elif len(invalid_clusters) < 3:
            invalid_clusters.append(row)

    mismatches = [item for item in count_checks if not item["match"]]
    if mismatches:
        status = "FAIL"
    elif broad_clusters or len(valid_clusters) < 3:
        status = "PARTIAL"
    else:
        status = "PASS"

    return SectionResult(
        name="Similar Comments Integrity",
        status=status,
        summary="Validate single-rule clustering and card=expander=CSV parity",
        details={
            "valid_clusters": valid_clusters,
            "incorrect_or_insufficient_clusters": invalid_clusters,
            "over_broad_cluster_examples": broad_clusters[:3],
            "count_mismatch_cases": mismatches[:10],
            "count_match_rate": round((len(count_checks) - len(mismatches)) / max(len(count_checks), 1), 4),
        },
    )


def section_resolution_quality(analysis_non_trash: pd.DataFrame, representatives: pd.DataFrame) -> SectionResult:
    reps = representatives.copy()
    action_texts: list[str] = []
    strong_examples: list[dict[str, Any]] = []
    weak_examples: list[dict[str, Any]] = []

    for _, rep in reps.iterrows():
        sentiment = str(rep.get("sentiment_final", rep.get("sentiment_label", "neutral")))
        similar_df, _ = _build_canonical_similar_comments(
            analysis_non_trash, rep, rep, sentiment, str(rep.get("comment_id", ""))
        )
        payload = _build_resolution_point(rep, sentiment, similar_comments=similar_df)
        action_text = str(payload.get("action_text", ""))
        action_texts.append(action_text)
        generic = any(pattern in action_text.lower() for pattern in GENERIC_ACTION_PATTERNS)
        insufficient = bool(payload.get("insufficient_evidence", False))

        row = {
            "comment_id": rep.get("comment_id"),
            "journey_stage": rep.get("journey_stage"),
            "action_domain": payload.get("action_domain"),
            "action_text": action_text,
            "insufficient_evidence": insufficient,
        }
        if not generic and not insufficient and len(strong_examples) < 3:
            strong_examples.append(row)
        if (generic or insufficient) and len(weak_examples) < 3:
            weak_examples.append(row)

    prefixes = [text[:36] for text in action_texts]
    prefix_counts = Counter(prefixes)
    max_prefix_repeat_ratio = (max(prefix_counts.values()) / len(prefixes)) if prefixes else 0.0
    distinct_action_ratio = len(set(action_texts)) / max(len(action_texts), 1)

    status = "PASS"
    if weak_examples or max_prefix_repeat_ratio >= 0.25:
        status = "PARTIAL"

    return SectionResult(
        name="Resolution Point Quality",
        status=status,
        summary="Check evidence-grounded actions and anti-template diversity",
        details={
            "strong_examples": strong_examples[:3],
            "weak_or_generic_examples": weak_examples[:3],
            "distinct_action_text_ratio": round(float(distinct_action_ratio), 4),
            "top_action_prefixes": dict(prefix_counts.most_common(5)),
            "max_prefix_repeat_ratio": round(float(max_prefix_repeat_ratio), 4),
        },
    )


def section_taxonomy(analysis_non_trash: pd.DataFrame, representatives: pd.DataFrame) -> SectionResult:
    journey_dist = (
        analysis_non_trash["journey_stage"].astype(str).value_counts(dropna=False).head(20).to_dict()
        if "journey_stage" in analysis_non_trash.columns
        else {}
    )

    axis_counter: Counter[str] = Counter()
    if "judgment_axes" in analysis_non_trash.columns:
        for value in analysis_non_trash["judgment_axes"].tolist():
            axis_counter.update(parse_list(value))
    top_axes = axis_counter.most_common(15)

    reps = representatives.copy()
    reps["text"] = reps.apply(pick_text, axis=1)
    reps["has_journey"] = reps["journey_stage"].astype(str).str.len() > 0 if "journey_stage" in reps.columns else False
    reps["has_axes"] = reps["judgment_axes"].map(lambda v: len(parse_list(v)) > 0) if "judgment_axes" in reps.columns else False
    correct = reps[reps["has_journey"] & reps["has_axes"]]
    incorrect = reps[~reps["has_journey"] | ~reps["has_axes"]]

    if incorrect.empty:
        incorrect = reps[
            reps["text"].str.contains("delivery|install|shipping", case=False, na=False)
            & ~reps["journey_stage"].astype(str).str.contains("delivery|onboarding", case=False, na=False)
        ]

    dominant_ratio = (top_axes[0][1] / sum(axis_counter.values())) if top_axes and sum(axis_counter.values()) > 0 else 0.0
    status = "PASS" if dominant_ratio < 0.75 and not correct.empty else "PARTIAL"

    return SectionResult(
        name="Taxonomy Validation",
        status=status,
        summary="Validate meaningful population and diversity of journey_stage + judgment_axes",
        details={
            "journey_distribution": journey_dist,
            "judgment_axis_top15": top_axes,
            "dominant_axis_ratio": round(float(dominant_ratio), 4),
            "correct_mapping_examples": sample_rows(correct, ["comment_id", "journey_stage", "judgment_axes", "text"], 5),
            "incorrect_mapping_examples": sample_rows(incorrect, ["comment_id", "journey_stage", "judgment_axes", "text"], 5),
        },
    )


def section_strategy(
    strategy_video: pd.DataFrame,
    strategy_video_candidates: pd.DataFrame,
    representatives: pd.DataFrame,
) -> SectionResult:
    valid = strategy_video[strategy_video["insufficient_evidence"] == False] if not strategy_video.empty else pd.DataFrame()  # noqa: E712
    problematic = strategy_video_candidates[strategy_video_candidates["insufficient_evidence"] == True] if not strategy_video_candidates.empty else pd.DataFrame()  # noqa: E712
    rep_map = {str(row["comment_id"]): pick_text(row) for _, row in representatives.iterrows()} if not representatives.empty else {}

    reword_suspects: list[dict[str, Any]] = []
    for _, row in valid.iterrows():
        summary = str(row.get("insight_summary", ""))
        for rid in parse_list(row.get("supporting_representatives")):
            source = rep_map.get(rid, "")
            if source and (source[:50] in summary or summary[:50] in source):
                reword_suspects.append({"level_id": row.get("level_id"), "rep_id": rid, "insight_summary": summary, "rep_text": source[:120]})
                break

    single_comment_generated = (
        bool((valid.get("supporting_cluster_count", pd.Series([], dtype=int)) < 2).any()) if not valid.empty else False
    )
    prefix_counts = Counter(valid["insight_summary"].astype(str).str.slice(0, 28).tolist()) if not valid.empty else Counter()
    max_prefix_repeat_ratio = (max(prefix_counts.values()) / len(valid)) if prefix_counts and len(valid) else 0.0

    status = "PASS"
    if valid.empty or single_comment_generated or reword_suspects or max_prefix_repeat_ratio >= 0.35:
        status = "PARTIAL"

    return SectionResult(
        name="Strategy Insight Validation",
        status=status,
        summary="Validate aggregate-only strategy output and insufficient-evidence guarding",
        details={
            "valid_insights": sample_rows(
                valid,
                ["level_id", "journey_stage", "judgment_axes", "insight_summary", "supporting_cluster_count", "priority_signal"],
                3,
            ),
            "problematic_insights": sample_rows(
                problematic,
                ["level_id", "journey_stage", "judgment_axes", "insufficient_evidence", "insight_summary"],
                3,
            ),
            "reworded_representative_suspects": reword_suspects[:3],
            "single_comment_generated": single_comment_generated,
            "top_insight_prefixes": dict(prefix_counts.most_common(5)),
            "max_prefix_repeat_ratio": round(float(max_prefix_repeat_ratio), 4),
        },
    )


def section_cross_level(strategy_video: pd.DataFrame, strategy_product: pd.DataFrame) -> SectionResult:
    comparison_case: dict[str, Any] = {}
    if not strategy_video.empty and not strategy_product.empty:
        for product_group, gdf in strategy_video.groupby("product_group"):
            videos = gdf["video_id"].dropna().astype(str).unique().tolist()
            if len(videos) < 2:
                continue
            a = gdf[gdf["video_id"] == videos[0]].head(3)
            b = gdf[gdf["video_id"] == videos[1]].head(3)
            if a.empty or b.empty:
                continue
            comparison_case = {
                "product_group": product_group,
                "video_a": videos[0],
                "video_b": videos[1],
                "video_a_journeys": a["journey_stage"].tolist(),
                "video_b_journeys": b["journey_stage"].tolist(),
                "video_a_insights": a["insight_summary"].tolist(),
                "video_b_insights": b["insight_summary"].tolist(),
            }
            break

    status = "PASS" if comparison_case else "PARTIAL"
    return SectionResult(
        name="Cross-Level Consistency",
        status=status,
        summary="Validate same product-group yields different video-level outputs",
        details={"comparison_case": comparison_case},
    )


def section_traceability(strategy_video: pd.DataFrame) -> SectionResult:
    traces: list[dict[str, Any]] = []
    if not strategy_video.empty:
        valid = strategy_video[strategy_video["insufficient_evidence"] == False]  # noqa: E712
        for _, row in valid.head(20).iterrows():
            rep_ids = parse_list(row.get("supporting_representatives"))
            if not rep_ids:
                continue
            traces.append(
                {
                    "level_id": row.get("level_id"),
                    "journey_stage": row.get("journey_stage"),
                    "insight_summary": row.get("insight_summary"),
                    "supporting_representatives": rep_ids[:5],
                    "supporting_cluster_count": int(row.get("supporting_cluster_count", 0) or 0),
                }
            )
            if len(traces) >= 2:
                break
    status = "PASS" if len(traces) >= 2 else "FAIL"
    return SectionResult(
        name="Traceability Validation",
        status=status,
        summary="Validate trace from strategy insight to representatives and clusters",
        details={"traceability_examples": traces},
    )


def build_sections(run_dir: Path) -> list[SectionResult]:
    comment_truth = load_df(run_dir, "comment_truth")
    analysis_non_trash = load_df(run_dir, "analysis_non_trash")
    representatives = load_df(run_dir, "representative_comments")
    nlp_keyword = load_df(run_dir, "nlp_keyword_insights")
    nlp_inquiry = load_df(run_dir, "nlp_inquiry_summary")
    strategy_video = load_df(run_dir, "strategy_video_insights")
    strategy_video_candidates = load_df(run_dir, "strategy_video_insight_candidates")
    strategy_product = load_df(run_dir, "strategy_product_group_insights")

    return [
        section_data_hygiene(comment_truth, representatives, nlp_keyword, nlp_inquiry, strategy_video),
        section_representative_quality(representatives),
        section_similar_integrity(analysis_non_trash, representatives),
        section_resolution_quality(analysis_non_trash, representatives),
        section_taxonomy(analysis_non_trash, representatives),
        section_strategy(strategy_video, strategy_video_candidates, representatives),
        section_cross_level(strategy_video, strategy_product),
        section_traceability(strategy_video),
    ]


def overall_verdict(sections: list[SectionResult]) -> str:
    statuses = [s.status for s in sections]
    if all(status == "PASS" for status in statuses):
        return "PASS"
    if any(status == "FAIL" for status in statuses):
        return "FAIL"
    return "PARTIAL"


def top_risks(sections: list[SectionResult]) -> list[str]:
    return [f"{s.name}: {s.summary}" for s in sections if s.status in {"PARTIAL", "FAIL"}][:5]


def run_validation(run_id: str, base_dir: Path) -> dict[str, Any]:
    run_dir = base_dir / "data" / "processed" / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    sections = build_sections(run_dir)
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "overall_verdict": overall_verdict(sections),
        "sections": [asdict(s) for s in sections],
        "critical_risks_top5": top_risks(sections),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="PHASE 6.7 Automated QA Validator")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parents[1]
    payload = run_validation(args.run_id, base_dir)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
