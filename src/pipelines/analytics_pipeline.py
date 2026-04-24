"""Analytics pipeline for descriptive and NLP outputs."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.analytics.brands import infer_brand
from src.analytics.comment_analysis import analyze_comment_with_context
from src.analytics.comment_truth import (
    CANONICAL_COMMENT_TRUTH_COLUMNS,
    FIXED_JOURNEY_STAGES,
    build_canonical_comment_truth,
)
from src.analytics.insight_engine import (
    build_inquiry_summary,
    build_keyword_insight_summary,
    build_nlp_sentiment_summary,
    build_product_mention_summary,
    build_topic_insight_summary,
)
from src.analytics.opinion_units import (
    build_opinion_units,
    build_representative_bundles,
    build_representative_section_payload,
    build_source_records,
)
from src.analytics.clustering import assign_clusters
from src.analytics.descriptive import build_descriptive_metrics
from src.analytics.keywords import extract_keywords
from src.analytics.sentiment import NLP_ANALYZER_AVAILABLE, _build_nlp_router, score_sentiment
from src.analytics.strategy_aggregator import build_strategy_insight_aggregates
from src.analytics.topics import infer_topic
from src.analytics.trends import build_issue_keyword_status, build_weekly_keyword_trend, build_weekly_sentiment_trend
from src.utils.io import ensure_dir, write_dataframe


def _canonical_representative_text(text: str) -> str:
    normalized = re.sub(r"[^\w\s\uac00-\ud7a3]", " ", str(text or "").lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized[:120]


def _looks_like_inquiry(text: str) -> bool:
    source = str(text or "")
    lowered = source.lower()
    return "?" in source or any(token in lowered for token in ["\uC5B4\uB5BB\uAC8C", "\uC5B4\uB514", "\uC65C", "\uAC00\uB2A5\uD55C\uAC00", "\uAC00\uB2A5\uD560\uAE4C\uC694", "how", "where", "why", "can i", "does it"])



class AnalyticsPipeline:
    """Compute business-oriented analysis outputs."""

    def __init__(
        self,
        processed_dir: Path,
        risk_keywords: list[str],
        top_n_keywords: int,
        topic_count: int,
        logger: logging.Logger | None = None,
        nlp_analyzer_config: dict | None = None,
        video_selection_config: dict | None = None,
    ) -> None:
        self.processed_dir = ensure_dir(processed_dir)
        self.risk_keywords = risk_keywords
        self.top_n_keywords = top_n_keywords
        self.topic_count = topic_count
        self.logger = logger or logging.getLogger(__name__)
        self.nlp_config = nlp_analyzer_config or {}
        self.video_selection = video_selection_config or {}
        self.video_selection_enabled = bool(self.video_selection.get("enabled", True))
        self.video_target_brand_terms = [str(item).lower() for item in self.video_selection.get("target_brand_terms", []) if str(item).strip()]
        self.video_competitor_terms = [str(item).lower() for item in self.video_selection.get("competitor_brand_terms", []) if str(item).strip()]
        self.video_product_terms = [str(item).lower() for item in self.video_selection.get("product_terms", []) if str(item).strip()]
        self.video_preferred_channels = [str(item).lower() for item in self.video_selection.get("preferred_channel_keywords", []) if str(item).strip()]
        self.video_blocked_channels = [str(item).lower() for item in self.video_selection.get("blocked_channel_keywords", []) if str(item).strip()]
        self.video_require_product = bool(self.video_selection.get("require_product_term", True))
        self.video_require_target = bool(self.video_selection.get("require_target_brand", False))
        self.video_exclude_competitors = bool(self.video_selection.get("exclude_competitor_brands", True))
        self.use_nlp = self.nlp_config.get("enabled", True) and NLP_ANALYZER_AVAILABLE
        self.nlp_provider = None
        if self.use_nlp:
            self.nlp_provider = _build_nlp_router(
                provider=self.nlp_config.get("provider", "claude"),
                model=self.nlp_config.get("model", "claude-sonnet-4-20250514"),
                fallback_provider=self.nlp_config.get("fallback_provider", "openai"),
                fallback_model=self.nlp_config.get("fallback_model", "gpt-4o-mini"),
            )
            self.logger.info("nlp_analyzer enabled (provider=%s)", self.nlp_config.get("provider", "claude"))

    def run(self, run_id: str, videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        videos_with_relevance = self._annotate_video_relevance(videos_df)
        working_comments = self._attach_video_context(comments_df, videos_with_relevance)
        for column in ["product", "region", "title", "video_url"]:
            if column not in working_comments.columns:
                working_comments[column] = pd.NA
        analysis_comments = working_comments[working_comments.get("analysis_included", False) == True].copy() if not working_comments.empty else working_comments.copy()
        if not analysis_comments.empty:
            parent_lookup = analysis_comments.set_index("comment_id")["text_display"].to_dict() if "comment_id" in analysis_comments.columns and "text_display" in analysis_comments.columns else {}
            analysis_comments["parent_comment_text"] = analysis_comments.get("parent_comment_id", pd.Series(index=analysis_comments.index, dtype=object)).map(lambda value: parent_lookup.get(value, "") if pd.notna(value) else "")
            _nlp_provider = self.nlp_provider
            _use_nlp = self.use_nlp
            analysis_results = analysis_comments.apply(
                lambda row: analyze_comment_with_context(
                    row.get("cleaned_text", ""),
                    validity=row.get("comment_validity", "valid"),
                    exclusion_reason=row.get("exclusion_reason"),
                    parent_comment=row.get("parent_comment_text", ""),
                    context_comments=["" if pd.isna(row.get("title", pd.NA)) else str(row.get("title", "")), "" if pd.isna(row.get("product", pd.NA)) else str(row.get("product", ""))],
                    is_reply=bool(row.get("is_reply", False)),
                    comment_id=str(row.get("comment_id", "")),
                    nlp_provider=_nlp_provider,
                    use_nlp=_use_nlp,
                    video_context={
                        "title": "" if pd.isna(row.get("title", pd.NA)) else str(row.get("title", "")),
                        "product": "" if pd.isna(row.get("product", pd.NA)) else str(row.get("product", "")),
                        "channel": "" if pd.isna(row.get("channel_title", pd.NA)) else str(row.get("channel_title", "")),
                        "video_lg_relevance_score": 0.0 if pd.isna(row.get("video_lg_relevance_score", pd.NA)) else float(row.get("video_lg_relevance_score", 0.0) or 0.0),
                        "is_lg_relevant_video": bool(row.get("is_lg_relevant_video", False)),
                    },
                ),
                axis=1,
            )
            # Core analysis fields
            analysis_comments["product_related"] = analysis_results.map(lambda item: item.product_related)
            analysis_comments["product_target"] = analysis_results.map(lambda item: item.product_target)
            analysis_comments["sentiment_label"] = analysis_results.map(lambda item: item.sentiment_label)
            analysis_comments["sentiment_score"] = analysis_results.map(lambda item: item.sentiment_score)
            analysis_comments["signal_strength"] = analysis_results.map(lambda item: item.signal_strength)
            analysis_comments["classification_type"] = analysis_results.map(lambda item: item.classification_type)
            analysis_comments["confidence_level"] = analysis_results.map(lambda item: item.confidence_level)
            analysis_comments["context_used"] = analysis_results.map(lambda item: item.context_used)
            analysis_comments["context_required_for_display"] = analysis_results.map(lambda item: item.context_required_for_display)
            analysis_comments["classification_reason"] = analysis_results.map(lambda item: item.reason)
            analysis_comments["needs_review"] = analysis_results.map(lambda item: item.needs_review)
            analysis_comments["insight_type"] = analysis_results.map(lambda item: item.insight_type)
            analysis_comments["lg_relevance_score"] = analysis_results.map(lambda item: item.lg_relevance_score)
            analysis_comments["lg_relevant_comment"] = analysis_results.map(lambda item: item.lg_relevant_comment)
            # nlp_analyzer enrichment fields
            analysis_comments["nlp_label"] = analysis_results.map(lambda item: item.nlp_label)
            analysis_comments["nlp_confidence"] = analysis_results.map(lambda item: item.nlp_confidence)
            analysis_comments["nlp_sentiment_reason"] = analysis_results.map(lambda item: item.nlp_sentiment_reason)
            analysis_comments["nlp_topics"] = analysis_results.map(lambda item: item.nlp_topics)
            analysis_comments["nlp_topic_sentiments"] = analysis_results.map(lambda item: item.nlp_topic_sentiments)
            analysis_comments["nlp_is_inquiry"] = analysis_results.map(lambda item: bool(item.nlp_is_inquiry))
            analysis_comments["nlp_is_rhetorical"] = analysis_results.map(lambda item: bool(item.nlp_is_rhetorical))
            analysis_comments["nlp_summary"] = analysis_results.map(lambda item: item.nlp_summary)
            analysis_comments["nlp_keywords"] = analysis_results.map(lambda item: item.nlp_keywords)
            analysis_comments["nlp_product_mentions"] = analysis_results.map(lambda item: item.nlp_product_mentions)
            analysis_comments["nlp_language"] = analysis_results.map(lambda item: item.nlp_language)
            analysis_comments["nlp_provider"] = analysis_results.map(lambda item: item.nlp_provider)
            analysis_comments["nlp_model"] = analysis_results.map(lambda item: item.nlp_model)
            analysis_comments["nlp_core_points"] = analysis_results.map(lambda item: item.nlp_core_points)
            analysis_comments["nlp_context_tags"] = analysis_results.map(lambda item: item.nlp_context_tags)
            analysis_comments["nlp_similarity_keys"] = analysis_results.map(lambda item: item.nlp_similarity_keys)
            analysis_comments["nlp_insight_summary"] = analysis_results.map(lambda item: item.nlp_insight_summary)
            analysis_comments["nlp_confidence_factors"] = analysis_results.map(lambda item: item.nlp_confidence_factors)
            analysis_comments["nlp_confidence_breakdown"] = analysis_results.map(lambda item: item.nlp_confidence_breakdown)
            analysis_comments["nlp_sentiment_intensity"] = analysis_results.map(lambda item: item.nlp_sentiment_intensity)
            analysis_comments["nlp_user_wants"] = analysis_results.map(lambda item: item.nlp_user_wants)
            representative_meta = analysis_comments.apply(self._representative_metadata, axis=1, result_type="expand")
            analysis_comments = pd.concat([analysis_comments, representative_meta], axis=1)
            analysis_comments["topic_label"] = analysis_comments["cleaned_text"].map(infer_topic)
            title_series = analysis_comments["title"].fillna("") if "title" in analysis_comments.columns else pd.Series("", index=analysis_comments.index)
            analysis_comments["brand_label"] = (analysis_comments["cleaned_text"].fillna("") + " " + title_series).map(infer_brand)
            analysis_comments = assign_clusters(analysis_comments, self.topic_count)
            analysis_comments["spam_like"] = analysis_comments.duplicated(subset=["video_id", "cleaned_text"], keep=False)
            analysis_comments["risk_flags"] = analysis_comments["cleaned_text"].map(self._flag_risks)
            _merge_cols = [
                "comment_id", "product_related", "product_target", "sentiment_label", "sentiment_score",
                "signal_strength", "classification_type", "confidence_level", "topic_label", "brand_label",
                "cluster_id", "spam_like", "risk_flags", "context_used", "context_required_for_display",
                "classification_reason", "needs_review", "insight_type", "parent_comment_text",
                "representative_score", "clarity_score", "product_specificity_score",
                "insight_value_score", "representativeness_score", "explainability_score",
                "context_independence_score", "representative_decision", "representative_reason",
                "lg_relevance_score", "lg_relevant_comment",
                "nlp_label", "nlp_confidence", "nlp_sentiment_reason", "nlp_topics",
                "nlp_topic_sentiments", "nlp_is_inquiry", "nlp_is_rhetorical", "nlp_summary",
                "nlp_keywords", "nlp_product_mentions", "nlp_language", "nlp_provider", "nlp_model",
                "nlp_core_points", "nlp_context_tags", "nlp_similarity_keys", "nlp_insight_summary",
                "nlp_confidence_factors", "nlp_confidence_breakdown", "nlp_sentiment_intensity",
                "nlp_user_wants",
            ]
            _available_merge_cols = [c for c in _merge_cols if c in analysis_comments.columns]
            working_comments = working_comments.merge(
                analysis_comments[_available_merge_cols],
                on="comment_id",
                how="left",
            )
        else:
            _empty_cols = [
                "product_related", "product_target", "sentiment_label", "sentiment_score",
                "signal_strength", "classification_type", "confidence_level", "topic_label",
                "brand_label", "cluster_id", "spam_like", "risk_flags", "context_used",
                "context_required_for_display", "classification_reason", "needs_review", "insight_type",
                "parent_comment_text", "representative_score", "clarity_score",
                "product_specificity_score", "insight_value_score", "representativeness_score",
                "explainability_score", "context_independence_score", "representative_decision",
                "representative_reason", "lg_relevance_score", "lg_relevant_comment",
                "nlp_label", "nlp_confidence", "nlp_sentiment_reason",
                "nlp_topics", "nlp_topic_sentiments", "nlp_is_inquiry", "nlp_is_rhetorical",
                "nlp_summary", "nlp_keywords", "nlp_product_mentions", "nlp_language",
                "nlp_provider", "nlp_model",
                "nlp_core_points", "nlp_context_tags", "nlp_similarity_keys", "nlp_insight_summary",
                "nlp_confidence_factors", "nlp_confidence_breakdown", "nlp_sentiment_intensity",
                "nlp_user_wants",
            ]
            for column in _empty_cols:
                working_comments[column] = pd.NA

        working_comments["sentiment_label"] = working_comments["sentiment_label"].where(
            working_comments.get("comment_validity", pd.Series(index=working_comments.index, dtype=object)).eq("valid"),
            pd.NA,
        )

        comment_truth = build_canonical_comment_truth(working_comments)
        working_comments = self._merge_canonical_truth(working_comments, comment_truth)
        analysis_comments = self._merge_canonical_truth(analysis_comments, comment_truth)
        analysis_included_series = (
            analysis_comments["analysis_included"]
            if "analysis_included" in analysis_comments.columns
            else pd.Series(False, index=analysis_comments.index)
        )
        analysis_comments = analysis_comments[analysis_included_series.fillna(False)].copy()
        if not analysis_comments.empty and "sentiment_final" in analysis_comments.columns:
            analysis_comments["sentiment_label"] = analysis_comments["sentiment_final"].fillna(analysis_comments.get("sentiment_label"))
            analysis_comments["journey_stage"] = analysis_comments.get("journey_stage", pd.Series(pd.NA, index=analysis_comments.index)).map(self._normalize_journey_stage_value)
            analysis_comments["judgment_axes"] = analysis_comments.get("judgment_axes", pd.Series([[] for _ in range(len(analysis_comments))], index=analysis_comments.index)).map(self._normalize_judgment_axes)
            analysis_comments["topic_label"] = analysis_comments.apply(self._topic_label_from_taxonomy, axis=1)
            working_comments = self._merge_taxonomy_backfill(working_comments, analysis_comments)

        hygiene_series = (
            analysis_comments["hygiene_class"]
            if "hygiene_class" in analysis_comments.columns
            else pd.Series("strategic", index=analysis_comments.index)
        )
        analysis_non_trash = analysis_comments[hygiene_series.fillna("trash").astype(str).str.lower().ne("trash")].copy()
        strategic_series = (
            analysis_non_trash["hygiene_class"]
            if "hygiene_class" in analysis_non_trash.columns
            else pd.Series("strategic", index=analysis_non_trash.index)
        )
        analysis_strategic = analysis_non_trash[strategic_series.fillna("").astype(str).str.lower().eq("strategic")].copy()

        metrics = build_descriptive_metrics(videos_with_relevance, working_comments)
        opinion_units = build_opinion_units(analysis_strategic)
        if not opinion_units.empty:
            source_counts = opinion_units.groupby("source_content_id").agg(opinion_count=("opinion_id", "count")).reset_index()
            analysis_non_trash = analysis_non_trash.merge(source_counts, left_on="comment_id", right_on="source_content_id", how="left")
            analysis_non_trash.drop(columns=["source_content_id"], inplace=True, errors="ignore")
            analysis_non_trash["source_count"] = 1
            analysis_non_trash["opinion_count"] = analysis_non_trash["opinion_count"].fillna(1).astype(int)
            analysis_comments = analysis_comments.merge(source_counts, left_on="comment_id", right_on="source_content_id", how="left")
            analysis_comments.drop(columns=["source_content_id"], inplace=True, errors="ignore")
            analysis_comments["source_count"] = 1
            analysis_comments["opinion_count"] = analysis_comments["opinion_count"].fillna(1).astype(int)
            working_comments = working_comments.merge(source_counts, left_on="comment_id", right_on="source_content_id", how="left")
            working_comments.drop(columns=["source_content_id"], inplace=True, errors="ignore")
            working_comments["source_count"] = 1
            working_comments["opinion_count"] = working_comments["opinion_count"].fillna(1).astype(int)
        else:
            analysis_non_trash["source_count"] = 1
            analysis_non_trash["opinion_count"] = 1
            analysis_comments["source_count"] = 1
            analysis_comments["opinion_count"] = 1
            working_comments["source_count"] = 1
            working_comments["opinion_count"] = 1

        representative_bundles = build_representative_bundles(opinion_units)
        source_records = build_source_records(opinion_units)
        representative_section = build_representative_section_payload(representative_bundles)
        monitoring_summary = self._monitoring_summary(opinion_units)
        reporting_summary = self._reporting_summary(opinion_units)
        keyword_df = extract_keywords(analysis_strategic["cleaned_text"] if not analysis_strategic.empty else pd.Series(dtype=str), self.top_n_keywords)
        sentiment_summary = self._summary_by_label(analysis_non_trash, "sentiment_label")
        topic_summary = self._summary_by_label(analysis_non_trash, "topic_label")
        representative_comments = self._representative_comments(analysis_strategic)
        outlier_comments = analysis_non_trash.sort_values(["like_count", "text_length"], ascending=False).head(20) if not analysis_non_trash.empty else pd.DataFrame()
        weekly_sentiment_trend = build_weekly_sentiment_trend(analysis_non_trash)
        weekly_keyword_trend = build_weekly_keyword_trend(analysis_non_trash)
        new_issue_keywords, persistent_issue_keywords = build_issue_keyword_status(weekly_keyword_trend)
        brand_ratio = self._brand_ratio(analysis_non_trash)
        cej_negative_rate = self._cej_negative_rate(analysis_non_trash)
        negative_density = self._negative_density(videos_with_relevance, analysis_non_trash)

        # insight_engine aggregation from nlp_analyzer results
        nlp_topic_insights = build_topic_insight_summary(analysis_non_trash)
        nlp_keyword_insights = build_keyword_insight_summary(analysis_non_trash)
        nlp_inquiry_summary = build_inquiry_summary(analysis_non_trash)
        nlp_product_mentions = build_product_mention_summary(analysis_non_trash)
        nlp_sentiment_dist = build_nlp_sentiment_summary(analysis_non_trash)
        strategy_outputs = build_strategy_insight_aggregates(
            analysis_non_trash=analysis_non_trash,
            representative_comments=representative_comments,
        )

        outputs = {
            **metrics,
            "comments": working_comments,
            "comment_truth": comment_truth,
            "analysis_comments": analysis_comments,
            "analysis_non_trash": analysis_non_trash,
            "analysis_strategic": analysis_strategic,
            "opinion_units": opinion_units,
            "representative_bundles": representative_bundles,
            "monitoring_summary": monitoring_summary,
            "reporting_summary": reporting_summary,
            "keyword_analysis": keyword_df,
            "sentiment_summary": sentiment_summary,
            "topic_summary": topic_summary,
            "representative_comments": representative_comments,
            "outlier_comments": outlier_comments,
            "weekly_sentiment_trend": weekly_sentiment_trend,
            "weekly_keyword_trend": weekly_keyword_trend,
            "brand_ratio": brand_ratio,
            "cej_negative_rate": cej_negative_rate,
            "negative_density": negative_density,
            "new_issue_keywords": new_issue_keywords,
            "persistent_issue_keywords": persistent_issue_keywords,
            "nlp_topic_insights": nlp_topic_insights,
            "nlp_keyword_insights": nlp_keyword_insights,
            "nlp_inquiry_summary": nlp_inquiry_summary,
            "nlp_product_mentions": nlp_product_mentions,
            "nlp_sentiment_distribution": nlp_sentiment_dist,
            **strategy_outputs,
        }

        run_dir = ensure_dir(self.processed_dir / run_id)
        write_dataframe(videos_with_relevance, run_dir / "videos_normalized.parquet")
        for name, frame in outputs.items():
            if isinstance(frame, pd.DataFrame):
                write_dataframe(frame, run_dir / f"{name}.parquet")
        (run_dir / "source_records.json").write_text(json.dumps({"source_records": source_records}, ensure_ascii=False, indent=2), encoding="utf-8")
        (run_dir / "representative_section.json").write_text(json.dumps({"representative_section": representative_section}, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Analytics stage completed", extra={"run_id": run_id, "stage": "analytics"})
        return outputs

    @staticmethod
    def _attach_video_context(comments_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
        if comments_df.empty or videos_df.empty:
            return comments_df.copy()
        video_columns = [
            column
            for column in [
                "video_id",
                "product",
                "keyword",
                "region",
                "title",
                "channel_title",
                "video_url",
                "is_lg_relevant_video",
                "video_lg_relevance_score",
                "lg_target_brand_hit",
                "lg_product_hit",
                "lg_preferred_channel_hit",
            ]
            if column in videos_df.columns
        ]
        if not video_columns:
            return comments_df.copy()
        return comments_df.merge(videos_df[video_columns].drop_duplicates(subset=["video_id"]), on="video_id", how="left")

    @staticmethod
    def _merge_canonical_truth(frame: pd.DataFrame, comment_truth: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or comment_truth.empty or "comment_id" not in frame.columns:
            return frame
        merge_columns = [column for column in CANONICAL_COMMENT_TRUTH_COLUMNS if column not in {"comment_id", "video_id"}]
        working = frame.drop(columns=merge_columns, errors="ignore").copy()
        payload_columns = [column for column in ["comment_id", *merge_columns] if column in comment_truth.columns]
        return working.merge(comment_truth[payload_columns], on="comment_id", how="left")

    @staticmethod
    def _normalize_journey_stage_value(value: Any) -> str:
        text = str(value).strip().lower() if pd.notna(value) else ""
        alias = {
            "aware": "awareness",
            "explore": "exploration",
            "decide": "decision",
            "deliver": "delivery",
            "install": "delivery",
            "installation": "delivery",
            "onboard": "onboarding",
            "on-board": "onboarding",
            "use": "usage",
            "maintain": "maintenance",
            "replace": "replacement",
            "s1 aware": "awareness",
            "s2 explore": "exploration",
            "s4 purchase": "purchase",
            "s5 deliver": "delivery",
            "s6 on-board": "onboarding",
            "s7 use": "usage",
            "s8 maintain": "maintenance",
            "s9 replace": "replacement",
        }
        normalized = alias.get(text, text)
        return normalized if normalized in FIXED_JOURNEY_STAGES else ""

    @staticmethod
    def _normalize_judgment_axes(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, tuple):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return []
            if raw.startswith("[") and raw.endswith("]"):
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, list):
                        return [str(item).strip() for item in parsed if str(item).strip()]
                except Exception:
                    pass
            return [part.strip() for part in re.split(r"[\|,;/]+", raw) if part.strip()]
        return []

    @classmethod
    def _primary_judgment_axis(cls, row: pd.Series) -> str:
        axes = cls._normalize_judgment_axes(row.get("judgment_axes", []))
        if axes:
            return axes[0]
        fallback = row.get("topic_label", row.get("classification_type", "general"))
        return str(fallback).strip() if pd.notna(fallback) and str(fallback).strip() else "general"

    @classmethod
    def _topic_label_from_taxonomy(cls, row: pd.Series) -> str:
        axis = cls._primary_judgment_axis(row)
        if "_" not in axis:
            return axis
        return axis.replace("_", " ").title()

    @classmethod
    def _merge_taxonomy_backfill(cls, frame: pd.DataFrame, source: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or source.empty or "comment_id" not in frame.columns:
            return frame
        keep = source[[column for column in ["comment_id", "journey_stage", "judgment_axes", "topic_label"] if column in source.columns]].drop_duplicates(subset=["comment_id"])
        merged = frame.drop(columns=["journey_stage", "judgment_axes", "topic_label"], errors="ignore").merge(keep, on="comment_id", how="left")
        if "topic_label" in frame.columns:
            merged["topic_label"] = merged["topic_label"].fillna(frame.get("topic_label"))
        return merged

    def _annotate_video_relevance(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        if videos_df.empty:
            return videos_df.copy()
        working = videos_df.copy()
        if not self.video_selection_enabled:
            working["video_lg_relevance_score"] = 0.0
            working["is_lg_relevant_video"] = True
            working["lg_target_brand_hit"] = False
            working["lg_product_hit"] = False
            working["lg_preferred_channel_hit"] = False
            return working

        title = working.get("title", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
        description = working.get("description", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
        channel = working.get("channel_title", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
        keyword = working.get("keyword", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
        product = working.get("product", pd.Series("", index=working.index)).fillna("").astype(str).str.lower()
        tags_blob = working.get("tags", pd.Series([[] for _ in range(len(working))], index=working.index)).map(
            lambda value: " ".join(str(item) for item in (value if isinstance(value, list) else [value]))
        ).fillna("").astype(str).str.lower()
        blob = (title + " " + description + " " + channel + " " + keyword + " " + product + " " + tags_blob).str.strip()

        target_hit = blob.map(lambda text: self._contains_term(text, self.video_target_brand_terms))
        product_hit = blob.map(lambda text: self._contains_term(text, self.video_product_terms))
        competitor_hit = blob.map(lambda text: self._contains_term(text, self.video_competitor_terms))
        preferred_hit = channel.map(lambda text: self._contains_term(text, self.video_preferred_channels))
        blocked_hit = channel.map(lambda text: self._contains_term(text, self.video_blocked_channels))

        score = (
            target_hit.astype(float) * 0.45
            + product_hit.astype(float) * 0.35
            + preferred_hit.astype(float) * 0.30
        )
        if self.video_exclude_competitors:
            score = score - ((competitor_hit & ~target_hit).astype(float) * 0.40)
        score = score.clip(lower=0.0, upper=1.0)

        is_relevant = score >= 0.45
        if self.video_require_product:
            is_relevant = is_relevant & (product_hit | preferred_hit)
        if self.video_require_target:
            is_relevant = is_relevant & (target_hit | preferred_hit)
        is_relevant = is_relevant & (~blocked_hit)

        working["video_lg_relevance_score"] = score.round(3)
        working["is_lg_relevant_video"] = is_relevant
        working["lg_target_brand_hit"] = target_hit
        working["lg_product_hit"] = product_hit
        working["lg_preferred_channel_hit"] = preferred_hit
        return working

    @staticmethod
    def _contains_term(text: str, terms: list[str]) -> bool:
        lowered = str(text or "").lower()
        for raw in terms:
            term = str(raw or "").strip().lower()
            if term and term in lowered:
                return True
        return False

    def _flag_risks(self, text: str) -> str:
        lowered = (text or "").lower()
        hits = [keyword for keyword in self.risk_keywords if keyword.lower() in lowered]
        return ", ".join(hits)

    @staticmethod
    def _summary_by_label(comments_df: pd.DataFrame, column: str) -> pd.DataFrame:
        if comments_df.empty or column not in comments_df.columns:
            return pd.DataFrame(columns=[column, "count"])
        return comments_df.groupby(column).size().reset_index(name="count").sort_values("count", ascending=False)

    @staticmethod
    def _representative_metadata(row: pd.Series) -> pd.Series:
        text_display = row.get("text_display", "")
        cleaned_text = row.get("cleaned_text", "")
        text = str(text_display) if pd.notna(text_display) and str(text_display).strip() else (str(cleaned_text) if pd.notna(cleaned_text) else "")
        product_related_raw = row.get("product_related", False)
        product_related = bool(product_related_raw) if pd.notna(product_related_raw) else False
        sentiment_raw = row.get("sentiment_label", "")
        sentiment = str(sentiment_raw) if pd.notna(sentiment_raw) else ""
        classification_raw = row.get("classification_type", "informational")
        classification_type = str(classification_raw) if pd.notna(classification_raw) and str(classification_raw).strip() else "informational"
        confidence_raw = row.get("confidence_level", "low")
        confidence = str(confidence_raw) if pd.notna(confidence_raw) and str(confidence_raw).strip() else "low"
        context_required_raw = row.get("context_required_for_display", False)
        context_required = bool(context_required_raw) if pd.notna(context_required_raw) else False
        needs_review_raw = row.get("needs_review", False)
        needs_review = bool(needs_review_raw) if pd.notna(needs_review_raw) else False
        target_raw = row.get("product_target", "")
        target = str(target_raw) if pd.notna(target_raw) else ""

        text_len = len(text.strip())
        clarity = 5 if 25 <= text_len <= 220 else 4 if 12 <= text_len <= 280 else 2
        product_specificity = 5 if target else 4 if product_related else 2
        if sentiment in {"positive", "negative"} and classification_type == "complaint":
            insight_value = 5
        elif classification_type in {"comparison", "preference"}:
            insight_value = 3
        elif product_related:
            insight_value = 3
        else:
            insight_value = 1
        representativeness = 5 if confidence == "high" and not needs_review else 4 if confidence == "medium" else 2
        explainability = 5 if not context_required and confidence == "high" else 4 if not context_required else 2
        context_independence = 5 if not context_required else 2
        total = int(clarity + product_specificity + insight_value + representativeness + explainability + context_independence)

        if sentiment in {"positive", "negative"} and total >= 24 and not context_required and classification_type != "informational":
            decision = "strong_candidate"
        elif total >= 18 and product_related:
            decision = "review_needed"
        else:
            decision = "not_suitable"

        if decision == "strong_candidate":
            reason = "\uc758\ubbf8\uac00 \uba85\ud655\ud558\uace0 \uc81c\ud488 \uad00\ub828\uc131\uc774 \ub192\uc544, \uc758\uc0ac\uacb0\uc815\uc790\uc5d0\uac8c \ubc14\ub85c \ubcf4\uc5ec\uc918\ub3c4 \uc65c \uc911\uc694\ud55c\uc9c0 \uc774\ud574\ud558\uae30 \uc26c\uc6b4 \ub313\uae00\uc785\ub2c8\ub2e4."
        elif decision == "review_needed":
            reason = "\uc758\ubbf8\ub294 \uc788\uc9c0\ub9cc \ub9e5\ub77d \uc758\uc874\uc131\uc774\ub098 \ud574\uc11d \ubcf4\uac15\uc774 \uc870\uae08 \ub354 \ud544\uc694\ud55c \ub313\uae00\uc774\ub77c, \ub300\ud45c \ucf54\uba58\ud2b8\ub85c \uc4f0\uae30 \uc804\uc5d0 \uac80\ud1a0\uac00 \ud544\uc694\ud569\ub2c8\ub2e4."
        else:
            reason = "\ub9e5\ub77d \uc758\uc874\uc131\uc774 \ub192\uac70\ub098 \uc778\uc0ac\uc774\ud2b8\uac00 \uc57d\ud574, \ub300\ud45c \ucf54\uba58\ud2b8\ub85c \ubc14\ub85c \uc81c\uc2dc\ud558\uae30\uc5d0\ub294 \uc124\uba85\ub825\uc774 \ubd80\uc871\ud55c \ub313\uae00\uc785\ub2c8\ub2e4."

        return pd.Series({
            "representative_score": total,
            "clarity_score": clarity,
            "product_specificity_score": product_specificity,
            "insight_value_score": insight_value,
            "representativeness_score": representativeness,
            "explainability_score": explainability,
            "context_independence_score": context_independence,
            "representative_decision": decision,
            "representative_reason": reason,
        })

    @staticmethod
    def _representative_comments(comments_df: pd.DataFrame) -> pd.DataFrame:
        if comments_df.empty:
            return pd.DataFrame()
        working = comments_df.copy()
        if "analysis_included" in working.columns:
            working = working[working["analysis_included"].fillna(False)].copy()
        if "hygiene_class" in working.columns:
            hygiene = working["hygiene_class"].fillna("").astype(str).str.lower()
            working = working[hygiene.ne("trash")].copy()
        if "representative_eligibility" in working.columns:
            eligibility = working["representative_eligibility"].fillna(False)
            # Canonical representative source must remain comment-level truth.
            working = working[eligibility.astype(bool)].copy()
        if "representative_decision" in working.columns:
            working = working[working["representative_decision"].isin(["strong_candidate", "review_needed"])].copy()
        if "classification_type" in working.columns:
            working = working[working["classification_type"].fillna("").astype(str).str.lower().ne("informational")].copy()
        if "product_related" in working.columns:
            working = working[working["product_related"].fillna(False)].copy()
        if working.empty:
            return pd.DataFrame()

        sentiment_canonical = (
            working.get("sentiment_final", working.get("sentiment_label", pd.Series("neutral", index=working.index)))
            .fillna(working.get("sentiment_label", pd.Series("neutral", index=working.index)))
            .astype(str)
            .str.lower()
        )
        allowed_sentiments = {"positive", "negative", "neutral", "mixed"}
        working["_sentiment_canonical"] = sentiment_canonical.where(sentiment_canonical.isin(allowed_sentiments), "neutral")
        working["sentiment_label"] = working["_sentiment_canonical"]

        frames = []
        decision_rank = {"strong_candidate": 0, "review_needed": 1, "not_suitable": 2}
        for label in ["negative", "positive", "mixed", "neutral"]:
            subset = working[working["_sentiment_canonical"] == label].copy()
            if subset.empty:
                continue
            if "representative_decision" in subset.columns:
                subset["_decision_rank"] = subset["representative_decision"].map(decision_rank).fillna(9)
            else:
                subset["_decision_rank"] = 9
            if "representative_score" not in subset.columns:
                subset["representative_score"] = 0
            cej_source = subset.get(
                "journey_stage",
                subset.get("customer_journey_stage", subset.get("cej_scene_code", pd.Series("", index=subset.index))),
            )
            subset["_journey_relevance"] = cej_source.fillna("").astype(str).map(lambda value: 0 if value.strip() in {"", "S10 Other", "기타"} else 1)
            subset["_sentiment_strength"] = pd.to_numeric(subset.get("sentiment_score", pd.Series(0.0, index=subset.index)), errors="coerce").fillna(0.0).abs()
            subset["_dedup_key"] = subset.apply(
                lambda row: "|".join([
                    str(row.get("product", "")) if pd.notna(row.get("product", "")) else "",
                    str(row.get("journey_stage", row.get("customer_journey_stage", row.get("cej_scene_code", "")))) if pd.notna(row.get("journey_stage", row.get("customer_journey_stage", row.get("cej_scene_code", "")))) else "",
                    AnalyticsPipeline._primary_judgment_axis(row),
                    _canonical_representative_text(row.get("cleaned_text", row.get("text_display", ""))),
                ]),
                axis=1,
            )
            subset = subset.sort_values(
                ["_decision_rank", "_journey_relevance", "representative_score", "like_count", "_sentiment_strength"],
                ascending=[True, False, False, False, False],
                na_position="last",
            )
            subset = subset.drop_duplicates(subset=["_dedup_key"], keep="first").head(8)
            frames.append(subset.drop(columns=["_decision_rank", "_journey_relevance", "_sentiment_strength", "_dedup_key"], errors="ignore"))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _brand_ratio(comments_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["product", "region", "brand_label", "count", "ratio"]
        if comments_df.empty or "brand_label" not in comments_df.columns:
            return pd.DataFrame(columns=columns)
        working = comments_df[comments_df["brand_label"].fillna("").astype(str).ne("")].copy()
        if working.empty:
            return pd.DataFrame(columns=columns)
        grouped = working.groupby(["product", "region", "brand_label"], dropna=False).size().reset_index(name="count")
        totals = grouped.groupby(["product", "region"], dropna=False)["count"].transform("sum")
        grouped["ratio"] = (grouped["count"] / totals).round(4)
        return grouped.sort_values(["product", "region", "count"], ascending=[True, True, False]).reset_index(drop=True)

    @staticmethod
    def _cej_negative_rate(comments_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["product", "region", "topic_label", "comments", "negative_comments", "negative_rate"]
        if comments_df.empty or "topic_label" not in comments_df.columns:
            return pd.DataFrame(columns=columns)
        grouped = comments_df.groupby(["product", "region", "topic_label"], dropna=False).agg(
            comments=("comment_id", "count"),
            negative_comments=("sentiment_label", lambda series: int((series == "negative").sum())),
        ).reset_index()
        grouped["negative_rate"] = (grouped["negative_comments"] / grouped["comments"]).round(4)
        return grouped.sort_values(["product", "region", "negative_rate", "comments"], ascending=[True, True, False, False]).reset_index(drop=True)

    @staticmethod
    def _monitoring_summary(opinion_units_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["source_count", "opinion_count", "positive_count", "negative_count", "neutral_count", "exclude_count", "neutral_rate", "exclude_rate", "split_rate"]
        if opinion_units_df.empty:
            return pd.DataFrame([{col: 0 for col in columns}])
        source_count = int(opinion_units_df["source_content_id"].nunique())
        opinion_count = int(len(opinion_units_df))
        positive_count = int((opinion_units_df["sentiment_code"] == "positive").sum())
        negative_count = int((opinion_units_df["sentiment_code"] == "negative").sum())
        neutral_count = int((opinion_units_df["sentiment_code"] == "neutral").sum())
        exclude_count = int((opinion_units_df["sentiment_code"] == "excluded").sum())
        split_sources = int((opinion_units_df.groupby("source_content_id")["opinion_id"].count() > 1).sum())
        return pd.DataFrame([{
            "source_count": source_count,
            "opinion_count": opinion_count,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "exclude_count": exclude_count,
            "neutral_rate": (neutral_count / opinion_count) if opinion_count else 0.0,
            "exclude_rate": (exclude_count / opinion_count) if opinion_count else 0.0,
            "split_rate": (split_sources / source_count) if source_count else 0.0,
        }])

    @staticmethod
    def _reporting_summary(opinion_units_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["positive_count", "negative_count", "kpi_opinion_count", "positive_share", "negative_share"]
        if opinion_units_df.empty:
            return pd.DataFrame([{col: 0 for col in columns}])
        reporting = opinion_units_df[opinion_units_df["sentiment_code"].isin(["positive", "negative"])].copy()
        positive_count = int((reporting["sentiment_code"] == "positive").sum())
        negative_count = int((reporting["sentiment_code"] == "negative").sum())
        total = int(len(reporting))
        return pd.DataFrame([{
            "positive_count": positive_count,
            "negative_count": negative_count,
            "kpi_opinion_count": total,
            "positive_share": (positive_count / total) if total else 0.0,
            "negative_share": (negative_count / total) if total else 0.0,
        }])

    @staticmethod
    def _negative_density(videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["product", "region", "video_id", "title", "video_url", "negative_comments", "total_comments", "negative_density"]
        if videos_df.empty:
            return pd.DataFrame(columns=columns)
        working = videos_df.copy()
        for column in ["product", "region", "title", "video_url"]:
            if column not in working.columns:
                working[column] = pd.NA
        if comments_df.empty:
            for column in ["negative_comments", "total_comments", "negative_density"]:
                working[column] = 0
            return working[columns].sort_values(["negative_density", "negative_comments"], ascending=False).reset_index(drop=True)
        negative_counts = comments_df.groupby("video_id")["sentiment_label"].apply(lambda series: int((series == "negative").sum())).rename("negative_comments")
        total_counts = comments_df.groupby("video_id").size().rename("total_comments")
        working = working.merge(negative_counts, on="video_id", how="left").merge(total_counts, on="video_id", how="left")
        working["negative_comments"] = working["negative_comments"].fillna(0).astype(int)
        working["total_comments"] = working["total_comments"].fillna(0).astype(int)
        density_base = working["total_comments"].replace(0, pd.NA).astype("Float64")
        working["negative_density"] = (working["negative_comments"].astype("Float64") / density_base).fillna(0).round(4)
        return working[columns].sort_values(["negative_density", "negative_comments"], ascending=False).reset_index(drop=True)
