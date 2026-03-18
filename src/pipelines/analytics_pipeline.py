"""Analytics pipeline for descriptive and NLP outputs."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analytics.brands import infer_brand
from src.analytics.clustering import assign_clusters
from src.analytics.descriptive import build_descriptive_metrics
from src.analytics.keywords import extract_keywords
from src.analytics.sentiment import score_sentiment
from src.analytics.topics import infer_topic
from src.analytics.trends import build_issue_keyword_status, build_weekly_keyword_trend, build_weekly_sentiment_trend
from src.utils.io import ensure_dir, write_dataframe


class AnalyticsPipeline:
    """Compute business-oriented analysis outputs."""

    def __init__(self, processed_dir: Path, risk_keywords: list[str], top_n_keywords: int, topic_count: int, logger: logging.Logger | None = None) -> None:
        self.processed_dir = ensure_dir(processed_dir)
        self.risk_keywords = risk_keywords
        self.top_n_keywords = top_n_keywords
        self.topic_count = topic_count
        self.logger = logger or logging.getLogger(__name__)

    def run(self, run_id: str, videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        working_comments = self._attach_video_context(comments_df, videos_df)
        analysis_comments = working_comments[working_comments.get("analysis_included", False) == True].copy() if not working_comments.empty else working_comments.copy()
        if not analysis_comments.empty:
            sentiment = analysis_comments["cleaned_text"].map(score_sentiment)
            analysis_comments[["sentiment_label", "sentiment_score"]] = pd.DataFrame(sentiment.tolist(), index=analysis_comments.index)
            analysis_comments["topic_label"] = analysis_comments["cleaned_text"].map(infer_topic)
            title_series = analysis_comments["title"].fillna("") if "title" in analysis_comments.columns else pd.Series("", index=analysis_comments.index)
            analysis_comments["brand_label"] = (analysis_comments["cleaned_text"].fillna("") + " " + title_series).map(infer_brand)
            analysis_comments = assign_clusters(analysis_comments, self.topic_count)
            analysis_comments["spam_like"] = analysis_comments.duplicated(subset=["video_id", "cleaned_text"], keep=False)
            analysis_comments["risk_flags"] = analysis_comments["cleaned_text"].map(self._flag_risks)
            working_comments = working_comments.merge(
                analysis_comments[["comment_id", "sentiment_label", "sentiment_score", "topic_label", "brand_label", "cluster_id", "spam_like", "risk_flags"]],
                on="comment_id",
                how="left",
            )
        else:
            for column in ["sentiment_label", "sentiment_score", "topic_label", "brand_label", "cluster_id", "spam_like", "risk_flags"]:
                working_comments[column] = pd.NA

        metrics = build_descriptive_metrics(videos_df, working_comments)
        keyword_df = extract_keywords(analysis_comments["cleaned_text"] if not analysis_comments.empty else pd.Series(dtype=str), self.top_n_keywords)
        sentiment_summary = self._summary_by_label(analysis_comments, "sentiment_label")
        topic_summary = self._summary_by_label(analysis_comments, "topic_label")
        representative_comments = self._representative_comments(analysis_comments)
        outlier_comments = analysis_comments.sort_values(["like_count", "text_length"], ascending=False).head(20) if not analysis_comments.empty else pd.DataFrame()
        weekly_sentiment_trend = build_weekly_sentiment_trend(analysis_comments)
        weekly_keyword_trend = build_weekly_keyword_trend(analysis_comments)
        new_issue_keywords, persistent_issue_keywords = build_issue_keyword_status(weekly_keyword_trend)
        brand_ratio = self._brand_ratio(analysis_comments)
        cej_negative_rate = self._cej_negative_rate(analysis_comments)
        negative_density = self._negative_density(videos_df, analysis_comments)

        outputs = {
            **metrics,
            "comments": working_comments,
            "analysis_comments": analysis_comments,
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
        }

        run_dir = ensure_dir(self.processed_dir / run_id)
        for name, frame in outputs.items():
            if isinstance(frame, pd.DataFrame):
                write_dataframe(frame, run_dir / f"{name}.parquet")
        self.logger.info("Analytics stage completed", extra={"run_id": run_id, "stage": "analytics"})
        return outputs

    @staticmethod
    def _attach_video_context(comments_df: pd.DataFrame, videos_df: pd.DataFrame) -> pd.DataFrame:
        if comments_df.empty or videos_df.empty:
            return comments_df.copy()
        video_columns = [column for column in ["video_id", "product", "keyword", "region", "title", "channel_title", "video_url"] if column in videos_df.columns]
        if not video_columns:
            return comments_df.copy()
        return comments_df.merge(videos_df[video_columns].drop_duplicates(subset=["video_id"]), on="video_id", how="left")

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
    def _representative_comments(comments_df: pd.DataFrame) -> pd.DataFrame:
        if comments_df.empty:
            return pd.DataFrame()
        frames = []
        for label in ["positive", "negative", "neutral"]:
            subset = comments_df[comments_df["sentiment_label"] == label].sort_values(["sentiment_score", "like_count"], ascending=[label != "negative", False]).head(6)
            if not subset.empty:
                frames.append(subset)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _brand_ratio(comments_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["product", "region", "brand_label", "count", "ratio"]
        if comments_df.empty or "brand_label" not in comments_df.columns:
            return pd.DataFrame(columns=columns)
        working = comments_df[comments_df["brand_label"] != "미언급"].copy()
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
    def _negative_density(videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
        columns = ["product", "region", "video_id", "title", "video_url", "negative_comments", "total_comments", "negative_density"]
        if videos_df.empty:
            return pd.DataFrame(columns=columns)
        working = videos_df.copy()
        if comments_df.empty:
            for column in ["negative_comments", "total_comments", "negative_density"]:
                working[column] = 0
            return working[columns] if set(columns).issubset(working.columns) else pd.DataFrame(columns=columns)
        negative_counts = comments_df.groupby("video_id")["sentiment_label"].apply(lambda series: int((series == "negative").sum())).rename("negative_comments")
        total_counts = comments_df.groupby("video_id").size().rename("total_comments")
        working = working.merge(negative_counts, on="video_id", how="left").merge(total_counts, on="video_id", how="left")
        working["negative_comments"] = working["negative_comments"].fillna(0).astype(int)
        working["total_comments"] = working["total_comments"].fillna(0).astype(int)
        density_base = working["total_comments"].replace(0, pd.NA).astype("Float64")
        working["negative_density"] = (working["negative_comments"].astype("Float64") / density_base).fillna(0).round(4)
        return working[columns].sort_values(["negative_density", "negative_comments"], ascending=False).reset_index(drop=True)
