"""Normalization pipeline for flat tables and derived columns."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analytics.text_cleaning import clean_text, classify_comment_quality, detect_language_hint
from src.utils.io import ensure_dir, write_dataframe


class NormalizePipeline:
    """Create analytics-ready normalized datasets."""

    def __init__(self, processed_dir: Path, logger: logging.Logger | None = None) -> None:
        self.processed_dir = ensure_dir(processed_dir)
        self.logger = logger or logging.getLogger(__name__)

    def run(self, run_id: str, videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        normalized_videos = videos_df.copy()
        normalized_comments = comments_df.copy()
        if not normalized_comments.empty:
            normalized_comments["text_original"] = normalized_comments["text_original"].fillna("")
            normalized_comments["text_display"] = normalized_comments["text_display"].fillna(normalized_comments["text_original"])
            normalized_comments["cleaned_text"] = normalized_comments["text_original"].map(clean_text)
            normalized_comments["text_length"] = normalized_comments["cleaned_text"].map(len)
            normalized_comments["language_detected"] = normalized_comments["cleaned_text"].map(detect_language_hint)
            normalized_comments["comment_quality"] = normalized_comments["text_original"].map(classify_comment_quality)
            normalized_comments["comment_validity"] = normalized_comments["comment_quality"].map(lambda value: "valid" if value == "valid" else "excluded")
            normalized_comments["exclusion_reason"] = normalized_comments["comment_quality"].where(normalized_comments["comment_validity"] == "excluded", pd.NA)
            normalized_comments["analysis_included"] = normalized_comments["comment_validity"].eq("valid") & normalized_comments["cleaned_text"].ne("")
            normalized_comments["removed_reason"] = normalized_comments["exclusion_reason"].fillna("")
            normalized_comments = normalized_comments.drop_duplicates(subset=["video_id", "comment_id"], keep="last").reset_index(drop=True)

        run_dir = ensure_dir(self.processed_dir / run_id)
        write_dataframe(normalized_videos, run_dir / "videos_normalized.parquet")
        write_dataframe(normalized_comments, run_dir / "comments_normalized.parquet")
        self.logger.info("Normalization stage completed", extra={"run_id": run_id, "stage": "normalize"})
        return normalized_videos, normalized_comments
