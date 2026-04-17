"""Normalization pipeline for flat tables and derived columns."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.analytics.text_cleaning import clean_text, classify_comment_quality, detect_language_hint
from src.utils.privacy import detect_pii, extract_external_links, mask_pii_text
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
            normalized_comments["hygiene_class"] = normalized_comments["comment_validity"].map(lambda value: "strategic" if value == "valid" else "trash")
            normalized_comments["analysis_included"] = (
                normalized_comments["comment_validity"].eq("valid")
                & normalized_comments["cleaned_text"].ne("")
                & normalized_comments["hygiene_class"].ne("trash")
            )
            normalized_comments["removed_reason"] = normalized_comments["exclusion_reason"].fillna("")
            normalized_comments["sentiment_final"] = pd.NA
            normalized_comments["mixed_flag"] = False
            normalized_comments["inquiry_flag"] = False
            normalized_comments["journey_stage"] = pd.NA
            normalized_comments["judgment_axes"] = pd.NA
            normalized_comments["representative_eligibility"] = pd.NA
            pii_payload = normalized_comments["text_original"].map(detect_pii)
            normalized_comments["pii_flag"] = pii_payload.map(lambda item: item[0])
            normalized_comments["pii_types"] = pii_payload.map(lambda item: item[1])
            normalized_comments["pii_raw_text"] = pii_payload.map(lambda item: item[2])
            normalized_comments["masked_text"] = normalized_comments.apply(lambda row: mask_pii_text(row.get("text_original", ""), row.get("pii_raw_text", "")), axis=1)
            normalized_comments["external_links"] = normalized_comments["text_original"].map(extract_external_links)
            normalized_comments["context_attachment"] = normalized_comments["external_links"].map(lambda items: items[0] if items else "")
            normalized_comments["source_type"] = normalized_comments.get("source_type", "YT_COMMENT")
            normalized_comments["live_chat_included"] = False
            normalized_comments = normalized_comments.drop_duplicates(subset=["video_id", "comment_id"], keep="last").reset_index(drop=True)

        run_dir = ensure_dir(self.processed_dir / run_id)
        write_dataframe(normalized_videos, run_dir / "videos_normalized.parquet")
        write_dataframe(normalized_comments, run_dir / "comments_normalized.parquet")
        self.logger.info("Normalization stage completed", extra={"run_id": run_id, "stage": "normalize"})
        return normalized_videos, normalized_comments
