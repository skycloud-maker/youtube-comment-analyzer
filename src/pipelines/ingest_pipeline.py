"""Ingestion pipeline orchestrating search and comment collection."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from src.collectors.comments_collector import CommentsCollector
from src.collectors.video_search import VideoSearchCollector
from src.models.schemas import RunConfig, RunManifest, utc_now_iso
from src.utils.io import ensure_dir, write_dataframe, write_json


class IngestPipeline:
    """Run search and raw comment collection with rerun safety."""

    def __init__(self, video_search: VideoSearchCollector, comments_collector: CommentsCollector, processed_dir: Path, logger: logging.Logger | None = None) -> None:
        self.video_search = video_search
        self.comments_collector = comments_collector
        self.processed_dir = ensure_dir(processed_dir)
        self.logger = logger or logging.getLogger(__name__)

    def run(self, config: RunConfig, manifest: RunManifest) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Path]:
        run_dir = ensure_dir(self.processed_dir / manifest.run_id)
        videos_path = run_dir / "videos_raw.parquet"
        comments_path = run_dir / "comments_raw.parquet"
        errors_path = run_dir / "errors.csv"
        manifest_path = run_dir / "run_manifest.json"

        search_keywords = config.keywords if config.keywords else [config.keyword]
        all_videos: list[dict] = []
        seen_video_ids: set[str] = set()
        per_keyword_limit = max(1, math.ceil(config.max_videos / len(search_keywords)))
        for kw in search_keywords:
            kw_config = config.model_copy(update={"keyword": kw, "max_videos": per_keyword_limit})
            kw_results = self.video_search.search(manifest.run_id, kw_config)
            for v in kw_results:
                vid = v["video_id"]
                if vid not in seen_video_ids:
                    seen_video_ids.add(vid)
                    all_videos.append(v)
            self.logger.info("Keyword '%s': %d videos (total unique: %d)", kw, len(kw_results), len(all_videos))
        # Preserve collection-time video universe order; dashboard handles narrowing/filtering.
        all_videos = sorted(
            all_videos,
            key=lambda item: (
                int(item.get("search_rank") or 999999),
                -int(item.get("comment_count") or 0),
            ),
        )
        videos = all_videos[:config.max_videos]
        videos_df = pd.DataFrame(videos)
        comment_rows: list[dict] = []
        error_rows: list[dict] = []

        existing_comments = self._load_existing_comments(comments_path)
        collected_video_ids = set(existing_comments["video_id"].unique().tolist()) if not existing_comments.empty else set()

        for video in videos:
            video_id = video["video_id"]
            if video_id in collected_video_ids and not config.refresh_existing:
                self.logger.info("Skipping previously collected video", extra={"run_id": manifest.run_id, "video_id": video_id, "stage": "ingest"})
                continue
            rows, video_errors = self.comments_collector.collect_video_comments(
                run_id=manifest.run_id,
                video_id=video_id,
                comments_per_video=config.comments_per_video,
                include_replies=config.include_replies,
                comments_order=config.comments_order,
            )
            comment_rows.extend(rows)
            error_rows.extend(video_errors)

        comments_df = self._merge_comments(existing_comments, pd.DataFrame(comment_rows))
        errors_df = pd.DataFrame(error_rows)

        manifest.finished_at = utc_now_iso()
        manifest.videos_collected = int(len(videos_df))
        manifest.comments_collected = int(len(comments_df))
        manifest.error_count = int(len(errors_df))
        manifest.failed_video_ids = errors_df["video_id"].dropna().astype(str).unique().tolist() if not errors_df.empty else []

        write_dataframe(videos_df, videos_path)
        final_comments_path = write_dataframe(comments_df, comments_path)
        if not errors_df.empty:
            write_dataframe(errors_df, errors_path)
        write_json(manifest_path, manifest.model_dump(mode="json"))
        self.logger.info("Ingestion stage completed", extra={"run_id": manifest.run_id, "stage": "ingest"})
        return videos_df, comments_df, errors_df, final_comments_path.parent

    @staticmethod
    def _load_existing_comments(comments_path: Path) -> pd.DataFrame:
        if comments_path.exists():
            try:
                return pd.read_parquet(comments_path)
            except Exception:
                pass
        csv_path = comments_path.with_suffix(".csv")
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()

    @staticmethod
    def _merge_comments(existing_comments: pd.DataFrame, new_comments: pd.DataFrame) -> pd.DataFrame:
        frames = [frame for frame in [existing_comments, new_comments] if not frame.empty]
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)
        return merged.drop_duplicates(subset=["video_id", "comment_id"], keep="last").reset_index(drop=True)
