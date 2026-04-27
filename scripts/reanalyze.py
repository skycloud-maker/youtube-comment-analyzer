"""Re-run the analytics pipeline on existing normalized data.

Loads comments_normalized.parquet + videos_normalized.parquet from an input
directory and runs only the analytics stage (LLM analysis + all aggregations).
Skips YouTube data collection and normalization.

Usage examples:
  # Re-analyze data that's already in the configured processed_dir
  python scripts/reanalyze.py --run-id my-run-2024

  # Load normalized parquets from a custom directory, write output under a new ID
  python scripts/reanalyze.py --input-dir _tmp_manual/run-1 --run-id reanalyzed-v2

  # Load from raw parquets (will normalize first)
  python scripts/reanalyze.py --input-dir data/processed/my-run --run-id reanalyzed-v2 --from-raw
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.logger import setup_logging
from src.pipelines.analytics_pipeline import AnalyticsPipeline
from src.pipelines.normalize_pipeline import NormalizePipeline


def _load_parquet(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_parquet(path)
    raise FileNotFoundError(f"Parquet not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run analytics on existing data without re-collecting from YouTube")
    parser.add_argument("--run-id", required=True, help="Output run ID (used as subdirectory name under processed_dir)")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Directory with normalized parquets. Defaults to processed_dir/run-id.",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Load comments_raw.parquet + videos_raw.parquet and normalize first",
    )
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.yaml.storage.logs_dir, config.yaml.log_level)
    logger = logging.getLogger(__name__)

    processed_dir = Path(config.yaml.storage.processed_dir).resolve()
    input_dir = Path(args.input_dir).resolve() if args.input_dir else processed_dir / args.run_id

    if args.from_raw:
        raw_videos_path = input_dir / "videos_raw.parquet"
        raw_comments_path = input_dir / "comments_raw.parquet"
        logger.info("Loading raw data from %s", input_dir)
        videos_df = _load_parquet(raw_videos_path)
        comments_df = _load_parquet(raw_comments_path)
        logger.info("Normalizing %d videos, %d comments...", len(videos_df), len(comments_df))
        normalize = NormalizePipeline(processed_dir, logger)
        videos_df, comments_df = normalize.run(args.run_id, videos_df, comments_df)
    else:
        videos_path = input_dir / "videos_normalized.parquet"
        comments_path = input_dir / "comments_normalized.parquet"
        logger.info("Loading normalized data from %s", input_dir)
        videos_df = _load_parquet(videos_path)
        comments_df = _load_parquet(comments_path)

    logger.info("Loaded %d videos, %d comments. Starting analytics (LLM analysis)...", len(videos_df), len(comments_df))

    analytics = AnalyticsPipeline(
        processed_dir=processed_dir,
        risk_keywords=config.yaml.analytics.risk_keywords,
        top_n_keywords=config.yaml.analytics.top_n_keywords,
        topic_count=config.yaml.analytics.topic_count,
        logger=logger,
        nlp_analyzer_config=config.yaml.analytics.nlp_analyzer.model_dump(),
        video_selection_config=config.yaml.collection.video_selection.model_dump(),
    )

    outputs = analytics.run(args.run_id, videos_df, comments_df)
    n_comments = len(outputs.get("comments", pd.DataFrame()))
    output_dir = processed_dir / args.run_id
    logger.info("Analytics complete. run_id=%s, comments=%d, output=%s", args.run_id, n_comments, output_dir)
    print(f"\n재분석 완료.")
    print(f"  Run ID  : {args.run_id}")
    print(f"  댓글 수 : {n_comments}")
    print(f"  출력 위치: {output_dir}")
    print(f"\nStreamlit을 재시작하면 새 결과를 확인할 수 있습니다.")


if __name__ == "__main__":
    main()
