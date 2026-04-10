"""Command line interface for YouTube comment analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.api.quota import QuotaLedger, estimate_quota_usage
from src.api.youtube_client import YoutubeClient
from src.collectors.comments_collector import CommentsCollector
from src.collectors.video_search import VideoSearchCollector
from src.config import AppConfig, load_config
from src.exporters.excel_writer import ExcelWriter
from src.exporters.report_writer import ReportWriter
from src.logger import setup_logging
from src.models.schemas import RunConfig, RunManifest
from src.pipelines.analytics_pipeline import AnalyticsPipeline
from src.pipelines.export_pipeline import ExportPipeline
from src.pipelines.ingest_pipeline import IngestPipeline
from src.pipelines.normalize_pipeline import NormalizePipeline
from src.utils.io import read_json, write_dataframe

LOGGER = logging.getLogger(__name__)


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YouTube comment analysis using official YouTube Data API v3")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ["run", "full-run", "search-videos", "collect-comments", "analyze", "export"]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--keyword", required=name in {"run", "full-run", "search-videos"})
        sub.add_argument("--product")
        sub.add_argument("--max-videos", type=int, default=100)
        sub.add_argument("--comments-per-video", type=int, default=0)
        sub.add_argument("--comments-order", choices=["relevance", "time"], default="time")
        sub.add_argument("--published-after")
        sub.add_argument("--published-before")
        sub.add_argument("--order", default="relevance")
        sub.add_argument("--language", default="ko")
        sub.add_argument("--region", default="KR")
        sub.add_argument("--channel-id")
        sub.add_argument("--include-replies", default="true")
        sub.add_argument("--refresh-existing", default="false")
        sub.add_argument("--output-prefix", default="youtube_analysis")
        sub.add_argument("--run-id")
    return parser


def _build_dependencies() -> tuple[AppConfig, QuotaLedger, YoutubeClient]:
    config = load_config()
    setup_logging(config.yaml.storage.logs_dir, config.yaml.log_level)
    quota_ledger = QuotaLedger(daily_limit=config.yaml.daily_quota_limit)
    client = YoutubeClient(
        api_key=config.env.youtube_api_key.get_secret_value(),
        base_url=config.yaml.api.base_url,
        timeout_seconds=config.yaml.api.timeout_seconds,
        max_retries=config.yaml.api.max_retries,
        backoff_seconds=config.yaml.api.backoff_seconds,
        raw_dir=config.yaml.storage.raw_dir,
        quota_ledger=quota_ledger,
        logger=LOGGER,
    )
    return config, quota_ledger, client


def _build_run_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        keyword=args.keyword or "",
        product=args.product,
        max_videos=args.max_videos,
        comments_per_video=args.comments_per_video,
        comments_order=args.comments_order,
        published_after=args.published_after,
        published_before=args.published_before,
        order=args.order,
        language=args.language,
        region=args.region,
        channel_id=args.channel_id,
        include_replies=_str_to_bool(args.include_replies),
        refresh_existing=_str_to_bool(args.refresh_existing),
        output_prefix=args.output_prefix,
    )


def _manifest_path(processed_dir: Path, run_id: str) -> Path:
    return processed_dir / run_id / "run_manifest.json"


def _load_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        return pd.read_csv(path) if path.exists() else pd.DataFrame()
    if path.exists():
        return pd.read_parquet(path)
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def _load_analytics_frames(run_dir: Path) -> dict[str, pd.DataFrame]:
    names = [
        "runs_summary",
        "video_ranking",
        "channel_comparison",
        "upload_distribution",
        "comment_time_trend",
        "quality_summary",
        "keyword_analysis",
        "sentiment_summary",
        "topic_summary",
        "representative_comments",
        "outlier_comments",
        "weekly_sentiment_trend",
        "weekly_keyword_trend",
        "brand_ratio",
        "cej_negative_rate",
        "negative_density",
        "new_issue_keywords",
        "persistent_issue_keywords",
        "comments",
    ]
    return {name: _load_frame(run_dir / f"{name}.parquet") for name in names}


def _persist_manifest(config: AppConfig, manifest: RunManifest) -> Path:
    path = _manifest_path(config.yaml.storage.processed_dir, manifest.run_id)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_pipeline(args: argparse.Namespace) -> dict[str, str]:
    config, quota_ledger, client = _build_dependencies()
    run_config = _build_run_config(args)
    quota_estimate = estimate_quota_usage(
        max_videos=run_config.max_videos,
        comments_per_video=run_config.comments_per_video,
        include_replies=run_config.include_replies,
    )
    if quota_ledger.warn_threshold_exceeded(quota_estimate["total_units"]):
        LOGGER.warning("Estimated quota exceeds the configured daily budget", extra={"stage": "quota"})

    manifest = RunManifest(keyword=run_config.keyword, config_json=run_config.model_dump(mode="json"), quota_estimate=quota_estimate)
    if args.run_id:
        manifest.run_id = args.run_id

    search_collector = VideoSearchCollector(client, config.yaml.storage.raw_dir / "search_cache", LOGGER)
    comments_collector = CommentsCollector(client, LOGGER)
    ingest = IngestPipeline(search_collector, comments_collector, config.yaml.storage.processed_dir, LOGGER)
    normalize = NormalizePipeline(config.yaml.storage.processed_dir, LOGGER)
    analytics = AnalyticsPipeline(
        config.yaml.storage.processed_dir,
        config.yaml.analytics.risk_keywords,
        config.yaml.analytics.top_n_keywords,
        config.yaml.analytics.topic_count,
        LOGGER,
        nlp_analyzer_config=config.yaml.analytics.nlp_analyzer.model_dump(),
    )
    export = ExportPipeline(
        config.yaml.storage.exports_dir,
        ExcelWriter(config.yaml.excel.datetime_format, config.yaml.excel.freeze_panes),
        ReportWriter(),
        LOGGER,
    )

    result: dict[str, str] = {"run_id": manifest.run_id}

    if args.command == "search-videos":
        videos = search_collector.search(manifest.run_id, run_config)
        videos_df = pd.DataFrame(videos)
        write_dataframe(videos_df, config.yaml.storage.processed_dir / manifest.run_id / "videos_raw.parquet")
        manifest.videos_collected = int(len(videos_df))
        manifest.api_calls_json = quota_ledger.actual_calls
        result["manifest_path"] = str(_persist_manifest(config, manifest))
        return result

    if args.command == "collect-comments":
        if not args.run_id:
            raise ValueError("--run-id is required for collect-comments")
        manifest = RunManifest.model_validate(read_json(_manifest_path(config.yaml.storage.processed_dir, args.run_id)))
        videos_df = _load_frame(config.yaml.storage.processed_dir / args.run_id / "videos_raw.parquet")
        existing_comments = ingest._load_existing_comments(config.yaml.storage.processed_dir / args.run_id / "comments_raw.parquet")
        comments_frames: list[pd.DataFrame] = []
        errors_frames: list[pd.DataFrame] = []
        for video_id in videos_df.get("video_id", pd.Series(dtype=str)).tolist():
            rows, errors = comments_collector.collect_video_comments(
                run_id=args.run_id,
                video_id=video_id,
                comments_per_video=run_config.comments_per_video,
                include_replies=run_config.include_replies,
                comments_order=run_config.comments_order,
            )
            comments_frames.append(pd.DataFrame(rows))
            errors_frames.append(pd.DataFrame(errors))
        new_comments = pd.concat(comments_frames, ignore_index=True) if comments_frames else pd.DataFrame()
        comments_df = ingest._merge_comments(existing_comments, new_comments)
        errors_df = pd.concat(errors_frames, ignore_index=True) if errors_frames else pd.DataFrame()
        write_dataframe(comments_df, config.yaml.storage.processed_dir / args.run_id / "comments_raw.parquet")
        if not errors_df.empty:
            write_dataframe(errors_df, config.yaml.storage.processed_dir / args.run_id / "errors.csv")
        manifest.comments_collected = int(len(comments_df))
        manifest.error_count = int(len(errors_df))
        manifest.api_calls_json = quota_ledger.actual_calls
        result["manifest_path"] = str(_persist_manifest(config, manifest))
        return result

    if args.command in {"run", "full-run"}:
        videos_df, comments_df, errors_df, _ = ingest.run(run_config, manifest)
        manifest.api_calls_json = quota_ledger.actual_calls
        result["manifest_path"] = str(_persist_manifest(config, manifest))
        normalized_videos_df, normalized_comments_df = normalize.run(manifest.run_id, videos_df, comments_df)
        analytics_outputs = analytics.run(manifest.run_id, normalized_videos_df, normalized_comments_df)
        workbook_path, report_path = export.run(
            run_id=manifest.run_id,
            output_prefix=run_config.output_prefix,
            manifest=manifest.model_dump(mode="json"),
            videos_df=normalized_videos_df,
            analytics_outputs=analytics_outputs,
            errors_df=errors_df,
        )
        result["workbook_path"] = str(workbook_path)
        result["report_path"] = str(report_path)
        return result

    if not args.run_id:
        raise ValueError("--run-id is required for analyze/export")

    manifest = RunManifest.model_validate(read_json(_manifest_path(config.yaml.storage.processed_dir, args.run_id)))
    videos_df = _load_frame(config.yaml.storage.processed_dir / args.run_id / "videos_raw.parquet")
    comments_df = _load_frame(config.yaml.storage.processed_dir / args.run_id / "comments_raw.parquet")

    if args.command == "analyze":
        normalized_videos_df, normalized_comments_df = normalize.run(args.run_id, videos_df, comments_df)
        analytics_outputs = analytics.run(args.run_id, normalized_videos_df, normalized_comments_df)
        result["analytics_dir"] = str(config.yaml.storage.processed_dir / args.run_id)
        result["comments_rows"] = str(len(analytics_outputs.get("comments", pd.DataFrame())))
        return result

    analytics_outputs = _load_analytics_frames(config.yaml.storage.processed_dir / args.run_id)
    errors_df = _load_frame(config.yaml.storage.processed_dir / args.run_id / "errors.parquet")
    if errors_df.empty:
        errors_df = _load_frame(config.yaml.storage.processed_dir / args.run_id / "errors.csv")
    workbook_path, report_path = export.run(
        run_id=args.run_id,
        output_prefix=run_config.output_prefix,
        manifest=manifest.model_dump(mode="json"),
        videos_df=_load_frame(config.yaml.storage.processed_dir / args.run_id / "videos_normalized.parquet"),
        analytics_outputs=analytics_outputs,
        errors_df=errors_df,
    )
    result["workbook_path"] = str(workbook_path)
    result["report_path"] = str(report_path)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    result = run_pipeline(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
