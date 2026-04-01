from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


COVERAGE_COLUMNS = [
    "search_rank",
    "video_id",
    "title",
    "channel_title",
    "published_at",
    "api_comment_count",
    "in_videos_raw",
    "in_comments_raw",
    "failed_video",
    "comment_audit_seen",
    "coverage_status",
    "reason_code",
]


def load_run_manifest(processed_dir: Path, run_id: str) -> dict[str, Any]:
    path = processed_dir / run_id / "run_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))



def load_comment_thread_audit_status(raw_dir: Path, run_id: str) -> dict[str, bool]:
    audit_dir = raw_dir / run_id / "commentThreads_list"
    status: dict[str, bool] = {}
    if not audit_dir.exists():
        return status
    for path in audit_dir.glob('*.json'):
        video_id = path.stem.split('_first')[0].split('_')[0]
        if video_id and video_id != 'global':
            status[video_id] = True
    return status


def load_search_candidates(raw_dir: Path, run_id: str) -> pd.DataFrame:
    path = raw_dir / run_id / "search_list" / "global_first.json"
    if not path.exists():
        return pd.DataFrame(columns=["search_rank", "video_id", "title", "channel_title", "published_at"])

    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("payload", {}).get("items", [])
    rows: list[dict[str, Any]] = []
    for rank, item in enumerate(items, start=1):
        snippet = item.get("snippet", {})
        rows.append(
            {
                "search_rank": rank,
                "video_id": item.get("id", {}).get("videoId"),
                "title": snippet.get("title", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "published_at": snippet.get("publishedAt", ""),
            }
        )
    return pd.DataFrame(rows)



def build_run_coverage_audit(
    processed_dir: Path,
    raw_dir: Path,
    run_id: str,
    *,
    hot_rank_cutoff: int = 10,
) -> pd.DataFrame:
    manifest = load_run_manifest(processed_dir, run_id)
    search_df = load_search_candidates(raw_dir, run_id)
    if search_df.empty:
        return pd.DataFrame(columns=COVERAGE_COLUMNS)

    videos_path = processed_dir / run_id / "videos_raw.parquet"
    comments_path = processed_dir / run_id / "comments_raw.parquet"
    videos_df = pd.read_parquet(videos_path) if videos_path.exists() else pd.DataFrame(columns=["video_id", "comment_count"])
    comments_df = pd.read_parquet(comments_path) if comments_path.exists() else pd.DataFrame(columns=["video_id"])

    collected_ids = set(videos_df.get("video_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    commented_ids = set(comments_df.get("video_id", pd.Series(dtype=str)).dropna().astype(str).tolist())
    api_comment_counts = {
        str(row.get("video_id")): int(row.get("comment_count") or 0)
        for _, row in videos_df.iterrows()
        if str(row.get("video_id") or "")
    }
    comment_audit_seen = load_comment_thread_audit_status(raw_dir, run_id)
    failed_ids = set(str(x) for x in manifest.get("failed_video_ids", []) if x)
    max_videos = int((manifest.get("config_json") or {}).get("max_videos") or len(collected_ids) or hot_rank_cutoff)

    rows: list[dict[str, Any]] = []
    hot_df = search_df[search_df["search_rank"] <= hot_rank_cutoff].copy()
    for _, row in hot_df.iterrows():
        video_id = str(row.get("video_id") or "")
        in_videos = video_id in collected_ids
        in_comments = video_id in commented_ids
        failed = video_id in failed_ids
        api_comment_count = int(api_comment_counts.get(video_id, 0))
        audit_seen = bool(comment_audit_seen.get(video_id))

        if in_videos and in_comments and not failed:
            status = "covered"
            reason_code = "COVERED"
        elif in_videos and failed:
            status = "missing_comments"
            reason_code = "INGEST_API_ERROR"
        elif in_videos and not in_comments:
            status = "missing_comments"
            if api_comment_count == 0 and audit_seen:
                reason_code = "COMMENTS_DISABLED_OR_ZERO"
            elif audit_seen:
                reason_code = "COMMENTS_NOT_COLLECTED"
            else:
                reason_code = "COMMENTS_AUDIT_MISSING"
        elif row.get("search_rank", 999999) > max_videos:
            status = "trimmed"
            reason_code = "TRIMMED_BY_MAX_VIDEOS"
        else:
            status = "missing"
            reason_code = "COVERAGE_MISSING_HOT_VIDEO"

        rows.append(
            {
                "search_rank": int(row.get("search_rank") or 0),
                "video_id": video_id,
                "title": str(row.get("title") or ""),
                "channel_title": str(row.get("channel_title") or ""),
                "published_at": str(row.get("published_at") or ""),
                "api_comment_count": api_comment_count,
                "in_videos_raw": in_videos,
                "in_comments_raw": in_comments,
                "failed_video": failed,
                "comment_audit_seen": audit_seen,
                "coverage_status": status,
                "reason_code": reason_code,
            }
        )

    return pd.DataFrame(rows, columns=COVERAGE_COLUMNS)



def summarize_coverage_audit(frame: pd.DataFrame) -> dict[str, int]:
    if frame is None or frame.empty:
        return {"hot_candidates": 0, "covered": 0, "missing": 0, "trimmed": 0, "missing_comments": 0}
    counts = frame["coverage_status"].value_counts().to_dict()
    return {
        "hot_candidates": int(len(frame)),
        "covered": int(counts.get("covered", 0)),
        "missing": int(counts.get("missing", 0)),
        "trimmed": int(counts.get("trimmed", 0)),
        "missing_comments": int(counts.get("missing_comments", 0)),
    }
