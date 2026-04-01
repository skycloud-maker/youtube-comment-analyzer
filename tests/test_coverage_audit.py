from pathlib import Path
import json
import shutil

import pandas as pd

from src.utils.coverage_audit import build_run_coverage_audit, summarize_coverage_audit


def test_build_run_coverage_audit_explains_trimmed_and_missing_cases():
    root = Path(r"C:\codex\test_artifacts\coverage_audit")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    processed_dir = root / "processed"
    raw_dir = root / "raw"
    run_id = "run_cov_001"
    (processed_dir / run_id).mkdir(parents=True, exist_ok=True)
    (raw_dir / run_id / "search_list").mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": run_id,
        "config_json": {"max_videos": 2},
        "failed_video_ids": ["v2"],
    }
    (processed_dir / run_id / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    payload = {
        "payload": {
            "items": [
                {"id": {"videoId": "v1"}, "snippet": {"title": "first", "channelTitle": "c1", "publishedAt": "2026-01-01T00:00:00Z"}},
                {"id": {"videoId": "v2"}, "snippet": {"title": "second", "channelTitle": "c2", "publishedAt": "2026-01-02T00:00:00Z"}},
                {"id": {"videoId": "v3"}, "snippet": {"title": "third", "channelTitle": "c3", "publishedAt": "2026-01-03T00:00:00Z"}},
            ]
        }
    }
    (raw_dir / run_id / "search_list" / "global_first.json").write_text(json.dumps(payload), encoding="utf-8")

    pd.DataFrame([
        {"video_id": "v1", "title": "first"},
        {"video_id": "v2", "title": "second"},
    ]).to_parquet(processed_dir / run_id / "videos_raw.parquet")
    pd.DataFrame([
        {"video_id": "v1", "comment_id": "c1"},
    ]).to_parquet(processed_dir / run_id / "comments_raw.parquet")

    frame = build_run_coverage_audit(processed_dir, raw_dir, run_id, hot_rank_cutoff=3)
    assert list(frame["reason_code"]) == ["COVERED", "INGEST_API_ERROR", "TRIMMED_BY_MAX_VIDEOS"]
    summary = summarize_coverage_audit(frame)
    assert summary == {
        "hot_candidates": 3,
        "covered": 1,
        "missing": 0,
        "trimmed": 1,
        "missing_comments": 1,
    }


def test_build_run_coverage_audit_marks_zero_comment_videos_explicitly():
    root = Path(r"C:\codex\test_artifacts\coverage_audit_zero")
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    processed_dir = root / "processed"
    raw_dir = root / "raw"
    run_id = "run_cov_002"
    (processed_dir / run_id).mkdir(parents=True, exist_ok=True)
    (raw_dir / run_id / "search_list").mkdir(parents=True, exist_ok=True)
    (raw_dir / run_id / "commentThreads_list").mkdir(parents=True, exist_ok=True)

    manifest = {"run_id": run_id, "config_json": {"max_videos": 1}, "failed_video_ids": []}
    (processed_dir / run_id / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    payload = {"payload": {"items": [{"id": {"videoId": "v1"}, "snippet": {"title": "first", "channelTitle": "c1", "publishedAt": "2026-01-01T00:00:00Z"}}]}}
    (raw_dir / run_id / "search_list" / "global_first.json").write_text(json.dumps(payload), encoding="utf-8")
    (raw_dir / run_id / "commentThreads_list" / "v1_first.json").write_text(json.dumps({"payload": {"items": []}}), encoding="utf-8")

    pd.DataFrame([{"video_id": "v1", "title": "first", "comment_count": 0}]).to_parquet(processed_dir / run_id / "videos_raw.parquet")
    pd.DataFrame([], columns=["video_id", "comment_id"]).to_parquet(processed_dir / run_id / "comments_raw.parquet")

    frame = build_run_coverage_audit(processed_dir, raw_dir, run_id, hot_rank_cutoff=1)
    assert frame.iloc[0]["reason_code"] == "COMMENTS_DISABLED_OR_ZERO"
    assert bool(frame.iloc[0]["comment_audit_seen"]) is True
