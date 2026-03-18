"""Video search collector."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from src.api.youtube_client import YoutubeClient
from src.models.schemas import RunConfig
from src.utils.io import read_json, write_json


class VideoSearchCollector:
    """Search and enrich videos using official YouTube API endpoints."""

    def __init__(self, client: YoutubeClient, cache_dir: Path, logger: logging.Logger | None = None) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)

    def _cache_key(self, config: RunConfig) -> str:
        payload = json.dumps(config.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def search(self, run_id: str, config: RunConfig) -> list[dict[str, Any]]:
        cache_path = self.cache_dir / f"search_{self._cache_key(config)}.json"
        if cache_path.exists() and not config.refresh_existing:
            self.logger.info("Using cached search results", extra={"run_id": run_id, "stage": "search_cache"})
            return read_json(cache_path)

        results: list[dict[str, Any]] = []
        page_token: str | None = None
        rank = 1

        while len(results) < config.max_videos:
            params: dict[str, Any] = {
                "part": "snippet",
                "q": config.keyword,
                "type": "video",
                "maxResults": 50,
                "order": config.order,
                "relevanceLanguage": config.language,
                "regionCode": config.region,
            }
            if config.published_after:
                params["publishedAfter"] = config.published_after
            if config.published_before:
                params["publishedBefore"] = config.published_before
            if config.channel_id:
                params["channelId"] = config.channel_id
            if page_token:
                params["pageToken"] = page_token

            payload = self.client.search_videos(run_id=run_id, **params)
            for item in payload.get("items", []):
                video_id = item.get("id", {}).get("videoId")
                if not video_id:
                    continue
                results.append({"video_id": video_id, "search_rank": rank, "snippet": item.get("snippet", {})})
                rank += 1
                if len(results) >= config.max_videos:
                    break
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        enriched = self._enrich_videos(run_id, config, results)
        write_json(cache_path, enriched)
        return enriched

    def _enrich_videos(self, run_id: str, config: RunConfig, search_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not search_results:
            return []
        videos_by_id = {item["video_id"]: item for item in search_results}
        video_ids = list(videos_by_id)
        enriched: list[dict[str, Any]] = []
        for index in range(0, len(video_ids), 50):
            batch = video_ids[index:index + 50]
            payload = self.client.list_videos(
                run_id=run_id,
                part="snippet,contentDetails,statistics",
                id=",".join(batch),
                maxResults=50,
            )
            for item in payload.get("items", []):
                video_id = item.get("id")
                base = videos_by_id.get(video_id, {})
                snippet = item.get("snippet", {})
                statistics = item.get("statistics", {})
                enriched.append(
                    {
                        "run_id": run_id,
                        "keyword": config.keyword,
                        "product": config.product or config.keyword,
                        "region": config.region,
                        "video_id": video_id,
                        "title": snippet.get("title"),
                        "description": snippet.get("description"),
                        "channel_id": snippet.get("channelId"),
                        "channel_title": snippet.get("channelTitle"),
                        "published_at": snippet.get("publishedAt"),
                        "duration": item.get("contentDetails", {}).get("duration"),
                        "view_count": int(statistics.get("viewCount", 0)) if statistics.get("viewCount") is not None else None,
                        "like_count": int(statistics.get("likeCount", 0)) if statistics.get("likeCount") is not None else None,
                        "comment_count": int(statistics.get("commentCount", 0)) if statistics.get("commentCount") is not None else None,
                        "tags": snippet.get("tags", []),
                        "category_id": snippet.get("categoryId"),
                        "search_rank": base.get("search_rank"),
                        "matched_keyword": config.keyword,
                        "video_url": f"https://www.youtube.com/watch?v={video_id}",
                    }
                )
        enriched.sort(key=lambda item: item.get("search_rank") or 0)
        return enriched
