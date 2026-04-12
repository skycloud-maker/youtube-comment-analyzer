"""Video search collector."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.api.youtube_client import YoutubeClient
from src.config import VideoSelectionSettings
from src.models.schemas import RunConfig
from src.utils.io import read_json, write_json


class VideoSearchCollector:
    """Search and enrich videos using official YouTube API endpoints."""

    def __init__(
        self,
        client: YoutubeClient,
        cache_dir: Path,
        logger: logging.Logger | None = None,
        selection_policy: VideoSelectionSettings | None = None,
    ) -> None:
        self.client = client
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or logging.getLogger(__name__)
        self.selection_policy = selection_policy or VideoSelectionSettings()

    def _cache_key(self, config: RunConfig) -> str:
        payload = json.dumps(
            {
                "run_config": config.model_dump(mode="json"),
                "selection_policy": self.selection_policy.model_dump(mode="json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def search(self, run_id: str, config: RunConfig) -> list[dict[str, Any]]:
        cache_path = self.cache_dir / f"search_{self._cache_key(config)}.json"
        if cache_path.exists() and not config.refresh_existing:
            self.logger.info("Using cached search results", extra={"run_id": run_id, "stage": "search_cache"})
            return read_json(cache_path)

        results: list[dict[str, Any]] = []
        page_token: str | None = None
        rank = 1
        oversample = max(1, int(config.search_oversample_factor or 1))
        target_pool_size = max(int(config.max_videos), int(config.max_videos) * oversample)

        while len(results) < target_pool_size:
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
                if len(results) >= target_pool_size:
                    break
            page_token = payload.get("nextPageToken")
            if not page_token:
                break

        enriched = self._enrich_videos(run_id, config, results)
        selected = self._apply_video_selection_policy(enriched)
        if not selected and enriched:
            self.logger.warning(
                "Video selection policy filtered out all videos; falling back to unfiltered results",
                extra={"run_id": run_id, "stage": "search_filter"},
            )
            selected = enriched
        selected = selected[: config.max_videos]
        write_json(cache_path, selected)
        return selected

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

    def _apply_video_selection_policy(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not videos:
            return []
        if not self.selection_policy.enabled:
            for item in videos:
                item["is_lg_relevant_video"] = True
                item["lg_target_brand_hit"] = False
                item["lg_product_hit"] = False
                item["lg_preferred_channel_hit"] = False
                item["selection_score"] = float(self._score_video(item, product_match=False, target_brand=False, preferred_channel=False))
            return sorted(videos, key=self._sort_key)

        selected: list[dict[str, Any]] = []
        dropped = 0
        for item in videos:
            title = str(item.get("title") or "")
            description = str(item.get("description") or "")
            channel = str(item.get("channel_title") or "")
            tags = item.get("tags") or []
            tags_blob = " ".join(str(tag) for tag in tags if tag)
            blob = f"{title} {description} {channel} {tags_blob}".lower()

            product_match = self._contains_any(blob, self.selection_policy.product_terms)
            target_brand = self._contains_any(blob, self.selection_policy.target_brand_terms)
            competitor_brand = self._contains_any(blob, self.selection_policy.competitor_brand_terms)
            preferred_channel = self._contains_any(channel.lower(), self.selection_policy.preferred_channel_keywords)
            blocked_channel = self._contains_any(channel.lower(), self.selection_policy.blocked_channel_keywords)

            keep = True
            if blocked_channel:
                keep = False
            elif self.selection_policy.require_product_term and not product_match and not preferred_channel:
                keep = False
            elif self.selection_policy.require_target_brand and not target_brand and not preferred_channel:
                keep = False
            elif self.selection_policy.exclude_competitor_brands and competitor_brand and not target_brand:
                keep = False

            if not keep:
                dropped += 1
                continue

            item["is_lg_relevant_video"] = True
            item["lg_target_brand_hit"] = bool(target_brand)
            item["lg_product_hit"] = bool(product_match)
            item["lg_preferred_channel_hit"] = bool(preferred_channel)
            item["selection_score"] = float(
                self._score_video(
                    item,
                    product_match=product_match,
                    target_brand=target_brand,
                    preferred_channel=preferred_channel,
                )
            )
            selected.append(item)

        if dropped:
            self.logger.info(
                "Video selection policy kept %d/%d videos",
                len(selected),
                len(videos),
            )
        return sorted(selected, key=self._sort_key)

    @staticmethod
    def _contains_any(text: str, terms: list[str]) -> bool:
        if not text:
            return False
        lowered = text.lower()
        for raw in terms:
            term = str(raw or "").strip().lower()
            if term and term in lowered:
                return True
        return False

    @staticmethod
    def _score_video(
        item: dict[str, Any],
        *,
        product_match: bool,
        target_brand: bool,
        preferred_channel: bool,
    ) -> float:
        search_rank = int(item.get("search_rank") or 9999)
        comment_count = int(item.get("comment_count") or 0)
        view_count = int(item.get("view_count") or 0)
        title = str(item.get("title") or "")
        keyword = str(item.get("keyword") or "")

        score = 0.0
        score += 40.0 if preferred_channel else 0.0
        score += 24.0 if target_brand else 0.0
        score += 12.0 if product_match else 0.0
        score += min(comment_count, 5000) / 120.0
        score += min(view_count, 2_000_000) / 200_000.0
        score += max(0.0, 25.0 - min(float(search_rank), 25.0))

        if keyword and title and re.search(re.escape(keyword.lower()), title.lower()):
            score += 5.0
        return score

    @staticmethod
    def _sort_key(item: dict[str, Any]) -> tuple[float, int, int]:
        return (
            -float(item.get("selection_score") or 0.0),
            int(item.get("search_rank") or 999999),
            -int(item.get("comment_count") or 0),
        )
