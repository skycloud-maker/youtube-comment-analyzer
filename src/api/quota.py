"""Quota estimation and tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil

YOUTUBE_QUOTA_COSTS = {
    "search.list": 100,
    "videos.list": 1,
    "commentThreads.list": 1,
    "comments.list": 1,
}


@dataclass
class QuotaLedger:
    """Track estimated and actual quota usage by API method."""

    daily_limit: int
    actual_calls: dict[str, int] = field(default_factory=dict)

    def record(self, method: str, units: int = 1) -> None:
        self.actual_calls[method] = self.actual_calls.get(method, 0) + units

    def actual_units(self) -> int:
        return sum(YOUTUBE_QUOTA_COSTS.get(method, 0) * count for method, count in self.actual_calls.items())

    def warn_threshold_exceeded(self, estimated_units: int) -> bool:
        return estimated_units > self.daily_limit



def estimate_quota_usage(*, max_videos: int, comments_per_video: int, include_replies: bool, page_size: int = 50) -> dict[str, int]:
    search_calls = ceil(max_videos / page_size) if max_videos else 0
    videos_calls = ceil(max_videos / page_size) if max_videos else 0
    effective_comments = comments_per_video if comments_per_video and comments_per_video > 0 else page_size * 20
    threads_calls = max_videos * ceil(effective_comments / page_size) if effective_comments else 0
    reply_calls = ceil(threads_calls * 0.3) if include_replies else 0
    total_units = (
        search_calls * YOUTUBE_QUOTA_COSTS["search.list"]
        + videos_calls * YOUTUBE_QUOTA_COSTS["videos.list"]
        + threads_calls * YOUTUBE_QUOTA_COSTS["commentThreads.list"]
        + reply_calls * YOUTUBE_QUOTA_COSTS["comments.list"]
    )
    return {
        "search.list": search_calls,
        "videos.list": videos_calls,
        "commentThreads.list": threads_calls,
        "comments.list": reply_calls,
        "total_units": total_units,
    }
