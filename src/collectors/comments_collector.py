"""Comment collection with full reply retrieval via official API."""

from __future__ import annotations

import logging
from typing import Any

from src.api.youtube_client import YoutubeApiError, YoutubeClient


class CommentsCollector:
    """Collect top-level comments and all replies for each video."""

    def __init__(self, client: YoutubeClient, logger: logging.Logger | None = None) -> None:
        self.client = client
        self.logger = logger or logging.getLogger(__name__)

    def collect_video_comments(
        self,
        *,
        run_id: str,
        video_id: str,
        comments_per_video: int,
        include_replies: bool,
        comments_order: str = "time",
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        thread_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        page_token: str | None = None
        fetched_threads = 0
        fetch_all = comments_per_video <= 0

        while fetch_all or fetched_threads < comments_per_video:
            params: dict[str, Any] = {
                "part": "snippet,replies",
                "videoId": video_id,
                "textFormat": "plainText",
                "maxResults": 100 if fetch_all else min(100, comments_per_video - fetched_threads),
                "order": comments_order,
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                payload = self.client.list_comment_threads(run_id=run_id, video_id=video_id, **params)
            except YoutubeApiError as exc:
                self.logger.warning("Skipping video due to comment API error", extra={"run_id": run_id, "video_id": video_id, "stage": "collect_comments"})
                error_rows.append({"run_id": run_id, "video_id": video_id, "error": str(exc)})
                break

            items = payload.get("items", [])
            for item in items:
                thread_rows.extend(self._normalize_thread(run_id, video_id, item, include_replies))
            fetched_threads += len(items)
            page_token = payload.get("nextPageToken")
            if not page_token or not items:
                break
        return thread_rows, error_rows

    def _normalize_thread(self, run_id: str, video_id: str, thread: dict[str, Any], include_replies: bool) -> list[dict[str, Any]]:
        snippet = thread.get("snippet", {})
        top_level = snippet.get("topLevelComment", {})
        top_snippet = top_level.get("snippet", {})
        top_comment_id = top_level.get("id")
        thread_id = thread.get("id") or top_comment_id
        reply_count = int(snippet.get("totalReplyCount", 0) or 0)

        rows = [self._comment_row(run_id=run_id, video_id=video_id, thread_id=thread_id, comment_id=top_comment_id, parent_comment_id=None, is_reply=False, snippet=top_snippet, reply_count=reply_count, total_reply_count=reply_count)]

        if not include_replies or reply_count == 0 or not top_comment_id:
            return rows

        embedded_replies = thread.get("replies", {}).get("comments", [])
        for reply in embedded_replies:
            rows.append(self._comment_row(run_id=run_id, video_id=video_id, thread_id=thread_id, comment_id=reply.get("id"), parent_comment_id=top_comment_id, is_reply=True, snippet=reply.get("snippet", {}), reply_count=0, total_reply_count=reply_count))

        if reply_count > len(embedded_replies):
            fetched_ids = {reply.get("id") for reply in embedded_replies}
            page_token: str | None = None
            while True:
                payload = self.client.list_comments(run_id=run_id, video_id=video_id, part="snippet", parentId=top_comment_id, textFormat="plainText", maxResults=100, pageToken=page_token)
                for reply in payload.get("items", []):
                    reply_id = reply.get("id")
                    if reply_id in fetched_ids:
                        continue
                    fetched_ids.add(reply_id)
                    rows.append(self._comment_row(run_id=run_id, video_id=video_id, thread_id=thread_id, comment_id=reply_id, parent_comment_id=top_comment_id, is_reply=True, snippet=reply.get("snippet", {}), reply_count=0, total_reply_count=reply_count))
                page_token = payload.get("nextPageToken")
                if not page_token:
                    break
        return rows

    @staticmethod
    def _comment_row(*, run_id: str, video_id: str, thread_id: str, comment_id: str | None, parent_comment_id: str | None, is_reply: bool, snippet: dict[str, Any], reply_count: int, total_reply_count: int) -> dict[str, Any]:
        author_channel = snippet.get("authorChannelId", {})
        author_channel_id = author_channel.get("value") if isinstance(author_channel, dict) else None
        text_original = snippet.get("textOriginal") or snippet.get("textDisplay") or ""
        text_display = snippet.get("textDisplay") or text_original
        return {
            "run_id": run_id,
            "video_id": video_id,
            "thread_id": thread_id,
            "comment_id": comment_id,
            "parent_comment_id": parent_comment_id,
            "is_reply": is_reply,
            "author_display_name": snippet.get("authorDisplayName"),
            "author_channel_id": author_channel_id,
            "text_original": text_original,
            "text_display": text_display,
            "like_count": int(snippet.get("likeCount", 0) or 0),
            "published_at": snippet.get("publishedAt"),
            "updated_at": snippet.get("updatedAt"),
            "reply_count": reply_count,
            "total_reply_count": total_reply_count,
            "viewer_rating": snippet.get("viewerRating"),
        }
