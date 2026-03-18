"""Thin YouTube Data API v3 client with quota tracking and raw audit persistence."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests

from src.api.quota import QuotaLedger
from src.models.schemas import ApiPageAudit
from src.utils.io import ensure_dir, write_json
from src.utils.retry import with_retry


class YoutubeApiError(RuntimeError):
    """Raised when the YouTube API returns a non-recoverable error."""


class YoutubeClient:
    """API wrapper for search, video metadata, comments, and replies."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        max_retries: int,
        backoff_seconds: float,
        raw_dir: Path,
        quota_ledger: QuotaLedger,
        logger: logging.Logger | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.backoff_seconds = backoff_seconds
        self.raw_dir = ensure_dir(raw_dir)
        self.quota_ledger = quota_ledger
        self.logger = logger or logging.getLogger(__name__)
        self.session = requests.Session()

    def _request(
        self,
        endpoint: str,
        params: dict[str, Any],
        *,
        run_id: str,
        page_token: str | None = None,
        video_id: str | None = None,
        quota_method: str,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        query = {**params, "key": self.api_key}

        def call() -> dict[str, Any]:
            response = self.session.get(url, params=query, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                if response.status_code in {403, 404}:
                    self.logger.warning("YouTube API returned a handled client error", extra={"endpoint": endpoint, "video_id": video_id, "page_token": page_token})
                response.raise_for_status()
            return response.json()

        try:
            payload = with_retry(call, logger=self.logger, max_retries=self.max_retries, backoff_seconds=self.backoff_seconds)
        except requests.HTTPError as exc:
            message = exc.response.text if exc.response is not None else str(exc)
            raise YoutubeApiError(message) from exc

        self.quota_ledger.record(quota_method)
        audit = ApiPageAudit(endpoint=quota_method, run_id=run_id, page_token=page_token, video_id=video_id, payload=payload)
        audit_path = self.raw_dir / run_id / quota_method.replace(".", "_") / f"{video_id or 'global'}_{page_token or 'first'}.json"
        write_json(audit_path, audit.model_dump(mode="json"))
        return payload

    def search_videos(self, *, run_id: str, **params: Any) -> dict[str, Any]:
        return self._request("search", params, run_id=run_id, page_token=params.get("pageToken"), quota_method="search.list")

    def list_videos(self, *, run_id: str, **params: Any) -> dict[str, Any]:
        return self._request("videos", params, run_id=run_id, quota_method="videos.list")

    def list_comment_threads(self, *, run_id: str, video_id: str, **params: Any) -> dict[str, Any]:
        return self._request("commentThreads", params, run_id=run_id, page_token=params.get("pageToken"), video_id=video_id, quota_method="commentThreads.list")

    def list_comments(self, *, run_id: str, video_id: str, **params: Any) -> dict[str, Any]:
        return self._request("comments", params, run_id=run_id, page_token=params.get("pageToken"), video_id=video_id, quota_method="comments.list")
