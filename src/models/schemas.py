"""Pydantic schemas and normalized domain objects."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field



def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class RunConfig(BaseModel):
    keyword: str
    keywords: list[str] = Field(default_factory=list)
    product: str | None = None
    max_videos: int = 100
    comments_per_video: int = 1000
    comments_order: str = "time"
    published_after: str | None = None
    published_before: str | None = None
    order: str = "relevance"
    language: str = "ko"
    region: str = "KR"
    channel_id: str | None = None
    include_replies: bool = True
    refresh_existing: bool = False
    search_oversample_factor: int = 1
    output_prefix: str = "youtube_analysis"


class RunManifest(BaseModel):
    run_id: str = Field(default_factory=lambda: uuid4().hex)
    keyword: str
    started_at: str = Field(default_factory=utc_now_iso)
    finished_at: str | None = None
    config_json: dict[str, Any]
    quota_estimate: dict[str, Any] = Field(default_factory=dict)
    api_calls_json: dict[str, int] = Field(default_factory=dict)
    error_count: int = 0
    videos_collected: int = 0
    comments_collected: int = 0
    failed_video_ids: list[str] = Field(default_factory=list)


class VideoRecord(BaseModel):
    run_id: str
    keyword: str
    product: str | None = None
    region: str | None = None
    video_id: str
    title: str | None = None
    description: str | None = None
    channel_id: str | None = None
    channel_title: str | None = None
    published_at: str | None = None
    duration: str | None = None
    view_count: int | None = None
    like_count: int | None = None
    comment_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    category_id: str | None = None
    search_rank: int | None = None
    matched_keyword: str | None = None


class CommentRecord(BaseModel):
    run_id: str
    video_id: str
    thread_id: str
    comment_id: str
    parent_comment_id: str | None = None
    is_reply: bool
    author_display_name: str | None = None
    author_channel_id: str | None = None
    text_original: str = ""
    text_display: str = ""
    like_count: int = 0
    published_at: str | None = None
    updated_at: str | None = None
    reply_count: int = 0
    total_reply_count: int = 0
    viewer_rating: str | None = None
    text_length: int = 0
    cleaned_text: str = ""
    language_detected: str | None = None
    sentiment_label: str | None = None
    sentiment_score: float | None = None
    topic_label: str | None = None
    cluster_id: int | None = None


class RunRow(BaseModel):
    run_id: str
    keyword: str
    started_at: str
    finished_at: str | None = None
    config_json: dict[str, Any]
    api_calls_json: dict[str, int]
    error_count: int
    videos_collected: int
    comments_collected: int


class ApiPageAudit(BaseModel):
    endpoint: str
    run_id: str
    page_token: str | None = None
    video_id: str | None = None
    collected_at: str = Field(default_factory=utc_now_iso)
    payload: dict[str, Any]
