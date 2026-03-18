import pandas as pd

from src.collectors.comments_collector import CommentsCollector
from src.pipelines.normalize_pipeline import NormalizePipeline
from src.pipelines.ingest_pipeline import IngestPipeline


class DummyClient:
    def __init__(self, reply_payload):
        self.reply_payload = reply_payload

    def list_comments(self, **kwargs):
        return self.reply_payload



def test_reply_completion_logic(tmp_path):
    client = DummyClient(
        {
            "items": [
                {
                    "id": "reply-2",
                    "snippet": {
                        "textOriginal": "추가 대댓글",
                        "textDisplay": "추가 대댓글",
                        "authorDisplayName": "사용자B",
                        "likeCount": 1,
                    },
                }
            ]
        }
    )
    collector = CommentsCollector(client)
    thread = {
        "id": "thread-1",
        "snippet": {
            "totalReplyCount": 2,
            "topLevelComment": {
                "id": "comment-1",
                "snippet": {
                    "textOriginal": "좋아요",
                    "textDisplay": "좋아요",
                    "authorDisplayName": "사용자A",
                    "likeCount": 3,
                },
            },
        },
        "replies": {
            "comments": [
                {
                    "id": "reply-1",
                    "snippet": {
                        "textOriginal": "맞아요",
                        "textDisplay": "맞아요",
                        "authorDisplayName": "사용자C",
                        "likeCount": 0,
                    },
                }
            ]
        },
    }
    rows = collector._normalize_thread("run-1", "video-1", thread, include_replies=True)
    reply_ids = {row["comment_id"] for row in rows if row["is_reply"]}
    assert reply_ids == {"reply-1", "reply-2"}



def test_normalization_flattens_and_cleans(tmp_path):
    pipeline = NormalizePipeline(tmp_path)
    videos_df = pd.DataFrame([{"run_id": "run-1", "video_id": "video-1", "title": "테스트"}])
    comments_df = pd.DataFrame(
        [
            {
                "run_id": "run-1",
                "video_id": "video-1",
                "thread_id": "thread-1",
                "comment_id": "comment-1",
                "parent_comment_id": None,
                "is_reply": False,
                "text_original": "정말 좋아요!!!! https://example.com",
                "text_display": "정말 좋아요!!!! https://example.com",
            },
            {
                "run_id": "run-1",
                "video_id": "video-1",
                "thread_id": "thread-1",
                "comment_id": "comment-1",
                "parent_comment_id": None,
                "is_reply": False,
                "text_original": "정말 좋아요!!!! https://example.com",
                "text_display": "정말 좋아요!!!! https://example.com",
            },
        ]
    )
    _, normalized = pipeline.run("run-1", videos_df, comments_df)
    assert len(normalized) == 1
    assert normalized.loc[0, "cleaned_text"] == "정말 좋아요!!"
    assert normalized.loc[0, "language_detected"] == "ko"
    assert normalized.loc[0, "analysis_included"] == False


class PagingClient:
    def __init__(self):
        self.calls = []

    def list_comment_threads(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            return {
                "items": [
                    {
                        "id": "thread-1",
                        "snippet": {
                            "totalReplyCount": 0,
                            "topLevelComment": {
                                "id": "comment-1",
                                "snippet": {"textOriginal": "첫 댓글", "textDisplay": "첫 댓글"},
                            },
                        },
                    }
                ],
                "nextPageToken": "page-2",
            }
        return {
            "items": [
                {
                    "id": "thread-2",
                    "snippet": {
                        "totalReplyCount": 0,
                        "topLevelComment": {
                            "id": "comment-2",
                            "snippet": {"textOriginal": "둘째 댓글", "textDisplay": "둘째 댓글"},
                        },
                    },
                }
            ]
        }



def test_collect_video_comments_fetches_all_pages_when_limit_zero():
    collector = CommentsCollector(PagingClient())
    rows, errors = collector.collect_video_comments(
        run_id="run-1",
        video_id="video-1",
        comments_per_video=0,
        include_replies=False,
    )
    assert errors == []
    assert {row["comment_id"] for row in rows} == {"comment-1", "comment-2"}



def test_merge_comments_keeps_new_rows_and_deduplicates():
    existing = pd.DataFrame([
        {"video_id": "video-1", "comment_id": "comment-1", "text_original": "old"},
        {"video_id": "video-1", "comment_id": "comment-2", "text_original": "stay"},
    ])
    new_rows = pd.DataFrame([
        {"video_id": "video-1", "comment_id": "comment-1", "text_original": "new"},
        {"video_id": "video-1", "comment_id": "comment-3", "text_original": "added"},
    ])
    merged = IngestPipeline._merge_comments(existing, new_rows)
    assert len(merged) == 3
    assert merged.loc[merged["comment_id"] == "comment-1", "text_original"].iloc[0] == "new"
