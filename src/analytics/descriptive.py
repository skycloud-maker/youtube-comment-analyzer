"""Descriptive analytics helpers."""

from __future__ import annotations

import pandas as pd



def build_descriptive_metrics(videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    replies_df = comments_df[comments_df["is_reply"] == True].copy() if not comments_df.empty else comments_df.copy()
    removed_count = int((comments_df.get("comment_validity", pd.Series(dtype=object)) == "excluded").sum()) if not comments_df.empty and "comment_validity" in comments_df else 0
    valid_count = int((comments_df.get("comment_validity", pd.Series(dtype=object)) == "valid").sum()) if not comments_df.empty and "comment_validity" in comments_df else int(len(comments_df))
    summary = pd.DataFrame(
        [
            {"metric": "total_videos", "value": int(len(videos_df))},
            {"metric": "total_comments", "value": int(len(comments_df))},
            {"metric": "valid_comments", "value": valid_count},
            {"metric": "removed_comments", "value": removed_count},
            {"metric": "total_replies", "value": int(len(replies_df))},
            {"metric": "avg_comments_per_video", "value": round(float(len(comments_df) / len(videos_df)), 2) if len(videos_df) else 0},
        ]
    )

    video_ranking = pd.DataFrame()
    if not comments_df.empty:
        comment_counts = comments_df.groupby("video_id").size().rename("comment_volume")
        engagement = comments_df.groupby("video_id")["like_count"].sum().rename("comment_like_sum")
        video_ranking = videos_df.merge(comment_counts, on="video_id", how="left").merge(engagement, on="video_id", how="left")
        video_ranking["comment_volume"] = video_ranking["comment_volume"].fillna(0).astype(int)
        video_ranking["comment_like_sum"] = video_ranking["comment_like_sum"].fillna(0).astype(int)
        video_ranking["engagement_proxy"] = video_ranking["comment_volume"] + video_ranking["comment_like_sum"]
        video_ranking = video_ranking.sort_values(["engagement_proxy", "comment_volume"], ascending=False)

    channel_comparison = pd.DataFrame()
    if not videos_df.empty:
        channel_comparison = videos_df.groupby("channel_title", dropna=False).agg(
            videos=("video_id", "nunique"),
            total_views=("view_count", "sum"),
            total_video_comments=("comment_count", "sum"),
        ).reset_index().sort_values("videos", ascending=False)

    upload_distribution = pd.DataFrame()
    if not videos_df.empty and "published_at" in videos_df:
        upload_distribution = videos_df.copy()
        upload_distribution["upload_date"] = pd.to_datetime(upload_distribution["published_at"], errors="coerce").dt.date
        upload_distribution = upload_distribution.groupby("upload_date").size().reset_index(name="video_count")

    comment_time_trend = pd.DataFrame()
    if not comments_df.empty:
        comment_time_trend = comments_df.copy()
        comment_time_trend["comment_date"] = pd.to_datetime(comment_time_trend["published_at"], errors="coerce").dt.date
        comment_time_trend = comment_time_trend.groupby("comment_date").size().reset_index(name="comment_count")

    quality_summary = pd.DataFrame()
    if not comments_df.empty and "comment_validity" in comments_df.columns:
        reason_series = comments_df["exclusion_reason"] if "exclusion_reason" in comments_df.columns else pd.Series(pd.NA, index=comments_df.index)
        quality_base = comments_df.assign(exclusion_reason=reason_series)
        quality_summary = quality_base.groupby(["comment_validity", "exclusion_reason"], dropna=False).size().reset_index(name="count").sort_values(["comment_validity", "count"], ascending=[True, False])

    return {
        "runs_summary": summary,
        "video_ranking": video_ranking,
        "channel_comparison": channel_comparison,
        "upload_distribution": upload_distribution,
        "comment_time_trend": comment_time_trend,
        "quality_summary": quality_summary,
    }
