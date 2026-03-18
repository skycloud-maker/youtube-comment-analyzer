"""Weekly trend builders for dashboard-friendly aggregates."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.analytics.keywords import filter_business_keywords, tokenize_text



def _week_start(series: pd.Series) -> pd.Series:
    naive = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
    return naive.dt.to_period("W-SUN").dt.start_time.dt.date.astype(str)



def add_week_labels(frame: pd.DataFrame, week_column: str = "week_start") -> pd.DataFrame:
    if frame.empty or week_column not in frame.columns:
        return frame
    working = frame.copy()
    ordered_weeks = sorted(working[week_column].dropna().astype(str).unique())
    week_map = {week: f"{index + 1}주차" for index, week in enumerate(ordered_weeks)}
    working["week_label"] = working[week_column].astype(str).map(week_map)
    return working



def build_weekly_sentiment_trend(comments_df: pd.DataFrame) -> pd.DataFrame:
    if comments_df.empty or "published_at" not in comments_df.columns:
        return pd.DataFrame(columns=["week_start", "week_label", "product", "region", "sentiment_label", "count", "ratio"])
    working = comments_df.copy()
    working["published_at"] = pd.to_datetime(working["published_at"], errors="coerce", utc=True)
    working = working.dropna(subset=["published_at", "sentiment_label"])
    if working.empty:
        return pd.DataFrame(columns=["week_start", "week_label", "product", "region", "sentiment_label", "count", "ratio"])
    working["week_start"] = _week_start(working["published_at"])
    grouped = working.groupby(["week_start", "product", "region", "sentiment_label"], dropna=False).size().reset_index(name="count")
    totals = grouped.groupby(["week_start", "product", "region"], dropna=False)["count"].transform("sum")
    grouped["ratio"] = (grouped["count"] / totals).round(4)
    return add_week_labels(grouped.sort_values(["week_start", "product", "region", "sentiment_label"]).reset_index(drop=True))



def build_weekly_keyword_trend(comments_df: pd.DataFrame, top_n_per_group: int = 10) -> pd.DataFrame:
    columns = ["week_start", "week_label", "product", "region", "sentiment_label", "keyword", "count"]
    if comments_df.empty or "published_at" not in comments_df.columns:
        return pd.DataFrame(columns=columns)
    working = comments_df.copy()
    working["published_at"] = pd.to_datetime(working["published_at"], errors="coerce", utc=True)
    working = working.dropna(subset=["published_at", "sentiment_label"])
    if working.empty:
        return pd.DataFrame(columns=columns)
    working["week_start"] = _week_start(working["published_at"])

    rows: list[dict[str, object]] = []
    for group_keys, group_df in working.groupby(["week_start", "product", "region", "sentiment_label"], dropna=False):
        counter: Counter[str] = Counter()
        for text in group_df["cleaned_text"].fillna(""):
            counter.update(tokenize_text(text))
        for keyword, count in filter_business_keywords(counter, top_n_per_group):
            rows.append(
                {
                    "week_start": group_keys[0],
                    "product": group_keys[1],
                    "region": group_keys[2],
                    "sentiment_label": group_keys[3],
                    "keyword": keyword,
                    "count": count,
                }
            )
    return add_week_labels(pd.DataFrame(rows, columns=[c for c in columns if c != "week_label"]))[columns]



def build_issue_keyword_status(keyword_trend_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["product", "region", "keyword", "latest_week", "count"]
    if keyword_trend_df.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns + ["weeks_active"])
    negative_df = keyword_trend_df[keyword_trend_df["sentiment_label"] == "negative"].copy()
    if negative_df.empty:
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=columns + ["weeks_active"])
    latest_week = negative_df["week_start"].max()
    latest_df = negative_df[negative_df["week_start"] == latest_week]
    prior_df = negative_df[negative_df["week_start"] < latest_week]

    latest_keys = latest_df.groupby(["product", "region", "keyword"], dropna=False)["count"].sum().reset_index()
    prior_keys = set(tuple(row) for row in prior_df[["product", "region", "keyword"]].drop_duplicates().itertuples(index=False, name=None))
    new_rows = latest_keys[~latest_keys.apply(lambda row: (row["product"], row["region"], row["keyword"]) in prior_keys, axis=1)].copy()
    new_rows["latest_week"] = latest_week

    week_counts = negative_df.groupby(["product", "region", "keyword"], dropna=False)["week_start"].nunique().reset_index(name="weeks_active")
    persistent = latest_keys.merge(week_counts, on=["product", "region", "keyword"], how="left")
    persistent = persistent[persistent["weeks_active"] >= 2].copy()
    persistent["latest_week"] = latest_week
    return new_rows[columns], persistent[columns + ["weeks_active"]]
