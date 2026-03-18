"""Narrative report writer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class ReportWriter:
    """Write a markdown summary report for business stakeholders."""

    def write_report(self, *, report_path: Path, manifest: dict, videos_df: pd.DataFrame, comments_df: pd.DataFrame, analytics_outputs: dict[str, pd.DataFrame], errors_df: pd.DataFrame) -> None:
        sentiment_summary = analytics_outputs.get("sentiment_summary", pd.DataFrame())
        weekly_sentiment = analytics_outputs.get("weekly_sentiment_trend", pd.DataFrame())
        cej_negative_rate = analytics_outputs.get("cej_negative_rate", pd.DataFrame()).head(5)
        brand_ratio = analytics_outputs.get("brand_ratio", pd.DataFrame()).head(5)
        negative_density = analytics_outputs.get("negative_density", pd.DataFrame()).head(5)
        new_issue_keywords = analytics_outputs.get("new_issue_keywords", pd.DataFrame()).head(5)
        persistent_issue_keywords = analytics_outputs.get("persistent_issue_keywords", pd.DataFrame()).head(5)
        representative = analytics_outputs.get("representative_comments", pd.DataFrame()).head(8)
        quality_summary = analytics_outputs.get("quality_summary", pd.DataFrame())

        lines = [
            f"# 가전 VoC 요약 보고서: {manifest.get('config_json', {}).get('product') or manifest.get('keyword')}",
            "",
            "## 수집 개요",
            f"- 제품: {manifest.get('config_json', {}).get('product') or manifest.get('keyword')}",
            f"- 국가: {manifest.get('config_json', {}).get('region')}",
            f"- 검색 키워드: {manifest.get('keyword')}",
            f"- 분석 영상 수: {len(videos_df)}개",
            f"- 수집 댓글 수: {len(comments_df)}개",
            f"- 오류 영상 수: {len(errors_df)}개",
        ]
        if not quality_summary.empty:
            removed = quality_summary[quality_summary["comment_quality"] != "meaningful"]["count"].sum()
            lines.append(f"- 제거 댓글 수: {int(removed)}개")

        lines.extend(["", "## 감성 및 주간 변화", ""])
        if not sentiment_summary.empty:
            lines.append("- 감성 분포: " + ", ".join(f"{row.sentiment_label} {row.count}건" for row in sentiment_summary.itertuples(index=False)))
        if not weekly_sentiment.empty:
            latest_week = str(weekly_sentiment["week_label"].dropna().iloc[-1])
            latest_rows = weekly_sentiment[weekly_sentiment["week_label"] == latest_week]
            lines.append("- 최신 주차 감성 비율: " + ", ".join(f"{row.sentiment_label} {round(float(row.ratio) * 100, 1)}%" for row in latest_rows.itertuples(index=False) if row.sentiment_label in ["positive", "negative"]))

        lines.extend(["", "## 개선 포인트", ""])
        if not cej_negative_rate.empty:
            lines.append("- CEJ 단계별 부정률 상위: " + ", ".join(f"{row.topic_label} {round(float(row.negative_rate) * 100, 1)}%" for row in cej_negative_rate.itertuples(index=False)))
        if not new_issue_keywords.empty:
            lines.append("- 신규 이슈 키워드: " + ", ".join(f"{row.keyword}({row.count})" for row in new_issue_keywords.itertuples(index=False)))
        if not persistent_issue_keywords.empty:
            lines.append("- 지속 이슈 키워드: " + ", ".join(f"{row.keyword}({row.weeks_active}주)" for row in persistent_issue_keywords.itertuples(index=False)))

        lines.extend(["", "## 보조 지표", ""])
        if not brand_ratio.empty:
            lines.append("- 브랜드 언급 비율: " + ", ".join(f"{row.brand_label} {round(float(row.ratio) * 100, 1)}%" for row in brand_ratio.itertuples(index=False)))
        if not negative_density.empty:
            lines.append("- 부정 댓글 밀도 상위 영상: " + ", ".join(f"{row.title} {round(float(row.negative_density) * 100, 1)}%" for row in negative_density.itertuples(index=False)))

        lines.extend(["", "## 대표 댓글", ""])
        if representative.empty:
            lines.append("- 대표 댓글을 추출할 데이터가 충분하지 않았습니다.")
        else:
            for row in representative.itertuples(index=False):
                lines.append(f"- [{row.sentiment_label}] [{row.topic_label}] {str(row.text_display)[:120]}")

        report_path.write_text("\n".join(lines), encoding="utf-8")
