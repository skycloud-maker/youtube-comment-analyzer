"""Excel workbook writer for business users."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, DoughnutChart, Reference
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter


class ExcelWriter:
    """Write a formatted Excel workbook with summary sheets and charts."""

    def __init__(self, datetime_format: str = "yyyy-mm-dd hh:mm:ss", freeze_panes: str = "A2") -> None:
        self.datetime_format = datetime_format
        self.freeze_panes = freeze_panes

    def write_workbook(self, *, workbook_path: Path, manifest: dict, videos_df: pd.DataFrame, comments_df: pd.DataFrame, analytics_outputs: dict[str, pd.DataFrame], errors_df: pd.DataFrame) -> None:
        workbook = Workbook()
        workbook.remove(workbook.active)
        workbook.properties.creator = "Codex"
        workbook.properties.title = "YouTube Comment Analysis"
        workbook.properties.subject = manifest.get("keyword", "YouTube comments")
        workbook.properties.description = "Official YouTube Data API v3 기반 분석 결과"

        self._write_readme_sheet(workbook, manifest)
        self._write_dataframe_sheet(workbook, "핵심지표", analytics_outputs.get("runs_summary", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "영상목록", videos_df)
        self._write_dataframe_sheet(workbook, "댓글원문", comments_df)
        self._write_dataframe_sheet(workbook, "영상반응순위", analytics_outputs.get("video_ranking", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "주간감성추이", analytics_outputs.get("weekly_sentiment_trend", pd.DataFrame()), sentiment=True)
        self._write_dataframe_sheet(workbook, "주간키워드", analytics_outputs.get("weekly_keyword_trend", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "긍정부정분포", analytics_outputs.get("sentiment_summary", pd.DataFrame()), sentiment=True)
        self._write_dataframe_sheet(workbook, "CEJ부정률", analytics_outputs.get("cej_negative_rate", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "브랜드언급비율", analytics_outputs.get("brand_ratio", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "영상당부정밀도", analytics_outputs.get("negative_density", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "신규이슈키워드", analytics_outputs.get("new_issue_keywords", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "지속이슈키워드", analytics_outputs.get("persistent_issue_keywords", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "제거댓글요약", analytics_outputs.get("quality_summary", pd.DataFrame()))
        self._write_dataframe_sheet(workbook, "대표댓글", analytics_outputs.get("representative_comments", pd.DataFrame()), sentiment=True)
        self._write_dataframe_sheet(workbook, "오류로그", errors_df)
        self._add_weekly_bar_chart(workbook)
        self._add_keyword_donut_chart(workbook, analytics_outputs)
        workbook.save(workbook_path)

    def _write_readme_sheet(self, workbook: Workbook, manifest: dict) -> None:
        sheet = workbook.create_sheet("README")
        rows = [
            ["항목", "설명"],
            ["프로젝트", "공식 YouTube Data API v3 기반 가전 VoC 분석"],
            ["제품", manifest.get("config_json", {}).get("product")],
            ["검색 키워드", manifest.get("keyword")],
            ["국가", manifest.get("config_json", {}).get("region")],
            ["실행 ID", manifest.get("run_id")],
            ["수집 시작", manifest.get("started_at")],
            ["수집 종료", manifest.get("finished_at")],
            ["설정", str(manifest.get("config_json"))],
            ["쿼터 추정", str(manifest.get("quota_estimate"))],
            ["실제 API 호출", str(manifest.get("api_calls_json"))],
        ]
        for row in rows:
            sheet.append(row)
        sheet.column_dimensions["A"].width = 20
        sheet.column_dimensions["B"].width = 120
        sheet.freeze_panes = "A2"
        sheet["A1"].font = Font(bold=True)
        sheet["B1"].font = Font(bold=True)

    def _write_dataframe_sheet(self, workbook: Workbook, title: str, df: pd.DataFrame, sentiment: bool = False) -> None:
        sheet = workbook.create_sheet(title)
        if df.empty:
            sheet.append(["message"])
            sheet.append(["No data available"])
            return
        normalized_df = df.apply(lambda column: column.map(self._normalize_cell_value))
        sheet.append(list(normalized_df.columns))
        for row in normalized_df.fillna("").itertuples(index=False, name=None):
            sheet.append(list(row))
        sheet.freeze_panes = self.freeze_panes
        sheet.auto_filter.ref = sheet.dimensions
        for cell in sheet[1]:
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill("solid", fgColor="0E5A47")
        for column_cells in sheet.columns:
            max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
            sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = min(max_length + 2, 60)
        if sentiment and "sentiment_label" in normalized_df.columns:
            sentiment_col = list(normalized_df.columns).index("sentiment_label") + 1
            col_letter = get_column_letter(sentiment_col)
            sheet.conditional_formatting.add(f"{col_letter}2:{col_letter}{sheet.max_row}", CellIsRule(operator="equal", formula=['"positive"'], fill=PatternFill("solid", fgColor="D7F5E8")))
            sheet.conditional_formatting.add(f"{col_letter}2:{col_letter}{sheet.max_row}", CellIsRule(operator="equal", formula=['"negative"'], fill=PatternFill("solid", fgColor="FCE0DD")))
            sheet.conditional_formatting.add(f"{col_letter}2:{col_letter}{sheet.max_row}", CellIsRule(operator="equal", formula=['"neutral"'], fill=PatternFill("solid", fgColor="F6F0D8")))

    @staticmethod
    def _normalize_cell_value(value):
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            value = value.tolist()
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return value

    def _add_weekly_bar_chart(self, workbook: Workbook) -> None:
        sheet_name = "주간감성추이"
        if sheet_name not in workbook.sheetnames:
            return
        source = workbook[sheet_name]
        if source.max_row < 2:
            return
        chart_sheet = workbook.create_sheet("주간감성차트")
        chart_sheet.append(["주차", "긍정 응답률", "부정 응답률"])
        headers = [cell.value for cell in source[1]]
        week_idx = headers.index("week_label") + 1 if "week_label" in headers else None
        sentiment_idx = headers.index("sentiment_label") + 1 if "sentiment_label" in headers else None
        ratio_idx = headers.index("ratio") + 1 if "ratio" in headers else None
        if not all([week_idx, sentiment_idx, ratio_idx]):
            return
        rows: dict[str, dict[str, float]] = {}
        for row_idx in range(2, source.max_row + 1):
            week_label = source.cell(row=row_idx, column=week_idx).value
            sentiment = source.cell(row=row_idx, column=sentiment_idx).value
            ratio = source.cell(row=row_idx, column=ratio_idx).value
            if not week_label or sentiment not in {"positive", "negative"}:
                continue
            rows.setdefault(str(week_label), {"positive": 0.0, "negative": 0.0})
            rows[str(week_label)][str(sentiment)] = float(ratio)
        for week_label, values in rows.items():
            chart_sheet.append([week_label, values.get("positive", 0.0), values.get("negative", 0.0)])
        chart = BarChart()
        chart.type = "col"
        chart.style = 10
        chart.title = "주차별 긍정/부정 응답률"
        chart.y_axis.title = "응답률"
        chart.x_axis.title = "주차"
        data = Reference(chart_sheet, min_col=2, min_row=1, max_col=3, max_row=chart_sheet.max_row)
        cats = Reference(chart_sheet, min_col=1, min_row=2, max_row=chart_sheet.max_row)
        chart.add_data(data, titles_from_data=True)
        chart.set_categories(cats)
        chart.height = 8
        chart.width = 16
        chart_sheet.add_chart(chart, "E2")

    def _add_keyword_donut_chart(self, workbook: Workbook, analytics_outputs: dict[str, pd.DataFrame]) -> None:
        keyword_df = analytics_outputs.get("weekly_keyword_trend", pd.DataFrame())
        if keyword_df.empty:
            return
        donut_sheet = workbook.create_sheet("키워드도넛차트")
        for offset, sentiment in enumerate(["negative", "positive"]):
            subset = keyword_df[keyword_df["sentiment_label"] == sentiment].groupby("keyword", as_index=False)["count"].sum().sort_values("count", ascending=False).head(6)
            if subset.empty:
                continue
            start_row = 1 if offset == 0 else 20
            donut_sheet.cell(row=start_row, column=1, value="키워드")
            donut_sheet.cell(row=start_row, column=2, value="비중")
            for idx, row in enumerate(subset.itertuples(index=False), start=1):
                donut_sheet.cell(row=start_row + idx, column=1, value=row.keyword)
                donut_sheet.cell(row=start_row + idx, column=2, value=row.count)
            chart = DoughnutChart()
            chart.title = "부정 키워드 비중" if sentiment == "negative" else "긍정 키워드 비중"
            labels = Reference(donut_sheet, min_col=1, min_row=start_row + 1, max_row=start_row + len(subset))
            data = Reference(donut_sheet, min_col=2, min_row=start_row, max_row=start_row + len(subset))
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(labels)
            chart.height = 7
            chart.width = 8
            donut_sheet.add_chart(chart, "D2" if sentiment == "negative" else "D20")
