"""Export pipeline for Excel workbook and markdown report."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.exporters.excel_writer import ExcelWriter
from src.exporters.report_writer import ReportWriter
from src.utils.io import ensure_dir


class ExportPipeline:
    """Render stakeholder-friendly outputs."""

    def __init__(self, exports_dir: Path, excel_writer: ExcelWriter, report_writer: ReportWriter, logger: logging.Logger | None = None) -> None:
        self.exports_dir = ensure_dir(exports_dir)
        self.excel_writer = excel_writer
        self.report_writer = report_writer
        self.logger = logger or logging.getLogger(__name__)

    def run(self, *, run_id: str, output_prefix: str, manifest: dict, videos_df: pd.DataFrame, analytics_outputs: dict[str, pd.DataFrame], errors_df: pd.DataFrame) -> tuple[Path, Path]:
        run_export_dir = ensure_dir(self.exports_dir / run_id)
        workbook_path = run_export_dir / f"{output_prefix}_{run_id}.xlsx"
        report_path = run_export_dir / f"{output_prefix}_{run_id}_summary.md"
        self.excel_writer.write_workbook(workbook_path=workbook_path, manifest=manifest, videos_df=videos_df, comments_df=analytics_outputs.get("comments", pd.DataFrame()), analytics_outputs=analytics_outputs, errors_df=errors_df)
        self.report_writer.write_report(report_path=report_path, manifest=manifest, videos_df=videos_df, comments_df=analytics_outputs.get("comments", pd.DataFrame()), analytics_outputs=analytics_outputs, errors_df=errors_df)
        self.logger.info("Export stage completed", extra={"run_id": run_id, "stage": "export"})
        return workbook_path, report_path
