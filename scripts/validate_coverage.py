from __future__ import annotations

import json
from pathlib import Path

from src.config import load_config
from src.dashboard_app import _pick_dashboard_run_dir
from src.utils.coverage_audit import build_run_coverage_audit, summarize_coverage_audit

BASE_DIR = Path(__file__).resolve().parents[1]
REPORT_PATH = BASE_DIR / "docs" / "harness" / "generated" / "dashboard-coverage-report.md"


def _latest_run_with_manifest(processed_dir: Path) -> str | None:
    run_dir = _pick_dashboard_run_dir(processed_dir)
    return run_dir.name if run_dir is not None else None


def main() -> None:
    config = load_config()
    processed_dir = config.yaml.storage.processed_dir
    raw_dir = config.yaml.storage.raw_dir
    run_id = _latest_run_with_manifest(processed_dir)
    if run_id is None:
        payload = {"failures": ["no run_manifest-backed processed run found"], "report": str(REPORT_PATH)}
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        raise SystemExit(1)

    audit_df = build_run_coverage_audit(processed_dir, raw_dir, run_id, hot_rank_cutoff=10)
    summary = summarize_coverage_audit(audit_df)
    failures = []
    if audit_df.empty:
        failures.append("coverage audit dataset is empty")
    if (audit_df["reason_code"] == "COVERAGE_MISSING_HOT_VIDEO").any():
        failures.append("at least one hot search candidate is missing without an explicit non-missing reason")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dashboard Coverage Report",
        "",
        f"- run_id: `{run_id}`",
        f"- hot candidates checked: {summary['hot_candidates']}",
        f"- covered: {summary['covered']}",
        f"- trimmed by max_videos: {summary['trimmed']}",
        f"- missing comments: {summary['missing_comments']}",
        f"- unexplained missing hot videos: {summary['missing']}",
        "",
        "## Top Coverage Audit Rows",
    ]
    if audit_df.empty:
        lines.append("- none")
    else:
        for _, row in audit_df.head(10).iterrows():
            lines.append(
                f"- rank {int(row['search_rank'])}: `{row['video_id']}` | {row['coverage_status']} | {row['reason_code']} | api_comment_count={int(row['api_comment_count'])} | {row['title']}"
            )
    lines.append("")
    lines.append("## Failures")
    if failures:
        lines.extend(f"- {failure}" for failure in failures)
    else:
        lines.append("- none")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {"run_id": run_id, "summary": summary, "failures": failures, "report": str(REPORT_PATH)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
