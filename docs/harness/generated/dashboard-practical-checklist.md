# Dashboard Practical Check Checklist

## Purpose
A short practical checklist for quickly verifying that the dashboard still behaves like a usable decision interface after code or harness changes.

## Quick Open
1. Run `run_dashboard.bat`.
2. Open `http://localhost:8501`.
3. Also open `http://localhost:8501/?mode=lite`.

## Filter Safety
- Remove one country and confirm the page does not crash.
- Switch `?? ??` between `??`, `?? ?? ???`, `?? ??`.
- Narrow products to one item and confirm cards, charts, and representative comments still render.
- Apply a keyword query and confirm empty results show an explicit empty state rather than unrelated data.

## Representative Comments
- Confirm representative cards show a clear issue-oriented comment, not a random short reaction.
- Confirm metadata and explanation text do not contain `???` or broken text.
- Confirm similar comments can be empty safely without throwing errors.

## Download
- Download the raw workbook.
- Confirm the workbook opens normally.
- Confirm major sheets are present and aligned with the visible dashboard scope.

## Lite / Full
- In full mode, the app should load without exception.
- In lite mode, the app should also load without exception.
- The analysis scope radio and major filters should remain interactive in both modes.

## Current Automated Backing
- Streamlit UI smoke passed on `2026-04-01`.
- Dashboard smoke validation passed on `2026-04-01` with 6 scenarios and 0 failures.
- Golden regression passed on `2026-04-01` with 18 passed and 1 skipped.
