# Dashboard Architecture (Refactored)

## Before

1. Streamlit startup depended on a manual "load data" button path.
2. Data visibility could break on rerun because only a flag was stored, not the full bundle.
3. Video filtering relevance rules could exclude videos too early in collection.
4. Dashboard representative-comment path re-ran local NLP/rule logic (`analyze_comment_with_context`) and diverged from pipeline output.

## After

## 1) Stable loading path

- Canonical loaders:
  - `load_dashboard_data()` for real mode
  - `load_sample_dashboard_data()` for sample mode
- Session state stores the actual bundle:
  - `dashboard_data_bundle`
  - `dashboard_data_mode`
  - `dashboard_data_signature`
- Buttons are now secondary controls:
  - sample load
  - real-data refresh
  - optional real analysis run

## 2) Video universe preservation

- Collection-time selection now annotates relevance flags without dropping rows.
- Ingestion keeps the collected video universe ordered by search rank.
- Dashboard-side filters perform narrowing/exploration.

## 3) NLP single-source alignment

- Comment-level fields are produced by analytics pipeline outputs.
- Dashboard no longer re-analyzes comments with an alternate NLP path for representative cards.
- Dashboard layer is limited to contextual interpretation and action-point generation.

## 4) Sample vs real mode separation

- Sample mode:
  - loads a compact bundle from recent runs
  - falls back to synthetic sample if no run exists
- Real mode:
  - loads all processed results
  - can be refreshed without restarting app

## Data flow summary

1. `src.main run` -> ingest -> normalize -> analytics -> `data/processed/<run_id>/*.parquet`
2. Streamlit real mode reads canonical processed outputs.
3. Streamlit sample mode reads compact/synthetic bundle for fast UI verification.
4. Dashboard applies user filters and contextual insight logic on top of pipeline outputs.
