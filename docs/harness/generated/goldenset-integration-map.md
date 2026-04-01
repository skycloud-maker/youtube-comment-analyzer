# Golden Set Integration Map

## Purpose
This note maps golden set cases to the current codebase so the placeholders and policy docs can be connected to real automated checks incrementally instead of all at once.

## Recommended Integration Order
1. Promo / exclusion hooks
2. Representative label and count-display hooks
3. Text safety hooks
4. Keyword quality hooks
5. Similarity integrity hooks
6. Coverage hooks

## Hook Map
### 1. Promo / exclusion
Primary files:
- [text_cleaning.py](C:/codex/src/analytics/text_cleaning.py)
- [normalize_pipeline.py](C:/codex/src/pipelines/normalize_pipeline.py)

Why this first:
- Promo leakage affects labels, representatives, keywords, and similarity at once.
- It is the cleanest first gate for GS-0001.

Expected first integration:
- A golden helper that feeds a promo-like snippet into the quality or exclusion path.
- A regression assertion that promo content is not eligible downstream.

### 2. Representative label and count-display invariants
Primary files:
- [analytics_pipeline.py](C:/codex/src/pipelines/analytics_pipeline.py)
- [opinion_units.py](C:/codex/src/analytics/opinion_units.py)
- [dashboard_app.py](C:/codex/src/dashboard_app.py)

Why next:
- These paths control representative comment construction and dashboard display.
- They are the main hook points for GS-0002, GS-0005, and GS-0006.

Expected first integration:
- Assert that `count(label) == 0` produces no representative row.
- Assert that representative rows stay in the same sentiment bucket.
- Assert that template-required metadata fields degrade safely.

### 3. Text safety / garbled render protection
Primary files:
- [dashboard_app.py](C:/codex/src/dashboard_app.py)
- [dashboard_rules.py](C:/codex/src/dashboard_rules.py)

Why here:
- Render-time sanitization is already part of the dashboard path.
- This is the clearest hook for GS-0003.

Expected first integration:
- Feed deliberately broken or replacement-character strings into the render guard.
- Assert that the UI-facing output never returns raw garbled text.

### 4. Keyword quality
Primary files:
- [keywords.py](C:/codex/src/analytics/keywords.py)
- any pipeline caller that builds label-specific keyword views

Why after core invariants:
- Keyword quality is important, but it depends on promo filtering and label purity being trustworthy first.
- This is the main hook for GS-0004.

Expected first integration:
- Build a small label-specific corpus fixture.
- Assert that promo or noise tokens do not survive top keyword extraction.

### 5. Similarity integrity
Primary files:
- current dashboard representative helpers in [dashboard_app.py](C:/codex/src/dashboard_app.py)
- any future dedicated similarity module

Why later:
- The current codebase does not yet expose a strong standalone semantic similarity contract.
- GS-0007 should start as a placeholder or heuristic test until the contract is explicit.

Expected first integration:
- Add an eligibility gate for gibberish or spam-like text.
- Assert empty similar-comment output with a reason-style explanation.

### 6. Coverage / missing hot video
Primary files:
- ingest and search orchestration paths under `src/`
- raw audit or manifest outputs in `data/raw` and `data/processed`

Why last:
- Coverage is usually a run-level contract, not a single-function unit contract.
- GS-0008 becomes meaningful once exclusion and audit logging are already stable.

Expected first integration:
- Compare expected tracked videos against actual processed outputs.
- Record an explainable exclusion or miss reason rather than silently dropping.

## What Is Ready Now
These can already be validated at the skeleton level:
- golden schema completeness
- required case anchor presence
- stable reason-code expectations
- privacy-safe snippet policy
- placeholder-aware skip behavior

## What Still Needs Real Hooking
These still need direct function-level or pipeline-level assertions:
- real promo detection result wiring
- representative selection hook from real analytics outputs
- keyword noise filtering from real extraction functions
- similarity rejection contract
- coverage miss reason logging

## Safe Execution Guidance
When connecting a new golden case to real code:
- hook one case family at a time
- add a narrow fixture first
- run only the relevant test file before broad suite reruns
- avoid mixing template, keyword, similarity, and dashboard layout changes in one patch
- if a hook changes the dataset contract, rerun dashboard validation and UI smoke before expanding scope
