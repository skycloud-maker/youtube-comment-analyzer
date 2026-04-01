# Golden Set TBD Replacement Guide

## Purpose
This guide explains how to replace `TBD_*` placeholders in [goldenset_seed.yaml](C:/codex/docs/harness/validation/goldenset_seed.yaml) with real `video_id` and `comment_id` values from processed dashboard data.

## Replacement Principles
- Replace placeholders only after confirming the case really matches the intended failure type.
- Prefer stable, reproducible examples over recent one-off examples.
- Keep one golden item mapped to one clear anchor case.
- Do not store full original comments in the seed file. Keep short snippets only.

## Required Source Data
Use processed parquet outputs under `C:/codex/data/processed/<run_id>/`.
The most useful files are:
- `comments_normalized.parquet`
- `videos_raw.parquet`
- any downstream analytics parquet that already contains sentiment, stage, inquiry, or representative bundle information

## Minimum Mapping Workflow
1. Choose the target golden item.
2. Identify the expected case tag and reason code.
3. Find a matching `video_id` in processed video data.
4. Find one or more matching `comment_id` rows in normalized comments.
5. Verify the case against the acceptance criteria.
6. Replace only the placeholder fields you can prove.
7. Leave uncertain fields as `TBD_*` rather than guessing.

## How To Find Real IDs
### GS-0001 Promo / pinned commercial
Look for comments containing affiliate or commercial markers such as:
- `??????`
- `?????`
- `??`
- `??`
- `????`

Candidate selection rule:
- The comment must clearly be commercial or link bait.
- It should be excluded from representative, keyword, and similarity pipelines.

### GS-0002 Count zero but representative must not appear
This is usually a scenario anchor rather than a single comment anchor.
Replacement rule:
- Find a real filter combination or video where one label count is truly zero.
- Use a stable `video_id` and a reference `comment_id` only if needed for traceability.
- Keep the expected reason code as `COUNT_ZERO`.

### GS-0003 Garbled text / encoding safety
Look for rows or rendered outputs where summary or metadata previously showed `???`, replacement characters, or broken text.
Replacement rule:
- Prefer cases reproducible from source data or a deterministic render path.
- If the problem is generated later in the UI, keep the golden anchored to the render field rather than a random comment.

### GS-0004 Keyword noise
Find a segment where top keywords would be polluted by promo or generic words without cleanup.
Replacement rule:
- Confirm the underlying comments include noise terms.
- Confirm the golden is tied to a label-specific corpus.

### GS-0005 Representative template compliance
Pick a representative-worthy comment with a clear product complaint or praise.
Replacement rule:
- The comment should support short evidence snippets.
- The linked video should have a stable title suitable for template checks.

### GS-0006 Label / representative mismatch
Find a historical mismatch case if available, or a stable case that would fail if label gating broke.
Replacement rule:
- The chosen anchor should make label mismatch detection obvious.

### GS-0007 Gibberish / similarity rejection
Find spam, gibberish, or meaningless text that should never receive similar comments.
Replacement rule:
- The source text must be clearly ineligible for meaningful similarity.

### GS-0008 Coverage / missing hot video
This is usually a run-level or coverage anchor.
Replacement rule:
- Map to a `video_id` only when you can explain why it was missing.
- Keep the case tied to a coverage reason code, not a sentiment example.

## Practical Query Hints
Useful columns in normalized comments commonly include:
- `comment_id`
- `video_id`
- `text_original`
- `text_display`
- `cleaned_text`
- `language_detected`
- `comment_quality`

Useful columns in raw videos commonly include:
- `video_id`
- `title`
- `channel_title`
- `keyword`
- `product`
- `region`
- `view_count`
- `comment_count`

## Replacement Rules
- Replace `TBD_VIDEO_*` only when the selected video is stable and reproducible.
- Replace `TBD_COMMENT_*` only when the selected comment clearly matches the case intent.
- Update `input_snippet` only with masked, short evidence.
- Do not silently change `case_tags` or expected `reason_code` during ID replacement.

## Review Checklist
- Does the real comment truly match the case tag?
- Would another reviewer reach the same conclusion?
- Is the snippet short and privacy-safe?
- Does the mapped item still satisfy the acceptance criteria?
- Is the mapping documented in the PR or validation note?
