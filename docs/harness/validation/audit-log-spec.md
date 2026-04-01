# Audit Log Spec ? YouTube Appliance VoC Dashboard Harness

## Purpose
? ??? ???? ?? ?????? ?? ??? ? ????? ??? ? ?? ??? ???? ????.
?? promo exclusion, count-display mismatch, garbled text ??, representative missing, similarity rejection, coverage miss ? ?? ???? ??? ?? ????.

## Event Families
- ingest / normalize
- sentiment labeling
- promo detection
- eligibility check
- representative selection
- summary text generation
- keyword extraction
- similarity matching
- rendering guardrail
- coverage audit

## Common Fields
- `event_id`
- `event_time`
- `run_id`
- `pipeline_version`
- `rule_version`
- `scope_type`
- `video_id`
- `comment_id`
- `filters_applied`
- `dataset_fingerprint`
- `decision`
- `reason_code`
- `reason_detail`
- `severity`

## Required Reason Codes
### Coverage
- `COVERAGE_MISSING_HOT_VIDEO`
- `CHANNEL_NOT_IN_SCOPE`
- `QUERY_NO_MATCH`
- `TRIMMED_BY_MAX_VIDEOS`
- `COMMENTS_DISABLED_OR_ZERO`
- `COMMENTS_NOT_COLLECTED`
- `COMMENTS_AUDIT_MISSING`
- `INGEST_API_ERROR`

### Promo / Exclusion
- `PROMO_DETECTED`
- `PINNED_COMMERCIAL`
- `EXCLUDED_BY_POLICY`
- `SPAM_GIBBERISH`

### Label / Representative Integrity
- `LABEL_MISMATCH`
- `COUNT_ZERO`
- `NO_VALID_CANDIDATE`
- `FILTERED_AS_PROMO`
- `DATASET_FINGERPRINT_MISMATCH`

### Text Safety
- `GARBLED_TEXT_DETECTED`
- `SANITIZED_REPLACED_WITH_NA`
- `ENCODING_ERROR`

### Keyword Quality
- `KEYWORD_CONTAINS_NOISE`
- `KEYWORD_CORPUS_EMPTY`
- `KEYWORD_NON_RELEVANT`

### Similarity
- `SIMILARITY_BELOW_THRESHOLD`
- `TOPIC_OVERLAP_LOW`
- `GIBBERISH_INPUT`
- `NO_ELIGIBLE_CANDIDATES`

## Logging Rules
- ?? ???? ?? ??? reason_code ? ???.
- ?? ???? N/A ? ???? ?? ? field ? reason_code ? ???.
- promo ??? representative / keyword / similarity ?? ????? ??.
- coverage miss ? run-level audit ?? ??? ??.

## Minimal JSON Example
```json
{
  "event_type": "PROMO_DETECTION",
  "decision": "EXCLUDE",
  "reason_code": "PROMO_DETECTED",
  "video_id": "abc123",
  "comment_id": "cmt001",
  "dataset_fingerprint": "ds:9f2a..."
}
```
