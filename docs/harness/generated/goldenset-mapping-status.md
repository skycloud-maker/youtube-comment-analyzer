# Golden Set Mapping Status

## Current Status

- `GS-0001`: Still placeholder
  case_tags: PROMO, PINNED_COMMERCIAL
  video_id: TBD_VIDEO_DRYER_COMPARE_202601
  comment_id: TBD_COMMENT_PINNED_AFFILIATE
- `GS-0002`: Still placeholder
  case_tags: COUNT_REP_MISMATCH
  video_id: TBD_VIDEO_WITH_POSITIVE_ZERO
  comment_id: TBD_SCENARIO_ANCHOR_POS_ZERO
- `GS-0003`: Mapped to real data
  case_tags: GARBLED_TEXT
  video_id: T28X0JpDziM
  comment_id: Ugz5zwLDjdcR93Uo5yZ4AaABAg
- `GS-0004`: Still placeholder
  case_tags: KEYWORD_NOISE
  video_id: TBD_VIDEO_KEYWORDS_NOISE
  comment_id: TBD_COMMENT_WITH_NOISE_TERMS
- `GS-0005`: Mapped to real data
  case_tags: TEMPLATE_NON_COMPLIANCE
  video_id: T28X0JpDziM
  comment_id: UgwpacANPoRQJX__2qt4AaABAg
- `GS-0006`: Still placeholder
  case_tags: LABEL_REP_MISMATCH
  video_id: TBD_VIDEO_LABEL_REP_MISMATCH
  comment_id: TBD_COMMENT_LABEL_MISMATCH
- `GS-0007`: Mapped to real data
  case_tags: SIMILARITY_IRRELEVANT, GIBBERISH_INPUT
  video_id: T28X0JpDziM
  comment_id: UgyspMuanvykyKevDJl4AaABAg
- `GS-0008`: Still placeholder
  case_tags: COVERAGE_MISSING_HOT_VIDEO
  video_id: TBD_VIDEO_HOT_MISSING
  comment_id: TBD_SCENARIO_COVERAGE

## Why Some Cases Remain Placeholder

- `GS-0001` promo/pinned commercial: no stable processed-data anchor was confirmed yet, so this stays placeholder rather than guessing.
- `GS-0002` count-zero scenario: this is better anchored to a deterministic filter scenario or analytics output, not a random single comment.
- `GS-0004` keyword noise: this should anchor to a label-specific segment and extracted top keywords, so a single comment is not enough.
- `GS-0006` label mismatch: this is best tied to a known regression scenario or representative-selection output.
- `GS-0008` coverage miss: this is a run-level contract and needs audit or manifest linkage before using a real `video_id`.

## Recommended Next Hook Targets

1. `GS-0003` -> render guard / garbled text sanitizer
2. `GS-0007` -> similarity and eligibility gate for meaningless input
3. `GS-0005` -> representative template compliance checks
4. `GS-0001` -> promo detection once a stable commercial anchor is located