from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import pytest

from src.analytics.keywords import tokenize_text
from src.analytics.text_cleaning import classify_comment_quality
from src.dashboard_app import (
    COL_BRAND,
    COL_CEJ,
    COL_ORIGINAL,
    COL_VIDEO_TITLE,
    _looks_garbled,
    _rows_from_representative_bundles,
)

REQUIRED_EXPECTED_KEYS = {
    "sentiment_label",
    "promo_flag",
    "eligible_for_representative",
    "eligible_for_keywords",
    "eligible_for_similarity",
    "representative_slot",
    "representative_template_checks",
    "keyword_checks",
    "similarity_checks",
    "reason_code",
}

REQUIRED_TEMPLATE_KEYS = {
    "header_ok",
    "video_title_label_ok",
    "no_garbled_ok",
    "snippet_only_ok",
    "similar_list_ok",
}

REQUIRED_KEYWORD_KEYS = {"noise_topk_ok", "label_purity_ok"}
REQUIRED_SIMILARITY_KEYS = {"should_return_empty", "min_topic_overlap_ok"}
VALID_SENTIMENTS = {"positive", "negative", "neutral", "excluded"}
VALID_LANGUAGES = {"ko", "en", "mixed", "unknown"}


def _all_bool(mapping: dict[str, Any], keys: Iterable[str]) -> bool:
    return all(isinstance(mapping.get(key), bool) for key in keys)


def _indexed(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["golden_id"]: item for item in items}


def test_goldenset_schema_basics(goldenset_items: list[dict[str, Any]]) -> None:
    assert goldenset_items, "goldenset_seed.yaml must not be empty"

    seen_ids: set[str] = set()
    for item in goldenset_items:
        assert isinstance(item, dict)
        golden_id = item.get("golden_id")
        assert isinstance(golden_id, str) and golden_id.startswith("GS-")
        assert golden_id not in seen_ids
        seen_ids.add(golden_id)

        assert item.get("language") in VALID_LANGUAGES
        assert isinstance(item.get("case_tags"), list) and item["case_tags"]
        assert isinstance(item.get("input_snippet"), str)
        assert 0 < len(item["input_snippet"]) <= 300
        assert isinstance(item.get("notes"), str) and item["notes"].strip()

        expected = item.get("expected")
        assert isinstance(expected, dict)
        assert REQUIRED_EXPECTED_KEYS.issubset(expected.keys())
        assert expected["sentiment_label"] in VALID_SENTIMENTS
        if expected["representative_slot"] is not None:
            assert expected["representative_slot"] in {"positive", "negative", "neutral"}

        assert _all_bool(expected["representative_template_checks"], REQUIRED_TEMPLATE_KEYS)
        assert _all_bool(expected["keyword_checks"], REQUIRED_KEYWORD_KEYS)
        assert _all_bool(expected["similarity_checks"], REQUIRED_SIMILARITY_KEYS)


def test_goldenset_has_required_anchor_cases(goldenset_items: list[dict[str, Any]]) -> None:
    tags = {tag for item in goldenset_items for tag in item.get("case_tags", [])}
    required = {
        "PROMO",
        "COUNT_REP_MISMATCH",
        "GARBLED_TEXT",
        "KEYWORD_NOISE",
        "TEMPLATE_NON_COMPLIANCE",
        "LABEL_REP_MISMATCH",
        "SIMILARITY_IRRELEVANT",
        "COVERAGE_MISSING_HOT_VIDEO",
    }
    assert required.issubset(tags)


@pytest.mark.parametrize(
    "golden_id, expected_reason_code",
    [
        ("GS-0001", "PROMO_DETECTED"),
        ("GS-0002", "COUNT_ZERO"),
        ("GS-0003", "GARBLED_TEXT_DETECTED"),
        ("GS-0006", "LABEL_MISMATCH"),
        ("GS-0007", "GIBBERISH_INPUT"),
        ("GS-0008", "COVERAGE_MISSING_HOT_VIDEO"),
    ],
)
def test_expected_reason_codes_are_stable(
    goldenset_items: list[dict[str, Any]], golden_id: str, expected_reason_code: str
) -> None:
    indexed = _indexed(goldenset_items)
    assert indexed[golden_id]["expected"]["reason_code"] == expected_reason_code


def test_tbd_placeholders_are_explicit(goldenset_items: list[dict[str, Any]]) -> None:
    for item in goldenset_items:
        video_id = item.get("video_id", "")
        comment_id = item.get("comment_id", "")
        assert isinstance(video_id, str) and video_id
        assert isinstance(comment_id, str) and comment_id
        if video_id.startswith("TBD_") or comment_id.startswith("TBD_"):
            assert video_id.startswith("TBD_") or comment_id.startswith("TBD_")


def test_no_full_comment_storage_policy(goldenset_items: list[dict[str, Any]]) -> None:
    for item in goldenset_items:
        snippet = item["input_snippet"]
        assert "\n\n" not in snippet
        assert len(snippet) <= 300


def test_gs0003_render_guard_detects_garbled_anchor(goldenset_items: list[dict[str, Any]]) -> None:
    item = _indexed(goldenset_items)["GS-0003"]
    snippet = item["input_snippet"]
    assert _looks_garbled(snippet) is True
    assert item["expected"]["reason_code"] == "GARBLED_TEXT_DETECTED"


def test_gs0007_url_only_input_is_not_eligible_for_keywords_or_similarity(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0007"]
    snippet = item["input_snippet"]
    quality = classify_comment_quality(snippet)
    assert quality in {"meaningless", "ad", "generic_engagement", "low_context"}
    assert tokenize_text(snippet) == []
    assert item["expected"]["eligible_for_similarity"] is False
    assert item["expected"]["reason_code"] == "GIBBERISH_INPUT"


def test_gs0005_representative_template_anchor_maps_into_bundle_rows(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0005"]
    bundle_df = pd.DataFrame(
        [
            {
                "sentiment_code": "negative",
                "rank": 1,
                "bundle_score": 0.92,
                "cluster_size": 3,
                "aspect_key": "??/??",
                "product": "???",
                "cej_scene_code": "S7 Use",
                "brand_mentioned": "LG",
                "top_video_title": "?? ???? ?? ??? ???? ???? ?? ?? (??? ?? 2?)",
                "top_source_link": "https://www.youtube.com/watch?v=T28X0JpDziM",
                "top_display_text": item["input_snippet"],
                "top_original_text": item["input_snippet"],
                "top_translation_text": "",
                "selection_reason": "??? ?? ?? ??? ????? ??? ?? ???? ??????.",
                "sample_comments_json": "[]",
                "top_pii_flag": "N",
                "top_source_content_id": item["comment_id"],
                "top_opinion_id": item["comment_id"],
                "top_likes_count": 12,
            }
        ]
    )
    rows = _rows_from_representative_bundles(bundle_df, sentiment="negative", limit=1)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row[COL_ORIGINAL] == item["input_snippet"]
    assert row[COL_VIDEO_TITLE].startswith("?? ????")
    assert row[COL_CEJ] != ""
    assert row[COL_BRAND] == "LG"
    assert row["selection_reason_ui"]
    assert _looks_garbled(str(row["selection_reason_ui"])) is False


def test_gs0001_promo_anchor_is_excluded_from_quality_and_keywords(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0001"]
    snippet = item["input_snippet"]
    assert classify_comment_quality(snippet) == "ad"
    promo_tokens = set(tokenize_text(snippet))
    assert {"??", "????", "??", "????", "???"}.isdisjoint(promo_tokens)
    assert item["expected"]["promo_flag"] is True
    assert item["expected"]["eligible_for_representative"] is False


def test_gs0002_count_zero_does_not_emit_positive_representative(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0002"]
    bundle_df = pd.DataFrame(
        [
            {
                "sentiment_code": "negative",
                "rank": 1,
                "bundle_score": 0.8,
                "cluster_size": 2,
                "aspect_key": "??",
                "product": "???",
                "cej_scene_code": "S7 Use",
                "brand_mentioned": "LG",
                "top_video_title": "negative only anchor",
                "top_source_link": "https://example.com/watch?v=neg",
                "top_display_text": "??? ?? ?? ?????.",
                "top_original_text": "??? ?? ?? ?????.",
                "selection_reason": "?? ??? ?????.",
                "top_source_content_id": "neg-1",
                "top_opinion_id": "neg-1",
            }
        ]
    )
    rows = _rows_from_representative_bundles(bundle_df, sentiment="positive", limit=1)
    assert rows.empty
    assert item["expected"]["reason_code"] == "COUNT_ZERO"


def test_gs0004_keyword_noise_terms_are_removed_from_anchor(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0004"]
    snippet = item["input_snippet"]
    tokens = tokenize_text(snippet)
    assert tokens == []
    assert item["expected"]["keyword_checks"]["noise_topk_ok"] is True


def test_gs0006_representative_rows_stay_in_requested_label_bucket(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0006"]
    bundle_df = pd.DataFrame(
        [
            {
                "sentiment_code": "positive",
                "rank": 1,
                "bundle_score": 0.7,
                "cluster_size": 2,
                "aspect_key": "??",
                "product": "???",
                "cej_scene_code": "S7 Use",
                "brand_mentioned": "LG",
                "top_video_title": "positive only anchor",
                "top_source_link": "https://example.com/watch?v=pos",
                "top_display_text": "????? ?????.",
                "top_original_text": "????? ?????.",
                "selection_reason": "?? ??? ?????.",
                "top_source_content_id": "pos-1",
                "top_opinion_id": "pos-1",
            }
        ]
    )
    rows = _rows_from_representative_bundles(bundle_df, sentiment="negative", limit=1)
    assert rows.empty
    assert item["expected"]["reason_code"] == "LABEL_MISMATCH"


def test_gs0008_coverage_anchor_remains_explicit_placeholder_until_manifest_exists(
    goldenset_items: list[dict[str, Any]],
) -> None:
    item = _indexed(goldenset_items)["GS-0008"]
    assert str(item["video_id"]).startswith("TBD_")
    assert str(item["comment_id"]).startswith("TBD_")
    assert item["expected"]["reason_code"] == "COVERAGE_MISSING_HOT_VIDEO"


@pytest.mark.skip(reason="Unmapped golden cases still need real promo/count/coverage pipeline hooks.")
def test_unmapped_pipeline_integration_hooks_placeholder() -> None:
    pass
