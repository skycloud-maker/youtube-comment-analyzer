from __future__ import annotations

import pandas as pd

from src.v5.common import normalize_judgment_axes, parse_list
from src.v5.product_normalizer import normalize_products
from src.v5.representative_selector import select_representatives
from src.v5.signal_splitter import split_comment_to_signals
from src.v5.topic_ranker import rank_topics


def test_signal_splitter_contrast_and_conditional_examples() -> None:
    row_en = pd.Series(
        {
            "comment_id": "c1",
            "video_id": "v1",
            "raw_text": "It's great but why do my glasses come out cloudy?",
            "product_semantic_id": "cat:dishwasher|line:na|name:na",
            "judgment_axes_seed": ["maintenance_burden"],
            "journey_stage_seed": "usage",
            "sentiment_seed": "mixed",
            "like_count": 1,
            "reply_count": 0,
        }
    )
    row_ko = pd.Series(
        {
            "comment_id": "c2",
            "video_id": "v1",
            "raw_text": "가격은 비싸지만 성능은 좋다",
            "product_semantic_id": "cat:washer|line:na|name:na",
            "judgment_axes_seed": ["price_value_fit"],
            "journey_stage_seed": "purchase",
            "sentiment_seed": "mixed",
            "like_count": 1,
            "reply_count": 0,
        }
    )
    row_cond = pd.Series(
        {
            "comment_id": "c3",
            "video_id": "v1",
            "raw_text": "가격만 낮으면 바로 구매한다",
            "product_semantic_id": "cat:washer|line:na|name:na",
            "judgment_axes_seed": ["price_value_fit"],
            "journey_stage_seed": "purchase",
            "sentiment_seed": "neutral",
            "like_count": 0,
            "reply_count": 0,
        }
    )

    out_en = split_comment_to_signals(row_en)
    out_ko = split_comment_to_signals(row_ko)
    out_cond = split_comment_to_signals(row_cond)

    assert len(out_en) >= 2
    assert len(out_ko) >= 2
    assert len(out_cond) >= 2
    assert any(item["split_rule"] == "contrast" for item in out_en + out_ko)
    assert any(item["split_rule"] == "conditional" for item in out_cond)


def test_axis_parser_handles_malformed_strings() -> None:
    malformed = "['maintenance_burden'"
    parsed = parse_list(malformed)
    normalized = normalize_judgment_axes(parsed)
    assert normalized == ["maintenance_burden"]

    invalid = "['unknown_axis', 'bad_axis']"
    parsed_invalid = parse_list(invalid)
    normalized_invalid = normalize_judgment_axes(parsed_invalid)
    assert normalized_invalid == ["usage_pattern_fit"]


def test_normalizer_overlap_and_aliases() -> None:
    context = pd.DataFrame(
        [
            {"comment_id": "c1", "video_id": "v1", "raw_text": "I love this dishwasher", "product_raw": "", "brand_raw": ""},
            {"comment_id": "c2", "video_id": "v1", "raw_text": "My washer leaks often", "product_raw": "", "brand_raw": ""},
            {"comment_id": "c3", "video_id": "v1", "raw_text": "틔운이 예쁘지만 관리가 어렵다", "product_raw": "", "brand_raw": "LG"},
            {"comment_id": "c4", "video_id": "v1", "raw_text": "I bought a hair dryer", "product_raw": "", "brand_raw": ""},
        ]
    )
    out = normalize_products(context)
    mapped = dict(zip(out["comment_id"], out["product_category"]))
    named = dict(zip(out["comment_id"], out["named_product"]))
    assert mapped["c1"] == "dishwasher"
    assert mapped["c2"] == "washer"
    assert mapped["c4"] != "dryer"
    assert named["c3"] == "tiiun"


def test_ranker_penalizes_low_coverage_topics() -> None:
    topics = pd.DataFrame(
        [
            {"topic_id": "t_big", "signal_count": 40, "comment_count": 20},
            {"topic_id": "t_small", "signal_count": 2, "comment_count": 1},
        ]
    )
    signals = pd.DataFrame(
        [{"topic_id": "t_big", "comment_id": f"c{i}", "polarity": "negative", "like_count": 0, "reply_count": 0} for i in range(20)]
        + [{"topic_id": "t_big", "comment_id": "c0", "polarity": "negative", "like_count": 0, "reply_count": 0} for _ in range(20)]
        + [{"topic_id": "t_small", "comment_id": "s1", "polarity": "negative", "like_count": 0, "reply_count": 0} for _ in range(2)]
    )
    ranked = rank_topics(topics, signals).set_index("topic_id")
    assert bool(ranked.loc["t_small", "insufficient_evidence"]) is True
    assert ranked.loc["t_big", "rank_score"] > ranked.loc["t_small", "rank_score"]


def test_representative_alignment_rejects_generic_comment() -> None:
    topics = pd.DataFrame(
        [
            {
                "topic_id": "t1",
                "journey_stage": "maintenance",
                "judgment_axis": "service_after_sales",
                "polarity_group": "negative",
                "product_semantic_id": "cat:dishwasher|line:na|name:na",
                "v5_simple_signature": "maintenance_service_after_sales_repair_support",
            }
        ]
    )
    signals = pd.DataFrame(
        [
            {
                "topic_id": "t1",
                "signal_id": "s1",
                "comment_id": "generic",
                "signal_type": "primary",
                "weight": 1.0,
                "signal_text": "Great video thanks!",
                "judgment_axis": "service_after_sales",
                "polarity": "negative",
                "product_semantic_id": "cat:dishwasher|line:na|name:na",
            },
            {
                "topic_id": "t1",
                "signal_id": "s2",
                "comment_id": "aligned",
                "signal_type": "primary",
                "weight": 1.0,
                "signal_text": "A/S response is too slow and support keeps delaying repairs.",
                "judgment_axis": "service_after_sales",
                "polarity": "negative",
                "product_semantic_id": "cat:dishwasher|line:na|name:na",
            },
        ]
    )

    selected = select_representatives(topics, signals)
    rep_ids = selected.loc[0, "representative_comment_ids"]
    assert rep_ids
    assert rep_ids[0] == "aligned"
