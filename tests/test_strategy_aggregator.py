import json

import pandas as pd

from src.analytics.strategy_aggregator import (
    REQUIRED_STRATEGY_COLUMNS,
    build_strategy_insight_aggregates,
)


def _analysis_row(
    comment_id: str,
    *,
    video_id: str,
    product: str,
    text: str,
    journey_stage: str,
    judgment_axes: list[str],
    sentiment_final: str,
    mixed_flag: bool = False,
    like_count: int = 0,
    reply_count: int = 0,
) -> dict:
    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "product": product,
        "cleaned_text": text,
        "text_display": text,
        "journey_stage": journey_stage,
        "judgment_axes": judgment_axes,
        "sentiment_final": sentiment_final,
        "mixed_flag": mixed_flag,
        "analysis_included": True,
        "hygiene_class": "strategic",
        "like_count": like_count,
        "reply_count": reply_count,
        "classification_type": "complaint" if sentiment_final == "negative" else "praise",
        "nlp_similarity_keys": judgment_axes,
        "nlp_core_points": judgment_axes,
    }


def _rep_row(
    comment_id: str,
    *,
    video_id: str,
    product: str,
    journey_stage: str,
    judgment_axes: list[str],
    sentiment_final: str,
) -> dict:
    return {
        "comment_id": comment_id,
        "video_id": video_id,
        "product": product,
        "journey_stage": journey_stage,
        "judgment_axes": judgment_axes,
        "sentiment_final": sentiment_final,
    }


def test_strategy_aggregates_produced_for_video_and_product_group() -> None:
    analysis_non_trash = pd.DataFrame(
        [
            _analysis_row(
                "c1",
                video_id="v1",
                product="세탁기",
                text="설치 공간이 좁아 문이 안 열립니다",
                journey_stage="delivery",
                judgment_axes=["installation_space_fit"],
                sentiment_final="negative",
                like_count=11,
                reply_count=2,
            ),
            _analysis_row(
                "c2",
                video_id="v1",
                product="세탁기",
                text="설치 기사 방문 후 공간 제약 설명이 없었습니다",
                journey_stage="delivery",
                judgment_axes=["installation_space_fit"],
                sentiment_final="negative",
                like_count=6,
                reply_count=1,
            ),
            _analysis_row(
                "c3",
                video_id="v1",
                product="세탁기",
                text="설치 전에 공간 체크가 안 되어 재방문했습니다",
                journey_stage="delivery",
                judgment_axes=["installation_space_fit"],
                sentiment_final="negative",
                like_count=4,
                reply_count=1,
            ),
        ]
    )
    representatives = pd.DataFrame(
        [
            _rep_row(
                "c1",
                video_id="v1",
                product="세탁기",
                journey_stage="delivery",
                judgment_axes=["installation_space_fit"],
                sentiment_final="negative",
            )
        ]
    )

    outputs = build_strategy_insight_aggregates(analysis_non_trash, representatives)
    video_insights = outputs["strategy_video_insights"]
    product_insights = outputs["strategy_product_group_insights"]

    assert not video_insights.empty
    assert not product_insights.empty
    assert set(REQUIRED_STRATEGY_COLUMNS).issubset(video_insights.columns)
    assert set(REQUIRED_STRATEGY_COLUMNS).issubset(product_insights.columns)
    assert (video_insights["level_type"] == "video").all()
    assert (product_insights["level_type"] == "product_group").all()


def test_single_comment_alone_does_not_create_strategy_insight() -> None:
    analysis_non_trash = pd.DataFrame(
        [
            _analysis_row(
                "solo",
                video_id="v-solo",
                product="냉장고",
                text="가격이 너무 비쌉니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            )
        ]
    )
    representatives = pd.DataFrame(
        [
            _rep_row(
                "solo",
                video_id="v-solo",
                product="냉장고",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            )
        ]
    )

    outputs = build_strategy_insight_aggregates(analysis_non_trash, representatives)
    assert outputs["strategy_video_insights"].empty

    candidates = outputs["strategy_video_insight_candidates"]
    assert not candidates.empty
    assert bool(candidates.iloc[0]["insufficient_evidence"]) is True
    reasons = json.loads(candidates.iloc[0]["insufficient_reasons"])
    assert "cluster_lt_3" in reasons


def test_same_product_group_two_videos_produce_distinct_journey_insights() -> None:
    analysis_non_trash = pd.DataFrame(
        [
            _analysis_row(
                "v1-a",
                video_id="v1",
                product="세탁기",
                text="가격이 부담돼서 구매를 망설입니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "v1-b",
                video_id="v1",
                product="세탁기",
                text="가격 인하가 없어 구매가 지연됩니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "v1-c",
                video_id="v1",
                product="세탁기",
                text="가격 대비 성능이 애매합니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "v2-a",
                video_id="v2",
                product="세탁기",
                text="밤에 소음이 커서 사용이 어렵습니다",
                journey_stage="usage",
                judgment_axes=["maintenance_burden"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "v2-b",
                video_id="v2",
                product="세탁기",
                text="필터 청소 주기가 짧아 관리 부담이 큽니다",
                journey_stage="usage",
                judgment_axes=["maintenance_burden"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "v2-c",
                video_id="v2",
                product="세탁기",
                text="필터 청소가 번거로워서 매일 확인합니다",
                journey_stage="usage",
                judgment_axes=["maintenance_burden"],
                sentiment_final="negative",
            ),
        ]
    )
    representatives = pd.DataFrame(
        [
            _rep_row(
                "v1-a",
                video_id="v1",
                product="세탁기",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _rep_row(
                "v2-a",
                video_id="v2",
                product="세탁기",
                journey_stage="usage",
                judgment_axes=["maintenance_burden"],
                sentiment_final="negative",
            ),
        ]
    )

    outputs = build_strategy_insight_aggregates(analysis_non_trash, representatives)
    video_insights = outputs["strategy_video_insights"]
    assert not video_insights.empty

    v1_stages = set(video_insights.loc[video_insights["level_id"] == "v1", "journey_stage"].tolist())
    v2_stages = set(video_insights.loc[video_insights["level_id"] == "v2", "journey_stage"].tolist())
    assert "purchase" in v1_stages
    assert "usage" in v2_stages


def test_same_topic_but_different_axes_produce_distinct_frames() -> None:
    analysis_non_trash = pd.DataFrame(
        [
            _analysis_row(
                "ax-1",
                video_id="v-ax",
                product="건조기",
                text="가격이 비싸서 구매가 망설여집니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "ax-2",
                video_id="v-ax",
                product="건조기",
                text="약정 조건이 길어 계약이 부담됩니다",
                journey_stage="purchase",
                judgment_axes=["purchase_strategy_contract"],
                sentiment_final="negative",
            ),
            _analysis_row(
                "ax-3",
                video_id="v-ax",
                product="건조기",
                text="가격과 계약 조건이 동시에 부담됩니다",
                journey_stage="purchase",
                judgment_axes=["price_value_fit", "purchase_strategy_contract"],
                sentiment_final="negative",
            ),
        ]
    )
    representatives = pd.DataFrame(
        [
            _rep_row(
                "ax-1",
                video_id="v-ax",
                product="건조기",
                journey_stage="purchase",
                judgment_axes=["price_value_fit"],
                sentiment_final="negative",
            ),
            _rep_row(
                "ax-2",
                video_id="v-ax",
                product="건조기",
                journey_stage="purchase",
                judgment_axes=["purchase_strategy_contract"],
                sentiment_final="negative",
            ),
        ]
    )
    outputs = build_strategy_insight_aggregates(analysis_non_trash, representatives)
    candidates = outputs["strategy_video_insight_candidates"]
    axes_seen = set()
    for raw in candidates["judgment_axes"].tolist():
        parsed = json.loads(raw)
        if parsed:
            axes_seen.add(parsed[0])

    assert "price_value_fit" in axes_seen
    assert "purchase_strategy_contract" in axes_seen


def test_insufficient_evidence_when_representative_link_missing() -> None:
    analysis_non_trash = pd.DataFrame(
        [
            _analysis_row(
                "m1",
                video_id="v-m",
                product="식기세척기",
                text="세척은 되지만 건조가 아쉽습니다",
                journey_stage="usage",
                judgment_axes=["automation_convenience"],
                sentiment_final="mixed",
                mixed_flag=True,
            ),
            _analysis_row(
                "m2",
                video_id="v-m",
                product="식기세척기",
                text="건조 품질은 좋아졌지만 소음이 큽니다",
                journey_stage="usage",
                judgment_axes=["automation_convenience"],
                sentiment_final="mixed",
                mixed_flag=True,
            ),
            _analysis_row(
                "m3",
                video_id="v-m",
                product="식기세척기",
                text="자동화는 편한데 세척 후 잔여물이 남습니다",
                journey_stage="usage",
                judgment_axes=["automation_convenience"],
                sentiment_final="mixed",
                mixed_flag=True,
            ),
        ]
    )
    # Representative polarity intentionally mismatched to force link-missing condition.
    representatives = pd.DataFrame(
        [
            _rep_row(
                "m1",
                video_id="v-m",
                product="식기세척기",
                journey_stage="usage",
                judgment_axes=["automation_convenience"],
                sentiment_final="positive",
            )
        ]
    )

    outputs = build_strategy_insight_aggregates(analysis_non_trash, representatives)
    assert outputs["strategy_video_insights"].empty
    candidates = outputs["strategy_video_insight_candidates"]
    assert not candidates.empty
    reasons = json.loads(candidates.iloc[0]["insufficient_reasons"])
    assert "representative_link_missing" in reasons
