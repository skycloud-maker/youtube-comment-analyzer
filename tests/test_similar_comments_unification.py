import pandas as pd

from src.dashboard_app import (
    COL_CEJ,
    COL_ORIGINAL,
    _build_canonical_similar_comments,
)


def _row(
    comment_id: str,
    text: str,
    *,
    sentiment: str = "negative",
    journey: str = "사용",
    core_points: list[str] | None = None,
    sim_keys: list[str] | None = None,
    classification_type: str = "complaint",
    hygiene_class: str = "strategic",
    analysis_included: bool = True,
    mixed_flag: bool = False,
    validity: str = "valid",
) -> dict:
    return {
        "comment_id": comment_id,
        COL_ORIGINAL: text,
        "sentiment_final": sentiment,
        COL_CEJ: journey,
        "nlp_core_points": core_points or [],
        "nlp_similarity_keys": sim_keys or [],
        "classification_type": classification_type,
        "hygiene_class": hygiene_class,
        "analysis_included": analysis_included,
        "mixed_flag": mixed_flag,
        "comment_validity": validity,
        "like_count": 1,
    }


def test_similar_comments_use_single_canonical_filters() -> None:
    representative = pd.Series(
        _row(
            "rep-1",
            "Filter replacement did not remove the odor.",
            core_points=["odor", "filter"],
            sim_keys=["odor", "filter_change"],
        )
    )
    pool = pd.DataFrame(
        [
            representative.to_dict(),
            _row(
                "s-1",
                "Odor remains after filter replacement.",
                core_points=["odor", "filter"],
                sim_keys=["odor", "filter_change"],
            ),
            _row(
                "s-2",
                "Filter was changed but smell still leaks.",
                core_points=["odor", "leak"],
                sim_keys=["odor", "filter_change"],
            ),
            _row(
                "s-3",
                "Strong odor persists in the same usage stage.",
                core_points=["odor", "usage"],
                sim_keys=["odor", "filter_change"],
            ),
            _row(
                "s-trash",
                "Affiliate link here",
                core_points=["odor"],
                sim_keys=["odor"],
                hygiene_class="trash",
            ),
            _row(
                "s-pos",
                "No issues at all after replacement.",
                sentiment="positive",
                core_points=["odor"],
                sim_keys=["odor"],
                classification_type="praise",
            ),
        ]
    )

    similar, meta = _build_canonical_similar_comments(pool, representative, representative, "negative", "rep-1")

    assert meta["insufficient_evidence"] is False
    assert meta["cluster_size"] == 3
    assert len(similar) == 3
    ids = set(similar["comment_id"].tolist())
    assert "rep-1" not in ids
    assert "s-trash" not in ids
    assert "s-pos" not in ids


def test_similar_comments_mark_insufficient_when_cluster_lt_three() -> None:
    representative = pd.Series(
        _row(
            "rep-2",
            "The machine shakes heavily.",
            core_points=["shake", "vibration"],
            sim_keys=["vibration"],
        )
    )
    pool = pd.DataFrame(
        [
            representative.to_dict(),
            _row(
                "s-4",
                "Vibration is severe at spin mode.",
                core_points=["vibration"],
                sim_keys=["vibration"],
            ),
            _row(
                "s-5",
                "Shaking remains after leveling.",
                core_points=["shake"],
                sim_keys=["vibration"],
            ),
        ]
    )

    similar, meta = _build_canonical_similar_comments(pool, representative, representative, "negative", "rep-2")

    assert similar.empty
    assert meta["insufficient_evidence"] is True
    assert meta["cluster_size"] == 0
    assert meta["candidate_count"] == 2


def test_mixed_representative_does_not_collapse_to_pure_direction() -> None:
    representative = pd.Series(
        _row(
            "rep-3",
            "Cleaning quality is good but odor issue remains.",
            sentiment="mixed",
            mixed_flag=True,
            core_points=["cleaning", "odor"],
            sim_keys=["odor", "tradeoff"],
            classification_type="mixed",
        )
    )
    pool = pd.DataFrame(
        [
            representative.to_dict(),
            _row(
                "s-mixed",
                "Overall useful, but smell still appears.",
                sentiment="mixed",
                mixed_flag=True,
                core_points=["odor", "tradeoff"],
                sim_keys=["odor", "tradeoff"],
                classification_type="mixed",
            ),
            _row(
                "s-neg",
                "Only odor complaint.",
                sentiment="negative",
                mixed_flag=False,
                core_points=["odor"],
                sim_keys=["odor"],
                classification_type="complaint",
            ),
            _row(
                "s-pos",
                "No odor and perfect quality.",
                sentiment="positive",
                mixed_flag=False,
                core_points=["quality"],
                sim_keys=["quality"],
                classification_type="praise",
            ),
        ]
    )

    similar, meta = _build_canonical_similar_comments(pool, representative, representative, "mixed", "rep-3")

    assert similar.empty
    assert meta["insufficient_evidence"] is True
    assert meta["cluster_size"] == 0
    assert meta["candidate_count"] == 1


def test_similar_comments_fallback_to_sentiment_label_when_sentiment_final_missing() -> None:
    representative = pd.Series(
        _row(
            "rep-4",
            "Installation delay and no callback from service.",
            sentiment="negative",
            core_points=["installation", "delay"],
            sim_keys=["installation_delay"],
        )
    )
    pool = pd.DataFrame(
        [
            representative.to_dict(),
            _row(
                "s-6",
                "Technician did not call back after installation request.",
                sentiment="negative",
                core_points=["installation", "callback"],
                sim_keys=["installation_delay"],
            ),
            _row(
                "s-7",
                "Delivery and install were postponed twice.",
                sentiment="negative",
                core_points=["installation", "postponed"],
                sim_keys=["installation_delay"],
            ),
            _row(
                "s-8",
                "No installation update for several days.",
                sentiment="negative",
                core_points=["installation", "delay"],
                sim_keys=["installation_delay"],
            ),
        ]
    )
    pool = pool.drop(columns=["sentiment_final"])
    representative = representative.drop(labels=["sentiment_final"])
    pool["sentiment_label"] = "negative"
    representative["sentiment_label"] = "negative"

    similar, meta = _build_canonical_similar_comments(pool, representative, representative, "negative", "rep-4")

    assert meta["insufficient_evidence"] is False
    assert meta["cluster_size"] == 3
    assert len(similar) == 3
