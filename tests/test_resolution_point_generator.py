import pandas as pd

from src.dashboard_app import COL_CEJ, COL_ORIGINAL, _build_resolution_point


def _build_row(
    text: str,
    *,
    stage: str = "\uC0AC\uC6A9",
    mixed: bool = False,
    product: str = "\uC138\uD0C1\uAE30",
    core_points: list[str] | None = None,
    context_tags: list[str] | None = None,
) -> pd.Series:
    return pd.Series(
        {
            "comment_id": "c-1",
            "video_id": "v-1",
            COL_ORIGINAL: text,
            COL_CEJ: stage,
            "mixed_flag": mixed,
            "product": product,
            "nlp_core_points": core_points or [],
            "nlp_context_tags": context_tags or [],
            "sentiment_final": "mixed" if mixed else "negative",
            "comment_validity": "valid",
            "analysis_included": True,
            "hygiene_class": "strategic",
        }
    )


def _build_similar(*texts: str) -> pd.DataFrame:
    return pd.DataFrame({"\uC6D0\uBB38": list(texts)})


def test_resolution_point_output_structure_and_evidence_grounding() -> None:
    row = _build_row(
        "\uD544\uD130\uB97C \uAD50\uCCB4\uD574\uB3C4 \uB0C4\uC0C8\uAC00 \uC9C0\uC18D\uB418\uACE0 \uBD80\uC5CC\uC5D0\uC11C \uC0AC\uC6A9\uD558\uAE30 \uD798\uB4ED\uB2C8\uB2E4.",
        stage="\uC0AC\uC6A9",
        core_points=["\uB0C4\uC0C8", "\uD544\uD130"],
    )
    similar = _build_similar(
        "\uD544\uD130 \uAD50\uCCB4 \uD6C4\uC5D0\uB3C4 \uC545\uCDE8\uAC00 \uACC4\uC18D \uB0A9\uB2C8\uB2E4.",
        "\uC6C0\uC9C1\uC77C \uB54C \uB0C4\uC0C8 \uB54C\uBB38\uC5D0 \uB2E4\uC6A9\uB3C4\uC2E4\uB85C \uC606\uACA8\uC2B5\uB2C8\uB2E4.",
        "\uC790\uB3D9 \uCC98\uB9AC \uAE30\uB300\uC600\uB294\uB370 \uB0C4\uC0C8 \uAD00\uB9AC\uB97C \uACC4\uC18D \uD574\uC57C \uD569\uB2C8\uB2E4.",
    )

    payload = _build_resolution_point(row, "negative", similar_comments=similar)
    assert isinstance(payload, dict)
    assert {"action_text", "action_domain", "evidence_trace", "insufficient_evidence"} <= set(payload.keys())
    assert payload["insufficient_evidence"] is False
    assert payload["action_domain"] in {
        "product",
        "UX / guide",
        "operations / service",
        "marketing / message",
        "business / policy",
    }
    assert payload["evidence_trace"]["similar_count"] >= 3


def test_resolution_points_change_when_problem_structure_differs() -> None:
    odor_row = _build_row(
        "\uC8FC\uBC29\uC5D0\uC11C \uB0C4\uC0C8\uAC00 \uC2EC\uD574 \uC0AC\uC6A9\uD558\uAE30 \uC5B4\uB835\uC2B5\uB2C8\uB2E4.",
        stage="\uC0AC\uC6A9",
        core_points=["\uB0C4\uC0C8", "\uC8FC\uBC29"],
    )
    service_row = _build_row(
        "A/S \uC811\uC218 \uD6C4 \uAE30\uC0AC \uBC29\uBB38\uC774 \uC5F0\uAE30\uB418\uC5B4 \uD574\uACB0\uC774 \uB2A6\uC5B4\uC9D1\uB2C8\uB2E4.",
        stage="\uAD00\uB9AC",
        core_points=["A/S", "\uC9C0\uC5F0"],
    )
    similar_odor = _build_similar(
        "\uB0C4\uC0C8 \uB54C\uBB38\uC5D0 \uC2E4\uB0B4 \uC0AC\uC6A9\uC744 \uD3EC\uAE30\uD588\uC2B5\uB2C8\uB2E4.",
        "\uD544\uD130\uB97C \uBC14\uAFD4\uB3C4 \uC545\uCDE8\uAC00 \uC90C \uC904\uC9C0 \uC54A\uC2B5\uB2C8\uB2E4.",
        "\uB0C4\uC0C8 \uBB38\uC81C\uAC00 \uC7AC\uBC1C\uD569\uB2C8\uB2E4.",
    )
    similar_service = _build_similar(
        "\uC11C\uBE44\uC2A4 \uC77C\uC815\uC774 \uC790\uAFB8 \uBC00\uB824\uC11C \uACE4\uB780\uD588\uC2B5\uB2C8\uB2E4.",
        "\uC218\uB9AC \uC9C4\uD589 \uACBD\uACFC \uC548\uB0B4\uAC00 \uC5C6\uC5C8\uC2B5\uB2C8\uB2E4.",
        "\uAE30\uC0AC \uBC29\uBB38 \uC5F0\uAE30\uB85C \uC2E4\uC0AC\uC6A9 \uD53C\uD574\uAC00 \uCEE4\uC84C\uC2B5\uB2C8\uB2E4.",
    )

    odor_payload = _build_resolution_point(odor_row, "negative", similar_comments=similar_odor)
    service_payload = _build_resolution_point(service_row, "negative", similar_comments=similar_service)
    assert isinstance(odor_payload, dict) and isinstance(service_payload, dict)
    assert odor_payload["action_text"] != service_payload["action_text"]
    assert odor_payload["action_domain"] != service_payload["action_domain"]


def test_journey_stage_changes_action_domain() -> None:
    base_text = "\uC57D\uC815 \uBE44\uC6A9 \uC815\uBCF4\uAC00 \uBD88\uD22C\uBA85\uD574 \uAD6C\uB9E4 \uACB0\uC815\uC744 \uBBF8\uB8E8\uAC8C \uB429\uB2C8\uB2E4."
    row_purchase = _build_row(base_text, stage="\uAD6C\uB9E4", core_points=["\uC57D\uC815", "\uBE44\uC6A9"])
    row_usage = _build_row(base_text, stage="\uC0AC\uC6A9", core_points=["\uC57D\uC815", "\uBE44\uC6A9"])
    similar = _build_similar(
        "\uC57D\uC815 \uC870\uAC74\uC774 \uBCF5\uC7A1\uD574 \uACB0\uC81C\uAC00 \uB9DD\uC124\uC5EC\uC9D1\uB2C8\uB2E4.",
        "\uD574\uC9C0 \uC704\uC57D\uAE08 \uC815\uBCF4\uAC00 \uBAA8\uD638\uD574 \uD310\uB2E8\uC774 \uC5B4\uB835\uC2B5\uB2C8\uB2E4.",
        "\uCD1D \uBE44\uC6A9\uC774 \uBD88\uBA85\uD655\uD574 \uAD6C\uB9E4 \uD3EC\uAE30 \uACE0\uBBFC\uC744 \uD569\uB2C8\uB2E4.",
    )

    purchase_payload = _build_resolution_point(row_purchase, "negative", similar_comments=similar)
    usage_payload = _build_resolution_point(row_usage, "negative", similar_comments=similar)
    assert isinstance(purchase_payload, dict) and isinstance(usage_payload, dict)
    assert purchase_payload["action_text"] != usage_payload["action_text"]


def test_mixed_comment_not_flattened_to_single_direction() -> None:
    row = _build_row(
        "\uAE30\uBCF8 \uC131\uB2A5\uC740 \uB9CC\uC871\uD558\uC9C0\uB9CC \uC18C\uC74C\uC774 \uCEE4\uC11C \uBC24\uC5D0 \uC0AC\uC6A9\uD558\uAE30 \uD798\uB4ED\uB2C8\uB2E4.",
        stage="\uC0AC\uC6A9",
        mixed=True,
        core_points=["\uC131\uB2A5", "\uC18C\uC74C"],
        context_tags=["\uBE44\uAD50 \uB9E5\uB77D"],
    )
    similar = _build_similar(
        "\uC131\uB2A5\uC740 \uC88B\uC740\uB370 \uC18C\uC74C \uBB38\uC81C\uAC00 \uB0A8\uC544 \uC788\uC2B5\uB2C8\uB2E4.",
        "\uACB0\uACFC\uB294 \uB9CC\uC871\uC2A4\uB7FD\uC9C0\uB9CC \uC57C\uAC04 \uC0AC\uC6A9\uC740 \uBD80\uB2F4\uC785\uB2C8\uB2E4.",
        "\uAE30\uB2A5 \uAC15\uC810\uACFC \uC0AC\uC6A9 \uBD88\uD3B8\uC774 \uD568\uAED8 \uBCF4\uC785\uB2C8\uB2E4.",
    )

    payload = _build_resolution_point(row, "mixed", similar_comments=similar)
    assert isinstance(payload, dict)
    assert payload["evidence_trace"]["polarity"] == "mixed"
    assert payload["evidence_trace"]["mixed_flag"] is True
    assert payload["insufficient_evidence"] is False


def test_missing_journey_stage_triggers_insufficient_evidence() -> None:
    row = _build_row(
        "\uC815\uBCF4\uAC00 \uBD80\uC871\uD574 \uC5B4\uB5A4 \uBB38\uC81C\uAC00 \uD575\uC2EC\uC778\uC9C0 \uD310\uB2E8\uD558\uAE30 \uC5B4\uB835\uC2B5\uB2C8\uB2E4.",
        stage="",
        core_points=[],
        context_tags=[],
    )
    similar = _build_similar(
        "\uC124\uBA85\uC774 \uC9E7\uC544\uC11C \uC6D0\uC778 \uD30C\uC545\uC774 \uC5B4\uB835\uC2B5\uB2C8\uB2E4.",
        "\uAD00\uCE21 \uADFC\uAC70\uAC00 \uC81C\uD55C\uC801\uC774\uC5B4\uC11C \uC2E4\uD589 \uC6B0\uC120\uC21C\uC704 \uD655\uC815\uC774 \uD798\uB4ED\uB2C8\uB2E4.",
        "\uCD94\uAC00 \uB9E5\uB77D \uC5C6\uC774\uB294 \uD655\uC815\uC801 \uC81C\uC548\uC744 \uB0B4\uAE30 \uC5B4\uB835\uC2B5\uB2C8\uB2E4.",
    )

    payload = _build_resolution_point(row, "negative", similar_comments=similar)
    assert isinstance(payload, dict)
    assert payload["insufficient_evidence"] is True


def test_resolution_point_accepts_new_fixed_journey_labels() -> None:
    row = _build_row(
        "\uC77C\uC0C1 \uC0AC\uC6A9\uC5D0\uC11C \uC790\uB3D9 \uCC98\uB9AC \uACB0\uACFC\uAC00 \uC77C\uAD00\uB418\uC9C0 \uC54A\uC544 \uBD88\uD3B8\uD569\uB2C8\uB2E4.",
        stage="usage",
        core_points=["\uC790\uB3D9", "\uC77C\uC0C1"],
    )
    similar = _build_similar(
        "\uC790\uB3D9 \uBAA8\uB4DC \uC77C\uAD00\uC131\uC774 \uB5A8\uC5B4\uC838 \uC7AC\uC2E4\uD589\uC774 \uD544\uC694\uD569\uB2C8\uB2E4.",
        "\uC0AC\uC6A9 \uB3C4\uC911 \uB9E5\uB77D\uC5D0 \uB530\uB77C \uACB0\uACFC \uD3B8\uCC28\uAC00 \uD07D\uB2C8\uB2E4.",
        "\uC77C\uC0C1 \uC0AC\uC6A9\uC790 \uD6C4\uAE30\uC5D0\uC11C \uAC19\uC740 \uC720\uD615\uC774 \uBC18\uBCF5\uB429\uB2C8\uB2E4.",
    )

    payload = _build_resolution_point(row, "negative", similar_comments=similar)
    assert isinstance(payload, dict)
    assert payload["insufficient_evidence"] is False
    assert payload["evidence_trace"]["journey_stage"] == "\uC0AC\uC6A9"
