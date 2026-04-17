import pandas as pd

from src.dashboard_app import COL_CEJ, COL_ORIGINAL, _build_resolution_point


def _build_row(
    text: str,
    *,
    stage: str = "사용",
    mixed: bool = False,
    product: str = "세탁기",
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
    return pd.DataFrame({"원문": list(texts)})


def test_resolution_point_output_structure_and_evidence_grounding() -> None:
    row = _build_row(
        "냄새가 심해서 부엌에서 못 쓰고 다용도실로 옮겼어요.",
        stage="사용",
        core_points=["냄새", "다용도실"],
    )
    similar = _build_similar(
        "필터를 바꿔도 냄새가 계속 납니다.",
        "냄새 때문에 밤에는 절대 못 돌려요.",
        "자동 처리 기대했는데 냄새만 더 심해졌어요.",
    )

    payload = _build_resolution_point(
        working_selected=row,
        sentiment_name="negative",
        similar_comments=similar,
    )
    assert isinstance(payload, dict)
    assert {"action_text", "action_domain", "evidence_trace", "insufficient_evidence"} <= set(payload.keys())
    assert payload["insufficient_evidence"] is False
    assert "냄새" in payload["action_text"]
    assert payload["action_domain"] in {
        "product",
        "UX / guide",
        "operations / service",
        "marketing / message",
        "business / policy",
    }


def test_resolution_points_change_when_problem_structure_differs() -> None:
    odor_row = _build_row(
        "냄새가 계속 올라와서 주방에서 사용할 수가 없습니다.",
        stage="사용",
        core_points=["냄새", "주방"],
    )
    service_row = _build_row(
        "A/S 접수 후 기사 방문이 계속 지연되어 해결이 안 됩니다.",
        stage="관리",
        core_points=["A/S", "지연"],
    )
    similar_odor = _build_similar(
        "냄새 때문에 환기를 계속 해야 합니다.",
        "활성탄 필터를 바꿔도 냄새가 안 줄어요.",
        "주방에서는 냄새 때문에 못 씁니다.",
    )
    similar_service = _build_similar(
        "수리 접수했는데 일정이 계속 밀립니다.",
        "센터 연결이 오래 걸리고 답이 없습니다.",
        "기사 방문이 연기되어 고장이 계속됩니다.",
    )

    odor_payload = _build_resolution_point(odor_row, "negative", similar_comments=similar_odor)
    service_payload = _build_resolution_point(service_row, "negative", similar_comments=similar_service)
    assert isinstance(odor_payload, dict) and isinstance(service_payload, dict)
    assert odor_payload["action_text"] != service_payload["action_text"]


def test_journey_stage_changes_action_domain() -> None:
    base_text = "약정 해지 비용이 불명확해서 구매 결정을 미루게 됩니다."
    row_purchase = _build_row(base_text, stage="구매", core_points=["약정", "해지"])
    row_usage = _build_row(base_text, stage="사용", core_points=["약정", "해지"])
    similar = _build_similar(
        "약정 조건이 복잡해서 계약을 망설입니다.",
        "해지 위약금이 unclear 해서 구매가 꺼려집니다.",
        "총비용이 안 보여서 결정을 못 하겠어요.",
    )

    purchase_payload = _build_resolution_point(row_purchase, "negative", similar_comments=similar)
    usage_payload = _build_resolution_point(row_usage, "negative", similar_comments=similar)
    assert isinstance(purchase_payload, dict) and isinstance(usage_payload, dict)
    assert purchase_payload["action_text"] != usage_payload["action_text"]
    assert "구매 단계" in purchase_payload["action_text"]
    assert "사용 단계" in usage_payload["action_text"]


def test_mixed_comment_not_flattened_to_single_direction() -> None:
    row = _build_row(
        "세척력은 만족하지만 소음이 커서 밤에는 못 씁니다.",
        stage="사용",
        mixed=True,
        core_points=["세척력", "소음"],
        context_tags=["비교 맥락"],
    )
    similar = _build_similar(
        "성능은 좋지만 소음 때문에 시간대를 가려서 씁니다.",
        "세척은 만족인데 야간 소음은 불편합니다.",
        "결과는 좋지만 소음 때문에 사용 패턴이 제한됩니다.",
    )

    payload = _build_resolution_point(row, "mixed", similar_comments=similar)
    assert isinstance(payload, dict)
    assert payload["evidence_trace"]["polarity"] == "mixed"
    assert payload["evidence_trace"]["mixed_flag"] is True
    assert any(token in payload["action_text"] for token in ["공존", "동시에", "이중 실행"])


def test_missing_journey_stage_triggers_insufficient_evidence() -> None:
    row = _build_row(
        "정보가 부족해서 어떤 문제가 핵심인지 판단이 어렵습니다.",
        stage="",
        core_points=[],
        context_tags=[],
    )
    similar = _build_similar(
        "문맥이 짧아서 원인 파악이 어렵네요.",
        "정보가 제한적이라 동일 패턴을 확정하기 어렵습니다.",
        "추가 설명 없이는 판단이 어렵습니다.",
    )

    payload = _build_resolution_point(row, "negative", similar_comments=similar)
    assert isinstance(payload, dict)
    assert payload["insufficient_evidence"] is True
    assert "근거 부족" in payload["action_text"]
