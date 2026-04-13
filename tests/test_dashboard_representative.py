import pandas as pd

from src.dashboard_app import _infer_inquiry_flag, _refine_aspect_label, _summarize_for_dashboard


def test_dashboard_summary_is_compressed_not_truncated() -> None:
    row = pd.Series(
        {
            "원문 댓글": "드럼세탁기가 너무 예민해서 불편하고 탈수가 잘 안돼요. 통돌이보다 옷감 손상이 커서 바꾸고 싶습니다.",
            "한국어 번역": "",
        }
    )
    summary = _summarize_for_dashboard(row, "negative")
    assert summary
    assert "요약 가능한 댓글 내용이 없습니다" not in summary
    assert len(summary) <= 170


def test_refine_aspect_label_from_general_to_specific() -> None:
    row = pd.Series({"원문 댓글": "AS 신청했는데 연락이 없고 수리 방문이 계속 지연됩니다."})
    label = _refine_aspect_label("General", row, ["수리", "서비스"], "AS 신청했는데 연락이 없어요")
    assert label == "A/S & Support"


def test_inquiry_fallback_detects_question_text() -> None:
    row = pd.Series({"nlp_is_inquiry": False})
    assert _infer_inquiry_flag(row, "이 모델은 소음이 몇 dB인가요?")
