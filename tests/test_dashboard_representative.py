import pandas as pd

from src.dashboard_app import (
    _build_resolution_point,
    _build_video_context_insight,
    _extract_card_keywords,
    _infer_inquiry_flag,
    _localize_aspect_label,
    _refine_aspect_label,
    _summarize_for_dashboard,
)


def test_dashboard_summary_is_semantic_not_truncated() -> None:
    row = pd.Series(
        {
            "원문 댓글": "구독 가입 과정에서 매니저가 계속 권유해서 불쾌했고, 수리 문의도 연결이 잘 안됐습니다.",
            "한국어 번역": "",
            "nlp_is_inquiry": False,
            "nlp_keywords": ["구독", "권유", "수리"],
        }
    )
    summary = _summarize_for_dashboard(row, "negative")
    assert summary
    assert "..." not in summary
    assert "불편·불만" in summary
    assert "원문 댓글" not in summary


def test_refine_aspect_label_from_general_to_specific() -> None:
    row = pd.Series({"원문 댓글": "AS 요청했는데 연락이 늦고 수리 방문 일정이 계속 밀렸습니다."})
    label = _refine_aspect_label("General", row, ["수리", "서비스"], "AS 요청했는데 연락이 늦었습니다.")
    assert label == "A/S & Support"


def test_aspect_localization_for_card_display() -> None:
    assert _localize_aspect_label("Performance") == "성능"
    assert _localize_aspect_label("A/S & Support") == "A/S·고객지원"
    assert _localize_aspect_label("General") == "제품 경험"


def test_inquiry_fallback_detects_question_text() -> None:
    row = pd.Series({"nlp_is_inquiry": False})
    assert _infer_inquiry_flag(row, "이 모델은 소음이 몇 dB인가요")


def test_extract_card_keywords_filters_noise_tokens() -> None:
    row = pd.Series(
        {
            "nlp_keywords": ["이게", "문제나", "한달", "통돌이는", "티비", "수리"],
            "원문 댓글": "통돌이는 수리 문의가 많고 티비 결합 판매가 불편했어요",
        }
    )
    keywords = _extract_card_keywords(row, "negative", limit=5)
    assert "이게" not in keywords
    assert "문제나" not in keywords
    assert "한달" not in keywords
    assert "통돌이" in keywords
    assert "TV" in keywords


def test_context_insight_and_action_are_specific_not_generic() -> None:
    row = pd.Series(
        {
            "comment_id": "c-1",
            "video_id": "v-1",
            "원문 댓글": "구독 가입할 때 매니저가 할부랑 위약금 설명을 제대로 안 해서 불쾌했어요.",
            "고객경험여정": "구매",
            "영상 제목": "가전 구독의 함정",
            "channel_title": "테스트채널",
        }
    )
    keywords = ["구독", "가입", "위약금"]
    insight = _build_video_context_insight(row, "negative", "구매", keywords)
    action = _build_resolution_point("negative", keywords, inquiry_flag=False)
    assert "구독/판매 프로세스" in insight or "판매" in insight
    assert "위약금" in action or "권유" in action
