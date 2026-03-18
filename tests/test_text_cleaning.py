from src.analytics.text_cleaning import clean_text, classify_comment_quality, detect_language_hint



def test_clean_text_removes_url_and_normalizes_repeats():
    result = clean_text("배송이 느려요ㅠㅠㅠㅠ https://example.com")
    assert "http" not in result
    assert result == "배송이 느려요ㅠㅠ"



def test_detect_language_hint_korean_priority():
    assert detect_language_hint("삼성 fridge 좋아요") == "ko"



def test_classify_comment_quality_marks_low_signal():
    assert classify_comment_quality("좋아요!!!") == "low_signal"
