from src.analytics.text_cleaning import clean_text, classify_comment_quality, detect_language_hint


def test_clean_text_removes_url_and_normalizes_repeats():
    result = clean_text("\ubc30\uc1a1\uc774 \ub290\ub824\uc694\u3160\u3160\u3160\u3160 https://example.com")
    assert "http" not in result
    assert result == "\ubc30\uc1a1\uc774 \ub290\ub824\uc694\u3160\u3160"


def test_detect_language_hint_korean_priority():
    assert detect_language_hint("\uc0bc\uc131 fridge \uc88b\uc544\uc694") == "ko"


def test_classify_comment_quality_marks_generic_or_low_context():
    assert classify_comment_quality("\uc88b\uc544\uc694!!!") in {"generic_engagement", "low_context", "meaningless"}
