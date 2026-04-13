from src.analytics.comment_analysis import _build_reason


def test_build_reason_uses_korean_strength_label() -> None:
    reason = _build_reason(
        sentiment="negative",
        classification_type="complaint",
        signal_strength="strong",
        context_used=False,
        target="washer",
    )
    assert "strong한" not in reason
    assert "높은" in reason
