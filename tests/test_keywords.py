from src.analytics.keywords import normalize_keyword


def test_keyword_normalization_drops_stopwords_and_generic_terms():
    assert normalize_keyword("are") == ""
    assert normalize_keyword("but") == ""
    assert normalize_keyword("문제") == ""
    assert normalize_keyword("후기") == ""


def test_keyword_normalization_keeps_business_terms():
    assert normalize_keyword("door") == "도어"
    assert normalize_keyword("noise") == "소음"
    assert normalize_keyword("냉각") == "냉각"


def test_keyword_normalization_drops_reported_noise_terms():
    assert normalize_keyword("샀는데") == ""
    assert normalize_keyword("있는데") == ""
    assert normalize_keyword("없어서") == ""
    assert normalize_keyword("없이") == ""
    assert normalize_keyword("광고했다고") == ""
    assert normalize_keyword("유튜버") == ""
