from src.api.quota import QuotaLedger, estimate_quota_usage



def test_quota_estimation_includes_search_cost():
    estimate = estimate_quota_usage(max_videos=100, comments_per_video=200, include_replies=True)
    assert estimate["search.list"] == 2
    assert estimate["videos.list"] == 2
    assert estimate["total_units"] > 200



def test_quota_ledger_actual_units():
    ledger = QuotaLedger(daily_limit=10000)
    ledger.record("search.list")
    ledger.record("videos.list", 2)
    assert ledger.actual_units() == 102



def test_quota_estimation_uses_default_window_for_full_collection():
    estimate = estimate_quota_usage(max_videos=10, comments_per_video=0, include_replies=False)
    assert estimate["commentThreads.list"] == 200
