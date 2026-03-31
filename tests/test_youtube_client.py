import logging
from pathlib import Path

from src.api.quota import QuotaLedger
from src.api.youtube_client import YoutubeApiError, YoutubeClient


class DummyResponse:
    def __init__(self, status_code: int, payload: dict, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests

            error = requests.HTTPError(self.text or f"status={self.status_code}")
            error.response = self
            raise error


class DummySession:
    def __init__(self, response: DummyResponse) -> None:
        self.response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self.response


def build_client(raw_dir: Path, response: DummyResponse) -> YoutubeClient:
    client = YoutubeClient(
        api_key="dummy",
        base_url="https://example.com/youtube/v3",
        timeout_seconds=3,
        max_retries=0,
        backoff_seconds=0.0,
        raw_dir=raw_dir,
        quota_ledger=QuotaLedger(daily_limit=10000),
        logger=logging.getLogger("test"),
    )
    client.session = DummySession(response)
    return client


def test_youtube_client_records_quota_and_writes_audit():
    temp_dir = Path(r"C:\codex\test_artifacts\yt_client_success")
    temp_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(temp_dir, DummyResponse(200, {"items": []}))
    payload = client.search_videos(run_id="run123", part="snippet", q="dryer")

    assert payload == {"items": []}
    assert client.quota_ledger.actual_calls["search.list"] == 1
    audit_files = list(temp_dir.rglob("*.json"))
    assert audit_files


def test_youtube_client_raises_domain_error_on_http_failure():
    temp_dir = Path(r"C:\codex\test_artifacts\yt_client_failure")
    temp_dir.mkdir(parents=True, exist_ok=True)
    client = build_client(temp_dir, DummyResponse(403, {"error": "quota"}, text="quota exceeded"))

    try:
        client.search_videos(run_id="run123", part="snippet", q="dryer")
    except YoutubeApiError as exc:
        assert "quota exceeded" in str(exc)
    else:
        raise AssertionError("YoutubeApiError was not raised")
