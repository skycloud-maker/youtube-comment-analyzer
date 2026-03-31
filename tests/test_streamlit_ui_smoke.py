import os
import tempfile
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(r"C:\codex\streamlit_app.py")
TMP_ROOT = Path(r"C:\codex\test_artifacts\streamlit_tmp")
ANALYSIS_SCOPE_LABEL = "\ubd84\uc11d \uc720\ud615"
INQUIRY_ONLY = "\ubb38\uc758 \ud3ec\ud568 \ub313\uae00\ub9cc"
INQUIRY_EXCLUDED = "\ubb38\uc758 \uc81c\uc678"


def _prepare_tempdir() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(TMP_ROOT)
    os.environ["TEMP"] = str(TMP_ROOT)
    tempfile.tempdir = str(TMP_ROOT)


def _run_app(mode: str | None = None) -> AppTest:
    _prepare_tempdir()
    app = AppTest.from_file(str(APP_PATH))
    if mode:
        app.query_params["mode"] = mode
    app.run(timeout=120)
    return app


def test_streamlit_full_mode_runs_without_exception():
    app = _run_app()

    assert not app.exception
    assert app.radio
    assert app.radio[0].label == ANALYSIS_SCOPE_LABEL


def test_streamlit_filter_scope_toggle_stays_stable():
    app = _run_app()
    assert not app.exception

    app.radio[0].set_value(INQUIRY_EXCLUDED)
    app.run(timeout=120)
    assert not app.exception

    app.radio[0].set_value(INQUIRY_ONLY)
    app.run(timeout=120)
    assert not app.exception


def test_streamlit_lite_mode_runs_without_exception():
    app = _run_app(mode="lite")

    assert not app.exception
