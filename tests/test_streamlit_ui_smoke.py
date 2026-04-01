import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

APP_PATH = Path(r"C:\codex\streamlit_app.py")
TMP_ROOT = Path(r"C:\codex\test_artifacts\streamlit_tmp")
ANALYSIS_SCOPE_LABEL = "분석 유형"
INQUIRY_ONLY = "문의 포함 댓글만"
INQUIRY_EXCLUDED = "문의 제외"
PYTHON = Path(r"C:\codex\python312\python.exe")


def _prepare_tempdir() -> None:
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(TMP_ROOT)
    os.environ["TEMP"] = str(TMP_ROOT)
    os.environ["TMPDIR"] = str(TMP_ROOT)
    tempfile.tempdir = str(TMP_ROOT)


def _run_app_subprocess(mode: str | None = None, toggle_sequence: list[str] | None = None) -> dict:
    _prepare_tempdir()
    script = r'''
import json
import os
import sys
import tempfile
from pathlib import Path

TMP_ROOT = Path(r"C:\codex\test_artifacts\streamlit_tmp")
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["TMP"] = str(TMP_ROOT)
os.environ["TEMP"] = str(TMP_ROOT)
os.environ["TMPDIR"] = str(TMP_ROOT)
tempfile.tempdir = str(TMP_ROOT)

from streamlit.testing.v1 import AppTest

app_path = sys.argv[1]
mode = sys.argv[2] if sys.argv[2] != "__NONE__" else None
sequence = json.loads(sys.argv[3])
app = AppTest.from_file(app_path)
if mode:
    app.query_params["mode"] = mode
app.run(timeout=120)
result = {
    "initial_exception": bool(app.exception),
    "radio_label": app.radio[0].label if app.radio else None,
    "sequence_ok": True,
}
for value in sequence:
    app.radio[0].set_value(value)
    app.run(timeout=120)
    if app.exception:
        result["sequence_ok"] = False
        break
print(json.dumps(result, ensure_ascii=False))
'''
    completed = subprocess.run(
        [str(PYTHON), "-c", script, str(APP_PATH), mode or "__NONE__", json.dumps(toggle_sequence or [], ensure_ascii=False)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        env={**os.environ, "TMP": str(TMP_ROOT), "TEMP": str(TMP_ROOT), "TMPDIR": str(TMP_ROOT)},
        timeout=180,
    )
    if completed.returncode != 0:
        raise AssertionError(f"subprocess smoke failed: {completed.stderr}\nSTDOUT={completed.stdout}")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1])
    return payload


def test_streamlit_full_mode_runs_without_exception():
    payload = _run_app_subprocess()
    assert payload["initial_exception"] is False
    assert isinstance(payload["radio_label"], str) and payload["radio_label"].strip()


def test_streamlit_filter_scope_toggle_stays_stable():
    payload = _run_app_subprocess(toggle_sequence=[INQUIRY_EXCLUDED, INQUIRY_ONLY])
    assert payload["initial_exception"] is False
    assert payload["sequence_ok"] is True


def test_streamlit_lite_mode_runs_without_exception():
    payload = _run_app_subprocess(mode="lite")
    assert payload["initial_exception"] is False
