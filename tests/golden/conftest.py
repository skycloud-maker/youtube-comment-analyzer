from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

GOLDENSET_PATH = Path(r"C:\codex\docs\harness\validation\goldenset_seed.yaml")


@pytest.fixture(scope="session")
def goldenset_items() -> list[dict[str, Any]]:
    data = yaml.safe_load(GOLDENSET_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("goldenset_seed.yaml must load as a list")
    return data
