from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

RULES_PATH = Path(__file__).resolve().parents[1] / 'config' / 'dashboard_rules.yaml'


@lru_cache(maxsize=1)
def load_dashboard_rules() -> dict[str, Any]:
    with RULES_PATH.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f'Invalid dashboard rules file: {RULES_PATH}')
    return payload


@lru_cache(maxsize=1)
def ui_labels() -> dict[str, str]:
    return dict(load_dashboard_rules().get('ui', {}))


@lru_cache(maxsize=1)
def representative_labels() -> dict[str, dict[str, str]]:
    return dict(load_dashboard_rules().get('representative', {}))
