"""I/O helpers for JSON, CSV, and parquet persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)



def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)



def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except (ImportError, ValueError, ModuleNotFoundError):
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False, encoding="utf-8-sig")
            return fallback
    if path.suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    raise ValueError(f"Unsupported dataframe output format: {path}")
