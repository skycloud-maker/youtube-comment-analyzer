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



def _coerce_complex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dict/list columns to JSON strings so Arrow can serialize them."""
    for col in df.columns:
        if df[col].dropna().empty:
            continue
        sample = df[col].dropna().iloc[0]
        if isinstance(sample, (dict, list)):
            df[col] = df[col].map(lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v)
    return df


def write_dataframe(df: pd.DataFrame, path: Path) -> Path:
    ensure_dir(path.parent)
    if path.suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
            return path
        except Exception:
            try:
                _coerce_complex_columns(df).to_parquet(path, index=False)
                return path
            except (ImportError, ModuleNotFoundError):
                pass
            fallback = path.with_suffix(".csv")
            df.to_csv(fallback, index=False, encoding="utf-8-sig")
            return fallback
    if path.suffix == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path
    raise ValueError(f"Unsupported dataframe output format: {path}")
