from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pandas as pd


def load_search_result_row(csv_path: str | Path, run_index: int) -> pd.Series:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Search results CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    index_column = None
    if "run_index" in df.columns:
        index_column = "run_index"
    elif "trial_index" in df.columns:
        index_column = "trial_index"
    if index_column is None:
        raise ValueError(f"CSV does not contain run_index column: {csv_path}")

    matched = df[df[index_column] == run_index]
    if matched.empty:
        raise ValueError(f"run_index={run_index} not found in {csv_path}")
    if len(matched) > 1:
        raise ValueError(f"run_index={run_index} is not unique in {csv_path}")
    return matched.iloc[0]


def extract_model_kwargs(row: pd.Series, allowed_keys: Iterable[str]) -> Dict[str, object]:
    overrides: Dict[str, object] = {}
    for key in allowed_keys:
        value = None
        if key in row.index:
            value = row[key]
        elif f"model_params.{key}" in row.index:
            value = row[f"model_params.{key}"]
        else:
            continue
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            value = value.item()
        overrides[key] = value
    return overrides
