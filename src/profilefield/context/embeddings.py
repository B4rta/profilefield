"""Validated precomputed embedding extraction."""

from __future__ import annotations

import pandas as pd


def select_embedding_columns(frame: pd.DataFrame, prefixes: list[str]) -> list[str]:
    columns = [column for column in frame.columns if any(column.startswith(p) for p in prefixes)]
    if not columns:
        raise ValueError(f"No context features found for prefixes {prefixes}")
    if frame[columns].isna().any().any():
        raise ValueError("Context features contain missing values")
    return columns
