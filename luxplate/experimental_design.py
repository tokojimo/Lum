"""Replicate-aware experimental-design helpers."""

import pandas as pd


def biological_n(data: pd.DataFrame, condition_id: str | None = None) -> int:
    """Count independent biological IDs, optionally for one condition.

    Files, plates, wells and technical replicates deliberately do not enter
    this calculation.
    """
    required = {"biological_replicate_id"}
    if condition_id is not None:
        required.add("condition_id")
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing experimental-design columns: {sorted(missing)}")
    selected = data
    if condition_id is not None:
        selected = data.loc[data["condition_id"] == condition_id]
    return int(selected["biological_replicate_id"].dropna().nunique())


def summarize_technical_replicates(
    data: pd.DataFrame, value: str
) -> pd.DataFrame:
    """Average technical wells within each biological unit and time point."""
    keys = ["condition_id", "biological_replicate_id", "temps_h"]
    missing = set(keys + [value]).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return data.groupby(keys, as_index=False, dropna=False)[value].mean()

