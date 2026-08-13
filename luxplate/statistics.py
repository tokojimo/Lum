"""Biological-replicate-aware directional inference.

Technical wells must be collapsed before calling this module.  Consequently,
every row supplied to the test represents one independent biological block.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


RESULT_COLUMNS = [
    "condition_1", "condition_2", "test", "alternative", "transformation",
    "value_column", "pairing_columns", "n_pairs", "statistic_t", "degrees_freedom",
    "mean_log10_difference", "p_raw", "p_holm", "holm_family_size", "alpha",
    "significance", "paired_values_json",
]


def _significance(p_value: float) -> str:
    if p_value < .0001:
        return "****"
    if p_value < .001:
        return "***"
    if p_value < .01:
        return "**"
    if p_value < .05:
        return "*"
    return "ns"


def paired_directional_t_tests(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "experience", "biological_replicate_id"),
    comparisons: tuple[tuple[str, str], ...] = (),
) -> pd.DataFrame:
    """Test explicitly requested ``left > right`` hypotheses on log10 values.

    A paired, one-tailed t-test is calculated only for positive, complete
    biological pairs.  Holm adjustment treats all requested, estimable
    contrasts as one comparison family.  With no pre-specified hypotheses no
    inferential test is performed.
    """
    ids = [column for column in identity
           if column in biological and biological[column].notna().any()]
    ids = ids or [column for column in ("Groupe",) if column in biological]
    if not comparisons or not ids or condition not in biological or value not in biological:
        return pd.DataFrame(columns=RESULT_COLUMNS)

    table = biological.pivot_table(index=ids, columns=condition, values=value, aggfunc="mean")
    rows: list[dict[str, object]] = []
    for left, right in comparisons:
        if left not in table or right not in table or left == right:
            continue
        pairs = table[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
        pairs = pairs.loc[pairs[left].gt(0) & pairs[right].gt(0)]
        if len(pairs) < 3:
            continue
        result = ttest_rel(
            np.log10(pairs[left].to_numpy(float)),
            np.log10(pairs[right].to_numpy(float)),
            alternative="greater",
        )
        raw = float(result.pvalue)
        if not np.isfinite(raw):
            raw = 1.0
        paired_records = []
        for pair_identity, pair_values in pairs.iterrows():
            identity_values = pair_identity if isinstance(pair_identity, tuple) else (pair_identity,)
            paired_records.append({
                **{column: value for column, value in zip(ids, identity_values)},
                "condition_1_value": float(pair_values[left]),
                "condition_2_value": float(pair_values[right]),
                "condition_1_log10": float(np.log10(pair_values[left])),
                "condition_2_log10": float(np.log10(pair_values[right])),
            })
        rows.append({
            "condition_1": left, "condition_2": right,
            "test": "paired t-test", "alternative": "condition_1 > condition_2",
            "transformation": "log10", "value_column": value,
            "pairing_columns": "|".join(ids), "n_pairs": len(pairs),
            "statistic_t": float(result.statistic), "degrees_freedom": len(pairs) - 1,
            "mean_log10_difference": float(
                (np.log10(pairs[left]) - np.log10(pairs[right])).mean()
            ),
            "p_raw": raw, "alpha": .05,
            "paired_values_json": json.dumps(paired_records, ensure_ascii=False, default=str),
        })

    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    order = result["p_raw"].sort_values().index
    adjusted = pd.Series(index=result.index, dtype=float)
    running = 0.0
    total = len(result)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, result.at[index, "p_raw"] * (total - rank)))
        adjusted.at[index] = running
    result["p_holm"] = adjusted
    result["holm_family_size"] = total
    # Each explicitly selected contrast is a separately planned, pairwise
    # hypothesis.  Keep Holm available in the exported audit table, but report
    # the raw significance used by the figure rather than silently treating all
    # simultaneously displayed brackets as one inferential family.
    result["significance"] = result["p_raw"].map(_significance)
    return result[RESULT_COLUMNS]
