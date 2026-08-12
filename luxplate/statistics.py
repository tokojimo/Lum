"""Biological-replicate-aware inference boundary.

Technical wells are deliberately collapsed before inference.  This prevents
pseudoreplication while still allowing the plotting layer to display every well.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


RESULT_COLUMNS = ["condition_1", "condition_2", "n_pairs", "p_raw", "p_holm"]


def paired_nonparametric_tests(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "replicat", "Groupe"),
    comparisons: tuple[tuple[str, str], ...] | None = None,
) -> tuple[float, pd.DataFrame]:
    """Return a Friedman p-value and Holm-adjusted paired Wilcoxon comparisons.

    Only complete biological blocks are used.  With fewer than three complete
    blocks, inferential p-values are not reported because such tests would be
    uninformative for a publication figure.
    """
    ids = [column for column in identity if column in biological and biological[column].notna().any()]
    ids = ids or [column for column in ("Groupe",) if column in biological]
    if not ids or condition not in biological or value not in biological:
        return np.nan, pd.DataFrame(columns=RESULT_COLUMNS)
    table = biological.pivot_table(index=ids, columns=condition, values=value, aggfunc="mean").dropna()
    if len(table) < 3 or table.shape[1] < 2:
        return np.nan, pd.DataFrame(columns=RESULT_COLUMNS)
    omnibus = float(friedmanchisquare(*(table[column] for column in table.columns)).pvalue) \
        if table.shape[1] >= 3 else np.nan
    rows = []
    available_pairs = list(combinations(table.columns, 2))
    if comparisons is not None:
        requested = {frozenset(pair) for pair in comparisons}
        available_pairs = [pair for pair in available_pairs if frozenset(pair) in requested]
    for left, right in available_pairs:
        try:
            raw = float(wilcoxon(table[left], table[right], alternative="two-sided").pvalue)
        except ValueError:
            raw = 1.0
        rows.append({"condition_1": left, "condition_2": right,
                     "n_pairs": len(table), "p_raw": raw})
    result = pd.DataFrame(rows)
    if result.empty:
        return omnibus, pd.DataFrame(columns=RESULT_COLUMNS)
    order = result["p_raw"].sort_values().index
    adjusted = pd.Series(index=result.index, dtype=float)
    running = 0.0
    total = len(result)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, result.at[index, "p_raw"] * (total - rank)))
        adjusted.at[index] = running
    result["p_holm"] = adjusted
    return omnibus, result
