"""Biological-replicate-aware inference boundary.

Technical wells are deliberately collapsed before inference.  This prevents
pseudoreplication while still allowing the plotting layer to display every well.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, ttest_rel, wilcoxon


RESULT_COLUMNS = ["condition_1", "condition_2", "n_pairs", "p_raw", "p_holm"]


def directional_paired_t_tests(
    biological: pd.DataFrame, *, value: str,
    comparisons: tuple[tuple[str, str], ...], condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "replicat", "Groupe"),
) -> pd.DataFrame:
    """Test prespecified ``higher > lower`` contrasts on log10 biological means.

    Technical observations must already have been collapsed into ``biological``.
    Each contrast is paired on complete biological blocks. Holm correction is
    applied only across the explicitly requested family of directional tests.
    """
    ids = [column for column in identity if column in biological and biological[column].notna().any()]
    ids = ids or [column for column in ("Groupe",) if column in biological]
    if not ids or condition not in biological or value not in biological:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    table = biological.pivot_table(index=ids, columns=condition, values=value, aggfunc="mean")
    rows = []
    for higher, lower in comparisons:
        if higher not in table or lower not in table:
            continue
        pairs = table[[higher, lower]].dropna()
        pairs = pairs.loc[pairs[higher].gt(0) & pairs[lower].gt(0)]
        if len(pairs) < 2:
            raw = np.nan
        else:
            raw = float(ttest_rel(
                np.log10(pairs[higher]), np.log10(pairs[lower]), alternative="greater",
            ).pvalue)
        rows.append({"condition_1": higher, "condition_2": lower,
                     "n_pairs": len(pairs), "p_raw": raw})
    result = pd.DataFrame(rows, columns=RESULT_COLUMNS[:-1])
    if result.empty:
        return pd.DataFrame(columns=RESULT_COLUMNS)
    finite = result["p_raw"].dropna()
    adjusted = pd.Series(np.nan, index=result.index, dtype=float)
    order = finite.sort_values().index
    running = 0.0
    total = len(order)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, result.at[index, "p_raw"] * (total - rank)))
        adjusted.at[index] = running
    result["p_holm"] = adjusted
    return result


def paired_nonparametric_tests(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "replicat", "Groupe"),
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
    for left, right in combinations(table.columns, 2):
        try:
            raw = float(wilcoxon(table[left], table[right], alternative="two-sided").pvalue)
        except ValueError:
            raw = 1.0
        rows.append({"condition_1": left, "condition_2": right,
                     "n_pairs": len(table), "p_raw": raw})
    result = pd.DataFrame(rows)
    order = result["p_raw"].sort_values().index
    adjusted = pd.Series(index=result.index, dtype=float)
    running = 0.0
    total = len(result)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, result.at[index, "p_raw"] * (total - rank)))
        adjusted.at[index] = running
    result["p_holm"] = adjusted
    return omnibus, result
