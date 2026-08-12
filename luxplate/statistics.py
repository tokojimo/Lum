"""Biological-replicate-aware directional inference.

Technical wells must be collapsed before calling this module.  Consequently,
every row supplied to the test represents one independent biological block.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


RESULT_COLUMNS = ["condition_1", "condition_2", "n_pairs", "p_raw", "p_holm"]


def paired_directional_t_tests(
    biological: pd.DataFrame, *, value: str, condition: str = "souche",
    identity: tuple[str, ...] = ("experience_id", "replicat", "Groupe"),
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
        rows.append({"condition_1": left, "condition_2": right,
                     "n_pairs": len(pairs), "p_raw": raw})

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
    return result[RESULT_COLUMNS]
