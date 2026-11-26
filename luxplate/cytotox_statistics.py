"""Paired comparisons of inter-manip AUC at the biological replicate level."""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


COMPARISON_COLUMNS = (
    "condition", "n_pairs", "n_unequal", "median_difference", "p_raw", "p_holm",
)


def biological_auc_medians(samples: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Reduce the technical wells to one median per workbook and condition."""
    return pd.DataFrame(
        {condition: {replicate: wells.median() for replicate, wells in groups.items()}
         for condition, groups in samples.items()},
        dtype=float,
    )


def paired_auc_comparisons(biological: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Run exact two-sided paired sign tests and adjust estimable p-values with Holm."""
    rows = []
    for condition in biological.columns:
        if condition == reference:
            continue
        paired = biological[[condition, reference]].dropna()
        differences = (paired[condition] - paired[reference]).to_numpy()
        # Ignore floating-point noise from the earlier AUC normalization.
        unequal = differences[~np.isclose(paired[condition], paired[reference], rtol=1e-12, atol=0)]
        p_value = (
            binomtest(np.count_nonzero(unequal > 0), len(unequal), alternative="two-sided").pvalue
            if len(unequal) else np.nan
        )
        rows.append((
            condition, len(paired), len(unequal),
            float(np.median(differences)) if len(differences) else np.nan, p_value, np.nan,
        ))

    result = pd.DataFrame(rows, columns=COMPARISON_COLUMNS)
    estimable = result["p_raw"].notna()
    if estimable.any():
        result.loc[estimable, "p_holm"] = multipletests(result.loc[estimable, "p_raw"], method="holm")[1]
    return result
