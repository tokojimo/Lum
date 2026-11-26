"""Randomized complete-block ANOVA and Tukey contrasts for inter-manip AUC."""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import f, studentized_range


def biological_auc_medians(samples: dict[str, dict[str, pd.Series]]) -> pd.DataFrame:
    """Reduce the technical wells to one median per workbook and condition."""
    return pd.DataFrame(
        {condition: {replicate: wells.median() for replicate, wells in groups.items()}
         for condition, groups in samples.items()},
        dtype=float,
    )


def blocked_auc_anova(
    biological: pd.DataFrame,
) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """Analyze an additive condition + workbook model on complete biological blocks."""
    observed = biological.replace([np.inf, -np.inf], np.nan)
    complete = observed.loc[:, observed.notna().any()].dropna()
    blocks, conditions = complete.shape
    comparisons = pd.DataFrame(
        combinations(biological.columns, 2), columns=["condition_a", "condition_b"],
    )
    comparisons["difference"] = np.nan
    comparisons["p_adjusted"] = np.nan
    anova = pd.DataFrame(columns=["source", "df", "ss", "ms", "f", "p"])
    if blocks < 2 or conditions < 2:
        return blocks, anova, comparisons

    values = complete.to_numpy(dtype=float)
    centered = values - values.mean()
    condition_effect = centered.mean(axis=0)
    block_effect = centered.mean(axis=1, keepdims=True)
    residual = centered - condition_effect - block_effect
    ss_condition = blocks * np.square(condition_effect).sum()
    ss_block = conditions * np.square(block_effect).sum()
    ss_residual = np.square(residual).sum()
    df_condition, df_block = conditions - 1, blocks - 1
    df_residual = df_condition * df_block
    ms_condition, ms_block, mse = ss_condition / df_condition, ss_block / df_block, ss_residual / df_residual
    # A residual at floating-point precision cannot estimate the model variance.
    tolerance = np.finfo(float).eps * np.abs(centered).max() * 16
    estimable = np.abs(residual).max() > tolerance
    f_condition, f_block = (ms_condition / mse, ms_block / mse) if estimable else (np.nan, np.nan)
    anova = pd.DataFrame([
        ("Condition", df_condition, ss_condition, ms_condition, f_condition, f.sf(f_condition, df_condition, df_residual)),
        ("Biological block", df_block, ss_block, ms_block, f_block, f.sf(f_block, df_block, df_residual)),
        ("Residual", df_residual, ss_residual, mse, np.nan, np.nan),
    ], columns=anova.columns)

    effects = pd.Series(condition_effect, index=complete.columns)
    comparisons["difference"] = comparisons["condition_a"].map(effects) - comparisons["condition_b"].map(effects)
    if estimable:
        # The standard error is from the block-adjusted residual, not a one-way Tukey test.
        q = comparisons["difference"].abs().to_numpy() / np.sqrt(mse / blocks)
        comparisons["p_adjusted"] = studentized_range.sf(q, conditions, df_residual)
    return blocks, anova, comparisons


def tukey_stars(p_value: float) -> str:
    """Annotate Tukey-adjusted probabilities using strict thresholds."""
    return "*" * sum(p_value < threshold for threshold in (.05, .01, .001, .0001))
