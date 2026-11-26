"""Regression checks for complete-block inference and figure annotations."""

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import ttest_rel
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

from luxplate.cytotox_statistics import biological_auc_medians, blocked_auc_anova, tukey_stars
from luxplate.cytotox_ui import _add_tukey_brackets


def test_technical_wells_supply_a_single_median_per_workbook():
    medians = biological_auc_medians({
        "Control": {"First": pd.Series([0.5, 1.0, 1.5]), "Second": pd.Series([1.0])},
        "Condition A": {"First": pd.Series([1.0, 2.0, 50.0]), "Third": pd.Series([4.0])},
    })
    assert medians.loc["First", "Condition A"] == 2
    assert medians.loc["Second", "Control"] == 1
    assert pd.isna(medians.loc["Third", "Control"])
    blocks, anova, comparisons = blocked_auc_anova(medians)
    assert blocks == 1
    assert anova.empty
    assert comparisons["p_adjusted"].isna().all()


def test_block_anova_matches_ols_and_tukey_with_two_groups_matches_paired_t():
    biological = pd.DataFrame({
        "A": [1., 10., 100., 220., np.nan],
        "B": [2., 12., 103., 224., 10.],
        "C": [10., 18., 111., 228., 11.],
        "All missing": [np.nan] * 5,
    })
    blocks, anova, comparisons = blocked_auc_anova(biological)
    long = biological[["A", "B", "C"]].dropna().rename_axis("block").reset_index().melt(
        id_vars="block", var_name="condition", value_name="auc",
    )
    expected = anova_lm(ols("auc ~ C(block) + C(condition)", data=long).fit(), typ=2)
    assert blocks == 4
    np.testing.assert_allclose(anova.loc[0, "f"], expected.loc["C(condition)", "F"])
    np.testing.assert_allclose(anova.loc[0, "p"], expected.loc["C(condition)", "PR(>F)"])
    np.testing.assert_allclose(anova.loc[2, "ms"], expected.loc["Residual", "sum_sq"] / expected.loc["Residual", "df"])
    assert comparisons["p_adjusted"].notna().sum() == 3
    assert len(comparisons) == 6

    _, _, pair = blocked_auc_anova(biological[["A", "B"]])
    expected_pair = ttest_rel(biological["A"].iloc[:4], biological["B"].iloc[:4])
    np.testing.assert_allclose(pair.loc[0, "p_adjusted"], expected_pair.pvalue)


def test_tukey_adjustment_and_brackets_only_show_strictly_significant_pairs():
    biological = pd.DataFrame({
        "A": [0, 1, 5, 9],
        "B": [1, 0, 4, 10],
        "C": [10, 11, 16, 20],
    })
    _, _, comparisons = blocked_auc_anova(biological)
    assert comparisons["p_adjusted"].iloc[0] >= .05
    assert (comparisons["p_adjusted"].iloc[1:] < .01).all()
    axis = Figure().subplots()
    axis.plot([1, 2, 3], [0, 1, 2], "o")
    assert _add_tukey_brackets(axis, list(biological), comparisons) == 2
    assert len(axis.lines) == 3
    assert [text.get_text() for text in axis.texts] == comparisons["p_adjusted"].iloc[1:].map(tukey_stars).tolist()
    assert [tukey_stars(value) for value in [.05, .01, .001, .0001, .00009, np.nan]] == [
        "", "*", "**", "***", "****", "",
    ]


def test_constant_paired_differences_have_no_residual_variance_estimate():
    _, anova, comparisons = blocked_auc_anova(pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}))
    assert anova["p"].isna().all()
    assert comparisons["p_adjusted"].isna().all()
