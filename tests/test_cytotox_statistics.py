"""Regression checks for biological pairing in Cytotox inter-manip statistics."""

import numpy as np
import pandas as pd

from luxplate.cytotox_statistics import biological_auc_medians, paired_auc_comparisons


def test_technical_wells_supply_a_single_median_per_workbook():
    medians = biological_auc_medians({
        "Témoin": {"Premier": pd.Series([0.5, 1.0, 1.5]), "Second": pd.Series([1.0])},
        "Condition A": {"Premier": pd.Series([1.0, 2.0, 50.0]), "Third": pd.Series([4.0])},
    })

    assert medians.loc["Premier", "Condition A"] == 2
    assert medians.loc["Second", "Témoin"] == 1
    assert pd.isna(medians.loc["Third", "Témoin"])
    comparison = paired_auc_comparisons(medians, "Témoin").iloc[0]
    assert comparison["n_pairs"] == 1
    assert comparison["median_difference"] == 1


def test_sign_tests_pair_workbooks_and_adjust_only_calculable_comparisons():
    values = pd.DataFrame({
        "Témoin": [1] * 6 + [np.nan],
        "A": [2] * 6 + [99],
        "B": [0, 2] * 3 + [np.nan],
        "Égalité": [1 + 1e-15] * 6 + [np.nan],
        "Sans paire": [np.nan] * 6 + [2],
    })
    result = paired_auc_comparisons(values, "Témoin").set_index("condition")

    assert result.loc["A", "n_pairs"] == 6
    assert result.loc["A", "median_difference"] == 1
    assert result.loc["A", "p_raw"] == 0.03125
    assert result.loc["A", "p_holm"] == 0.0625
    assert result.loc["B", "p_holm"] == 1
    assert result.loc["Égalité", "n_pairs"] == 6
    assert result.loc["Égalité", "n_unequal"] == 0
    assert pd.isna(result.loc["Égalité", "p_raw"])
    assert result.loc["Sans paire", "n_pairs"] == 0
    assert pd.isna(result.loc["Sans paire", "p_holm"])
