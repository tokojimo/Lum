import json

import numpy as np
import pandas as pd
import pytest

from luxplate.statistics import paired_directional_t_tests


def _biological(values):
    return pd.DataFrame([
        {"experience_id": experiment, "souche": strain, "value": value}
        for experiment, row in enumerate(values, start=1)
        for strain, value in zip(("control", "reporter", "other"), row)
    ])


def test_directional_tests_use_log10_paired_biological_values_and_holm():
    biological = _biological([
        (10, 20, 11), (100, 210, 105), (1000, 2200, 1010), (10000, 23000, 9900)
    ])
    comparisons = paired_directional_t_tests(
        biological, value="value",
        comparisons=(("reporter", "control"), ("other", "control")),
    )

    expected = pytest.importorskip("scipy.stats").ttest_rel(
        np.log10([20, 210, 2200, 23000]), np.log10([10, 100, 1000, 10000]),
        alternative="greater",
    ).pvalue
    assert comparisons.loc[0, "p_raw"] == pytest.approx(expected)
    assert comparisons["p_holm"].between(0, 1).all()
    assert comparisons["n_pairs"].eq(4).all()
    assert comparisons.loc[0, "test"] == "paired t-test"
    assert comparisons.loc[0, "degrees_freedom"] == 3
    assert comparisons.loc[0, "holm_family_size"] == 2
    pairs = json.loads(comparisons.loc[0, "paired_values_json"])
    assert len(pairs) == 4
    assert pairs[0]["condition_1_value"] == 20.0
    assert pairs[0]["condition_2_log10"] == 1.0


def test_separately_planned_pairwise_significance_uses_raw_p_value():
    biological = pd.DataFrame([
        {"experience_id": experiment, "souche": strain, "value": value}
        for experiment, untreated, treated in (
            (1, 1.243e8, 3.019e8),
            (2, 1.303e8, 2.033e8),
            (3, 1.101e8, 1.921e8),
        )
        for strain, value in (("without SPD", untreated), ("with SPD", treated))
    ])
    # Five brackets may be displayed together, but each is a separately
    # planned 2-by-2 test. Repeating the contrast creates the same Holm family
    # size without changing the underlying test result.
    comparisons = paired_directional_t_tests(
        biological, value="value",
        comparisons=(("with SPD", "without SPD"),) * 5,
    )

    assert comparisons.loc[0, "p_raw"] < .05
    assert comparisons.loc[0, "p_holm"] >= .05
    assert comparisons.loc[0, "significance"] == "*"


def test_direction_is_preserved_and_no_selection_means_no_test():
    biological = _biological([(1, 2, 3), (2, 4, 6), (4, 9, 12)])
    assert paired_directional_t_tests(biological, value="value").empty
    forward = paired_directional_t_tests(
        biological, value="value", comparisons=(("reporter", "control"),)
    )
    reverse = paired_directional_t_tests(
        biological, value="value", comparisons=(("control", "reporter"),)
    )
    assert forward.loc[0, "p_raw"] < .05
    assert reverse.loc[0, "p_raw"] > .95


def test_nonpositive_values_are_excluded_and_three_pairs_are_required():
    biological = _biological([(1, 2, 3), (2, 4, 6), (0, 8, 12), (4, 9, 12)])
    comparisons = paired_directional_t_tests(
        biological, value="value", comparisons=(("reporter", "control"),)
    )
    assert comparisons.loc[0, "n_pairs"] == 3

    biological.loc[biological["experience_id"].eq(4) & biological["souche"].eq("control"), "value"] = 0
    assert paired_directional_t_tests(
        biological, value="value", comparisons=(("reporter", "control"),)
    ).empty
