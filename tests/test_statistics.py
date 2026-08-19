import json

import numpy as np
import pandas as pd
import pytest

from luxplate.statistics import (all_pairwise_comparisons, directional_test_diagnostics,
                                 paired_directional_t_tests)


@pytest.mark.parametrize(("conditions", "expected"), [
    (("A", "B", "C"), (("A", "B"), ("A", "C"), ("B", "C"))),
    (("A", "B", "C", "D"), (
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "C"), ("B", "D"), ("C", "D"),
    )),
])
def test_all_pairwise_comparisons_are_unique(conditions, expected):
    pairs = all_pairwise_comparisons(conditions)
    assert pairs == expected
    assert len({frozenset(pair) for pair in pairs}) == len(pairs)


def test_default_all_pairs_preserves_existing_pair_statistic_and_p_value():
    biological = _biological([(10, 20, 30), (12, 25, 32), (9, 22, 28), (14, 31, 35)])
    explicitly_selected = paired_directional_t_tests(
        biological, value="value", comparisons=(("control", "reporter"),)
    ).iloc[0]
    all_pairs = paired_directional_t_tests(biological, value="value")

    assert len(all_pairs) == 3
    same_pair = all_pairs.loc[
        all_pairs["condition_1"].eq("control")
        & all_pairs["condition_2"].eq("reporter")
    ].iloc[0]
    assert same_pair["statistic_t"] == explicitly_selected["statistic_t"]
    assert same_pair["p_raw"] == explicitly_selected["p_raw"]


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
    assert paired_directional_t_tests(biological, value="value", comparisons=()).empty
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
    insufficient = paired_directional_t_tests(
        biological, value="value", comparisons=(("reporter", "control"),)
    )
    assert insufficient.loc[0, "n_pairs"] == 2
    assert np.isnan(insufficient.loc[0, "p_raw"])
    assert insufficient.loc[0, "significance"] == "NA"
    assert insufficient.loc[0, "calculation_status"] == "non calculable"
    assert "moins de 3 paires" in insufficient.loc[0, "non_calculable_reason"]


def test_conditions_are_matched_canonically_instead_of_silently_disappearing():
    biological = pd.DataFrame([
        {"experience_id": experiment, "condition": condition, "value": value}
        for experiment, reporter, control in (
            (1, 20, 10), (2, 45, 20), (3, 95, 40), (4, 210, 80)
        )
        for condition, value in (
            ("PspeD2-1A\0SCFM2 (Spd)", reporter),
            ("P0\0SCFM2 (Spd)", control),
        )
    ])

    result = paired_directional_t_tests(
        biological, value="value", condition="condition",
        comparisons=(("  psped2-1a\0Experiment 2 | SCFM2  (Spd) ",
                      "p0\0SCFM2 (Spd)"),),
    )

    assert len(result) == 1
    assert result.loc[0, "condition_1"] == "PspeD2-1A\0SCFM2 (Spd)"
    assert result.loc[0, "n_pairs"] == 4
    assert result.loc[0, "calculation_status"] == "calculé"


def test_real_bm2_construct_names_match_reporter_metric_columns():
    """Regression for guided hypotheses made from full Varioskan strain names."""
    reporters = ("P0-lux", "PspeD2-1A-lux", "PspeD2-3B-lux")
    biological = pd.DataFrame([
        {"experience_id": f"exp{experiment}", "condition": reporter + "\0BM2",
         "value": value}
        for experiment, values in enumerate(((10, 20, 15), (12, 25, 19), (9, 21, 16)), start=1)
        for reporter, value in zip(reporters, values)
    ])
    full = {reporter: f"14.1Ac attB::{reporter}\0BM2" for reporter in reporters}

    result = paired_directional_t_tests(
        biological, value="value", condition="condition",
        comparisons=((full["PspeD2-1A-lux"], full["P0-lux"]),
                     (full["PspeD2-3B-lux"], full["P0-lux"]),
                     (full["PspeD2-1A-lux"], full["PspeD2-3B-lux"])),
    )

    assert len(result) == 3
    assert result["n_pairs"].eq(3).all()
    assert result["calculation_status"].eq("calculé").all()
    assert result["non_calculable_reason"].eq("").all()


def test_unmatched_validated_condition_is_reported_with_a_reason():
    biological = _biological([(1, 2, 3), (2, 4, 6), (4, 9, 12)])

    result = paired_directional_t_tests(
        biological, value="value",
        comparisons=(("missing", "control"),),
    )

    assert len(result) == 1
    assert result.loc[0, "condition_1"] == "missing"
    assert result.loc[0, "calculation_status"] == "non calculable"
    assert "condition A introuvable" in result.loc[0, "non_calculable_reason"]


def test_directional_diagnostics_expose_exact_runtime_identifiers():
    biological = pd.DataFrame([
        {"experience": f"exp{experiment}", "souche": strain,
         "Groupe": "Experiment 2 | BM2", "_comparison": strain + "\0BM2",
         "lum_norm_peak": value, "lum_norm_auc": value * 2}
        for experiment in range(1, 4)
        for strain, value in (("P0-lux", 10), ("PspeD2-1A-lux", 20), ("other", 30))
    ])
    requested = ("14.1Ac attB::PspeD2-1A-lux\0BM2", "14.1Ac attB::P0-lux\0BM2")

    diagnostics = directional_test_diagnostics(
        biological, value="lum_norm_peak", condition="_comparison",
        identity=("experience",), comparisons=(requested,),
    )

    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["requested_left"] == requested[0]
    assert diagnostic["canonical_left"] == "psped2-1a-lux\0bm2"
    assert diagnostic["condition_column"] == "_comparison"
    assert diagnostic["identity_columns"] == ["experience"]
    assert {item["repr"] for item in diagnostic["pivot_columns"]} >= {
        repr("P0-lux\0BM2"), repr("PspeD2-1A-lux\0BM2")
    }
    assert diagnostic["unique_values"]["Groupe"] == ["Experiment 2 | BM2"]
    assert len(diagnostic["biological_rows"]) == 6
