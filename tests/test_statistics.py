import pandas as pd
import pytest

from luxplate.statistics import directional_paired_t_tests, paired_nonparametric_tests


def test_directional_paired_t_tests_use_log_biological_pairs_and_holm():
    biological = pd.DataFrame([
        {"experience_id": experiment, "condition": condition, "value": value}
        for experiment, (control, mutant_ratio, other_ratio) in enumerate(
            ((10.0, 3.8, 1.8), (20.0, 4.0, 2.0), (40.0, 4.2, 2.2)), start=1,
        )
        for condition, value in (("control", control), ("mutant", control * mutant_ratio),
                                 ("other", control * other_ratio))
    ])

    result = directional_paired_t_tests(
        biological, value="value", condition="condition",
        comparisons=(("mutant", "control"), ("other", "control")),
    )

    assert result["n_pairs"].eq(3).all()
    assert result["p_raw"].lt(.05).all()
    assert result["p_holm"].ge(result["p_raw"]).all()


def test_directional_paired_t_tests_ignore_nonpositive_and_incomplete_pairs():
    biological = pd.DataFrame([
        {"experience_id": "E1", "condition": "higher", "value": 4.0},
        {"experience_id": "E1", "condition": "lower", "value": 2.0},
        {"experience_id": "E2", "condition": "higher", "value": 5.0},
        {"experience_id": "E2", "condition": "lower", "value": 0.0},
        {"experience_id": "E3", "condition": "higher", "value": 6.0},
    ])

    result = directional_paired_t_tests(
        biological, value="value", condition="condition",
        comparisons=(("higher", "lower"),),
    )

    assert result.loc[0, "n_pairs"] == 1
    assert pd.isna(result.loc[0, "p_raw"])


def test_paired_tests_use_complete_biological_blocks_and_holm_correction():
    biological = pd.DataFrame([
        {"experience_id": experiment, "replicat": 1, "souche": strain, "value": value}
        for experiment, values in {
            "E1": (1.0, 5.0, 9.0), "E2": (2.0, 6.0, 10.0), "E3": (3.0, 7.0, 11.0)
        }.items()
        for strain, value in zip(("P0-lux", "A-lux", "B-lux"), values)
    ])

    omnibus, comparisons = paired_nonparametric_tests(biological, value="value")

    assert 0 <= omnibus <= 1
    assert len(comparisons) == 3
    assert comparisons["p_holm"].between(0, 1).all()
    assert comparisons["n_pairs"].eq(3).all()


def test_paired_tests_report_significance_for_a_clear_repeated_effect():
    biological = pd.DataFrame([
        {"experience_id": experiment, "souche": strain, "value": value}
        for experiment in range(10)
        for strain, value in (("control", experiment), ("reporter", experiment + 10))
    ])

    _, comparisons = paired_nonparametric_tests(biological, value="value")

    assert comparisons.loc[0, "p_raw"] == pytest.approx(0.001953125)
    assert comparisons.loc[0, "p_holm"] < 0.05


def test_paired_tests_do_not_claim_inference_below_three_biological_blocks():
    biological = pd.DataFrame([
        {"experience_id": "E1", "souche": "P0-lux", "value": 1.0},
        {"experience_id": "E1", "souche": "A-lux", "value": 2.0},
    ])

    omnibus, comparisons = paired_nonparametric_tests(biological, value="value")

    assert pd.isna(omnibus)
    assert comparisons.empty
