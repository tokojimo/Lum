import pandas as pd
import pytest

from luxplate.statistics import paired_nonparametric_tests


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
