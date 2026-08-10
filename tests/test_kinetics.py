import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from luxplate.kinetics import (
    calculate_auc,
    calculate_growth_metrics,
    calculate_peak,
    run_kinetics,
)


def kinetic_table(experience="E1", header="S1_A1", scale=1.0):
    time = np.arange(5, dtype=float)
    od = 0.1 * np.exp(np.log(2) * time / 2)
    return pd.DataFrame({
        "experience_id": experience, "temps_h": time, "souche": "S1", "Groupe": "G1",
        "sample_header": header, "puits": "A1", "replicat": 1, "type": "souche",
        "DO_corr": od, "Lum_corr": scale * np.array([1, 3, 5, 4, 2], dtype=float),
        "Lum_norm": scale * np.array([10, 30, 50, 40, 20], dtype=float),
    })


def test_auc_known_curve_and_no_implicit_interpolation_or_input_mutation():
    time = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([0.0, 1.0, np.nan, 3.0])
    original = values.copy()
    assert calculate_auc(time, values) == pytest.approx(4.5)
    np.testing.assert_equal(values, original)


def test_peak_uses_first_observed_time_when_maximum_is_tied():
    assert calculate_peak([3, 1, 2], [7, 7, 2]) == (7, 1)


def test_exponential_growth_rate_and_doubling_time_are_known():
    metrics = calculate_growth_metrics(kinetic_table(), growth_window_points=3)
    assert metrics["max_growth_rate_per_h"] == pytest.approx(np.log(2) / 2)
    assert metrics["doubling_time_h"] == pytest.approx(2)


def test_too_few_points_are_rejected_and_invalid_window_is_not_bridged():
    data = kinetic_table().iloc[:2].copy()
    result = run_kinetics(data, growth_window_points=3, minimum_auc_points=3)
    assert result.series_metrics.empty
    assert "insufficient_growth_points" in result.rejected_series.iloc[0]["reason"]

    data = kinetic_table()
    data.loc[2, "DO_corr"] = np.nan
    result = run_kinetics(data, growth_window_points=3)
    assert np.isnan(result.series_metrics.iloc[0]["max_growth_rate_per_h"])
    assert set(result.warnings["code"]) == {"insufficient_growth_window"}


def test_nonpositive_infinite_and_missing_values_are_excluded():
    data = kinetic_table()
    data["DO_corr"] = [0.1, 0, -1, np.inf, 0.4]
    data["Lum_norm"] = [10, np.nan, np.inf, 20, 30]
    result = run_kinetics(data)
    row = result.series_metrics.iloc[0]
    assert row["n_growth_points"] == 2
    assert row["n_lum_norm_points"] == 3
    assert row["od_auc"] == pytest.approx(-1.05)


def test_duplicate_times_reject_series_with_reason():
    data = kinetic_table()
    data.loc[2, "temps_h"] = 1
    result = run_kinetics(data)
    assert result.series_metrics.empty
    assert result.rejected_series.iloc[0]["reason"] == "duplicate_time"


def test_experiments_are_strictly_separate_and_inputs_unchanged():
    data = pd.concat([kinetic_table("E1", "S1_A1"), kinetic_table("E2", "S1_A1", 2)])
    original = data.copy(deep=True)
    result = run_kinetics(data)
    assert len(result.series_metrics) == 2
    assert len(result.strain_summary) == 2
    assert set(result.strain_summary["experience_id"]) == {"E1", "E2"}
    pdt.assert_frame_equal(data, original)


def test_normalized_output_integrates_with_kinetics():
    from luxplate.blanks import run_blank_correction
    from luxplate.normalization import run_normalization

    rows = []
    for kind, strain, header, ods in [
        ("blanc", "B", "B1", [0.1] * 5),
        ("blanc", "B", "B2", [0.1] * 5),
        ("souche", "S", "S1", [0.2, 0.25, 0.3, 0.4, 0.5]),
    ]:
        for time, od in enumerate(ods):
            rows.append({"experience_id": "E1", "temps_h": time, "souche": strain,
                         "Groupe": "G", "replicat": 1, "sample_header": header,
                         "puits": header, "DO_brute": od, "Lum_brute": od * (time + 1) * 100,
                         "type": kind})
    corrected = run_blank_correction(pd.DataFrame(rows)).corrected_data
    normalized = run_normalization(corrected, consecutive_points=3).normalized_data
    result = run_kinetics(normalized, growth_window_points=3)
    assert len(result.series_metrics) == 1
    assert result.series_metrics.iloc[0]["lum_norm_peak"] > 0
