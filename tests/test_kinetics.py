import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from luxplate.kinetics import (
    REJECTED_SERIES_COLUMNS,
    SERIES_METRIC_COLUMNS,
    TECHNICAL_SUMMARY_COLUMNS,
    WARNING_COLUMNS,
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
    growth = calculate_growth_metrics(data)
    assert growth["n_growth_points"] == 2
    assert growth["od_auc"] == pytest.approx(-1.05)

    result = run_kinetics(data)
    assert result.series_metrics.empty
    assert "non_positive_od_auc" in result.rejected_series.iloc[0]["reason"]


def test_duplicate_times_warn_and_break_growth_windows_without_rejecting_global_metrics():
    data = kinetic_table()
    data.loc[2, "temps_h"] = 1
    result = run_kinetics(data)
    assert len(result.series_metrics) == 1
    assert result.series_metrics.iloc[0]["od_auc"] == pytest.approx(calculate_auc(data.temps_h, data.DO_corr))
    assert "duplicate_time" in set(result.warnings["code"])


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


def test_auc_spans_single_and_successive_missing_observations():
    assert calculate_auc([0, 1, 2, 3], [0, 1, np.nan, 3]) == pytest.approx(4.5)
    assert calculate_auc([0, 1, 2, 3, 4], [0, 1, np.nan, np.nan, 4]) == pytest.approx(8.0)


def test_normalized_auc_is_ratio_of_corrected_lum_and_od_aucs_on_common_window():
    data = kinetic_table()
    result = run_kinetics(data)
    row = result.series_metrics.iloc[0]
    expected_lum_auc = calculate_auc(data["temps_h"], data["Lum_corr"])
    expected_od_auc = calculate_auc(data["temps_h"], data["DO_corr"])
    assert row["lum_corr_auc"] == pytest.approx(expected_lum_auc)
    assert row["lum_norm_auc"] == pytest.approx(expected_lum_auc / expected_od_auc)
    assert row["n_auc_points"] == len(data)


def test_auc_ratio_rejects_wells_without_the_shared_experiment_window():
    complete = kinetic_table(header="complete")
    incomplete = kinetic_table(header="incomplete").query("temps_h > 0")
    result = run_kinetics(pd.concat([complete, incomplete], ignore_index=True))
    assert result.series_metrics["sample_header"].tolist() == ["complete"]
    rejected = result.rejected_series.set_index("sample_header")
    assert "incomplete_common_auc_window" in rejected.loc["incomplete", "reason"]


def test_all_experiments_use_the_complete_span_of_the_shortest_experiment():
    long = kinetic_table(experience="long", header="long")
    short = kinetic_table(experience="short", header="short").query("temps_h <= 2")
    short["temps_h"] += 10
    result = run_kinetics(pd.concat([long, short], ignore_index=True))
    rows = result.series_metrics.set_index("experience_id")

    expected_long_lum = calculate_auc(long.loc[long["temps_h"] <= 2, "temps_h"],
                                      long.loc[long["temps_h"] <= 2, "Lum_corr"])
    expected_long_od = calculate_auc(long.loc[long["temps_h"] <= 2, "temps_h"],
                                     long.loc[long["temps_h"] <= 2, "DO_corr"])
    assert rows.loc["long", "lum_norm_auc"] == pytest.approx(expected_long_lum / expected_long_od)
    assert rows.loc["long", "od_auc"] == pytest.approx(expected_long_od)
    assert rows.loc["long", "auc_window_duration_h"] == pytest.approx(2)
    assert rows.loc["short", "auc_window_duration_h"] == pytest.approx(2)
    assert rows.loc["long", "auc_window_start_h"] == pytest.approx(0)
    assert rows.loc["short", "auc_window_start_h"] == pytest.approx(10)
    assert rows.loc["long", "n_auc_points"] == 3
    assert rows.loc["short", "n_auc_points"] == 3


def test_same_header_in_two_wells_remains_two_series_and_replicate_is_summarized_separately():
    a1 = kinetic_table(header="shared")
    a2 = kinetic_table(header="shared", scale=2); a2["puits"] = "A2"
    biological = kinetic_table(header="shared", scale=3); biological["puits"] = "A3"; biological["replicat"] = 2
    result = run_kinetics(pd.concat([a1, a2, biological], ignore_index=True))
    assert len(result.series_metrics) == 3
    assert set(result.series_metrics["puits"]) == {"A1", "A2", "A3"}
    assert result.strain_summary["n_technical_series"].tolist() == [2, 1]
    assert result.strain_summary["replicat"].tolist() == [1, 2]


def test_irregular_window_uses_real_time_and_reports_quality_and_nonpositive_reason():
    data = kinetic_table().iloc[:3].copy()
    data["temps_h"] = [0.0, 0.5, 3.0]
    data["DO_corr"] = 0.1 * np.exp(0.4 * data["temps_h"])
    result = run_kinetics(data, growth_window_min_duration_h=2, growth_rate_min_r_squared=0.99)
    row = result.series_metrics.iloc[0]
    assert row["max_growth_rate_per_h"] == pytest.approx(0.4)
    assert row["growth_rate_r_squared"] == pytest.approx(1)

    data["DO_corr"] = [0.3, 0.2, 0.1]
    row = run_kinetics(data).series_metrics.iloc[0]
    assert np.isnan(row["doubling_time_h"])
    assert row["growth_rate_publishability_reason"] == "non_positive_growth_rate"


def test_low_regression_quality_is_explicit_and_output_schemas_are_stable():
    data = kinetic_table().iloc[:3].copy(); data["DO_corr"] = [0.1, 0.5, 0.11]
    result = run_kinetics(data, growth_rate_min_r_squared=0.99)
    assert result.series_metrics.iloc[0]["growth_rate_publishability_reason"] == "insufficient_regression_quality"
    assert "insufficient_regression_quality" in set(result.warnings["code"])

    rejected = kinetic_table().iloc[:1]
    empty = run_kinetics(rejected)
    assert list(empty.series_metrics.columns) == list(SERIES_METRIC_COLUMNS)
    assert list(empty.strain_summary.columns) == list(TECHNICAL_SUMMARY_COLUMNS)
    assert list(empty.rejected_series.columns) == list(REJECTED_SERIES_COLUMNS)
    assert list(run_kinetics(kinetic_table()).warnings.columns) == list(WARNING_COLUMNS)
