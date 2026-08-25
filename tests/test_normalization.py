import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from luxplate.blanks import run_blank_correction
from luxplate.normalization import (
    calculate_blank_od_threshold,
    find_first_consecutive_valid_time,
    run_normalization,
    validate_normalization_inputs,
)


def corrected_table(experience="E1"):
    rows = []
    for header, kind, strain, ods, lums in [
        ("blank_1", "blanc", "blank", [-0.01, 0.01, -0.01, 0.01, 0.0], [0, 0, 0, 0, 0]),
        ("strain_1", "souche", "S1", [0.02, 0.06, 0.07, 0.08, 0.09], [2, 6, 14, 24, 36]),
    ]:
        for time, (od, lum) in enumerate(zip(ods, lums)):
            rows.append({"experience_id": experience, "temps_h": time, "souche": strain,
                         "sample_header": header, "replicat": 1, "type": kind,
                         "DO_corr": od, "Lum_corr": lum})
    return pd.DataFrame(rows)


def test_blank_threshold_uses_mean_plus_sample_sd():
    data = corrected_table()
    details = calculate_blank_od_threshold(data, blank_sd_multiplier=3)
    expected = data.query("type == 'blanc'")["DO_corr"]
    assert details.iloc[0]["blank_threshold"] == pytest.approx(expected.mean() + 3 * expected.std(ddof=1))


def test_minimum_od_controls_threshold_and_first_exact_block():
    result = run_normalization(corrected_table(), minimum_od=0.05, consecutive_points=3)
    threshold = result.threshold_details.iloc[0]
    assert threshold["effective_threshold"] == pytest.approx(0.05)
    series = result.series_validation.iloc[0]
    assert series["normalization_start_time_h"] == 1
    strain = result.normalized_data.query("type == 'souche'").set_index("temps_h")
    assert np.isnan(strain.loc[0, "Lum_norm"])
    assert strain.loc[1, "Lum_norm"] == pytest.approx(100)
    assert strain.loc[2, "Lum_norm"] == pytest.approx(200)


def test_blank_threshold_wins_when_above_minimum():
    data = corrected_table()
    data.loc[data["type"].eq("blanc"), "DO_corr"] = [0.1, 0.2, 0.1, 0.2, 0.15]
    result = run_normalization(data, blank_sd_multiplier=1, minimum_od=0.05)
    row = result.threshold_details.iloc[0]
    assert row["effective_threshold"] == pytest.approx(row["blank_threshold"])
    assert row["effective_threshold"] > 0.05


def test_invalid_od_and_short_series_are_never_divided():
    data = corrected_table()
    data.loc[data["sample_header"].eq("strain_1"), "DO_corr"] = [0.1, 0.0, np.inf, -1, np.nan]
    result = run_normalization(data, consecutive_points=3)
    strains = result.normalized_data.query("type == 'souche'")
    assert strains["Lum_norm"].isna().all()
    assert "series_not_validated" in set(strains["normalization_reason"])
    short = corrected_table().query("temps_h < 2")
    short_result = run_normalization(short, consecutive_points=3)
    assert short_result.series_validation.iloc[0]["reason"] == "series_too_short"


def _endpoint_table(rows):
    return pd.DataFrame([
        {"experience_id": experience, "temps_h": time, "souche": strain,
         "sample_header": header, "replicat": replicate, "type": kind,
         "DO_corr": od, "Lum_corr": lum, "Groupe": "G"}
        for experience, time, strain, header, replicate, kind, od, lum in rows
    ])


def test_single_time_endpoint_normalizes_one_valid_measurement():
    data = _endpoint_table([("E1", 4, "S1", "S1_1", 1, "souche", .20, 10_000)])

    row = run_normalization(data, minimum_od=.05, consecutive_points=3).normalized_data.iloc[0]

    assert row["Lum_norm"] == pytest.approx(50_000)
    assert bool(row["normalization_ok"])
    assert row["normalization_reason"] == ""


def test_single_time_endpoint_normalizes_all_strains_and_replicates():
    data = _endpoint_table([
        ("E1", 4, strain, f"{strain}_{replicate}", replicate, "souche", od, lum)
        for strain, od, lum in (("S1", .2, 10_000), ("S2", .4, 12_000))
        for replicate in (1, 2)
    ])

    normalized = run_normalization(data, consecutive_points=3).normalized_data

    np.testing.assert_allclose(normalized["Lum_norm"],
                               normalized["Lum_corr"] / normalized["DO_corr"])
    assert normalized["normalization_ok"].all()


def test_single_time_endpoint_respects_effective_threshold():
    data = _endpoint_table([
        ("E1", 4, "blank", "B1", 1, "blanc", .10, 0),
        ("E1", 4, "S1", "S1_1", 1, "souche", .10, 10_000),
    ])

    row = run_normalization(data, blank_sd_multiplier=0, minimum_od=.05).normalized_data.query(
        "type == 'souche'"
    ).iloc[0]

    assert np.isnan(row["Lum_norm"])
    assert not bool(row["normalization_ok"])
    assert row["normalization_reason"] == "od_not_above_threshold"


@pytest.mark.parametrize(("od", "reason"), [(0, "non_positive_od"), (-.1, "non_positive_od")])
def test_single_time_endpoint_never_divides_by_non_positive_od(od, reason):
    data = _endpoint_table([("E1", 4, "S1", "S1_1", 1, "souche", od, 10_000)])

    row = run_normalization(data, minimum_od=0).normalized_data.iloc[0]

    assert np.isnan(row["Lum_norm"])
    assert not bool(row["normalization_ok"])
    assert row["normalization_reason"] == reason


def test_independent_single_time_experiments_allow_small_clock_differences():
    data = _endpoint_table([
        ("E1", 4.0, "S1", "E1_S1", 1, "souche", .2, 10_000),
        ("E2", 4.05, "S1", "E2_S1", 1, "souche", .25, 10_000),
    ])

    normalized = run_normalization(data, consecutive_points=3).normalized_data

    assert normalized["normalization_ok"].all()


def test_multi_time_kinetic_still_requires_consecutive_points():
    data = _endpoint_table([
        ("E1", time, "S1", "S1_1", 1, "souche", od, 1_000)
        for time, od in ((0, .2), (1, .01), (2, .2))
    ])

    result = run_normalization(data, minimum_od=.05, consecutive_points=3)

    assert result.normalized_data["Lum_norm"].isna().all()
    assert not result.series_validation.iloc[0]["series_valid"]
    assert result.series_validation.iloc[0]["reason"] == "no_consecutive_points_above_threshold"


def test_no_blank_falls_back_to_minimum_and_warns():
    data = corrected_table().query("type == 'souche'")
    result = run_normalization(data)
    assert result.threshold_details.iloc[0]["effective_threshold"] == pytest.approx(0.05)
    assert set(result.warnings["code"]) == {"no_valid_corrected_blanks"}


def test_experiments_have_independent_thresholds_and_series():
    first = corrected_table("E1")
    second = corrected_table("E2")
    second.loc[second["type"].eq("blanc"), "DO_corr"] += 0.2
    data = pd.concat([first, second], ignore_index=True)
    result = run_normalization(data, blank_sd_multiplier=0)
    thresholds = result.threshold_details.set_index("experience_id")["effective_threshold"]
    assert thresholds["E1"] == pytest.approx(0.05)
    assert thresholds["E2"] == pytest.approx(0.2)
    assert len(result.series_validation) == 2


def test_rejected_rows_keep_reason_and_input_is_unchanged():
    data = corrected_table()
    original = data.copy(deep=True)
    result = run_normalization(data)
    assert not result.rejected_rows.empty
    assert result.rejected_rows["normalization_reason"].ne("").all()
    pdt.assert_frame_equal(data, original)


def test_duplicate_time_breaks_consecutive_run_and_validation_returns_copy():
    series = pd.DataFrame({"temps_h": [0, 1, 1, 2, 3, 4], "DO_corr": [1] * 6})
    assert find_first_consecutive_valid_time(series, 0.05, 3) == 2
    data = corrected_table()
    prepared = validate_normalization_inputs(data)
    assert prepared is not data


def test_blank_correction_output_integrates_with_normalization():
    rows = []
    for kind, strain, header, ods in [
        ("blanc", "B", "B1", [0.10] * 4),
        ("blanc", "B", "B2", [0.12] * 4),
        ("souche", "S", "S1", [0.20, 0.25, 0.30, 0.35]),
    ]:
        for time, od in enumerate(ods):
            rows.append({"temps_h": time, "souche": strain, "Groupe": "G", "replicat": 1,
                         "sample_header": header, "DO_brute": od, "Lum_brute": od * 1000, "type": kind})
    correction = run_blank_correction(pd.DataFrame(rows))
    assert correction.corrected_data["type"].eq("blanc").any()
    normalized = run_normalization(correction.corrected_data)
    assert normalized.threshold_details.iloc[0]["n_valid_blank_points"] == 8
