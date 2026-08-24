import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from luxplate.blanks import run_blank_correction
from luxplate.crosstalk import (
    MODEL_RESOURCE,
    PLATE_WELLS,
    correct_plate_crosstalk,
    load_crosstalk_model,
)


def plate(signal, *, time=0.0, experience="exp-1"):
    model = load_crosstalk_model()
    raw = model.Dbest @ np.asarray(signal, dtype=float) + model.background_rlu
    return pd.DataFrame({
        "experience": experience, "temps_h": time, "puits": PLATE_WELLS,
        "Lum_brute": raw, "DO_brute": 1.0, "Groupe": "medium",
        "type": ["blanc"] * 2 + ["souche"] * 94,
        "souche": ["blank"] * 2 + ["strain"] * 94,
        "replicat": range(96), "sample_header": [f"sample-{i}" for i in range(96)],
    })


def corrected_in_plate_order(result):
    return result.set_index("puits").loc[list(PLATE_WELLS), "RLU_corrected"].to_numpy()


def test_artifacts_load_as_official_kernel():
    assert MODEL_RESOURCE.joinpath("kernel_D_best.npy").is_file()
    assert not MODEL_RESOURCE.joinpath("kernel_D.npy").is_file()
    model = load_crosstalk_model()
    assert model.Dbest.shape == (96, 96)
    assert model.metadata["kernel_id"] == "MAURI_E06_BEST"
    assert np.isfinite(model.condition_number)


def test_canonical_well_order():
    assert PLATE_WELLS == tuple(f"{row}{col:02d}" for row in "ABCDEFGH" for col in range(1, 13))


def test_synthetic_mathematical_identity_and_output_traceability():
    truth = np.linspace(1.0, 100_000.0, 96)
    raw = plate(truth).sample(frac=1, random_state=42)
    result = correct_plate_crosstalk(raw)
    np.testing.assert_allclose(corrected_in_plate_order(result), truth, rtol=2e-12, atol=2e-10)
    assert np.array_equal(result["RLU_raw"], result["Lum_brute"])
    assert np.array_equal(result["Lum_analysis"], result["RLU_corrected"])
    assert result["crosstalk_method"].eq("MAURI_DBEST").all()
    assert result["crosstalk_kernel_id"].eq("MAURI_E06_BEST").all()
    assert result["max_abs_reconstruction_residual"].max() < 1e-8


def test_multiple_simultaneous_sources():
    truth = np.zeros(96)
    truth[[0, 17, 53, 70, 95]] = [1, 1e2, 1e6, 3.5, 8e4]
    np.testing.assert_allclose(
        corrected_in_plate_order(correct_plate_crosstalk(plate(truth))),
        truth,
        atol=1e-9,
    )


def test_times_and_experiments_are_solved_independently():
    signals = [np.arange(96), np.arange(96) * 10, np.arange(96)[::-1] * 3]
    data = pd.concat([plate(signals[0], time=0, experience="a"),
                      plate(signals[1], time=1, experience="a"),
                      plate(signals[2], time=0, experience="b")], ignore_index=True)
    result = correct_plate_crosstalk(data)
    for (experience, time), truth in zip([("a", 0), ("a", 1), ("b", 0)], signals):
        group = result.query("experience == @experience and temps_h == @time")
        np.testing.assert_allclose(corrected_in_plate_order(group), truth, atol=2e-12)


def test_source_workbooks_are_solved_as_independent_plates_at_same_time():
    first_truth = np.arange(48, dtype=float) + 10
    second_truth = np.arange(48, dtype=float) + 100
    model = load_crosstalk_model()

    def half_plate(workbook, wells, truth):
        indices = [PLATE_WELLS.index(well) for well in wells]
        raw = model.Dbest[np.ix_(indices, indices)] @ truth + model.background_rlu
        return pd.DataFrame({
            "experience": "shared-run",
            "source_workbook": workbook,
            "temps_h": 1.0,
            "puits": wells,
            "Lum_brute": raw,
        })

    # Both workbooks use the same physical wells and real time.  Combining
    # them before deconvolution would either duplicate wells or manufacture a
    # synthetic plate; the workbook boundary must therefore win.
    wells = PLATE_WELLS[:48]
    data = pd.concat([
        half_plate("run_t1.xlsx", wells, first_truth),
        half_plate("run_t2.xlsx", wells, second_truth),
    ], ignore_index=True)

    result = correct_plate_crosstalk(data)

    for workbook, truth in (("run_t1.xlsx", first_truth), ("run_t2.xlsx", second_truth)):
        corrected = (result.query("source_workbook == @workbook")
                     .set_index("puits").loc[list(wells), "RLU_corrected"])
        np.testing.assert_allclose(corrected, truth, rtol=2e-12, atol=2e-10)


def test_input_is_not_mutated():
    data = plate(np.ones(96))
    original = data.copy(deep=True)
    correct_plate_crosstalk(data)
    pdt.assert_frame_equal(data, original)


def test_negative_results_are_constrained_to_zero():
    truth = np.ones(96)
    truth[20] = -123.456
    result = correct_plate_crosstalk(plate(truth))
    corrected = corrected_in_plate_order(result)
    assert np.all(corrected >= 0)
    assert corrected[20] == pytest.approx(0, abs=1e-8)


def test_unmeasured_well_is_implicitly_non_luminescent():
    truth = np.linspace(1.0, 1_000.0, 96)
    truth[-1] = 0.0
    partial = plate(truth).iloc[:-1]

    result = correct_plate_crosstalk(partial)

    expected = pd.Series(truth, index=PLATE_WELLS).drop(index="H12")
    actual = result.set_index("puits")["RLU_corrected"].loc[expected.index]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)


def test_partial_plate_with_known_water_wells_uses_reduced_kernel():
    truth = np.linspace(10.0, 1_000.0, 96)
    water = {"A02": "water", "H12": "eau"}
    truth[[1, 95]] = 0.0
    partial = plate(truth).query("puits not in @water").sample(frac=1, random_state=7)

    result = correct_plate_crosstalk(partial, unmeasured_well_statuses=water)

    expected = pd.Series(truth, index=PLATE_WELLS).drop(index=list(water))
    actual = result.set_index("puits")["RLU_corrected"].loc[expected.index]
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-10)
    assert set(result["puits"]).isdisjoint(water)


def test_dispersed_non_luminescent_wells_preserve_canonical_matrix_order():
    water_wells = ["A01", "B07", "D04", "F11", "H03"]
    measured = [well for well in PLATE_WELLS if well not in water_wells]
    measured_indices = [PLATE_WELLS.index(well) for well in measured]
    truth = np.linspace(50.0, 50_000.0, len(measured))
    model = load_crosstalk_model()
    optical = model.Dbest[np.ix_(measured_indices, measured_indices)] @ truth
    data = pd.DataFrame({
        "temps_h": 0.0,
        "puits": measured,
        "Lum_brute": optical + model.background_rlu,
    }).sample(frac=1, random_state=11)

    result = correct_plate_crosstalk(
        data,
        unmeasured_well_statuses={well: "non-luminescent" for well in water_wells},
    )

    actual = result.set_index("puits").loc[measured, "RLU_corrected"]
    np.testing.assert_allclose(actual, truth, rtol=2e-12, atol=2e-10)


def test_invalid_absent_well_status_is_rejected():
    with pytest.raises(ValueError, match="Statut invalide.*H12"):
        correct_plate_crosstalk(
            plate(np.ones(96)).iloc[:-1],
            unmeasured_well_statuses={"H12": "unknown"},
        )


def test_missing_or_non_numeric_luminescence_is_rejected():
    data = plate(np.ones(96))
    data.loc[3, "Lum_brute"] = np.nan
    with pytest.raises(ValueError, match="Lum_brute manquante ou non numérique"):
        correct_plate_crosstalk(data)


def test_duplicate_well_is_rejected():
    data = plate(np.ones(96))
    data.loc[95, "puits"] = "A01"
    with pytest.raises(ValueError, match="dupliqués"):
        correct_plate_crosstalk(data)


def test_blank_correction_uses_dbest_lum_analysis_downstream():
    truth = np.arange(96, dtype=float) + 100
    corrected = correct_plate_crosstalk(plate(truth))
    result = run_blank_correction(corrected)
    # Two blank wells contain 100 and 101 RLU; their experimental mean is 100.5.
    strain = result.corrected_data.query("puits == 'A03'").iloc[0]
    assert strain["Lum_corr"] == pytest.approx(102 - 100.5)
    blanks = result.corrected_data.query("type == 'blanc'").sort_values("puits")
    assert blanks["Lum_corr"].tolist() == pytest.approx([100, 101])
    assert blanks["Lum_blank_residual"].tolist() == pytest.approx([-0.5, 0.5])
