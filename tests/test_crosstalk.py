import numpy as np
import pandas as pd
import pytest

from luxplate.crosstalk import (CROSSTALK_COEFFICIENTS, INSTRUMENT_BACKGROUND_RLU,
                                PLATE_WELLS, correct_plate_crosstalk)


def plate(source_well="B06", source_value=1_000_000.0):
    return pd.DataFrame({
        "puits": PLATE_WELLS,
        "temps_h": 0.0,
        "Lum_brute": [source_value if well == source_well else INSTRUMENT_BACKGROUND_RLU
                      for well in PLATE_WELLS],
    })


def test_single_source_uses_source_relative_direction_and_only_adjacent_wells():
    raw = plate()
    corrected = correct_plate_crosstalk(raw)
    light = 1_000_000.0 - INSTRUMENT_BACKGROUND_RLU
    predicted = corrected.set_index("puits")["CrossTalk_predicted"]
    assert predicted["A06"] == pytest.approx(CROSSTALK_COEFFICIENTS["S"] * light)
    assert predicted["C06"] == pytest.approx(CROSSTALK_COEFFICIENTS["N"] * light)
    assert predicted["B05"] == pytest.approx(CROSSTALK_COEFFICIENTS["E"] * light)
    assert predicted["B07"] == pytest.approx(CROSSTALK_COEFFICIENTS["O"] * light)
    assert predicted["A05"] == pytest.approx(CROSSTALK_COEFFICIENTS["SE"] * light)
    assert predicted["A07"] == pytest.approx(CROSSTALK_COEFFICIENTS["SO"] * light)
    assert predicted["C05"] == pytest.approx(CROSSTALK_COEFFICIENTS["NE"] * light)
    assert predicted["C07"] == pytest.approx(CROSSTALK_COEFFICIENTS["NO"] * light)
    assert np.count_nonzero(predicted.to_numpy()) == 8


def test_correction_is_non_mutating_keeps_negative_values_and_separates_times():
    raw = pd.concat([plate(source_value=24.0), plate(source_value=124.0).assign(temps_h=1.0)], ignore_index=True)
    snapshot = raw.copy(deep=True)
    corrected = correct_plate_crosstalk(raw)
    pd.testing.assert_frame_equal(raw, snapshot)
    assert "RLU_corrected" not in raw
    assert corrected.loc[corrected["temps_h"].eq(0), "CrossTalk_predicted"].eq(0).all()
    assert corrected.loc[(corrected["temps_h"].eq(1)) & (corrected["puits"].eq("A06")),
                         "RLU_corrected"].iloc[0] < 0


def test_missing_wells_and_values_are_zero_signal_but_duplicates_are_rejected():
    partial = plate().loc[lambda frame: ~frame["puits"].isin(["A01", "A02"])]
    partial.loc[partial["puits"].eq("H12"), "Lum_brute"] = np.nan
    corrected = correct_plate_crosstalk(partial)

    assert len(corrected) == len(partial)
    assert corrected["CrossTalk_predicted"].notna().all()
    # A missing neighbour contributes no signal, rather than -24 RLU after
    # subtraction of the instrumental background.
    assert corrected.loc[corrected["puits"].eq("A03"), "CrossTalk_predicted"].iloc[0] == 0
    assert corrected.loc[corrected["puits"].eq("H12"), "RLU_corrected"].iloc[0] == 0

    duplicated = pd.concat([plate(), plate().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="doublons=A01"):
        correct_plate_crosstalk(duplicated)
