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


def test_incomplete_or_duplicate_plate_is_rejected():
    with pytest.raises(ValueError, match="96 puits uniques"):
        correct_plate_crosstalk(plate().iloc[:-1])
    duplicated = pd.concat([plate(), plate().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="doublons=A01"):
        correct_plate_crosstalk(duplicated)


def test_lecture_separates_acquisitions_with_the_same_rounded_time():
    first = plate(source_value=24.0).assign(experience="exp", lecture=1)
    second = plate(source_value=124.0).assign(experience="exp", lecture=2)
    raw = pd.concat([first, second], ignore_index=True)

    corrected = correct_plate_crosstalk(raw)

    assert corrected.loc[corrected["lecture"].eq(1), "CrossTalk_predicted"].eq(0).all()
    predicted_a06 = corrected.loc[
        corrected["lecture"].eq(2) & corrected["puits"].eq("A06"),
        "CrossTalk_predicted",
    ].iloc[0]
    assert predicted_a06 == pytest.approx(
        CROSSTALK_COEFFICIENTS["S"] * (124.0 - INSTRUMENT_BACKGROUND_RLU)
    )


@pytest.mark.parametrize("invalid_lecture", [None, "", "not-a-number", np.inf])
def test_invalid_lecture_is_rejected_before_plate_grouping(invalid_lecture):
    raw = plate().assign(lecture=pd.Series([1] * len(PLATE_WELLS), dtype=object))
    raw.loc[0, "lecture"] = invalid_lecture

    with pytest.raises(ValueError, match="'lecture'.*manquantes ou invalides"):
        correct_plate_crosstalk(raw)


def test_directional_details_do_not_depend_on_unique_coefficient_values(monkeypatch):
    import luxplate.crosstalk as crosstalk

    coefficients = dict(CROSSTALK_COEFFICIENTS)
    coefficients["S"] = coefficients["N"]
    monkeypatch.setattr(crosstalk, "CROSSTALK_COEFFICIENTS", coefficients)
    monkeypatch.setattr(crosstalk, "CROSSTALK_MATRIX", crosstalk.build_crosstalk_matrix())

    corrected = crosstalk.correct_plate_crosstalk(plate()).set_index("puits")

    light = 1_000_000.0 - INSTRUMENT_BACKGROUND_RLU
    assert corrected.loc["A06", "CT_S"] == pytest.approx(coefficients["S"] * light)
    assert corrected.loc["A06", "CT_N"] == 0
    assert corrected.loc["C06", "CT_N"] == pytest.approx(coefficients["N"] * light)
    assert corrected.loc["C06", "CT_S"] == 0
