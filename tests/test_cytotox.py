"""Regression checks for the agreed Cytotox workbook layout."""

from io import BytesIO

import numpy as np
from openpyxl import Workbook
import pandas as pd
import pytest

from luxplate.cytotox import (
    CytotoxSignal, auc_by_well, common_auc_window, normalize_ip, read_cytotox_workbook, relative_auc_by_well,
)
from luxplate.cytotox_ui import _signal_figure, _wells_by_strain


@pytest.fixture
def workbook():
    workbook = Workbook()
    gfp = workbook.active
    gfp.title = "GFP"
    gfp.append(["Metric: Confluence (%)"])
    gfp.append(["Date Time", "Elapsed", "A1", "A2"])
    gfp.append([None, 0, 10, 20])
    gfp.append([None, 0.5, 12, 22])
    gfp.append([None, "Elapsed", "A1", "A2"])
    gfp.append([None, 0, 0, 0])
    workbook.create_sheet("IP").append(["Elapsed", "A1", "A2"])
    workbook["IP"].append([0, 21, 42])
    final_gfp = workbook.create_sheet("GFP pour R ")
    final_gfp.append(["Elapsed", "A1", "A2"])
    final_gfp.append([0, 0, 0])
    final_gfp.append([0.5, -0.01, 0.02])
    plate = workbook.create_sheet("Plaque")
    plate.append([None, *range(1, 13)])
    for row in "ABCDEFGH":
        plate.append([row])
    plate["B2"] = " Souche Δ "
    plate["C2"] = "Souche Δ"
    plate["F9"] = "None"
    return workbook


def _read(workbook):
    buffer = BytesIO()
    workbook.save(buffer)
    return read_cytotox_workbook(buffer.getvalue())


def test_reads_first_raw_gfp_block_and_final_values_from_their_own_sheet(workbook):
    result = _read(workbook)
    gfp, ip, final_gfp = result.signals
    assert gfp.metric == "Confluence (%)"
    assert gfp.data.index.tolist() == [0, 0.5]
    assert gfp.data["A1"].tolist() == [10, 12]
    assert ip.data.loc[0, "A2"] == 42
    assert final_gfp.data.loc[0.5, "A1"] == -0.01
    assert result.plate.loc["A", [1, 2]].tolist() == ["Souche Δ", "Souche Δ"]
    assert result.plate.loc["H", 5] == "None"
    assert result.plate.shape == (8, 12)


def test_rejects_duplicate_wells_before_label_mapping(workbook):
    workbook["GFP"]["D2"] = "a01"
    with pytest.raises(ValueError, match="colonnes de temps ou de puits en double"):
        _read(workbook)


def test_raw_panels_show_each_well_with_shared_replicate_colors_and_minutes(workbook):
    workbook["Plaque"]["B3"] = "Autre souche"
    workbook["GFP"]["E2"] = "B1"
    workbook["GFP"]["E3"] = 30
    workbook["GFP"]["E4"] = 32
    result = _read(workbook)
    groups = _wells_by_strain(result.plate)
    selected = {strain: wells for strain, wells in groups.items() if strain != "None"}
    figure = _signal_figure(result.signals[0], selected, "Essai — GFP raw")
    first, second = figure.axes
    assert [axis.get_title() for axis in figure.axes] == ["Souche Δ", "Autre souche"]
    assert [line.get_label() for line in first.lines] == ["Rep 1", "Rep 2"]
    assert first.lines[0].get_color() == second.lines[0].get_color()
    np.testing.assert_array_equal(first.lines[1].get_xdata(), [0, 30])
    np.testing.assert_array_equal(first.lines[1].get_ydata(), [20, 22])
    assert first.get_shared_y_axes().joined(first, second)


def test_ip_normalization_aligns_wells_and_keeps_the_cutoff_inclusive(workbook):
    workbook["GFP"]["D3"] = 0
    workbook["IP"].append([0.5, 31, 50])
    workbook["IP"].append([1, 41, 55])
    workbook["IP"].append([1.5, 51, 60])
    gfp, ip, final_gfp = _read(workbook).signals
    normalized = normalize_ip(gfp.data, ip.data[["A2", "A1"]], 60)
    assert normalized.index.tolist() == [0, 0.5, 1]
    assert normalized.columns.tolist() == ["A2", "A1"]
    np.testing.assert_array_equal(normalized["A1"], [0, 1, 2])
    assert normalized["A2"].isna().all()
    assert ip.data.loc[1.5, "A1"] == 51
    assert final_gfp.data.loc[0.5, "A1"] == -0.01


def test_ip_normalization_requires_time_zero_in_both_sheets(workbook):
    gfp, ip, _ = _read(workbook).signals
    with pytest.raises(ValueError, match="t = 0"):
        normalize_ip(gfp.data.drop(index=0), ip.data, 1200)


def test_auc_uses_retained_points_including_when_endpoints_are_excluded():
    ip = pd.DataFrame(
        {"A1": [0, 3, 4], "A2": [0, np.nan, 4], "A3": [0, 3, np.nan], "A4": [np.nan, 3, 4]},
        index=[0, 0.5, 1],
    )
    auc = auc_by_well(ip)
    np.testing.assert_allclose(auc, [150, 120, 45, 105])
    assert auc_by_well(ip.iloc[[0]]).isna().all()


def test_intermanip_auc_uses_the_common_window_and_each_workbook_reference():
    first = pd.DataFrame(
        {"A1": [0, 0.5, 1, 1.5], "A2": [0, 1.5, 3, 4.5], "B1": [0, 2, 4, 6]},
        index=[0, .5, 1, 1.5],
    )
    second = pd.DataFrame(
        {"A1": [0, 1.5, 2.5], "A2": [0, 4.5, 7.5], "B1": [0, 9, 15]},
        index=[0, .75, 1.25],
    )
    window = common_auc_window([first, second])
    assert window == (0, 75)
    np.testing.assert_allclose(relative_auc_by_well(first, ["A1", "A2"], window), [.5, 1.5, 2])
    np.testing.assert_allclose(relative_auc_by_well(second, ["A1", "A2"], window), [.5, 1.5, 3])

    second.loc[1.25, "B1"] = np.nan
    first.loc[0, "B1"] = np.nan
    window = common_auc_window([first, second])
    assert window == (30, 45)
    np.testing.assert_allclose(relative_auc_by_well(first, ["A1", "A2"], window), [.5, 1.5, 2])
    np.testing.assert_allclose(relative_auc_by_well(second, ["A1", "A2"], window), [.5, 1.5, 3])
    assert relative_auc_by_well(second, ["H12"], window) is None
    assert relative_auc_by_well(first * 0, ["A1", "A2"], window) is None
    assert common_auc_window([first, second.set_axis([2, 3, 4])]) is None


def test_normalized_curve_keeps_individual_wells_and_exclusion_gaps():
    data = pd.DataFrame({"A1": [2, np.nan, 3], "A2": [4, 5, np.nan]}, index=[0, .5, 1])
    signal = CytotoxSignal("GFP normalisé", "GFP pour R", "GFP", data)
    figure = _signal_figure(signal, {"Souche": ["A1", "A2"]}, "Courbe finale")
    axis = figure.axes[0]
    assert [line.get_label() for line in axis.lines] == ["Rep 1", "Rep 2"]
    np.testing.assert_allclose(axis.lines[0].get_ydata(), [2, np.nan, 3])
    np.testing.assert_allclose(axis.lines[1].get_ydata(), [4, 5, np.nan])
    assert not axis.collections
