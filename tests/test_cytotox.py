"""Regression checks for the agreed Cytotox workbook layout."""

from io import BytesIO

import numpy as np
from openpyxl import Workbook
import pytest

from luxplate.cytotox import read_cytotox_workbook
from luxplate.cytotox_ui import _raw_figure, _wells_by_strain


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
    figure = _raw_figure(result.signals[0], selected, "Essai — GFP raw")
    first, second = figure.axes
    assert [axis.get_title() for axis in figure.axes] == ["Souche Δ", "Autre souche"]
    assert [line.get_label() for line in first.lines] == ["Rep 1", "Rep 2"]
    assert first.lines[0].get_color() == second.lines[0].get_color()
    np.testing.assert_array_equal(first.lines[1].get_xdata(), [0, 30])
    np.testing.assert_array_equal(first.lines[1].get_ydata(), [20, 22])
    assert first.get_shared_y_axes().joined(first, second)
