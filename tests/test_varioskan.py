from io import BytesIO

import numpy as np
import pytest
from openpyxl import Workbook, load_workbook

from luxplate.varioskan import (
    combine_kinetic_tables,
    inspect_workbook,
    normalize_strain_name,
    parse_kinetic_workbook,
)


def synthetic_workbook(*, second_luminescence=False) -> BytesIO:
    workbook = Workbook()
    workbook.remove(workbook.active)
    headers = ["Lecture en cours", "temps moy. [s]", "PAO1 (A01)", "PAO1 (A02)", "Blanc (B01)"]
    for name, scale in [("Absorbance 600", 0.01), ("Luminescence 1", 100.0)]:
        sheet = workbook.create_sheet(name)
        sheet.append(["export synthétique"])
        sheet.append(headers)
        sheet.append([1, 0, scale, scale * 2, scale / 2])
        sheet.append([2, 3600, scale * 3, scale * 4, scale / 2])
    if second_luminescence:
        sheet = workbook.create_sheet("Luminescence 2")
        sheet.append(headers)
        sheet.append([1, 0, 10, 20, 5])
        sheet.append([2, 3600, 30, 40, 5])
    plan = workbook.create_sheet("Plan de plaque")
    plan.append([])
    plan.append([])
    plan.append([])
    plan.append([None, *range(1, 13)])
    plan.append(["A", "PAO1", "PAO1"])
    plan.append([None, "Groupe 1", "Groupe 1"])
    plan.append(["B", "Blanc"])
    plan.append([None, "Groupe 1"])
    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output


def workbook_with_strain_headers(headers: list[str]) -> BytesIO:
    """Build one plate whose headers are distinct technical wells."""
    if len(headers) > 96:
        raise ValueError("Une plaque synthétique ne peut pas dépasser 96 puits.")
    source = synthetic_workbook()
    workbook = load_workbook(source)
    wells = [f"{chr(ord('A') + index // 12)}{index % 12 + 1:02d}" for index in range(len(headers))]
    sample_headers = [f"{header} ({well})" for header, well in zip(headers, wells)]
    for sheet_name in ("Absorbance 600", "Luminescence 1"):
        sheet = workbook[sheet_name]
        sheet.delete_cols(3, sheet.max_column - 2)
        for column, header in enumerate(sample_headers, 3):
            sheet.cell(2, column).value = header
            sheet.cell(3, column).value = column
            sheet.cell(4, column).value = column * 2
    plan = workbook["Plan de plaque"]
    plan.delete_rows(5, max(plan.max_row - 4, 0))
    for row_index in range((len(headers) + 11) // 12):
        sample_row = 5 + row_index * 2
        plan.cell(sample_row, 1).value = chr(ord("A") + row_index)
        for column in range(2, min(14, len(headers) - row_index * 12 + 2)):
            plan.cell(sample_row, column).value = headers[row_index * 12 + column - 2]
            plan.cell(sample_row + 1, column).value = "Groupe 1"
    output = BytesIO()
    workbook.save(output)
    output.seek(0)
    return output


def test_parse_kinetic_workbook_preserves_raw_technical_wells():
    result = parse_kinetic_workbook(synthetic_workbook())
    assert len(result) == 6
    assert result["sample_header"].nunique() == 3
    assert result.loc[result["souche"].eq("PAO1"), "replicat"].unique().tolist() == [1, 2]
    assert result.loc[result["souche"].eq("PAO1"), "Groupe"].eq("Groupe 1").all()
    np.testing.assert_allclose(sorted(result["temps_h"].unique()), [0, 1])
    assert result.loc[result["type"].eq("blanc"), "souche"].eq("Blanc").all()


def test_multiple_luminescence_sheets_require_explicit_selection():
    source = synthetic_workbook(second_luminescence=True)
    absorbance, luminescence = inspect_workbook(source)
    assert absorbance == "Absorbance 600"
    assert luminescence == ["Luminescence 1", "Luminescence 2"]
    source.seek(0)
    with pytest.raises(ValueError, match="choisissez-en une explicitement"):
        parse_kinetic_workbook(source)
    source.seek(0)
    selected = parse_kinetic_workbook(source, "Luminescence 2")
    assert selected["Lum_brute"].max() == 40


def test_medium_prefix_is_separated_and_numbered_blank_is_associated():
    source = synthetic_workbook()
    workbook = load_workbook(source)
    renamed = {
        "PAO1 (A01)": "SCFM2 (Po) 14.1Ac attB::P0-lux (A01)",
        "PAO1 (A02)": "SCFM2 (Po) 14.1Ac attB::P0-lux (A02)",
        "Blanc (B01)": "Blanc1 (B01)",
    }
    for sheet_name in ("Absorbance 600", "Luminescence 1"):
        sheet = workbook[sheet_name]
        for cell in sheet[2]:
            if cell.value in renamed:
                cell.value = renamed[cell.value]
    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    result = parse_kinetic_workbook(output)

    strains = result.loc[result["type"].eq("souche")]
    blanks = result.loc[result["type"].eq("blanc")]
    assert strains["souche"].unique().tolist() == ["14.1Ac attB::P0-lux"]
    assert strains["Groupe"].unique().tolist() == ["SCFM2 (Po)"]
    assert blanks["Groupe"].unique().tolist() == ["SCFM2 (Po)"]


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("14.1Ac attB::MiniCTXlux(PspeD2-1A-lux) (A01)", "14.1Ac attB::PspeD2-1A-lux"),
        ("14.1Ac attB::PspeD2-1A-Lux (A01)", "14.1Ac attB::PspeD2-1A-lux"),
        ("14.1Ac attB::PspeD2-1A-lux (A01)", "14.1Ac attB::PspeD2-1A-lux"),
        ("14.1Ac attB::MiniCTXlux(P0-lux) (A01)", "14.1Ac attB::P0-lux"),
    ],
)
def test_equivalent_lux_reporter_spellings_share_a_canonical_strain(header, expected):
    source = synthetic_workbook()
    workbook = load_workbook(source)
    for sheet_name in ("Absorbance 600", "Luminescence 1"):
        sheet = workbook[sheet_name]
        sheet.cell(2, 3).value = header
    output = BytesIO()
    workbook.save(output)
    output.seek(0)

    result = parse_kinetic_workbook(output)

    assert result.loc[result["puits"].eq("A01"), "souche"].unique().tolist() == [expected]


def test_varioskan_0001_suffix_is_removed_without_losing_biological_qualifiers():
    assert normalize_strain_name("14.1Ac attB::P0-lux0001") == normalize_strain_name(
        "14.1Ac attB::P0-Lux"
    )
    assert normalize_strain_name("14.1Ac attB::PspeD2-1A-Lux") == normalize_strain_name(
        "14.1Ac attB::PspeD2-1A-lux"
    )
    assert normalize_strain_name("14.1Ac attB::PspeD2-3B-Lux") == normalize_strain_name(
        "14.1Ac attB::PspeD2-3B-lux"
    )
    assert normalize_strain_name("14.1Ac attB::P0-lux (M1)0001") == (
        "14.1Ac attB::P0-lux (M1)"
    )
    assert normalize_strain_name("14.1Ac attB::P0-lux (M1)0001") != normalize_strain_name(
        "14.1Ac attB::P0-lux (M2)0001"
    )


def test_lone_varioskan_0001_header_is_canonicalized_during_import():
    headers = ["LB 14.1Ac attB::P0-lux0001"] * 3

    strains = parse_kinetic_workbook(workbook_with_strain_headers(headers)).query("type == 'souche'")

    assert strains["souche"].unique().tolist() == ["14.1Ac attB::P0-lux"]
    assert strains["Groupe"].unique().tolist() == ["LB"]
    assert strains["replicat"].unique().tolist() == [1, 2, 3]
    assert strains["puits"].nunique() == 3


def test_morphotypes_remain_distinct_during_import():
    headers = [
        "SCFM2-KPI 14.1Ac attB::P0-lux (M1)0001",
        "SCFM2-KPI 14.1Ac attB::P0-lux (M2)0001",
    ]

    strains = parse_kinetic_workbook(workbook_with_strain_headers(headers)).query("type == 'souche'")

    assert strains["souche"].unique().tolist() == [
        "14.1Ac attB::P0-lux (M1)",
        "14.1Ac attB::P0-lux (M2)",
    ]
    assert strains["Groupe"].unique().tolist() == ["SCFM2-KPI"]


@pytest.mark.parametrize(
    "headers",
    [
        ["P0-lux", "P0-lux0002", "P0-lux0003"],
        ["PspeD2-1A-lux", "PspeD2-1A-lux0002"],
        ["PspeD2-3B-lux", "PspeD2-3B-lux0025"],
        ["P0-lux", "P0-lux2", "P0-lux_2", "P0-lux-2", "P0-lux_rep2", "P0-lux-rep2"],
    ],
)
def test_source_generated_technical_suffixes_share_one_strain(headers):
    result = parse_kinetic_workbook(workbook_with_strain_headers(headers))

    strains = result.loc[result["type"].eq("souche")]
    assert strains["souche"].unique().tolist() == [headers[0]]
    assert strains["replicat"].unique().tolist() == list(range(1, len(headers) + 1))
    assert strains["sample_header"].nunique() == len(headers)
    assert strains["puits"].nunique() == len(headers)


@pytest.mark.parametrize("promoter", ["PspeD2-1A", "PspeD2-3B"])
def test_numbered_lux_series_is_collapsed_without_unsuffixed_header(promoter):
    headers = [f"{promoter}-lux{number:04d}" for number in (1, 2, 3, 25)]

    strains = parse_kinetic_workbook(workbook_with_strain_headers(headers)).query("type == 'souche'")

    assert strains["souche"].unique().tolist() == [f"{promoter}-lux"]
    assert strains["replicat"].unique().tolist() == [1, 2, 3, 4]
    assert strains["sample_header"].nunique() == 4
    assert strains["puits"].nunique() == 4


@pytest.mark.parametrize("promoter", ["PspeD2-1A", "PspeD2-3B"])
def test_complete_numbered_lux_series_has_25_replicates(promoter):
    headers = [f"{promoter}-lux{number:04d}" for number in range(1, 26)]

    strains = parse_kinetic_workbook(workbook_with_strain_headers(headers)).query("type == 'souche'")

    assert strains["souche"].unique().tolist() == [f"{promoter}-lux"]
    assert strains["replicat"].unique().tolist() == list(range(1, 26))


def test_ma_complex_reference_structure_has_exactly_three_strains():
    strain_prefix = "14.1Ac attb::"
    headers = (
        [f"{strain_prefix}P0-lux"] * 22
        + [f"{strain_prefix}PspeD2-1A-lux{number:04d}" for number in range(1, 26)]
        + [f"{strain_prefix}PspeD2-3B-lux{number:04d}" for number in range(1, 26)]
    )

    strains = parse_kinetic_workbook(workbook_with_strain_headers(headers)).query("type == 'souche'")

    assert strains["souche"].unique().tolist() == [
        f"{strain_prefix}P0-lux",
        f"{strain_prefix}PspeD2-1A-lux",
        f"{strain_prefix}PspeD2-3B-lux",
    ]
    assert strains.groupby("souche")["sample_header"].nunique().to_dict() == {
        f"{strain_prefix}P0-lux": 22,
        f"{strain_prefix}PspeD2-1A-lux": 25,
        f"{strain_prefix}PspeD2-3B-lux": 25,
    }


@pytest.mark.parametrize("natural_name", ["PspeD2", "14.1Ac", "14.3B", "PA14", "3-B", "1-A"])
def test_natural_terminal_digits_are_not_stripped_without_cohort_evidence(natural_name):
    result = parse_kinetic_workbook(workbook_with_strain_headers([natural_name]))

    assert result.loc[result["type"].eq("souche"), "souche"].unique().tolist() == [natural_name]


def test_combining_workbooks_namespaces_internal_blank_links():
    first = parse_kinetic_workbook(synthetic_workbook())
    second = parse_kinetic_workbook(synthetic_workbook())

    combined = combine_kinetic_tables([("jour1.xlsx", first), ("jour2.xlsx", second)])

    assert set(combined["experience"]) == {"jour1", "jour2"}
    assert combined["Groupe"].nunique() == 2
    assert combined["sample_header"].nunique() == first["sample_header"].nunique() * 2
