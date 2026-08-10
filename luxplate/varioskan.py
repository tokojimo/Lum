"""Import Varioskan kinetic workbooks into the historical long-table schema.

This module is the application-facing integration of
``examples/synthetic/01_mise_en_forme_donnees.py``.  It deliberately keeps the
raw values and the French column names used by the subsequent legacy stages.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Iterable

import pandas as pd
from openpyxl import load_workbook

HEADER_READING = "Lecture en cours"
HEADER_TIME = "temps moy. [s]"
PLATE_ROWS = set("ABCDEFGH")


def clean_text(value: object) -> str:
    """Normalize spreadsheet text without changing meaningful labels."""
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def find_measurement_sheets(sheet_names: Iterable[str]) -> tuple[str, list[str]]:
    """Return the first absorbance sheet and every luminescence candidate."""
    names = list(sheet_names)
    absorbance = [n for n in names if clean_text(n).lower().startswith("absorbance")]
    luminescence = [n for n in names if clean_text(n).lower().startswith("luminescence")]
    if not absorbance:
        raise ValueError("Aucune feuille d'absorbance détectée dans le fichier.")
    if not luminescence:
        raise ValueError("Aucune feuille de luminescence détectée dans le fichier.")
    return absorbance[0], luminescence


def inspect_workbook(source: str | Path | BinaryIO) -> tuple[str, list[str]]:
    """Inspect a workbook so the UI can require an explicit Lum sheet choice."""
    workbook = load_workbook(source, read_only=True, data_only=True)
    try:
        return find_measurement_sheets(workbook.sheetnames)
    finally:
        workbook.close()


def _header_row(sheet) -> int:
    for row in range(1, min(sheet.max_row, 30) + 1):
        values = [clean_text(sheet.cell(row, col).value) for col in range(1, min(sheet.max_column, 10) + 1)]
        if HEADER_READING in values and HEADER_TIME in values:
            return row
    raise ValueError(f"Impossible de trouver la ligne d'entête dans la feuille {sheet.title!r}.")


def _wide_table(sheet) -> tuple[pd.DataFrame, list[str]]:
    header_row = _header_row(sheet)
    headers = [clean_text(sheet.cell(header_row, col).value) for col in range(1, sheet.max_column + 1)]
    useful = [index for index, name in enumerate(headers, start=1) if name]
    rows = []
    for row in range(header_row + 1, sheet.max_row + 1):
        values = [sheet.cell(row, col).value for col in useful]
        if not values or all(value is None for value in values):
            continue
        rows.append(values)
    if not rows:
        raise ValueError(f"Aucune donnée détectée dans {sheet.title!r}.")

    frame = pd.DataFrame(rows, columns=[headers[index - 1] for index in useful]).rename(
        columns={HEADER_READING: "lecture", HEADER_TIME: "temps_sec"}
    )
    frame["lecture"] = pd.to_numeric(frame["lecture"], errors="coerce")
    frame["temps_sec"] = pd.to_numeric(frame["temps_sec"], errors="coerce")
    frame = frame.dropna(subset=["lecture", "temps_sec"]).copy()
    frame["lecture"] = frame["lecture"].astype(int)
    measurements = [column for column in frame.columns if column not in {"lecture", "temps_sec"}]
    for column in measurements:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, measurements


def _sample_metadata(header: str) -> dict[str, str]:
    normalized = clean_text(header)
    match = re.match(r"^(.*?)(?:\s*\(([A-H]\d{2})\))?$", normalized)
    name = clean_text(match.group(1)) if match else normalized
    well = (match.group(2) or "") if match else ""
    return {
        "sample_header": header,
        "souche": name,
        "puits": well,
        "type": "blanc" if any(word in name.lower() for word in ("blanc", "blank")) else "souche",
    }


def _plate_groups(sheet) -> dict[str, str]:
    groups: dict[str, str] = {}
    number_row = None
    for row in range(1, min(sheet.max_row, 20) + 1):
        numbers = set()
        for col in range(2, min(sheet.max_column, 13) + 1):
            try:
                numbers.add(int(sheet.cell(row, col).value))
            except (TypeError, ValueError):
                pass
        if set(range(1, 13)).issubset(numbers):
            number_row = row
            break
    if number_row is None:
        raise ValueError("Impossible de trouver l'entête 1..12 dans le Plan de plaque.")

    for row in range(1, sheet.max_row + 1):
        letter = clean_text(sheet.cell(row, 1).value).upper()
        if letter not in PLATE_ROWS or row + 1 > sheet.max_row:
            continue
        for col in range(2, min(sheet.max_column, 13) + 1):
            try:
                number = int(sheet.cell(number_row, col).value)
            except (TypeError, ValueError):
                continue
            group = clean_text(sheet.cell(row + 1, col).value)
            if group:
                groups[f"{letter}{number:02d}"] = group
    return groups


def parse_kinetic_workbook(
    source: str | Path | BinaryIO,
    luminescence_sheet: str | None = None,
) -> pd.DataFrame:
    """Parse one kinetic workbook into an unaveraged, raw long table.

    When multiple luminescence sheets exist, callers must select one rather
    than allowing the scientific choice to be made silently.
    """
    workbook = load_workbook(source, read_only=False, data_only=True)
    try:
        absorbance_sheet, luminescence_sheets = find_measurement_sheets(workbook.sheetnames)
        if luminescence_sheet is None:
            if len(luminescence_sheets) > 1:
                raise ValueError("Plusieurs feuilles de luminescence détectées : choisissez-en une explicitement.")
            luminescence_sheet = luminescence_sheets[0]
        if luminescence_sheet not in luminescence_sheets:
            raise ValueError(f"Feuille de luminescence inconnue : {luminescence_sheet!r}.")
        plan_names = [name for name in workbook.sheetnames if clean_text(name).lower() == "plan de plaque"]
        if not plan_names:
            raise ValueError("La feuille 'Plan de plaque' est absente.")

        absorbance, absorbance_columns = _wide_table(workbook[absorbance_sheet])
        luminescence, luminescence_columns = _wide_table(workbook[luminescence_sheet])
        common = [column for column in absorbance_columns if column in set(luminescence_columns)]
        if not common:
            raise ValueError("Aucun échantillon commun entre l'absorbance et la luminescence.")

        do_long = absorbance[["lecture", "temps_sec", *common]].melt(
            id_vars=["lecture", "temps_sec"], var_name="sample_header", value_name="DO_brute"
        ).rename(columns={"temps_sec": "temps_sec_do"})
        lum_long = luminescence[["lecture", "temps_sec", *common]].melt(
            id_vars=["lecture", "temps_sec"], var_name="sample_header", value_name="Lum_brute"
        ).rename(columns={"temps_sec": "temps_sec_lum"})
        result = do_long.merge(lum_long, on=["lecture", "sample_header"], how="inner", validate="one_to_one")
        result["ecart_temps_s"] = (result["temps_sec_do"] - result["temps_sec_lum"]).abs()
        result["temps_h"] = result[["temps_sec_do", "temps_sec_lum"]].mean(axis=1) / 3600.0

        metadata = pd.DataFrame([_sample_metadata(column) for column in common])
        metadata["Groupe"] = metadata["puits"].map(_plate_groups(workbook[plan_names[0]])).fillna("")
        metadata["replicat"] = metadata.groupby("souche", sort=False).cumcount() + 1
        result = result.merge(metadata, on="sample_header", how="left", validate="many_to_one")
        columns = [
            "temps_h", "souche", "Groupe", "replicat", "DO_brute", "Lum_brute",
            "type", "puits", "lecture", "sample_header", "temps_sec_do",
            "temps_sec_lum", "ecart_temps_s",
        ]
        return result[columns].sort_values(["type", "souche", "replicat", "temps_h"]).reset_index(drop=True)
    finally:
        workbook.close()
