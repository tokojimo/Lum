"""Import Cytotox workbooks and prepare their normalized signals."""

from dataclasses import dataclass
from io import BytesIO
from math import isfinite
import re
from zipfile import BadZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException

from .kinetics import calculate_auc
from .varioskan import clean_text


PLATE_ROWS = tuple("ABCDEFGH")
PLATE_COLUMNS = tuple(range(1, 13))
WELL_PATTERN = re.compile(r"([A-H])0?([1-9]|1[0-2])")
TIME_HEADERS = {"elapsed", "elapsed (h)", "temps (h)", "temps_h"}


@dataclass
class CytotoxSignal:
    name: str
    sheet_name: str
    metric: str
    data: pd.DataFrame


@dataclass
class CytotoxWorkbook:
    plate: pd.DataFrame
    signals: tuple[CytotoxSignal, ...]


def normalize_ip(gfp: pd.DataFrame, ip: pd.DataFrame, end_minutes: float) -> pd.DataFrame:
    """Calculate (IP(t) - IP(0)) / GFP(0) per well through the inclusive cutoff."""
    if 0 not in gfp.index or 0 not in ip.index:
        raise ValueError("Les feuilles GFP brut et IP brut doivent contenir une mesure à t = 0.")
    end_hours = np.nextafter(end_minutes / 60, np.inf)
    window = ip.loc[(ip.index >= 0) & (ip.index <= end_hours)]
    gfp_zero = gfp.loc[0].reindex(ip.columns)
    return window.sub(ip.loc[0], axis="columns").div(
        gfp_zero.where(gfp_zero.ne(0)), axis="columns",
    ).replace([np.inf, -np.inf], np.nan)


def ip_auc_by_well(ip: pd.DataFrame) -> pd.Series:
    """Integrate filtered IP in minutes over the same measured interval for every well."""
    auc = pd.Series(np.nan, index=ip.columns, dtype=float)
    if len(ip.index) < 2:
        return auc
    # A missing endpoint would shorten the window for this well. Interior
    # gaps are handled by the shared trapezoidal integration.
    endpoints = ip.iloc[[0, -1]].notna().all()
    for well in endpoints[endpoints].index:
        auc.loc[well] = calculate_auc(ip.index * 60, ip[well])
    return auc


def _read_plate(sheet) -> pd.DataFrame:
    rows = list(sheet.iter_rows(max_row=9, max_col=13, values_only=True))
    columns = rows[0][1:] if rows else ()
    row_labels = [clean_text(row[0]).upper() for row in rows[1:]]
    if set(columns) != set(PLATE_COLUMNS) or set(row_labels) != set(PLATE_ROWS):
        raise ValueError(
            f"Feuille « {sheet.title} » : le plan doit porter les numéros 1 à 12 "
            "sur la première ligne et les lettres A à H dans la première colonne."
        )
    plate = pd.DataFrame(
        [[clean_text(cell) for cell in row[1:]] for row in rows[1:]],
        index=row_labels, columns=[int(column) for column in columns],
    ).reindex(index=PLATE_ROWS, columns=PLATE_COLUMNS)
    if not plate.ne("").any().any():
        raise ValueError(f"Feuille « {sheet.title} » : renseignez les libellés des puits.")
    return plate


def _number(value, sheet_name: str, line: int, column: str) -> float:
    try:
        if not isinstance(value, bool):
            number = float(value.replace(",", ".") if isinstance(value, str) else value)
            if isfinite(number):
                return number
    except (TypeError, ValueError, OverflowError):
        pass
    raise ValueError(f"Feuille « {sheet_name} », ligne {line} : valeur non numérique ({column}).")


def _read_signal(sheet, name: str) -> CytotoxSignal:
    metric = ""
    rows = enumerate(sheet.values, start=1)
    for _, header in rows:
        labels = [clean_text(cell) for cell in header]
        time_columns = [index for index, label in enumerate(labels) if label.casefold() in TIME_HEADERS]
        wells = [(index, f"{match[1]}{int(match[2])}") for index, label in enumerate(labels)
                 if (match := WELL_PATTERN.fullmatch(label.upper()))]
        if time_columns and wells:
            break
        for label in labels:
            if label.casefold().startswith("metric:"):
                metric = label.partition(":")[2].strip()
    else:
        raise ValueError(f"Feuille « {sheet.title} » : en-tête Elapsed et colonnes de puits introuvables.")

    well_names = [well for _, well in wells]
    if len(time_columns) != 1 or len(set(well_names)) != len(wells):
        raise ValueError(f"Feuille « {sheet.title} » : colonnes de temps ou de puits en double.")
    time_column = time_columns[0]
    times = []
    values = []
    for line, row in rows:
        elapsed = row[time_column] if time_column < len(row) else None
        readings = [row[column] if column < len(row) else None for column, _ in wells]
        # The raw GFP sheet may have additional blocks after the first table.
        if clean_text(elapsed).casefold() in TIME_HEADERS:
            break
        if elapsed is None and all(value is None for value in readings):
            if times:
                break
            continue
        time = _number(elapsed, sheet.title, line, "Elapsed")
        if times and time <= times[-1]:
            raise ValueError(f"Feuille « {sheet.title} », ligne {line} : les temps doivent être croissants.")
        times.append(time)
        values.append([
            None if value is None else _number(value, sheet.title, line, well)
            for value, well in zip(readings, well_names)
        ])

    data = pd.DataFrame(values, index=pd.Index(times, name="temps_h"), columns=well_names, dtype=float)
    if not data.notna().any().any():
        raise ValueError(
            f"Feuille « {sheet.title} » : aucune mesure lisible. "
            "Si les cellules contiennent des formules, enregistrez le classeur avec leurs résultats."
        )
    return CytotoxSignal(name, sheet.title, metric or name, data)


def read_cytotox_workbook(payload: bytes) -> CytotoxWorkbook:
    """Read GFP, IP, final GFP and the 96-well plate, in that order."""
    try:
        workbook = load_workbook(BytesIO(payload), read_only=True, data_only=True)
    except (BadZipFile, InvalidFileException, KeyError, OSError, ValueError) as error:
        raise ValueError("Impossible de lire ce classeur Excel.") from error
    try:
        if len(workbook.worksheets) != 4:
            raise ValueError("Le classeur doit contenir quatre feuilles : GFP, IP, courbes finales GFP et plaque.")
        gfp, ip, final_gfp, plate = workbook.worksheets
        return CytotoxWorkbook(
            plate=_read_plate(plate),
            signals=(
                _read_signal(gfp, "GFP brut"),
                _read_signal(ip, "IP"),
                _read_signal(final_gfp, "Courbes finales GFP"),
            ),
        )
    finally:
        workbook.close()
