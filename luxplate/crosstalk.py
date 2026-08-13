"""Fixed, non-iterative optical cross-talk correction for 96-well plates."""

from __future__ import annotations

from types import MappingProxyType
import re

import numpy as np
import pandas as pd


INSTRUMENT_BACKGROUND_RLU = 24.0
# Direction means the position of the source relative to the target well.
CROSSTALK_COEFFICIENTS = MappingProxyType({
    "N": 0.013177,
    "S": 0.009386,
    "E": 0.014344,
    "O": 0.012281,
    "NE": 0.002256,
    "NO": 0.001075,
    "SE": 0.000874,
    "SO": 0.001521,
})
_DIRECTION_OFFSETS = MappingProxyType({
    "N": (-1, 0), "S": (1, 0), "E": (0, 1), "O": (0, -1),
    "NE": (-1, 1), "NO": (-1, -1), "SE": (1, 1), "SO": (1, -1),
})
PLATE_WELLS = tuple(f"{row}{column:02d}" for row in "ABCDEFGH" for column in range(1, 13))
_WELL_RE = re.compile(r"^[A-H](?:0[1-9]|1[0-2])$")


def build_crosstalk_matrix() -> np.ndarray:
    """Return A where ``A[target, source]`` is the fixed directional coefficient."""
    indices = {well: index for index, well in enumerate(PLATE_WELLS)}
    matrix = np.zeros((96, 96), dtype=float)
    for target_index, target in enumerate(PLATE_WELLS):
        row, column = ord(target[0]) - ord("A"), int(target[1:]) - 1
        for direction, (row_offset, column_offset) in _DIRECTION_OFFSETS.items():
            source_row, source_column = row + row_offset, column + column_offset
            if 0 <= source_row < 8 and 0 <= source_column < 12:
                source = f"{chr(ord('A') + source_row)}{source_column + 1:02d}"
                matrix[target_index, indices[source]] = CROSSTALK_COEFFICIENTS[direction]
    matrix.setflags(write=False)
    return matrix


CROSSTALK_MATRIX = build_crosstalk_matrix()


def correct_plate_crosstalk(data: pd.DataFrame) -> pd.DataFrame:
    """Add raw, predicted and corrected RLU columns without mutating ``data``.

    Every experiment and time is treated as a separate optical plate. Each
    such plate must contain exactly one finite measurement for every A01--H12
    well. Sources are always the simultaneous measured values minus the fixed
    instrumental background; corrected values are never fed back as sources.
    ``Lum_analysis`` is the explicit downstream signal used by blank correction.
    """
    required = {"puits", "temps_h", "Lum_brute"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Colonnes manquantes pour le cross-talk : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau de luminescence est vide.")

    output = data.copy(deep=True)
    output["puits"] = output["puits"].fillna("").astype(str).str.strip().str.upper()
    invalid = sorted(output.loc[~output["puits"].str.fullmatch(_WELL_RE), "puits"].unique())
    if invalid:
        raise ValueError("Positions de puits invalides (format attendu A01 à H12) : " + ", ".join(invalid))
    output["RLU_raw"] = pd.to_numeric(output["Lum_brute"], errors="coerce")
    output["Instrument_background_RLU"] = INSTRUMENT_BACKGROUND_RLU
    for direction in CROSSTALK_COEFFICIENTS:
        output[f"CT_{direction}"] = np.nan
    output["CrossTalk_predicted"] = np.nan
    output["RLU_corrected"] = np.nan

    plate_keys = ["experience"] if "experience" in output.columns else []
    group_keys = [*plate_keys, "temps_h"]
    grouper = group_keys[0] if len(group_keys) == 1 else group_keys
    expected = set(PLATE_WELLS)
    for key, plate in output.groupby(grouper, sort=False, dropna=False):
        observed = set(plate["puits"])
        duplicates = sorted(plate.loc[plate["puits"].duplicated(keep=False), "puits"].unique())
        missing_wells = sorted(expected - observed)
        extra_wells = sorted(observed - expected)
        if len(plate) != 96 or duplicates or missing_wells or extra_wells:
            details = []
            if missing_wells:
                details.append("manquants=" + ",".join(missing_wells))
            if duplicates:
                details.append("doublons=" + ",".join(duplicates))
            if extra_wells:
                details.append("hors plaque=" + ",".join(extra_wells))
            raise ValueError(f"Plaque incomplète au groupe {key!r}: 96 puits uniques requis ({'; '.join(details)}).")
        ordered = plate.set_index("puits").loc[list(PLATE_WELLS)]
        measured = ordered["RLU_raw"].to_numpy(dtype=float)
        if not np.isfinite(measured).all():
            bad = list(ordered.index[~np.isfinite(measured)])
            raise ValueError(f"Luminescence absente/non numérique au groupe {key!r}: {', '.join(bad)}.")
        light = measured - INSTRUMENT_BACKGROUND_RLU
        contributions = {}
        for direction, coefficient in CROSSTALK_COEFFICIENTS.items():
            directional = np.zeros(96, dtype=float)
            coefficient_mask = CROSSTALK_MATRIX == coefficient
            directional[:] = coefficient_mask @ light * coefficient
            contributions[direction] = directional
        predicted = CROSSTALK_MATRIX @ light
        corrected = light - predicted
        # Resolve output row numbers within this plate (experiments repeat well names).
        plate_rows = plate.reset_index(names="_output_row").set_index("puits")
        row_indices = plate_rows.loc[list(PLATE_WELLS), "_output_row"].to_numpy()
        for direction, values in contributions.items():
            output.loc[row_indices, f"CT_{direction}"] = values
        output.loc[row_indices, "CrossTalk_predicted"] = predicted
        output.loc[row_indices, "RLU_corrected"] = corrected

    output["Lum_analysis"] = output["RLU_corrected"]
    return output
