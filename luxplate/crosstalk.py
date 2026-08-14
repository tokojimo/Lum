"""Application of the frozen Mauri E06 Dbest optical cross-talk model."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files
import json
import math
import re
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


KERNEL_ID = "MAURI_E06_BEST"
MODEL_RESOURCE = files("luxplate").joinpath("resources", "crosstalk", KERNEL_ID)
PLATE_WELLS = tuple(
    f"{row}{column:02d}" for row in "ABCDEFGH" for column in range(1, 13)
)
_WELL_RE = re.compile(r"^[A-H](?:0[1-9]|1[0-2])$")
_NON_LUMINESCENT_STATUSES = {"water", "eau", "non_luminescent"}


def _normalize_status(value: object) -> str:
    """Normalize the deliberately small vocabulary accepted for absent wells."""
    return re.sub(r"[\s-]+", "_", str(value).strip().casefold())


@dataclass(frozen=True)
class CrosstalkModel:
    """Validated, immutable view of the packaged calibration artifacts."""

    Dbest: np.ndarray
    background_rlu: float
    metadata: dict[str, Any]
    condition_number: float


def load_crosstalk_model() -> CrosstalkModel:
    """Load and validate the exact packaged Dbest calibration artifacts."""
    try:
        with MODEL_RESOURCE.joinpath("02_background_estimate.json").open(
            "r", encoding="utf-8"
        ) as stream:
            background_data = json.load(stream)
        with MODEL_RESOURCE.joinpath("kernel_metadata.json").open(
            "r", encoding="utf-8"
        ) as stream:
            metadata = json.load(stream)
        with as_file(MODEL_RESOURCE.joinpath("kernel_D_best.npy")) as kernel_path:
            kernel = np.load(kernel_path, allow_pickle=False)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, KeyError) as error:
        raise RuntimeError(f"Artefacts Dbest absents ou illisibles : {error}") from error

    if metadata.get("kernel_id") != KERNEL_ID:
        raise RuntimeError(
            f"Kernel Dbest invalide : kernel_id={metadata.get('kernel_id')!r}, "
            f"attendu={KERNEL_ID!r}."
        )
    if kernel.shape != (96, 96):
        raise RuntimeError(f"Kernel Dbest invalide : forme {kernel.shape}, attendue (96, 96).")
    if not np.issubdtype(kernel.dtype, np.number) or not np.isfinite(kernel).all():
        raise RuntimeError("Kernel Dbest invalide : toutes les valeurs doivent être numériques et finies.")

    background = background_data.get("background_rlu")
    if isinstance(background, bool) or not isinstance(background, (int, float)):
        raise RuntimeError("Background Dbest invalide : background_rlu doit être numérique.")
    background = float(background)
    if not math.isfinite(background):
        raise RuntimeError("Background Dbest invalide : background_rlu doit être fini.")

    kernel = np.asarray(kernel, dtype=float)
    try:
        # A solve against zero validates that the matrix is square and non-singular.
        np.linalg.solve(kernel, np.zeros(96, dtype=float))
        condition_number = float(np.linalg.cond(kernel))
    except np.linalg.LinAlgError as error:
        raise RuntimeError("Kernel Dbest inutilisable par np.linalg.solve.") from error
    if not math.isfinite(condition_number):
        raise RuntimeError("Kernel Dbest invalide : condition number non fini.")
    kernel.setflags(write=False)
    return CrosstalkModel(kernel, background, metadata, condition_number)


def correct_plate_crosstalk(
    data: pd.DataFrame,
    *,
    unmeasured_well_statuses: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Apply the Dbest solve to complete or partial plates.

    Every position absent from a plate/time group is considered non-luminescent:
    its *true* source signal is assumed to be zero. ``unmeasured_well_statuses``
    remains available as optional documentation and, when supplied, is
    validated against the accepted ``water``, ``eau`` and ``non_luminescent``
    vocabulary. A missing raw reading on a row that is present is never
    manufactured.

    Each experiment/time group is solved independently using the principal
    submatrix in canonical A01–H12 order. The input is never mutated and
    negative deconvolved values are kept.
    """
    required = {"puits", "temps_h", "Lum_brute"}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError("Colonnes manquantes pour le cross-talk : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau de luminescence est vide.")

    for well, status in (unmeasured_well_statuses or {}).items():
        normalized_well = str(well).strip().upper()
        if not _WELL_RE.fullmatch(normalized_well):
            raise ValueError(
                "Position de puits non mesuré invalide (format attendu A01 à H12) : "
                + normalized_well
            )
        normalized_status = _normalize_status(status)
        if normalized_status not in _NON_LUMINESCENT_STATUSES:
            raise ValueError(
                f"Statut invalide pour le puits non mesuré {normalized_well} : {status!r}. "
                "Statuts autorisés : water, eau, non_luminescent."
            )
    model = load_crosstalk_model()
    output = data.copy(deep=True)
    output["_crosstalk_row_position"] = np.arange(len(output))
    wells = output["puits"].fillna("").astype(str).str.strip().str.upper()
    invalid = sorted(wells.loc[~wells.str.fullmatch(_WELL_RE)].unique())
    if invalid:
        raise ValueError(
            "Positions de puits invalides (format attendu A01 à H12) : " + ", ".join(invalid)
        )
    output["puits"] = wells
    output["RLU_raw"] = pd.to_numeric(output["Lum_brute"], errors="coerce")

    result_columns = (
        "RLU_background_subtracted", "RLU_corrected", "RLU_correction_delta",
        "Lum_analysis", "max_abs_reconstruction_residual", "RMSE_reconstruction",
    )
    for column in result_columns:
        output[column] = np.nan
    output["crosstalk_method"] = "MAURI_DBEST"
    output["crosstalk_kernel_id"] = model.metadata["kernel_id"]
    output["crosstalk_condition_number"] = model.condition_number

    group_keys = (["experience"] if "experience" in output.columns else []) + ["temps_h"]
    grouper = group_keys[0] if len(group_keys) == 1 else group_keys
    for key, plate in output.groupby(grouper, sort=False, dropna=False):
        duplicates = sorted(plate.loc[plate["puits"].duplicated(keep=False), "puits"].unique())
        if duplicates:
            raise ValueError(f"Correction Dbest impossible : puits dupliqués au groupe {key!r}: " + ", ".join(duplicates))
        measured_wells = set(plate["puits"])
        if plate["RLU_raw"].isna().any() or not np.isfinite(plate["RLU_raw"].to_numpy()).all():
            raise ValueError(f"Correction Dbest impossible : Lum_brute manquante ou non numérique au groupe {key!r}.")

        measured_order = [well for well in PLATE_WELLS if well in measured_wells]
        measured_indices = [PLATE_WELLS.index(well) for well in measured_order]
        ordered = plate.set_index("puits").loc[measured_order]
        raw = ordered["RLU_raw"].to_numpy(dtype=float)
        optical = raw - model.background_rlu
        reduced_kernel = model.Dbest[np.ix_(measured_indices, measured_indices)]
        try:
            corrected = np.linalg.solve(reduced_kernel, optical)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"Sous-kernel Dbest inutilisable au groupe {key!r}.") from error
        residual = optical - reduced_kernel @ corrected
        max_residual = float(np.max(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(np.square(residual))))

        values = pd.DataFrame({
            "RLU_background_subtracted": optical,
            "RLU_corrected": corrected,
            "RLU_correction_delta": optical - corrected,
            "Lum_analysis": corrected,
        }, index=measured_order)
        result_positions = [output.columns.get_loc(column) for column in values.columns]
        for _, row in plate.iterrows():
            position = int(row["_crosstalk_row_position"])
            output.iloc[position, result_positions] = values.loc[row["puits"]].to_numpy()
        positions = plate["_crosstalk_row_position"].to_numpy(dtype=int)
        output.iloc[positions, output.columns.get_loc("max_abs_reconstruction_residual")] = max_residual
        output.iloc[positions, output.columns.get_loc("RMSE_reconstruction")] = rmse

    return output.drop(columns="_crosstalk_row_position")
