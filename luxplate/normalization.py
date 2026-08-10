"""Normalize blank-corrected luminescence by a defensible OD threshold."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .models import ProcessingSettings


REQUIRED_COLUMNS = ("temps_h", "souche", "DO_corr", "Lum_corr", "type")
WARNING_COLUMNS = ("code", "experience_id", "message")
VALIDATION_COLUMNS = (
    "experience_id", "souche", "sample_header", "replicat", "n_points_total",
    "n_points_above_threshold", "threshold_effective", "consecutive_points",
    "normalization_start_time_h", "series_valid", "reason",
)


@dataclass(frozen=True)
class NormalizationResult:
    """Independent tables produced by an OD normalization run."""

    normalized_data: pd.DataFrame
    rejected_rows: pd.DataFrame
    series_validation: pd.DataFrame
    threshold_details: pd.DataFrame
    warnings: pd.DataFrame
    summary: pd.DataFrame


def validate_normalization_inputs(data: pd.DataFrame) -> pd.DataFrame:
    """Validate parameters encoded in the table and return a normalized copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes pour la normalisation : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau à normaliser est vide.")
    result = data.copy(deep=True)
    for column in ("temps_h", "DO_corr", "Lum_corr"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["temps_h"].isna().any() or not np.isfinite(result["temps_h"]).all():
        raise ValueError("temps_h doit contenir uniquement des valeurs numériques finies.")
    result["type"] = result["type"].fillna("").astype(str).str.strip().str.lower()
    result["souche"] = result["souche"].fillna("").astype(str).str.strip()
    if "experience_id" in result:
        result["experience_id"] = result["experience_id"].fillna("").astype(str).str.strip()
    if "sample_header" not in result:
        if "puits" in result:
            result["sample_header"] = result["souche"] + " (" + result["puits"].astype(str) + ")"
        elif "replicat" in result:
            replicate = pd.to_numeric(result["replicat"], errors="coerce").astype("Int64").astype(str)
            result["sample_header"] = result["souche"] + "_rep" + replicate
        else:
            raise ValueError("Une colonne sample_header, puits ou replicat est requise pour identifier les séries.")
    result["sample_header"] = result["sample_header"].fillna("").astype(str).str.strip()
    return result.reset_index(drop=True)


def _experience_groups(data: pd.DataFrame):
    if "experience_id" in data:
        return data.groupby("experience_id", dropna=False, sort=False)
    return [("", data)]


def calculate_blank_od_threshold(data: pd.DataFrame, blank_sd_multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate mean + k×sample-SD of finite corrected blank ODs per experiment."""
    if not np.isfinite(blank_sd_multiplier) or blank_sd_multiplier < 0:
        raise ValueError("blank_sd_multiplier doit être un nombre fini supérieur ou égal à zéro.")
    rows = []
    for experience, group in _experience_groups(data):
        values = pd.to_numeric(
            group.loc[group["type"].eq("blanc"), "DO_corr"], errors="coerce"
        ).to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        mean = float(values.mean()) if len(values) else np.nan
        sd = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        threshold = mean + blank_sd_multiplier * sd if len(values) > 1 else mean
        rows.append({
            "experience_id": experience, "n_valid_blank_points": len(values),
            "blank_od_mean": mean, "blank_od_sd": sd,
            "blank_sd_multiplier": float(blank_sd_multiplier), "blank_threshold": threshold,
        })
    return pd.DataFrame(rows)


def find_first_consecutive_valid_time(
    series: pd.DataFrame, threshold: float, consecutive_points: int = 3
) -> float:
    """Return the first time starting a run of valid, strictly above-threshold ODs.

    Duplicate or decreasing timestamps break a run; missing/invalid OD values do
    too. No interpolation is performed, so observed rows remain the time grid.
    """
    if consecutive_points < 1:
        raise ValueError("consecutive_points doit être supérieur ou égal à 1.")
    ordered = series.sort_values("temps_h", kind="stable")
    ods = pd.to_numeric(ordered["DO_corr"], errors="coerce").to_numpy(dtype=float)
    times = pd.to_numeric(ordered["temps_h"], errors="coerce").to_numpy(dtype=float)
    run = 0
    for index, (time, od) in enumerate(zip(times, ods)):
        increasing = index == 0 or time > times[index - 1]
        run = run + 1 if increasing and np.isfinite(od) and od > threshold else 0
        if run == consecutive_points:
            return float(times[index - consecutive_points + 1])
    return np.nan


def _series_columns(data: pd.DataFrame) -> list[str]:
    columns = ["souche", "sample_header"]
    if "experience_id" in data:
        columns.insert(0, "experience_id")
    return columns


def validate_series_for_normalization(
    data: pd.DataFrame, threshold_details: pd.DataFrame,
    minimum_od: float = 0.05, consecutive_points: int = 3,
) -> pd.DataFrame:
    """Validate each strain/well series independently and record its start time."""
    if not np.isfinite(minimum_od) or minimum_od < 0:
        raise ValueError("minimum_od doit être un nombre fini supérieur ou égal à zéro.")
    if not isinstance(consecutive_points, (int, np.integer)) or consecutive_points < 1:
        raise ValueError("consecutive_points doit être un entier supérieur ou égal à 1.")
    threshold_map = threshold_details.set_index("experience_id")["blank_threshold"].to_dict()
    rows = []
    strains = data.loc[data["type"].eq("souche")]
    for keys, series in strains.groupby(_series_columns(data), dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(_series_columns(data), keys))
        experience = identity.get("experience_id", "")
        blank_threshold = threshold_map.get(experience, np.nan)
        effective = max(float(minimum_od), float(blank_threshold)) if pd.notna(blank_threshold) else float(minimum_od)
        start = find_first_consecutive_valid_time(series, effective, int(consecutive_points))
        ods = pd.to_numeric(series["DO_corr"], errors="coerce")
        reason = "" if pd.notna(start) else (
            "series_too_short" if len(series) < consecutive_points else "no_consecutive_points_above_threshold"
        )
        rows.append({
            "experience_id": experience, "souche": identity["souche"],
            "sample_header": identity["sample_header"],
            "replicat": series["replicat"].iloc[0] if "replicat" in series else pd.NA,
            "n_points_total": len(series),
            "n_points_above_threshold": int((np.isfinite(ods) & (ods > effective)).sum()),
            "threshold_effective": effective, "consecutive_points": int(consecutive_points),
            "normalization_start_time_h": start, "series_valid": bool(pd.notna(start)), "reason": reason,
        })
    return pd.DataFrame(rows, columns=VALIDATION_COLUMNS)


def normalize_luminescence_by_od(
    data: pd.DataFrame, series_validation: pd.DataFrame, threshold_details: pd.DataFrame,
    minimum_od: float = 0.05,
) -> pd.DataFrame:
    """Return all input rows annotated with normalization status and reason."""
    output = data.copy(deep=True)
    merge_columns = _series_columns(data)
    annotations = series_validation[list(dict.fromkeys(merge_columns + [
        "threshold_effective", "normalization_start_time_h", "series_valid"
    ]))]
    output = output.merge(annotations, on=merge_columns, how="left", validate="many_to_one")
    threshold_map = threshold_details.set_index("experience_id")["blank_threshold"].to_dict()
    experience = output["experience_id"] if "experience_id" in output else pd.Series("", index=output.index)
    output["blank_threshold"] = experience.map(threshold_map)
    output["minimum_od"] = float(minimum_od)
    reasons = []
    for row in output.itertuples(index=False):
        kind, od, lum = row.type, row.DO_corr, row.Lum_corr
        if kind != "souche": reason = "blank_row" if kind == "blanc" else "non_strain_row"
        elif not np.isfinite(od): reason = "invalid_od"
        elif od <= 0: reason = "non_positive_od"
        elif not np.isfinite(lum): reason = "invalid_luminescence"
        elif pd.isna(row.series_valid) or not bool(row.series_valid): reason = "series_not_validated"
        elif row.temps_h < row.normalization_start_time_h: reason = "before_normalization_start"
        elif od <= row.threshold_effective: reason = "od_not_above_threshold"
        else: reason = ""
        reasons.append(reason)
    output["normalization_reason"] = reasons
    output["normalization_ok"] = output["normalization_reason"].eq("")
    output["Lum_norm"] = np.nan
    valid = output["normalization_ok"]
    output.loc[valid, "Lum_norm"] = output.loc[valid, "Lum_corr"] / output.loc[valid, "DO_corr"]
    return output


def run_normalization(
    data: pd.DataFrame, blank_sd_multiplier: float = ProcessingSettings.blank_sd_multiplier,
    minimum_od: float = ProcessingSettings.minimum_od,
    consecutive_points: int = ProcessingSettings.consecutive_points,
) -> NormalizationResult:
    """Run the complete, non-mutating OD normalization workflow."""
    prepared = validate_normalization_inputs(data)
    details = calculate_blank_od_threshold(prepared, blank_sd_multiplier)
    details["minimum_od"] = float(minimum_od)
    details["effective_threshold"] = details["blank_threshold"].fillna(float(minimum_od)).clip(lower=float(minimum_od))
    details["consecutive_points"] = consecutive_points
    validation = validate_series_for_normalization(prepared, details, minimum_od, consecutive_points)
    normalized = normalize_luminescence_by_od(prepared, validation, details, minimum_od)
    rejected = normalized.loc[~normalized["normalization_ok"]].copy().reset_index(drop=True)
    warnings = pd.DataFrame([
        {"code": "no_valid_corrected_blanks", "experience_id": row.experience_id,
         "message": "Aucun blanc corrigé valide : le seuil effectif utilise uniquement la DO minimale."}
        for row in details.loc[details["n_valid_blank_points"].eq(0)].itertuples(index=False)
    ], columns=WARNING_COLUMNS)
    summary = pd.DataFrame([
        ("rows_total", len(normalized)),
        ("strain_rows", int(normalized["type"].eq("souche").sum())),
        ("normalized_rows", int(normalized["normalization_ok"].sum())),
        ("rejected_rows", len(rejected)),
        ("valid_series", int(validation["series_valid"].sum()) if not validation.empty else 0),
        ("invalid_series", int((~validation["series_valid"]).sum()) if not validation.empty else 0),
    ], columns=["metric", "value"])
    return NormalizationResult(normalized, rejected, validation, details, warnings, summary)
