"""Extraction of kinetic parameters from normalized plate-reader series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("temps_h", "souche", "DO_corr", "Lum_norm")
IDENTITY_COLUMNS = ("experience_id", "souche", "Groupe", "sample_header", "puits", "replicat")
METRIC_COLUMNS = (
    "od_max", "od_max_time_h", "od_auc", "max_growth_rate_per_h",
    "max_growth_rate_start_h", "max_growth_rate_end_h", "doubling_time_h",
    "lum_norm_peak", "lum_norm_peak_time_h", "lum_norm_auc",
    "lum_corr_peak", "lum_corr_peak_time_h", "lum_corr_auc",
)


@dataclass(frozen=True)
class KineticsResult:
    """Independent, exportable tables produced by a kinetics run."""

    series_metrics: pd.DataFrame
    strain_summary: pd.DataFrame
    rejected_series: pd.DataFrame
    warnings: pd.DataFrame
    summary: pd.DataFrame


def _finite_sorted(time_h, values) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_h, dtype=float)
    value = np.asarray(values, dtype=float)
    if time.shape != value.shape:
        raise ValueError("Time and value arrays must have identical shapes.")
    valid = np.isfinite(time) & np.isfinite(value)
    order = np.argsort(time[valid], kind="stable")
    return time[valid][order], value[valid][order]


def calculate_auc(time_h, values) -> float:
    """Trapezoidal AUC on observed finite points, without interpolation."""
    time, value = _finite_sorted(time_h, values)
    if time.size < 2:
        return float("nan")
    return float(np.trapezoid(value, time))


def calculate_peak(time_h, values) -> tuple[float, float]:
    """Return ``(peak, first observed peak time)``, without interpolation."""
    time, value = _finite_sorted(time_h, values)
    if time.size == 0:
        return float("nan"), float("nan")
    index = int(np.argmax(value))
    return float(value[index]), float(time[index])


def validate_kinetics_inputs(data: pd.DataFrame, growth_window_points: int = 3) -> pd.DataFrame:
    """Validate the kinetics contract and return a deep, numeric copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes pour la cinétique : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau cinétique est vide.")
    if not isinstance(growth_window_points, (int, np.integer)) or growth_window_points < 2:
        raise ValueError("growth_window_points doit être un entier supérieur ou égal à 2.")
    result = data.copy(deep=True)
    for column in ("temps_h", "DO_corr", "Lum_norm", "Lum_corr"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["temps_h"].isna().any() or not np.isfinite(result["temps_h"]).all():
        raise ValueError("temps_h doit contenir uniquement des valeurs numériques finies.")
    if "sample_header" not in result:
        if "puits" in result:
            result["sample_header"] = result["souche"].astype(str) + " (" + result["puits"].astype(str) + ")"
        else:
            raise ValueError("Une colonne sample_header ou puits est requise pour identifier les séries.")
    return result.reset_index(drop=True)


def calculate_growth_metrics(series: pd.DataFrame, growth_window_points: int = 3) -> dict[str, float | int]:
    """Calculate OD metrics and the steepest observed-window log(OD) slope.

    A growth window consists of consecutive rows after stable time sorting. Invalid
    or non-positive ODs and non-increasing timestamps break a window; points are
    never interpolated.
    """
    times, ods = _finite_sorted(series["temps_h"], series["DO_corr"])
    positive = ods > 0
    od_times, positive_ods = times[positive], ods[positive]
    peak, peak_time = calculate_peak(times, ods)
    metrics: dict[str, float | int] = {
        "n_growth_points": int(len(positive_ods)), "od_max": peak, "od_max_time_h": peak_time,
        "od_auc": calculate_auc(times, ods), "max_growth_rate_per_h": np.nan,
        "max_growth_rate_start_h": np.nan, "max_growth_rate_end_h": np.nan,
        "doubling_time_h": np.nan,
    }
    ordered = series.sort_values("temps_h", kind="stable")
    raw_t = pd.to_numeric(ordered["temps_h"], errors="coerce").to_numpy(float)
    raw_od = pd.to_numeric(ordered["DO_corr"], errors="coerce").to_numpy(float)
    candidates = []
    for start in range(len(ordered) - growth_window_points + 1):
        t = raw_t[start:start + growth_window_points]
        od = raw_od[start:start + growth_window_points]
        if np.isfinite(t).all() and np.isfinite(od).all() and (od > 0).all() and (np.diff(t) > 0).all():
            slope = float(np.polyfit(t, np.log(od), 1)[0])
            candidates.append((slope, float(t[0]), float(t[-1])))
    if candidates:
        slope, start, end = max(candidates, key=lambda item: item[0])
        metrics.update(max_growth_rate_per_h=slope, max_growth_rate_start_h=start,
                       max_growth_rate_end_h=end)
        if slope > 0:
            metrics["doubling_time_h"] = float(np.log(2) / slope)
    return metrics


def calculate_luminescence_metrics(series: pd.DataFrame) -> dict[str, float | int]:
    """Calculate normalized (and, when present, corrected) luminescence metrics."""
    peak, peak_time = calculate_peak(series["temps_h"], series["Lum_norm"])
    finite_norm = np.isfinite(pd.to_numeric(series["Lum_norm"], errors="coerce"))
    result: dict[str, float | int] = {
        "n_lum_norm_points": int(finite_norm.sum()), "lum_norm_peak": peak,
        "lum_norm_peak_time_h": peak_time,
        "lum_norm_auc": calculate_auc(series["temps_h"], series["Lum_norm"]),
    }
    if "Lum_corr" in series:
        corr_peak, corr_time = calculate_peak(series["temps_h"], series["Lum_corr"])
        result.update(lum_corr_peak=corr_peak, lum_corr_peak_time_h=corr_time,
                      lum_corr_auc=calculate_auc(series["temps_h"], series["Lum_corr"]))
    else:
        result.update(lum_corr_peak=np.nan, lum_corr_peak_time_h=np.nan, lum_corr_auc=np.nan)
    return result


def _series_group_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in ("experience_id", "sample_header") if column in data]


def extract_series_kinetics(
    data: pd.DataFrame, growth_window_points: int = 3, minimum_auc_points: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract per-well metrics, rejected series, and non-fatal warnings."""
    if not isinstance(minimum_auc_points, (int, np.integer)) or minimum_auc_points < 2:
        raise ValueError("minimum_auc_points doit être un entier supérieur ou égal à 2.")
    metrics, rejected, warnings = [], [], []
    strains = data.loc[data["type"].eq("souche")] if "type" in data else data
    for _, series in strains.groupby(_series_group_columns(data), dropna=False, sort=False):
        identity = {column: series[column].iloc[0] for column in IDENTITY_COLUMNS if column in series}
        times = series["temps_h"]
        if times.duplicated().any():
            rejected.append({**identity, "reason": "duplicate_time", "n_points_total": len(series)})
            continue
        growth = calculate_growth_metrics(series, growth_window_points)
        lum = calculate_luminescence_metrics(series)
        reasons = []
        if growth["n_growth_points"] < minimum_auc_points:
            reasons.append("insufficient_growth_points")
        if lum["n_lum_norm_points"] < minimum_auc_points:
            reasons.append("insufficient_lum_norm_points")
        if reasons:
            rejected.append({**identity, "reason": ";".join(reasons), "n_points_total": len(series)})
            continue
        if not np.isfinite(growth["max_growth_rate_per_h"]):
            warnings.append({**identity, "code": "insufficient_growth_window",
                             "message": "Aucune fenêtre de croissance valide n'a été trouvée."})
        finite_any = np.isfinite(series[["DO_corr", "Lum_norm"]].to_numpy(dtype=float)).all(axis=1)
        used_times = series.loc[finite_any, "temps_h"]
        metrics.append({**identity, "n_points_total": len(series),
                        "n_points_jointly_finite": int(finite_any.sum()),
                        "analysis_start_h": float(used_times.min()) if len(used_times) else np.nan,
                        "analysis_end_h": float(used_times.max()) if len(used_times) else np.nan,
                        "growth_window_points": growth_window_points,
                        "minimum_auc_points": minimum_auc_points, **growth, **lum})
    return pd.DataFrame(metrics), pd.DataFrame(rejected), pd.DataFrame(warnings)


def summarize_technical_replicates(series_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize wells within each experiment/strain/condition, never across experiments."""
    group_columns = [column for column in ("experience_id", "souche", "Groupe") if column in series_metrics]
    if series_metrics.empty:
        return pd.DataFrame(columns=group_columns + ["n_technical_series"])
    rows = []
    for keys, group in series_metrics.groupby(group_columns, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys))
        row["n_technical_series"] = len(group)
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = values.mean()
            row[f"{metric}_sd"] = values.std(ddof=1)
            row[f"{metric}_n"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def run_kinetics(
    data: pd.DataFrame, growth_window_points: int = 3, minimum_auc_points: int = 2,
) -> KineticsResult:
    """Run the complete, non-mutating kinetic-parameter workflow."""
    prepared = validate_kinetics_inputs(data, growth_window_points)
    metrics, rejected, warnings = extract_series_kinetics(
        prepared, growth_window_points, minimum_auc_points
    )
    strain_summary = summarize_technical_replicates(metrics)
    summary = pd.DataFrame([
        ("series_total", len(metrics) + len(rejected)),
        ("series_analyzed", len(metrics)),
        ("series_rejected", len(rejected)),
        ("warnings", len(warnings)),
        ("growth_window_points", growth_window_points),
        ("minimum_auc_points", minimum_auc_points),
    ], columns=["metric", "value"])
    return KineticsResult(metrics, strain_summary, rejected, warnings, summary)
