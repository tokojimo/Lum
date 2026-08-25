"""Extraction of kinetic parameters from normalized plate-reader series."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("temps_h", "souche", "DO_corr", "Lum_corr", "Lum_norm")
# ``combine_kinetic_tables`` records the uploaded workbook in ``experience``.
# Keep that legacy-but-real biological identity all the way into the metric
# table; otherwise only ``replicat`` survives and technical well numbers can be
# mistaken for independent experiments by the statistical figures.
IDENTITY_COLUMNS = (
    "biological_pair_id", "experience_id", "experience", "souche", "Groupe", "sample_header", "puits", "replicat",
)
METRIC_COLUMNS = (
    "od_max", "od_max_time_h", "od_auc", "max_growth_rate_per_h",
    "max_growth_rate_start_h", "max_growth_rate_end_h", "doubling_time_h",
    "lum_norm_peak", "lum_norm_peak_time_h", "lum_norm_auc",
    "lum_corr_peak", "lum_corr_peak_time_h", "lum_corr_auc",
)
SERIES_METRIC_COLUMNS = (*IDENTITY_COLUMNS, "n_points_total", "n_points_jointly_finite",
    "analysis_start_h", "analysis_end_h", "growth_window_points", "growth_window_min_duration_h",
    "growth_rate_min_r_squared", "minimum_auc_points", "auc_window_start_h", "auc_window_end_h",
    "auc_window_duration_h", "n_auc_points", "od_max",
    "od_max_time_h", "od_auc", "max_growth_rate_per_h", "growth_rate_r_squared",
    "max_growth_rate_start_h", "max_growth_rate_end_h", "doubling_time_h",
    "growth_rate_publishability_reason", "n_lum_norm_points", "n_lum_norm_auc_points", "lum_norm_peak",
    "lum_norm_peak_time_h", "lum_norm_auc", "lum_norm_auc_reason",
    "lum_norm_auc_do_cutoff", "lum_corr_peak", "lum_corr_peak_time_h",
    "lum_corr_auc")
REJECTED_SERIES_COLUMNS = (*IDENTITY_COLUMNS, "reason", "n_points_total")
WARNING_COLUMNS = (*IDENTITY_COLUMNS, "code", "message")
SUMMARY_COLUMNS = ("metric", "value")
TECHNICAL_SUMMARY_COLUMNS = (
    "biological_pair_id", "experience_id", "experience", "souche", "Groupe", "replicat", "n_technical_series",
    *(name for metric in METRIC_COLUMNS for name in (f"{metric}_mean", f"{metric}_sd", f"{metric}_n")),
)


@dataclass(frozen=True)
class KineticsResult:
    series_metrics: pd.DataFrame
    strain_summary: pd.DataFrame
    rejected_series: pd.DataFrame
    warnings: pd.DataFrame
    summary: pd.DataFrame


def _table(rows, columns) -> pd.DataFrame:
    """Build an output table with a stable public schema, including when empty."""
    if rows and not isinstance(rows[0], dict):
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows).reindex(columns=columns)


def _finite_sorted(time_h, values) -> tuple[np.ndarray, np.ndarray]:
    time = np.asarray(time_h, dtype=float)
    value = np.asarray(values, dtype=float)
    if time.shape != value.shape:
        raise ValueError("Time and value arrays must have identical shapes.")
    valid = np.isfinite(time) & np.isfinite(value)
    order = np.argsort(time[valid], kind="stable")
    return time[valid][order], value[valid][order]


def calculate_auc(time_h, values) -> float:
    """AUC over every finite observation; trapezoids deliberately span missing points."""
    time, value = _finite_sorted(time_h, values)
    if time.size < 2:
        return float("nan")
    return float(np.trapezoid(value, time))


def calculate_baseline_shifted_auc(time_h, values) -> float:
    """AUC after translating the lowest finite value to zero."""
    time, value = _finite_sorted(time_h, values)
    if time.size < 2:
        return float("nan")
    return float(np.trapezoid(value - np.min(value), time))


def calculate_peak(time_h, values) -> tuple[float, float]:
    """Return ``(peak, first chronological observed peak time)``."""
    time, value = _finite_sorted(time_h, values)
    if time.size == 0:
        return float("nan"), float("nan")
    index = int(np.argmax(value))
    return float(value[index]), float(time[index])


def validate_kinetics_inputs(data: pd.DataFrame, growth_window_points: int = 3) -> pd.DataFrame:
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
        if "puits" not in result:
            raise ValueError("Une colonne sample_header ou puits est requise pour identifier les séries.")
        result["sample_header"] = result["souche"].astype(str) + " (" + result["puits"].astype(str) + ")"
    return result.reset_index(drop=True)


def calculate_growth_metrics(series: pd.DataFrame, growth_window_points: int = 3,
                             growth_window_min_duration_h: float = 0.0,
                             growth_rate_min_r_squared: float = 0.0) -> dict[str, float | int | str]:
    """Calculate OD metrics and the best publishable consecutive log(OD) window.

    Invalid/non-positive OD and duplicate or decreasing times break a window. Uneven
    sampling is accepted because regression uses actual times. Candidate windows must
    meet the duration and R² thresholds; the highest slope is retained.
    """
    times, ods = _finite_sorted(series["temps_h"], series["DO_corr"])
    positive_ods = ods[ods > 0]
    peak, peak_time = calculate_peak(times, ods)
    metrics: dict[str, float | int | str] = {
        "n_growth_points": int(len(positive_ods)), "od_max": peak, "od_max_time_h": peak_time,
        "od_auc": calculate_auc(times, ods), "max_growth_rate_per_h": np.nan,
        "growth_rate_r_squared": np.nan, "max_growth_rate_start_h": np.nan,
        "max_growth_rate_end_h": np.nan, "doubling_time_h": np.nan,
        "growth_rate_publishability_reason": "insufficient_growth_window",
    }
    ordered = series.sort_values("temps_h", kind="stable")
    raw_t = pd.to_numeric(ordered["temps_h"], errors="coerce").to_numpy(float)
    raw_od = pd.to_numeric(ordered["DO_corr"], errors="coerce").to_numpy(float)
    candidates = []
    for start in range(len(ordered) - growth_window_points + 1):
        t = raw_t[start:start + growth_window_points]
        od = raw_od[start:start + growth_window_points]
        if not (np.isfinite(t).all() and np.isfinite(od).all() and (od > 0).all()
                and (np.diff(t) > 0).all() and t[-1] - t[0] >= growth_window_min_duration_h):
            continue
        log_od = np.log(od)
        slope, intercept = np.polyfit(t, log_od, 1)
        residual = float(np.sum((log_od - (slope * t + intercept)) ** 2))
        total = float(np.sum((log_od - log_od.mean()) ** 2))
        r_squared = 1.0 if total == 0 and residual == 0 else (1 - residual / total if total else 0.0)
        candidates.append((float(slope), float(r_squared), float(t[0]), float(t[-1])))
    quality = [candidate for candidate in candidates if candidate[1] >= growth_rate_min_r_squared]
    if quality:
        slope, r_squared, start, end = max(quality, key=lambda item: item[0])
        metrics.update(max_growth_rate_per_h=slope, growth_rate_r_squared=r_squared,
                       max_growth_rate_start_h=start, max_growth_rate_end_h=end)
        if slope > 0:
            metrics.update(doubling_time_h=float(np.log(2) / slope), growth_rate_publishability_reason="")
        else:
            metrics["growth_rate_publishability_reason"] = "non_positive_growth_rate"
    elif candidates:
        metrics["growth_rate_publishability_reason"] = "insufficient_regression_quality"
    return metrics


def calculate_luminescence_metrics(
    series: pd.DataFrame, window_start_h: float | None = None, window_end_h: float | None = None,
    lum_norm_auc_do_cutoff: float | None = None,
) -> dict[str, float | int | str]:
    """Calculate instantaneous and integrated luminescence metrics.

    ``Lum_norm`` is used only for its instantaneous peak.  The integrated
    normalized metric is ``AUC(Lum_corr) / AUC(DO_corr - min(DO_corr))``.
    When an OD cutoff is supplied, both areas use the same curve, truncated at
    its first upward crossing with one linearly interpolated boundary point.
    """
    peak, peak_time = calculate_peak(series["temps_h"], series["Lum_norm"])
    finite_norm = np.isfinite(pd.to_numeric(series["Lum_norm"], errors="coerce"))
    time = pd.to_numeric(series["temps_h"], errors="coerce").to_numpy(float)
    od = pd.to_numeric(series["DO_corr"], errors="coerce").to_numpy(float)
    lum = pd.to_numeric(series["Lum_corr"], errors="coerce").to_numpy(float)
    start = float(np.nanmin(time)) if window_start_h is None else float(window_start_h)
    end = float(np.nanmax(time)) if window_end_h is None else float(window_end_h)
    in_window = ((time > start) | np.isclose(time, start)) & (
        (time < end) | np.isclose(time, end)
    )
    common = np.isfinite(time) & np.isfinite(od) & np.isfinite(lum) & in_window
    auc_time, auc_od, auc_lum = time[common], od[common], lum[common]
    order = np.argsort(auc_time, kind="stable")
    auc_time, auc_od, auc_lum = (
        values[order] for values in (auc_time, auc_od, auc_lum)
    )
    # Blank correction can make the first OD observations negative. Translate
    # the complete common-window OD curve so that its minimum is zero rather
    # than allowing those observations to subtract from the integrated biomass.
    auc_reason = ""
    integration_time, integration_od, integration_lum = auc_time, auc_od, auc_lum
    if lum_norm_auc_do_cutoff is not None:
        cutoff = float(lum_norm_auc_do_cutoff)
        crossings = np.flatnonzero(auc_od >= cutoff)
        crossing = int(crossings[0]) if len(crossings) else -1
        if crossing <= 0:
            # There is no observed below-threshold segment on which to
            # interpolate a crossing or integrate a meaningful interval.
            auc_reason = "do_cutoff_not_reached"
        else:
            t0, t1 = auc_time[crossing - 1], auc_time[crossing]
            od0, od1 = auc_od[crossing - 1], auc_od[crossing]
            lum0, lum1 = auc_lum[crossing - 1], auc_lum[crossing]
            if np.isclose(od1, cutoff):
                boundary_t, boundary_od, boundary_lum = t1, od1, lum1
            elif od1 > od0:
                fraction = (cutoff - od0) / (od1 - od0)
                boundary_t = t0 + fraction * (t1 - t0)
                boundary_od = cutoff
                boundary_lum = lum0 + fraction * (lum1 - lum0)
            else:
                auc_reason = "do_cutoff_not_reached"
            if not auc_reason:
                integration_time = np.append(auc_time[:crossing], boundary_t)
                integration_od = np.append(auc_od[:crossing], boundary_od)
                integration_lum = np.append(auc_lum[:crossing], boundary_lum)
    od_auc = calculate_baseline_shifted_auc(integration_time, integration_od)
    lum_auc = calculate_auc(integration_time, integration_lum)
    if not auc_reason and len(integration_time) < 2:
        auc_reason = "insufficient_auc_points"
    if not auc_reason and (not np.isfinite(od_auc) or od_auc <= 0):
        auc_reason = "non_positive_od_auc"
    norm_auc = lum_auc / od_auc if not auc_reason else np.nan
    covers_window = bool(
        len(auc_time) >= 2 and np.isclose(auc_time, start).any() and np.isclose(auc_time, end).any()
    )
    result = {"n_lum_norm_points": int(finite_norm.sum()),
              "n_lum_norm_auc_points": int(len(integration_time)) if not auc_reason else 0,
              "n_auc_points": int(common.sum()),
              "auc_covers_common_window": covers_window, "lum_norm_peak": peak,
              "lum_norm_peak_time_h": peak_time, "lum_norm_auc": norm_auc,
              "lum_norm_auc_reason": auc_reason,
              "lum_norm_auc_do_cutoff": lum_norm_auc_do_cutoff,
              "lum_corr_auc": lum_auc, "fixed_window_od_auc": od_auc,
              "effective_auc_window_end_h": (float(integration_time[-1])
                                              if len(integration_time) else np.nan)}
    if "Lum_corr" in series:
        corr_peak, corr_time = calculate_peak(series["temps_h"], series["Lum_corr"])
        result.update(lum_corr_peak=corr_peak, lum_corr_peak_time_h=corr_time)
    else:
        result.update(lum_corr_peak=np.nan, lum_corr_peak_time_h=np.nan)
    return result


def _series_group_columns(data: pd.DataFrame) -> list[str]:
    """Return the stable biological identity of a time series.

    A physical well is acquisition metadata, not a series identifier: split-file
    imports may move the same logical replicate to a different well at every
    time point.  ``sample_header`` (or the remaining biological fields) carries
    the stable identity instead.  The well column remains present in the input
    and public metric schema for traceability, but is not a split-file key.
    """
    preferred = (
        "biological_pair_id", "experience_id", "experience", "souche",
        "Groupe", "sample_header", "replicat",
    )
    columns = [column for column in preferred if column in data]
    # Endpoint imports are explicitly tagged by ``time_index`` and retain the
    # originating workbook.  In historical all-times-in-one-workbook tables,
    # distinct technical wells can legitimately share a header and must remain
    # distinct series, so preserve the legacy well key there.
    is_split_time_import = "time_index" in data and "source_workbook" in data
    if not is_split_time_import and "puits" in data:
        columns.append("puits")
    if not columns:
        raise ValueError("Impossible de construire une clé canonique de série.")
    return columns


def _experiment_identity_column(data: pd.DataFrame) -> str | None:
    """Return the first populated independent-experiment identifier."""
    for column in ("experience_id", "experience"):
        if column in data and data[column].notna().any():
            return column
    return None


def _common_auc_windows(
    strains: pd.DataFrame,
) -> tuple[str | None, dict[object, tuple[float, float]]]:
    """Return equal-duration AUC windows anchored at each experiment's first time.

    The common duration is the longest duration supported by every experiment:
    the complete acquisition span of the shortest experiment. With no
    ``experience_id``, the uploaded workbook name stored in legacy
    ``experience`` is used.  Without either identity, the sole dataset's
    complete span is used.
    """
    identity_column = _experiment_identity_column(strains)
    if identity_column is not None:
        bounds = strains.groupby(identity_column, dropna=False)["temps_h"].agg(["min", "max"])
    else:
        bounds = pd.DataFrame(
            {"min": [strains["temps_h"].min()], "max": [strains["temps_h"].max()]},
            index=[None],
        )
    common_duration = float((bounds["max"] - bounds["min"]).min())
    return identity_column, {
        experience: (float(row["min"]), float(row["min"] + common_duration))
        for experience, row in bounds.iterrows()
    }


def extract_series_kinetics(data: pd.DataFrame, growth_window_points: int = 3,
    minimum_auc_points: int = 2, growth_window_min_duration_h: float = 0.0,
    growth_rate_min_r_squared: float = 0.0,
    lum_norm_auc_do_cutoff: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not isinstance(minimum_auc_points, (int, np.integer)) or minimum_auc_points < 2:
        raise ValueError("minimum_auc_points doit être un entier supérieur ou égal à 2.")
    metrics, rejected, warnings = [], [], []
    strains = data.loc[data["type"].eq("souche")] if "type" in data else data
    auc_identity, auc_windows = _common_auc_windows(strains)
    for _, series in strains.groupby(_series_group_columns(data), dropna=False, sort=False):
        identity = {column: series[column].iloc[0] for column in IDENTITY_COLUMNS if column in series}
        window_start, window_end = auc_windows[identity.get(auc_identity) if auc_identity else None]
        if series["temps_h"].duplicated().any():
            warnings.append({**identity, "code": "duplicate_time",
                "message": "Temps dupliqué : les métriques globales sont conservées, aucune fenêtre ne traverse le doublon."})
        growth = calculate_growth_metrics(series, growth_window_points, growth_window_min_duration_h,
                                          growth_rate_min_r_squared)
        lum = calculate_luminescence_metrics(
            series, window_start, window_end, lum_norm_auc_do_cutoff
        )
        reasons = []
        if growth["n_growth_points"] < minimum_auc_points:
            reasons.append("insufficient_growth_points")
        if lum["n_auc_points"] < minimum_auc_points:
            reasons.append("insufficient_auc_points")
        if not lum["auc_covers_common_window"]:
            reasons.append("incomplete_common_auc_window")
        if not np.isfinite(lum["fixed_window_od_auc"]) or lum["fixed_window_od_auc"] <= 0:
            reasons.append("non_positive_od_auc")
        if reasons:
            rejected.append({**identity, "reason": ";".join(reasons), "n_points_total": len(series)})
            continue
        reason = growth["growth_rate_publishability_reason"]
        if reason:
            warnings.append({**identity, "code": reason,
                             "message": "Le taux de croissance et/ou le temps de doublement n'est pas publiable."})
        if lum["lum_norm_auc_reason"]:
            warnings.append({**identity, "code": lum["lum_norm_auc_reason"],
                             "message": ("L'AUC de luminescence normalisée n'est pas "
                                         "calculable sur la fenêtre demandée.")})
        finite_any = np.isfinite(series[["DO_corr", "Lum_norm"]].to_numpy(dtype=float)).all(axis=1)
        used_times = series.loc[finite_any, "temps_h"]
        growth["od_auc"] = lum["fixed_window_od_auc"]
        metrics.append({**identity, "n_points_total": len(series),
            "n_points_jointly_finite": int(finite_any.sum()),
            "analysis_start_h": float(used_times.min()) if len(used_times) else np.nan,
            "analysis_end_h": float(used_times.max()) if len(used_times) else np.nan,
            "growth_window_points": growth_window_points,
            "growth_window_min_duration_h": growth_window_min_duration_h,
            "growth_rate_min_r_squared": growth_rate_min_r_squared,
            "minimum_auc_points": minimum_auc_points,
            "auc_window_start_h": window_start,
            "auc_window_end_h": lum["effective_auc_window_end_h"],
            "auc_window_duration_h": lum["effective_auc_window_end_h"] - window_start,
            **growth,
            **{key: value for key, value in lum.items()
               if key not in {"auc_covers_common_window", "fixed_window_od_auc",
                              "effective_auc_window_end_h"}}})
    return (_table(metrics, SERIES_METRIC_COLUMNS), _table(rejected, REJECTED_SERIES_COLUMNS),
            _table(warnings, WARNING_COLUMNS))


def summarize_technical_replicates(series_metrics: pd.DataFrame) -> pd.DataFrame:
    """Summarize technical wells, retaining biological replicate identity."""
    group_columns = [column for column in
                     ("biological_pair_id", "experience_id", "experience", "souche", "Groupe", "replicat")
                     if column in series_metrics]
    rows = []
    for keys, group in series_metrics.groupby(group_columns, dropna=False, sort=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(group_columns, keys)); row["n_technical_series"] = len(group)
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(group[metric], errors="coerce")
            row.update({f"{metric}_mean": values.mean(), f"{metric}_sd": values.std(ddof=1),
                        f"{metric}_n": int(values.notna().sum())})
        rows.append(row)
    return _table(rows, TECHNICAL_SUMMARY_COLUMNS)


def run_kinetics(data: pd.DataFrame, growth_window_points: int = 3, minimum_auc_points: int = 2,
                 growth_window_min_duration_h: float = 0.0,
                 growth_rate_min_r_squared: float = 0.0,
                 lum_norm_auc_do_cutoff: float | None = None) -> KineticsResult:
    """Run the complete, non-mutating kinetic-parameter workflow."""
    if not np.isfinite(growth_window_min_duration_h) or growth_window_min_duration_h < 0:
        raise ValueError("growth_window_min_duration_h doit être positif ou nul.")
    if not np.isfinite(growth_rate_min_r_squared) or not 0 <= growth_rate_min_r_squared <= 1:
        raise ValueError("growth_rate_min_r_squared doit être compris entre 0 et 1.")
    if lum_norm_auc_do_cutoff is not None and (
        not np.isfinite(lum_norm_auc_do_cutoff) or lum_norm_auc_do_cutoff < 0
    ):
        raise ValueError("lum_norm_auc_do_cutoff doit être positif, nul ou None.")
    prepared = validate_kinetics_inputs(data, growth_window_points)
    metrics, rejected, warnings = extract_series_kinetics(prepared, growth_window_points,
        minimum_auc_points, growth_window_min_duration_h, growth_rate_min_r_squared,
        lum_norm_auc_do_cutoff)
    strain_summary = summarize_technical_replicates(metrics)
    summary = _table([
        ("series_total", len(metrics) + len(rejected)), ("series_analyzed", len(metrics)),
        ("series_rejected", len(rejected)), ("warnings", len(warnings)),
        ("growth_window_points", growth_window_points),
        ("growth_window_min_duration_h", growth_window_min_duration_h),
        ("growth_rate_min_r_squared", growth_rate_min_r_squared),
        ("minimum_auc_points", minimum_auc_points),
    ], SUMMARY_COLUMNS)
    return KineticsResult(metrics, strain_summary, rejected, warnings, summary)
