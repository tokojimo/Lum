"""Reusable, non-destructive quality control for long kinetic tables."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "temps_h",
    "souche",
    "replicat",
    "DO_brute",
    "Lum_brute",
    "type",
)

DECISION_COLUMNS = (
    "decision_id",
    "scope",
    "type",
    "souche",
    "replicat",
    "sample_header",
    "puits",
    "temps_h",
    "variable_cible",
    "detection_type",
    "motif_auto",
    "z_do",
    "z_lum",
    "n_points_aberrants_serie",
    "flags_serie",
    "decision_utilisateur",
    "raison_utilisateur",
)


@dataclass(frozen=True)
class QCResult:
    """All QC products; the source observations are never removed or altered."""

    data: pd.DataFrame
    global_summary: pd.DataFrame
    series_summary: pd.DataFrame
    anomalies: pd.DataFrame
    decisions: pd.DataFrame


def validate_and_prepare(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the long schema and return a typed, consistently sorted copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes dans le tableau long : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau long est vide.")

    result = data.copy()
    for column in ("temps_h", "replicat", "DO_brute", "Lum_brute"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["souche"] = result["souche"].astype("string").str.strip()
    result["type"] = result["type"].astype("string").str.strip().str.lower()
    if result["temps_h"].isna().any():
        raise ValueError("La colonne temps_h contient des valeurs absentes ou non numériques.")
    if result["souche"].isna().any() or result["souche"].eq("").any():
        raise ValueError("La colonne souche contient des valeurs absentes ou vides.")
    if result["type"].isna().any() or result["type"].eq("").any():
        raise ValueError("La colonne type contient des valeurs absentes ou vides.")

    result["replicat"] = result["replicat"].astype("Int64")
    if "puits" not in result:
        result["puits"] = ""
    else:
        result["puits"] = result["puits"].fillna("").astype(str)
    if "sample_header" not in result:
        suffix = result["replicat"].astype(str)
        result["sample_header"] = result["souche"].astype(str) + "_rep" + suffix
    else:
        result["sample_header"] = result["sample_header"].fillna("").astype(str)
        empty = result["sample_header"].eq("")
        result.loc[empty, "sample_header"] = (
            result.loc[empty, "souche"].astype(str)
            + "_rep"
            + result.loc[empty, "replicat"].astype(str)
        )
    return result.sort_values(
        ["type", "souche", "sample_header", "temps_h"], kind="stable"
    ).reset_index(drop=True)


def robust_z_scores(values: pd.Series) -> pd.Series:
    """Return median/MAD z-scores, with sample z-scores as a zero-MAD fallback."""
    numeric = pd.to_numeric(values, errors="coerce")
    median = numeric.median(skipna=True)
    mad = (numeric - median).abs().median(skipna=True)
    if pd.isna(median):
        return pd.Series(np.nan, index=values.index, dtype=float)
    if pd.isna(mad) or mad == 0:
        std = numeric.std(ddof=1, skipna=True)
        if pd.isna(std) or std == 0:
            return pd.Series(np.nan, index=values.index, dtype=float)
        return (numeric - numeric.mean(skipna=True)) / std
    return 0.6745 * (numeric - median) / mad


def detect_anomalies(data: pd.DataFrame, z_threshold: float = 3.5) -> pd.DataFrame:
    """Propose point anomalies only where at least three technical series exist."""
    if z_threshold <= 0:
        raise ValueError("Le seuil z doit être strictement positif.")
    detected: list[pd.DataFrame] = []
    for _, block in data.groupby(["type", "souche", "temps_h"], dropna=False, sort=False):
        if block["sample_header"].nunique() < 3:
            continue
        block = block.copy()
        block["z_do"] = robust_z_scores(block["DO_brute"])
        block["z_lum"] = robust_z_scores(block["Lum_brute"])
        block["point_aberrant_DO"] = block["z_do"].abs() > z_threshold
        block["point_aberrant_Lum"] = block["z_lum"].abs() > z_threshold
        block["point_aberrant_any"] = block[
            ["point_aberrant_DO", "point_aberrant_Lum"]
        ].any(axis=1)
        detected.append(block.loc[block["point_aberrant_any"]])
    extra = ["z_do", "z_lum", "point_aberrant_DO", "point_aberrant_Lum", "point_aberrant_any"]
    if not detected:
        return pd.DataFrame(columns=[*data.columns, *extra])
    return pd.concat(detected, ignore_index=True).sort_values(
        ["type", "souche", "temps_h", "sample_header"], kind="stable"
    ).reset_index(drop=True)


def summarize_series(data: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Summarize completeness and elementary plausibility for every technical series."""
    theoretical_times = data["temps_h"].nunique()
    anomaly_counts = anomalies.groupby("sample_header").size() if not anomalies.empty else pd.Series(dtype=int)
    rows: list[dict[str, object]] = []
    keys = ["type", "souche", "replicat", "sample_header", "puits"]
    for key, block in data.groupby(keys, dropna=False, sort=False):
        block = block.sort_values("temps_h")
        n_points = len(block)
        unique_times = block["temps_h"].nunique()
        do_diff = block["DO_brute"].diff()
        lum_diff = block["Lum_brute"].diff()
        lum_std = lum_diff.std(skipna=True, ddof=0)
        values = {
            "n_temps_manquants": max(0, theoretical_times - unique_times),
            "n_temps_doubles": n_points - unique_times,
            "n_DO_NA": int(block["DO_brute"].isna().sum()),
            "n_Lum_NA": int(block["Lum_brute"].isna().sum()),
            "n_DO_neg": int(block["DO_brute"].lt(0).sum()),
            "n_Lum_neg": int(block["Lum_brute"].lt(0).sum()),
            "n_baisses_DO_importantes": int(do_diff.lt(-0.02).sum()),
            "n_grands_sauts_Lum": int(lum_diff.abs().gt(3 * lum_std).sum()) if lum_std > 0 else 0,
            "n_points_aberrants": int(anomaly_counts.get(key[3], 0)),
        }
        flags = []
        labels = {
            "n_temps_manquants": "temps_manquants",
            "n_temps_doubles": "temps_doubles",
            "n_DO_NA": "DO_NA",
            "n_Lum_NA": "Lum_NA",
            "n_DO_neg": "DO_neg",
        }
        flags.extend(f"{label}={values[name]}" for name, label in labels.items() if values[name])
        if key[0] == "souche" and n_points and values["n_Lum_neg"] / n_points >= 0.25:
            flags.append(f"Lum_neg_frequentes={values['n_Lum_neg']}")
        if values["n_baisses_DO_importantes"] >= 3:
            flags.append(f"baisses_DO={values['n_baisses_DO_importantes']}")
        if values["n_points_aberrants"] >= 3:
            flags.append(f"points_aberrants={values['n_points_aberrants']}")
        rows.append(dict(zip(keys, key)) | {
            "n_points": n_points,
            "n_temps_uniques": unique_times,
            "n_temps_theorique": theoretical_times,
            "DO_min": block["DO_brute"].min(), "DO_max": block["DO_brute"].max(),
            "Lum_min": block["Lum_brute"].min(), "Lum_max": block["Lum_brute"].max(),
            **values, "flags": "; ".join(flags) if flags else "OK",
        })
    return pd.DataFrame(rows)


def summarize_global(data: pd.DataFrame, series: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Build a two-column global QC dashboard table."""
    times = np.sort(data["temps_h"].unique())
    rows = [
        ("n_lignes", len(data)), ("n_souches_uniques_total", data["souche"].nunique()),
        ("n_series_total", len(series)), ("n_series_flaggees", series["flags"].ne("OK").sum()),
        ("n_points_aberrants", len(anomalies)), ("n_temps_uniques", len(times)),
        ("pas_median_h", float(np.median(np.diff(times))) if len(times) > 1 else np.nan),
        ("temps_min_h", float(times.min())), ("temps_max_h", float(times.max())),
        ("n_DO_NA_total", data["DO_brute"].isna().sum()),
        ("n_Lum_NA_total", data["Lum_brute"].isna().sum()),
        ("n_DO_neg_total", data["DO_brute"].lt(0).sum()),
        ("n_Lum_neg_total", data["Lum_brute"].lt(0).sum()),
    ]
    return pd.DataFrame(rows, columns=["indicateur", "valeur"])


def build_decision_log(anomalies: pd.DataFrame, series: pd.DataFrame) -> pd.DataFrame:
    """Create review proposals. Every generated decision is explicitly ``review``."""
    rows: list[dict[str, object]] = []
    lookup = series.set_index("sample_header") if not series.empty else pd.DataFrame()
    for _, point in anomalies.iterrows():
        do_flag, lum_flag = bool(point["point_aberrant_DO"]), bool(point["point_aberrant_Lum"])
        target = "both" if do_flag and lum_flag else ("DO" if do_flag else "Lum")
        header = point["sample_header"]
        info = lookup.loc[header] if header in lookup.index else {}
        rows.append({
            "decision_id": f"point|{header}|{point['temps_h']:g}|{target}", "scope": "point",
            **{column: point.get(column, "") for column in ("type", "souche", "replicat", "sample_header", "puits", "temps_h")},
            "variable_cible": target, "detection_type": "point_aberrant",
            "motif_auto": "; ".join(name for name, flag in (("point_aberrant_DO", do_flag), ("point_aberrant_Lum", lum_flag)) if flag),
            "z_do": point["z_do"], "z_lum": point["z_lum"],
            "n_points_aberrants_serie": info.get("n_points_aberrants", np.nan),
            "flags_serie": info.get("flags", ""), "decision_utilisateur": "review", "raison_utilisateur": "",
        })
    for _, item in series.loc[series["flags"].ne("OK")].iterrows():
        rows.append({
            "decision_id": f"serie|{item['sample_header']}", "scope": "serie",
            **{column: item.get(column, "") for column in ("type", "souche", "replicat", "sample_header", "puits")},
            "temps_h": np.nan, "variable_cible": "both", "detection_type": "serie_a_inspecter",
            "motif_auto": item["flags"], "z_do": np.nan, "z_lum": np.nan,
            "n_points_aberrants_serie": item["n_points_aberrants"], "flags_serie": item["flags"],
            "decision_utilisateur": "review", "raison_utilisateur": "",
        })
    return pd.DataFrame(rows, columns=DECISION_COLUMNS).drop_duplicates("decision_id")


def run_quality_control(data: pd.DataFrame, z_threshold: float = 3.5) -> QCResult:
    """Run the complete proposal-only QC workflow."""
    prepared = validate_and_prepare(data)
    anomalies = detect_anomalies(prepared, z_threshold)
    series = summarize_series(prepared, anomalies)
    summary = summarize_global(prepared, series, anomalies)
    return QCResult(prepared, summary, series, anomalies, build_decision_log(anomalies, series))
