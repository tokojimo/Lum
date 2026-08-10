"""Orchestration helpers for the guided end-to-end Streamlit workflow."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .blanks import BlankCorrectionResult, run_blank_correction
from .kinetics import KineticsResult, run_kinetics
from .normalization import NormalizationResult, run_normalization
from .qc import DECISION_COLUMNS


@dataclass(frozen=True)
class CompleteAnalysisResult:
    blank_correction: BlankCorrectionResult
    normalization: NormalizationResult
    kinetics: KineticsResult


def filter_experiment_data(
    data: pd.DataFrame, strains: list[str], groups: list[str] | None = None
) -> pd.DataFrame:
    """Keep selected strains and automatically include their associated blanks.

    When ``groups`` is supplied, only the requested media are retained. Their
    associated blanks are included automatically.
    """
    if not strains:
        raise ValueError("Sélectionnez au moins une souche.")
    strain_rows = data["type"].astype(str).str.lower().eq("souche")
    selected_strains = data["souche"].astype(str).isin(strains)
    inferred_groups = data.loc[strain_rows & selected_strains, "Groupe"].astype(str).unique()
    if groups is not None:
        inferred_groups = [group for group in inferred_groups if group in set(map(str, groups))]
    if not len(inferred_groups):
        raise ValueError("La sélection ne contient aucune série de souche.")
    groups = list(inferred_groups)
    selected_groups = data["Groupe"].astype(str).isin(groups)
    required_blanks = data["type"].astype(str).str.lower().eq("blanc")
    result = data.loc[selected_groups & (selected_strains | required_blanks)].copy()
    if not result["type"].astype(str).str.lower().eq("souche").any():
        raise ValueError("La sélection ne contient aucune série de souche.")
    return result.reset_index(drop=True)


def build_manual_decisions(
    data: pd.DataFrame,
    point_indices: list[int] | None = None,
    series_headers: list[str] | None = None,
) -> pd.DataFrame:
    """Create explicit, auditable removal decisions from manual UI selections."""
    rows: list[dict[str, object]] = []
    point_indices = point_indices or []
    series_headers = series_headers or []
    for index in point_indices:
        if index not in data.index:
            continue
        point = data.loc[index]
        rows.append({
            "decision_id": f"manual-point|{point['sample_header']}|{point['temps_h']}|{index}",
            "scope": "point", "type": point.get("type", ""),
            "souche": point.get("souche", ""), "replicat": point.get("replicat", pd.NA),
            "sample_header": point.get("sample_header", ""), "puits": point.get("puits", ""),
            "temps_h": point.get("temps_h", pd.NA), "variable_cible": "both",
            "detection_type": "selection_manuelle", "motif_auto": "",
            "z_do": pd.NA, "z_lum": pd.NA, "n_points_aberrants_serie": pd.NA,
            "flags_serie": "", "decision_utilisateur": "exclure",
            "raison_utilisateur": "Sélection manuelle dans le workflow guidé",
        })
    for header in dict.fromkeys(series_headers):
        matches = data.loc[data["sample_header"].astype(str).eq(str(header))]
        if matches.empty:
            continue
        point = matches.iloc[0]
        rows.append({
            "decision_id": f"manual-series|{header}", "scope": "serie",
            "type": point.get("type", ""), "souche": point.get("souche", ""),
            "replicat": point.get("replicat", pd.NA), "sample_header": header,
            "puits": point.get("puits", ""), "temps_h": pd.NA,
            "variable_cible": "both", "detection_type": "selection_manuelle",
            "motif_auto": "", "z_do": pd.NA, "z_lum": pd.NA,
            "n_points_aberrants_serie": pd.NA, "flags_serie": "",
            "decision_utilisateur": "exclure",
            "raison_utilisateur": "Courbe entière sélectionnée dans le workflow guidé",
        })
    return pd.DataFrame(rows, columns=DECISION_COLUMNS)


def build_bulk_point_decisions(
    data: pd.DataFrame,
    *,
    experience: str,
    times: list[float],
    sample_type: str = "all",
) -> pd.DataFrame:
    """Create point decisions for a simple experiment/time/type selection.

    One decision is emitted per technical curve and selected observed time.  The
    resulting journal uses the same auditable format as selections made in the
    detailed table.
    """
    if sample_type not in {"all", "blanc", "souche"}:
        raise ValueError("sample_type doit valoir 'all', 'blanc' ou 'souche'.")
    experience_column = "experience" if "experience" in data else "experience_id"
    if experience_column not in data or not times:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    selected = data.loc[data[experience_column].astype(str).eq(str(experience))].copy()
    if sample_type != "all":
        selected = selected.loc[selected["type"].astype(str).str.lower().eq(sample_type)]
    numeric_times = pd.to_numeric(selected["temps_h"], errors="coerce")
    selected = selected.loc[numeric_times.apply(
        lambda value: pd.notna(value) and any(abs(float(value) - float(time)) <= 1e-6 for time in times)
    )]
    point_indices = selected.groupby(["sample_header", "temps_h"], sort=False).head(1).index.tolist()
    decisions = build_manual_decisions(data, point_indices, [])
    if not decisions.empty:
        decisions["detection_type"] = "selection_groupee"
        decisions["raison_utilisateur"] = (
            f"Exclusion groupée — expérience {experience}, type {sample_type}"
        )
    return decisions


def run_complete_analysis(
    data: pd.DataFrame,
    decisions: pd.DataFrame | None = None,
    *,
    blank_sd_multiplier: float = 3.0,
    minimum_od: float = 0.05,
    consecutive_points: int = 3,
    growth_window_points: int = 3,
    growth_window_min_duration_h: float = 0.0,
    growth_rate_min_r_squared: float = 0.0,
    minimum_auc_points: int = 2,
) -> CompleteAnalysisResult:
    """Run blank correction, normalization and kinetics with one explicit action."""
    correction = run_blank_correction(data, decisions)
    normalization = run_normalization(
        correction.corrected_data, blank_sd_multiplier, minimum_od, consecutive_points
    )
    kinetics = run_kinetics(
        normalization.normalized_data,
        growth_window_points=growth_window_points,
        minimum_auc_points=minimum_auc_points,
        growth_window_min_duration_h=growth_window_min_duration_h,
        growth_rate_min_r_squared=growth_rate_min_r_squared,
    )
    return CompleteAnalysisResult(correction, normalization, kinetics)
