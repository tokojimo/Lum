"""Reusable QC filtering and group-specific blank correction.

An observation is an atomic pair of OD and luminescence measurements.  Therefore
an exclusion aimed at ``DO`` or ``Lum`` removes the complete observation rather
than leaving a scientifically ambiguous half-row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "temps_h", "souche", "Groupe", "replicat", "DO_brute", "Lum_brute", "type"
)
EXCLUSION_DECISIONS = frozenset(
    {"drop", "exclude", "remove", "exclure", "supprimer", "discard", "delete", "retirer", "retire", "rejeter", "reject"}
)
WARNING_COLUMNS = ("code", "Groupe", "temps_h", "message")


@dataclass(frozen=True)
class BlankCorrectionResult:
    """All products of correction; each table is a new DataFrame."""

    retained_data: pd.DataFrame
    excluded_data: pd.DataFrame
    blank_profiles: pd.DataFrame
    corrected_data: pd.DataFrame
    warnings: pd.DataFrame
    summary: pd.DataFrame


def validate_blank_inputs(data: pd.DataFrame, decisions: pd.DataFrame | None = None) -> pd.DataFrame:
    """Validate and normalize a long table, returning a copy."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("Colonnes manquantes pour la correction des blancs : " + ", ".join(missing))
    if data.empty:
        raise ValueError("Le tableau à corriger est vide.")
    if decisions is not None and not decisions.empty:
        required_decisions = {"scope", "decision_utilisateur", "sample_header", "replicat"}
        missing_decisions = sorted(required_decisions.difference(decisions.columns))
        if missing_decisions:
            raise ValueError("Colonnes manquantes dans le journal QC : " + ", ".join(missing_decisions))

    result = data.copy(deep=True)
    for column in ("temps_h", "replicat", "DO_brute", "Lum_brute"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if result["temps_h"].isna().any() or result["replicat"].isna().any():
        raise ValueError("temps_h et replicat doivent contenir uniquement des valeurs numériques.")
    result["replicat"] = result["replicat"].astype("Int64")
    for column in ("souche", "Groupe", "type"):
        result[column] = result[column].fillna("").astype(str).str.strip()
    if result["Groupe"].eq("").any():
        raise ValueError("La colonne Groupe contient des valeurs vides.")
    result["type"] = result["type"].str.lower()
    if "sample_header" not in result:
        result["sample_header"] = result["souche"] + "_rep" + result["replicat"].astype(str)
    else:
        result["sample_header"] = result["sample_header"].fillna("").astype(str).str.strip()
    return result.reset_index(drop=True)


def _exclusion_mask(decisions: pd.DataFrame) -> pd.Series:
    values = decisions.get("decision_utilisateur", pd.Series("", index=decisions.index))
    return values.fillna("").astype(str).str.strip().str.lower().isin(EXCLUSION_DECISIONS)


def apply_qc_decisions(data: pd.DataFrame, decisions: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply explicit point/series exclusions and return retained/excluded copies.

    Targeting only OD or luminescence still removes the whole observation because
    both measurements belong to one plate-reader row.
    """
    working = data.copy(deep=True)
    excluded = pd.Series(False, index=working.index)
    if decisions is None or decisions.empty:
        return working.reset_index(drop=True), working.iloc[0:0].copy()

    for _, decision in decisions.loc[_exclusion_mask(decisions)].iterrows():
        scope = str(decision.get("scope", "")).strip().lower()
        if scope not in {"point", "serie"}:
            continue
        mask = working["sample_header"].eq(str(decision.get("sample_header", "")).strip())
        replicate = pd.to_numeric(pd.Series([decision.get("replicat")]), errors="coerce").iloc[0]
        if pd.notna(replicate):
            mask &= working["replicat"].eq(replicate)
        if scope == "point":
            time = pd.to_numeric(pd.Series([decision.get("temps_h")]), errors="coerce").iloc[0]
            if pd.isna(time):
                continue
            mask &= np.isclose(working["temps_h"].astype(float), float(time), rtol=0, atol=1e-6)
        excluded |= mask
    excluded_data = working.loc[excluded].copy().reset_index(drop=True)
    retained_data = working.loc[~excluded].copy().reset_index(drop=True)
    return retained_data, excluded_data


def associate_strains_with_blanks(data: pd.DataFrame) -> pd.DataFrame:
    """Annotate whether each strain has blank observations in its own group."""
    result = data.copy(deep=True)
    blank_groups = set(result.loc[result["type"].eq("blanc"), "Groupe"])
    result["blanc_disponible"] = result["Groupe"].isin(blank_groups)
    return result


def calculate_blank_profiles(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate blank statistics independently for every group and time."""
    blanks = data.loc[data["type"].eq("blanc")]
    columns = [
        "Groupe", "temps_h", "n_blancs", "DO_blanc_moyenne", "DO_blanc_sd",
        "DO_blanc_min", "DO_blanc_max", "Lum_blanc_moyenne", "Lum_blanc_sd",
        "Lum_blanc_min", "Lum_blanc_max",
    ]
    if blanks.empty:
        return pd.DataFrame(columns=columns)
    profiles = blanks.groupby(["Groupe", "temps_h"], dropna=False).agg(
        n_blancs=("DO_brute", "size"),
        DO_blanc_moyenne=("DO_brute", "mean"), DO_blanc_sd=("DO_brute", "std"),
        DO_blanc_min=("DO_brute", "min"), DO_blanc_max=("DO_brute", "max"),
        Lum_blanc_moyenne=("Lum_brute", "mean"), Lum_blanc_sd=("Lum_brute", "std"),
        Lum_blanc_min=("Lum_brute", "min"), Lum_blanc_max=("Lum_brute", "max"),
    ).reset_index()
    return profiles[columns].sort_values(["Groupe", "temps_h"], kind="stable").reset_index(drop=True)


def correct_blanks(data: pd.DataFrame, blank_profiles: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Correct every retained strain and blank against its group/time profile.

    Keeping corrected blanks is part of the public contract: the normalization
    step uses their residual OD distribution to derive its detection threshold.
    """
    observations = data.loc[data["type"].isin(["souche", "blanc"])].copy()
    corrected = observations.merge(blank_profiles, on=["Groupe", "temps_h"], how="left", validate="many_to_one")
    corrected["DO_corr"] = corrected["DO_brute"] - corrected["DO_blanc_moyenne"]
    corrected["Lum_corr"] = corrected["Lum_brute"] - corrected["Lum_blanc_moyenne"]
    missing = corrected.loc[
        corrected["type"].eq("souche")
        & corrected[["DO_blanc_moyenne", "Lum_blanc_moyenne"]].isna().any(axis=1),
        ["Groupe", "temps_h"],
    ].drop_duplicates()
    warnings = pd.DataFrame([
        {"code": "blank_missing", "Groupe": row.Groupe, "temps_h": row.temps_h,
         "message": f"Aucun blanc disponible pour le groupe {row.Groupe} au temps {row.temps_h}."}
        for row in missing.itertuples(index=False)
    ], columns=WARNING_COLUMNS)
    return corrected.reset_index(drop=True), warnings


def run_blank_correction(data: pd.DataFrame, decisions: pd.DataFrame | None = None) -> BlankCorrectionResult:
    """Validate, apply QC, derive profiles, and correct strains and blanks."""
    prepared = validate_blank_inputs(data, decisions)
    retained, excluded = apply_qc_decisions(prepared, decisions)
    retained = associate_strains_with_blanks(retained)
    profiles = calculate_blank_profiles(retained)
    corrected, warnings = correct_blanks(retained, profiles)
    summary = pd.DataFrame(
        [
            ("lignes_avant_qc", len(prepared)),
            ("lignes_exclues", len(excluded)),
            ("lignes_apres_qc", len(retained)),
            ("lignes_souches_corrigees", int(
                (corrected["type"].eq("souche") & corrected[["DO_corr", "Lum_corr"]].notna().all(axis=1)).sum()
            )),
            ("lignes_blancs_corrigees", int(
                (corrected["type"].eq("blanc") & corrected[["DO_corr", "Lum_corr"]].notna().all(axis=1)).sum()
            )),
            ("groupes_sans_blanc", int(warnings["Groupe"].nunique()) if not warnings.empty else 0),
        ],
        columns=["metrique", "valeur"],
    )
    return BlankCorrectionResult(retained, excluded, profiles, corrected, warnings, summary)
