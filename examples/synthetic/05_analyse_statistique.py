#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
05_analyse_statistique.py

Objectif
--------
Extraire des paramètres cinétiques à partir du fichier normalisé produit par
04_normalisation_DO.py, puis résumer proprement les résultats :
    1. par puits / série (réplicat technique)
    2. par souche au sein de chaque expérience indépendante
    3. par souche à travers les expériences indépendantes

Le script est pensé pour deux usages :
- maintenant : une seule plaque / une seule expérience -> analyse descriptive
- plus tard : plusieurs expériences indépendantes -> mêmes sorties + tests
  exploratoires au niveau des expériences indépendantes

Philosophie statistique
-----------------------
- Les paramètres sont extraits par courbe, pas testés temps par temps.
- Les réplicats techniques (puits) servent au résumé intra-plaque.
- Les tests entre souches ne sont lancés que si plusieurs expériences
  indépendantes sont disponibles.

Paramètres extraits par puits
-----------------------------
- AUC_Lum_norm
- Lum_norm_max
- temps_pic_h
- Lum_norm_final
- AUC_DO
- DO_max
- temps_DO_max_h
- temps_debut_norm_h
- temps_fin_norm_h

Usage
-----
python 05_analyse_statistique.py "exp1_normalise_DO.csv"
python 05_analyse_statistique.py "exp1_normalise_DO.csv" "exp2_normalise_DO.csv" "exp3_normalise_DO.csv"
python 05_analyse_statistique.py "exp1_normalise_DO.csv" "exp2_normalise_DO.csv" --experience-ids manip1 manip2
python 05_analyse_statistique.py "..._normalise_DO.csv" --experience-col experience_id
python 05_analyse_statistique.py "..._normalise_DO.csv" --min-bio-reps 2
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


COLONNES_OBLIGATOIRES = [
    "temps_h",
    "souche",
    "type",
    "DO_corr",
]

COLONNES_NUMERIQUES = [
    "temps_h",
    "replicat",
    "lecture",
    "DO_brute",
    "Lum_brute",
    "DO_blanc_moy",
    "Lum_blanc_moy",
    "DO_blanc_sd",
    "Lum_blanc_sd",
    "n_blancs",
    "n_lignes_blanc",
    "DO_corr",
    "Lum_corr",
    "Lum_norm",
    "do_threshold_utilise",
]

COLONNES_TEXTE = [
    "souche",
    "Groupe",
    "type",
    "puits",
    "sample_header",
    "blanc_associe",
    "normalisation_ok",
    "raison_exclusion_norm",
]

METRICS_PUITS = [
    "AUC_Lum_corr",
    "Lum_corr_max",
    "temps_pic_Lum_corr_h",
    "AUC_Lum_norm",
    "Lum_norm_max",
    "temps_pic_h",
    "Lum_norm_final",
    "AUC_DO",
    "ratio_AUC_Lum_corr_sur_AUC_DO",
    "DO_max",
    "temps_DO_max_h",
    "temps_debut_norm_h",
    "temps_fin_norm_h",
    "duree_normalisee_h",
]

METRICS_TESTS = [
    "AUC_Lum_corr_moy",
    "Lum_corr_max_moy",
    "temps_pic_Lum_corr_h_moy",
    "AUC_Lum_norm_moy",
    "Lum_norm_max_moy",
    "temps_pic_h_moy",
    "Lum_norm_final_moy",
    "AUC_DO_moy",
    "ratio_AUC_Lum_corr_sur_AUC_DO_moy",
    "DO_max_moy",
]


# -----------------------------------------------------------------------------
# Utilitaires généraux
# -----------------------------------------------------------------------------

def lire_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffixe = path.suffix.lower()
    if suffixe == ".csv":
        return pd.read_csv(path)
    if suffixe in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Format non supporté : {path.suffix}")


def nettoyer_nom_fichier(texte: object) -> str:
    texte = str(texte)
    texte = re.sub(r"[^A-Za-z0-9._-]+", "_", texte).strip("_")
    return texte or "sans_nom"


def cleaner_basename(path: Path) -> str:
    return nettoyer_nom_fichier(path.stem.replace("_normalise_DO", ""))


def verifier_colonnes(df: pd.DataFrame, colonnes: Iterable[str]) -> None:
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))


def inferer_output_dir(input_paths: Sequence[Path], output_dir_arg: Path | None) -> Path:
    if output_dir_arg is not None:
        out = output_dir_arg.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if not input_paths:
        raise ValueError("Aucun fichier d'entrée fourni.")

    if len(input_paths) == 1:
        input_path = input_paths[0]
        parent_name = input_path.parent.name
        if parent_name.startswith("NORM_"):
            out = (input_path.parent.parent / parent_name.replace("NORM_", "STAT_", 1)).resolve()
        else:
            base_name = input_path.stem.replace("_normalise_DO", "")
            out = (input_path.parent / f"STAT_{base_name}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    import os
    if all(p.parent.name.startswith("NORM_") for p in input_paths):
        out_root = input_paths[0].parent.parent
    else:
        out_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))

    stems = [cleaner_basename(p) for p in input_paths]
    if len(stems) <= 3:
        suffix = "__".join(stems)
    else:
        suffix = f"{stems[0]}__plus_{len(stems)-1}_fichiers"
    out = (out_root / f"STAT_multi_{suffix}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out

def preparer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in COLONNES_NUMERIQUES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in COLONNES_TEXTE:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str).str.strip()

    if "type" in df.columns:
        df["type"] = df["type"].str.lower()

    if "normalisation_ok" in df.columns:
        df["normalisation_ok"] = df["normalisation_ok"].astype(str).str.strip().str.lower()
        df["normalisation_ok"] = df["normalisation_ok"].isin(["true", "1", "oui", "yes"])
    else:
        df["normalisation_ok"] = df["Lum_norm"].notna() if "Lum_norm" in df.columns else False

    if "sample_header" not in df.columns:
        if "puits" in df.columns:
            df["sample_header"] = df["souche"].astype(str) + " (" + df["puits"].astype(str) + ")"
        else:
            rep_txt = df["replicat"].astype("Int64").astype(str) if "replicat" in df.columns else "NA"
            df["sample_header"] = df["souche"].astype(str) + "_rep" + rep_txt

    if "puits" not in df.columns:
        df["puits"] = ""
    if "replicat" not in df.columns:
        df["replicat"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    else:
        df["replicat"] = pd.to_numeric(df["replicat"], errors="coerce").astype("Int64")

    df = df.sort_values([c for c in ["souche", "sample_header", "temps_h"] if c in df.columns]).reset_index(drop=True)
    return df


def ajouter_experience_id(df: pd.DataFrame, experience_col: str, default_experience_id: str) -> pd.DataFrame:
    df = df.copy()
    if experience_col in df.columns:
        df[experience_col] = df[experience_col].fillna("").astype(str).str.strip()
        df.loc[df[experience_col].eq(""), experience_col] = default_experience_id
    else:
        df[experience_col] = default_experience_id
    return df


def charger_entrees_multiples(
    input_files: Sequence[Path],
    experience_col: str,
    experience_ids: Sequence[str] | None,
    fallback_prefix: str,
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    if not input_files:
        raise ValueError("Aucun fichier d'entrée fourni.")

    resolved_paths: list[Path] = []
    frames: list[pd.DataFrame] = []
    assigned_ids: list[str] = []

    if experience_ids is not None and len(experience_ids) != len(input_files):
        raise ValueError(
            "Le nombre de valeurs passées à --experience-ids doit correspondre au nombre de fichiers d'entrée."
        )

    seen_ids: set[str] = set()

    for i, raw_path in enumerate(input_files, start=1):
        path = raw_path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Fichier d'entrée introuvable : {path}")

        df = lire_table(path)
        verifier_colonnes(df, COLONNES_OBLIGATOIRES)
        df = preparer_df(df)

        if experience_ids is not None:
            exp_id = str(experience_ids[i - 1]).strip()
        elif len(input_files) == 1:
            exp_id = fallback_prefix
        else:
            exp_id = cleaner_basename(path)
            if not exp_id:
                exp_id = f"exp{i}"

        if exp_id in seen_ids:
            exp_id = f"{exp_id}__{i}"
        seen_ids.add(exp_id)

        if len(input_files) > 1:
            df[experience_col] = exp_id
        else:
            df = ajouter_experience_id(df, experience_col, exp_id)
        df["experience_id"] = df[experience_col]

        resolved_paths.append(path)
        frames.append(df)
        assigned_ids.append(exp_id)

    df_all = pd.concat(frames, ignore_index=True)
    return df_all, resolved_paths, assigned_ids


def auc_trapezoid(x: Sequence[float], y: Sequence[float]) -> float:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < 2:
        return np.nan
    order = np.argsort(x_arr)
    x_arr = x_arr[order]
    y_arr = y_arr[order]
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_arr, x_arr))
    return float(np.trapz(y_arr, x_arr))


def first_time_of_max(df: pd.DataFrame, value_col: str, time_col: str = "temps_h") -> float:
    sub = df[[time_col, value_col]].dropna().copy()
    if sub.empty:
        return np.nan
    vmax = sub[value_col].max()
    sub = sub.loc[sub[value_col] == vmax].sort_values(time_col)
    if sub.empty:
        return np.nan
    return float(sub.iloc[0][time_col])


# -----------------------------------------------------------------------------
# Extraction des paramètres par puits
# -----------------------------------------------------------------------------

def extraire_parametres_serie(sub: pd.DataFrame) -> dict[str, object]:
    sub = sub.sort_values("temps_h").copy()
    meta = {
        "experience_id": sub["experience_id"].iloc[0] if "experience_id" in sub.columns else "exp1",
        "souche": sub["souche"].iloc[0] if "souche" in sub.columns else "",
        "sample_header": sub["sample_header"].iloc[0] if "sample_header" in sub.columns else "",
        "puits": sub["puits"].iloc[0] if "puits" in sub.columns else "",
        "replicat": sub["replicat"].iloc[0] if "replicat" in sub.columns else pd.NA,
        "Groupe": sub["Groupe"].iloc[0] if "Groupe" in sub.columns else "",
        "blanc_associe": sub["blanc_associe"].iloc[0] if "blanc_associe" in sub.columns else "",
        "n_points_total": int(len(sub)),
    }

    # DO corrigée : on travaille sur les souches uniquement.
    do_sub = sub.loc[(sub["type"] == "souche") & sub["DO_corr"].notna(), ["temps_h", "DO_corr"]].copy()
    meta["n_points_DO"] = int(len(do_sub))
    if not do_sub.empty:
        meta["AUC_DO"] = auc_trapezoid(do_sub["temps_h"], do_sub["DO_corr"])
        meta["DO_max"] = float(do_sub["DO_corr"].max())
        meta["temps_DO_max_h"] = first_time_of_max(do_sub, "DO_corr")
        meta["temps_debut_DO_h"] = float(do_sub["temps_h"].min())
        meta["temps_fin_DO_h"] = float(do_sub["temps_h"].max())
    else:
        meta["AUC_DO"] = np.nan
        meta["DO_max"] = np.nan
        meta["temps_DO_max_h"] = np.nan
        meta["temps_debut_DO_h"] = np.nan
        meta["temps_fin_DO_h"] = np.nan

    # Luminescence corrigée : utile comme lecture principale et pour un ratio d'AUC plus stable.
    lum_corr_sub = sub.loc[(sub["type"] == "souche") & sub["Lum_corr"].notna(), ["temps_h", "Lum_corr"]].copy() if "Lum_corr" in sub.columns else pd.DataFrame(columns=["temps_h", "Lum_corr"])
    meta["n_points_Lum_corr"] = int(len(lum_corr_sub))
    if not lum_corr_sub.empty:
        meta["AUC_Lum_corr"] = auc_trapezoid(lum_corr_sub["temps_h"], lum_corr_sub["Lum_corr"])
        meta["Lum_corr_max"] = float(lum_corr_sub["Lum_corr"].max())
        meta["temps_pic_Lum_corr_h"] = first_time_of_max(lum_corr_sub, "Lum_corr")
    else:
        meta["AUC_Lum_corr"] = np.nan
        meta["Lum_corr_max"] = np.nan
        meta["temps_pic_Lum_corr_h"] = np.nan

    # Luminescence normalisée : seulement les points effectivement normalisés.
    if "Lum_norm" in sub.columns:
        lum_sub = sub.loc[
            (sub["type"] == "souche") &
            sub["Lum_norm"].notna() &
            sub["normalisation_ok"],
            ["temps_h", "Lum_norm"]
        ].copy()
    else:
        lum_sub = pd.DataFrame(columns=["temps_h", "Lum_norm"])

    meta["n_points_Lum_norm"] = int(len(lum_sub))
    if not lum_sub.empty:
        meta["AUC_Lum_norm"] = auc_trapezoid(lum_sub["temps_h"], lum_sub["Lum_norm"])
        meta["Lum_norm_max"] = float(lum_sub["Lum_norm"].max())
        meta["temps_pic_h"] = first_time_of_max(lum_sub, "Lum_norm")
        meta["Lum_norm_final"] = float(lum_sub.sort_values("temps_h").iloc[-1]["Lum_norm"])
        meta["temps_debut_norm_h"] = float(lum_sub["temps_h"].min())
        meta["temps_fin_norm_h"] = float(lum_sub["temps_h"].max())
        meta["duree_normalisee_h"] = float(meta["temps_fin_norm_h"] - meta["temps_debut_norm_h"])
    else:
        meta["AUC_Lum_norm"] = np.nan
        meta["Lum_norm_max"] = np.nan
        meta["temps_pic_h"] = np.nan
        meta["Lum_norm_final"] = np.nan
        meta["temps_debut_norm_h"] = np.nan
        meta["temps_fin_norm_h"] = np.nan
        meta["duree_normalisee_h"] = np.nan

    if pd.notna(meta.get("AUC_Lum_corr")) and pd.notna(meta.get("AUC_DO")) and float(meta["AUC_DO"]) > 0:
        meta["ratio_AUC_Lum_corr_sur_AUC_DO"] = float(meta["AUC_Lum_corr"] / meta["AUC_DO"])
    else:
        meta["ratio_AUC_Lum_corr_sur_AUC_DO"] = np.nan

    return meta


def construire_parametres_par_puits(df: pd.DataFrame) -> pd.DataFrame:
    souches = df.loc[df["type"] == "souche"].copy()
    if souches.empty:
        raise ValueError("Aucune ligne de type 'souche' trouvée dans le fichier d'entrée.")

    group_cols = ["experience_id", "souche", "sample_header"]
    rows = []
    for _, sub in souches.groupby(group_cols, dropna=False):
        rows.append(extraire_parametres_serie(sub))

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["experience_id", "souche", "replicat", "sample_header"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


# -----------------------------------------------------------------------------
# Résumés descriptifs
# -----------------------------------------------------------------------------

def _safe_stat(series: pd.Series, mode: str) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    if valid.empty:
        return np.nan
    if mode == "mean":
        return float(valid.mean())
    if mode == "sd":
        return float(valid.std(ddof=1)) if len(valid) >= 2 else np.nan
    if mode == "median":
        return float(valid.median())
    if mode == "min":
        return float(valid.min())
    if mode == "max":
        return float(valid.max())
    raise ValueError(f"Mode statistique inconnu : {mode}")


def construire_resume_par_experience_souche(param_puits: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (exp_id, souche), sub in param_puits.groupby(["experience_id", "souche"], dropna=False):
        row: dict[str, object] = {
            "experience_id": exp_id,
            "souche": souche,
            "n_puits_techniques": int(len(sub)),
            "puits_ids": " | ".join(sorted(sub["sample_header"].dropna().astype(str).unique())),
        }
        for metric in METRICS_PUITS:
            if metric not in sub.columns:
                continue
            row[f"{metric}_moy"] = _safe_stat(sub[metric], "mean")
            row[f"{metric}_sd"] = _safe_stat(sub[metric], "sd")
            row[f"{metric}_mediane"] = _safe_stat(sub[metric], "median")
            row[f"{metric}_min"] = _safe_stat(sub[metric], "min")
            row[f"{metric}_max"] = _safe_stat(sub[metric], "max")
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values(["experience_id", "souche"]).reset_index(drop=True)


def construire_resume_par_souche(resume_exp: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for souche, sub in resume_exp.groupby("souche", dropna=False):
        row: dict[str, object] = {
            "souche": souche,
            "n_experiences": int(sub["experience_id"].nunique()),
            "n_lignes_experience": int(len(sub)),
            "experiences_ids": " | ".join(sorted(sub["experience_id"].dropna().astype(str).unique())),
        }
        for metric in METRICS_TESTS:
            if metric not in sub.columns:
                continue
            row[f"{metric}_moyenne_inter_exp"] = _safe_stat(sub[metric], "mean")
            row[f"{metric}_sd_inter_exp"] = _safe_stat(sub[metric], "sd")
            row[f"{metric}_mediane_inter_exp"] = _safe_stat(sub[metric], "median")
            row[f"{metric}_min_inter_exp"] = _safe_stat(sub[metric], "min")
            row[f"{metric}_max_inter_exp"] = _safe_stat(sub[metric], "max")
        rows.append(row)

    out = pd.DataFrame(rows)
    return out.sort_values("souche").reset_index(drop=True)


# -----------------------------------------------------------------------------
# Tests exploratoires au niveau des expériences indépendantes
# -----------------------------------------------------------------------------

def holm_correction(pvalues: Sequence[float]) -> list[float]:
    pvals = np.asarray(list(pvalues), dtype=float)
    n = len(pvals)
    if n == 0:
        return []
    order = np.argsort(pvals)
    ranked = pvals[order]
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for i, p in enumerate(ranked):
        adj = min((n - i) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[i] = running_max
    back = np.empty(n, dtype=float)
    back[order] = adjusted
    return back.tolist()


def lancer_tests_exploratoires(
    resume_exp: pd.DataFrame,
    min_bio_reps: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    notes: list[str] = []

    if stats is None:
        notes.append("Scipy indisponible : tests exploratoires non réalisés.")
        return pd.DataFrame(), pd.DataFrame(), notes

    n_exp_total = int(resume_exp["experience_id"].nunique()) if not resume_exp.empty else 0
    if n_exp_total < 2:
        notes.append(
            "Une seule expérience indépendante détectée : tests entre souches non réalisés. "
            "Les sorties restent descriptives."
        )
        return pd.DataFrame(), pd.DataFrame(), notes

    global_rows: list[dict[str, object]] = []
    posthoc_rows: list[dict[str, object]] = []

    for metric in METRICS_TESTS:
        if metric not in resume_exp.columns:
            continue

        dat = resume_exp[["souche", "experience_id", metric]].copy()
        dat[metric] = pd.to_numeric(dat[metric], errors="coerce")
        dat = dat.dropna(subset=[metric])
        if dat.empty:
            global_rows.append({
                "metrique": metric,
                "test_global": "kruskal",
                "n_souches_eligibles": 0,
                "n_experiences_total": n_exp_total,
                "statistique": np.nan,
                "p_value": np.nan,
                "conclusion": "aucune_valeur_non_manquante",
            })
            continue

        counts = dat.groupby("souche")[metric].size().sort_index()
        eligibles = counts[counts >= min_bio_reps].index.tolist()
        dat_eligible = dat[dat["souche"].isin(eligibles)].copy()

        if len(eligibles) < 2:
            global_rows.append({
                "metrique": metric,
                "test_global": "kruskal",
                "n_souches_eligibles": len(eligibles),
                "n_experiences_total": n_exp_total,
                "statistique": np.nan,
                "p_value": np.nan,
                "conclusion": f"insuffisant_pour_test_global_min_bio_reps_{min_bio_reps}",
            })
            continue

        groups = [
            dat_eligible.loc[dat_eligible["souche"] == souche, metric].to_numpy(dtype=float)
            for souche in eligibles
        ]

        try:
            stat_kruskal, p_kruskal = stats.kruskal(*groups, nan_policy="omit")
        except Exception as exc:
            global_rows.append({
                "metrique": metric,
                "test_global": "kruskal",
                "n_souches_eligibles": len(eligibles),
                "n_experiences_total": n_exp_total,
                "statistique": np.nan,
                "p_value": np.nan,
                "conclusion": f"erreur_kruskal: {exc}",
            })
            continue

        conclusion = "significatif" if p_kruskal < 0.05 else "non_significatif"
        global_rows.append({
            "metrique": metric,
            "test_global": "kruskal",
            "n_souches_eligibles": len(eligibles),
            "n_experiences_total": n_exp_total,
            "statistique": float(stat_kruskal),
            "p_value": float(p_kruskal),
            "conclusion": conclusion,
        })

        # Comparaisons par paires : Mann-Whitney U + Holm
        pair_results: list[dict[str, object]] = []
        for souche_a, souche_b in itertools.combinations(eligibles, 2):
            va = dat_eligible.loc[dat_eligible["souche"] == souche_a, metric].to_numpy(dtype=float)
            vb = dat_eligible.loc[dat_eligible["souche"] == souche_b, metric].to_numpy(dtype=float)
            if len(va) < min_bio_reps or len(vb) < min_bio_reps:
                continue
            try:
                stat_u, p_u = stats.mannwhitneyu(va, vb, alternative="two-sided")
            except Exception:
                stat_u, p_u = np.nan, np.nan
            pair_results.append({
                "metrique": metric,
                "test_pairwise": "mannwhitneyu",
                "souche_a": souche_a,
                "souche_b": souche_b,
                "n_exp_a": int(len(va)),
                "n_exp_b": int(len(vb)),
                "statistique": float(stat_u) if pd.notna(stat_u) else np.nan,
                "p_value_brute": float(p_u) if pd.notna(p_u) else np.nan,
            })

        if pair_results:
            pvals = [row["p_value_brute"] for row in pair_results]
            adj = holm_correction(pvals)
            for row, p_adj in zip(pair_results, adj):
                row["p_value_holm"] = p_adj
                row["conclusion"] = "significatif" if pd.notna(p_adj) and p_adj < 0.05 else "non_significatif"
                posthoc_rows.append(row)

    return pd.DataFrame(global_rows), pd.DataFrame(posthoc_rows), notes


# -----------------------------------------------------------------------------
# Rapport texte
# -----------------------------------------------------------------------------

def construire_resume_global(
    df: pd.DataFrame,
    param_puits: pd.DataFrame,
    resume_exp: pd.DataFrame,
    tests_globaux: pd.DataFrame,
    notes_tests: list[str],
    min_bio_reps: int,
) -> pd.DataFrame:
    lignes = []

    def add(metric: str, value: object) -> None:
        lignes.append({"metrique": metric, "valeur": value})

    add("lignes_entree", len(df))
    add("souches_uniques", int(df.loc[df["type"] == "souche", "souche"].nunique()))
    add("series_uniques", int(param_puits["sample_header"].nunique()) if not param_puits.empty else 0)
    add("experiences_uniques", int(df["experience_id"].nunique()) if "experience_id" in df.columns else 0)
    add("puits_avec_Lum_norm", int(param_puits["n_points_Lum_norm"].gt(0).sum()) if "n_points_Lum_norm" in param_puits.columns else 0)
    add("puits_sans_Lum_norm", int(param_puits["n_points_Lum_norm"].eq(0).sum()) if "n_points_Lum_norm" in param_puits.columns else 0)
    add("min_bio_reps_pour_tests", min_bio_reps)

    if "do_threshold_utilise" in df.columns:
        seuils = pd.to_numeric(df["do_threshold_utilise"], errors="coerce").dropna().unique()
        if len(seuils) == 1:
            add("do_threshold_utilise", float(seuils[0]))
        elif len(seuils) > 1:
            add("do_threshold_utilise", "multiple")

    if not tests_globaux.empty:
        add("tests_globaux_realises", int(len(tests_globaux)))
        add("tests_globaux_significatifs", int((tests_globaux["conclusion"] == "significatif").sum()))
    else:
        add("tests_globaux_realises", 0)
        add("tests_globaux_significatifs", 0)

    for idx, note in enumerate(notes_tests, start=1):
        add(f"note_test_{idx}", note)

    return pd.DataFrame(lignes)


def ecrire_rapport_txt(
    path: Path,
    df: pd.DataFrame,
    param_puits: pd.DataFrame,
    resume_exp: pd.DataFrame,
    resume_souche: pd.DataFrame,
    tests_globaux: pd.DataFrame,
    posthoc: pd.DataFrame,
    notes_tests: list[str],
    min_bio_reps: int,
) -> None:
    n_souches = int(df.loc[df["type"] == "souche", "souche"].nunique())
    n_exp = int(df["experience_id"].nunique()) if "experience_id" in df.columns else 0
    n_series = int(len(param_puits))

    lines = []
    lines.append("ANALYSE STATISTIQUE - RESUME")
    lines.append("=" * 72)
    lines.append(f"Lignes en entrée : {len(df)}")
    lines.append(f"Souches uniques : {n_souches}")
    lines.append(f"Expériences indépendantes détectées : {n_exp}")
    lines.append(f"Séries / puits analysés : {n_series}")
    lines.append("")
    lines.append("Approche")
    lines.append("- Extraction de paramètres par puits / réplicat technique.")
    lines.append("- Résumé intra-expérience par souche.")
    lines.append("- Tests entre souches uniquement au niveau des expériences indépendantes.")
    lines.append(f"- Seuil minimum pour lancer les tests : {min_bio_reps} expériences par souche.")
    lines.append("")

    if notes_tests:
        lines.append("Notes sur les tests")
        for note in notes_tests:
            lines.append(f"- {note}")
        lines.append("")

    if not resume_exp.empty:
        lines.append("Résumé principal par expérience et par souche")
        preview_cols = [c for c in [
            "experience_id", "souche", "n_puits_techniques",
            "AUC_Lum_corr_moy", "ratio_AUC_Lum_corr_sur_AUC_DO_moy",
            "AUC_Lum_norm_moy", "Lum_norm_max_moy", "temps_pic_h_moy",
            "AUC_DO_moy", "DO_max_moy"
        ] if c in resume_exp.columns]
        if preview_cols:
            lines.append(resume_exp[preview_cols].to_string(index=False))
            lines.append("")

    if not tests_globaux.empty:
        lines.append("Tests globaux")
        lines.append(tests_globaux.to_string(index=False))
        lines.append("")

    if not posthoc.empty:
        lines.append("Comparaisons post-hoc exploratoires")
        lines.append(posthoc.to_string(index=False))
        lines.append("")

    if not resume_souche.empty:
        lines.append("Résumé inter-expériences par souche")
        preview_cols = [c for c in [
            "souche", "n_experiences",
            "AUC_Lum_corr_moy_moyenne_inter_exp",
            "ratio_AUC_Lum_corr_sur_AUC_DO_moy_moyenne_inter_exp",
            "AUC_Lum_norm_moy_moyenne_inter_exp",
            "Lum_norm_max_moy_moyenne_inter_exp",
            "temps_pic_h_moy_moyenne_inter_exp",
            "AUC_DO_moy_moyenne_inter_exp",
            "DO_max_moy_moyenne_inter_exp",
        ] if c in resume_souche.columns]
        if preview_cols:
            lines.append(resume_souche[preview_cols].to_string(index=False))
            lines.append("")

    lines.append("Interprétation")
    if n_exp < 2:
        lines.append(
            "- Une seule expérience indépendante est disponible : les résultats doivent être lus comme descriptifs. "
            "Les triplicats techniques résument la plaque mais ne remplacent pas des répétitions biologiques."
        )
    else:
        lines.append(
            "- Les tests entre souches reposent sur les moyennes par souche au sein de chaque expérience indépendante."
        )
        lines.append(
            "- Les comparaisons par paires sont exploratoires et corrigées par Holm."
        )

    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse statistique descriptive des cinétiques normalisées.")
    parser.add_argument(
        "input_files",
        type=Path,
        nargs="+",
        help="Un ou plusieurs fichiers CSV/XLSX issus de 04_normalisation_DO.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie. Par défaut : STAT_<nom> si 1 fichier, ou STAT_multi_<...> si plusieurs fichiers.",
    )
    parser.add_argument(
        "--experience-col",
        type=str,
        default="experience_id",
        help="Nom de la colonne identifiant l'expérience indépendante. Défaut : experience_id",
    )
    parser.add_argument(
        "--experience-id",
        type=str,
        default="exp1",
        help="Identifiant à utiliser si un seul fichier est fourni et que la colonne d'expérience est absente ou vide. Défaut : exp1",
    )
    parser.add_argument(
        "--experience-ids",
        type=str,
        nargs="+",
        default=None,
        help="Liste optionnelle d'identifiants d'expérience, un par fichier d'entrée, dans le même ordre.",
    )
    parser.add_argument(
        "--min-bio-reps",
        type=int,
        default=2,
        help="Nombre minimum d'expériences indépendantes par souche pour lancer les tests exploratoires. Défaut : 2",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parser_args()

    if args.min_bio_reps < 2:
        raise ValueError("--min-bio-reps doit être >= 2.")

    df, input_paths, assigned_ids = charger_entrees_multiples(
        input_files=args.input_files,
        experience_col=args.experience_col,
        experience_ids=args.experience_ids,
        fallback_prefix=args.experience_id,
    )
    output_dir = inferer_output_dir(input_paths, args.output_dir)

    param_puits = construire_parametres_par_puits(df)
    resume_exp = construire_resume_par_experience_souche(param_puits)
    resume_souche = construire_resume_par_souche(resume_exp)
    tests_globaux, posthoc, notes_tests = lancer_tests_exploratoires(
        resume_exp=resume_exp,
        min_bio_reps=args.min_bio_reps,
    )
    resume_global = construire_resume_global(
        df=df,
        param_puits=param_puits,
        resume_exp=resume_exp,
        tests_globaux=tests_globaux,
        notes_tests=notes_tests,
        min_bio_reps=args.min_bio_reps,
    )

    if len(input_paths) == 1:
        base = cleaner_basename(input_paths[0])
    else:
        base = "multi_experiences"
    path_param = output_dir / f"{base}_parametres_par_puits.csv"
    path_resume_exp = output_dir / "resume_par_experience_souche.csv"
    path_resume_souche = output_dir / "resume_par_souche.csv"
    path_tests = output_dir / "tests_globaux.csv"
    path_posthoc = output_dir / "comparaisons_posthoc.csv"
    path_resume_global = output_dir / "resume_analyse_statistique.csv"
    path_report = output_dir / "analyse_statistique_resume.txt"

    param_puits.to_csv(path_param, index=False, encoding="utf-8-sig")
    resume_exp.to_csv(path_resume_exp, index=False, encoding="utf-8-sig")
    resume_souche.to_csv(path_resume_souche, index=False, encoding="utf-8-sig")
    tests_globaux.to_csv(path_tests, index=False, encoding="utf-8-sig")
    posthoc.to_csv(path_posthoc, index=False, encoding="utf-8-sig")
    resume_global.to_csv(path_resume_global, index=False, encoding="utf-8-sig")
    ecrire_rapport_txt(
        path=path_report,
        df=df,
        param_puits=param_puits,
        resume_exp=resume_exp,
        resume_souche=resume_souche,
        tests_globaux=tests_globaux,
        posthoc=posthoc,
        notes_tests=notes_tests,
        min_bio_reps=args.min_bio_reps,
    )

    print("\n[OK] Analyse statistique terminée.")
    print(f"[OK] Fichiers d'entrée : {len(input_paths)}")
    for path, exp_id in zip(input_paths, assigned_ids):
        print(f"[INFO] {exp_id} <= {path}")
    print(f"[OK] Paramètres par puits : {path_param}")
    print(f"[OK] Résumé par expérience et souche : {path_resume_exp}")
    print(f"[OK] Résumé par souche : {path_resume_souche}")
    print(f"[OK] Tests globaux : {path_tests}")
    print(f"[OK] Comparaisons post-hoc : {path_posthoc}")
    print(f"[OK] Résumé global : {path_resume_global}")
    print(f"[OK] Rapport texte : {path_report}")
    print(f"[INFO] Expériences détectées : {df['experience_id'].nunique()}")
    print(f"[INFO] Séries / puits analysés : {len(param_puits)}")
    print(f"[INFO] Souches analysées : {df.loc[df['type'] == 'souche', 'souche'].nunique()}")
    if notes_tests:
        for note in notes_tests:
            print(f"[INFO] {note}")


if __name__ == "__main__":
    main()
