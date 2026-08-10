#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03_correction_blancs.py

Objectif
--------
Appliquer la correction des blancs sur le tableau long produit par
01_mise_en_forme_donnees.py, après revue manuelle des outliers détectés au QC.

Le script :
    1. lit le tableau long (CSV/XLSX)
    2. localise et lit outliers_decisions.csv
    3. demande une confirmation explicite à l'utilisateur avant de continuer
    4. exclut les séries / points marqués à retirer dans outliers_decisions.csv
    5. associe chaque souche à son blanc via la colonne Groupe
    6. calcule le blanc moyen par temps
    7. produit DO_corr et Lum_corr
    8. écrit les sorties en CSV

Usage
-----
python 03_correction_blancs.py "Manip_Optimisation_1_format_long_Luminescence_20_02.csv"
python 03_correction_blancs.py "...csv" --outliers-decisions "QC_.../outliers_decisions.csv"
python 03_correction_blancs.py "...csv" --output-dir "CORR_manip1"

Notes sur outliers_decisions.csv
--------------------------------
Le script n'exclut des données que si decision_utilisateur contient explicitement
une valeur d'exclusion, par exemple :
    - drop
    - exclude
    - remove
    - exclure
    - supprimer
    - discard

Les valeurs "keep", "review", vides, etc. n'excluent rien.
"""

from __future__ import annotations

import argparse
import sys
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLONNES_OBLIGATOIRES = [
    "temps_h",
    "souche",
    "Groupe",
    "replicat",
    "DO_brute",
    "Lum_brute",
    "type",
]

EXCLUSION_VALUES = {
    "drop",
    "exclude",
    "remove",
    "exclure",
    "supprimer",
    "discard",
    "delete",
    "retirer",
    "retire",
    "rejeter",
    "reject",
}


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


def verifier_colonnes(df: pd.DataFrame, colonnes: Iterable[str]) -> None:
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError("Colonnes manquantes : " + ", ".join(manquantes))


def preparer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["temps_h", "replicat", "DO_brute", "Lum_brute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["souche", "Groupe", "type"]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["type"] = df["type"].str.lower()
    df.loc[df["Groupe"].eq(""), "Groupe"] = "Sans_groupe"

    if "puits" not in df.columns:
        df["puits"] = ""
    else:
        df["puits"] = df["puits"].fillna("").astype(str).str.strip()

    if "sample_header" not in df.columns:
        if "puits" in df.columns:
            df["sample_header"] = df["souche"] + " (" + df["puits"] + ")"
        else:
            df["sample_header"] = df["souche"] + "_rep" + df["replicat"].astype("Int64").astype(str)
    else:
        df["sample_header"] = df["sample_header"].fillna("").astype(str).str.strip()

    if "lecture" not in df.columns:
        df["lecture"] = np.nan

    df = df.dropna(subset=["temps_h", "replicat"]).copy()
    df["replicat"] = df["replicat"].astype("Int64")
    df = df.sort_values(["type", "souche", "sample_header", "temps_h"]).reset_index(drop=True)
    return df


def inferer_outliers_path(input_path: Path) -> Path:
    candidat_1 = input_path.parent / f"QC_{input_path.stem}" / "outliers_decisions.csv"
    if candidat_1.exists():
        return candidat_1

    candidat_2 = input_path.parent / "QC" / "outliers_decisions.csv"
    if candidat_2.exists():
        return candidat_2

    matches = sorted(input_path.parent.rglob("outliers_decisions.csv"))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        for m in matches:
            if input_path.stem in str(m.parent):
                return m
        return matches[0]

    return candidat_1


def demander_confirmation(outliers_path: Path) -> None:
    print("\n" + "=" * 72)
    print("ATTENTION AVANT CORRECTION DES BLANCS")
    print("Le script va utiliser le fichier de décisions suivant :")
    print(f"  {outliers_path}")
    print("Vérifie que tu as relu ce fichier et modifié decision_utilisateur si besoin.")
    print("Le script ne continue que si tu réponds 'oui'.")
    print("=" * 72)
    reponse = input("As-tu vérifié le fichier outliers_decisions.csv ? Tape 'oui' pour continuer : ").strip().lower()
    if reponse not in {"oui", "o", "yes", "y"}:
        print("[INFO] Opération annulée. Vérifie/modifie outliers_decisions.csv puis relance le script.")
        sys.exit(0)


def charger_outliers(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Fichier outliers_decisions.csv introuvable. "
            "Lance le QC qui le génère, ou passe --outliers-decisions explicitement.\n"
            f"Chemin attendu : {path}"
        )

    df = pd.read_csv(path)
    colonnes_attendues = [
        "scope",
        "sample_header",
        "replicat",
        "decision_utilisateur",
    ]
    verifier_colonnes(df, colonnes_attendues)

    if "temps_h" not in df.columns:
        df["temps_h"] = np.nan
    if "variable_cible" not in df.columns:
        df["variable_cible"] = "both"

    df["scope"] = df["scope"].fillna("").astype(str).str.strip().str.lower()
    df["sample_header"] = df["sample_header"].fillna("").astype(str).str.strip()
    df["replicat"] = pd.to_numeric(df["replicat"], errors="coerce").astype("Int64")
    df["temps_h"] = pd.to_numeric(df["temps_h"], errors="coerce")
    df["variable_cible"] = df["variable_cible"].fillna("both").astype(str).str.strip().str.lower()
    df["decision_utilisateur"] = df["decision_utilisateur"].fillna("").astype(str).str.strip().str.lower()
    return df


def est_exclusion(valeur: object) -> bool:
    txt = str(valeur).strip().lower()
    return txt in EXCLUSION_VALUES


def appliquer_decisions_outliers(df: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df2 = df.copy()

    decisions_exclues = decisions[decisions["decision_utilisateur"].map(est_exclusion)].copy()
    if decisions_exclues.empty:
        df2["exclu_outlier"] = False
        exclusions = pd.DataFrame(columns=["scope", "sample_header", "replicat", "temps_h", "variable_cible"])
        resume = pd.DataFrame([
            {"metrique": "series_exclues", "valeur": 0},
            {"metrique": "points_exclus", "valeur": 0},
            {"metrique": "lignes_total_input", "valeur": len(df)},
            {"metrique": "lignes_retenues_apres_exclusion", "valeur": len(df2)},
        ])
        return df2, exclusions, resume

    df2["exclu_outlier"] = False
    exclusions_lignes = []

    decisions_series = decisions_exclues[decisions_exclues["scope"] == "serie"].copy()
    if not decisions_series.empty:
        for _, row in decisions_series.iterrows():
            mask = (
                df2["sample_header"].eq(row["sample_header"]) &
                df2["replicat"].eq(row["replicat"])
            )
            if mask.any():
                df2.loc[mask, "exclu_outlier"] = True
                exclusions_lignes.append({
                    "scope": "serie",
                    "sample_header": row["sample_header"],
                    "replicat": row["replicat"],
                    "temps_h": np.nan,
                    "variable_cible": "both",
                })

    decisions_points = decisions_exclues[decisions_exclues["scope"] == "point"].copy()
    if not decisions_points.empty:
        for _, row in decisions_points.iterrows():
            mask = (
                df2["sample_header"].eq(row["sample_header"]) &
                df2["replicat"].eq(row["replicat"]) &
                np.isclose(df2["temps_h"].astype(float), float(row["temps_h"]), rtol=0, atol=1e-6)
            )
            if mask.any():
                df2.loc[mask, "exclu_outlier"] = True
                exclusions_lignes.append({
                    "scope": "point",
                    "sample_header": row["sample_header"],
                    "replicat": row["replicat"],
                    "temps_h": row["temps_h"],
                    "variable_cible": row.get("variable_cible", "both"),
                })

    exclusions = pd.DataFrame(exclusions_lignes).drop_duplicates()
    df_filtre = df2.loc[~df2["exclu_outlier"]].copy()

    resume = pd.DataFrame([
        {"metrique": "series_exclues", "valeur": int((exclusions["scope"] == "serie").sum()) if not exclusions.empty else 0},
        {"metrique": "points_exclus", "valeur": int((exclusions["scope"] == "point").sum()) if not exclusions.empty else 0},
        {"metrique": "lignes_total_input", "valeur": len(df)},
        {"metrique": "lignes_retenues_apres_exclusion", "valeur": len(df_filtre)},
    ])
    return df_filtre, exclusions, resume


def decouper_groupes(groupe: object) -> list[str]:
    texte = str(groupe).strip()
    if not texte:
        return []
    return [x.strip() for x in re.split(r"\s*,\s*|\s*;\s*", texte) if x.strip()]


def construire_mapping_groupes_blancs(df: pd.DataFrame) -> pd.DataFrame:
    blancs = (
        df.loc[df["type"] == "blanc", ["souche", "Groupe"]]
        .drop_duplicates()
        .rename(columns={"souche": "blanc_associe", "Groupe": "Groupe_blanc"})
        .copy()
    )
    if blancs.empty:
        raise ValueError("Aucun blanc trouvé dans la colonne type.")

    lignes = []
    for _, row in blancs.iterrows():
        tokens = decouper_groupes(row["Groupe_blanc"])
        if not tokens:
            continue
        for token in tokens:
            lignes.append({
                "Groupe": token,
                "blanc_associe": row["blanc_associe"],
                "Groupe_blanc": row["Groupe_blanc"],
            })

    mapping = pd.DataFrame(lignes).drop_duplicates()
    if mapping.empty:
        raise ValueError(
            "Impossible de construire le mapping Groupe -> blanc. "
            "Vérifie la colonne Groupe des blancs."
        )

    doublons = mapping.groupby("Groupe")["blanc_associe"].nunique()
    conflits = doublons[doublons > 1]
    if not conflits.empty:
        details = mapping[mapping["Groupe"].isin(conflits.index)].sort_values(["Groupe", "blanc_associe"])
        raise ValueError(
            "Conflit dans l'association Groupe -> blanc. Un même groupe pointe vers plusieurs blancs.\n"
            + details.to_string(index=False)
        )

    return mapping.sort_values(["Groupe", "blanc_associe"]).reset_index(drop=True)


def associer_blancs(df: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    m = mapping[["Groupe", "blanc_associe"]].drop_duplicates()

    df2 = df2.merge(m, on="Groupe", how="left")
    df2.loc[df2["type"] == "blanc", "blanc_associe"] = df2.loc[df2["type"] == "blanc", "souche"]

    non_assignes = df2.loc[df2["blanc_associe"].isna(), ["type", "souche", "Groupe", "sample_header"]].drop_duplicates()
    if not non_assignes.empty:
        raise ValueError(
            "Certaines lignes n'ont pas de blanc associé. Vérifie la colonne Groupe.\n"
            + non_assignes.to_string(index=False)
        )

    return df2


def calculer_blanc_moyen(df: pd.DataFrame) -> pd.DataFrame:
    blancs = df.loc[df["type"] == "blanc"].copy()
    if blancs.empty:
        raise ValueError("Aucune ligne de type 'blanc' disponible après exclusion des outliers.")

    agg = (
        blancs.groupby(["blanc_associe", "temps_h"], dropna=False)
        .agg(
            DO_blanc_moy=("DO_brute", "mean"),
            Lum_blanc_moy=("Lum_brute", "mean"),
            DO_blanc_sd=("DO_brute", "std"),
            Lum_blanc_sd=("Lum_brute", "std"),
            n_blancs=("sample_header", "nunique"),
            n_lignes_blanc=("sample_header", "size"),
        )
        .reset_index()
        .sort_values(["blanc_associe", "temps_h"])
        .reset_index(drop=True)
    )
    return agg


def appliquer_correction_blancs(df: pd.DataFrame, blanc_moyen: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(blanc_moyen, on=["blanc_associe", "temps_h"], how="left")

    manquants = out.loc[
        out[["DO_blanc_moy", "Lum_blanc_moy"]].isna().any(axis=1),
        ["type", "souche", "sample_header", "Groupe", "blanc_associe", "temps_h"]
    ]
    if not manquants.empty:
        raise ValueError(
            "Certains temps n'ont pas de blanc moyen associé. Vérifie les exclusions et les temps.\n"
            + manquants.head(20).to_string(index=False)
        )

    out["DO_corr"] = out["DO_brute"] - out["DO_blanc_moy"]
    out["Lum_corr"] = out["Lum_brute"] - out["Lum_blanc_moy"]
    return out


def construire_resume(df_input: pd.DataFrame, df_filtre: pd.DataFrame, df_corr: pd.DataFrame, exclusions: pd.DataFrame, blanc_moyen: pd.DataFrame) -> pd.DataFrame:
    resume = [
        {"metrique": "lignes_input", "valeur": len(df_input)},
        {"metrique": "lignes_apres_exclusions", "valeur": len(df_filtre)},
        {"metrique": "lignes_sortie_corrigee", "valeur": len(df_corr)},
        {"metrique": "series_uniques_sortie", "valeur": int(df_corr["sample_header"].nunique())},
        {"metrique": "souches_uniques_sortie", "valeur": int(df_corr.loc[df_corr["type"] == "souche", "souche"].nunique())},
        {"metrique": "blancs_uniques_sortie", "valeur": int(df_corr.loc[df_corr["type"] == "blanc", "souche"].nunique())},
        {"metrique": "blancs_associes_uniques", "valeur": int(df_corr["blanc_associe"].nunique())},
        {"metrique": "series_exclues", "valeur": int((exclusions["scope"] == "serie").sum()) if not exclusions.empty else 0},
        {"metrique": "points_exclus", "valeur": int((exclusions["scope"] == "point").sum()) if not exclusions.empty else 0},
        {"metrique": "n_lignes_blanc_moyen", "valeur": len(blanc_moyen)},
    ]
    return pd.DataFrame(resume)


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correction des blancs après QC et revue des outliers.")
    parser.add_argument("input_file", type=Path, help="Tableau long CSV/XLSX issu du script 01")
    parser.add_argument(
        "--outliers-decisions",
        type=Path,
        default=None,
        help="Chemin vers outliers_decisions.csv. Par défaut, le script tente de l'inférer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie. Par défaut : CORR_<nom_du_fichier>",
    )
    return parser.parse_args()


def main() -> None:
    args = parser_args()

    input_path = args.input_file.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {input_path}")

    outliers_path = args.outliers_decisions.resolve() if args.outliers_decisions else inferer_outliers_path(input_path).resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else (input_path.parent / f"CORR_{input_path.stem}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    demander_confirmation(outliers_path)

    df_input = lire_table(input_path)
    verifier_colonnes(df_input, COLONNES_OBLIGATOIRES)
    df_input = preparer_df(df_input)

    decisions = charger_outliers(outliers_path)
    df_filtre, exclusions, resume_exclusions = appliquer_decisions_outliers(df_input, decisions)

    mapping = construire_mapping_groupes_blancs(df_input)
    df_filtre = associer_blancs(df_filtre, mapping)
    blanc_moyen = calculer_blanc_moyen(df_filtre)
    df_corr = appliquer_correction_blancs(df_filtre, blanc_moyen)
    resume = construire_resume(df_input, df_filtre, df_corr, exclusions, blanc_moyen)
    resume = pd.concat([resume, resume_exclusions], ignore_index=True).drop_duplicates(subset=["metrique"], keep="first")

    base = nettoyer_nom_fichier(input_path.stem)
    path_corr = output_dir / f"{base}_corrige_blancs.csv"
    path_mapping = output_dir / "mapping_groupes_blancs.csv"
    path_blanc = output_dir / "blanc_moyen_par_temps.csv"
    path_exclusions = output_dir / "exclusions_appliquees.csv"
    path_resume = output_dir / "resume_correction_blancs.csv"

    df_corr.to_csv(path_corr, index=False, encoding="utf-8-sig")
    mapping.to_csv(path_mapping, index=False, encoding="utf-8-sig")
    blanc_moyen.to_csv(path_blanc, index=False, encoding="utf-8-sig")
    exclusions.to_csv(path_exclusions, index=False, encoding="utf-8-sig")
    resume.to_csv(path_resume, index=False, encoding="utf-8-sig")

    print("\n[OK] Correction des blancs terminée.")
    print(f"[OK] Données corrigées : {path_corr}")
    print(f"[OK] Mapping groupes/blancs : {path_mapping}")
    print(f"[OK] Blanc moyen par temps : {path_blanc}")
    print(f"[OK] Exclusions appliquées : {path_exclusions}")
    print(f"[OK] Résumé : {path_resume}")

    n_series = int((exclusions["scope"] == "serie").sum()) if not exclusions.empty else 0
    n_points = int((exclusions["scope"] == "point").sum()) if not exclusions.empty else 0
    print(f"[INFO] Séries exclues : {n_series}")
    print(f"[INFO] Points exclus : {n_points}")
    print(f"[INFO] Blancs associés détectés : {df_corr['blanc_associe'].nunique()}")


if __name__ == "__main__":
    main()
