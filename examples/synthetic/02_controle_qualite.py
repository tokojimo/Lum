#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_controle_qualite.py

Objectif
--------
Réaliser un contrôle qualité de premier niveau sur le tableau long produit par
01_mise_en_forme_donnees.py.

Le script :
    1. lit un fichier CSV/XLSX au format long
    2. vérifie les colonnes attendues
    3. produit des graphes bruts DO et luminescence par souche / blanc
    4. résume la qualité des séries (par puits / réplicat)
    5. repère des points aberrants potentiels entre réplicats
    6. crée et complète automatiquement un fichier outliers_decisions.csv
       que l'utilisateur peut ensuite modifier manuellement

Entrée attendue
---------------
Un tableau long avec au minimum les colonnes :
    - temps_h
    - souche
    - replicat
    - DO_brute
    - Lum_brute
    - type

Colonnes facultatives mais recommandées :
    - puits
    - lecture
    - sample_header

Exemples d'utilisation
----------------------
python 02_controle_qualite.py "Manip_Optimisation_1_format_long_Luminescence_20_02.csv"
python 02_controle_qualite.py "Manip_Optimisation_1_format_long_Luminescence_20_02.xlsx"
python 02_controle_qualite.py "...csv" --output-dir "QC_manip1"

Sorties
-------
Un dossier contenant :
    - resume_global_qc.csv
    - qc_series.csv
    - qc_points_aberrants.csv
    - outliers_decisions.csv
    - graphes_png/
        - DO_souches/*.png
        - Lum_souches/*.png
        - DO_blancs/*.png
        - Lum_blancs/*.png
        - resume_global/*.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLONNES_OBLIGATOIRES = [
    "temps_h",
    "souche",
    "replicat",
    "DO_brute",
    "Lum_brute",
    "type",
]

COLONNES_OUTLIERS = [
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
    "commentaire_utilisateur",
]


def nettoyer_nom_fichier(texte: object) -> str:
    texte = str(texte)
    texte = re.sub(r"[^A-Za-z0-9._-]+", "_", texte).strip("_")
    return texte or "sans_nom"


def fmt_id_part(valeur: object) -> str:
    if valeur is None or (isinstance(valeur, float) and np.isnan(valeur)):
        return "NA"
    if pd.isna(valeur):
        return "NA"
    if isinstance(valeur, (int, np.integer)):
        return str(int(valeur))
    if isinstance(valeur, (float, np.floating)):
        if np.isnan(valeur):
            return "NA"
        return f"{float(valeur):.4f}".rstrip("0").rstrip(".")
    return nettoyer_nom_fichier(str(valeur))


def lire_table_longue(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffixe = path.suffix.lower()
    if suffixe == ".csv":
        df = pd.read_csv(path)
    elif suffixe in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    else:
        raise ValueError(f"Format non supporté : {path.suffix}. Utiliser CSV ou XLSX.")
    return df


def verifier_colonnes(df: pd.DataFrame) -> None:
    manquantes = [c for c in COLONNES_OBLIGATOIRES if c not in df.columns]
    if manquantes:
        raise ValueError(
            "Colonnes manquantes dans le tableau long : " + ", ".join(manquantes)
        )


def preparer_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in ["temps_h", "replicat", "DO_brute", "Lum_brute"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["souche"] = df["souche"].astype(str).str.strip()
    df["type"] = df["type"].astype(str).str.strip().str.lower()

    if "sample_header" not in df.columns:
        if "puits" in df.columns:
            df["sample_header"] = df["souche"].astype(str) + " (" + df["puits"].astype(str) + ")"
        else:
            df["sample_header"] = df["souche"].astype(str) + "_rep" + df["replicat"].astype("Int64").astype(str)

    if "puits" not in df.columns:
        df["puits"] = ""

    if "lecture" not in df.columns:
        df["lecture"] = np.nan

    df["replicat"] = df["replicat"].astype("Int64")
    df = df.dropna(subset=["temps_h"]).copy()
    df = df.sort_values(["type", "souche", "sample_header", "temps_h"]).reset_index(drop=True)
    return df


def creer_dossiers(output_dir: Path) -> dict[str, Path]:
    dossiers = {
        "root": output_dir,
        "png_root": output_dir / "graphes_png",
        "do_souches": output_dir / "graphes_png" / "DO_souches",
        "lum_souches": output_dir / "graphes_png" / "Lum_souches",
        "do_blancs": output_dir / "graphes_png" / "DO_blancs",
        "lum_blancs": output_dir / "graphes_png" / "Lum_blancs",
        "resume": output_dir / "graphes_png" / "resume_global",
    }
    for d in dossiers.values():
        d.mkdir(parents=True, exist_ok=True)
    return dossiers


def temps_theoriques(df: pd.DataFrame) -> np.ndarray:
    return np.sort(df["temps_h"].dropna().unique())


def robust_z_scores(valeurs: pd.Series) -> pd.Series:
    s = pd.to_numeric(valeurs, errors="coerce")
    med = np.nanmedian(s)
    mad = np.nanmedian(np.abs(s - med))

    if np.isnan(med):
        return pd.Series(np.nan, index=valeurs.index)

    if mad == 0 or np.isnan(mad):
        std = np.nanstd(s, ddof=1)
        if std == 0 or np.isnan(std):
            return pd.Series(np.nan, index=valeurs.index)
        return (s - np.nanmean(s)) / std

    return 0.6745 * (s - med) / mad


def detecter_points_aberrants(df: pd.DataFrame, seuil_z: float = 3.5) -> pd.DataFrame:
    donnees = []

    group_cols = ["type", "souche", "temps_h"]
    for _, bloc in df.groupby(group_cols, dropna=False):
        if bloc["sample_header"].nunique() < 3:
            continue

        bloc = bloc.copy()
        bloc["z_do"] = robust_z_scores(bloc["DO_brute"])
        bloc["z_lum"] = robust_z_scores(bloc["Lum_brute"])

        bloc["point_aberrant_DO"] = bloc["z_do"].abs() > seuil_z
        bloc["point_aberrant_Lum"] = bloc["z_lum"].abs() > seuil_z
        bloc["point_aberrant_any"] = bloc[["point_aberrant_DO", "point_aberrant_Lum"]].any(axis=1)

        donnees.append(bloc)

    if not donnees:
        return pd.DataFrame(columns=list(df.columns) + [
            "z_do", "z_lum", "point_aberrant_DO", "point_aberrant_Lum", "point_aberrant_any"
        ])

    out = pd.concat(donnees, ignore_index=True)
    out = out[out["point_aberrant_any"]].copy()
    return out.sort_values(["type", "souche", "temps_h", "sample_header"]).reset_index(drop=True)


def resumer_series(df: pd.DataFrame, temps_ref: np.ndarray, points_aberrants: pd.DataFrame) -> pd.DataFrame:
    n_temps_theorique = len(temps_ref)
    aberrants_par_serie = pd.Series(dtype=int)
    if not points_aberrants.empty:
        aberrants_par_serie = points_aberrants.groupby("sample_header")["point_aberrant_any"].sum().astype(int)

    lignes = []
    for cle, bloc in df.groupby(["type", "souche", "replicat", "sample_header", "puits"], dropna=False):
        type_, souche, replicat, sample_header, puits = cle
        bloc = bloc.sort_values("temps_h").copy()

        n_points = len(bloc)
        n_temps_uniques = bloc["temps_h"].nunique()
        n_doublons_temps = n_points - n_temps_uniques
        n_manquants_theoriques = max(0, n_temps_theorique - n_temps_uniques)

        n_do_nan = int(bloc["DO_brute"].isna().sum())
        n_lum_nan = int(bloc["Lum_brute"].isna().sum())
        n_do_neg = int((bloc["DO_brute"] < 0).sum())
        n_lum_neg = int((bloc["Lum_brute"] < 0).sum())

        do_diff = bloc["DO_brute"].diff()
        lum_diff = bloc["Lum_brute"].diff()
        n_baisse_do = int((do_diff < -0.02).sum())
        lum_std = np.nanstd(lum_diff)
        n_saut_lum = int((lum_diff.abs() > (3 * lum_std if lum_std > 0 else np.inf)).sum())

        n_aberrants = int(aberrants_par_serie.get(sample_header, 0))

        flags: List[str] = []
        if n_manquants_theoriques > 0:
            flags.append(f"temps_manquants={n_manquants_theoriques}")
        if n_doublons_temps > 0:
            flags.append(f"temps_doubles={n_doublons_temps}")
        if n_do_nan > 0:
            flags.append(f"DO_NA={n_do_nan}")
        if n_lum_nan > 0:
            flags.append(f"Lum_NA={n_lum_nan}")
        if n_do_neg > 0:
            flags.append(f"DO_neg={n_do_neg}")
        if type_ == "souche" and n_points > 0 and n_lum_neg / n_points >= 0.25:
            flags.append(f"Lum_neg_frequentes={n_lum_neg}")
        if n_baisse_do >= 3:
            flags.append(f"baisses_DO={n_baisse_do}")
        if n_aberrants >= 3:
            flags.append(f"points_aberrants={n_aberrants}")

        lignes.append(
            {
                "type": type_,
                "souche": souche,
                "replicat": replicat,
                "sample_header": sample_header,
                "puits": puits,
                "n_points": n_points,
                "n_temps_uniques": n_temps_uniques,
                "n_temps_theorique": n_temps_theorique,
                "n_temps_manquants": n_manquants_theoriques,
                "n_temps_doubles": n_doublons_temps,
                "DO_min": bloc["DO_brute"].min(),
                "DO_max": bloc["DO_brute"].max(),
                "Lum_min": bloc["Lum_brute"].min(),
                "Lum_max": bloc["Lum_brute"].max(),
                "n_DO_NA": n_do_nan,
                "n_Lum_NA": n_lum_nan,
                "n_DO_neg": n_do_neg,
                "n_Lum_neg": n_lum_neg,
                "n_baisses_DO_importantes": n_baisse_do,
                "n_grands_sauts_Lum": n_saut_lum,
                "n_points_aberrants": n_aberrants,
                "flags": "; ".join(flags) if flags else "OK",
            }
        )

    out = pd.DataFrame(lignes)
    out = out.sort_values(["type", "souche", "replicat", "sample_header"]).reset_index(drop=True)
    return out


def resume_global(df: pd.DataFrame, qc_series: pd.DataFrame, points_aberrants: pd.DataFrame) -> pd.DataFrame:
    temps = np.sort(df["temps_h"].dropna().unique())
    pas_median = np.nan
    if len(temps) >= 2:
        pas_median = float(np.median(np.diff(temps)))

    lignes = [
        ("n_lignes", int(len(df))),
        ("n_souches_uniques_total", int(df["souche"].nunique())),
        ("n_sample_header", int(df["sample_header"].nunique())),
        ("n_types", int(df["type"].nunique())),
        ("n_souches_type_souche", int(df.loc[df["type"] == "souche", "souche"].nunique())),
        ("n_souches_type_blanc", int(df.loc[df["type"] == "blanc", "souche"].nunique())),
        ("n_temps_uniques", int(len(temps))),
        ("pas_median_h", pas_median),
        ("temps_min_h", float(np.nanmin(temps)) if len(temps) else np.nan),
        ("temps_max_h", float(np.nanmax(temps)) if len(temps) else np.nan),
        ("n_series_total", int(len(qc_series))),
        ("n_series_flaggees", int((qc_series["flags"] != "OK").sum())),
        ("n_points_aberrants", int(len(points_aberrants))),
        ("n_DO_neg_total", int((df["DO_brute"] < 0).sum())),
        ("n_Lum_neg_total", int((df["Lum_brute"] < 0).sum())),
        ("n_DO_NA_total", int(df["DO_brute"].isna().sum())),
        ("n_Lum_NA_total", int(df["Lum_brute"].isna().sum())),
    ]
    return pd.DataFrame(lignes, columns=["indicateur", "valeur"])


def tracer_courbes_par_groupe(
    df: pd.DataFrame,
    variable: str,
    type_filtre: str,
    output_dir: Path,
    prefixe: str,
) -> None:
    sous_df = df[df["type"] == type_filtre].copy()
    if sous_df.empty:
        return

    for souche, bloc in sous_df.groupby("souche", dropna=False):
        fig, ax = plt.subplots(figsize=(8, 5))
        bloc = bloc.sort_values(["sample_header", "temps_h"])

        for sample_header, serie in bloc.groupby("sample_header", dropna=False):
            etiquette = str(sample_header)
            if "replicat" in serie.columns and serie["replicat"].notna().any():
                rep = serie["replicat"].dropna().iloc[0]
                etiquette = f"rep {int(rep)} - {etiquette}"
            ax.plot(serie["temps_h"], serie[variable], marker="o", markersize=2.5, linewidth=1.2, label=etiquette)

        ax.set_title(f"{variable} brute - {souche}")
        ax.set_xlabel("Temps (h)")
        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()

        nom = nettoyer_nom_fichier(f"{prefixe}_{souche}.png")
        fig.savefig(output_dir / nom, dpi=200)
        plt.close(fig)


def tracer_resume_global(df: pd.DataFrame, output_dir: Path) -> None:
    for variable in ["DO_brute", "Lum_brute"]:
        for type_filtre in ["souche", "blanc"]:
            bloc = df[df["type"] == type_filtre].copy()
            if bloc.empty:
                continue

            fig, ax = plt.subplots(figsize=(9, 5.5))
            for _, serie in bloc.groupby("sample_header", dropna=False):
                ax.plot(serie["temps_h"], serie[variable], linewidth=1.0, alpha=0.8)
            ax.set_title(f"Vue globale - {variable} - {type_filtre}")
            ax.set_xlabel("Temps (h)")
            ax.set_ylabel(variable)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(output_dir / nettoyer_nom_fichier(f"resume_{variable}_{type_filtre}.png"), dpi=200)
            plt.close(fig)

    counts = []
    for _, bloc in df.groupby("sample_header"):
        counts.append(len(bloc))
    if counts:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.hist(counts, bins=min(20, max(5, len(set(counts)))))
        ax.set_title("Distribution du nombre de points par série")
        ax.set_xlabel("Nombre de points")
        ax.set_ylabel("Nombre de séries")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "distribution_points_par_serie.png", dpi=200)
        plt.close(fig)


def construire_outliers_decisions(
    points_aberrants: pd.DataFrame,
    qc_series: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    series_lookup = {}
    if not qc_series.empty:
        series_lookup = qc_series.set_index("sample_header")[["n_points_aberrants", "flags"]].to_dict("index")

    lignes = []

    if not points_aberrants.empty:
        for _, row in points_aberrants.iterrows():
            if bool(row.get("point_aberrant_DO")) and bool(row.get("point_aberrant_Lum")):
                variable_cible = "both"
                motif = "point_aberrant_DO; point_aberrant_Lum"
            elif bool(row.get("point_aberrant_DO")):
                variable_cible = "DO"
                motif = "point_aberrant_DO"
            else:
                variable_cible = "Lum"
                motif = "point_aberrant_Lum"

            info_serie = series_lookup.get(row["sample_header"], {})
            decision_id = "|".join([
                "point",
                fmt_id_part(row["sample_header"]),
                fmt_id_part(row["temps_h"]),
                variable_cible,
            ])

            lignes.append(
                {
                    "decision_id": decision_id,
                    "scope": "point",
                    "type": row.get("type", ""),
                    "souche": row.get("souche", ""),
                    "replicat": row.get("replicat", pd.NA),
                    "sample_header": row.get("sample_header", ""),
                    "puits": row.get("puits", ""),
                    "temps_h": row.get("temps_h", np.nan),
                    "variable_cible": variable_cible,
                    "detection_type": "point_aberrant",
                    "motif_auto": motif,
                    "z_do": row.get("z_do", np.nan),
                    "z_lum": row.get("z_lum", np.nan),
                    "n_points_aberrants_serie": info_serie.get("n_points_aberrants", np.nan),
                    "flags_serie": info_serie.get("flags", ""),
                    "decision_utilisateur": "review",
                    "raison_utilisateur": "",
                    "commentaire_utilisateur": "",
                }
            )

    qc_series_flag = qc_series[qc_series["flags"] != "OK"].copy() if not qc_series.empty else pd.DataFrame()
    if not qc_series_flag.empty:
        for _, row in qc_series_flag.iterrows():
            decision_id = "|".join([
                "serie",
                fmt_id_part(row["sample_header"]),
            ])
            lignes.append(
                {
                    "decision_id": decision_id,
                    "scope": "serie",
                    "type": row.get("type", ""),
                    "souche": row.get("souche", ""),
                    "replicat": row.get("replicat", pd.NA),
                    "sample_header": row.get("sample_header", ""),
                    "puits": row.get("puits", ""),
                    "temps_h": np.nan,
                    "variable_cible": "both",
                    "detection_type": "serie_a_inspecter",
                    "motif_auto": row.get("flags", ""),
                    "z_do": np.nan,
                    "z_lum": np.nan,
                    "n_points_aberrants_serie": row.get("n_points_aberrants", np.nan),
                    "flags_serie": row.get("flags", ""),
                    "decision_utilisateur": "review",
                    "raison_utilisateur": "",
                    "commentaire_utilisateur": "",
                }
            )

    auto_df = pd.DataFrame(lignes, columns=COLONNES_OUTLIERS)
    if auto_df.empty:
        auto_df = pd.DataFrame(columns=COLONNES_OUTLIERS)

    chemin = output_dir / "outliers_decisions.csv"
    if chemin.exists():
        ancien = pd.read_csv(chemin)
        for col in COLONNES_OUTLIERS:
            if col not in ancien.columns:
                ancien[col] = np.nan
        ancien = ancien[COLONNES_OUTLIERS].copy()

        colonnes_user = ["decision_utilisateur", "raison_utilisateur", "commentaire_utilisateur"]
        ancien_user = ancien[["decision_id"] + colonnes_user].drop_duplicates("decision_id")
        auto_df = auto_df.merge(ancien_user, on="decision_id", how="left", suffixes=("", "_old"))

        for col in colonnes_user:
            old_col = f"{col}_old"
            auto_df[col] = auto_df[old_col].combine_first(auto_df[col])
            auto_df = auto_df.drop(columns=[old_col])

        ids_auto = set(auto_df["decision_id"].astype(str))
        lignes_manuelles = ancien[~ancien["decision_id"].astype(str).isin(ids_auto)].copy()
        if not lignes_manuelles.empty:
            auto_df = pd.concat([auto_df, lignes_manuelles[COLONNES_OUTLIERS]], ignore_index=True)

    if not auto_df.empty:
        auto_df["decision_utilisateur"] = auto_df["decision_utilisateur"].fillna("review")
        auto_df["raison_utilisateur"] = auto_df["raison_utilisateur"].fillna("")
        auto_df["commentaire_utilisateur"] = auto_df["commentaire_utilisateur"].fillna("")
        auto_df = auto_df.drop_duplicates(subset=["decision_id"], keep="first").copy()
        auto_df["scope_order"] = auto_df["scope"].map({"serie": 0, "point": 1}).fillna(9)
        auto_df = auto_df.sort_values(
            ["scope_order", "type", "souche", "replicat", "sample_header", "temps_h", "variable_cible"]
        ).drop(columns=["scope_order"]).reset_index(drop=True)

    auto_df.to_csv(chemin, index=False, encoding="utf-8-sig")
    return auto_df


def ecrire_sorties(
    output_dir: Path,
    resume: pd.DataFrame,
    qc_series: pd.DataFrame,
    points_aberrants: pd.DataFrame,
    outliers_decisions: pd.DataFrame,
) -> None:
    resume.to_csv(output_dir / "resume_global_qc.csv", index=False)
    qc_series.to_csv(output_dir / "qc_series.csv", index=False)
    points_aberrants.to_csv(output_dir / "qc_points_aberrants.csv", index=False)

    with pd.ExcelWriter(output_dir / "QC_resume_et_series.xlsx", engine="openpyxl") as writer:
        resume.to_excel(writer, sheet_name="resume_global", index=False)
        qc_series.to_excel(writer, sheet_name="qc_series", index=False)
        points_aberrants.to_excel(writer, sheet_name="points_aberrants", index=False)
        outliers_decisions.to_excel(writer, sheet_name="outliers_decisions", index=False)


def imprimer_resume_console(
    resume: pd.DataFrame,
    qc_series: pd.DataFrame,
    points_aberrants: pd.DataFrame,
    outliers_decisions: pd.DataFrame,
    output_dir: Path,
) -> None:
    d = dict(zip(resume["indicateur"], resume["valeur"]))
    print("[OK] Contrôle qualité terminé.")
    print(f"[INFO] Dossier de sortie : {output_dir}")
    print(f"[INFO] Lignes analysées : {int(d.get('n_lignes', 0))}")
    print(f"[INFO] Séries analysées : {int(d.get('n_series_total', 0))}")
    print(f"[INFO] Séries flaggées : {int(d.get('n_series_flaggees', 0))}")
    print(f"[INFO] Points aberrants potentiels : {int(d.get('n_points_aberrants', 0))}")
    print(f"[INFO] Fichier de décision créé/mis à jour : {output_dir / 'outliers_decisions.csv'}")
    if not outliers_decisions.empty:
        n_series = int((outliers_decisions["scope"] == "serie").sum())
        n_points = int((outliers_decisions["scope"] == "point").sum())
        print(f"[INFO] Entrées dans outliers_decisions.csv : {len(outliers_decisions)} ({n_series} séries, {n_points} points)")
        print("[INFO] Édite ensuite la colonne 'decision_utilisateur' avec par ex. : keep / remove / review")
    if not qc_series.empty:
        top = qc_series[qc_series["flags"] != "OK"].head(10)
        if not top.empty:
            print("[INFO] Premières séries à inspecter :")
            for _, row in top.iterrows():
                print(f"    - {row['sample_header']} -> {row['flags']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Contrôle qualité des données longues Varioskan.")
    parser.add_argument("fichier_long", help="Chemin vers le fichier long CSV/XLSX produit à l'étape 1.")
    parser.add_argument("--sheet-name", default=0, help="Nom ou index de feuille si entrée XLSX (défaut : 0).")
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie. Par défaut : QC_<nom_du_fichier>.")
    parser.add_argument("--seuil-z", type=float, default=3.5, help="Seuil de détection des points aberrants robustes (défaut : 3.5).")
    args = parser.parse_args()

    fichier = Path(args.fichier_long)
    if not fichier.exists():
        raise FileNotFoundError(f"Fichier introuvable : {fichier}")

    sheet_name = args.sheet_name
    if isinstance(sheet_name, str) and sheet_name.isdigit():
        sheet_name = int(sheet_name)

    output_dir = Path(args.output_dir) if args.output_dir else fichier.with_name(f"QC_{fichier.stem}")
    dossiers = creer_dossiers(output_dir)

    df = lire_table_longue(fichier, sheet_name=sheet_name)
    verifier_colonnes(df)
    df = preparer_df(df)

    temps_ref = temps_theoriques(df)
    points_aberrants = detecter_points_aberrants(df, seuil_z=args.seuil_z)
    qc_series = resumer_series(df, temps_ref, points_aberrants)
    resume = resume_global(df, qc_series, points_aberrants)

    tracer_courbes_par_groupe(df, "DO_brute", "souche", dossiers["do_souches"], "DO")
    tracer_courbes_par_groupe(df, "Lum_brute", "souche", dossiers["lum_souches"], "Lum")
    tracer_courbes_par_groupe(df, "DO_brute", "blanc", dossiers["do_blancs"], "DO_blanc")
    tracer_courbes_par_groupe(df, "Lum_brute", "blanc", dossiers["lum_blancs"], "Lum_blanc")
    tracer_resume_global(df, dossiers["resume"])

    outliers_decisions = construire_outliers_decisions(points_aberrants, qc_series, output_dir)
    ecrire_sorties(output_dir, resume, qc_series, points_aberrants, outliers_decisions)
    imprimer_resume_console(resume, qc_series, points_aberrants, outliers_decisions, output_dir)


if __name__ == "__main__":
    main()
