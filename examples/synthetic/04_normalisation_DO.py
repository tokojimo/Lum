#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
04_normalisation_DO.py

Objectif
--------
Normaliser la luminescence corrigée des blancs par la DO corrigée des blancs,
avec une règle plus robuste au début de culture.

Approche
--------
1. Le seuil DO n'est pas fixé arbitrairement.
   Il est calculé à partir des blancs corrigés :
       seuil_blanc = moyenne(DO_corr des blancs) + k * SD(DO_corr des blancs)
2. On impose aussi une DO minimale pratique (--do-min, défaut 0.05).
3. Le seuil effectivement utilisé est donc :
       seuil_effectif = max(seuil_blanc, do_min)
4. Pour chaque série/puits, la normalisation ne commence qu'au premier temps où
   la série dépasse le seuil_effectif sur N temps consécutifs.
5. Ensuite, on calcule :
       Lum_norm = Lum_corr / DO_corr
   seulement pour les lignes qui respectent toutes les conditions.

Sorties
-------
- ..._normalise_DO.csv
- resume_normalisation_DO.csv
- resume_normalisation_par_souche.csv
- lignes_non_normalisees.csv
- seuil_DO_details.csv
- validation_series_normalisation.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLONNES_OBLIGATOIRES = [
    "temps_h",
    "souche",
    "DO_corr",
    "Lum_corr",
    "type",
]

COLONNES_NUMERIQUES = [
    "temps_h",
    "replicat",
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
]

COLONNES_TEXTE = [
    "souche",
    "Groupe",
    "type",
    "puits",
    "sample_header",
    "blanc_associe",
]


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


def inferer_output_dir(input_path: Path, output_dir_arg: Path | None) -> Path:
    if output_dir_arg is not None:
        out = output_dir_arg.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    parent_name = input_path.parent.name
    if parent_name.startswith("CORR_"):
        out = (input_path.parent.parent / parent_name.replace("CORR_", "NORM_", 1)).resolve()
    else:
        base_name = input_path.stem.replace("_corrige_blancs", "")
        out = (input_path.parent / f"NORM_{base_name}").resolve()
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
        df["type"] = df["type"].fillna("").astype(str).str.strip().str.lower()

    if "sample_header" not in df.columns:
        if "puits" in df.columns:
            df["sample_header"] = df["souche"].astype(str) + " (" + df["puits"].astype(str) + ")"
        else:
            rep_txt = pd.to_numeric(df.get("replicat"), errors="coerce").astype("Int64").astype(str)
            df["sample_header"] = df["souche"].astype(str) + "_rep" + rep_txt

    # Nettoyage des colonnes produites par d'anciennes versions du script.
    for col in [
        "Lum_norm",
        "normalisation_ok",
        "raison_exclusion_norm",
        "do_threshold_utilise",
        "do_threshold_blanc",
        "do_min_utilise",
        "n_points_consecutifs_utilise",
        "serie_validee_pour_normalisation",
        "temps_debut_normalisation_serie_h",
    ]:
        if col in df.columns:
            df = df.drop(columns=[col])

    sort_cols = [c for c in ["type", "souche", "sample_header", "temps_h"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def calculer_seuil_depuis_blancs(df: pd.DataFrame, k_sd: float) -> tuple[float, pd.DataFrame]:
    blancs = df.loc[(df["type"] == "blanc") & df["DO_corr"].notna(), ["temps_h", "DO_corr"]].copy()
    details = []

    if blancs.empty:
        details.append({
            "metrique": "n_points_blancs_valides",
            "valeur": 0,
        })
        details.append({"metrique": "moyenne_DO_corr_blancs", "valeur": np.nan})
        details.append({"metrique": "sd_DO_corr_blancs", "valeur": np.nan})
        details.append({"metrique": "k_sd", "valeur": k_sd})
        details.append({"metrique": "seuil_blanc_calcule", "valeur": np.nan})
        return np.nan, pd.DataFrame(details)

    mean_blank = float(blancs["DO_corr"].mean())
    sd_blank = float(blancs["DO_corr"].std(ddof=1)) if len(blancs) >= 2 else np.nan

    if np.isnan(sd_blank):
        seuil_blanc = mean_blank
    else:
        seuil_blanc = mean_blank + (k_sd * sd_blank)

    details.append({"metrique": "n_points_blancs_valides", "valeur": int(len(blancs))})
    details.append({"metrique": "moyenne_DO_corr_blancs", "valeur": mean_blank})
    details.append({"metrique": "sd_DO_corr_blancs", "valeur": sd_blank})
    details.append({"metrique": "k_sd", "valeur": k_sd})
    details.append({"metrique": "seuil_blanc_calcule", "valeur": seuil_blanc})
    return float(seuil_blanc), pd.DataFrame(details)


def groupe_serie_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    if "experience_id" in df.columns:
        cols.append("experience_id")
    cols.extend([c for c in ["souche", "sample_header"] if c in df.columns])
    return cols


def premier_temps_valide_consecutif(sub: pd.DataFrame, seuil_effectif: float, n_consecutive: int) -> float:
    sub = sub.sort_values("temps_h")
    do_vals = pd.to_numeric(sub["DO_corr"], errors="coerce").to_numpy(dtype=float)
    time_vals = pd.to_numeric(sub["temps_h"], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(do_vals) & (do_vals > seuil_effectif)
    if len(valid) < n_consecutive:
        return np.nan

    for i in range(0, len(valid) - n_consecutive + 1):
        window = valid[i:i + n_consecutive]
        if bool(np.all(window)):
            return float(time_vals[i])
    return np.nan


def construire_validation_series(df: pd.DataFrame, seuil_blanc: float, do_min: float, n_consecutive: int) -> pd.DataFrame:
    seuil_blanc_safe = seuil_blanc if pd.notna(seuil_blanc) else -np.inf
    seuil_effectif = float(max(seuil_blanc_safe, do_min))

    rows: list[dict[str, object]] = []
    souches = df.loc[df["type"] == "souche"].copy()
    if souches.empty:
        return pd.DataFrame(columns=[
            "experience_id", "souche", "sample_header", "puits", "replicat",
            "seuil_blanc", "do_min", "seuil_effectif", "n_points_consecutifs_utilise",
            "temps_debut_normalisation_serie_h", "serie_validee_pour_normalisation",
            "n_points_total_serie", "n_points_au_dessus_seuil_effectif",
        ])

    for _, sub in souches.groupby(groupe_serie_cols(souches), dropna=False):
        sub = sub.sort_values("temps_h")
        t0 = premier_temps_valide_consecutif(sub, seuil_effectif=seuil_effectif, n_consecutive=n_consecutive)
        do_vals = pd.to_numeric(sub["DO_corr"], errors="coerce")
        rows.append({
            "experience_id": sub["experience_id"].iloc[0] if "experience_id" in sub.columns else "",
            "souche": sub["souche"].iloc[0] if "souche" in sub.columns else "",
            "sample_header": sub["sample_header"].iloc[0] if "sample_header" in sub.columns else "",
            "puits": sub["puits"].iloc[0] if "puits" in sub.columns else "",
            "replicat": sub["replicat"].iloc[0] if "replicat" in sub.columns else pd.NA,
            "seuil_blanc": seuil_blanc,
            "do_min": do_min,
            "seuil_effectif": seuil_effectif,
            "n_points_consecutifs_utilise": n_consecutive,
            "temps_debut_normalisation_serie_h": t0,
            "serie_validee_pour_normalisation": pd.notna(t0),
            "n_points_total_serie": int(len(sub)),
            "n_points_au_dessus_seuil_effectif": int(((do_vals > seuil_effectif) & do_vals.notna()).sum()),
        })

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["experience_id", "souche", "replicat", "sample_header"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def determiner_raison_exclusion(row: pd.Series) -> str:
    type_val = str(row.get("type", "")).strip().lower()
    do_corr = row.get("DO_corr", np.nan)
    lum_corr = row.get("Lum_corr", np.nan)
    seuil_effectif = row.get("do_threshold_utilise", np.nan)
    t0 = row.get("temps_debut_normalisation_serie_h", np.nan)
    time_val = row.get("temps_h", np.nan)
    serie_ok = bool(row.get("serie_validee_pour_normalisation", False))

    if type_val == "blanc":
        return "type_blanc"
    if pd.isna(do_corr):
        return "DO_corr_manquante"
    if pd.isna(lum_corr):
        return "Lum_corr_manquante"
    if not serie_ok:
        return "serie_pas_validee"
    if pd.notna(t0) and pd.notna(time_val) and float(time_val) < float(t0):
        return "avant_debut_normalisation"
    if pd.notna(seuil_effectif) and float(do_corr) <= float(seuil_effectif):
        return "DO_corr<=seuil_effectif"
    return ""


def appliquer_normalisation(
    df: pd.DataFrame,
    validation_series: pd.DataFrame,
    seuil_blanc: float,
    do_min: float,
    n_consecutive: int,
) -> pd.DataFrame:
    out = df.copy()
    seuil_blanc_safe = seuil_blanc if pd.notna(seuil_blanc) else -np.inf
    seuil_effectif = float(max(seuil_blanc_safe, do_min))

    merge_cols = [c for c in ["experience_id", "souche", "sample_header"] if c in out.columns and c in validation_series.columns]
    extra_cols = [
        "temps_debut_normalisation_serie_h",
        "serie_validee_pour_normalisation",
        "seuil_effectif",
        "n_points_consecutifs_utilise",
    ]
    if merge_cols:
        out = out.merge(validation_series[merge_cols + extra_cols], on=merge_cols, how="left")
    else:
        out["temps_debut_normalisation_serie_h"] = np.nan
        out["serie_validee_pour_normalisation"] = False
        out["seuil_effectif"] = seuil_effectif
        out["n_points_consecutifs_utilise"] = n_consecutive

    out["do_threshold_blanc"] = seuil_blanc
    out["do_min_utilise"] = do_min
    out["do_threshold_utilise"] = pd.to_numeric(out.get("seuil_effectif"), errors="coerce")
    out["raison_exclusion_norm"] = out.apply(determiner_raison_exclusion, axis=1)
    out["normalisation_ok"] = out["raison_exclusion_norm"].eq("")
    out["Lum_norm"] = np.where(out["normalisation_ok"], out["Lum_corr"] / out["DO_corr"], np.nan)
    out["DO_corr_positive"] = out["DO_corr"] > 0
    out["Lum_corr_positive"] = out["Lum_corr"] > 0
    return out


def construire_resume(df_norm: pd.DataFrame, seuil_blanc: float, do_min: float, n_consecutive: int) -> pd.DataFrame:
    lignes = []

    def add(metric: str, value: object) -> None:
        lignes.append({"metrique": metric, "valeur": value})

    add("seuil_blanc_calcule", seuil_blanc)
    add("do_min_utilise", do_min)
    add("seuil_effectif_utilise", max(seuil_blanc if pd.notna(seuil_blanc) else -np.inf, do_min))
    add("n_points_consecutifs_utilise", n_consecutive)
    add("lignes_total", len(df_norm))
    add("lignes_souches", int((df_norm["type"] == "souche").sum()))
    add("lignes_blancs", int((df_norm["type"] == "blanc").sum()))
    if "serie_validee_pour_normalisation" in df_norm.columns:
        series_valid = df_norm[["sample_header", "serie_validee_pour_normalisation"]].drop_duplicates().copy()
        series_valid["serie_validee_pour_normalisation"] = series_valid["serie_validee_pour_normalisation"].astype(bool)
        add("series_validees", int(series_valid["serie_validee_pour_normalisation"].sum()))
    else:
        add("series_validees", 0)
    add("lignes_normalisees", int(df_norm["normalisation_ok"].sum()))
    add("lignes_non_normalisees", int((~df_norm["normalisation_ok"]).sum()))

    raisons = (
        df_norm.loc[~df_norm["normalisation_ok"], "raison_exclusion_norm"]
        .value_counts(dropna=False)
        .sort_index()
    )
    for raison, n in raisons.items():
        add("raison_" + (str(raison) if str(raison) else "aucune"), int(n))

    valid = df_norm.loc[df_norm["normalisation_ok"], "Lum_norm"].dropna()
    add("Lum_norm_min", float(valid.min()) if not valid.empty else np.nan)
    add("Lum_norm_mediane", float(valid.median()) if not valid.empty else np.nan)
    add("Lum_norm_moyenne", float(valid.mean()) if not valid.empty else np.nan)
    add("Lum_norm_max", float(valid.max()) if not valid.empty else np.nan)
    return pd.DataFrame(lignes)


def construire_resume_par_souche(df_norm: pd.DataFrame) -> pd.DataFrame:
    souches = df_norm.loc[df_norm["type"] == "souche"].copy()
    if souches.empty:
        return pd.DataFrame()

    rows = []
    for souche, sub in souches.groupby("souche", dropna=False):
        valid = sub.loc[sub["normalisation_ok"], "Lum_norm"].dropna()
        n_total = len(sub)
        n_ok = int(sub["normalisation_ok"].sum())
        series = sub[[c for c in ["sample_header", "serie_validee_pour_normalisation"] if c in sub.columns]].drop_duplicates()
        if not series.empty and "serie_validee_pour_normalisation" in series.columns:
            n_series_valid = int(series["serie_validee_pour_normalisation"].astype(bool).sum())
        else:
            n_series_valid = 0
        rows.append({
            "souche": souche,
            "n_lignes": n_total,
            "n_lignes_normalisees": n_ok,
            "n_lignes_non_normalisees": n_total - n_ok,
            "fraction_lignes_normalisees": (n_ok / n_total) if n_total else np.nan,
            "n_series_validees": n_series_valid,
            "Lum_norm_min": float(valid.min()) if not valid.empty else np.nan,
            "Lum_norm_mediane": float(valid.median()) if not valid.empty else np.nan,
            "Lum_norm_moyenne": float(valid.mean()) if not valid.empty else np.nan,
            "Lum_norm_max": float(valid.max()) if not valid.empty else np.nan,
        })
    return pd.DataFrame(rows).sort_values("souche").reset_index(drop=True)


def construire_lignes_non_normalisees(df_norm: pd.DataFrame) -> pd.DataFrame:
    cols_prefer = [
        "temps_h",
        "souche",
        "Groupe",
        "replicat",
        "type",
        "puits",
        "sample_header",
        "blanc_associe",
        "DO_corr",
        "Lum_corr",
        "do_threshold_utilise",
        "temps_debut_normalisation_serie_h",
        "raison_exclusion_norm",
    ]
    cols = [c for c in cols_prefer if c in df_norm.columns]
    sort_cols = [c for c in ["type", "souche", "sample_header", "temps_h"] if c in cols]
    out = df_norm.loc[~df_norm["normalisation_ok"], cols].copy()
    if sort_cols:
        out = out.sort_values(sort_cols)
    return out.reset_index(drop=True)


def parser_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalisation robuste de la luminescence corrigée par la DO corrigée.")
    parser.add_argument("input_file", type=Path, help="Fichier CSV/XLSX issu de 03_correction_blancs.py")
    parser.add_argument(
        "--k-sd",
        type=float,
        default=3.0,
        help="Nombre de SD utilisé pour calculer le seuil à partir des blancs corrigés. Défaut : 3",
    )
    parser.add_argument(
        "--do-min",
        type=float,
        default=0.05,
        help="DO minimale pratique imposée en plus du seuil dérivé des blancs. Défaut : 0.05",
    )
    parser.add_argument(
        "--n-consecutive",
        type=int,
        default=3,
        help="Nombre de points consécutifs au-dessus du seuil requis pour démarrer la normalisation. Défaut : 3",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Dossier de sortie. Par défaut : dossier frère NORM_<...> si l'entrée vient d'un dossier CORR_<...>.",
    )
    return parser.parse_args()


def main() -> None:
    args = parser_args()

    if args.k_sd < 0:
        raise ValueError("--k-sd doit être >= 0.")
    if args.do_min < 0:
        raise ValueError("--do-min doit être >= 0.")
    if args.n_consecutive < 1:
        raise ValueError("--n-consecutive doit être >= 1.")

    input_path = args.input_file.resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier d'entrée introuvable : {input_path}")

    output_dir = inferer_output_dir(input_path, args.output_dir)

    df = lire_table(input_path)
    verifier_colonnes(df, COLONNES_OBLIGATOIRES)
    df = preparer_df(df)

    seuil_blanc, details_seuil = calculer_seuil_depuis_blancs(df, k_sd=args.k_sd)
    validation_series = construire_validation_series(
        df=df,
        seuil_blanc=seuil_blanc,
        do_min=args.do_min,
        n_consecutive=args.n_consecutive,
    )
    df_norm = appliquer_normalisation(
        df=df,
        validation_series=validation_series,
        seuil_blanc=seuil_blanc,
        do_min=args.do_min,
        n_consecutive=args.n_consecutive,
    )
    resume = construire_resume(
        df_norm=df_norm,
        seuil_blanc=seuil_blanc,
        do_min=args.do_min,
        n_consecutive=args.n_consecutive,
    )
    resume_par_souche = construire_resume_par_souche(df_norm)
    lignes_non_normalisees = construire_lignes_non_normalisees(df_norm)

    base = nettoyer_nom_fichier(input_path.stem)
    path_norm = output_dir / f"{base}_normalise_DO.csv"
    path_resume = output_dir / "resume_normalisation_DO.csv"
    path_resume_souche = output_dir / "resume_normalisation_par_souche.csv"
    path_non_norm = output_dir / "lignes_non_normalisees.csv"
    path_details = output_dir / "seuil_DO_details.csv"
    path_validation = output_dir / "validation_series_normalisation.csv"

    df_norm.to_csv(path_norm, index=False, encoding="utf-8-sig")
    resume.to_csv(path_resume, index=False, encoding="utf-8-sig")
    resume_par_souche.to_csv(path_resume_souche, index=False, encoding="utf-8-sig")
    lignes_non_normalisees.to_csv(path_non_norm, index=False, encoding="utf-8-sig")
    details_seuil.to_csv(path_details, index=False, encoding="utf-8-sig")
    validation_series.to_csv(path_validation, index=False, encoding="utf-8-sig")

    seuil_effectif = max(seuil_blanc if pd.notna(seuil_blanc) else -np.inf, args.do_min)
    print("\n[OK] Normalisation DO terminée.")
    print(f"[OK] Données normalisées : {path_norm}")
    print(f"[OK] Résumé global : {path_resume}")
    print(f"[OK] Résumé par souche : {path_resume_souche}")
    print(f"[OK] Lignes non normalisées : {path_non_norm}")
    print(f"[OK] Détails seuil DO : {path_details}")
    print(f"[OK] Validation des séries : {path_validation}")
    print(f"[INFO] Seuil blanc calculé : {seuil_blanc}")
    print(f"[INFO] DO minimale imposée : {args.do_min}")
    print(f"[INFO] Seuil effectif utilisé : {seuil_effectif}")
    print(f"[INFO] Points consécutifs requis : {args.n_consecutive}")
    print(f"[INFO] Lignes normalisées : {int(df_norm['normalisation_ok'].sum())}")
    print(f"[INFO] Lignes non normalisées : {int((~df_norm['normalisation_ok']).sum())}")


if __name__ == "__main__":
    main()
