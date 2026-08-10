#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_courbes_mixte_double_axe_par_milieu.py

Objectif
--------
Générer un panneau PNG combiné par milieu avec 2 axes Y :

- axe Y gauche  : croissance (DO_corr)
- axe Y droite  : luminescence (Lum_corr, échelle linéaire)

Organisation
------------
- exactement la même logique de panneau que "panel_croissance_par_milieu"
  et "panel_LUMINESCENCE_par_milieu"
- un sous-graphe par milieu
- dans chaque sous-graphe : une couleur par souche
- croissance en trait plein
- luminescence en pointillés

Style
-----
- aucune courbe de réplicat n'est affichée dans la figure principale
- points + lignes = moyenne
- barres d'erreur = SD
- si plusieurs expériences :
    résumé = moyenne ± SD des expériences biologiques
- si une seule expérience :
    résumé = moyenne ± SD des séries techniques

Entrée
------
- un ou plusieurs fichiers CSV/XLSX/XLS pour la luminescence, contenant au minimum :
  temps_h, souche, type, Lum_corr
- un fichier Excel séparé pour la croissance en tube, fourni avec --do-file

Formats attendus dans la colonne "souche"
-----------------------------------------
- LB [14.1Ac]
- SCFM1 [14.1Ac]
- Old SCFM2 [14.1Ac]
- New SCFM2 [14.1Ac]

ou tolérés :
- LB 14.1Ac
- SCFM1 14.1Ac
- Old SCFM2 14.1Ac
- New SCFM2 14.1Ac
"""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import LogFormatterMathtext



# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

COLONNES_MIN_LUM = ["temps_h", "souche", "type", "Lum_corr"]

MEDIUM_ORDER = {
    "LB": 0,
    "SCFM1": 1,
    "Old SCFM2": 2,
    "New SCFM2": 3,
    "SCFM2 (Po)": 4,
    "SCFM2 ØPo": 4,
    "SCFM2 ØPO": 4,
    "SCFM2 0Po": 4,
}

# Couleurs fixées pour rester identiques à la figure luminescence seule :
# P0 = bleu, PspeD2-1A = rouge, PspeD2-3B = vert.
STRAIN_COLORS = {
    "14.1Ac": "#1f77b4",  # bleu
    "14.1Ac attB::MiniCTXlux(P0-lux)": "#1f77b4",  # bleu
    "14.1Ac attB::MiniCTXlux(PspeD-lux)": "#2ca02c",  # vert
    "14.1Ac attB::MiniCTXlux(PspeD2-1A-lux)": "#d62728",  # rouge
    "14.1Ac attB::MiniCTXlux(PspeD2-3B-lux)": "#2ca02c",  # vert
    "14.1Ac attB::MiniCTXlux(PspeE-lux)": "#8c564b",  # marron
    # variantes non canonisées, au cas où elles apparaissent directement
    "14.1Ac attB::P0-lux": "#1f77b4",
    "14.1Ac attB::PspeD2-1A-lux": "#d62728",
    "14.1Ac attB::PspeD2-3B-lux": "#2ca02c",
}

FALLBACK_PALETTE = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#9467bd",
    "#ff7f0e",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

LUM_DASH = (0, (3.0, 2.0))


# -----------------------------------------------------------------------------
# Lecture / préparation
# -----------------------------------------------------------------------------


def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Format non supporté : {path.suffix}")



def clean_text(value: object) -> str:
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text or "table"



def cleaner_basename(path: Path) -> str:
    name = path.stem
    name = name.replace("_normalise_DO", "")
    name = name.replace("_normalise_LUM", "")
    name = name.replace("_corrige_blancs", "")
    return clean_text(name)



def infer_output_dir(input_paths: Sequence[Path], output_dir: Path | None) -> Path:
    """Déterminer le dossier de sortie sans créer un chemin Windows trop long.

    La version précédente concaténait les noms complets des 3 fichiers dans le
    nom du dossier de sortie. Sous Windows, ce chemin peut dépasser la limite
    classique de longueur et provoquer WinError 3. Ici, le nom est volontairement
    court.
    """
    if output_dir is not None:
        out = output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if len(input_paths) == 1:
        input_path = input_paths[0]
        parent_name = input_path.parent.name
        if parent_name.startswith("NORM_"):
            out = (input_path.parent.parent / parent_name.replace("NORM_", "FIG_", 1)).resolve()
        else:
            out = (input_path.parent / "FIG_MIXTE").resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if all(p.parent.name.startswith("NORM_") for p in input_paths):
        out_root = input_paths[0].parent.parent
    else:
        out_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))

    # Nom court pour éviter les erreurs Windows de chemin trop long.
    out = (out_root / f"FIG_MIXTE_multi_{len(input_paths)}rep").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out



def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")



def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["temps_h", "replicat", "DO_corr", "Lum_corr", "Lum_norm", "DO_brute", "Lum_brute"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")

    if "normalisation_ok" in out.columns:
        try:
            out["normalisation_ok"] = out["normalisation_ok"].astype("boolean")
        except Exception:
            pass

    sort_cols = [c for c in ["experience_id", "souche", "sample_header", "puits", "temps_h"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    return out



def add_experience_ids(
    dfs: list[pd.DataFrame],
    input_paths: Sequence[Path],
    explicit_ids: Sequence[str] | None,
) -> pd.DataFrame:
    out = []

    if explicit_ids is not None and len(explicit_ids) != len(input_paths):
        raise ValueError("Le nombre de --experience-ids doit correspondre au nombre de fichiers d'entrée.")

    for i, (df, path) in enumerate(zip(dfs, input_paths), start=1):
        cur = df.copy()

        if explicit_ids is not None:
            exp_id = explicit_ids[i - 1]
        elif "experience_id" in cur.columns and cur["experience_id"].notna().any():
            values = cur["experience_id"].dropna().astype(str).unique().tolist()
            exp_id = values[0] if values else f"exp{i}"
        else:
            exp_id = cleaner_basename(path)

        cur["experience_id"] = str(exp_id)
        cur["source_file"] = str(path)
        out.append(cur)

    return pd.concat(out, ignore_index=True)



def infer_series_id(df: pd.DataFrame) -> pd.Series:
    if "puits" in df.columns and df["puits"].notna().any():
        base = df["puits"].astype(str)
    elif "sample_header" in df.columns and df["sample_header"].notna().any():
        base = df["sample_header"].astype(str)
    else:
        souche = df["souche"].astype(str) if "souche" in df.columns else pd.Series(["serie"] * len(df), index=df.index)
        rep = (
            df["replicat"].astype("Int64").astype(str)
            if "replicat" in df.columns
            else pd.Series(["NA"] * len(df), index=df.index)
        )
        base = souche + "__rep" + rep

    exp = (
        df["experience_id"].astype(str)
        if "experience_id" in df.columns
        else pd.Series(["exp"] * len(df), index=df.index)
    )
    return exp + "::__" + base


# -----------------------------------------------------------------------------
# Parsing de la colonne "souche"
# -----------------------------------------------------------------------------


def normalize_condition_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



def extract_medium_and_strain(value: object) -> tuple[str | None, str | None]:
    if pd.isna(value):
        return None, None

    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)

    m = re.fullmatch(r"(.+?)\s*\[\s*(.+?)\s*\]", text)
    if m:
        medium = m.group(1).strip()
        strain = m.group(2).strip()
        return medium or None, strain or None

    m = re.search(r"\b14\.1Ac\b.*$", text)
    if m:
        medium = text[:m.start()].strip()
        strain = text[m.start():].strip()
        return (medium or None), (strain or None)

    return None, None



def is_blank_condition(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return bool(re.match(r"^(blanc|blank)\s*\d*$", text))



def canonicalize_medium_label(value: object) -> str | None:
    """Harmoniser les noms de milieux entre le fichier DO tube et les CSV luminescence.

    Exemple : "SCFM2 ØPO", "SCFM2 ØPo", "SCFM2 0Po" et "SCFM2 (Po)"
    sont ramenés au même libellé, afin d'éviter deux panels séparés.
    """
    if pd.isna(value):
        return None
    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    norm = text.replace("Ø", "0").replace("ø", "0")
    norm_simple = re.sub(r"[\s()_\-]+", "", norm).lower()

    # Variantes observées pour le même milieu.
    if norm_simple in {"scfm2po", "scfm20po", "scfm2opo"}:
        return "SCFM2 (Po)"

    return text



def canonicalize_strain_label(value: object) -> str | None:
    """Harmoniser les noms de souches/promoteurs entre fichiers luminescence et DO tube.

    Le fichier de croissance en tube peut contenir des libellés courts comme :
    - 14.1Ac attB::ØP-lux
    - 14.1Ac attB::PspeD2-1A-lux

    Les fichiers luminescence utilisent souvent :
    - 14.1Ac attB::MiniCTXlux(P0-lux)
    - 14.1Ac attB::MiniCTXlux(PspeD2-1A-lux)
    """
    if pd.isna(value):
        return None

    text = str(value).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None

    # Déjà au format canonique MiniCTXlux(...)
    if "MiniCTXlux" in text:
        return text

    # Souche sauvage simple
    if text == "14.1Ac":
        return text

    m = re.fullmatch(r"14\.1Ac\s+attB::(.+)", text)
    if m:
        promoter = m.group(1).strip()
        promoter_clean = promoter.replace(" ", "")
        promoter_key = promoter_clean.replace("Ø", "0").replace("O", "0")

        # Contrôle P0/ØP, selon les notations utilisées dans les fichiers.
        p0_aliases = {
            "P0-lux",
            "0P-lux",
            "P0lux",
            "0Plux",
            "P0",
            "0P",
            "ØP-lux",
            "PØ-lux",
            "ØPo-lux",
        }
        if promoter_clean in p0_aliases or promoter_key in {"P0-lux", "0P-lux", "P0lux", "0Plux", "P0", "0P"}:
            promoter_clean = "P0-lux"

        return f"14.1Ac attB::MiniCTXlux({promoter_clean})"

    return text



def parse_do_tube_header(label: object, forced_medium: str | None = None) -> tuple[str | None, str | None]:
    if pd.isna(label):
        return None, None

    text = str(label).replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return None, None

    # Format attendu dans le fichier DO tube : "SCFM2 ØPo + 14.1Ac attB::..."
    parts = re.split(r"\s+\+\s+", text, maxsplit=1)
    if len(parts) == 2:
        medium, strain = parts[0].strip(), parts[1].strip()
    else:
        medium, strain = extract_medium_and_strain(text)

    if forced_medium:
        medium = forced_medium

    return canonicalize_medium_label(medium), canonicalize_strain_label(strain)



def read_do_tube_excel(
    path: Path,
    sheet_name: str | int = 0,
    forced_medium: str | None = None,
) -> pd.DataFrame:
    """Lire le fichier Excel de croissance en tube organisé par blocs REP_1/REP_2/REP_3.

    Structure attendue :
    - une ligne d'en-tête par réplicat biologique, avec REP_1 en colonne A
    - la colonne A des lignes suivantes contient le temps
    - les colonnes B:J contiennent les réplicats techniques de DO
    """
    raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
    rows: list[dict[str, object]] = []

    current_rep: str | None = None
    current_headers: dict[int, tuple[str | None, str | None, str]] = {}

    for idx, row in raw.iterrows():
        first = row.iloc[0]

        # Nouvelle ligne de bloc REP_1 / REP_2 / REP_3
        if isinstance(first, str) and re.match(r"^\s*REP[_\s-]*\d+\s*$", first, flags=re.IGNORECASE):
            current_rep = re.sub(r"\s+", "_", first.strip().upper())
            current_headers = {}
            for col_idx in range(1, len(row)):
                header = row.iloc[col_idx]
                if pd.isna(header):
                    continue
                medium, strain = parse_do_tube_header(header, forced_medium=forced_medium)
                if medium is None or strain is None:
                    continue
                current_headers[col_idx] = (medium, strain, str(header))
            continue

        if current_rep is None or not current_headers:
            continue

        time_value = pd.to_numeric(first, errors="coerce")
        if pd.isna(time_value):
            continue

        for col_idx, (medium, strain, raw_header) in current_headers.items():
            value = pd.to_numeric(row.iloc[col_idx], errors="coerce")
            if pd.isna(value):
                continue

            rows.append({
                "experience_id": current_rep,
                "source_file": str(path),
                "temps_h": float(time_value),
                "souche": f"{medium} [{strain}]",
                "condition_norm": f"{medium} [{strain}]",
                "medium": medium,
                "strain": strain,
                "type": "souche",
                "DO_corr": float(value),
                "sample_header": raw_header,
                "puits": f"tube_col_{col_idx}",
                "series_id": f"{current_rep}::__tube_col_{col_idx}",
            })

    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(
            "Aucune valeur de DO exploitable détectée dans le fichier --do-file. "
            "Vérifie que le fichier est organisé en blocs REP_1/REP_2/REP_3."
        )

    return prepare_dataframe(out)



def prepare_growth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "condition_norm" not in out.columns:
        out["condition_norm"] = out["souche"].map(normalize_condition_label)
    else:
        out["condition_norm"] = out["condition_norm"].map(normalize_condition_label)

    if "medium" not in out.columns or "strain" not in out.columns or out["medium"].isna().all() or out["strain"].isna().all():
        parsed = out["condition_norm"].map(extract_medium_and_strain)
        out["medium"] = parsed.map(lambda x: x[0])
        out["strain"] = parsed.map(lambda x: x[1])

    out["medium"] = out["medium"].map(canonicalize_medium_label)
    out["strain"] = out["strain"].map(canonicalize_strain_label)

    if "type" in out.columns:
        out = out.loc[out["type"].astype(str).str.lower() == "souche"].copy()

    value_cols = [c for c in ["DO_corr", "Lum_corr"] if c in out.columns]
    if value_cols:
        mask_value = False
        for c in value_cols:
            mask_value = mask_value | out[c].notna()
        out = out.loc[mask_value].copy()

    out = out.loc[out["condition_norm"].notna()].copy()
    out = out.loc[~out["condition_norm"].astype(str).map(is_blank_condition)].copy()
    out = out.loc[out["medium"].notna() & out["strain"].notna()].copy()

    # Conservation du comportement des scripts d'origine
    out = out.loc[out["medium"] != "SCFM1"].copy()

    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Ordres et couleurs
# -----------------------------------------------------------------------------


def build_medium_order(values: Sequence[str]) -> list[str]:
    uniq = pd.Series(list(values), dtype="string").dropna().astype(str).drop_duplicates().tolist()
    uniq = sorted(uniq, key=lambda x: (MEDIUM_ORDER.get(x, 99), x.lower()))
    return uniq



def build_strain_order(values: Sequence[str]) -> list[str]:
    uniq = pd.Series(list(values), dtype="string").dropna().astype(str).drop_duplicates().tolist()

    known = [s for s in STRAIN_COLORS if s in uniq]
    extra = sorted([s for s in uniq if s not in STRAIN_COLORS], key=str.lower)
    return known + extra



def build_complete_color_map(order: Sequence[str], fixed_map: dict[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    fallback_iter = iter(FALLBACK_PALETTE * 10)

    for item in order:
        if item in fixed_map:
            out[item] = fixed_map[item]
        else:
            color = next(fallback_iter)
            while color in out.values():
                color = next(fallback_iter)
            out[item] = color
    return out


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------


def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "legend.frameon": False,
    })



def mm_to_inch(mm: float) -> float:
    return mm / 25.4



def save_png(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)



def compute_panel_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    if n_panels == 1:
        return 1, 1
    if n_panels == 2:
        return 1, 2
    if n_panels <= 4:
        return 2, 2

    ncols = math.ceil(math.sqrt(n_panels))
    nrows = math.ceil(n_panels / ncols)
    return nrows, ncols


# -----------------------------------------------------------------------------
# Agrégation
# -----------------------------------------------------------------------------


def aggregate_curve_data(
    df: pd.DataFrame,
    panel_col: str,
    compare_col: str,
    value_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if value_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), "missing"

    work = df.loc[df[value_col].notna()].copy()
    work = work.loc[work[panel_col].notna() & work[compare_col].notna()].copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), "empty"

    n_exp = int(work["experience_id"].nunique()) if "experience_id" in work.columns else 1

    indiv_lines = (
        work.groupby([panel_col, compare_col, "series_id", "temps_h"], dropna=False, observed=True)[value_col]
        .mean()
        .reset_index(name="line_value")
    )

    if n_exp <= 1:
        summary = (
            indiv_lines.groupby([panel_col, compare_col, "temps_h"], dropna=False, observed=True)["line_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={
                "mean": "mean_value",
                "std": "sd_value",
                "count": "n_units",
            })
        )
        return summary, indiv_lines, "technique"

    exp_lines = (
        work.groupby(["experience_id", panel_col, compare_col, "temps_h"], dropna=False, observed=True)[value_col]
        .mean()
        .reset_index(name="bio_value")
    )

    summary = (
        exp_lines.groupby([panel_col, compare_col, "temps_h"], dropna=False, observed=True)["bio_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={
            "mean": "mean_value",
            "std": "sd_value",
            "count": "n_units",
        })
    )

    return summary, indiv_lines, "biologique"


# -----------------------------------------------------------------------------
# Tracé
# -----------------------------------------------------------------------------


def make_handles(order: Sequence[str], color_map: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color_map[item], lw=2.0, label=item)
        for item in order
        if item in color_map
    ]



def format_panel_title(prefix: str, value: str) -> tuple[str, float]:
    value = str(value).strip()

    if len(value) >= 42:
        return value, 6.0
    if len(value) >= 30:
        return value, 6.4
    return value, 7.0



def clip_log_band(y: np.ndarray, sd: np.ndarray, floor: float) -> tuple[np.ndarray, np.ndarray]:
    low = np.maximum(y - sd, floor)
    high = np.maximum(y + sd, floor)
    return low, high



def make_lum_formatter(scale_power: int = 8) -> FuncFormatter:
    factor = float(10 ** scale_power)

    def _fmt(x: float, pos: int) -> str:
        if not np.isfinite(x) or x <= 0:
            return ""
        return f"{x / factor:.4f}".replace(".", ",")

    return FuncFormatter(_fmt)



def draw_subplot_dual_axis(
    ax_left: plt.Axes,
    summary_do: pd.DataFrame,
    lines_do: pd.DataFrame,
    summary_lum: pd.DataFrame,
    lines_lum: pd.DataFrame,
    panel_value: str,
    compare_order: Sequence[str],
    compare_col: str,
    color_map: dict[str, str],
    title: str,
    title_fontsize: float,
    ylim_left: tuple[float, float] | None = None,
    ylim_right: tuple[float, float] | None = None,
) -> plt.Axes:
    """Tracer uniquement les moyennes ± SD.

    Les courbes de réplicats techniques ou biologiques ne sont pas affichées.
    Les fichiers *_lines.csv restent exportés pour contrôle qualité, mais ils ne
    sont pas utilisés dans la figure principale.
    """
    ax_right = ax_left.twinx()

    sub_sum_do_all = summary_do.loc[summary_do.iloc[:, 0] == panel_value].copy()
    sub_sum_lum_all = summary_lum.loc[summary_lum.iloc[:, 0] == panel_value].copy()

    for compare_value in compare_order:
        color = color_map.get(compare_value, "#333333")

        # -------------------------
        # Croissance (axe gauche) : moyenne ± SD uniquement
        # -------------------------
        sub_sum_do = sub_sum_do_all.loc[sub_sum_do_all[compare_col] == compare_value].sort_values("temps_h")
        if not sub_sum_do.empty:
            x_do = sub_sum_do["temps_h"].to_numpy(dtype=float)
            y_do = sub_sum_do["mean_value"].to_numpy(dtype=float)
            sd_do = sub_sum_do["sd_value"].fillna(0).to_numpy(dtype=float)

            ax_left.errorbar(
                x_do,
                y_do,
                yerr=sd_do,
                color=color,
                marker="o",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.7,
                markersize=2.5,
                linewidth=1.35,
                linestyle="-",
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=3,
            )

        # -------------------------
        # Luminescence (axe droite) : moyenne ± SD uniquement
        # -------------------------
        sub_sum_lum = sub_sum_lum_all.loc[sub_sum_lum_all[compare_col] == compare_value].sort_values("temps_h")
        if not sub_sum_lum.empty:
            x_lum = sub_sum_lum["temps_h"].to_numpy(dtype=float)
            y_lum = sub_sum_lum["mean_value"].to_numpy(dtype=float)
            sd_lum = sub_sum_lum["sd_value"].fillna(0).to_numpy(dtype=float)

            ax_right.errorbar(
                x_lum,
                y_lum,
                yerr=sd_lum,
                color=color,
                marker="s",
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=0.7,
                markersize=2.3,
                linewidth=1.25,
                linestyle=LUM_DASH,
                elinewidth=0.7,
                capsize=1.8,
                capthick=0.7,
                zorder=3,
            )

    ax_left.set_title(title, loc="left", fontweight="bold", fontsize=title_fontsize, pad=15)
    ax_left.set_xlabel("Time (h)")
    ax_left.set_ylabel("Optical density")
    ax_left.grid(False)

    ax_right.set_ylabel("Luminescence")

    x_candidates = []
    if not sub_sum_do_all.empty:
        x_candidates.append(sub_sum_do_all["temps_h"].dropna().to_numpy(dtype=float))
    if not sub_sum_lum_all.empty:
        x_candidates.append(sub_sum_lum_all["temps_h"].dropna().to_numpy(dtype=float))

    if x_candidates:
        x_all = np.concatenate([x for x in x_candidates if len(x) > 0])
        if len(x_all) > 0:
            ax_left.set_xlim(np.nanmin(x_all), np.nanmax(x_all))
    ax_left.margins(x=0.0)

    if ylim_left is not None:
        ax_left.set_ylim(*ylim_left)
    if ylim_right is not None:
        ax_right.set_ylim(*ylim_right)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(True)

    return ax_right


def plot_dual_panel_par_milieu(
    summary_do: pd.DataFrame,
    lines_do: pd.DataFrame,
    summary_lum: pd.DataFrame,
    lines_lum: pd.DataFrame,
    mode_do: str,
    mode_lum: str,
    panel_order: Sequence[str],
    compare_order: Sequence[str],
    compare_col: str,
    color_map: dict[str, str],
    output_path: Path,
    figure_title: str,
    ylim_left: tuple[float, float] | None = None,
    ylim_right: tuple[float, float] | None = None,
) -> None:
    if summary_do.empty and summary_lum.empty:
        return

    n_panels = len(panel_order)

    if n_panels == 3:
        nrows, ncols = 1, 3
        fig_width_mm = 280
        fig_height_mm = 150
    else:
        nrows, ncols = compute_panel_grid(n_panels)
        fig_width_mm = 190
        fig_height_mm = max(95, 72 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for i, panel_value in enumerate(panel_order):
        ax = axes_flat[i]
        panel_title, panel_title_fontsize = format_panel_title("Milieu", panel_value)
        draw_subplot_dual_axis(
            ax_left=ax,
            summary_do=summary_do,
            lines_do=lines_do,
            summary_lum=summary_lum,
            lines_lum=lines_lum,
            panel_value=panel_value,
            compare_order=compare_order,
            compare_col=compare_col,
            color_map=color_map,
            title=panel_title,
            title_fontsize=panel_title_fontsize,
            ylim_left=ylim_left,
            ylim_right=ylim_right,
        )

    for j in range(len(panel_order), len(axes_flat)):
        axes_flat[j].axis("off")

    handles = make_handles(compare_order, color_map)

    do_text = "mean of biological replicates" if mode_do == "biologique" else "mean of technical replicates"
    lum_text = "mean of biological replicates" if mode_lum == "biologique" else "mean of technical replicates"
    subtitle = (
        f"Solid lines with open circles = growth (left axis, {do_text}); "
        f"dashed lines with open squares = luminescence (right axis, {lum_text}); "
        "error bars = SD. Individual replicate curves are not displayed."
    )

    fig.suptitle(figure_title, y=0.998, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.975, subtitle, ha="center", va="top", fontsize=7.2, color="#444444")

    if handles:
        fig.legend(
            handles=handles,
            title="Souche",
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            frameon=False,
            handlelength=2.5,
            columnspacing=1.6,
            labelspacing=0.8,
        )
        fig.subplots_adjust(
            top=0.79,
            bottom=0.08,
            left=0.07,
            right=0.98,
            hspace=0.75,
            wspace=0.45,
        )
    else:
        fig.subplots_adjust(
            top=0.86,
            bottom=0.08,
            left=0.07,
            right=0.98,
            hspace=0.75,
            wspace=0.45,
        )

    save_png(fig, output_path)


# -----------------------------------------------------------------------------
# Rapport
# -----------------------------------------------------------------------------


def write_diagnostics(
    output_dir: Path,
    df_lum: pd.DataFrame,
    df_do: pd.DataFrame,
    do_file: Path,
    medium_order: Sequence[str],
    strain_order: Sequence[str],
    mode_do: str,
    mode_lum: str,
    ylim_do: tuple[float, float],
    ylim_lum: tuple[float, float],
) -> None:
    lines = []
    lines.append("ANALYSE COURBES MIXTES CROISSANCE TUBE + LUMINESCENCE")
    lines.append("")
    lines.append(f"Dossier de sortie : {output_dir}")
    lines.append(f"Fichier DO tube : {do_file}")
    lines.append(f"Nombre de lignes luminescence retenues : {len(df_lum)}")
    lines.append(f"Nombre de lignes DO tube retenues : {len(df_do)}")
    lines.append(f"Nombre d'expériences luminescence : {int(df_lum['experience_id'].nunique())}")
    lines.append(f"Nombre d'expériences DO tube : {int(df_do['experience_id'].nunique())}")
    lines.append("")
    lines.append("Milieux détectés :")
    for x in medium_order:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("Souches détectées :")
    for x in strain_order:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"Mode croissance : {mode_do}")
    lines.append(f"Mode luminescence : {mode_lum}")
    lines.append(f"Axe Y gauche (DO) : {ylim_do}")
    lines.append(f"Axe Y droite (Lum, linéaire) : {ylim_lum}")
    lines.append("")
    lines.append("Convention graphique :")
    lines.append("- trait plein + cercles ouverts = croissance moyenne")
    lines.append("- pointillés + carrés ouverts = luminescence moyenne")
    lines.append("- barres d'erreur = SD")
    lines.append("- couleurs = souches")
    lines.append("- les courbes de réplicats individuels ne sont pas affichées")
    lines.append("")
    lines.append("Exports PNG :")
    lines.append("- panel_croissance_tube_luminescence_double_axe_par_milieu.png")

    (output_dir / "resume_courbes_mixte_double_axe.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def run_analysis(df_lum: pd.DataFrame, df_do: pd.DataFrame, output_dir: Path, do_file: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    medium_order = build_medium_order(
        df_lum["medium"].dropna().astype(str).tolist()
        + df_do["medium"].dropna().astype(str).tolist()
    )
    strain_order = build_strain_order(
        df_lum["strain"].dropna().astype(str).tolist()
        + df_do["strain"].dropna().astype(str).tolist()
    )

    if not medium_order:
        raise ValueError("Aucun milieu exploitable détecté.")
    if not strain_order:
        raise ValueError("Aucune souche exploitable détectée.")

    strain_color_map = build_complete_color_map(strain_order, STRAIN_COLORS)

    do_vals = pd.to_numeric(df_do["DO_corr"], errors="coerce")
    do_vals = do_vals[do_vals.notna()]
    if do_vals.empty:
        raise ValueError("Impossible de tracer DO_corr : aucune valeur exploitable dans le fichier --do-file.")
    global_do_ylim = (0.0, float(do_vals.max()) * 1.05)

    lum_vals = pd.to_numeric(df_lum["Lum_corr"], errors="coerce")
    lum_vals = lum_vals[lum_vals.notna()]
    if lum_vals.empty:
        raise ValueError("Impossible de tracer Lum_corr : aucune valeur exploitable dans les fichiers luminescence.")

    lum_min = float(lum_vals.min())
    lum_max = float(lum_vals.max())
    if lum_min == lum_max:
        lum_pad = abs(lum_max) * 0.05 if lum_max != 0 else 1.0
    else:
        lum_pad = (lum_max - lum_min) * 0.05
    global_lum_ylim = (lum_min - lum_pad, lum_max + lum_pad)

    # Données croissance : prises dans le fichier Excel DO tube.
    summary_do, lines_do, mode_do = aggregate_curve_data(
        df=df_do,
        panel_col="medium",
        compare_col="strain",
        value_col="DO_corr",
    )

    summary_do.to_csv(
        output_dir / "data_panel_mixte_par_milieu_DO_TUBE_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines_do.to_csv(
        output_dir / "data_panel_mixte_par_milieu_DO_TUBE_lines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Données luminescence : prises dans les fichiers de cinétique normalisés.
    summary_lum, lines_lum, mode_lum = aggregate_curve_data(
        df=df_lum,
        panel_col="medium",
        compare_col="strain",
        value_col="Lum_corr",
    )

    summary_lum.to_csv(
        output_dir / "data_panel_mixte_par_milieu_LUM_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines_lum.to_csv(
        output_dir / "data_panel_mixte_par_milieu_LUM_lines.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plot_dual_panel_par_milieu(
        summary_do=summary_do,
        lines_do=lines_do,
        summary_lum=summary_lum,
        lines_lum=lines_lum,
        mode_do=mode_do,
        mode_lum=mode_lum,
        panel_order=medium_order,
        compare_order=strain_order,
        compare_col="strain",
        color_map=strain_color_map,
        output_path=output_dir / "panel_croissance_tube_luminescence_double_axe_par_milieu.png",
        figure_title="Growth and luminescence dynamics",
        ylim_left=global_do_ylim,
        ylim_right=global_lum_ylim,
    )

    write_diagnostics(
        output_dir=output_dir,
        df_lum=df_lum,
        df_do=df_do,
        do_file=do_file,
        medium_order=medium_order,
        strain_order=strain_order,
        mode_do=mode_do,
        mode_lum=mode_lum,
        ylim_do=global_do_ylim,
        ylim_lum=global_lum_ylim,
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générer un panneau combiné DO tube + luminescence par milieu avec double axe Y."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Fichiers CSV/XLSX/XLS de luminescence contenant Lum_corr",
    )
    parser.add_argument(
        "--do-file",
        required=True,
        help="Fichier Excel contenant la DO de croissance en tube, organisé en blocs REP_1/REP_2/REP_3",
    )
    parser.add_argument(
        "--do-sheet",
        default="0",
        help="Nom ou index de la feuille du fichier DO tube. Défaut : 0 = première feuille.",
    )
    parser.add_argument(
        "--do-medium-label",
        default=None,
        help=(
            "Optionnel : forcer le nom du milieu pour toutes les données DO tube. "
            "Utile si le fichier DO indique 'SCFM2 ØPo' mais que les fichiers luminescence utilisent 'Old SCFM2' ou 'New SCFM2'."
        ),
    )
    parser.add_argument(
        "--experience-ids",
        nargs="*",
        default=None,
        help="IDs d'expériences à associer aux fichiers de luminescence",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie (optionnel)",
    )
    return parser.parse_args()



def parse_sheet_arg(value: str) -> str | int:
    value = str(value)
    if re.fullmatch(r"\d+", value):
        return int(value)
    return value



def main() -> None:
    configure_matplotlib()
    args = parse_args()

    input_paths = [Path(p).resolve() for p in args.input_files]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Fichier luminescence introuvable : {p}")

    do_path = Path(args.do_file).resolve()
    if not do_path.exists():
        raise FileNotFoundError(f"Fichier DO tube introuvable : {do_path}")

    # Lecture des fichiers luminescence : on utilise Lum_corr uniquement pour l'axe droit.
    dfs_lum = [prepare_dataframe(read_table(p)) for p in input_paths]
    for df in dfs_lum:
        ensure_columns(df, COLONNES_MIN_LUM)

    df_lum = add_experience_ids(dfs_lum, input_paths, args.experience_ids)
    df_lum = prepare_dataframe(df_lum)
    df_lum["series_id"] = infer_series_id(df_lum)
    df_lum = prepare_growth_dataframe(df_lum)

    # Lecture de la DO depuis le fichier Excel de croissance en tube.
    df_do = read_do_tube_excel(
        do_path,
        sheet_name=parse_sheet_arg(args.do_sheet),
        forced_medium=args.do_medium_label,
    )
    df_do = prepare_growth_dataframe(df_do)

    if df_lum.empty:
        raise ValueError(
            "Aucune donnée luminescence exploitable détectée. "
            "Vérifie la colonne 'souche', la colonne 'type' et la présence de Lum_corr."
        )

    if df_do.empty:
        raise ValueError(
            "Aucune donnée DO tube exploitable détectée. "
            "Vérifie le fichier passé avec --do-file."
        )

    output_dir = infer_output_dir(
        input_paths=input_paths,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    run_analysis(df_lum=df_lum, df_do=df_do, output_dir=output_dir, do_file=do_path)

    print("=" * 80)
    print("FIGURE MIXTE DO TUBE + LUMINESCENCE TERMINEE")
    print("=" * 80)
    print(f"Dossier de sortie : {output_dir}")
    print(f"Fichier DO tube : {do_path}")
    print(f"Expériences luminescence détectées : {int(df_lum['experience_id'].nunique())}")
    print(f"Expériences DO tube détectées : {int(df_do['experience_id'].nunique())}")
    print(f"Milieux détectés : {len(set(df_lum['medium'].dropna().astype(str)) | set(df_do['medium'].dropna().astype(str)))}")
    print(f"Souches détectées : {len(set(df_lum['strain'].dropna().astype(str)) | set(df_do['strain'].dropna().astype(str)))}")
    print("Exports : PNG uniquement")
    print("- panel_croissance_tube_luminescence_double_axe_par_milieu.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
