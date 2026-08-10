
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_figures_par_milieu_et_par_souche.py

Objectif
--------
Produire des figures PNG séparées :
- une figure par milieu (courbes colorées par souche)
- une figure par souche (courbes colorées par milieu)

Sorties générées
----------------
Pour chaque milieu ET pour chaque souche :
- courbes de croissance (DO_corr)
- courbes de luminescence corrigée / non normalisée (Lum_corr)
- courbes de luminescence normalisée (Lum_norm)
- points AUC de luminescence normalisée
- points pic de luminescence normalisée
- histogrammes du temps de doublement (estimé à partir de la pente max de ln(DO_corr))

Remarques
---------
- plus de multipanel
- export PNG uniquement
- en mode merge, les lignes fines représentent les séries/puits,
  la ligne épaisse résume les expériences biologiques
- pour Lum_norm_max, l'axe Y est en log et les valeurs <= 0 sont exclues
"""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

COLONNES_MIN = ["temps_h", "souche", "type", "DO_corr"]

CANONICAL_MEDIA = {
    "lb": "LB",
    "scfm1": "SCFM1",
    "old scfm2": "Old SCFM2",
    "new scfm2": "New SCFM2",
}

MEDIUM_ORDER = {
    "LB": 0,
    "SCFM1": 1,
    "Old SCFM2": 2,
    "New SCFM2": 3,
}

MEDIUM_COLORS = {
    "LB": "#1f77b4",
    "SCFM1": "#d62728",
    "Old SCFM2": "#2ca02c",
    "New SCFM2": "#9467bd",
}

MEDIUM_MARKERS = {
    "LB": "s",
    "SCFM1": "D",
    "Old SCFM2": "o",
    "New SCFM2": "^",
}

STRAIN_COLOR_FIXED = {
    "14.1Ac": "#1f77b4",  # bleu
    "14.1Ac attB::MiniCTXlux(P0-lux)": "#d62728",  # rouge
    "14.1Ac attB::MiniCTXlux(PspeD-lux)": "#2ca02c",  # vert
    "14.1Ac attB::MiniCTXlux(PspeD2-1A-lux)": "#9467bd",  # violet
    "14.1Ac attB::MiniCTXlux(PspeD2-3B-lux)": "#ff7f0e",  # orange
    "14.1Ac attB::MiniCTXlux(PspeE-lux)": "#8c564b",  # marron
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

CURVE_SPECS = {
    "DO_corr": ("Cinétique de croissance", "DO corrigée"),
    "Lum_corr": ("Cinétique de luminescence corrigée", "Luminescence corrigée"),
    "Lum_norm": ("Cinétique de luminescence normalisée", "Luminescence normalisée / DO corrigée"),
}

POINT_SPECS = {
    "AUC_Lum_norm": ("AUC de luminescence normalisée", "AUC luminescence normalisée"),
    "Lum_norm_max": ("Pic de luminescence normalisée", "Pic luminescence normalisée"),
}

BAR_SPECS = {
    "doubling_time_h": ("Temps de doublement", "Temps de doublement (h)"),
}

DOUBLING_MIN_POINTS = 3


# -----------------------------------------------------------------------------
# Utilitaires généraux
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
    name = name.replace("_corrige_blancs", "")
    return clean_text(name)


def infer_output_dir(input_paths: Sequence[Path], output_dir: Path | None) -> Path:
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
            out = (input_path.parent / f"FIG_{cleaner_basename(input_path)}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if all(p.parent.name.startswith("NORM_") for p in input_paths):
        out_root = input_paths[0].parent.parent
    else:
        out_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))

    stems = [cleaner_basename(p) for p in input_paths]
    suffix = "__".join(stems[:3]) if len(stems) <= 3 else f"{stems[0]}__plus_{len(stems)-1}_autres"
    out = (out_root / f"FIG_multi_{suffix}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "temps_h", "replicat", "DO_corr", "Lum_corr", "Lum_norm",
        "DO_brute", "Lum_brute",
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].astype("string")

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
# Parsing des conditions dans "souche"
# -----------------------------------------------------------------------------

def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_condition_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = normalize_spaces(str(value).replace("_", " "))
    if not text:
        return None

    m = re.fullmatch(
        r"(LB|SCFM1|Old\s*SCFM2|New\s*SCFM2)\s*\[\s*(.*?)\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        medium_raw = normalize_spaces(m.group(1)).lower()
        strain = normalize_spaces(m.group(2))
        medium = CANONICAL_MEDIA.get(medium_raw, m.group(1).strip())
        return f"{medium} [{strain}]"

    m = re.fullmatch(
        r"(LB|SCFM1|Old\s*SCFM2|New\s*SCFM2)\s+(.*)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        medium_raw = normalize_spaces(m.group(1)).lower()
        strain = normalize_spaces(m.group(2))
        medium = CANONICAL_MEDIA.get(medium_raw, m.group(1).strip())
        return f"{medium} [{strain}]"

    return text


def extract_medium_and_strain(value: object) -> tuple[str | None, str | None]:
    label = normalize_condition_label(value)
    if label is None:
        return None, None

    m = re.fullmatch(r"(LB|SCFM1|Old SCFM2|New SCFM2)\s*\[\s*(.*?)\s*\]", label)
    if m:
        return m.group(1), normalize_spaces(m.group(2))

    return None, label


def detect_conditions_from_souche(df: pd.DataFrame) -> pd.DataFrame:
    if "souche" not in df.columns:
        raise ValueError("La colonne 'souche' est absente du tableau.")

    parsed = df["souche"].map(extract_medium_and_strain)
    out = df.copy()
    out["medium"] = parsed.map(lambda x: x[0])
    out["strain"] = parsed.map(lambda x: x[1])
    out["figure_group"] = df["souche"].map(normalize_condition_label)
    return out


def is_blank_condition(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return bool(re.match(r"^(blanc|blank)\s*\d*$", text))


def filter_dataframe_for_figures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["type"].astype(str).str.lower() == "souche"].copy()
    out = out.loc[out["figure_group"].notna()].copy()
    out = out.loc[out["medium"].notna() & out["strain"].notna()].copy()
    out = out.loc[~out["figure_group"].astype(str).map(is_blank_condition)].copy()
    out = out.loc[~out["strain"].astype(str).map(is_blank_condition)].copy()
    return out.reset_index(drop=True)


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
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def export_png(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Ordres et couleurs
# -----------------------------------------------------------------------------

def build_medium_order(values: Sequence[str]) -> list[str]:
    unique = []
    seen = set()
    for v in values:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return sorted(unique, key=lambda x: (MEDIUM_ORDER.get(x, 99), x.lower()))


def build_strain_order(values: Sequence[str]) -> list[str]:
    unique = []
    seen = set()
    fixed_order = list(STRAIN_COLOR_FIXED.keys())

    for v in fixed_order:
        if v in values and v not in seen:
            seen.add(v)
            unique.append(v)

    others = sorted([v for v in values if v not in seen], key=lambda x: x.lower())
    unique.extend(others)
    return unique


def build_strain_color_map(strain_order: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    fallback_i = 0
    for strain in strain_order:
        if strain in STRAIN_COLOR_FIXED:
            out[strain] = STRAIN_COLOR_FIXED[strain]
        else:
            out[strain] = FALLBACK_PALETTE[fallback_i % len(FALLBACK_PALETTE)]
            fallback_i += 1
    return out


def build_medium_color_map(medium_order: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    fallback_i = 0
    for medium in medium_order:
        if medium in MEDIUM_COLORS:
            out[medium] = MEDIUM_COLORS[medium]
        else:
            out[medium] = FALLBACK_PALETTE[fallback_i % len(FALLBACK_PALETTE)]
            fallback_i += 1
    return out


def make_line_handles(group_order: Sequence[str], colors: dict[str, str]) -> list[Line2D]:
    return [Line2D([0], [0], color=colors[g], lw=2.0, label=str(g)) for g in group_order if g in colors]


def make_patch_handles(group_order: Sequence[str], colors: dict[str, str]) -> list[Patch]:
    return [Patch(facecolor=colors[g], edgecolor="none", label=str(g), alpha=0.9) for g in group_order if g in colors]


# -----------------------------------------------------------------------------
# Agrégations
# -----------------------------------------------------------------------------

def aggregate_curve_data(df: pd.DataFrame, value_col: str, group_col: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if value_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame(), "missing"

    work = df.loc[df[value_col].notna()].copy()
    work = work.loc[work[group_col].notna()].copy()

    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), "empty"

    n_exp = int(work["experience_id"].nunique()) if "experience_id" in work.columns else 1

    indiv_lines = (
        work.groupby([group_col, "series_id", "temps_h"], dropna=False, observed=True)[value_col]
        .mean()
        .reset_index(name="line_value")
        .rename(columns={group_col: "group_name"})
    )

    if n_exp <= 1:
        summary = (
            indiv_lines.groupby(["group_name", "temps_h"], dropna=False, observed=True)["line_value"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": "mean_value", "std": "sd_value", "count": "n_units"})
        )
        return summary, indiv_lines, "technique"

    exp_lines = (
        work.groupby(["experience_id", group_col, "temps_h"], dropna=False, observed=True)[value_col]
        .mean()
        .reset_index(name="bio_value")
        .rename(columns={group_col: "group_name"})
    )

    summary = (
        exp_lines.groupby(["group_name", "temps_h"], dropna=False, observed=True)["bio_value"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_value", "std": "sd_value", "count": "n_units"})
    )

    return summary, indiv_lines, "biologique"


def trapezoid_auc(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(valid) < 2:
        return np.nan
    return float(np.trapezoid(valid["y"].to_numpy(dtype=float), valid["x"].to_numpy(dtype=float)))


def estimate_growth_metrics(sub: pd.DataFrame) -> tuple[float, float, int]:
    """
    Estime mu_max par régressions linéaires sur fenêtres glissantes de 3 points
    dans ln(DO_corr) vs temps_h.
    Retourne (mu_max_h_inv, doubling_time_h, n_points_positifs_utilises)
    """
    if "DO_corr" not in sub.columns:
        return np.nan, np.nan, 0

    valid = sub.loc[sub["temps_h"].notna() & sub["DO_corr"].notna() & (sub["DO_corr"] > 0)].copy()
    valid = valid.sort_values("temps_h")
    n_valid = int(len(valid))

    if n_valid < DOUBLING_MIN_POINTS:
        return np.nan, np.nan, n_valid

    x = valid["temps_h"].to_numpy(dtype=float)
    y = np.log(valid["DO_corr"].to_numpy(dtype=float))

    slopes = []
    window = DOUBLING_MIN_POINTS
    for start in range(0, len(valid) - window + 1):
        xw = x[start:start + window]
        yw = y[start:start + window]
        if np.unique(xw).size < 2:
            continue
        slope, _intercept = np.polyfit(xw, yw, 1)
        if np.isfinite(slope):
            slopes.append(float(slope))

    if not slopes:
        return np.nan, np.nan, n_valid

    mu_max = float(np.nanmax(slopes))
    if not np.isfinite(mu_max) or mu_max <= 0:
        return mu_max, np.nan, n_valid

    doubling_time = float(np.log(2.0) / mu_max)
    return mu_max, doubling_time, n_valid


def extract_parameters_per_series(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = [c for c in ["experience_id", "medium", "strain", "series_id"] if c in work.columns]

    for keys, sub in work.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        sub = sub.sort_values("temps_h").copy()

        row: dict[str, object] = {**meta}

        if "sample_header" in sub.columns and sub["sample_header"].notna().any():
            row["sample_header"] = str(sub["sample_header"].dropna().iloc[0])
        if "puits" in sub.columns and sub["puits"].notna().any():
            row["puits"] = str(sub["puits"].dropna().iloc[0])
        if "souche" in sub.columns and sub["souche"].notna().any():
            row["souche"] = str(sub["souche"].dropna().iloc[0])

        row["n_points_total"] = int(len(sub))

        row["AUC_DO"] = trapezoid_auc(sub["temps_h"], sub["DO_corr"]) if "DO_corr" in sub.columns else np.nan
        row["DO_max"] = float(sub["DO_corr"].max()) if "DO_corr" in sub.columns and sub["DO_corr"].notna().any() else np.nan
        row["temps_DO_max_h"] = (
            float(sub.loc[sub["DO_corr"].idxmax(), "temps_h"])
            if "DO_corr" in sub.columns and sub["DO_corr"].notna().any()
            else np.nan
        )

        mu_max, doubling_time_h, n_pos = estimate_growth_metrics(sub)
        row["mu_max_h_inv"] = mu_max
        row["doubling_time_h"] = doubling_time_h
        row["n_points_DO_pos"] = n_pos

        if "Lum_norm" in sub.columns:
            valid = sub.loc[sub["Lum_norm"].notna()].copy()
            row["n_points_Lum_norm"] = int(len(valid))
            row["temps_debut_norm_h"] = float(valid["temps_h"].min()) if not valid.empty else np.nan
            row["temps_fin_norm_h"] = float(valid["temps_h"].max()) if not valid.empty else np.nan
            row["AUC_Lum_norm"] = trapezoid_auc(valid["temps_h"], valid["Lum_norm"]) if not valid.empty else np.nan
            row["Lum_norm_max"] = float(valid["Lum_norm"].max()) if not valid.empty else np.nan
            row["Lum_norm_final"] = float(valid.sort_values("temps_h")["Lum_norm"].iloc[-1]) if not valid.empty else np.nan
            row["temps_pic_h"] = float(valid.loc[valid["Lum_norm"].idxmax(), "temps_h"]) if not valid.empty else np.nan
        else:
            row["n_points_Lum_norm"] = 0
            row["temps_debut_norm_h"] = np.nan
            row["temps_fin_norm_h"] = np.nan
            row["AUC_Lum_norm"] = np.nan
            row["Lum_norm_max"] = np.nan
            row["Lum_norm_final"] = np.nan
            row["temps_pic_h"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    sort_cols = [c for c in ["medium", "strain", "experience_id", "series_id"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if not out.empty else out


def summarize_points(params: pd.DataFrame, metric: str, group_col: str) -> tuple[pd.DataFrame, str]:
    if params.empty or metric not in params.columns:
        return pd.DataFrame(), "missing"

    work = params.loc[params[group_col].notna() & params[metric].notna()].copy()
    if work.empty:
        return pd.DataFrame(), "empty"

    n_exp = int(work["experience_id"].nunique()) if "experience_id" in work.columns else 1

    if n_exp <= 1:
        points = work[[group_col, metric, "series_id"]].copy()
        points = points.rename(columns={group_col: "group_name"})
        points["unit_id"] = points["series_id"].astype(str)
        points["niveau"] = "puits"
        return points.reset_index(drop=True), "puits"

    grouped = (
        work.groupby(["experience_id", group_col], dropna=False, observed=True)[metric]
        .mean()
        .reset_index(name=metric)
        .rename(columns={group_col: "group_name"})
    )
    grouped["unit_id"] = grouped["experience_id"].astype(str)
    grouped["niveau"] = "experience"
    return grouped.reset_index(drop=True), "experience"


# -----------------------------------------------------------------------------
# Tracé
# -----------------------------------------------------------------------------

def draw_timecourse(
    ax: plt.Axes,
    summary: pd.DataFrame,
    lines: pd.DataFrame,
    group_order: Sequence[str],
    colors: dict[str, str],
    ylabel: str,
) -> None:
    if summary.empty:
        return

    line_group_col = "series_id" if "series_id" in lines.columns else "experience_id"

    for group_name in group_order:
        color = colors[group_name]

        sub_lines = lines.loc[lines["group_name"] == group_name]
        if not sub_lines.empty:
            for _, line_df in sub_lines.groupby(line_group_col, dropna=False):
                line_df = line_df.sort_values("temps_h")
                ax.plot(
                    line_df["temps_h"],
                    line_df["line_value"],
                    color=color,
                    alpha=0.18,
                    linewidth=0.8,
                    zorder=1,
                )

        sub_sum = summary.loc[summary["group_name"] == group_name].sort_values("temps_h")
        if sub_sum.empty:
            continue

        x = sub_sum["temps_h"].to_numpy(dtype=float)
        y = sub_sum["mean_value"].to_numpy(dtype=float)
        sd = sub_sum["sd_value"].fillna(0).to_numpy(dtype=float)

        ax.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0, zorder=2)
        ax.plot(x, y, color=color, linewidth=1.8, zorder=3)

    ax.set_xlabel("Temps (h)")
    ax.set_ylabel(ylabel)
    ax.grid(False)

    x_all = summary["temps_h"].dropna().to_numpy(dtype=float)
    if len(x_all) > 0:
        ax.set_xlim(np.nanmin(x_all), np.nanmax(x_all))
    ax.margins(x=0.0)



def jitter_positions(n: int, center: float, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - width, center + width, n)


def make_timecourse_subtitle(mode: str) -> str:
    return (
        "Lignes fines = puits/séries | ligne épaisse = moyenne des puits"
        if mode == "technique"
        else "Lignes fines = puits/séries | ligne épaisse = moyenne inter-expériences"
    )


def make_points_subtitle(points: pd.DataFrame, *, log_scale: bool = False, filtered_out: int = 0) -> str:
    subtitle = (
        "Points = puits techniques"
        if points["niveau"].astype(str).eq("puits").all()
        else "Points = moyennes d'expériences biologiques"
    )
    if log_scale:
        subtitle += " | axe Y en log"
        if filtered_out > 0:
            subtitle += f" | {filtered_out} valeur(s) <= 0 exclue(s)"
    subtitle += " | valeur affichée = moyenne"
    return subtitle


def make_bar_subtitle(points: pd.DataFrame) -> str:
    return (
        "Barres = moyenne ± SD | points = puits techniques"
        if points["niveau"].astype(str).eq("puits").all()
        else "Barres = moyenne ± SD | points = moyennes d'expériences biologiques"
    )


def add_figure_header(
    fig: plt.Figure,
    *,
    title: str,
    subtitle: str | None = None,
    legend_handles: Sequence[Line2D | Patch] | None = None,
    legend_title: str | None = None,
    legend_ncol: int = 2,
) -> tuple[float, float, float, float]:
    fig.text(0.08, 0.982, title, ha="left", va="top", fontsize=10, fontweight="bold")

    y_sub = 0.945
    if subtitle:
        fig.text(0.08, y_sub, subtitle, ha="left", va="top", fontsize=7, color="#444444")

    if legend_handles:
        fig.legend(
            handles=list(legend_handles),
            loc="upper left",
            bbox_to_anchor=(0.08, 0.895, 0.89, 0.01),
            mode="expand",
            ncol=max(1, legend_ncol),
            frameon=False,
            title=legend_title,
            handlelength=2.6,
            columnspacing=1.1,
            labelspacing=0.7,
            borderaxespad=0.0,
        )
        return 0.08, 0.12, 0.98, 0.64

    return 0.08, 0.12, 0.98, 0.82


def draw_dot_metric(
    ax: plt.Axes,
    points: pd.DataFrame,
    metric: str,
    ylabel: str,
    group_order: Sequence[str],
    colors: dict[str, str],
    log_scale: bool = False,
) -> int:
    if points.empty:
        return 0

    work = points.copy()
    filtered_out = 0

    if log_scale:
        before = len(work.loc[work[metric].notna()])
        work = work.loc[work[metric].notna() & (work[metric] > 0)].copy()
        filtered_out = before - len(work)
        if not work.empty:
            ax.set_yscale("log")

    positions = {g: i for i, g in enumerate(group_order)}

    ymin_log = None
    ymax_log = None
    if log_scale and not work.empty:
        vals = work[metric].to_numpy(dtype=float)
        ymin_log = float(np.nanmin(vals)) / 1.15
        ymax_log = float(np.nanmax(vals)) * 2.40

    for idx, group_name in enumerate(group_order):
        x_center = positions[group_name]
        sub = work.loc[(work["group_name"] == group_name) & (work[metric].notna())]
        if sub.empty:
            continue

        y = sub[metric].to_numpy(dtype=float)
        xj = jitter_positions(len(sub), x_center)

        ax.scatter(
            xj,
            y,
            s=28,
            color=colors[group_name],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )

        mean_val = float(np.nanmean(y)) if len(y) else np.nan
        sd_val = float(np.nanstd(y, ddof=1)) if len(y) >= 2 else np.nan

        if np.isfinite(mean_val):
            ax.plot(
                [x_center - 0.20, x_center + 0.20],
                [mean_val, mean_val],
                color="black",
                linewidth=1.1,
                zorder=4,
            )
            if np.isfinite(sd_val):
                low = mean_val - sd_val
                high = mean_val + sd_val
                if (not log_scale) or (low > 0 and high > 0):
                    ax.vlines(x_center, low, high, color="black", linewidth=0.9, zorder=4)
                    ax.hlines([low, high], x_center - 0.06, x_center + 0.06, color="black", linewidth=0.9, zorder=4)

    ax.set_xticks([positions[g] for g in group_order])
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    ax.set_ylabel(ylabel if not log_scale else f"{ylabel} (log)")
    ax.grid(False)

    if log_scale and ymin_log is not None and ymax_log is not None:
        ax.set_ylim(ymin_log, ymax_log)

    return filtered_out


def draw_bar_metric(
    ax: plt.Axes,
    points: pd.DataFrame,
    metric: str,
    ylabel: str,
    group_order: Sequence[str],
    colors: dict[str, str],
) -> None:
    if points.empty:
        return

    positions = {g: i for i, g in enumerate(group_order)}

    means = []
    sds = []
    for group_name in group_order:
        sub = points.loc[(points["group_name"] == group_name) & (points[metric].notna()), metric]
        means.append(float(np.nanmean(sub)) if len(sub) else np.nan)
        sds.append(float(np.nanstd(sub, ddof=1)) if len(sub) >= 2 else 0.0)

    ax.bar(
        [positions[g] for g in group_order],
        means,
        yerr=sds,
        color=[colors[g] for g in group_order],
        edgecolor="none",
        alpha=0.88,
        width=0.68,
        linewidth=0,
        capsize=3,
        zorder=2,
    )

    for group_name in group_order:
        sub = points.loc[(points["group_name"] == group_name) & (points[metric].notna())]
        if sub.empty:
            continue
        xj = jitter_positions(len(sub), positions[group_name], width=0.16)
        y = sub[metric].to_numpy(dtype=float)
        ax.scatter(
            xj,
            y,
            s=24,
            color="white",
            edgecolor="black",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )

    ax.set_xticks([positions[g] for g in group_order])
    ax.set_xticklabels(group_order, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(False, axis="y")


def plot_timecourse_figure(
    summary: pd.DataFrame,
    lines: pd.DataFrame,
    mode: str,
    group_order: Sequence[str],
    colors: dict[str, str],
    title: str,
    ylabel: str,
    outpath: Path,
) -> None:
    if summary.empty:
        return

    fig_w_mm = max(180, 80 + 28 * len(group_order))
    fig, ax = plt.subplots(figsize=(mm_to_inch(fig_w_mm), mm_to_inch(92)))
    draw_timecourse(ax, summary, lines, group_order, colors, ylabel)

    handles = make_line_handles(group_order, colors)
    left, bottom, right, top = add_figure_header(
        fig,
        title=title,
        subtitle=make_timecourse_subtitle(mode),
        legend_handles=handles,
        legend_title="Conditions" if handles else None,
        legend_ncol=2 if len(group_order) > 3 else len(group_order),
    )
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

    export_png(fig, outpath)


def plot_dot_figure(
    points: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    group_order: Sequence[str],
    colors: dict[str, str],
    outpath: Path,
) -> None:
    if points.empty:
        return

    fig_w_mm = max(96, 30 + 18 * len(group_order))
    fig_h_mm = 125 if metric == "Lum_norm_max" else 90
    fig, ax = plt.subplots(figsize=(mm_to_inch(fig_w_mm), mm_to_inch(fig_h_mm)))

    filtered_out = draw_dot_metric(
        ax=ax,
        points=points,
        metric=metric,
        ylabel=ylabel,
        group_order=group_order,
        colors=colors,
        log_scale=(metric == "Lum_norm_max"),
    )

    left, bottom, right, top = add_figure_header(
        fig,
        title=title,
        subtitle=make_points_subtitle(points, log_scale=(metric == "Lum_norm_max"), filtered_out=filtered_out),
    )
    fig.subplots_adjust(top=top, bottom=0.30, left=max(left, 0.10), right=0.99)
    export_png(fig, outpath)


def plot_bar_figure(
    points: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    group_order: Sequence[str],
    colors: dict[str, str],
    outpath: Path,
) -> None:
    if points.empty:
        return

    fig_w_mm = max(96, 30 + 18 * len(group_order))
    fig, ax = plt.subplots(figsize=(mm_to_inch(fig_w_mm), mm_to_inch(90)))

    draw_bar_metric(
        ax=ax,
        points=points,
        metric=metric,
        ylabel=ylabel,
        group_order=group_order,
        colors=colors,
    )

    left, bottom, right, top = add_figure_header(
        fig,
        title=title,
        subtitle=make_bar_subtitle(points),
    )
    fig.subplots_adjust(top=top, bottom=0.30, left=max(left, 0.10), right=0.99)
    export_png(fig, outpath)



def build_condition_label(medium: str, strain: str) -> str:
    return f"{strain} ({medium})"



def format_scientific_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mant = value / (10 ** exp)
    return rf"${mant:.2f}\\times 10^{{{exp}}}$"



def build_global_condition_layout(df: pd.DataFrame) -> dict[str, object]:
    pairs = (
        df[["strain", "medium"]]
        .dropna()
        .drop_duplicates()
        .copy()
    )
    if pairs.empty:
        return {
            "condition_order": [],
            "positions": {},
            "colors": {},
            "markers": {},
            "strain_map": {},
            "medium_map": {},
        }

    strain_order = build_strain_order(pairs["strain"].astype(str).tolist())
    medium_order = build_medium_order(pairs["medium"].astype(str).tolist())
    strain_rank = {x: i for i, x in enumerate(strain_order)}
    medium_rank = {x: i for i, x in enumerate(medium_order)}
    strain_colors = build_strain_color_map(strain_order)

    pairs["strain"] = pairs["strain"].astype(str)
    pairs["medium"] = pairs["medium"].astype(str)
    pairs = pairs.sort_values(
        by=["strain", "medium"],
        key=lambda s: s.map(strain_rank) if s.name == "strain" else s.map(medium_rank),
    ).reset_index(drop=True)

    condition_order: list[str] = []
    positions: dict[str, float] = {}
    colors: dict[str, str] = {}
    markers: dict[str, str] = {}
    strain_map: dict[str, str] = {}
    medium_map: dict[str, str] = {}

    x = 0.0
    prev_strain = None
    for i, row in pairs.iterrows():
        strain = str(row["strain"])
        medium = str(row["medium"])
        label = build_condition_label(medium, strain)
        if i == 0:
            x = 0.0
        else:
            x += 0.82 if strain == prev_strain else 1.45
        condition_order.append(label)
        positions[label] = float(x)
        colors[label] = strain_colors[strain]
        markers[label] = MEDIUM_MARKERS.get(medium, "o")
        strain_map[label] = strain
        medium_map[label] = medium
        prev_strain = strain

    return {
        "condition_order": condition_order,
        "positions": positions,
        "colors": colors,
        "markers": markers,
        "strain_map": strain_map,
        "medium_map": medium_map,
    }



def summarize_points_by_condition(params: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    if params.empty or metric not in params.columns:
        return pd.DataFrame(), "missing"

    work = params.loc[
        params["strain"].notna() & params["medium"].notna() & params[metric].notna()
    ].copy()
    if work.empty:
        return pd.DataFrame(), "empty"

    work["condition_label"] = [build_condition_label(m, s) for m, s in zip(work["medium"].astype(str), work["strain"].astype(str))]

    n_exp = int(work["experience_id"].nunique()) if "experience_id" in work.columns else 1
    if n_exp <= 1:
        points = work[["condition_label", "strain", "medium", metric, "series_id"]].copy()
        points["unit_id"] = points["series_id"].astype(str)
        points["niveau"] = "puits"
        return points.reset_index(drop=True), "puits"

    grouped = (
        work.groupby(["experience_id", "condition_label", "strain", "medium"], dropna=False, observed=True)[metric]
        .mean()
        .reset_index(name=metric)
    )
    grouped["unit_id"] = grouped["experience_id"].astype(str)
    grouped["niveau"] = "experience"
    return grouped.reset_index(drop=True), "experience"



def draw_global_condition_dot_metric(
    ax: plt.Axes,
    points: pd.DataFrame,
    metric: str,
    ylabel: str,
    layout: dict[str, object],
    log_scale: bool = False,
) -> int:
    if points.empty:
        return 0

    order = list(layout["condition_order"])
    positions = dict(layout["positions"])
    colors = dict(layout["colors"])
    markers = dict(layout["markers"])

    work = points.copy()
    filtered_out = 0
    ymin_log = None
    ymax_log = None

    if log_scale:
        before = len(work.loc[work[metric].notna()])
        work = work.loc[work[metric].notna() & (work[metric] > 0)].copy()
        filtered_out = before - len(work)
        if not work.empty:
            ax.set_yscale("log")
            vals = work[metric].to_numpy(dtype=float)
            ymin_log = float(np.nanmin(vals)) / 1.15
            ymax_log = float(np.nanmax(vals)) * 2.40

    for idx, cond in enumerate(order):
        x_center = positions[cond]
        sub = work.loc[(work["condition_label"] == cond) & (work[metric].notna())]
        if sub.empty:
            continue

        y = sub[metric].to_numpy(dtype=float)
        xj = jitter_positions(len(sub), x_center, width=0.18)
        marker = markers.get(cond, "o")
        color = colors.get(cond, "#333333")

        ax.scatter(
            xj,
            y,
            s=28,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )

        mean_val = float(np.nanmean(y)) if len(y) else np.nan
        sd_val = float(np.nanstd(y, ddof=1)) if len(y) >= 2 else np.nan

        if np.isfinite(mean_val):
            ax.plot(
                [x_center - 0.20, x_center + 0.20],
                [mean_val, mean_val],
                color="black",
                linewidth=1.1,
                zorder=4,
            )

            if np.isfinite(sd_val):
                low = mean_val - sd_val
                high = mean_val + sd_val
                if (not log_scale) or (low > 0 and high > 0):
                    ax.vlines(x_center, low, high, color="black", linewidth=0.9, zorder=4)
                    ax.hlines([low, high], x_center - 0.06, x_center + 0.06, color="black", linewidth=0.9, zorder=4)

            if log_scale:
                y_text = float(np.nanmax(y)) * [1.10, 1.28, 1.48][idx % 3]
            else:
                y_span = float(np.nanmax(y) - np.nanmin(y)) if len(y) > 1 else max(abs(mean_val) * 0.1, 0.05)
                if y_span == 0:
                    y_span = max(abs(mean_val) * 0.1, 0.05)
                y_text = float(np.nanmax(y)) + [0.08, 0.16, 0.24][idx % 3] * y_span

            x_text = x_center
            if idx == 0:
                x_text += 0.10
            elif idx == len(order) - 1:
                x_text -= 0.10

            ax.text(
                x_text,
                y_text,
                format_scientific_label(mean_val),
                ha="center",
                va="bottom",
                fontsize=4.0,
                color="black",
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15),
            )

    xticks = [positions[c] for c in order]
    ax.set_xticks(xticks)
    ax.set_xticklabels(order, rotation=35, ha="right")
    ax.set_ylabel(ylabel if not log_scale else f"{ylabel} (log)")
    ax.grid(False)

    if positions:
        xvals = list(positions.values())
        ax.set_xlim(min(xvals) - 0.70, max(xvals) + 0.70)
    if log_scale and ymin_log is not None and ymax_log is not None:
        ax.set_ylim(ymin_log, ymax_log)

    return filtered_out


def plot_global_recap_figure(
    points: pd.DataFrame,
    metric: str,
    title: str,
    ylabel: str,
    layout: dict[str, object],
    outpath: Path,
) -> None:
    if points.empty or not layout.get("condition_order"):
        return

    n_conditions = len(layout["condition_order"])
    fig_w_mm = max(240, 24 + 22 * n_conditions)
    fig_h_mm = 125 if metric == "Lum_norm_max" else 95
    fig, ax = plt.subplots(figsize=(mm_to_inch(fig_w_mm), mm_to_inch(fig_h_mm)))

    filtered_out = draw_global_condition_dot_metric(
        ax=ax,
        points=points,
        metric=metric,
        ylabel=ylabel,
        layout=layout,
        log_scale=(metric == "Lum_norm_max"),
    )

    ax.set_title(title, loc="left", fontweight="bold")
    ax.text(
        0.0,
        1.02,
        make_points_subtitle(points, log_scale=(metric == "Lum_norm_max"), filtered_out=filtered_out),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.8,
        color="#444444",
    )

    fig.subplots_adjust(top=0.90, bottom=0.30, left=0.10, right=0.995)
    export_png(fig, outpath)


# -----------------------------------------------------------------------------
# Génération par vue
# -----------------------------------------------------------------------------

def sanitize_name(text: str) -> str:
    return clean_text(text).replace(".", "_")


def generate_view_figures(
    df_view: pd.DataFrame,
    params_view: pd.DataFrame,
    *,
    group_col: str,
    group_order: Sequence[str],
    colors: dict[str, str],
    entity_name: str,
    entity_value: str,
    output_dir: Path,
) -> list[str]:
    created_files: list[str] = []

    if df_view.empty or not group_order:
        return created_files

    entity_slug = sanitize_name(entity_value)

    # Courbes
    for metric, (title_short, ylabel) in CURVE_SPECS.items():
        summary, lines, mode = aggregate_curve_data(df_view, metric, group_col=group_col)
        if summary.empty:
            continue

        summary.to_csv(output_dir / f"data_{entity_name}_{entity_slug}_{metric}_summary.csv", index=False, encoding="utf-8-sig")
        lines.to_csv(output_dir / f"data_{entity_name}_{entity_slug}_{metric}_lines.csv", index=False, encoding="utf-8-sig")

        outpath = output_dir / f"{entity_name}_{entity_slug}_{metric}"
        plot_timecourse_figure(
            summary=summary,
            lines=lines,
            mode=mode,
            group_order=group_order,
            colors=colors,
            title=f"{title_short} - {entity_value}",
            ylabel=ylabel,
            outpath=outpath,
        )
        created_files.append(str(outpath.with_suffix(".png").name))

    # Points (AUC et pic)
    for metric, (title_short, ylabel) in POINT_SPECS.items():
        points, level = summarize_points(params_view, metric, group_col=group_col)
        if points.empty:
            continue

        points.to_csv(output_dir / f"data_{entity_name}_{entity_slug}_{metric}_points.csv", index=False, encoding="utf-8-sig")

        outpath = output_dir / f"{entity_name}_{entity_slug}_{metric}"
        plot_dot_figure(
            points=points,
            metric=metric,
            title=f"{title_short} - {entity_value}",
            ylabel=ylabel,
            group_order=group_order,
            colors=colors,
            outpath=outpath,
        )
        created_files.append(str(outpath.with_suffix(".png").name))

    # Barres (temps de doublement)
    for metric, (title_short, ylabel) in BAR_SPECS.items():
        points, level = summarize_points(params_view, metric, group_col=group_col)
        if points.empty:
            continue

        points.to_csv(output_dir / f"data_{entity_name}_{entity_slug}_{metric}_points.csv", index=False, encoding="utf-8-sig")

        outpath = output_dir / f"{entity_name}_{entity_slug}_{metric}"
        plot_bar_figure(
            points=points,
            metric=metric,
            title=f"{title_short} - {entity_value}",
            ylabel=ylabel,
            group_order=group_order,
            colors=colors,
            outpath=outpath,
        )
        created_files.append(str(outpath.with_suffix(".png").name))

    return created_files


# -----------------------------------------------------------------------------
# Rapport
# -----------------------------------------------------------------------------

def build_report(
    df_fig: pd.DataFrame,
    params: pd.DataFrame,
    output_dir: Path,
    created_files: Sequence[str],
) -> str:
    media_present = build_medium_order(df_fig["medium"].dropna().astype(str).unique().tolist())
    strains_present = build_strain_order(df_fig["strain"].dropna().astype(str).unique().tolist())

    lines = []
    lines.append("ANALYSE DES FIGURES")
    lines.append("")
    lines.append(f"Dossier de sortie : {output_dir}")
    lines.append(f"Nombre de lignes analysées : {len(df_fig)}")
    lines.append(f"Nombre d'expériences : {int(df_fig['experience_id'].nunique()) if 'experience_id' in df_fig.columns else 1}")
    lines.append(f"Nombre de séries/puits : {int(df_fig['series_id'].nunique()) if 'series_id' in df_fig.columns else 0}")
    lines.append("")
    lines.append("Milieux détectés :")
    for x in media_present:
        lines.append(f"- {x}")
    lines.append("")
    lines.append("Souches détectées :")
    for x in strains_present:
        lines.append(f"- {x}")
    lines.append("")
    lines.append(f"Nombre de fichiers PNG générés : {len(created_files)}")
    lines.append("")
    lines.append("Temps de doublement :")
    n_dt = int(params["doubling_time_h"].notna().sum()) if "doubling_time_h" in params.columns else 0
    lines.append(f"- Séries avec temps de doublement estimé : {n_dt}")
    lines.append(f"- Méthode : pente maximale de ln(DO_corr) sur fenêtres glissantes de {DOUBLING_MIN_POINTS} points")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    params = extract_parameters_per_series(df)
    params.to_csv(output_dir / "data_parametres_par_serie.csv", index=False, encoding="utf-8-sig")

    media_present = build_medium_order(df["medium"].dropna().astype(str).unique().tolist())
    strains_present = build_strain_order(df["strain"].dropna().astype(str).unique().tolist())

    strain_color_map = build_strain_color_map(strains_present)
    medium_color_map = build_medium_color_map(media_present)

    created_files: list[str] = []

    # Figures par milieu : courbes colorées par souche
    out_by_medium = output_dir / "figures_par_milieu"
    out_by_medium.mkdir(parents=True, exist_ok=True)

    for medium in media_present:
        df_view = df.loc[df["medium"] == medium].copy()
        params_view = params.loc[params["medium"] == medium].copy()
        group_order = build_strain_order(df_view["strain"].dropna().astype(str).unique().tolist())
        colors = {g: strain_color_map[g] for g in group_order}
        created_files.extend(
            generate_view_figures(
                df_view=df_view,
                params_view=params_view,
                group_col="strain",
                group_order=group_order,
                colors=colors,
                entity_name="milieu",
                entity_value=medium,
                output_dir=out_by_medium,
            )
        )

    # Figures par souche : courbes colorées par milieu
    out_by_strain = output_dir / "figures_par_souche"
    out_by_strain.mkdir(parents=True, exist_ok=True)

    for strain in strains_present:
        df_view = df.loc[df["strain"] == strain].copy()
        params_view = params.loc[params["strain"] == strain].copy()
        group_order = build_medium_order(df_view["medium"].dropna().astype(str).unique().tolist())
        colors = {g: medium_color_map[g] for g in group_order}
        created_files.extend(
            generate_view_figures(
                df_view=df_view,
                params_view=params_view,
                group_col="medium",
                group_order=group_order,
                colors=colors,
                entity_name="souche",
                entity_value=strain,
                output_dir=out_by_strain,
            )
        )

    # Deux grands récapitulatifs globaux comme dans le script initial
    global_layout = build_global_condition_layout(df)
    for metric, (title_short, ylabel) in POINT_SPECS.items():
        points_global, level = summarize_points_by_condition(params, metric)
        if points_global.empty:
            continue
        points_global.to_csv(output_dir / f"data_global_{metric}_points.csv", index=False, encoding="utf-8-sig")
        outpath = output_dir / f"figure_{metric}"
        plot_global_recap_figure(
            points=points_global,
            metric=metric,
            title=f"Toutes conditions - {ylabel}",
            ylabel=ylabel,
            layout=global_layout,
            outpath=outpath,
        )
        created_files.append(str(outpath.with_suffix('.png').name))

    report = build_report(df, params, output_dir, created_files)
    (output_dir / "figures_resume.txt").write_text(report, encoding="utf-8")

    diag_lines = []
    diag_lines.append("Couleurs fixes par souche :")
    for k, v in STRAIN_COLOR_FIXED.items():
        diag_lines.append(f"- {k}: {v}")
    diag_lines.append("")
    diag_lines.append("Couleurs par milieu (figures par souche) :")
    for k in media_present:
        diag_lines.append(f"- {k}: {medium_color_map[k]}")
    (output_dir / "diagnostic_couleurs.txt").write_text("\n".join(diag_lines) + "\n", encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générer des figures PNG par milieu et par souche à partir de la colonne 'souche'."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Un ou plusieurs fichiers normalisés issus de 04_normalisation_DO.py",
    )
    parser.add_argument(
        "--experience-ids",
        nargs="*",
        default=None,
        help="IDs d'expériences à associer aux fichiers d'entrée",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie (optionnel)",
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()

    input_paths = [Path(p).resolve() for p in args.input_files]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable : {p}")

    dfs = [prepare_dataframe(read_table(p)) for p in input_paths]
    for df in dfs:
        ensure_columns(df, COLONNES_MIN)

    df = add_experience_ids(dfs, input_paths, args.experience_ids)
    df = prepare_dataframe(df)
    df["series_id"] = infer_series_id(df)

    output_dir = infer_output_dir(input_paths, Path(args.output_dir) if args.output_dir else None)

    df = detect_conditions_from_souche(df)
    df_fig = filter_dataframe_for_figures(df)

    if df_fig.empty:
        raise ValueError(
            "Aucune condition exploitable détectée dans la colonne 'souche'. "
            "Formats attendus : 'LB [14.1Ac]', 'SCFM1 [14.1Ac]', 'Old SCFM2 [14.1Ac]', 'New SCFM2 [14.1Ac]'."
        )

    df_fig = df_fig.sort_values(["medium", "strain", "temps_h", "series_id"]).reset_index(drop=True)

    run_analysis(df_fig, output_dir)

    media_present = build_medium_order(df_fig["medium"].dropna().astype(str).unique().tolist())
    strains_present = build_strain_order(df_fig["strain"].dropna().astype(str).unique().tolist())

    print("=" * 80)
    print("FIGURES TERMINEES - PAR MILIEU ET PAR SOUCHE")
    print("=" * 80)
    print(f"Dossier de sortie : {output_dir}")
    print(f"Expériences détectées : {int(df_fig['experience_id'].nunique())}")
    print(f"Milieux détectés : {len(media_present)}")
    for x in media_present:
        print(f"  - {x}")
    print(f"Souches détectées : {len(strains_present)}")
    for x in strains_present:
        print(f"  - {x}")
    print("Exports : PNG uniquement")
    print("=" * 80)


if __name__ == "__main__":
    main()
