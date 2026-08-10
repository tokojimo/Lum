#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
06_auc_lum_norm_global.py

Objectif
--------
Générer UNE figure globale de l'AUC de luminescence normalisée (AUC_Lum_norm),
avec la même logique de tracé que le script global du pic de luminescence
normalisée, mais avec axe Y en log.

Sorties
-------
- figure_AUC_Lum_norm.png
- figure_data_parametres_par_serie.csv
- figure_data_AUC_Lum_norm_points.csv
- diagnostic_conditions_detectees.txt
- resume_auc_lum_norm.txt
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


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

COLONNES_MIN = ["temps_h", "souche", "type", "DO_corr"]

PALETTE = [
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

MEDIUM_MARKERS = {
    "LB": "s",
    "SCFM1": "D",
    "Old SCFM2": "o",
    "New SCFM2": "^",
}


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
# Conditions issues de la colonne "souche"
# -----------------------------------------------------------------------------

def normalize_condition_label(value: object) -> str | None:
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    m = re.fullmatch(
        r"(LB|SCFM1|Old\s*SCFM2|New\s*SCFM2)\s*\[\s*(.*?)\s*\]",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        medium_raw = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        strain = re.sub(r"\s+", " ", m.group(2)).strip()
        medium = CANONICAL_MEDIA.get(medium_raw, m.group(1).strip())
        return f"{medium} [{strain}]"

    m = re.fullmatch(
        r"(LB|SCFM1|Old\s*SCFM2|New\s*SCFM2)\s+(.*)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        medium_raw = re.sub(r"\s+", " ", m.group(1)).strip().lower()
        strain = re.sub(r"\s+", " ", m.group(2)).strip()
        medium = CANONICAL_MEDIA.get(medium_raw, m.group(1).strip())
        return f"{medium} [{strain}]"

    return text


def extract_medium_and_strain(value: object) -> tuple[str | None, str | None]:
    label = normalize_condition_label(value)
    if label is None:
        return None, None

    m = re.fullmatch(r"(LB|SCFM1|Old SCFM2|New SCFM2)\s*\[\s*(.*?)\s*\]", label)
    if m:
        return m.group(1), m.group(2)

    return None, label


def detect_conditions_from_souche(df: pd.DataFrame) -> pd.Series:
    if "souche" not in df.columns:
        raise ValueError("La colonne 'souche' est absente du tableau.")
    return df["souche"].map(normalize_condition_label)


def get_strain_name(category: object) -> str:
    medium, strain = extract_medium_and_strain(category)
    return strain if strain is not None else str(category)


def build_strain_order(category_order: Sequence[str]) -> list[str]:
    strains = []
    seen = set()
    for category in category_order:
        strain = get_strain_name(category)
        if strain not in seen:
            seen.add(strain)
            strains.append(strain)
    return strains


def build_strain_color_map(category_order: Sequence[str]) -> dict[str, str]:
    strain_order = build_strain_order(category_order)
    return {strain: PALETTE[i % len(PALETTE)] for i, strain in enumerate(strain_order)}


def build_xtick_labels(category_order: Sequence[str]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for category in category_order:
        medium, strain = extract_medium_and_strain(category)
        if medium is not None and strain is not None:
            labels[str(category)] = f"{strain} ({medium})"
        else:
            labels[str(category)] = str(category)
    return labels


def build_category_order(categories: Sequence[str]) -> list[str]:
    parsed = []

    for category in categories:
        medium, strain = extract_medium_and_strain(category)
        strain_key = (strain or str(category)).strip().lower()
        medium_rank = MEDIUM_ORDER.get(medium, 99)

        parsed.append({
            "category": str(category),
            "strain_key": strain_key,
            "medium_rank": medium_rank,
        })

    parsed = sorted(parsed, key=lambda x: (x["strain_key"], x["medium_rank"], x["category"].lower()))
    return [x["category"] for x in parsed]


def build_grouped_positions(category_order: Sequence[str]) -> dict[str, float]:
    positions: dict[str, float] = {}
    if not category_order:
        return positions

    x = 0.0
    prev_strain = None

    gap_same_strain = 0.78
    gap_new_strain = 1.35

    for i, category in enumerate(category_order):
        _, strain = extract_medium_and_strain(category)
        cur_strain = strain if strain is not None else str(category)

        if i == 0:
            x = 0.0
        else:
            x += gap_same_strain if cur_strain == prev_strain else gap_new_strain

        positions[str(category)] = float(x)
        prev_strain = cur_strain

    return positions


def prepare_category_layout(df: pd.DataFrame, category_col: str = "figure_group") -> dict[str, object]:
    category_order = build_category_order(
        df[category_col].dropna().astype(str).drop_duplicates().tolist()
    )
    return {
        "category_order": category_order,
        "strain_colors": build_strain_color_map(category_order),
        "xtick_labels": build_xtick_labels(category_order),
        "positions": build_grouped_positions(category_order),
    }


def is_blank_condition(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return bool(re.match(r"^(blanc|blank)\s*\d*$", text))


def filter_dataframe_for_figures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.loc[out["type"].astype(str).str.lower() == "souche"].copy()
    out = out.loc[out["figure_group"].notna()].copy()
    out = out.loc[~out["figure_group"].astype(str).map(is_blank_condition)].copy()
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Style et export
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


def export_png(fig: plt.Figure, basepath: Path, png_dpi: int = 600) -> None:
    fig.savefig(basepath.with_suffix(".png"), dpi=png_dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Paramètres par série
# -----------------------------------------------------------------------------

def trapezoid_auc(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(valid) < 2:
        return np.nan
    return float(np.trapezoid(valid["y"].to_numpy(dtype=float), valid["x"].to_numpy(dtype=float)))


def extract_parameters_per_series(df: pd.DataFrame, category_col: str) -> pd.DataFrame:
    work = df.loc[df["type"].astype(str).str.lower() == "souche"].copy()
    work = work.loc[work[category_col].notna()].copy()

    if work.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    group_cols = [c for c in ["experience_id", category_col, "series_id"] if c in work.columns]

    for keys, sub in work.groupby(group_cols, dropna=False, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        meta = dict(zip(group_cols, keys))
        sub = sub.sort_values("temps_h").copy()

        row: dict[str, object] = {**meta, "category": meta.get(category_col)}

        if "sample_header" in sub.columns and sub["sample_header"].notna().any():
            row["sample_header"] = str(sub["sample_header"].dropna().iloc[0])
        if "puits" in sub.columns and sub["puits"].notna().any():
            row["puits"] = str(sub["puits"].dropna().iloc[0])
        if "souche" in sub.columns and sub["souche"].notna().any():
            row["souche"] = str(sub["souche"].dropna().iloc[0])

        row["n_points_total"] = int(len(sub))
        row["AUC_DO"] = trapezoid_auc(sub["temps_h"], sub["DO_corr"])
        row["DO_max"] = float(sub["DO_corr"].max()) if sub["DO_corr"].notna().any() else np.nan
        row["temps_DO_max_h"] = float(sub.loc[sub["DO_corr"].idxmax(), "temps_h"]) if sub["DO_corr"].notna().any() else np.nan

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
    sort_cols = [c for c in ["experience_id", "category", "series_id"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True) if not out.empty else out


def summary_points_for_metric(params: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    if params.empty or metric not in params.columns:
        return pd.DataFrame(), "missing"

    n_exp = int(params["experience_id"].nunique()) if "experience_id" in params.columns else 1
    work = params.loc[params[metric].notna()].copy()

    if work.empty:
        return pd.DataFrame(), "empty"

    if n_exp <= 1:
        points = work[["category", metric, "series_id"]].copy()
        points["unit_id"] = points["series_id"].astype(str)
        points["niveau"] = "puits"
        return points.reset_index(drop=True), "puits"

    grouped = (
        work.groupby(["experience_id", "category"], dropna=False, observed=True)[metric]
        .mean()
        .reset_index(name=metric)
    )
    grouped["unit_id"] = grouped["experience_id"].astype(str)
    grouped["niveau"] = "experience"
    return grouped.reset_index(drop=True), "experience"


# -----------------------------------------------------------------------------
# Tracé
# -----------------------------------------------------------------------------

def jitter_positions(n: int, center: float, width: float = 0.18) -> np.ndarray:
    if n <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - width, center + width, n)


def draw_dot_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    metric: str,
    ylabel: str,
    category_order: Sequence[str],
    strain_colors: dict[str, str],
    positions: dict[str, float],
    xtick_labels_map: dict[str, str],
    point_level: str,
    title: str,
    stat_label: str = "mean",
) -> str:
    if points.empty:
        return ""

    use_log_scale = (metric in {"Lum_norm_max", "AUC_Lum_norm"})
    filtered_out_nonpositive = 0
    work = points.copy()

    ymin_log = None
    ymax_log = None

    if use_log_scale:
        n_before = len(work.loc[work[metric].notna()])
        work = work.loc[work[metric].notna() & (work[metric] > 0)].copy()
        filtered_out_nonpositive = n_before - len(work)

        if not work.empty:
            ax.set_yscale("log")
            vals = work[metric].to_numpy(dtype=float)
            ymin_data = float(np.nanmin(vals))
            ymax_data = float(np.nanmax(vals))
            ymin_log = ymin_data / 1.15
            ymax_log = ymax_data * 2.40

    for idx, category in enumerate(category_order):
        x_center = positions[category]
        sub = work.loc[(work["category"] == category) & (work[metric].notna())]
        if sub.empty:
            continue

        medium, strain = extract_medium_and_strain(category)
        strain_name = strain if strain is not None else str(category)

        point_color = strain_colors.get(strain_name, "#333333")
        point_marker = MEDIUM_MARKERS.get(medium, "o")

        xj = jitter_positions(len(sub), x_center)
        y = sub[metric].to_numpy(dtype=float)

        ax.scatter(
            xj,
            y,
            s=28,
            color=point_color,
            marker=point_marker,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )

        mean_val = float(np.nanmean(y)) if len(y) else np.nan
        median_val = float(np.nanmedian(y)) if len(y) else np.nan
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

                if (not use_log_scale) or (low > 0 and high > 0):
                    ax.vlines(x_center, low, high, color="black", linewidth=0.9, zorder=4)
                    ax.hlines([low, high], x_center - 0.06, x_center + 0.06, color="black", linewidth=0.9, zorder=4)

        stat_val = median_val if stat_label == "median" else mean_val

        if np.isfinite(stat_val):
            levels_log = [1.10, 1.28, 1.48]
            levels_lin = [0.08, 0.16, 0.24]

            if use_log_scale:
                y_text = float(np.nanmax(y)) * levels_log[idx % len(levels_log)]
            else:
                y_span = float(np.nanmax(y) - np.nanmin(y)) if len(y) > 1 else max(abs(stat_val) * 0.1, 0.05)
                if y_span == 0:
                    y_span = max(abs(stat_val) * 0.1, 0.05)
                y_text = float(np.nanmax(y)) + levels_lin[idx % len(levels_lin)] * y_span

            if stat_val > 0:
                exp = int(np.floor(np.log10(stat_val)))
                mant = stat_val / (10 ** exp)
                label_text = rf"${mant:.2f}\times 10^{{{exp}}}$"
            else:
                label_text = "NA"

            x_text = x_center
            if idx == 0:
                x_text += 0.10
            elif idx == len(category_order) - 1:
                x_text -= 0.10

            ax.text(
                x_text,
                y_text,
                label_text,
                ha="center",
                va="bottom",
                fontsize=4.0,
                color="black",
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.15),
            )

    xticks = [positions[c] for c in category_order]
    xticklabels = [xtick_labels_map[c] for c in category_order]

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=35, ha="right")
    ax.set_ylabel(ylabel if not use_log_scale else f"{ylabel} (log)")
    ax.set_title("")
    ax.grid(False)

    if positions:
        xvals = list(positions.values())
        ax.set_xlim(min(xvals) - 0.70, max(xvals) + 0.70)

    if use_log_scale and ymin_log is not None and ymax_log is not None:
        ax.set_ylim(ymin_log, ymax_log)

    subtitle = "Points = puits techniques" if point_level == "puits" else "Points = moyennes d'expériences biologiques"
    if use_log_scale:
        subtitle += " | axe Y en log"
        if filtered_out_nonpositive > 0:
            subtitle += f" | {filtered_out_nonpositive} valeur(s) <= 0 exclue(s)"
    subtitle += " | valeur affichée = médiane" if stat_label == "median" else " | valeur affichée = moyenne"

    return subtitle


def plot_metric_dots(
    points: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    basepath: Path,
    category_order: Sequence[str],
    strain_colors: dict[str, str],
    point_level: str,
    positions: dict[str, float],
    xtick_labels_map: dict[str, str],
) -> None:
    if points.empty:
        return

    fig_width_mm = max(96, 10 * len(category_order))
    fig_height_mm = 125 if metric in {"Lum_norm_max", "AUC_Lum_norm"} else 90
    fig, ax = plt.subplots(figsize=(mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm)))

    subtitle = draw_dot_panel(
        ax=ax,
        points=points,
        metric=metric,
        ylabel=ylabel,
        category_order=category_order,
        strain_colors=strain_colors,
        positions=positions,
        xtick_labels_map=xtick_labels_map,
        point_level=point_level,
        title=title,
        stat_label="mean",
    )

    fig.suptitle(title, y=0.985, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.955, subtitle, ha="center", va="top", fontsize=6.8, color="#444444")
    fig.subplots_adjust(top=0.88, bottom=0.30, left=0.10, right=0.995)
    export_png(fig, basepath)


# -----------------------------------------------------------------------------
# Rapport
# -----------------------------------------------------------------------------

def build_report(
    df: pd.DataFrame,
    points: pd.DataFrame,
    point_level: str,
    output_dir: Path,
    title: str,
    category_order: Sequence[str],
) -> str:
    lines = []
    lines.append("ANALYSE AUC LUMINESCENCE NORMALISÉE\n")
    lines.append(f"Titre : {title}")
    lines.append(f"Dossier de sortie : {output_dir}")
    lines.append(f"Nombre de lignes d'entrée : {len(df)}")
    lines.append(f"Nombre d'expériences : {int(df['experience_id'].nunique()) if 'experience_id' in df.columns else 1}")
    lines.append(f"Nombre de conditions tracées : {len(category_order)}")
    lines.append(f"Conditions : {', '.join(category_order)}")
    lines.append("")
    lines.append(f"AUC_Lum_norm : {'générée' if not points.empty else 'non générée'} (niveau des points = {point_level})")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------------
# Pipeline global
# -----------------------------------------------------------------------------

def run_analysis(df: pd.DataFrame, title: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    layout = prepare_category_layout(df, category_col="figure_group")
    category_order = layout["category_order"]
    strain_colors = layout["strain_colors"]
    xtick_labels_map = layout["xtick_labels"]
    positions = layout["positions"]

    params = extract_parameters_per_series(df, category_col="figure_group")
    params.to_csv(output_dir / "figure_data_parametres_par_serie.csv", index=False, encoding="utf-8-sig")

    points, point_level = summary_points_for_metric(params, "AUC_Lum_norm")
    points.to_csv(output_dir / "figure_data_AUC_Lum_norm_points.csv", index=False, encoding="utf-8-sig")

    plot_metric_dots(
        points=points,
        metric="AUC_Lum_norm",
        ylabel="AUC luminescence normalisée",
        title=f"{title} - AUC luminescence normalisée",
        basepath=output_dir / "figure_AUC_Lum_norm",
        category_order=category_order,
        strain_colors=strain_colors,
        point_level=point_level,
        positions=positions,
        xtick_labels_map=xtick_labels_map,
    )

    report = build_report(df, points, point_level, output_dir, title, category_order)
    (output_dir / "resume_auc_lum_norm.txt").write_text(report, encoding="utf-8")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Générer la figure globale de l'AUC de luminescence normalisée (AUC_Lum_norm)."
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

    df["figure_group"] = detect_conditions_from_souche(df)

    # enlève toutes les conditions SCFM1
    df = df.loc[
        ~df["figure_group"].astype(str).str.startswith("SCFM1 [", na=False)
    ].copy()

    df_fig = filter_dataframe_for_figures(df)    

    detected_conditions = build_category_order(
        df_fig["figure_group"].dropna().astype(str).drop_duplicates().tolist()
    )
    if not detected_conditions:
        raise ValueError(
            "Aucune condition exploitable détectée dans la colonne 'souche'. "
            "Formats attendus par exemple : 'LB [PAO1]', 'SCFM1 [PAO1]', 'Old SCFM2 [PAO1]', 'New SCFM2 [PAO1]'."
        )

    diag_lines = []
    diag_lines.append("Colonne utilisée : souche")
    diag_lines.append("")
    diag_lines.append("Conditions tracées (ordre d'affichage) :")
    for val in detected_conditions:
        diag_lines.append(f"- {val}")

    (output_dir / "diagnostic_conditions_detectees.txt").write_text(
        "\n".join(diag_lines) + "\n",
        encoding="utf-8",
    )

    df_fig["figure_group"] = pd.Categorical(
        df_fig["figure_group"],
        categories=detected_conditions,
        ordered=True,
    )
    df_fig = df_fig.sort_values(["figure_group", "temps_h", "series_id"]).reset_index(drop=True)

    run_analysis(df_fig, "Toutes conditions", output_dir)

    print("=" * 80)
    print("AUC LUMINESCENCE NORMALISÉE TERMINÉE")
    print("=" * 80)
    print(f"Dossier de sortie : {output_dir}")
    print(f"Expériences détectées : {int(df_fig['experience_id'].nunique())}")
    print("Colonne utilisée : souche")
    print(f"Conditions détectées : {len(detected_conditions)}")
    for cond in detected_conditions:
        print(f"  - {cond}")
    print("Export : PNG uniquement")
    print("- figure_AUC_Lum_norm.png")
    print("=" * 80)


if __name__ == "__main__":
    main()