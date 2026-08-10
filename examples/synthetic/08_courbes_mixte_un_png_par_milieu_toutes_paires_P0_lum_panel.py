#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
08_courbes_mixte_un_png_par_milieu_toutes_paires_P0.py

Objectif
--------
Créer un fichier PNG par milieu.

Dans chaque PNG :
- un panneau par souche rapporteur ;
- chaque panneau contient uniquement la paire : P0-lux + souche rapporteur ;
- croissance : trait plein + cercles ouverts, axe Y gauche ;
- luminescence : pointillés + carrés ouverts, axe Y droit ;
- barres d'erreur : SD.

Exemple
-------
Si 4 milieux et 4 souches rapporteurs sont présents :
- 4 PNG sont créés (un par milieu) ;
- dans chaque PNG, 4 panneaux sont affichés ;
- chaque panneau compare P0-lux à une souche gene-lux.
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


# -----------------------------------------------------------------------------
# Constantes
# -----------------------------------------------------------------------------

COLONNES_MIN = ["temps_h", "souche", "type", "DO_corr", "Lum_corr"]

DEFAULT_CONTROL_STRAIN = "14.1Ac attB::MiniCTXlux(P0-lux)"
DEFAULT_PARENT_STRAIN = "14.1Ac"

MEDIUM_ORDER = {
    "2.5% SCFM2-KPi + 97.5% DMEM-SVF": 0,
    "2.5% SCFM2-KPi + 97.5% DMEM-SVF-KPi": 1,
    "SCFM2-KPi": 2,
    "SCFM2-KPi (Spd)": 3,
    "LB": 10,
    "SCFM1": 11,
    "Old SCFM2": 12,
    "New SCFM2": 13,
}

REPORTER_COLORS = {
    "P0-lux": "#1f77b4",
    "PspeD-lux": "#d62728",
    "PspeD2-1A-lux": "#2ca02c",
    "PspeD2-3B-lux": "#9467bd",
    "PspeE-lux": "#ff7f0e",
    "14.1Ac": "#7f7f7f",
}

REPORTER_ORDER = {
    "P0-lux": 0,
    "PspeD-lux": 1,
    "PspeD2-1A-lux": 2,
    "PspeD2-3B-lux": 3,
    "PspeE-lux": 4,
    "14.1Ac": 99,
}

FALLBACK_PALETTE = [
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
    for suffix in ["_normalise_DO", "_normalise_LUM", "_corrige_blancs"]:
        name = name.replace(suffix, "")
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
            base = parent_name.replace("NORM_", "FIG_PAR_MILIEU_P0_VS_SOUCHES_", 1)
            out = (input_path.parent.parent / base).resolve()
        else:
            out = (input_path.parent / f"FIG_PAR_MILIEU_P0_VS_SOUCHES_{cleaner_basename(input_path)}").resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if all(p.parent.name.startswith("NORM_") for p in input_paths):
        out_root = input_paths[0].parent.parent
    else:
        out_root = Path(os.path.commonpath([str(p.parent) for p in input_paths]))

    stems = [cleaner_basename(p) for p in input_paths]
    suffix = "__".join(stems[:3]) if len(stems) <= 3 else f"{stems[0]}__plus_{len(stems)-1}_autres"
    out = (out_root / f"FIG_PAR_MILIEU_P0_VS_SOUCHES_multi_{suffix}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out



def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")



def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in ["temps_h", "replicat", "lecture", "DO_corr", "Lum_corr", "Lum_norm", "DO_brute", "Lum_brute"]:
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
    if explicit_ids is not None and len(explicit_ids) != len(input_paths):
        raise ValueError("Le nombre de --experience-ids doit correspondre au nombre de fichiers d'entrée.")

    out = []
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
        souche = df["souche"].astype(str)
        rep = df["replicat"].astype("Int64").astype(str) if "replicat" in df.columns else pd.Series(["NA"] * len(df), index=df.index)
        base = souche + "__rep" + rep

    exp = df["experience_id"].astype(str)
    return exp + "::__" + base


# -----------------------------------------------------------------------------
# Parsing de la colonne souche
# -----------------------------------------------------------------------------


def normalize_condition_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = re.sub(r"[_]+", " ", str(value).strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None



def extract_medium_and_strain(value: object) -> tuple[str | None, str | None]:
    if pd.isna(value):
        return None, None

    text = re.sub(r"\s+", " ", str(value).strip())

    match = re.fullmatch(r"(.+?)\s*\[\s*(.+?)\s*\]", text)
    if match:
        medium = match.group(1).strip()
        strain = match.group(2).strip()
        return medium or None, strain or None

    match = re.search(r"\b14\.1Ac\b.*$", text)
    if match:
        medium = text[:match.start()].strip()
        strain = text[match.start():].strip()
        return medium or None, strain or None

    return None, None



def is_blank_condition(value: object) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return bool(re.match(r"^(blanc|blank)\s*\d*$", text))



def prepare_growth_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["condition_norm"] = out["souche"].map(normalize_condition_label)
    parsed = out["condition_norm"].map(extract_medium_and_strain)
    out["medium"] = parsed.map(lambda x: x[0])
    out["strain"] = parsed.map(lambda x: x[1])

    out = out.loc[out["type"].astype(str).str.lower() == "souche"].copy()
    out = out.loc[out["DO_corr"].notna() | out["Lum_corr"].notna()].copy()
    out = out.loc[out["condition_norm"].notna()].copy()
    out = out.loc[~out["condition_norm"].astype(str).map(is_blank_condition)].copy()
    out = out.loc[out["medium"].notna() & out["strain"].notna()].copy()
    return out.reset_index(drop=True)


# -----------------------------------------------------------------------------
# Ordres / couleurs / libellés
# -----------------------------------------------------------------------------


def build_medium_order(values: Sequence[str]) -> list[str]:
    uniq = pd.Series(list(values), dtype="string").dropna().astype(str).drop_duplicates().tolist()
    return sorted(uniq, key=lambda x: (MEDIUM_ORDER.get(x, 99), x.lower()))



def compact_strain_label(strain: str) -> str:
    text = re.sub(r"\s+", " ", str(strain).strip())

    if re.fullmatch(r"14\.1Ac", text, flags=re.IGNORECASE):
        return "14.1Ac"

    match = re.search(r"attB::\s*(.+)$", text, flags=re.IGNORECASE)
    reporter = match.group(1).strip() if match else text

    mini_match = re.fullmatch(r"MiniCTXlux\s*\(\s*(.+?)\s*\)", reporter, flags=re.IGNORECASE)
    if mini_match:
        reporter = mini_match.group(1).strip()

    return reporter.strip("() ")



def strain_match_key(strain: str) -> str:
    compact = compact_strain_label(strain)
    compact = compact.replace("–", "-").replace("—", "-")
    return re.sub(r"[^a-z0-9]+", "", compact.lower())



def build_strain_order(values: Sequence[str]) -> list[str]:
    uniq = pd.Series(list(values), dtype="string").dropna().astype(str).drop_duplicates().tolist()
    return sorted(
        uniq,
        key=lambda strain: (
            REPORTER_ORDER.get(compact_strain_label(strain), 50),
            compact_strain_label(strain).lower(),
            strain.lower(),
        ),
    )



def build_complete_color_map(order: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    fallback = iter(FALLBACK_PALETTE * 20)

    for strain in order:
        compact = compact_strain_label(strain)
        if compact in REPORTER_COLORS:
            out[strain] = REPORTER_COLORS[compact]
            continue
        color = next(fallback)
        while color in out.values():
            color = next(fallback)
        out[strain] = color

    return out



def resolve_detected_strain(requested: str, detected: Sequence[str], role: str) -> str:
    detected_list = list(detected)

    if requested in detected_list:
        return requested

    requested_key = strain_match_key(requested)
    matches = [strain for strain in detected_list if strain_match_key(strain) == requested_key]

    if len(matches) == 1:
        return matches[0]

    detected_text = "\n".join(f"- {strain}" for strain in detected_list)
    if not matches:
        raise ValueError(
            f"La souche {role} n'a pas été trouvée.\n"
            f"Libellé recherché : {requested}\n"
            "Libellés détectés dans les données :\n"
            f"{detected_text}"
        )

    matches_text = "\n".join(f"- {strain}" for strain in matches)
    raise ValueError(
        f"Le libellé de la souche {role} est ambigu : {requested}\n"
        "Correspondances possibles :\n"
        f"{matches_text}\n"
        "Utilise le libellé complet avec l'option correspondante."
    )



def medium_filename(medium: str) -> str:
    return clean_text(medium)


# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
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
        }
    )



def mm_to_inch(mm: float) -> float:
    return mm / 25.4



def save_png(fig: plt.Figure, path: Path, dpi: int = 600) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)



def compute_panel_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 1:
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
) -> tuple[pd.DataFrame, str]:
    if value_col not in df.columns:
        return pd.DataFrame(), "missing"

    work = df.loc[df[value_col].notna()].copy()
    work = work.loc[work[panel_col].notna() & work[compare_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(), "empty"

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
            .rename(columns={"mean": "mean_value", "std": "sd_value", "count": "n_units"})
        )
        return summary, "technique"

    if "lecture" in work.columns:
        exp_lines = (
            work.groupby(["experience_id", panel_col, compare_col, "lecture"], dropna=False, observed=True)
            .agg(bio_value=(value_col, "mean"), temps_h=("temps_h", "median"))
            .reset_index()
        )
        summary = (
            exp_lines.groupby([panel_col, compare_col, "lecture"], dropna=False, observed=True)
            .agg(mean_value=("bio_value", "mean"), sd_value=("bio_value", "std"), n_units=("bio_value", "count"), temps_h=("temps_h", "median"))
            .reset_index()[[panel_col, compare_col, "temps_h", "mean_value", "sd_value", "n_units"]]
        )
        return summary, "biologique"

    raise ValueError("La colonne 'lecture' est requise pour agréger plusieurs expériences biologiques.")


# -----------------------------------------------------------------------------
# Tracé
# -----------------------------------------------------------------------------


def make_pair_handles(pair_order: Sequence[str], color_map: dict[str, str]) -> list[Line2D]:
    return [
        Line2D([0], [0], color=color_map[strain], lw=2.0, label=compact_strain_label(strain))
        for strain in pair_order
    ]


def compute_panel_lum_ylim(summary_lum: pd.DataFrame, pair_order: Sequence[str]) -> tuple[float, float]:
    sub = summary_lum.loc[summary_lum['strain'].isin(pair_order)].copy()
    if sub.empty:
        return (0.0, 1.0)

    y = pd.to_numeric(sub['mean_value'], errors='coerce').to_numpy(dtype=float)
    sd = pd.to_numeric(sub['sd_value'], errors='coerce').fillna(0).to_numpy(dtype=float)
    low = np.nanmin(y - sd)
    high = np.nanmax(y + sd)

    if not np.isfinite(low):
        low = float(np.nanmin(y)) if len(y) else 0.0
    if not np.isfinite(high):
        high = float(np.nanmax(y)) if len(y) else 1.0

    span = high - low
    pad = 0.05 * span if span > 0 else max(abs(high) * 0.05, 1.0)
    return (low - pad, high + pad)



def draw_subplot_dual_axis(
    ax_left: plt.Axes,
    summary_do: pd.DataFrame,
    summary_lum: pd.DataFrame,
    target_strain: str,
    pair_order: Sequence[str],
    color_map: dict[str, str],
    ylim_left: tuple[float, float],
    ylim_right_default: tuple[float, float] | None,
) -> None:
    ax_right = ax_left.twinx()

    sub_do_all = summary_do.loc[summary_do["strain"].isin(pair_order)].copy()
    sub_lum_all = summary_lum.loc[summary_lum["strain"].isin(pair_order)].copy()

    for strain in pair_order:
        color = color_map[strain]
        sub_do = sub_do_all.loc[sub_do_all["strain"] == strain].sort_values("temps_h")
        if not sub_do.empty:
            ax_left.errorbar(
                sub_do["temps_h"].to_numpy(dtype=float),
                sub_do["mean_value"].to_numpy(dtype=float),
                yerr=sub_do["sd_value"].fillna(0).to_numpy(dtype=float),
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

        sub_lum = sub_lum_all.loc[sub_lum_all["strain"] == strain].sort_values("temps_h")
        if not sub_lum.empty:
            ax_right.errorbar(
                sub_lum["temps_h"].to_numpy(dtype=float),
                sub_lum["mean_value"].to_numpy(dtype=float),
                yerr=sub_lum["sd_value"].fillna(0).to_numpy(dtype=float),
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

    ax_left.set_title(compact_strain_label(target_strain), loc="left", fontweight="bold", fontsize=7.4, pad=12)
    ax_left.set_xlabel("Time (h)")
    ax_left.set_ylabel("Optical density")
    ax_right.set_ylabel("Luminescence")
    ax_left.grid(False)

    x_arrays = []
    if not sub_do_all.empty:
        x_arrays.append(sub_do_all["temps_h"].dropna().to_numpy(dtype=float))
    if not sub_lum_all.empty:
        x_arrays.append(sub_lum_all["temps_h"].dropna().to_numpy(dtype=float))
    x_arrays = [arr for arr in x_arrays if len(arr) > 0]
    if x_arrays:
        x_all = np.concatenate(x_arrays)
        ax_left.set_xlim(float(np.nanmin(x_all)), float(np.nanmax(x_all)))
    ax_left.margins(x=0.0)

    ax_left.set_ylim(*ylim_left)
    panel_ylim_right = compute_panel_lum_ylim(summary_lum=sub_lum_all, pair_order=pair_order)
    if ylim_right_default is not None and panel_ylim_right is None:
        panel_ylim_right = ylim_right_default
    ax_right.set_ylim(*panel_ylim_right)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["right"].set_visible(True)



def plot_medium_panel(
    medium_df: pd.DataFrame,
    medium: str,
    control_strain: str,
    targets: Sequence[str],
    color_map: dict[str, str],
    output_path: Path,
    ylim_left: tuple[float, float],
    ylim_right: tuple[float, float],
    dpi: int,
) -> tuple[str, str]:
    summary_do, mode_do = aggregate_curve_data(
        medium_df,
        panel_col="medium",
        compare_col="strain",
        value_col="DO_corr",
    )
    summary_lum, mode_lum = aggregate_curve_data(
        medium_df,
        panel_col="medium",
        compare_col="strain",
        value_col="Lum_corr",
    )

    if summary_do.empty and summary_lum.empty:
        raise ValueError(f"Aucune donnée exploitable pour le milieu {medium}.")

    n_panels = len(targets)
    nrows, ncols = compute_panel_grid(n_panels)
    if n_panels == 4:
        fig_width_mm = 190
        fig_height_mm = 150
    elif n_panels == 3:
        nrows, ncols = 1, 3
        fig_width_mm = 280
        fig_height_mm = 115
    else:
        fig_width_mm = max(95, 92 * ncols)
        fig_height_mm = max(90, 70 * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(mm_to_inch(fig_width_mm), mm_to_inch(fig_height_mm)),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for index, target_strain in enumerate(targets):
        pair_order = [control_strain, target_strain]
        draw_subplot_dual_axis(
            ax_left=axes_flat[index],
            summary_do=summary_do,
            summary_lum=summary_lum,
            target_strain=target_strain,
            pair_order=pair_order,
            color_map=color_map,
            ylim_left=ylim_left,
            ylim_right_default=ylim_right,
        )

    for index in range(len(targets), len(axes_flat)):
        axes_flat[index].axis("off")

    do_text = "mean of biological replicates" if mode_do == "biologique" else "mean of technical replicates"
    lum_text = "mean of biological replicates" if mode_lum == "biologique" else "mean of technical replicates"
    subtitle = (
        f"Solid lines with open circles = growth (left axis, {do_text}); "
        f"dashed lines with open squares = luminescence (right axis, {lum_text}); "
        "error bars = SD."
    )

    figure_title = f"Growth and luminescence dynamics across P0-lux pairs in medium: {medium}"
    fig.suptitle(figure_title, y=0.995, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.969, subtitle, ha="center", va="top", fontsize=7.1, color="#444444")

    fig.legend(
        handles=make_pair_handles([control_strain, targets[0]], color_map),
        title="Line code",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.935),
        ncol=2,
        frameon=False,
        handlelength=2.8,
        columnspacing=2.2,
    )

    fig.subplots_adjust(
        top=0.80,
        bottom=0.09,
        left=0.08,
        right=0.93,
        hspace=0.78,
        wspace=0.55,
    )

    save_png(fig, output_path, dpi=dpi)
    return mode_do, mode_lum


# -----------------------------------------------------------------------------
# Pipeline
# -----------------------------------------------------------------------------


def compute_global_limits(df: pd.DataFrame) -> tuple[tuple[float, float], tuple[float, float]]:
    do_values = pd.to_numeric(df["DO_corr"], errors="coerce").dropna()
    lum_values = pd.to_numeric(df["Lum_corr"], errors="coerce").dropna()

    if do_values.empty:
        raise ValueError("Impossible de tracer DO_corr : aucune valeur exploitable.")
    if lum_values.empty:
        raise ValueError("Impossible de tracer Lum_corr : aucune valeur exploitable.")

    do_min = min(0.0, float(do_values.min()))
    do_max = float(do_values.max())
    do_span = do_max - do_min
    do_pad = 0.05 * do_span if do_span > 0 else 0.05
    ylim_do = (do_min, do_max + do_pad)

    lum_min = float(lum_values.min())
    lum_max = float(lum_values.max())
    lum_span = lum_max - lum_min
    lum_pad = 0.05 * lum_span if lum_span > 0 else max(abs(lum_max) * 0.05, 1.0)
    ylim_lum = (lum_min - lum_pad, lum_max + lum_pad)

    return ylim_do, ylim_lum



def resolve_target_strains(
    detected_order: Sequence[str],
    control_strain: str,
    requested_targets: Sequence[str] | None,
    include_parent: bool,
) -> list[str]:
    detected = list(detected_order)

    if requested_targets:
        targets: list[str] = []
        for requested in requested_targets:
            resolved = resolve_detected_strain(requested=requested, detected=detected, role="cible")
            if resolved != control_strain and resolved not in targets:
                targets.append(resolved)
    else:
        targets = [strain for strain in detected if strain != control_strain]
        if not include_parent:
            parent_key = strain_match_key(DEFAULT_PARENT_STRAIN)
            targets = [strain for strain in targets if strain_match_key(strain) != parent_key]

    if not targets:
        raise ValueError("Aucune souche cible à comparer à P0-lux.")

    return targets



def write_summary(
    output_dir: Path,
    control_strain: str,
    targets: Sequence[str],
    media: Sequence[str],
    exported_files: Sequence[Path],
    ylim_do: tuple[float, float],
    ylim_lum: tuple[float, float],
) -> None:
    lines = [
        "COMPARAISONS PAR MILIEU : TOUTES LES PAIRES P0-LUX VS SOUCHES",
        "",
        f"Contrôle : {control_strain}",
        f"Nombre de souches cibles/panneaux par PNG : {len(targets)}",
        f"Nombre de milieux/PNG : {len(media)}",
        f"Axe Y DO commun : {ylim_do}",
        "Axe Y luminescence : ajusté individuellement à chaque panneau",
        "",
        "Souches cibles :",
    ]
    lines.extend(f"- {target}" for target in targets)
    lines.extend(["", "PNG créés :"])
    lines.extend(f"- {path.name}" for path in exported_files)

    (output_dir / "resume_par_milieu_toutes_paires_P0.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")



def run_analysis(
    df: pd.DataFrame,
    output_dir: Path,
    control_strain: str,
    requested_targets: Sequence[str] | None,
    include_parent: bool,
    dpi: int,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    medium_order = build_medium_order(df["medium"].dropna().astype(str).tolist())
    strain_order = build_strain_order(df["strain"].dropna().astype(str).tolist())

    if not medium_order:
        raise ValueError("Aucun milieu exploitable détecté.")
    if not strain_order:
        raise ValueError("Aucune souche exploitable détectée.")

    control_strain = resolve_detected_strain(
        requested=control_strain,
        detected=strain_order,
        role="contrôle P0-lux",
    )

    targets = resolve_target_strains(
        detected_order=strain_order,
        control_strain=control_strain,
        requested_targets=requested_targets,
        include_parent=include_parent,
    )

    selected_df = df.loc[df["strain"].isin([control_strain, *targets])].copy()
    ylim_do, ylim_lum = compute_global_limits(selected_df)
    color_map = build_complete_color_map([control_strain, *targets])

    exported_files: list[Path] = []
    for medium in medium_order:
        medium_df = df.loc[(df["medium"] == medium) & (df["strain"].isin([control_strain, *targets]))].copy()
        if medium_df.empty:
            continue

        output_path = output_dir / f"panel_{medium_filename(medium)}_toutes_paires_P0-lux.png"
        plot_medium_panel(
            medium_df=medium_df,
            medium=medium,
            control_strain=control_strain,
            targets=targets,
            color_map=color_map,
            output_path=output_path,
            ylim_left=ylim_do,
            ylim_right=ylim_lum,
            dpi=dpi,
        )
        exported_files.append(output_path)

    if not exported_files:
        raise ValueError("Aucun PNG n'a été généré : aucun milieu ne contient les souches demandées.")

    write_summary(
        output_dir=output_dir,
        control_strain=control_strain,
        targets=targets,
        media=medium_order,
        exported_files=exported_files,
        ylim_do=ylim_do,
        ylim_lum=ylim_lum,
    )

    return exported_files


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Créer un PNG par milieu, avec un panneau par souche, "
            "en comparant toujours la souche à P0-lux."
        )
    )
    parser.add_argument("input_files", nargs="+", help="Un ou plusieurs fichiers CSV/XLSX/XLS")
    parser.add_argument("--experience-ids", nargs="*", default=None, help="IDs d'expériences à associer aux fichiers d'entrée")
    parser.add_argument("--output-dir", default=None, help="Dossier de sortie (optionnel)")
    parser.add_argument(
        "--control-strain",
        default=DEFAULT_CONTROL_STRAIN,
        help=("Libellé de la souche P0-lux. Les variantes MiniCTXlux(P0-lux) et attB::P0-lux sont reconnues automatiquement."),
    )
    parser.add_argument(
        "--target-strains",
        nargs="*",
        default=None,
        help=(
            "Liste optionnelle des souches à comparer à P0-lux. "
            "Sans cette option, toutes les souches sauf P0-lux et 14.1Ac sont utilisées."
        ),
    )
    parser.add_argument("--include-parent", action="store_true", help="Inclure aussi 14.1Ac parmi les panneaux si présent")
    parser.add_argument("--dpi", type=int, default=600, help="Résolution des PNG, 600 par défaut")
    return parser.parse_args()



def main() -> None:
    configure_matplotlib()
    args = parse_args()

    input_paths = [Path(path).resolve() for path in args.input_files]
    for path in input_paths:
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

    dataframes = [prepare_dataframe(read_table(path)) for path in input_paths]
    for dataframe in dataframes:
        ensure_columns(dataframe, COLONNES_MIN)

    df = add_experience_ids(dataframes, input_paths, args.experience_ids)
    df = prepare_dataframe(df)
    df["series_id"] = infer_series_id(df)

    output_dir = infer_output_dir(
        input_paths=input_paths,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )

    df_growth = prepare_growth_dataframe(df)
    if df_growth.empty:
        raise ValueError(
            "Aucune donnée exploitable détectée. Vérifie les colonnes souche, type, DO_corr et Lum_corr."
        )

    exported_files = run_analysis(
        df=df_growth,
        output_dir=output_dir,
        control_strain=args.control_strain,
        requested_targets=args.target_strains,
        include_parent=args.include_parent,
        dpi=args.dpi,
    )

    print("=" * 80)
    print("FIGURES PAR MILIEU : TOUTES LES PAIRES P0-LUX TERMINEES")
    print("=" * 80)
    print(f"Dossier de sortie : {output_dir}")
    print(f"Nombre de PNG créés (milieux) : {len(exported_files)}")
    print(f"Nombre de panneaux par PNG (souches cibles) : {len(pd.Series(df_growth['strain']).drop_duplicates()) - 1}")
    for path in exported_files:
        print(f"- {path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
