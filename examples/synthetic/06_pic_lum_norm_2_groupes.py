#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère deux figures séparées pour le pic de luminescence normalisée.

Ordre imposé des souches dans chaque figure :
    P0, speD, speE, speD2-1A, speD2-3B

Groupe 1, ordre des milieux :
    1. 2.5% SCFM2-KPi + 97.5% DMEM-SVF
    2. 2.5% SCFM2-KPi + 97.5% DMEM-SVF-KPi

Groupe 2, ordre des milieux :
    1. SCFM2-KPi
    2. SCFM2-KPi (Spd)

Les points représentent les puits/séries techniques lorsqu'une seule expérience
est fournie. Avec plusieurs expériences, chaque point représente la moyenne des
puits d'une expérience biologique.
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


# =============================================================================
# Configuration propre à ce script
# =============================================================================

METRIC = "Lum_norm_max"
Y_LABEL = "Normalized luminescence peak"
USE_LOG_Y = True
FILE_PREFIX = "figure_Lum_norm_max"
POINTS_FILENAME = "figure_data_Lum_norm_max_points.csv"
PARAMS_FILENAME = "figure_data_parametres_par_serie_pic.csv"
REPORT_FILENAME = "resume_pic_lum_norm_2_groupes.txt"
ANALYSIS_LABEL = "PIC DE LUMINESCENCE NORMALISÉE"
TITLE_PREFIX = (
    "Peak normalized luminescence of polyamine-associated promoter–lux "
    "reporters in Pseudomonas aeruginosa 14.1Ac"
)


# =============================================================================
# Ordres demandés
# =============================================================================

STRAIN_ORDER = ["P0", "speD", "speE", "speD2-1A", "speD2-3B"]

GROUP_MEDIA: dict[str, list[str]] = {
    "Groupe 1": [
        "2.5% SCFM2-KPi + 97.5% DMEM-SVF",
        "2.5% SCFM2-KPi + 97.5% DMEM-SVF-KPi",
    ],
    "Groupe 2": [
        "SCFM2-KPi",
        "SCFM2-KPi (Spd)",
    ],
}

GROUP_SLUG = {"Groupe 1": "groupe1", "Groupe 2": "groupe2"}

STRAIN_COLORS = {
    "P0": "#1f77b4",
    "speD": "#d62728",
    "speE": "#ff7f0e",
    "speD2-1A": "#2ca02c",
    "speD2-3B": "#9467bd",
}

# Le premier milieu de chaque groupe utilise un cercle, le second un triangle.
MEDIUM_MARKERS: dict[str, str] = {
    media[0]: "o" for media in GROUP_MEDIA.values()
}
MEDIUM_MARKERS.update({media[1]: "^" for media in GROUP_MEDIA.values()})

COLONNES_MIN = ["temps_h", "souche", "type", "DO_corr", "Lum_norm"]


# =============================================================================
# Lecture et préparation
# =============================================================================

def read_table(path: Path, sheet_name: str | int = 0) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet_name)
    raise ValueError(f"Format non supporté : {path.suffix}")


def clean_text(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return text or "table"


def cleaner_basename(path: Path) -> str:
    name = path.stem.replace("_normalise_DO", "").replace("_corrige_blancs", "")
    return clean_text(name)


def infer_output_dir(input_paths: Sequence[Path], output_dir: Path | None) -> Path:
    if output_dir is not None:
        out = output_dir.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if len(input_paths) == 1:
        input_path = input_paths[0]
        if input_path.parent.name.startswith("NORM_"):
            out = input_path.parent.parent / input_path.parent.name.replace("NORM_", "FIG_", 1)
        else:
            out = input_path.parent / f"FIG_{cleaner_basename(input_path)}"
        out = out.resolve()
        out.mkdir(parents=True, exist_ok=True)
        return out

    if all(path.parent.name.startswith("NORM_") for path in input_paths):
        out_root = input_paths[0].parent.parent
    else:
        out_root = Path(os.path.commonpath([str(path.parent) for path in input_paths]))

    stems = [cleaner_basename(path) for path in input_paths]
    suffix = "__".join(stems[:3]) if len(stems) <= 3 else f"{stems[0]}__plus_{len(stems)-1}_autres"
    out = (out_root / f"FIG_multi_{suffix}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    numeric_columns = [
        "temps_h", "replicat", "DO_corr", "Lum_corr", "Lum_norm",
        "DO_brute", "Lum_brute",
    ]
    for column in numeric_columns:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype("string")

    sort_columns = [
        column for column in
        ["experience_id", "souche", "sample_header", "puits", "temps_h"]
        if column in out.columns
    ]
    if sort_columns:
        out = out.sort_values(sort_columns).reset_index(drop=True)
    return out


def add_experience_ids(
    dfs: list[pd.DataFrame],
    input_paths: Sequence[Path],
    explicit_ids: Sequence[str] | None,
) -> pd.DataFrame:
    if explicit_ids is not None and len(explicit_ids) != len(input_paths):
        raise ValueError(
            "Le nombre de --experience-ids doit correspondre au nombre de fichiers d'entrée."
        )

    merged: list[pd.DataFrame] = []
    for index, (df, path) in enumerate(zip(dfs, input_paths), start=1):
        current = df.copy()
        fallback_id = cleaner_basename(path) or f"exp{index}"
        if explicit_ids is not None:
            current["experience_id"] = str(explicit_ids[index - 1])
        elif "experience_id" in current.columns and current["experience_id"].notna().any():
            # Conserve les différents IDs biologiques déjà présents dans un fichier fusionné.
            current["experience_id"] = (
                current["experience_id"].astype("string").fillna(fallback_id).astype(str)
            )
        else:
            current["experience_id"] = fallback_id

        current["source_file"] = str(path)
        merged.append(current)

    return pd.concat(merged, ignore_index=True)


def infer_series_id(df: pd.DataFrame) -> pd.Series:
    if "puits" in df.columns and df["puits"].notna().any():
        base = df["puits"].astype(str)
    elif "sample_header" in df.columns and df["sample_header"].notna().any():
        base = df["sample_header"].astype(str)
    else:
        strain = df["souche"].astype(str)
        if "replicat" in df.columns:
            replicate = df["replicat"].astype("Int64").astype(str)
        else:
            replicate = pd.Series(["NA"] * len(df), index=df.index)
        base = strain + "__rep" + replicate

    return df["experience_id"].astype(str) + "::__" + base


# =============================================================================
# Reconnaissance des milieux et des souches
# =============================================================================

def normalize_key(value: object) -> str:
    text = str(value).strip()
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = re.sub(r"[_]+", " ", text)
    text = re.sub(r"\s*\+\s*", " + ", text)
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.casefold()


ALL_MEDIA = [medium for media in GROUP_MEDIA.values() for medium in media]
MEDIA_BY_KEY = {normalize_key(medium): medium for medium in ALL_MEDIA}


def canonicalize_medium(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1].strip()
    return MEDIA_BY_KEY.get(normalize_key(text))


def canonicalize_strain(value: object) -> str | None:
    if pd.isna(value):
        return None

    key = normalize_key(value).replace(" ", "")
    if "sped2-1a" in key:
        return "speD2-1A"
    if "sped2-3b" in key:
        return "speD2-3B"
    if "spee" in key:
        return "speE"
    if re.search(r"p[o0](?:-?lux)?", key):
        return "P0"
    if "sped" in key:
        return "speD"
    return None


def split_condition(value: object) -> tuple[str | None, str | None]:
    """Retourne (milieu canonique, souche canonique)."""
    if pd.isna(value):
        return None, None

    text = re.sub(r"\s+", " ", str(value).strip())
    if not text:
        return None, None

    # Format principal : "Milieu [Souche]".
    bracket_match = re.fullmatch(r"(.+?)\s*\[\s*(.+?)\s*\]", text)
    if bracket_match:
        medium = canonicalize_medium(bracket_match.group(1))
        strain = canonicalize_strain(bracket_match.group(2))
        if medium is not None and strain is not None:
            return medium, strain

    # Format d'affichage : "Souche (Milieu)". On teste chaque parenthèse
    # ouvrante pour gérer le milieu imbriqué "SCFM2-KPi (Spd)".
    if text.endswith(")"):
        for match in re.finditer(r"\(", text):
            split_index = match.start()
            medium_candidate = text[split_index + 1:-1].strip()
            medium = canonicalize_medium(medium_candidate)
            if medium is not None:
                strain = canonicalize_strain(text[:split_index].strip())
                if strain is not None:
                    return medium, strain

    # Format éventuel : "Milieu 14.1Ac ...".
    strain_start = re.search(r"\b14\.1Ac\b", text, flags=re.IGNORECASE)
    if strain_start:
        medium = canonicalize_medium(text[:strain_start.start()].strip())
        strain = canonicalize_strain(text[strain_start.start():].strip())
        if medium is not None and strain is not None:
            return medium, strain

    return None, None


def add_plot_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["souche"].map(split_condition)
    out = df.copy()
    out["plot_medium"] = parsed.map(lambda pair: pair[0])
    out["plot_strain"] = parsed.map(lambda pair: pair[1])

    medium_to_group = {
        medium: group_name
        for group_name, media in GROUP_MEDIA.items()
        for medium in media
    }
    out["plot_group"] = out["plot_medium"].map(medium_to_group)
    out["plot_category"] = np.where(
        out["plot_medium"].notna() & out["plot_strain"].notna(),
        out["plot_strain"].astype(str) + "||" + out["plot_medium"].astype(str),
        pd.NA,
    )
    return out


def filter_dataframe_for_figures(df: pd.DataFrame) -> pd.DataFrame:
    out = df.loc[df["type"].astype(str).str.casefold() == "souche"].copy()
    out = out.loc[
        out["plot_group"].notna()
        & out["plot_medium"].notna()
        & out["plot_strain"].isin(STRAIN_ORDER)
    ].copy()
    return out.reset_index(drop=True)


def category_id(strain: str, medium: str) -> str:
    return f"{strain}||{medium}"


def category_order_for_group(points: pd.DataFrame, group_name: str) -> list[str]:
    present = set(points.loc[points["plot_group"] == group_name, "plot_category"].dropna().astype(str))
    return [
        category_id(strain, medium)
        for strain in STRAIN_ORDER
        for medium in GROUP_MEDIA[group_name]
        if category_id(strain, medium) in present
    ]


def build_positions(category_order: Sequence[str]) -> dict[str, float]:
    positions: dict[str, float] = {}
    previous_strain: str | None = None
    x = 0.0
    for index, category in enumerate(category_order):
        strain, _ = category.split("||", 1)
        if index == 0:
            x = 0.0
        elif strain == previous_strain:
            x += 0.76
        else:
            x += 1.34
        positions[category] = x
        previous_strain = strain
    return positions


def short_medium_label(medium: str) -> str:
    if medium == "2.5% SCFM2-KPi + 97.5% DMEM-SVF":
        return "2.5% SCFM2-KPi +\n97.5% DMEM-SVF"
    if medium == "2.5% SCFM2-KPi + 97.5% DMEM-SVF-KPi":
        return "2.5% SCFM2-KPi +\n97.5% DMEM-SVF-KPi"
    return medium


# =============================================================================
# Calcul des paramètres par série
# =============================================================================

def trapezoid_auc(x: pd.Series, y: pd.Series) -> float:
    valid = pd.DataFrame({"x": x, "y": y}).dropna().sort_values("x")
    if len(valid) < 2:
        return np.nan
    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=float)
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y_values, x_values))
    return float(np.trapz(y_values, x_values))


def extract_parameters_per_series(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    group_columns = [
        "experience_id", "plot_group", "plot_medium", "plot_strain",
        "plot_category", "series_id",
    ]
    rows: list[dict[str, object]] = []

    for keys, sub in df.groupby(group_columns, dropna=False, observed=True):
        metadata = dict(zip(group_columns, keys))
        sub = sub.sort_values("temps_h").copy()
        valid_lum = sub.loc[sub["Lum_norm"].notna()].copy()

        row: dict[str, object] = dict(metadata)
        row["n_points_total"] = int(len(sub))
        row["n_points_Lum_norm"] = int(len(valid_lum))
        row["AUC_DO"] = trapezoid_auc(sub["temps_h"], sub["DO_corr"])
        row["DO_max"] = float(sub["DO_corr"].max()) if sub["DO_corr"].notna().any() else np.nan
        row["AUC_Lum_norm"] = (
            trapezoid_auc(valid_lum["temps_h"], valid_lum["Lum_norm"])
            if not valid_lum.empty else np.nan
        )
        row["Lum_norm_max"] = (
            float(valid_lum["Lum_norm"].max()) if not valid_lum.empty else np.nan
        )
        row["Lum_norm_final"] = (
            float(valid_lum.iloc[-1]["Lum_norm"]) if not valid_lum.empty else np.nan
        )
        row["temps_pic_h"] = (
            float(valid_lum.loc[valid_lum["Lum_norm"].idxmax(), "temps_h"])
            if not valid_lum.empty else np.nan
        )

        for optional_column in ["sample_header", "puits", "souche"]:
            if optional_column in sub.columns and sub[optional_column].notna().any():
                row[optional_column] = str(sub.loc[sub[optional_column].notna(), optional_column].iloc[0])
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_columns).reset_index(drop=True)


def summary_points_for_metric(params: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, str]:
    if params.empty or metric not in params.columns:
        return pd.DataFrame(), "missing"

    work = params.loc[params[metric].notna()].copy()
    if work.empty:
        return pd.DataFrame(), "empty"

    n_experiments = int(work["experience_id"].nunique())
    identifying_columns = [
        "plot_group", "plot_medium", "plot_strain", "plot_category",
    ]

    if n_experiments <= 1:
        points = work[identifying_columns + [metric, "series_id"]].copy()
        points["unit_id"] = points["series_id"].astype(str)
        points["niveau"] = "puits"
        return points.reset_index(drop=True), "puits"

    points = (
        work.groupby(["experience_id"] + identifying_columns, dropna=False, observed=True)[metric]
        .mean()
        .reset_index()
    )
    points["unit_id"] = points["experience_id"].astype(str)
    points["niveau"] = "experience"
    return points.reset_index(drop=True), "experience"


# =============================================================================
# Tracé
# =============================================================================

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def jitter_positions(n: int, center: float, width: float = 0.16) -> np.ndarray:
    if n <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - width, center + width, n)


def scientific_label(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"${mantissa:.2f}\times 10^{{{exponent}}}$"


def draw_dot_panel(
    ax: plt.Axes,
    points: pd.DataFrame,
    metric: str,
    group_name: str,
    category_order: Sequence[str],
    positions: dict[str, float],
    use_log_y: bool,
) -> int:
    work = points.loc[points["plot_group"] == group_name].copy()
    excluded_nonpositive = 0

    if use_log_y:
        before = int(work[metric].notna().sum())
        work = work.loc[work[metric].notna() & (work[metric] > 0)].copy()
        excluded_nonpositive = before - len(work)
        if not work.empty:
            ax.set_yscale("log")

    all_values = work[metric].dropna().to_numpy(dtype=float)
    if len(all_values) == 0:
        return excluded_nonpositive

    global_min = float(np.nanmin(all_values))
    global_max = float(np.nanmax(all_values))
    linear_span = global_max - global_min
    if not np.isfinite(linear_span) or linear_span <= 0:
        linear_span = max(abs(global_max) * 0.25, 1.0)

    for index, category in enumerate(category_order):
        strain, medium = category.split("||", 1)
        sub = work.loc[work["plot_category"] == category]
        if sub.empty:
            continue

        x_center = positions[category]
        y = sub[metric].to_numpy(dtype=float)
        x_jittered = jitter_positions(len(sub), x_center)

        ax.scatter(
            x_jittered,
            y,
            s=29,
            color=STRAIN_COLORS[strain],
            marker=MEDIUM_MARKERS[medium],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.95,
            zorder=3,
        )

        mean_value = float(np.nanmean(y))
        sd_value = float(np.nanstd(y, ddof=1)) if len(y) >= 2 else np.nan

        ax.plot(
            [x_center - 0.20, x_center + 0.20],
            [mean_value, mean_value],
            color="black",
            linewidth=1.1,
            zorder=4,
        )

        if np.isfinite(sd_value):
            low = mean_value - sd_value
            high = mean_value + sd_value
            if (not use_log_y) or low > 0:
                ax.vlines(x_center, low, high, color="black", linewidth=0.9, zorder=4)
                ax.hlines(
                    [low, high], x_center - 0.06, x_center + 0.06,
                    color="black", linewidth=0.9, zorder=4,
                )

        if use_log_y:
            text_factors = [1.12, 1.28, 1.47, 1.18]
            y_text = float(np.nanmax(y)) * text_factors[index % len(text_factors)]
        else:
            text_offsets = [0.06, 0.13, 0.20, 0.09]
            y_text = float(np.nanmax(y)) + text_offsets[index % len(text_offsets)] * linear_span

        ax.text(
            x_center,
            y_text,
            scientific_label(mean_value),
            ha="center",
            va="bottom",
            fontsize=4.4,
            color="black",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.12),
        )

    strain_ticks: list[float] = []
    strain_labels: list[str] = []
    for strain in STRAIN_ORDER:
        strain_categories = [
            category for category in category_order
            if category.split("||", 1)[0] == strain
        ]
        if not strain_categories:
            continue
        strain_ticks.append(float(np.mean([positions[category] for category in strain_categories])))
        strain_labels.append(strain)

    ax.set_xticks(strain_ticks)
    ax.set_xticklabels(strain_labels, rotation=0, ha="center")
    ax.set_xlabel("Promoter–lux reporter")
    ax.set_ylabel(f"{Y_LABEL} (log)" if use_log_y else Y_LABEL)
    ax.grid(False)

    x_values = list(positions.values())
    ax.set_xlim(min(x_values) - 0.65, max(x_values) + 0.65)

    if use_log_y:
        ax.set_ylim(global_min / 1.25, global_max * 2.35)
    else:
        lower = min(0.0, global_min - 0.08 * linear_span)
        upper = global_max + 0.30 * linear_span
        ax.set_ylim(lower, upper)

    return excluded_nonpositive


def add_legends(ax: plt.Axes, group_name: str) -> None:
    medium_handles = [
        Line2D(
            [0], [0], marker=MEDIUM_MARKERS[medium], linestyle="none", markersize=5,
            markerfacecolor="#666666", markeredgecolor="white", label=medium,
        )
        for medium in GROUP_MEDIA[group_name]
    ]
    ax.legend(
        handles=medium_handles,
        title="Milieu (ordre gauche → droite)",
        loc="upper left",
        bbox_to_anchor=(1.005, 1.0),
        borderaxespad=0.0,
    )


def plot_group(
    points: pd.DataFrame,
    group_name: str,
    point_level: str,
    output_dir: Path,
    use_log_y: bool,
) -> Path | None:
    category_order = category_order_for_group(points, group_name)
    if not category_order:
        return None

    positions = build_positions(category_order)
    figure_width_mm = max(170, 17 * len(category_order))
    figure_height_mm = 118
    fig, ax = plt.subplots(
        figsize=(mm_to_inch(figure_width_mm), mm_to_inch(figure_height_mm))
    )

    excluded = draw_dot_panel(
        ax=ax,
        points=points,
        metric=METRIC,
        group_name=group_name,
        category_order=category_order,
        positions=positions,
        use_log_y=use_log_y,
    )
    add_legends(ax, group_name)

    point_description = (
        "Points = puits/séries techniques"
        if point_level == "puits"
        else "Points = moyennes d'expériences biologiques"
    )
    subtitle = point_description + " | barre = moyenne ± écart-type"
    if excluded:
        subtitle += f" | {excluded} valeur(s) ≤ 0 exclue(s)"

    fig.suptitle(f"{TITLE_PREFIX} — {group_name}", y=0.985, fontsize=10, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=6.8, color="#444444")
    fig.subplots_adjust(top=0.86, bottom=0.18, left=0.10, right=0.78)

    output_path = output_dir / f"{FILE_PREFIX}_{GROUP_SLUG[group_name]}.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return output_path


# =============================================================================
# Diagnostic et pipeline
# =============================================================================

def write_diagnostic(
    original_df: pd.DataFrame,
    plotted_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    all_conditions = original_df["souche"].dropna().astype(str).drop_duplicates().tolist()
    recognized = plotted_df[
        ["souche", "plot_group", "plot_strain", "plot_medium"]
    ].drop_duplicates()

    recognized_raw = set(recognized["souche"].astype(str))
    unrecognized = [condition for condition in all_conditions if condition not in recognized_raw]

    lines = [
        "Colonne utilisée : souche",
        "",
        "Ordre des souches imposé : " + ", ".join(STRAIN_ORDER),
        "",
        "Conditions reconnues et tracées :",
    ]
    if recognized.empty:
        lines.append("- aucune")
    else:
        for row in recognized.itertuples(index=False):
            lines.append(
                f"- {row.souche} -> {row.plot_group} | {row.plot_strain} | {row.plot_medium}"
            )

    lines.extend(["", "Conditions non retenues ou non reconnues :"])
    if unrecognized:
        lines.extend(f"- {condition}" for condition in unrecognized)
    else:
        lines.append("- aucune")

    (output_dir / "diagnostic_conditions_2_groupes.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run_analysis(df: pd.DataFrame, output_dir: Path, use_log_y: bool) -> list[Path]:
    params = extract_parameters_per_series(df)
    params.to_csv(output_dir / PARAMS_FILENAME, index=False, encoding="utf-8-sig")

    points, point_level = summary_points_for_metric(params, METRIC)
    points.to_csv(output_dir / POINTS_FILENAME, index=False, encoding="utf-8-sig")

    generated: list[Path] = []
    for group_name in GROUP_MEDIA:
        output_path = plot_group(
            points=points,
            group_name=group_name,
            point_level=point_level,
            output_dir=output_dir,
            use_log_y=use_log_y,
        )
        if output_path is not None:
            generated.append(output_path)

    report_lines = [
        ANALYSIS_LABEL,
        "",
        f"Nombre de lignes retenues : {len(df)}",
        f"Nombre d'expériences : {df['experience_id'].nunique() if not df.empty else 0}",
        f"Niveau des points : {point_level}",
        f"Métrique : {METRIC}",
        f"Axe Y logarithmique : {'oui' if use_log_y else 'non'}",
        "",
        "Figures générées :",
    ]
    report_lines.extend(f"- {path.name}" for path in generated)
    if not generated:
        report_lines.append("- aucune")

    (output_dir / REPORT_FILENAME).write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )
    return generated


# =============================================================================
# Interface en ligne de commande
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Générer deux figures du pic de luminescence normalisée, "
            "avec les milieux et les souches dans l'ordre imposé."
        )
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Un ou plusieurs CSV/XLSX normalisés issus du pipeline de normalisation.",
    )
    parser.add_argument(
        "--experience-ids",
        nargs="*",
        default=None,
        help="IDs d'expériences à associer aux fichiers d'entrée.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dossier de sortie. Par défaut, il est déduit du ou des fichiers d'entrée.",
    )
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

    merged = add_experience_ids(dataframes, input_paths, args.experience_ids)
    merged = prepare_dataframe(merged)
    merged["series_id"] = infer_series_id(merged)
    merged = add_plot_columns(merged)

    output_dir = infer_output_dir(
        input_paths,
        Path(args.output_dir) if args.output_dir else None,
    )
    plotted = filter_dataframe_for_figures(merged)
    write_diagnostic(merged, plotted, output_dir)

    if plotted.empty:
        raise ValueError(
            "Aucune condition demandée n'a été reconnue. Consultez "
            "diagnostic_conditions_2_groupes.txt dans le dossier de sortie."
        )

    generated = run_analysis(plotted, output_dir, use_log_y=USE_LOG_Y)

    print("=" * 80)
    print(f"{ANALYSIS_LABEL} TERMINÉ")
    print("=" * 80)
    print(f"Dossier de sortie : {output_dir}")
    print(f"Expériences détectées : {plotted['experience_id'].nunique()}")
    print("Ordre des souches : " + " -> ".join(STRAIN_ORDER))
    print("Figures générées :")
    for path in generated:
        print(f"- {path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
