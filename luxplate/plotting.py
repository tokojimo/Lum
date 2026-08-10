"""Matplotlib scientific figures used by the application."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PUBLICATION_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000")


def _publication_style(axis):
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(False)
    axis.tick_params(width=0.8, length=3)


def _series_label(row: pd.Series) -> str:
    return str(row.get("souche", row.get("sample_header", "Série")))


def plot_publication_panels(data: pd.DataFrame, *, value: str, group_by: str = "Groupe",
                            title: str | None = None):
    """Make one polished mean ± SD time-course panel per medium or strain.

    Fine lines retain the technical-series information used throughout the example
    scripts; the heavy line and translucent ribbon show mean and standard deviation.
    """
    required = {"temps_h", "souche", "sample_header", value, group_by}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data[value].notna()].copy()
    if "type" in work:
        work = work.loc[work["type"].eq("souche")]
    panels = list(dict.fromkeys(work[group_by].dropna().astype(str)))
    if not panels:
        raise ValueError("Aucune donnée exploitable pour construire les figures finales.")
    ncols = min(3, len(panels)); nrows = int(np.ceil(len(panels) / ncols))
    figure, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.15 * nrows), squeeze=False)
    ylabel = {"DO_corr": "Densité optique corrigée", "Lum_corr": "Luminescence corrigée (RLU)",
              "Lum_norm": "Luminescence normalisée (RLU / DO)"}.get(value, value)
    for panel_index, panel in enumerate(panels):
        axis = axes.flat[panel_index]
        subset = work.loc[work[group_by].astype(str).eq(panel)]
        strains = list(dict.fromkeys(subset["souche"].astype(str)))
        for index, strain in enumerate(strains):
            color = PUBLICATION_COLORS[index % len(PUBLICATION_COLORS)]
            strain_data = subset.loc[subset["souche"].astype(str).eq(strain)]
            for _, curve in strain_data.groupby("sample_header", sort=False):
                curve = curve.sort_values("temps_h")
                axis.plot(curve["temps_h"], curve[value], color=color, lw=0.7, alpha=0.20)
            summary = strain_data.groupby("temps_h", as_index=False)[value].agg(["mean", "std"]).reset_index()
            x = summary["temps_h"].to_numpy(float); y = summary["mean"].to_numpy(float)
            sd = summary["std"].fillna(0).to_numpy(float)
            axis.fill_between(x, y - sd, y + sd, color=color, alpha=0.14, linewidth=0)
            axis.plot(x, y, color=color, lw=2, label=strain)
        axis.set(title=panel, xlabel="Temps (h)", ylabel=ylabel)
        axis.title.set_fontweight("bold"); axis.title.set_ha("left"); axis.title.set_position((0, 1.0))
        _publication_style(axis)
    for axis in axes.flat[len(panels):]:
        axis.remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, title="Souche", frameon=False, loc="upper center",
                      bbox_to_anchor=(0.5, 0.93), ncol=min(5, len(labels)))
    figure.suptitle(title or f"{ylabel} au cours du temps", fontsize=13, fontweight="bold", y=0.995)
    figure.text(0.5, 0.955, "Lignes fines = puits techniques · ligne épaisse = moyenne · ruban = ± écart-type",
                ha="center", color="#555555", fontsize=8)
    figure.subplots_adjust(top=0.82, bottom=0.12, hspace=0.48, wspace=0.34)
    return figure


def build_publication_figures(data: pd.DataFrame, *, title: str = "Analyse LuxPlate") -> list[tuple[str, object]]:
    """Build the curve families represented in the historical example scripts."""
    choices = (("DO_corr", "croissance"), ("Lum_corr", "luminescence"), ("Lum_norm", "luminescence_normalisee"))
    figures = []
    for value, suffix in choices:
        if value in data and data[value].notna().any():
            figures.append((suffix, plot_publication_panels(data, value=value, title=f"{title} — {suffix.replace('_', ' ')}")))
    return figures


def plot_raw_curves(data: pd.DataFrame):
    """Plot every raw technical curve; no averaging or interpolation."""
    required = {"temps_h", "souche", "sample_header", "DO_brute", "Lum_brute"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for header, curve in data.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["DO_brute"], marker="o", markersize=3, alpha=0.75, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_brute"], marker="o", markersize=3, alpha=0.75, label=label)
    axes[0].set(title="Croissance brute", xlabel="Temps (h)", ylabel="DO brute")
    axes[1].set(title="Luminescence brute", xlabel="Temps (h)", ylabel="Luminescence brute (RLU)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def build_guided_raw_figures(data: pd.DataFrame, *, sample_type: str) -> list[tuple[str, object]]:
    """Build one static DO/luminescence figure per biological replicate.

    Technical wells belonging to the same strain, medium and biological replicate
    stay together in a figure.  Keeping these previews as Matplotlib figures avoids
    the much heavier interactive charts in the guided import screen.
    """
    required = {"temps_h", "souche", "Groupe", "replicat", "sample_header", "type",
                "DO_brute", "Lum_brute"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    work = data.loc[data["type"].astype(str).str.lower().eq(sample_type.lower())].copy()
    figures: list[tuple[str, object]] = []
    for keys, replicate in work.groupby(["Groupe", "souche", "replicat"], dropna=False, sort=False):
        medium, strain, biological_replicate = keys
        title = f"{medium} · {strain} · réplicat biologique {biological_replicate}"
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        for header, curve in replicate.groupby("sample_header", sort=False):
            curve = curve.sort_values("temps_h", kind="stable")
            label = str(header)
            axes[0].plot(curve["temps_h"], curve["DO_brute"], lw=1.4, label=label)
            axes[1].plot(curve["temps_h"], curve["Lum_brute"], lw=1.4, label=label)
        axes[0].set(title="DO", xlabel="Temps (h)", ylabel="DO brute")
        axes[1].set(title="Luminescence", xlabel="Temps (h)", ylabel="Luminescence brute (RLU)")
        for axis in axes:
            _publication_style(axis)
            axis.legend(fontsize="x-small", frameon=False, loc="best")
        figure.suptitle(title, fontsize=11, fontweight="bold")
        figures.append((title, figure))
    return figures


def plot_qc_curves(data: pd.DataFrame, anomalies: pd.DataFrame):
    """Plot unmodified curves and overlay proposed anomalies as red crosses."""
    figure = plot_raw_curves(data)
    axes = figure.axes
    if not anomalies.empty:
        axes[0].scatter(
            anomalies["temps_h"], anomalies["DO_brute"], marker="x", s=90,
            linewidths=2.2, color="crimson", zorder=10, label="Anomalie proposée",
        )
        axes[1].scatter(
            anomalies["temps_h"], anomalies["Lum_brute"], marker="x", s=90,
            linewidths=2.2, color="crimson", zorder=10, label="Anomalie proposée",
        )
        for axis in axes:
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles))
            axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
    return figure


def plot_blank_correction(data: pd.DataFrame):
    """Compare raw and blank-corrected strain curves."""
    required = {"temps_h", "souche", "sample_header", "DO_brute", "Lum_brute", "DO_corr", "Lum_corr"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for _, curve in data.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["DO_brute"], color="0.7", alpha=0.45)
        axes[0].plot(curve["temps_h"], curve["DO_corr"], alpha=0.8, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_brute"], color="0.7", alpha=0.45)
        axes[1].plot(curve["temps_h"], curve["Lum_corr"], alpha=0.8, label=label)
    axes[0].set(title="DO : brute (gris) / corrigée", xlabel="Temps (h)", ylabel="DO")
    axes[1].set(title="Luminescence : brute (gris) / corrigée", xlabel="Temps (h)", ylabel="Luminescence (RLU)")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def plot_normalization(data: pd.DataFrame):
    """Compare corrected and OD-normalized luminescence for strain series."""
    required = {"temps_h", "souche", "sample_header", "type", "Lum_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    strains = data.loc[data["type"].eq("souche")]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for _, curve in strains.groupby("sample_header", sort=False):
        curve = curve.sort_values("temps_h")
        label = str(curve["souche"].iloc[0])
        axes[0].plot(curve["temps_h"], curve["Lum_corr"], alpha=0.8, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_norm"], alpha=0.8, label=label)
    axes[0].set(title="Luminescence corrigée", xlabel="Temps (h)", ylabel="Luminescence (RLU)")
    axes[1].set(title="Luminescence normalisée par la DO", xlabel="Temps (h)", ylabel="Lum_corr / DO_corr")
    for axis in axes:
        handles, labels = axis.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        if unique:
            axis.legend(unique.values(), unique.keys(), fontsize="small", loc="best")
        axis.grid(alpha=0.2)
    return figure


def plot_kinetics(data: pd.DataFrame, series_metrics: pd.DataFrame):
    """Plot each well and annotate OD/normalized-luminescence kinetic landmarks."""
    required = {"temps_h", "sample_header", "DO_corr", "Lum_norm"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes pour la figure : {sorted(missing)}")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    group_columns = [column for column in
                     ("experience_id", "souche", "Groupe", "sample_header", "puits", "replicat")
                     if column in data]
    for keys, curve in data.groupby(group_columns, dropna=False, sort=False):
        curve = curve.sort_values("temps_h", kind="stable")
        keys = keys if isinstance(keys, tuple) else (keys,)
        identity = dict(zip(group_columns, keys))
        label = " · ".join(str(identity[column]) for column in group_columns)
        axes[0].plot(curve["temps_h"], curve["DO_corr"], marker="o", markersize=2, label=label)
        axes[1].plot(curve["temps_h"], curve["Lum_norm"], marker="o", markersize=2, label=label)
        selected = series_metrics
        for column, value in identity.items():
            selected = selected.loc[selected[column].eq(value)]
        if selected.empty:
            continue
        metric = selected.iloc[0]
        axes[0].scatter(metric["od_max_time_h"], metric["od_max"], marker="*", s=90, zorder=5)
        axes[1].scatter(metric["lum_norm_peak_time_h"], metric["lum_norm_peak"], marker="*", s=90, zorder=5)
        if pd.notna(metric["max_growth_rate_start_h"]):
            axes[0].axvspan(metric["max_growth_rate_start_h"], metric["max_growth_rate_end_h"],
                            alpha=0.10, color="green")
    axes[0].set(title="Croissance et fenêtre maximale", xlabel="Temps (h)", ylabel="DO corrigée")
    axes[1].set(title="Luminescence et pic", xlabel="Temps (h)", ylabel="Luminescence normalisée")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize="x-small", loc="best")
    return figure
