"""Matplotlib scientific figures used by the application."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


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
