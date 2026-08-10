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
