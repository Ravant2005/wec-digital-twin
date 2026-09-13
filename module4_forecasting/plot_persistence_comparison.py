"""
plot_persistence_comparison.py — Lead-wise GRU+wind vs persistence plots.

Produces:
  M42_7_gru_vs_persistence_hs_lead.png
  M42_8_gru_vs_persistence_tp_lead.png
  M42_9_gru_vs_persistence_direction_lead.png

Does NOT modify any existing Module 4.2 plots.

Run:
  PYTHONPATH=. .venv/bin/python module4_forecasting/plot_persistence_comparison.py
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COMP_CSV = "results/forecasting/exp_gru_wind/per_lead_persistence_comparison.csv"
OUT_DIR  = "outputs/module4_forecasting"
os.makedirs(OUT_DIR, exist_ok=True)


def _load() -> pd.DataFrame:
    df = pd.read_csv(COMP_CSV)
    assert len(df) == 48
    assert list(df["lead_hours"]) == list(range(1, 49))
    return df


def plot_hs(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df["lead_hours"], df["gru_hs_rmse"],         label="GRU wave+wind",
            color="#d62728", linewidth=1.8)
    ax.plot(df["lead_hours"], df["persistence_hs_rmse"], label="Persistence",
            color="black", linewidth=1.4, linestyle="--")
    ax.set_ylabel("Hs RMSE (m)")
    ax.set_title("Hs RMSE vs forecast lead time — GRU wave+wind vs persistence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["lead_hours"], df["hs_skill_vs_persistence"],
             color="#2ca02c", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Forecast lead time (h)")
    ax2.set_ylabel("Hs skill")
    ax2.set_title("Hs skill score vs persistence  (positive = GRU better)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_7_gru_vs_persistence_hs_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


def plot_tp(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df["lead_hours"], df["gru_tp_rmse"],         label="GRU wave+wind",
            color="#d62728", linewidth=1.8)
    ax.plot(df["lead_hours"], df["persistence_tp_rmse"], label="Persistence",
            color="black", linewidth=1.4, linestyle="--")
    ax.set_ylabel("Tp RMSE (s)")
    ax.set_title("Tp RMSE vs forecast lead time — GRU wave+wind vs persistence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["lead_hours"], df["tp_skill_vs_persistence"],
             color="#2ca02c", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Forecast lead time (h)")
    ax2.set_ylabel("Tp skill")
    ax2.set_title("Tp skill score vs persistence  (positive = GRU better)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_8_gru_vs_persistence_tp_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


def plot_direction(df: pd.DataFrame) -> None:
    import numpy as np
    dir_skill = 1.0 - df["gru_direction_mae"].values / df["persistence_direction_mae"].values

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(df["lead_hours"], df["gru_direction_mae"],         label="GRU wave+wind",
            color="#d62728", linewidth=1.8)
    ax.plot(df["lead_hours"], df["persistence_direction_mae"], label="Persistence",
            color="black", linewidth=1.4, linestyle="--")
    ax.set_ylabel("Direction MAE (°, circular)")
    ax.set_title("Direction MAE vs forecast lead time — GRU wave+wind vs persistence")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(df["lead_hours"], dir_skill, color="#2ca02c", linewidth=1.5)
    ax2.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax2.set_xlabel("Forecast lead time (h)")
    ax2.set_ylabel("Dir skill")
    ax2.set_title("Direction skill score vs persistence  (positive = GRU better)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_9_gru_vs_persistence_direction_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


if __name__ == "__main__":
    print("Generating lead-wise persistence comparison plots...", flush=True)
    df = _load()
    plot_hs(df)
    plot_tp(df)
    plot_direction(df)
    print(f"\nAll plots saved to: {OUT_DIR}/", flush=True)
