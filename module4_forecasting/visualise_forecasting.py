"""
visualise_forecasting.py — Diagnostic plots for Module 4.2 results.

Produces exactly 6 plots:
  1. Training vs validation loss curve (best model: GRU wave+wind)
  2. Hs RMSE vs forecast lead time (all 4 models + persistence)
  3. Tp RMSE vs forecast lead time (all 4 models + persistence)
  4. Direction MAE vs forecast lead time (all 4 models + persistence)
  5. Example 48-hour Hs forecast vs truth (GRU wave+wind, test sample 500)
  6. Example 48-hour Tp forecast vs truth (GRU wave+wind, test sample 500)

Saved to: outputs/module4_forecasting/

Run:
  PYTHONPATH=. .venv/bin/python module4_forecasting/visualise_forecasting.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_ROOT = "results/forecasting"
OUT_DIR      = "outputs/module4_forecasting"
os.makedirs(OUT_DIR, exist_ok=True)

EXPERIMENTS = ["exp_lstm_wave", "exp_lstm_wind", "exp_gru_wave", "exp_gru_wind"]
LABELS      = ["LSTM wave-only", "LSTM wave+wind", "GRU wave-only", "GRU wave+wind"]
COLORS      = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
BEST_EXP    = "exp_gru_wind"
BEST_LABEL  = "GRU wave+wind"

# Persistence per-lead: flat line at aggregate RMSE values
PERSIST_HS_RMSE  = 0.2513
PERSIST_TP_RMSE  = 2.6427
PERSIST_DIR_MAE  = 19.69


def _load_history(exp: str) -> dict:
    path = os.path.join(RESULTS_ROOT, exp, "training_history.json")
    with open(path) as f:
        return json.load(f)


def _load_lead(exp: str) -> pd.DataFrame:
    return pd.read_csv(os.path.join(RESULTS_ROOT, exp, "per_lead_metrics.csv"))


def _load_predictions(exp: str):
    data = np.load(os.path.join(RESULTS_ROOT, exp, "predictions.npz"))
    return data["predictions"], data["targets"]


# ---------------------------------------------------------------------------
# Plot 1 — Training vs validation loss (best model)
# ---------------------------------------------------------------------------
def plot_loss_curve():
    h = _load_history(BEST_EXP)
    train_loss = h["train_loss"]
    val_loss   = h["val_loss"]
    best_epoch = h["best_epoch"]

    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, label="Train loss", color="#1f77b4")
    ax.plot(epochs, val_loss,   label="Val loss",   color="#ff7f0e")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.8,
               label=f"Best epoch ({best_epoch})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted MSE loss")
    ax.set_title(f"Training history — {BEST_LABEL}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_1_loss_curve.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 2 — Hs RMSE vs lead time
# ---------------------------------------------------------------------------
def plot_hs_rmse():
    fig, ax = plt.subplots(figsize=(9, 4))
    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        df = _load_lead(exp)
        ax.plot(df["lead_h"], df["hs_rmse"], label=label, color=color)
    ax.axhline(PERSIST_HS_RMSE, color="black", linestyle="--",
               linewidth=1.2, label=f"Persistence ({PERSIST_HS_RMSE:.3f} m)")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Hs RMSE (m)")
    ax.set_title("Hs RMSE vs forecast lead time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_2_hs_rmse_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 3 — Tp RMSE vs lead time
# ---------------------------------------------------------------------------
def plot_tp_rmse():
    fig, ax = plt.subplots(figsize=(9, 4))
    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        df = _load_lead(exp)
        ax.plot(df["lead_h"], df["tp_rmse"], label=label, color=color)
    ax.axhline(PERSIST_TP_RMSE, color="black", linestyle="--",
               linewidth=1.2, label=f"Persistence ({PERSIST_TP_RMSE:.3f} s)")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Tp RMSE (s)")
    ax.set_title("Tp RMSE vs forecast lead time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_3_tp_rmse_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Plot 4 — Direction MAE vs lead time
# ---------------------------------------------------------------------------
def plot_dir_mae():
    fig, ax = plt.subplots(figsize=(9, 4))
    for exp, label, color in zip(EXPERIMENTS, LABELS, COLORS):
        df = _load_lead(exp)
        ax.plot(df["lead_h"], df["dir_mae"], label=label, color=color)
    ax.axhline(PERSIST_DIR_MAE, color="black", linestyle="--",
               linewidth=1.2, label=f"Persistence ({PERSIST_DIR_MAE:.1f}°)")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Direction MAE (°, circular)")
    ax.set_title("Direction MAE vs forecast lead time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_4_dir_mae_lead.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Plots 5 & 6 — Example 48-hour forecast vs truth
# ---------------------------------------------------------------------------
def plot_example_forecasts(sample_idx: int = 500):
    preds, targets = _load_predictions(BEST_EXP)
    # preds/targets shape: (N, 48, 4)  — Hs[0], Tp[1], sin[2], cos[3]
    n = preds.shape[0]
    idx = min(sample_idx, n - 1)

    leads = np.arange(1, 49)
    hs_pred = preds[idx, :, 0]
    hs_true = targets[idx, :, 0]
    tp_pred = preds[idx, :, 1]
    tp_true = targets[idx, :, 1]

    # Plot 5 — Hs
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(leads, hs_true, label="Truth",      color="black",   linewidth=1.5)
    ax.plot(leads, hs_pred, label=BEST_LABEL,   color="#d62728", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Hs (m)")
    ax.set_title(f"Example 48-h Hs forecast — {BEST_LABEL} (test sample {idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_5_example_hs.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)

    # Plot 6 — Tp
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(leads, tp_true, label="Truth",      color="black",   linewidth=1.5)
    ax.plot(leads, tp_pred, label=BEST_LABEL,   color="#d62728", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Forecast lead time (h)")
    ax.set_ylabel("Tp (s)")
    ax.set_title(f"Example 48-h Tp forecast — {BEST_LABEL} (test sample {idx})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "M42_6_example_tp.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating Module 4.2 diagnostic plots...", flush=True)
    plot_loss_curve()
    plot_hs_rmse()
    plot_tp_rmse()
    plot_dir_mae()
    plot_example_forecasts(sample_idx=500)
    print(f"\nAll plots saved to: {OUT_DIR}/", flush=True)
