"""
evaluation.py — Test-set evaluation for Tier-1 wave forecasters (Module 4.2).

Responsibilities
----------------
- Batch inference on test set (no future targets used)
- Inverse transform predictions to physical units
- Per-lead-time metrics (Hs MAE/RMSE, Tp MAE/RMSE, direction circular MAE/RMSE)
- Aggregate metrics across all 48 horizons
- Baseline comparison (persistence, climatology)
- Forecast error distribution extraction (required for Module 5)
- Saving all results to CSV/JSON

Scientific guarantees
---------------------
- Predictions are generated from the frozen best checkpoint only
- No test data was used during training or early stopping
- Direction errors are always circular
- All metrics reported in physical units (m, s, degrees)
- Empirical error distributions are preserved (not fitted to Gaussian)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from module4_forecasting.dataset_v2 import ForecastDatasetV2, WaveScalerV2
from module4_forecasting.metrics import (
    ForecastMetrics,
    HorizonMetrics,
    circular_error,
    evaluate_by_horizon,
    evaluate_forecast,
    skill_score,
)
from module4_forecasting.models import _WaveForecasterBase
from module4_forecasting.training import _TorchDataset, get_device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model:   _WaveForecasterBase,
    ds:      ForecastDatasetV2,
    scaler:  WaveScalerV2,
    batch_size: int = 256,
    device:  Optional[torch.device] = None,
) -> np.ndarray:
    """
    Run inference on a dataset and return predictions in physical units.

    Parameters
    ----------
    model : trained forecaster (eval mode expected)
    ds : ForecastDatasetV2
    scaler : WaveScalerV2 fitted on training data
    batch_size : int
    device : torch.device or None

    Returns
    -------
    np.ndarray
        Shape (n_samples, forecast_horizon, 4).
        Channels: Hs [m], Tp [s], sin_direction, cos_direction.
        Hs and Tp are in physical units (inverse-transformed).
        sin/cos are raw network outputs (not normalised to unit circle).
    """
    if device is None:
        device = get_device()

    print(f"  Building inference dataset ({len(ds):,} windows)...", end="", flush=True)
    torch_ds = _TorchDataset(ds, scaler)
    print(" done.", flush=True)
    loader   = DataLoader(torch_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    n_batches = len(loader)
    print(f"  Running inference ({n_batches} batches)...", end="", flush=True)
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)           # [B, H, 4] normalised
            all_preds.append(pred.cpu().numpy())
    print(" done.", flush=True)

    preds_norm = np.concatenate(all_preds, axis=0)   # (N, H, 4)

    # Inverse transform Hs (idx 0) and Tp (idx 1) to physical units
    preds_phys = scaler.inverse_transform_target(preds_norm)
    return preds_phys


# ---------------------------------------------------------------------------
# Direction normalisation
# ---------------------------------------------------------------------------

def _normalise_sincos(sin_d: np.ndarray, cos_d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalise predicted sin/cos to unit circle."""
    norm = np.sqrt(sin_d**2 + cos_d**2)
    norm = np.where(norm < 1e-9, 1.0, norm)
    return sin_d / norm, cos_d / norm


def _sincos_to_deg(sin_d: np.ndarray, cos_d: np.ndarray) -> np.ndarray:
    """Convert sin/cos to degrees [0, 360)."""
    return np.degrees(np.arctan2(sin_d, cos_d)) % 360.0


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model:       _WaveForecasterBase,
    test_ds:     ForecastDatasetV2,
    scaler:      WaveScalerV2,
    results_dir: str,
    batch_size:  int = 256,
    device:      Optional[torch.device] = None,
) -> Tuple[ForecastMetrics, HorizonMetrics, np.ndarray]:
    """
    Run full test-set evaluation and save all results.

    Parameters
    ----------
    model : trained forecaster
    test_ds : ForecastDatasetV2 (test split)
    scaler : WaveScalerV2
    results_dir : str
    batch_size : int
    device : torch.device or None

    Returns
    -------
    agg_metrics : ForecastMetrics  (aggregate over all leads)
    horizon_metrics : HorizonMetrics  (per-lead)
    predictions : np.ndarray  (N, H, 4) in physical units
    """
    os.makedirs(results_dir, exist_ok=True)

    # --- Inference ---
    preds = predict(model, test_ds, scaler, batch_size=batch_size, device=device)
    targets = test_ds.get_all_targets()   # (N, H, 4) physical units

    # --- Aggregate metrics ---
    agg = evaluate_forecast(preds, targets)
    print(f"  Hs  MAE={agg.hs_mae:.4f} m   RMSE={agg.hs_rmse:.4f} m   bias={agg.hs_bias:+.4f} m")
    print(f"  Tp  MAE={agg.tp_mae:.4f} s   RMSE={agg.tp_rmse:.4f} s   bias={agg.tp_bias:+.4f} s")
    print(f"  Dir MAE={agg.dir_mae:.2f}°  RMSE={agg.dir_rmse:.2f}°")

    # --- Per-lead metrics ---
    hm = evaluate_by_horizon(preds, targets)

    # --- Save aggregate metrics ---
    agg_dict = {
        "hs_mae":  agg.hs_mae,  "hs_rmse":  agg.hs_rmse,  "hs_bias":  agg.hs_bias,
        "tp_mae":  agg.tp_mae,  "tp_rmse":  agg.tp_rmse,  "tp_bias":  agg.tp_bias,
        "dir_mae": agg.dir_mae, "dir_rmse": agg.dir_rmse,
        "n_samples": agg.n_samples, "forecast_horizon": agg.forecast_horizon,
    }
    with open(os.path.join(results_dir, "test_metrics.json"), "w") as f:
        json.dump(agg_dict, f, indent=2)

    # --- Save per-lead metrics ---
    lead_df = pd.DataFrame({
        "lead_h":   hm.horizons,
        "hs_mae":   hm.hs_mae,
        "hs_rmse":  hm.hs_rmse,
        "tp_mae":   hm.tp_mae,
        "tp_rmse":  hm.tp_rmse,
        "dir_mae":  hm.dir_mae,
        "dir_rmse": hm.dir_rmse,
    })
    lead_df.to_csv(os.path.join(results_dir, "per_lead_metrics.csv"), index=False)

    # --- Save predictions (compressed) ---
    np.savez_compressed(
        os.path.join(results_dir, "predictions.npz"),
        predictions=preds.astype(np.float32),
        targets=targets.astype(np.float32),
    )

    return agg, hm, preds


# ---------------------------------------------------------------------------
# Forecast error distribution (required for Module 5)
# ---------------------------------------------------------------------------

def extract_error_distribution(
    predictions: np.ndarray,
    targets:     np.ndarray,
    test_ds:     ForecastDatasetV2,
    results_dir: str,
) -> pd.DataFrame:
    """
    Extract empirical forecast errors for every lead time.

    Stores the raw error distribution for later use in Module 5
    (realistic forecast noise injection).  Does NOT fit a Gaussian.

    Parameters
    ----------
    predictions : np.ndarray  (N, H, 4) physical units
    targets     : np.ndarray  (N, H, 4) physical units
    test_ds     : ForecastDatasetV2 (for origin timestamps)
    results_dir : str

    Returns
    -------
    pd.DataFrame with columns:
        lead_h, sample_idx, origin_idx,
        Hs_pred, Hs_true, Hs_error,
        Tp_pred, Tp_true, Tp_error,
        dir_pred_deg, dir_true_deg, dir_error_deg
    """
    os.makedirs(results_dir, exist_ok=True)

    N, H, _ = predictions.shape
    origins  = test_ds.get_all_origins()

    rows = []
    for h in range(H):
        hs_pred = predictions[:, h, 0]
        hs_true = targets[:, h, 0]
        tp_pred = predictions[:, h, 1]
        tp_true = targets[:, h, 1]

        sin_p, cos_p = _normalise_sincos(predictions[:, h, 2], predictions[:, h, 3])
        sin_t, cos_t = targets[:, h, 2], targets[:, h, 3]
        dir_pred = _sincos_to_deg(sin_p, cos_p)
        dir_true = _sincos_to_deg(sin_t, cos_t)
        dir_err  = circular_error(dir_pred, dir_true)

        for i in range(N):
            rows.append({
                "lead_h":       h + 1,
                "sample_idx":   i,
                "origin_idx":   int(origins[i]),
                "Hs_pred":      float(hs_pred[i]),
                "Hs_true":      float(hs_true[i]),
                "Hs_error":     float(hs_pred[i] - hs_true[i]),
                "Tp_pred":      float(tp_pred[i]),
                "Tp_true":      float(tp_true[i]),
                "Tp_error":     float(tp_pred[i] - tp_true[i]),
                "dir_pred_deg": float(dir_pred[i]),
                "dir_true_deg": float(dir_true[i]),
                "dir_error_deg": float(dir_err[i]),
            })

    errors_df = pd.DataFrame(rows)
    errors_df.to_csv(
        os.path.join(results_dir, "forecast_errors.csv.gz"),
        index=False, compression="gzip",
    )

    # --- Per-lead summary statistics ---
    summary_rows = []
    for h in range(1, H + 1):
        sub = errors_df[errors_df["lead_h"] == h]
        for var, col in [("Hs", "Hs_error"), ("Tp", "Tp_error"), ("dir", "dir_error_deg")]:
            e = sub[col].dropna().values
            if len(e) == 0:
                continue
            summary_rows.append({
                "lead_h":  h,
                "variable": var,
                "mean":    float(np.mean(e)),
                "std":     float(np.std(e)),
                "median":  float(np.median(e)),
                "mae":     float(np.mean(np.abs(e))),
                "rmse":    float(np.sqrt(np.mean(e**2))),
                "p05":     float(np.percentile(e,  5)),
                "p25":     float(np.percentile(e, 25)),
                "p50":     float(np.percentile(e, 50)),
                "p75":     float(np.percentile(e, 75)),
                "p95":     float(np.percentile(e, 95)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(results_dir, "error_distribution_summary.csv"),
        index=False,
    )

    return errors_df


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------

def compare_baselines(
    model_results:  Dict[str, ForecastMetrics],
    results_dir:    str,
) -> pd.DataFrame:
    """
    Build a comparison table: Persistence, Climatology, LSTM, GRU variants.

    Parameters
    ----------
    model_results : dict mapping experiment name → ForecastMetrics
    results_dir : str

    Returns
    -------
    pd.DataFrame
    """
    rows = []
    for name, m in model_results.items():
        rows.append({
            "Model":    name,
            "Hs_MAE":   round(m.hs_mae,  4),
            "Hs_RMSE":  round(m.hs_rmse, 4),
            "Tp_MAE":   round(m.tp_mae,  4),
            "Tp_RMSE":  round(m.tp_rmse, 4),
            "Dir_MAE":  round(m.dir_mae, 2),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "baseline_comparison.csv"), index=False)
    return df


# ---------------------------------------------------------------------------
# Persistence baseline on v2 test set
# ---------------------------------------------------------------------------

def run_persistence_v2(test_ds: ForecastDatasetV2) -> np.ndarray:
    """
    Compute persistence forecasts for the v2 test set.

    Repeats the last valid observation in the input window for all horizons.
    Returns predictions in physical units (Hs [m], Tp [s], sin_dir, cos_dir).

    Parameters
    ----------
    test_ds : ForecastDatasetV2

    Returns
    -------
    np.ndarray  (N, H, 4)
    """
    H = test_ds.config.forecast_horizon
    N = len(test_ds)
    preds = []
    print(f"  Computing persistence ({N:,} windows)...", end="", flush=True)
    for i in range(N):
        s = test_ds[i]
        valid_rows = np.where(s.input_mask)[0]
        if len(valid_rows) == 0:
            preds.append(np.full((H, 4), np.nan))
            continue
        last = valid_rows[-1]
        row = s.X[last, :4].copy()   # Hs, Tp, sin_dir, cos_dir (unscaled)
        preds.append(np.tile(row, (H, 1)))
    print(" done.", flush=True)
    return np.stack(preds, axis=0)


# ---------------------------------------------------------------------------
# Climatology baseline on v2 datasets
# ---------------------------------------------------------------------------

def run_climatology_v2(
    train_ds: ForecastDatasetV2,
    test_ds:  ForecastDatasetV2,
) -> np.ndarray:
    """
    Compute climatology forecasts using training-set mean.

    Fitted on training targets only.  Returns predictions in physical units.

    Parameters
    ----------
    train_ds : ForecastDatasetV2 (training split)
    test_ds  : ForecastDatasetV2 (test split)

    Returns
    -------
    np.ndarray  (N_test, H, 4)
    """
    print(f"  Computing climatology (fitting on {len(train_ds):,} train windows)...", end="", flush=True)
    targets_train = train_ds.get_all_targets()   # (N_train, H, 4)
    flat = targets_train.reshape(-1, 4)

    clim = np.zeros(4)
    for i in range(4):
        col = flat[:, i]
        col = col[~np.isnan(col)]
        clim[i] = float(np.mean(col))

    print(" done.", flush=True)
    N = len(test_ds)
    H = test_ds.config.forecast_horizon
    row  = clim[np.newaxis, :]           # (1, 4)
    step = np.tile(row, (H, 1))          # (H, 4)
    return np.tile(step[np.newaxis], (N, 1, 1))
