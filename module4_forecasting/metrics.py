"""
metrics.py — Wave forecast evaluation metrics (Module 4.1).

Metrics
-------
Hs, Tp : MAE, RMSE, bias
Direction : circular MAE, circular RMSE (degrees)

Skill score against a reference baseline:
    Skill = 1 - RMSE_model / RMSE_reference

Direction errors are always computed circularly.
Percentage error is NOT reported for direction.

All functions accept arrays of shape (n_samples, forecast_horizon, N_TARGETS)
or (n_samples, forecast_horizon) for a single variable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from module4_forecasting.dataset import TARGET_NAMES, N_TARGETS, sincos_to_direction


# ---------------------------------------------------------------------------
# Circular direction error
# ---------------------------------------------------------------------------

def circular_error(pred_deg: np.ndarray, true_deg: np.ndarray) -> np.ndarray:
    """
    Signed circular error in degrees: pred − true, wrapped to (−180, 180].

    Parameters
    ----------
    pred_deg, true_deg : array_like
        Directions in degrees.

    Returns
    -------
    np.ndarray
        Signed errors in (−180, 180].
    """
    diff = np.asarray(pred_deg, dtype=float) - np.asarray(true_deg, dtype=float)
    err  = diff % 360.0
    return np.where(err > 180.0, err - 360.0, err)


def _sincos_to_deg_safe(sin_d: np.ndarray, cos_d: np.ndarray) -> np.ndarray:
    """Convert sin/cos to degrees, propagating NaN."""
    deg = sincos_to_direction(sin_d, cos_d)
    nan_mask = np.isnan(sin_d) | np.isnan(cos_d)
    deg[nan_mask] = np.nan
    return deg


# ---------------------------------------------------------------------------
# ForecastMetrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class ForecastMetrics:
    """
    Aggregated forecast metrics over all samples and horizons.

    Attributes
    ----------
    hs_mae, hs_rmse, hs_bias : float
    tp_mae, tp_rmse, tp_bias : float
    dir_mae, dir_rmse : float  (degrees, circular)
    n_samples : int
    forecast_horizon : int
    """

    hs_mae:   float = np.nan
    hs_rmse:  float = np.nan
    hs_bias:  float = np.nan
    tp_mae:   float = np.nan
    tp_rmse:  float = np.nan
    tp_bias:  float = np.nan
    dir_mae:  float = np.nan
    dir_rmse: float = np.nan
    n_samples: int = 0
    forecast_horizon: int = 0


@dataclass
class HorizonMetrics:
    """
    Per-horizon metrics.

    Each list has length forecast_horizon.
    """

    hs_mae:   List[float] = field(default_factory=list)
    hs_rmse:  List[float] = field(default_factory=list)
    tp_mae:   List[float] = field(default_factory=list)
    tp_rmse:  List[float] = field(default_factory=list)
    dir_mae:  List[float] = field(default_factory=list)
    dir_rmse: List[float] = field(default_factory=list)
    horizons: List[int]   = field(default_factory=list)


# ---------------------------------------------------------------------------
# evaluate_forecast
# ---------------------------------------------------------------------------

def evaluate_forecast(
    predictions: np.ndarray,
    targets: np.ndarray,
    reference: Optional[np.ndarray] = None,
) -> ForecastMetrics:
    """
    Compute aggregate forecast metrics.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (n_samples, forecast_horizon, N_TARGETS).
        TARGET_NAMES order: Hs, Tp, sin_direction, cos_direction.
    targets : np.ndarray
        Same shape as predictions.  True future values.
    reference : np.ndarray or None
        Reference forecast (e.g. persistence) for skill score.
        Same shape.  Not currently stored in ForecastMetrics but
        available for external skill computation.

    Returns
    -------
    ForecastMetrics
    """
    predictions = np.asarray(predictions, dtype=float)
    targets     = np.asarray(targets,     dtype=float)

    n_samples, H, _ = predictions.shape

    def _mae_rmse_bias(pred, true):
        err = pred - true
        valid = ~(np.isnan(err))
        if not np.any(valid):
            return np.nan, np.nan, np.nan
        e = err[valid]
        return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e**2))), float(np.mean(e))

    # Hs
    hs_mae, hs_rmse, hs_bias = _mae_rmse_bias(
        predictions[..., 0], targets[..., 0]
    )

    # Tp
    tp_mae, tp_rmse, tp_bias = _mae_rmse_bias(
        predictions[..., 1], targets[..., 1]
    )

    # Direction (circular)
    pred_deg = _sincos_to_deg_safe(predictions[..., 2], predictions[..., 3])
    true_deg = _sincos_to_deg_safe(targets[..., 2],     targets[..., 3])
    dir_err  = circular_error(pred_deg, true_deg)
    valid_d  = ~np.isnan(dir_err)
    if np.any(valid_d):
        dir_mae  = float(np.mean(np.abs(dir_err[valid_d])))
        dir_rmse = float(np.sqrt(np.mean(dir_err[valid_d]**2)))
    else:
        dir_mae = dir_rmse = np.nan

    return ForecastMetrics(
        hs_mae=hs_mae, hs_rmse=hs_rmse, hs_bias=hs_bias,
        tp_mae=tp_mae, tp_rmse=tp_rmse, tp_bias=tp_bias,
        dir_mae=dir_mae, dir_rmse=dir_rmse,
        n_samples=n_samples, forecast_horizon=H,
    )


# ---------------------------------------------------------------------------
# evaluate_by_horizon
# ---------------------------------------------------------------------------

def evaluate_by_horizon(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> HorizonMetrics:
    """
    Compute metrics separately for each forecast horizon step.

    Parameters
    ----------
    predictions : np.ndarray
        Shape (n_samples, forecast_horizon, N_TARGETS).
    targets : np.ndarray
        Same shape.

    Returns
    -------
    HorizonMetrics
    """
    predictions = np.asarray(predictions, dtype=float)
    targets     = np.asarray(targets,     dtype=float)

    H = predictions.shape[1]
    hm = HorizonMetrics(horizons=list(range(1, H + 1)))

    for h in range(H):
        pred_h = predictions[:, h, :]
        true_h = targets[:, h, :]

        # Hs
        err_hs = pred_h[:, 0] - true_h[:, 0]
        v = ~np.isnan(err_hs)
        hm.hs_mae.append(float(np.mean(np.abs(err_hs[v]))) if v.any() else np.nan)
        hm.hs_rmse.append(float(np.sqrt(np.mean(err_hs[v]**2))) if v.any() else np.nan)

        # Tp
        err_tp = pred_h[:, 1] - true_h[:, 1]
        v = ~np.isnan(err_tp)
        hm.tp_mae.append(float(np.mean(np.abs(err_tp[v]))) if v.any() else np.nan)
        hm.tp_rmse.append(float(np.sqrt(np.mean(err_tp[v]**2))) if v.any() else np.nan)

        # Direction (circular)
        pred_deg = _sincos_to_deg_safe(pred_h[:, 2], pred_h[:, 3])
        true_deg = _sincos_to_deg_safe(true_h[:, 2], true_h[:, 3])
        d_err    = circular_error(pred_deg, true_deg)
        v = ~np.isnan(d_err)
        hm.dir_mae.append(float(np.mean(np.abs(d_err[v]))) if v.any() else np.nan)
        hm.dir_rmse.append(float(np.sqrt(np.mean(d_err[v]**2))) if v.any() else np.nan)

    return hm


# ---------------------------------------------------------------------------
# skill_score
# ---------------------------------------------------------------------------

def skill_score(model_rmse: float, reference_rmse: float) -> float:
    """
    Skill score: 1 - model_rmse / reference_rmse.

    Positive = model better than reference.
    Zero     = same as reference.
    Negative = model worse than reference.

    Parameters
    ----------
    model_rmse : float
    reference_rmse : float

    Returns
    -------
    float
    """
    if reference_rmse == 0.0 or np.isnan(reference_rmse):
        return np.nan
    return 1.0 - model_rmse / reference_rmse
