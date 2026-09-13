"""
baselines.py — Non-ML forecasting baselines (Module 4.1).

============================================================
PERSISTENCE BASELINE
============================================================

At forecast origin t, predict the most recent valid observation
for all future horizons:

    Hs_hat(t+k)        = Hs_obs(t*)
    Tp_hat(t+k)        = Tp_obs(t*)
    direction_hat(t+k) = direction_obs(t*)

where t* is the most recent valid observation at or before t,
found by searching backward up to max_lookback steps.

If no valid observation is found within max_lookback, the forecast
is marked invalid (NaN).

The persistence baseline NEVER reads future truth.

============================================================
CLIMATOLOGY BASELINE
============================================================

Predict the training-set mean (or median) for all horizons:

    Hs_hat(t+k)        = mean(Hs_true_train)
    Tp_hat(t+k)        = mean(Tp_true_train)
    direction_hat(t+k) = circular_mean(direction_true_train)

Climatology is computed ONLY from training data.
Validation/test data do NOT affect the climatology.

Circular mean direction is computed via atan2(mean(sin), mean(cos)).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from module4_forecasting.dataset import (
    ForecastDataset, ForecastSample, TARGET_NAMES, N_TARGETS,
    direction_to_sincos, sincos_to_direction,
)


# ---------------------------------------------------------------------------
# PersistenceForecaster
# ---------------------------------------------------------------------------

class PersistenceForecaster:
    """
    Naive persistence baseline: repeat the most recent valid observation.

    Parameters
    ----------
    max_lookback : int
        Maximum number of steps to search backward for a valid observation.
        Default 24.  If no valid observation is found, forecast is NaN.
    """

    def __init__(self, max_lookback: int = 24) -> None:
        if max_lookback < 1:
            raise ValueError(f"max_lookback must be >= 1, got {max_lookback}")
        self.max_lookback = max_lookback

    def predict_sample(self, sample: ForecastSample, obs_Hs: np.ndarray,
                       obs_Tp: np.ndarray, obs_dir: np.ndarray,
                       valid_H: np.ndarray, valid_T: np.ndarray,
                       valid_D: np.ndarray) -> np.ndarray:
        """
        Produce a persistence forecast for one sample.

        Searches backward from the origin step within the input window
        for the most recent valid observation.

        Parameters
        ----------
        sample : ForecastSample
        obs_Hs, obs_Tp, obs_dir : np.ndarray
            Full observation arrays (length = total time series).
        valid_H, valid_T, valid_D : np.ndarray
            Full validity arrays.

        Returns
        -------
        np.ndarray
            Shape (forecast_horizon, N_TARGETS).  NaN if no valid obs found.
        """
        H = sample.target.shape[0]
        origin = sample.origin_index

        # Search backward for most recent valid observation
        hs_val = tp_val = dir_val = np.nan
        found = False
        for lag in range(self.max_lookback):
            idx = origin - lag
            if idx < 0:
                break
            if valid_H[idx] and valid_T[idx] and valid_D[idx]:
                hs_val  = obs_Hs[idx]
                tp_val  = obs_Tp[idx]
                dir_val = obs_dir[idx]
                found   = True
                break

        if not found:
            return np.full((H, N_TARGETS), np.nan)

        sin_d, cos_d = direction_to_sincos(np.array([dir_val]))
        row = np.array([hs_val, tp_val, float(sin_d[0]), float(cos_d[0])])
        return np.tile(row, (H, 1))

    def predict_dataset(self, ds: ForecastDataset) -> np.ndarray:
        """
        Produce persistence forecasts for all samples in a dataset.

        Parameters
        ----------
        ds : ForecastDataset

        Returns
        -------
        np.ndarray
            Shape (n_samples, forecast_horizon, N_TARGETS).
        """
        obs = ds.obs
        preds = []
        for i in range(len(ds)):
            sample = ds[i]
            pred = self.predict_sample(
                sample,
                obs.Hs_obs, obs.Tp_obs, obs.direction_obs,
                obs.valid_Hs, obs.valid_Tp, obs.valid_direction,
            )
            preds.append(pred)
        return np.stack(preds, axis=0)


# ---------------------------------------------------------------------------
# ClimatologyForecaster
# ---------------------------------------------------------------------------

class ClimatologyForecaster:
    """
    Training-set climatology baseline.

    Predicts the training-set mean for all horizons.
    Fitted ONLY on training data.

    Parameters
    ----------
    use_median : bool
        If True, use median instead of mean for Hs and Tp.  Default False.
    """

    def __init__(self, use_median: bool = False) -> None:
        self.use_median = use_median
        self._clim: Optional[np.ndarray] = None   # shape (N_TARGETS,)

    def fit(self, train_ds: ForecastDataset) -> "ClimatologyForecaster":
        """
        Compute climatology from training targets.

        Parameters
        ----------
        train_ds : ForecastDataset
            Must be the training split.

        Returns
        -------
        self
        """
        targets = train_ds.get_all_targets()   # (n_train, H, N_TARGETS)
        # Flatten over samples and horizons
        flat = targets.reshape(-1, N_TARGETS)

        clim = np.zeros(N_TARGETS)

        # Hs (index 0)
        hs_vals = flat[:, 0]
        hs_vals = hs_vals[~np.isnan(hs_vals)]
        clim[0] = float(np.median(hs_vals) if self.use_median else np.mean(hs_vals))

        # Tp (index 1)
        tp_vals = flat[:, 1]
        tp_vals = tp_vals[~np.isnan(tp_vals)]
        clim[1] = float(np.median(tp_vals) if self.use_median else np.mean(tp_vals))

        # Direction: circular mean via mean(sin), mean(cos)
        sin_vals = flat[:, 2]
        cos_vals = flat[:, 3]
        valid = ~(np.isnan(sin_vals) | np.isnan(cos_vals))
        clim[2] = float(np.mean(sin_vals[valid]))
        clim[3] = float(np.mean(cos_vals[valid]))

        self._clim = clim
        return self

    def predict(self, n_samples: int, forecast_horizon: int) -> np.ndarray:
        """
        Produce climatology forecasts.

        Parameters
        ----------
        n_samples : int
        forecast_horizon : int

        Returns
        -------
        np.ndarray
            Shape (n_samples, forecast_horizon, N_TARGETS).
        """
        if self._clim is None:
            raise RuntimeError("ClimatologyForecaster has not been fitted.  Call fit() first.")
        row = self._clim[np.newaxis, :]                    # (1, N_TARGETS)
        step = np.tile(row, (forecast_horizon, 1))         # (H, N_TARGETS)
        return np.tile(step[np.newaxis, :, :], (n_samples, 1, 1))

    def predict_dataset(self, ds: ForecastDataset) -> np.ndarray:
        """Predict for all samples in a dataset."""
        return self.predict(len(ds), ds.config.forecast_horizon)

    @property
    def climatology(self) -> Optional[np.ndarray]:
        """Fitted climatology vector, shape (N_TARGETS,)."""
        return self._clim
