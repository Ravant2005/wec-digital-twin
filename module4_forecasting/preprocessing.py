"""
preprocessing.py — Wave forecasting data scaler (Module 4.1).

============================================================
DESIGN PRINCIPLES
============================================================

1. Normalization statistics are computed ONLY from training data.
   Validation and test data are transformed using training statistics.

2. The scaler is fitted once via fit() and must not be re-fitted on
   validation or test data.

3. Direction sin/cos components are in [-1, 1] by construction and are
   NOT standardized (standardizing them would distort the circular
   representation).

4. Validity mask columns (Hs_valid, Tp_valid, direction_valid) are
   binary {0, 1} and are NOT standardized.

5. Only Hs and Tp are standardized (z-score: subtract mean, divide by std).

6. The scaler is serializable via its fitted_stats property.

============================================================
FEATURE INDICES (from dataset.FEATURE_NAMES)
============================================================

0  Hs              → standardized
1  Tp              → standardized
2  sin_direction   → NOT standardized (already in [-1,1])
3  cos_direction   → NOT standardized (already in [-1,1])
4  Hs_valid        → NOT standardized (binary)
5  Tp_valid        → NOT standardized (binary)
6  direction_valid → NOT standardized (binary)

============================================================
TARGET INDICES (from dataset.TARGET_NAMES)
============================================================

0  Hs              → standardized (same stats as feature Hs)
1  Tp              → standardized (same stats as feature Tp)
2  sin_direction   → NOT standardized
3  cos_direction   → NOT standardized
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from module4_forecasting.dataset import FEATURE_NAMES, TARGET_NAMES, N_FEATURES, N_TARGETS

# Indices of features/targets that are standardized
_SCALE_FEATURE_IDX = [0, 1]   # Hs, Tp in features
_SCALE_TARGET_IDX  = [0, 1]   # Hs, Tp in targets


@dataclass(frozen=True)
class ScalerStats:
    """Fitted normalization statistics."""
    mean: np.ndarray   # shape (N_FEATURES,), NaN for unscaled features
    std: np.ndarray    # shape (N_FEATURES,), NaN for unscaled features


class WaveScaler:
    """
    Z-score scaler for wave forecasting features and targets.

    Standardizes Hs and Tp only.  Direction sin/cos and validity masks
    are passed through unchanged.

    Usage
    -----
    scaler = WaveScaler()
    scaler.fit(X_train)          # X_train shape: (n, L, N_FEATURES)
    X_scaled = scaler.transform(X_train)
    X_orig   = scaler.inverse_transform(X_scaled)

    For targets:
    y_scaled = scaler.transform_target(y_train)
    y_orig   = scaler.inverse_transform_target(y_scaled)
    """

    def __init__(self) -> None:
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "WaveScaler":
        """
        Fit scaler on training features.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, input_length, N_FEATURES) or (n_samples, N_FEATURES).
            Must be training data ONLY.  NaN values are ignored (nanmean/nanstd).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            # Flatten samples × timesteps for statistics
            X_flat = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            X_flat = X
        else:
            raise ValueError(f"X must be 2-D or 3-D, got {X.ndim}-D")

        self._mean = np.zeros(N_FEATURES)
        self._std  = np.ones(N_FEATURES)

        for idx in _SCALE_FEATURE_IDX:
            col = X_flat[:, idx]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) == 0:
                raise ValueError(f"Feature {FEATURE_NAMES[idx]} has no valid values for fitting")
            self._mean[idx] = float(np.mean(col_valid))
            s = float(np.std(col_valid))
            self._std[idx] = s if s > 1e-12 else 1.0

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("WaveScaler has not been fitted.  Call fit() first.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to features.

        Parameters
        ----------
        X : np.ndarray
            Shape (..., N_FEATURES).

        Returns
        -------
        np.ndarray
            Same shape as X.  NaN values are preserved.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        for idx in _SCALE_FEATURE_IDX:
            X[..., idx] = (X[..., idx] - self._mean[idx]) / self._std[idx]
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse z-score normalization for features.

        Parameters
        ----------
        X : np.ndarray
            Shape (..., N_FEATURES).

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        for idx in _SCALE_FEATURE_IDX:
            X[..., idx] = X[..., idx] * self._std[idx] + self._mean[idx]
        return X

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to targets.

        Parameters
        ----------
        y : np.ndarray
            Shape (..., N_TARGETS).

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float).copy()
        for t_idx, f_idx in zip(_SCALE_TARGET_IDX, _SCALE_FEATURE_IDX):
            y[..., t_idx] = (y[..., t_idx] - self._mean[f_idx]) / self._std[f_idx]
        return y

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Reverse z-score normalization for targets.

        Parameters
        ----------
        y : np.ndarray
            Shape (..., N_TARGETS).

        Returns
        -------
        np.ndarray
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float).copy()
        for t_idx, f_idx in zip(_SCALE_TARGET_IDX, _SCALE_FEATURE_IDX):
            y[..., t_idx] = y[..., t_idx] * self._std[f_idx] + self._mean[f_idx]
        return y

    @property
    def fitted_stats(self) -> Dict[str, np.ndarray]:
        """Return fitted mean and std arrays (for inspection/serialization)."""
        self._check_fitted()
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    @property
    def is_fitted(self) -> bool:
        return self._fitted
