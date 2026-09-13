"""
dataset_v2.py — Tier-1 wave forecasting dataset (Module 4.2 pipeline).

This is the v2 data pipeline built on the full 2010–2025 ERA5 archive.
It sits ALONGSIDE the frozen v1 dataset.py and does NOT modify it.

============================================================
FORECASTING TASK
============================================================

Given the previous L hours of observed wave conditions (and optionally
ERA5 wind), predict the next H hours of TRUE wave state.

Default Tier-1 configuration:
    input_length     = 48 hours  (2 days of context)
    forecast_horizon = 48 hours  (2 days ahead)

============================================================
FEATURE MODES
============================================================

wave_only (N_FEATURES_WAVE = 7):
    Index  Name
      0    Hs              observed Hs [m]  (NaN if missing)
      1    Tp              observed Tp [s]  (NaN if missing)
      2    sin_direction   sin(direction_obs * pi/180)
      3    cos_direction   cos(direction_obs * pi/180)
      4    Hs_valid        1.0 if Hs_obs is valid, else 0.0
      5    Tp_valid        1.0 if Tp_obs is valid, else 0.0
      6    direction_valid 1.0 if direction_obs is valid, else 0.0

wave_wind (N_FEATURES_WIND = 11):
    Indices 0–6 as above, plus:
      7    u10             ERA5 10 m zonal wind [m/s]
      8    v10             ERA5 10 m meridional wind [m/s]
      9    u10_valid       1.0 if u10 is finite, else 0.0
     10    v10_valid       1.0 if v10 is finite, else 0.0

Wind is ERA5 reanalysis data, NOT passed through the sensor model.
Wind validity flags mark NaN/non-finite values only.

============================================================
TARGET ORDER (TARGET_NAMES_V2, N_TARGETS_V2 = 4)
============================================================

    Index  Name
      0    Hs              true future Hs [m]
      1    Tp              true future Tp [s]
      2    sin_direction   sin(true future direction * pi/180)
      3    cos_direction   cos(true future direction * pi/180)

============================================================
YEAR-BASED CHRONOLOGICAL SPLIT
============================================================

The time series is split on calendar-year boundaries BEFORE generating
sliding windows.  No window ever crosses a split boundary.

    TRAIN : 2010-01-01 00:00 through 2021-12-31 23:00
    VAL   : 2022-01-01 00:00 through 2023-12-31 23:00
    TEST  : 2024-01-01 00:00 through 2025-12-31 23:00

This is a hard year-boundary split, not a fraction-based split.
Random shuffling of the time series is FORBIDDEN.

============================================================
MISSING DATA POLICY
============================================================

"mask_only" (default):
    Missing wave observations remain NaN in X.
    The valid_* columns carry the mask.
    Downstream models must handle NaN explicitly.

"train_median":
    Missing values in X are replaced by the training-set median,
    computed ONLY from training data.  Valid masks are retained.
    Must NEVER use validation/test statistics.

Wind NaN values are flagged via u10_valid/v10_valid but are NOT
imputed by the train_median policy (wind is not a sensor observation).

============================================================
SCALING
============================================================

WaveScalerV2 standardizes continuous variables using training statistics:
    Hs, Tp  : z-score (subtract mean, divide by std)
    u10, v10: z-score (subtract mean, divide by std)
    sin/cos direction : NOT standardized (already in [-1, 1])
    validity masks    : NOT standardized (binary)

Scaler is fitted ONLY on training data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from module3_sensor.sensor import SensorObservation, SensorParameters, WaveSensor
from module4_forecasting.dataset import direction_to_sincos, sincos_to_direction

# ---------------------------------------------------------------------------
# Feature / target schema
# ---------------------------------------------------------------------------

FEATURE_NAMES_WAVE: List[str] = [
    "Hs",
    "Tp",
    "sin_direction",
    "cos_direction",
    "Hs_valid",
    "Tp_valid",
    "direction_valid",
]

FEATURE_NAMES_WIND: List[str] = FEATURE_NAMES_WAVE + [
    "u10",
    "v10",
    "u10_valid",
    "v10_valid",
]

TARGET_NAMES_V2: List[str] = [
    "Hs",
    "Tp",
    "sin_direction",
    "cos_direction",
]

N_FEATURES_WAVE = len(FEATURE_NAMES_WAVE)   # 7
N_FEATURES_WIND = len(FEATURE_NAMES_WIND)   # 11
N_TARGETS_V2    = len(TARGET_NAMES_V2)      # 4

# Indices of features that are z-scored
_SCALE_WAVE_IDX = [0, 1]        # Hs, Tp
_SCALE_WIND_IDX = [0, 1, 7, 8]  # Hs, Tp, u10, v10  (wave_wind mode)


# ---------------------------------------------------------------------------
# Year-based split
# ---------------------------------------------------------------------------

TRAIN_END_YEAR = 2021   # inclusive
VAL_END_YEAR   = 2023   # inclusive
# TEST: 2024–2025


@dataclass(frozen=True)
class YearSplit:
    """
    Calendar-year split indices for a time series with a DatetimeIndex.

    Attributes
    ----------
    train_end : int
        Exclusive end index of training portion.
        All timestamps < 2022-01-01 00:00 UTC.
    val_end : int
        Exclusive end index of validation portion.
        All timestamps < 2024-01-01 00:00 UTC.
    n_total : int
        Total length of the time series.
    train_end_date : pd.Timestamp
    val_end_date   : pd.Timestamp
    """

    train_end:      int
    val_end:        int
    n_total:        int
    train_end_date: pd.Timestamp
    val_end_date:   pd.Timestamp

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def val_slice(self) -> slice:
        return slice(self.train_end, self.val_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.val_end, self.n_total)

    @property
    def n_train(self) -> int:
        return self.train_end

    @property
    def n_val(self) -> int:
        return self.val_end - self.train_end

    @property
    def n_test(self) -> int:
        return self.n_total - self.val_end


def make_year_split(
    times: np.ndarray,
    train_end_year: int = TRAIN_END_YEAR,
    val_end_year:   int = VAL_END_YEAR,
) -> YearSplit:
    """
    Compute year-boundary split indices for a time series.

    The split is on calendar-year boundaries:
        TRAIN : times < {train_end_year+1}-01-01
        VAL   : {train_end_year+1}-01-01 <= times < {val_end_year+1}-01-01
        TEST  : times >= {val_end_year+1}-01-01

    The split is computed BEFORE window generation.
    No window is allowed to cross a split boundary.

    Parameters
    ----------
    times : array_like
        Timestamps (convertible to pd.DatetimeIndex).
        Must be sorted and monotonically increasing.
    train_end_year : int
        Last year (inclusive) of the training set.  Default 2021.
    val_end_year : int
        Last year (inclusive) of the validation set.  Default 2023.

    Returns
    -------
    YearSplit
    """
    t = pd.to_datetime(times)
    n = len(t)

    train_boundary = pd.Timestamp(f"{train_end_year + 1}-01-01", tz=None)
    val_boundary   = pd.Timestamp(f"{val_end_year   + 1}-01-01", tz=None)

    # Handle timezone-aware timestamps
    if t.tz is not None:
        train_boundary = train_boundary.tz_localize(t.tz)
        val_boundary   = val_boundary.tz_localize(t.tz)

    # Find first index >= boundary (exclusive end of train/val)
    train_end = int(np.searchsorted(t, train_boundary, side="left"))
    val_end   = int(np.searchsorted(t, val_boundary,   side="left"))

    # Clamp to valid range
    train_end = max(0, min(train_end, n))
    val_end   = max(train_end, min(val_end, n))

    return YearSplit(
        train_end=train_end,
        val_end=val_end,
        n_total=n,
        train_end_date=train_boundary,
        val_end_date=val_boundary,
    )


# ---------------------------------------------------------------------------
# ForecastConfigV2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastConfigV2:
    """
    Configuration for the v2 wave forecasting dataset.

    Parameters
    ----------
    input_length : int
        Number of past hourly steps used as input.  Default 48 (2 days).
    forecast_horizon : int
        Number of future hourly steps to predict.  Default 48 (2 days).
    feature_mode : str
        "wave_only" or "wave_wind".
    missing_policy : str
        "mask_only" or "train_median".
    """

    input_length:     int = 48
    forecast_horizon: int = 48
    feature_mode:     str = "wave_only"
    missing_policy:   str = "mask_only"

    def __post_init__(self) -> None:
        if self.input_length < 1:
            raise ValueError(f"input_length must be >= 1, got {self.input_length}")
        if self.forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {self.forecast_horizon}")
        if self.feature_mode not in ("wave_only", "wave_wind"):
            raise ValueError(
                f"feature_mode must be 'wave_only' or 'wave_wind', "
                f"got '{self.feature_mode}'"
            )
        if self.missing_policy not in ("mask_only", "train_median"):
            raise ValueError(
                f"missing_policy must be 'mask_only' or 'train_median', "
                f"got '{self.missing_policy}'"
            )

    @property
    def n_features(self) -> int:
        return N_FEATURES_WIND if self.feature_mode == "wave_wind" else N_FEATURES_WAVE

    @property
    def feature_names(self) -> List[str]:
        return FEATURE_NAMES_WIND if self.feature_mode == "wave_wind" else FEATURE_NAMES_WAVE

    @property
    def window_length(self) -> int:
        return self.input_length + self.forecast_horizon


# ---------------------------------------------------------------------------
# ForecastSampleV2
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastSampleV2:
    """
    A single supervised forecasting sample (v2).

    Attributes
    ----------
    X : np.ndarray
        Input feature matrix, shape (input_length, n_features).
        May contain NaN where observations are missing.
    target : np.ndarray
        Target matrix, shape (forecast_horizon, N_TARGETS_V2).
        Contains TRUE future wave state.  NEVER used as model input.
    origin_index : int
        Index in the original time series of the last input step (t).
    input_mask : np.ndarray
        Boolean mask, shape (input_length,).
        True where ALL wave observations (Hs, Tp, direction) are valid.
    """

    X:            np.ndarray
    target:       np.ndarray
    origin_index: int
    input_mask:   np.ndarray


# ---------------------------------------------------------------------------
# ForecastDatasetV2
# ---------------------------------------------------------------------------

class ForecastDatasetV2:
    """
    Sliding-window supervised forecasting dataset (v2).

    Operates on a combined wave+wind DataFrame produced by the ERA5 archive
    loader, after passing wave observations through the Module 3 sensor model.

    Parameters
    ----------
    obs : SensorObservation
        Wave sensor output from Module 3.  Provides observed Hs/Tp/direction
        and validity masks.  Also carries true values for target construction.
    config : ForecastConfigV2
        Dataset configuration.
    split : YearSplit
        Year-boundary split indices.
    subset : str
        "train", "val", or "test".
    u10 : np.ndarray or None
        ERA5 10 m zonal wind [m/s], shape (N,).  Required for wave_wind mode.
    v10 : np.ndarray or None
        ERA5 10 m meridional wind [m/s], shape (N,).  Required for wave_wind mode.
    train_medians : np.ndarray or None
        Training-set medians for "train_median" missing policy.
        Shape (n_features,).  Must be provided for val/test subsets.
    """

    def __init__(
        self,
        obs:           SensorObservation,
        config:        ForecastConfigV2,
        split:         YearSplit,
        subset:        str,
        u10:           Optional[np.ndarray] = None,
        v10:           Optional[np.ndarray] = None,
        train_medians: Optional[np.ndarray] = None,
    ) -> None:
        if subset not in ("train", "val", "test"):
            raise ValueError(f"subset must be 'train', 'val', or 'test', got '{subset}'")

        if config.feature_mode == "wave_wind":
            if u10 is None or v10 is None:
                raise ValueError(
                    "u10 and v10 must be provided for feature_mode='wave_wind'."
                )
            if len(u10) != len(obs.time) or len(v10) != len(obs.time):
                raise ValueError(
                    f"u10/v10 length ({len(u10)}) must match obs length ({len(obs.time)})."
                )

        self.obs    = obs
        self.config = config
        self.split  = split
        self.subset = subset
        self.u10    = u10
        self.v10    = v10

        # Index range for this subset
        if subset == "train":
            start, end = 0, split.train_end
        elif subset == "val":
            start, end = split.train_end, split.val_end
        else:
            start, end = split.val_end, split.n_total

        self._start = start
        self._end   = end

        # Build full-length raw arrays
        self._X_raw, self._target_raw = self._build_arrays()

        # Training medians (computed from train subset only)
        if subset == "train":
            self._train_medians = self._compute_train_medians(start, end)
        else:
            self._train_medians = train_medians

        # Window origin indices
        # Origin = last input step.  Window: [origin-L+1 : origin+H+1)
        # All indices must lie within [start, end).
        L, H = config.input_length, config.forecast_horizon
        self._origins: List[int] = []
        for origin in range(start + L - 1, end - H):
            self._origins.append(origin)

    def _build_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build full-length feature and target arrays.

        X_raw   shape: (N, n_features)
        target  shape: (N, N_TARGETS_V2)

        X uses ONLY observable fields (no truth leakage).
        Target uses TRUE values (for supervised evaluation only).
        """
        obs = self.obs
        sin_obs, cos_obs = direction_to_sincos(obs.direction_obs)

        wave_cols = [
            obs.Hs_obs,                         # 0: Hs
            obs.Tp_obs,                         # 1: Tp
            sin_obs,                            # 2: sin_direction
            cos_obs,                            # 3: cos_direction
            obs.valid_Hs.astype(float),         # 4: Hs_valid
            obs.valid_Tp.astype(float),         # 5: Tp_valid
            obs.valid_direction.astype(float),  # 6: direction_valid
        ]

        if self.config.feature_mode == "wave_wind":
            u10 = np.asarray(self.u10, dtype=float)
            v10 = np.asarray(self.v10, dtype=float)
            u10_valid = np.isfinite(u10).astype(float)
            v10_valid = np.isfinite(v10).astype(float)
            # Replace non-finite wind with NaN for consistency
            u10 = np.where(np.isfinite(u10), u10, np.nan)
            v10 = np.where(np.isfinite(v10), v10, np.nan)
            X_raw = np.column_stack(wave_cols + [u10, v10, u10_valid, v10_valid])
        else:
            X_raw = np.column_stack(wave_cols)

        # Target: TRUE future values
        sin_true, cos_true = direction_to_sincos(obs.direction_true)
        target_raw = np.column_stack([
            obs.Hs_true,   # 0: Hs
            obs.Tp_true,   # 1: Tp
            sin_true,      # 2: sin_direction
            cos_true,      # 3: cos_direction
        ])

        return X_raw, target_raw

    def _compute_train_medians(self, start: int, end: int) -> np.ndarray:
        """Compute per-feature medians from training data only (NaN-safe)."""
        X_train = self._X_raw[start:end]
        return np.nanmedian(X_train, axis=0)

    def __len__(self) -> int:
        return len(self._origins)

    def __getitem__(self, idx: int) -> ForecastSampleV2:
        origin = self._origins[idx]
        L, H   = self.config.input_length, self.config.forecast_horizon

        # Input window: [origin - L + 1 : origin + 1]
        X = self._X_raw[origin - L + 1 : origin + 1].copy()

        # Apply missing policy to wave features only (indices 0, 1)
        if self.config.missing_policy == "train_median" and self._train_medians is not None:
            for f in [0, 1]:  # Hs, Tp only
                nan_mask = np.isnan(X[:, f])
                if np.any(nan_mask):
                    X[nan_mask, f] = self._train_medians[f]

        # Target window: [origin + 1 : origin + H + 1] — TRUE future values
        target = self._target_raw[origin + 1 : origin + H + 1].copy()

        # Input validity mask: all three wave obs valid at each step
        valid_H = X[:, 4].astype(bool)
        valid_T = X[:, 5].astype(bool)
        valid_D = X[:, 6].astype(bool)
        input_mask = valid_H & valid_T & valid_D

        return ForecastSampleV2(
            X=X,
            target=target,
            origin_index=origin,
            input_mask=input_mask,
        )

    @property
    def train_medians(self) -> Optional[np.ndarray]:
        """Training-set medians, shape (n_features,).  None if not computed."""
        return self._train_medians

    def get_all_X(self) -> np.ndarray:
        """Return all input windows stacked, shape (n_samples, input_length, n_features)."""
        return np.stack([self[i].X for i in range(len(self))], axis=0)

    def get_all_targets(self) -> np.ndarray:
        """Return all targets stacked, shape (n_samples, forecast_horizon, N_TARGETS_V2)."""
        return np.stack([self[i].target for i in range(len(self))], axis=0)

    def get_all_origins(self) -> np.ndarray:
        """Return all origin indices, shape (n_samples,)."""
        return np.array(self._origins)


# ---------------------------------------------------------------------------
# WaveScalerV2
# ---------------------------------------------------------------------------

class WaveScalerV2:
    """
    Z-score scaler for v2 wave forecasting features and targets.

    Standardizes continuous variables only:
        wave_only mode : Hs (idx 0), Tp (idx 1)
        wave_wind mode : Hs (idx 0), Tp (idx 1), u10 (idx 7), v10 (idx 8)

    Direction sin/cos, validity masks are passed through unchanged.

    Fitted ONLY on training data.  Never re-fitted on val/test.
    NaN values are ignored during fitting (nanmean/nanstd).
    NaN values are preserved during transform.
    """

    def __init__(self, feature_mode: str = "wave_only") -> None:
        if feature_mode not in ("wave_only", "wave_wind"):
            raise ValueError(f"feature_mode must be 'wave_only' or 'wave_wind'")
        self.feature_mode = feature_mode
        self._scale_idx = (
            _SCALE_WIND_IDX if feature_mode == "wave_wind" else _SCALE_WAVE_IDX
        )
        n_feat = N_FEATURES_WIND if feature_mode == "wave_wind" else N_FEATURES_WAVE
        self._n_features = n_feat
        self._mean: Optional[np.ndarray] = None
        self._std:  Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray) -> "WaveScalerV2":
        """
        Fit scaler on training features.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, input_length, n_features) or (n_samples, n_features).
            Must be training data ONLY.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        elif X.ndim == 2:
            X_flat = X
        else:
            raise ValueError(f"X must be 2-D or 3-D, got {X.ndim}-D")

        self._mean = np.zeros(self._n_features)
        self._std  = np.ones(self._n_features)

        feature_names = (
            FEATURE_NAMES_WIND if self.feature_mode == "wave_wind"
            else FEATURE_NAMES_WAVE
        )
        for idx in self._scale_idx:
            col = X_flat[:, idx]
            col_valid = col[~np.isnan(col)]
            if len(col_valid) == 0:
                raise ValueError(
                    f"Feature '{feature_names[idx]}' has no valid values for fitting."
                )
            self._mean[idx] = float(np.mean(col_valid))
            s = float(np.std(col_valid))
            self._std[idx] = s if s > 1e-12 else 1.0

        self._fitted = True
        return self

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("WaveScalerV2 has not been fitted.  Call fit() first.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply z-score normalization.  NaN values are preserved."""
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        for idx in self._scale_idx:
            X[..., idx] = (X[..., idx] - self._mean[idx]) / self._std[idx]
        return X

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization."""
        self._check_fitted()
        X = np.asarray(X, dtype=float).copy()
        for idx in self._scale_idx:
            X[..., idx] = X[..., idx] * self._std[idx] + self._mean[idx]
        return X

    def transform_target(self, y: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to targets.

        Target indices 0 (Hs) and 1 (Tp) are standardized using the
        same statistics as the corresponding feature columns.
        """
        self._check_fitted()
        y = np.asarray(y, dtype=float).copy()
        # Target always has Hs at 0, Tp at 1 regardless of feature mode
        hs_mean, hs_std = self._mean[0], self._std[0]
        tp_mean, tp_std = self._mean[1], self._std[1]
        y[..., 0] = (y[..., 0] - hs_mean) / hs_std
        y[..., 1] = (y[..., 1] - tp_mean) / tp_std
        return y

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        """Reverse z-score normalization for targets."""
        self._check_fitted()
        y = np.asarray(y, dtype=float).copy()
        y[..., 0] = y[..., 0] * self._std[0] + self._mean[0]
        y[..., 1] = y[..., 1] * self._std[1] + self._mean[1]
        return y

    @property
    def fitted_stats(self) -> dict:
        """Return fitted mean and std arrays."""
        self._check_fitted()
        return {"mean": self._mean.copy(), "std": self._std.copy()}

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ---------------------------------------------------------------------------
# build_datasets_v2 — convenience factory
# ---------------------------------------------------------------------------

def build_datasets_v2(
    obs:    SensorObservation,
    config: ForecastConfigV2,
    split:  YearSplit,
    u10:    Optional[np.ndarray] = None,
    v10:    Optional[np.ndarray] = None,
) -> Tuple[ForecastDatasetV2, ForecastDatasetV2, ForecastDatasetV2, WaveScalerV2]:
    """
    Build train, validation, and test ForecastDatasetV2 objects.

    The year-based split is applied BEFORE window generation.
    Training medians are computed from training data and passed to val/test.
    The scaler is fitted on training data only.

    Parameters
    ----------
    obs : SensorObservation
        Full time series of sensor observations.
    config : ForecastConfigV2
        Dataset configuration.
    split : YearSplit
        Pre-computed year-boundary split.
    u10, v10 : np.ndarray or None
        ERA5 wind components.  Required for wave_wind mode.

    Returns
    -------
    train_ds, val_ds, test_ds, scaler
    """
    train_ds = ForecastDatasetV2(obs, config, split, "train", u10=u10, v10=v10)
    val_ds   = ForecastDatasetV2(obs, config, split, "val",   u10=u10, v10=v10,
                                  train_medians=train_ds.train_medians)
    test_ds  = ForecastDatasetV2(obs, config, split, "test",  u10=u10, v10=v10,
                                  train_medians=train_ds.train_medians)

    scaler = WaveScalerV2(feature_mode=config.feature_mode)
    if len(train_ds) > 0:
        X_train = train_ds.get_all_X()
        scaler.fit(X_train)

    return train_ds, val_ds, test_ds, scaler
