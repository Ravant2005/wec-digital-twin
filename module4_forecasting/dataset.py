"""
dataset.py — Wave forecasting dataset builder (Module 4.1).

============================================================
FORECASTING TASK
============================================================

Given the previous L hours of observed wave conditions:

    [Hs_obs, Tp_obs, direction_obs]_(t-L+1 ... t)

predict the next H hours of TRUE wave state:

    [Hs_true, Tp_true, direction_true]_(t+1 ... t+H)

The input uses ONLY sensor observations (Module 3 output).
The target uses TRUE future values for supervised evaluation.
Future truth NEVER appears in the input features.

============================================================
FEATURE ORDER  (FEATURE_NAMES)
============================================================

Index  Name
  0    Hs              observed Hs [m]  (NaN if missing)
  1    Tp              observed Tp [s]  (NaN if missing)
  2    sin_direction   sin(direction_obs * pi/180)
  3    cos_direction   cos(direction_obs * pi/180)
  4    Hs_valid        1.0 if Hs_obs is valid, else 0.0
  5    Tp_valid        1.0 if Tp_obs is valid, else 0.0
  6    direction_valid 1.0 if direction_obs is valid, else 0.0

Shape of X per sample: (input_length, 7)

============================================================
TARGET ORDER  (TARGET_NAMES)
============================================================

Index  Name
  0    Hs              true future Hs [m]
  1    Tp              true future Tp [s]
  2    sin_direction   sin(true future direction * pi/180)
  3    cos_direction   cos(true future direction * pi/180)

Shape of target per sample: (forecast_horizon, 4)

============================================================
CHRONOLOGICAL SPLIT
============================================================

The time series is split chronologically BEFORE generating windows:

    [0 : train_end)          → train
    [train_end : val_end)    → validation
    [val_end : N)            → test

Windows are generated independently within each split.
No window ever crosses a split boundary.
Random shuffling of the time series is FORBIDDEN.

============================================================
DIRECTION REPRESENTATION
============================================================

Raw direction degrees are converted to (sin, cos) to avoid the
discontinuity at 0°/360°.  The inverse transform recovers degrees
via atan2.

============================================================
MISSING DATA POLICY
============================================================

"mask_only" (default):
    Missing values remain NaN in X.  The valid_* columns carry the mask.
    Downstream models must handle NaN explicitly.

"train_median":
    Missing values in X are replaced by the training-set median of each
    feature, computed ONLY from training data.  Valid masks are retained.
    This policy must be applied AFTER fitting the scaler on training data.
    It must NEVER use validation/test statistics.

============================================================
DATASET SIZE WARNING
============================================================

The ERA5 January 2024 dataset contains 744 hourly samples.
With input_length=24 and forecast_horizon=6, the maximum number of
non-overlapping windows is approximately 744 / 30 ≈ 24.
Even with overlapping windows the total is ~714 samples.
This is INSUFFICIENT for meaningful deep-learning training.
Results from any neural network trained on this dataset should be
treated as proof-of-concept only, not production-quality forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from module3_sensor.sensor import SensorObservation

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_NAMES: List[str] = [
    "Hs",
    "Tp",
    "sin_direction",
    "cos_direction",
    "Hs_valid",
    "Tp_valid",
    "direction_valid",
]

TARGET_NAMES: List[str] = [
    "Hs",
    "Tp",
    "sin_direction",
    "cos_direction",
]

N_FEATURES = len(FEATURE_NAMES)   # 7
N_TARGETS  = len(TARGET_NAMES)    # 4


# ---------------------------------------------------------------------------
# Direction utilities
# ---------------------------------------------------------------------------

def direction_to_sincos(deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert direction in degrees to (sin, cos) pair.

    Parameters
    ----------
    deg : array_like
        Direction(s) in degrees [0, 360).

    Returns
    -------
    sin_d, cos_d : np.ndarray
        sin(deg * pi/180), cos(deg * pi/180).
        NaN inputs produce NaN outputs.
    """
    rad = np.asarray(deg, dtype=float) * (np.pi / 180.0)
    return np.sin(rad), np.cos(rad)


def sincos_to_direction(sin_d: np.ndarray, cos_d: np.ndarray) -> np.ndarray:
    """
    Recover direction in degrees [0, 360) from (sin, cos).

    Parameters
    ----------
    sin_d, cos_d : array_like
        sin and cos of the direction angle.

    Returns
    -------
    np.ndarray
        Direction in degrees, wrapped to [0, 360).
        NaN inputs produce NaN outputs.
    """
    deg = np.degrees(np.arctan2(
        np.asarray(sin_d, dtype=float),
        np.asarray(cos_d, dtype=float),
    ))
    return deg % 360.0


# ---------------------------------------------------------------------------
# ForecastConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastConfig:
    """
    Configuration for the wave forecasting dataset.

    Parameters
    ----------
    input_length : int
        Number of past observation steps used as input (L).  Default 24.
    forecast_horizon : int
        Number of future steps to predict (H).  Default 6.
    train_fraction : float
        Fraction of time series used for training.  Default 0.70.
    val_fraction : float
        Fraction used for validation.  Default 0.15.
        Test fraction = 1 - train_fraction - val_fraction.
    missing_policy : str
        How to handle NaN in X.  "mask_only" or "train_median".
    """

    input_length: int = 24
    forecast_horizon: int = 6
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    missing_policy: str = "mask_only"

    def __post_init__(self) -> None:
        if self.input_length < 1:
            raise ValueError(f"input_length must be >= 1, got {self.input_length}")
        if self.forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {self.forecast_horizon}")
        if not (0 < self.train_fraction < 1):
            raise ValueError(f"train_fraction must be in (0,1), got {self.train_fraction}")
        if not (0 < self.val_fraction < 1):
            raise ValueError(f"val_fraction must be in (0,1), got {self.val_fraction}")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError("train_fraction + val_fraction must be < 1")
        if self.missing_policy not in ("mask_only", "train_median"):
            raise ValueError(f"missing_policy must be 'mask_only' or 'train_median', "
                             f"got '{self.missing_policy}'")

    @property
    def test_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.val_fraction

    @property
    def window_length(self) -> int:
        """Total timesteps consumed per sample (input + target)."""
        return self.input_length + self.forecast_horizon


# ---------------------------------------------------------------------------
# TimeSeriesSplit
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TimeSeriesSplit:
    """
    Chronological split indices for a time series of length N.

    Attributes
    ----------
    train_end : int
        Exclusive end index of training portion.
    val_end : int
        Exclusive end index of validation portion.
    n_total : int
        Total length of the time series.
    """

    train_end: int
    val_end: int
    n_total: int

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


def make_split(n: int, config: ForecastConfig) -> TimeSeriesSplit:
    """
    Compute chronological split indices.

    Parameters
    ----------
    n : int
        Total number of time steps.
    config : ForecastConfig

    Returns
    -------
    TimeSeriesSplit
    """
    train_end = int(np.floor(n * config.train_fraction))
    val_end   = int(np.floor(n * (config.train_fraction + config.val_fraction)))
    return TimeSeriesSplit(train_end=train_end, val_end=val_end, n_total=n)


# ---------------------------------------------------------------------------
# ForecastSample
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForecastSample:
    """
    A single supervised forecasting sample.

    Attributes
    ----------
    X : np.ndarray
        Input feature matrix, shape (input_length, N_FEATURES).
        Feature order: FEATURE_NAMES.
        May contain NaN where observations are missing.
    target : np.ndarray
        Target matrix, shape (forecast_horizon, N_TARGETS).
        Target order: TARGET_NAMES.
        Contains TRUE future wave state (for supervised evaluation).
        NEVER used as model input.
    origin_index : int
        Index in the original time series of the last input step (t).
    input_mask : np.ndarray
        Boolean mask, shape (input_length,).
        True where ALL three observations (Hs, Tp, direction) are valid.
    """

    X: np.ndarray
    target: np.ndarray
    origin_index: int
    input_mask: np.ndarray


# ---------------------------------------------------------------------------
# ForecastDataset
# ---------------------------------------------------------------------------

class ForecastDataset:
    """
    Sliding-window supervised forecasting dataset from sensor observations.

    Generates samples by sliding a window of length
    (input_length + forecast_horizon) over the time series.

    Windows are generated within a specified index range [start, end),
    ensuring no window crosses split boundaries.

    Parameters
    ----------
    obs : SensorObservation
        Sensor output from Module 3.
    config : ForecastConfig
        Dataset configuration.
    split : TimeSeriesSplit
        Chronological split indices.
    subset : str
        One of "train", "val", "test".  Determines which portion of the
        time series is used for window generation.
    train_medians : np.ndarray or None
        Training-set medians for "train_median" missing policy.
        Shape (N_FEATURES,).  Must be provided if config.missing_policy
        == "train_median" and subset != "train".
    """

    def __init__(
        self,
        obs: SensorObservation,
        config: ForecastConfig,
        split: TimeSeriesSplit,
        subset: str,
        train_medians: Optional[np.ndarray] = None,
    ) -> None:
        if subset not in ("train", "val", "test"):
            raise ValueError(f"subset must be 'train', 'val', or 'test', got '{subset}'")

        self.obs    = obs
        self.config = config
        self.split  = split
        self.subset = subset

        # Determine index range for this subset
        if subset == "train":
            start, end = 0, split.train_end
        elif subset == "val":
            start, end = split.train_end, split.val_end
        else:
            start, end = split.val_end, split.n_total

        self._start = start
        self._end   = end

        # Build raw feature and target arrays from the full observation
        self._X_raw, self._target_raw = self._build_arrays(obs)

        # Compute training medians if this is the train subset
        if subset == "train":
            self._train_medians = self._compute_train_medians(start, end)
        else:
            self._train_medians = train_medians

        # Generate window origin indices (last input step = origin)
        # Window spans [origin - input_length + 1 : origin + forecast_horizon + 1)
        # All indices must lie within [start, end)
        L, H = config.input_length, config.forecast_horizon
        self._origins: List[int] = []
        for origin in range(start + L - 1, end - H):
            self._origins.append(origin)

    def _build_arrays(
        self, obs: SensorObservation
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build full-length feature and target arrays from SensorObservation.

        X_raw shape: (N, N_FEATURES)
        target_raw shape: (N, N_TARGETS)

        X uses ONLY observable fields (no truth).
        target uses TRUE values (for supervised evaluation only).
        """
        n = len(obs.time)

        sin_d, cos_d = direction_to_sincos(obs.direction_obs)

        X_raw = np.column_stack([
            obs.Hs_obs,                          # 0: Hs
            obs.Tp_obs,                          # 1: Tp
            sin_d,                               # 2: sin_direction
            cos_d,                               # 3: cos_direction
            obs.valid_Hs.astype(float),          # 4: Hs_valid
            obs.valid_Tp.astype(float),          # 5: Tp_valid
            obs.valid_direction.astype(float),   # 6: direction_valid
        ])

        # Target: TRUE future values (for supervised training/evaluation)
        sin_true, cos_true = direction_to_sincos(obs.direction_true)
        target_raw = np.column_stack([
            obs.Hs_true,    # 0: Hs
            obs.Tp_true,    # 1: Tp
            sin_true,       # 2: sin_direction
            cos_true,       # 3: cos_direction
        ])

        return X_raw, target_raw

    def _compute_train_medians(self, start: int, end: int) -> np.ndarray:
        """
        Compute per-feature medians from training data only.

        Uses only the training portion [start, end).
        NaN values are ignored (nanmedian).
        """
        X_train = self._X_raw[start:end]
        return np.nanmedian(X_train, axis=0)

    def __len__(self) -> int:
        return len(self._origins)

    def __getitem__(self, idx: int) -> ForecastSample:
        origin = self._origins[idx]
        L, H   = self.config.input_length, self.config.forecast_horizon

        # Input window: [origin - L + 1 : origin + 1]
        X = self._X_raw[origin - L + 1 : origin + 1].copy()

        # Apply missing policy
        if self.config.missing_policy == "train_median" and self._train_medians is not None:
            for f in range(N_FEATURES):
                nan_mask = np.isnan(X[:, f])
                if np.any(nan_mask):
                    X[nan_mask, f] = self._train_medians[f]

        # Target window: [origin + 1 : origin + H + 1]  — TRUE future values
        target = self._target_raw[origin + 1 : origin + H + 1].copy()

        # Input validity mask: all three obs valid at each step
        valid_H = X[:, 4].astype(bool)
        valid_T = X[:, 5].astype(bool)
        valid_D = X[:, 6].astype(bool)
        input_mask = valid_H & valid_T & valid_D

        return ForecastSample(
            X=X,
            target=target,
            origin_index=origin,
            input_mask=input_mask,
        )

    @property
    def train_medians(self) -> Optional[np.ndarray]:
        """Training-set medians, shape (N_FEATURES,).  None if not computed."""
        return self._train_medians

    def get_all_X(self) -> np.ndarray:
        """Return all input windows stacked, shape (n_samples, input_length, N_FEATURES)."""
        return np.stack([self[i].X for i in range(len(self))], axis=0)

    def get_all_targets(self) -> np.ndarray:
        """Return all targets stacked, shape (n_samples, forecast_horizon, N_TARGETS)."""
        return np.stack([self[i].target for i in range(len(self))], axis=0)

    def get_all_origins(self) -> np.ndarray:
        """Return all origin indices, shape (n_samples,)."""
        return np.array(self._origins)


def build_datasets(
    obs: SensorObservation,
    config: ForecastConfig,
) -> Tuple[ForecastDataset, ForecastDataset, ForecastDataset, TimeSeriesSplit]:
    """
    Build train, validation, and test ForecastDatasets from a SensorObservation.

    The split is chronological.  The scaler is fitted on training data only.
    Training medians are computed from training data and passed to val/test.

    Parameters
    ----------
    obs : SensorObservation
    config : ForecastConfig

    Returns
    -------
    train_ds, val_ds, test_ds, split
    """
    n     = len(obs.time)
    split = make_split(n, config)

    train_ds = ForecastDataset(obs, config, split, "train")
    val_ds   = ForecastDataset(obs, config, split, "val",
                               train_medians=train_ds.train_medians)
    test_ds  = ForecastDataset(obs, config, split, "test",
                               train_medians=train_ds.train_medians)

    return train_ds, val_ds, test_ds, split
