"""
sensor.py — Synthetic wave sensor / camera observation model (Module 3).

============================================================
PURPOSE
============================================================

Models the OUTPUT of a hypothetical camera-based wave estimation system
mounted on the WEC.  This module does NOT implement computer vision,
image processing, wave segmentation, or tracking.

The sensor represents the measurement uncertainty of such a system:

    camera imagery
        ↓
    wave estimation algorithm   (NOT modelled here)
        ↓
    Hs / Tp / direction observations   (THIS module)

============================================================
SENSOR MODEL
============================================================

For each observation at time index n, the sensor produces:

    Hs_obs[n]        = Hs_true[n - delay] + bias_Hs + N(0, σ_Hs)
    Tp_obs[n]        = Tp_true[n - delay] + bias_Tp + N(0, σ_Tp)
    dir_obs[n]       = wrap360(dir_true[n - delay] + bias_dir + N(0, σ_dir))

Physical bounds are enforced AFTER noise addition:
    Hs_obs >= 0   (clipped; clip events are counted)
    Tp_obs > 0    (clipped; clip events are counted)
    dir_obs in [0, 360)  (circular wrap, no clipping needed)

Direction arithmetic is ALWAYS circular.  Example:
    359° + 5° → 4°    (not 364°)
    1°  − 5° → 356°   (not −4°)

============================================================
CAUSAL GUARANTEE
============================================================

Observation at index n uses truth at index n − delay_steps.
For n < delay_steps the observation is marked UNAVAILABLE (NaN + mask=False).
No future truth is ever accessed.

============================================================
MISSING DATA
============================================================

Each observation is independently dropped with probability missing_probability
using a seeded RNG.  Missing observations are represented as NaN with
valid_* = False.  No forward-fill or imputation is performed here.

============================================================
LEAKAGE PREVENTION
============================================================

SensorObservation.as_model_input() returns ONLY the observable fields
(time, Hs_obs, Tp_obs, direction_obs, valid_*).  True values are
retained in the full object for validation only and are never returned
by as_model_input().

============================================================
SCIENTIFIC LIMITATIONS
============================================================

- Synthetic sensor only.  Noise parameters are development assumptions.
- No actual camera image processing is implemented.
- No wave segmentation or tracking is implemented.
- No environmental occlusion model (sea-spray, glare, rain, fog).
- No spatial camera geometry is modelled.
- No uncertainty calibration against real buoy-camera data.
- Noise is assumed i.i.d. Gaussian; real camera errors are correlated.

============================================================
UNITS
============================================================
    Hs          : m
    Tp          : s
    direction   : degrees (ERA5 convention, clockwise from North)
    time        : any (passed through unchanged)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Circular direction helpers
# ---------------------------------------------------------------------------

def _wrap360(deg: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [0, 360)."""
    return np.asarray(deg, dtype=float) % 360.0


def _circular_add(base: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Add delta to base direction and wrap result to [0, 360)."""
    return _wrap360(np.asarray(base, dtype=float) + np.asarray(delta, dtype=float))


def circular_direction_error(obs: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Signed circular error in degrees: obs − true, wrapped to (−180, 180].

    Parameters
    ----------
    obs : array_like
        Observed directions [degrees].
    true : array_like
        True directions [degrees].

    Returns
    -------
    np.ndarray
        Signed errors in (−180, 180].  Exactly ±180 maps to +180.
    """
    diff = np.asarray(obs, dtype=float) - np.asarray(true, dtype=float)
    # Wrap to (-180, 180]: use modulo then shift; -180 → +180
    err = diff % 360.0          # [0, 360)
    err = np.where(err > 180.0, err - 360.0, err)   # (-180, 180]
    return err


# ---------------------------------------------------------------------------
# SensorParameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorParameters:
    """
    Configuration for the synthetic wave sensor.

    Parameters
    ----------
    sampling_interval : float
        Sensor sampling interval [same units as input time].  Currently
        informational; the sensor processes every row of the input.
    hs_noise_std : float
        Standard deviation of Gaussian Hs noise [m].  >= 0.
    tp_noise_std : float
        Standard deviation of Gaussian Tp noise [s].  >= 0.
    direction_noise_std : float
        Standard deviation of Gaussian direction noise [degrees].  >= 0.
    hs_bias : float
        Systematic Hs bias [m].  Added before noise.
    tp_bias : float
        Systematic Tp bias [s].  Added before noise.
    direction_bias : float
        Systematic direction bias [degrees].  Added circularly.
    delay_steps : int
        Number of samples of sensor latency.  >= 0.
        Observation at index n uses truth at index n − delay_steps.
        Indices 0 .. delay_steps−1 are marked unavailable.
    missing_probability : float
        Probability in [0, 1] that any given observation is missing.
    random_seed : int or None
        Seed for the internal RNG.  None = non-reproducible.
    """

    sampling_interval: float = 3600.0
    hs_noise_std: float = 0.1
    tp_noise_std: float = 0.5
    direction_noise_std: float = 10.0
    hs_bias: float = 0.0
    tp_bias: float = 0.0
    direction_bias: float = 0.0
    delay_steps: int = 0
    missing_probability: float = 0.0
    random_seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hs_noise_std < 0:
            raise ValueError(f"hs_noise_std must be >= 0, got {self.hs_noise_std}")
        if self.tp_noise_std < 0:
            raise ValueError(f"tp_noise_std must be >= 0, got {self.tp_noise_std}")
        if self.direction_noise_std < 0:
            raise ValueError(f"direction_noise_std must be >= 0, got {self.direction_noise_std}")
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {self.delay_steps}")
        if not (0.0 <= self.missing_probability <= 1.0):
            raise ValueError(f"missing_probability must be in [0,1], got {self.missing_probability}")


# ---------------------------------------------------------------------------
# SensorObservation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorObservation:
    """
    Sensor output for a sequence of observations.

    True fields are retained for validation only.
    Call as_model_input() to obtain only the observable fields.

    Parameters
    ----------
    time : np.ndarray
        Timestamps, shape (n,).
    Hs_true : np.ndarray
        True significant wave height [m], shape (n,).
    Tp_true : np.ndarray
        True peak period [s], shape (n,).
    direction_true : np.ndarray
        True mean direction [degrees], shape (n,).
    Hs_obs : np.ndarray
        Observed Hs [m], shape (n,).  NaN where not valid.
    Tp_obs : np.ndarray
        Observed Tp [s], shape (n,).  NaN where not valid.
    direction_obs : np.ndarray
        Observed direction [degrees], shape (n,).  NaN where not valid.
    valid_Hs : np.ndarray
        Boolean mask: True where Hs_obs is a valid observation.
    valid_Tp : np.ndarray
        Boolean mask: True where Tp_obs is a valid observation.
    valid_direction : np.ndarray
        Boolean mask: True where direction_obs is a valid observation.
    delay_steps : int
        Sensor delay used.
    params : SensorParameters
        Parameters used to generate this observation.
    n_hs_clipped : int
        Number of Hs observations clipped to 0 (would have been negative).
    n_tp_clipped : int
        Number of Tp observations clipped to 0 (would have been <= 0).
    """

    time: np.ndarray
    Hs_true: np.ndarray
    Tp_true: np.ndarray
    direction_true: np.ndarray
    Hs_obs: np.ndarray
    Tp_obs: np.ndarray
    direction_obs: np.ndarray
    valid_Hs: np.ndarray
    valid_Tp: np.ndarray
    valid_direction: np.ndarray
    delay_steps: int
    params: SensorParameters
    n_hs_clipped: int = 0
    n_tp_clipped: int = 0

    def as_model_input(self) -> dict:
        """
        Return ONLY the observable fields available to a forecasting model.

        Does NOT include true values.  Only contains information that would
        be available at or before the observation time.

        Returns
        -------
        dict with keys:
            time, Hs_obs, Tp_obs, direction_obs,
            valid_Hs, valid_Tp, valid_direction
        """
        return {
            "time": self.time,
            "Hs_obs": self.Hs_obs,
            "Tp_obs": self.Tp_obs,
            "direction_obs": self.direction_obs,
            "valid_Hs": self.valid_Hs,
            "valid_Tp": self.valid_Tp,
            "valid_direction": self.valid_direction,
        }

    @property
    def missing_fraction(self) -> float:
        """Fraction of observations that are missing (any variable)."""
        n = len(self.time)
        if n == 0:
            return 0.0
        missing = ~(self.valid_Hs & self.valid_Tp & self.valid_direction)
        return float(np.mean(missing))

    @property
    def hs_error(self) -> np.ndarray:
        """Hs_obs − Hs_true where valid, else NaN."""
        err = self.Hs_obs - self.Hs_true
        err[~self.valid_Hs] = np.nan
        return err

    @property
    def tp_error(self) -> np.ndarray:
        """Tp_obs − Tp_true where valid, else NaN."""
        err = self.Tp_obs - self.Tp_true
        err[~self.valid_Tp] = np.nan
        return err

    @property
    def direction_error(self) -> np.ndarray:
        """Circular direction error (obs − true) in (−180, 180] where valid, else NaN."""
        err = circular_direction_error(self.direction_obs, self.direction_true)
        err[~self.valid_direction] = np.nan
        return err


# ---------------------------------------------------------------------------
# WaveSensor
# ---------------------------------------------------------------------------

class WaveSensor:
    """
    Synthetic wave sensor producing imperfect observations of Hs, Tp, direction.

    ORACLE/REFERENCE LATCHING BASELINE — uses true excitation signal.

    This sensor models the measurement uncertainty of a hypothetical
    camera-based wave estimation system.  It does NOT implement computer
    vision or image processing.

    Parameters
    ----------
    params : SensorParameters
        Sensor configuration.
    """

    def __init__(self, params: SensorParameters) -> None:
        self.params = params
        self._rng = np.random.default_rng(params.random_seed)

    def observe(
        self,
        time: np.ndarray,
        Hs_true: np.ndarray,
        Tp_true: np.ndarray,
        direction_true: np.ndarray,
    ) -> SensorObservation:
        """
        Generate synthetic observations from true wave state.

        Causal guarantee: observation at index n uses truth at index
        n − delay_steps.  Indices 0 .. delay_steps−1 are unavailable.

        Parameters
        ----------
        time : array_like
            Timestamps, shape (n,).
        Hs_true : array_like
            True significant wave height [m], shape (n,).
        Tp_true : array_like
            True peak period [s], shape (n,).
        direction_true : array_like
            True mean direction [degrees, 0–360), shape (n,).

        Returns
        -------
        SensorObservation
        """
        time          = np.asarray(time,          dtype=float)
        Hs_true       = np.asarray(Hs_true,       dtype=float)
        Tp_true       = np.asarray(Tp_true,       dtype=float)
        direction_true = np.asarray(direction_true, dtype=float)

        n = len(time)
        p = self.params
        d = p.delay_steps

        # Output arrays — initialised to NaN
        Hs_obs  = np.full(n, np.nan)
        Tp_obs  = np.full(n, np.nan)
        dir_obs = np.full(n, np.nan)
        valid_H = np.zeros(n, dtype=bool)
        valid_T = np.zeros(n, dtype=bool)
        valid_D = np.zeros(n, dtype=bool)

        n_hs_clipped = 0
        n_tp_clipped = 0

        # Draw all noise and missing masks at once for reproducibility
        hs_noise  = self._rng.normal(0.0, p.hs_noise_std,        n)
        tp_noise  = self._rng.normal(0.0, p.tp_noise_std,        n)
        dir_noise = self._rng.normal(0.0, p.direction_noise_std, n)
        missing   = self._rng.random(n) < p.missing_probability

        for i in range(n):
            src = i - d          # index into truth arrays
            if src < 0:
                # Before delay period: unavailable
                continue
            if missing[i]:
                # Randomly missing
                continue

            # Hs
            hs_raw = Hs_true[src] + p.hs_bias + hs_noise[i]
            if hs_raw < 0.0:
                n_hs_clipped += 1
                hs_raw = 0.0
            Hs_obs[i]  = hs_raw
            valid_H[i] = True

            # Tp
            tp_raw = Tp_true[src] + p.tp_bias + tp_noise[i]
            if tp_raw <= 0.0:
                n_tp_clipped += 1
                tp_raw = 1e-3   # smallest physically meaningful value
            Tp_obs[i]  = tp_raw
            valid_T[i] = True

            # Direction (circular)
            dir_obs[i] = _circular_add(
                direction_true[src],
                p.direction_bias + dir_noise[i],
            )
            valid_D[i] = True

        return SensorObservation(
            time=time,
            Hs_true=Hs_true,
            Tp_true=Tp_true,
            direction_true=direction_true,
            Hs_obs=Hs_obs,
            Tp_obs=Tp_obs,
            direction_obs=dir_obs,
            valid_Hs=valid_H,
            valid_Tp=valid_T,
            valid_direction=valid_D,
            delay_steps=d,
            params=p,
            n_hs_clipped=n_hs_clipped,
            n_tp_clipped=n_tp_clipped,
        )
