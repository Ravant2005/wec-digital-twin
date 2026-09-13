"""
forecast_uncertainty.py — Empirical forecast-error injection (Module 5.2B).

============================================================
PURPOSE
============================================================

Provides the ForecastErrorSampler class, which injects realistic forecast
uncertainty into the deterministic GRU+wind output during RL training.

The sampled errors come exclusively from the held-out Module 4.2 test
period (2024–2025) as stored in:

    results/forecasting/exp_gru_wind/forecast_errors.csv.gz

This is NOT Gaussian noise injection.  The empirical error distributions
are strongly non-Gaussian (kurtosis up to 25 at short leads) and vary
significantly with lead time.  Gaussian approximation would be scientifically
unjustifiable and is explicitly prohibited by the research plan.

============================================================
ANTI-LEAKAGE GUARANTEE
============================================================

The ForecastErrorSampler draws errors from the Module 4.2 held-out test
statistics — a population of forecast mistakes observed on 17,449 historical
forecasts.  These errors do NOT come from the current RL episode's future
truth.  Adding them to the deterministic GRU output does NOT expose the
agent to future truth.

The distinction:
    - Sampled errors: from the population of historical GRU mistakes
    - Current episode's true future: used ONLY for physics/reward/diagnostics

These are structurally separate and never combined in the RL observation.

============================================================
ERROR SIGN CONVENTION
============================================================

The CSV stores:
    Hs_error   = Hs_pred  − Hs_true    (positive = model overestimate)
    Tp_error   = Tp_pred  − Tp_true    (positive = model overestimate)
    dir_error_deg = circular(dir_pred_deg − dir_true_deg)  (−180, 180]

Application to a new deterministic forecast:

    Hs_realistic   = Hs_det   + sampled_Hs_error
    Tp_realistic   = Tp_det   + sampled_Tp_error
    dir_realistic  = wrap360(dir_det + sampled_dir_error)

This means: if the sampled error is +0.2 m (typical overestimate), the
realistic forecast will be 0.2 m higher than the deterministic prediction.
Equivalently, the agent sees a biased-high forecast, just as it would when
the real GRU makes that error on actual data.

============================================================
TRAJECTORY SAMPLING STRATEGY
============================================================

The CSV contains 17,449 complete forecast trajectories × 48 lead hours.
Each trajectory (identified by sample_idx) has a row for every lead from
1 to 48.

Approach: sample ONE trajectory index uniformly from [0, 17448], and use
the errors at leads 1..48 from that trajectory.

Why trajectories, not independent lead draws:
1. Temporal correlation is high (Hs lead1–lead2 r = 0.956).  Independent
   draws would produce physically implausible step-changes in forecast error.
2. Joint variable structure (Hs, Tp, dir) is preserved within each trajectory
   even though cross-variable correlations are modest (r ~ 0.03–0.13).
3. The resulting error profiles are guaranteed to have been observed on real
   GRU+wind forecasts on real ERA5 data — the strongest possible justification.

============================================================
MEMORY LAYOUT
============================================================

At load time the three error arrays are stored as float32:

    _err_hs  : shape (17449, 48)  — Hs errors, rows=trajectories, cols=leads
    _err_tp  : shape (17449, 48)  — Tp errors
    _err_dir : shape (17449, 48)  — direction errors [degrees]

Column index k corresponds to lead_h = k+1 (0-indexed).
Total memory: ~10 MB.

============================================================
PHYSICAL VALIDITY ENFORCEMENT
============================================================

After error injection, rare combinations can produce physically invalid values.
The following corrections are applied silently with documented counts:

    Hs_realistic < 0   → clipped to 0.0  (0.01% of realistic predictions in test data)
    Tp_realistic ≤ 0   → clipped to 0.1  (0.04% of realistic predictions in test data)
    dir_realistic      → wrapped to [0, 360) (always, by circular arithmetic)

The clipping fractions are small enough that they do not materially bias the
distribution.  They are documented as a limitation in the audit.

============================================================
FORECAST REPRESENTATION IN THE OBSERVATION
============================================================

The GRU outputs (and this module modifies) forecasts in the format:
    index 0: Hs      [m]
    index 1: Tp      [s]
    index 2: sin(direction)
    index 3: cos(direction)

Direction error is applied in degree space:
    1. Reconstruct dir_pred_deg = atan2(sin_dir, cos_dir) × 180/π wrapped to [0,360)
    2. dir_realistic_deg = wrap360(dir_pred_deg + sampled_dir_error_deg)
    3. sin_realistic, cos_realistic = direction_to_sincos(dir_realistic_deg)

This preserves the (sin, cos) representation expected by the observation
builder, consistent with Module 4's training convention.

============================================================
UNITS
============================================================
    Hs_error        : m
    Tp_error        : s
    dir_error_deg   : degrees  (circular, (−180, 180])
    lead_h          : hours  (1..48)

============================================================
MODULE DEPENDENCIES (read-only, no modifications)
============================================================
    module4_forecasting.dataset  — direction_to_sincos()
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from module4_forecasting.dataset import direction_to_sincos

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default path to the empirical forecast-error CSV.
DEFAULT_ERROR_CSV: str = os.path.join(
    "results", "forecasting", "exp_gru_wind", "forecast_errors.csv.gz"
)

#: Number of forecast leads (must match Module 4.2 forecast_horizon).
N_LEADS: int = 48

#: Minimum physical Hs after injection [m].
HS_MIN: float = 0.0

#: Minimum physical Tp after injection [s].
TP_MIN: float = 0.1


# ---------------------------------------------------------------------------
# Helper: direction utilities
# ---------------------------------------------------------------------------

def _wrap360(deg: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [0, 360)."""
    return np.asarray(deg, dtype=float) % 360.0


def _sincos_to_deg(sin_dir: np.ndarray, cos_dir: np.ndarray) -> np.ndarray:
    """
    Convert (sin, cos) direction representation to degrees in [0, 360).

    Parameters
    ----------
    sin_dir, cos_dir : array_like
        Sin and cos of direction.

    Returns
    -------
    np.ndarray : direction in degrees, [0, 360).
    """
    deg = np.degrees(np.arctan2(
        np.asarray(sin_dir, dtype=float),
        np.asarray(cos_dir, dtype=float),
    ))
    return _wrap360(deg)


# ---------------------------------------------------------------------------
# ErrorTrajectoryBank — stores pre-loaded error arrays
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorTrajectoryBank:
    """
    Pre-loaded empirical forecast-error trajectories from Module 4.2 test data.

    Stores three arrays of shape (n_trajectories, n_leads):

        err_hs   : Hs errors [m]          = Hs_pred − Hs_true
        err_tp   : Tp errors [s]          = Tp_pred − Tp_true
        err_dir  : direction errors [deg] = circular(dir_pred − dir_true)

    Stored as float32 (~10 MB total) with leads 0-indexed (column k = lead_h k+1).

    Parameters
    ----------
    err_hs : np.ndarray, shape (n_traj, 48), float32
    err_tp : np.ndarray, shape (n_traj, 48), float32
    err_dir : np.ndarray, shape (n_traj, 48), float32
    n_trajectories : int
    n_leads : int
    csv_path : str  path from which this bank was loaded
    """

    err_hs:           np.ndarray   # shape (n_traj, 48)
    err_tp:           np.ndarray   # shape (n_traj, 48)
    err_dir:          np.ndarray   # shape (n_traj, 48)
    n_trajectories:   int
    n_leads:          int
    csv_path:         str

    def get_trajectory(self, traj_idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return the error trajectory for a given sample index.

        Parameters
        ----------
        traj_idx : int
            Trajectory index in [0, n_trajectories).

        Returns
        -------
        (hs_errors, tp_errors, dir_errors)
            Each shape (n_leads,), dtype float64.
            Indexing: element k corresponds to lead_h = k+1.
        """
        i = int(traj_idx) % self.n_trajectories
        return (
            self.err_hs[i].astype(np.float64),
            self.err_tp[i].astype(np.float64),
            self.err_dir[i].astype(np.float64),
        )

    def lead_statistics(self, lead_h: int) -> dict:
        """
        Return descriptive statistics for a specific lead time.

        Parameters
        ----------
        lead_h : int  Lead time in hours (1..48).

        Returns
        -------
        dict  with keys: lead_h, Hs_mean, Hs_std, Hs_p05, Hs_p25, Hs_p50,
              Hs_p75, Hs_p95, Tp_mean, Tp_std, Tp_p05, Tp_p25, Tp_p50,
              Tp_p75, Tp_p95, dir_mean, dir_std, dir_p05, dir_p25, dir_p50,
              dir_p75, dir_p95.
        """
        k = lead_h - 1
        hs  = self.err_hs[:, k].astype(float)
        tp  = self.err_tp[:, k].astype(float)
        dr  = self.err_dir[:, k].astype(float)

        def pct(arr, q):
            return float(np.percentile(arr, q))

        return {
            "lead_h":   lead_h,
            "Hs_mean":  float(np.mean(hs)),  "Hs_std":  float(np.std(hs)),
            "Hs_p05":   pct(hs, 5),  "Hs_p25": pct(hs, 25),
            "Hs_p50":   pct(hs, 50), "Hs_p75": pct(hs, 75),  "Hs_p95": pct(hs, 95),
            "Tp_mean":  float(np.mean(tp)),  "Tp_std":  float(np.std(tp)),
            "Tp_p05":   pct(tp, 5),  "Tp_p25": pct(tp, 25),
            "Tp_p50":   pct(tp, 50), "Tp_p75": pct(tp, 75),  "Tp_p95": pct(tp, 95),
            "dir_mean": float(np.mean(dr)),  "dir_std": float(np.std(dr)),
            "dir_p05":  pct(dr, 5),  "dir_p25": pct(dr, 25),
            "dir_p50":  pct(dr, 50), "dir_p75": pct(dr, 75), "dir_p95": pct(dr, 95),
        }


# ---------------------------------------------------------------------------
# load_error_bank — builds ErrorTrajectoryBank from CSV
# ---------------------------------------------------------------------------

def load_error_bank(
    csv_path: str = DEFAULT_ERROR_CSV,
) -> ErrorTrajectoryBank:
    """
    Load the Module 4.2 empirical forecast-error dataset and return a bank.

    Reads the CSV, pivots each variable into (n_trajectories × n_leads)
    arrays, validates completeness, and stores as float32.

    Parameters
    ----------
    csv_path : str
        Path to forecast_errors.csv.gz.

    Returns
    -------
    ErrorTrajectoryBank

    Raises
    ------
    FileNotFoundError
        If csv_path does not exist.
    ValueError
        If required columns are missing or lead structure is incomplete.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Forecast-error CSV not found: {csv_path}\n"
            "Ensure Module 4.2 evaluation has been run."
        )

    required = {"lead_h", "sample_idx", "Hs_error", "Tp_error", "dir_error_deg"}
    df = pd.read_csv(csv_path)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"forecast_errors CSV missing columns: {sorted(missing)}")

    leads = sorted(df["lead_h"].unique())
    if leads != list(range(1, N_LEADS + 1)):
        raise ValueError(
            f"Expected lead_h values 1..{N_LEADS}, got {leads[:5]}...{leads[-5:]}"
        )

    # Pivot into (n_traj, n_leads) arrays — sorted by sample_idx then lead_h
    pivot_hs  = df.pivot(index="sample_idx", columns="lead_h", values="Hs_error")
    pivot_tp  = df.pivot(index="sample_idx", columns="lead_h", values="Tp_error")
    pivot_dir = df.pivot(index="sample_idx", columns="lead_h", values="dir_error_deg")

    if pivot_hs.isnull().any().any():
        raise ValueError("Hs_error pivot has NaN — some (sample_idx, lead_h) combinations missing.")
    if pivot_tp.isnull().any().any():
        raise ValueError("Tp_error pivot has NaN — some (sample_idx, lead_h) combinations missing.")
    if pivot_dir.isnull().any().any():
        raise ValueError("dir_error_deg pivot has NaN — some (sample_idx, lead_h) combinations missing.")

    # Ensure columns are sorted leads 1..48
    col_order = list(range(1, N_LEADS + 1))
    arr_hs  = pivot_hs[col_order].values.astype(np.float32)
    arr_tp  = pivot_tp[col_order].values.astype(np.float32)
    arr_dir = pivot_dir[col_order].values.astype(np.float32)

    n_traj = arr_hs.shape[0]

    return ErrorTrajectoryBank(
        err_hs         = arr_hs,
        err_tp         = arr_tp,
        err_dir        = arr_dir,
        n_trajectories = n_traj,
        n_leads        = N_LEADS,
        csv_path       = csv_path,
    )


# ---------------------------------------------------------------------------
# ForecastErrorSampler — main public API
# ---------------------------------------------------------------------------

class ForecastErrorSampler:
    """
    Samples empirical forecast-error trajectories and applies them to a
    deterministic GRU+wind forecast.

    ============================================================
    USAGE
    ============================================================

    1. Construction (once per RL training run):

        sampler = ForecastErrorSampler.from_csv(
            csv_path="results/forecasting/exp_gru_wind/forecast_errors.csv.gz",
            seed=42,
        )

    2. Per-episode reset (gives each episode an independent error draw):

        sampler.reset()   # advances the RNG, draws a new trajectory index

    3. Per-step application (inject errors into GRU deterministic output):

        realistic = sampler.apply(det_forecast)
        # det_forecast : np.ndarray, shape (48, 4), [Hs, Tp, sin_dir, cos_dir]
        # realistic    : same shape, with sampled errors applied

    ============================================================
    SAMPLING MECHANICS
    ============================================================

    reset() draws a single trajectory index from [0, n_trajectories) using
    the internal RNG.  apply() uses the errors from that trajectory,
    indexing by the position in the 48-step forecast horizon.

    The RNG is NOT re-seeded on every reset() — it advances naturally,
    so consecutive episodes see different error trajectories even with
    the same initial seed.  This is the correct behaviour for RL training.

    ============================================================
    ERROR INJECTION FORMULA
    ============================================================

    For forecast lead h (1..48):

        Hs_realistic[h]  = clip(Hs_det[h]  + err_hs[traj, h-1], HS_MIN, ∞)
        Tp_realistic[h]  = clip(Tp_det[h]  + err_tp[traj, h-1], TP_MIN, ∞)
        dir_realistic[h] = wrap360(dir_det[h] + err_dir[traj, h-1])
        sin_r, cos_r     = direction_to_sincos(dir_realistic)

    err_hs, err_tp, err_dir are from a single trajectory drawn once per
    episode, preserving temporal and joint-variable correlation structure.

    ============================================================
    ZERO-ERROR MODE
    ============================================================

    Setting scale=0.0 disables injection entirely:
        realistic == det_forecast  (exact equality)

    This is the identity transformation, useful for ablation studies.

    Parameters
    ----------
    bank : ErrorTrajectoryBank
        Pre-loaded trajectory bank.
    seed : int or None
        Seed for the internal RNG.  The RNG is seeded once at construction
        and advances naturally across episodes.
    scale : float
        Error magnitude scale factor.  Default 1.0 (full empirical errors).
        0.0 = no injection.  Values in (0, 1) give partial injection.
    """

    def __init__(
        self,
        bank:  ErrorTrajectoryBank,
        seed:  Optional[int] = None,
        scale: float         = 1.0,
    ) -> None:
        if scale < 0.0:
            raise ValueError(f"scale must be >= 0.0, got {scale}")
        self._bank       = bank
        self._scale      = scale
        self._rng        = np.random.default_rng(seed)
        # Current trajectory index (set by reset())
        self._traj_idx:  int           = 0
        self._hs_traj:   np.ndarray    = np.zeros(N_LEADS)
        self._tp_traj:   np.ndarray    = np.zeros(N_LEADS)
        self._dir_traj:  np.ndarray    = np.zeros(N_LEADS)
        self._has_been_reset: bool     = False
        # Clip counters (cumulative across all apply() calls)
        self._n_hs_clipped:  int = 0
        self._n_tp_clipped:  int = 0
        # Draw initial trajectory
        self.reset()

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_csv(
        cls,
        csv_path: str = DEFAULT_ERROR_CSV,
        seed:     Optional[int] = None,
        scale:    float         = 1.0,
    ) -> "ForecastErrorSampler":
        """
        Construct a ForecastErrorSampler by loading the error CSV.

        Parameters
        ----------
        csv_path : str
            Path to forecast_errors.csv.gz.
        seed : int or None
            Seed for the internal RNG.
        scale : float
            Error magnitude scale.  Default 1.0.

        Returns
        -------
        ForecastErrorSampler
        """
        bank = load_error_bank(csv_path)
        return cls(bank=bank, seed=seed, scale=scale)

    # ------------------------------------------------------------------
    # reset — draw a new trajectory for this episode
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Draw a new trajectory index from the error bank.

        This should be called at the start of each RL episode.  The RNG
        advances naturally, so consecutive resets produce different trajectories.

        The drawn trajectory index is used by all subsequent apply() calls
        until the next reset().
        """
        self._traj_idx = int(self._rng.integers(0, self._bank.n_trajectories))
        hs, tp, dr     = self._bank.get_trajectory(self._traj_idx)
        self._hs_traj  = hs * self._scale
        self._tp_traj  = tp * self._scale
        self._dir_traj = dr * self._scale
        self._has_been_reset = True

    # ------------------------------------------------------------------
    # apply — inject errors into a deterministic GRU forecast
    # ------------------------------------------------------------------

    def apply(
        self,
        det_forecast: np.ndarray,
    ) -> np.ndarray:
        """
        Apply sampled empirical errors to a deterministic GRU forecast.

        Parameters
        ----------
        det_forecast : np.ndarray, shape (H_FORE, 4)
            Deterministic GRU output in physical units:
            columns [Hs [m], Tp [s], sin(dir), cos(dir)].
            H_FORE must equal N_LEADS = 48.

        Returns
        -------
        np.ndarray, shape (48, 4)
            Realistic forecast with sampled errors applied.
            Columns: [Hs [m], Tp [s], sin(dir_realistic), cos(dir_realistic)].
            Hs >= 0, Tp >= TP_MIN, direction in [0, 360).

        Raises
        ------
        ValueError
            If det_forecast has wrong shape.
        """
        if not self._has_been_reset:
            self.reset()

        det_forecast = np.asarray(det_forecast, dtype=np.float64)
        if det_forecast.shape != (N_LEADS, 4):
            raise ValueError(
                f"det_forecast must have shape ({N_LEADS}, 4), got {det_forecast.shape}"
            )

        # If scale == 0, return exact copy of deterministic forecast
        if self._scale == 0.0:
            return det_forecast.copy()

        out = det_forecast.copy()

        # ---- Hs (column 0) ----
        Hs_det  = out[:, 0]
        Hs_real = Hs_det + self._hs_traj
        n_clipped_hs = int(np.sum(Hs_real < HS_MIN))
        self._n_hs_clipped += n_clipped_hs
        Hs_real = np.maximum(Hs_real, HS_MIN)
        out[:, 0] = Hs_real

        # ---- Tp (column 1) ----
        Tp_det  = out[:, 1]
        Tp_real = Tp_det + self._tp_traj
        n_clipped_tp = int(np.sum(Tp_real <= TP_MIN))
        self._n_tp_clipped += n_clipped_tp
        Tp_real = np.maximum(Tp_real, TP_MIN)
        out[:, 1] = Tp_real

        # ---- Direction (columns 2, 3) ----
        # Reconstruct dir_pred_deg from (sin, cos)
        sin_det = out[:, 2]
        cos_det = out[:, 3]
        dir_det_deg = _sincos_to_deg(sin_det, cos_det)

        # Apply circular error and wrap to [0, 360)
        dir_real_deg = _wrap360(dir_det_deg + self._dir_traj)

        # Convert back to (sin, cos) using Module 4 convention
        sin_real, cos_real = direction_to_sincos(dir_real_deg)
        out[:, 2] = sin_real
        out[:, 3] = cos_real

        # Final NaN/Inf guard (should never trigger in practice)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

        return out

    # ------------------------------------------------------------------
    # Zero-error probe (for testing)
    # ------------------------------------------------------------------

    def apply_zero_error(self, det_forecast: np.ndarray) -> np.ndarray:
        """
        Return the deterministic forecast unchanged (zero-error identity).

        Useful for verifying that apply(scale=0) == det_forecast.
        """
        return np.asarray(det_forecast, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def bank(self) -> ErrorTrajectoryBank:
        """The underlying ErrorTrajectoryBank."""
        return self._bank

    @property
    def scale(self) -> float:
        """Current error magnitude scale factor."""
        return self._scale

    @property
    def current_trajectory_index(self) -> int:
        """The trajectory index drawn at the last reset()."""
        return self._traj_idx

    @property
    def n_hs_clipped(self) -> int:
        """Cumulative count of Hs values clipped to 0."""
        return self._n_hs_clipped

    @property
    def n_tp_clipped(self) -> int:
        """Cumulative count of Tp values clipped to TP_MIN."""
        return self._n_tp_clipped

    @property
    def current_hs_errors(self) -> np.ndarray:
        """
        Hs error trajectory for the current episode, shape (48,).
        Element k is the error applied at forecast lead k+1.
        Returns scaled errors.
        """
        return self._hs_traj.copy()

    @property
    def current_tp_errors(self) -> np.ndarray:
        """Tp error trajectory for the current episode, shape (48,)."""
        return self._tp_traj.copy()

    @property
    def current_dir_errors(self) -> np.ndarray:
        """Direction error trajectory [deg] for the current episode, shape (48,)."""
        return self._dir_traj.copy()

    def summary(self) -> dict:
        """
        Return a summary dict describing the sampler configuration and state.
        """
        return {
            "csv_path":             self._bank.csv_path,
            "n_trajectories":       self._bank.n_trajectories,
            "n_leads":              self._bank.n_leads,
            "scale":                self._scale,
            "current_traj_idx":     self._traj_idx,
            "n_hs_clipped_total":   self._n_hs_clipped,
            "n_tp_clipped_total":   self._n_tp_clipped,
            "sampling_method":      "trajectory_resample_with_replacement",
            "temporal_structure":   "preserved (full 48-lead trajectory per draw)",
            "joint_structure":      "preserved (Hs/Tp/dir errors from same trajectory)",
            "error_sign":           "pred_minus_true (positive=overestimate)",
            "direction_convention": "circular, applied in degree space, result in [0,360)",
        }
