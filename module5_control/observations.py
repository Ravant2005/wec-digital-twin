"""
observations.py — Observation space definition and builder (Module 5.1).

============================================================
OBSERVATION STRUCTURE
============================================================

The Tier-1 RL observation is a flat float32 vector composed of four
conceptual blocks:

    Block A — Observed history (H_OBS = 48 hourly steps)
    -------------------------------------------------------
    Per step (wave_wind mode, 11 features each):
        0   Hs_obs          [m]      (NaN → 0 when invalid)
        1   Tp_obs          [s]      (NaN → 0 when invalid)
        2   sin(dir_obs)    [-1,1]
        3   cos(dir_obs)    [-1,1]
        4   Hs_valid        {0,1}    validity mask
        5   Tp_valid        {0,1}    validity mask
        6   dir_valid       {0,1}    validity mask
        7   u10             [m/s]    ERA5 wind (not sensor-noisy)
        8   v10             [m/s]    ERA5 wind (not sensor-noisy)
        9   u10_valid       {0,1}    1 if finite, 0 otherwise
       10   v10_valid       {0,1}    1 if finite, 0 otherwise
    Flattened: H_OBS × 11 = 48 × 11 = 528 values

    Block B — Forecast (H_FORE = 48 hourly steps)
    -----------------------------------------------
    Per step (4 values each):
        0   Hs_forecast     [m]      (zero-padded if unavailable)
        1   Tp_forecast     [s]
        2   sin(dir_fore)   [-1,1]
        3   cos(dir_fore)   [-1,1]
    Flattened: H_FORE × 4 = 48 × 4 = 192 values

    Block C — WEC state (3 values)
    --------------------------------
        0   heave_position x    [m]   normalised by x_scale
        1   heave_velocity v    [m/s] normalised by v_scale
        2   latch_status        {0,1} (float)

    -------------------------------------------------------
    TOTAL DIMENSION: 528 + 192 + 3 = 723
    -------------------------------------------------------

Direction representation:
    Direction is always encoded as (sin, cos) of the angle in radians,
    consistent with the Module 4 dataset convention:
        sin_dir = sin(direction_deg × π / 180)
        cos_dir = cos(direction_deg × π / 180)
    Raw degree angles are NEVER placed in the observation vector.

Normalisation:
    Hs_obs / Tp_obs are z-score normalised using the Module 4 scaler
    statistics (if a scaler is provided), keeping the network on the
    same scale it was trained on.
    wind u10/v10 are z-score normalised similarly.
    Forecast components use the same scaler statistics.
    WEC state uses fixed physical scale factors (x_scale, v_scale).

Information mode enforcement:
    - "perfect_forecast":  Block B filled from true future ERA5 values.
    - "realistic_forecast": Block B filled from GRU+wind deterministic
                             prediction (no stochastic error injection yet).
    - "reactive":           Block B is all zeros (forecast withheld).

    True wave values NEVER appear in Block A or B in realistic/reactive
    modes.  They are available in the info dict for debug/evaluation only.

============================================================
DIMENSIONALITY SUMMARY
============================================================

    H_OBS  = 48  (2-day lookback at 1-hour resolution)
    F_OBS  = 11  (wave_wind feature count from Module 4)
    H_FORE = 48  (2-day forecast horizon)
    F_FORE = 4   (Hs, Tp, sin_dir, cos_dir)
    N_WEC  = 3   (x, v, latch)

    OBS_DIM = H_OBS × F_OBS + H_FORE × F_FORE + N_WEC
            = 528 + 192 + 3 = 723

============================================================
DEPENDENCIES (read-only — no module modifications)
============================================================
    module4_forecasting.dataset_v2  — WaveScalerV2
    module4_forecasting.dataset     — direction_to_sincos
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from module4_forecasting.dataset import direction_to_sincos
from module4_forecasting.dataset_v2 import WaveScalerV2

# ---------------------------------------------------------------------------
# Dimensionality constants
# ---------------------------------------------------------------------------

#: Observation history length [hours].  Must match Module 4 input_length.
H_OBS: int = 48

#: Forecast horizon [hours].  Must match Module 4 forecast_horizon.
H_FORE: int = 48

#: Number of wave_wind features per history step (Module 4 schema).
F_OBS_WAVE_WIND: int = 11

#: Number of forecast output channels per step.
F_FORE: int = 4  # Hs, Tp, sin_dir, cos_dir

#: Number of WEC state components.
N_WEC: int = 3  # x, v, latch_status

#: Total observation dimension (wave_wind mode).
OBS_DIM: int = H_OBS * F_OBS_WAVE_WIND + H_FORE * F_FORE + N_WEC
# = 528 + 192 + 3 = 723

# Slice indices for each block in the flat observation vector
_BLOCK_A_START = 0
_BLOCK_A_END   = H_OBS * F_OBS_WAVE_WIND          # 528
_BLOCK_B_START = _BLOCK_A_END
_BLOCK_B_END   = _BLOCK_A_END + H_FORE * F_FORE   # 720
_BLOCK_C_START = _BLOCK_B_END
_BLOCK_C_END   = _BLOCK_B_END + N_WEC              # 723

assert OBS_DIM == _BLOCK_C_END, "Observation dimension mismatch"

# ---------------------------------------------------------------------------
# Physical scale factors for WEC state normalisation
# ---------------------------------------------------------------------------

#: Heave displacement scale [m].  ±2× this value covers typical buoy range.
X_SCALE: float = 2.0

#: Heave velocity scale [m/s].  ±2× this value covers typical velocity range.
V_SCALE: float = 1.0


# ---------------------------------------------------------------------------
# ObservationConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObservationConfig:
    """
    Configuration for the observation builder.

    Parameters
    ----------
    obs_history_len : int
        Number of past hourly sea-state steps in Block A.
        Must equal Module 4 input_length (48).
    forecast_len : int
        Number of forecast steps in Block B.
        Must equal Module 4 forecast_horizon (48).
    x_scale : float
        WEC heave position normalisation scale [m].
        Observation value = x / x_scale.  Clip to [-1, 1] not enforced
        (the agent sees out-of-range values as signal of unusual state).
    v_scale : float
        WEC heave velocity normalisation scale [m/s].
    use_wind : bool
        If True, Block A includes wind features (F_OBS = 11).
        If False, Block A uses wave_only features (F_OBS = 7) and
        OBS_DIM = 48×7 + 48×4 + 3 = 531.
        Currently always True (canonical GRU checkpoint uses wave_wind).
    """

    obs_history_len: int   = H_OBS
    forecast_len:    int   = H_FORE
    x_scale:         float = X_SCALE
    v_scale:         float = V_SCALE
    use_wind:        bool  = True

    @property
    def n_obs_features(self) -> int:
        """Number of features per history step (11 wave_wind, 7 wave_only)."""
        return F_OBS_WAVE_WIND if self.use_wind else 7

    @property
    def obs_dim(self) -> int:
        """Total flattened observation dimension."""
        return (
            self.obs_history_len * self.n_obs_features
            + self.forecast_len  * F_FORE
            + N_WEC
        )

    def __post_init__(self) -> None:
        if self.obs_history_len != H_OBS:
            raise ValueError(
                f"obs_history_len must equal H_OBS={H_OBS} "
                f"(Module 4 input_length), got {self.obs_history_len}"
            )
        if self.forecast_len != H_FORE:
            raise ValueError(
                f"forecast_len must equal H_FORE={H_FORE} "
                f"(Module 4 forecast_horizon), got {self.forecast_len}"
            )
        if self.x_scale <= 0:
            raise ValueError(f"x_scale must be > 0, got {self.x_scale}")
        if self.v_scale <= 0:
            raise ValueError(f"v_scale must be > 0, got {self.v_scale}")


# ---------------------------------------------------------------------------
# build_observation_space
# ---------------------------------------------------------------------------

def build_observation_space(config: ObservationConfig) -> spaces.Box:
    """
    Build the Gymnasium Box observation space.

    The observation is a flat float32 vector of length config.obs_dim.
    Bounds are set to [-inf, +inf] because normalisation may place
    extreme sea states outside any finite bound.  The agent (PPO/DQN)
    will clip or handle these internally.

    Parameters
    ----------
    config : ObservationConfig

    Returns
    -------
    gymnasium.spaces.Box
        dtype float32, shape (obs_dim,), bounds (-inf, +inf).
    """
    dim = config.obs_dim
    return spaces.Box(
        low  = np.full(dim, -np.inf, dtype=np.float32),
        high = np.full(dim,  np.inf, dtype=np.float32),
        shape=(dim,),
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# ObservationBuilder
# ---------------------------------------------------------------------------

class ObservationBuilder:
    """
    Builds the flat observation vector from raw environment state.

    This class is stateless with respect to the RL episode — it does not
    maintain a rolling buffer itself.  The caller (WECControlEnv) passes
    the current history and forecast arrays.

    Parameters
    ----------
    config : ObservationConfig
    scaler : WaveScalerV2 or None
        Fitted scaler from Module 4 for normalising Hs, Tp, u10, v10.
        If None, features are used raw (not recommended in practice).
    """

    def __init__(
        self,
        config: ObservationConfig,
        scaler: Optional[WaveScalerV2] = None,
    ) -> None:
        self.config = config
        self.scaler = scaler

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        obs_history:   np.ndarray,
        forecast:      np.ndarray,
        x:             float,
        v:             float,
        latch_status:  int,
    ) -> np.ndarray:
        """
        Assemble the flat observation vector.

        Parameters
        ----------
        obs_history : np.ndarray, shape (H_OBS, n_obs_features)
            Block A: rolling window of sensor observations.
            Features follow the Module 4 wave_wind schema:
              [Hs_obs, Tp_obs, sin_dir_obs, cos_dir_obs,
               Hs_valid, Tp_valid, dir_valid,
               u10, v10, u10_valid, v10_valid]  (wave_wind)
            NaN values represent missing observations.
        forecast : np.ndarray, shape (H_FORE, 4)
            Block B: forecast of (Hs, Tp, sin_dir, cos_dir).
            Zeros if mode="reactive".
        x : float
            WEC heave displacement [m].
        v : float
            WEC heave velocity [m/s].
        latch_status : int
            0 = FREE, 1 = LATCHED.

        Returns
        -------
        np.ndarray, shape (obs_dim,), dtype float32
            Flat observation vector with no NaN values.
        """
        cfg = self.config

        # --- Block A: history ---
        A = self._normalise_history(obs_history.copy())   # (H_OBS, F_OBS)
        # Replace any remaining NaN with 0.0 (training-mean proxy after z-score)
        A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
        A_flat = A.reshape(-1)  # (H_OBS * F_OBS,)

        # --- Block B: forecast ---
        B = self._normalise_forecast(forecast.copy())     # (H_FORE, 4)
        B = np.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
        B_flat = B.reshape(-1)  # (H_FORE * 4,)

        # --- Block C: WEC state ---
        C = np.array([
            x / cfg.x_scale,
            v / cfg.v_scale,
            float(latch_status),
        ], dtype=np.float64)
        C = np.nan_to_num(C, nan=0.0)

        # --- Concatenate and cast ---
        obs = np.concatenate([A_flat, B_flat, C]).astype(np.float32)

        assert len(obs) == cfg.obs_dim, (
            f"Observation dimension mismatch: {len(obs)} != {cfg.obs_dim}"
        )
        return obs

    # ------------------------------------------------------------------
    # Direction encoding
    # ------------------------------------------------------------------

    @staticmethod
    def direction_to_sincos(direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert direction [degrees] to (sin, cos) pair.

        Delegates to the Module 4 utility to guarantee consistency with
        the training convention.

        Parameters
        ----------
        direction_deg : np.ndarray
            Direction in degrees (ERA5 convention).  May contain NaN.

        Returns
        -------
        sin_dir, cos_dir : np.ndarray
            Each shape matching input.  NaN in → NaN out.
        """
        # direction_to_sincos from dataset.py handles NaN via numpy math
        return direction_to_sincos(direction_deg)

    # ------------------------------------------------------------------
    # Normalisation helpers
    # ------------------------------------------------------------------

    def _normalise_history(self, A: np.ndarray) -> np.ndarray:
        """
        Z-score normalise the continuous features in Block A.

        Normalises Hs (idx 0), Tp (idx 1), u10 (idx 7), v10 (idx 8)
        using the Module 4 scaler statistics.  All other features
        (sin/cos direction, validity masks) are passed through unchanged.

        NaN values are preserved for the nan_to_num call in build().
        """
        if self.scaler is None or not self.scaler.is_fitted:
            return A
        # scaler.transform expects (..., n_features); A is (H_OBS, F_OBS)
        return self.scaler.transform(A)

    def _normalise_forecast(self, B: np.ndarray) -> np.ndarray:
        """
        Normalise the forecast block (shape H_FORE × 4).

        Forecast target layout (from Module 4):
            idx 0 : Hs    — z-score using scaler mean[0] / std[0]
            idx 1 : Tp    — z-score using scaler mean[1] / std[1]
            idx 2 : sin_dir — no normalisation (already [-1,1])
            idx 3 : cos_dir — no normalisation (already [-1,1])

        Uses the scaler's transform_target() which applies the same
        Hs/Tp statistics.
        """
        if self.scaler is None or not self.scaler.is_fitted:
            return B
        return self.scaler.transform_target(B)

    # ------------------------------------------------------------------
    # Zero observation (for reset / padding)
    # ------------------------------------------------------------------

    def zero_observation(self) -> np.ndarray:
        """
        Return a zero observation vector.

        Used to initialise the observation buffer at episode start before
        any real observations are available.
        """
        return np.zeros(self.config.obs_dim, dtype=np.float32)
