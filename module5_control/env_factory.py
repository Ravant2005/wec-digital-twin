"""
env_factory.py — Environment factory for PPO training (Module 5.3B).

============================================================
PURPOSE
============================================================

Factory functions that construct WECControlEnv instances and
SB3-compatible vectorised environments with VecNormalize.

Centralises:
  1. BEM / Cummins / PTO parameter construction (cached once).
  2. EpisodeReplay construction for train / validation / test splits.
  3. ForecastErrorSampler construction with error-bank safety check.
  4. SB3 DummyVecEnv + VecNormalize wrapping.
  5. GRU checkpoint path management.

This module does NOT train anything. It is the reusable plumbing used
by train_ppo.py, evaluate_ppo.py, and the test suite.

============================================================
FORECAST MODES
============================================================

reactive
    Block B = all zeros. No forecast information. No GRU. No error sampler.

perfect_forecast
    Block B = true future ERA5 (oracle upper bound). No GRU. No error sampler.
    Requires replay with H_OBS + episode + H_FORE hours.

realistic_forecast
    Block B = frozen GRU+wind deterministic output + 5.2B sampled errors.
    Requires:
      - GRU checkpoint (canonical best_model.pt)
      - Training-safe error bank (forecast_errors_val2022_2023.csv.gz)
        assert_safe_training_error_bank() guards against 2024–2025 leakage.

============================================================
VECNORMALIZE CONTRACT
============================================================

Training wrapper:
    VecNormalize(env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    norm_obs=False because:
      Block A/B Hs/Tp/u10/v10 are already z-scored by Module 4 scaler.
      sin/cos/direction/latch validity are already in [-1,1] or {0,1}.
      Block C x/v already divided by X_SCALE / V_SCALE.
      Double-normalisation would distort the feature scale.

    norm_reward=True because:
      Per-step reward ~ O(50–70) at P_ref=1000W — too large for PPO.
      VecNormalize adaptively whitens the reward signal.

    clip_reward=10.0 clips normalised rewards to [-10, 10] to prevent
      rare outlier step-rewards from destabilising value-function learning.

Evaluation wrapper:
    VecNormalize.load(...) with training=False and norm_reward=False.
    Statistics frozen — no update on validation/test data.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import gymnasium as gym

try:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

from module2_wec.geometry import REFERENCE_BUOY
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.hydrodynamic_coefficients import (
    compute_hydrodynamic_coefficients,
    HydrodynamicCoefficients,
)
from module2_wec.radiation import compute_radiation_kernel
from module2_wec.cummins import CumminsParameters
from module2_wec.pto import PTOParameters
from module3_sensor.sensor import SensorParameters

from module5_control.environment import (
    WECControlEnv,
    EnvironmentConfig,
    build_env_from_checkpoint,
)
from module5_control.replay import (
    EpisodeReplay,
    build_episode_replay,
    build_synthetic_replay,
    DT_PHYSICS,
    DT_CONTROL,
)
from module5_control.observations import ObservationConfig, OBS_DIM
from module5_control.rewards import RewardConfig
from module5_control.forecast_uncertainty import ForecastErrorSampler
from module5_control.rl_config import (
    RL_DATA_SPLIT,
    RL_TRAIN_YEARS,
    RL_VALIDATION_YEARS,
    RL_TEST_YEARS,
    RL_EPISODE_HOURS,
    RL_MIN_REPLAY_HOURS_FORECAST,
    RL_MIN_REPLAY_HOURS_REACTIVE,
    PPOConfig,
    assert_safe_training_error_bank,
    ERROR_BANK_TRAIN_DEFAULT,
    GRU_CHECKPOINT_PATH,
)


# ---------------------------------------------------------------------------
# Cached WEC physical parameters (BEM is expensive — ~15 s)
# ---------------------------------------------------------------------------

_WEC_PARAMS_CACHE: Optional[Dict[str, Any]] = None
_ERA5_ARCHIVE_CACHE: Optional[Any] = None


class SeededResetWrapper(gym.Wrapper):
    """Provide a deterministic reset seed when SB3 omits ``seed``."""

    def __init__(self, env: gym.Env, reset_seed: int) -> None:
        super().__init__(env)
        self._reset_seed = int(reset_seed)

    def reset(self, **kwargs):
        kwargs.setdefault("seed", self._reset_seed)
        return self.env.reset(**kwargs)

    def __getattr__(self, name: str):
        return getattr(self.env, name)


def build_wec_params(
    resolution=(6, 24, 16),
    omega_min=0.2,
    omega_max=1.4,
    n_omega=20,
) -> Dict[str, Any]:
    """
    Build all WEC physical parameters needed by the environment.

    Results are cached after the first call. BEM computation is ~15 s
    on a modern CPU — caching avoids repeated Capytaine runs.

    Returns
    -------
    dict with keys:
        cummins   : CumminsParameters
        pto       : PTOParameters   (B_PTO=200 kN·s/m, K_PTO=0)
        hydro_amp : np.ndarray      excitation_force_heave_amplitude
        hydro_omega : np.ndarray    omega grid
        hydro     : HydrodynamicCoefficients (for Module 5.2A excitation)
    """
    global _WEC_PARAMS_CACHE
    if _WEC_PARAMS_CACHE is not None:
        return _WEC_PARAMS_CACHE

    hydro: HydrodynamicCoefficients = compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY,
        resolution=resolution,
        omega_min=omega_min,
        omega_max=omega_max,
        n_omega=n_omega,
        progress_bar=False,
    )
    hs_obj = Hydrostatics(REFERENCE_BUOY)
    rk = compute_radiation_kernel(hydro)

    cummins = CumminsParameters(
        mass                  = REFERENCE_BUOY.mass,
        hydrostatic_stiffness = hs_obj.hydrostatic_stiffness,
        added_mass_infinity   = rk.A_infinity,
        kernel_time           = rk.time,
        kernel_values         = rk.kernel,
    )
    pto = PTOParameters(damping=200_000.0, stiffness=0.0)

    _WEC_PARAMS_CACHE = {
        "cummins":     cummins,
        "pto":         pto,
        "hydro_amp":   hydro.excitation_force_heave_amplitude,
        "hydro_omega": hydro.omega,
        "hydro":       hydro,
    }
    return _WEC_PARAMS_CACHE


def clear_wec_params_cache() -> None:
    """Force next build_wec_params() call to recompute BEM."""
    global _WEC_PARAMS_CACHE
    _WEC_PARAMS_CACHE = None


# ---------------------------------------------------------------------------
# Replay builder
# ---------------------------------------------------------------------------

def _load_era5_archive_cached() -> Any:
    """Load the canonical 2010–2025 ERA5 archive once per process."""
    global _ERA5_ARCHIVE_CACHE
    if _ERA5_ARCHIVE_CACHE is None:
        from module4_forecasting.era5_archive import load_era5_archive

        _ERA5_ARCHIVE_CACHE = load_era5_archive(
            start_year=2010,
            end_year=2025,
            allow_missing_months=False,
        )
    return _ERA5_ARCHIVE_CACHE


def _era5_split_frame(
    split: str,
    era5_df: Optional[Any] = None,
) -> Any:
    """Return a chronological split without allowing cross-split windows."""
    split_years = {
        "train": RL_TRAIN_YEARS,
        "validation": RL_VALIDATION_YEARS,
        "test": RL_TEST_YEARS,
    }
    if split not in split_years:
        raise ValueError(f"split must be one of {sorted(split_years)}, got {split!r}")

    df = _load_era5_archive_cached() if era5_df is None else era5_df
    times = df["time"]
    start_year, end_year = split_years[split]
    mask = (times.dt.year >= start_year) & (times.dt.year <= end_year)
    result = df.loc[mask].reset_index(drop=True)
    if len(result) == 0:
        raise ValueError(f"No ERA5 rows found for split={split!r}.")
    return result


def _sample_replay_start_indices(
    n_rows: int,
    n_hours: int,
    n_envs: int,
    seed: int,
) -> List[int]:
    """Sample distinct continuous replay windows within one chronological split."""
    # 48 hours of context are required before the first RL action. Forecast
    # modes also retain 48 hours after the episode for Block B.
    min_start = 48
    max_start = n_rows - n_hours
    if max_start < min_start:
        raise ValueError(
            f"Split has {n_rows} hourly rows, but {n_hours} hours are required "
            f"for one replay window."
        )

    candidates = np.arange(min_start, max_start + 1, dtype=int)
    if n_envs > len(candidates):
        raise ValueError(
            f"Cannot create {n_envs} distinct replay windows from {len(candidates)} "
            f"valid start indices."
        )
    rng = np.random.default_rng(seed)
    return [int(value) for value in rng.choice(candidates, size=n_envs, replace=False)]


def build_replay_for_mode(
    mode: str,
    episode_hours: int = RL_EPISODE_HOURS,
    seed: int = 0,
    use_synthetic: bool = True,
    split: str = "train",
    start_index: Optional[int] = None,
    era5_df: Optional[Any] = None,
) -> EpisodeReplay:
    """
    Build a replay with enough context/horizon for the forecast mode.

    ``split`` is enforced for real ERA5 replays. ``start_index`` is relative to
    the selected split, so no replay window can cross train/validation/test
    boundaries. Smoke tests may explicitly select ``use_synthetic=True``.
    """
    if episode_hours <= 0:
        raise ValueError(f"episode_hours must be positive, got {episode_hours}")

    if mode == "reactive":
        n_hours = max(
            RL_MIN_REPLAY_HOURS_REACTIVE,
            48 + episode_hours,
        )
    elif mode in ("perfect_forecast", "realistic_forecast"):
        n_hours = max(
            RL_MIN_REPLAY_HOURS_FORECAST,
            48 + episode_hours + 48,
        )
    else:
        raise ValueError(
            "mode must be 'reactive', 'perfect_forecast', or 'realistic_forecast', "
            f"got {mode!r}"
        )

    if use_synthetic:
        return build_synthetic_replay(
            n_hours=n_hours,
            Hs=1.5,
            Tp=8.0,
            seed=seed,
        )

    split_df = _era5_split_frame(split=split, era5_df=era5_df)
    if start_index is None:
        start_index = _sample_replay_start_indices(
            n_rows=len(split_df),
            n_hours=n_hours,
            n_envs=1,
            seed=seed,
        )[0]
    elif start_index < 48 or start_index + n_hours > len(split_df):
        raise ValueError(
            f"start_index={start_index} is outside the {split} split for a "
            f"{n_hours}-hour replay (valid range 48..{len(split_df) - n_hours})."
        )

    return build_episode_replay(
        df=split_df,
        start_index=start_index,
        n_hours=n_hours,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# ForecastErrorSampler builder with safety check
# ---------------------------------------------------------------------------

def build_forecast_error_sampler(
    mode: str,
    error_bank_path: Optional[str] = None,
    seed: int = 42,
    allow_final_test_bank: bool = False,
) -> Optional[ForecastErrorSampler]:
    """
    Build a ForecastErrorSampler for realistic_forecast mode.

    For reactive/perfect modes returns None (no error injection needed).

    Parameters
    ----------
    mode : str
        "reactive", "perfect_forecast", or "realistic_forecast".
    error_bank_path : str or None
        If None and mode == "realistic_forecast", defaults to the
        training-safe 2022–2023 bank.
    seed : int
        RNG seed for the sampler.
    allow_final_test_bank : bool
        If True, skip the safety assertion. Used ONLY for final frozen-policy
        evaluation on the 2024–2025 bank. Default False.

    Returns
    -------
    ForecastErrorSampler or None
    """
    if mode != "realistic_forecast":
        return None

    csv_path = error_bank_path or ERROR_BANK_TRAIN_DEFAULT

    if not allow_final_test_bank:
        assert_safe_training_error_bank(csv_path)

    return ForecastErrorSampler.from_csv(csv_path=csv_path, seed=seed)


# ---------------------------------------------------------------------------
# Single environment builder
# ---------------------------------------------------------------------------

def build_single_env(
    mode: str = "reactive",
    max_episode_steps: int = 7200,
    replay_seed: int = 0,
    env_rng_seed: Optional[int] = None,
    error_bank_path: Optional[str] = None,
    gru_checkpoint: Optional[str] = None,
    allow_final_test_bank: bool = False,
    error_sampler_seed: int = 42,
    use_synthetic: bool = True,
    split: str = "train",
    start_index: Optional[int] = None,
    episode_hours: int = RL_EPISODE_HOURS,
) -> gym.Env:
    """
    Build a single WECControlEnv instance.

    ``use_synthetic=True`` is reserved for smoke tests and unit tests. Production
    callers should pass ``use_synthetic=False`` and a chronological ``split``.
    """
    wec = build_wec_params()
    replay = build_replay_for_mode(
        mode=mode,
        episode_hours=episode_hours,
        seed=replay_seed,
        use_synthetic=use_synthetic,
        split=split,
        start_index=start_index,
    )

    sampler = build_forecast_error_sampler(
        mode=mode,
        error_bank_path=error_bank_path,
        seed=error_sampler_seed,
        allow_final_test_bank=allow_final_test_bank,
    )

    gru_path = gru_checkpoint if mode == "realistic_forecast" else None
    if mode == "realistic_forecast" and gru_path is None:
        gru_path = GRU_CHECKPOINT_PATH

    env = build_env_from_checkpoint(
        replay                    = replay,
        cummins_params            = wec["cummins"],
        pto_params                = wec["pto"],
        hydro_excitation_amplitude= wec["hydro_amp"],
        hydro_omega               = wec["hydro_omega"],
        mode                      = mode,
        gru_checkpoint_path       = gru_path,
        max_episode_steps         = max_episode_steps,
        hydro                     = wec["hydro"],
        forecast_error_sampler    = sampler,
    )

    reset_seed = replay_seed if env_rng_seed is None else env_rng_seed
    return SeededResetWrapper(env, reset_seed=reset_seed)


# ---------------------------------------------------------------------------
# SB3 vectorised environment factory
# ---------------------------------------------------------------------------

def make_env_fn(
    mode: str = "reactive",
    max_episode_steps: int = 7200,
    replay_seed_offset: int = 0,
    error_bank_path: Optional[str] = None,
    gru_checkpoint: Optional[str] = None,
    allow_final_test_bank: bool = False,
    seed: int = 42,
    split: str = "train",
    use_synthetic: bool = True,
    episode_hours: int = RL_EPISODE_HOURS,
    replay_start_index: Optional[int] = None,
) -> Callable[[], gym.Env]:
    """Return a thunk that builds a fresh, independently seeded environment."""
    def _thunk() -> gym.Env:
        return build_single_env(
            mode                  = mode,
            max_episode_steps     = max_episode_steps,
            replay_seed           = seed + replay_seed_offset,
            env_rng_seed          = seed + replay_seed_offset,
            error_bank_path       = error_bank_path,
            gru_checkpoint        = gru_checkpoint,
            allow_final_test_bank = allow_final_test_bank,
            error_sampler_seed    = seed + replay_seed_offset,
            use_synthetic         = use_synthetic,
            split                 = split,
            episode_hours         = episode_hours,
            start_index           = replay_start_index,
        )
    return _thunk


def build_vec_env(
    mode: str = "reactive",
    n_envs: int = 4,
    max_episode_steps: int = 7200,
    error_bank_path: Optional[str] = None,
    gru_checkpoint: Optional[str] = None,
    allow_final_test_bank: bool = False,
    use_subproc: bool = False,
    seed: int = 42,
    split: str = "train",
    use_synthetic: bool = True,
    episode_hours: int = RL_EPISODE_HOURS,
    replay_start_indices: Optional[Sequence[int]] = None,
) -> "DummyVecEnv":
    """Build a vectorised environment with distinct replay windows and seeds."""
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable_baselines3 is not installed. "
            "Run: pip install stable-baselines3>=2.0.0,<3.0.0"
        )
    if n_envs <= 0:
        raise ValueError(f"n_envs must be positive, got {n_envs}")

    if replay_start_indices is None:
        if use_synthetic:
            start_indices = [None] * n_envs
        else:
            # The replay length is selected below; use a conservative forecast
            # length so the sampled indices are valid for every mode.
            n_hours = (
                RL_MIN_REPLAY_HOURS_FORECAST
                if mode != "reactive"
                else RL_MIN_REPLAY_HOURS_REACTIVE
            )
            split_df = _era5_split_frame(split=split)
            start_indices = _sample_replay_start_indices(
                n_rows=len(split_df),
                n_hours=n_hours,
                n_envs=n_envs,
                seed=seed,
            )
    else:
        start_indices = list(replay_start_indices)
        if len(start_indices) != n_envs:
            raise ValueError(
                f"replay_start_indices has length {len(start_indices)}, expected {n_envs}"
            )

    fns = [
        make_env_fn(
            mode                  = mode,
            max_episode_steps     = max_episode_steps,
            replay_seed_offset    = i,
            error_bank_path       = error_bank_path,
            gru_checkpoint        = gru_checkpoint,
            allow_final_test_bank = allow_final_test_bank,
            seed                  = seed,
            split                 = split,
            use_synthetic         = use_synthetic,
            episode_hours         = episode_hours,
            replay_start_index    = start_indices[i],
        )
        for i in range(n_envs)
    ]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(fns, start_method="fork")
    return DummyVecEnv(fns)


def wrap_vec_normalize(
    vec_env: "DummyVecEnv",
    norm_obs: bool = False,
    norm_reward: bool = True,
    clip_obs: float = 10.0,
    clip_reward: float = 10.0,
    training: bool = True,
) -> "VecNormalize":
    """
    Wrap a vectorised environment with SB3 VecNormalize.

    Training default: norm_obs=False, norm_reward=True, clip_reward=10.0.
    Observation normalisation is OFF because the 723-dim observation is
    already z-scored by the Module 4 scaler. Double-normalisation would
    distort the feature scale and is explicitly prohibited.

    Evaluation default (training=False, norm_reward=False):
    Load saved statistics, freeze them — no adaptation on eval data.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable_baselines3 is not installed.")

    return VecNormalize(
        vec_env,
        norm_obs   = norm_obs,
        norm_reward= norm_reward,
        clip_obs   = clip_obs,
        clip_reward= clip_reward,
        training   = training,
    )


def build_training_env(
    mode: str = "reactive",
    n_envs: int = 4,
    max_episode_steps: int = 7200,
    error_bank_path: Optional[str] = None,
    seed: int = 42,
    split: str = "train",
    use_synthetic: bool = False,
    episode_hours: int = RL_EPISODE_HOURS,
    replay_start_indices: Optional[Sequence[int]] = None,
) -> "VecNormalize":
    """Build the production training environment on the requested split."""
    if split != "train" and not use_synthetic:
        raise ValueError("Training environments must use split='train'.")
    vec_env = build_vec_env(
        mode                  = mode,
        n_envs                = n_envs,
        max_episode_steps     = max_episode_steps,
        error_bank_path       = error_bank_path,
        allow_final_test_bank = False,
        seed                  = seed,
        split                 = split,
        use_synthetic         = use_synthetic,
        episode_hours         = episode_hours,
        replay_start_indices  = replay_start_indices,
    )
    return wrap_vec_normalize(
        vec_env,
        norm_obs   = False,
        norm_reward= True,
        clip_reward= 10.0,
        training   = True,
    )


def build_eval_env(
    mode: str = "reactive",
    n_envs: int = 1,
    max_episode_steps: int = 7200,
    vec_normalize_path: Optional[str] = None,
    error_bank_path: Optional[str] = None,
    allow_final_test_bank: bool = False,
    seed: int = 999,
    split: str = "validation",
    use_synthetic: bool = False,
    episode_hours: int = RL_EPISODE_HOURS,
    replay_start_index: Optional[int] = None,
    gru_checkpoint: Optional[str] = None,
) -> "VecNormalize":
    """
    Build a frozen-statistics evaluation environment.

    A provided ``vec_normalize_path`` must exist; missing statistics are never
    silently replaced with zero-initialised statistics.
    """
    if not _SB3_AVAILABLE:
        raise ImportError("stable_baselines3 is not installed.")

    vec_env = build_vec_env(
        mode                  = mode,
        n_envs                = n_envs,
        max_episode_steps     = max_episode_steps,
        error_bank_path       = error_bank_path,
        allow_final_test_bank = allow_final_test_bank,
        seed                  = seed,
        split                 = split,
        use_synthetic         = use_synthetic,
        episode_hours         = episode_hours,
        replay_start_indices  = None if replay_start_index is None else [replay_start_index],
        gru_checkpoint        = gru_checkpoint,
    )

    if vec_normalize_path is not None:
        if not os.path.exists(vec_normalize_path):
            raise FileNotFoundError(
                f"VecNormalize statistics not found: {vec_normalize_path}"
            )
        wrapped = VecNormalize.load(vec_normalize_path, vec_env)
        wrapped.training = False
        wrapped.norm_reward = False
        wrapped.norm_obs = False
        return wrapped

    wrapped = wrap_vec_normalize(
        vec_env,
        norm_obs   = False,
        norm_reward= False,
        clip_reward= 10.0,
        training   = False,
    )
    return wrapped
