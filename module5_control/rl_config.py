"""
rl_config.py — RL experiment configuration for Module 5.3 (design-only, Module 5.3A).

============================================================
PURPOSE
============================================================

Declarative configuration for the three PPO experiments.

DO NOT train in this file.
This is design/configuration only.

The three experiments:

    Experiment 1:  REACTIVE_PPO        — no forecast information
    Experiment 2:  PERFECT_FORECAST_PPO — oracle upper bound
    Experiment 3:  REALISTIC_FORECAST_PPO — GRU+wind + Module 5.2B errors

============================================================
REWARD SCALING NOTE
============================================================

With B_PTO = 200 kN·s/m, mean PTO power in a 1.5 m/8 s sea state is
~67 kW (passive) or ~47 kW (threshold latching).  Using P_ref = 1000 W
gives per-step rewards of O(67) — too large for PPO gradient stability.

Recommended approach: use SB3 VecNormalize for reward normalisation.
This avoids modifying the frozen RewardConfig and adapts automatically
to the actual power range observed during training.

Alternatively, if VecNormalize is not desired, set P_ref = 50_000 W
in the training RewardConfig (per-step passive reward → ~1.34).

See rl_config.REWARD_SCALING_RECOMMENDATION below.

============================================================
REFERENCES
============================================================
- Schulman et al. (2017), Proximal Policy Optimization.
- Raffin et al. (2021), Stable-Baselines3: Reliable RL Implementations.
- Falnes (2002), Ocean Waves and Oscillating Systems, §6.
- Research plan (WEC-Research-Plan-Combined-2.docx), Tier 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Data split (mirrors Module 4.2 chronological split)
# ---------------------------------------------------------------------------

RL_TRAIN_YEARS:       Tuple[int, int] = (2010, 2021)   # inclusive
RL_VALIDATION_YEARS:  Tuple[int, int] = (2022, 2023)   # inclusive
RL_TEST_YEARS:        Tuple[int, int] = (2024, 2025)   # HOLD OUT — touch last

# ---------------------------------------------------------------------------
# Episode design
# ---------------------------------------------------------------------------

#: RL training episode length in ERA5 hours.
#: 2 hours = 7200 control steps = 720 s at 1 Hz.
#: Chosen to balance:
#:   - enough context for the GRU+wind 48h forecast to be meaningful
#:   - short enough for fast episode turnover during PPO rollout
#:   - consistent with Module 5.2C benchmark episodes
RL_EPISODE_HOURS: int = 2

#: Minimum replay segment length for forecast pathways.
#: perfect_forecast needs H_OBS (48h lookback) + episode + H_FORE (48h ahead).
#: => at minimum 48 + 2 + 48 = 98 hours.
RL_MIN_REPLAY_HOURS_FORECAST: int = 98
RL_MIN_REPLAY_HOURS_REACTIVE: int = 50   # 48h lookback + 2h episode

# ---------------------------------------------------------------------------
# Reward scaling
# ---------------------------------------------------------------------------

#: If True, use SB3 VecNormalize for reward normalisation (recommended).
#: If False, set REWARD_P_REF_W to a manually chosen reference power.
REWARD_USE_VECNORMALIZE: bool = True

#: P_ref [W] for RewardConfig.
#: Used only if REWARD_USE_VECNORMALIZE = False.
#: 50_000 W gives per-step passive reward ~1.3 in the Hs=1.5m/Tp=8s reference sea state.
REWARD_P_REF_W: float = 50_000.0

#: When VecNormalize is used, the existing RewardConfig default (P_ref=1000 W)
#: is kept and VecNormalize handles the per-step scaling.
REWARD_SCALING_RECOMMENDATION: str = (
    "Use VecNormalize(norm_reward=True, clip_reward=10.0) to normalise rewards "
    "adaptively. This avoids modifying frozen RewardConfig and handles the "
    "~50x difference between the default P_ref=1000W and actual mean power "
    "(~50 kW for this buoy/PTO configuration). "
    "Clip at 10.0 to prevent outlier step rewards from destabilising value learning."
)

# ---------------------------------------------------------------------------
# PPO architecture and hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    """
    PPO hyperparameter configuration for one experiment.

    Categories:
        (P) = Fixed by physics/problem — do not change.
        (D) = Reasonable default — validated during Module 5.3A review.
        (T) = Candidate for tuning on validation data.

    References:
        Schulman et al. (2017) — original PPO paper.
        SB3 documentation — default recommendations.
        Andrychowicz et al. (2021) — hyperparameter sensitivity studies.
    """

    # ------------------------------------------------------------------
    # Experiment identification
    # ------------------------------------------------------------------
    experiment_name:    str   = "reactive_ppo"
    forecast_mode:      str   = "reactive"   # "reactive"|"perfect_forecast"|"realistic_forecast"
    oracle_label:       str   = ""           # set to "ORACLE_UPPER_BOUND" for perfect_forecast

    # ------------------------------------------------------------------
    # Policy architecture  (D)
    # ------------------------------------------------------------------
    policy_type:        str         = "MlpPolicy"
    net_arch:           Tuple[int, ...] = (256, 256, 128)
    # Rationale: 723-dimensional observation with temporal structure.
    # 256→256→128 MLP is a moderate depth for this input size.
    # Larger network risks overfit on the ~2-hour episodes; smaller
    # may underfit the 48-step history.
    activation_fn:      str         = "tanh"
    # Tanh preferred over ReLU for PPO: bounded activations reduce
    # gradient variance in policy loss; ReLU can cause dead neurons
    # in the value function head.

    # ------------------------------------------------------------------
    # Rollout and batch  (D, T)
    # ------------------------------------------------------------------
    n_steps:            int   = 3600
    # One full ERA5 hour per rollout buffer per env.
    # Rationale: 3600 steps = 1h of control at 1 Hz.
    # Long enough to include multiple latch/release cycles (~10–20 events
    # at 4 s per latch, 3 s dead time). Short enough for frequent updates.
    batch_size:         int   = 256
    # Mini-batch size. 256 is a standard default for MlpPolicy.
    n_epochs:           int   = 10
    # Number of epochs per update. Standard PPO default.
    n_envs:             int   = 4
    # Parallel environments for rollout collection.
    # Each runs the same sea-state distribution with different seeds.

    # ------------------------------------------------------------------
    # Discount and GAE  (D, T)
    # ------------------------------------------------------------------
    gamma:              float = 0.995
    # High gamma (0.995) appropriate for long episodes (7200 steps).
    # Equivalent to ~200-step effective horizon for discounted return.
    # At 1 Hz, 200 s ≈ 3.3 minutes — captures multi-wave coherence effects.
    gae_lambda:         float = 0.95
    # Standard GAE lambda. Balances bias/variance in advantage estimation.

    # ------------------------------------------------------------------
    # Policy gradient  (D, T)
    # ------------------------------------------------------------------
    learning_rate:      float = 3e-4
    # Adam learning rate. Standard PPO default.
    # Should be decayed as training converges; can use linear schedule.
    lr_schedule:        str   = "linear"   # "constant" or "linear" decay to 0
    clip_range:         float = 0.2
    # PPO clip range. Standard default.
    clip_range_vf:      Optional[float] = None   # None = no value clip
    # Value function clip not recommended by default (Schulman 2017).

    # ------------------------------------------------------------------
    # Entropy and exploration  (D, T)
    # ------------------------------------------------------------------
    ent_coef:           float = 0.01
    # Small entropy bonus encourages exploration of latch timing.
    # 0.01 is mild — can increase if policy collapses to all-RELEASE.
    # For the reactive experiment, start with 0.01.
    # For forecast experiments, may need 0.005 if policy already converges.

    # ------------------------------------------------------------------
    # Value function  (D)
    # ------------------------------------------------------------------
    vf_coef:            float = 0.5
    # Standard PPO default.
    max_grad_norm:      float = 0.5
    # Gradient clipping. Standard PPO default.

    # ------------------------------------------------------------------
    # Training budget  (D, T)
    # ------------------------------------------------------------------
    total_timesteps:    int   = 5_000_000
    # 5M steps / 4 envs / 7200 steps/ep ≈ 174 episodes per env ≈ 695 total.
    # At ~4 s/episode (from benchmark), 4 envs * 174 eps = ~11 minutes wall clock.
    # This is a STARTING budget. Double if policy has not converged.
    # Maximum budget for Tier 1: 20M steps.

    # ------------------------------------------------------------------
    # Evaluation  (D)
    # ------------------------------------------------------------------
    eval_freq:          int   = 50_000    # evaluate every 50k steps
    n_eval_episodes:    int   = 7         # one per benchmark segment
    eval_seed:          int   = 999       # deterministic eval seed

    # ------------------------------------------------------------------
    # Reproducibility  (D)
    # ------------------------------------------------------------------
    seed:               int   = 42        # global seed
    torch_deterministic: bool = True

    # ------------------------------------------------------------------
    # Checkpointing  (D)
    # ------------------------------------------------------------------
    checkpoint_freq:    int   = 200_000   # save every 200k steps
    best_model_metric:  str   = "mean_reward"

    # ------------------------------------------------------------------
    # VecNormalize  (D)
    # ------------------------------------------------------------------
    use_vecnormalize:   bool  = True
    vecnorm_norm_obs:   bool  = False     # obs already normalised
    vecnorm_norm_rew:   bool  = True      # reward needs normalisation
    vecnorm_clip_obs:   float = 10.0
    vecnorm_clip_rew:   float = 10.0
    # obs normalisation is OFF because:
    #   - Block A Hs/Tp/u10/v10 already z-scored by Module 4 scaler
    #   - Block A sin/cos/validity already in [-1,1] or {0,1}
    #   - Block B same as Block A
    #   - Block C x/v scaled by X_SCALE/V_SCALE; latch {0,1}
    # Applying VecNormalize obs normalisation would DOUBLE-normalise
    # the already-scaled features.
    # See rl_config.py normalisation table for detail.


# ---------------------------------------------------------------------------
# Three experiment configurations
# ---------------------------------------------------------------------------

EXPERIMENT_REACTIVE = PPOConfig(
    experiment_name  = "reactive_ppo",
    forecast_mode    = "reactive",
    oracle_label     = "",
    ent_coef         = 0.01,
    total_timesteps  = 5_000_000,
)

EXPERIMENT_PERFECT_FORECAST = PPOConfig(
    experiment_name  = "perfect_forecast_ppo",
    forecast_mode    = "perfect_forecast",
    oracle_label     = "ORACLE_UPPER_BOUND",
    ent_coef         = 0.01,
    total_timesteps  = 5_000_000,
    # NOTE: Requires replay with >= 98 hours (48 lookback + 2 episode + 48 forecast)
    # GRU checkpoint NOT required for this mode.
)

EXPERIMENT_REALISTIC_FORECAST = PPOConfig(
    experiment_name  = "realistic_forecast_ppo",
    forecast_mode    = "realistic_forecast",
    oracle_label     = "",
    ent_coef         = 0.01,
    total_timesteps  = 10_000_000,     # more budget: harder problem (noisy forecast)
    # Requires GRU checkpoint and ForecastErrorSampler
    # IMPORTANT: ForecastErrorSampler must use TRAINING-PERIOD error bank,
    # NOT the 2024-2025 test-period bank, to avoid leakage.
    # See forecast_error_leakage note in module5_3a_rl_design.md
)

ALL_EXPERIMENTS = [
    EXPERIMENT_REACTIVE,
    EXPERIMENT_PERFECT_FORECAST,
    EXPERIMENT_REALISTIC_FORECAST,
]

# ---------------------------------------------------------------------------
# Data split configuration
# ---------------------------------------------------------------------------

ERROR_BANK_TRAIN_DEFAULT: str = (
    "results/forecasting/exp_gru_wind/forecast_errors_val2022_2023.csv.gz"
)

ERROR_BANK_FINAL_TEST: str = (
    "results/forecasting/exp_gru_wind/forecast_errors.csv.gz"
)

GRU_CHECKPOINT_PATH: str = (
    "results/forecasting/exp_gru_wind/best_model.pt"
)


def assert_safe_training_error_bank(csv_path: str) -> None:
    """
    Assert that the given error-bank path is safe for RL TRAINING.

    Raises AssertionError if:
      - The path points to the 2024–2025 final-test bank (leakage).
      - The file does not exist.

    This is a hard safety guard: the 2024–2025 bank is FINAL TEST ONLY.
    """
    import os
    from pathlib import Path

    resolved_path = Path(csv_path).resolve()
    forbidden_resolved = Path(ERROR_BANK_FINAL_TEST).resolve()

    if resolved_path == forbidden_resolved:
        raise AssertionError(
            f"ERROR-BANK LEAKAGE PREVENTED: "
            f"Refusing to use the final-test error bank for training.\n"
            f"  Forbidden path: {ERROR_BANK_FINAL_TEST}\n"
            f"  Safe training bank: {ERROR_BANK_TRAIN_DEFAULT}\n"
            f"The 2024–2025 bank is reserved for FINAL TEST EVALUATION ONLY. "
            f"Using it during training would leak test-period forecast-error "
            f"statistics into the RL policy. For training use the 2022–2023 "
            f"validation-period bank instead."
        )

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Training error bank not found: {csv_path}\n"
            f"Expected: {ERROR_BANK_TRAIN_DEFAULT}\n"
            f"Verify Module 5.3A-DATA validation error bank has been generated."
        )


@dataclass
class RLDataSplit:
    """
    Chronological data split for RL experiments.

    TRAIN:      2010–2021  — RL training episodes drawn from this period.
    VALIDATION: 2022–2023  — RL hyperparameter selection and early stopping.
    TEST:       2024–2025  — COMPLETELY HELD OUT. Touch only for final paper results.

    FORECAST ERROR BANK:
        Training-safe bank (forecast_errors_val2022_2023.csv.gz) = 2022–2023 GRU errors.
        This is the ONLY bank permitted during PPO training and validation.

        Final-test bank (forecast_errors.csv.gz) = 2024–2025 GRU errors.
        RESERVED EXCLUSIVELY for frozen-policy final evaluation.
        assert_safe_training_error_bank() prevents accidental use during training.
    """
    train_years:      Tuple[int, int] = RL_TRAIN_YEARS
    validation_years: Tuple[int, int] = RL_VALIDATION_YEARS
    test_years:       Tuple[int, int] = RL_TEST_YEARS

    error_bank_train:      str = ERROR_BANK_TRAIN_DEFAULT
    error_bank_final_eval: str = ERROR_BANK_FINAL_TEST

    gru_checkpoint_path:   str = GRU_CHECKPOINT_PATH

    def training_era5_years(self):
        return list(range(self.train_years[0], self.train_years[1] + 1))

    def validation_era5_years(self):
        return list(range(self.validation_years[0], self.validation_years[1] + 1))

    def __post_init__(self) -> None:
        assert self.train_years[1] < self.validation_years[0], (
            f"Train years {self.train_years} must end before validation "
            f"years {self.validation_years} begin."
        )
        assert self.validation_years[1] < self.test_years[0], (
            f"Validation years {self.validation_years} must end before test "
            f"years {self.test_years} begin."
        )


RL_DATA_SPLIT = RLDataSplit()

# ---------------------------------------------------------------------------
# Evaluation protocol
# ---------------------------------------------------------------------------

@dataclass
class EvaluationProtocol:
    """
    Deterministic evaluation protocol for RL policy assessment.

    All policies are evaluated under identical conditions.
    Final test results must only be computed once, after the policy is frozen.
    """
    eval_seeds:             Tuple[int, ...] = (100, 101, 102, 103, 104)
    n_eval_segments:        int             = 7    # all BENCHMARK_SEGMENTS from 5.2C
    segment_hours:          int             = 2
    deterministic_inference: bool           = True   # no exploration during eval
    eval_modes:             Tuple[str, ...] = (
        "reactive",           # all policies evaluated without forecast
        "perfect_forecast",   # oracle upper bound
        "realistic_forecast", # realistic operating condition
    )
    metrics: Tuple[str, ...] = (
        "total_energy_Wh",
        "mean_power_W",
        "max_step_power_W",
        "n_latch_events",
        "latch_fraction",
        "max_abs_displacement_m",
        "max_abs_velocity_ms",
        "total_reward",
        "is_finite",
    )

EVALUATION_PROTOCOL = EvaluationProtocol()
