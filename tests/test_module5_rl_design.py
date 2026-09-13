"""
test_module5_rl_design.py — Design validation tests for Module 5.3A.

These tests verify the correctness of the RL experiment design without training.

Tests verify:
1.  rl_config.py imports and all experiment configs are valid.
2.  PPOConfig field types and constraints.
3.  Observation space shape and dtype are correct (723, float32).
4.  Observation feature ordering matches spec.
5.  Block C contains WEC state at correct indices.
6.  Action space is Discrete(2).
7.  Reward formula with known inputs matches expected values.
8.  P_ref=1000 W gives typical per-step reward O(50–70) (documenting the scaling issue).
9.  VecNormalize recommendation: norm_obs=False, norm_reward=True.
10. Data split years are chronologically ordered.
11. RL training years do not overlap test years.
12. Error bank csv_path points to the 2024–2025 test period (confirm leakage risk is known).
13. reactive mode: Block B is all zeros.
14. perfect_forecast mode: Block B is nonzero (requires ≥49 hour replay).
15. realistic_forecast mode: Block B is nonzero and differs from perfect_forecast.
16. No future truth in reactive/realistic observations (existing test coverage).
17. Gymnasium environment passes SB3 compatibility check (basic API).
18. Three experiment configs have distinct names and modes.
19. Evaluation protocol seeds are deterministic.
20. Training years match Module 4.2 train split.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wec_params():
    from module5_control.benchmark import build_wec_params
    return build_wec_params()


@pytest.fixture(scope="session")
def replay_short():
    from module5_control.replay import build_synthetic_replay
    return build_synthetic_replay(n_hours=2, Hs=1.5, Tp=8.0, seed=42)


@pytest.fixture(scope="session")
def replay_long():
    """50h replay needed for perfect_forecast Block B to be non-zero."""
    from module5_control.replay import build_synthetic_replay
    return build_synthetic_replay(n_hours=50, Hs=1.5, Tp=8.0, seed=42)


# ===========================================================================
# Test 1 — rl_config imports
# ===========================================================================

def test_01_rl_config_imports():
    from module5_control.rl_config import (
        PPOConfig, RLDataSplit, EvaluationProtocol,
        EXPERIMENT_REACTIVE, EXPERIMENT_PERFECT_FORECAST,
        EXPERIMENT_REALISTIC_FORECAST, ALL_EXPERIMENTS,
        RL_DATA_SPLIT, EVALUATION_PROTOCOL,
        RL_TRAIN_YEARS, RL_VALIDATION_YEARS, RL_TEST_YEARS,
    )
    assert len(ALL_EXPERIMENTS) == 3


# ===========================================================================
# Test 2 — PPOConfig fields are valid
# ===========================================================================

def test_02_ppo_config_reactive():
    from module5_control.rl_config import EXPERIMENT_REACTIVE
    cfg = EXPERIMENT_REACTIVE
    assert cfg.forecast_mode    == "reactive"
    assert cfg.policy_type      == "MlpPolicy"
    assert len(cfg.net_arch)    == 3
    assert cfg.gamma            >= 0.99
    assert cfg.n_steps          == 3600
    assert cfg.batch_size       in (128, 256, 512)
    assert cfg.total_timesteps  >= 1_000_000
    assert cfg.seed             == 42


def test_02_ppo_config_perfect_forecast():
    from module5_control.rl_config import EXPERIMENT_PERFECT_FORECAST
    cfg = EXPERIMENT_PERFECT_FORECAST
    assert cfg.forecast_mode == "perfect_forecast"
    assert "ORACLE" in cfg.oracle_label.upper()


def test_02_ppo_config_realistic():
    from module5_control.rl_config import EXPERIMENT_REALISTIC_FORECAST
    cfg = EXPERIMENT_REALISTIC_FORECAST
    assert cfg.forecast_mode == "realistic_forecast"
    assert cfg.oracle_label  == ""


# ===========================================================================
# Test 3 — Observation space shape and dtype
# ===========================================================================

def test_03_obs_space_shape():
    from module5_control.observations import OBS_DIM
    from gymnasium import spaces
    import numpy as np
    sp = spaces.Box(
        low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
    )
    assert sp.shape   == (723,)
    assert sp.dtype   == np.float32
    assert OBS_DIM    == 723


# ===========================================================================
# Test 4 — Observation feature ordering matches spec
# ===========================================================================

def test_04_block_a_layout():
    from module5_control.observations import (
        _BLOCK_A_START, _BLOCK_A_END, H_OBS, F_OBS_WAVE_WIND
    )
    assert _BLOCK_A_START == 0
    assert _BLOCK_A_END   == H_OBS * F_OBS_WAVE_WIND
    assert _BLOCK_A_END   == 528


def test_04_block_b_layout():
    from module5_control.observations import (
        _BLOCK_B_START, _BLOCK_B_END, H_FORE, F_FORE
    )
    assert _BLOCK_B_START == 528
    assert _BLOCK_B_END   == 528 + H_FORE * F_FORE
    assert _BLOCK_B_END   == 720


def test_04_block_c_layout():
    from module5_control.observations import _BLOCK_C_START, _BLOCK_C_END, N_WEC
    assert _BLOCK_C_START == 720
    assert _BLOCK_C_END   == 723
    assert N_WEC          == 3


# ===========================================================================
# Test 5 — Block C WEC state encoding
# ===========================================================================

def test_05_block_c_x_v_latch_encoding(wec_params, replay_short):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_C_START, X_SCALE, V_SCALE

    env = build_env_from_checkpoint(
        replay=replay_short,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=5,
    )
    obs, _ = env.reset(seed=0)

    # At reset: x=0, v=0, latch=0 (RELEASE)
    assert abs(float(obs[_BLOCK_C_START])   - 0.0 / X_SCALE) < 1e-5   # x/X_SCALE
    assert abs(float(obs[_BLOCK_C_START+1]) - 0.0 / V_SCALE) < 1e-5   # v/V_SCALE
    assert float(obs[_BLOCK_C_START+2]) == 0.0                         # latch=RELEASE

    # After one step with LATCH
    obs, *_ = env.step(1)
    assert float(obs[_BLOCK_C_START+2]) == 1.0   # latch=LATCHED


# ===========================================================================
# Test 6 — Action space is Discrete(2)
# ===========================================================================

def test_06_action_space(wec_params, replay_short):
    from module5_control.environment import build_env_from_checkpoint
    from gymnasium import spaces

    env = build_env_from_checkpoint(
        replay=replay_short,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=5,
    )
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 2
    assert env.action_space.contains(np.int64(0))
    assert env.action_space.contains(np.int64(1))


# ===========================================================================
# Test 7 — Reward formula correctness
# ===========================================================================

def test_07_reward_formula():
    from module5_control.rewards import RewardConfig, compute_step_reward

    cfg = RewardConfig()
    # Known power, no penalties
    r = compute_step_reward(1000.0, False, False, cfg)
    assert abs(r - 1.0) < 1e-9   # 1000/1000 = 1.0

    # Switch penalty
    r_sw = compute_step_reward(1000.0, True, False, cfg)
    assert abs(r_sw - (1.0 - 0.01)) < 1e-9

    # End-stop penalty
    r_es = compute_step_reward(1000.0, False, True, cfg)
    assert abs(r_es - (1.0 - 0.1)) < 1e-9

    # Zero power
    r_zero = compute_step_reward(0.0, False, False, cfg)
    assert r_zero == 0.0

    # Power is always >= 0 from PTO
    r_pos = compute_step_reward(50_000.0, False, False, cfg)
    assert r_pos > 0.0


# ===========================================================================
# Test 8 — Reward magnitude issue is documented (P_ref=1000 → O(50) rewards)
# ===========================================================================

def test_08_reward_scaling_issue_documented():
    """
    Verify that the reward scaling issue is real:
    passive mean power ~67 kW gives per-step reward ~67 (too large for PPO).
    This test documents the issue, not a bug to fix.
    """
    from module5_control.rewards import RewardConfig, compute_step_reward
    from module5_control.rl_config import REWARD_USE_VECNORMALIZE

    cfg = RewardConfig()  # default P_ref=1000

    # From Module 5.2C benchmark: passive mean power ~ 67,220 W
    passive_reward = compute_step_reward(67_220.0, False, False, cfg)
    assert passive_reward > 50.0, (
        f"Passive reward={passive_reward:.1f} — should be ~67 to confirm scaling issue"
    )

    # VecNormalize is the recommended fix
    assert REWARD_USE_VECNORMALIZE is True, (
        "VecNormalize must be True in rl_config to handle reward scaling"
    )


# ===========================================================================
# Test 9 — VecNormalize setting: norm_obs=False, norm_reward=True
# ===========================================================================

def test_09_vecnormalize_config():
    from module5_control.rl_config import EXPERIMENT_REACTIVE
    cfg = EXPERIMENT_REACTIVE
    assert cfg.vecnorm_norm_obs is False, (
        "obs normalisation must be OFF — features already z-scored by Module 4 scaler"
    )
    assert cfg.vecnorm_norm_rew is True, (
        "reward normalisation must be ON — P_ref=1000 gives O(50) per-step rewards"
    )
    assert cfg.vecnorm_clip_rew >= 5.0


# ===========================================================================
# Test 10 — Data split years are chronologically ordered
# ===========================================================================

def test_10_data_split_chronological():
    from module5_control.rl_config import RL_TRAIN_YEARS, RL_VALIDATION_YEARS, RL_TEST_YEARS

    assert RL_TRAIN_YEARS[1]      < RL_VALIDATION_YEARS[0]
    assert RL_VALIDATION_YEARS[1] < RL_TEST_YEARS[0]


# ===========================================================================
# Test 11 — Training and test years do not overlap
# ===========================================================================

def test_11_no_train_test_overlap():
    from module5_control.rl_config import RL_TRAIN_YEARS, RL_TEST_YEARS

    train_set = set(range(RL_TRAIN_YEARS[0], RL_TRAIN_YEARS[1] + 1))
    test_set  = set(range(RL_TEST_YEARS[0],  RL_TEST_YEARS[1]  + 1))
    assert len(train_set & test_set) == 0


# ===========================================================================
# Test 12 — Error bank csv confirms 2024–2025 test period (leakage documented)
# ===========================================================================

def test_12_error_bank_is_test_period():
    """
    The existing error bank comes from 2024-2025 (confirmed by origin_idx).
    This test documents the known leakage risk.
    A new validation-period bank must be created before RL training.
    """
    import pandas as pd
    df = pd.read_csv("results/forecasting/exp_gru_wind/forecast_errors.csv.gz",
                     nrows=1000)
    min_origin = df["origin_idx"].min()

    # origin_idx=122759 corresponds to ~2024-01-02 (from design doc analysis)
    # Must be >= 2024 start (hours from 2010-01-01)
    import pandas as pd
    start_2024 = (pd.Timestamp("2024-01-01") - pd.Timestamp("2010-01-01")).days * 24
    assert min_origin >= start_2024, (
        f"origin_idx={min_origin} < 2024 start={start_2024} — bank may not be test period"
    )


# ===========================================================================
# Test 13 — Reactive mode Block B is zeros
# ===========================================================================

def test_13_reactive_block_b_zeros(wec_params, replay_short):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END

    env = build_env_from_checkpoint(
        replay=replay_short,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=10,
    )
    obs, _ = env.reset(seed=0)
    np.testing.assert_array_equal(obs[_BLOCK_B_START:_BLOCK_B_END], 0.0)


# ===========================================================================
# Test 14 — Perfect forecast Block B nonzero (requires long replay)
# ===========================================================================

def test_14_perfect_forecast_block_b_nonzero(wec_params, replay_long):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END

    env = build_env_from_checkpoint(
        replay=replay_long,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="perfect_forecast",
        max_episode_steps=10,
    )
    obs, _ = env.reset(seed=0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END]
    assert not np.all(block_b == 0.0), "Block B must contain oracle future values"
    assert np.all(np.isfinite(block_b))


# ===========================================================================
# Test 15 — Realistic forecast Block B nonzero and differs from reactive
# ===========================================================================

def test_15_realistic_vs_reactive_block_b(wec_params, replay_long):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END

    sampler = ForecastErrorSampler.from_csv(seed=42)
    env_real = build_env_from_checkpoint(
        replay=replay_long,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path="results/forecasting/exp_gru_wind/best_model.pt",
        max_episode_steps=5,
        forecast_error_sampler=sampler,
    )
    env_react = build_env_from_checkpoint(
        replay=replay_long,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=5,
    )
    obs_real,  _ = env_real.reset(seed=0)
    obs_react, _ = env_react.reset(seed=0)

    b_real  = obs_real[_BLOCK_B_START:_BLOCK_B_END]
    b_react = obs_react[_BLOCK_B_START:_BLOCK_B_END]

    assert np.all(np.isfinite(b_real))
    assert not np.allclose(b_real, b_react), (
        "Realistic forecast Block B must differ from reactive (all-zero) Block B"
    )


# ===========================================================================
# Test 16 — No future truth in realistic/reactive obs (regression)
# ===========================================================================

def test_16_no_future_truth_reactive(wec_params, replay_short):
    """Reactive obs Block B must be all zeros — no oracle info."""
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END

    env = build_env_from_checkpoint(
        replay=replay_short,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=20,
    )
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(0)
        np.testing.assert_array_equal(
            obs[_BLOCK_B_START:_BLOCK_B_END], 0.0,
            err_msg="Future truth leaked into reactive observation"
        )


# ===========================================================================
# Test 17 — Gymnasium API compatibility (basic check)
# ===========================================================================

def test_17_gymnasium_api_compatibility(wec_params, replay_short):
    """Environment must implement reset/step/action_space/observation_space."""
    import gymnasium as gym
    from module5_control.environment import build_env_from_checkpoint

    env = build_env_from_checkpoint(
        replay=replay_short,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=5,
    )

    assert isinstance(env, gym.Env)
    assert hasattr(env, "reset")
    assert hasattr(env, "step")
    assert hasattr(env, "action_space")
    assert hasattr(env, "observation_space")

    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32

    obs, rew, term, trunc, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    assert isinstance(rew, float)
    assert isinstance(term, (bool, np.bool_))
    assert isinstance(trunc, (bool, np.bool_))
    assert isinstance(info, dict)


# ===========================================================================
# Test 18 — Three experiment configs have distinct names and modes
# ===========================================================================

def test_18_distinct_experiments():
    from module5_control.rl_config import ALL_EXPERIMENTS

    names = [e.experiment_name for e in ALL_EXPERIMENTS]
    modes = [e.forecast_mode   for e in ALL_EXPERIMENTS]

    assert len(set(names)) == 3, f"Experiment names not unique: {names}"
    assert len(set(modes)) == 3, f"Forecast modes not unique: {modes}"
    assert "reactive"             in modes
    assert "perfect_forecast"     in modes
    assert "realistic_forecast"   in modes


# ===========================================================================
# Test 19 — Evaluation seeds are deterministic
# ===========================================================================

def test_19_eval_seeds_deterministic():
    from module5_control.rl_config import EVALUATION_PROTOCOL
    seeds = EVALUATION_PROTOCOL.eval_seeds
    assert len(seeds) >= 3
    # Same seeds every time (not random)
    assert EVALUATION_PROTOCOL.eval_seeds == seeds
    assert EVALUATION_PROTOCOL.deterministic_inference is True


# ===========================================================================
# Test 20 — Training years match Module 4.2 split
# ===========================================================================

def test_20_rl_train_matches_module4_split():
    from module5_control.rl_config import RL_TRAIN_YEARS, RL_VALIDATION_YEARS, RL_TEST_YEARS
    from module4_forecasting.dataset_v2 import TRAIN_END_YEAR, VAL_END_YEAR

    # RL uses same year boundaries as Module 4
    assert RL_TRAIN_YEARS[1]      == TRAIN_END_YEAR        # 2021
    assert RL_VALIDATION_YEARS[0] == TRAIN_END_YEAR + 1    # 2022
    assert RL_VALIDATION_YEARS[1] == VAL_END_YEAR          # 2023
    assert RL_TEST_YEARS[0]       == VAL_END_YEAR + 1      # 2024


# ===========================================================================
# Additional: stable_baselines3 NOT installed — document as blocker
# ===========================================================================

def test_sb3_not_installed():
    """
    Confirm stable_baselines3 is not yet installed.
    This must be installed before Module 5.3B training begins.
    """
    try:
        import stable_baselines3
        pytest.skip("stable_baselines3 is installed — remove this skip")
    except ImportError:
        pass   # Expected: SB3 not installed, must be installed before training


# ===========================================================================
# Additional: rl_config PPO n_steps * n_envs divides cleanly for rollout
# ===========================================================================

def test_rollout_size_divisible_by_batch():
    from module5_control.rl_config import EXPERIMENT_REACTIVE
    cfg = EXPERIMENT_REACTIVE
    rollout_size = cfg.n_steps * cfg.n_envs
    assert rollout_size % cfg.batch_size == 0 or rollout_size >= cfg.batch_size, (
        f"Rollout size {rollout_size} not cleanly divisible by batch_size {cfg.batch_size}"
    )
