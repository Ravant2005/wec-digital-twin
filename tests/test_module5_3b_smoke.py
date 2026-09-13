"""
test_module5_3b_smoke.py — Module 5.3B PPO Infrastructure Smoke Test.

Comprehensive smoke-test covering all 12 required items from Module 5.3B
section 15, plus performance timing and configuration validation.

IMPORTANT: No PPO training beyond 512 timesteps. Uses synthetic replay.
Realistic mode uses TRAINING-SAFE 2022-2023 error bank only.
"""

from __future__ import annotations

import math
import os
import time
import unittest.mock
from typing import Optional

import numpy as np
import pytest

# ============================================================
# Constants
# ============================================================
SMOKE_EPISODE_STEPS = 50
SMOKE_N_ENVS = 1
SMOKE_BATCH_SIZE = 16
SMOKE_N_STEPS = 16
SMOKE_TIMESTEPS = 32


# ============================================================
# Session fixtures
# ============================================================

@pytest.fixture(scope="session")
def wec_params():
    from module5_control.env_factory import build_wec_params
    return build_wec_params()


@pytest.fixture(scope="session")
def train_error_bank():
    from module5_control.rl_config import ERROR_BANK_TRAIN_DEFAULT, ERROR_BANK_FINAL_TEST
    assert os.path.normpath(ERROR_BANK_TRAIN_DEFAULT) != os.path.normpath(ERROR_BANK_FINAL_TEST)
    if not os.path.exists(ERROR_BANK_TRAIN_DEFAULT):
        pytest.skip(f"Training error bank not found: {ERROR_BANK_TRAIN_DEFAULT}")
    return ERROR_BANK_TRAIN_DEFAULT


# ============================================================
# 1 — Environment initialises (all 3 modes)
# ============================================================

class TestSmokeEnvInit:
    @pytest.mark.parametrize("mode", ["reactive", "perfect_forecast", "realistic_forecast"])
    def test_env_initialises(self, mode, wec_params, train_error_bank):
        from module5_control.env_factory import build_single_env
        error_bank = train_error_bank if mode == "realistic_forecast" else None
        env = build_single_env(mode=mode, max_episode_steps=SMOKE_EPISODE_STEPS,
                               replay_seed=0, error_bank_path=error_bank, use_synthetic=True)
        obs, info = env.reset(seed=0)
        assert obs is not None
        assert isinstance(info, dict)


# ============================================================
# 2 — Observation shape (723,), dtype float32
# ============================================================

class TestSmokeObsShape:
    @pytest.mark.parametrize("mode", ["reactive", "perfect_forecast", "realistic_forecast"])
    def test_obs_shape_dtype(self, mode, wec_params, train_error_bank):
        from module5_control.env_factory import build_single_env
        from module5_control.observations import OBS_DIM
        error_bank = train_error_bank if mode == "realistic_forecast" else None
        env = build_single_env(mode=mode, max_episode_steps=SMOKE_EPISODE_STEPS,
                               replay_seed=0, error_bank_path=error_bank, use_synthetic=True)
        obs, _ = env.reset(seed=0)
        assert OBS_DIM == 723
        assert obs.shape == (723,), f"mode={mode}: shape={obs.shape}"
        assert obs.dtype == np.float32, f"mode={mode}: dtype={obs.dtype}"


# ============================================================
# 3 — Action space = Discrete(2)
# ============================================================

class TestSmokeActionSpace:
    def test_discrete_2(self, wec_params):
        from gymnasium import spaces
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=10, use_synthetic=True)
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == 2

    def test_action_0_release(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=10, use_synthetic=True)
        env.reset(seed=0)
        obs, r, t, tr, info = env.step(0)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        assert info["latch_status"] == 0

    def test_action_1_latch(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=10, use_synthetic=True)
        env.reset(seed=0)
        obs, r, t, tr, info = env.step(1)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(r)
        assert info["latch_status"] == 1


# ============================================================
# 4 — PPO initialises
# ============================================================

class TestSmokePPOInit:
    def test_ppo_init_reactive(self, wec_params):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[64, 32], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        assert model.policy is not None
        assert model.observation_space.shape == (723,)
        assert model.action_space.n == 2

    def test_ppo_init_production_arch(self, wec_params):
        """PPO with the full production architecture from rl_config."""
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        from module5_control.rl_config import EXPERIMENT_REACTIVE
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO(
            EXPERIMENT_REACTIVE.policy_type, env,
            n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE, n_epochs=1,
            gamma=EXPERIMENT_REACTIVE.gamma, gae_lambda=EXPERIMENT_REACTIVE.gae_lambda,
            clip_range=EXPERIMENT_REACTIVE.clip_range, ent_coef=EXPERIMENT_REACTIVE.ent_coef,
            vf_coef=EXPERIMENT_REACTIVE.vf_coef, max_grad_norm=EXPERIMENT_REACTIVE.max_grad_norm,
            policy_kwargs=dict(net_arch=list(EXPERIMENT_REACTIVE.net_arch),
                               activation_fn=torch.nn.Tanh),
            verbose=0, seed=42,
        )
        assert model.policy is not None


# ============================================================
# 5 — Rollout works
# ============================================================

class TestSmokeRollout:
    @pytest.mark.parametrize("mode", ["reactive", "perfect_forecast"])
    def test_rollout_completes(self, mode, wec_params):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode=mode, n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS, reset_num_timesteps=True)
        assert model.num_timesteps > 0

    def test_rollout_realistic(self, wec_params, train_error_bank):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode="realistic_forecast", n_envs=1,
                                 max_episode_steps=SMOKE_EPISODE_STEPS,
                                 error_bank_path=train_error_bank, use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS, reset_num_timesteps=True)
        assert model.num_timesteps > 0


# ============================================================
# 6 — PPO gradient step executes
# ============================================================

class TestSmokePPOUpdate:
    def test_gradient_step_changes_policy(self, wec_params):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=2, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        params_before = [p.clone().detach() for p in model.policy.parameters()]
        model.learn(total_timesteps=SMOKE_TIMESTEPS, reset_num_timesteps=True)
        params_after = [p.clone().detach() for p in model.policy.parameters()]
        changed = any(not torch.allclose(b, a, atol=1e-9)
                      for b, a in zip(params_before, params_after))
        assert changed, "Gradient step should update at least some parameters"


# ============================================================
# 7 — No NaN/Inf in obs/rewards (all 3 modes, 100 steps)
# ============================================================

class TestSmokeNoNanInf:
    @pytest.mark.parametrize("env_mode", ["reactive", "perfect_forecast", "realistic_forecast"])
    def test_100_steps_no_nan_inf(self, env_mode, wec_params, train_error_bank):
        from module5_control.env_factory import build_single_env
        error_bank = train_error_bank if env_mode == "realistic_forecast" else None
        env = build_single_env(mode=env_mode, max_episode_steps=200, replay_seed=99,
                               error_bank_path=error_bank, use_synthetic=True)
        obs, _ = env.reset(seed=99)
        assert np.all(np.isfinite(obs)), f"reset() non-finite obs in mode={env_mode}"

        nan_obs, inf_obs, nan_rew, inf_rew = 0, 0, 0, 0
        for i in range(100):
            obs, r, term, trunc, _ = env.step(i % 2)
            nan_obs += int(np.sum(np.isnan(obs)))
            inf_obs += int(np.sum(np.isinf(obs)))
            if np.isnan(float(r)): nan_rew += 1
            if np.isinf(float(r)): inf_rew += 1
            if term or trunc:
                obs, _ = env.reset(seed=i)

        assert nan_obs == 0, f"mode={env_mode}: {nan_obs} NaN obs entries"
        assert inf_obs == 0, f"mode={env_mode}: {inf_obs} Inf obs entries"
        assert nan_rew == 0, f"mode={env_mode}: {nan_rew} NaN rewards"
        assert inf_rew == 0, f"mode={env_mode}: {inf_rew} Inf rewards"


# ============================================================
# 8 — Checkpoint saves and reloads
# ============================================================

class TestSmokeCheckpointSave:
    def test_checkpoint_save_load(self, wec_params, tmp_path):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS)
        zip_path = str(tmp_path / "smoke.zip")
        model.save(zip_path)
        assert os.path.exists(zip_path)
        assert os.path.getsize(zip_path) > 0
        del model
        m = PPO.load(zip_path, env=None)
        assert m.policy is not None
        assert m.observation_space.shape == (723,)
        assert m.action_space.n == 2


# ============================================================
# 9 — Reload → identical deterministic predictions
# ============================================================

class TestSmokeCheckpointDeterminism:
    def test_two_loads_same_predictions(self, wec_params, tmp_path):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env, build_eval_env
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS)
        zip_path = str(tmp_path / "det.zip")
        model.save(zip_path)
        del model
        ma = PPO.load(zip_path, env=None)
        mb = PPO.load(zip_path, env=None)
        eval_env = build_eval_env(mode="reactive", n_envs=1,
                                  max_episode_steps=5, use_synthetic=True)
        obs = eval_env.reset()
        a_a, _ = ma.predict(obs, deterministic=True)
        a_b, _ = mb.predict(obs, deterministic=True)
        assert np.array_equal(a_a, a_b), f"Two loads differ: {a_a} vs {a_b}"


# ============================================================
# 10 — VecNormalize saves/loads with frozen statistics
# ============================================================

class TestSmokeVecNormalizeSaveLoad:
    def test_vecnorm_save_load_frozen(self, wec_params, tmp_path):
        from stable_baselines3.common.vec_env import VecNormalize
        from module5_control.env_factory import build_training_env, build_eval_env
        train_env = build_training_env(mode="reactive", n_envs=1,
                                       max_episode_steps=SMOKE_EPISODE_STEPS, use_synthetic=True)
        # Run a few steps to update reward stats
        obs = train_env.reset()
        for _ in range(10):
            obs, r, done, info = train_env.step(np.array([0]))
        pkl = str(tmp_path / "vn.pkl")
        train_env.save(pkl)
        assert os.path.exists(pkl)

        eval_env = build_eval_env(mode="reactive", n_envs=1,
                                   max_episode_steps=SMOKE_EPISODE_STEPS,
                                   vec_normalize_path=pkl, use_synthetic=True)
        assert isinstance(eval_env, VecNormalize)
        assert eval_env.training is False
        assert eval_env.norm_reward is False
        assert eval_env.norm_obs is False
        obs = eval_env.reset()
        assert obs.shape == (1, 723) and np.all(np.isfinite(obs))

    def test_eval_stats_frozen(self, wec_params, tmp_path):
        """Stats must not update when stepping eval env."""
        from module5_control.env_factory import build_training_env, build_eval_env
        train_env = build_training_env(mode="reactive", n_envs=1,
                                       max_episode_steps=SMOKE_EPISODE_STEPS, use_synthetic=True)
        pkl = str(tmp_path / "vn_frz.pkl")
        train_env.save(pkl)
        eval_env = build_eval_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                   vec_normalize_path=pkl, use_synthetic=True)
        # ret_rms mean should not change
        mean_before = float(eval_env.ret_rms.mean) if hasattr(eval_env, 'ret_rms') else 0.0
        eval_env.reset()
        for _ in range(20):
            eval_env.step(np.array([0]))
        mean_after = float(eval_env.ret_rms.mean) if hasattr(eval_env, 'ret_rms') else 0.0
        assert mean_before == mean_after


# ============================================================
# 11 — Deterministic inference works after reload
# ============================================================

class TestSmokeDeterministicInference:
    def test_deterministic_consistent(self, wec_params, tmp_path):
        import torch
        from stable_baselines3 import PPO
        from module5_control.env_factory import build_training_env, build_eval_env
        env = build_training_env(mode="reactive", n_envs=1, max_episode_steps=SMOKE_EPISODE_STEPS,
                                 use_synthetic=True)
        model = PPO("MlpPolicy", env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS)
        zip_path = str(tmp_path / "det_inf.zip")
        model.save(zip_path)
        del model
        m2 = PPO.load(zip_path, env=None)
        eval_env = build_eval_env(mode="reactive", n_envs=1,
                                   max_episode_steps=SMOKE_EPISODE_STEPS, use_synthetic=True)
        obs = eval_env.reset()
        actions = [m2.predict(obs, deterministic=True)[0] for _ in range(5)]
        for i, a in enumerate(actions[1:], 1):
            assert np.array_equal(actions[0], a), f"Call {i} differs from call 0"


# ============================================================
# 12 — Environment continues after reload
# ============================================================

class TestSmokeEnvContinuesAfterReload:
    def test_steps_after_reload(self, wec_params, tmp_path):
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize
        from module5_control.env_factory import build_training_env, build_eval_env
        train_env = build_training_env(mode="reactive", n_envs=1,
                                       max_episode_steps=SMOKE_EPISODE_STEPS, use_synthetic=True)
        model = PPO("MlpPolicy", train_env, n_steps=SMOKE_N_STEPS, batch_size=SMOKE_BATCH_SIZE,
                    n_epochs=1, policy_kwargs=dict(net_arch=[32, 16], activation_fn=torch.nn.Tanh),
                    verbose=0, seed=42)
        model.learn(total_timesteps=SMOKE_TIMESTEPS)
        zip_path = str(tmp_path / "m.zip")
        pkl_path = str(tmp_path / "vn.pkl")
        model.save(zip_path)
        train_env.save(pkl_path)
        del model, train_env
        m2 = PPO.load(zip_path, env=None)
        eval_env = build_eval_env(mode="reactive", n_envs=1,
                                   max_episode_steps=SMOKE_EPISODE_STEPS,
                                   vec_normalize_path=pkl_path, use_synthetic=True)
        assert isinstance(eval_env, VecNormalize)
        assert eval_env.training is False
        obs = eval_env.reset()
        for i in range(20):
            act, _ = m2.predict(obs, deterministic=True)
            obs, r, done, info = eval_env.step(act)
            assert np.all(np.isfinite(obs)), f"Step {i}: non-finite obs after reload"
            assert np.isfinite(r).all(), f"Step {i}: non-finite reward after reload"


# ============================================================
# PERFORMANCE MEASUREMENT
# ============================================================

class TestSmokePerformance:
    def test_step_throughput_and_estimates(self, wec_params):
        """Measures raw step throughput and prints 5M/10M estimates."""
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=500,
                               replay_seed=0, use_synthetic=True)
        env.reset(seed=0)
        N = 200
        t0 = time.time()
        for i in range(N):
            obs, r, term, trunc, info = env.step(i % 2)
            if term or trunc:
                env.reset(seed=i)
        elapsed = time.time() - t0
        sps = N / elapsed
        est_5m_1env = 5_000_000 / sps
        est_10m_1env = 10_000_000 / sps
        est_5m_4env = est_5m_1env / 4
        est_10m_4env = est_10m_1env / 4
        print(f"\n[PERF] Step throughput (single env): {sps:.1f} steps/s")
        print(f"[PERF] Estimate  5M steps (1 env):  {est_5m_1env/60:.1f} min")
        print(f"[PERF] Estimate 10M steps (1 env):  {est_10m_1env/60:.1f} min")
        print(f"[PERF] Estimate  5M steps (4 envs): {est_5m_4env/60:.1f} min")
        print(f"[PERF] Estimate 10M steps (4 envs): {est_10m_4env/60:.1f} min")
        assert sps > 0.5, f"Throughput {sps:.2f} steps/s is too low"

    def test_reset_time(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=100,
                               replay_seed=0, use_synthetic=True)
        N = 3
        t0 = time.time()
        for i in range(N):
            env.reset(seed=i)
        avg_reset_s = (time.time() - t0) / N
        print(f"\n[PERF] Avg reset time: {avg_reset_s*1000:.1f} ms")
        assert avg_reset_s < 30.0, f"Reset time {avg_reset_s:.2f}s too slow"


# ============================================================
# ERROR BANK SAFETY
# ============================================================

class TestSmokeErrorBankSafety:
    def test_train_bank_differs_from_final_test_bank(self):
        from module5_control.rl_config import ERROR_BANK_TRAIN_DEFAULT, ERROR_BANK_FINAL_TEST
        assert os.path.normpath(ERROR_BANK_TRAIN_DEFAULT) != os.path.normpath(ERROR_BANK_FINAL_TEST)

    def test_safety_guard_refuses_final_test_bank(self):
        from module5_control.rl_config import assert_safe_training_error_bank, ERROR_BANK_FINAL_TEST
        if not os.path.exists(ERROR_BANK_FINAL_TEST):
            pytest.skip("Final test bank not present")
        with pytest.raises(AssertionError, match="LEAKAGE"):
            assert_safe_training_error_bank(ERROR_BANK_FINAL_TEST)

    def test_realistic_env_uses_train_bank(self, wec_params, train_error_bank):
        from module5_control.env_factory import build_forecast_error_sampler
        from module5_control.rl_config import ERROR_BANK_TRAIN_DEFAULT
        sampler = build_forecast_error_sampler(mode="realistic_forecast",
                                               error_bank_path=None,
                                               seed=42, allow_final_test_bank=False)
        assert sampler is not None
        assert os.path.normpath(sampler.bank.csv_path) == os.path.normpath(ERROR_BANK_TRAIN_DEFAULT)

    def test_train_bank_structure(self, train_error_bank):
        from module5_control.env_factory import build_forecast_error_sampler
        sampler = build_forecast_error_sampler(mode="realistic_forecast",
                                               error_bank_path=train_error_bank,
                                               seed=0, allow_final_test_bank=False)
        assert sampler.bank.n_trajectories > 0
        assert sampler.bank.n_leads == 48
        assert sampler.bank.err_hs.shape[1] == 48


# ============================================================
# VECNORMALIZE CONFIG
# ============================================================

class TestSmokeVecNormalizeConfig:
    def test_training_config(self, wec_params):
        from stable_baselines3.common.vec_env import VecNormalize
        from module5_control.env_factory import build_training_env
        env = build_training_env(mode="reactive", n_envs=1,
                                  max_episode_steps=10, use_synthetic=True)
        assert isinstance(env, VecNormalize)
        assert env.norm_obs is False
        assert env.norm_reward is True
        assert env.clip_reward == 10.0
        assert env.training is True


# ============================================================
# PPO CONFIG VALIDATION
# ============================================================

class TestSmokePPOConfig:
    def test_config_values_match_spec(self):
        from module5_control.rl_config import EXPERIMENT_REACTIVE, EXPERIMENT_PERFECT_FORECAST, EXPERIMENT_REALISTIC_FORECAST
        for cfg in [EXPERIMENT_REACTIVE, EXPERIMENT_PERFECT_FORECAST, EXPERIMENT_REALISTIC_FORECAST]:
            assert cfg.learning_rate == 3e-4
            assert cfg.lr_schedule == "linear"
            assert cfg.n_steps == 3600
            assert cfg.batch_size == 256
            assert cfg.n_epochs == 10
            assert cfg.gamma == 0.995
            assert cfg.gae_lambda == 0.95
            assert cfg.clip_range == 0.2
            assert cfg.ent_coef == 0.01
            assert cfg.vf_coef == 0.5
            assert cfg.max_grad_norm == 0.5
            assert cfg.net_arch == (256, 256, 128)
            assert cfg.activation_fn == "tanh"
            assert cfg.seed == 42
            assert cfg.use_vecnormalize is True
            assert cfg.vecnorm_norm_obs is False
            assert cfg.vecnorm_norm_rew is True
            assert cfg.vecnorm_clip_rew == 10.0

    def test_production_episode_7200(self):
        from module5_control.rl_config import RL_EPISODE_HOURS
        from module5_control.replay import N_CONTROL_PER_HOUR
        assert RL_EPISODE_HOURS == 2
        assert N_CONTROL_PER_HOUR == 3600
        assert RL_EPISODE_HOURS * N_CONTROL_PER_HOUR == 7200

    def test_data_split_chronological(self):
        from module5_control.rl_config import RL_TRAIN_YEARS, RL_VALIDATION_YEARS, RL_TEST_YEARS
        assert RL_TRAIN_YEARS == (2010, 2021)
        assert RL_VALIDATION_YEARS == (2022, 2023)
        assert RL_TEST_YEARS == (2024, 2025)
        assert RL_TRAIN_YEARS[1] < RL_VALIDATION_YEARS[0]
        assert RL_VALIDATION_YEARS[1] < RL_TEST_YEARS[0]


# ============================================================
# REWARD CONFIG
# ============================================================

class TestSmokeRewardConfig:
    def test_defaults_match_spec(self):
        from module5_control.rewards import RewardConfig
        cfg = RewardConfig()
        assert cfg.p_ref_W == 1000.0
        assert cfg.lambda_switch == 0.01
        assert cfg.lambda_end_stop == 0.1
        assert cfg.x_end_stop_m == 3.0

    def test_immutable(self):
        from module5_control.rewards import RewardConfig
        cfg = RewardConfig()
        with pytest.raises((AttributeError, TypeError)):
            cfg.p_ref_W = 999.0


# ============================================================
# PHYSICS TIMESTEP
# ============================================================

class TestSmokePhysicsTimestep:
    def test_constants(self):
        from module5_control.replay import DT_PHYSICS, DT_CONTROL, N_PHYSICS_PER_CONTROL
        assert DT_PHYSICS == 0.1
        assert DT_CONTROL == 1.0
        assert N_PHYSICS_PER_CONTROL == 10

    def test_10_substeps_per_control(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=5, use_synthetic=True)
        env.reset(seed=0)
        _, _, _, _, info = env.step(0)
        assert info["control_step"] == 1
        assert info["physics_step"] == 10


# ============================================================
# SAFETY GUARD: ACCIDENTAL TRAINING PREVENTION
# ============================================================

class TestSmokeAccidentalTrainingPrevention:
    def test_default_timesteps_small(self):
        import sys
        import unittest.mock
        from module5_control.train_ppo import _parse_args
        with unittest.mock.patch('sys.argv', ['train_ppo']):
            args = _parse_args()
        assert args.timesteps <= 10_000, (
            f"Default --timesteps={args.timesteps} must be ≤ 10,000 (smoke test scale)"
        )

    def test_warn_on_large_timesteps_code_exists(self):
        import inspect
        from module5_control import train_ppo
        src = inspect.getsource(train_ppo.train)
        assert "10_000" in src or "10000" in src


# ============================================================
# GYMNASIUM API
# ============================================================

class TestSmokeGymnasiumAPI:
    def test_reset_2tuple(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=5, use_synthetic=True)
        result = env.reset(seed=0)
        assert isinstance(result, tuple) and len(result) == 2
        obs, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)

    def test_step_5tuple(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=5, use_synthetic=True)
        env.reset(seed=0)
        result = env.step(0)
        assert isinstance(result, tuple) and len(result) == 5
        obs, rew, term, trunc, info = result
        assert isinstance(obs, np.ndarray)
        assert isinstance(rew, (float, int, np.floating, np.integer))
        assert isinstance(term, (bool, np.bool_))
        assert isinstance(trunc, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_is_gymnasium_env(self, wec_params):
        import gymnasium as gym
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=5, use_synthetic=True)
        assert isinstance(env, gym.Env)

    def test_action_space_membership(self, wec_params):
        from module5_control.env_factory import build_single_env
        env = build_single_env(mode="reactive", max_episode_steps=5, use_synthetic=True)
        assert env.action_space.contains(np.int64(0))
        assert env.action_space.contains(np.int64(1))
        assert not env.action_space.contains(np.int64(2))
