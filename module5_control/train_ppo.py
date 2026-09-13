"""
train_ppo.py — PPO training script for WEC latching control (Module 5.3B).

============================================================
PURPOSE
============================================================

Trains an SB3 PPO agent on the WECControlEnv with the three
forecast-information modes:

    --mode reactive            Block B = zeros (no forecast)
    --mode perfect             Block B = true future ERA5 (oracle upper bound)
    --mode realistic           Block B = GRU+wind + 5.2B sampled errors

Production training uses the Module 5 data split:
    TRAIN = 2010–2021
    VAL   = 2022–2023  (checkpoint selection)
    TEST  = 2024–2025  (HELD OUT — never used here)

Checkpointing:
    latest_model.zip           after every save_freq steps
    best_model.zip             selected on validation mean reward
    vecnormalize.pkl           training reward normalisation statistics

IMPORTANT SAFETY GUARDS
=======================

1. Default --timesteps is 1000 (smoke-test scale). An accidental
   `python train_ppo.py` will NOT launch 5M/10M training.
2. For --mode realistic the error bank defaults to the TRAINING-SAFE
   2022–2023 validation-period bank. The 2024–2025 final-test bank
   is explicitly refused via assert_safe_training_error_bank().
3. VecNormalize uses norm_obs=False (observations already scaled)
   and norm_reward=True (adaptive reward whitening, clip_reward=10.0).
4. --episode-steps defaults to 7200 (2 hr production episode).
   Smoke tests should override with a smaller value (e.g. 50–200).

============================================================
USAGE
============================================================

Smoke test (reactive, ~1000 steps, short episodes):
    python -m module5_control.train_ppo \\
        --mode reactive --timesteps 1000 --episode-steps 200

Smoke test (perfect_forecast):
    python -m module5_control.train_ppo \\
        --mode perfect --timesteps 1000 --episode-steps 200

Smoke test (realistic — validates error bank safety):
    python -m module5_control.train_ppo \\
        --mode realistic --timesteps 1000 --episode-steps 200

Production reactive PPO (NOT to be run in 5.3B — approved in 5.3C):
    python -m module5_control.train_ppo \\
        --mode reactive --timesteps 5_000_000 --n-envs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PPO training for WEC latching control (Module 5.3B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        type=str,
        default="reactive",
        choices=["reactive", "perfect", "realistic"],
        help="Forecast information mode.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train"],
        help="Data split. Only 'train' is supported by this script.",
    )
    p.add_argument(
        "--timesteps",
        type=int,
        default=1000,
        help="Total PPO timesteps. DEFAULT=1000 (smoke test scale). "
             "Set explicitly to 5_000_000 for production reactive PPO.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="PPO and environment master seed.",
    )
    p.add_argument(
        "--episode-steps",
        type=int,
        default=7200,
        help="RL control steps per episode. Production=7200 (2 hours at 1 Hz). "
             "Override to a smaller value (50–200) for smoke tests.",
    )
    p.add_argument(
        "--error-bank",
        type=str,
        default=None,
        help="Path to forecast-error bank for --mode realistic. "
             "If None, defaults to the TRAINING-SAFE 2022–2023 bank "
             "(results/forecasting/exp_gru_wind/forecast_errors_val2022_2023.csv.gz). "
             "The 2024–2025 final-test bank is explicitly refused.",
    )
    p.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel SB3 vector environments. "
             "Smoke test: 1–2. Production: 4.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for checkpoints. If None, uses "
             "results/rl/<mode>_ppo_<timestamp>.",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=3600,
        help="PPO rollout buffer size (steps per env per update).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="PPO mini-batch size per epoch.",
    )
    p.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="PPO epochs per rollout.",
    )
    p.add_argument(
        "--save-freq",
        type=int,
        default=None,
        help="Save latest checkpoint every N timesteps. "
             "Default: max(1000, timesteps // 10).",
    )
    p.add_argument(
        "--use-synthetic",
        action="store_true",
        help="Use synthetic replay for fast smoke tests (default: real ERA5).",
    )
    return p.parse_args()


def _mode_to_env_mode(mode_arg: str) -> str:
    """Translate CLI --mode to the internal mode strings used by the env."""
    mapping = {
        "reactive":  "reactive",
        "perfect":   "perfect_forecast",
        "realistic": "realistic_forecast",
    }
    return mapping[mode_arg]


def _ensure_output_dir(args: argparse.Namespace) -> str:
    if args.output_dir is not None:
        out = args.output_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join("results", "rl", f"{args.mode}_ppo_{ts}")
    os.makedirs(out, exist_ok=True)
    return out


def _linear_schedule(initial_value: float):
    """Linear learning-rate schedule from initial_value → 0."""
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Run PPO training.

    Returns a summary dict with:
      timesteps, runtime_s, steps_per_sec, output_dir,
      latest_saved, best_saved, vecnormalize_saved,
      obs_shape, action_shape, reward_stats, nan_flag, inf_flag
    """
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import (
            BaseCallback,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 and torch are required for train_ppo. "
            f"Install with: pip install stable-baselines3>=2.0.0,<3.0.0\n{e}"
        )

    from module5_control.env_factory import (
        build_training_env,
        build_eval_env,
        build_wec_params,
    )
    from module5_control.rl_config import (
        PPOConfig,
        EXPERIMENT_REACTIVE,
        EXPERIMENT_PERFECT_FORECAST,
        EXPERIMENT_REALISTIC_FORECAST,
        RL_DATA_SPLIT,
    )

    env_mode = _mode_to_env_mode(args.mode)
    output_dir = _ensure_output_dir(args)

    if args.timesteps > 10_000:
        print(
            f"[WARN] timesteps={args.timesteps} is larger than smoke-test "
            f"scale. Module 5.3B requires approval before multi-million-step "
            f"runs. If this is intentional, approve Module 5.3C first.",
            file=sys.stderr,
        )

    _ = build_wec_params()

    error_bank_path = args.error_bank
    if args.mode == "realistic" and error_bank_path is None:
        error_bank_path = RL_DATA_SPLIT.error_bank_train

    save_freq = args.save_freq
    if save_freq is None:
        save_freq = max(1000, args.timesteps // 10)

    # Adjust n_steps for smoke tests to ensure batch_size divides n_steps * n_envs
    # and that n_steps * n_envs >= batch_size (SB3 requirement).
    n_steps = args.n_steps
    if args.timesteps <= 10_000:
        # Smoke test: pick the smallest n_steps such that:
        #   (a) n_steps * n_envs is a positive multiple of batch_size
        #   (b) n_steps * n_envs >= batch_size (at least one mini-batch)
        # This simplifies to: n_steps = ceil(batch_size / n_envs)
        import math
        n_steps = math.ceil(args.batch_size / max(1, args.n_envs))
        # Cap at total timesteps so we don't collect more than requested
        n_steps = min(n_steps, max(1, args.timesteps // max(1, args.n_envs)))
        # Recheck: if batch_size > n_steps * n_envs, bump n_steps up
        while n_steps * args.n_envs < args.batch_size:
            n_steps += 1
    else:
        n_steps = min(args.n_steps, args.timesteps)

    # ------------------------------------------------------------------
    # Build training environment
    # ------------------------------------------------------------------
    train_env = build_training_env(
        mode              = env_mode,
        n_envs            = args.n_envs,
        max_episode_steps = args.episode_steps,
        error_bank_path   = error_bank_path,
        seed              = args.seed,
        use_synthetic     = args.use_synthetic,
    )

    obs_example = train_env.reset()
    obs_shape = tuple(obs_example.shape[1:]) if obs_example.ndim > 1 else obs_example.shape

    # ------------------------------------------------------------------
    # Build evaluation environment (validation split — 2022–2023)
    # VecNormalize stats are loaded from training env via save/load path.
    # ------------------------------------------------------------------
    vecnorm_tmp_path = os.path.join(output_dir, "vecnormalize_init.pkl")
    train_env.save(vecnorm_tmp_path)

    eval_env = build_eval_env(
        mode                = env_mode,
        n_envs              = 1,
        max_episode_steps   = args.episode_steps,
        vec_normalize_path  = vecnorm_tmp_path,
        error_bank_path     = error_bank_path,
        allow_final_test_bank = False,
        seed                = args.seed,
        split               = "validation",
        use_synthetic       = args.use_synthetic,
    )
    eval_env.training = False
    eval_env.norm_reward = False

    # ------------------------------------------------------------------
    # PPO hyperparameters — from rl_config.PPOConfig
    # ------------------------------------------------------------------
    experiment_cfg: PPOConfig = {
        "reactive":  EXPERIMENT_REACTIVE,
        "perfect":   EXPERIMENT_PERFECT_FORECAST,
        "realistic": EXPERIMENT_REALISTIC_FORECAST,
    }[args.mode]

    lr_schedule = _linear_schedule(experiment_cfg.learning_rate)

    policy_kwargs = dict(
        net_arch      = list(experiment_cfg.net_arch),
        activation_fn = torch.nn.Tanh,
    )

    model = PPO(
        policy            = experiment_cfg.policy_type,
        env               = train_env,
        learning_rate     = lr_schedule,
        n_steps           = n_steps,
        batch_size        = args.batch_size,
        n_epochs          = args.n_epochs,
        gamma             = experiment_cfg.gamma,
        gae_lambda        = experiment_cfg.gae_lambda,
        clip_range        = experiment_cfg.clip_range,
        ent_coef          = experiment_cfg.ent_coef,
        vf_coef           = experiment_cfg.vf_coef,
        max_grad_norm     = experiment_cfg.max_grad_norm,
        seed              = args.seed,
        policy_kwargs     = policy_kwargs,
        verbose           = 1,
        device            = "auto",
    )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    latest_path = os.path.join(output_dir, "checkpoints")
    os.makedirs(latest_path, exist_ok=True)
    checkpoint_cb = CheckpointCallback(
        save_freq      = max(1, save_freq // max(1, args.n_envs)),
        save_path      = latest_path,
        name_prefix    = "rl_model",
        save_replay_buffer = False,
        save_vecnormalize   = True,
        verbose        = 1,
    )

    best_path = os.path.join(output_dir, "best")
    os.makedirs(best_path, exist_ok=True)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = best_path,
        log_path             = best_path,
        eval_freq            = max(1, save_freq // max(1, args.n_envs)),
        n_eval_episodes      = 1 if args.timesteps <= 10_000 else 3,
        deterministic        = True,
        render               = False,
        verbose              = 1,
    )

    # ------------------------------------------------------------------
    # NaN / Inf tracking during training
    # ------------------------------------------------------------------
    class NanInfTracker(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)
            self.obs_nan_count = 0
            self.rew_nan_count = 0
            self.obs_inf_count = 0
            self.rew_inf_count = 0

        def _on_step(self) -> bool:
            obs = self.locals.get("obs")
            rewards = self.locals.get("rewards")
            if obs is not None:
                arr = np.asarray(obs)
                if not np.all(np.isfinite(arr)):
                    self.obs_nan_count += int(np.sum(~np.isfinite(arr) & ~np.isinf(arr)))
                    self.obs_inf_count += int(np.sum(np.isinf(arr)))
            if rewards is not None:
                arr = np.asarray(rewards)
                if not np.all(np.isfinite(arr)):
                    self.rew_nan_count += int(np.sum(~np.isfinite(arr) & ~np.isinf(arr)))
                    self.rew_inf_count += int(np.sum(np.isinf(arr)))
            return True

    nan_tracker = NanInfTracker(verbose=0)

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    t0 = time.time()
    timesteps_before = model.num_timesteps

    model.learn(
        total_timesteps = args.timesteps if args.timesteps > 0 else args.n_steps,
        callback        = [checkpoint_cb, eval_cb, nan_tracker],
        log_interval    = 1,
        reset_num_timesteps = True,
    )

    runtime = time.time() - t0
    actual_steps = model.num_timesteps - timesteps_before
    steps_per_sec = actual_steps / runtime if runtime > 0 else 0.0

    # ------------------------------------------------------------------
    # Save final models + VecNormalize
    # ------------------------------------------------------------------
    latest_model_path = os.path.join(output_dir, "latest_model.zip")
    model.save(latest_model_path)

    vecnorm_path = os.path.join(output_dir, "vecnormalize.pkl")
    train_env.save(vecnorm_path)

    best_model_candidates = [
        os.path.join(best_path, "best_model.zip"),
    ]
    best_saved = best_model_candidates[0] if os.path.exists(best_model_candidates[0]) else None

    # ------------------------------------------------------------------
    # Quick reload + deterministic inference sanity check
    # ------------------------------------------------------------------
    del model
    model2 = PPO.load(latest_model_path, env=None, device="auto")
    test_env = build_eval_env(
        mode                = env_mode,
        n_envs              = 1,
        max_episode_steps   = min(10, args.episode_steps),
        vec_normalize_path  = vecnorm_path,
        error_bank_path     = error_bank_path,
        allow_final_test_bank = False,
        seed                = args.seed,
        split               = "validation",
        use_synthetic       = args.use_synthetic,
    )
    test_obs = test_env.reset()
    action, _ = model2.predict(test_obs, deterministic=True)
    next_obs, reward, done, info = test_env.step(action)

    def _has(arr):
        a = np.asarray(arr)
        return bool(np.any(~np.isfinite(a)))

    nan_flag = bool(
        nan_tracker.obs_nan_count > 0
        or nan_tracker.rew_nan_count > 0
        or _has(test_obs)
        or _has(reward)
        or _has(next_obs)
    )
    inf_flag = bool(
        nan_tracker.obs_inf_count > 0
        or nan_tracker.rew_inf_count > 0
    )

    action_shape = int(getattr(model2.action_space, "n", 0))

    summary = dict(
        mode              = args.mode,
        env_mode          = env_mode,
        timesteps         = int(actual_steps),
        runtime_s         = round(runtime, 3),
        steps_per_sec     = round(steps_per_sec, 2),
        output_dir        = output_dir,
        latest_saved      = latest_model_path if os.path.exists(latest_model_path) else None,
        best_saved        = best_saved,
        vecnormalize_saved= vecnorm_path if os.path.exists(vecnorm_path) else None,
        obs_shape         = list(obs_shape),
        action_n          = action_shape,
        nan_flag          = nan_flag,
        inf_flag          = inf_flag,
        nan_obs_count     = nan_tracker.obs_nan_count,
        nan_rew_count     = nan_tracker.rew_nan_count,
        inf_obs_count     = nan_tracker.obs_inf_count,
        inf_rew_count     = nan_tracker.rew_inf_count,
        error_bank_used   = error_bank_path,
        episode_steps     = args.episode_steps,
        n_envs            = args.n_envs,
        seed              = args.seed,
    )

    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[train_ppo] Summary written to {summary_path}")
    return summary


def main() -> int:
    args = _parse_args()

    if args.timesteps <= 0:
        print("[ERROR] --timesteps must be positive.", file=sys.stderr)
        return 2

    summary = train(args)

    print("\n========== TRAINING SUMMARY (Module 5.3B smoke test) ==========")
    for k, v in summary.items():
        print(f"  {k:25s}: {v}")
    print("================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
