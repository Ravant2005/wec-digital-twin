"""
evaluate_ppo.py — PPO policy evaluation (Module 5.3B).

============================================================
PURPOSE
============================================================

Loads a trained PPO checkpoint + VecNormalize statistics and
evaluates the policy deterministically on the validation split
(2022–2023) or, if EXPLICITLY requested, on the final test split
(2024–2025).

This script is SAFE BY DEFAULT:
  - It never uses the 2024–2025 ERA5 data or the final-test error bank
    unless --final-test is passed as an explicit, deliberate opt-in.
  - It always calls model.predict(..., deterministic=True).
  - It loads VecNormalize with training=False, norm_reward=False
    so statistics are FROZEN — no adaptation to eval data.

Per-episode metrics tracked:
    mean_power_W            mean PTO absorbed power
    total_energy_Wh         cumulative PTO energy
    total_reward            raw (un-normalised) cumulative reward
    n_latch_events          number of latch state changes
    max_abs_displacement_m  max |x|
    mean_abs_displacement_m mean |x|
    max_abs_velocity_ms     max |v|
    end_stop_events         steps where |x| > 3 m
    obs_nan_count           total non-finite observation entries
    rew_nan_count           total non-finite rewards
    steps                   episode length in control steps
    terminated / truncated  Gymnasium terminal flags

============================================================
USAGE
============================================================

Evaluate latest_model on validation (2022–2023 conceptually —
smoke tests use synthetic replays):
    python -m module5_control.evaluate_ppo \\
        --model results/rl/reactive_ppo_XXX/latest_model.zip \\
        --vecnormalize results/rl/reactive_ppo_XXX/vecnormalize.pkl \\
        --mode reactive --n-episodes 3

Final test (2024–2025) — EXPLICIT opt-in only:
    python -m module5_control.evaluate_ppo \\
        --model .../best/best_model.zip \\
        --vecnormalize .../vecnormalize.pkl \\
        --mode realistic --n-episodes 7 --final-test

Module 5.3B SMOKE TEST EVALUATION:
    Do NOT use --final-test. Always use validation-safe defaults.
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
        description="PPO policy evaluation for WEC latching (Module 5.3B).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=str, required=True,
                   help="Path to PPO model .zip (e.g. latest_model.zip).")
    p.add_argument("--vecnormalize", type=str, required=True,
                   help="Path to VecNormalize .pkl statistics. Required for frozen stats.")
    p.add_argument("--mode", type=str, default="reactive",
                   choices=["reactive", "perfect", "realistic"],
                   help="Forecast information mode.")
    p.add_argument("--error-bank", type=str, default=None,
                   help="Custom error-bank path for realistic_forecast mode.")
    p.add_argument("--final-test", action="store_true",
                   help="EXPLICIT opt-in: use 2024–2025 final-test split. "
                        "Refused for --mode realistic unless combined with "
                        "--allow-final-test-error-bank. "
                        "SAFE DEFAULT: validation 2022–2023.")
    p.add_argument("--allow-final-test-error-bank", action="store_true",
                   help="Permit using the 2024–2025 error bank with "
                        "--mode realistic --final-test. For frozen policy "
                        "evaluation only. NEVER during training or "
                        "hyperparameter selection.")
    p.add_argument("--n-episodes", type=int, default=3,
                   help="Number of evaluation episodes.")
    p.add_argument("--episode-steps", type=int, default=7200,
                   help="Maximum RL control steps per episode.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel evaluation envs (1 = deterministic default).")
    p.add_argument("--seed", type=int, default=999,
                   help="Evaluation master seed.")
    p.add_argument("--output", type=str, default=None,
                   help="Output path for evaluation JSON. Defaults to "
                        "<model_dir>/eval_results.json.")
    return p.parse_args()


def _mode_to_env_mode(mode_arg: str) -> str:
    mapping = {
        "reactive":  "reactive",
        "perfect":   "perfect_forecast",
        "realistic": "realistic_forecast",
    }
    return mapping[mode_arg]


def _resolve_error_bank(args: argparse.Namespace) -> Optional[str]:
    """
    Resolve which error bank to use for realistic_forecast mode.

    Training / validation (default) → 2022–2023 safe bank.
    Final test + explicit allow      → 2024–2025 bank.
    """
    from module5_control.rl_config import (
        RL_DATA_SPLIT,
        assert_safe_training_error_bank,
    )

    if args.mode != "realistic":
        return None

    if args.final_test and args.allow_final_test_error_bank:
        return RL_DATA_SPLIT.error_bank_final_eval

    bank = args.error_bank or RL_DATA_SPLIT.error_bank_train
    assert_safe_training_error_bank(bank)
    return bank


def _run_single_episode(model, vec_env, max_steps: int) -> Dict[str, Any]:
    """
    Run one deterministic evaluation episode.

    Returns per-episode metrics dict.
    """
    obs = vec_env.reset()
    done = False
    total_reward = 0.0
    total_energy_Wh = 0.0
    n_latches = 0
    prev_latch = None
    max_abs_x = 0.0
    sum_abs_x = 0.0
    max_abs_v = 0.0
    end_stop_events = 0
    power_values: List[float] = []
    obs_nan_count = 0
    rew_nan_count = 0
    steps = 0
    terminated = False
    truncated = False

    END_STOP_M = 3.0

    while not done and steps < max_steps:
        action, _ = model.predict(obs, deterministic=True)

        # Latched vs previous — count switches
        # action comes as array for vec envs
        act_scalar = int(action.flat[0]) if hasattr(action, "flat") else int(action)
        if prev_latch is not None and act_scalar != prev_latch:
            n_latches += 1
        prev_latch = act_scalar

        next_obs, reward, done_flag, info_list = vec_env.step(action)

        steps += 1

        obs_arr = np.asarray(obs)
        if not np.all(np.isfinite(obs_arr)):
            obs_nan_count += int(np.sum(~np.isfinite(obs_arr)))

        rew_arr = np.asarray(reward)
        if not np.all(np.isfinite(rew_arr)):
            rew_nan_count += int(np.sum(~np.isfinite(rew_arr)))

        raw_reward = float(np.asarray(reward).flat[0])
        total_reward += raw_reward

        info = info_list[0] if isinstance(info_list, list) and len(info_list) else {}
        if isinstance(info, dict):
            total_energy_Wh = float(info.get("cumulative_energy_Wh", total_energy_Wh))
            x = float(info.get("x", 0.0))
            v = float(info.get("v", 0.0))
            mean_p = float(info.get("mean_step_power_W", 0.0))
            power_values.append(mean_p)
            ax = abs(x)
            max_abs_x = max(max_abs_x, ax)
            sum_abs_x += ax
            max_abs_v = max(max_abs_v, abs(v))
            if ax > END_STOP_M:
                end_stop_events += 1

        done_scalar = bool(np.asarray(done_flag).flat[0]) if hasattr(done_flag, "flat") else bool(done_flag)
        if done_scalar:
            truncated = True
            done = True

        obs = next_obs

    mean_power_W = float(np.mean(power_values)) if power_values else 0.0
    mean_abs_x = (sum_abs_x / steps) if steps > 0 else 0.0

    return dict(
        steps=steps,
        mean_power_W=round(mean_power_W, 4),
        total_energy_Wh=round(total_energy_Wh, 6),
        total_reward=round(total_reward, 4),
        n_latch_events=n_latches,
        max_abs_displacement_m=round(max_abs_x, 4),
        mean_abs_displacement_m=round(mean_abs_x, 4),
        max_abs_velocity_ms=round(max_abs_v, 4),
        end_stop_events=end_stop_events,
        obs_nan_count=obs_nan_count,
        rew_nan_count=rew_nan_count,
        terminated=terminated,
        truncated=truncated,
    )


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    try:
        from stable_baselines3 import PPO
    except ImportError as e:
        raise ImportError(
            "stable_baselines3 required. pip install stable-baselines3>=2.0.0,<3.0.0"
        ) from e

    from module5_control.env_factory import build_eval_env, build_wec_params

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"PPO model not found: {args.model}")

    env_mode = _mode_to_env_mode(args.mode)
    error_bank_path = _resolve_error_bank(args)

    if args.mode == "realistic" and args.final_test and not args.allow_final_test_error_bank:
        raise SystemExit(
            "[evaluate_ppo] SAFETY: --mode realistic --final-test requires "
            "--allow-final-test-error-bank as a second explicit opt-in. "
            "The 2024–2025 error bank is reserved for frozen-policy evaluation "
            "ONLY. If this is indeed the final evaluation, add the flag."
        )

    _ = build_wec_params()

    allow_final = bool(args.final_test and args.allow_final_test_error_bank)
    split = "test" if args.final_test else "validation"

    vec_env = build_eval_env(
        mode                  = env_mode,
        n_envs                = args.n_envs,
        max_episode_steps     = args.episode_steps,
        vec_normalize_path    = args.vecnormalize,
        error_bank_path       = error_bank_path,
        allow_final_test_bank = allow_final,
        seed                  = args.seed,
        split                 = split,
        use_synthetic         = False,
    )
    vec_env.training = False
    vec_env.norm_reward = False

    model = PPO.load(args.model, env=None, device="auto")

    obs = vec_env.reset()
    obs_shape = list(obs.shape[1:]) if obs.ndim > 1 else list(obs.shape)
    action_n = int(getattr(model.action_space, "n", 0))

    episodes: List[Dict[str, Any]] = []
    t0 = time.time()
    for ep_i in range(args.n_episodes):
        ep_metrics = _run_single_episode(model, vec_env, args.episode_steps)
        ep_metrics["episode_index"] = ep_i
        episodes.append(ep_metrics)
    runtime_s = round(time.time() - t0, 3)

    def _agg(key):
        vals = [e[key] for e in episodes]
        return round(float(np.mean(vals)), 6), round(float(np.std(vals)), 6)

    m_power_mean, m_power_std   = _agg("mean_power_W")
    m_energy_mean, m_energy_std = _agg("total_energy_Wh")
    m_reward_mean, m_reward_std = _agg("total_reward")
    m_latch_mean, m_latch_std   = _agg("n_latch_events")
    m_maxx_mean, m_maxx_std     = _agg("max_abs_displacement_m")
    m_endstop_sum = int(np.sum([e["end_stop_events"] for e in episodes]))
    m_obsnan_sum  = int(np.sum([e["obs_nan_count"] for e in episodes]))
    m_rew_nan_sum = int(np.sum([e["rew_nan_count"] for e in episodes]))

    summary = dict(
        model_path            = args.model,
        vecnormalize_path     = args.vecnormalize,
        mode                  = args.mode,
        env_mode              = env_mode,
        split                 = split,
        used_final_test_bank  = allow_final,
        error_bank_path       = error_bank_path,
        seed                  = args.seed,
        n_episodes            = args.n_episodes,
        episode_steps         = args.episode_steps,
        runtime_s             = runtime_s,
        obs_shape             = obs_shape,
        action_n              = action_n,
        deterministic         = True,
        episodes              = episodes,
        aggregate             = dict(
            mean_power_W_mean         = m_power_mean,
            mean_power_W_std          = m_power_std,
            total_energy_Wh_mean      = m_energy_mean,
            total_energy_Wh_std       = m_energy_std,
            total_reward_mean         = m_reward_mean,
            total_reward_std          = m_reward_std,
            n_latch_events_mean       = m_latch_mean,
            n_latch_events_std        = m_latch_std,
            max_abs_displacement_m_mean = m_maxx_mean,
            max_abs_displacement_m_std  = m_maxx_std,
            total_end_stop_events     = m_endstop_sum,
            total_obs_nan             = m_obsnan_sum,
            total_rew_nan             = m_rew_nan_sum,
        ),
    )

    out_path = args.output
    if out_path is None:
        model_dir = os.path.dirname(args.model) or "."
        tag = "final_test" if args.final_test else "validation"
        out_path = os.path.join(model_dir, f"eval_{tag}_results.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[evaluate_ppo] Wrote results → {out_path}")
    return summary


def main() -> int:
    args = _parse_args()
    summary = evaluate(args)

    agg = summary["aggregate"]
    print("\n========== EVALUATION SUMMARY (Module 5.3B) ==========")
    print(f"  mode                 : {summary['mode']} ({summary['env_mode']})")
    print(f"  split                : {summary['split']}")
    print(f"  final_test_bank_used : {summary['used_final_test_bank']}")
    print(f"  episodes             : {summary['n_episodes']} × {summary['episode_steps']} steps")
    print(f"  runtime              : {summary['runtime_s']} s")
    print(f"  mean_power_W         : {agg['mean_power_W_mean']:.2f} ± {agg['mean_power_W_std']:.2f}")
    print(f"  total_energy_Wh      : {agg['total_energy_Wh_mean']:.4f} ± {agg['total_energy_Wh_std']:.4f}")
    print(f"  total_reward         : {agg['total_reward_mean']:.2f} ± {agg['total_reward_std']:.2f}")
    print(f"  n_latch_events       : {agg['n_latch_events_mean']:.1f} ± {agg['n_latch_events_std']:.1f}")
    print(f"  max|x| mean          : {agg['max_abs_displacement_m_mean']:.3f} m")
    print(f"  end_stop_events total: {agg['total_end_stop_events']}")
    print(f"  total obs NaN        : {agg['total_obs_nan']}")
    print(f"  total rew NaN        : {agg['total_rew_nan']}")
    print("=======================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
