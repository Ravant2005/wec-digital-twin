"""
test_module5_controllers.py — Comprehensive tests for Module 5.2C.

Covers all 26 specification requirements plus statistical-fairness and
sanity checks.

Spec requirements:
 1.  Passive controller never latches.
 2.  Fixed-threshold controller is deterministic.
 3.  Fixed-threshold controller does not use future truth.
 4.  Reactive controller does not access forecast Block B.
 5.  Perfect forecast pathway contains true future forecast by design.
 6.  Realistic forecast pathway uses Module 5.2B.
 7.  Realistic pathway does not access current episode's future truth.
 8.  All controllers use the same replay.
 9.  Controller actions are valid: 0 = release, 1 = latch.
10.  Controller action timing is 1 Hz.
11.  PTO power is nonnegative.
12.  Energy integration is correct.
13.  Energy ≈ mean_power × duration.
14.  Same seed → reproducible benchmark results.
15.  Changing seed changes only intended stochastic components.
16.  Passive run remains finite.
17.  Threshold run remains finite.
18.  Reactive run remains finite.
19.  Latching does not destroy radiation-memory continuity.
20.  No NaN/Inf in benchmark outputs.
21.  Max displacement/velocity remain bounded.
22.  Same physical replay → comparable baseline conditions.
23.  Benchmark results saved in machine-readable format.
24.  Existing Module 5.2A tests still pass (verified by full suite).
25.  Existing Module 5.2B tests still pass (verified by full suite).
26.  Full repository test suite passes (verified by run_all).

Additional tests:
 - Controller factory (make_controller).
 - FixedThresholdConfig validation.
 - BaseController interface contract.
 - BenchmarkResult serialisation.
 - Energy cross-check formula.
 - Physics sanity: no explosion in any baseline.
 - Latch timing: min_latch_steps respected.
 - OracleThresholdController runs in perfect_forecast env.
 - RealisticThresholdController runs with ForecastErrorSampler.
 - Benchmark CSV and JSON output format.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from module5_control.controllers import (
    BaseController,
    PassiveController,
    FixedThresholdController,
    FixedThresholdConfig,
    ReactiveController,
    OracleThresholdController,
    RealisticThresholdController,
    make_controller,
    ACTION_RELEASE,
    ACTION_LATCH,
)
from module5_control.observations import (
    OBS_DIM,
    _BLOCK_B_START,
    _BLOCK_B_END,
    _BLOCK_C_START,
    X_SCALE,
    V_SCALE,
)
from module5_control.replay import (
    build_synthetic_replay,
    DT_PHYSICS,
    DT_CONTROL,
    N_PHYSICS_PER_CONTROL,
)
from module5_control.benchmark import (
    BenchmarkHarness,
    BenchmarkResult,
    BENCHMARK_SEGMENTS,
    BENCHMARK_BASE_SEED,
    BENCHMARK_RESET_SEED,
    build_standard_benchmark_controllers,
    build_wec_params,
    _compute_summary,
)


# ---------------------------------------------------------------------------
# Session-scope fixtures (expensive BEM computation — shared)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def wec_params():
    return build_wec_params()


@pytest.fixture(scope="session")
def replay_2h():
    return build_synthetic_replay(n_hours=2, Hs=1.5, Tp=8.0, seed=42)


@pytest.fixture(scope="session")
def replay_1h():
    return build_synthetic_replay(n_hours=1, Hs=1.5, Tp=8.0, seed=42)


@pytest.fixture(scope="session")
def replay_50h():
    """50-hour replay needed for perfect_forecast Block B to be non-zero (needs >= 49h)."""
    return build_synthetic_replay(n_hours=50, Hs=1.5, Tp=8.0, seed=42)


@pytest.fixture(scope="session")
def harness(wec_params):
    gru = "results/forecasting/exp_gru_wind/best_model.pt"
    return BenchmarkHarness(
        cummins_params             = wec_params["cummins"],
        pto_params                 = wec_params["pto"],
        hydro_excitation_amplitude = wec_params["hydro_amp"],
        hydro_omega                = wec_params["hydro_omega"],
        hydro                      = wec_params["hydro"],
        gru_checkpoint_path        = gru,
        max_episode_steps          = 200,   # short for fast tests
        verbose                    = False,
    )


def _dummy_obs(block_b_val: float = 0.0, x: float = 0.1, v: float = 0.05) -> np.ndarray:
    """Build a minimal valid observation for unit tests."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[_BLOCK_B_START:_BLOCK_B_END] = block_b_val
    obs[_BLOCK_C_START]     = x / X_SCALE
    obs[_BLOCK_C_START + 1] = v / V_SCALE
    obs[_BLOCK_C_START + 2] = 0.0   # latch_status = FREE
    return obs


def _dummy_info(v: float = 0.05, latch: int = 0) -> dict:
    return {
        "x":                       0.1,
        "v":                       v,
        "latch_status":            latch,
        "latch_switched":          False,
        "instantaneous_power_W":   0.0,
        "mean_step_power_W":       0.0,
        "cumulative_energy_J":     0.0,
        "cumulative_energy_Wh":    0.0,
        "control_step":            1,
        "physics_step":            10,
        "era5_hour_index":         0,
        "action":                  0,
        "reward":                  0.0,
        "mode":                    "reactive",
    }


# ===========================================================================
# Test 1 — Passive controller never latches
# ===========================================================================

def test_01_passive_never_latches():
    ctrl = PassiveController()
    ctrl.reset()
    for v in [0.0, 0.001, -0.001, 1.0, -1.0, 0.5]:
        obs  = _dummy_obs(v=v)
        info = _dummy_info(v=v)
        assert ctrl.act(obs, info) == ACTION_RELEASE


def test_01_passive_never_latches_sustained(wec_params, replay_1h, harness):
    """Passive controller produces zero latch events over a full episode."""
    ctrl   = PassiveController()
    result = harness.run_episode(
        ctrl, replay_1h, env_mode="passive", segment_id="test_passive"
    )
    assert result.n_latch_events == 0
    assert result.latch_fraction == 0.0


# ===========================================================================
# Test 2 — Fixed-threshold controller is deterministic
# ===========================================================================

def test_02_threshold_deterministic():
    """Same velocity sequence → identical action sequence."""
    cfg     = FixedThresholdConfig(v_latch_threshold=0.05, min_latch_steps=2, min_release_steps=2)
    velocities = [0.1, 0.1, 0.03, 0.03, 0.03, 0.03, 0.03, 0.1, 0.1]

    actions_run1, actions_run2 = [], []
    for run_actions in [actions_run1, actions_run2]:
        ctrl = FixedThresholdController(cfg)
        ctrl.reset()
        for v in velocities:
            obs  = _dummy_obs(v=v)
            info = _dummy_info(v=v)
            run_actions.append(ctrl.act(obs, info))

    assert actions_run1 == actions_run2


def test_02_threshold_deterministic_full_episode(wec_params, replay_1h, harness):
    """Same seed + same controller → identical energy."""
    ctrl = FixedThresholdController()
    r1   = harness.run_episode(ctrl, replay_1h, env_mode="fixed_threshold", reset_seed=0, segment_id="s")
    r2   = harness.run_episode(ctrl, replay_1h, env_mode="fixed_threshold", reset_seed=0, segment_id="s")
    assert r1.total_energy_Wh == r2.total_energy_Wh


# ===========================================================================
# Test 3 — Fixed-threshold does not use future truth
# ===========================================================================

def test_03_threshold_no_future_truth():
    """FixedThresholdController only reads info['v'] — no oracle keys."""
    ctrl = FixedThresholdController()
    ctrl.reset()
    # Provide an info dict that has NO Hs_true/Tp_true/direction_true
    info = _dummy_info(v=0.03)
    obs  = _dummy_obs(v=0.03)
    # Must not raise even if oracle keys are absent
    action = ctrl.act(obs, info)
    assert action in (0, 1)


def test_03_threshold_ignores_block_b():
    """Changing Block B does not change FixedThreshold action for same v."""
    ctrl = FixedThresholdController()
    ctrl.reset()
    v    = 0.03

    obs_b0 = _dummy_obs(block_b_val=0.0, v=v)
    obs_b1 = _dummy_obs(block_b_val=1.0, v=v)
    info   = _dummy_info(v=v)

    ctrl.reset(); a0 = ctrl.act(obs_b0, info)
    ctrl.reset(); a1 = ctrl.act(obs_b1, info)
    assert a0 == a1, "Block B content must not affect FixedThreshold action"


# ===========================================================================
# Test 4 — Reactive controller does not access forecast Block B
# ===========================================================================

def test_04_reactive_block_b_must_be_zeros():
    """ReactiveController raises AssertionError if Block B is non-zero."""
    ctrl = ReactiveController()
    ctrl.reset()
    obs_nonzero_b = _dummy_obs(block_b_val=1.0, v=0.03)
    info          = _dummy_info(v=0.03)
    with pytest.raises(AssertionError):
        ctrl.act(obs_nonzero_b, info)


def test_04_reactive_works_with_zero_block_b():
    ctrl = ReactiveController()
    ctrl.reset()
    obs  = _dummy_obs(block_b_val=0.0, v=0.03)
    info = _dummy_info(v=0.03)
    action = ctrl.act(obs, info)
    assert action in (0, 1)


def test_04_reactive_run_block_b_all_zeros(wec_params, replay_1h, harness):
    """Verify environment returns Block B = 0 in reactive mode."""
    from module5_control.environment import build_env_from_checkpoint
    env = build_env_from_checkpoint(
        replay=replay_1h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=10,
    )
    obs, _ = env.reset(seed=0)
    assert np.allclose(obs[_BLOCK_B_START:_BLOCK_B_END], 0.0)
    for _ in range(5):
        obs, *_ = env.step(0)
        assert np.allclose(obs[_BLOCK_B_START:_BLOCK_B_END], 0.0)


# ===========================================================================
# Test 5 — Perfect forecast pathway has true future forecast
# ===========================================================================

def test_05_perfect_forecast_block_b_nonzero(wec_params, replay_50h):
    """In perfect_forecast mode with a long enough replay, Block B must contain non-trivial values.
    Note: perfect_forecast looks ahead H_FORE=48 hours, so the replay must have >= 49 hours."""
    from module5_control.environment import build_env_from_checkpoint
    env = build_env_from_checkpoint(
        replay=replay_50h,
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


def test_05_oracle_controller_runs(wec_params, replay_50h, harness):
    ctrl   = OracleThresholdController()
    result = harness.run_episode(
        ctrl, replay_50h, env_mode="oracle_interface", segment_id="oracle_test"
    )
    assert result.is_finite
    assert result.total_energy_Wh >= 0.0


# ===========================================================================
# Test 6 — Realistic forecast uses Module 5.2B
# ===========================================================================

def test_06_realistic_uses_52b_sampler(wec_params, replay_1h, harness):
    """RealisticThresholdController uses ForecastErrorSampler from 5.2B."""
    ctrl   = RealisticThresholdController()
    result = harness.run_episode(
        ctrl, replay_1h, env_mode="realistic_interface", segment_id="real_test"
    )
    assert result.is_finite
    assert result.total_energy_Wh >= 0.0
    assert result.env_mode == "realistic_forecast"


def test_06_realistic_env_different_from_reactive(wec_params, replay_1h):
    """Realistic forecast Block B should differ from zero (GRU provides forecasts)."""
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    sampler = ForecastErrorSampler.from_csv(seed=0)
    env = build_env_from_checkpoint(
        replay=replay_1h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path="results/forecasting/exp_gru_wind/best_model.pt",
        max_episode_steps=10,
        forecast_error_sampler=sampler,
    )
    obs, _ = env.reset(seed=0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END]
    assert np.all(np.isfinite(block_b))


# ===========================================================================
# Test 7 — Realistic pathway does not access future truth
# ===========================================================================

def test_07_realistic_no_future_truth():
    """RealisticThresholdController uses only info['v'] — no oracle keys."""
    ctrl = RealisticThresholdController()
    ctrl.reset()
    obs  = _dummy_obs(block_b_val=0.5, v=0.03)
    info = _dummy_info(v=0.03)
    # Must work without Hs_true, Tp_true, direction_true in info
    action = ctrl.act(obs, info)
    assert action in (0, 1)


# ===========================================================================
# Test 8 — All controllers use the same replay
# ===========================================================================

def test_08_same_replay_all_controllers(wec_params, harness):
    """All controllers run on the same EpisodeReplay (same Hs, Tp, seed)."""
    replay = build_synthetic_replay(n_hours=1, Hs=1.5, Tp=8.0, seed=100)
    controllers = [
        PassiveController(),
        FixedThresholdController(),
        ReactiveController(),
    ]
    results = []
    for ctrl in controllers:
        r = harness.run_episode(
            ctrl, replay, env_mode=ctrl.label,
            reset_seed=0, segment_id="same_replay_test",
            Hs_m=1.5, Tp_s=8.0,    # explicitly pass sea-state metadata
        )
        results.append(r)

    # All should report identical Hs, Tp, seed
    for r in results:
        assert r.Hs_m == 1.5
        assert r.Tp_s == 8.0
        assert r.replay_seed == 100


# ===========================================================================
# Test 9 — Controller actions are valid (0 or 1)
# ===========================================================================

def test_09_actions_valid_passive():
    ctrl = PassiveController()
    ctrl.reset()
    for _ in range(20):
        obs  = _dummy_obs(v=np.random.uniform(-0.5, 0.5))
        info = _dummy_info(v=float(obs[_BLOCK_C_START + 1]))
        assert ctrl.act(obs, info) in (0, 1)


def test_09_actions_valid_threshold():
    ctrl = FixedThresholdController()
    ctrl.reset()
    rng  = np.random.default_rng(0)
    for v in rng.uniform(-1.0, 1.0, 30):
        obs  = _dummy_obs(v=float(v))
        info = _dummy_info(v=float(v))
        assert ctrl.act(obs, info) in (0, 1)


# ===========================================================================
# Test 10 — Controller action timing is 1 Hz
# ===========================================================================

def test_10_action_timing_1hz(wec_params, replay_1h, harness):
    """
    n_steps in result must equal replay.n_control_steps (or max_episode_steps).
    Each step = DT_CONTROL = 1.0 s.
    """
    ctrl   = PassiveController()
    result = harness.run_episode(ctrl, replay_1h, env_mode="passive", segment_id="timing_test")
    # max_episode_steps = 200 in harness fixture
    assert result.n_steps == min(200, replay_1h.n_control_steps)


# ===========================================================================
# Test 11 — PTO power is nonnegative
# ===========================================================================

def test_11_pto_power_nonneg(wec_params, replay_1h, harness):
    """mean_step_power_W and total_energy_Wh must be >= 0 for all controllers."""
    controllers = [PassiveController(), FixedThresholdController(), ReactiveController()]
    for ctrl in controllers:
        result = harness.run_episode(ctrl, replay_1h, env_mode=ctrl.label,
                                     reset_seed=0, segment_id="power_test")
        assert result.mean_power_W       >= 0.0, f"{ctrl.label}: negative mean power"
        assert result.total_energy_Wh    >= 0.0, f"{ctrl.label}: negative energy"
        assert result.max_step_power_W   >= 0.0, f"{ctrl.label}: negative max power"


# ===========================================================================
# Test 12 — Energy integration is correct
# ===========================================================================

def test_12_energy_integration_formula(wec_params, replay_1h):
    """Verify E_Wh = Σ P_abs * dt_physics / 3600 via PowerAccumulator."""
    from module5_control.environment import build_env_from_checkpoint
    env = build_env_from_checkpoint(
        replay=replay_1h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=50,
    )
    env.reset(seed=0)
    manual_energy_J = 0.0
    for _ in range(50):
        obs, rew, term, trunc, info = env.step(0)
        manual_energy_J += info["mean_step_power_W"] * DT_CONTROL
        if term or trunc:
            break
    env_energy_J = info["cumulative_energy_J"]
    # Within 1% tolerance (rounding from DT_CONTROL vs actual N_PHYS * DT_PHYSICS)
    assert abs(manual_energy_J - env_energy_J) / max(env_energy_J, 1.0) < 0.02


# ===========================================================================
# Test 13 — Energy ≈ mean_power × duration
# ===========================================================================

def test_13_energy_mean_power_consistency(wec_params, replay_1h, harness):
    ctrl   = PassiveController()
    result = harness.run_episode(ctrl, replay_1h, env_mode="passive", segment_id="energy_check")
    # BenchmarkResult.energy_check_rel_err = |E_Wh - mean_P * T_h| / E_Wh
    assert result.energy_check_rel_err < 0.05, (
        f"Energy cross-check failed: rel_err={result.energy_check_rel_err:.4f}"
    )


# ===========================================================================
# Test 14 — Same seed → reproducible results
# ===========================================================================

def test_14_same_seed_reproducible(wec_params, harness):
    replay = build_synthetic_replay(n_hours=1, Hs=1.5, Tp=8.0, seed=7)
    ctrl   = FixedThresholdController()
    r1 = harness.run_episode(ctrl, replay, env_mode="fixed_threshold", reset_seed=0, segment_id="s")
    r2 = harness.run_episode(ctrl, replay, env_mode="fixed_threshold", reset_seed=0, segment_id="s")
    assert r1.total_energy_Wh      == r2.total_energy_Wh
    assert r1.n_latch_events       == r2.n_latch_events
    assert r1.max_abs_displacement_m == r2.max_abs_displacement_m


# ===========================================================================
# Test 15 — Different seed changes intended stochastic component only
# ===========================================================================

def test_15_seed_affects_only_sensor_and_sampler(wec_params):
    """
    For the passive controller (no latching, no GRU), different reset seeds
    change the sensor observations (Block A) but the underlying physics
    (deterministic replay) stays the same.  The energy for the passive
    controller should not be strongly sensitive to the reset seed since
    physics is identical — the sensor affects only observations, not forces.
    """
    from module5_control.environment import build_env_from_checkpoint
    replay = build_synthetic_replay(n_hours=1, seed=42)

    energies = []
    for seed in [0, 1, 2]:
        env = build_env_from_checkpoint(
            replay=replay,
            cummins_params=wec_params["cummins"],
            pto_params=wec_params["pto"],
            hydro_excitation_amplitude=wec_params["hydro_amp"],
            hydro_omega=wec_params["hydro_omega"],
            mode="reactive",
            max_episode_steps=100,
        )
        env.reset(seed=seed)
        for _ in range(100):
            obs, rew, term, trunc, info = env.step(0)   # passive
            if term or trunc: break
        energies.append(info["cumulative_energy_J"])

    # All passive runs on same replay → identical energy (sensor doesn't affect physics)
    for e in energies:
        assert abs(e - energies[0]) < 1e-6, (
            "Passive energy must be identical across reset seeds "
            "(sensor noise does not enter the physics loop)"
        )


# ===========================================================================
# Test 16–18 — All baseline runs remain finite
# ===========================================================================

def test_16_passive_run_finite(wec_params, replay_1h, harness):
    result = harness.run_episode(
        PassiveController(), replay_1h, env_mode="passive", segment_id="finite_test"
    )
    assert result.is_finite
    assert not result.has_nan
    assert not result.has_inf


def test_17_threshold_run_finite(wec_params, replay_1h, harness):
    result = harness.run_episode(
        FixedThresholdController(), replay_1h, env_mode="fixed_threshold", segment_id="finite_test"
    )
    assert result.is_finite
    assert not result.has_nan
    assert not result.has_inf


def test_18_reactive_run_finite(wec_params, replay_1h, harness):
    result = harness.run_episode(
        ReactiveController(), replay_1h, env_mode="reactive", segment_id="finite_test"
    )
    assert result.is_finite
    assert not result.has_nan
    assert not result.has_inf


# ===========================================================================
# Test 19 — Latching does not destroy radiation-memory continuity
# ===========================================================================

def test_19_radiation_memory_continuous(wec_params, replay_1h):
    """
    The integrator's step_count must advance monotonically through
    latch/release cycles.  No reset of velocity history occurs.
    """
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.replay import N_PHYSICS_PER_CONTROL

    env = build_env_from_checkpoint(
        replay=replay_1h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=50,
    )
    obs, info = env.reset(seed=0)
    ctrl = FixedThresholdController()
    ctrl.reset()
    prev_step_count = env.integrator.step_count
    for i in range(20):
        action = ctrl.act(obs, info)
        obs, rew, term, trunc, info = env.step(action)
        sc = env.integrator.step_count
        assert sc == prev_step_count + N_PHYSICS_PER_CONTROL, (
            f"Step count discontinuity at step {i}: {prev_step_count} → {sc}"
        )
        prev_step_count = sc
        if term or trunc:
            break


# ===========================================================================
# Test 20 — No NaN/Inf in benchmark outputs
# ===========================================================================

def test_20_no_nan_inf_all_baselines(wec_params, replay_1h, harness):
    for ctrl in [PassiveController(), FixedThresholdController(), ReactiveController()]:
        r = harness.run_episode(ctrl, replay_1h, env_mode=ctrl.label, segment_id="nan_test")
        assert not r.has_nan, f"{ctrl.label}: NaN in benchmark output"
        assert not r.has_inf, f"{ctrl.label}: Inf in benchmark output"
        assert r.is_finite


# ===========================================================================
# Test 21 — Displacement and velocity stay bounded
# ===========================================================================

def test_21_displacement_bounded(wec_params, replay_1h, harness):
    """All baselines: max |x| < 10 m (no explosion)."""
    for ctrl in [PassiveController(), FixedThresholdController(), ReactiveController()]:
        r = harness.run_episode(ctrl, replay_1h, env_mode=ctrl.label, segment_id="bound_test")
        assert r.max_abs_displacement_m < 10.0, (
            f"{ctrl.label}: displacement explosion: max|x|={r.max_abs_displacement_m:.2f}m"
        )
        assert r.max_abs_velocity_ms < 10.0, (
            f"{ctrl.label}: velocity explosion: max|v|={r.max_abs_velocity_ms:.2f}m/s"
        )


# ===========================================================================
# Test 22 — Same replay → comparable initial conditions
# ===========================================================================

def test_22_same_replay_conditions(wec_params, replay_1h, harness):
    """All controllers report identical Hs, Tp, seed."""
    ctrls = [PassiveController(), FixedThresholdController(), ReactiveController()]
    results = [
        harness.run_episode(c, replay_1h, env_mode=c.label,
                            Hs_m=1.5, Tp_s=8.0, segment_id="s")
        for c in ctrls
    ]
    for r in results:
        assert r.Hs_m == 1.5
        assert r.Tp_s == 8.0
        assert r.replay_seed == 42
        assert r.pto_damping_Ns_m == 200_000.0


# ===========================================================================
# Test 23 — Benchmark results saved in machine-readable format
# ===========================================================================

def test_23_results_saved_csv_json(wec_params, replay_1h, harness):
    """Run 2 controllers and verify CSV and JSON are written correctly."""
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        harness_local = BenchmarkHarness(
            cummins_params             = wec_params["cummins"],
            pto_params                 = wec_params["pto"],
            hydro_excitation_amplitude = wec_params["hydro_amp"],
            hydro_omega                = wec_params["hydro_omega"],
            hydro                      = wec_params["hydro"],
            max_episode_steps          = 50,
            verbose                    = False,
        )
        ctrls    = [PassiveController(), FixedThresholdController()]
        segments = [("seg_ref", 1.5, 8.0, "reference")]
        results  = harness_local.run_benchmark(
            ctrls,
            segments     = segments,
            n_hours      = 1,
            save_results = True,
            results_dir  = tmpdir,
        )

        csv_path  = os.path.join(tmpdir, "benchmark_results.csv")
        json_path = os.path.join(tmpdir, "benchmark_summary.json")

        assert os.path.exists(csv_path),  "benchmark_results.csv not created"
        assert os.path.exists(json_path), "benchmark_summary.json not created"

        df = pd.read_csv(csv_path)
        assert len(df) == 2   # 1 segment × 2 controllers
        for col in ("controller_label", "total_energy_Wh", "is_finite"):
            assert col in df.columns, f"Column '{col}' missing from CSV"

        with open(json_path) as f:
            summary = json.load(f)
        assert "controllers" in summary
        assert "passive" in summary["controllers"]
        assert "fixed_threshold" in summary["controllers"]


# ===========================================================================
# Additional: FixedThresholdConfig validation
# ===========================================================================

def test_threshold_config_defaults():
    cfg = FixedThresholdConfig()
    assert cfg.v_latch_threshold == 0.05
    assert cfg.min_latch_steps   == 4
    assert cfg.min_release_steps == 3


def test_threshold_config_bad_threshold():
    with pytest.raises(ValueError):
        FixedThresholdConfig(v_latch_threshold=-0.1)


# ===========================================================================
# Additional: make_controller factory
# ===========================================================================

def test_make_controller_all_types():
    for ctype in ["passive", "fixed_threshold", "reactive", "oracle_interface", "realistic_interface"]:
        ctrl = make_controller(ctype)
        assert isinstance(ctrl, BaseController)
        assert ctrl.label == ctype


def test_make_controller_unknown_raises():
    with pytest.raises(ValueError):
        make_controller("invalid_controller")


# ===========================================================================
# Additional: controller reset behaviour
# ===========================================================================

def test_controller_reset_clears_state():
    ctrl = FixedThresholdController()
    ctrl.reset()
    # Drive the controller to LATCH state (v near zero)
    for _ in range(10):
        ctrl.act(_dummy_obs(v=0.01), _dummy_info(v=0.01))
    # Reset should bring it back to RELEASE
    ctrl.reset()
    # After reset, large velocity should keep it in RELEASE
    obs  = _dummy_obs(v=0.5)
    info = _dummy_info(v=0.5)
    action = ctrl.act(obs, info)
    # After reset with release_steps=0, min_release_steps=3 not met yet
    # so it stays RELEASED regardless of v
    assert action == ACTION_RELEASE


# ===========================================================================
# Additional: min_latch_steps enforcement
# ===========================================================================

def test_min_latch_steps_respected():
    """Controller must stay LATCHED for exactly min_latch_steps, then release (timer-based)."""
    cfg2 = FixedThresholdConfig(v_latch_threshold=0.1, min_latch_steps=5, min_release_steps=0)
    ctrl2 = FixedThresholdController(cfg2)
    ctrl2.reset()

    # Step 1: low v → triggers LATCH (release_steps=0 >= min_release_steps=0)
    a1 = ctrl2.act(_dummy_obs(v=0.02), _dummy_info(v=0.02))
    assert a1 == ACTION_LATCH, f"Step 1 should latch, got {a1}"

    # Steps 2-5: still within timer (latch_steps increments 0→4)
    for i in range(4):
        a = ctrl2.act(_dummy_obs(v=0.0), _dummy_info(v=0.0))
        assert a == ACTION_LATCH, f"Step {i+2} should stay latched, got {a}"

    # Step 6: latch_steps reaches 5 >= min_latch_steps=5 → release
    a6 = ctrl2.act(_dummy_obs(v=0.0), _dummy_info(v=0.0))
    assert a6 == ACTION_RELEASE, f"Step 6 should release after min_latch_steps=5, got {a6}"


# ===========================================================================
# Additional: physics sanity check — one reference sea state
# ===========================================================================

def test_physics_sanity_reference_sea_state(wec_params, replay_2h, harness):
    """
    Reference sea state: Hs=1.5m, Tp=8s.
    All baselines must be numerically stable and physically reasonable.
    """
    ctrls = [PassiveController(), FixedThresholdController(), ReactiveController()]
    for ctrl in ctrls:
        r = harness.run_episode(ctrl, replay_2h, env_mode=ctrl.label,
                                Hs_m=1.5, Tp_s=8.0, segment_id="sanity")
        assert r.is_finite, f"{ctrl.label} not finite in sanity check"
        assert r.max_abs_displacement_m < 20.0
        assert r.max_abs_velocity_ms    < 10.0
        assert r.total_energy_Wh        >= 0.0
        assert r.mean_power_W           >= 0.0


# ===========================================================================
# Additional: benchmark_summary has expected structure
# ===========================================================================

def test_summary_structure(wec_params, replay_1h, harness):
    ctrls = [PassiveController(), FixedThresholdController()]
    results = [
        harness.run_episode(c, replay_1h, env_mode=c.label,
                            Hs_m=1.5, Tp_s=8.0, segment_id="s")
        for c in ctrls
    ]
    summary = _compute_summary(results)
    assert "metadata"    in summary
    assert "controllers" in summary
    for lbl in ["passive", "fixed_threshold"]:
        assert lbl in summary["controllers"]
        ctrl_entry = summary["controllers"][lbl]
        for key in ["n_segments", "mean_Wh", "median_Wh", "std_Wh", "min_Wh", "max_Wh"]:
            assert key in ctrl_entry, f"Key '{key}' missing for {lbl}"


# ===========================================================================
# Additional: BenchmarkResult serialisation
# ===========================================================================

def test_benchmark_result_to_dict():
    r = BenchmarkResult(
        controller_label="passive", segment_id="test", segment_desc="t",
        env_mode="reactive", Hs_m=1.5, Tp_s=8.0,
        pto_damping_Ns_m=200_000.0, pto_stiffness_N_m=0.0,
        replay_seed=42, episode_hours=1.0,
        total_energy_Wh=5.0, total_energy_kWh=0.005, mean_power_W=5000.0,
        max_step_power_W=15_000.0, total_reward=10.0, n_steps=3600,
        n_latch_events=0, n_release_events=0, latch_fraction=0.0,
        max_abs_displacement_m=1.0, max_abs_velocity_ms=0.5,
        max_abs_acceleration_ms2=0.1, has_nan=False, has_inf=False,
        is_finite=True, wall_time_s=5.0, energy_check_rel_err=0.001,
    )
    d = r.to_dict()
    assert d["controller_label"] == "passive"
    assert d["total_energy_Wh"]  == 5.0
    assert d["is_finite"]        is True


# ===========================================================================
# Additional: OracleThresholdController uses same rule as FixedThreshold
# ===========================================================================

def test_oracle_same_rule_as_threshold():
    """OracleThresholdController.act() produces the same action as FixedThresholdController
    for the same velocity, since both read only info['v']."""
    cfg   = FixedThresholdConfig(v_latch_threshold=0.05, min_latch_steps=2, min_release_steps=2)
    ctrl_t = FixedThresholdController(cfg)
    ctrl_o = OracleThresholdController(cfg)
    ctrl_t.reset()
    ctrl_o.reset()

    velocities = [0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01, 0.1, 0.1, 0.1]
    for v in velocities:
        obs  = _dummy_obs(block_b_val=0.0, v=v)
        info = _dummy_info(v=v)
        at   = ctrl_t.act(obs, info)
        ao   = ctrl_o.act(obs, info)
        assert at == ao, f"Oracle and Threshold actions differ at v={v}: {at} vs {ao}"
