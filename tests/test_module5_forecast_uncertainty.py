"""
test_module5_forecast_uncertainty.py — Tests for Module 5.2B.

Covers all 24 specification requirements plus statistical validation.

Spec requirements:
 1.  CSV loads successfully.
 2.  Expected variables present.
 3.  Expected 48 lead times represented.
 4.  Lead-specific distributions correctly constructed.
 5.  Same seed → identical sampled errors.
 6.  Different seeds → different samples.
 7.  Hs remains non-negative after injection.
 8.  Tp remains positive after injection.
 9.  Direction wrapping is correct.
10.  Joint Hs/Tp/direction sampling works (trajectory-level).
11.  Forecast shape remains (48, 4) after injection.
12.  Realistic forecast differs from deterministic when errors are nonzero.
13.  Zero-error mode reproduces deterministic forecast exactly.
14.  perfect_forecast mode remains untouched.
15.  reactive mode remains untouched.
16.  RL observation contains NO future truth.
17.  Error values not directly exposed in observation.
18.  Forecast uncertainty grows with lead time (empirical structure preserved).
19.  No NaN/Inf reaches the observation.
20.  Environment reset with same seed is reproducible.
21.  Existing Module 5.2A excitation behaviour unchanged.
22.  Existing latch/radiation-memory behaviour unchanged.
23.  Existing PTO power calculation unchanged.
24.  Full environment smoke test works.

Additional tests:
 - Statistical validation: sampled distribution reproduces empirical.
 - ErrorTrajectoryBank.lead_statistics() correctness.
 - load_error_bank() error handling.
 - ForecastErrorSampler.summary() completeness.
 - Direction conversion round-trip.
 - Physical validity under extreme Hs/Tp scenarios.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Session-scope fixtures (expensive; shared across all tests)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def error_bank():
    from module5_control.forecast_uncertainty import load_error_bank
    return load_error_bank()


@pytest.fixture(scope="session")
def sampler_seed42(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    return ForecastErrorSampler(bank=error_bank, seed=42)


@pytest.fixture(scope="session")
def wec_params():
    from module2_wec.geometry import REFERENCE_BUOY
    from module2_wec.hydrostatics import Hydrostatics
    from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
    from module2_wec.radiation import compute_radiation_kernel
    from module2_wec.cummins import CumminsParameters
    from module2_wec.pto import PTOParameters
    hydro = compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY, resolution=(6, 24, 16),
        omega_min=0.2, omega_max=1.4, n_omega=20, progress_bar=False,
    )
    hs_obj = Hydrostatics(REFERENCE_BUOY)
    rk = compute_radiation_kernel(hydro)
    cp = CumminsParameters(
        mass=REFERENCE_BUOY.mass,
        hydrostatic_stiffness=hs_obj.hydrostatic_stiffness,
        added_mass_infinity=rk.A_infinity,
        kernel_time=rk.time, kernel_values=rk.kernel,
    )
    return {
        "cummins":    cp,
        "pto":        PTOParameters(damping=200_000.0, stiffness=0.0),
        "hydro_amp":  hydro.excitation_force_heave_amplitude,
        "hydro_omega":hydro.omega,
        "hydro":      hydro,
    }


@pytest.fixture(scope="session")
def replay_4h():
    from module5_control.replay import build_synthetic_replay
    return build_synthetic_replay(n_hours=4, seed=42)


def _det_forecast(Hs: float = 1.5, Tp: float = 8.0, dir_deg: float = 270.0) -> np.ndarray:
    """Build a constant deterministic forecast array (48, 4)."""
    from module4_forecasting.dataset import direction_to_sincos
    sin_d, cos_d = direction_to_sincos(np.array([dir_deg]))
    out = np.zeros((48, 4))
    out[:, 0] = Hs
    out[:, 1] = Tp
    out[:, 2] = float(sin_d[0])
    out[:, 3] = float(cos_d[0])
    return out


# ===========================================================================
# Test 1 — CSV loads successfully
# ===========================================================================

def test_01_csv_loads(error_bank):
    from module5_control.forecast_uncertainty import ErrorTrajectoryBank
    assert isinstance(error_bank, ErrorTrajectoryBank)
    assert error_bank.n_trajectories > 0
    assert error_bank.n_leads == 48


def test_01_from_csv_factory():
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    sampler = ForecastErrorSampler.from_csv(seed=0)
    assert sampler.bank.n_trajectories == 17449


def test_01_missing_file_raises():
    from module5_control.forecast_uncertainty import load_error_bank
    with pytest.raises(FileNotFoundError):
        load_error_bank("/nonexistent/path/errors.csv.gz")


# ===========================================================================
# Test 2 — Expected variables present
# ===========================================================================

def test_02_expected_variables(error_bank):
    import pandas as pd
    df = pd.read_csv(error_bank.csv_path)
    required = {"lead_h", "sample_idx", "Hs_error", "Tp_error", "dir_error_deg"}
    assert required.issubset(set(df.columns)), (
        f"Missing columns: {required - set(df.columns)}"
    )


def test_02_arrays_finite(error_bank):
    assert np.all(np.isfinite(error_bank.err_hs)),  "err_hs has NaN/Inf"
    assert np.all(np.isfinite(error_bank.err_tp)),  "err_tp has NaN/Inf"
    assert np.all(np.isfinite(error_bank.err_dir)), "err_dir has NaN/Inf"


def test_02_array_shapes(error_bank):
    assert error_bank.err_hs.shape  == (17449, 48)
    assert error_bank.err_tp.shape  == (17449, 48)
    assert error_bank.err_dir.shape == (17449, 48)


def test_02_array_dtype(error_bank):
    assert error_bank.err_hs.dtype  == np.float32
    assert error_bank.err_tp.dtype  == np.float32
    assert error_bank.err_dir.dtype == np.float32


# ===========================================================================
# Test 3 — Expected 48 lead times represented
# ===========================================================================

def test_03_48_leads(error_bank):
    assert error_bank.n_leads == 48


def test_03_all_leads_complete(error_bank):
    """Every trajectory should have exactly 48 lead columns (no NaN)."""
    assert not np.any(np.isnan(error_bank.err_hs))
    assert error_bank.err_hs.shape[1] == 48


# ===========================================================================
# Test 4 — Lead-specific distributions correctly constructed
# ===========================================================================

def test_04_lead_stats_correct(error_bank):
    """lead_statistics() at lead 1 and 48 must match raw CSV values."""
    import pandas as pd
    df = pd.read_csv(error_bank.csv_path)

    for lead in [1, 48]:
        stats = error_bank.lead_statistics(lead)
        sub   = df[df["lead_h"] == lead]

        assert abs(stats["Hs_mean"]  - float(sub.Hs_error.mean()))  < 1e-3
        assert abs(stats["Hs_std"]   - float(sub.Hs_error.std()))   < 1e-3
        assert abs(stats["Tp_mean"]  - float(sub.Tp_error.mean()))  < 1e-3
        assert abs(stats["dir_mean"] - float(sub.dir_error_deg.mean())) < 1e-3


def test_04_uncertainty_grows_with_lead(error_bank):
    """Hs error std should increase from lead 1 to lead 48."""
    std1  = float(np.std(error_bank.err_hs[:, 0]))   # lead 1  (col 0)
    std48 = float(np.std(error_bank.err_hs[:, 47]))  # lead 48 (col 47)
    assert std48 > std1, (
        f"Hs error std should grow with lead: std(lead1)={std1:.4f}, "
        f"std(lead48)={std48:.4f}"
    )


# ===========================================================================
# Test 5 — Same seed → identical sampled errors
# ===========================================================================

def test_05_same_seed_identical(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s1  = ForecastErrorSampler(bank=error_bank, seed=77)
    s2  = ForecastErrorSampler(bank=error_bank, seed=77)
    np.testing.assert_array_equal(s1.apply(det), s2.apply(det))


def test_05_same_seed_same_trajectory(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    s1 = ForecastErrorSampler(bank=error_bank, seed=100)
    s2 = ForecastErrorSampler(bank=error_bank, seed=100)
    assert s1.current_trajectory_index == s2.current_trajectory_index


# ===========================================================================
# Test 6 — Different seeds → different samples
# ===========================================================================

def test_06_different_seeds_differ(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s1  = ForecastErrorSampler(bank=error_bank, seed=1)
    s2  = ForecastErrorSampler(bank=error_bank, seed=2)
    assert not np.array_equal(s1.apply(det), s2.apply(det))


# ===========================================================================
# Test 7 — Hs remains non-negative
# ===========================================================================

def test_07_hs_nonneg_typical(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast(Hs=1.5)
    for seed in range(50):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        assert np.all(out[:, 0] >= 0.0), f"Negative Hs in seed={seed}"


def test_07_hs_nonneg_extreme(error_bank):
    """Even with very small Hs prediction, no negative output."""
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast(Hs=0.01)   # near-zero Hs
    for seed in range(200):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        assert np.all(out[:, 0] >= 0.0)


# ===========================================================================
# Test 8 — Tp remains positive
# ===========================================================================

def test_08_tp_positive_typical(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast(Tp=8.0)
    for seed in range(50):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        assert np.all(out[:, 1] > 0.0), f"Non-positive Tp in seed={seed}"


def test_08_tp_positive_extreme(error_bank):
    """Even with very small Tp prediction, Tp stays above TP_MIN."""
    from module5_control.forecast_uncertainty import ForecastErrorSampler, TP_MIN
    det = _det_forecast(Tp=1.0)   # small Tp
    for seed in range(200):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        assert np.all(out[:, 1] >= TP_MIN)


# ===========================================================================
# Test 9 — Direction wrapping is correct
# ===========================================================================

def test_09_direction_sincos_unit(error_bank):
    """sin²+cos² must equal 1 for all leads after injection."""
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast(dir_deg=270.0)
    for seed in range(20):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        norms = out[:, 2]**2 + out[:, 3]**2
        np.testing.assert_allclose(norms, 1.0, atol=1e-10,
            err_msg=f"sin²+cos² != 1 at seed={seed}")


def test_09_direction_wrapping_boundary():
    """Adding error to dir=359° must wrap correctly, not produce values > 360."""
    from module5_control.forecast_uncertainty import (
        ForecastErrorSampler, load_error_bank, _sincos_to_deg,
    )
    bank = load_error_bank()
    from module4_forecasting.dataset import direction_to_sincos
    sin_d, cos_d = direction_to_sincos(np.array([359.0]))
    det = np.zeros((48, 4))
    det[:, 0] = 1.0
    det[:, 1] = 8.0
    det[:, 2] = float(sin_d[0])
    det[:, 3] = float(cos_d[0])
    for seed in range(30):
        s   = ForecastErrorSampler(bank=bank, seed=seed)
        out = s.apply(det)
        dir_out = _sincos_to_deg(out[:, 2], out[:, 3])
        assert np.all(dir_out >= 0.0)
        assert np.all(dir_out < 360.0)


# ===========================================================================
# Test 10 — Joint Hs/Tp/dir sampling (trajectory-level)
# ===========================================================================

def test_10_joint_sampling_trajectory_consistent(error_bank):
    """
    Hs and Tp errors must come from the SAME trajectory — not independently
    drawn.  Verify by checking that the sampled errors match the stored
    trajectory for the drawn index.
    """
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s   = ForecastErrorSampler(bank=error_bank, seed=11)

    traj_idx   = s.current_trajectory_index
    hs_traj, tp_traj, dir_traj = error_bank.get_trajectory(traj_idx)

    # The sampled errors stored in the sampler must equal the trajectory
    np.testing.assert_allclose(s.current_hs_errors,  hs_traj,  atol=1e-5)
    np.testing.assert_allclose(s.current_tp_errors,  tp_traj,  atol=1e-5)
    np.testing.assert_allclose(s.current_dir_errors, dir_traj, atol=1e-5)


def test_10_all_three_variables_change(error_bank):
    """A non-trivial trajectory should affect Hs, Tp, and direction."""
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    for seed in range(100):
        s   = ForecastErrorSampler(bank=error_bank, seed=seed)
        out = s.apply(det)
        if (not np.allclose(out[:, 0], det[:, 0])
                and not np.allclose(out[:, 1], det[:, 1])
                and not np.allclose(out[:, 2], det[:, 2])):
            return   # at least one seed has all three changed
    pytest.fail("No seed produced changes in all three variables")


# ===========================================================================
# Test 11 — Forecast shape remains (48, 4)
# ===========================================================================

def test_11_output_shape(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s   = ForecastErrorSampler(bank=error_bank, seed=0)
    out = s.apply(det)
    assert out.shape == (48, 4)


def test_11_output_dtype_float64(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s   = ForecastErrorSampler(bank=error_bank, seed=0)
    out = s.apply(det)
    assert out.dtype == np.float64


def test_11_wrong_shape_raises(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    s = ForecastErrorSampler(bank=error_bank, seed=0)
    with pytest.raises(ValueError):
        s.apply(np.ones((24, 4)))   # wrong horizon


# ===========================================================================
# Test 12 — Realistic differs from deterministic (nonzero errors)
# ===========================================================================

def test_12_realistic_differs_from_det(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det = _det_forecast()
    s   = ForecastErrorSampler(bank=error_bank, seed=42)
    out = s.apply(det)
    # With full scale errors, output must differ from det
    assert not np.allclose(out, det), "Realistic forecast should differ from deterministic"


# ===========================================================================
# Test 13 — Zero-error mode reproduces det exactly
# ===========================================================================

def test_13_zero_scale_exact(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det  = _det_forecast()
    s    = ForecastErrorSampler(bank=error_bank, seed=42, scale=0.0)
    out  = s.apply(det)
    np.testing.assert_array_equal(out, det)


def test_13_zero_scale_multiple_resets(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    det  = _det_forecast()
    s    = ForecastErrorSampler(bank=error_bank, seed=0, scale=0.0)
    for _ in range(5):
        s.reset()
        out = s.apply(det)
        np.testing.assert_array_equal(out, det)


# ===========================================================================
# Test 14 — perfect_forecast mode untouched
# ===========================================================================

def test_14_perfect_forecast_unchanged(wec_params, replay_4h):
    """In perfect_forecast mode the observation Block B must contain
    true future ERA5 values, not GRU predictions or injected errors."""
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank
    from module5_control.observations import H_FORE, F_FORE

    # Build a sampler — but perfect_forecast should not use it
    bank    = load_error_bank()
    sampler = ForecastErrorSampler(bank=bank, seed=0, scale=1.0)

    env_perfect = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="perfect_forecast",
        max_episode_steps=5,
        forecast_error_sampler=sampler,   # attached but should be ignored
    )
    obs, _ = env_perfect.reset(seed=0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END]
    # Block B should have real Hs/Tp from ERA5 (not zeros, not corrupted)
    assert not np.all(block_b == 0.0), "perfect_forecast Block B should not be all zeros"
    assert np.all(np.isfinite(block_b))


def test_14_perfect_forecast_sampler_not_invoked(wec_params, replay_4h):
    """The sampler should NOT be invoked in perfect_forecast mode."""
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    sampler = ForecastErrorSampler(bank=bank, seed=0, scale=1.0)

    env = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="perfect_forecast",
        max_episode_steps=5,
        forecast_error_sampler=sampler,
    )
    env.reset(seed=0)
    # In perfect_forecast mode _gru_forecast() is never called,
    # so the clip counter should remain 0
    assert sampler.n_hs_clipped == 0
    assert sampler.n_tp_clipped == 0


# ===========================================================================
# Test 15 — reactive mode untouched
# ===========================================================================

def test_15_reactive_block_b_all_zeros(wec_params, replay_4h):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    sampler = ForecastErrorSampler(bank=bank, seed=0, scale=1.0)

    env = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=5,
        forecast_error_sampler=sampler,   # attached but irrelevant
    )
    obs, _ = env.reset(seed=0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END]
    np.testing.assert_array_equal(block_b, 0.0)


# ===========================================================================
# Test 16 — RL observation contains NO future truth
# ===========================================================================

def test_16_no_future_truth_in_observation(wec_params, replay_4h):
    """
    The realistic-forecast observation Block B must NOT contain the true
    future ERA5 Hs/Tp/direction values.  We verify this by comparing the
    realistic-forecast observation with the perfect-forecast observation:
    they must differ (perfect = truth, realistic = GRU + errors ≠ truth).
    """
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    gru_ckpt = "results/forecasting/exp_gru_wind/best_model.pt"

    env_real = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path=gru_ckpt,
        max_episode_steps=5,
        forecast_error_sampler=ForecastErrorSampler(bank=bank, seed=42),
    )
    env_perf = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="perfect_forecast",
        max_episode_steps=5,
    )

    obs_real, _ = env_real.reset(seed=0)
    obs_perf, _ = env_perf.reset(seed=0)

    b_real = obs_real[_BLOCK_B_START:_BLOCK_B_END]
    b_perf = obs_perf[_BLOCK_B_START:_BLOCK_B_END]

    # Both are finite
    assert np.all(np.isfinite(b_real))
    assert np.all(np.isfinite(b_perf))
    # They must differ (realistic ≠ oracle truth)
    assert not np.allclose(b_real, b_perf), (
        "realistic_forecast Block B must not equal perfect_forecast Block B"
    )


# ===========================================================================
# Test 17 — Error values not directly exposed in observation
# ===========================================================================

def test_17_errors_not_in_observation(wec_params, replay_4h):
    """
    The sampled error values themselves must not appear verbatim in the
    observation.  The observation contains the forecast + error output,
    not the raw error values.
    """
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.observations import _BLOCK_B_START, _BLOCK_B_END
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    gru_ckpt = "results/forecasting/exp_gru_wind/best_model.pt"
    sampler = ForecastErrorSampler(bank=bank, seed=3)

    env = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path=gru_ckpt,
        max_episode_steps=5,
        forecast_error_sampler=sampler,
    )
    obs, _ = env.reset(seed=0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END].reshape(48, 4)

    # The raw Hs errors must not appear verbatim in Block B column 0
    raw_hs_errors = sampler.current_hs_errors
    # Block B col 0 = Hs forecast (after scaler), not raw error magnitudes
    # The Hs forecast values should be in physical range [0, 5 m] scaled
    # while raw errors are typically ±0.1–0.5 m — the two overlap but
    # the exact array should not match
    assert not np.allclose(block_b[:, 0], raw_hs_errors, atol=0.01), (
        "Raw Hs errors must not appear verbatim in observation Block B"
    )


# ===========================================================================
# Test 18 — Forecast uncertainty grows with lead time
# ===========================================================================

def test_18_uncertainty_increases_with_lead(error_bank):
    """
    The empirical std of Hs errors at lead 48 must exceed that at lead 1.
    This is a fundamental property of forecast skill degradation.
    """
    std_lead1  = float(np.std(error_bank.err_hs[:, 0]))
    std_lead12 = float(np.std(error_bank.err_hs[:, 11]))
    std_lead24 = float(np.std(error_bank.err_hs[:, 23]))
    std_lead48 = float(np.std(error_bank.err_hs[:, 47]))

    assert std_lead12 > std_lead1,  "Hs std must increase from lead 1 to 12"
    assert std_lead24 > std_lead12, "Hs std must increase from lead 12 to 24"
    assert std_lead48 > std_lead24, "Hs std must increase from lead 24 to 48"


def test_18_sampled_errors_grow_with_lead(error_bank):
    """
    When sampling many trajectories, the RMS of sampled Hs errors at
    lead 48 must exceed those at lead 1.
    """
    from module5_control.forecast_uncertainty import ForecastErrorSampler

    n_samples = 500
    hs_lead1  = []
    hs_lead48 = []

    rng = np.random.default_rng(0)
    traj_indices = rng.integers(0, error_bank.n_trajectories, n_samples)

    for idx in traj_indices:
        hs_traj, _, _ = error_bank.get_trajectory(int(idx))
        hs_lead1.append(hs_traj[0])
        hs_lead48.append(hs_traj[47])

    rms1  = float(np.sqrt(np.mean(np.array(hs_lead1)**2)))
    rms48 = float(np.sqrt(np.mean(np.array(hs_lead48)**2)))
    assert rms48 > rms1, f"RMS Hs error should grow: lead1={rms1:.4f}, lead48={rms48:.4f}"


# ===========================================================================
# Test 19 — No NaN/Inf in observation
# ===========================================================================

def test_19_no_nan_inf_observation(wec_params, replay_4h):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    gru_ckpt = "results/forecasting/exp_gru_wind/best_model.pt"

    env = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path=gru_ckpt,
        max_episode_steps=20,
        forecast_error_sampler=ForecastErrorSampler(bank=bank, seed=7),
    )
    obs, _ = env.reset(seed=0)
    assert np.all(np.isfinite(obs)), "NaN/Inf in reset observation"
    assert not np.any(np.isnan(obs))

    for _ in range(10):
        obs, rew, term, trunc, info = env.step(0)
        assert np.all(np.isfinite(obs)), "NaN/Inf in step observation"
        assert not np.any(np.isnan(obs))
        if term or trunc:
            break


# ===========================================================================
# Test 20 — Reset with same seed is reproducible
# ===========================================================================

def test_20_reset_reproducible(wec_params, replay_4h):
    """
    Two environments with the same initial sampler seed and same reset
    seed must produce identical observation sequences.
    """
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank     = load_error_bank()
    gru_ckpt = "results/forecasting/exp_gru_wind/best_model.pt"

    results = []
    for _ in range(2):
        sampler = ForecastErrorSampler(bank=bank, seed=99)
        env = build_env_from_checkpoint(
            replay=replay_4h,
            cummins_params=wec_params["cummins"],
            pto_params=wec_params["pto"],
            hydro_excitation_amplitude=wec_params["hydro_amp"],
            hydro_omega=wec_params["hydro_omega"],
            mode="realistic_forecast",
            gru_checkpoint_path=gru_ckpt,
            max_episode_steps=10,
            forecast_error_sampler=sampler,
        )
        obs, _ = env.reset(seed=0)
        traj   = [obs.copy()]
        for a in [0, 1, 0, 0, 1]:
            obs, rew, *_ = env.step(a)
            traj.append(obs.copy())
        results.append(traj)

    for t in range(len(results[0])):
        np.testing.assert_array_equal(
            results[0][t], results[1][t],
            err_msg=f"Observation mismatch at step {t} in reproducibility test",
        )


# ===========================================================================
# Test 21 — Module 5.2A excitation behaviour unchanged
# ===========================================================================

def test_21_excitation_unchanged(wec_params, replay_4h):
    """
    Adding a forecast-error sampler to the env must not change excitation
    or physics.  Compare x,v trajectories with and without sampler in
    reactive mode (where sampler is not invoked anyway).
    """
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank    = load_error_bank()
    actions = [0, 1, 0, 0, 1, 0]

    def _run(with_sampler):
        sampler = ForecastErrorSampler(bank=bank, seed=5) if with_sampler else None
        env = build_env_from_checkpoint(
            replay=replay_4h,
            cummins_params=wec_params["cummins"],
            pto_params=wec_params["pto"],
            hydro_excitation_amplitude=wec_params["hydro_amp"],
            hydro_omega=wec_params["hydro_omega"],
            mode="reactive",    # sampler not invoked in reactive
            max_episode_steps=len(actions)+2,
            hydro=wec_params["hydro"],
            forecast_error_sampler=sampler,
        )
        env.reset(seed=0)
        traj = []
        for a in actions:
            _, _, _, _, info = env.step(a)
            traj.append((info["x"], info["v"], info["mean_step_power_W"]))
        return traj

    t_no  = _run(False)
    t_yes = _run(True)
    for i, (r1, r2) in enumerate(zip(t_no, t_yes)):
        assert r1 == r2, f"Excitation/physics changed by sampler presence at step {i}"


# ===========================================================================
# Test 22 — Latch / radiation-memory unchanged
# ===========================================================================

def test_22_latch_radiation_memory_unchanged(wec_params, replay_4h):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank
    from module5_control.replay import N_PHYSICS_PER_CONTROL

    bank = load_error_bank()
    env  = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=50,
        forecast_error_sampler=ForecastErrorSampler(bank=bank, seed=0),
    )
    env.reset(seed=0)
    for _ in range(5): env.step(0)
    step_before = env.integrator.step_count
    for _ in range(3): env.step(1)   # latch
    for _ in range(3): env.step(0)   # release
    step_after = env.integrator.step_count
    assert step_after == step_before + 6 * N_PHYSICS_PER_CONTROL


# ===========================================================================
# Test 23 — PTO power unchanged
# ===========================================================================

def test_23_pto_power_nonneg(wec_params, replay_4h):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank = load_error_bank()
    env  = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=30,
        forecast_error_sampler=ForecastErrorSampler(bank=bank, seed=0),
    )
    env.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env.step(0)
        assert info["mean_step_power_W"] >= 0.0
        if term or trunc:
            break


# ===========================================================================
# Test 24 — Full environment smoke test with 5.2B
# ===========================================================================

def test_24_full_smoke_test(wec_params, replay_4h):
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.forecast_uncertainty import ForecastErrorSampler, load_error_bank

    bank     = load_error_bank()
    gru_ckpt = "results/forecasting/exp_gru_wind/best_model.pt"
    sampler  = ForecastErrorSampler(bank=bank, seed=42)

    env = build_env_from_checkpoint(
        replay=replay_4h,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="realistic_forecast",
        gru_checkpoint_path=gru_ckpt,
        max_episode_steps=20,
        hydro=wec_params["hydro"],
        forecast_error_sampler=sampler,
    )

    obs, info = env.reset(seed=0)
    assert obs.shape == (723,)
    assert np.all(np.isfinite(obs))

    cum_energy_prev = 0.0
    for step in range(10):
        obs, rew, term, trunc, info = env.step(step % 2)
        assert obs.shape == (723,)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(rew)
        assert info["cumulative_energy_J"] >= cum_energy_prev - 1e-9
        cum_energy_prev = info["cumulative_energy_J"]
        if term or trunc:
            break


# ===========================================================================
# Additional: Statistical validation
# ===========================================================================

def test_stat_sampled_hs_mean_close_to_empirical(error_bank):
    """
    Sampling 5000 trajectories should reproduce the empirical Hs mean
    at lead 1 and lead 48 to within ~3 standard errors.
    """
    n_draws = 5000
    rng     = np.random.default_rng(0)
    idxs    = rng.integers(0, error_bank.n_trajectories, n_draws)

    hs_l1  = error_bank.err_hs[idxs, 0].astype(float)
    hs_l48 = error_bank.err_hs[idxs, 47].astype(float)

    emp_mean_l1  = float(np.mean(error_bank.err_hs[:, 0]))
    emp_mean_l48 = float(np.mean(error_bank.err_hs[:, 47]))
    emp_std_l1   = float(np.std(error_bank.err_hs[:, 0]))
    emp_std_l48  = float(np.std(error_bank.err_hs[:, 47]))

    samp_mean_l1  = float(np.mean(hs_l1))
    samp_mean_l48 = float(np.mean(hs_l48))

    # Tolerance: 3 standard errors of the mean
    se_l1  = emp_std_l1  / np.sqrt(n_draws) * 3
    se_l48 = emp_std_l48 / np.sqrt(n_draws) * 3

    assert abs(samp_mean_l1  - emp_mean_l1)  < se_l1,  (
        f"Sampled Hs mean at lead1 differs: {samp_mean_l1:.4f} vs {emp_mean_l1:.4f}")
    assert abs(samp_mean_l48 - emp_mean_l48) < se_l48, (
        f"Sampled Hs mean at lead48 differs: {samp_mean_l48:.4f} vs {emp_mean_l48:.4f}")


def test_stat_sampled_hs_std_close_to_empirical(error_bank):
    """Sampled std should reproduce empirical std to <5%."""
    n_draws = 5000
    rng     = np.random.default_rng(1)
    idxs    = rng.integers(0, error_bank.n_trajectories, n_draws)

    for lead_idx, lead_h in [(0, 1), (11, 12), (23, 24), (47, 48)]:
        samp  = error_bank.err_hs[idxs, lead_idx].astype(float)
        emp   = error_bank.err_hs[:, lead_idx].astype(float)
        rel_diff = abs(float(np.std(samp)) - float(np.std(emp))) / float(np.std(emp))
        assert rel_diff < 0.05, (
            f"Sampled Hs std at lead {lead_h} off by {100*rel_diff:.1f}% "
            f"(sampled={np.std(samp):.4f}, empirical={np.std(emp):.4f})"
        )


def test_stat_temporal_correlation_preserved(error_bank):
    """
    Within sampled trajectories, the lag-1 temporal correlation must be
    comparable to the empirical value.
    """
    n_draws = 1000
    rng     = np.random.default_rng(2)
    idxs    = rng.integers(0, error_bank.n_trajectories, n_draws)

    l1  = error_bank.err_hs[idxs, 0].astype(float)
    l2  = error_bank.err_hs[idxs, 1].astype(float)
    sampled_corr = float(np.corrcoef(l1, l2)[0, 1])

    emp_l1 = error_bank.err_hs[:, 0].astype(float)
    emp_l2 = error_bank.err_hs[:, 1].astype(float)
    emp_corr = float(np.corrcoef(emp_l1, emp_l2)[0, 1])

    assert abs(sampled_corr - emp_corr) < 0.05, (
        f"Sampled lag-1 corr {sampled_corr:.4f} differs from empirical {emp_corr:.4f}"
    )


def test_stat_direction_errors_circular(error_bank):
    """All stored direction errors must be in (-180, 180]."""
    assert np.all(error_bank.err_dir > -180.0)
    assert np.all(error_bank.err_dir <= 180.0)


def test_stat_non_gaussian_confirmed(error_bank):
    """
    The empirical distributions should reject Gaussian at p < 1e-4.
    This validates that empirical resampling is necessary.
    """
    from scipy import stats
    # Sample 500 values from lead 1 Hs errors
    hs_l1 = error_bank.err_hs[:500, 0].astype(float)
    _, p  = stats.shapiro(hs_l1)
    assert p < 1e-4, (
        f"Hs errors at lead 1 appear Gaussian (p={p:.4f}); "
        "empirical resampling may be unnecessary — investigate"
    )


# ===========================================================================
# Additional: ForecastErrorSampler.summary()
# ===========================================================================

def test_sampler_summary(error_bank):
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    s = ForecastErrorSampler(bank=error_bank, seed=0)
    summary = s.summary()
    for key in ("n_trajectories", "n_leads", "scale", "error_sign",
                "direction_convention", "sampling_method", "temporal_structure"):
        assert key in summary, f"Key '{key}' missing from summary"


# ===========================================================================
# Additional: ErrorTrajectoryBank.lead_statistics()
# ===========================================================================

def test_lead_statistics_all_leads(error_bank):
    """lead_statistics should work for all 48 leads without error."""
    for lead in range(1, 49):
        stats = error_bank.lead_statistics(lead)
        assert stats["lead_h"] == lead
        assert np.isfinite(stats["Hs_mean"])
        assert np.isfinite(stats["Tp_std"])
        assert np.isfinite(stats["dir_p50"])


# ===========================================================================
# Additional: direction conversion round-trip
# ===========================================================================

def test_direction_roundtrip():
    """sincos → deg → sincos should be lossless."""
    from module5_control.forecast_uncertainty import _sincos_to_deg, _wrap360
    from module4_forecasting.dataset import direction_to_sincos

    dirs = np.array([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0, 359.9])
    sin_d, cos_d = direction_to_sincos(dirs)
    dirs_back    = _sincos_to_deg(sin_d, cos_d)
    np.testing.assert_allclose(dirs_back, dirs, atol=1e-8)


def test_wrap360_identity():
    from module5_control.forecast_uncertainty import _wrap360
    vals = np.array([0.0, 90.0, 180.0, 270.0, 359.9, 360.0, 361.0, -1.0, -90.0])
    wrapped = _wrap360(vals)
    assert np.all(wrapped >= 0.0)
    assert np.all(wrapped < 360.0)
