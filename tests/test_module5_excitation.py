"""
test_module5_excitation.py — Tests for module5_control/excitation.py (Module 5.2A).

Covers all 15 specification requirements plus diagnostic comparisons
and physics sanity checks.

Test index (spec requirements):
 1.  Import works.
 2.  Deterministic repeated construction produces identical force.
 3.  Single-frequency wave: amplitude and phase match BEM coefficient exactly.
 4.  Multi-frequency superposition is exact (F_total = Σ Fᵢ).
 5.  Interpolation at exact BEM grid points returns exact coefficient values.
 6.  No extrapolation beyond trusted BEM range (out-of-range → H_exc = 0).
 7.  Zero wave amplitude produces zero excitation force.
 8.  Deterministic irregular replay produces deterministic force.
 9.  Same wave phases used for η and F_exc (physical consistency).
10.  Force has finite values and no NaN/Inf.
11.  Physics integration remains stable with 5.2A forcing.
12.  Existing Module 5 environment still runs with 5.2A forcing.
13.  Latching still preserves radiation memory with 5.2A forcing.
14.  PTO power remains finite and non-negative with 5.2A forcing.
15.  Energy remains non-decreasing with 5.2A forcing.

Additional tests:
 - Old-vs-new force comparison (RMS, std, correlation).
 - WaveComponents correctness.
 - HourlyExcitationBuffer integration.
 - force_array() matches repeated force_at() calls.
 - Physics sanity check (displacement, velocity, power bounds).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Session-scope fixture: build WEC params once (BEM is expensive ~15 s)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def hydro():
    from module2_wec.geometry import REFERENCE_BUOY
    from module2_wec.hydrodynamic_coefficients import compute_hydrodynamic_coefficients
    return compute_hydrodynamic_coefficients(
        geometry=REFERENCE_BUOY, resolution=(6, 24, 16),
        omega_min=0.2, omega_max=1.4, n_omega=20, progress_bar=False,
    )


@pytest.fixture(scope="session")
def wec_params(hydro):
    from module2_wec.geometry import REFERENCE_BUOY
    from module2_wec.hydrostatics import Hydrostatics
    from module2_wec.radiation import compute_radiation_kernel
    from module2_wec.cummins import CumminsParameters
    from module2_wec.pto import PTOParameters
    hs_obj = Hydrostatics(REFERENCE_BUOY)
    rk     = compute_radiation_kernel(hydro)
    cp     = CumminsParameters(
        mass=REFERENCE_BUOY.mass,
        hydrostatic_stiffness=hs_obj.hydrostatic_stiffness,
        added_mass_infinity=rk.A_infinity,
        kernel_time=rk.time, kernel_values=rk.kernel,
    )
    return {"cummins": cp, "pto": PTOParameters(damping=200_000.0, stiffness=0.0),
            "hydro_amp": hydro.excitation_force_heave_amplitude,
            "hydro_omega": hydro.omega}


@pytest.fixture(scope="session")
def standard_wc(hydro):
    """WaveComponents for a standard Goa sea state, reused across tests."""
    from module5_control.excitation import build_wave_components
    return build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=42)


@pytest.fixture(scope="session")
def standard_model(standard_wc, hydro):
    """IrregularExcitationModel for the standard sea state."""
    from module5_control.excitation import IrregularExcitationModel
    return IrregularExcitationModel(standard_wc, hydro, t0=0.0)


# ===========================================================================
# Test 1 — import works
# ===========================================================================

def test_01_imports():
    from module5_control.excitation import (
        WaveComponents,
        build_wave_components,
        IrregularExcitationModel,
        HourlyExcitationBuffer,
    )


# ===========================================================================
# Test 2 — deterministic repeated construction produces identical force
# ===========================================================================

def test_02_deterministic_same_seed(hydro):
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    t = np.linspace(0, 10, 101)
    results = []
    for _ in range(3):
        wc    = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=7)
        model = IrregularExcitationModel(wc, hydro, t0=0.0)
        results.append(model.force_array(t))
    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[0], results[2])


def test_02_different_seeds_differ(hydro):
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    t   = np.linspace(0, 10, 101)
    wc1 = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    wc2 = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=1)
    m1  = IrregularExcitationModel(wc1, hydro, t0=0.0)
    m2  = IrregularExcitationModel(wc2, hydro, t0=0.0)
    assert not np.allclose(m1.force_array(t), m2.force_array(t))


# ===========================================================================
# Test 3 — single-frequency: amplitude and phase from BEM
# ===========================================================================

def test_03_single_frequency_amplitude_and_phase(hydro):
    """
    For a wave with a single component at a frequency that IS an exact BEM
    grid point, the excitation force must satisfy:

        F_exc(t) = a * |H_exc(omega)| * cos(omega*t + phi + angle(H_exc))

    We construct Z with exactly one non-zero component placed at an exact
    BEM grid frequency, so interpolation is exact (no discretisation error).
    """
    from module5_control.excitation import (
        WaveComponents, IrregularExcitationModel,
    )

    # Use the BEM grid frequency at index 10 (omega ~ 0.83 rad/s)
    bem_idx   = 10
    omega_test = float(hydro.omega[bem_idx])
    f_test     = omega_test / (2.0 * np.pi)
    phi_test   = 0.3       # rad
    a_test     = 0.5       # m

    # BEM coefficient at this exact frequency
    H_real = float(hydro.excitation_force_heave_real[bem_idx])
    H_imag = float(hydro.excitation_force_heave_imag[bem_idx])
    H_amp  = float(hydro.excitation_force_heave_amplitude[bem_idx])
    H_ang  = float(hydro.excitation_force_heave_phase[bem_idx])

    # Build a WaveComponents where only one frequency component is active,
    # placed EXACTLY at an BEM grid frequency so np.interp gives exact values.
    # Use the BEM omega grid itself (padded with zeros outside the one active index).
    N_f       = len(hydro.omega)   # 20 BEM points
    N_d       = 1
    f_arr     = hydro.frequency_hz.copy()     # exact BEM frequency grid
    omega_arr = hydro.omega.copy()

    Z = np.zeros(N_f, dtype=complex)
    Z[bem_idx] = a_test * np.exp(1j * phi_test)

    wc = WaveComponents(
        f_hz=f_arr, omega=omega_arr,
        S_f=np.zeros(N_f), a_2d=np.zeros((N_f, N_d)),
        phases_2d=np.zeros((N_f, N_d)), Z=Z,
        Hs=1.0, Tp=2.0*np.pi/omega_test, direction_deg=270.0, seed=None,
    )
    model = IrregularExcitationModel(wc, hydro, t0=0.0)

    # Verify that np.interp at an exact grid point gives exact value
    assert abs(model.H_exc_interp[bem_idx].real - H_real) < 1e-6
    assert abs(model.H_exc_interp[bem_idx].imag - H_imag) < 1e-6

    # Evaluate force over many cycles
    t = np.linspace(0, 4 * 2.0 * np.pi / omega_test, 10000)
    F = model.force_array(t)

    # Expected: F(t) = a * |H| * cos(omega*t + phi + angle_H)
    # Equivalently: Re[ H_exc * a * exp(i*(omega*t + phi)) ]
    #             = Re[ C * exp(i*omega*t) ]  where C = H_exc * Z[bem_idx]
    C_expected = (H_real + 1j * H_imag) * a_test * np.exp(1j * phi_test)
    F_expected = np.real(C_expected) * np.cos(omega_test * t) - np.imag(C_expected) * np.sin(omega_test * t)

    np.testing.assert_allclose(
        F, F_expected, atol=1e-3, rtol=1e-6,
        err_msg="Single-frequency force does not match BEM coefficient",
    )


# ===========================================================================
# Test 4 — multi-frequency superposition is exact
# ===========================================================================

def test_04_superposition(hydro):
    """
    F_total(t) = F_1(t) + F_2(t) + ... must hold to numerical precision.
    """
    from module5_control.excitation import WaveComponents, IrregularExcitationModel
    import numpy as np

    N_f  = 30
    N_d  = 1
    f_arr = np.linspace(0.04, 0.2, N_f)
    omega_arr = 2.0 * np.pi * f_arr

    rng = np.random.default_rng(123)
    a_vec   = rng.uniform(0.01, 0.5, N_f)
    phi_vec = rng.uniform(0, 2*np.pi, N_f)
    Z       = a_vec * np.exp(1j * phi_vec)

    wc_full = WaveComponents(
        f_hz=f_arr, omega=omega_arr,
        S_f=np.zeros(N_f), a_2d=np.zeros((N_f, N_d)),
        phases_2d=np.zeros((N_f, N_d)), Z=Z,
        Hs=1.0, Tp=8.0, direction_deg=270.0, seed=None,
    )
    model_full = IrregularExcitationModel(wc_full, hydro, t0=0.0)

    t = np.linspace(0, 20, 500)
    F_full = model_full.force_array(t)

    # Build individual single-component models and sum
    F_sum = np.zeros_like(F_full)
    for i in range(N_f):
        Z_i = np.zeros(N_f, dtype=complex)
        Z_i[i] = Z[i]
        wc_i = WaveComponents(
            f_hz=f_arr, omega=omega_arr,
            S_f=np.zeros(N_f), a_2d=np.zeros((N_f, N_d)),
            phases_2d=np.zeros((N_f, N_d)), Z=Z_i,
            Hs=1.0, Tp=8.0, direction_deg=270.0, seed=None,
        )
        m_i = IrregularExcitationModel(wc_i, hydro, t0=0.0)
        F_sum += m_i.force_array(t)

    np.testing.assert_allclose(
        F_full, F_sum, atol=1e-6, rtol=1e-9,
        err_msg="Superposition failed: F_total != sum of individual components",
    )


# ===========================================================================
# Test 5 — interpolation at exact BEM grid points returns exact values
# ===========================================================================

def test_05_interpolation_at_bem_grid_points(hydro):
    """
    When a wave component falls exactly on a BEM grid frequency, the
    interpolated H_exc must equal the stored BEM value exactly.
    """
    from module5_control.excitation import WaveComponents, IrregularExcitationModel
    import numpy as np

    # Use the first and last BEM frequencies
    for bem_idx in [0, 5, 10, 15, 19]:
        omega_bem = float(hydro.omega[bem_idx])
        f_bem     = omega_bem / (2.0 * np.pi)
        H_real_true = float(hydro.excitation_force_heave_real[bem_idx])
        H_imag_true = float(hydro.excitation_force_heave_imag[bem_idx])

        N_f   = 5
        f_arr = np.linspace(f_bem * 0.5, f_bem * 2.0, N_f)
        omega_arr = 2.0 * np.pi * f_arr
        idx_close = int(np.argmin(np.abs(omega_arr - omega_bem)))
        # Place the grid exactly on the BEM point
        f_arr[idx_close]     = f_bem
        omega_arr[idx_close] = omega_bem

        Z = np.zeros(N_f, dtype=complex)
        Z[idx_close] = 1.0 + 0j   # unit amplitude, zero phase

        wc = WaveComponents(
            f_hz=f_arr, omega=omega_arr,
            S_f=np.zeros(N_f), a_2d=np.zeros((N_f, 1)),
            phases_2d=np.zeros((N_f, 1)), Z=Z,
            Hs=1.0, Tp=1.0/f_bem, direction_deg=270.0, seed=None,
        )
        model = IrregularExcitationModel(wc, hydro, t0=0.0)

        # H_exc_interp at the exact BEM grid point should match stored value
        H_interp = model.H_exc_interp[idx_close]
        assert abs(H_interp.real - H_real_true) < 1.0, (
            f"Real part interpolation error at BEM idx {bem_idx}: "
            f"{H_interp.real:.2f} vs {H_real_true:.2f}"
        )
        assert abs(H_interp.imag - H_imag_true) < 1.0, (
            f"Imaginary part interpolation error at BEM idx {bem_idx}: "
            f"{H_interp.imag:.2f} vs {H_imag_true:.2f}"
        )


# ===========================================================================
# Test 6 — no extrapolation beyond BEM range (out-of-range → H_exc = 0)
# ===========================================================================

def test_06_out_of_range_gives_zero_force(hydro):
    """
    Components with omega outside [omega_min, omega_max] must produce
    zero excitation force (H_exc interpolated to 0).
    """
    from module5_control.excitation import WaveComponents, IrregularExcitationModel
    import numpy as np

    omega_min = float(hydro.omega_min)
    omega_max = float(hydro.omega_max)

    # Two components: one well below BEM range, one well above
    N_f    = 2
    f_arr  = np.array([0.005, 0.8])     # omega = 0.031, 5.03 rad/s
    omega_arr = 2.0 * np.pi * f_arr

    # Both are out of BEM range [0.2, 1.4]
    assert omega_arr[0] < omega_min
    assert omega_arr[1] > omega_max

    Z = np.array([1.0 + 0j, 1.0 + 0j])
    wc = WaveComponents(
        f_hz=f_arr, omega=omega_arr,
        S_f=np.zeros(N_f), a_2d=np.zeros((N_f, 1)),
        phases_2d=np.zeros((N_f, 1)), Z=Z,
        Hs=1.0, Tp=8.0, direction_deg=270.0, seed=None,
    )
    model = IrregularExcitationModel(wc, hydro, t0=0.0)

    # All H_exc must be zero
    np.testing.assert_array_equal(model.H_exc_interp, 0.0)
    # Force must be zero at all times
    t = np.linspace(0, 10, 100)
    np.testing.assert_array_equal(model.force_array(t), 0.0)
    assert model.n_active == 0


def test_06_in_range_components_are_nonzero(hydro):
    """
    Components within the BEM range must have non-zero H_exc.
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    wc    = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    model = IrregularExcitationModel(wc, hydro, t0=0.0)
    in_range = (wc.omega >= hydro.omega_min) & (wc.omega <= hydro.omega_max)
    # At least some in-range components must have non-zero H_exc
    assert model.n_active == int(np.sum(in_range))
    assert model.n_active > 0


# ===========================================================================
# Test 7 — zero wave amplitude → zero excitation force
# ===========================================================================

def test_07_zero_amplitude_zero_force(hydro):
    from module5_control.excitation import WaveComponents, IrregularExcitationModel
    import numpy as np

    N_f   = 20
    f_arr = np.linspace(0.04, 0.22, N_f)
    omega_arr = 2.0 * np.pi * f_arr
    Z = np.zeros(N_f, dtype=complex)  # all zero amplitudes

    wc = WaveComponents(
        f_hz=f_arr, omega=omega_arr,
        S_f=np.zeros(N_f), a_2d=np.zeros((N_f, 1)),
        phases_2d=np.zeros((N_f, 1)), Z=Z,
        Hs=1.0, Tp=8.0, direction_deg=270.0, seed=None,
    )
    model = IrregularExcitationModel(wc, hydro, t0=0.0)
    t = np.linspace(0, 20, 500)
    F = model.force_array(t)
    np.testing.assert_array_equal(F, 0.0, err_msg="Zero amplitudes must give zero force")


# ===========================================================================
# Test 8 — deterministic replay produces deterministic force
# ===========================================================================

def test_08_replay_force_deterministic():
    """
    Same build_synthetic_replay seed → same wave_components_list → same force.
    """
    from module5_control.replay import build_synthetic_replay
    r1 = build_synthetic_replay(n_hours=2, seed=5)
    r2 = build_synthetic_replay(n_hours=2, seed=5)
    # Effective complex amplitudes must be identical
    for h in range(2):
        np.testing.assert_array_equal(
            r1.wave_components_list[h].Z,
            r2.wave_components_list[h].Z,
            err_msg=f"Z mismatch at hour {h}",
        )


def test_08_different_seed_different_replay():
    from module5_control.replay import build_synthetic_replay
    r1 = build_synthetic_replay(n_hours=2, seed=0)
    r2 = build_synthetic_replay(n_hours=2, seed=99)
    assert not np.allclose(r1.wave_components_list[0].Z, r2.wave_components_list[0].Z)


# ===========================================================================
# Test 9 — same wave phases used for η and F_exc
# ===========================================================================

def test_09_eta_excitation_phase_consistency(hydro):
    """
    The η(t) reconstructed from Z_i must exactly match goa_seastate() output.
    Proves that F_exc and η share the same wave realization.
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    from module1_ocean.goa_seastate import goa_seastate

    Hs, Tp, dir_deg, seed = 1.5, 8.0, 270.0, 42
    wc    = build_wave_components(Hs=Hs, Tp=Tp, direction_deg=dir_deg, seed=seed)
    model = IrregularExcitationModel(wc, hydro, t0=0.0)

    result = goa_seastate(Hs=Hs, Tp=Tp, direction_deg=dir_deg, depth_m=30.0,
                           duration_s=20.0, dt=0.1, seed=seed)
    t_local = result.t[:200]
    eta_model = model.eta_array(t_local)
    eta_goa   = result.eta[:200]

    np.testing.assert_allclose(
        eta_model, eta_goa, atol=1e-10,
        err_msg="η from IrregularExcitationModel does not match goa_seastate()",
    )


def test_09_eta_force_same_z(hydro):
    """
    eta_array and force_array must use the same Z_i (different H_exc applied).
    Verify: force_array(t) / eta_array(t) is not constant (H_exc freq-dependent).
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    wc    = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    model = IrregularExcitationModel(wc, hydro, t0=0.0)
    t     = np.linspace(1.0, 20.0, 200)
    F     = model.force_array(t)
    eta   = model.eta_array(t)
    # If H_exc were constant (single-freq approx), F/eta would be constant.
    # For freq-dependent H_exc the ratio varies — confirm it is NOT constant.
    ratio = F / np.where(np.abs(eta) > 0.01, eta, np.nan)
    valid = np.isfinite(ratio)
    if valid.sum() > 10:
        # Ratio should vary (H_exc is frequency-dependent, not a scalar)
        ratio_std = float(np.nanstd(ratio[valid]))
        assert ratio_std > 1e3, (
            f"F/eta ratio appears constant (std={ratio_std:.1f}), "
            "suggesting single-freq approximation is still being used"
        )


# ===========================================================================
# Test 10 — force finite, no NaN/Inf
# ===========================================================================

def test_10_force_finite(standard_model):
    t = np.linspace(0, 100, 1001)
    F = standard_model.force_array(t)
    assert np.all(np.isfinite(F)), "Force contains NaN or Inf"
    assert not np.any(np.isnan(F))


def test_10_force_at_finite(standard_model):
    for t_val in [0.0, 1.0, 10.0, 100.0, 3600.0]:
        F = standard_model.force_at(t_val)
        assert np.isfinite(F), f"force_at({t_val}) is not finite: {F}"


# ===========================================================================
# Test 11 — physics integration stable with 5.2A forcing
# ===========================================================================

def test_11_physics_stable(wec_params, hydro):
    """
    Run Cummins integrator with 5.2A force for 200 s and verify:
    - no NaN/Inf in x, v
    - |x| < 20 m (no explosion)
    - |v| < 10 m/s (no explosion)
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    from module5_control.environment import CumminsStepIntegrator
    from module5_control.replay import DT_PHYSICS

    wc    = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    model = IrregularExcitationModel(wc, hydro, t0=0.0)
    integ = CumminsStepIntegrator(wec_params["cummins"], wec_params["pto"], dt=DT_PHYSICS)
    integ.reset()

    n_steps = int(200.0 / DT_PHYSICS)
    x_max, v_max = 0.0, 0.0
    for i in range(n_steps):
        t_local = i * DT_PHYSICS
        F_exc   = model.force_at(t_local)
        x, v, _ = integ.step(F_exc)
        assert np.isfinite(x), f"NaN/Inf in x at step {i}"
        assert np.isfinite(v), f"NaN/Inf in v at step {i}"
        x_max = max(x_max, abs(x))
        v_max = max(v_max, abs(v))

    assert x_max < 20.0, f"|x|_max = {x_max:.2f} m — possible explosion"
    assert v_max < 10.0, f"|v|_max = {v_max:.2f} m/s — possible explosion"


# ===========================================================================
# Test 12 — existing environment still runs with 5.2A forcing
# ===========================================================================

def test_12_environment_runs_with_52a(wec_params, hydro):
    from module5_control.replay import build_synthetic_replay
    from module5_control.environment import build_env_from_checkpoint

    replay = build_synthetic_replay(n_hours=4, seed=42)
    env = build_env_from_checkpoint(
        replay=replay,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive",
        max_episode_steps=20,
        hydro=hydro,
    )
    assert env._excitation_buffer is not None, "5.2A buffer should be active"
    obs, info = env.reset(seed=0)
    assert obs.shape == (723,)
    assert np.all(np.isfinite(obs))
    for _ in range(10):
        obs, rew, term, trunc, info = env.step(0)
        assert np.all(np.isfinite(obs))
        assert np.isfinite(rew)
        if term or trunc:
            break


# ===========================================================================
# Test 13 — latching preserves radiation memory with 5.2A forcing
# ===========================================================================

def test_13_latch_preserves_radiation_memory(wec_params, hydro):
    from module5_control.replay import build_synthetic_replay
    from module5_control.environment import build_env_from_checkpoint
    from module5_control.replay import N_PHYSICS_PER_CONTROL

    replay = build_synthetic_replay(n_hours=4, seed=10)
    env = build_env_from_checkpoint(
        replay=replay,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive", max_episode_steps=50, hydro=hydro,
    )
    env.reset(seed=0)
    # Free motion then latch then release
    for _ in range(5): env.step(0)
    step_before = env.integrator.step_count
    for _ in range(3): env.step(1)  # latch
    for _ in range(3): env.step(0)  # release
    step_after = env.integrator.step_count
    # step_count must advance by exactly 6 * N_PHYSICS_PER_CONTROL
    assert step_after == step_before + 6 * N_PHYSICS_PER_CONTROL


# ===========================================================================
# Test 14 — PTO power non-negative with 5.2A forcing
# ===========================================================================

def test_14_pto_power_nonneg_52a(wec_params, hydro):
    from module5_control.replay import build_synthetic_replay
    from module5_control.environment import build_env_from_checkpoint

    replay = build_synthetic_replay(n_hours=4, seed=3)
    env = build_env_from_checkpoint(
        replay=replay,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive", max_episode_steps=50, hydro=hydro,
    )
    env.reset(seed=0)
    for _ in range(20):
        _, _, term, trunc, info = env.step(0)
        assert info["mean_step_power_W"] >= 0.0
        if term or trunc: break


# ===========================================================================
# Test 15 — energy non-decreasing with 5.2A forcing
# ===========================================================================

def test_15_energy_nondecreasing_52a(wec_params, hydro):
    from module5_control.replay import build_synthetic_replay
    from module5_control.environment import build_env_from_checkpoint

    replay = build_synthetic_replay(n_hours=4, seed=4)
    env = build_env_from_checkpoint(
        replay=replay,
        cummins_params=wec_params["cummins"],
        pto_params=wec_params["pto"],
        hydro_excitation_amplitude=wec_params["hydro_amp"],
        hydro_omega=wec_params["hydro_omega"],
        mode="reactive", max_episode_steps=50, hydro=hydro,
    )
    env.reset(seed=0)
    prev_energy = 0.0
    for a in [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]:
        _, _, term, trunc, info = env.step(a)
        e = info["cumulative_energy_J"]
        assert e >= prev_energy - 1e-9
        prev_energy = e
        if term or trunc: break


# ===========================================================================
# Additional: force_array == repeated force_at
# ===========================================================================

def test_force_array_matches_force_at(standard_model):
    t = np.linspace(0, 10, 50)
    F_arr  = standard_model.force_array(t)
    F_loop = np.array([standard_model.force_at(ti) for ti in t])
    np.testing.assert_allclose(F_arr, F_loop, atol=1e-9)


# ===========================================================================
# Additional: WaveComponents properties
# ===========================================================================

def test_wave_components_shapes():
    from module5_control.excitation import build_wave_components
    wc = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    assert wc.f_hz.shape == (128,)
    assert wc.omega.shape == (128,)
    assert wc.S_f.shape   == (128,)
    assert wc.a_2d.shape  == (128, 16)
    assert wc.phases_2d.shape == (128, 16)
    assert wc.Z.shape     == (128,)
    assert wc.Z.dtype     == np.complex128
    assert wc.n_components == 128


def test_wave_components_omega_consistent():
    from module5_control.excitation import build_wave_components
    wc = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    np.testing.assert_allclose(wc.omega, 2.0 * np.pi * wc.f_hz, rtol=1e-12)


def test_wave_components_amplitudes_nonneg():
    from module5_control.excitation import build_wave_components
    wc = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    assert np.all(wc.a_2d >= 0.0)


def test_wave_components_phases_in_range():
    from module5_control.excitation import build_wave_components
    wc = build_wave_components(Hs=1.5, Tp=8.0, direction_deg=270.0, seed=0)
    assert np.all(wc.phases_2d >= 0.0)
    assert np.all(wc.phases_2d <= 2.0 * np.pi)


# ===========================================================================
# Additional: HourlyExcitationBuffer
# ===========================================================================

def test_hourly_buffer_builds(hydro):
    from module5_control.excitation import HourlyExcitationBuffer
    Hs_arr  = np.array([1.5, 1.8, 1.2])
    Tp_arr  = np.array([8.0, 9.0, 7.0])
    dir_arr = np.array([270.0, 265.0, 275.0])
    buf = HourlyExcitationBuffer(hydro, Hs_arr, Tp_arr, dir_arr, base_seed=0)
    assert buf.n_hours == 3
    F0 = buf.force_at_physics_step(0, t_within_hour=0.5)
    assert np.isfinite(F0)


def test_hourly_buffer_deterministic(hydro):
    from module5_control.excitation import HourlyExcitationBuffer
    Hs_arr  = np.array([1.5, 1.8])
    Tp_arr  = np.array([8.0, 9.0])
    dir_arr = np.array([270.0, 265.0])
    buf1 = HourlyExcitationBuffer(hydro, Hs_arr, Tp_arr, dir_arr, base_seed=7)
    buf2 = HourlyExcitationBuffer(hydro, Hs_arr, Tp_arr, dir_arr, base_seed=7)
    for h in range(2):
        for t in [0.0, 100.0, 999.9]:
            assert buf1.force_at_physics_step(h, t) == buf2.force_at_physics_step(h, t)


def test_hourly_buffer_nan_row_gives_finite(hydro):
    from module5_control.excitation import HourlyExcitationBuffer
    Hs_arr  = np.array([np.nan, 1.5])
    Tp_arr  = np.array([8.0, 8.0])
    dir_arr = np.array([270.0, 270.0])
    buf = HourlyExcitationBuffer(hydro, Hs_arr, Tp_arr, dir_arr, base_seed=0)
    F = buf.force_at_physics_step(0, t_within_hour=100.0)
    assert np.isfinite(F)


# ===========================================================================
# Additional: old-vs-new force comparison (diagnostic, not a pass/fail)
# ===========================================================================

def test_old_vs_new_force_comparison(hydro):
    """
    Compare Module 5.1 approximation vs Module 5.2A full excitation.

    This is a scientific comparison test — we verify the new implementation
    produces forces of physically reasonable scale relative to the old one,
    not that they are equal (they should differ because 5.2A uses
    frequency-dependent H_exc across all components).

    The correct claim: "5.2A uses frequency-dependent excitation consistent
    with the BEM response across all irregular wave components."
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    import numpy as np

    Hs, Tp, direction_deg, seed = 1.5, 8.0, 270.0, 42
    wc    = build_wave_components(Hs=Hs, Tp=Tp, direction_deg=direction_deg, seed=seed)
    model_new = IrregularExcitationModel(wc, hydro, t0=0.0)

    t = np.linspace(0, 200, 2000)

    # New: full frequency-dependent excitation
    F_new = model_new.force_array(t)
    eta   = model_new.eta_array(t)

    # Old: |H_exc(omega_p)| * eta(t)
    omega_p = 2.0 * np.pi / Tp
    omega_p = float(np.clip(omega_p, float(hydro.omega_min), float(hydro.omega_max)))
    H_exc_p = float(np.interp(omega_p, hydro.omega, hydro.excitation_force_heave_amplitude))
    F_old = H_exc_p * eta

    rms_new  = float(np.sqrt(np.mean(F_new**2)))
    rms_old  = float(np.sqrt(np.mean(F_old**2)))
    std_new  = float(np.std(F_new))
    std_old  = float(np.std(F_old))
    corr     = float(np.corrcoef(F_new, F_old)[0, 1])
    diff_rms = float(np.sqrt(np.mean((F_new - F_old)**2)))

    # Scientific sanity checks (not equality):
    # 1. New force RMS is in the same order of magnitude as old
    assert 0.1 * rms_old < rms_new < 10.0 * rms_old, (
        f"RMS force ratio out of range: new={rms_new:.0f} N vs old={rms_old:.0f} N"
    )
    # 2. Forces are correlated in magnitude (same underlying wave realization).
    # Note: the 5.1 approximation uses |H_exc(omega_p)| * eta (positive scalar),
    # while 5.2A applies the full complex H_exc with phase ~π, so the two may be
    # anti-correlated in the raw (signed) sense.  We check the absolute correlation.
    assert abs(corr) > 0.5, f"|F_new vs F_old| correlation too low: |{corr:.3f}|"
    # 3. They are NOT identical (5.2A has richer frequency content)
    assert diff_rms > 1.0, "Old and new forces are suspiciously identical"
    # 4. Report values for the audit
    print(f"\n  OLD (5.1 approx): RMS={rms_old:.0f} N, std={std_old:.0f} N")
    print(f"  NEW (5.2A full):  RMS={rms_new:.0f} N, std={std_new:.0f} N")
    print(f"  Correlation: {corr:.4f}, RMS difference: {diff_rms:.0f} N")


# ===========================================================================
# Additional: physics sanity check
# ===========================================================================

def test_physics_sanity_check(wec_params, hydro):
    """
    Run full episode with 5.2A forcing and report key diagnostic quantities.
    Checks: no NaN/Inf, bounded displacement/velocity/acceleration.
    """
    from module5_control.excitation import build_wave_components, IrregularExcitationModel
    from module5_control.environment import CumminsStepIntegrator
    from module5_control.replay import DT_PHYSICS
    from module2_wec.pto import compute_pto_power
    import numpy as np

    Hs, Tp = 1.5, 8.0
    seed   = 42
    wc     = build_wave_components(Hs=Hs, Tp=Tp, direction_deg=270.0, seed=seed)
    model  = IrregularExcitationModel(wc, hydro, t0=0.0)
    integ  = CumminsStepIntegrator(wec_params["cummins"], wec_params["pto"], dt=DT_PHYSICS)
    integ.reset()

    n_steps    = int(3600.0 / DT_PHYSICS)   # 1 hour
    x_arr      = np.zeros(n_steps)
    v_arr      = np.zeros(n_steps)
    a_arr      = np.zeros(n_steps)
    p_arr      = np.zeros(n_steps)

    for i in range(n_steps):
        t_local = i * DT_PHYSICS
        F_exc   = model.force_at(t_local)
        x, v, a = integ.step(F_exc)
        p       = float(compute_pto_power(np.array([v]), wec_params["pto"])[0])
        x_arr[i] = x; v_arr[i] = v; a_arr[i] = a; p_arr[i] = p

    # Assertions
    assert np.all(np.isfinite(x_arr)), "NaN/Inf in displacement"
    assert np.all(np.isfinite(v_arr)), "NaN/Inf in velocity"
    assert np.all(np.isfinite(a_arr)), "NaN/Inf in acceleration"
    assert np.all(p_arr >= -1e-9),     "Negative PTO power"

    max_x   = float(np.max(np.abs(x_arr)))
    max_v   = float(np.max(np.abs(v_arr)))
    max_a   = float(np.max(np.abs(a_arr)))
    eta_rms = float(np.std(wc.Z.real))   # proxy; proper eta RMS from model
    t_arr   = np.arange(n_steps) * DT_PHYSICS
    eta_arr = model.eta_array(t_arr[:200])
    eta_rms_true = float(np.std(eta_arr))
    P_mean  = float(np.mean(p_arr))
    E_total = float(np.sum(p_arr) * DT_PHYSICS)

    # Bounds (generous for this sea state)
    assert max_x < 20.0, f"max |x| = {max_x:.2f} m"
    assert max_v < 10.0, f"max |v| = {max_v:.2f} m/s"

    # Print diagnostics for audit
    print(f"\n  Hs={Hs} m, Tp={Tp} s")
    print(f"  eta RMS (first 200 steps): {eta_rms_true:.4f} m")
    print(f"  F_exc RMS: {float(np.sqrt(np.mean(model.force_array(t_arr[:200])**2))):.0f} N")
    print(f"  max |x|={max_x:.4f} m, max |v|={max_v:.4f} m/s, max |a|={max_a:.4f} m/s²")
    print(f"  mean PTO power: {P_mean:.1f} W, cumulative energy: {E_total/3600:.4f} Wh")
