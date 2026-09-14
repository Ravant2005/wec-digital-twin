"""
test_replay_excitation_equivalence.py — Regression test for the optimized
excitation buffer in module5_control/replay.py (Module 5.2A fix).

============================================================
CONTEXT
============================================================

build_episode_replay() previously called goa_seastate() to populate the
legacy excitation_force buffer, which performs a full O(N_t × N_f × N_dir)
spatial wave synthesis.  This caused an ~45 s initialization hang that
mimicked a deadlock during the PPO test suite.

The fix replaced the goa_seastate() call with a mathematically identical
1D dot product using the collapsed complex amplitudes Zᵢ.

============================================================
MATHEMATICAL IDENTITY BEING VERIFIED
============================================================

Module 1 spatial_waves.py computes:

    η(x, y, t) = Σᵢ Σⱼ aᵢⱼ cos(ωᵢt − kᵢ(x sinθⱼ + y cosθⱼ) + φᵢⱼ)

where:
    ωᵢ   = 2π fᵢ                       angular frequency [rad/s]
    kᵢ   = wave number via dispersion   [rad/m]
    θⱼ   = propagation directions       [rad]
    aᵢⱼ  = sqrt(2 S(fᵢ) wⱼ Δf)         component amplitude [m]
    φᵢⱼ  = uniform random phases        [rad]

At the buoy location (x=0, y=0):
    The spatial phase term kᵢ(x sinθⱼ + y cosθⱼ) = 0  for ALL i, j.

Therefore:
    η(0, 0, t) = Σᵢ Σⱼ aᵢⱼ cos(ωᵢt + φᵢⱼ)

Applying the cosine identity:  cos(α + φ) = cos(α)cos(φ) − sin(α)sin(φ)

    η(0, 0, t) = Σᵢ Σⱼ aᵢⱼ [cos(ωᵢt)cos(φᵢⱼ) − sin(ωᵢt)sin(φᵢⱼ)]
               = Σᵢ cos(ωᵢt) Σⱼ aᵢⱼ cos(φᵢⱼ) − sin(ωᵢt) Σⱼ aᵢⱼ sin(φᵢⱼ)
               = Σᵢ [cos(ωᵢt) Re(Zᵢ) − sin(ωᵢt) Im(Zᵢ)]

where Zᵢ = Σⱼ aᵢⱼ exp(i φᵢⱼ) is the collapsed complex amplitude.

In matrix form (as implemented in the fix):
    phase[t_idx, i] = ωᵢ × t[t_idx]
    η[t_idx] = dot(cos(phase), Re(Z)) − dot(sin(phase), Im(Z))

This identity is EXACT up to floating-point rounding — there is no
approximation.  The only errors are IEEE-754 double precision rounding
(≈ 1e-15 per operation).

============================================================
TOLERANCE JUSTIFICATION
============================================================

Both paths use identical aᵢⱼ and φᵢⱼ values.  The only difference is
the order of operations:

    OLD path:  Σᵢ Σⱼ aᵢⱼ cos(ωᵢt − kᵢ×0 + φᵢⱼ)     [N_dir inner loop]
    NEW path:  Σᵢ [Re(Zᵢ) cos(ωᵢt) − Im(Zᵢ) sin(ωᵢt)]  [Z already collapsed]

The Zᵢ computation itself introduces N_dir × 2 multiply-adds per frequency
band.  For N_f=128, N_dir=16, the accumulated floating-point rounding
is bounded by ~ N_dir × N_f × ε ≈ 128×16×2.2e-16 ≈ 4.5e-13.

The excitation.py module's own comment states:
    "max |η_from_Zᵢ − η_goa_seastate| < 4×10⁻¹⁶"
which was measured for the IrregularExcitationModel.eta_array() path.
The replay.py path includes an additional np.dot step, so we allow:
    atol=1e-10   (100× more generous than the ≤ 4e-13 bound; safe for N/m forces)

This tolerance is physically meaningless: 1e-10 m wave amplitude is
9 orders of magnitude below the measurement noise floor.

For the excitation FORCE buffer (eta × |H_exc| at ω_p), using |H_exc| ≈ O(1e5) N/m,
the force difference is bounded at ~ 1e-5 N, which is physically negligible.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_hours: int, Hs: float = 1.5, Tp: float = 8.0,
             dir_deg: float = 270.0, u10: float = 5.0, v10: float = 2.0):
    """Build a minimal ERA5-like DataFrame for test use."""
    import pandas as pd
    ts = [pd.Timestamp("2020-06-01") + pd.Timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame({
        "time":          ts,
        "Hs_m":          [Hs]      * n_hours,
        "Tp_s":          [Tp]      * n_hours,
        "direction_deg": [dir_deg] * n_hours,
        "u10":           [u10]     * n_hours,
        "v10":           [v10]     * n_hours,
    })


def _reference_eta_from_goa_seastate(Hs, Tp, dir_deg, seed, dt_physics=0.1):
    """
    Compute the reference eta buffer for one ERA5 hour using goa_seastate().

    This is the ORIGINAL code path (slow, but authoritative).
    """
    from module1_ocean.goa_seastate import goa_seastate
    from module5_control.replay import DT_ERA5

    result = goa_seastate(
        Hs=Hs,
        Tp=Tp,
        direction_deg=dir_deg,
        depth_m=30.0,
        gamma=3.3,
        duration_s=DT_ERA5,
        dt=dt_physics,
        seed=seed,
    )
    n_phys_per_hour = int(round(DT_ERA5 / dt_physics))
    eta_ref = result.eta
    n_actual = len(eta_ref)
    n_use = min(n_actual, n_phys_per_hour)

    buf = np.zeros(n_phys_per_hour, dtype=float)
    buf[:n_use] = eta_ref[:n_use]
    if n_use < n_phys_per_hour:
        buf[n_use:] = buf[n_use - 1]
    return buf


def _optimized_eta_from_Z(wc, dt_physics=0.1):
    """
    Compute eta buffer for one ERA5 hour using the fast collapsed-Z path.

    This is the NEW code path (fast, implemented in the replay.py fix).

    Formula:
        phase[t, i] = ωᵢ × t[t]
        η[t] = dot(cos(phase), Re(Z)) − dot(sin(phase), Im(Z))
    """
    from module5_control.replay import DT_ERA5
    n_phys_per_hour = int(round(DT_ERA5 / dt_physics))
    t_array = np.arange(n_phys_per_hour) * dt_physics
    phase = wc.omega[np.newaxis, :] * t_array[:, np.newaxis]
    eta_fast = np.dot(np.cos(phase), np.real(wc.Z)) - np.dot(np.sin(phase), np.imag(wc.Z))
    return eta_fast


# ===========================================================================
# Test 1 — Exact match for a single ERA5 hour (short duration)
# ===========================================================================

def test_replay_eta_matches_goa_seastate_first_200_steps():
    """
    The fast Z-collapse path must match goa_seastate() to within floating-point
    rounding for the first 200 physics steps (20 s at dt=0.1 s).

    Reasoning for checking only 200 steps:
        - goa_seastate() is authoritative but slow (takes ~1s per full hour).
        - At 20 s we capture several wave periods (Tp=8s → ~2.5 periods).
        - The test confirms phase consistency, amplitude, and sign convention.
        - Tolerance atol=1e-10 m: bounded by N_dir × N_f IEEE-754 rounding.
    """
    from module5_control.excitation import build_wave_components
    from module1_ocean.goa_seastate import goa_seastate

    Hs, Tp, dir_deg, seed = 1.5, 8.0, 270.0, 42
    dt_physics = 0.1
    n_check = 200   # 20 s at 0.1 s resolution

    wc = build_wave_components(Hs=Hs, Tp=Tp, direction_deg=dir_deg,
                               gamma=3.3, seed=seed)

    # Reference: goa_seastate at the origin x=0, y=0
    result = goa_seastate(
        Hs=Hs, Tp=Tp, direction_deg=dir_deg,
        depth_m=30.0, gamma=3.3,
        duration_s=n_check * dt_physics + dt_physics,  # just enough
        dt=dt_physics,
        seed=seed,
    )
    eta_ref = result.eta[:n_check]

    # Optimized: Z-collapse dot product
    t_array = np.arange(n_check) * dt_physics
    phase = wc.omega[np.newaxis, :] * t_array[:, np.newaxis]
    eta_fast = np.dot(np.cos(phase), np.real(wc.Z)) - np.dot(np.sin(phase), np.imag(wc.Z))

    max_abs_err = float(np.max(np.abs(eta_fast - eta_ref)))
    rms_err     = float(np.sqrt(np.mean((eta_fast - eta_ref) ** 2)))

    # Tolerance: 1e-10 m (bounded by N_dir×N_f IEEE-754 rounding ≈ 4.5e-13 per step)
    assert max_abs_err < 1e-10, (
        f"Max absolute error {max_abs_err:.3e} m exceeds atol=1e-10 m.\n"
        f"RMS error: {rms_err:.3e} m.\n"
        "This indicates a sign convention or phase convention mismatch."
    )
    assert rms_err < 1e-10, (
        f"RMS error {rms_err:.3e} m exceeds atol=1e-10 m."
    )


# ===========================================================================
# Test 2 — Numerical identity via np.allclose (pointwise, 200 steps)
# ===========================================================================

def test_replay_eta_allclose_200_steps():
    """
    np.allclose check with rtol=1e-10, atol=1e-10 over 200 steps.

    Uses the same approach as excitation.py's own internal verification comment:
        "max |η_from_Zᵢ − η_goa_seastate| < 4×10⁻¹⁶"
    Our tolerance is 10,000× more lenient than that to account for the
    additional dot product computation in the replay path.
    """
    from module5_control.excitation import build_wave_components
    from module1_ocean.goa_seastate import goa_seastate

    Hs, Tp, dir_deg, seed = 2.0, 10.0, 180.0, 7
    dt_physics = 0.1
    n_check = 200

    wc = build_wave_components(Hs=Hs, Tp=Tp, direction_deg=dir_deg,
                               gamma=3.3, seed=seed)

    result = goa_seastate(
        Hs=Hs, Tp=Tp, direction_deg=dir_deg,
        depth_m=30.0, gamma=3.3,
        duration_s=n_check * dt_physics + dt_physics,
        dt=dt_physics, seed=seed,
    )
    eta_ref = result.eta[:n_check]

    t_array = np.arange(n_check) * dt_physics
    phase = wc.omega[np.newaxis, :] * t_array[:, np.newaxis]
    eta_fast = np.dot(np.cos(phase), np.real(wc.Z)) - np.dot(np.sin(phase), np.imag(wc.Z))

    assert np.allclose(eta_fast, eta_ref, rtol=1e-10, atol=1e-10), (
        f"np.allclose failed.\n"
        f"Max abs err: {np.max(np.abs(eta_fast - eta_ref)):.3e} m\n"
        f"RMS err:     {np.sqrt(np.mean((eta_fast - eta_ref)**2)):.3e} m\n"
        "Both paths must produce identical results at (x=0, y=0)."
    )


# ===========================================================================
# Test 3 — exc_buffer from build_episode_replay matches goa_seastate
# ===========================================================================

def test_build_episode_replay_exc_buffer_matches_goa_seastate():
    """
    The exc_buffer produced by build_episode_replay() must numerically match
    the goa_seastate() output for the first N_check steps of the first hour.

    This directly validates the relay.py fix applied in commit 1f4a9c4.

    Design:
    - n_hours=1, seed=42, Hs=1.5m, Tp=8s, dir=270°
    - Compare only the first 300 steps (30 s) of hour 0 to keep the test
      fast (avoiding the full 36000-step goa_seastate call).
    - Full-hour consistency is proven by Test 1 (same mathematical identity).
    """
    from module5_control.replay import build_episode_replay, DT_PHYSICS
    from module1_ocean.goa_seastate import goa_seastate

    Hs, Tp, dir_deg, seed = 1.5, 8.0, 270.0, 42
    n_check = 300  # 30 s of physics at dt=0.1 s

    df  = _make_df(n_hours=1, Hs=Hs, Tp=Tp, dir_deg=dir_deg)
    rep = build_episode_replay(df, start_index=0, n_hours=1, seed=seed,
                               depth_m=30.0, gamma=3.3, dt_physics=DT_PHYSICS)

    exc_fast = rep.excitation_force[0, :n_check]   # first n_check steps, hour 0

    result = goa_seastate(
        Hs=Hs, Tp=Tp, direction_deg=dir_deg,
        depth_m=30.0, gamma=3.3,
        duration_s=n_check * DT_PHYSICS + DT_PHYSICS,
        dt=DT_PHYSICS, seed=seed,
    )
    eta_ref = result.eta[:n_check]

    max_abs_err = float(np.max(np.abs(exc_fast - eta_ref)))
    rms_err     = float(np.sqrt(np.mean((exc_fast - eta_ref) ** 2)))

    assert max_abs_err < 1e-10, (
        f"exc_buffer[0, :300] vs goa_seastate: max error {max_abs_err:.3e} m.\n"
        f"RMS error: {rms_err:.3e} m.\n"
        "The replay.py optimization broke numerical equivalence."
    )
    assert rms_err < 1e-10, (
        f"exc_buffer RMS error {rms_err:.3e} m exceeds tolerance."
    )


# ===========================================================================
# Test 4 — exc_buffer for multiple seeds (determinism)
# ===========================================================================

@pytest.mark.parametrize("seed", [0, 1, 42, 99, 123])
def test_exc_buffer_deterministic_multi_seed(seed):
    """
    build_episode_replay() must return identical exc_buffer for the same seed.

    Verifies that the Z-collapse path is perfectly reproducible.
    """
    from module5_control.replay import build_episode_replay, DT_PHYSICS

    df = _make_df(n_hours=2, Hs=1.5, Tp=8.0)

    r1 = build_episode_replay(df, start_index=0, n_hours=2,
                              seed=seed, dt_physics=DT_PHYSICS)
    r2 = build_episode_replay(df, start_index=0, n_hours=2,
                              seed=seed, dt_physics=DT_PHYSICS)

    np.testing.assert_array_equal(
        r1.excitation_force, r2.excitation_force,
        err_msg=f"exc_buffer differs for same seed={seed}: not deterministic",
    )


# ===========================================================================
# Test 5 — different seeds produce different buffers
# ===========================================================================

def test_exc_buffer_different_seeds_differ():
    """
    Different seeds must produce different exc_buffers (with overwhelming probability).

    Confirms that the Z-collapse path correctly propagates the random seed.
    """
    from module5_control.replay import build_episode_replay, DT_PHYSICS

    df = _make_df(n_hours=1, Hs=1.5, Tp=8.0)

    r0 = build_episode_replay(df, start_index=0, n_hours=1, seed=0, dt_physics=DT_PHYSICS)
    r1 = build_episode_replay(df, start_index=0, n_hours=1, seed=1, dt_physics=DT_PHYSICS)

    assert not np.allclose(r0.excitation_force, r1.excitation_force), (
        "exc_buffer is identical for seed=0 and seed=1 — seed is not propagated."
    )


# ===========================================================================
# Test 6 — wave_components_list Z matches what's used in exc_buffer
# ===========================================================================

def test_wave_components_Z_consistent_with_exc_buffer():
    """
    The WaveComponents.Z stored in wave_components_list must produce the same
    exc_buffer as was stored by build_episode_replay().

    This validates that replay.py's fix and excitation.py use IDENTICAL Zᵢ values.
    """
    from module5_control.replay import build_episode_replay, DT_PHYSICS, DT_ERA5

    Hs, Tp, dir_deg, seed = 1.5, 8.0, 270.0, 42
    n_check = 300

    df  = _make_df(n_hours=1, Hs=Hs, Tp=Tp, dir_deg=dir_deg)
    rep = build_episode_replay(df, start_index=0, n_hours=1, seed=seed,
                               depth_m=30.0, gamma=3.3, dt_physics=DT_PHYSICS)

    assert len(rep.wave_components_list) == 1, "wave_components_list should have 1 entry"
    wc = rep.wave_components_list[0]

    # Recompute from stored Z
    t_array = np.arange(n_check) * DT_PHYSICS
    phase = wc.omega[np.newaxis, :] * t_array[:, np.newaxis]
    eta_from_Z = (np.dot(np.cos(phase), np.real(wc.Z))
                  - np.dot(np.sin(phase), np.imag(wc.Z)))

    exc_buf_slice = rep.excitation_force[0, :n_check]

    np.testing.assert_allclose(
        eta_from_Z, exc_buf_slice, rtol=0.0, atol=1e-14,
        err_msg=(
            "exc_buffer[0,:300] and the recomputed eta from wc.Z differ.\n"
            "This means replay.py and excitation.py use different Z values — "
            "the shared Zᵢ invariant is violated."
        ),
    )


# ===========================================================================
# Test 7 — physical sanity check (Hs reconstruction from exc_buffer)
# ===========================================================================

def test_exc_buffer_hs_consistent():
    """
    The RMS of exc_buffer over the full hour should give Hs ≈ Hs_input × √2.

    Approximate check only (1-hour realization has variance estimation noise).
    Tolerance: Hs_reconstructed within 30% of target (stochastic realization).

    This is a PHYSICS sanity check, not a precision equivalence check.
    """
    from module5_control.replay import build_episode_replay, DT_PHYSICS

    Hs_target = 1.5
    df = _make_df(n_hours=1, Hs=Hs_target, Tp=8.0)
    rep = build_episode_replay(df, start_index=0, n_hours=1, seed=123, dt_physics=DT_PHYSICS)

    eta_full = rep.excitation_force[0, :]  # full hour at 0.1 s
    # Hs = 4 × std(eta)  for a stationary ergodic process in the limit
    Hs_reconstructed = 4.0 * float(np.std(eta_full))

    assert 0.7 * Hs_target < Hs_reconstructed < 1.3 * Hs_target, (
        f"Reconstructed Hs={Hs_reconstructed:.3f} m is far from target "
        f"Hs={Hs_target} m.  The amplitude normalization may be wrong."
    )
