"""
tests/test_module1_validation.py — Regression tests for Module 1 validation.

Tests the entire physics chain with quantitative tolerances.
Run with:  pytest tests/test_module1_validation.py -v
"""

import numpy as np
import pytest

from module1_ocean.spectrum import (
    jonswap, pierson_moskowitz,
    significant_wave_height_from_spectrum, peak_period,
)
from module1_ocean.waves import frequency_grid, significant_wave_height_from_timeseries
from module1_ocean.direction import directional_grid, directional_weights, cosine_squared_spreading
from module1_ocean.dispersion import (
    solve_wave_number, dispersion_residual, phase_velocity, group_velocity,
)
from module1_ocean.goa_seastate import goa_seastate

G = 9.81

# ---------------------------------------------------------------------------
# 1. Spectral Hs reconstruction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Hs,Tp", [(0.5, 6.0), (1.0, 10.0), (1.5, 14.0)])
def test_spectral_hs_within_tolerance(Hs, Tp):
    """Spectral Hs must match requested Hs within 0.1% on a fine grid."""
    f = frequency_grid(0.01, 2.0, 2000)
    S = jonswap(f, Hs, Tp, gamma=3.3)
    Hs_spec = significant_wave_height_from_spectrum(f, S)
    assert abs(Hs_spec - Hs) / Hs < 1e-3, (
        f"Hs={Hs}: spectral Hs={Hs_spec:.6f}, rel_err={abs(Hs_spec-Hs)/Hs:.2e}"
    )


@pytest.mark.parametrize("Hs,Tp", [(0.5, 6.0), (1.0, 10.0), (1.5, 14.0)])
def test_spectral_tp_within_grid_resolution(Hs, Tp):
    """
    Spectral Tp error is induced by frequency-grid discretization.

    The spectral peak is at fp=1/Tp analytically, but the numerical grid
    can only resolve it to the nearest grid point.  The worst-case error
    is |1/(fp - df/2) - Tp| / Tp, which for df=0.001Hz and Tp=6s is ~0.28%.
    The measured errors (0.22-0.41%) are within this bound.

    Tolerance set to 1% to accommodate the grid-induced error without
    artificially tightening or loosening it.
    """
    f = frequency_grid(0.01, 2.0, 2000)
    df = f[1] - f[0]
    S = jonswap(f, Hs, Tp, gamma=3.3)
    Tp_spec = peak_period(f, S)
    fp_target = 1.0 / Tp
    # Worst-case grid-induced Tp error
    Tp_worst = 1.0 / (fp_target - df / 2.0)
    Tp_err_bound = abs(Tp_worst - Tp) / Tp
    Tp_err_actual = abs(Tp_spec - Tp) / Tp
    assert Tp_err_actual <= Tp_err_bound * 1.05, (
        f"Tp={Tp}: actual err={Tp_err_actual:.4f} exceeds grid bound {Tp_err_bound:.4f}"
    )
    assert Tp_err_actual < 0.01, (
        f"Tp={Tp}: spectral Tp={Tp_spec:.4f}, rel_err={Tp_err_actual:.2e} > 1%"
    )


# ---------------------------------------------------------------------------
# 2. JONSWAP / PM
# ---------------------------------------------------------------------------

def test_jonswap_gamma1_equals_pm_exactly():
    """gamma=1 must produce S_J == S_PM to floating-point equality."""
    f = frequency_grid(0.01, 2.0, 2000)
    S_pm = pierson_moskowitz(f, 1.0, 10.0)
    S_j1 = jonswap(f, 1.0, 10.0, gamma=1.0)
    np.testing.assert_array_equal(S_j1, S_pm)


def test_jonswap_energy_conservation_explicit():
    """
    Report m0_PM, m0_JONSWAP_raw, m0_JONSWAP_normalized.
    Normalized m0 must equal m0_PM to within 0.1%.
    """
    f = frequency_grid(0.01, 2.0, 2000)
    Hs, Tp = 1.0, 10.0
    S_pm = pierson_moskowitz(f, Hs, Tp)
    S_j = jonswap(f, Hs, Tp, gamma=3.3)
    m0_pm = float(np.trapezoid(S_pm, f))
    m0_j  = float(np.trapezoid(S_j, f))
    assert abs(m0_j - m0_pm) / m0_pm < 1e-3, (
        f"m0_PM={m0_pm:.6f}, m0_J={m0_j:.6f}, rel_err={abs(m0_j-m0_pm)/m0_pm:.2e}"
    )


def test_jonswap_gamma33_peak_exceeds_pm():
    """gamma=3.3 must produce higher spectral density at fp than PM."""
    f = frequency_grid(0.01, 2.0, 2000)
    Hs, Tp = 1.0, 10.0
    S_pm = pierson_moskowitz(f, Hs, Tp)
    S_j  = jonswap(f, Hs, Tp, gamma=3.3)
    fp_idx = np.argmin(np.abs(f - 1.0/Tp))
    assert S_j[fp_idx] > S_pm[fp_idx]


# ---------------------------------------------------------------------------
# 3. Directional energy — circular statistics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg", [0, 45, 90, 180, 225, 270, 315])
def test_directional_weights_sum_to_one_fine_grid(deg):
    """sum(w_j) must equal 1 to machine precision for all test directions."""
    theta = directional_grid(360)
    w = directional_weights(theta, np.deg2rad(deg))
    assert abs(w.sum() - 1.0) < 1e-12, f"dir={deg}: sum(w)={w.sum():.15f}"


@pytest.mark.parametrize("deg", [0, 45, 90, 180, 225, 270, 315])
def test_circular_mean_matches_requested_direction(deg):
    """
    Circular mean of directional weights must match requested direction.
    Uses weighted circular statistics: theta_mean = atan2(sum(w*sin), sum(w*cos)).
    NOT arithmetic mean.
    """
    theta = directional_grid(360)
    w = directional_weights(theta, np.deg2rad(deg))
    C = float(np.sum(w * np.cos(theta)))
    S = float(np.sum(w * np.sin(theta)))
    circ_mean_deg = np.rad2deg(np.arctan2(S, C)) % 360.0
    diff = abs(((circ_mean_deg - deg) + 180) % 360 - 180)
    assert diff < 0.01, (
        f"dir={deg}: circular mean={circ_mean_deg:.4f}, diff={diff:.4f} deg"
    )


# ---------------------------------------------------------------------------
# 4. Direction degree/radian API safety
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("deg,expected_rad", [
    (0,   0.0),
    (90,  np.pi / 2),
    (180, np.pi),
    (270, 3 * np.pi / 2),
])
def test_goa_api_direction_conversion(deg, expected_rad):
    """goa_seastate must convert direction_deg to radians exactly once."""
    result = goa_seastate(1.0, 10.0, float(deg), 50.0,
                          duration_s=60.0, dt=0.5, seed=0)
    assert abs(result.direction_rad - expected_rad) < 1e-12, (
        f"{deg} deg -> {result.direction_rad:.8f} rad, expected {expected_rad:.8f}"
    )


def test_direction_rad_in_valid_range():
    """direction_rad must be in [0, 2*pi) — confirms degrees were not passed raw."""
    for deg in [0, 90, 180, 270]:
        result = goa_seastate(1.0, 10.0, float(deg), 50.0,
                              duration_s=60.0, dt=0.5, seed=0)
        assert 0.0 <= result.direction_rad < 2 * np.pi, (
            f"direction_rad={result.direction_rad:.4f} outside [0, 2pi) for deg={deg}"
        )


def test_passing_degrees_as_radians_detectable():
    """
    270 degrees passed as radians = 270 rad >> 2*pi.
    The internal direction_rad for direction_deg=270 must be ~4.71 rad,
    NOT 270 rad.  This documents the API contract explicitly.
    """
    result = goa_seastate(1.0, 10.0, 270.0, 50.0,
                          duration_s=60.0, dt=0.5, seed=0)
    # If degrees were accidentally used as radians, direction_rad would be ~270
    assert result.direction_rad < 2 * np.pi, (
        "direction_rad >= 2*pi: degrees may have been passed as radians"
    )
    assert abs(result.direction_rad - 3 * np.pi / 2) < 1e-10


# ---------------------------------------------------------------------------
# 5. Dispersion validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("depth", [5.0, 20.0, 50.0, 200.0])
def test_dispersion_residual_below_1e10(depth):
    """omega^2 = g*k*tanh(k*d) must hold to relative precision < 1e-10."""
    f = np.linspace(0.05, 0.5, 20)
    k = solve_wave_number(f, depth_m=depth)
    R = dispersion_residual(f, k, depth_m=depth)
    omega_sq = (2 * np.pi * f) ** 2
    rel_res = np.abs(R) / omega_sq
    assert rel_res.max() < 1e-10, (
        f"depth={depth}: max rel residual={rel_res.max():.2e}"
    )


def test_deep_water_limit():
    """At d=200m, f=0.1Hz: k must agree with omega^2/g to within 0.01%."""
    f0, d = 0.1, 200.0
    k_fd = float(solve_wave_number(f0, depth_m=d))
    k_dw = (2 * np.pi * f0) ** 2 / G
    assert abs(k_fd - k_dw) / k_dw < 1e-4


def test_shallow_water_phase_velocity():
    """At d=0.5m, f=0.05Hz: phase velocity must be within 0.1% of sqrt(g*d)."""
    f0, d = 0.05, 0.5
    k = float(solve_wave_number(f0, depth_m=d))
    c = float(phase_velocity(f0, k))
    c_sw = np.sqrt(G * d)
    assert abs(c - c_sw) / c_sw < 1e-3


def test_shallow_water_group_velocity_approaches_phase():
    """In shallow water, Cg/c must be > 0.99 (approaches 1)."""
    f0, d = 0.05, 0.5
    k = float(solve_wave_number(f0, depth_m=d))
    c  = float(phase_velocity(f0, k))
    cg = float(group_velocity(f0, k, depth_m=d))
    assert cg / c > 0.99


# ---------------------------------------------------------------------------
# 6. Spatial propagation — analytical phase verification
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta_deg,dx,dy", [
    (0,   0.0,  1.0),   # North: crest moves in +y
    (90,  1.0,  0.0),   # East:  crest moves in +x
    (180, 0.0, -1.0),   # South: crest moves in -y
    (270, -1.0, 0.0),   # West:  crest moves in -x
])
def test_crest_tracks_in_propagation_direction(theta_deg, dx, dy):
    """
    For a single-frequency wave, the crest at t=0 at origin must remain
    a crest at t=dt at position (dx*c*dt, dy*c*dt).
    Verifies the spatial phase formula: -k(x*sin(theta) + y*cos(theta)).
    """
    f0 = 0.1
    depth = 50.0
    k0 = float(solve_wave_number(f0, depth_m=depth))
    omega0 = 2 * np.pi * f0
    c = omega0 / k0
    theta = np.deg2rad(theta_deg)
    dt = 1.0
    x_c = dx * c * dt
    y_c = dy * c * dt
    eta_crest = np.cos(omega0 * dt - k0 * (x_c * np.sin(theta) + y_c * np.cos(theta)))
    assert abs(eta_crest - 1.0) < 1e-12, (
        f"theta={theta_deg}: crest eta={eta_crest:.8f}, expected 1.0"
    )


@pytest.mark.parametrize("theta_deg,perp_x,perp_y", [
    (0,   50.0, 0.0),   # North: perpendicular is East (x)
    (90,  0.0, 50.0),   # East:  perpendicular is North (y)
    (180, 50.0, 0.0),   # South: perpendicular is East (x)
    (270, 0.0, 50.0),   # West:  perpendicular is North (y)
])
def test_perpendicular_displacement_has_no_phase_effect(theta_deg, perp_x, perp_y):
    """
    Displacement perpendicular to propagation direction must not change phase.
    For North (theta=0): sin(0)=0, so x has no effect.
    For East (theta=pi/2): cos(pi/2)=0, so y has no effect.
    """
    f0 = 0.1
    depth = 50.0
    k0 = float(solve_wave_number(f0, depth_m=depth))
    theta = np.deg2rad(theta_deg)
    phase_origin = -k0 * (0 * np.sin(theta) + 0 * np.cos(theta))
    phase_perp   = -k0 * (perp_x * np.sin(theta) + perp_y * np.cos(theta))
    assert abs(phase_perp - phase_origin) < 1e-10, (
        f"theta={theta_deg}: perp phase diff={abs(phase_perp-phase_origin):.2e}"
    )


@pytest.mark.parametrize("theta_deg,dx,dy", [
    (0,   0.0,  1.0),
    (90,  1.0,  0.0),
    (180, 0.0, -1.0),
    (270, -1.0, 0.0),
])
def test_numerical_propagation_speed(theta_deg, dx, dy):
    """
    Numerical propagation-speed test.

    For eta = cos(omega*t - k*(x*sin(th) + y*cos(th))), track the crest
    position along the propagation direction at multiple times by scanning
    a window of width lambda/2 centred on the expected crest position c*t.
    Fit s_crest(t) = c_measured * t + intercept and verify c_measured
    matches c_theory = omega/k to within 0.1%.

    The window must be narrower than one wavelength to avoid aliasing
    (multiple crests in the scan range).
    """
    f0 = 0.1
    depth = 50.0
    k0 = float(solve_wave_number(f0, depth_m=depth))
    omega0 = 2 * np.pi * f0
    c_theory = omega0 / k0
    lam = 2.0 * np.pi / k0
    theta = np.deg2rad(theta_deg)

    n_times = 20
    n_space = 500   # points per window
    half_win = lam * 0.4   # window half-width < lambda/2 to avoid aliasing
    t_arr = np.linspace(0.5 / f0, 2.0 / f0, n_times)  # start after t=0 to avoid s=0 ambiguity

    crest_positions = []
    for t_val in t_arr:
        s_centre = c_theory * t_val   # expected crest position
        s_win = np.linspace(s_centre - half_win, s_centre + half_win, n_space)
        x_line = dx * s_win
        y_line = dy * s_win
        eta_line = np.cos(
            omega0 * t_val
            - k0 * (x_line * np.sin(theta) + y_line * np.cos(theta))
        )
        crest_positions.append(s_win[np.argmax(eta_line)])

    crest_positions = np.array(crest_positions)
    A = np.column_stack([t_arr, np.ones_like(t_arr)])
    coeffs, _, _, _ = np.linalg.lstsq(A, crest_positions, rcond=None)
    c_measured = float(coeffs[0])
    rel_err = abs(c_measured - c_theory) / c_theory
    assert rel_err < 1e-3, (
        f"theta={theta_deg}: c_theory={c_theory:.4f}, "
        f"c_measured={c_measured:.4f}, rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 7. Spatial energy conservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Hs,Tp,n_dir", [
    (0.5,  6.0,  8),
    (1.0, 10.0, 16),
    (1.5, 14.0, 32),
])
def test_spatial_energy_conservation(Hs, Tp, n_dir):
    """
    sum_ij(a_ij^2 / 2) must equal sum_i(S_i * df) to machine precision.
    Verifies that directional spreading does not change total variance.
    """
    f = frequency_grid(0.02, 0.5, 128)
    S_f = jonswap(f, Hs, Tp, gamma=3.3)
    df = f[1] - f[0]
    spectral_var = float(np.sum(S_f * df))
    theta = directional_grid(n_dir)
    w = directional_weights(theta, np.deg2rad(270.0))
    S_2d = S_f[:, np.newaxis] * w[np.newaxis, :]
    a = np.sqrt(2.0 * S_2d * df)
    synthesis_var = float(np.sum(a**2 / 2.0))
    rel_err = abs(synthesis_var - spectral_var) / spectral_var
    assert rel_err < 1e-12, (
        f"Hs={Hs},Tp={Tp},n_dir={n_dir}: rel_err={rel_err:.2e}"
    )


# ---------------------------------------------------------------------------
# 8. Random-realization statistics (statistical, not deterministic)
# ---------------------------------------------------------------------------

def test_hs_convergence_mean_close_to_spectral():
    """
    Mean Hs over 50 seeds for 3600s must be within 1% of spectral Hs.
    This is a statistical test — individual realizations may deviate more.
    """
    from module1_ocean.waves import synthesize_surface_elevation
    Hs_target, Tp = 1.0, 10.0
    f = frequency_grid(0.02, 0.5, 128)
    S_f = jonswap(f, Hs_target, Tp, gamma=3.3)
    Hs_spectral = significant_wave_height_from_spectrum(f, S_f)
    t = np.arange(0.0, 3600.0, 0.5)
    hs_vals = [
        significant_wave_height_from_timeseries(
            synthesize_surface_elevation(t, f, S_f, seed=s)
        )
        for s in range(50)
    ]
    mean_hs = np.mean(hs_vals)
    assert abs(mean_hs - Hs_spectral) / Hs_spectral < 0.01, (
        f"Mean Hs over 50 seeds = {mean_hs:.4f}, spectral = {Hs_spectral:.4f}"
    )


def test_hs_std_decreases_with_duration():
    """
    std(Hs_est) must decrease as duration increases.
    Tests convergence property: longer records -> better Hs estimates.
    """
    from module1_ocean.waves import synthesize_surface_elevation
    Hs_target, Tp = 1.0, 10.0
    f = frequency_grid(0.02, 0.5, 128)
    S_f = jonswap(f, Hs_target, Tp, gamma=3.3)
    n_seeds = 30
    stds = []
    for dur in [300.0, 3600.0]:
        t = np.arange(0.0, dur, 0.5)
        hs_vals = [
            significant_wave_height_from_timeseries(
                synthesize_surface_elevation(t, f, S_f, seed=s)
            )
            for s in range(n_seeds)
        ]
        stds.append(np.std(hs_vals))
    assert stds[1] < stds[0], (
        f"std(Hs) did not decrease: 300s={stds[0]:.4f}, 3600s={stds[1]:.4f}"
    )


# ---------------------------------------------------------------------------
# 9. ERA5 replay validation
# ---------------------------------------------------------------------------

import os
import pandas as pd

ERA5_FILE = "data/raw/era5_goa_jan2024.nc"


@pytest.fixture(scope="module")
def era5_replay():
    """Load ERA5 and run a 6-segment replay. Skip if file absent."""
    if not os.path.exists(ERA5_FILE):
        pytest.skip(f"ERA5 file not found: {ERA5_FILE}")
    from module1_ocean.era5 import load_era5, extract_goa_timeseries
    from module1_ocean.goa_seastate import generate_goa_replay
    ds = load_era5(ERA5_FILE)
    df = extract_goa_timeseries(ds)
    df_sub = df.head(6).copy()
    replay = generate_goa_replay(df_sub, depth_m=50.0, base_seed=42,
                                  duration_s=600.0, dt=0.5)
    return replay, df_sub


def test_era5_direction_conversion_no_pi_offset(era5_replay):
    """direction_rad must equal np.deg2rad(direction_deg) with NO pi offset."""
    replay, df_sub = era5_replay
    for i, seg in enumerate(replay.segments):
        expected = np.deg2rad(replay.era5_direction_deg[i])
        assert abs(seg.direction_rad - expected) < 1e-10, (
            f"Segment {i}: direction_rad={seg.direction_rad:.6f}, "
            f"expected={expected:.6f}"
        )


def test_era5_timestamps_preserved(era5_replay):
    """ERA5 timestamps must pass through the replay pipeline unchanged."""
    replay, df_sub = era5_replay
    for i, ts in enumerate(replay.era5_timestamps):
        assert ts == df_sub["time"].iloc[i]


def test_era5_same_seed_reproducible(era5_replay):
    """Fixed base_seed must produce identical segment realizations."""
    replay, df_sub = era5_replay
    from module1_ocean.goa_seastate import generate_goa_replay
    replay2 = generate_goa_replay(df_sub, depth_m=50.0, base_seed=42,
                                   duration_s=600.0, dt=0.5)
    for s1, s2 in zip(replay.segments, replay2.segments):
        np.testing.assert_array_equal(s1.eta, s2.eta)


def test_era5_different_seed_differs(era5_replay):
    """Different base_seed must produce different realizations."""
    replay, df_sub = era5_replay
    from module1_ocean.goa_seastate import generate_goa_replay
    replay3 = generate_goa_replay(df_sub, depth_m=50.0, base_seed=99,
                                   duration_s=600.0, dt=0.5)
    assert not np.array_equal(replay.segments[0].eta, replay3.segments[0].eta)


def test_era5_spectral_hs_close_to_input(era5_replay):
    """Spectral Hs for each segment must be within 0.1% of ERA5 input Hs."""
    replay, _ = era5_replay
    for i, seg in enumerate(replay.segments):
        Hs_in = replay.era5_Hs[i]
        rel_err = abs(seg.Hs_spectral - Hs_in) / Hs_in
        assert rel_err < 1e-3, (
            f"Segment {i}: Hs_in={Hs_in:.4f}, Hs_spec={seg.Hs_spectral:.4f}, "
            f"rel_err={rel_err:.2e}"
        )


def test_era5_no_double_conversion(era5_replay):
    """
    direction_rad must be in [0, 2*pi), confirming no double conversion.
    If direction_deg were converted twice (deg->rad->rad), the result
    would be a tiny angle near 0, not the expected value.
    """
    replay, _ = era5_replay
    for i, seg in enumerate(replay.segments):
        assert 0.0 <= seg.direction_rad < 2 * np.pi, (
            f"Segment {i}: direction_rad={seg.direction_rad:.4f} outside [0,2pi)"
        )


# ---------------------------------------------------------------------------
# 10. Polar plot convention regression test
# ---------------------------------------------------------------------------

def test_polar_plot_convention_north_zero():
    """
    Verify that the polar plot helper applies N=0, clockwise convention.
    The directional spreading lobe for mean_dir=270 deg must peak at
    theta=3*pi/2 rad (West), not at any other angle.
    """
    theta = directional_grid(360)
    D = cosine_squared_spreading(theta, np.deg2rad(270.0))
    peak_idx = np.argmax(D)
    peak_deg = np.rad2deg(theta[peak_idx])
    assert abs(peak_deg - 270.0) < 1.0, (
        f"Peak of D at {peak_deg:.2f} deg, expected 270 deg (West)"
    )


def test_polar_plot_convention_all_cardinal():
    """
    For each cardinal direction, the spreading lobe must peak at the
    correct angle in the project convention (N=0, E=90, S=180, W=270).
    """
    theta = directional_grid(360)
    for deg in [0, 90, 180, 270]:
        D = cosine_squared_spreading(theta, np.deg2rad(deg))
        peak_deg = np.rad2deg(theta[np.argmax(D)])
        diff = abs(((peak_deg - deg) + 180) % 360 - 180)
        assert diff < 1.0, (
            f"dir={deg}: peak at {peak_deg:.2f} deg, diff={diff:.2f} deg"
        )


def test_matplotlib_polar_convention_labels():
    """
    Verify that set_theta_zero_location('N') + set_theta_direction(-1)
    produces tick labels [0, 90, 180, 270] at [N, E, S, W] positions.
    This is a regression test for the visualization convention.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids([0, 90, 180, 270],
                      labels=["N\n0°", "E\n90°", "S\n180°", "W\n270°"])
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert "N\n0°" in labels
    assert "E\n90°" in labels
    assert "S\n180°" in labels
    assert "W\n270°" in labels
    plt.close(fig)

