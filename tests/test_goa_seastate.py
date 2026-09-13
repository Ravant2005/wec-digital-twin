"""
tests/test_goa_seastate.py — Tests for module1_ocean/goa_seastate.py

Run with:  pytest tests/test_goa_seastate.py -v
"""

import numpy as np
import pandas as pd
import pytest

from module1_ocean.goa_seastate import goa_seastate, generate_goa_replay, SeaStateResult
from module1_ocean.spectrum import peak_frequency

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

HS    = 0.74    # m  — representative Goa January mean
TP    = 10.0    # s
DIR   = 270.0   # degrees — westerly
DEPTH = 50.0    # m — demonstration depth, not a real Goa value
SEED  = 42

# Minimal ERA5-like DataFrame for replay tests
_HS_VALS  = [0.6, 0.7, 0.8, 0.75]
_TP_VALS  = [9.0, 10.0, 11.0, 10.5]
_DIR_VALS = [260.0, 270.0, 280.0, 265.0]

def _make_df(n=4):
    times = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "time":          times,
        "Hs_m":          _HS_VALS[:n],
        "Tp_s":          _TP_VALS[:n],
        "direction_deg": _DIR_VALS[:n],
    })


# ---------------------------------------------------------------------------
# ERA5 degrees → radians conversion
# ---------------------------------------------------------------------------

def test_direction_conversion_stored_correctly():
    """direction_rad must equal np.deg2rad(direction_deg)."""
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    assert result.direction_rad == pytest.approx(np.deg2rad(DIR), rel=1e-12)


def test_direction_conversion_north():
    """0° (North) must convert to 0 rad."""
    result = goa_seastate(HS, TP, 0.0, DEPTH, seed=SEED)
    assert result.direction_rad == pytest.approx(0.0, abs=1e-12)


def test_direction_conversion_east():
    """90° (East) must convert to π/2 rad."""
    result = goa_seastate(HS, TP, 90.0, DEPTH, seed=SEED)
    assert result.direction_rad == pytest.approx(np.pi / 2.0, rel=1e-12)


def test_direction_degrees_not_used_as_radians():
    """
    270° in radians would be ~15.7 rad — far outside [0, 2π).
    The internal direction_rad must be in [0, 2π), confirming degrees
    were converted and not passed raw.
    """
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    assert 0.0 <= result.direction_rad < 2.0 * np.pi


# ---------------------------------------------------------------------------
# Invalid input validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("Hs", [0.0, -1.0])
def test_invalid_hs_raises(Hs):
    with pytest.raises(ValueError, match="Hs"):
        goa_seastate(Hs, TP, DIR, DEPTH)


@pytest.mark.parametrize("Tp", [0.0, -5.0])
def test_invalid_tp_raises(Tp):
    with pytest.raises(ValueError, match="Tp"):
        goa_seastate(HS, Tp, DIR, DEPTH)


@pytest.mark.parametrize("depth", [0.0, -10.0])
def test_invalid_depth_raises(depth):
    with pytest.raises(ValueError, match="depth_m"):
        goa_seastate(HS, TP, DIR, depth)


def test_invalid_gamma_raises():
    with pytest.raises(ValueError, match="gamma"):
        goa_seastate(HS, TP, DIR, DEPTH, gamma=0.0)


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_result_is_seastate_result():
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    assert isinstance(result, SeaStateResult)


def test_eta_shape_matches_t():
    result = goa_seastate(HS, TP, DIR, DEPTH, duration_s=600.0, dt=0.5, seed=SEED)
    assert result.eta.shape == result.t.shape


def test_t_starts_at_zero():
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    assert result.t[0] == pytest.approx(0.0)


def test_t_step_matches_dt():
    dt = 0.25
    result = goa_seastate(HS, TP, DIR, DEPTH, dt=dt, duration_s=100.0, seed=SEED)
    diffs = np.diff(result.t)
    assert np.allclose(diffs, dt, rtol=1e-12)


def test_all_outputs_finite():
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    assert np.all(np.isfinite(result.eta))


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_same_seed_identical_eta():
    r1 = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    r2 = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    np.testing.assert_array_equal(r1.eta, r2.eta)


def test_different_seeds_different_eta():
    r1 = goa_seastate(HS, TP, DIR, DEPTH, seed=1)
    r2 = goa_seastate(HS, TP, DIR, DEPTH, seed=2)
    assert not np.array_equal(r1.eta, r2.eta)


# ---------------------------------------------------------------------------
# Hs reconstruction
# ---------------------------------------------------------------------------

def test_spectral_hs_close_to_requested():
    """Spectral Hs must be within 0.1% of requested Hs."""
    result = goa_seastate(HS, TP, DIR, DEPTH, seed=SEED)
    rel_err = abs(result.Hs_spectral - HS) / HS
    assert rel_err < 1e-3, f"Spectral Hs error {rel_err:.2e} > 0.1%"


def test_timeseries_hs_close_to_requested():
    """
    Time-series Hs (4·std) must be within 15% of requested Hs for a
    1-hour realization.  Tolerance accounts for random phase variance.
    """
    result = goa_seastate(HS, TP, DIR, DEPTH, duration_s=3600.0, seed=SEED)
    rel_err = abs(result.Hs_timeseries - HS) / HS
    assert rel_err < 0.15, (
        f"Time-series Hs={result.Hs_timeseries:.4f} m deviates {rel_err:.1%} "
        f"from requested Hs={HS} m"
    )


# ---------------------------------------------------------------------------
# Tp / spectral peak
# ---------------------------------------------------------------------------

def test_spectral_peak_close_to_tp():
    """
    The spectral peak frequency must be within one frequency bin of fp=1/Tp.
    """
    from module1_ocean.spectrum import jonswap
    from module1_ocean.waves import frequency_grid

    fp_target = 1.0 / TP
    f = frequency_grid(0.02, 0.5, 128)
    S = jonswap(f, HS, TP, gamma=3.3)
    fp_numerical = peak_frequency(f, S)
    df = f[1] - f[0]
    assert abs(fp_numerical - fp_target) <= df, (
        f"Spectral peak {fp_numerical:.5f} Hz deviates more than Δf={df:.5f} Hz "
        f"from fp={fp_target:.5f} Hz"
    )


# ---------------------------------------------------------------------------
# Directional normalization preserved
# ---------------------------------------------------------------------------

def test_directional_weights_sum_to_one():
    """Directional weights must sum to 1 for any ERA5 direction."""
    from module1_ocean.direction import directional_grid, directional_weights
    for dir_deg in [0.0, 90.0, 180.0, 270.0, 45.0, 315.0]:
        theta = directional_grid(16)
        w = directional_weights(theta, np.deg2rad(dir_deg))
        assert abs(w.sum() - 1.0) < 1e-12, f"Weights don't sum to 1 for dir={dir_deg}°"


# ---------------------------------------------------------------------------
# ERA5 replay
# ---------------------------------------------------------------------------

def test_replay_returns_correct_segment_count():
    df = _make_df(4)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    assert len(replay.segments) == 4


def test_replay_preserves_timestamps():
    df = _make_df(4)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    for i, ts in enumerate(replay.era5_timestamps):
        assert ts == df["time"].iloc[i]


def test_replay_preserves_hs_values():
    df = _make_df(4)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    np.testing.assert_array_equal(replay.era5_Hs, df["Hs_m"].values)


def test_replay_preserves_tp_values():
    df = _make_df(4)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    np.testing.assert_array_equal(replay.era5_Tp, df["Tp_s"].values)


def test_replay_preserves_direction_values():
    df = _make_df(4)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    np.testing.assert_array_equal(
        replay.era5_direction_deg, df["direction_deg"].values
    )


def test_replay_segments_are_deterministic():
    """Same base_seed must produce identical segments."""
    df = _make_df(3)
    r1 = generate_goa_replay(df, depth_m=DEPTH, base_seed=10, duration_s=600.0)
    r2 = generate_goa_replay(df, depth_m=DEPTH, base_seed=10, duration_s=600.0)
    for s1, s2 in zip(r1.segments, r2.segments):
        np.testing.assert_array_equal(s1.eta, s2.eta)


def test_replay_different_base_seeds_differ():
    df = _make_df(2)
    r1 = generate_goa_replay(df, depth_m=DEPTH, base_seed=1, duration_s=600.0)
    r2 = generate_goa_replay(df, depth_m=DEPTH, base_seed=99, duration_s=600.0)
    assert not np.array_equal(r1.segments[0].eta, r2.segments[0].eta)


def test_replay_skips_nan_rows():
    df = _make_df(3)
    df.loc[1, "Hs_m"] = float("nan")
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0, skip_nan=True)
    assert len(replay.segments) == 2


def test_replay_raises_on_nan_when_skip_false():
    df = _make_df(2)
    df.loc[0, "Tp_s"] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                             duration_s=600.0, skip_nan=False)


def test_replay_missing_column_raises():
    df = _make_df(2).drop(columns=["direction_deg"])
    with pytest.raises(ValueError, match="direction_deg"):
        generate_goa_replay(df, depth_m=DEPTH)


def test_replay_invalid_depth_raises():
    df = _make_df(2)
    with pytest.raises(ValueError, match="depth_m"):
        generate_goa_replay(df, depth_m=0.0)


def test_replay_direction_stored_in_degrees_and_radians():
    """Each segment must store direction in both degrees and radians."""
    df = _make_df(2)
    replay = generate_goa_replay(df, depth_m=DEPTH, base_seed=0,
                                  duration_s=600.0)
    for seg, dir_deg in zip(replay.segments, replay.era5_direction_deg):
        assert seg.direction_deg == pytest.approx(dir_deg)
        assert seg.direction_rad == pytest.approx(np.deg2rad(dir_deg), rel=1e-12)
