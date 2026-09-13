"""
test_module5_replay.py — Tests for module5_control.replay (Module 5.1).

Tests cover:
- EpisodeReplay construction and properties
- build_synthetic_replay() correctness and determinism
- build_episode_replay() from DataFrame
- ERA5/physics/control step index mapping
- Seed-based determinism
- Error handling for invalid inputs
"""

import numpy as np
import pandas as pd
import pytest

from module5_control.replay import (
    EpisodeReplay,
    build_episode_replay,
    build_synthetic_replay,
    DT_PHYSICS,
    DT_CONTROL,
    N_PHYSICS_PER_CONTROL,
    N_CONTROL_PER_HOUR,
    N_PHYSICS_PER_HOUR,
    DEPTH_GOA_M,
)


# ---------------------------------------------------------------------------
# Constants sanity check
# ---------------------------------------------------------------------------

def test_timestep_constants_consistent():
    """DT_CONTROL / DT_PHYSICS must equal N_PHYSICS_PER_CONTROL (integer)."""
    assert N_PHYSICS_PER_CONTROL == int(round(DT_CONTROL / DT_PHYSICS))
    assert abs(N_PHYSICS_PER_CONTROL * DT_PHYSICS - DT_CONTROL) < 1e-9


def test_n_physics_per_hour():
    """N_PHYSICS_PER_HOUR must be 3600 / DT_PHYSICS."""
    assert N_PHYSICS_PER_HOUR == int(round(3600.0 / DT_PHYSICS))


def test_n_control_per_hour():
    """N_CONTROL_PER_HOUR must be 3600 / DT_CONTROL."""
    assert N_CONTROL_PER_HOUR == int(round(3600.0 / DT_CONTROL))


def test_physics_dt_below_nyquist():
    """DT_PHYSICS must satisfy Nyquist: dt < pi / omega_max = pi / 1.4."""
    nyquist_limit = np.pi / 1.4
    assert DT_PHYSICS < nyquist_limit, (
        f"DT_PHYSICS={DT_PHYSICS} violates Nyquist limit {nyquist_limit:.4f} s"
    )


def test_control_dt_sensible():
    """DT_CONTROL must be > DT_PHYSICS and << typical wave period."""
    assert DT_CONTROL > DT_PHYSICS
    assert DT_CONTROL <= 5.0, "DT_CONTROL must be small relative to wave period"


# ---------------------------------------------------------------------------
# build_synthetic_replay
# ---------------------------------------------------------------------------

def test_build_synthetic_replay_returns_episode_replay():
    r = build_synthetic_replay(n_hours=2, seed=0)
    assert isinstance(r, EpisodeReplay)


def test_build_synthetic_replay_shapes():
    n = 3
    r = build_synthetic_replay(n_hours=n, seed=0)
    assert r.n_hours == n
    assert r.Hs_true.shape == (n,)
    assert r.Tp_true.shape == (n,)
    assert r.direction_true.shape == (n,)
    assert r.u10.shape == (n,)
    assert r.v10.shape == (n,)
    assert r.excitation_force.shape == (n, N_PHYSICS_PER_HOUR)
    assert len(r.timestamps) == n


def test_build_synthetic_replay_constant_seastate():
    """All hours should have the requested Hs and Tp."""
    Hs, Tp = 2.0, 10.0
    r = build_synthetic_replay(n_hours=2, Hs=Hs, Tp=Tp, seed=1)
    np.testing.assert_allclose(r.Hs_true, Hs)
    np.testing.assert_allclose(r.Tp_true, Tp)


def test_build_synthetic_replay_excitation_finite():
    """Excitation force buffer must be finite (no NaN / inf)."""
    r = build_synthetic_replay(n_hours=2, seed=0)
    assert np.all(np.isfinite(r.excitation_force))


def test_build_synthetic_replay_deterministic_same_seed():
    """Same seed → identical excitation force buffers."""
    r1 = build_synthetic_replay(n_hours=2, seed=99)
    r2 = build_synthetic_replay(n_hours=2, seed=99)
    np.testing.assert_array_equal(r1.excitation_force, r2.excitation_force)


def test_build_synthetic_replay_different_seeds_differ():
    """Different seeds → different excitation forces (with overwhelming probability)."""
    r1 = build_synthetic_replay(n_hours=2, seed=0)
    r2 = build_synthetic_replay(n_hours=2, seed=1)
    assert not np.allclose(r1.excitation_force, r2.excitation_force)


def test_build_synthetic_replay_depth_stored():
    depth = 25.0
    r = build_synthetic_replay(n_hours=1, seed=0, depth_m=depth)
    assert r.depth_m == depth


def test_build_synthetic_replay_dt_stored():
    r = build_synthetic_replay(n_hours=1, seed=0)
    assert r.dt_physics == DT_PHYSICS
    assert r.dt_control == DT_CONTROL


# ---------------------------------------------------------------------------
# EpisodeReplay derived properties
# ---------------------------------------------------------------------------

def test_n_control_steps():
    r = build_synthetic_replay(n_hours=3, seed=0)
    assert r.n_control_steps == 3 * N_CONTROL_PER_HOUR


def test_n_physics_steps():
    r = build_synthetic_replay(n_hours=3, seed=0)
    assert r.n_physics_steps == 3 * N_PHYSICS_PER_HOUR


def test_episode_duration_s():
    n = 4
    r = build_synthetic_replay(n_hours=n, seed=0)
    assert r.episode_duration_s == n * 3600.0


def test_get_excitation_at_physics_step_in_range():
    """Can retrieve excitation force at every physics step."""
    r = build_synthetic_replay(n_hours=2, seed=0)
    for step in [0, 1000, N_PHYSICS_PER_HOUR - 1, N_PHYSICS_PER_HOUR,
                 2 * N_PHYSICS_PER_HOUR - 1]:
        val = r.get_excitation_at_physics_step(step)
        assert np.isfinite(val)


def test_get_excitation_matches_buffer():
    """get_excitation_at_physics_step must match the raw buffer."""
    r = build_synthetic_replay(n_hours=2, seed=0)
    for hour in range(2):
        for sub in [0, 100, 999]:
            phys = hour * N_PHYSICS_PER_HOUR + sub
            val  = r.get_excitation_at_physics_step(phys)
            expected = float(r.excitation_force[hour, sub])
            assert val == expected


def test_get_era5_hour_index():
    r = build_synthetic_replay(n_hours=3, seed=0)
    assert r.get_era5_hour_index(0) == 0
    assert r.get_era5_hour_index(N_PHYSICS_PER_HOUR - 1) == 0
    assert r.get_era5_hour_index(N_PHYSICS_PER_HOUR) == 1
    assert r.get_era5_hour_index(2 * N_PHYSICS_PER_HOUR) == 2


def test_get_control_step_hour():
    r = build_synthetic_replay(n_hours=3, seed=0)
    assert r.get_control_step_hour(0) == 0
    assert r.get_control_step_hour(N_CONTROL_PER_HOUR - 1) == 0
    assert r.get_control_step_hour(N_CONTROL_PER_HOUR) == 1
    assert r.get_control_step_hour(2 * N_CONTROL_PER_HOUR) == 2


# ---------------------------------------------------------------------------
# build_episode_replay from DataFrame
# ---------------------------------------------------------------------------

def _make_df(n_hours: int) -> pd.DataFrame:
    """Helper: synthetic ERA5 DataFrame."""
    ts = [pd.Timestamp("2020-06-01") + pd.Timedelta(hours=i) for i in range(n_hours)]
    return pd.DataFrame({
        "time":          ts,
        "Hs_m":          [1.5] * n_hours,
        "Tp_s":          [8.0] * n_hours,
        "direction_deg": [270.0] * n_hours,
        "u10":           [5.0]  * n_hours,
        "v10":           [2.0]  * n_hours,
    })


def test_build_episode_replay_basic():
    df = _make_df(3)
    r  = build_episode_replay(df, start_index=0, n_hours=3, seed=42)
    assert isinstance(r, EpisodeReplay)
    assert r.n_hours == 3
    assert r.era5_start_index == 0
    assert r.seed == 42


def test_build_episode_replay_start_index():
    df = _make_df(5)
    r  = build_episode_replay(df, start_index=2, n_hours=2, seed=0)
    assert r.n_hours == 2
    assert r.era5_start_index == 2
    # Check Hs value is from the sliced rows
    np.testing.assert_allclose(r.Hs_true, 1.5)


def test_build_episode_replay_bad_start_raises():
    df = _make_df(3)
    with pytest.raises(ValueError, match="exceeds DataFrame length"):
        build_episode_replay(df, start_index=2, n_hours=3, seed=0)


def test_build_episode_replay_missing_columns_raises():
    df = pd.DataFrame({"time": [], "Hs_m": []})
    with pytest.raises(ValueError, match="missing required columns"):
        build_episode_replay(df, start_index=0, n_hours=1, seed=0)


def test_build_episode_replay_invalid_depth_raises():
    df = _make_df(2)
    with pytest.raises(ValueError):
        build_episode_replay(df, start_index=0, n_hours=2, seed=0, depth_m=-5.0)


def test_build_episode_replay_n_hours_zero_raises():
    df = _make_df(2)
    with pytest.raises(ValueError):
        build_episode_replay(df, start_index=0, n_hours=0, seed=0)


def test_build_episode_replay_nan_row_handled():
    """NaN rows should be handled gracefully (calm-sea fallback) and not crash."""
    df = _make_df(2)
    df.loc[1, "Hs_m"] = np.nan
    r = build_episode_replay(df, start_index=0, n_hours=2, seed=0)
    # Module 5.2A: NaN → calm-sea fallback (Hs=0.01), produces tiny but non-zero eta.
    # The important thing is no crash and excitation_force is finite.
    assert r.n_hours == 2
    assert np.all(np.isfinite(r.excitation_force[1, :]))
    # The calm-sea fallback amplitude is very small
    assert np.max(np.abs(r.excitation_force[1, :])) < 0.1


def test_build_episode_replay_deterministic():
    df = _make_df(3)
    r1 = build_episode_replay(df, start_index=0, n_hours=3, seed=7)
    r2 = build_episode_replay(df, start_index=0, n_hours=3, seed=7)
    np.testing.assert_array_equal(r1.excitation_force, r2.excitation_force)
