"""
test_module5_observations.py — Tests for module5_control.observations (Module 5.1).

Tests cover:
- ObservationConfig validation
- build_observation_space() Gymnasium Box properties
- ObservationBuilder.build() shape, dtype, finiteness
- NaN handling in history (must produce finite output)
- Normalisation via scaler (Hs, Tp z-scored)
- Direction encoding (sin/cos, no raw angles)
- Zero observation utility
- Observation dimension audit (exact 723 for wave_wind)
"""

import numpy as np
import pytest

from module5_control.observations import (
    ObservationConfig,
    ObservationBuilder,
    build_observation_space,
    H_OBS,
    H_FORE,
    F_OBS_WAVE_WIND,
    F_FORE,
    N_WEC,
    OBS_DIM,
    _BLOCK_A_START,
    _BLOCK_A_END,
    _BLOCK_B_START,
    _BLOCK_B_END,
    _BLOCK_C_START,
    _BLOCK_C_END,
)
from module4_forecasting.dataset_v2 import WaveScalerV2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_history(nan_frac: float = 0.0) -> np.ndarray:
    """Return a (H_OBS, F_OBS) history array, optionally with NaNs."""
    rng = np.random.default_rng(0)
    A   = np.zeros((H_OBS, F_OBS_WAVE_WIND))
    A[:, 0] = rng.uniform(0.5, 3.0, H_OBS)   # Hs
    A[:, 1] = rng.uniform(5.0, 14.0, H_OBS)  # Tp
    A[:, 2] = rng.uniform(-1, 1, H_OBS)       # sin_dir
    A[:, 3] = rng.uniform(-1, 1, H_OBS)       # cos_dir
    A[:, 4] = 1.0                              # Hs_valid
    A[:, 5] = 1.0                              # Tp_valid
    A[:, 6] = 1.0                              # dir_valid
    A[:, 7] = rng.uniform(-10, 10, H_OBS)     # u10
    A[:, 8] = rng.uniform(-10, 10, H_OBS)     # v10
    A[:, 9] = 1.0
    A[:, 10] = 1.0
    if nan_frac > 0:
        n_nan = int(nan_frac * H_OBS)
        idx   = rng.choice(H_OBS, n_nan, replace=False)
        A[idx, 0] = np.nan
        A[idx, 4] = 0.0
    return A


def _make_forecast() -> np.ndarray:
    rng = np.random.default_rng(1)
    B   = np.zeros((H_FORE, F_FORE))
    B[:, 0] = rng.uniform(0.5, 3.0, H_FORE)   # Hs
    B[:, 1] = rng.uniform(5.0, 14.0, H_FORE)  # Tp
    B[:, 2] = rng.uniform(-1, 1, H_FORE)       # sin_dir
    B[:, 3] = rng.uniform(-1, 1, H_FORE)       # cos_dir
    return B


def _make_fitted_scaler() -> WaveScalerV2:
    scaler = WaveScalerV2(feature_mode="wave_wind")
    rng   = np.random.default_rng(2)
    X     = rng.uniform(0, 5, (100, H_OBS, F_OBS_WAVE_WIND))
    X[..., 0] = rng.uniform(0.5, 3.0, (100, H_OBS))  # Hs
    X[..., 1] = rng.uniform(5.0, 14.0, (100, H_OBS)) # Tp
    X[..., 7] = rng.uniform(-15, 15, (100, H_OBS))   # u10
    X[..., 8] = rng.uniform(-15, 15, (100, H_OBS))   # v10
    scaler.fit(X)
    return scaler


# ---------------------------------------------------------------------------
# OBS_DIM audit
# ---------------------------------------------------------------------------

def test_obs_dim_exact_value():
    """Total obs dimension must be exactly 723 in wave_wind mode."""
    assert OBS_DIM == 723, f"Expected 723, got {OBS_DIM}"


def test_obs_dim_formula():
    """OBS_DIM = H_OBS*F_OBS + H_FORE*F_FORE + N_WEC."""
    expected = H_OBS * F_OBS_WAVE_WIND + H_FORE * F_FORE + N_WEC
    assert OBS_DIM == expected


def test_block_slices_non_overlapping():
    """Block A, B, C slices must be contiguous and non-overlapping."""
    assert _BLOCK_A_START == 0
    assert _BLOCK_A_END   == H_OBS * F_OBS_WAVE_WIND
    assert _BLOCK_B_START == _BLOCK_A_END
    assert _BLOCK_B_END   == _BLOCK_B_START + H_FORE * F_FORE
    assert _BLOCK_C_START == _BLOCK_B_END
    assert _BLOCK_C_END   == OBS_DIM


# ---------------------------------------------------------------------------
# ObservationConfig
# ---------------------------------------------------------------------------

def test_obs_config_defaults():
    cfg = ObservationConfig()
    assert cfg.obs_history_len == H_OBS
    assert cfg.forecast_len    == H_FORE
    assert cfg.use_wind        is True
    assert cfg.obs_dim         == OBS_DIM


def test_obs_config_wrong_history_raises():
    with pytest.raises(ValueError, match="obs_history_len"):
        ObservationConfig(obs_history_len=24)


def test_obs_config_wrong_forecast_raises():
    with pytest.raises(ValueError, match="forecast_len"):
        ObservationConfig(forecast_len=24)


def test_obs_config_bad_x_scale_raises():
    with pytest.raises(ValueError):
        ObservationConfig(x_scale=0.0)


def test_obs_config_bad_v_scale_raises():
    with pytest.raises(ValueError):
        ObservationConfig(v_scale=-1.0)


# ---------------------------------------------------------------------------
# build_observation_space
# ---------------------------------------------------------------------------

def test_build_observation_space_shape():
    import gymnasium as gym
    from gymnasium import spaces
    cfg   = ObservationConfig()
    space = build_observation_space(cfg)
    assert isinstance(space, spaces.Box)
    assert space.shape == (OBS_DIM,)


def test_build_observation_space_dtype():
    cfg   = ObservationConfig()
    space = build_observation_space(cfg)
    assert space.dtype == np.float32


def test_build_observation_space_unbounded():
    """Bounds should be ±inf so extreme sea states are not clipped."""
    cfg   = ObservationConfig()
    space = build_observation_space(cfg)
    assert np.all(space.low  == -np.inf)
    assert np.all(space.high ==  np.inf)


# ---------------------------------------------------------------------------
# ObservationBuilder.build()
# ---------------------------------------------------------------------------

def test_build_correct_shape():
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    obs     = builder.build(_make_history(), _make_forecast(), 0.1, 0.05, 0)
    assert obs.shape == (OBS_DIM,)


def test_build_dtype_float32():
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    obs     = builder.build(_make_history(), _make_forecast(), 0.0, 0.0, 0)
    assert obs.dtype == np.float32


def test_build_no_nan_clean_input():
    """No NaN in → no NaN out."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    obs     = builder.build(_make_history(nan_frac=0.0), _make_forecast(), 0.0, 0.0, 0)
    assert not np.any(np.isnan(obs))


def test_build_no_nan_with_missing_history():
    """NaN values in history must be replaced by 0 in the observation."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    obs     = builder.build(_make_history(nan_frac=0.3), _make_forecast(), 0.0, 0.0, 0)
    assert not np.any(np.isnan(obs))
    assert np.all(np.isfinite(obs))


def test_build_no_nan_all_missing_history():
    """All-NaN history must still produce finite observation."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    A       = np.full((H_OBS, F_OBS_WAVE_WIND), np.nan)
    obs     = builder.build(A, _make_forecast(), 0.0, 0.0, 0)
    assert not np.any(np.isnan(obs))
    assert np.all(np.isfinite(obs))


def test_build_reactive_forecast_is_zeros():
    """When forecast block is zeros (reactive mode) Block B should be zeros."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    B_zeros = np.zeros((H_FORE, F_FORE))
    obs     = builder.build(_make_history(), B_zeros, 0.0, 0.0, 0)
    block_b = obs[_BLOCK_B_START:_BLOCK_B_END]
    np.testing.assert_array_equal(block_b, 0.0)


def test_build_wec_state_in_block_c():
    """WEC state (x, v, latch) must appear in Block C."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    x_scale = cfg.x_scale
    v_scale = cfg.v_scale
    obs     = builder.build(_make_history(), _make_forecast(), x=1.0, v=0.5, latch_status=1)
    c       = obs[_BLOCK_C_START:_BLOCK_C_END]
    assert abs(float(c[0]) - 1.0 / x_scale) < 1e-5
    assert abs(float(c[1]) - 0.5 / v_scale) < 1e-5
    assert float(c[2]) == 1.0


def test_build_latch_off_in_block_c():
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    obs     = builder.build(_make_history(), _make_forecast(), x=0.0, v=0.0, latch_status=0)
    c       = obs[_BLOCK_C_START:_BLOCK_C_END]
    assert float(c[2]) == 0.0


def test_build_no_raw_angles():
    """Direction values in history block must be sin/cos, not raw degrees."""
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    A       = _make_history()
    # Set all sin/cos to known values; raw degree would be large (e.g. 270)
    A[:, 2] = 0.5   # sin
    A[:, 3] = 0.866 # cos
    obs     = builder.build(A, _make_forecast(), 0.0, 0.0, 0)
    # Block A should not contain values > 1 for direction slots (indices 2,3)
    block_a = obs[_BLOCK_A_START:_BLOCK_A_END].reshape(H_OBS, F_OBS_WAVE_WIND)
    assert np.all(np.abs(block_a[:, 2]) <= 2.0)  # sin bounded after z-score (no scale applied)
    assert np.all(np.abs(block_a[:, 3]) <= 2.0)  # cos bounded


def test_build_with_fitted_scaler_finite():
    """With a fitted scaler obs must remain finite."""
    cfg     = ObservationConfig()
    scaler  = _make_fitted_scaler()
    builder = ObservationBuilder(cfg, scaler=scaler)
    obs     = builder.build(_make_history(), _make_forecast(), 0.0, 0.0, 0)
    assert np.all(np.isfinite(obs))


def test_build_with_fitted_scaler_hs_normalised():
    """With scaler, Hs in Block A should be z-scored (not raw)."""
    cfg    = ObservationConfig()
    scaler = _make_fitted_scaler()
    stats  = scaler.fitted_stats
    hs_mean, hs_std = stats["mean"][0], stats["std"][0]
    builder = ObservationBuilder(cfg, scaler=scaler)
    A = _make_history()
    A[:, 0] = 1.5   # Hs = 1.5 m for all steps
    obs     = builder.build(A, _make_forecast(), 0.0, 0.0, 0)
    block_a = obs[_BLOCK_A_START:_BLOCK_A_END].reshape(H_OBS, F_OBS_WAVE_WIND)
    expected_hs_scaled = (1.5 - hs_mean) / hs_std
    np.testing.assert_allclose(block_a[:, 0], expected_hs_scaled, atol=1e-4)


def test_zero_observation_shape():
    cfg     = ObservationConfig()
    builder = ObservationBuilder(cfg, scaler=None)
    z       = builder.zero_observation()
    assert z.shape == (OBS_DIM,)
    assert z.dtype == np.float32
    np.testing.assert_array_equal(z, 0.0)


# ---------------------------------------------------------------------------
# Direction encoding consistency with Module 4
# ---------------------------------------------------------------------------

def test_direction_to_sincos_consistency():
    """direction_to_sincos must match Module 4's convention."""
    from module4_forecasting.dataset import direction_to_sincos as m4_sincos
    from module5_control.observations import ObservationBuilder

    dirs = np.array([0.0, 90.0, 180.0, 270.0, 360.0])
    s5, c5 = ObservationBuilder.direction_to_sincos(dirs)
    s4, c4 = m4_sincos(dirs)
    np.testing.assert_allclose(s5, s4, atol=1e-12)
    np.testing.assert_allclose(c5, c4, atol=1e-12)


def test_direction_0_deg():
    s, c = ObservationBuilder.direction_to_sincos(np.array([0.0]))
    np.testing.assert_allclose(s[0], 0.0, atol=1e-10)
    np.testing.assert_allclose(c[0], 1.0, atol=1e-10)


def test_direction_90_deg():
    s, c = ObservationBuilder.direction_to_sincos(np.array([90.0]))
    np.testing.assert_allclose(s[0], 1.0, atol=1e-10)
    np.testing.assert_allclose(c[0], 0.0, atol=1e-10)
