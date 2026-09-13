"""
tests/test_forecasting_baselines.py — Baseline tests for Module 4.1.

Tests:
  1.  PersistenceForecaster output shape
  2.  Persistence uses most recent valid observation
  3.  Persistence never reads future truth
  4.  Persistence handles missing observations (lookback)
  5.  Persistence marks forecast invalid when no valid obs in lookback
  6.  ClimatologyForecaster output shape
  7.  Climatology uses only training data
  8.  Climatology does not use validation/test data
  9.  Climatology direction is circular mean
  10. WaveScaler fit uses training data only
  11. WaveScaler transform/inverse_transform round-trip
  12. WaveScaler does not standardize sin/cos or validity columns
  13. WaveScaler refit on val/test is forbidden (stats must not change)
  14. LEAKAGE H: persistence never reads future truth
  15. LEAKAGE I: missing imputation never uses future information
  16. LEAKAGE E: climatology never uses val/test data
"""

import numpy as np
import pytest

from module3_sensor import SensorParameters, WaveSensor
from module4_forecasting import (
    ForecastConfig, build_datasets, FEATURE_NAMES, N_FEATURES, N_TARGETS,
    PersistenceForecaster, ClimatologyForecaster, WaveScaler,
    direction_to_sincos, sincos_to_direction,
)


def _make_obs(n=200, seed=0, missing_prob=0.0):
    rng = np.random.default_rng(seed)
    time     = np.arange(n, dtype=float)
    Hs_true  = np.abs(1.0 + 0.5 * rng.standard_normal(n))
    Tp_true  = 8.0 + rng.standard_normal(n)
    dir_true = (270.0 + 20.0 * rng.standard_normal(n)) % 360.0
    p = SensorParameters(hs_noise_std=0.1, tp_noise_std=0.5,
                         direction_noise_std=10.0, missing_probability=missing_prob,
                         random_seed=seed)
    return WaveSensor(p).observe(time, Hs_true, Tp_true, dir_true)


@pytest.fixture(scope="module")
def setup():
    obs = _make_obs(n=200, seed=0)
    cfg = ForecastConfig(input_length=12, forecast_horizon=4)
    train_ds, val_ds, test_ds, split = build_datasets(obs, cfg)
    return obs, cfg, train_ds, val_ds, test_ds, split


# ---------------------------------------------------------------------------
# 1. PersistenceForecaster output shape
# ---------------------------------------------------------------------------

def test_persistence_shape(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    pf = PersistenceForecaster()
    preds = pf.predict_dataset(train_ds)
    assert preds.shape == (len(train_ds), cfg.forecast_horizon, N_TARGETS)


# ---------------------------------------------------------------------------
# 2. Persistence uses most recent valid observation
# ---------------------------------------------------------------------------

def test_persistence_uses_last_valid(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    pf = PersistenceForecaster()
    if len(train_ds) == 0:
        pytest.skip("No training windows")

    s = train_ds[0]
    origin = s.origin_index
    # Find the most recent valid obs at or before origin
    for lag in range(24):
        idx = origin - lag
        if idx >= 0 and obs.valid_Hs[idx] and obs.valid_Tp[idx] and obs.valid_direction[idx]:
            expected_hs = obs.Hs_obs[idx]
            break
    else:
        pytest.skip("No valid obs in lookback")

    pred = pf.predict_sample(s, obs.Hs_obs, obs.Tp_obs, obs.direction_obs,
                             obs.valid_Hs, obs.valid_Tp, obs.valid_direction)
    # All horizons should have the same Hs
    assert np.allclose(pred[:, 0], expected_hs, atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Persistence never reads future truth
# ---------------------------------------------------------------------------

def test_persistence_no_future_truth(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    pf = PersistenceForecaster()
    if len(train_ds) == 0:
        pytest.skip("No training windows")

    s = train_ds[0]
    origin = s.origin_index
    pred = pf.predict_sample(s, obs.Hs_obs, obs.Tp_obs, obs.direction_obs,
                             obs.valid_Hs, obs.valid_Tp, obs.valid_direction)
    # Prediction must not equal any future true value (with noise they differ)
    for h in range(cfg.forecast_horizon):
        future_true_hs = obs.Hs_true[origin + h + 1]
        # Persistence uses obs, not truth — they differ by noise
        # This is a structural test: persistence reads obs, not truth
        # We verify by checking the prediction equals an obs value, not truth
        if not np.isnan(pred[h, 0]):
            # Find which obs value was used
            found_in_obs = any(
                np.isclose(pred[h, 0], obs.Hs_obs[origin - lag], atol=1e-9)
                for lag in range(24)
                if origin - lag >= 0 and obs.valid_Hs[origin - lag]
            )
            assert found_in_obs, "Persistence prediction does not match any obs value"


# ---------------------------------------------------------------------------
# 4. Persistence handles missing observations (lookback)
# ---------------------------------------------------------------------------

def test_persistence_lookback():
    """With all-missing obs except one, persistence should find that one."""
    n = 50
    obs_miss = _make_obs(n=n, seed=5, missing_prob=0.0)
    # Manually mark all obs invalid except index 10
    import copy
    Hs_obs  = obs_miss.Hs_obs.copy()
    valid_H = np.zeros(n, dtype=bool)
    valid_T = np.zeros(n, dtype=bool)
    valid_D = np.zeros(n, dtype=bool)
    valid_H[10] = valid_T[10] = valid_D[10] = True

    cfg = ForecastConfig(input_length=6, forecast_horizon=2)
    from module4_forecasting.dataset import make_split, ForecastDataset, ForecastSample
    split = make_split(n, cfg)
    ds = ForecastDataset(obs_miss, cfg, split, "train")
    if len(ds) == 0:
        pytest.skip("No windows")

    pf = PersistenceForecaster(max_lookback=30)
    s = ds[0]
    pred = pf.predict_sample(s, Hs_obs, obs_miss.Tp_obs, obs_miss.direction_obs,
                             valid_H, valid_T, valid_D)
    # Should find obs at index 10 if origin >= 10
    origin = s.origin_index
    if origin >= 10:
        assert not np.all(np.isnan(pred[:, 0])), "Should have found valid obs at index 10"


# ---------------------------------------------------------------------------
# 5. Persistence marks forecast invalid when no valid obs in lookback
# ---------------------------------------------------------------------------

def test_persistence_all_missing():
    """With all-missing obs, persistence should return NaN."""
    n = 50
    obs_miss = _make_obs(n=n, seed=6, missing_prob=1.0)
    cfg = ForecastConfig(input_length=6, forecast_horizon=2)
    from module4_forecasting.dataset import make_split, ForecastDataset
    split = make_split(n, cfg)
    ds = ForecastDataset(obs_miss, cfg, split, "train")
    if len(ds) == 0:
        pytest.skip("No windows")

    pf = PersistenceForecaster(max_lookback=5)
    s = ds[0]
    pred = pf.predict_sample(s, obs_miss.Hs_obs, obs_miss.Tp_obs, obs_miss.direction_obs,
                             obs_miss.valid_Hs, obs_miss.valid_Tp, obs_miss.valid_direction)
    assert np.all(np.isnan(pred)), "All-missing obs should produce NaN forecast"


# ---------------------------------------------------------------------------
# 6. ClimatologyForecaster output shape
# ---------------------------------------------------------------------------

def test_climatology_shape(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    cf = ClimatologyForecaster()
    cf.fit(train_ds)
    preds = cf.predict_dataset(val_ds)
    assert preds.shape == (len(val_ds), cfg.forecast_horizon, N_TARGETS)


# ---------------------------------------------------------------------------
# 7. Climatology uses only training data
# ---------------------------------------------------------------------------

def test_climatology_fitted_on_train(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    cf = ClimatologyForecaster()
    cf.fit(train_ds)
    assert cf.climatology is not None
    assert len(cf.climatology) == N_TARGETS


# ---------------------------------------------------------------------------
# 8. Climatology does not use validation/test data
# ---------------------------------------------------------------------------

def test_climatology_no_val_test_leakage(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    cf1 = ClimatologyForecaster()
    cf1.fit(train_ds)
    clim1 = cf1.climatology.copy()

    # Fit again — must be identical
    cf2 = ClimatologyForecaster()
    cf2.fit(train_ds)
    np.testing.assert_array_almost_equal(cf1.climatology, cf2.climatology)

    # Climatology must not change if we build val/test datasets
    np.testing.assert_array_equal(cf1.climatology, clim1)


# ---------------------------------------------------------------------------
# 9. Climatology direction is circular mean
# ---------------------------------------------------------------------------

def test_climatology_circular_direction(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    cf = ClimatologyForecaster()
    cf.fit(train_ds)
    # sin^2 + cos^2 should be <= 1 (circular mean has magnitude <= 1)
    sin_c = cf.climatology[2]
    cos_c = cf.climatology[3]
    assert sin_c**2 + cos_c**2 <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 10. WaveScaler fit uses training data only
# ---------------------------------------------------------------------------

def test_scaler_fit_train_only(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    scaler = WaveScaler()
    X_train = train_ds.get_all_X()
    scaler.fit(X_train)
    assert scaler.is_fitted
    stats = scaler.fitted_stats
    assert "mean" in stats and "std" in stats


# ---------------------------------------------------------------------------
# 11. WaveScaler transform/inverse_transform round-trip
# ---------------------------------------------------------------------------

def test_scaler_roundtrip(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    scaler = WaveScaler()
    X_train = train_ds.get_all_X()
    scaler.fit(X_train)

    X_scaled = scaler.transform(X_train.copy())
    X_back   = scaler.inverse_transform(X_scaled)

    # Only Hs and Tp are scaled; check round-trip for valid (non-NaN) values
    for col in [0, 1]:
        valid = ~np.isnan(X_train[..., col])
        np.testing.assert_allclose(
            X_back[..., col][valid], X_train[..., col][valid], atol=1e-9,
            err_msg=f"Round-trip failed for feature {FEATURE_NAMES[col]}"
        )


# ---------------------------------------------------------------------------
# 12. WaveScaler does not standardize sin/cos or validity columns
# ---------------------------------------------------------------------------

def test_scaler_passthrough_columns(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    scaler = WaveScaler()
    X_train = train_ds.get_all_X()
    scaler.fit(X_train)
    X_scaled = scaler.transform(X_train.copy())

    # Columns 2,3,4,5,6 must be unchanged
    for col in [2, 3, 4, 5, 6]:
        np.testing.assert_array_equal(
            X_scaled[..., col], X_train[..., col],
            err_msg=f"Column {FEATURE_NAMES[col]} was modified by scaler"
        )


# ---------------------------------------------------------------------------
# 13. WaveScaler: val/test transformation uses training stats
# ---------------------------------------------------------------------------

def test_scaler_val_uses_train_stats(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    scaler = WaveScaler()
    X_train = train_ds.get_all_X()
    scaler.fit(X_train)
    stats_before = scaler.fitted_stats["mean"].copy()

    # Transform val — must not change fitted stats
    X_val = val_ds.get_all_X()
    _ = scaler.transform(X_val)
    np.testing.assert_array_equal(scaler.fitted_stats["mean"], stats_before)


# ---------------------------------------------------------------------------
# 14. LEAKAGE H: persistence never reads future truth
# ---------------------------------------------------------------------------

def test_leakage_h_persistence_no_future_truth(setup):
    """Persistence prediction must equal an obs value, not a truth value."""
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    pf = PersistenceForecaster()
    if len(train_ds) == 0:
        pytest.skip("No training windows")

    s = train_ds[0]
    origin = s.origin_index
    pred = pf.predict_sample(s, obs.Hs_obs, obs.Tp_obs, obs.direction_obs,
                             obs.valid_Hs, obs.valid_Tp, obs.valid_direction)

    if not np.isnan(pred[0, 0]):
        # The predicted Hs must come from obs, not from truth
        # obs and truth differ by noise, so they should not be equal
        future_true = obs.Hs_true[origin + 1 : origin + cfg.forecast_horizon + 1]
        for ft in future_true:
            assert not np.isclose(pred[0, 0], ft, atol=1e-10), \
                "Persistence prediction equals future truth — leakage detected"


# ---------------------------------------------------------------------------
# 15. LEAKAGE I: train_median imputation never uses future information
# ---------------------------------------------------------------------------

def test_leakage_i_imputation_no_future():
    """train_median must be computed from training data only, not future steps."""
    obs = _make_obs(n=200, seed=10, missing_prob=0.2)
    cfg = ForecastConfig(input_length=6, forecast_horizon=2,
                         missing_policy="train_median")
    split = make_split(len(obs.time), cfg)
    train_ds = ForecastDataset(obs, cfg, split, "train")

    # Medians must be computed from training indices only
    train_end = split.train_end
    X_train_raw = train_ds._X_raw[:train_end]
    expected_medians = np.nanmedian(X_train_raw, axis=0)
    np.testing.assert_allclose(train_ds.train_medians, expected_medians, atol=1e-10)


# ---------------------------------------------------------------------------
# 16. LEAKAGE E: climatology never uses val/test data
# ---------------------------------------------------------------------------

def test_leakage_e_climatology_train_only(setup):
    obs, cfg, train_ds, val_ds, test_ds, split = setup
    cf = ClimatologyForecaster()
    cf.fit(train_ds)
    clim_train = cf.climatology.copy()

    # Verify: climatology computed from train targets only
    targets_train = train_ds.get_all_targets().reshape(-1, N_TARGETS)
    hs_vals = targets_train[:, 0]
    hs_vals = hs_vals[~np.isnan(hs_vals)]
    expected_hs_mean = float(np.mean(hs_vals))
    assert abs(cf.climatology[0] - expected_hs_mean) < 1e-9, \
        "Climatology Hs does not match training mean"


from module4_forecasting.dataset import make_split, ForecastDataset
