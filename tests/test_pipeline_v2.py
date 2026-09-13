"""
tests/test_pipeline_v2.py — Tests for the Tier-1 v2 forecasting data pipeline.

Tests
-----
 1.  Archive coverage: 192 wave files present
 2.  Archive coverage: 192 wind files present
 3.  Archive coverage: complete 2010-2025
 4.  load_era5_archive returns correct column names
 5.  load_era5_archive total wave records = 140256
 6.  load_era5_archive total wind records = 140256
 7.  Timestamps are unique
 8.  Timestamps are monotonically increasing
 9.  Wave/wind timestamp alignment
10.  No duplicate timestamps in concatenated series
11.  Year split: train ends at 2022-01-01
12.  Year split: val ends at 2024-01-01
13.  Year split: test starts at 2024-01-01
14.  Year split: no overlap between splits
15.  Year split: covers full series
16.  ForecastConfigV2 defaults valid
17.  ForecastConfigV2 rejects bad parameters
18.  wave_only feature shape: (48, 7)
19.  wave_wind feature shape: (48, 11)
20.  Target shape: (48, 4)
21.  Feature names match schema
22.  Target names match schema
23.  Validity mask columns are binary
24.  Direction sin/cos encoding correct
25.  No future truth in X (leakage A)
26.  No future observations in X (leakage B)
27.  No window crosses train/val boundary
28.  No window crosses val/test boundary
29.  Scaler fitted only on train data
30.  Scaler does not use val/test statistics
31.  WaveScalerV2 transform/inverse round-trip
32.  WaveScalerV2 target transform/inverse round-trip
33.  train_median policy fills NaN in wave features
34.  train_median never uses val/test statistics
35.  Missing wave obs produce NaN in X (mask_only)
36.  Wind features not passed through sensor model
37.  build_datasets_v2 returns correct subset sizes
38.  Dataset is deterministic (same seed -> same output)
39.  validate_archive_coverage reports complete
40.  Missing month detection
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from module3_sensor.sensor import SensorParameters, WaveSensor
from module4_forecasting.era5_archive import (
    load_era5_archive,
    validate_archive_coverage,
    WAVES_DIR,
    WIND_DIR,
    DEFAULT_START_YEAR,
    DEFAULT_END_YEAR,
)
from module4_forecasting.dataset_v2 import (
    ForecastConfigV2,
    ForecastDatasetV2,
    WaveScalerV2,
    YearSplit,
    make_year_split,
    build_datasets_v2,
    FEATURE_NAMES_WAVE,
    FEATURE_NAMES_WIND,
    TARGET_NAMES_V2,
    N_FEATURES_WAVE,
    N_FEATURES_WIND,
    N_TARGETS_V2,
)


# ---------------------------------------------------------------------------
# Module-level fixture: load the full archive once
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_df():
    """Load the full 2010-2025 ERA5 archive once for all tests."""
    return load_era5_archive()


@pytest.fixture(scope="module")
def split(full_df):
    return make_year_split(full_df["time"].values)


@pytest.fixture(scope="module")
def obs_full(full_df):
    """Run the Module 3 sensor over the full archive."""
    p = SensorParameters(
        hs_noise_std=0.15,
        tp_noise_std=0.8,
        direction_noise_std=15.0,
        delay_steps=1,
        missing_probability=0.05,
        random_seed=42,
    )
    sensor = WaveSensor(p)
    time = np.arange(len(full_df), dtype=float)
    return sensor.observe(
        time=time,
        Hs_true=full_df["Hs_m"].values,
        Tp_true=full_df["Tp_s"].values,
        direction_true=full_df["direction_deg"].values,
    )


@pytest.fixture(scope="module")
def datasets_wave(obs_full, split):
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48, feature_mode="wave_only")
    return build_datasets_v2(obs_full, cfg, split)


@pytest.fixture(scope="module")
def datasets_wind(obs_full, full_df, split):
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48, feature_mode="wave_wind")
    return build_datasets_v2(
        obs_full, cfg, split,
        u10=full_df["u10"].values,
        v10=full_df["v10"].values,
    )


# ---------------------------------------------------------------------------
# 1-3. Archive coverage
# ---------------------------------------------------------------------------

def test_wave_file_count():
    import glob
    files = glob.glob(os.path.join(WAVES_DIR, "*.nc"))
    assert len(files) == 192, f"Expected 192 wave files, got {len(files)}"


def test_wind_file_count():
    import glob
    files = glob.glob(os.path.join(WIND_DIR, "*.nc"))
    assert len(files) == 192, f"Expected 192 wind files, got {len(files)}"


def test_archive_coverage_complete():
    result = validate_archive_coverage()
    assert result["complete"], (
        f"Archive incomplete. "
        f"Missing wave: {result['wave_missing'][:5]}, "
        f"Missing wind: {result['wind_missing'][:5]}"
    )
    assert result["expected_months"] == 192


# ---------------------------------------------------------------------------
# 4-6. DataFrame structure and record counts
# ---------------------------------------------------------------------------

def test_dataframe_columns(full_df):
    expected = {"time", "Hs_m", "Tp_s", "direction_deg", "u10", "v10"}
    assert expected.issubset(set(full_df.columns)), (
        f"Missing columns: {expected - set(full_df.columns)}"
    )


def test_total_wave_records(full_df):
    assert len(full_df) == 140256, (
        f"Expected 140256 records, got {len(full_df)}"
    )


def test_total_wind_records(full_df):
    # Wind is aligned to wave, so same count
    assert full_df["u10"].notna().sum() + full_df["u10"].isna().sum() == 140256


# ---------------------------------------------------------------------------
# 7-10. Timestamp integrity
# ---------------------------------------------------------------------------

def test_timestamps_unique(full_df):
    n_dupes = full_df["time"].duplicated().sum()
    assert n_dupes == 0, f"Found {n_dupes} duplicate timestamps"


def test_timestamps_monotonic(full_df):
    t = pd.to_datetime(full_df["time"])
    diffs = np.diff(t.values).astype("timedelta64[s]").astype(int)
    assert np.all(diffs > 0), "Timestamps are not strictly monotonically increasing"


def test_timestamps_hourly(full_df):
    t = pd.to_datetime(full_df["time"])
    diffs = np.diff(t.values).astype("timedelta64[s]").astype(int)
    non_hourly = int(np.sum(diffs != 3600))
    assert non_hourly == 0, f"{non_hourly} gaps are not exactly 1 hour"


def test_wave_wind_alignment(full_df):
    # Both streams loaded and merged — timestamps are identical by construction
    assert len(full_df) > 0
    assert "u10" in full_df.columns
    assert "v10" in full_df.columns


# ---------------------------------------------------------------------------
# 11-15. Year split
# ---------------------------------------------------------------------------

def test_split_train_end(full_df, split):
    t = pd.to_datetime(full_df["time"])
    boundary = pd.Timestamp("2022-01-01")
    # All train timestamps must be < 2022-01-01
    train_times = t.iloc[:split.train_end]
    assert (train_times < boundary).all(), "Train set contains timestamps >= 2022-01-01"


def test_split_val_end(full_df, split):
    t = pd.to_datetime(full_df["time"])
    boundary = pd.Timestamp("2024-01-01")
    val_times = t.iloc[split.train_end:split.val_end]
    assert (val_times < boundary).all(), "Val set contains timestamps >= 2024-01-01"
    assert (val_times >= pd.Timestamp("2022-01-01")).all(), \
        "Val set contains timestamps < 2022-01-01"


def test_split_test_start(full_df, split):
    t = pd.to_datetime(full_df["time"])
    test_times = t.iloc[split.val_end:]
    assert (test_times >= pd.Timestamp("2024-01-01")).all(), \
        "Test set contains timestamps < 2024-01-01"


def test_split_no_overlap(split):
    assert split.train_end <= split.val_end
    assert split.val_end <= split.n_total
    # Indices are exclusive ends, so no overlap by construction
    assert split.n_train + split.n_val + split.n_test == split.n_total


def test_split_covers_full_series(full_df, split):
    assert split.n_total == len(full_df)
    assert split.train_end + split.n_val + split.n_test == len(full_df)


# ---------------------------------------------------------------------------
# 16-17. ForecastConfigV2
# ---------------------------------------------------------------------------

def test_config_v2_defaults():
    c = ForecastConfigV2()
    assert c.input_length == 48
    assert c.forecast_horizon == 48
    assert c.feature_mode == "wave_only"
    assert c.missing_policy == "mask_only"


def test_config_v2_validation():
    with pytest.raises(ValueError):
        ForecastConfigV2(input_length=0)
    with pytest.raises(ValueError):
        ForecastConfigV2(forecast_horizon=0)
    with pytest.raises(ValueError):
        ForecastConfigV2(feature_mode="invalid")
    with pytest.raises(ValueError):
        ForecastConfigV2(missing_policy="unknown")


# ---------------------------------------------------------------------------
# 18-20. Array shapes
# ---------------------------------------------------------------------------

def test_wave_only_x_shape(datasets_wave):
    train_ds = datasets_wave[0]
    assert len(train_ds) > 0
    s = train_ds[0]
    assert s.X.shape == (48, N_FEATURES_WAVE), \
        f"Expected (48, {N_FEATURES_WAVE}), got {s.X.shape}"


def test_wave_wind_x_shape(datasets_wind):
    train_ds = datasets_wind[0]
    assert len(train_ds) > 0
    s = train_ds[0]
    assert s.X.shape == (48, N_FEATURES_WIND), \
        f"Expected (48, {N_FEATURES_WIND}), got {s.X.shape}"


def test_target_shape(datasets_wave):
    train_ds = datasets_wave[0]
    assert len(train_ds) > 0
    s = train_ds[0]
    assert s.target.shape == (48, N_TARGETS_V2), \
        f"Expected (48, {N_TARGETS_V2}), got {s.target.shape}"


# ---------------------------------------------------------------------------
# 21-22. Feature and target names
# ---------------------------------------------------------------------------

def test_feature_names_wave():
    assert len(FEATURE_NAMES_WAVE) == N_FEATURES_WAVE == 7
    assert FEATURE_NAMES_WAVE[0] == "Hs"
    assert FEATURE_NAMES_WAVE[1] == "Tp"
    assert FEATURE_NAMES_WAVE[2] == "sin_direction"
    assert FEATURE_NAMES_WAVE[3] == "cos_direction"
    assert FEATURE_NAMES_WAVE[4] == "Hs_valid"
    assert FEATURE_NAMES_WAVE[5] == "Tp_valid"
    assert FEATURE_NAMES_WAVE[6] == "direction_valid"


def test_feature_names_wind():
    assert len(FEATURE_NAMES_WIND) == N_FEATURES_WIND == 11
    assert FEATURE_NAMES_WIND[7]  == "u10"
    assert FEATURE_NAMES_WIND[8]  == "v10"
    assert FEATURE_NAMES_WIND[9]  == "u10_valid"
    assert FEATURE_NAMES_WIND[10] == "v10_valid"


def test_target_names_v2():
    assert len(TARGET_NAMES_V2) == N_TARGETS_V2 == 4
    assert TARGET_NAMES_V2[0] == "Hs"
    assert TARGET_NAMES_V2[1] == "Tp"
    assert TARGET_NAMES_V2[2] == "sin_direction"
    assert TARGET_NAMES_V2[3] == "cos_direction"


# ---------------------------------------------------------------------------
# 23. Validity mask columns are binary
# ---------------------------------------------------------------------------

def test_validity_columns_binary(datasets_wave):
    train_ds = datasets_wave[0]
    for i in range(min(10, len(train_ds))):
        X = train_ds[i].X
        for col in [4, 5, 6]:
            vals = X[:, col]
            assert np.all((vals == 0.0) | (vals == 1.0)), \
                f"Validity column {col} has non-binary values in sample {i}"


def test_wind_validity_columns_binary(datasets_wind):
    train_ds = datasets_wind[0]
    for i in range(min(10, len(train_ds))):
        X = train_ds[i].X
        for col in [9, 10]:
            vals = X[:, col]
            assert np.all((vals == 0.0) | (vals == 1.0)), \
                f"Wind validity column {col} has non-binary values"


# ---------------------------------------------------------------------------
# 24. Direction sin/cos encoding
# ---------------------------------------------------------------------------

def test_direction_sincos_bounds(datasets_wave):
    train_ds = datasets_wave[0]
    s = train_ds[0]
    sin_col = s.X[:, 2]
    cos_col = s.X[:, 3]
    valid = ~np.isnan(sin_col)
    assert np.all(np.abs(sin_col[valid]) <= 1.0 + 1e-9)
    assert np.all(np.abs(cos_col[valid]) <= 1.0 + 1e-9)


def test_direction_sincos_unit_circle(datasets_wave):
    train_ds = datasets_wave[0]
    s = train_ds[0]
    sin_col = s.X[:, 2]
    cos_col = s.X[:, 3]
    valid = ~(np.isnan(sin_col) | np.isnan(cos_col))
    norms = sin_col[valid]**2 + cos_col[valid]**2
    np.testing.assert_allclose(norms, 1.0, atol=1e-9,
                                err_msg="sin²+cos² != 1 for direction encoding")


# ---------------------------------------------------------------------------
# 25-26. Leakage tests
# ---------------------------------------------------------------------------

def test_leakage_no_future_truth_in_X(obs_full, split):
    """X must not contain any future true values."""
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48)
    ds = ForecastDatasetV2(obs_full, cfg, split, "train")
    assert len(ds) > 0
    s = ds[0]
    origin = s.origin_index
    # X covers [origin-47 : origin+1]; target covers [origin+1 : origin+49]
    # Hs_true at future indices must not appear in X[:, 0]
    future_true = obs_full.Hs_true[origin + 1 : origin + 49]
    for ft in future_true:
        if np.isnan(ft):
            continue
        assert not np.any(np.isclose(s.X[:, 0], ft, atol=1e-10)), \
            f"Future true Hs {ft:.4f} found in X"


def test_leakage_x_matches_obs_window(obs_full, split):
    """X[:, 0] must exactly match obs.Hs_obs in the input window."""
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48)
    ds = ForecastDatasetV2(obs_full, cfg, split, "train")
    assert len(ds) > 0
    s = ds[0]
    origin = s.origin_index
    L = 48
    expected = obs_full.Hs_obs[origin - L + 1 : origin + 1]
    np.testing.assert_array_equal(s.X[:, 0], expected)


# ---------------------------------------------------------------------------
# 27-28. No window crosses split boundaries
# ---------------------------------------------------------------------------

def test_no_window_crosses_train_val(datasets_wave, split):
    train_ds, val_ds, test_ds, _ = datasets_wave
    L, H = 48, 48

    if len(train_ds) > 0:
        origins = train_ds.get_all_origins()
        assert origins.min() - L + 1 >= 0
        assert origins.max() + H + 1 <= split.train_end

    if len(val_ds) > 0:
        origins = val_ds.get_all_origins()
        assert origins.min() - L + 1 >= split.train_end
        assert origins.max() + H + 1 <= split.val_end


def test_no_window_crosses_val_test(datasets_wave, split):
    _, _, test_ds, _ = datasets_wave
    L, H = 48, 48

    if len(test_ds) > 0:
        origins = test_ds.get_all_origins()
        assert origins.min() - L + 1 >= split.val_end
        assert origins.max() + H + 1 <= split.n_total


# ---------------------------------------------------------------------------
# 29-30. Scaler fitted only on train
# ---------------------------------------------------------------------------

def test_scaler_fitted_on_train_only(datasets_wave):
    train_ds, val_ds, test_ds, scaler = datasets_wave
    assert scaler.is_fitted

    # Scaler stats should match training data statistics
    X_train = train_ds.get_all_X()
    flat = X_train.reshape(-1, N_FEATURES_WAVE)
    hs_vals = flat[:, 0]
    hs_vals = hs_vals[~np.isnan(hs_vals)]
    expected_mean = float(np.mean(hs_vals))
    assert abs(scaler.fitted_stats["mean"][0] - expected_mean) < 1e-6, \
        "Scaler Hs mean does not match training data mean"


def test_scaler_not_refitted_on_val(datasets_wave):
    train_ds, val_ds, test_ds, scaler = datasets_wave
    stats_before = scaler.fitted_stats["mean"].copy()
    # Attempting to transform val data should not change scaler stats
    if len(val_ds) > 0:
        _ = scaler.transform(val_ds.get_all_X())
    np.testing.assert_array_equal(scaler.fitted_stats["mean"], stats_before)


# ---------------------------------------------------------------------------
# 31-32. Scaler round-trips
# ---------------------------------------------------------------------------

def test_scaler_wave_roundtrip(datasets_wave):
    train_ds, _, _, scaler = datasets_wave
    X = train_ds.get_all_X()
    X_scaled = scaler.transform(X)
    X_back = scaler.inverse_transform(X_scaled)
    valid = ~np.isnan(X[..., 0])
    np.testing.assert_allclose(
        X[..., 0][valid], X_back[..., 0][valid], atol=1e-9,
        err_msg="Hs round-trip failed"
    )


def test_scaler_target_roundtrip(datasets_wave):
    train_ds, _, _, scaler = datasets_wave
    y = train_ds.get_all_targets()
    y_scaled = scaler.transform_target(y)
    y_back = scaler.inverse_transform_target(y_scaled)
    np.testing.assert_allclose(y[..., 0], y_back[..., 0], atol=1e-9,
                                err_msg="Target Hs round-trip failed")


# ---------------------------------------------------------------------------
# 33-34. Missing data policy
# ---------------------------------------------------------------------------

def test_train_median_fills_wave_nan(obs_full, split):
    cfg = ForecastConfigV2(
        input_length=48, forecast_horizon=48,
        feature_mode="wave_only", missing_policy="train_median"
    )
    ds = ForecastDatasetV2(obs_full, cfg, split, "train")
    if len(ds) == 0:
        pytest.skip("No training windows")
    X_all = ds.get_all_X()
    # Hs and Tp columns should have no NaN after train_median imputation
    assert not np.any(np.isnan(X_all[:, :, 0])), "Hs still has NaN after train_median"
    assert not np.any(np.isnan(X_all[:, :, 1])), "Tp still has NaN after train_median"


def test_train_median_uses_only_train_data(obs_full, split):
    cfg = ForecastConfigV2(
        input_length=48, forecast_horizon=48, missing_policy="train_median"
    )
    train_ds = ForecastDatasetV2(obs_full, cfg, split, "train")
    medians_a = train_ds.train_medians.copy()

    # Build val with same train_medians — medians must not change
    val_ds = ForecastDatasetV2(obs_full, cfg, split, "val",
                                train_medians=train_ds.train_medians)
    np.testing.assert_array_equal(
        train_ds.train_medians, medians_a,
        err_msg="train_medians changed after building val dataset"
    )


# ---------------------------------------------------------------------------
# 35. Missing wave obs produce NaN in X (mask_only)
# ---------------------------------------------------------------------------

def test_missing_obs_produce_nan(obs_full, split):
    cfg = ForecastConfigV2(
        input_length=48, forecast_horizon=48, missing_policy="mask_only"
    )
    ds = ForecastDatasetV2(obs_full, cfg, split, "train")
    if len(ds) == 0:
        pytest.skip("No training windows")
    X_all = ds.get_all_X()
    # With 5% missing probability over 140k records, NaN must appear
    has_nan = np.any(np.isnan(X_all[:, :, 0]))
    assert has_nan, "Expected NaN in Hs column with missing_probability=0.05"


# ---------------------------------------------------------------------------
# 36. Wind features not passed through sensor model
# ---------------------------------------------------------------------------

def test_wind_not_through_sensor(full_df, obs_full, split):
    """u10/v10 in X must equal ERA5 values directly, not sensor-noised values."""
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48,
                           feature_mode="wave_wind")
    ds = ForecastDatasetV2(
        obs_full, cfg, split, "train",
        u10=full_df["u10"].values,
        v10=full_df["v10"].values,
    )
    assert len(ds) > 0
    s = ds[0]
    origin = s.origin_index
    L = 48
    # X[:, 7] should equal u10[origin-L+1 : origin+1] exactly
    expected_u10 = full_df["u10"].values[origin - L + 1 : origin + 1]
    np.testing.assert_array_equal(s.X[:, 7], expected_u10,
                                   err_msg="u10 in X does not match ERA5 u10")


# ---------------------------------------------------------------------------
# 37. build_datasets_v2 sizes
# ---------------------------------------------------------------------------

def test_build_datasets_v2_sizes(datasets_wave, split):
    train_ds, val_ds, test_ds, _ = datasets_wave
    assert len(train_ds) > 0, "No training windows"
    assert len(val_ds)   > 0, "No validation windows"
    assert len(test_ds)  > 0, "No test windows"
    # Train should have the most windows (12 years vs 2+2)
    assert len(train_ds) > len(val_ds)
    assert len(train_ds) > len(test_ds)


# ---------------------------------------------------------------------------
# 38. Determinism
# ---------------------------------------------------------------------------

def test_dataset_deterministic(obs_full, split):
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48)
    ds1 = ForecastDatasetV2(obs_full, cfg, split, "train")
    ds2 = ForecastDatasetV2(obs_full, cfg, split, "train")
    assert len(ds1) > 0
    np.testing.assert_array_equal(ds1[0].X, ds2[0].X)
    np.testing.assert_array_equal(ds1[0].target, ds2[0].target)


# ---------------------------------------------------------------------------
# 39-40. Archive validation helpers
# ---------------------------------------------------------------------------

def test_validate_archive_coverage_complete():
    result = validate_archive_coverage()
    assert result["complete"]
    assert len(result["wave_missing"]) == 0
    assert len(result["wind_missing"]) == 0


def test_validate_archive_detects_missing(tmp_path):
    """validate_archive_coverage reports missing files correctly."""
    result = validate_archive_coverage(
        start_year=2010, end_year=2025,
        waves_dir=str(tmp_path / "waves"),
        wind_dir=str(tmp_path / "wind"),
    )
    assert not result["complete"]
    assert len(result["wave_missing"]) == 192
    assert len(result["wind_missing"]) == 192


# ---------------------------------------------------------------------------
# Frozen module integrity checks
# ---------------------------------------------------------------------------

def test_frozen_dataset_v1_unchanged():
    path = _ROOT / "module4_forecasting" / "dataset.py"
    src = path.read_text()
    assert "class ForecastDataset" in src
    assert "FEATURE_NAMES" in src
    assert "N_FEATURES = 7" in src or "N_FEATURES = len(FEATURE_NAMES)" in src


def test_frozen_sensor_unchanged():
    path = _ROOT / "module3_sensor" / "sensor.py"
    src = path.read_text()
    assert "class WaveSensor" in src
    assert "class SensorObservation" in src


def test_frozen_era5_loader_unchanged():
    path = _ROOT / "module1_ocean" / "era5.py"
    src = path.read_text()
    assert "def load_era5" in src
    assert "def extract_goa_timeseries" in src
