"""
tests/test_era5_visualization.py — Tests for plot_era5.py and analyze_era5.py.

Run with:  pytest tests/test_era5_visualization.py -v
"""

import copy

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from module1_ocean.plot_era5 import (
    load_goa_dataframe,
    plot_wave_height,
    plot_peak_period,
    plot_wave_direction,
    plot_all,
)
from module1_ocean.analyze_era5 import (
    wave_height_stats,
    peak_period_stats,
    wave_direction_stats,
    hs_tp_correlation,
    summarise_all,
)

TEST_FILE = "data/raw/era5_goa_jan2024.nc"
EXPECTED_COLUMNS = ["time", "Hs_m", "Tp_s", "direction_deg"]


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Load the January 2024 Goa time series once for all tests."""
    return load_goa_dataframe(TEST_FILE)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test to avoid resource warnings."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_returns_dataframe(df):
    """load_goa_dataframe must return a pandas DataFrame."""
    assert isinstance(df, pd.DataFrame)


def test_dataframe_has_expected_columns(df):
    """DataFrame must have exactly the four expected columns."""
    assert list(df.columns) == EXPECTED_COLUMNS


def test_dataframe_has_744_records(df):
    """January 2024 ERA5 dataset must contain exactly 744 hourly records."""
    assert len(df) == 744


# ---------------------------------------------------------------------------
# Plotting — execution without errors
# ---------------------------------------------------------------------------

def test_plot_wave_height_runs(df):
    """plot_wave_height must execute without raising an exception."""
    fig = plot_wave_height(df)
    assert fig is not None


def test_plot_peak_period_runs(df):
    """plot_peak_period must execute without raising an exception."""
    fig = plot_peak_period(df)
    assert fig is not None


def test_plot_wave_direction_runs(df):
    """plot_wave_direction must execute without raising an exception."""
    fig = plot_wave_direction(df)
    assert fig is not None


def test_plot_all_runs(df):
    """plot_all must execute without raising an exception."""
    fig = plot_all(df)
    assert fig is not None


def test_plot_all_has_three_axes(df):
    """plot_all must produce a figure with exactly three subplots."""
    fig = plot_all(df)
    assert len(fig.axes) == 3


# ---------------------------------------------------------------------------
# DataFrame immutability — plotting must not modify the input
# ---------------------------------------------------------------------------

def test_plot_wave_height_does_not_modify_df(df):
    snapshot = df.copy(deep=True)
    plot_wave_height(df)
    pd.testing.assert_frame_equal(df, snapshot)


def test_plot_all_does_not_modify_df(df):
    snapshot = df.copy(deep=True)
    plot_all(df)
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# Wave height statistics
# ---------------------------------------------------------------------------

def test_hs_stats_keys(df):
    """wave_height_stats must return all required keys."""
    stats = wave_height_stats(df)
    for key in ("count", "mean_m", "std_m", "min_m", "max_m", "median_m"):
        assert key in stats, f"Missing key: {key}"


def test_hs_stats_finite(df):
    """All Hs statistics must be finite (no NaN or inf in this dataset)."""
    stats = wave_height_stats(df)
    for key, val in stats.items():
        assert np.isfinite(val), f"Hs stat '{key}' is not finite: {val}"


def test_hs_count(df):
    """Hs count must equal 744 (no NaNs in this dataset)."""
    assert wave_height_stats(df)["count"] == 744


def test_hs_mean_range(df):
    """Hs mean must be within the physically plausible range for this dataset."""
    mean = wave_height_stats(df)["mean_m"]
    assert 0.5 < mean < 2.0, f"Hs mean {mean:.3f} m outside expected range"


def test_hs_min_positive(df):
    """Hs minimum must be positive (wave height cannot be zero or negative)."""
    assert wave_height_stats(df)["min_m"] > 0.0


# ---------------------------------------------------------------------------
# Peak period statistics
# ---------------------------------------------------------------------------

def test_tp_stats_keys(df):
    """peak_period_stats must return all required keys."""
    stats = peak_period_stats(df)
    for key in ("count", "mean_s", "std_s", "min_s", "max_s", "median_s"):
        assert key in stats, f"Missing key: {key}"


def test_tp_stats_finite(df):
    """All Tp statistics must be finite."""
    stats = peak_period_stats(df)
    for key, val in stats.items():
        assert np.isfinite(val), f"Tp stat '{key}' is not finite: {val}"


def test_tp_mean_range(df):
    """Tp mean must be within a physically plausible range for ocean swell."""
    mean = peak_period_stats(df)["mean_s"]
    assert 3.0 < mean < 25.0, f"Tp mean {mean:.2f} s outside expected range"


# ---------------------------------------------------------------------------
# Wave direction statistics (circular)
# ---------------------------------------------------------------------------

def test_direction_stats_keys(df):
    """wave_direction_stats must return all required keys."""
    stats = wave_direction_stats(df)
    for key in ("count", "circular_mean_deg", "circular_std_deg",
                "mean_resultant_len", "min_deg", "max_deg", "median_deg"):
        assert key in stats, f"Missing key: {key}"


def test_circular_mean_in_range(df):
    """Circular mean direction must be in [0, 360)."""
    mean_dir = wave_direction_stats(df)["circular_mean_deg"]
    assert 0.0 <= mean_dir < 360.0


def test_mean_resultant_len_in_range(df):
    """Mean resultant length R̄ must be in [0, 1]."""
    r_bar = wave_direction_stats(df)["mean_resultant_len"]
    assert 0.0 <= r_bar <= 1.0


def test_direction_stats_does_not_modify_df(df):
    """wave_direction_stats must not modify the input DataFrame."""
    snapshot = df.copy(deep=True)
    wave_direction_stats(df)
    pd.testing.assert_frame_equal(df, snapshot)


# ---------------------------------------------------------------------------
# Hs / Tp correlation
# ---------------------------------------------------------------------------

def test_correlation_is_float(df):
    """hs_tp_correlation must return a Python float."""
    assert isinstance(hs_tp_correlation(df), float)


def test_correlation_in_valid_range(df):
    """Pearson r must be in [-1, 1]."""
    r = hs_tp_correlation(df)
    assert -1.0 <= r <= 1.0


def test_correlation_finite(df):
    """Pearson r must be finite for this dataset."""
    assert np.isfinite(hs_tp_correlation(df))


# ---------------------------------------------------------------------------
# summarise_all
# ---------------------------------------------------------------------------

def test_summarise_all_keys(df):
    """summarise_all must return all top-level keys."""
    summary = summarise_all(df)
    for key in ("wave_height", "peak_period", "wave_direction", "hs_tp_pearson_r"):
        assert key in summary, f"Missing key: {key}"
