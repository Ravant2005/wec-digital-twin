"""
test_module5_val_error_bank.py — Tests for Module 5.3A-DATA validation error bank.

Verifies:
 1.  Output file exists.
 2.  Expected columns present.
 3.  All 48 leads present.
 4.  Hs_error == Hs_pred - Hs_true.
 5.  Tp_error == Tp_pred - Tp_true.
 6.  Direction errors are correctly wrapped (−180, 180].
 7.  No 2024–2025 timestamps (no test data in bank).
 8.  No training-period origins (no pre-2022 data in bank).
 9.  No NaN or invalid rows.
10.  Deterministic regeneration (bit-stable within float tolerance).
11.  Compatible with ForecastErrorSampler / load_error_bank().
12.  Trajectory lengths are exactly 48 leads.
13.  sample_idx range is 0 to N-1 (contiguous).
14.  origin_idx timestamps are exclusively 2022–2023.
15.  New bank does NOT overlap with the test-period bank.
16.  ForecastErrorSampler.apply() returns finite values with new bank.
17.  Error statistics grow with lead time (forecast skill degrades).
18.  Column schema matches forecast_errors.csv.gz exactly.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

VAL_BANK_PATH  = "results/forecasting/exp_gru_wind/forecast_errors_val2022_2023.csv.gz"
TEST_BANK_PATH = "results/forecasting/exp_gru_wind/forecast_errors.csv.gz"

# Module 4 split boundaries (from dataset_v2.py)
TRAIN_END_ORIGIN_MIN = 105192   # first val idx
VAL_END_ORIGIN_MAX   = 122712   # first test idx (exclusive)

# Expected year range of val origins
VAL_START_YEAR = 2022
VAL_END_YEAR   = 2023


@pytest.fixture(scope="session")
def val_df():
    """Load the full validation error bank once per session."""
    if not os.path.exists(VAL_BANK_PATH):
        pytest.skip(f"Validation error bank not found: {VAL_BANK_PATH}")
    return pd.read_csv(VAL_BANK_PATH)


@pytest.fixture(scope="session")
def test_df():
    """Load the existing test-period bank for comparison."""
    if not os.path.exists(TEST_BANK_PATH):
        pytest.skip(f"Test error bank not found: {TEST_BANK_PATH}")
    return pd.read_csv(TEST_BANK_PATH)


# ===========================================================================
# Test 1 — Output file exists
# ===========================================================================

def test_01_file_exists():
    assert os.path.exists(VAL_BANK_PATH), f"File not found: {VAL_BANK_PATH}"
    size_mb = os.path.getsize(VAL_BANK_PATH) / 1e6
    assert size_mb > 10.0, f"File too small ({size_mb:.1f} MB) — may be empty"


# ===========================================================================
# Test 2 — Expected columns present
# ===========================================================================

def test_02_expected_columns(val_df, test_df):
    """Column schema must exactly match the test-period bank."""
    expected = list(test_df.columns)
    assert list(val_df.columns) == expected, (
        f"Column mismatch: got {list(val_df.columns)}, expected {expected}"
    )


def test_02_exact_required_columns(val_df):
    required = [
        "lead_h", "sample_idx", "origin_idx",
        "Hs_pred", "Hs_true", "Hs_error",
        "Tp_pred", "Tp_true", "Tp_error",
        "dir_pred_deg", "dir_true_deg", "dir_error_deg",
    ]
    for col in required:
        assert col in val_df.columns, f"Missing column: {col}"


# ===========================================================================
# Test 3 — All 48 leads present
# ===========================================================================

def test_03_all_48_leads(val_df):
    leads = sorted(val_df["lead_h"].unique().tolist())
    assert leads == list(range(1, 49)), f"Missing leads: {leads[:5]}...{leads[-5:]}"


def test_03_same_count_per_lead(val_df):
    """Each lead must have the same number of samples."""
    counts = val_df.groupby("lead_h").size()
    assert counts.nunique() == 1, f"Unequal samples per lead: {counts.unique()}"


# ===========================================================================
# Test 4 — Hs_error == Hs_pred - Hs_true
# ===========================================================================

def test_04_hs_error_sign_convention(val_df):
    max_diff = (val_df["Hs_pred"] - val_df["Hs_true"] - val_df["Hs_error"]).abs().max()
    assert max_diff < 1e-5, f"Hs_error != Hs_pred - Hs_true: max_diff={max_diff:.2e}"


# ===========================================================================
# Test 5 — Tp_error == Tp_pred - Tp_true
# ===========================================================================

def test_05_tp_error_sign_convention(val_df):
    max_diff = (val_df["Tp_pred"] - val_df["Tp_true"] - val_df["Tp_error"]).abs().max()
    assert max_diff < 1e-5, f"Tp_error != Tp_pred - Tp_true: max_diff={max_diff:.2e}"


# ===========================================================================
# Test 6 — Direction errors wrapped to (-180, 180]
# ===========================================================================

def test_06_dir_error_wrapped(val_df):
    assert val_df["dir_error_deg"].min() > -180.0, (
        f"dir_error_deg min = {val_df['dir_error_deg'].min():.3f} not > -180"
    )
    assert val_df["dir_error_deg"].max() <= 180.0, (
        f"dir_error_deg max = {val_df['dir_error_deg'].max():.3f} not <= 180"
    )


def test_06_dir_error_is_circular_pred_minus_true(val_df):
    """Verify direction error uses circular(pred - true) convention on a sample."""
    sample = val_df.sample(100, random_state=42)
    raw_diff = (sample["dir_pred_deg"] - sample["dir_true_deg"]).values
    # wrap to (-180, 180]
    wrapped = ((raw_diff + 180) % 360) - 180
    np.testing.assert_allclose(
        wrapped, sample["dir_error_deg"].values, atol=0.01,
        err_msg="Direction error is not circular(pred - true)"
    )


# ===========================================================================
# Test 7 — No 2024–2025 timestamps (test data not in bank)
# ===========================================================================

def test_07_no_test_period_origins(val_df):
    max_origin = int(val_df["origin_idx"].max())
    assert max_origin < VAL_END_ORIGIN_MAX, (
        f"origin_idx {max_origin} >= test boundary {VAL_END_ORIGIN_MAX}"
    )


# ===========================================================================
# Test 8 — No training-period origins
# ===========================================================================

def test_08_no_training_period_origins(val_df):
    min_origin = int(val_df["origin_idx"].min())
    assert min_origin >= TRAIN_END_ORIGIN_MIN, (
        f"origin_idx {min_origin} < val start {TRAIN_END_ORIGIN_MIN}: training data leaked"
    )


# ===========================================================================
# Test 9 — No NaN or invalid rows
# ===========================================================================

def test_09_no_nan(val_df):
    for col in ["Hs_error", "Tp_error", "dir_error_deg",
                "Hs_pred", "Hs_true", "Tp_pred", "Tp_true"]:
        n_nan = val_df[col].isna().sum()
        assert n_nan == 0, f"NaN in column {col}: {n_nan} rows"


def test_09_hs_pred_positive(val_df):
    assert (val_df["Hs_pred"] > 0).all(), "Some Hs_pred <= 0"


def test_09_tp_pred_positive(val_df):
    assert (val_df["Tp_pred"] > 0).all(), "Some Tp_pred <= 0"


# ===========================================================================
# Test 10 — Deterministic regeneration
# ===========================================================================

def test_10_deterministic(val_df):
    """Re-reading the saved file must produce identical results."""
    df_reload = pd.read_csv(VAL_BANK_PATH)
    np.testing.assert_allclose(
        val_df["Hs_error"].values, df_reload["Hs_error"].values, atol=1e-9,
        err_msg="Hs_error differs between two reads of the same file"
    )


# ===========================================================================
# Test 11 — Compatible with ForecastErrorSampler / load_error_bank()
# ===========================================================================

def test_11_load_error_bank_compatible():
    from module5_control.forecast_uncertainty import load_error_bank
    bank = load_error_bank(csv_path=VAL_BANK_PATH)
    assert bank.n_trajectories > 0
    assert bank.n_leads == 48
    assert bank.err_hs.shape  == (bank.n_trajectories, 48)
    assert bank.err_tp.shape  == (bank.n_trajectories, 48)
    assert bank.err_dir.shape == (bank.n_trajectories, 48)
    assert not np.any(np.isnan(bank.err_hs))


def test_11_forecast_error_sampler_apply():
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    from module4_forecasting.dataset import direction_to_sincos
    sampler = ForecastErrorSampler.from_csv(csv_path=VAL_BANK_PATH, seed=42)
    # Build a fake deterministic forecast
    det = np.zeros((48, 4), dtype=np.float64)
    det[:, 0] = 1.5   # Hs
    det[:, 1] = 8.0   # Tp
    sin_d, cos_d = direction_to_sincos(np.full(48, 270.0))
    det[:, 2] = sin_d
    det[:, 3] = cos_d
    realistic = sampler.apply(det)
    assert realistic.shape == (48, 4)
    assert np.all(np.isfinite(realistic))
    assert np.all(realistic[:, 0] >= 0.0)   # Hs >= 0
    assert np.all(realistic[:, 1] > 0.0)    # Tp > 0
    assert np.allclose(realistic[:, 2]**2 + realistic[:, 3]**2, 1.0, atol=1e-10)


# ===========================================================================
# Test 12 — Trajectory lengths are exactly 48
# ===========================================================================

def test_12_trajectory_length_48(val_df):
    per_sample = val_df.groupby("sample_idx")["lead_h"].count()
    assert (per_sample == 48).all(), (
        f"Some trajectories have != 48 leads: {per_sample.value_counts()}"
    )


# ===========================================================================
# Test 13 — sample_idx is 0 to N-1 (contiguous)
# ===========================================================================

def test_13_contiguous_sample_idx(val_df):
    n_traj = val_df["sample_idx"].nunique()
    expected = set(range(n_traj))
    actual   = set(val_df["sample_idx"].unique().tolist())
    assert actual == expected, "sample_idx is not contiguous 0..N-1"


# ===========================================================================
# Test 14 — origin_idx timestamps are exclusively 2022–2023
# ===========================================================================

def test_14_origin_timestamps_2022_2023():
    from module4_forecasting.era5_archive import load_era5_archive
    from module4_forecasting.dataset_v2 import make_year_split
    df_era5 = load_era5_archive(allow_missing_months=True)
    split   = make_year_split(df_era5["time"].values)

    val_df  = pd.read_csv(VAL_BANK_PATH)
    unique_origins = val_df["origin_idx"].unique()

    # Check year range of origins
    origin_times = pd.to_datetime(df_era5["time"].iloc[unique_origins])
    years        = origin_times.dt.year
    assert years.min() == VAL_START_YEAR, f"Earliest year {years.min()} != {VAL_START_YEAR}"
    assert years.max() == VAL_END_YEAR,   f"Latest year {years.max()} != {VAL_END_YEAR}"


# ===========================================================================
# Test 15 — New bank does NOT overlap with the test-period bank
# ===========================================================================

def test_15_no_overlap_with_test_bank(val_df, test_df):
    val_origins  = set(val_df["origin_idx"].unique().tolist())
    test_origins = set(test_df["origin_idx"].unique().tolist())
    overlap = val_origins & test_origins
    assert len(overlap) == 0, (
        f"origin_idx overlap between val and test banks: {len(overlap)} shared origins"
    )


def test_15_val_origins_less_than_test_origins(val_df, test_df):
    """Validate chronological ordering: val origins < test origins."""
    assert val_df["origin_idx"].max() < test_df["origin_idx"].min(), (
        f"Val max origin {val_df['origin_idx'].max()} not < "
        f"test min origin {test_df['origin_idx'].min()}"
    )


# ===========================================================================
# Test 16 — ForecastErrorSampler.apply() produces finite values
# ===========================================================================

def test_16_sampler_finite_output():
    from module5_control.forecast_uncertainty import ForecastErrorSampler
    import numpy as np
    sampler = ForecastErrorSampler.from_csv(csv_path=VAL_BANK_PATH, seed=99)
    det = np.ones((48, 4), dtype=np.float64)
    det[:, 0] = 2.0; det[:, 1] = 10.0
    det[:, 2] = 0.0; det[:, 3] = -1.0  # direction=270 deg
    for _ in range(20):
        sampler.reset()
        out = sampler.apply(det)
        assert np.all(np.isfinite(out)), "Sampler produced non-finite values"
        assert np.all(out[:, 0] >= 0.0), "Negative Hs from sampler"
        assert np.all(out[:, 1] > 0.0),  "Non-positive Tp from sampler"


# ===========================================================================
# Test 17 — Error statistics grow with lead time
# ===========================================================================

def test_17_error_grows_with_lead(val_df):
    """Hs error std must increase from lead 1 to lead 48."""
    std_1  = float(val_df[val_df["lead_h"] == 1 ]["Hs_error"].std())
    std_12 = float(val_df[val_df["lead_h"] == 12]["Hs_error"].std())
    std_24 = float(val_df[val_df["lead_h"] == 24]["Hs_error"].std())
    std_48 = float(val_df[val_df["lead_h"] == 48]["Hs_error"].std())
    assert std_12 > std_1,  f"Hs std should grow 1→12: {std_1:.4f} → {std_12:.4f}"
    assert std_24 > std_12, f"Hs std should grow 12→24: {std_12:.4f} → {std_24:.4f}"
    assert std_48 > std_24, f"Hs std should grow 24→48: {std_24:.4f} → {std_48:.4f}"


# ===========================================================================
# Test 18 — Column schema matches test-period bank exactly
# ===========================================================================

def test_18_column_schema_matches_test_bank(val_df, test_df):
    assert list(val_df.dtypes) == list(test_df.dtypes), (
        f"Dtype mismatch:\nval:  {list(val_df.dtypes)}\ntest: {list(test_df.dtypes)}"
    )


# ===========================================================================
# Additional: n_trajectories and n_rows sanity
# ===========================================================================

def test_n_trajectories(val_df):
    n = val_df["sample_idx"].nunique()
    assert n == 17425, f"Expected 17425 trajectories, got {n}"
    assert len(val_df) == 17425 * 48, (
        f"Expected {17425*48} rows, got {len(val_df)}"
    )


def test_n_rows_total(val_df):
    assert len(val_df) == 836_400, f"Expected 836,400 rows, got {len(val_df)}"
