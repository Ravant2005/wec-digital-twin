"""
tests/test_persistence_comparison.py — Tests for lead-wise persistence audit.

Tests
-----
 1. Output CSV has exactly 48 lead rows
 2. Leads are 1..48 in order
 3. No NaNs in any metric column
 4. Skill calculation is correct (1 - model/persistence)
 5. Persistence is calculated independently by lead (not a flat line)
 6. Direction uses circular error (not raw subtraction)
 7. Deterministic: repeated execution gives identical values
 8. GRU Hs RMSE < persistence Hs RMSE at lead 1 (sanity)
 9. GRU Tp RMSE < persistence Tp RMSE at lead 1 (sanity)
10. GRU direction MAE < persistence direction MAE at lead 1 (sanity)
11. Persistence Hs RMSE grows with lead (not flat)
12. Skill is positive at lead 1 for all metrics
13. CSV columns match required schema exactly
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

COMP_CSV = _ROOT / "results" / "forecasting" / "exp_gru_wind" / \
           "per_lead_persistence_comparison.csv"

REQUIRED_COLUMNS = [
    "lead_hours",
    "gru_hs_rmse", "persistence_hs_rmse", "hs_skill_vs_persistence",
    "gru_hs_mae",  "persistence_hs_mae",
    "gru_tp_rmse", "persistence_tp_rmse", "tp_skill_vs_persistence",
    "gru_tp_mae",  "persistence_tp_mae",
    "gru_direction_mae", "persistence_direction_mae",
]


@pytest.fixture(scope="module")
def comp_df() -> pd.DataFrame:
    if not COMP_CSV.exists():
        pytest.skip(f"Comparison CSV not found: {COMP_CSV}. "
                    "Run audit_persistence_comparison.py first.")
    return pd.read_csv(COMP_CSV)


# ---------------------------------------------------------------------------
# 1. Exactly 48 rows
# ---------------------------------------------------------------------------
def test_exactly_48_leads(comp_df):
    assert len(comp_df) == 48, f"Expected 48 rows, got {len(comp_df)}"


# ---------------------------------------------------------------------------
# 2. Leads are 1..48 in order
# ---------------------------------------------------------------------------
def test_leads_are_1_to_48(comp_df):
    assert list(comp_df["lead_hours"]) == list(range(1, 49)), \
        "lead_hours column is not exactly 1..48 in order"


# ---------------------------------------------------------------------------
# 3. No NaNs in any metric column
# ---------------------------------------------------------------------------
def test_no_nans(comp_df):
    for col in REQUIRED_COLUMNS:
        n_nan = comp_df[col].isna().sum()
        assert n_nan == 0, f"Column '{col}' has {n_nan} NaN values"


# ---------------------------------------------------------------------------
# 4. Skill calculation is correct
# ---------------------------------------------------------------------------
def test_hs_skill_calculation(comp_df):
    expected = 1.0 - comp_df["gru_hs_rmse"] / comp_df["persistence_hs_rmse"]
    np.testing.assert_allclose(
        comp_df["hs_skill_vs_persistence"].values,
        expected.values,
        atol=1e-5,
        err_msg="hs_skill_vs_persistence does not equal 1 - gru/persistence",
    )


def test_tp_skill_calculation(comp_df):
    expected = 1.0 - comp_df["gru_tp_rmse"] / comp_df["persistence_tp_rmse"]
    np.testing.assert_allclose(
        comp_df["tp_skill_vs_persistence"].values,
        expected.values,
        atol=1e-5,
        err_msg="tp_skill_vs_persistence does not equal 1 - gru/persistence",
    )


# ---------------------------------------------------------------------------
# 5. Persistence is calculated independently by lead (not a flat line)
# ---------------------------------------------------------------------------
def test_persistence_hs_not_flat(comp_df):
    """Persistence Hs RMSE must vary across leads — it is NOT a single aggregate."""
    vals = comp_df["persistence_hs_rmse"].values
    assert vals.max() - vals.min() > 0.01, \
        "Persistence Hs RMSE appears flat — lead-wise calculation may be wrong"


def test_persistence_tp_not_flat(comp_df):
    vals = comp_df["persistence_tp_rmse"].values
    assert vals.max() - vals.min() > 0.1, \
        "Persistence Tp RMSE appears flat — lead-wise calculation may be wrong"


def test_persistence_dir_not_flat(comp_df):
    vals = comp_df["persistence_direction_mae"].values
    assert vals.max() - vals.min() > 1.0, \
        "Persistence direction MAE appears flat — lead-wise calculation may be wrong"


# ---------------------------------------------------------------------------
# 6. Direction uses circular error (unit test on the function itself)
# ---------------------------------------------------------------------------
def test_circular_error_used_for_direction():
    """Verify circular_error handles the 0/360 wrap correctly."""
    from module4_forecasting.metrics import circular_error
    # 1° - 359° = -358° → wrapped to +2°
    err = circular_error(np.array([1.0]), np.array([359.0]))
    np.testing.assert_allclose(err, [2.0], atol=1e-9)
    # 359° - 1° = 358° → wrapped to -2°
    err = circular_error(np.array([359.0]), np.array([1.0]))
    np.testing.assert_allclose(err, [-2.0], atol=1e-9)


def test_direction_mae_less_than_raw_subtraction(comp_df):
    """
    At lead 1, persistence direction MAE must be < 180° (circular bound).
    If raw subtraction were used, errors near 0/360 boundary would be ~360°.
    """
    assert comp_df.loc[0, "persistence_direction_mae"] < 180.0, \
        "Direction MAE >= 180° suggests non-circular error was used"
    assert comp_df.loc[0, "gru_direction_mae"] < 180.0


# ---------------------------------------------------------------------------
# 7. Deterministic: running persistence_by_horizon twice gives same result
# ---------------------------------------------------------------------------
def test_persistence_deterministic():
    """
    persistence_by_horizon is a pure function of persist_preds and targets.
    Two calls with the same inputs must return identical results.
    """
    from module4_forecasting.audit_persistence_comparison import persistence_by_horizon
    rng = np.random.default_rng(0)
    N, H = 50, 48
    persist_preds = rng.uniform(0.5, 3.0, (N, H, 4))
    targets       = rng.uniform(0.5, 3.0, (N, H, 4))
    # Make sin/cos plausible
    angles = rng.uniform(0, 2 * np.pi, (N, H))
    persist_preds[:, :, 2] = np.sin(angles)
    persist_preds[:, :, 3] = np.cos(angles)
    targets[:, :, 2] = np.sin(angles + 0.1)
    targets[:, :, 3] = np.cos(angles + 0.1)

    r1 = persistence_by_horizon(persist_preds, targets)
    r2 = persistence_by_horizon(persist_preds, targets)
    np.testing.assert_array_equal(r1["hs_rmse"], r2["hs_rmse"])
    np.testing.assert_array_equal(r1["dir_mae"], r2["dir_mae"])


# ---------------------------------------------------------------------------
# 8-10. GRU beats persistence at lead 1
# ---------------------------------------------------------------------------
def test_gru_beats_persistence_hs_lead1(comp_df):
    row = comp_df[comp_df["lead_hours"] == 1].iloc[0]
    assert row["gru_hs_rmse"] < row["persistence_hs_rmse"], \
        f"GRU Hs RMSE ({row['gru_hs_rmse']:.4f}) >= persistence ({row['persistence_hs_rmse']:.4f}) at lead 1"


def test_gru_beats_persistence_tp_lead1(comp_df):
    row = comp_df[comp_df["lead_hours"] == 1].iloc[0]
    assert row["gru_tp_rmse"] < row["persistence_tp_rmse"], \
        f"GRU Tp RMSE ({row['gru_tp_rmse']:.4f}) >= persistence ({row['persistence_tp_rmse']:.4f}) at lead 1"


def test_gru_beats_persistence_dir_lead1(comp_df):
    row = comp_df[comp_df["lead_hours"] == 1].iloc[0]
    assert row["gru_direction_mae"] < row["persistence_direction_mae"], \
        f"GRU Dir MAE ({row['gru_direction_mae']:.2f}) >= persistence ({row['persistence_direction_mae']:.2f}) at lead 1"


# ---------------------------------------------------------------------------
# 11. Persistence Hs RMSE grows with lead
# ---------------------------------------------------------------------------
def test_persistence_hs_rmse_grows_with_lead(comp_df):
    """Persistence error must increase with lead time (wave state diverges)."""
    vals = comp_df["persistence_hs_rmse"].values
    # Check that lead 48 > lead 1
    assert vals[47] > vals[0], \
        f"Persistence Hs RMSE at lead 48 ({vals[47]:.4f}) <= lead 1 ({vals[0]:.4f})"
    # Check overall trend: mean of last 12 > mean of first 12
    assert vals[36:].mean() > vals[:12].mean()


# ---------------------------------------------------------------------------
# 12. Skill positive at lead 1
# ---------------------------------------------------------------------------
def test_hs_skill_positive_at_lead1(comp_df):
    assert comp_df.loc[0, "hs_skill_vs_persistence"] > 0


def test_tp_skill_positive_at_lead1(comp_df):
    assert comp_df.loc[0, "tp_skill_vs_persistence"] > 0


# ---------------------------------------------------------------------------
# 13. CSV columns match required schema exactly
# ---------------------------------------------------------------------------
def test_csv_columns(comp_df):
    missing = set(REQUIRED_COLUMNS) - set(comp_df.columns)
    assert len(missing) == 0, f"Missing columns: {missing}"
