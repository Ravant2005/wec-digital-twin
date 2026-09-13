"""
tests/test_forecasting_metrics.py — Metrics tests for Module 4.1.

Tests:
  1.  circular_error wraps correctly
  2.  circular_error known values
  3.  evaluate_forecast returns ForecastMetrics
  4.  evaluate_forecast Hs MAE is correct
  5.  evaluate_forecast Tp RMSE is correct
  6.  evaluate_forecast direction MAE is circular
  7.  evaluate_by_horizon returns correct horizon count
  8.  evaluate_by_horizon per-horizon MAE is correct
  9.  skill_score = 0 when model equals reference
  10. skill_score > 0 when model is better
  11. skill_score < 0 when model is worse
  12. Perfect forecast gives zero errors
  13. NaN predictions are handled gracefully
"""

import numpy as np
import pytest

from module4_forecasting import (
    evaluate_forecast, evaluate_by_horizon, circular_error, skill_score,
    direction_to_sincos, N_TARGETS,
)


def _make_targets(n=50, H=4, seed=0):
    rng = np.random.default_rng(seed)
    Hs  = np.abs(1.0 + 0.3 * rng.standard_normal((n, H)))
    Tp  = 8.0 + 0.5 * rng.standard_normal((n, H))
    deg = (270.0 + 10.0 * rng.standard_normal((n, H))) % 360.0
    sin_d, cos_d = direction_to_sincos(deg)
    return np.stack([Hs, Tp, sin_d, cos_d], axis=-1)


# ---------------------------------------------------------------------------
# 1. circular_error wraps correctly
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("pred,true,expected", [
    (4.0,   359.0,   5.0),
    (356.0,   1.0,  -5.0),
    (10.0,  180.0, -170.0),
    (330.0, 270.0,  60.0),
    (0.0,   0.0,    0.0),
    (180.0, 0.0,  180.0),
])
def test_circular_error_wrap(pred, true, expected):
    result = float(circular_error(np.array([pred]), np.array([true]))[0])
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# 2. circular_error known values
# ---------------------------------------------------------------------------

def test_circular_error_array():
    pred = np.array([0.0, 90.0, 180.0, 270.0])
    true = np.array([0.0,  0.0,   0.0,   0.0])
    err  = circular_error(pred, true)
    np.testing.assert_allclose(err, [0.0, 90.0, 180.0, -90.0], atol=1e-9)


# ---------------------------------------------------------------------------
# 3. evaluate_forecast returns ForecastMetrics
# ---------------------------------------------------------------------------

def test_evaluate_forecast_returns_metrics():
    targets = _make_targets()
    preds   = targets + 0.01  # small error
    m = evaluate_forecast(preds, targets)
    assert hasattr(m, "hs_mae")
    assert hasattr(m, "tp_rmse")
    assert hasattr(m, "dir_mae")
    assert m.n_samples == 50
    assert m.forecast_horizon == 4


# ---------------------------------------------------------------------------
# 4. evaluate_forecast Hs MAE is correct
# ---------------------------------------------------------------------------

def test_evaluate_forecast_hs_mae():
    n, H = 100, 3
    targets = _make_targets(n=n, H=H)
    # Add known constant error to Hs
    preds = targets.copy()
    preds[:, :, 0] += 0.2
    m = evaluate_forecast(preds, targets)
    assert abs(m.hs_mae - 0.2) < 1e-9, f"Expected Hs MAE=0.2, got {m.hs_mae}"
    assert abs(m.hs_bias - 0.2) < 1e-9, f"Expected Hs bias=0.2, got {m.hs_bias}"


# ---------------------------------------------------------------------------
# 5. evaluate_forecast Tp RMSE is correct
# ---------------------------------------------------------------------------

def test_evaluate_forecast_tp_rmse():
    n, H = 100, 3
    targets = _make_targets(n=n, H=H)
    preds = targets.copy()
    preds[:, :, 1] += 1.0   # constant Tp error
    m = evaluate_forecast(preds, targets)
    assert abs(m.tp_rmse - 1.0) < 1e-9, f"Expected Tp RMSE=1.0, got {m.tp_rmse}"


# ---------------------------------------------------------------------------
# 6. evaluate_forecast direction MAE is circular
# ---------------------------------------------------------------------------

def test_evaluate_forecast_direction_circular():
    n, H = 50, 2
    # True direction near 0°, predicted near 358° — circular error should be ~2°
    sin_true, cos_true = direction_to_sincos(np.full((n, H), 1.0))
    sin_pred, cos_pred = direction_to_sincos(np.full((n, H), 359.0))
    targets = np.stack([np.ones((n, H)), np.ones((n, H)) * 8.0,
                        sin_true, cos_true], axis=-1)
    preds   = np.stack([np.ones((n, H)), np.ones((n, H)) * 8.0,
                        sin_pred, cos_pred], axis=-1)
    m = evaluate_forecast(preds, targets)
    assert abs(m.dir_mae - 2.0) < 0.01, \
        f"Circular direction MAE should be ~2°, got {m.dir_mae}"


# ---------------------------------------------------------------------------
# 7. evaluate_by_horizon returns correct horizon count
# ---------------------------------------------------------------------------

def test_evaluate_by_horizon_count():
    H = 6
    targets = _make_targets(H=H)
    preds   = targets.copy()
    hm = evaluate_by_horizon(preds, targets)
    assert len(hm.hs_mae)  == H
    assert len(hm.tp_rmse) == H
    assert len(hm.dir_mae) == H
    assert hm.horizons == list(range(1, H + 1))


# ---------------------------------------------------------------------------
# 8. evaluate_by_horizon per-horizon MAE is correct
# ---------------------------------------------------------------------------

def test_evaluate_by_horizon_per_step():
    n, H = 100, 4
    targets = _make_targets(n=n, H=H)
    preds   = targets.copy()
    # Add horizon-dependent error: error at horizon h = h * 0.1
    for h in range(H):
        preds[:, h, 0] += (h + 1) * 0.1
    hm = evaluate_by_horizon(preds, targets)
    for h in range(H):
        expected = (h + 1) * 0.1
        assert abs(hm.hs_mae[h] - expected) < 1e-9, \
            f"Horizon {h+1}: expected MAE={expected}, got {hm.hs_mae[h]}"


# ---------------------------------------------------------------------------
# 9. skill_score = 0 when model equals reference
# ---------------------------------------------------------------------------

def test_skill_score_zero():
    assert skill_score(1.0, 1.0) == 0.0


# ---------------------------------------------------------------------------
# 10. skill_score > 0 when model is better
# ---------------------------------------------------------------------------

def test_skill_score_positive():
    assert skill_score(0.5, 1.0) == 0.5


# ---------------------------------------------------------------------------
# 11. skill_score < 0 when model is worse
# ---------------------------------------------------------------------------

def test_skill_score_negative():
    assert skill_score(2.0, 1.0) == -1.0


# ---------------------------------------------------------------------------
# 12. Perfect forecast gives zero errors
# ---------------------------------------------------------------------------

def test_perfect_forecast_zero_error():
    targets = _make_targets()
    preds   = targets.copy()
    m = evaluate_forecast(preds, targets)
    assert abs(m.hs_mae)  < 1e-9
    assert abs(m.tp_mae)  < 1e-9
    assert abs(m.dir_mae) < 1e-6


# ---------------------------------------------------------------------------
# 13. NaN predictions are handled gracefully
# ---------------------------------------------------------------------------

def test_nan_predictions_handled():
    targets = _make_targets(n=20, H=3)
    preds   = targets.copy()
    preds[:10, :, 0] = np.nan   # half the Hs predictions are NaN
    m = evaluate_forecast(preds, targets)
    assert np.isfinite(m.hs_mae), "NaN predictions should be ignored, not crash"
