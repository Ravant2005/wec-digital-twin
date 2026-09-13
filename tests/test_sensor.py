"""
tests/test_sensor.py — Tests for module3_sensor (Module 3).

Tests cover:
  1.  Initial state: SensorParameters defaults are valid
  2.  Gaussian Hs noise: mean ≈ 0, std ≈ configured
  3.  Gaussian Tp noise: mean ≈ 0, std ≈ configured
  4.  Gaussian direction noise: circular mean ≈ 0, std ≈ configured
  5.  Hs bias shifts mean correctly
  6.  Tp bias shifts mean correctly
  7.  Direction bias shifts circular mean correctly
  8.  Direction wraps correctly: 359+5→4, 1-5→356, 180+190→10, 270-300→330
  9.  Hs_obs >= 0 always
  10. Tp_obs > 0 always
  11. direction_obs in [0, 360) always
  12. Delay: observation[n] uses truth[n-delay]
  13. Delay: indices 0..delay-1 are unavailable (NaN, valid=False)
  14. missing_probability=1.0 → all missing
  15. missing_probability=0.0 → none missing
  16. Reproducibility: same seed → identical observations
  17. Different seeds → different observations
  18. LEAKAGE A: future truth cannot appear in observation history
  19. LEAKAGE B: model_input() contains only observable fields (no truth keys)
  20. LEAKAGE C: changing future truth after cutoff does not change past input
  21. LEAKAGE D: delay never causes future information to appear
  22. LEAKAGE E: missing values remain NaN (no imputation)
  23. True state is unchanged after observe()
  24. as_model_input() does not contain Hs_true, Tp_true, direction_true
  25. Zero noise + zero bias → obs equals truth (after delay)
  26. ERA5 real data passes through sensor without error
  27. SensorParameters validation rejects negative noise std
  28. SensorParameters validation rejects delay < 0
  29. SensorParameters validation rejects missing_probability outside [0,1]
  30. circular_direction_error utility is correct
"""

import numpy as np
import pytest

from module3_sensor import (
    SensorParameters,
    SensorObservation,
    WaveSensor,
    circular_direction_error,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_truth(n, Hs=1.5, Tp=8.0, direction=270.0):
    time      = np.arange(n, dtype=float)
    Hs_true   = np.full(n, Hs)
    Tp_true   = np.full(n, Tp)
    dir_true  = np.full(n, direction)
    return time, Hs_true, Tp_true, dir_true


def _sensor(seed=42, **kwargs):
    p = SensorParameters(random_seed=seed, **kwargs)
    return WaveSensor(p)


# ---------------------------------------------------------------------------
# 1. SensorParameters defaults are valid
# ---------------------------------------------------------------------------

def test_default_parameters_valid():
    p = SensorParameters()
    assert p.hs_noise_std >= 0
    assert p.tp_noise_std >= 0
    assert p.direction_noise_std >= 0
    assert p.delay_steps >= 0
    assert 0.0 <= p.missing_probability <= 1.0


# ---------------------------------------------------------------------------
# 2. Gaussian Hs noise: mean ≈ 0, std ≈ configured
# ---------------------------------------------------------------------------

def test_hs_noise_statistics():
    n = 5000
    std = 0.15
    s = _sensor(seed=0, hs_noise_std=std, tp_noise_std=0, direction_noise_std=0,
                missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Hs=2.0)
    obs = s.observe(t, H, T, D)
    err = obs.Hs_obs[obs.valid_Hs] - H[obs.valid_Hs]
    assert abs(np.mean(err)) < 3 * std / np.sqrt(n), "Hs noise mean not near 0"
    assert abs(np.std(err) - std) < 0.05 * std + 0.01, "Hs noise std wrong"


# ---------------------------------------------------------------------------
# 3. Gaussian Tp noise: mean ≈ 0, std ≈ configured
# ---------------------------------------------------------------------------

def test_tp_noise_statistics():
    n = 5000
    std = 0.5
    s = _sensor(seed=1, hs_noise_std=0, tp_noise_std=std, direction_noise_std=0,
                missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Tp=10.0)
    obs = s.observe(t, H, T, D)
    err = obs.Tp_obs[obs.valid_Tp] - T[obs.valid_Tp]
    assert abs(np.mean(err)) < 3 * std / np.sqrt(n), "Tp noise mean not near 0"
    assert abs(np.std(err) - std) < 0.05 * std + 0.01, "Tp noise std wrong"


# ---------------------------------------------------------------------------
# 4. Gaussian direction noise: circular mean ≈ 0, std ≈ configured
# ---------------------------------------------------------------------------

def test_direction_noise_statistics():
    n = 5000
    std = 10.0
    s = _sensor(seed=2, hs_noise_std=0, tp_noise_std=0, direction_noise_std=std,
                missing_probability=0.0)
    t, H, T, D = _constant_truth(n, direction=90.0)
    obs = s.observe(t, H, T, D)
    err = circular_direction_error(
        obs.direction_obs[obs.valid_direction],
        D[obs.valid_direction],
    )
    assert abs(np.mean(err)) < 3 * std / np.sqrt(n), "Direction noise mean not near 0"
    assert abs(np.std(err) - std) < 0.1 * std + 0.5, "Direction noise std wrong"


# ---------------------------------------------------------------------------
# 5. Hs bias shifts mean correctly
# ---------------------------------------------------------------------------

def test_hs_bias():
    n = 2000
    bias = 0.1
    s = _sensor(seed=3, hs_noise_std=0.0, tp_noise_std=0, direction_noise_std=0,
                hs_bias=bias, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Hs=1.5)
    obs = s.observe(t, H, T, D)
    mean_obs = np.mean(obs.Hs_obs[obs.valid_Hs])
    assert abs(mean_obs - (1.5 + bias)) < 1e-9, f"Hs bias wrong: {mean_obs}"


# ---------------------------------------------------------------------------
# 6. Tp bias shifts mean correctly
# ---------------------------------------------------------------------------

def test_tp_bias():
    n = 2000
    bias = 0.5
    s = _sensor(seed=4, hs_noise_std=0, tp_noise_std=0.0, direction_noise_std=0,
                tp_bias=bias, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Tp=8.0)
    obs = s.observe(t, H, T, D)
    mean_obs = np.mean(obs.Tp_obs[obs.valid_Tp])
    assert abs(mean_obs - (8.0 + bias)) < 1e-9, f"Tp bias wrong: {mean_obs}"


# ---------------------------------------------------------------------------
# 7. Direction bias shifts circular mean correctly
# ---------------------------------------------------------------------------

def test_direction_bias():
    n = 2000
    bias = 10.0
    s = _sensor(seed=5, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0.0,
                direction_bias=bias, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, direction=270.0)
    obs = s.observe(t, H, T, D)
    # With zero noise, every obs should be exactly 280°
    assert np.allclose(obs.direction_obs[obs.valid_direction], 280.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 8. Direction circular wrapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("true_dir,delta,expected", [
    (359.0,  5.0,   4.0),
    (  1.0, -5.0, 356.0),
    (180.0, 190.0,  10.0),
    (270.0, -300.0, 330.0),
])
def test_direction_circular_wrap(true_dir, delta, expected):
    n = 10
    s = _sensor(seed=6, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0.0,
                direction_bias=delta, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, direction=true_dir)
    obs = s.observe(t, H, T, D)
    result = obs.direction_obs[obs.valid_direction][0]
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"


# ---------------------------------------------------------------------------
# 9. Hs_obs >= 0 always
# ---------------------------------------------------------------------------

def test_hs_non_negative():
    # Use large negative bias to force clipping
    n = 500
    s = _sensor(seed=7, hs_noise_std=5.0, tp_noise_std=0, direction_noise_std=0,
                hs_bias=-10.0, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Hs=0.5)
    obs = s.observe(t, H, T, D)
    assert np.all(obs.Hs_obs[obs.valid_Hs] >= 0.0), "Hs_obs has negative values"
    assert obs.n_hs_clipped > 0, "Expected some Hs clipping events"


# ---------------------------------------------------------------------------
# 10. Tp_obs > 0 always
# ---------------------------------------------------------------------------

def test_tp_positive():
    n = 500
    s = _sensor(seed=8, hs_noise_std=0, tp_noise_std=20.0, direction_noise_std=0,
                tp_bias=-50.0, missing_probability=0.0)
    t, H, T, D = _constant_truth(n, Tp=1.0)
    obs = s.observe(t, H, T, D)
    assert np.all(obs.Tp_obs[obs.valid_Tp] > 0.0), "Tp_obs has non-positive values"
    assert obs.n_tp_clipped > 0, "Expected some Tp clipping events"


# ---------------------------------------------------------------------------
# 11. direction_obs in [0, 360) always
# ---------------------------------------------------------------------------

def test_direction_in_range():
    n = 1000
    s = _sensor(seed=9, hs_noise_std=0, tp_noise_std=0, direction_noise_std=90.0,
                missing_probability=0.0)
    t, H, T, D = _constant_truth(n, direction=180.0)
    obs = s.observe(t, H, T, D)
    d = obs.direction_obs[obs.valid_direction]
    assert np.all(d >= 0.0) and np.all(d < 360.0), "direction_obs out of [0,360)"


# ---------------------------------------------------------------------------
# 12. Delay: observation[n] uses truth[n-delay]
# ---------------------------------------------------------------------------

def test_delay_correct_source():
    delay = 2
    n = 10
    # Ramp truth so each index has a unique value
    time     = np.arange(n, dtype=float)
    Hs_true  = np.arange(1, n + 1, dtype=float)   # [1,2,3,...,10]
    Tp_true  = np.ones(n) * 8.0
    dir_true = np.zeros(n)

    s = _sensor(seed=10, hs_noise_std=0.0, tp_noise_std=0, direction_noise_std=0,
                delay_steps=delay, missing_probability=0.0)
    obs = s.observe(time, Hs_true, Tp_true, dir_true)

    for i in range(delay, n):
        assert obs.valid_Hs[i], f"Index {i} should be valid"
        assert obs.Hs_obs[i] == Hs_true[i - delay], (
            f"obs[{i}]={obs.Hs_obs[i]} != truth[{i-delay}]={Hs_true[i-delay]}"
        )


# ---------------------------------------------------------------------------
# 13. Delay: indices 0..delay-1 are unavailable
# ---------------------------------------------------------------------------

def test_delay_initial_unavailable():
    delay = 3
    n = 10
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=11, delay_steps=delay, missing_probability=0.0)
    obs = s.observe(t, H, T, D)

    for i in range(delay):
        assert not obs.valid_Hs[i],       f"Index {i} should be unavailable (Hs)"
        assert not obs.valid_Tp[i],       f"Index {i} should be unavailable (Tp)"
        assert not obs.valid_direction[i], f"Index {i} should be unavailable (dir)"
        assert np.isnan(obs.Hs_obs[i]),       f"Hs_obs[{i}] should be NaN"
        assert np.isnan(obs.Tp_obs[i]),       f"Tp_obs[{i}] should be NaN"
        assert np.isnan(obs.direction_obs[i]), f"dir_obs[{i}] should be NaN"


# ---------------------------------------------------------------------------
# 14. missing_probability=1.0 → all missing
# ---------------------------------------------------------------------------

def test_all_missing():
    n = 50
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=12, missing_probability=1.0)
    obs = s.observe(t, H, T, D)
    assert not np.any(obs.valid_Hs),       "All Hs should be missing"
    assert not np.any(obs.valid_Tp),       "All Tp should be missing"
    assert not np.any(obs.valid_direction), "All direction should be missing"


# ---------------------------------------------------------------------------
# 15. missing_probability=0.0 → none missing (after delay)
# ---------------------------------------------------------------------------

def test_none_missing():
    n = 50
    delay = 2
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=13, missing_probability=0.0, delay_steps=delay)
    obs = s.observe(t, H, T, D)
    # After delay, all should be valid
    assert np.all(obs.valid_Hs[delay:]),       "No Hs should be missing after delay"
    assert np.all(obs.valid_Tp[delay:]),       "No Tp should be missing after delay"
    assert np.all(obs.valid_direction[delay:]), "No direction should be missing after delay"


# ---------------------------------------------------------------------------
# 16. Reproducibility: same seed → identical observations
# ---------------------------------------------------------------------------

def test_reproducibility_same_seed():
    n = 100
    t, H, T, D = _constant_truth(n)
    s1 = _sensor(seed=42)
    s2 = _sensor(seed=42)
    obs1 = s1.observe(t, H, T, D)
    obs2 = s2.observe(t, H, T, D)
    np.testing.assert_array_equal(obs1.Hs_obs, obs2.Hs_obs)
    np.testing.assert_array_equal(obs1.Tp_obs, obs2.Tp_obs)
    np.testing.assert_array_equal(obs1.direction_obs, obs2.direction_obs)
    np.testing.assert_array_equal(obs1.valid_Hs, obs2.valid_Hs)


# ---------------------------------------------------------------------------
# 17. Different seeds → different observations
# ---------------------------------------------------------------------------

def test_different_seeds_differ():
    n = 100
    t, H, T, D = _constant_truth(n)
    s1 = _sensor(seed=1)
    s2 = _sensor(seed=2)
    obs1 = s1.observe(t, H, T, D)
    obs2 = s2.observe(t, H, T, D)
    # With noise, observations should differ
    assert not np.allclose(obs1.Hs_obs[obs1.valid_Hs], obs2.Hs_obs[obs2.valid_Hs]), \
        "Different seeds should produce different Hs observations"


# ---------------------------------------------------------------------------
# 18. LEAKAGE A: future truth cannot appear in observation history
# ---------------------------------------------------------------------------

def test_leakage_a_no_future_truth():
    """
    Observation at index n must only use truth at index n-delay.
    Verify by using a ramp truth: obs[n] == truth[n-delay] exactly.
    """
    delay = 2
    n = 20
    time     = np.arange(n, dtype=float)
    Hs_true  = np.arange(n, dtype=float) * 0.1   # unique per index
    Tp_true  = np.ones(n) * 8.0
    dir_true = np.zeros(n)

    s = _sensor(seed=0, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0,
                delay_steps=delay, missing_probability=0.0)
    obs = s.observe(time, Hs_true, Tp_true, dir_true)

    for i in range(delay, n):
        expected = Hs_true[i - delay]
        assert obs.Hs_obs[i] == expected, (
            f"obs[{i}]={obs.Hs_obs[i]:.4f} uses future truth "
            f"(expected truth[{i-delay}]={expected:.4f})"
        )
    # Confirm obs[n] != truth[n] for n > delay (would indicate no delay)
    for i in range(delay, n - 1):
        if Hs_true[i] != Hs_true[i - delay]:
            assert obs.Hs_obs[i] != Hs_true[i], \
                f"obs[{i}] equals current truth — delay not applied"


# ---------------------------------------------------------------------------
# 19. LEAKAGE B: as_model_input() contains only observable fields
# ---------------------------------------------------------------------------

def test_leakage_b_model_input_no_truth():
    n = 20
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=0)
    obs = s.observe(t, H, T, D)
    mi = obs.as_model_input()

    forbidden = {"Hs_true", "Tp_true", "direction_true"}
    for key in forbidden:
        assert key not in mi, f"as_model_input() must not contain '{key}'"

    required = {"time", "Hs_obs", "Tp_obs", "direction_obs",
                "valid_Hs", "valid_Tp", "valid_direction"}
    for key in required:
        assert key in mi, f"as_model_input() must contain '{key}'"


# ---------------------------------------------------------------------------
# 20. LEAKAGE C: changing future truth after cutoff does not change past input
# ---------------------------------------------------------------------------

def test_leakage_c_future_truth_immutable():
    """
    Modifying truth values after observation generation must not affect
    the already-generated observations.
    """
    n = 20
    t, H, T, D = _constant_truth(n)
    H_copy = H.copy()

    s = _sensor(seed=0, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0,
                missing_probability=0.0)
    obs = s.observe(t, H_copy, T, D)
    Hs_obs_before = obs.Hs_obs.copy()

    # Mutate the truth array after the fact
    H_copy[10:] = 999.0

    # Observations must be unchanged (they were computed from the original values)
    np.testing.assert_array_equal(obs.Hs_obs, Hs_obs_before,
        err_msg="Observations changed when future truth was mutated")


# ---------------------------------------------------------------------------
# 21. LEAKAGE D: delay never causes future information to appear
# ---------------------------------------------------------------------------

def test_leakage_d_delay_causal():
    """
    With delay=2, obs[n] must use truth[n-2], never truth[n] or truth[n+1].
    """
    delay = 2
    n = 15
    time     = np.arange(n, dtype=float)
    # Each index has a unique Hs value
    Hs_true  = np.arange(100, 100 + n, dtype=float)
    Tp_true  = np.ones(n) * 8.0
    dir_true = np.zeros(n)

    s = _sensor(seed=0, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0,
                delay_steps=delay, missing_probability=0.0)
    obs = s.observe(time, Hs_true, Tp_true, dir_true)

    for i in range(delay, n):
        # Must equal truth[i-delay], not truth[i] or truth[i+1]
        assert obs.Hs_obs[i] == Hs_true[i - delay], \
            f"obs[{i}] does not use truth[{i-delay}]"
        assert obs.Hs_obs[i] != Hs_true[i], \
            f"obs[{i}] uses current truth (no delay applied)"


# ---------------------------------------------------------------------------
# 22. LEAKAGE E: missing values remain NaN (no imputation)
# ---------------------------------------------------------------------------

def test_leakage_e_missing_stays_nan():
    n = 50
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=0, missing_probability=0.5)
    obs = s.observe(t, H, T, D)

    # Where valid=False, obs must be NaN
    assert np.all(np.isnan(obs.Hs_obs[~obs.valid_Hs])), \
        "Missing Hs values must be NaN"
    assert np.all(np.isnan(obs.Tp_obs[~obs.valid_Tp])), \
        "Missing Tp values must be NaN"
    assert np.all(np.isnan(obs.direction_obs[~obs.valid_direction])), \
        "Missing direction values must be NaN"


# ---------------------------------------------------------------------------
# 23. True state is unchanged after observe()
# ---------------------------------------------------------------------------

def test_true_state_unchanged():
    n = 50
    t, H, T, D = _constant_truth(n)
    H_orig = H.copy()
    T_orig = T.copy()
    D_orig = D.copy()

    s = _sensor(seed=0)
    obs = s.observe(t, H, T, D)

    np.testing.assert_array_equal(H, H_orig, err_msg="Hs_true was modified")
    np.testing.assert_array_equal(T, T_orig, err_msg="Tp_true was modified")
    np.testing.assert_array_equal(D, D_orig, err_msg="direction_true was modified")


# ---------------------------------------------------------------------------
# 24. as_model_input() does not contain truth keys
# ---------------------------------------------------------------------------

def test_model_input_excludes_truth_keys():
    n = 20
    t, H, T, D = _constant_truth(n)
    s = _sensor(seed=0)
    obs = s.observe(t, H, T, D)
    mi = obs.as_model_input()
    assert "Hs_true" not in mi
    assert "Tp_true" not in mi
    assert "direction_true" not in mi


# ---------------------------------------------------------------------------
# 25. Zero noise + zero bias → obs equals truth (after delay)
# ---------------------------------------------------------------------------

def test_zero_noise_zero_bias():
    n = 30
    delay = 1
    t, H, T, D = _constant_truth(n, Hs=2.0, Tp=9.0, direction=135.0)
    s = _sensor(seed=0, hs_noise_std=0, tp_noise_std=0, direction_noise_std=0,
                hs_bias=0, tp_bias=0, direction_bias=0,
                delay_steps=delay, missing_probability=0.0)
    obs = s.observe(t, H, T, D)

    for i in range(delay, n):
        assert obs.Hs_obs[i] == H[i - delay]
        assert obs.Tp_obs[i] == T[i - delay]
        assert obs.direction_obs[i] == D[i - delay]


# ---------------------------------------------------------------------------
# 26. ERA5 real data passes through sensor without error
# ---------------------------------------------------------------------------

def test_era5_data_passthrough():
    """ERA5 Goa data must pass through the sensor without exceptions."""
    try:
        from module1_ocean.era5 import load_era5, extract_goa_timeseries
        df = extract_goa_timeseries(load_era5("data/raw/era5_goa_jan2024.nc"))
    except Exception:
        pytest.skip("ERA5 data not available")

    time      = np.arange(len(df), dtype=float)
    Hs_true   = df["Hs_m"].values.astype(float)
    Tp_true   = df["Tp_s"].values.astype(float)
    dir_true  = df["direction_deg"].values.astype(float)

    # Replace NaN with reasonable values for sensor test
    Hs_true  = np.where(np.isnan(Hs_true),  1.0, Hs_true)
    Tp_true  = np.where(np.isnan(Tp_true),  8.0, Tp_true)
    dir_true = np.where(np.isnan(dir_true), 270.0, dir_true)

    p = SensorParameters(
        hs_noise_std=0.1, tp_noise_std=0.5, direction_noise_std=10.0,
        delay_steps=1, missing_probability=0.05, random_seed=0,
    )
    s = WaveSensor(p)
    obs = s.observe(time, Hs_true, Tp_true, dir_true)

    assert len(obs.Hs_obs) == len(time)
    assert np.all(obs.Hs_obs[obs.valid_Hs] >= 0.0)
    assert np.all(obs.Tp_obs[obs.valid_Tp] > 0.0)
    d = obs.direction_obs[obs.valid_direction]
    assert np.all(d >= 0.0) and np.all(d < 360.0)


# ---------------------------------------------------------------------------
# 27. SensorParameters rejects negative noise std
# ---------------------------------------------------------------------------

def test_invalid_noise_std():
    with pytest.raises(ValueError):
        SensorParameters(hs_noise_std=-0.1)
    with pytest.raises(ValueError):
        SensorParameters(tp_noise_std=-0.1)
    with pytest.raises(ValueError):
        SensorParameters(direction_noise_std=-1.0)


# ---------------------------------------------------------------------------
# 28. SensorParameters rejects delay < 0
# ---------------------------------------------------------------------------

def test_invalid_delay():
    with pytest.raises(ValueError):
        SensorParameters(delay_steps=-1)


# ---------------------------------------------------------------------------
# 29. SensorParameters rejects missing_probability outside [0,1]
# ---------------------------------------------------------------------------

def test_invalid_missing_probability():
    with pytest.raises(ValueError):
        SensorParameters(missing_probability=-0.1)
    with pytest.raises(ValueError):
        SensorParameters(missing_probability=1.1)


# ---------------------------------------------------------------------------
# 30. circular_direction_error utility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("obs,true,expected", [
    (4.0,   359.0,   5.0),
    (356.0,   1.0,  -5.0),
    (10.0,  180.0, -170.0),
    (330.0, 270.0,  60.0),
    (0.0,   0.0,    0.0),
    (180.0, 0.0,  180.0),
])
def test_circular_direction_error(obs, true, expected):
    result = float(circular_direction_error(np.array([obs]), np.array([true]))[0])
    assert abs(result - expected) < 1e-9, f"Expected {expected}, got {result}"
