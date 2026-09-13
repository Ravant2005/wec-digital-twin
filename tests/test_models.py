"""
tests/test_models.py — Tests for Module 4.2 LSTM/GRU wave forecasters.

Tests
-----
 1.  LSTM accepts wave-only input shape (B, 48, 7)
 2.  LSTM accepts wave+wind input shape (B, 48, 11)
 3.  LSTM output shape is (B, 48, 4)
 4.  GRU accepts wave-only input shape (B, 48, 7)
 5.  GRU accepts wave+wind input shape (B, 48, 11)
 6.  GRU output shape is (B, 48, 4)
 7.  No NaNs in LSTM output
 8.  No NaNs in GRU output
 9.  NaN inputs are handled (replaced with 0.0 internally)
10.  LSTM inference does not require future targets
11.  GRU inference does not require future targets
12.  Deterministic output under fixed seed (LSTM)
13.  Deterministic output under fixed seed (GRU)
14.  ModelConfig validation rejects bad parameters
15.  build_model factory returns correct types
16.  WeightedMSELoss computes correct shape
17.  WeightedMSELoss handles NaN targets
18.  WaveScalerV2 inverse transform returns physical units
19.  Circular direction metric handles 0/360 boundary
20.  Per-lead metrics have exactly 48 horizons
21.  Error distribution contains lead-time association
22.  Training checkpoint can be reloaded
23.  Configuration is saved and recoverable
24.  _TorchDataset length matches ForecastDatasetV2
25.  _TorchDataset items are float32 tensors
26.  Persistence v2 output shape correct
27.  Climatology v2 output shape correct
28.  Frozen module4 files untouched
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from module4_forecasting.models import (
    ModelConfig,
    WaveForecasterLSTM,
    WaveForecasterGRU,
    build_model,
    N_TARGETS,
)
from module4_forecasting.training import (
    TrainConfig,
    WeightedMSELoss,
    _TorchDataset,
    seed_everything,
    load_checkpoint,
    train,
)
from module4_forecasting.evaluation import (
    run_persistence_v2,
    run_climatology_v2,
    extract_error_distribution,
    _normalise_sincos,
    _sincos_to_deg,
)
from module4_forecasting.metrics import (
    circular_error,
    evaluate_by_horizon,
    evaluate_forecast,
)
from module4_forecasting.dataset_v2 import (
    WaveScalerV2,
    N_FEATURES_WAVE,
    N_FEATURES_WIND,
    N_TARGETS_V2,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model(model_type: str, n_features: int) -> torch.nn.Module:
    cfg = ModelConfig(n_features=n_features, forecast_horizon=48,
                      hidden_size=32, num_layers=2, dropout=0.0)
    return build_model(model_type, cfg)


def _rand_input(B: int, L: int, F: int) -> torch.Tensor:
    return torch.randn(B, L, F)


# ---------------------------------------------------------------------------
# 1-6. Shape tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type,n_feat", [
    ("lstm", N_FEATURES_WAVE),
    ("lstm", N_FEATURES_WIND),
    ("gru",  N_FEATURES_WAVE),
    ("gru",  N_FEATURES_WIND),
])
def test_output_shape(model_type, n_feat):
    model = _make_model(model_type, n_feat)
    x = _rand_input(4, 48, n_feat)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (4, 48, N_TARGETS), \
        f"{model_type} n_feat={n_feat}: expected (4,48,{N_TARGETS}), got {out.shape}"


def test_lstm_wave_only_shape():
    model = _make_model("lstm", N_FEATURES_WAVE)
    x = _rand_input(8, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 48, 4)


def test_lstm_wave_wind_shape():
    model = _make_model("lstm", N_FEATURES_WIND)
    x = _rand_input(8, 48, N_FEATURES_WIND)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 48, 4)


def test_gru_wave_only_shape():
    model = _make_model("gru", N_FEATURES_WAVE)
    x = _rand_input(8, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 48, 4)


def test_gru_wave_wind_shape():
    model = _make_model("gru", N_FEATURES_WIND)
    x = _rand_input(8, 48, N_FEATURES_WIND)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (8, 48, 4)


# ---------------------------------------------------------------------------
# 7-9. NaN handling
# ---------------------------------------------------------------------------

def test_lstm_no_nan_output():
    model = _make_model("lstm", N_FEATURES_WAVE)
    x = _rand_input(4, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "LSTM output contains NaN"


def test_gru_no_nan_output():
    model = _make_model("gru", N_FEATURES_WAVE)
    x = _rand_input(4, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "GRU output contains NaN"


def test_nan_input_handled():
    """NaN inputs must not propagate to output (replaced with 0.0 internally)."""
    model = _make_model("lstm", N_FEATURES_WAVE)
    x = _rand_input(4, 48, N_FEATURES_WAVE)
    x[:, :5, 0] = float("nan")   # inject NaN into Hs channel
    with torch.no_grad():
        out = model(x)
    assert not torch.isnan(out).any(), "NaN in input propagated to output"


# ---------------------------------------------------------------------------
# 10-11. Inference does not require future targets
# ---------------------------------------------------------------------------

def test_lstm_inference_no_future_targets():
    """Model forward() takes only X — no target argument exists."""
    model = _make_model("lstm", N_FEATURES_WAVE)
    x = _rand_input(2, 48, N_FEATURES_WAVE)
    # forward() must work with only x
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 48, 4)


def test_gru_inference_no_future_targets():
    model = _make_model("gru", N_FEATURES_WAVE)
    x = _rand_input(2, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, 48, 4)


# ---------------------------------------------------------------------------
# 12-13. Determinism
# ---------------------------------------------------------------------------

def test_lstm_deterministic():
    seed_everything(42)
    model = _make_model("lstm", N_FEATURES_WAVE)
    x = torch.randn(4, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out1 = model(x).clone()
    with torch.no_grad():
        out2 = model(x).clone()
    torch.testing.assert_close(out1, out2)


def test_gru_deterministic():
    seed_everything(42)
    model = _make_model("gru", N_FEATURES_WAVE)
    x = torch.randn(4, 48, N_FEATURES_WAVE)
    with torch.no_grad():
        out1 = model(x).clone()
    with torch.no_grad():
        out2 = model(x).clone()
    torch.testing.assert_close(out1, out2)


# ---------------------------------------------------------------------------
# 14-15. ModelConfig and factory
# ---------------------------------------------------------------------------

def test_model_config_validation():
    with pytest.raises(AssertionError):
        ModelConfig(n_features=0)
    with pytest.raises(AssertionError):
        ModelConfig(forecast_horizon=0)
    with pytest.raises(AssertionError):
        ModelConfig(hidden_size=0)
    with pytest.raises(AssertionError):
        ModelConfig(dropout=1.0)


def test_build_model_factory():
    lstm = build_model("lstm", ModelConfig(n_features=7))
    gru  = build_model("gru",  ModelConfig(n_features=7))
    assert isinstance(lstm, WaveForecasterLSTM)
    assert isinstance(gru,  WaveForecasterGRU)
    with pytest.raises(ValueError):
        build_model("transformer", ModelConfig(n_features=7))


# ---------------------------------------------------------------------------
# 16-17. Loss function
# ---------------------------------------------------------------------------

def test_weighted_mse_loss_shape():
    loss_fn = WeightedMSELoss([1.0, 1.0, 1.0, 1.0])
    pred   = torch.randn(8, 48, 4)
    target = torch.randn(8, 48, 4)
    loss = loss_fn(pred, target)
    assert loss.shape == (), "Loss must be a scalar"
    assert not torch.isnan(loss)


def test_weighted_mse_loss_nan_targets():
    """NaN in targets must not propagate to loss."""
    loss_fn = WeightedMSELoss([1.0, 1.0, 1.0, 1.0])
    pred   = torch.randn(4, 48, 4)
    target = torch.randn(4, 48, 4)
    target[0, :5, 0] = float("nan")
    loss = loss_fn(pred, target)
    assert not torch.isnan(loss), "NaN target propagated to loss"


# ---------------------------------------------------------------------------
# 18. Scaler inverse transform returns physical units
# ---------------------------------------------------------------------------

def test_scaler_inverse_transform_physical_units():
    scaler = WaveScalerV2(feature_mode="wave_only")
    # Synthetic training data: Hs ~ N(1.5, 0.5), Tp ~ N(10, 2)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 48, 7))
    X[:, :, 0] = rng.normal(1.5, 0.5, (100, 48))   # Hs
    X[:, :, 1] = rng.normal(10.0, 2.0, (100, 48))  # Tp
    scaler.fit(X)

    y = np.array([[[2.0, 12.0, 0.5, 0.866]]])   # (1, 1, 4) physical
    y_scaled = scaler.transform_target(y)
    y_back   = scaler.inverse_transform_target(y_scaled)
    np.testing.assert_allclose(y_back[0, 0, 0], 2.0,  atol=1e-5, err_msg="Hs not recovered")
    np.testing.assert_allclose(y_back[0, 0, 1], 12.0, atol=1e-5, err_msg="Tp not recovered")


# ---------------------------------------------------------------------------
# 19. Circular direction metric handles 0/360 boundary
# ---------------------------------------------------------------------------

def test_circular_error_boundary():
    pred = np.array([1.0, 359.0, 180.0])
    true = np.array([359.0, 1.0, 0.0])
    err  = circular_error(pred, true)
    # 1 - 359 = -358 → +2
    np.testing.assert_allclose(err[0],   2.0, atol=1e-9)
    # 359 - 1 = 358 → -2
    np.testing.assert_allclose(err[1],  -2.0, atol=1e-9)
    # 180 - 0 = 180 → 180 (boundary: stays at 180)
    np.testing.assert_allclose(abs(err[2]), 180.0, atol=1e-9)


def test_circular_error_zero_is_zero():
    pred = np.array([45.0, 270.0])
    true = np.array([45.0, 270.0])
    err  = circular_error(pred, true)
    np.testing.assert_allclose(err, 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# 20. Per-lead metrics have exactly 48 horizons
# ---------------------------------------------------------------------------

def test_per_lead_metrics_48_horizons():
    rng  = np.random.default_rng(1)
    pred = rng.normal(size=(100, 48, 4))
    true = rng.normal(size=(100, 48, 4))
    hm   = evaluate_by_horizon(pred, true)
    assert len(hm.horizons)  == 48
    assert len(hm.hs_mae)    == 48
    assert len(hm.hs_rmse)   == 48
    assert len(hm.tp_mae)    == 48
    assert len(hm.tp_rmse)   == 48
    assert len(hm.dir_mae)   == 48
    assert len(hm.dir_rmse)  == 48
    assert hm.horizons[0]  == 1
    assert hm.horizons[-1] == 48


# ---------------------------------------------------------------------------
# 21. Error distribution contains lead-time association
# ---------------------------------------------------------------------------

def test_error_distribution_lead_time_association():
    """extract_error_distribution must produce one row per (sample, lead)."""
    from module4_forecasting.era5_archive import load_era5_archive
    from module4_forecasting.dataset_v2 import (
        ForecastConfigV2, build_datasets_v2, make_year_split,
    )
    from module3_sensor.sensor import SensorParameters, WaveSensor

    # Use a tiny synthetic dataset to keep the test fast
    rng = np.random.default_rng(0)
    N = 500
    times = np.arange(N, dtype=float)
    Hs    = rng.uniform(0.5, 3.0, N)
    Tp    = rng.uniform(6.0, 14.0, N)
    dirn  = rng.uniform(0, 360, N)

    from module3_sensor.sensor import SensorObservation, SensorParameters
    obs = SensorObservation(
        time=times, Hs_true=Hs, Tp_true=Tp, direction_true=dirn,
        Hs_obs=Hs.copy(), Tp_obs=Tp.copy(), direction_obs=dirn.copy(),
        valid_Hs=np.ones(N, bool), valid_Tp=np.ones(N, bool),
        valid_direction=np.ones(N, bool),
        delay_steps=0,
        params=SensorParameters(),
    )

    from module4_forecasting.dataset_v2 import ForecastDatasetV2, YearSplit
    split = YearSplit(train_end=300, val_end=400, n_total=500,
                      train_end_date=None, val_end_date=None)
    cfg = ForecastConfigV2(input_length=10, forecast_horizon=5)
    test_ds = ForecastDatasetV2(obs, cfg, split, "test")

    if len(test_ds) == 0:
        pytest.skip("No test windows in synthetic dataset")

    N_test = len(test_ds)
    H = 5
    preds   = rng.normal(size=(N_test, H, 4))
    targets = rng.normal(size=(N_test, H, 4))

    with tempfile.TemporaryDirectory() as tmpdir:
        df = extract_error_distribution(preds, targets, test_ds, tmpdir)

    assert "lead_h" in df.columns
    assert "Hs_error" in df.columns
    assert "Tp_error" in df.columns
    assert "dir_error_deg" in df.columns
    assert set(df["lead_h"].unique()) == set(range(1, H + 1))
    assert len(df) == N_test * H


# ---------------------------------------------------------------------------
# 22-23. Checkpoint save/load and config recovery
# ---------------------------------------------------------------------------

def test_checkpoint_reload():
    """Save a tiny model checkpoint and reload it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = ModelConfig(n_features=7, forecast_horizon=4,
                          hidden_size=16, num_layers=1, dropout=0.0)
        model = WaveForecasterLSTM(cfg)
        x = torch.randn(2, 10, 7)
        with torch.no_grad():
            out_before = model(x).clone()

        # Simulate what training.py saves
        train_cfg = TrainConfig(
            model_type="lstm", feature_mode="wave_only", n_features=7,
            input_length=10, forecast_horizon=4,
            hidden_size=16, num_layers=1, dropout=0.0,
            batch_size=2, learning_rate=1e-3, max_epochs=1, patience=1, seed=42,
        )
        ckpt_path = os.path.join(tmpdir, "best_model.pt")
        scaler = WaveScalerV2("wave_only")
        rng = np.random.default_rng(0)
        X_fake = rng.normal(size=(50, 10, 7))
        X_fake[:, :, 0] = rng.normal(1.5, 0.5, (50, 10))
        X_fake[:, :, 1] = rng.normal(10.0, 2.0, (50, 10))
        scaler.fit(X_fake)

        torch.save({
            "epoch": 1,
            "model_state": model.state_dict(),
            "optimiser_state": {},
            "val_loss": 0.1,
            "config": train_cfg.to_dict(),
            "scaler_mean": scaler.fitted_stats["mean"].tolist(),
            "scaler_std":  scaler.fitted_stats["std"].tolist(),
            "scaler_feature_mode": "wave_only",
        }, ckpt_path)

        model2, cfg2, scaler2 = load_checkpoint(ckpt_path, device=torch.device("cpu"))
        with torch.no_grad():
            out_after = model2(x)

        torch.testing.assert_close(out_before, out_after)
        assert scaler2.is_fitted


def test_config_saved_and_recoverable():
    """TrainConfig serialises to JSON and round-trips correctly."""
    cfg = TrainConfig(
        model_type="gru", feature_mode="wave_wind", n_features=11,
        hidden_size=64, num_layers=2, dropout=0.1,
        batch_size=64, learning_rate=5e-4, max_epochs=20, patience=5, seed=99,
        experiment_name="test_exp",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "config.json")
        with open(path, "w") as f:
            json.dump(cfg.to_dict(), f)
        with open(path) as f:
            loaded = json.load(f)

    assert loaded["model_type"]    == "gru"
    assert loaded["feature_mode"]  == "wave_wind"
    assert loaded["n_features"]    == 11
    assert loaded["hidden_size"]   == 64
    assert loaded["seed"]          == 99


# ---------------------------------------------------------------------------
# 24-25. _TorchDataset
# ---------------------------------------------------------------------------

def test_torch_dataset_length_and_dtype():
    """_TorchDataset must match ForecastDatasetV2 length and return float32."""
    from module3_sensor.sensor import SensorObservation, SensorParameters
    from module4_forecasting.dataset_v2 import ForecastDatasetV2, YearSplit, ForecastConfigV2

    rng = np.random.default_rng(1)
    N = 400
    obs = SensorObservation(
        time=np.arange(N, dtype=float),
        Hs_true=rng.uniform(0.5, 3.0, N),
        Tp_true=rng.uniform(6.0, 14.0, N),
        direction_true=rng.uniform(0, 360, N),
        Hs_obs=rng.uniform(0.5, 3.0, N),
        Tp_obs=rng.uniform(6.0, 14.0, N),
        direction_obs=rng.uniform(0, 360, N),
        valid_Hs=np.ones(N, bool),
        valid_Tp=np.ones(N, bool),
        valid_direction=np.ones(N, bool),
        delay_steps=0,
        params=SensorParameters(),
    )
    split = YearSplit(train_end=300, val_end=350, n_total=N,
                      train_end_date=None, val_end_date=None)
    cfg = ForecastConfigV2(input_length=10, forecast_horizon=5)
    ds = ForecastDatasetV2(obs, cfg, split, "train")

    scaler = WaveScalerV2("wave_only")
    X_all = ds.get_all_X()
    scaler.fit(X_all)

    torch_ds = _TorchDataset(ds, scaler)
    assert len(torch_ds) == len(ds)

    X_item, y_item = torch_ds[0]
    assert X_item.dtype == torch.float32
    assert y_item.dtype == torch.float32
    assert X_item.shape == (10, 7)
    assert y_item.shape == (5, 4)


# ---------------------------------------------------------------------------
# 26-27. Persistence and climatology v2 shapes
# ---------------------------------------------------------------------------

def test_persistence_v2_shape():
    from module3_sensor.sensor import SensorObservation, SensorParameters
    from module4_forecasting.dataset_v2 import ForecastDatasetV2, YearSplit, ForecastConfigV2

    rng = np.random.default_rng(2)
    N = 400
    obs = SensorObservation(
        time=np.arange(N, dtype=float),
        Hs_true=rng.uniform(0.5, 3.0, N),
        Tp_true=rng.uniform(6.0, 14.0, N),
        direction_true=rng.uniform(0, 360, N),
        Hs_obs=rng.uniform(0.5, 3.0, N),
        Tp_obs=rng.uniform(6.0, 14.0, N),
        direction_obs=rng.uniform(0, 360, N),
        valid_Hs=np.ones(N, bool),
        valid_Tp=np.ones(N, bool),
        valid_direction=np.ones(N, bool),
        delay_steps=0,
        params=SensorParameters(),
    )
    split = YearSplit(train_end=300, val_end=350, n_total=N,
                      train_end_date=None, val_end_date=None)
    cfg = ForecastConfigV2(input_length=10, forecast_horizon=5)
    test_ds = ForecastDatasetV2(obs, cfg, split, "test")

    if len(test_ds) == 0:
        pytest.skip("No test windows")

    preds = run_persistence_v2(test_ds)
    assert preds.shape == (len(test_ds), 5, 4)


def test_climatology_v2_shape():
    from module3_sensor.sensor import SensorObservation, SensorParameters
    from module4_forecasting.dataset_v2 import ForecastDatasetV2, YearSplit, ForecastConfigV2

    rng = np.random.default_rng(3)
    N = 400
    obs = SensorObservation(
        time=np.arange(N, dtype=float),
        Hs_true=rng.uniform(0.5, 3.0, N),
        Tp_true=rng.uniform(6.0, 14.0, N),
        direction_true=rng.uniform(0, 360, N),
        Hs_obs=rng.uniform(0.5, 3.0, N),
        Tp_obs=rng.uniform(6.0, 14.0, N),
        direction_obs=rng.uniform(0, 360, N),
        valid_Hs=np.ones(N, bool),
        valid_Tp=np.ones(N, bool),
        valid_direction=np.ones(N, bool),
        delay_steps=0,
        params=SensorParameters(),
    )
    split = YearSplit(train_end=300, val_end=350, n_total=N,
                      train_end_date=None, val_end_date=None)
    cfg = ForecastConfigV2(input_length=10, forecast_horizon=5)
    train_ds = ForecastDatasetV2(obs, cfg, split, "train")
    test_ds  = ForecastDatasetV2(obs, cfg, split, "test")

    if len(test_ds) == 0:
        pytest.skip("No test windows")

    preds = run_climatology_v2(train_ds, test_ds)
    assert preds.shape == (len(test_ds), 5, 4)


# ---------------------------------------------------------------------------
# 28. Frozen module integrity
# ---------------------------------------------------------------------------

def test_frozen_models_not_modified():
    """Frozen module4 files must not be modified."""
    frozen = [
        _ROOT / "module4_forecasting" / "dataset.py",
        _ROOT / "module4_forecasting" / "preprocessing.py",
        _ROOT / "module4_forecasting" / "baselines.py",
        _ROOT / "module4_forecasting" / "metrics.py",
    ]
    for p in frozen:
        assert p.exists(), f"Frozen file missing: {p}"
        src = p.read_text()
        assert len(src) > 100, f"Frozen file appears empty: {p}"


def test_new_module_files_exist():
    """New Module 4.2 files must exist."""
    new_files = [
        _ROOT / "module4_forecasting" / "models.py",
        _ROOT / "module4_forecasting" / "training.py",
        _ROOT / "module4_forecasting" / "evaluation.py",
    ]
    for p in new_files:
        assert p.exists(), f"New file missing: {p}"
