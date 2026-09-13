"""
run_experiments.py — Train all Tier-1 LSTM/GRU experiments (Module 4.2).

Experiments
-----------
  exp_lstm_wave   : LSTM, wave_only (7 features)
  exp_lstm_wind   : LSTM, wave_wind (11 features)
  exp_gru_wave    : GRU,  wave_only (7 features)
  exp_gru_wind    : GRU,  wave_wind (11 features)

All experiments use identical hyperparameters, split, seed, and sensor
parameters.  The only difference is model_type and feature_mode.

Results are saved to:
  results/forecasting/<experiment_name>/

Run:
  PYTHONPATH=. .venv/bin/python module4_forecasting/run_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

from module3_sensor.sensor import SensorParameters, WaveSensor
from module4_forecasting.dataset_v2 import (
    ForecastConfigV2,
    build_datasets_v2,
    make_year_split,
    N_FEATURES_WAVE,
    N_FEATURES_WIND,
)
from module4_forecasting.era5_archive import load_era5_archive
from module4_forecasting.evaluation import (
    compare_baselines,
    evaluate,
    extract_error_distribution,
    run_climatology_v2,
    run_persistence_v2,
)
from module4_forecasting.metrics import evaluate_forecast
from module4_forecasting.training import TrainConfig, get_device, load_checkpoint, train

RESULTS_ROOT = "results/forecasting"

# ---------------------------------------------------------------------------
# Sensor parameters (same as demo_pipeline_v2.py)
# ---------------------------------------------------------------------------
SENSOR_PARAMS = SensorParameters(
    hs_noise_std=0.15,
    tp_noise_std=0.8,
    direction_noise_std=15.0,
    delay_steps=1,
    missing_probability=0.05,
    random_seed=42,
)

# ---------------------------------------------------------------------------
# Training hyperparameters (identical across all experiments)
# ---------------------------------------------------------------------------
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.1
BATCH_SIZE  = 128
LR          = 1e-3
MAX_EPOCHS  = 50
PATIENCE    = 7
SEED        = 42


def p(msg="", **kwargs):
    """Print with immediate flush."""
    print(msg, flush=True, **kwargs)


def main():
    t_total = time.time()

    p("=" * 68)
    p("MODULE 4.2 — TIER-1 LSTM/GRU WAVE FORECASTER")
    p("=" * 68)

    device = get_device()
    p(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load ERA5 archive and run sensor
    # ------------------------------------------------------------------
    p("\n--- Loading ERA5 archive ---")
    df = load_era5_archive()
    p(f"  Records: {len(df):,}")

    split = make_year_split(df["time"].values)
    p(f"  Train: {split.n_train:,}  Val: {split.n_val:,}  Test: {split.n_test:,}")

    sensor = WaveSensor(SENSOR_PARAMS)
    p("  Running sensor model...", end="")
    obs = sensor.observe(
        time=np.arange(len(df), dtype=float),
        Hs_true=df["Hs_m"].values,
        Tp_true=df["Tp_s"].values,
        direction_true=df["direction_deg"].values,
    )
    p(f" done.  Missing fraction: {obs.missing_fraction:.3f}")

    # ------------------------------------------------------------------
    # 2. Build datasets (wave_only and wave_wind)
    # ------------------------------------------------------------------
    p("\n--- Building datasets ---")
    p("  Building wave_only datasets (this takes ~1-2 min)...")
    cfg_wave = ForecastConfigV2(input_length=48, forecast_horizon=48,
                                feature_mode="wave_only")
    train_w, val_w, test_w, scaler_w = build_datasets_v2(obs, cfg_wave, split)
    p(f"  wave_only  train={len(train_w):,}  val={len(val_w):,}  test={len(test_w):,}")

    p("  Building wave_wind datasets (this takes ~1-2 min)...")
    cfg_wind = ForecastConfigV2(input_length=48, forecast_horizon=48,
                                feature_mode="wave_wind")
    train_ww, val_ww, test_ww, scaler_ww = build_datasets_v2(
        obs, cfg_wind, split,
        u10=df["u10"].values, v10=df["v10"].values,
    )
    p(f"  wave_wind  train={len(train_ww):,}  val={len(val_ww):,}  test={len(test_ww):,}")

    # ------------------------------------------------------------------
    # 3. Baselines on test set
    # ------------------------------------------------------------------
    p("\n--- Computing baselines ---")
    persist_preds = run_persistence_v2(test_w)
    clim_preds    = run_climatology_v2(train_w, test_w)
    p("  Getting test targets...", end="")
    targets_test  = test_w.get_all_targets()
    p(" done.")

    persist_m = evaluate_forecast(persist_preds, targets_test)
    clim_m    = evaluate_forecast(clim_preds,    targets_test)

    p(f"  Persistence  Hs RMSE={persist_m.hs_rmse:.4f} m  "
      f"Tp RMSE={persist_m.tp_rmse:.4f} s  Dir MAE={persist_m.dir_mae:.2f}°")
    p(f"  Climatology  Hs RMSE={clim_m.hs_rmse:.4f} m  "
      f"Tp RMSE={clim_m.tp_rmse:.4f} s  Dir MAE={clim_m.dir_mae:.2f}°")

    # ------------------------------------------------------------------
    # 4. Define experiments
    # ------------------------------------------------------------------
    experiments = [
        dict(name="exp_lstm_wave", model_type="lstm", feature_mode="wave_only",
             n_features=N_FEATURES_WAVE, train_ds=train_w, val_ds=val_w,
             test_ds=test_w, scaler=scaler_w),
        dict(name="exp_lstm_wind", model_type="lstm", feature_mode="wave_wind",
             n_features=N_FEATURES_WIND, train_ds=train_ww, val_ds=val_ww,
             test_ds=test_ww, scaler=scaler_ww),
        dict(name="exp_gru_wave",  model_type="gru",  feature_mode="wave_only",
             n_features=N_FEATURES_WAVE, train_ds=train_w, val_ds=val_w,
             test_ds=test_w, scaler=scaler_w),
        dict(name="exp_gru_wind",  model_type="gru",  feature_mode="wave_wind",
             n_features=N_FEATURES_WIND, train_ds=train_ww, val_ds=val_ww,
             test_ds=test_ww, scaler=scaler_ww),
    ]

    all_metrics = {
        "Persistence": persist_m,
        "Climatology": clim_m,
    }
    best_model_name = None
    best_hs_rmse    = float("inf")

    # ------------------------------------------------------------------
    # 5. Train each experiment
    # ------------------------------------------------------------------
    for exp in experiments:
        name = exp["name"]
        p(f"\n{'='*68}")
        p(f"EXPERIMENT: {name}")
        p(f"{'='*68}")

        results_dir = os.path.join(RESULTS_ROOT, name)
        os.makedirs(results_dir, exist_ok=True)

        cfg = TrainConfig(
            model_type=exp["model_type"],
            feature_mode=exp["feature_mode"],
            n_features=exp["n_features"],
            input_length=48,
            forecast_horizon=48,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=SEED,
            experiment_name=name,
        )

        result = train(
            train_ds=exp["train_ds"],
            val_ds=exp["val_ds"],
            scaler=exp["scaler"],
            config=cfg,
            results_dir=results_dir,
        )

        # --- Evaluate best checkpoint ---
        p(f"\n  Evaluating best checkpoint (epoch {result.best_epoch})...")
        model, _, scaler_loaded = load_checkpoint(
            result.checkpoint_path, device=device
        )
        agg, hm, preds = evaluate(
            model=model,
            test_ds=exp["test_ds"],
            scaler=scaler_loaded,
            results_dir=results_dir,
            device=device,
        )
        all_metrics[name] = agg

        if agg.hs_rmse < best_hs_rmse:
            best_hs_rmse    = agg.hs_rmse
            best_model_name = name

        # --- Error distribution ---
        p("  Extracting error distribution...", end="")
        targets_arr = exp["test_ds"].get_all_targets()
        extract_error_distribution(preds, targets_arr, exp["test_ds"], results_dir)
        p(" done.")

        p(f"  Hs RMSE={agg.hs_rmse:.4f} m  (persistence={persist_m.hs_rmse:.4f} m)")
        skill = 1.0 - agg.hs_rmse / persist_m.hs_rmse
        p(f"  Hs skill vs persistence: {skill:+.4f}")

    # ------------------------------------------------------------------
    # 6. Comparison table
    # ------------------------------------------------------------------
    p(f"\n{'='*68}")
    p("BASELINE COMPARISON TABLE")
    p(f"{'='*68}")
    comp_df = compare_baselines(all_metrics, RESULTS_ROOT)
    p(comp_df.to_string(index=False))
    p(f"\nBest model: {best_model_name}  (Hs RMSE={best_hs_rmse:.4f} m)")

    # ------------------------------------------------------------------
    # 7. Wave-only vs wave+wind comparison
    # ------------------------------------------------------------------
    p(f"\n{'='*68}")
    p("WAVE-ONLY vs WAVE+WIND")
    p(f"{'='*68}")
    for mtype in ["lstm", "gru"]:
        wave_key = f"exp_{mtype}_wave"
        wind_key = f"exp_{mtype}_wind"
        if wave_key in all_metrics and wind_key in all_metrics:
            mw  = all_metrics[wave_key]
            mww = all_metrics[wind_key]
            delta = mww.hs_rmse - mw.hs_rmse
            p(f"  {mtype.upper()} wave_only Hs RMSE={mw.hs_rmse:.4f}  "
              f"wave_wind Hs RMSE={mww.hs_rmse:.4f}  "
              f"delta={delta:+.4f} m  "
              f"({'wind helps' if delta < 0 else 'wind hurts or neutral'})")

    total_time = time.time() - t_total
    p(f"\nTotal runtime: {total_time/60:.1f} min")
    p(f"Results saved to: {RESULTS_ROOT}/")
    p("=" * 68)
    p("MODULE 4.2 TRAINING COMPLETE")
    p("=" * 68)


if __name__ == "__main__":
    main()
