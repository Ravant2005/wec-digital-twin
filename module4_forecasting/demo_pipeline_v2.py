"""
demo_pipeline_v2.py — Tier-1 forecasting data pipeline demonstration.

Shows:
  - Archive coverage (192 wave + 192 wind files)
  - Total record counts
  - Wave/wind alignment
  - Year-based train/val/test split
  - Feature names and shapes
  - Missing-data statistics
  - Baseline metrics on the full dataset

Does NOT train a neural network.
"""

import numpy as np
import pandas as pd

from module3_sensor.sensor import SensorParameters, WaveSensor
from module4_forecasting.era5_archive import load_era5_archive, validate_archive_coverage
from module4_forecasting.dataset_v2 import (
    ForecastConfigV2,
    make_year_split,
    build_datasets_v2,
    FEATURE_NAMES_WAVE,
    FEATURE_NAMES_WIND,
    TARGET_NAMES_V2,
)
from module4_forecasting.metrics import evaluate_forecast, evaluate_by_horizon, skill_score


def main():
    print("=" * 68)
    print("TIER-1 FORECASTING DATA PIPELINE — v2 DEMONSTRATION")
    print("=" * 68)

    # ------------------------------------------------------------------
    # 1. Archive coverage
    # ------------------------------------------------------------------
    print("\n--- ARCHIVE COVERAGE ---")
    cov = validate_archive_coverage()
    print(f"  Expected months  : {cov['expected_months']} per stream")
    print(f"  Wave present     : {len(cov['wave_present'])}")
    print(f"  Wave missing     : {len(cov['wave_missing'])}")
    print(f"  Wind present     : {len(cov['wind_present'])}")
    print(f"  Wind missing     : {len(cov['wind_missing'])}")
    print(f"  Complete         : {cov['complete']}")

    # ------------------------------------------------------------------
    # 2. Load archive
    # ------------------------------------------------------------------
    print("\n--- LOADING ERA5 ARCHIVE ---")
    df = load_era5_archive()
    print(f"  Total records    : {len(df):,}")
    print(f"  Columns          : {list(df.columns)}")
    t = pd.to_datetime(df["time"])
    print(f"  First timestamp  : {t.iloc[0]}")
    print(f"  Last timestamp   : {t.iloc[-1]}")
    print(f"  Hs range         : {df['Hs_m'].min():.2f} – {df['Hs_m'].max():.2f} m")
    print(f"  Tp range         : {df['Tp_s'].min():.2f} – {df['Tp_s'].max():.2f} s")
    print(f"  Direction range  : {df['direction_deg'].min():.1f} – {df['direction_deg'].max():.1f} deg")
    print(f"  u10 range        : {df['u10'].min():.2f} – {df['u10'].max():.2f} m/s")
    print(f"  v10 range        : {df['v10'].min():.2f} – {df['v10'].max():.2f} m/s")

    # ------------------------------------------------------------------
    # 3. Year-based split
    # ------------------------------------------------------------------
    print("\n--- YEAR-BASED SPLIT ---")
    split = make_year_split(df["time"].values)
    train_times = t.iloc[:split.train_end]
    val_times   = t.iloc[split.train_end:split.val_end]
    test_times  = t.iloc[split.val_end:]
    print(f"  TRAIN : {train_times.iloc[0]} → {train_times.iloc[-1]}")
    print(f"          {split.n_train:,} records")
    print(f"  VAL   : {val_times.iloc[0]} → {val_times.iloc[-1]}")
    print(f"          {split.n_val:,} records")
    print(f"  TEST  : {test_times.iloc[0]} → {test_times.iloc[-1]}")
    print(f"          {split.n_test:,} records")

    # ------------------------------------------------------------------
    # 4. Sensor model
    # ------------------------------------------------------------------
    print("\n--- MODULE 3 SENSOR ---")
    sensor_params = SensorParameters(
        hs_noise_std=0.15,
        tp_noise_std=0.8,
        direction_noise_std=15.0,
        delay_steps=1,
        missing_probability=0.05,
        random_seed=42,
    )
    sensor = WaveSensor(sensor_params)
    time_idx = np.arange(len(df), dtype=float)
    obs = sensor.observe(
        time=time_idx,
        Hs_true=df["Hs_m"].values,
        Tp_true=df["Tp_s"].values,
        direction_true=df["direction_deg"].values,
    )
    print(f"  Observations     : {len(obs.time):,}")
    print(f"  Missing fraction : {obs.missing_fraction:.3f}")
    print(f"  Hs clipped       : {obs.n_hs_clipped}")
    print(f"  Tp clipped       : {obs.n_tp_clipped}")
    valid_hs = obs.valid_Hs.sum()
    print(f"  Valid Hs obs     : {valid_hs:,} / {len(obs.time):,}")

    # ------------------------------------------------------------------
    # 5. wave_only dataset
    # ------------------------------------------------------------------
    print("\n--- WAVE-ONLY DATASET (48→48) ---")
    cfg_wave = ForecastConfigV2(
        input_length=48, forecast_horizon=48, feature_mode="wave_only"
    )
    train_w, val_w, test_w, scaler_w = build_datasets_v2(obs, cfg_wave, split)
    print(f"  Feature mode     : wave_only")
    print(f"  Feature names    : {FEATURE_NAMES_WAVE}")
    print(f"  Target names     : {TARGET_NAMES_V2}")
    print(f"  Train windows    : {len(train_w):,}")
    print(f"  Val windows      : {len(val_w):,}")
    print(f"  Test windows     : {len(test_w):,}")
    if len(train_w) > 0:
        s = train_w[0]
        print(f"  X shape/sample   : {s.X.shape}")
        print(f"  Target shape/smp : {s.target.shape}")
    print(f"  Scaler fitted    : {scaler_w.is_fitted}")
    if scaler_w.is_fitted:
        stats = scaler_w.fitted_stats
        print(f"  Hs  mean={stats['mean'][0]:.4f} m,  std={stats['std'][0]:.4f} m")
        print(f"  Tp  mean={stats['mean'][1]:.4f} s,  std={stats['std'][1]:.4f} s")

    # ------------------------------------------------------------------
    # 6. wave_wind dataset
    # ------------------------------------------------------------------
    print("\n--- WAVE+WIND DATASET (48→48) ---")
    cfg_wind = ForecastConfigV2(
        input_length=48, forecast_horizon=48, feature_mode="wave_wind"
    )
    train_ww, val_ww, test_ww, scaler_ww = build_datasets_v2(
        obs, cfg_wind, split,
        u10=df["u10"].values,
        v10=df["v10"].values,
    )
    print(f"  Feature mode     : wave_wind")
    print(f"  Feature names    : {FEATURE_NAMES_WIND}")
    print(f"  Train windows    : {len(train_ww):,}")
    print(f"  Val windows      : {len(val_ww):,}")
    print(f"  Test windows     : {len(test_ww):,}")
    if len(train_ww) > 0:
        s = train_ww[0]
        print(f"  X shape/sample   : {s.X.shape}")

    # ------------------------------------------------------------------
    # 7. Missing data statistics
    # ------------------------------------------------------------------
    print("\n--- MISSING DATA STATISTICS ---")
    if len(train_w) > 0:
        X_train = train_w.get_all_X()
        for i, name in enumerate(FEATURE_NAMES_WAVE[:4]):
            n_nan = int(np.sum(np.isnan(X_train[:, :, i])))
            total = X_train.shape[0] * X_train.shape[1]
            print(f"  {name:<20}: {n_nan:,} NaN / {total:,} ({100*n_nan/total:.2f}%)")

    # ------------------------------------------------------------------
    # 8. Persistence baseline on test set
    # ------------------------------------------------------------------
    print("\n--- PERSISTENCE BASELINE (test set, wave_only) ---")
    if len(test_w) > 0:
        # Build persistence predictions: repeat last valid obs for all horizons
        preds = []
        for i in range(len(test_w)):
            s = test_w[i]
            # Find last valid row in input window
            valid_rows = np.where(s.input_mask)[0]
            if len(valid_rows) == 0:
                preds.append(np.full((48, 4), np.nan))
                continue
            last = valid_rows[-1]
            row = s.X[last, :4].copy()  # Hs, Tp, sin_dir, cos_dir
            preds.append(np.tile(row, (48, 1)))
        preds = np.stack(preds, axis=0)
        targets = test_w.get_all_targets()

        from module4_forecasting.metrics import evaluate_forecast, evaluate_by_horizon
        m = evaluate_forecast(preds, targets)
        print(f"  Hs  MAE={m.hs_mae:.4f} m   RMSE={m.hs_rmse:.4f} m   bias={m.hs_bias:+.4f} m")
        print(f"  Tp  MAE={m.tp_mae:.4f} s   RMSE={m.tp_rmse:.4f} s   bias={m.tp_bias:+.4f} s")
        print(f"  Dir MAE={m.dir_mae:.2f} deg  RMSE={m.dir_rmse:.2f} deg")
        print(f"  n_samples={m.n_samples:,}  horizon={m.forecast_horizon}")

        hm = evaluate_by_horizon(preds, targets)
        print(f"\n  Horizon-dependent Hs RMSE (first 6 steps):")
        for h in range(min(6, len(hm.hs_rmse))):
            print(f"    h={h+1:2d}h  Hs_RMSE={hm.hs_rmse[h]:.4f} m")

    print("\n" + "=" * 68)
    print("PIPELINE v2 DEMONSTRATION COMPLETE")
    print("=" * 68)
    print("\nNext step: implement Module 4.2 LSTM/GRU forecaster.")


if __name__ == "__main__":
    main()
