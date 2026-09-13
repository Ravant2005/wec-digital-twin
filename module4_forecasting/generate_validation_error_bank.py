"""
generate_validation_error_bank.py — Module 5.3A-DATA

Generate a leakage-safe forecast-error bank from the 2022–2023 validation period
for use during realistic-forecast PPO training (Module 5.3).

============================================================
SCIENTIFIC PURPOSE
============================================================

The existing forecast_errors.csv.gz was produced from the 2024–2025 TEST period
(origin_idx 122759–140207).  Using that bank during RL training would inject
test-period error statistics into the training process — a methodological concern
identified in Module 5.3A design review.

This script generates an equivalent bank using ONLY the 2022–2023 VALIDATION
period, which is chronologically between training (2010–2021) and the held-out
test set (2024–2025).

The 2024–2025 bank remains completely untouched and reserved for final evaluation.

============================================================
EXACT METHODOLOGY
============================================================

Reproduces the canonical Module 4.2 inference pipeline from run_experiments.py:

    1. Load full ERA5 archive (2010–2025).
    2. Apply the SAME sensor model as the original training (SENSOR_PARAMS).
    3. Build ForecastDatasetV2 on the VALIDATION split (2022–2023) using
       the SAME ForecastConfigV2 (wave_wind, input_length=48, forecast_horizon=48).
    4. Load the frozen best_model.pt checkpoint (GRU+wind).
    5. Run evaluation.predict() in eval mode, no gradients, no teacher forcing.
    6. Extract errors using evaluation.extract_error_distribution() with
       modified output path — identical logic to the canonical version.

The only difference from the canonical test-set generation:
    - Uses test_ds (2022–2023 val) instead of test_ds (2024–2025 test).
    - Saves to forecast_errors_val2022_2023.csv.gz.

============================================================
LEAKAGE BOUNDARIES
============================================================

A forecast window at origin o requires:
    - Input:  o - 47 ... o  (48 hourly steps = lookback)
    - Target: o + 1 ... o + 48  (48 hourly steps = forecast)

Val period indices: 105192 to 122711 (inclusive).
Valid origins:      105239 to 122663
    - First: 105192 + 48 - 1 = 105239  (first origin with full 48h lookback in val)
    - Last:  122712 - 48 - 1 = 122663  (last origin with full 48h forecast in val)

These boundaries are enforced by ForecastDatasetV2 with subset="val".

The input lookback for origins near the val start (105239) begins at index 105192,
which is the first val row.  No training-period data enters the lookback.

The forecast target for the last valid origin (122663) ends at 122711,
which is the last val row.  No test-period data enters the targets.

============================================================
FROZEN ARTIFACTS
============================================================

The following are never written to or modified:
    - results/forecasting/exp_gru_wind/best_model.pt
    - results/forecasting/exp_gru_wind/forecast_errors.csv.gz
    - Any Module 1–5.2C file.

============================================================
USAGE
============================================================

    python -m module4_forecasting.generate_validation_error_bank

Or with explicit output directory:

    python -m module4_forecasting.generate_validation_error_bank --results_dir results/forecasting/exp_gru_wind

============================================================
REFERENCES
============================================================
- Module 4.2 run_experiments.py — canonical training/evaluation pipeline.
- Module 4.2 evaluation.py     — predict(), extract_error_distribution().
- Module 5.3A design document  — leakage analysis.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

from module3_sensor.sensor import SensorParameters, WaveSensor
from module4_forecasting.dataset_v2 import (
    ForecastConfigV2,
    make_year_split,
    build_datasets_v2,
)
from module4_forecasting.era5_archive import load_era5_archive
from module4_forecasting.evaluation import (
    predict,
    extract_error_distribution,
    _sincos_to_deg,
    _normalise_sincos,
)
from module4_forecasting.metrics import circular_error, evaluate_forecast
from module4_forecasting.training import load_checkpoint, get_device

# ---------------------------------------------------------------------------
# Canonical sensor parameters (identical to run_experiments.py SENSOR_PARAMS)
# ---------------------------------------------------------------------------
SENSOR_PARAMS = SensorParameters(
    hs_noise_std       = 0.15,
    tp_noise_std       = 0.8,
    direction_noise_std= 15.0,
    delay_steps        = 1,
    missing_probability= 0.05,
    random_seed        = 42,
)

# ---------------------------------------------------------------------------
# Canonical GRU+wind checkpoint
# ---------------------------------------------------------------------------
CANONICAL_CHECKPOINT = os.path.join(
    "results", "forecasting", "exp_gru_wind", "best_model.pt"
)

# ---------------------------------------------------------------------------
# Output file
# ---------------------------------------------------------------------------
OUTPUT_FILENAME = "forecast_errors_val2022_2023.csv.gz"

# Validation period (must match Module 4.2 split)
VAL_START_YEAR = 2022
VAL_END_YEAR   = 2023


def p(msg: str = "", **kwargs) -> None:
    print(msg, flush=True, **kwargs)


def generate_validation_error_bank(
    results_dir: str = os.path.join("results", "forecasting", "exp_gru_wind"),
    checkpoint_path: str = CANONICAL_CHECKPOINT,
    batch_size: int = 256,
    verify_reproducibility: bool = True,
) -> pd.DataFrame:
    """
    Generate forecast errors for the 2022–2023 validation period.

    Uses the frozen GRU+wind model and the exact same sensor/preprocessing
    pipeline as the canonical Module 4.2 evaluation.

    Parameters
    ----------
    results_dir : str
        Directory where the output CSV will be saved.
    checkpoint_path : str
        Path to the frozen best_model.pt.
    batch_size : int
        Inference batch size.
    verify_reproducibility : bool
        If True, run inference twice and assert identical results.

    Returns
    -------
    pd.DataFrame
        Error bank with columns identical to forecast_errors.csv.gz.
    """
    t0 = time.time()
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, OUTPUT_FILENAME)

    # Guard: never overwrite the canonical test-period bank
    canonical_errors = os.path.join(results_dir, "forecast_errors.csv.gz")
    assert output_path != canonical_errors, (
        "SAFETY STOP: output path must not be the canonical test-period error bank."
    )

    p("=" * 66)
    p("MODULE 5.3A-DATA: Validation Error Bank Generation")
    p(f"Output: {output_path}")
    p("=" * 66)

    # ------------------------------------------------------------------
    # Step 1: Load ERA5 archive and sensor model
    # ------------------------------------------------------------------
    p("\n--- Step 1: ERA5 archive + sensor ---")
    df = load_era5_archive()
    p(f"  Archive: {len(df):,} records  [{df.time.min()} … {df.time.max()}]")

    # Compute split — uses make_year_split which replicates Module 4 boundaries
    split = make_year_split(df["time"].values)
    p(f"  Train:  {split.n_train:,}  (up to {df.time.iloc[split.train_end-1]})")
    p(f"  Val:    {split.n_val:,}   ({df.time.iloc[split.train_end]} … {df.time.iloc[split.val_end-1]})")
    p(f"  Test:   {split.n_test:,}  ({df.time.iloc[split.val_end]} … {df.time.iloc[-1]})")

    # Verify val period boundaries
    val_start_ts = pd.Timestamp(df.time.iloc[split.train_end])
    val_end_ts   = pd.Timestamp(df.time.iloc[split.val_end - 1])
    assert val_start_ts.year == VAL_START_YEAR, (
        f"Val start year {val_start_ts.year} != expected {VAL_START_YEAR}"
    )
    assert val_end_ts.year == VAL_END_YEAR, (
        f"Val end year {val_end_ts.year} != expected {VAL_END_YEAR}"
    )

    # Run sensor on full archive (same as run_experiments.py)
    sensor = WaveSensor(SENSOR_PARAMS)
    p("  Running sensor model...", end="")
    obs = sensor.observe(
        time           = np.arange(len(df), dtype=float),
        Hs_true        = df["Hs_m"].values,
        Tp_true        = df["Tp_s"].values,
        direction_true = df["direction_deg"].values,
    )
    p(f" done.  missing_fraction={obs.missing_fraction:.3f}")

    # ------------------------------------------------------------------
    # Step 2: Build VALIDATION dataset (val split only)
    # ------------------------------------------------------------------
    p("\n--- Step 2: Build validation dataset (wave_wind) ---")
    fc_cfg = ForecastConfigV2(
        input_length     = 48,
        forecast_horizon = 48,
        feature_mode     = "wave_wind",   # must match the canonical GRU checkpoint
        missing_policy   = "mask_only",
    )

    # We build all three splits so the scaler is fitted on training data only
    # (same as the canonical pipeline — scaler from build_datasets_v2 is fitted on train)
    _, val_ds, _, scaler = build_datasets_v2(
        obs   = obs,
        config= fc_cfg,
        split = split,
        u10   = df["u10"].values,
        v10   = df["v10"].values,
    )
    p(f"  Val windows: {len(val_ds):,}")
    p(f"  Feature mode: {fc_cfg.feature_mode}, n_features={fc_cfg.n_features}")

    # Verify val origin range
    origins = val_ds.get_all_origins()
    p(f"  origin_idx range: {origins.min()} … {origins.max()}")
    p(f"    → timestamps:  {df.time.iloc[origins.min()]} … {df.time.iloc[origins.max()]}")

    # Leakage assertion: all origins must be within val indices
    assert int(origins.min()) >= split.train_end, (
        f"Earliest origin {origins.min()} < train_end {split.train_end} — training data leaked!"
    )
    assert int(origins.max()) < split.val_end, (
        f"Latest origin {origins.max()} >= val_end {split.val_end} — test data leaked!"
    )

    # Forecast target range: target goes from origin+1 to origin+48
    max_target_idx = int(origins.max()) + fc_cfg.forecast_horizon
    assert max_target_idx <= split.val_end, (
        f"Target end {max_target_idx} > val_end {split.val_end} — test data leaked into targets!"
    )
    p(f"  Leakage assertion PASSED: all origins and targets within val period.")

    # ------------------------------------------------------------------
    # Step 3: Load frozen checkpoint
    # ------------------------------------------------------------------
    p("\n--- Step 3: Load frozen checkpoint ---")
    assert os.path.exists(checkpoint_path), (
        f"Checkpoint not found: {checkpoint_path}"
    )
    device = get_device()
    model, train_cfg, scaler_loaded = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    p(f"  Loaded: {train_cfg.model_type} | feature_mode={train_cfg.feature_mode} "
      f"| n_features={train_cfg.n_features}")
    p(f"  Device: {device}")

    # Verify checkpoint feature mode matches dataset
    assert train_cfg.feature_mode == fc_cfg.feature_mode, (
        f"Feature mode mismatch: checkpoint={train_cfg.feature_mode}, "
        f"dataset={fc_cfg.feature_mode}"
    )

    # Use scaler from checkpoint (same as built above — cross-check)
    # The scaler in the checkpoint was fitted on the 2010-2021 training data,
    # identical to what build_datasets_v2 produces for the same run.
    p(f"  Scaler mean (Hs): {scaler_loaded._mean[0]:.4f}, std: {scaler_loaded._std[0]:.4f}")

    # ------------------------------------------------------------------
    # Step 4: Run inference (eval mode, no gradients)
    # ------------------------------------------------------------------
    p("\n--- Step 4: Inference on validation set ---")
    preds = predict(model, val_ds, scaler_loaded, batch_size=batch_size, device=device)
    p(f"  Predictions shape: {preds.shape}  (N, H=48, 4)")
    p(f"  Hs pred range: {preds[:,:,0].min():.3f} … {preds[:,:,0].max():.3f} m")
    p(f"  Tp pred range: {preds[:,:,1].min():.3f} … {preds[:,:,1].max():.3f} s")

    # ------------------------------------------------------------------
    # Step 5: Reproducibility check
    # ------------------------------------------------------------------
    if verify_reproducibility:
        p("\n--- Step 5: Reproducibility check ---")
        preds2 = predict(model, val_ds, scaler_loaded, batch_size=batch_size, device=device)
        max_diff = float(np.max(np.abs(preds - preds2)))
        assert max_diff == 0.0, f"Reproducibility FAILED: max|preds1 - preds2| = {max_diff}"
        p(f"  PASSED: identical predictions both runs (max diff = {max_diff})")

    # ------------------------------------------------------------------
    # Step 6: Extract error distribution
    # ------------------------------------------------------------------
    p("\n--- Step 6: Extract errors ---")
    targets = val_ds.get_all_targets()   # (N, 48, 4) physical units
    p(f"  Targets shape: {targets.shape}")

    # Build the error DataFrame using identical logic to evaluation.py
    errors_df = _build_error_df(preds, targets, val_ds)
    p(f"  Rows generated: {len(errors_df):,}")
    p(f"  Unique sample_idx: {errors_df.sample_idx.nunique()}")
    p(f"  lead_h range: {errors_df.lead_h.min()} … {errors_df.lead_h.max()}")
    p(f"  origin_idx range: {errors_df.origin_idx.min()} … {errors_df.origin_idx.max()}")

    # ------------------------------------------------------------------
    # Step 7: Leakage verification
    # ------------------------------------------------------------------
    p("\n--- Step 7: Leakage verification ---")

    # No 2024-2025 timestamps
    min_origin = int(errors_df.origin_idx.min())
    max_origin = int(errors_df.origin_idx.max())
    assert min_origin >= split.train_end, (
        f"origin_idx {min_origin} < train_end {split.train_end}: training data in bank!"
    )
    assert max_origin < split.val_end, (
        f"origin_idx {max_origin} >= val_end {split.val_end}: test data in bank!"
    )

    # Verify timestamps are 2022-2023 only
    origin_times = df.time.iloc[errors_df.origin_idx.unique()].dt.year
    assert origin_times.min() == VAL_START_YEAR, (
        f"Earliest origin year {origin_times.min()} != {VAL_START_YEAR}"
    )
    assert origin_times.max() == VAL_END_YEAR, (
        f"Latest origin year {origin_times.max()} != {VAL_END_YEAR}"
    )
    p(f"  Origin year range: {origin_times.min()} – {origin_times.max()}  ✓")

    # No NaN in errors
    for col in ["Hs_error", "Tp_error", "dir_error_deg"]:
        n_nan = errors_df[col].isna().sum()
        assert n_nan == 0, f"NaN in {col}: {n_nan} rows"
    p(f"  No NaN in error columns  ✓")

    # Verify error = pred - true
    hs_check = (errors_df["Hs_pred"] - errors_df["Hs_true"] - errors_df["Hs_error"]).abs().max()
    tp_check = (errors_df["Tp_pred"] - errors_df["Tp_true"] - errors_df["Tp_error"]).abs().max()
    assert hs_check < 1e-5, f"Hs_error != pred - true: max diff = {hs_check}"
    assert tp_check < 1e-5, f"Tp_error != pred - true: max diff = {tp_check}"
    p(f"  error = pred - true verified (max|diff| = {max(hs_check, tp_check):.2e})  ✓")

    # Verify direction errors are in (-180, 180]
    dir_min = errors_df["dir_error_deg"].min()
    dir_max = errors_df["dir_error_deg"].max()
    assert dir_min > -180.0,  f"dir_error_deg min {dir_min:.3f} not > -180"
    assert dir_max <= 180.0,  f"dir_error_deg max {dir_max:.3f} not <= 180"
    p(f"  dir_error_deg range: [{dir_min:.2f}, {dir_max:.2f}]  ✓")

    # All 48 leads present
    leads = sorted(errors_df.lead_h.unique().tolist())
    assert leads == list(range(1, 49)), f"Missing leads: got {leads[:5]}...{leads[-5:]}"
    p(f"  All 48 leads present  ✓")

    # ------------------------------------------------------------------
    # Step 8: Save
    # ------------------------------------------------------------------
    p(f"\n--- Step 8: Save to {output_path} ---")
    errors_df.to_csv(output_path, index=False, compression="gzip")
    file_size_mb = os.path.getsize(output_path) / 1e6
    p(f"  Saved: {output_path}  ({file_size_mb:.1f} MB)")

    # ------------------------------------------------------------------
    # Step 9: Summary statistics
    # ------------------------------------------------------------------
    p("\n--- Summary statistics ---")
    _print_summary_stats(errors_df)

    duration = time.time() - t0
    p(f"\nTotal time: {duration:.1f} s")
    p("\n=== GENERATION COMPLETE ===")
    p(f"Output: {output_path}")
    p("Frozen files: best_model.pt and forecast_errors.csv.gz NOT modified.")

    return errors_df


# ---------------------------------------------------------------------------
# Error DataFrame builder (mirrors evaluation.extract_error_distribution logic)
# ---------------------------------------------------------------------------

def _build_error_df(
    predictions: np.ndarray,
    targets:     np.ndarray,
    val_ds,
) -> pd.DataFrame:
    """
    Build error DataFrame in the canonical format.

    Identical logic to evaluation.extract_error_distribution() but:
    - Saves to a different path (handled by caller).
    - Returns the DataFrame for in-memory verification.

    Error sign convention:
        Hs_error   = Hs_pred  − Hs_true
        Tp_error   = Tp_pred  − Tp_true
        dir_error  = circular(dir_pred − dir_true), wrapped to (-180, 180]

    Direction handling:
        sin/cos from the model are normalised to unit circle before
        converting to degrees (same as evaluation.py).
    """
    N, H, _ = predictions.shape
    origins  = val_ds.get_all_origins()

    rows = []
    for h in range(H):
        hs_pred  = predictions[:, h, 0]
        hs_true  = targets[:, h, 0]
        tp_pred  = predictions[:, h, 1]
        tp_true  = targets[:, h, 1]

        # Normalise predicted sin/cos to unit circle (identical to evaluation.py)
        sin_p, cos_p = _normalise_sincos(predictions[:, h, 2], predictions[:, h, 3])
        sin_t, cos_t = targets[:, h, 2], targets[:, h, 3]

        dir_pred = _sincos_to_deg(sin_p, cos_p)   # [0, 360)
        dir_true = _sincos_to_deg(sin_t, cos_t)   # [0, 360)
        dir_err  = circular_error(dir_pred, dir_true)   # (-180, 180]

        for i in range(N):
            rows.append({
                "lead_h":        h + 1,
                "sample_idx":    i,
                "origin_idx":    int(origins[i]),
                "Hs_pred":       float(hs_pred[i]),
                "Hs_true":       float(hs_true[i]),
                "Hs_error":      float(hs_pred[i] - hs_true[i]),
                "Tp_pred":       float(tp_pred[i]),
                "Tp_true":       float(tp_true[i]),
                "Tp_error":      float(tp_pred[i] - tp_true[i]),
                "dir_pred_deg":  float(dir_pred[i]),
                "dir_true_deg":  float(dir_true[i]),
                "dir_error_deg": float(dir_err[i]),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-lead statistics printer
# ---------------------------------------------------------------------------

def _print_summary_stats(df: pd.DataFrame) -> None:
    """Print per-lead error statistics at key horizons."""
    print(f"  {'Lead':>4}  {'Hs_mean':>8} {'Hs_std':>7} {'Hs_MAE':>7} {'Tp_mean':>8} {'Tp_std':>7} {'dir_std':>8}")
    print(f"  {'-'*55}")
    for lead in [1, 6, 12, 24, 36, 48]:
        sub = df[df["lead_h"] == lead]
        hs  = sub["Hs_error"].values
        tp  = sub["Tp_error"].values
        dr  = sub["dir_error_deg"].values
        print(f"  {lead:>4}  {np.mean(hs):>+8.4f} {np.std(hs):>7.4f} {np.mean(np.abs(hs)):>7.4f} "
              f"{np.mean(tp):>+8.4f} {np.std(tp):>7.4f} {np.std(dr):>8.4f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate 2022-2023 validation forecast-error bank."
    )
    parser.add_argument(
        "--results_dir",
        default=os.path.join("results", "forecasting", "exp_gru_wind"),
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint",
        default=CANONICAL_CHECKPOINT,
        help="Path to frozen GRU+wind best_model.pt",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--no_repro_check",
        action="store_true",
        help="Skip reproducibility check (saves ~50%% time)",
    )
    args = parser.parse_args()

    generate_validation_error_bank(
        results_dir             = args.results_dir,
        checkpoint_path         = args.checkpoint,
        batch_size              = args.batch_size,
        verify_reproducibility  = not args.no_repro_check,
    )


if __name__ == "__main__":
    main()
