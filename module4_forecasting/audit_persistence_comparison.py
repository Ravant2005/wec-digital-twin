"""
audit_persistence_comparison.py — Lead-wise GRU+wind vs persistence audit.

This script does NOT retrain any model and does NOT modify any existing
artifact.  It:

  1. Reconstructs the test dataset (deterministic — same ERA5 archive,
     same sensor seed, same year split as the original training run).
  2. Loads the frozen GRU+wind per-lead metrics from per_lead_metrics.csv.
  3. Computes persistence predictions via run_persistence_v2() on the
     same 17,449 test windows, then evaluates per-lead metrics.
  4. Writes per_lead_persistence_comparison.csv.
  5. Prints the skill-horizon summary.

Persistence definition (unchanged from evaluation.py / run_persistence_v2):
  For each test window, the persistence forecast is the last valid
  observation in the 48-step input window (Hs, Tp, sin_dir, cos_dir),
  held constant across all 48 forecast leads.

Alignment guarantee:
  ForecastDatasetV2 is deterministic — window ordering depends only on
  the time series and split boundaries, not on any random state.
  origin_idx range [122759, 140207] matches the saved forecast_errors.csv.gz,
  confirming the reconstruction is identical to the original evaluation.

Run:
  PYTHONPATH=. .venv/bin/python module4_forecasting/audit_persistence_comparison.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from module3_sensor.sensor import SensorParameters, WaveSensor
from module4_forecasting.dataset_v2 import (
    ForecastConfigV2,
    build_datasets_v2,
    make_year_split,
    N_FEATURES_WIND,
)
from module4_forecasting.era5_archive import load_era5_archive
from module4_forecasting.evaluation import (
    _normalise_sincos,
    _sincos_to_deg,
    run_persistence_v2,
)
from module4_forecasting.metrics import circular_error, evaluate_by_horizon

RESULTS_DIR = "results/forecasting/exp_gru_wind"
OUT_CSV     = os.path.join(RESULTS_DIR, "per_lead_persistence_comparison.csv")

# Sensor parameters — identical to run_experiments.py
SENSOR_PARAMS = SensorParameters(
    hs_noise_std=0.15,
    tp_noise_std=0.8,
    direction_noise_std=15.0,
    delay_steps=1,
    missing_probability=0.05,
    random_seed=42,
)


def p(msg="", **kw):
    print(msg, flush=True, **kw)


# ---------------------------------------------------------------------------
# Persistence per-lead evaluation
# ---------------------------------------------------------------------------

def persistence_by_horizon(persist_preds: np.ndarray,
                            targets: np.ndarray) -> dict:
    """
    Compute per-lead persistence metrics.

    persist_preds : (N, H, 4)  — Hs, Tp, sin_dir, cos_dir  (physical units)
    targets       : (N, H, 4)  — same layout

    Returns dict with lists of length H:
        hs_mae, hs_rmse, tp_mae, tp_rmse, dir_mae
    """
    N, H, _ = persist_preds.shape
    hs_mae_l, hs_rmse_l = [], []
    tp_mae_l, tp_rmse_l = [], []
    dir_mae_l = []

    for h in range(H):
        # Hs
        err_hs = persist_preds[:, h, 0] - targets[:, h, 0]
        v = ~np.isnan(err_hs)
        hs_mae_l.append(float(np.mean(np.abs(err_hs[v]))))
        hs_rmse_l.append(float(np.sqrt(np.mean(err_hs[v] ** 2))))

        # Tp
        err_tp = persist_preds[:, h, 1] - targets[:, h, 1]
        v = ~np.isnan(err_tp)
        tp_mae_l.append(float(np.mean(np.abs(err_tp[v]))))
        tp_rmse_l.append(float(np.sqrt(np.mean(err_tp[v] ** 2))))

        # Direction — circular
        sin_p, cos_p = _normalise_sincos(persist_preds[:, h, 2],
                                         persist_preds[:, h, 3])
        sin_t, cos_t = targets[:, h, 2], targets[:, h, 3]
        dir_pred = _sincos_to_deg(sin_p, cos_p)
        dir_true = _sincos_to_deg(sin_t, cos_t)
        d_err = circular_error(dir_pred, dir_true)
        v = ~np.isnan(d_err)
        dir_mae_l.append(float(np.mean(np.abs(d_err[v]))))

    return dict(hs_mae=hs_mae_l, hs_rmse=hs_rmse_l,
                tp_mae=tp_mae_l, tp_rmse=tp_rmse_l,
                dir_mae=dir_mae_l)


def _skill(model_err: float, persist_err: float) -> float:
    if persist_err == 0.0 or np.isnan(persist_err):
        return np.nan
    return 1.0 - model_err / persist_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p("=" * 68)
    p("MODULE 4.2 AUDIT — LEAD-WISE PERSISTENCE COMPARISON")
    p("=" * 68)

    # ------------------------------------------------------------------
    # 1. Reconstruct test dataset (deterministic)
    # ------------------------------------------------------------------
    p("\n--- Reconstructing test dataset ---")
    p("  Loading ERA5 archive...", end="")
    df = load_era5_archive()
    p(f" done. {len(df):,} records.")

    split = make_year_split(df["time"].values)
    p(f"  Split: train={split.n_train:,}  val={split.n_val:,}  test={split.n_test:,}")

    p("  Running sensor (seed=42)...", end="")
    sensor = WaveSensor(SENSOR_PARAMS)
    obs = sensor.observe(
        time=np.arange(len(df), dtype=float),
        Hs_true=df["Hs_m"].values,
        Tp_true=df["Tp_s"].values,
        direction_true=df["direction_deg"].values,
    )
    p(f" done. Missing={obs.missing_fraction:.3f}")

    p("  Building wave_wind test dataset...", end="")
    cfg = ForecastConfigV2(input_length=48, forecast_horizon=48,
                           feature_mode="wave_wind")
    _, _, test_ds, _ = build_datasets_v2(
        obs, cfg, split,
        u10=df["u10"].values, v10=df["v10"].values,
    )
    p(f" done. {len(test_ds):,} windows.")

    # Verify alignment with saved artifacts
    origins = test_ds.get_all_origins()
    assert origins[0]  == 122759, f"origin[0]={origins[0]}, expected 122759"
    assert origins[-1] == 140207, f"origin[-1]={origins[-1]}, expected 140207"
    assert len(test_ds) == 17449, f"len={len(test_ds)}, expected 17449"
    p("  Alignment check: PASSED (origin_idx 122759..140207, 17,449 windows)")

    # ------------------------------------------------------------------
    # 2. Load saved GRU+wind per-lead metrics
    # ------------------------------------------------------------------
    p("\n--- Loading GRU+wind per-lead metrics ---")
    gru_df = pd.read_csv(os.path.join(RESULTS_DIR, "per_lead_metrics.csv"))
    assert len(gru_df) == 48, f"Expected 48 lead rows, got {len(gru_df)}"
    p(f"  Loaded {len(gru_df)} lead rows.")

    # ------------------------------------------------------------------
    # 3. Compute persistence per-lead
    # ------------------------------------------------------------------
    p("\n--- Computing persistence per-lead ---")
    persist_preds = run_persistence_v2(test_ds)   # (17449, 48, 4) unscaled
    p(f"  persist_preds shape: {persist_preds.shape}")

    # Load targets from saved npz (identical to test_ds.get_all_targets())
    p("  Loading saved targets from predictions.npz...", end="")
    data    = np.load(os.path.join(RESULTS_DIR, "predictions.npz"))
    targets = data["targets"].astype(np.float64)   # (17449, 48, 4)
    p(" done.")

    # Cross-check: targets[0,0,0] must match forecast_errors sample_idx=0 lead=1
    ef_check = pd.read_csv(
        os.path.join(RESULTS_DIR, "forecast_errors.csv.gz"), nrows=1
    )
    assert abs(float(ef_check["Hs_true"].iloc[0]) - targets[0, 0, 0]) < 1e-4, \
        "Target cross-check failed — targets.npz does not match forecast_errors"
    p("  Target cross-check: PASSED")

    p("  Evaluating persistence by lead...", end="")
    pm = persistence_by_horizon(persist_preds, targets)
    p(" done.")

    # ------------------------------------------------------------------
    # 4. Build comparison DataFrame
    # ------------------------------------------------------------------
    p("\n--- Building comparison table ---")
    rows = []
    for i, lead in enumerate(range(1, 49)):
        g_hs_rmse  = gru_df.loc[i, "hs_rmse"]
        g_hs_mae   = gru_df.loc[i, "hs_mae"]
        g_tp_rmse  = gru_df.loc[i, "tp_rmse"]
        g_tp_mae   = gru_df.loc[i, "tp_mae"]
        g_dir_mae  = gru_df.loc[i, "dir_mae"]

        p_hs_rmse  = pm["hs_rmse"][i]
        p_hs_mae   = pm["hs_mae"][i]
        p_tp_rmse  = pm["tp_rmse"][i]
        p_tp_mae   = pm["tp_mae"][i]
        p_dir_mae  = pm["dir_mae"][i]

        rows.append({
            "lead_hours":               lead,
            "gru_hs_rmse":              round(g_hs_rmse,  6),
            "persistence_hs_rmse":      round(p_hs_rmse,  6),
            "hs_skill_vs_persistence":  round(_skill(g_hs_rmse, p_hs_rmse), 6),
            "gru_hs_mae":               round(g_hs_mae,   6),
            "persistence_hs_mae":       round(p_hs_mae,   6),
            "gru_tp_rmse":              round(g_tp_rmse,  6),
            "persistence_tp_rmse":      round(p_tp_rmse,  6),
            "tp_skill_vs_persistence":  round(_skill(g_tp_rmse, p_tp_rmse), 6),
            "gru_tp_mae":               round(g_tp_mae,   6),
            "persistence_tp_mae":       round(p_tp_mae,   6),
            "gru_direction_mae":        round(g_dir_mae,  6),
            "persistence_direction_mae": round(p_dir_mae, 6),
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_CSV, index=False)
    p(f"  Saved: {OUT_CSV}")

    # ------------------------------------------------------------------
    # 5. Skill horizon summary
    # ------------------------------------------------------------------
    p("\n" + "=" * 68)
    p("SKILL HORIZON SUMMARY")
    p("=" * 68)

    hs_skill  = comp_df["hs_skill_vs_persistence"].values
    tp_skill  = comp_df["tp_skill_vs_persistence"].values
    dir_skill = 1.0 - comp_df["gru_direction_mae"].values / comp_df["persistence_direction_mae"].values

    def _first_negative(skill_arr, name):
        neg = np.where(skill_arr < 0)[0]
        if len(neg) == 0:
            p(f"  {name}: GRU better than persistence at ALL 48 leads")
            return None
        first = int(neg[0]) + 1
        p(f"  {name}: first lead where GRU worse than persistence = lead {first}h")
        return first

    def _last_positive(skill_arr, name):
        pos = np.where(skill_arr > 0)[0]
        if len(pos) == 0:
            p(f"  {name}: GRU never better than persistence")
            return None
        last = int(pos[-1]) + 1
        p(f"  {name}: last lead where GRU better than persistence = lead {last}h")
        return last

    p()
    _first_negative(hs_skill,  "Hs RMSE ")
    _first_negative(tp_skill,  "Tp RMSE ")
    _first_negative(dir_skill, "Dir MAE ")
    p()
    _last_positive(hs_skill,  "Hs RMSE ")
    _last_positive(tp_skill,  "Tp RMSE ")
    _last_positive(dir_skill, "Dir MAE ")

    # ------------------------------------------------------------------
    # 6. Values at key leads
    # ------------------------------------------------------------------
    p("\n" + "=" * 68)
    p("VALUES AT KEY LEADS")
    p("=" * 68)
    p(f"{'Lead':>5}  {'GRU Hs':>8}  {'Pers Hs':>8}  {'Hs Skill':>9}  "
      f"{'GRU Tp':>8}  {'Pers Tp':>8}  {'Tp Skill':>9}  "
      f"{'GRU Dir':>8}  {'Pers Dir':>8}")
    p(f"{'(h)':>5}  {'RMSE(m)':>8}  {'RMSE(m)':>8}  {'':>9}  "
      f"{'RMSE(s)':>8}  {'RMSE(s)':>8}  {'':>9}  "
      f"{'MAE(°)':>8}  {'MAE(°)':>8}")
    p("-" * 90)
    for lead in [1, 6, 12, 24, 36, 48]:
        row = comp_df[comp_df["lead_hours"] == lead].iloc[0]
        ds = 1.0 - row["gru_direction_mae"] / row["persistence_direction_mae"]
        p(f"{lead:>5}  {row['gru_hs_rmse']:>8.4f}  {row['persistence_hs_rmse']:>8.4f}  "
          f"{row['hs_skill_vs_persistence']:>+9.4f}  "
          f"{row['gru_tp_rmse']:>8.4f}  {row['persistence_tp_rmse']:>8.4f}  "
          f"{row['tp_skill_vs_persistence']:>+9.4f}  "
          f"{row['gru_direction_mae']:>8.2f}  {row['persistence_direction_mae']:>8.2f}")

    p()
    p("Note: persistence Hs/Tp RMSE is NOT flat across leads.")
    p("Persistence repeats the last valid obs for all leads, but the")
    p("TRUE values at each lead differ — so persistence error grows with lead.")


if __name__ == "__main__":
    main()
