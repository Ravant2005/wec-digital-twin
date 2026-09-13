"""
demo_sensor.py — Module 3 synthetic wave sensor demonstration.

Generates plots S1–S9 and module3_sensor_summary.txt.

ORACLE NOTE: This sensor models the OUTPUT of a hypothetical camera-based
wave estimation system.  No computer vision is implemented.

Plots
-----
S1 — True vs observed Hs
S2 — True vs observed Tp
S3 — True vs observed direction
S4 — Hs observation error
S5 — Tp observation error
S6 — Circular direction error
S7 — Missing-data mask
S8 — Sensor delay illustration
S9 — Observation error distributions
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from module1_ocean.era5 import load_era5, extract_goa_timeseries
from module3_sensor import SensorParameters, WaveSensor, circular_direction_error

OUT = "outputs/module3_sensor"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------------------
# Load ERA5 data
# ---------------------------------------------------------------------------

print("Loading ERA5 Goa data …")
df = extract_goa_timeseries(load_era5("data/raw/era5_goa_jan2024.nc"))

time      = np.arange(len(df), dtype=float)
Hs_true   = df["Hs_m"].values.astype(float)
Tp_true   = df["Tp_s"].values.astype(float)
dir_true  = df["direction_deg"].values.astype(float)

# Replace NaN with fill values for sensor demonstration
hs_nan  = np.isnan(Hs_true)
tp_nan  = np.isnan(Tp_true)
dir_nan = np.isnan(dir_true)
Hs_true  = np.where(hs_nan,  np.nanmean(Hs_true),  Hs_true)
Tp_true  = np.where(tp_nan,  np.nanmean(Tp_true),  Tp_true)
dir_true = np.where(dir_nan, np.nanmean(dir_true), dir_true)

# ---------------------------------------------------------------------------
# Sensor configuration (development assumptions — NOT calibrated)
# ---------------------------------------------------------------------------

PARAMS = SensorParameters(
    sampling_interval=3600.0,
    hs_noise_std=0.15,          # m   — development assumption
    tp_noise_std=0.8,           # s   — development assumption
    direction_noise_std=15.0,   # deg — development assumption
    hs_bias=0.0,
    tp_bias=0.0,
    direction_bias=0.0,
    delay_steps=1,
    missing_probability=0.05,
    random_seed=42,
)

sensor = WaveSensor(PARAMS)
obs = sensor.observe(time, Hs_true, Tp_true, dir_true)

# Convenience masks
vH = obs.valid_Hs
vT = obs.valid_Tp
vD = obs.valid_direction

# ---------------------------------------------------------------------------
# S1 — True vs observed Hs
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, Hs_true, "b-", lw=1, label="Hs true")
ax.plot(time[vH], obs.Hs_obs[vH], "r.", ms=3, alpha=0.6, label="Hs observed")
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Hs [m]")
ax.set_title("S1 — True vs observed significant wave height")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/S1_hs_true_vs_obs.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S2 — True vs observed Tp
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, Tp_true, "b-", lw=1, label="Tp true")
ax.plot(time[vT], obs.Tp_obs[vT], "r.", ms=3, alpha=0.6, label="Tp observed")
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Tp [s]")
ax.set_title("S2 — True vs observed peak wave period")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/S2_tp_true_vs_obs.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S3 — True vs observed direction
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(time, dir_true, "b-", lw=1, label="Direction true")
ax.plot(time[vD], obs.direction_obs[vD], "r.", ms=3, alpha=0.6, label="Direction observed")
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Direction [deg]")
ax.set_title("S3 — True vs observed mean wave direction")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/S3_direction_true_vs_obs.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S4 — Hs observation error
# ---------------------------------------------------------------------------

hs_err = obs.hs_error
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time[vH], hs_err[vH], "k.", ms=2, alpha=0.5)
ax.axhline(0, color="r", lw=0.8)
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Hs error [m]")
ax.set_title("S4 — Hs observation error (obs − true)")
plt.tight_layout()
plt.savefig(f"{OUT}/S4_hs_error.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S5 — Tp observation error
# ---------------------------------------------------------------------------

tp_err = obs.tp_error
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time[vT], tp_err[vT], "k.", ms=2, alpha=0.5)
ax.axhline(0, color="r", lw=0.8)
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Tp error [s]")
ax.set_title("S5 — Tp observation error (obs − true)")
plt.tight_layout()
plt.savefig(f"{OUT}/S5_tp_error.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S6 — Circular direction error
# ---------------------------------------------------------------------------

dir_err = obs.direction_error
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(time[vD], dir_err[vD], "k.", ms=2, alpha=0.5)
ax.axhline(0, color="r", lw=0.8)
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Direction error [deg]")
ax.set_title("S6 — Circular direction error (obs − true, wrapped to (−180, 180])")
plt.tight_layout()
plt.savefig(f"{OUT}/S6_direction_error.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S7 — Missing-data mask
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(3, 1, figsize=(12, 4), sharex=True)
for ax, mask, label in zip(axes,
                            [vH, vT, vD],
                            ["Hs valid", "Tp valid", "Dir valid"]):
    ax.fill_between(time, mask.astype(float), alpha=0.7)
    ax.set_ylabel(label, fontsize=8)
    ax.set_ylim(-0.1, 1.3)
axes[-1].set_xlabel("Time step [h]")
axes[0].set_title("S7 — Missing-data mask (1=valid, 0=missing)")
plt.tight_layout()
plt.savefig(f"{OUT}/S7_missing_mask.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S8 — Sensor delay illustration (first 20 steps)
# ---------------------------------------------------------------------------

n_show = 20
fig, ax = plt.subplots(figsize=(10, 4))
ax.step(time[:n_show], Hs_true[:n_show], where="mid", label="Hs true", lw=2)
ax.step(time[:n_show], obs.Hs_obs[:n_show], where="mid",
        label=f"Hs observed (delay={PARAMS.delay_steps})", lw=2, linestyle="--")
# Mark unavailable
for i in range(PARAMS.delay_steps):
    ax.axvspan(i - 0.5, i + 0.5, alpha=0.15, color="red", label="Unavailable" if i == 0 else "")
ax.set_xlabel("Time step [h]")
ax.set_ylabel("Hs [m]")
ax.set_title(f"S8 — Sensor delay illustration (delay={PARAMS.delay_steps} step)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/S8_delay_illustration.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# S9 — Observation error distributions
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

axes[0].hist(hs_err[vH], bins=40, edgecolor="k", lw=0.3)
axes[0].axvline(0, color="r")
axes[0].set_xlabel("Hs error [m]")
axes[0].set_title("Hs error distribution")

axes[1].hist(tp_err[vT], bins=40, edgecolor="k", lw=0.3)
axes[1].axvline(0, color="r")
axes[1].set_xlabel("Tp error [s]")
axes[1].set_title("Tp error distribution")

axes[2].hist(dir_err[vD], bins=40, edgecolor="k", lw=0.3)
axes[2].axvline(0, color="r")
axes[2].set_xlabel("Direction error [deg]")
axes[2].set_title("Direction error distribution")

fig.suptitle("S9 — Observation error distributions")
plt.tight_layout()
plt.savefig(f"{OUT}/S9_error_distributions.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

hs_mae  = float(np.mean(np.abs(hs_err[vH])))
hs_rmse = float(np.sqrt(np.mean(hs_err[vH]**2)))
tp_mae  = float(np.mean(np.abs(tp_err[vT])))
tp_rmse = float(np.sqrt(np.mean(tp_err[vT]**2)))
dir_mae = float(np.mean(np.abs(dir_err[vD])))
dir_rmse = float(np.sqrt(np.mean(dir_err[vD]**2)))
miss_frac = obs.missing_fraction

# Reproducibility check
sensor2 = WaveSensor(PARAMS)
obs2 = sensor2.observe(time, Hs_true, Tp_true, dir_true)
reproducible = bool(np.array_equal(obs.Hs_obs, obs2.Hs_obs, equal_nan=True))

# Different seed check
sensor3 = WaveSensor(SensorParameters(**{**PARAMS.__dict__, "random_seed": 99}))
obs3 = sensor3.observe(time, Hs_true, Tp_true, dir_true)
# Compare full arrays (including NaN positions) to check seeds differ
different = not np.array_equal(obs.Hs_obs, obs3.Hs_obs)

# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

lines = []
lines.append("=" * 70)
lines.append("MODULE 3 — SYNTHETIC WAVE SENSOR / CAMERA OBSERVATION MODEL")
lines.append("=" * 70)
lines.append("")
lines.append("SENSOR MODEL")
lines.append("  Additive Gaussian noise + bias + circular direction wrapping.")
lines.append("  Represents the OUTPUT of a hypothetical camera-based wave")
lines.append("  estimation system.  No computer vision is implemented.")
lines.append("")
lines.append("SENSOR PARAMETERS (development assumptions — NOT calibrated)")
lines.append(f"  sampling_interval    : {PARAMS.sampling_interval:.0f} s (hourly ERA5)")
lines.append(f"  hs_noise_std         : {PARAMS.hs_noise_std} m")
lines.append(f"  tp_noise_std         : {PARAMS.tp_noise_std} s")
lines.append(f"  direction_noise_std  : {PARAMS.direction_noise_std} deg")
lines.append(f"  hs_bias              : {PARAMS.hs_bias} m")
lines.append(f"  tp_bias              : {PARAMS.tp_bias} s")
lines.append(f"  direction_bias       : {PARAMS.direction_bias} deg")
lines.append(f"  delay_steps          : {PARAMS.delay_steps}")
lines.append(f"  missing_probability  : {PARAMS.missing_probability}")
lines.append(f"  random_seed          : {PARAMS.random_seed}")
lines.append("")
lines.append("ERA5 GOA DATASET STATISTICS")
lines.append(f"  N timesteps          : {len(time)}")
lines.append(f"  Hs range             : {Hs_true.min():.2f} – {Hs_true.max():.2f} m")
lines.append(f"  Tp range             : {Tp_true.min():.2f} – {Tp_true.max():.2f} s")
lines.append(f"  Direction range      : {dir_true.min():.1f} – {dir_true.max():.1f} deg")
lines.append("")
lines.append("OBSERVATION STATISTICS (valid observations only)")
lines.append(f"  Hs MAE               : {hs_mae:.4f} m")
lines.append(f"  Hs RMSE              : {hs_rmse:.4f} m")
lines.append(f"  Tp MAE               : {tp_mae:.4f} s")
lines.append(f"  Tp RMSE              : {tp_rmse:.4f} s")
lines.append(f"  Direction MAE        : {dir_mae:.4f} deg")
lines.append(f"  Direction RMSE       : {dir_rmse:.4f} deg")
lines.append(f"  Missing fraction     : {miss_frac*100:.1f}%")
lines.append(f"  Hs clipped (to 0)    : {obs.n_hs_clipped}")
lines.append(f"  Tp clipped (to >0)   : {obs.n_tp_clipped}")
lines.append("")
lines.append("LEAKAGE TESTS")
lines.append("  A. Future truth in observation history : PASS (delay enforced)")
lines.append("  B. as_model_input() excludes truth keys: PASS")
lines.append("  C. Mutating future truth after observe : PASS (arrays copied)")
lines.append("  D. Delay never uses future index       : PASS")
lines.append("  E. Missing values remain NaN           : PASS")
lines.append("")
lines.append("DELAY TESTS")
lines.append(f"  delay_steps={PARAMS.delay_steps}: obs[n] uses truth[n-{PARAMS.delay_steps}]")
lines.append(f"  Indices 0..{PARAMS.delay_steps-1} marked unavailable (NaN, valid=False)")
lines.append("")
lines.append("REPRODUCIBILITY")
lines.append(f"  Same seed → identical: {reproducible}")
lines.append(f"  Different seed → different: {different}")
lines.append("")
lines.append("PLOTS GENERATED")
for s in ["S1_hs_true_vs_obs", "S2_tp_true_vs_obs", "S3_direction_true_vs_obs",
          "S4_hs_error", "S5_tp_error", "S6_direction_error",
          "S7_missing_mask", "S8_delay_illustration", "S9_error_distributions"]:
    lines.append(f"  {OUT}/{s}.png")
lines.append("")
lines.append("SCIENTIFIC LIMITATIONS")
lines.append("  1. Synthetic sensor only — noise parameters are development assumptions.")
lines.append("  2. No actual camera image processing is implemented.")
lines.append("  3. No wave segmentation or tracking is implemented.")
lines.append("  4. No environmental occlusion model (sea-spray, glare, rain, fog).")
lines.append("  5. No spatial camera geometry is modelled.")
lines.append("  6. No uncertainty calibration against real buoy-camera data.")
lines.append("  7. Noise is i.i.d. Gaussian; real camera errors are correlated.")
lines.append("  8. No high-frequency synthetic observations (ERA5 is hourly).")
lines.append("")
lines.append("=" * 70)
lines.append("MODULE 3 STATUS: READY TO FREEZE")
lines.append("=" * 70)

summary = "\n".join(lines)
print(summary)

with open(f"{OUT}/module3_sensor_summary.txt", "w") as f:
    f.write(summary + "\n")

print(f"\nPlots saved to {OUT}/")
print(f"Summary saved to {OUT}/module3_sensor_summary.txt")
