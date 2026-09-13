"""
demo_goa_replay.py — ERA5-conditioned synthetic sea-state demonstration.

Loads the ERA5 Goa January 2024 dataset, selects a short demonstration
period, generates synthetic sea-surface realizations, and saves plots.

IMPORTANT:
- The synthetic surface elevation is NOT the actual historical sea surface.
  It is a statistically consistent realization conditioned on ERA5 bulk
  parameters (Hs, Tp, direction) for each hour.
- depth_m = 50 m is a demonstration value only.  It does NOT represent
  the actual Goa deployment depth.
- gamma = 3.3 is the conventional JONSWAP default.  It is NOT calibrated
  for the Goa coast.

Run from the project root:
    python -m module1_ocean.demo_goa_replay
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from module1_ocean.era5 import load_era5, extract_goa_timeseries
from module1_ocean.goa_seastate import goa_seastate, generate_goa_replay

ERA5_FILE  = "data/raw/era5_goa_jan2024.nc"
OUTPUT_DIR = "outputs"

# Demonstration parameters — not real Goa values
DEPTH_M    = 50.0    # m — demonstration depth only
GAMMA      = 3.3     # conventional default, not Goa-calibrated
BASE_SEED  = 42
DT         = 0.5     # s
DURATION_S = 3600.0  # s — one hour per segment

# Select a short demonstration window: first 6 hours of January 2024
N_HOURS    = 6


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load ERA5 ---
    print(f"Loading ERA5: {ERA5_FILE}")
    ds = load_era5(ERA5_FILE)
    df_full = extract_goa_timeseries(ds)
    df = df_full.iloc[:N_HOURS].copy()

    print(f"Selected {N_HOURS} ERA5 records: "
          f"{df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    print()

    # --- Print ERA5 parameters for selected window ---
    print("ERA5 sea-state parameters for selected window:")
    print(f"  {'Time':>22}  {'Hs(m)':>7}  {'Tp(s)':>7}  {'Dir(°)':>8}")
    print("  " + "-" * 52)
    for _, row in df.iterrows():
        print(f"  {str(row['time']):>22}  {row['Hs_m']:>7.3f}  "
              f"{row['Tp_s']:>7.2f}  {row['direction_deg']:>8.1f}")
    print()

    # --- Generate replay ---
    print(f"Generating {N_HOURS} synthetic sea-state segments...")
    print(f"  depth_m = {DEPTH_M} m  (demonstration value only)")
    print(f"  gamma   = {GAMMA}  (conventional default, not Goa-calibrated)")
    print()

    replay = generate_goa_replay(
        df,
        depth_m=DEPTH_M,
        gamma=GAMMA,
        duration_s=DURATION_S,
        dt=DT,
        base_seed=BASE_SEED,
    )

    # --- Print per-segment statistics ---
    print(f"  {'Segment':>8}  {'ERA5 Hs':>8}  {'Spec Hs':>8}  "
          f"{'TS Hs':>8}  {'Mean η':>9}  {'Std η':>8}")
    print("  " + "-" * 62)
    for i, (seg, ts) in enumerate(zip(replay.segments, replay.era5_timestamps)):
        print(f"  {i:>8}  {seg.Hs_requested:>8.3f}  {seg.Hs_spectral:>8.4f}  "
              f"{seg.Hs_timeseries:>8.4f}  {np.mean(seg.eta):>+9.5f}  "
              f"{np.std(seg.eta):>8.5f}")
    print()

    # --- Single-segment deep-dive: first hour ---
    seg0 = replay.segments[0]
    ts0  = replay.era5_timestamps[0]

    print("=" * 60)
    print(f"  Single-segment detail: {ts0}")
    print("=" * 60)
    print(f"  ERA5 Hs          = {seg0.Hs_requested:.4f} m")
    print(f"  ERA5 Tp          = {seg0.Tp_requested:.4f} s")
    print(f"  ERA5 direction   = {seg0.direction_deg:.1f}°")
    print(f"  direction (rad)  = {seg0.direction_rad:.6f} rad")
    print(f"  Spectral Hs      = {seg0.Hs_spectral:.6f} m")
    print(f"  Time-series Hs   = {seg0.Hs_timeseries:.6f} m")
    print(f"  Mean η           = {np.mean(seg0.eta):+.6f} m")
    print(f"  Std(η)           = {np.std(seg0.eta):.6f} m")
    print(f"  Min η            = {np.min(seg0.eta):.4f} m")
    print(f"  Max η            = {np.max(seg0.eta):.4f} m")
    print(f"  Depth            = {seg0.depth_m} m (demonstration only)")
    print(f"  gamma            = {seg0.gamma} (not Goa-calibrated)")
    print("=" * 60)
    print()

    # --- Plot A: ERA5 parameters over selected window ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    times = df["time"].values

    axes[0].plot(times, df["Hs_m"], "o-", color="#1f77b4", markersize=5)
    axes[0].set_ylabel("Hs [m]")
    axes[0].set_title(
        f"ERA5 Goa Sea-State Parameters — {N_HOURS} hours from "
        f"{df['time'].iloc[0].strftime('%Y-%m-%d %H:%M UTC')}"
    )
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(times, df["Tp_s"], "o-", color="#ff7f0e", markersize=5)
    axes[1].set_ylabel("Tp [s]")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(times, df["direction_deg"], "o-", color="#2ca02c", markersize=5)
    axes[2].set_ylabel("Direction [°]")
    axes[2].set_ylim(0, 360)
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    axes[2].set_xlabel("Time (UTC)")
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    path_a = os.path.join(OUTPUT_DIR, "goa_replay_era5_params.png")
    fig.savefig(path_a, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_a}")

    # --- Plot B: Synthetic surface elevation for each segment ---
    fig, axes = plt.subplots(N_HOURS, 1, figsize=(12, 2.0 * N_HOURS),
                              sharex=False)
    if N_HOURS == 1:
        axes = [axes]

    for i, (seg, ts) in enumerate(zip(replay.segments, replay.era5_timestamps)):
        # Show first 300 s of each segment
        window = seg.t <= 300.0
        axes[i].plot(seg.t[window], seg.eta[window],
                     color="#1f77b4", linewidth=0.7)
        axes[i].axhline(0, color="gray", linewidth=0.4, linestyle="--")
        axes[i].set_ylabel("η [m]", fontsize=8)
        axes[i].set_title(
            f"Segment {i}: {ts.strftime('%H:%M UTC')}  "
            f"Hs={seg.Hs_requested:.2f}m  Tp={seg.Tp_requested:.1f}s  "
            f"Dir={seg.direction_deg:.0f}°  "
            f"Hs_ts={seg.Hs_timeseries:.3f}m",
            fontsize=8
        )
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time within segment [s]")
    fig.suptitle(
        "Synthetic Sea-Surface Realizations (ERA5-conditioned)\n"
        f"depth={DEPTH_M}m (demo only), γ={GAMMA} (not Goa-calibrated)",
        fontsize=10
    )
    fig.tight_layout()
    path_b = os.path.join(OUTPUT_DIR, "goa_replay_segments.png")
    fig.savefig(path_b, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_b}")

    # --- Plot C: Hs comparison (ERA5 vs spectral vs time-series) ---
    seg_indices = np.arange(len(replay.segments))
    hs_era5 = replay.era5_Hs
    hs_spec  = np.array([s.Hs_spectral   for s in replay.segments])
    hs_ts    = np.array([s.Hs_timeseries for s in replay.segments])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(seg_indices, hs_era5, "o-", label="ERA5 Hs",       color="#1f77b4")
    ax.plot(seg_indices, hs_spec,  "s--", label="Spectral Hs",  color="#ff7f0e")
    ax.plot(seg_indices, hs_ts,    "^:", label="Time-series Hs (4·std)",
            color="#2ca02c")
    ax.set_xlabel("Segment index")
    ax.set_ylabel("Hs [m]")
    ax.set_title("Hs Comparison: ERA5 vs Spectral vs Time-series")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path_c = os.path.join(OUTPUT_DIR, "goa_replay_hs_comparison.png")
    fig.savefig(path_c, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {path_c}")


if __name__ == "__main__":
    main()
