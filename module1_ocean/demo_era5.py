"""
demo_era5.py — Demonstration script for Module 1 ERA5 wave data.

Loads the ERA5 Goa January 2024 dataset, prints descriptive statistics,
and generates the three wave-parameter plots.

Run from the project root:
    python -m module1_ocean.demo_era5
"""

import os
import matplotlib
matplotlib.use("Agg")  # save to file without requiring a display

import matplotlib.pyplot as plt

from module1_ocean.plot_era5 import load_goa_dataframe, plot_all
from module1_ocean.analyze_era5 import summarise_all

DATA_FILE = "data/raw/era5_goa_jan2024.nc"
OUTPUT_DIR = "outputs"
PLOT_FILE = os.path.join(OUTPUT_DIR, "era5_goa_jan2024_wave_parameters.png")


def _print_stats(summary: dict) -> None:
    hs = summary["wave_height"]
    tp = summary["peak_period"]
    wd = summary["wave_direction"]
    r  = summary["hs_tp_pearson_r"]

    print("\n" + "=" * 60)
    print("  ERA5 Wave Statistics — Goa (15.5°N, 73.5°E) — Jan 2024")
    print("=" * 60)

    print(f"\n  Significant Wave Height (Hs) [{hs['count']} observations]")
    print(f"    Mean   : {hs['mean_m']:.3f} m")
    print(f"    Std    : {hs['std_m']:.3f} m")
    print(f"    Min    : {hs['min_m']:.3f} m")
    print(f"    Median : {hs['median_m']:.3f} m")
    print(f"    Max    : {hs['max_m']:.3f} m")

    print(f"\n  Peak Wave Period (Tp) [{tp['count']} observations]")
    print(f"    Mean   : {tp['mean_s']:.2f} s")
    print(f"    Std    : {tp['std_s']:.2f} s")
    print(f"    Min    : {tp['min_s']:.2f} s")
    print(f"    Median : {tp['median_s']:.2f} s")
    print(f"    Max    : {tp['max_s']:.2f} s")

    print(f"\n  Mean Wave Direction [{wd['count']} observations]")
    print(f"    Circular mean : {wd['circular_mean_deg']:.1f}° true (FROM)")
    print(f"    Circular std  : {wd['circular_std_deg']:.1f}°")
    print(f"    Mean resultant length (R̄) : {wd['mean_resultant_len']:.4f}")
    print(f"    Raw range     : {wd['min_deg']:.1f}° – {wd['max_deg']:.1f}°")
    print(f"    Note: min/max/median are raw values, not circular statistics.")

    print(f"\n  Hs–Tp Pearson correlation : {r:+.4f}")
    print("=" * 60 + "\n")


def main() -> None:
    print(f"Loading: {DATA_FILE}")
    df = load_goa_dataframe(DATA_FILE)
    print(f"Loaded {len(df)} hourly records "
          f"({df['time'].iloc[0]} → {df['time'].iloc[-1]})")

    summary = summarise_all(df)
    _print_stats(summary)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig = plot_all(df)
    fig.savefig(PLOT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {PLOT_FILE}")


if __name__ == "__main__":
    main()
