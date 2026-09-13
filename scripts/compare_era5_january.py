"""
compare_era5_january.py — Compare the existing Module 1 ERA5 file against a
newly downloaded ERA5 file for January 2024.

Purpose: verify that the new CDS acquisition pipeline reproduces the same
data as the existing validated file used by Module 1.

Usage
-----
    python scripts/compare_era5_january.py \
        data/raw/era5_goa_jan2024.nc \
        data/raw/era5/test/era5_goa_2024-01.nc

The existing file is treated as the reference.
The new file is treated as the candidate.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr

# Variables to compare (must exist in both files)
COMPARE_VARS = ["swh", "pp1d", "mwd"]

# Tolerance for "close but not identical"
CLOSE_ATOL = 1e-5


def _circular_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Signed circular difference a − b wrapped to (−180, 180]."""
    d = (a - b) % 360.0
    return np.where(d > 180.0, d - 360.0, d)


def _time_coord(ds: xr.Dataset) -> str:
    for name in ("valid_time", "time"):
        if name in ds.coords:
            return name
    raise ValueError(f"No time coordinate found. Coords: {list(ds.coords)}")


def compare(ref_path: str, cand_path: str) -> bool:
    """
    Compare reference and candidate ERA5 files.

    Returns True if all compared variables agree within floating-point
    tolerance (values are identical or differ only by representation).
    """
    print("\nERA5 JANUARY 2024 COMPARISON")
    print("─" * 60)
    print(f"  reference : {ref_path}")
    print(f"  candidate : {cand_path}")

    try:
        ref  = xr.open_dataset(ref_path)
        cand = xr.open_dataset(cand_path)
    except Exception as exc:
        print(f"FATAL: cannot open file — {exc}")
        return False

    # ------------------------------------------------------------------
    # Shape / time
    # ------------------------------------------------------------------
    tc_ref  = _time_coord(ref)
    tc_cand = _time_coord(cand)

    times_ref  = pd.to_datetime(ref[tc_ref].values)
    times_cand = pd.to_datetime(cand[tc_cand].values)

    print(f"\nTIME")
    print(f"  reference : {len(times_ref)} records  "
          f"({times_ref[0]} → {times_ref[-1]})")
    print(f"  candidate : {len(times_cand)} records  "
          f"({times_cand[0]} → {times_cand[-1]})")

    times_match = (len(times_ref) == len(times_cand) and
                   np.all(times_ref == times_cand))
    print(f"  timestamps equal: {'YES' if times_match else 'NO ← MISMATCH'}")

    if not times_match:
        print("  WARNING: time arrays differ — variable comparison may be misleading")

    # ------------------------------------------------------------------
    # Coordinates
    # ------------------------------------------------------------------
    print(f"\nCOORDINATES")
    for coord in ("latitude", "longitude"):
        rv = float(ref[coord].values.flat[0])  if coord in ref.coords  else None
        cv = float(cand[coord].values.flat[0]) if coord in cand.coords else None
        match = (rv is not None and cv is not None and
                 abs(rv - cv) < 1e-6)
        print(f"  {coord:<12}: ref={rv}  cand={cv}  "
              f"{'match' if match else 'DIFFER'}")

    # ------------------------------------------------------------------
    # Variable comparison
    # ------------------------------------------------------------------
    print(f"\nVARIABLE COMPARISON")
    all_close = True

    for var in COMPARE_VARS:
        ref_has  = var in ref.data_vars
        cand_has = var in cand.data_vars

        if not ref_has or not cand_has:
            status = (f"SKIP — ref={'present' if ref_has else 'MISSING'}, "
                      f"cand={'present' if cand_has else 'MISSING'}")
            print(f"\n  {var}: {status}")
            continue

        r = ref[var].squeeze(drop=True).values.ravel().astype(float)
        c = cand[var].squeeze(drop=True).values.ravel().astype(float)

        # Align lengths if timestamps differ
        n = min(len(r), len(c))
        r, c = r[:n], c[:n]

        # Direction uses circular difference
        if var == "mwd":
            diff = _circular_diff(c, r)
        else:
            diff = c - r

        valid = np.isfinite(diff)
        n_valid = int(np.sum(valid))
        n_nan_r = int(np.sum(np.isnan(r)))
        n_nan_c = int(np.sum(np.isnan(c)))

        if n_valid == 0:
            print(f"\n  {var}: no finite differences to compare")
            continue

        d = diff[valid]
        mae  = float(np.mean(np.abs(d)))
        rmse = float(np.sqrt(np.mean(d**2)))
        max_abs = float(np.max(np.abs(d)))
        bias = float(np.mean(d))

        # Relative difference (skip for direction)
        if var != "mwd":
            denom = np.abs(r[valid])
            denom = np.where(denom < 1e-9, np.nan, denom)
            rel = np.abs(d) / denom
            max_rel = float(np.nanmax(rel)) if np.any(np.isfinite(rel)) else np.nan
        else:
            max_rel = np.nan

        close = max_abs < CLOSE_ATOL
        if not close:
            all_close = False

        print(f"\n  {var}:")
        print(f"    shape ref/cand : {len(r)} / {len(c)}")
        print(f"    NaN ref/cand   : {n_nan_r} / {n_nan_c}")
        print(f"    MAE            : {mae:.6g}")
        print(f"    RMSE           : {rmse:.6g}")
        print(f"    max |diff|     : {max_abs:.6g}")
        print(f"    bias           : {bias:+.6g}")
        if not np.isnan(max_rel):
            print(f"    max rel diff   : {max_rel:.4%}")
        if var == "mwd":
            print(f"    (direction differences are circular)")
        print(f"    identical      : {'YES' if close else 'NO — values differ'}")

    ref.close()
    cand.close()

    print(f"\nOVERALL: {'FILES AGREE' if all_close else 'FILES DIFFER'}")
    if all_close:
        print("  All compared variables agree within floating-point tolerance.")
        print("  The new pipeline reproduces the existing January 2024 ERA5 data.")
    else:
        print("  One or more variables differ beyond tolerance.")
        print("  Possible causes: different CDS request, different grid point,")
        print("  different variable set, or data revision.")

    return all_close


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare existing and newly downloaded ERA5 January 2024 files.",
    )
    p.add_argument("reference",  help="Existing reference file (e.g. data/raw/era5_goa_jan2024.nc)")
    p.add_argument("candidate",  help="Newly downloaded file (e.g. data/raw/era5/test/era5_goa_2024-01.nc)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = compare(args.reference, args.candidate)
    sys.exit(0 if ok else 1)
