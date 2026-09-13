"""
audit_era5_archive.py — Scan the ERA5 production archive and report coverage.

Usage
-----
  python scripts/audit_era5_archive.py
  python scripts/audit_era5_archive.py --start-year 2010 --end-year 2025
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration (mirrors download_era5_batch.py)
# ---------------------------------------------------------------------------

PRODUCTION_ROOT = "data/raw/era5/production"
WAVES_DIR       = os.path.join(PRODUCTION_ROOT, "waves")
WIND_DIR        = os.path.join(PRODUCTION_ROOT, "wind")
MANIFEST_PATH   = os.path.join(PRODUCTION_ROOT, "manifest.json")

DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR   = 2025


def _wave_path(year: int, month: int) -> str:
    return os.path.join(WAVES_DIR, f"era5_waves_goa_{year}-{month:02d}.nc")


def _wind_path(year: int, month: int) -> str:
    return os.path.join(WIND_DIR, f"era5_wind_goa_{year}-{month:02d}.nc")


def _expected_hours(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit(start_year: int = DEFAULT_START_YEAR,
          end_year:   int = DEFAULT_END_YEAR) -> dict:
    """
    Scan the production archive and return a summary dict.
    """
    all_months = [
        (y, m)
        for y in range(start_year, end_year + 1)
        for m in range(1, 13)
    ]
    n_expected = len(all_months)

    # Load manifest if present
    manifest_records: dict = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as fh:
            manifest_records = json.load(fh).get("records", {})

    results: dict = {
        "period": f"{start_year}-01 → {end_year}-12",
        "expected_months": n_expected,
        "waves": {"present": [], "missing": [], "failed": []},
        "wind":  {"present": [], "missing": [], "failed": []},
        "total_wave_records": 0,
        "total_wind_records": 0,
        "coordinate_issues": [],
        "variable_issues": [],
    }

    for stream, path_fn, stream_key in [
        ("waves", _wave_path, "waves"),
        ("wind",  _wind_path, "wind"),
    ]:
        for y, m in all_months:
            path = path_fn(y, m)
            label = f"{y}-{m:02d}"
            mkey  = f"{stream}_{label}"

            if os.path.exists(path):
                results[stream_key]["present"].append(label)
                # Check manifest for record count
                rec = manifest_records.get(mkey, {})
                n   = rec.get("n_records", 0)
                if stream == "waves":
                    results["total_wave_records"] += n
                else:
                    results["total_wind_records"] += n
                # Coordinate check from manifest
                alat = rec.get("actual_latitude",  None)
                alon = rec.get("actual_longitude", None)
                if alat is not None and (abs(alat - 15.5) > 0.26 or
                                         abs(alon - 73.5) > 0.26):
                    results["coordinate_issues"].append(
                        f"{stream} {label}: lat={alat}, lon={alon}"
                    )
            else:
                # Check if manifest says it failed
                rec = manifest_records.get(mkey, {})
                if rec.get("status") == "failed":
                    results[stream_key]["failed"].append(label)
                else:
                    results[stream_key]["missing"].append(label)

    # Duplicate file detection (same month appearing twice — shouldn't happen
    # with the naming scheme, but check anyway)
    for stream_key, dir_ in [("waves", WAVES_DIR), ("wind", WIND_DIR)]:
        if not os.path.isdir(dir_):
            continue
        seen: dict[str, list[str]] = {}
        for fname in os.listdir(dir_):
            if not fname.endswith(".nc"):
                continue
            # Extract YYYY-MM from filename
            parts = fname.replace(".nc", "").split("_")
            ym = parts[-1] if parts else fname
            seen.setdefault(ym, []).append(fname)
        for ym, files in seen.items():
            if len(files) > 1:
                results["variable_issues"].append(
                    f"{stream_key} duplicate for {ym}: {files}"
                )

    return results


def print_report(r: dict) -> None:
    print("\nERA5 ARCHIVE AUDIT")
    print("=" * 60)
    print(f"Period   : {r['period']}")
    print(f"Expected : {r['expected_months']} months per stream")
    print()

    for stream in ("waves", "wind"):
        s = r[stream]
        n_present = len(s["present"])
        n_missing = len(s["missing"])
        n_failed  = len(s["failed"])
        n_exp     = r["expected_months"]
        print(f"{stream.upper()} STREAM")
        print(f"  expected : {n_exp}")
        print(f"  present  : {n_present}")
        print(f"  missing  : {n_missing}")
        print(f"  failed   : {n_failed}")
        if s["missing"]:
            # Print in compact groups
            print(f"  missing months: {s['missing'][:12]}"
                  + (" …" if len(s["missing"]) > 12 else ""))
        if s["failed"]:
            print(f"  failed months : {s['failed']}")
        print()

    total_wave = r["total_wave_records"]
    total_wind = r["total_wind_records"]
    print(f"RECORDS (from manifest)")
    print(f"  wave records : {total_wave:,}")
    print(f"  wind records : {total_wind:,}")
    print()

    if r["coordinate_issues"]:
        print("COORDINATE ISSUES:")
        for issue in r["coordinate_issues"]:
            print(f"  {issue}")
        print()

    if r["variable_issues"]:
        print("VARIABLE / DUPLICATE ISSUES:")
        for issue in r["variable_issues"]:
            print(f"  {issue}")
        print()

    wave_complete = len(r["waves"]["missing"]) == 0 and len(r["waves"]["failed"]) == 0
    wind_complete = len(r["wind"]["missing"])  == 0 and len(r["wind"]["failed"])  == 0
    status = "COMPLETE" if (wave_complete and wind_complete) else "INCOMPLETE"
    print(f"STATUS: {status}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Audit the ERA5 production archive for completeness.",
    )
    p.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year",   type=int, default=DEFAULT_END_YEAR)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    r = audit(args.start_year, args.end_year)
    print_report(r)
    wave_ok = len(r["waves"]["missing"]) == 0 and len(r["waves"]["failed"]) == 0
    wind_ok = len(r["wind"]["missing"])  == 0 and len(r["wind"]["failed"])  == 0
    sys.exit(0 if (wave_ok and wind_ok) else 1)
