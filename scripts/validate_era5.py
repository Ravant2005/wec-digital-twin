"""
validate_era5.py — Validate downloaded ERA5 NetCDF files.

Supports three modes:

  auto   — detect stream from variables present in the file (default)
  waves  — expect swh, mwp, mwd, pp1d
  wind   — expect u10, v10

Usage
-----
  python scripts/validate_era5.py path/to/file.nc
  python scripts/validate_era5.py path/to/file.nc --stream waves
  python scripts/validate_era5.py path/to/file.nc --stream wind
  python scripts/validate_era5.py path/to/file.nc --strict
"""

from __future__ import annotations

import argparse
import calendar
import sys
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Expected configuration
# ---------------------------------------------------------------------------

EXPECTED_LATITUDE  = 15.5
EXPECTED_LONGITUDE = 73.5
COORD_TOLERANCE    = 0.26   # half a 0.5-degree grid cell

WAVE_VARS = ["swh", "pp1d", "mwd", "mwp"]

# Wind variables: CDS may store them under several short names
WIND_VAR_CANDIDATES = {
    "u10": ["u10", "u10n", "10u"],
    "v10": ["v10", "v10n", "10v"],
}

# Physical bounds (finite values only)
PHYSICAL_BOUNDS: dict[str, tuple[Optional[float], Optional[float], bool]] = {
    # var: (lo, hi, strictly_positive)
    "swh":  (0.0,    None,  False),
    "pp1d": (None,   None,  True),
    "mwd":  (0.0,    360.0, False),
    "mwp":  (None,   None,  True),
    "u10":  (-100.,  100.,  False),
    "v10":  (-100.,  100.,  False),
}

# January 2024 reference (used by legacy single-file tests)
JAN_2024_RECORDS = 744


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_wind_vars(ds: xr.Dataset) -> dict[str, Optional[str]]:
    resolved = {}
    for logical, candidates in WIND_VAR_CANDIDATES.items():
        resolved[logical] = next((c for c in candidates if c in ds.data_vars), None)
    return resolved


def _time_coord(ds: xr.Dataset) -> Optional[str]:
    for name in ("valid_time", "time"):
        if name in ds.coords:
            return name
    return None


def _detect_stream(ds: xr.Dataset) -> str:
    """Infer stream from variables present: 'waves', 'wind', or 'mixed'."""
    has_wave = any(v in ds.data_vars for v in WAVE_VARS)
    wind_map = _resolve_wind_vars(ds)
    has_wind = any(v is not None for v in wind_map.values())
    if has_wave and not has_wind:
        return "waves"
    if has_wind and not has_wave:
        return "wind"
    if has_wave and has_wind:
        return "mixed"
    return "unknown"


def expected_hours(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1] * 24


# ---------------------------------------------------------------------------
# Core validation
# ---------------------------------------------------------------------------

def validate(
    filepath: str,
    stream: str = "auto",
    strict: bool = False,
    year: Optional[int] = None,
    month: Optional[int] = None,
) -> bool:
    """
    Validate an ERA5 NetCDF file.

    Parameters
    ----------
    filepath : str
    stream : str
        "auto" (detect from file), "waves", "wind", or "mixed".
    strict : bool
        If True, warnings also cause FAIL.
    year, month : int or None
        If provided, check the exact expected record count for that month.
        If None, infer from the time coordinate when possible.

    Returns
    -------
    bool  — True = PASS, False = FAIL
    """
    errors   = []
    warnings = []
    info: dict = {"file": filepath}

    # ------------------------------------------------------------------
    # A. File opens
    # ------------------------------------------------------------------
    try:
        ds = xr.open_dataset(filepath)
    except Exception as exc:
        print(f"\nERA5 VALIDATION\n{'─'*50}")
        print(f"file: {filepath}")
        print(f"FATAL: cannot open file — {exc}")
        print("STATUS: FAIL")
        return False

    # ------------------------------------------------------------------
    # Detect stream
    # ------------------------------------------------------------------
    if stream == "auto":
        stream = _detect_stream(ds)
    info["stream"] = stream

    wind_map = _resolve_wind_vars(ds)

    # ------------------------------------------------------------------
    # B. Required variables
    # ------------------------------------------------------------------
    if stream in ("waves", "mixed", "auto", "unknown"):
        missing_wave = [v for v in WAVE_VARS if v not in ds.data_vars]
        if missing_wave:
            errors.append(f"Missing required wave variables: {missing_wave}")

    if stream in ("wind", "mixed"):
        for logical, actual in wind_map.items():
            if actual is None:
                errors.append(
                    f"Missing wind variable '{logical}' "
                    f"(tried: {WIND_VAR_CANDIDATES[logical]})"
                )

    if stream == "wind":
        # Wind-only file must NOT contain wave variables
        present_wave = [v for v in WAVE_VARS if v in ds.data_vars]
        if present_wave:
            warnings.append(
                f"Wind file contains wave variables {present_wave} — "
                "wave and wind should be in separate files"
            )

    if stream == "waves":
        # Wave-only file must NOT contain wind variables
        present_wind = [a for a in wind_map.values() if a is not None]
        if present_wind:
            warnings.append(
                f"Wave file contains wind variables {present_wind} — "
                "wave and wind should be in separate files"
            )

    # ------------------------------------------------------------------
    # C. Time coordinate
    # ------------------------------------------------------------------
    tc = _time_coord(ds)
    if tc is None:
        errors.append("No time coordinate found (expected 'valid_time' or 'time')")
        times = None
    else:
        times = pd.to_datetime(ds[tc].values)
        info.update({"time_coord": tc, "n_records": len(times),
                     "time_start": str(times[0]), "time_end": str(times[-1])})

    # ------------------------------------------------------------------
    # D. Record count for the expected year/month
    # ------------------------------------------------------------------
    if times is not None and len(times) > 0:
        # Infer year/month from data if not supplied
        inferred_year  = year  if year  is not None else int(times[0].year)
        inferred_month = month if month is not None else int(times[0].month)

        # Only check if all timestamps belong to the same month
        all_same_month = (
            times[0].year  == times[-1].year and
            times[0].month == times[-1].month
        )
        if all_same_month:
            expected = expected_hours(inferred_year, inferred_month)
            if len(times) != expected:
                errors.append(
                    f"{inferred_year}-{inferred_month:02d} should have "
                    f"{expected} hourly records, got {len(times)}"
                )

    # ------------------------------------------------------------------
    # E. Hourly and monotonic
    # ------------------------------------------------------------------
    if times is not None and len(times) > 1:
        diffs_h = np.diff(times).astype("timedelta64[h]").astype(int)
        if not np.all(diffs_h > 0):
            errors.append("Time coordinate is not strictly monotonically increasing")
        non_hourly = int(np.sum(diffs_h != 1))
        if non_hourly > 0:
            errors.append(f"{non_hourly} time step(s) are not exactly 1 hour apart")

    # ------------------------------------------------------------------
    # F. Duplicate timestamps
    # ------------------------------------------------------------------
    if times is not None:
        n_dupes = len(times) - len(np.unique(times))
        if n_dupes > 0:
            errors.append(f"{n_dupes} duplicate timestamp(s) found")

    # ------------------------------------------------------------------
    # G. Coordinate presence
    # ------------------------------------------------------------------
    for coord_name in ("latitude", "longitude"):
        if coord_name not in ds.coords:
            errors.append(f"Coordinate '{coord_name}' not found")

    # ------------------------------------------------------------------
    # H. Grid point accuracy
    # ------------------------------------------------------------------
    if "latitude" in ds.coords and "longitude" in ds.coords:
        actual_lat = float(ds["latitude"].values.flat[0])
        actual_lon = float(ds["longitude"].values.flat[0])
        info["latitude"]  = actual_lat
        info["longitude"] = actual_lon
        if abs(actual_lat - EXPECTED_LATITUDE) > COORD_TOLERANCE:
            errors.append(
                f"Latitude {actual_lat} differs from expected "
                f"{EXPECTED_LATITUDE} by more than {COORD_TOLERANCE}°"
            )
        if abs(actual_lon - EXPECTED_LONGITUDE) > COORD_TOLERANCE:
            errors.append(
                f"Longitude {actual_lon} differs from expected "
                f"{EXPECTED_LONGITUDE} by more than {COORD_TOLERANCE}°"
            )

    # ------------------------------------------------------------------
    # I. NaN counts (reported, not filled)
    # ------------------------------------------------------------------
    nan_counts: dict[str, int] = {}
    vars_to_check = []
    if stream != "wind":
        vars_to_check += [v for v in WAVE_VARS if v in ds.data_vars]
    if stream != "waves":
        vars_to_check += [a for a in wind_map.values() if a is not None]

    for var in vars_to_check:
        vals = ds[var].values.ravel()
        nan_counts[var] = int(np.sum(np.isnan(vals)))
    info["nan_counts"] = nan_counts

    # ------------------------------------------------------------------
    # J. Physical sanity (failures become errors)
    # ------------------------------------------------------------------
    phys_issues: list[str] = []

    def _check(var_name: str) -> None:
        if var_name not in ds.data_vars:
            return
        lo, hi, strictly_pos = PHYSICAL_BOUNDS.get(var_name, (None, None, False))
        vals   = ds[var_name].values.ravel()
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            phys_issues.append(f"{var_name}: no finite values")
            return
        if lo is not None and np.any(finite < lo):
            phys_issues.append(
                f"{var_name}: {int(np.sum(finite < lo))} value(s) below {lo}"
            )
        if hi is not None and np.any(finite > hi):
            phys_issues.append(
                f"{var_name}: {int(np.sum(finite > hi))} value(s) above {hi}"
            )
        if strictly_pos and np.any(finite <= 0.0):
            phys_issues.append(
                f"{var_name}: {int(np.sum(finite <= 0.0))} value(s) <= 0 (must be > 0)"
            )

    for v in vars_to_check:
        _check(v)
    # Also check resolved wind names against their logical bounds
    for logical, actual in wind_map.items():
        if actual and actual not in vars_to_check:
            _check(actual)

    errors.extend(phys_issues)

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    print(f"\nERA5 VALIDATION")
    print("─" * 50)
    print(f"file          : {info['file']}")
    print(f"stream        : {info['stream']}")
    if times is not None:
        print(f"time records  : {info['n_records']}")
        print(f"time start    : {info['time_start']}")
        print(f"time end      : {info['time_end']}")
    if "latitude" in info:
        print(f"latitude      : {info['latitude']}")
        print(f"longitude     : {info['longitude']}")

    print("\nvariables:")
    if stream != "wind":
        for var in WAVE_VARS:
            print(f"  {var:<8}: {'OK' if var in ds.data_vars else 'MISSING'}")
    if stream != "waves":
        for logical, actual in wind_map.items():
            if actual:
                print(f"  {logical:<8}: OK  (stored as '{actual}')")
            else:
                print(f"  {logical:<8}: NOT FOUND")

    print("\nmissing values (NaN counts):")
    if nan_counts:
        for var, n in nan_counts.items():
            flag = "  ← WARNING" if n > 0 else ""
            print(f"  {var:<8}: {n}{flag}")
    else:
        print("  (none checked)")

    print("\nphysical checks:")
    if phys_issues:
        for issue in phys_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  all checks passed")

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  WARN: {w}")

    if errors:
        print("\nerrors:")
        for e in errors:
            print(f"  ERROR: {e}")

    ds.close()

    passed = len(errors) == 0 and (not strict or len(warnings) == 0)
    print(f"\nSTATUS: {'PASS' if passed else 'FAIL'}")
    return passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a downloaded ERA5 NetCDF file.",
    )
    p.add_argument("filepath", help="Path to the ERA5 NetCDF file")
    p.add_argument("--stream", choices=["auto", "waves", "wind", "mixed"],
                   default="auto", help="Expected stream type (default: auto-detect)")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 on warnings as well as errors")
    p.add_argument("--year",  type=int, default=None,
                   help="Expected year (for record-count check)")
    p.add_argument("--month", type=int, default=None,
                   help="Expected month (for record-count check)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ok = validate(args.filepath, stream=args.stream, strict=args.strict,
                  year=args.year, month=args.month)
    sys.exit(0 if ok else 1)
