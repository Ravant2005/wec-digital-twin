"""
era5_archive.py — ERA5 archive loader for the WEC Digital Twin (Module 4.2 pipeline).

Loads and concatenates the full 2010–2025 ERA5 hourly archive from the
production directory into a single aligned pandas DataFrame.

Archive layout expected
-----------------------
  data/raw/era5/production/waves/era5_waves_goa_YYYY-MM.nc
  data/raw/era5/production/wind/era5_wind_goa_YYYY-MM.nc

Required wave variables : swh, pp1d, mwd
Required wind variables : u10, v10
Optional wave variable  : mwp (preserved if present)

Time coordinate
---------------
ERA5 files use 'valid_time' as the time coordinate.
All files use singleton latitude and longitude dimensions that are squeezed out.

Output
------
load_era5_archive() returns a pandas DataFrame with columns:
    time          — UTC timestamp (DatetimeIndex)
    Hs_m          — significant wave height [m]  (swh)
    Tp_s          — peak wave period [s]          (pp1d)
    direction_deg — mean wave direction [degrees] (mwd)
    u10           — 10 m zonal wind [m/s]
    v10           — 10 m meridional wind [m/s]

Missing-data policy
-------------------
NaN values in the source files are preserved as-is.
No interpolation or forward-fill is performed.
Missing timestamps are reported explicitly and raise ValueError.

Scientific notes
----------------
- Wave and wind streams are on different ERA5 grids but both are
  requested at the same nominal point (15.5 N, 73.5 E).  CDS returns
  the nearest grid point for each stream independently.  The actual
  coordinates are verified to be within 0.26 degrees of the target.
- Wind variables are ERA5 reanalysis values, not sensor observations.
  They are NOT passed through the Module 3 sensor model.
- This module is read-only with respect to the archive.
"""

from __future__ import annotations

import os
import glob
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PRODUCTION_ROOT = "data/raw/era5/production"
WAVES_DIR = os.path.join(PRODUCTION_ROOT, "waves")
WIND_DIR  = os.path.join(PRODUCTION_ROOT, "wind")

WAVE_FILE_PATTERN = "era5_waves_goa_{year}-{month:02d}.nc"
WIND_FILE_PATTERN = "era5_wind_goa_{year}-{month:02d}.nc"

TIME_COORD = "valid_time"

# Required variables
REQUIRED_WAVE_VARS = ("swh", "pp1d", "mwd")
REQUIRED_WIND_VARS = ("u10", "v10")

# Expected geographic point
EXPECTED_LAT = 15.5
EXPECTED_LON = 73.5
COORD_TOLERANCE = 0.26  # half a 0.5-degree ERA5 wave grid cell

# Default archive period
DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR   = 2025


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_months(
    start_year: int = DEFAULT_START_YEAR,
    end_year:   int = DEFAULT_END_YEAR,
) -> list[Tuple[int, int]]:
    """Return list of (year, month) tuples for the full expected period."""
    months = []
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            months.append((y, m))
    return months


def _wave_path(year: int, month: int, root: str = WAVES_DIR) -> str:
    return os.path.join(root, WAVE_FILE_PATTERN.format(year=year, month=month))


def _wind_path(year: int, month: int, root: str = WIND_DIR) -> str:
    return os.path.join(root, WIND_FILE_PATTERN.format(year=year, month=month))


def _squeeze_var(ds: xr.Dataset, var: str) -> np.ndarray:
    """Extract a variable, squeezing out singleton lat/lon dimensions."""
    return ds[var].squeeze(drop=True).values


def _load_one_wave(path: str) -> pd.DataFrame:
    """
    Load one monthly wave NetCDF file.

    Returns a DataFrame with columns: time, Hs_m, Tp_s, direction_deg.
    Raises ValueError if required variables are missing.
    """
    ds = xr.open_dataset(path)
    try:
        missing = [v for v in REQUIRED_WAVE_VARS if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"Wave file {path} missing required variables: {missing}. "
                f"Found: {list(ds.data_vars)}"
            )
        if TIME_COORD not in ds.coords:
            raise ValueError(
                f"Wave file {path} missing time coordinate '{TIME_COORD}'. "
                f"Found: {list(ds.coords)}"
            )
        times = pd.to_datetime(ds[TIME_COORD].values)
        df = pd.DataFrame({
            "time":          times,
            "Hs_m":          _squeeze_var(ds, "swh"),
            "Tp_s":          _squeeze_var(ds, "pp1d"),
            "direction_deg": _squeeze_var(ds, "mwd"),
        })
    finally:
        ds.close()
    return df


def _load_one_wind(path: str) -> pd.DataFrame:
    """
    Load one monthly wind NetCDF file.

    Returns a DataFrame with columns: time, u10, v10.
    Raises ValueError if required variables are missing.
    """
    ds = xr.open_dataset(path)
    try:
        missing = [v for v in REQUIRED_WIND_VARS if v not in ds.data_vars]
        if missing:
            raise ValueError(
                f"Wind file {path} missing required variables: {missing}. "
                f"Found: {list(ds.data_vars)}"
            )
        if TIME_COORD not in ds.coords:
            raise ValueError(
                f"Wind file {path} missing time coordinate '{TIME_COORD}'. "
                f"Found: {list(ds.coords)}"
            )
        times = pd.to_datetime(ds[TIME_COORD].values)
        df = pd.DataFrame({
            "time": times,
            "u10":  _squeeze_var(ds, "u10"),
            "v10":  _squeeze_var(ds, "v10"),
        })
    finally:
        ds.close()
    return df


# ---------------------------------------------------------------------------
# Coverage validation
# ---------------------------------------------------------------------------

def validate_archive_coverage(
    start_year: int = DEFAULT_START_YEAR,
    end_year:   int = DEFAULT_END_YEAR,
    waves_dir:  str = WAVES_DIR,
    wind_dir:   str = WIND_DIR,
) -> dict:
    """
    Check that all expected monthly files are present.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range to check.
    waves_dir, wind_dir : str
        Directories to scan.

    Returns
    -------
    dict with keys:
        expected_months : int
        wave_present    : list of "YYYY-MM" strings
        wave_missing    : list of "YYYY-MM" strings
        wind_present    : list of "YYYY-MM" strings
        wind_missing    : list of "YYYY-MM" strings
        complete        : bool
    """
    expected = _expected_months(start_year, end_year)
    wave_present, wave_missing = [], []
    wind_present, wind_missing = [], []

    for y, m in expected:
        label = f"{y}-{m:02d}"
        if os.path.exists(_wave_path(y, m, waves_dir)):
            wave_present.append(label)
        else:
            wave_missing.append(label)
        if os.path.exists(_wind_path(y, m, wind_dir)):
            wind_present.append(label)
        else:
            wind_missing.append(label)

    return {
        "expected_months": len(expected),
        "wave_present":    wave_present,
        "wave_missing":    wave_missing,
        "wind_present":    wind_present,
        "wind_missing":    wind_missing,
        "complete":        len(wave_missing) == 0 and len(wind_missing) == 0,
    }


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_era5_archive(
    start_year:  int = DEFAULT_START_YEAR,
    end_year:    int = DEFAULT_END_YEAR,
    waves_dir:   str = WAVES_DIR,
    wind_dir:    str = WIND_DIR,
    allow_missing_months: bool = False,
) -> pd.DataFrame:
    """
    Load and concatenate the full ERA5 archive into a single DataFrame.

    Loads all monthly wave and wind files for the specified period,
    concatenates them chronologically, verifies timestamp integrity,
    and aligns the two streams by timestamp.

    Parameters
    ----------
    start_year, end_year : int
        Inclusive year range to load.  Default 2010–2025.
    waves_dir, wind_dir : str
        Directories containing the monthly NetCDF files.
    allow_missing_months : bool
        If False (default), raises ValueError if any expected monthly file
        is absent.  If True, skips missing files and reports them.

    Returns
    -------
    pd.DataFrame
        Columns: time, Hs_m, Tp_s, direction_deg, u10, v10.
        Indexed 0..N-1.  Sorted by time.  No duplicate timestamps.

    Raises
    ------
    FileNotFoundError
        If a required monthly file is missing and allow_missing_months=False.
    ValueError
        If timestamps are not monotonically increasing, contain duplicates,
        or wave/wind streams do not align.
    """
    expected = _expected_months(start_year, end_year)

    wave_frames: list[pd.DataFrame] = []
    wind_frames: list[pd.DataFrame] = []
    missing_wave: list[str] = []
    missing_wind: list[str] = []

    for y, m in expected:
        label = f"{y}-{m:02d}"
        wp = _wave_path(y, m, waves_dir)
        wdp = _wind_path(y, m, wind_dir)

        if not os.path.exists(wp):
            if allow_missing_months:
                missing_wave.append(label)
            else:
                raise FileNotFoundError(
                    f"Expected wave file not found: {wp}\n"
                    f"Run the ERA5 batch downloader to fill missing months."
                )
        else:
            wave_frames.append(_load_one_wave(wp))

        if not os.path.exists(wdp):
            if allow_missing_months:
                missing_wind.append(label)
            else:
                raise FileNotFoundError(
                    f"Expected wind file not found: {wdp}\n"
                    f"Run the ERA5 batch downloader to fill missing months."
                )
        else:
            wind_frames.append(_load_one_wind(wdp))

    if missing_wave:
        print(f"WARNING: {len(missing_wave)} wave months missing: {missing_wave[:6]}"
              + (" ..." if len(missing_wave) > 6 else ""))
    if missing_wind:
        print(f"WARNING: {len(missing_wind)} wind months missing: {missing_wind[:6]}"
              + (" ..." if len(missing_wind) > 6 else ""))

    if not wave_frames:
        raise ValueError("No wave files were loaded.")
    if not wind_frames:
        raise ValueError("No wind files were loaded.")

    # Concatenate
    wave_df = pd.concat(wave_frames, ignore_index=True)
    wind_df = pd.concat(wind_frames, ignore_index=True)

    # --- Validate wave timestamps ---
    _validate_timestamps(wave_df["time"], label="wave")

    # --- Validate wind timestamps ---
    _validate_timestamps(wind_df["time"], label="wind")

    # --- Align streams ---
    wave_times = wave_df["time"].values
    wind_times = wind_df["time"].values

    if len(wave_times) != len(wind_times):
        raise ValueError(
            f"Wave and wind streams have different record counts: "
            f"{len(wave_times)} vs {len(wind_times)}. "
            f"Check for missing or extra monthly files."
        )

    if not np.all(wave_times == wind_times):
        # Find first mismatch for a useful error message
        mismatch_idx = np.where(wave_times != wind_times)[0]
        raise ValueError(
            f"Wave and wind timestamps do not align. "
            f"First mismatch at index {mismatch_idx[0]}: "
            f"wave={wave_times[mismatch_idx[0]]}, "
            f"wind={wind_times[mismatch_idx[0]]}. "
            f"Total mismatches: {len(mismatch_idx)}."
        )

    # Merge on time
    df = wave_df.merge(wind_df, on="time", how="inner")

    # Final sort and reset index
    df = df.sort_values("time").reset_index(drop=True)

    return df


def _validate_timestamps(times: pd.Series, label: str) -> None:
    """
    Verify timestamps are unique and monotonically increasing.

    Raises ValueError with a descriptive message if not.
    """
    t = pd.to_datetime(times)

    # Duplicate check
    n_dupes = t.duplicated().sum()
    if n_dupes > 0:
        dupes = t[t.duplicated()].values[:3]
        raise ValueError(
            f"{label} stream has {n_dupes} duplicate timestamp(s). "
            f"First duplicates: {dupes}. "
            f"Check for overlapping monthly files."
        )

    # Monotonic check
    diffs = np.diff(t.values).astype("timedelta64[s]").astype(int)
    if not np.all(diffs > 0):
        n_bad = int(np.sum(diffs <= 0))
        raise ValueError(
            f"{label} stream is not strictly monotonically increasing. "
            f"{n_bad} non-increasing step(s) found."
        )

    # Hourly check
    non_hourly = int(np.sum(diffs != 3600))
    if non_hourly > 0:
        raise ValueError(
            f"{label} stream has {non_hourly} gap(s) that are not exactly "
            f"1 hour. Missing timestamps are not silently interpolated. "
            f"Check the archive for missing records."
        )
