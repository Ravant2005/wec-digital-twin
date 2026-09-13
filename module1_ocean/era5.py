"""
era5.py — ERA5 wave data loader for the WEC Digital Twin project.

Handles loading and extracting ERA5 reanalysis wave parameters from a NetCDF
file downloaded via the Copernicus Climate Data Store (CDS).

Variables extracted:
  Hs  (significant wave height, metres)   — the average height of the highest
        one-third of waves; the standard measure of sea state severity.
  Tp  (peak wave period, seconds)         — the period of the most energetic
        waves in the spectrum; governs resonance with the buoy.
  direction (mean wave direction, degrees from North, clockwise) — the
        direction FROM which the dominant waves are travelling.
"""

import xarray as xr
import pandas as pd

# ERA5 variable names as stored in the NetCDF file
_REQUIRED_VARS = ("swh", "pp1d", "mwd")

# ERA5 uses 'valid_time' as the time coordinate (not 'time')
_TIME_COORD = "valid_time"


def load_era5(filepath: str) -> xr.Dataset:
    """
    Open an ERA5 NetCDF file and verify it contains the required wave variables.

    Parameters
    ----------
    filepath : str
        Path to the ERA5 NetCDF file.

    Returns
    -------
    xr.Dataset
        The opened dataset, unmodified.

    Raises
    ------
    ValueError
        If any of the required variables (swh, pp1d, mwd) are missing.
    FileNotFoundError
        If the file does not exist (raised by xarray).
    """
    ds = xr.open_dataset(filepath)

    missing = [v for v in _REQUIRED_VARS if v not in ds.data_vars]
    if missing:
        raise ValueError(
            f"ERA5 dataset is missing required variable(s): {missing}. "
            f"Found: {list(ds.data_vars)}"
        )

    return ds


def extract_goa_timeseries(ds: xr.Dataset) -> pd.DataFrame:
    """
    Extract the hourly wave time series from a single-point ERA5 dataset.

    Squeezes out the singleton latitude and longitude dimensions so the result
    is a plain 1-D time series.  Values are preserved exactly as ERA5 provides
    them — no interpolation, smoothing, or gap-filling is applied.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset returned by :func:`load_era5`.

    Returns
    -------
    pd.DataFrame
        Columns:
          time          — UTC timestamp (pandas Timestamp)
          Hs_m          — Significant wave height [metres].  Hs is defined as
                          the mean height of the highest one-third of waves and
                          is the primary measure of sea state energy.
          Tp_s          — Peak wave period [seconds].  The period corresponding
                          to the peak of the wave energy spectrum; determines
                          the dominant wave frequency experienced by the buoy.
          direction_deg — Mean wave direction [degrees, clockwise from North].
                          The compass bearing FROM which the waves are coming.

    Raises
    ------
    ValueError
        If the time coordinate is missing, duplicate timestamps are found, or
        the expected output columns are absent.
    """
    if _TIME_COORD not in ds.coords:
        raise ValueError(
            f"Expected time coordinate '{_TIME_COORD}' not found in dataset. "
            f"Available coordinates: {list(ds.coords)}"
        )

    time_values = pd.to_datetime(ds[_TIME_COORD].values)

    # .squeeze() removes the (1,) latitude and (1,) longitude dimensions,
    # leaving a pure 1-D array indexed only by valid_time.
    hs = ds["swh"].squeeze(drop=True).values
    tp = ds["pp1d"].squeeze(drop=True).values
    direction = ds["mwd"].squeeze(drop=True).values

    df = pd.DataFrame(
        {
            "time": time_values,
            "Hs_m": hs,
            "Tp_s": tp,
            "direction_deg": direction,
        }
    )

    # --- Validation ---
    required_columns = ["time", "Hs_m", "Tp_s", "direction_deg"]
    missing_cols = [c for c in required_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Output DataFrame is missing columns: {missing_cols}")

    if df["time"].duplicated().any():
        n_dupes = df["time"].duplicated().sum()
        raise ValueError(
            f"ERA5 time series contains {n_dupes} duplicate timestamp(s). "
            "The dataset may be corrupted or contain overlapping files."
        )

    return df
