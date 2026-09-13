"""
download_era5.py — Reproducible ERA5 downloader for the WEC Digital Twin project.

Downloads hourly ERA5 single-level wave and wind variables for the Goa offshore
point (15.5 N, 73.5 E) via the Copernicus Climate Data Store (CDS) API.

Credentials are read exclusively from ~/.cdsapirc — never from command-line
arguments or environment variables set by this script.

Usage
-----
# Download January 2024 to an explicit path:
    python scripts/download_era5.py --year 2024 --month 1 \
        --output data/raw/era5/test/era5_goa_2024-01.nc

# Download January 2024 to the auto-generated path:
    python scripts/download_era5.py --year 2024 --month 1

# Overwrite an existing file:
    python scripts/download_era5.py --year 2024 --month 1 --force

Schema version: 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration constants — edit here, not on the command line
# ---------------------------------------------------------------------------

CDS_DATASET = "reanalysis-era5-single-levels"

# Nearest ERA5 0.5-degree ocean-wave grid point to the project location
# Project physical location: 15.45 N, 73.55 E
LATITUDE  = 15.5
LONGITUDE = 73.5

# CDS area box: [N, W, S, E] — single point expressed as a tight box
# Using a 0.01-degree margin so the CDS server selects exactly one grid cell
AREA = [LATITUDE, LONGITUDE, LATITUDE, LONGITUDE]

# Variables to request (CDS long-name identifiers)
VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",  # → swh
    "peak_wave_period",                                      # → pp1d
    "mean_wave_direction",                                   # → mwd
    "mean_wave_period",                                      # → mwp
    "10m_u_component_of_wind",                               # → u10
    "10m_v_component_of_wind",                               # → v10
]

DEFAULT_OUTPUT_DIR = "data/raw/era5/production"
TEST_OUTPUT_DIR    = "data/raw/era5/test"

SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _output_path(year: int, month: int, directory: str) -> str:
    return os.path.join(directory, f"era5_goa_{year}-{month:02d}.nc")


def _metadata_path(nc_path: str) -> str:
    base, _ = os.path.splitext(nc_path)
    return base + ".json"


def _build_request(year: int, month: int) -> dict:
    """Build the CDS API request dict for one calendar month."""
    import calendar
    n_days = calendar.monthrange(year, month)[1]
    days   = [f"{d:02d}" for d in range(1, n_days + 1)]
    hours  = [f"{h:02d}:00" for h in range(24)]

    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year":  [str(year)],
        "month": [f"{month:02d}"],
        "day":   days,
        "time":  hours,
        "area":  AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _write_metadata(nc_path: str, year: int, month: int, actual_lat: float,
                    actual_lon: float) -> str:
    """Write a JSON provenance sidecar next to the NetCDF file."""
    meta = {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": "ERA5 Reanalysis — Single Levels",
        "cds_dataset_id": CDS_DATASET,
        "project_location": {"latitude_N": 15.45, "longitude_E": 73.55,
                             "description": "Goa offshore WEC site"},
        "era5_wave_grid_resolution_deg": 0.5,
        "requested_latitude": LATITUDE,
        "requested_longitude": LONGITUDE,
        "actual_latitude_in_file": actual_lat,
        "actual_longitude_in_file": actual_lon,
        "variables_requested": VARIABLES,
        "year": year,
        "month": month,
        "temporal_resolution": "hourly",
        "data_format": "netcdf",
        "source": "Copernicus Climate Data Store / ECMWF ERA5",
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_file": os.path.basename(nc_path),
        "immutable_note": (
            "This raw file must not be modified. "
            "All processing must operate on copies."
        ),
    }
    meta_path = _metadata_path(nc_path)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


def _read_actual_coords(nc_path: str) -> tuple[float, float]:
    """Return (latitude, longitude) as stored in the downloaded file."""
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat = float(ds["latitude"].values.flat[0])
    lon = float(ds["longitude"].values.flat[0])
    ds.close()
    return lat, lon


# ---------------------------------------------------------------------------
# Main download function
# ---------------------------------------------------------------------------

def download(year: int, month: int, output: str, force: bool = False) -> None:
    """
    Download one month of ERA5 data to *output*.

    Parameters
    ----------
    year, month : int
    output : str
        Destination NetCDF path.
    force : bool
        If False (default), refuse to overwrite an existing file.
    """
    # Safety check — refuse to overwrite without --force
    if os.path.exists(output) and not force:
        print(f"Target already exists; refusing to overwrite: {output}")
        print("Use --force to replace it.")
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    request = _build_request(year, month)

    print("=" * 60)
    print("ERA5 DOWNLOAD")
    print("=" * 60)
    print(f"  Dataset  : {CDS_DATASET}")
    print(f"  Year     : {year}")
    print(f"  Month    : {month:02d}")
    print(f"  Latitude : {LATITUDE} N")
    print(f"  Longitude: {LONGITUDE} E")
    print(f"  Variables: {VARIABLES}")
    print(f"  Output   : {output}")
    print("=" * 60)

    try:
        import cdsapi
        client = cdsapi.Client()
        client.retrieve(CDS_DATASET, request, output)
    except Exception as exc:
        # Never print the exception message verbatim if it might contain keys
        msg = str(exc)
        if "key" in msg.lower() or "token" in msg.lower() or "auth" in msg.lower():
            print("ERROR: CDS authentication failed. "
                  "Check that ~/.cdsapirc contains valid url and key entries.")
        else:
            print(f"ERROR: CDS request failed — {msg}")
        sys.exit(1)

    print(f"\nDownload complete: {output}")

    # Write provenance metadata
    try:
        actual_lat, actual_lon = _read_actual_coords(output)
        meta_path = _write_metadata(output, year, month, actual_lat, actual_lon)
        print(f"Metadata written : {meta_path}")
        if abs(actual_lat - LATITUDE) > 0.26 or abs(actual_lon - LONGITUDE) > 0.26:
            print(f"WARNING: actual grid point ({actual_lat}, {actual_lon}) "
                  f"differs from requested ({LATITUDE}, {LONGITUDE}) by more "
                  f"than half a grid cell.")
    except Exception as exc:
        print(f"WARNING: could not write metadata sidecar — {exc}")

    print("STATUS: SUCCESS")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download ERA5 wave/wind data for the Goa WEC site.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--year",  type=int, required=True, help="Calendar year (e.g. 2024)")
    p.add_argument("--month", type=int, required=True, choices=range(1, 13),
                   metavar="MONTH", help="Calendar month 1–12")
    p.add_argument("--output", type=str, default=None,
                   help="Output NetCDF path (auto-generated if omitted)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing output file")
    p.add_argument("--test-dir", action="store_true",
                   help=f"Write to {TEST_OUTPUT_DIR}/ instead of {DEFAULT_OUTPUT_DIR}/")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.output:
        out = args.output
    else:
        directory = TEST_OUTPUT_DIR if args.test_dir else DEFAULT_OUTPUT_DIR
        out = _output_path(args.year, args.month, directory)

    download(args.year, args.month, out, force=args.force)
