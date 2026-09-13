"""
download_era5_production.py — Production ERA5 bulk download for the WEC Digital Twin.

Downloads the full 2010–2025 hourly ERA5 archive for the Goa offshore point
(15.5 N, 73.5 E) as a single NetCDF file via one CDS API request.

Output
------
    data/raw/era5/production/era5_goa_2010_2025_v1.nc
    data/raw/era5/production/era5_goa_2010_2025_v1.json   ← provenance sidecar

Credentials
-----------
Read exclusively from ~/.cdsapirc by cdsapi.Client().
Never passed on the command line or hard-coded here.

Usage
-----
    # Dry-run: print the request without downloading
    python scripts/download_era5_production.py --dry-run

    # Execute the download (user must initiate explicitly)
    python scripts/download_era5_production.py

    # Overwrite an existing output file
    python scripts/download_era5_production.py --force

Design notes
------------
* A single multi-year CDS request is used rather than 192 monthly calls.
  The CDS backend handles the aggregation; the result is one NetCDF file.
* Overwrite protection is on by default (--force required to replace).
* A JSON provenance sidecar is written after a successful download.
* The existing per-month script (download_era5.py) is NOT modified.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Production configuration — authoritative constants
# ---------------------------------------------------------------------------

CDS_DATASET = "reanalysis-era5-single-levels"

LATITUDE  = 15.5
LONGITUDE = 73.5
AREA      = [LATITUDE, LONGITUDE, LATITUDE, LONGITUDE]   # [N, W, S, E]

YEARS  = [str(y) for y in range(2010, 2026)]   # 2010 … 2025 inclusive
MONTHS = [f"{m:02d}" for m in range(1, 13)]     # 01 … 12
DAYS   = [f"{d:02d}" for d in range(1, 32)]     # 01 … 31 (CDS ignores invalid dates)
HOURS  = [f"{h:02d}:00" for h in range(24)]     # 00:00 … 23:00

VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",  # → swh
    "mean_wave_period",                                      # → mwp
    "mean_wave_direction",                                   # → mwd
    "peak_wave_period",                                      # → pp1d
    "10m_u_component_of_wind",                               # → u10
    "10m_v_component_of_wind",                               # → v10
]

OUTPUT_PATH   = "data/raw/era5/production/era5_goa_2010_2025_v1.nc"
SCHEMA_VERSION = 1

# Expected record count: 16 years × 365.25 days/yr × 24 h ≈ 140 256 hourly steps
# Exact value depends on leap years; used only for documentation.
EXPECTED_APPROX_RECORDS = 140_256


# ---------------------------------------------------------------------------
# Request builder
# ---------------------------------------------------------------------------

def _build_request() -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable":     VARIABLES,
        "year":         YEARS,
        "month":        MONTHS,
        "day":          DAYS,
        "time":         HOURS,
        "area":         AREA,
        "data_format":        "netcdf",
        "download_format":    "unarchived",
    }


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------

def _write_metadata(nc_path: str, actual_lat: float, actual_lon: float) -> str:
    meta = {
        "schema_version":            SCHEMA_VERSION,
        "dataset_name":              "ERA5 Reanalysis — Single Levels",
        "cds_dataset_id":            CDS_DATASET,
        "project_location":          {"latitude_N": 15.45, "longitude_E": 73.55,
                                      "description": "Goa offshore WEC site"},
        "era5_wave_grid_resolution_deg": 0.5,
        "requested_latitude":        LATITUDE,
        "requested_longitude":       LONGITUDE,
        "actual_latitude_in_file":   actual_lat,
        "actual_longitude_in_file":  actual_lon,
        "variables_requested":       VARIABLES,
        "years":                     YEARS,
        "months":                    MONTHS,
        "temporal_resolution":       "hourly",
        "data_format":               "netcdf",
        "source":                    "Copernicus Climate Data Store / ECMWF ERA5",
        "download_timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "raw_file":                  os.path.basename(nc_path),
        "expected_approx_records":   EXPECTED_APPROX_RECORDS,
        "immutable_note": (
            "This raw file must not be modified. "
            "All processing must operate on copies."
        ),
    }
    base, _ = os.path.splitext(nc_path)
    meta_path = base + ".json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta_path


def _read_actual_coords(nc_path: str) -> tuple[float, float]:
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat = float(ds["latitude"].values.flat[0])
    lon = float(ds["longitude"].values.flat[0])
    ds.close()
    return lat, lon


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(force: bool = False, dry_run: bool = False) -> None:
    output  = OUTPUT_PATH
    request = _build_request()

    print("=" * 64)
    print("ERA5 PRODUCTION DOWNLOAD — 2010–2025")
    print("=" * 64)
    print(f"  Dataset   : {CDS_DATASET}")
    print(f"  Years     : {YEARS[0]} – {YEARS[-1]}  ({len(YEARS)} years)")
    print(f"  Months    : all 12")
    print(f"  Days      : all (01–31; CDS skips invalid dates)")
    print(f"  Hours     : all 24 (00:00–23:00)")
    print(f"  Latitude  : {LATITUDE} N")
    print(f"  Longitude : {LONGITUDE} E")
    print(f"  Variables : {VARIABLES}")
    print(f"  Output    : {output}")
    print(f"  Approx records expected: ~{EXPECTED_APPROX_RECORDS:,}")
    print("=" * 64)

    if dry_run:
        print("\nDRY-RUN: request validated. No download performed.")
        print("Request dict:")
        print(json.dumps(request, indent=2))
        return

    if os.path.exists(output) and not force:
        print(f"\nTarget already exists; refusing to overwrite: {output}")
        print("Use --force to replace it.")
        sys.exit(0)

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    try:
        import cdsapi
        client = cdsapi.Client()
        client.retrieve(CDS_DATASET, request, output)
    except Exception as exc:
        msg = str(exc)
        if any(w in msg.lower() for w in ("key", "token", "auth")):
            print("ERROR: CDS authentication failed. "
                  "Check that ~/.cdsapirc contains valid url and key entries.")
        else:
            print(f"ERROR: CDS request failed — {msg}")
        sys.exit(1)

    print(f"\nDownload complete: {output}")

    try:
        actual_lat, actual_lon = _read_actual_coords(output)
        meta_path = _write_metadata(output, actual_lat, actual_lon)
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
        description="Download ERA5 2010–2025 production archive for the Goa WEC site.",
    )
    p.add_argument("--force",   action="store_true",
                   help="Overwrite existing output file")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the request and exit without downloading")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(force=args.force, dry_run=args.dry_run)
