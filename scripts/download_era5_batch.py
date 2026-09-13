"""
download_era5_batch.py — Production ERA5 batch downloader for the WEC Digital Twin.

Downloads ERA5 hourly data for the Goa offshore point (15.5 N, 73.5 E) as
separate monthly files for two independent streams:

  waves  — ocean-wave variables on the 0.5° wave grid
  wind   — atmospheric wind variables on the native atmospheric grid

Wave and wind variables are intentionally kept in separate requests because
ERA5 ocean-wave and atmospheric fields use different spatial grids and CDS
rejects mixed-grid NetCDF requests.

Output layout
-------------
  data/raw/era5/production/waves/era5_waves_goa_YYYY-MM.nc   + .json sidecar
  data/raw/era5/production/wind/era5_wind_goa_YYYY-MM.nc     + .json sidecar
  data/raw/era5/production/manifest.json

Credentials
-----------
Read exclusively from ~/.cdsapirc by cdsapi.Client().  Never passed on the
command line or hard-coded here.

Usage
-----
  # Dry-run (no download, prints all planned requests)
  python scripts/download_era5_batch.py --dry-run

  # Download everything 2010-2025, both streams
  python scripts/download_era5_batch.py

  # Waves only, 2020-2022
  python scripts/download_era5_batch.py --stream waves --start-year 2020 --end-year 2022

  # Resume from a specific month
  python scripts/download_era5_batch.py --start-month 2015-06 --end-month 2015-12

  # Replace existing files
  python scripts/download_era5_batch.py --force

  # Adjust pause between requests and retry count
  python scripts/download_era5_batch.py --pause-seconds 5 --retry 3
"""

from __future__ import annotations

import argparse
import calendar
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Iterator

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

CDS_DATASET = "reanalysis-era5-single-levels"

# Nearest ERA5 0.5-degree ocean-wave grid point to the project location
WAVE_LATITUDE  = 15.5
WAVE_LONGITUDE = 73.5

# For atmospheric wind, request the same nominal point.
# ERA5 atmospheric data have a finer native grid; CDS will return the nearest
# grid point.  Raw files preserve whatever coordinate CDS resolves to.
WIND_LATITUDE  = 15.5
WIND_LONGITUDE = 73.5

WAVE_VARIABLES = [
    "significant_height_of_combined_wind_waves_and_swell",  # → swh
    "mean_wave_period",                                      # → mwp
    "mean_wave_direction",                                   # → mwd
    "peak_wave_period",                                      # → pp1d
]

WIND_VARIABLES = [
    "10m_u_component_of_wind",   # → u10
    "10m_v_component_of_wind",   # → v10
]

PRODUCTION_ROOT = "data/raw/era5/production"
WAVES_DIR       = os.path.join(PRODUCTION_ROOT, "waves")
WIND_DIR        = os.path.join(PRODUCTION_ROOT, "wind")
MANIFEST_PATH   = os.path.join(PRODUCTION_ROOT, "manifest.json")

DEFAULT_START_YEAR = 2010
DEFAULT_END_YEAR   = 2025
DEFAULT_PAUSE_S    = 3
DEFAULT_RETRIES    = 3
SCHEMA_VERSION     = 1

HOURS = [f"{h:02d}:00" for h in range(24)]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def wave_path(year: int, month: int) -> str:
    return os.path.join(WAVES_DIR, f"era5_waves_goa_{year}-{month:02d}.nc")


def wind_path(year: int, month: int) -> str:
    return os.path.join(WIND_DIR, f"era5_wind_goa_{year}-{month:02d}.nc")


def _sidecar(nc_path: str) -> str:
    base, _ = os.path.splitext(nc_path)
    return base + ".json"


def expected_hours(year: int, month: int) -> int:
    """Return the exact number of hourly records for a given year/month."""
    return calendar.monthrange(year, month)[1] * 24


# ---------------------------------------------------------------------------
# Month enumeration
# ---------------------------------------------------------------------------

def iter_months(
    start_year: int, start_month: int,
    end_year: int,   end_month: int,
) -> Iterator[tuple[int, int]]:
    """Yield (year, month) pairs in chronological order, inclusive."""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


# ---------------------------------------------------------------------------
# CDS request builders
# ---------------------------------------------------------------------------

def _build_wave_request(year: int, month: int) -> dict:
    n_days = calendar.monthrange(year, month)[1]
    return {
        "product_type":    ["reanalysis"],
        "variable":        WAVE_VARIABLES,
        "year":            [str(year)],
        "month":           [f"{month:02d}"],
        "day":             [f"{d:02d}" for d in range(1, n_days + 1)],
        "time":            HOURS,
        "area":            [WAVE_LATITUDE, WAVE_LONGITUDE,
                            WAVE_LATITUDE, WAVE_LONGITUDE],
        "data_format":     "netcdf",
        "download_format": "unarchived",
    }


def _build_wind_request(year: int, month: int) -> dict:
    n_days = calendar.monthrange(year, month)[1]
    return {
        "product_type":    ["reanalysis"],
        "variable":        WIND_VARIABLES,
        "year":            [str(year)],
        "month":           [f"{month:02d}"],
        "day":             [f"{d:02d}" for d in range(1, n_days + 1)],
        "time":            HOURS,
        "area":            [WIND_LATITUDE, WIND_LONGITUDE,
                            WIND_LATITUDE, WIND_LONGITUDE],
        "data_format":     "netcdf",
        "download_format": "unarchived",
    }


# ---------------------------------------------------------------------------
# Provenance sidecar
# ---------------------------------------------------------------------------

def _write_sidecar(
    nc_path: str, stream: str, year: int, month: int,
    actual_lat: float, actual_lon: float,
    req_lat: float, req_lon: float,
    variables: list[str],
) -> str:
    meta = {
        "schema_version":            SCHEMA_VERSION,
        "dataset_name":              "ERA5 Reanalysis — Single Levels",
        "cds_dataset_id":            CDS_DATASET,
        "stream":                    stream,
        "project_location":          {"latitude_N": 15.45, "longitude_E": 73.55,
                                      "description": "Goa offshore WEC site"},
        "era5_wave_grid_resolution_deg": 0.5,
        "requested_latitude":        req_lat,
        "requested_longitude":       req_lon,
        "actual_latitude_in_file":   actual_lat,
        "actual_longitude_in_file":  actual_lon,
        "variables_requested":       variables,
        "year":                      year,
        "month":                     month,
        "temporal_resolution":       "hourly",
        "data_format":               "netcdf",
        "source":                    "Copernicus Climate Data Store / ECMWF ERA5",
        "download_timestamp_utc":    datetime.now(timezone.utc).isoformat(),
        "raw_file":                  os.path.basename(nc_path),
        "immutable_note": (
            "This raw file must not be modified. "
            "All processing must operate on copies."
        ),
    }
    path = _sidecar(nc_path)
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    return path


def _read_coords(nc_path: str) -> tuple[float, float]:
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    lat = float(ds["latitude"].values.flat[0])
    lon = float(ds["longitude"].values.flat[0])
    ds.close()
    return lat, lon


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH) as fh:
            return json.load(fh)
    return {"schema_version": SCHEMA_VERSION, "records": {}}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(MANIFEST_PATH)), exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        json.dump(manifest, fh, indent=2)


def _manifest_key(stream: str, year: int, month: int) -> str:
    return f"{stream}_{year}-{month:02d}"


def _manifest_update(
    manifest: dict, stream: str, year: int, month: int,
    status: str, path: str = "",
    n_records: int = 0, actual_lat: float = 0.0, actual_lon: float = 0.0,
    variables: list | None = None, error: str = "",
    validation_status: str = "",
) -> None:
    key = _manifest_key(stream, year, month)
    manifest["records"][key] = {
        "year":              year,
        "month":             month,
        "stream":            stream,
        "path":              path,
        "status":            status,
        "n_records":         n_records,
        "actual_latitude":   actual_lat,
        "actual_longitude":  actual_lon,
        "variables":         variables or [],
        "download_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "validation_status": validation_status,
        "error":             error,
    }


# ---------------------------------------------------------------------------
# Single-month download with retry
# ---------------------------------------------------------------------------

def _download_one(
    stream: str, year: int, month: int,
    output: str, force: bool,
    retries: int, pause_s: float,
) -> tuple[str, str]:
    """
    Download one month for one stream.

    Returns
    -------
    (status, error_message)
    status: "ok" | "skipped" | "failed"
    """
    if os.path.exists(output) and not force:
        print(f"  SKIP existing: {output}")
        return "skipped", ""

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)

    if stream == "waves":
        request = _build_wave_request(year, month)
        req_lat, req_lon = WAVE_LATITUDE, WAVE_LONGITUDE
        variables = WAVE_VARIABLES
    else:
        request = _build_wind_request(year, month)
        req_lat, req_lon = WIND_LATITUDE, WIND_LONGITUDE
        variables = WIND_VARIABLES

    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            import cdsapi
            client = cdsapi.Client()
            client.retrieve(CDS_DATASET, request, output)
            last_error = ""
            break
        except Exception as exc:
            last_error = str(exc)
            is_auth = any(w in last_error.lower() for w in ("key", "token", "auth", "403"))
            if is_auth:
                print(f"  AUTH ERROR — not retrying: check ~/.cdsapirc")
                return "failed", "authentication error"
            if attempt < retries:
                delay = pause_s * (2 ** (attempt - 1))
                print(f"  attempt {attempt}/{retries} failed, retrying in {delay:.0f}s …")
                time.sleep(delay)
            else:
                print(f"  all {retries} attempts failed: {last_error}")
                return "failed", last_error

    # Write sidecar
    try:
        actual_lat, actual_lon = _read_coords(output)
        _write_sidecar(output, stream, year, month,
                       actual_lat, actual_lon, req_lat, req_lon, variables)
    except Exception as exc:
        print(f"  WARNING: sidecar write failed — {exc}")
        actual_lat, actual_lon = req_lat, req_lon

    return "ok", ""


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(
    start_year: int, start_month: int,
    end_year: int,   end_month: int,
    streams: list[str],
    force: bool      = False,
    dry_run: bool    = False,
    retries: int     = DEFAULT_RETRIES,
    pause_s: float   = DEFAULT_PAUSE_S,
) -> int:
    """
    Execute the batch download.

    Returns exit code: 0 = all succeeded/skipped, 1 = any failure.
    """
    months = list(iter_months(start_year, start_month, end_year, end_month))
    total_requests = len(months) * len(streams)

    print("=" * 64)
    print("ERA5 BATCH DOWNLOAD")
    print("=" * 64)
    print(f"  Period  : {start_year}-{start_month:02d} → {end_year}-{end_month:02d}")
    print(f"  Streams : {streams}")
    print(f"  Months  : {len(months)}")
    print(f"  Requests: {total_requests}  ({len(months)} months × {len(streams)} streams)")
    print(f"  Force   : {force}")
    print(f"  Dry-run : {dry_run}")
    print(f"  Retries : {retries}")
    print(f"  Pause   : {pause_s}s between requests")
    print("=" * 64)

    if dry_run:
        print("\nDRY-RUN — planned requests:\n")
        for y, m in months:
            for stream in streams:
                if stream == "waves":
                    path = wave_path(y, m)
                    req  = _build_wave_request(y, m)
                else:
                    path = wind_path(y, m)
                    req  = _build_wind_request(y, m)
                exists = "EXISTS" if os.path.exists(path) else "missing"
                print(f"  {stream:5s}  {y}-{m:02d}  →  {path}  [{exists}]")
        print(f"\nTotal: {total_requests} requests planned.")
        return 0

    manifest = _load_manifest()
    n_ok = n_skip = n_fail = 0
    request_count = 0

    for y, m in months:
        for stream in streams:
            output = wave_path(y, m) if stream == "waves" else wind_path(y, m)
            variables = WAVE_VARIABLES if stream == "waves" else WIND_VARIABLES
            expected = expected_hours(y, m)

            print(f"\n[{request_count+1}/{total_requests}] {stream} {y}-{m:02d}  →  {output}")

            status, error = _download_one(
                stream, y, m, output, force, retries, pause_s
            )

            n_records = 0
            actual_lat = actual_lon = 0.0
            if status == "ok":
                try:
                    actual_lat, actual_lon = _read_coords(output)
                    import xarray as xr
                    ds = xr.open_dataset(output)
                    tc = "valid_time" if "valid_time" in ds.coords else "time"
                    n_records = int(ds.dims.get(tc, 0))
                    ds.close()
                    if n_records != expected:
                        print(f"  WARNING: expected {expected} records, got {n_records}")
                except Exception:
                    pass
                n_ok += 1
                print(f"  OK  ({n_records} records)")
            elif status == "skipped":
                n_skip += 1
            else:
                n_fail += 1
                print(f"  FAILED: {error}")

            _manifest_update(
                manifest, stream, y, m,
                status=status, path=output,
                n_records=n_records,
                actual_lat=actual_lat, actual_lon=actual_lon,
                variables=variables, error=error,
            )
            _save_manifest(manifest)
            request_count += 1

            if status == "ok" and request_count < total_requests:
                time.sleep(pause_s)

    print("\n" + "=" * 64)
    print(f"BATCH COMPLETE")
    print(f"  successful : {n_ok}")
    print(f"  skipped    : {n_skip}")
    print(f"  failed     : {n_fail}")
    print(f"  manifest   : {MANIFEST_PATH}")
    print("=" * 64)

    return 1 if n_fail > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_month(s: str) -> tuple[int, int]:
    """Parse 'YYYY-MM' into (year, month)."""
    try:
        dt = datetime.strptime(s, "%Y-%m")
        return dt.year, dt.month
    except ValueError:
        raise argparse.ArgumentTypeError(f"Expected YYYY-MM, got '{s}'")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-download ERA5 wave and wind data for the Goa WEC site.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--start-year",  type=int, default=DEFAULT_START_YEAR)
    p.add_argument("--end-year",    type=int, default=DEFAULT_END_YEAR)
    p.add_argument("--stream",      choices=["waves", "wind", "all"],
                   default="all", dest="stream")
    p.add_argument("--start-month", type=_parse_month, default=None,
                   metavar="YYYY-MM",
                   help="Override start to a specific month (overrides --start-year)")
    p.add_argument("--end-month",   type=_parse_month, default=None,
                   metavar="YYYY-MM",
                   help="Override end to a specific month (overrides --end-year)")
    p.add_argument("--dry-run",     action="store_true")
    p.add_argument("--force",       action="store_true",
                   help="Overwrite existing files")
    p.add_argument("--retry",       type=int, default=DEFAULT_RETRIES,
                   dest="retries", metavar="N")
    p.add_argument("--pause-seconds", type=float, default=DEFAULT_PAUSE_S,
                   dest="pause_s")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    sy, sm = (args.start_month if args.start_month
              else (args.start_year, 1))
    ey, em = (args.end_month   if args.end_month
              else (args.end_year, 12))

    streams = ["waves", "wind"] if args.stream == "all" else [args.stream]

    code = run_batch(
        start_year=sy,  start_month=sm,
        end_year=ey,    end_month=em,
        streams=streams,
        force=args.force,
        dry_run=args.dry_run,
        retries=args.retries,
        pause_s=args.pause_s,
    )
    sys.exit(code)
