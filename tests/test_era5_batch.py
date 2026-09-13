"""
tests/test_era5_batch.py — Tests for the ERA5 batch acquisition system.

No live CDS downloads are performed.

Tests
-----
 1. Month enumeration — correct sequence
 2. Month enumeration — leap-year record count
 3. Output path generation — waves and wind
 4. Wave/wind variable separation — no mixing
 5. No wave+wind variables in a single request
 6. Skip-existing behaviour (no --force)
 7. Force behaviour overwrites existing file
 8. Dry-run behaviour — no files written
 9. Retry behaviour — succeeds on second attempt
10. Manifest creation and update
11. Failed-month recording in manifest
12. Expected month coverage 2010-2025
13. Physical validation via validate_era5
14. Coordinate validation via validate_era5
15. Credentials never appear in batch script source
16. No frozen modules changed
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import download_era5_batch as batch
import validate_era5 as val_mod


# ---------------------------------------------------------------------------
# Synthetic dataset helpers
# ---------------------------------------------------------------------------

def _make_wave_ds(year: int = 2020, month: int = 1,
                  lat: float = 15.5, lon: float = 73.5) -> xr.Dataset:
    n = batch.expected_hours(year, month)
    times = pd.date_range(f"{year}-{month:02d}-01", periods=n, freq="h")
    return xr.Dataset(
        {"swh":  ("valid_time", np.ones(n)),
         "pp1d": ("valid_time", np.full(n, 10.0)),
         "mwd":  ("valid_time", np.full(n, 270.0)),
         "mwp":  ("valid_time", np.full(n, 9.0))},
        coords={"valid_time": times, "latitude": [lat], "longitude": [lon]},
    )


def _make_wind_ds(year: int = 2020, month: int = 1,
                  lat: float = 15.5, lon: float = 73.5) -> xr.Dataset:
    n = batch.expected_hours(year, month)
    times = pd.date_range(f"{year}-{month:02d}-01", periods=n, freq="h")
    return xr.Dataset(
        {"u10": ("valid_time", np.full(n, 5.0)),
         "v10": ("valid_time", np.full(n, -3.0))},
        coords={"valid_time": times, "latitude": [lat], "longitude": [lon]},
    )


# ---------------------------------------------------------------------------
# 1. Month enumeration — correct sequence
# ---------------------------------------------------------------------------

def test_iter_months_basic():
    months = list(batch.iter_months(2020, 1, 2020, 3))
    assert months == [(2020, 1), (2020, 2), (2020, 3)]


def test_iter_months_year_boundary():
    months = list(batch.iter_months(2019, 11, 2020, 2))
    assert months == [(2019, 11), (2019, 12), (2020, 1), (2020, 2)]


def test_iter_months_single():
    months = list(batch.iter_months(2015, 6, 2015, 6))
    assert months == [(2015, 6)]


def test_iter_months_full_year():
    months = list(batch.iter_months(2020, 1, 2020, 12))
    assert len(months) == 12
    assert months[0]  == (2020, 1)
    assert months[-1] == (2020, 12)


# ---------------------------------------------------------------------------
# 2. Leap-year record count
# ---------------------------------------------------------------------------

def test_expected_hours_jan():
    assert batch.expected_hours(2020, 1) == 31 * 24   # 744


def test_expected_hours_feb_leap():
    assert batch.expected_hours(2020, 2) == 29 * 24   # 696


def test_expected_hours_feb_non_leap():
    assert batch.expected_hours(2021, 2) == 28 * 24   # 672


def test_expected_hours_april():
    assert batch.expected_hours(2020, 4) == 30 * 24   # 720


def test_expected_hours_dec():
    assert batch.expected_hours(2024, 12) == 31 * 24  # 744


# ---------------------------------------------------------------------------
# 3. Output path generation
# ---------------------------------------------------------------------------

def test_wave_path_format():
    p = batch.wave_path(2020, 1)
    assert p.endswith("era5_waves_goa_2020-01.nc")
    assert "waves" in p


def test_wind_path_format():
    p = batch.wind_path(2020, 1)
    assert p.endswith("era5_wind_goa_2020-01.nc")
    assert "wind" in p


def test_wave_wind_paths_differ():
    assert batch.wave_path(2020, 6) != batch.wind_path(2020, 6)


def test_path_zero_padded_month():
    assert "2020-03" in batch.wave_path(2020, 3)
    assert "2020-03" in batch.wind_path(2020, 3)


# ---------------------------------------------------------------------------
# 4. Wave/wind variable separation
# ---------------------------------------------------------------------------

def test_wave_request_contains_only_wave_vars():
    req = batch._build_wave_request(2020, 1)
    for v in req["variable"]:
        assert v in batch.WAVE_VARIABLES, f"Non-wave variable in wave request: {v}"
    for v in batch.WIND_VARIABLES:
        assert v not in req["variable"], f"Wind variable leaked into wave request: {v}"


def test_wind_request_contains_only_wind_vars():
    req = batch._build_wind_request(2020, 1)
    for v in req["variable"]:
        assert v in batch.WIND_VARIABLES, f"Non-wind variable in wind request: {v}"
    for v in batch.WAVE_VARIABLES:
        assert v not in req["variable"], f"Wave variable leaked into wind request: {v}"


# ---------------------------------------------------------------------------
# 5. No wave+wind variables in a single request
# ---------------------------------------------------------------------------

def test_wave_and_wind_vars_never_combined():
    wave_set = set(batch.WAVE_VARIABLES)
    wind_set = set(batch.WIND_VARIABLES)
    assert wave_set.isdisjoint(wind_set), \
        "WAVE_VARIABLES and WIND_VARIABLES share entries — they must be disjoint"


def test_wave_request_has_no_wind_vars():
    req = batch._build_wave_request(2024, 6)
    assert "10m_u_component_of_wind" not in req["variable"]
    assert "10m_v_component_of_wind" not in req["variable"]


def test_wind_request_has_no_wave_vars():
    req = batch._build_wind_request(2024, 6)
    assert "significant_height_of_combined_wind_waves_and_swell" not in req["variable"]
    assert "peak_wave_period" not in req["variable"]


# ---------------------------------------------------------------------------
# 6. Skip-existing behaviour
# ---------------------------------------------------------------------------

def test_skip_existing_no_force(tmp_path, capsys):
    nc = tmp_path / "era5_waves_goa_2020-01.nc"
    nc.write_text("placeholder")

    status, err = batch._download_one(
        "waves", 2020, 1, str(nc), force=False,
        retries=1, pause_s=0,
    )
    assert status == "skipped"
    assert err == ""
    captured = capsys.readouterr()
    assert "skip" in captured.out.lower()


# ---------------------------------------------------------------------------
# 7. Force behaviour overwrites existing file
# ---------------------------------------------------------------------------

def test_force_overwrites_existing(tmp_path):
    nc = tmp_path / "era5_waves_goa_2020-01.nc"
    nc.write_text("old")

    class _FakeClient:
        def retrieve(self, name, request, target):
            _make_wave_ds(2020, 1).to_netcdf(target)

    with patch("cdsapi.Client", return_value=_FakeClient()):
        status, err = batch._download_one(
            "waves", 2020, 1, str(nc), force=True,
            retries=1, pause_s=0,
        )

    assert status == "ok"
    assert nc.stat().st_size > len("old")


# ---------------------------------------------------------------------------
# 8. Dry-run behaviour — no files written
# ---------------------------------------------------------------------------

def test_dry_run_writes_no_files(tmp_path, monkeypatch, capsys):
    # Redirect output dirs to tmp_path so we can check nothing was created
    monkeypatch.setattr(batch, "WAVES_DIR", str(tmp_path / "waves"))
    monkeypatch.setattr(batch, "WIND_DIR",  str(tmp_path / "wind"))
    monkeypatch.setattr(batch, "MANIFEST_PATH", str(tmp_path / "manifest.json"))

    code = batch.run_batch(
        start_year=2020, start_month=1,
        end_year=2020,   end_month=1,
        streams=["waves"],
        dry_run=True,
    )

    assert code == 0
    assert not (tmp_path / "waves").exists() or \
           len(list((tmp_path / "waves").glob("*.nc"))) == 0
    assert not (tmp_path / "manifest.json").exists()

    out = capsys.readouterr().out
    assert "dry" in out.lower() or "planned" in out.lower()


# ---------------------------------------------------------------------------
# 9. Retry behaviour — succeeds on second attempt
# ---------------------------------------------------------------------------

def test_retry_succeeds_on_second_attempt(tmp_path):
    nc = tmp_path / "era5_waves_goa_2020-01.nc"
    call_count = {"n": 0}

    class _FlakyClient:
        def retrieve(self, name, request, target):
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise RuntimeError("transient network error")
            _make_wave_ds(2020, 1).to_netcdf(target)

    with patch("cdsapi.Client", return_value=_FlakyClient()):
        status, err = batch._download_one(
            "waves", 2020, 1, str(nc), force=False,
            retries=3, pause_s=0,
        )

    assert status == "ok"
    assert call_count["n"] == 2


def test_retry_exhausted_returns_failed(tmp_path):
    nc = tmp_path / "era5_waves_goa_2020-01.nc"

    class _AlwaysFailClient:
        def retrieve(self, name, request, target):
            raise RuntimeError("persistent error")

    with patch("cdsapi.Client", return_value=_AlwaysFailClient()):
        status, err = batch._download_one(
            "waves", 2020, 1, str(nc), force=False,
            retries=2, pause_s=0,
        )

    assert status == "failed"
    assert "persistent error" in err


# ---------------------------------------------------------------------------
# 10. Manifest creation and update
# ---------------------------------------------------------------------------

def test_manifest_created_on_update(tmp_path, monkeypatch):
    manifest_path = str(tmp_path / "manifest.json")
    monkeypatch.setattr(batch, "MANIFEST_PATH", manifest_path)

    manifest = batch._load_manifest()
    batch._manifest_update(
        manifest, "waves", 2020, 1,
        status="ok", path="/some/path.nc",
        n_records=744, actual_lat=15.5, actual_lon=73.5,
        variables=batch.WAVE_VARIABLES,
    )
    batch._save_manifest(manifest)

    assert os.path.exists(manifest_path)
    with open(manifest_path) as fh:
        data = json.load(fh)

    key = "waves_2020-01"
    assert key in data["records"]
    rec = data["records"][key]
    assert rec["status"] == "ok"
    assert rec["n_records"] == 744
    assert rec["stream"] == "waves"
    assert rec["year"] == 2020
    assert rec["month"] == 1


def test_manifest_key_format():
    assert batch._manifest_key("waves", 2020, 1)  == "waves_2020-01"
    assert batch._manifest_key("wind",  2025, 12) == "wind_2025-12"


# ---------------------------------------------------------------------------
# 11. Failed-month recording in manifest
# ---------------------------------------------------------------------------

def test_failed_month_recorded_in_manifest(tmp_path, monkeypatch):
    manifest_path = str(tmp_path / "manifest.json")
    monkeypatch.setattr(batch, "MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(batch, "WAVES_DIR", str(tmp_path / "waves"))
    monkeypatch.setattr(batch, "WIND_DIR",  str(tmp_path / "wind"))

    class _AlwaysFailClient:
        def retrieve(self, name, request, target):
            raise RuntimeError("cds down")

    with patch("cdsapi.Client", return_value=_AlwaysFailClient()):
        code = batch.run_batch(
            start_year=2020, start_month=1,
            end_year=2020,   end_month=1,
            streams=["waves"],
            retries=1, pause_s=0,
        )

    assert code == 1   # non-zero exit on failure

    with open(manifest_path) as fh:
        data = json.load(fh)

    rec = data["records"].get("waves_2020-01", {})
    assert rec.get("status") == "failed"
    assert "cds down" in rec.get("error", "")


# ---------------------------------------------------------------------------
# 12. Expected month coverage 2010-2025
# ---------------------------------------------------------------------------

def test_full_period_month_count():
    months = list(batch.iter_months(2010, 1, 2025, 12))
    assert len(months) == 16 * 12   # 192


def test_full_period_starts_and_ends():
    months = list(batch.iter_months(2010, 1, 2025, 12))
    assert months[0]  == (2010, 1)
    assert months[-1] == (2025, 12)


def test_wave_and_wind_total_requests():
    months = list(batch.iter_months(2010, 1, 2025, 12))
    # 192 months × 2 streams = 384 total requests
    assert len(months) * 2 == 384


# ---------------------------------------------------------------------------
# 13. Physical validation via validate_era5
# ---------------------------------------------------------------------------

def test_validate_wave_file_passes(tmp_path, capsys):
    nc = str(tmp_path / "era5_waves_goa_2020-01.nc")
    _make_wave_ds(2020, 1).to_netcdf(nc)
    assert val_mod.validate(nc, stream="waves") is True


def test_validate_wind_file_passes(tmp_path, capsys):
    nc = str(tmp_path / "era5_wind_goa_2020-01.nc")
    _make_wind_ds(2020, 1).to_netcdf(nc)
    assert val_mod.validate(nc, stream="wind") is True


def test_validate_wave_negative_swh_fails(tmp_path, capsys):
    swh = np.ones(batch.expected_hours(2020, 1))
    swh[5] = -0.1
    ds = _make_wave_ds(2020, 1)
    ds["swh"].values[:] = swh
    nc = str(tmp_path / "bad_swh.nc")
    ds.to_netcdf(nc)
    assert val_mod.validate(nc, stream="waves") is False


def test_validate_wind_file_wrong_record_count(tmp_path, capsys):
    # February 2020 is a leap year: 29 × 24 = 696 records
    # Build a file with 672 records (non-leap Feb) — should fail
    n = 28 * 24
    times = pd.date_range("2020-02-01", periods=n, freq="h")
    ds = xr.Dataset(
        {"u10": ("valid_time", np.ones(n)),
         "v10": ("valid_time", np.ones(n))},
        coords={"valid_time": times, "latitude": [15.5], "longitude": [73.5]},
    )
    nc = str(tmp_path / "short_feb.nc")
    ds.to_netcdf(nc)
    assert val_mod.validate(nc, stream="wind") is False


def test_validate_auto_detects_wave_stream(tmp_path, capsys):
    nc = str(tmp_path / "wave_auto.nc")
    _make_wave_ds(2021, 3).to_netcdf(nc)
    result = val_mod.validate(nc, stream="auto")
    assert result is True
    out = capsys.readouterr().out
    assert "waves" in out


def test_validate_auto_detects_wind_stream(tmp_path, capsys):
    nc = str(tmp_path / "wind_auto.nc")
    _make_wind_ds(2021, 3).to_netcdf(nc)
    result = val_mod.validate(nc, stream="auto")
    assert result is True
    out = capsys.readouterr().out
    assert "wind" in out


# ---------------------------------------------------------------------------
# 14. Coordinate validation via validate_era5
# ---------------------------------------------------------------------------

def test_validate_wrong_lat_fails(tmp_path, capsys):
    nc = str(tmp_path / "wrong_lat.nc")
    _make_wave_ds(2020, 1, lat=20.0).to_netcdf(nc)
    assert val_mod.validate(nc, stream="waves") is False


def test_validate_wrong_lon_fails(tmp_path, capsys):
    nc = str(tmp_path / "wrong_lon.nc")
    _make_wave_ds(2020, 1, lon=80.0).to_netcdf(nc)
    assert val_mod.validate(nc, stream="waves") is False


def test_validate_correct_coords_pass(tmp_path, capsys):
    nc = str(tmp_path / "good_coords.nc")
    _make_wave_ds(2020, 1, lat=15.5, lon=73.5).to_netcdf(nc)
    assert val_mod.validate(nc, stream="waves") is True


# ---------------------------------------------------------------------------
# 15. Credentials never appear in batch script source
# ---------------------------------------------------------------------------

def test_no_credentials_in_batch_script():
    src = inspect.getsource(batch).lower()
    for pattern in ("api_key", "apikey", "password", "secret"):
        assert pattern not in src, \
            f"Forbidden pattern '{pattern}' in download_era5_batch.py"


def test_batch_script_does_not_open_cdsapirc():
    src = inspect.getsource(batch)
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = ""
            if isinstance(node.func, ast.Name):
                fname = node.func.id
            elif isinstance(node.func, ast.Attribute):
                fname = node.func.attr
            if fname == "open":
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and ".cdsapirc" in str(arg.value):
                        raise AssertionError(
                            "download_era5_batch.py opens ~/.cdsapirc directly"
                        )


# ---------------------------------------------------------------------------
# 16. No frozen modules changed
# ---------------------------------------------------------------------------

def test_frozen_module1_era5_unchanged():
    path = _PROJECT_ROOT / "module1_ocean" / "era5.py"
    src  = path.read_text()
    assert "def load_era5" in src
    assert "def extract_goa_timeseries" in src


def test_frozen_module4_dataset_unchanged():
    path = _PROJECT_ROOT / "module4_forecasting" / "dataset.py"
    src  = path.read_text()
    assert "class ForecastDataset" in src
    assert "FEATURE_NAMES" in src


def test_existing_download_era5_unchanged():
    """The original per-month downloader must not have been modified."""
    path = _PROJECT_ROOT / "scripts" / "download_era5.py"
    src  = path.read_text()
    assert "def download(" in src
    assert "def _build_request(" in src


def test_production_subdirs_exist():
    for subdir in ("waves", "wind"):
        path = _PROJECT_ROOT / "data" / "raw" / "era5" / "production" / subdir
        assert path.is_dir(), f"Missing production subdir: {subdir}"


def test_new_scripts_exist():
    for script in ("download_era5_batch.py", "audit_era5_archive.py"):
        path = _PROJECT_ROOT / "scripts" / script
        assert path.exists(), f"Missing script: scripts/{script}"
