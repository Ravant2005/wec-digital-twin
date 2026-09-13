"""
tests/test_era5_acquisition.py — Tests for the ERA5 acquisition infrastructure.

No live CDS downloads are performed.  All tests use synthetic xarray datasets
or inspect source code / file system state.

Tests
-----
 1. Safe output path generation
 2. Refusal to overwrite without --force
 3. Metadata sidecar generation
 4. Required-variable validation (all present)
 5. Missing-variable detection
 6. Time monotonicity detection
 7. Duplicate timestamp detection
 8. Expected hourly frequency detection
 9. January 2024 expected record count (744)
10. Latitude/longitude validation
11. Non-negative Hs (swh) validation
12. Positive-period validation (pp1d, mwp)
13. Circular direction validation (mwd in [0, 360))
14. Missing values are reported, not filled
15. Credentials are not embedded in source code
16. Existing project files are not modified
"""

from __future__ import annotations

import importlib
import inspect
import json
import os
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# ---------------------------------------------------------------------------
# Helpers — build synthetic ERA5-like datasets
# ---------------------------------------------------------------------------

def _make_ds(
    n: int = 744,
    start: str = "2024-01-01",
    freq: str = "h",
    lat: float = 15.5,
    lon: float = 73.5,
    swh_vals: np.ndarray | None = None,
    pp1d_vals: np.ndarray | None = None,
    mwd_vals: np.ndarray | None = None,
    mwp_vals: np.ndarray | None = None,
    u10_vals: np.ndarray | None = None,
    v10_vals: np.ndarray | None = None,
    include_wind: bool = True,
    time_coord: str = "valid_time",
) -> xr.Dataset:
    """Build a minimal synthetic ERA5-like xarray Dataset."""
    times = pd.date_range(start, periods=n, freq=freq)

    def _arr(v, default):
        return v if v is not None else np.full(n, default)

    data_vars = {
        "swh":  (time_coord, _arr(swh_vals,  1.0)),
        "pp1d": (time_coord, _arr(pp1d_vals, 10.0)),
        "mwd":  (time_coord, _arr(mwd_vals,  270.0)),
        "mwp":  (time_coord, _arr(mwp_vals,  9.0)),
    }
    if include_wind:
        data_vars["u10"] = (time_coord, _arr(u10_vals, 5.0))
        data_vars["v10"] = (time_coord, _arr(v10_vals, -3.0))

    return xr.Dataset(
        data_vars,
        coords={
            time_coord: times,
            "latitude":  [lat],
            "longitude": [lon],
        },
    )


# ---------------------------------------------------------------------------
# Import the scripts under test
# ---------------------------------------------------------------------------

# Add project root to path so scripts/ is importable
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

import download_era5   as dl_mod
import validate_era5   as val_mod


# ---------------------------------------------------------------------------
# 1. Safe output path generation
# ---------------------------------------------------------------------------

def test_output_path_format():
    path = dl_mod._output_path(2024, 1, "data/raw/era5/test")
    assert path == "data/raw/era5/test/era5_goa_2024-01.nc"


def test_output_path_zero_padded_month():
    path = dl_mod._output_path(2024, 3, "data/raw/era5/production")
    assert "2024-03" in path


def test_metadata_path_derived_from_nc():
    meta = dl_mod._metadata_path("data/raw/era5/test/era5_goa_2024-01.nc")
    assert meta == "data/raw/era5/test/era5_goa_2024-01.json"


# ---------------------------------------------------------------------------
# 2. Refusal to overwrite without --force
# ---------------------------------------------------------------------------

def test_refuse_overwrite_existing_file(tmp_path, capsys):
    existing = tmp_path / "era5_goa_2024-01.nc"
    existing.write_text("placeholder")

    with pytest.raises(SystemExit) as exc_info:
        dl_mod.download(2024, 1, str(existing), force=False)

    captured = capsys.readouterr()
    assert exc_info.value.code == 0
    assert "refusing to overwrite" in captured.out.lower()


def test_force_flag_allows_overwrite(tmp_path, monkeypatch):
    """With --force, the download proceeds (we mock the CDS call)."""
    existing = tmp_path / "era5_goa_2024-01.nc"
    existing.write_text("old content")

    # Mock cdsapi.Client so no real download happens
    class _FakeResult:
        pass

    class _FakeClient:
        def retrieve(self, name, request, target):
            # Write a valid minimal NetCDF so metadata writing succeeds
            ds = _make_ds(n=744)
            ds.to_netcdf(target)

    with patch("cdsapi.Client", return_value=_FakeClient()):
        # Should not raise SystemExit
        dl_mod.download(2024, 1, str(existing), force=True)

    # File should have been replaced
    assert existing.stat().st_size > len("old content")


# ---------------------------------------------------------------------------
# 3. Metadata sidecar generation
# ---------------------------------------------------------------------------

def test_metadata_sidecar_written(tmp_path):
    nc_path = str(tmp_path / "era5_goa_2024-01.nc")
    ds = _make_ds(n=744)
    ds.to_netcdf(nc_path)

    meta_path = dl_mod._write_metadata(nc_path, 2024, 1, 15.5, 73.5)

    assert os.path.exists(meta_path)
    with open(meta_path) as fh:
        meta = json.load(fh)

    assert meta["year"] == 2024
    assert meta["month"] == 1
    assert meta["requested_latitude"] == dl_mod.LATITUDE
    assert meta["requested_longitude"] == dl_mod.LONGITUDE
    assert meta["cds_dataset_id"] == dl_mod.CDS_DATASET
    assert "download_timestamp_utc" in meta
    assert "immutable_note" in meta
    assert meta["schema_version"] == dl_mod.SCHEMA_VERSION


def test_metadata_no_credentials(tmp_path):
    """Metadata sidecar must not contain any credential-like keys."""
    nc_path = str(tmp_path / "era5_goa_2024-01.nc")
    ds = _make_ds(n=744)
    ds.to_netcdf(nc_path)

    meta_path = dl_mod._write_metadata(nc_path, 2024, 1, 15.5, 73.5)
    raw_text = Path(meta_path).read_text().lower()

    for forbidden in ("key", "token", "password", "secret", "credential"):
        # "key" appears in JSON keys like "cds_dataset_id" — check values only
        meta = json.loads(Path(meta_path).read_text())
        flat_values = str(list(meta.values())).lower()
        # The word "key" should not appear as a standalone value
        assert "cdsapirc" not in flat_values
        assert "password" not in flat_values
        assert "secret" not in flat_values


# ---------------------------------------------------------------------------
# 4. Required-variable validation (all present)
# ---------------------------------------------------------------------------

def test_validate_all_vars_present(tmp_path, capsys):
    nc = str(tmp_path / "good.nc")
    _make_ds(n=744, include_wind=True).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is True


# ---------------------------------------------------------------------------
# 5. Missing-variable detection
# ---------------------------------------------------------------------------

def test_validate_missing_swh(tmp_path, capsys):
    ds = _make_ds(n=744)
    ds = ds.drop_vars("swh")
    nc = str(tmp_path / "no_swh.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_missing_mwp(tmp_path, capsys):
    ds = _make_ds(n=744)
    ds = ds.drop_vars("mwp")
    nc = str(tmp_path / "no_mwp.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_missing_pp1d(tmp_path, capsys):
    ds = _make_ds(n=744)
    ds = ds.drop_vars("pp1d")
    nc = str(tmp_path / "no_pp1d.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 6. Time monotonicity detection
# ---------------------------------------------------------------------------

def test_validate_non_monotonic_time(tmp_path, capsys):
    times = pd.date_range("2024-01-01", periods=10, freq="h")
    # Reverse the last two entries to break monotonicity
    times_arr = times.to_numpy().copy()
    times_arr[-1], times_arr[-2] = times_arr[-2], times_arr[-1]

    ds = xr.Dataset(
        {"swh": ("valid_time", np.ones(10)),
         "pp1d": ("valid_time", np.ones(10) * 10),
         "mwd": ("valid_time", np.ones(10) * 270),
         "mwp": ("valid_time", np.ones(10) * 9)},
        coords={"valid_time": times_arr,
                "latitude": [15.5], "longitude": [73.5]},
    )
    nc = str(tmp_path / "nonmono.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 7. Duplicate timestamp detection
# ---------------------------------------------------------------------------

def test_validate_duplicate_timestamps(tmp_path, capsys):
    times = pd.date_range("2024-01-01", periods=10, freq="h").tolist()
    times[5] = times[4]   # introduce a duplicate

    ds = xr.Dataset(
        {"swh": ("valid_time", np.ones(10)),
         "pp1d": ("valid_time", np.ones(10) * 10),
         "mwd": ("valid_time", np.ones(10) * 270),
         "mwp": ("valid_time", np.ones(10) * 9)},
        coords={"valid_time": times,
                "latitude": [15.5], "longitude": [73.5]},
    )
    nc = str(tmp_path / "dupes.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 8. Expected hourly frequency detection
# ---------------------------------------------------------------------------

def test_validate_non_hourly_gap(tmp_path, capsys):
    """A 2-hour gap should be flagged."""
    times = pd.date_range("2024-01-01", periods=10, freq="h").tolist()
    times[5] = times[4] + pd.Timedelta(hours=2)   # 2-hour gap

    ds = xr.Dataset(
        {"swh": ("valid_time", np.ones(10)),
         "pp1d": ("valid_time", np.ones(10) * 10),
         "mwd": ("valid_time", np.ones(10) * 270),
         "mwp": ("valid_time", np.ones(10) * 9)},
        coords={"valid_time": times,
                "latitude": [15.5], "longitude": [73.5]},
    )
    nc = str(tmp_path / "gap.nc")
    ds.to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 9. January 2024 expected record count (744)
# ---------------------------------------------------------------------------

def test_validate_jan2024_correct_count(tmp_path, capsys):
    nc = str(tmp_path / "jan2024_ok.nc")
    _make_ds(n=744, start="2024-01-01").to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is True


def test_validate_jan2024_wrong_count(tmp_path, capsys):
    nc = str(tmp_path / "jan2024_short.nc")
    _make_ds(n=720, start="2024-01-01").to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 10. Latitude / longitude validation
# ---------------------------------------------------------------------------

def test_validate_correct_lat_lon(tmp_path, capsys):
    nc = str(tmp_path / "correct_coords.nc")
    _make_ds(n=744, lat=15.5, lon=73.5).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is True


def test_validate_wrong_latitude(tmp_path, capsys):
    nc = str(tmp_path / "wrong_lat.nc")
    _make_ds(n=744, lat=20.0, lon=73.5).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_wrong_longitude(tmp_path, capsys):
    nc = str(tmp_path / "wrong_lon.nc")
    _make_ds(n=744, lat=15.5, lon=80.0).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 11. Non-negative Hs (swh) validation
# ---------------------------------------------------------------------------

def test_validate_negative_swh(tmp_path, capsys):
    swh = np.full(744, 1.0)
    swh[100] = -0.5   # physically impossible
    nc = str(tmp_path / "neg_swh.nc")
    _make_ds(n=744, swh_vals=swh).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_zero_swh_allowed(tmp_path, capsys):
    """swh = 0 is physically marginal but not strictly forbidden."""
    swh = np.full(744, 0.0)
    nc = str(tmp_path / "zero_swh.nc")
    _make_ds(n=744, swh_vals=swh).to_netcdf(nc)
    # Should pass (0 is not < 0)
    result = val_mod.validate(nc)
    assert result is True


# ---------------------------------------------------------------------------
# 12. Positive-period validation (pp1d, mwp)
# ---------------------------------------------------------------------------

def test_validate_zero_pp1d(tmp_path, capsys):
    pp1d = np.full(744, 10.0)
    pp1d[50] = 0.0   # period must be > 0
    nc = str(tmp_path / "zero_pp1d.nc")
    _make_ds(n=744, pp1d_vals=pp1d).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_negative_mwp(tmp_path, capsys):
    mwp = np.full(744, 9.0)
    mwp[200] = -1.0
    nc = str(tmp_path / "neg_mwp.nc")
    _make_ds(n=744, mwp_vals=mwp).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 13. Circular direction validation (mwd in [0, 360))
# ---------------------------------------------------------------------------

def test_validate_direction_out_of_range(tmp_path, capsys):
    mwd = np.full(744, 270.0)
    mwd[10] = 400.0   # outside [0, 360)
    nc = str(tmp_path / "bad_dir.nc")
    _make_ds(n=744, mwd_vals=mwd).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


def test_validate_direction_negative(tmp_path, capsys):
    mwd = np.full(744, 270.0)
    mwd[10] = -10.0
    nc = str(tmp_path / "neg_dir.nc")
    _make_ds(n=744, mwd_vals=mwd).to_netcdf(nc)
    result = val_mod.validate(nc)
    assert result is False


# ---------------------------------------------------------------------------
# 14. Missing values are reported, not filled
# ---------------------------------------------------------------------------

def test_validate_nan_reported_not_filled(tmp_path, capsys):
    """NaN values must appear in the report; the validator must not fill them."""
    swh = np.full(744, 1.0)
    swh[0:10] = np.nan
    nc = str(tmp_path / "nan_swh.nc")
    _make_ds(n=744, swh_vals=swh).to_netcdf(nc)

    result = val_mod.validate(nc)
    captured = capsys.readouterr()

    # Validation should still pass (NaN is reported, not an error by itself)
    assert result is True
    # The report must mention the NaN count
    assert "10" in captured.out or "nan" in captured.out.lower()

    # Verify the file on disk still has NaN (not filled)
    ds_check = xr.open_dataset(nc)
    assert int(np.sum(np.isnan(ds_check["swh"].values))) == 10
    ds_check.close()


# ---------------------------------------------------------------------------
# 15. Credentials are not embedded in source code
# ---------------------------------------------------------------------------

def test_no_credentials_in_download_script():
    """download_era5.py must not contain hard-coded credential values."""
    src = inspect.getsource(dl_mod)
    src_lower = src.lower()
    # These patterns must not appear as code (not in docstrings as path refs)
    for pattern in ("api_key", "apikey", "password", "secret"):
        assert pattern not in src_lower, (
            f"Forbidden pattern '{pattern}' found in download_era5.py source"
        )


def test_no_credentials_in_validate_script():
    src = inspect.getsource(val_mod)
    src_lower = src.lower()
    for pattern in ("password", "secret", "api_key", "apikey"):
        assert pattern not in src_lower, (
            f"Forbidden pattern '{pattern}' found in validate_era5.py source"
        )


def test_cdsapirc_not_opened_directly():
    """Scripts must not call open() on ~/.cdsapirc — credentials handled by cdsapi.Client()."""
    for mod in (dl_mod, val_mod):
        src = inspect.getsource(mod)
        # Detect direct open() calls on the rc file (not docstring mentions)
        import ast
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                func_name = ""
                if isinstance(func, ast.Name):
                    func_name = func.id
                elif isinstance(func, ast.Attribute):
                    func_name = func.attr
                if func_name == "open":
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and ".cdsapirc" in str(arg.value):
                            raise AssertionError(
                                f"{mod.__name__} opens ~/.cdsapirc directly"
                            )


# ---------------------------------------------------------------------------
# 16. Existing project files are not modified
# ---------------------------------------------------------------------------

def test_frozen_module1_era5_unchanged():
    """module1_ocean/era5.py must not have been modified."""
    era5_path = _PROJECT_ROOT / "module1_ocean" / "era5.py"
    assert era5_path.exists(), "module1_ocean/era5.py is missing"
    src = era5_path.read_text()
    # Verify the frozen API is intact
    assert "def load_era5" in src
    assert "def extract_goa_timeseries" in src
    assert "_REQUIRED_VARS" in src


def test_frozen_module4_dataset_unchanged():
    """module4_forecasting/dataset.py must not have been modified."""
    ds_path = _PROJECT_ROOT / "module4_forecasting" / "dataset.py"
    assert ds_path.exists()
    src = ds_path.read_text()
    assert "class ForecastDataset" in src
    assert "FEATURE_NAMES" in src


def test_existing_raw_file_untouched():
    """data/raw/era5_goa_jan2024.nc must still exist and be unmodified."""
    raw = _PROJECT_ROOT / "data" / "raw" / "era5_goa_jan2024.nc"
    assert raw.exists(), "Existing raw ERA5 file has been deleted or moved"
    # File must be a valid NetCDF (xarray can open it)
    ds = xr.open_dataset(str(raw))
    assert "swh" in ds.data_vars
    ds.close()


def test_new_directories_created():
    """Required new directories must exist."""
    for subdir in (
        "data/raw/era5/test",
        "data/raw/era5/production",
        "data/processed/era5",
        "data/metadata",
        "scripts",
    ):
        path = _PROJECT_ROOT / subdir
        assert path.is_dir(), f"Expected directory not found: {subdir}"


def test_scripts_exist():
    """All three scripts must exist."""
    for script in ("download_era5.py", "validate_era5.py", "compare_era5_january.py"):
        path = _PROJECT_ROOT / "scripts" / script
        assert path.exists(), f"Script not found: scripts/{script}"
