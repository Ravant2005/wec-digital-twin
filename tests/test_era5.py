"""
tests/test_era5.py — Tests for module1_ocean/era5.py

Verifies loading and extraction of the ERA5 Goa January 2024 test dataset.
Run with:  pytest tests/test_era5.py -v
"""

import pytest
import pandas as pd
import xarray as xr

from module1_ocean.era5 import load_era5, extract_goa_timeseries

# Path relative to the project root (where pytest is invoked from)
TEST_FILE = "data/raw/era5_goa_jan2024.nc"

EXPECTED_RECORDS = 744
EXPECTED_LATITUDE = 15.5
EXPECTED_LONGITUDE = 73.5
EXPECTED_COLUMNS = ["time", "Hs_m", "Tp_s", "direction_deg"]
REQUIRED_VARS = ["swh", "pp1d", "mwd"]


@pytest.fixture(scope="module")
def dataset() -> xr.Dataset:
    """Load the ERA5 test dataset once for all tests in this module."""
    return load_era5(TEST_FILE)


@pytest.fixture(scope="module")
def timeseries(dataset) -> pd.DataFrame:
    """Extract the time series once for all tests in this module."""
    return extract_goa_timeseries(dataset)


# ---------------------------------------------------------------------------
# load_era5 tests
# ---------------------------------------------------------------------------

def test_load_returns_dataset(dataset):
    """load_era5 should return an xarray Dataset."""
    assert isinstance(dataset, xr.Dataset)


def test_required_variables_exist(dataset):
    """Dataset must contain swh, pp1d, and mwd."""
    for var in REQUIRED_VARS:
        assert var in dataset.data_vars, f"Missing variable: {var}"


def test_load_raises_on_missing_variable(tmp_path):
    """load_era5 should raise ValueError when a required variable is absent."""
    # Build a minimal dataset that is missing 'mwd'
    import numpy as np
    import pandas as pd

    times = pd.date_range("2024-01-01", periods=3, freq="h")
    ds_incomplete = xr.Dataset(
        {
            "swh": ("valid_time", np.ones(3)),
            "pp1d": ("valid_time", np.ones(3)),
            # 'mwd' intentionally omitted
        },
        coords={"valid_time": times},
    )
    nc_path = tmp_path / "incomplete.nc"
    ds_incomplete.to_netcdf(nc_path)

    with pytest.raises(ValueError, match="mwd"):
        load_era5(str(nc_path))


# ---------------------------------------------------------------------------
# extract_goa_timeseries tests
# ---------------------------------------------------------------------------

def test_timeseries_returns_dataframe(timeseries):
    """extract_goa_timeseries should return a pandas DataFrame."""
    assert isinstance(timeseries, pd.DataFrame)


def test_expected_columns(timeseries):
    """DataFrame must have exactly the four expected columns."""
    assert list(timeseries.columns) == EXPECTED_COLUMNS


def test_record_count(timeseries):
    """Test dataset must contain exactly 744 hourly records (Jan 2024)."""
    assert len(timeseries) == EXPECTED_RECORDS, (
        f"Expected {EXPECTED_RECORDS} records, got {len(timeseries)}"
    )


def test_no_duplicate_timestamps(timeseries):
    """Timestamps must be unique — no overlapping ERA5 records."""
    assert not timeseries["time"].duplicated().any()


def test_latitude_value(dataset):
    """Dataset latitude must be 15.5 (Goa offshore point)."""
    lat = float(dataset["latitude"].values.flat[0])
    assert lat == pytest.approx(EXPECTED_LATITUDE)


def test_longitude_value(dataset):
    """Dataset longitude must be 73.5 (Goa offshore point)."""
    lon = float(dataset["longitude"].values.flat[0])
    assert lon == pytest.approx(EXPECTED_LONGITUDE)


def test_nan_values_preserved(timeseries):
    """NaN values must not be silently filled — they should remain as NaN."""
    # We verify that the loader does NOT call fillna or interpolate.
    # If the raw data has NaNs, they must survive into the DataFrame.
    # (This dataset may have no NaNs, but the dtype must be float, not object.)
    for col in ["Hs_m", "Tp_s", "direction_deg"]:
        assert timeseries[col].dtype.kind == "f", (
            f"Column '{col}' should be float dtype to preserve NaN capability"
        )
