"""
goa_seastate.py — ERA5-conditioned synthetic sea-state realizations.

Connects the ERA5 Goa time series to the existing physics engine
(spectrum → dispersion → directional spreading → spatial synthesis).

What this module does
---------------------
ERA5 provides hourly bulk sea-state parameters:
    Hs(t)         — significant wave height [m]
    Tp(t)         — peak wave period [s]
    direction(t)  — mean wave direction [degrees, toward convention]

These parameters describe the *statistics* of the sea surface over each
one-hour window.  They do NOT provide the actual second-by-second surface
elevation history.

This module generates a *synthetic realization* conditioned on those
statistics: a plausible surface elevation time series that is statistically
consistent with the ERA5 parameters for each hour, using the JONSWAP
spectrum, cos²-type directional spreading, and finite-depth dispersion.

Sea-state segment approach
--------------------------
Each ERA5 record defines one stationary sea-state segment.  A separate
synthetic realization is generated for each segment independently.  The
segments are NOT physically continuous at their boundaries — there is no
phase-matching or tapering between adjacent hours.

This is the correct approach for a first implementation because:
1. ERA5 hourly parameters already assume stationarity within each hour.
2. Physically continuous transitions would require a spectral evolution
   model that is beyond the current scope.
3. The boundary discontinuity is documented and explicit, not hidden.

For applications that require a continuous record, the segments can be
concatenated with a short cosine taper at each boundary.  That is left
for a future milestone.

Direction convention
--------------------
ERA5 mwd is in degrees, clockwise from North, direction waves propagate
TOWARD.  This module converts to radians exactly once at the API boundary:

    direction_rad = np.deg2rad(direction_deg)

The internal physics engine uses radians throughout.  No π offset is added.

Depth
-----
depth_m is an explicit required parameter.  This module does NOT invent
or hard-code a Goa bathymetry value.

gamma
-----
gamma=3.3 is the conventional JONSWAP default from the North Sea campaign.
It is NOT calibrated for the Goa coast.  Future work will estimate gamma
from the ERA5 Goa dataset.

Units
-----
    Hs            : m
    Tp            : s
    direction_deg : degrees (ERA5 input)
    direction_rad : radians (internal)
    depth_m       : m
    duration_s    : s
    dt            : s
    eta           : m
    t             : s (relative, starting at 0)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from module1_ocean.spectrum import jonswap, significant_wave_height_from_spectrum
from module1_ocean.waves import frequency_grid, significant_wave_height_from_timeseries
from module1_ocean.spatial_waves import synthesize_spatial_surface


# ---------------------------------------------------------------------------
# Default frequency grid parameters
# ---------------------------------------------------------------------------
_F_MIN       = 0.02   # Hz — low enough to capture long swell (Tp up to 50 s)
_F_MAX       = 0.5    # Hz — high enough for wind sea
_N_FREQ      = 128    # number of frequency components
_N_DIR       = 16     # number of directional bins


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SeaStateResult:
    """
    Container for a single synthetic sea-state realization.

    Attributes
    ----------
    t : np.ndarray
        Time array [s], starting at 0.
    eta : np.ndarray
        Synthetic surface elevation [m] at (x, y).
    Hs_requested : float
        ERA5 Hs used to construct the spectrum [m].
    Tp_requested : float
        ERA5 Tp used to construct the spectrum [s].
    direction_deg : float
        ERA5 mean wave direction [degrees].
    direction_rad : float
        Mean wave direction converted to radians (internal value used).
    Hs_spectral : float
        Hs reconstructed from the JONSWAP spectrum integral [m].
    Hs_timeseries : float
        Hs estimated from the synthesized η(t) via 4·std [m].
    depth_m : float
        Water depth used [m].
    gamma : float
        JONSWAP peak enhancement factor used.
    seed : Optional[int]
        Random seed used (None if unseeded).
    """
    t:              np.ndarray
    eta:            np.ndarray
    Hs_requested:   float
    Tp_requested:   float
    direction_deg:  float
    direction_rad:  float
    Hs_spectral:    float
    Hs_timeseries:  float
    depth_m:        float
    gamma:          float
    seed:           Optional[int]


@dataclass
class ReplayResult:
    """
    Container for a multi-segment ERA5 replay.

    Attributes
    ----------
    segments : list of SeaStateResult
        One entry per ERA5 record processed.
    era5_timestamps : list of pd.Timestamp
        ERA5 valid_time for each segment.
    era5_Hs : np.ndarray
        ERA5 Hs values [m] for each segment.
    era5_Tp : np.ndarray
        ERA5 Tp values [s] for each segment.
    era5_direction_deg : np.ndarray
        ERA5 direction values [degrees] for each segment.
    """
    segments:           list
    era5_timestamps:    list
    era5_Hs:            np.ndarray
    era5_Tp:            np.ndarray
    era5_direction_deg: np.ndarray


# ---------------------------------------------------------------------------
# 1. Single sea-state synthesis
# ---------------------------------------------------------------------------

def goa_seastate(
    Hs: float,
    Tp: float,
    direction_deg: float,
    depth_m: float,
    gamma: float = 3.3,
    duration_s: float = 3600.0,
    dt: float = 0.5,
    x: float = 0.0,
    y: float = 0.0,
    n_freq: int = _N_FREQ,
    n_directions: int = _N_DIR,
    seed: Optional[int] = None,
) -> SeaStateResult:
    """
    Generate a synthetic sea-surface realization for a single ERA5 sea state.

    This is a synthetic realization conditioned on ERA5 bulk parameters.
    It is NOT the actual historical surface elevation.

    Direction conversion
    --------------------
    direction_deg is converted to radians exactly once here:

        direction_rad = np.deg2rad(direction_deg)

    No π offset is applied.  ERA5 mwd is the direction waves propagate TOWARD.

    Parameters
    ----------
    Hs : float
        Significant wave height [m].  Must be > 0.
    Tp : float
        Peak wave period [s].  Must be > 0.
    direction_deg : float
        Mean wave direction [degrees, clockwise from North, toward convention].
        ERA5 mwd value.  Will be converted to radians internally.
    depth_m : float
        Water depth [m].  Must be > 0.  Not invented — must be supplied.
    gamma : float, optional
        JONSWAP peak enhancement factor.  Default 3.3 (conventional, not
        Goa-calibrated).
    duration_s : float, optional
        Simulation duration [s].  Default 3600 (one hour).
    dt : float, optional
        Time step [s].  Default 0.5.
    x : float, optional
        East coordinate of evaluation point [m].  Default 0.
    y : float, optional
        North coordinate of evaluation point [m].  Default 0.
    n_freq : int, optional
        Number of frequency components.  Default 128.
    n_directions : int, optional
        Number of directional bins.  Default 16.
    seed : int or None, optional
        Random seed for reproducibility.

    Returns
    -------
    SeaStateResult
        Contains t, eta, and all diagnostic quantities.

    Raises
    ------
    ValueError
        If Hs <= 0, Tp <= 0, depth_m <= 0, or gamma <= 0.
    """
    # --- Input validation ---
    if Hs <= 0.0:
        raise ValueError(f"Hs must be positive, got {Hs}")
    if Tp <= 0.0:
        raise ValueError(f"Tp must be positive, got {Tp}")
    if depth_m <= 0.0:
        raise ValueError(f"depth_m must be positive, got {depth_m}")
    if gamma <= 0.0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    # --- Direction conversion: degrees → radians, exactly once ---
    direction_rad = np.deg2rad(float(direction_deg))

    # --- Frequency grid ---
    # Ensure fp = 1/Tp is well-resolved: f_min < fp < f_max
    fp = 1.0 / Tp
    f_min = min(_F_MIN, fp * 0.5)
    f_max = max(_F_MAX, fp * 3.0)
    f = frequency_grid(f_min, f_max, n_freq)

    # --- JONSWAP spectrum ---
    S_f = jonswap(f, Hs, Tp, gamma=gamma)
    Hs_spectral = significant_wave_height_from_spectrum(f, S_f)

    # --- Time array (relative, starting at 0) ---
    t = np.arange(0.0, duration_s, dt)

    # --- Synthesize surface elevation ---
    eta = synthesize_spatial_surface(
        x=x,
        y=y,
        t=t,
        f=f,
        S_f=S_f,
        mean_direction=direction_rad,
        depth_m=depth_m,
        n_directions=n_directions,
        seed=seed,
    )

    Hs_ts = significant_wave_height_from_timeseries(eta)

    return SeaStateResult(
        t=t,
        eta=eta,
        Hs_requested=float(Hs),
        Tp_requested=float(Tp),
        direction_deg=float(direction_deg),
        direction_rad=float(direction_rad),
        Hs_spectral=Hs_spectral,
        Hs_timeseries=Hs_ts,
        depth_m=float(depth_m),
        gamma=float(gamma),
        seed=seed,
    )


# ---------------------------------------------------------------------------
# 2. ERA5 time-series replay
# ---------------------------------------------------------------------------

def generate_goa_replay(
    df: pd.DataFrame,
    depth_m: float,
    gamma: float = 3.3,
    duration_s: float = 3600.0,
    dt: float = 0.5,
    x: float = 0.0,
    y: float = 0.0,
    n_freq: int = _N_FREQ,
    n_directions: int = _N_DIR,
    base_seed: Optional[int] = None,
    skip_nan: bool = True,
) -> ReplayResult:
    """
    Generate synthetic sea-state realizations for each row of an ERA5 DataFrame.

    Each ERA5 record defines one stationary sea-state segment.  A separate
    synthetic realization is generated for each segment independently.

    Segment boundary behavior
    -------------------------
    Segments are NOT physically continuous at their boundaries.  Each segment
    starts with its own independent random phases.  This is consistent with
    the ERA5 assumption that each hourly record represents a stationary sea
    state.  Physically continuous transitions are not implemented at this stage.

    Reproducibility
    ---------------
    If base_seed is provided, segment i uses seed = base_seed + i, ensuring
    each segment is independently reproducible while the full replay is
    deterministic.

    Parameters
    ----------
    df : pd.DataFrame
        ERA5 time series from extract_goa_timeseries().
        Required columns: time, Hs_m, Tp_s, direction_deg.
    depth_m : float
        Water depth [m].  Must be > 0.  Not invented by this function.
    gamma : float, optional
        JONSWAP peak enhancement factor.  Default 3.3 (not Goa-calibrated).
    duration_s : float, optional
        Duration of each segment [s].  Default 3600 (one hour).
    dt : float, optional
        Time step [s].  Default 0.5.
    x : float, optional
        East coordinate [m].  Default 0.
    y : float, optional
        North coordinate [m].  Default 0.
    n_freq : int, optional
        Number of frequency components.  Default 128.
    n_directions : int, optional
        Number of directional bins.  Default 16.
    base_seed : int or None, optional
        Base random seed.  Segment i uses seed = base_seed + i.
    skip_nan : bool, optional
        If True (default), skip rows where Hs, Tp, or direction is NaN.
        If False, raise ValueError on NaN.

    Returns
    -------
    ReplayResult
        Contains one SeaStateResult per processed segment, plus the
        original ERA5 timestamps and parameters.

    Raises
    ------
    ValueError
        If required columns are missing, depth_m <= 0, or (skip_nan=False
        and NaN values are present).
    """
    required = ["time", "Hs_m", "Tp_s", "direction_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")

    if depth_m <= 0.0:
        raise ValueError(f"depth_m must be positive, got {depth_m}")

    segments = []
    timestamps = []
    hs_list, tp_list, dir_list = [], [], []

    for i, row in enumerate(df.itertuples(index=False)):
        Hs  = float(row.Hs_m)
        Tp  = float(row.Tp_s)
        dir_deg = float(row.direction_deg)
        ts  = row.time

        # Handle NaN
        if np.isnan(Hs) or np.isnan(Tp) or np.isnan(dir_deg):
            if skip_nan:
                continue
            raise ValueError(
                f"NaN in ERA5 row {i} (time={ts}): "
                f"Hs={Hs}, Tp={Tp}, direction={dir_deg}. "
                f"Set skip_nan=True to skip NaN rows."
            )

        seg_seed = (base_seed + i) if base_seed is not None else None

        result = goa_seastate(
            Hs=Hs,
            Tp=Tp,
            direction_deg=dir_deg,
            depth_m=depth_m,
            gamma=gamma,
            duration_s=duration_s,
            dt=dt,
            x=x,
            y=y,
            n_freq=n_freq,
            n_directions=n_directions,
            seed=seg_seed,
        )

        segments.append(result)
        timestamps.append(ts)
        hs_list.append(Hs)
        tp_list.append(Tp)
        dir_list.append(dir_deg)

    return ReplayResult(
        segments=segments,
        era5_timestamps=timestamps,
        era5_Hs=np.array(hs_list),
        era5_Tp=np.array(tp_list),
        era5_direction_deg=np.array(dir_list),
    )
