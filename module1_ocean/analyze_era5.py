"""
analyze_era5.py — Descriptive statistics for the ERA5 Goa wave time series.

Provides scalar statistics for Hs and Tp, circular statistics for wave
direction, and the Pearson correlation between Hs and Tp.

Scientific notes
----------------
- Hs and Tp are scalar quantities; ordinary arithmetic statistics apply.
- direction_deg is an angular (circular) quantity.  Arithmetic mean of angles
  is undefined across the 0°/360° wrap-around (e.g. the mean of 359° and 1°
  is 0°, not 180°).  This module uses circular statistics for direction.
- No values are interpolated, smoothed, or filled.  NaNs are excluded from
  calculations via numpy's nan-aware functions (nanmean, nanstd, etc.).
- Units are kept explicit throughout: Hs in metres, Tp in seconds,
  direction in degrees true.
"""

import numpy as np
import pandas as pd


def wave_height_stats(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for significant wave height (Hs).

    Hs is the mean height of the highest one-third of waves.  It is a scalar
    quantity, so ordinary arithmetic statistics are appropriate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column 'Hs_m' [metres].

    Returns
    -------
    dict
        Keys: count, mean_m, std_m, min_m, max_m, median_m.
        All values are in metres.  NaNs are excluded from all calculations.
    """
    v = df["Hs_m"].to_numpy(dtype=float)
    return {
        "count":    int(np.sum(~np.isnan(v))),
        "mean_m":   float(np.nanmean(v)),
        "std_m":    float(np.nanstd(v, ddof=1)),
        "min_m":    float(np.nanmin(v)),
        "max_m":    float(np.nanmax(v)),
        "median_m": float(np.nanmedian(v)),
    }


def peak_period_stats(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for peak wave period (Tp).

    Tp is the period of the most energetic waves in the spectrum.  It is a
    scalar quantity measured in seconds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column 'Tp_s' [seconds].

    Returns
    -------
    dict
        Keys: count, mean_s, std_s, min_s, max_s, median_s.
        All values are in seconds.  NaNs are excluded from all calculations.
    """
    v = df["Tp_s"].to_numpy(dtype=float)
    return {
        "count":    int(np.sum(~np.isnan(v))),
        "mean_s":   float(np.nanmean(v)),
        "std_s":    float(np.nanstd(v, ddof=1)),
        "min_s":    float(np.nanmin(v)),
        "max_s":    float(np.nanmax(v)),
        "median_s": float(np.nanmedian(v)),
    }


def wave_direction_stats(df: pd.DataFrame) -> dict:
    """
    Compute circular descriptive statistics for mean wave direction.

    Wave direction is an angular quantity in degrees true (clockwise from
    North), indicating the bearing FROM which waves are travelling.

    Arithmetic mean is NOT used for direction because it breaks at the
    0°/360° boundary.  Instead, circular statistics are applied:

      circular mean  = atan2(mean(sin θ), mean(cos θ))  [mod 360°]
      circular std   = sqrt(-2 · ln(R̄))  where R̄ is the mean resultant length
                       (Mardia & Jupp, 2000).  Units: degrees.

    The min, max, and median reported here are the raw arithmetic values from
    the ERA5 dataset and are provided for reference only — they should NOT be
    interpreted as directional statistics across the 0°/360° boundary.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the column 'direction_deg' [degrees true].

    Returns
    -------
    dict
        Keys:
          count              — number of non-NaN observations
          circular_mean_deg  — circular mean direction [degrees true]
          circular_std_deg   — circular standard deviation [degrees]
          mean_resultant_len — R̄ ∈ [0, 1]; 1 = perfectly unimodal,
                               0 = uniformly distributed
          min_deg            — minimum raw value [degrees, reference only]
          max_deg            — maximum raw value [degrees, reference only]
          median_deg         — median raw value [degrees, reference only]
    """
    v = df["direction_deg"].to_numpy(dtype=float)
    valid = v[~np.isnan(v)]

    rad = np.deg2rad(valid)
    sin_mean = np.mean(np.sin(rad))
    cos_mean = np.mean(np.cos(rad))

    # Mean resultant length: measure of directional concentration
    r_bar = float(np.sqrt(sin_mean**2 + cos_mean**2))

    # Circular mean: atan2 handles the 0/360 wrap correctly
    circ_mean = float(np.rad2deg(np.arctan2(sin_mean, cos_mean)) % 360)

    # Circular standard deviation (Mardia & Jupp, 2000, eq. 2.3.11)
    # Clamp r_bar to [0, 1] to guard against floating-point overshoot
    r_clamped = min(r_bar, 1.0)
    circ_std = float(np.rad2deg(np.sqrt(-2.0 * np.log(r_clamped)))) if r_clamped > 0 else float("inf")

    return {
        "count":               int(len(valid)),
        "circular_mean_deg":   round(circ_mean, 4),
        "circular_std_deg":    round(circ_std, 4),
        "mean_resultant_len":  round(r_bar, 6),
        "min_deg":             float(np.min(valid)),
        "max_deg":             float(np.max(valid)),
        "median_deg":          float(np.median(valid)),
    }


def hs_tp_correlation(df: pd.DataFrame) -> float:
    """
    Compute the Pearson correlation coefficient between Hs and Tp.

    Both Hs and Tp are scalar quantities, so Pearson's r is appropriate.
    Rows where either value is NaN are excluded pairwise.

    A positive correlation means larger waves tend to have longer periods.
    A negative or near-zero correlation indicates the two parameters vary
    somewhat independently at this location and time period.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'Hs_m' and 'Tp_s'.

    Returns
    -------
    float
        Pearson r ∈ [-1, 1].
    """
    pair = df[["Hs_m", "Tp_s"]].dropna()
    hs = pair["Hs_m"].to_numpy(dtype=float)
    tp = pair["Tp_s"].to_numpy(dtype=float)
    return float(np.corrcoef(hs, tp)[0, 1])


def summarise_all(df: pd.DataFrame) -> dict:
    """
    Return a combined summary of all wave statistics.

    Convenience wrapper that calls wave_height_stats(), peak_period_stats(),
    wave_direction_stats(), and hs_tp_correlation() and merges the results
    under labelled keys.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: Hs_m, Tp_s, direction_deg.

    Returns
    -------
    dict
        Keys: 'wave_height', 'peak_period', 'wave_direction',
              'hs_tp_pearson_r'.
    """
    return {
        "wave_height":    wave_height_stats(df),
        "peak_period":    peak_period_stats(df),
        "wave_direction": wave_direction_stats(df),
        "hs_tp_pearson_r": hs_tp_correlation(df),
    }
