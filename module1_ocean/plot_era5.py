"""
plot_era5.py — Visualization functions for the ERA5 Goa wave time series.

All plotting functions accept a DataFrame produced by extract_goa_timeseries()
and render the three primary wave parameters:

  Hs_m          — Significant wave height [m]
  Tp_s          — Peak wave period [s]
  direction_deg — Mean wave direction [degrees true, FROM]

The underlying DataFrame is never modified by any function in this module.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from module1_ocean.era5 import load_era5, extract_goa_timeseries


def load_goa_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load an ERA5 NetCDF file and return the Goa wave time series as a DataFrame.

    Delegates entirely to the existing load_era5() and extract_goa_timeseries()
    functions — no ERA5 loading logic is duplicated here.

    Parameters
    ----------
    filepath : str
        Path to the ERA5 NetCDF file.

    Returns
    -------
    pd.DataFrame
        Columns: time, Hs_m, Tp_s, direction_deg.
    """
    return extract_goa_timeseries(load_era5(filepath))


def plot_wave_height(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Figure:
    """
    Plot significant wave height (Hs) versus time.

    Hs is the mean height of the highest one-third of waves [metres].
    Higher values indicate more energetic sea states.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'time' and 'Hs_m'.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax)
    ax.plot(df["time"], df["Hs_m"], color="#1f77b4", linewidth=0.9)
    ax.set_ylabel("Hs [m]")
    ax.set_title("Significant Wave Height — Goa (ERA5, Jan 2024)")
    _format_time_axis(ax)
    fig.tight_layout()
    return fig


def plot_peak_period(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Figure:
    """
    Plot peak wave period (Tp) versus time.

    Tp is the period of the most energetic waves in the spectrum [seconds].
    It governs the resonance frequency relevant to the point-absorber buoy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'time' and 'Tp_s'.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax)
    ax.plot(df["time"], df["Tp_s"], color="#ff7f0e", linewidth=0.9)
    ax.set_ylabel("Tp [s]")
    ax.set_title("Peak Wave Period — Goa (ERA5, Jan 2024)")
    _format_time_axis(ax)
    fig.tight_layout()
    return fig


def plot_wave_direction(df: pd.DataFrame, ax: plt.Axes = None) -> plt.Figure:
    """
    Plot mean wave direction versus time.

    Direction is reported in degrees true (clockwise from North), indicating
    the compass bearing FROM which the dominant waves are travelling.
    ERA5 values are preserved without conversion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'time' and 'direction_deg'.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  A new figure is created if not provided.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax)
    ax.plot(df["time"], df["direction_deg"], color="#2ca02c", linewidth=0.9)
    ax.set_ylabel("Direction [° true, FROM]")
    ax.set_title("Mean Wave Direction — Goa (ERA5, Jan 2024)")
    ax.set_ylim(0, 360)
    ax.set_yticks([0, 90, 180, 270, 360])
    ax.set_yticklabels(["N (0°)", "E (90°)", "S (180°)", "W (270°)", "N (360°)"])
    _format_time_axis(ax)
    fig.tight_layout()
    return fig


def plot_all(df: pd.DataFrame) -> plt.Figure:
    """
    Produce all three wave-parameter plots stacked in a single figure.

    The DataFrame is not modified.  Each subplot shares the same time axis
    for easy visual comparison of Hs, Tp, and direction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: time, Hs_m, Tp_s, direction_deg.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing three vertically stacked subplots.
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axes[0].plot(df["time"], df["Hs_m"], color="#1f77b4", linewidth=0.9)
    axes[0].set_ylabel("Hs [m]")
    axes[0].set_title("ERA5 Wave Parameters — Goa (15.5°N, 73.5°E) — January 2024")

    axes[1].plot(df["time"], df["Tp_s"], color="#ff7f0e", linewidth=0.9)
    axes[1].set_ylabel("Tp [s]")

    axes[2].plot(df["time"], df["direction_deg"], color="#2ca02c", linewidth=0.9)
    axes[2].set_ylabel("Direction\n[° true, FROM]")
    axes[2].set_ylim(0, 360)
    axes[2].set_yticks([0, 90, 180, 270, 360])

    _format_time_axis(axes[2])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_axes(ax):
    """Return (fig, ax), creating a new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.get_figure()
    return fig, ax


def _format_time_axis(ax: plt.Axes) -> None:
    """Apply a readable month/day date format to the x-axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.set_xlabel("Date (UTC)")
    ax.grid(True, alpha=0.3)
