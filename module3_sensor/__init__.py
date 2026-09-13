"""
module3_sensor — Synthetic wave sensor / camera observation model.

Models the measurement uncertainty of a hypothetical camera-based wave
estimation system.  Does NOT implement computer vision or image processing.

Exports
-------
SensorParameters
    Configuration dataclass for the sensor.
SensorObservation
    Observation output dataclass.  Call .as_model_input() to obtain only
    the observable fields (no truth leakage).
WaveSensor
    Main sensor class.  Call .observe(time, Hs_true, Tp_true, dir_true).
circular_direction_error
    Utility: signed circular error in degrees.
"""

from module3_sensor.sensor import (
    SensorParameters,
    SensorObservation,
    WaveSensor,
    circular_direction_error,
)

__all__ = [
    "SensorParameters",
    "SensorObservation",
    "WaveSensor",
    "circular_direction_error",
]
