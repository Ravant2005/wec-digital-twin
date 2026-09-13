"""
module2_wec — Physics-based heaving point-absorber WEC dynamics model.

Module structure (milestones):
    geometry.py      — buoy geometry and physical parameters  [THIS MILESTONE]
    hydrostatics.py  — hydrostatic equilibrium quantities     [THIS MILESTONE]
    hydrodynamics.py — Capytaine BEM coefficients             [FUTURE]
    cummins.py       — Cummins equation time-domain solver    [FUTURE]
    pto.py           — PTO force models                       [FUTURE]
    control.py       — Latching and RL controllers            [FUTURE]

Units (SI throughout):
    length   : m
    mass     : kg
    force    : N
    pressure : Pa  (N/m²)
    density  : kg/m³
    time     : s
    frequency: Hz  (rad/s where noted)
"""

from module2_wec.geometry import CylinderGeometry
from module2_wec.hydrostatics import Hydrostatics
from module2_wec.capytaine_model import CapytaineModel
from module2_wec.hydrodynamic_coefficients import (
    HydrodynamicCoefficients,
    compute_hydrodynamic_coefficients,
)
from module2_wec.radiation import (
    RadiationKernel,
    compute_radiation_kernel,
)
from module2_wec.excitation import (
    ExcitationSignal,
    compute_excitation_force,
    interpolate_excitation,
)
from module2_wec.cummins import (
    CumminsParameters,
    CumminsResult,
    solve_cummins,
    frequency_domain_rao,
    frequency_domain_rao_pto,
)
from module2_wec.pto import (
    PTOParameters,
    PTOResult,
    compute_pto_force,
    compute_pto_power,
    compute_pto_result,
)

__all__ = [
    "CylinderGeometry",
    "Hydrostatics",
    "CapytaineModel",
    "HydrodynamicCoefficients",
    "compute_hydrodynamic_coefficients",
    "RadiationKernel",
    "compute_radiation_kernel",
    "ExcitationSignal",
    "compute_excitation_force",
    "interpolate_excitation",
    "CumminsParameters",
    "CumminsResult",
    "solve_cummins",
    "frequency_domain_rao",
    "frequency_domain_rao_pto",
    "PTOParameters",
    "PTOResult",
    "compute_pto_force",
    "compute_pto_power",
    "compute_pto_result",
]
