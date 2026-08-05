# ============================================================
# Material parameters
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA device
#
# Layer stack:
#
# CoreOxide
#     ↓
# MoS2 channel
#     ↓
# Al2O3 tunnel dielectric
#     ↓
# HfO2 charge-trap layer
#     ↓
# Al2O3 blocking dielectric
#     ↓
# Metal gate
#
# Notes:
#   - Values are baseline simulation parameters.
#   - They are not yet process-calibrated.
#   - Trap capture/emission is not included here.
# ============================================================

import sys
from math import exp, pi, sqrt
from pathlib import Path


PROJECT_ROOT_DIRECTORY = Path(__file__).resolve().parent.parent
PROJECT_ROOT_TEXT = str(PROJECT_ROOT_DIRECTORY)

if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)

import compact_handoff_parameters as compact_parameters

if Path(compact_parameters.__file__).resolve() != (
    PROJECT_ROOT_DIRECTORY / "compact_handoff_parameters.py"
).resolve():
    raise ImportError("compact_handoff_parameters resolved outside this repository.")


# ============================================================
# Fundamental constants
# ============================================================

q = 1.602176634e-19
# C

k_B = 1.380649e-23
# J/K

temperature = 300.0
# K

Vt = k_B * temperature / q
# V

eps0 = 8.8541878128e-14
# F/cm


# ============================================================
# Layer thicknesses
# ============================================================

mos2_thickness = (
    compact_parameters.MOS2_THICKNESS_NM
    * compact_parameters.NM_TO_CM
)
# cm
# 2 nm

tunnel_oxide_thickness = (
    compact_parameters.TUNNEL_OXIDE_THICKNESS_NM
    * compact_parameters.NM_TO_CM
)
# cm
# 3 nm Al2O3

charge_trap_thickness = (
    compact_parameters.CHARGE_TRAP_THICKNESS_NM
    * compact_parameters.NM_TO_CM
)
# cm
# 5 nm HfO2

blocking_oxide_thickness = (
    compact_parameters.BLOCKING_OXIDE_THICKNESS_NM
    * compact_parameters.NM_TO_CM
)
# cm
# 16 nm Al2O3


# ============================================================
# Relative dielectric constants
# ============================================================

relative_permittivity_core_oxide = (
    compact_parameters.CORE_OXIDE_RELATIVE_PERMITTIVITY
)
# Initial SiO2-like core approximation

relative_permittivity_mos2 = compact_parameters.MOS2_RELATIVE_PERMITTIVITY
# Initial out-of-plane MoS2 approximation

relative_permittivity_al2o3 = compact_parameters.AL2O3_RELATIVE_PERMITTIVITY
# Literature nominal Al2O3 value

relative_permittivity_hfo2 = compact_parameters.HFO2_RELATIVE_PERMITTIVITY
# Literature nominal HfO2 value

relative_permittivity_air = compact_parameters.AIR_RELATIVE_PERMITTIVITY


# ============================================================
# Absolute dielectric permittivities
# ============================================================

eps_core_oxide = (
    relative_permittivity_core_oxide
    * eps0
)

eps_mos2 = (
    relative_permittivity_mos2
    * eps0
)

eps_tunnel_oxide = (
    relative_permittivity_al2o3
    * eps0
)
# Al2O3

eps_charge_trap = (
    relative_permittivity_hfo2
    * eps0
)
# HfO2

eps_blocking_oxide = (
    relative_permittivity_al2o3
    * eps0
)
# Al2O3

eps_air = (
    relative_permittivity_air
    * eps0
)


# ============================================================
# MoS2 band parameters
# ============================================================

Eg_MoS2 = 1.67
# eV
# Effective transport bandgap baseline

ElectronAffinity_MoS2 = 4.21
# eV
# Baseline electron affinity


# ============================================================
# Density-of-states constants
# ============================================================

hbar = 1.054571817e-34
# J s

m0 = 9.1093837139e-31
# kg


# ============================================================
# MoS2 DOS effective masses
# ============================================================

dos_effective_mass_ratio_n_MoS2 = 1.0
# Electron DOS effective mass ratio

dos_effective_mass_ratio_p_MoS2 = 0.8
# Hole DOS effective mass ratio

effective_mass_n_MoS2 = (
    dos_effective_mass_ratio_n_MoS2
    * m0
)

effective_mass_p_MoS2 = (
    dos_effective_mass_ratio_p_MoS2
    * m0
)


# ============================================================
# MoS2 degeneracy factors
# ============================================================

spin_degeneracy_MoS2 = 2.0

valley_degeneracy_n_MoS2 = 6.0
# Q-valley baseline assumption

valley_degeneracy_p_MoS2 = 1.0
# Gamma-valley baseline assumption


# ============================================================
# Two-dimensional effective density of states
# ============================================================

Nc_2D_MoS2_m2 = (
    spin_degeneracy_MoS2
    * valley_degeneracy_n_MoS2
    * effective_mass_n_MoS2
    * k_B
    * temperature
    / (2.0 * pi * hbar**2)
)
# m^-2

Nv_2D_MoS2_m2 = (
    spin_degeneracy_MoS2
    * valley_degeneracy_p_MoS2
    * effective_mass_p_MoS2
    * k_B
    * temperature
    / (2.0 * pi * hbar**2)
)
# m^-2


# ============================================================
# Convert m^-2 to cm^-2
# ============================================================

Nc_2D_MoS2_cm2 = (
    Nc_2D_MoS2_m2
    * 1.0e-4
)

Nv_2D_MoS2_cm2 = (
    Nv_2D_MoS2_m2
    * 1.0e-4
)


# ============================================================
# Effective volume DOS used by DEVSIM
# ============================================================

Nc_MoS2 = (
    Nc_2D_MoS2_cm2
    / mos2_thickness
)
# cm^-3

Nv_MoS2 = (
    Nv_2D_MoS2_cm2
    / mos2_thickness
)
# cm^-3


# ============================================================
# Intrinsic carrier concentration
# ============================================================

BoltzmannConstant_eV = 8.617333262e-5
# eV/K

ThermalEnergy_eV = (
    BoltzmannConstant_eV
    * temperature
)
# eV

ni_MoS2 = (
    sqrt(
        Nc_MoS2
        * Nv_MoS2
    )
    * exp(
        -Eg_MoS2
        / (
            2.0
            * ThermalEnergy_eV
        )
    )
)
# cm^-3


# ============================================================
# Carrier mobilities
# ============================================================

mu_n_MoS2 = 50.0
# cm^2/Vs
# Baseline constant electron mobility

mu_p_MoS2 = 10.0
# cm^2/Vs
# Retained for future ambipolar simulation


# ============================================================
# Doping
# ============================================================

channel_doping = 1.0e15
# cm^-3
# Uniform n-type baseline

source_drain_doping = 1.0e19
# cm^-3
# Not yet spatially implemented


# ============================================================
# Gate parameters
# ============================================================

gate_work_function = 4.5
# eV
# Placeholder metal work function

flat_band_voltage = 0.0
# V
# Current framework assumes zero work-function offset


# ============================================================
# HfO2 charge-trap baseline parameters
# ============================================================

trap_density_max = 2.0e18
# cm^-3
# Upper test value for static trapped-electron states

trap_density_states = (
    0.0,
    2.0e17,
    5.0e17,
    1.0e18,
    2.0e18,
)
# cm^-3
# Static occupied-electron densities used in memory-window sweep


# ============================================================
# Equivalent sheet-density helper
# ============================================================

def trap_volume_to_sheet_density(
    trap_density_cm3,
):

    return (
        trap_density_cm3
        * charge_trap_thickness
    )
# cm^-2
