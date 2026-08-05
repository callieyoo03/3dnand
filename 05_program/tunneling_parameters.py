# ============================================================
# Tunneling parameters
# MoS2 / Al2O3 / HfO2 charge-trap memory
#
# Unit convention
# ---------------
# Electric field supplied to the tunneling model:
#     V/cm
#
# Fowler-Nordheim internal calculation:
#     SI units
#
# Returned current density:
#     A/cm^2
#
# Important
# ---------
# The barrier height and tunneling effective mass are initial
# model parameters. They must later be calibrated against
# literature or experimental data for the actual interface.
# ============================================================

import math


# ============================================================
# Fundamental constants
# ============================================================

q = 1.602176634e-19
# Elementary charge
# C

hbar = 1.054571817e-34
# Reduced Planck constant
# J s

m0 = 9.1093837139e-31
# Free-electron mass
# kg


# ============================================================
# Unit conversion constants
# ============================================================

EV_TO_J = q

V_CM_TO_V_M = 1.0e2

A_M2_TO_A_CM2 = 1.0e-4


# ============================================================
# Tunnel-oxide parameters
# ============================================================

BARRIER_HEIGHT_EV = 3.56
# eV
#
# Literature nominal reference value.
# This is a CBO-based MoS2 / Al2O3 barrier candidate,
# not a final I-V calibrated effective FN barrier.

TUNNEL_EFFECTIVE_MASS_RATIO = 0.28
# m* / m0
#
# Literature nominal Al2O3 tunneling-mass reference.
# It was not co-fitted with the MoS2 / Al2O3 CBO above.

TUNNEL_OXIDE_THICKNESS_NM = 4.0
# nm


# ============================================================
# Derived SI parameters
# ============================================================

BARRIER_HEIGHT_J = (
    BARRIER_HEIGHT_EV
    * EV_TO_J
)

TUNNEL_EFFECTIVE_MASS_KG = (
    TUNNEL_EFFECTIVE_MASS_RATIO
    * m0
)

TUNNEL_OXIDE_THICKNESS_M = (
    TUNNEL_OXIDE_THICKNESS_NM
    * 1.0e-9
)


# ============================================================
# Fowler-Nordheim coefficients
# ============================================================
#
# SI form:
#
# J = FN_A_SI * E^2 * exp(-FN_B_SI / E)
#
# E:
#     V/m
#
# J:
#     A/m^2
#
# FN_A_SI:
#     A/V^2
#
# FN_B_SI:
#     V/m
# ============================================================

FN_A_SI = (
    q**3
    / (
        16.0
        * math.pi**2
        * hbar
        * BARRIER_HEIGHT_J
    )
)

FN_B_SI = (
    4.0
    * math.sqrt(
        2.0
        * TUNNEL_EFFECTIVE_MASS_KG
    )
    * BARRIER_HEIGHT_J**1.5
    / (
        3.0
        * q
        * hbar
    )
)


# ============================================================
# Numerical protection
# ============================================================

MINIMUM_FIELD_V_CM = 1.0
# Below this field, return zero current density.

MAXIMUM_EXPONENT_MAGNITUDE = 700.0
# Avoid floating-point underflow/overflow in exp().


# ============================================================
# Program/erase settings
# ============================================================

PROGRAM_TIME_S = 1.0e-6

ERASE_TIME_S = 1.0e-6


# ============================================================
# Backward-compatible aliases
# ============================================================
#
# Existing project code may still refer to the old lowercase
# or generic names. These aliases prevent import breakage.
# ============================================================

barrier_height = BARRIER_HEIGHT_EV

effective_mass = TUNNEL_EFFECTIVE_MASS_RATIO

FN_A = FN_A_SI

FN_B = FN_B_SI

program_time = PROGRAM_TIME_S

erase_time = ERASE_TIME_S


# ============================================================
# Parameter reporting
# ============================================================

def get_tunneling_parameter_summary():
    """
    Return the current tunneling parameters.

    Returns
    -------
    dict
        Parameter names and values.
    """

    return {
        "barrier_height_eV": (
            BARRIER_HEIGHT_EV
        ),
        "barrier_height_J": (
            BARRIER_HEIGHT_J
        ),
        "effective_mass_ratio": (
            TUNNEL_EFFECTIVE_MASS_RATIO
        ),
        "effective_mass_kg": (
            TUNNEL_EFFECTIVE_MASS_KG
        ),
        "tunnel_oxide_thickness_nm": (
            TUNNEL_OXIDE_THICKNESS_NM
        ),
        "FN_A_SI": (
            FN_A_SI
        ),
        "FN_B_SI_V_m": (
            FN_B_SI
        ),
        "program_time_s": (
            PROGRAM_TIME_S
        ),
        "erase_time_s": (
            ERASE_TIME_S
        ),
    }


def print_tunneling_parameter_summary():
    """
    Print the active Fowler-Nordheim parameters.
    """

    summary = (
        get_tunneling_parameter_summary()
    )

    print()
    print("=" * 70)
    print("TUNNELING PARAMETER SUMMARY")
    print("=" * 70)

    print(
        f'Barrier height          = '
        f'{summary["barrier_height_eV"]:.6f} eV'
    )

    print(
        f'Effective mass ratio    = '
        f'{summary["effective_mass_ratio"]:.6f}'
    )

    print(
        f'Tunnel oxide thickness  = '
        f'{summary["tunnel_oxide_thickness_nm"]:.6f} nm'
    )

    print(
        f'FN A coefficient        = '
        f'{summary["FN_A_SI"]:.6e} A/V^2'
    )

    print(
        f'FN B coefficient        = '
        f'{summary["FN_B_SI_V_m"]:.6e} V/m'
    )