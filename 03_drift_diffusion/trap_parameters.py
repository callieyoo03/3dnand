# ============================================================
# Static charge-trap parameters
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
#
# Purpose:
#   Define several static trapped-electron states in HfO2.
#
# Important:
#   This stage does not calculate electron injection,
#   capture, emission, program time, or retention.
#
#   Instead, the trapped-electron density is prescribed
#   externally and its electrostatic effect on I_D-V_G
#   and threshold voltage is calculated.
# ============================================================


# ============================================================
# Fundamental constant
# ============================================================

q = 1.602176634e-19
# C


# ============================================================
# Charge-trap region
# ============================================================

CHARGE_TRAP_REGION = "ChargeTrap"


# ============================================================
# HfO2 charge-trap thickness
# ============================================================

charge_trap_thickness = 5.0e-7
# cm
# 5 nm


# ============================================================
# Static trapped-electron-density states
#
# Positive values below represent the number density of
# trapped electrons.
#
# The corresponding physical charge density is negative:
#
# rho_trap = -q * N_trap
# ============================================================

TRAPPED_ELECTRON_DENSITY_STATES = (
    0.0,
    2.0e17,
    5.0e17,
    1.0e18,
    2.0e18,
)
# cm^-3


# ============================================================
# Human-readable state labels
# ============================================================

TRAP_STATE_LABELS = (
    "State_0_Empty",
    "State_1",
    "State_2",
    "State_3",
    "State_4",
)


# ============================================================
# Equivalent trapped-electron sheet density
#
# N_sheet = N_volume * trap thickness
# ============================================================

def volume_to_sheet_density(
    volume_density_cm3,
):

    return (
        float(volume_density_cm3)
        * charge_trap_thickness
    )
# cm^-2


# ============================================================
# Charge-density conversion
#
# Physical trapped charge:
#
# rho = -q * N_trap
# ============================================================

def electron_density_to_charge_density(
    electron_density_cm3,
):

    return (
        -q
        * float(electron_density_cm3)
    )
# C/cm^3


# ============================================================
# Threshold-voltage extraction criterion
#
# Vth is defined here using a constant drain-current method:
#
# |ID| = THRESHOLD_CURRENT
#
# The exact value can later be changed, but every trap state
# must use the same criterion.
# ============================================================

THRESHOLD_CURRENT = 1.0e-10
# A


# ============================================================
# I_D-V_G measurement conditions
# ============================================================

SOURCE_VOLTAGE = 0.0
# V

DRAIN_VOLTAGE = 0.05
# V

GATE_START_VOLTAGE = -1.0
# V

GATE_STOP_VOLTAGE = 3.0
# V

GATE_OUTPUT_STEP = 0.1
# V


# ============================================================
# Adaptive-bias-ramp parameters
# ============================================================

DRAIN_INITIAL_STEP = 0.01
# V

DRAIN_MINIMUM_STEP = 5.0e-4
# V

GATE_INITIAL_STEP = 0.05
# V

GATE_MINIMUM_STEP = 2.5e-3
# V


# ============================================================
# Solver parameters
# ============================================================

SOLVER_ABSOLUTE_ERROR = 1.0e10

SOLVER_RELATIVE_ERROR = 1.0e-9

SOLVER_MAXIMUM_ITERATIONS = 150


# ============================================================
# Output files
# ============================================================

MEMORY_WINDOW_CSV = "memory_window_id_vg.csv"

THRESHOLD_SUMMARY_CSV = "memory_window_vth.csv"

FINAL_VTK_NAME = "mos2_gaa_charge_trap_final"


# ============================================================
# Parameter validation
# ============================================================

def validate_trap_parameters():

    if len(
        TRAPPED_ELECTRON_DENSITY_STATES
    ) != len(
        TRAP_STATE_LABELS
    ):

        raise ValueError(
            "The number of trap-density states must match "
            "the number of trap-state labels."
        )

    if charge_trap_thickness <= 0.0:

        raise ValueError(
            "Charge-trap thickness must be positive."
        )

    if THRESHOLD_CURRENT <= 0.0:

        raise ValueError(
            "Threshold-current criterion must be positive."
        )

    previous_density = None

    for density in TRAPPED_ELECTRON_DENSITY_STATES:

        if density < 0.0:

            raise ValueError(
                "Trapped-electron density must not be negative."
            )

        if (
            previous_density is not None
            and density < previous_density
        ):

            raise ValueError(
                "Trap-density states must be arranged "
                "in ascending order."
            )

        previous_density = density