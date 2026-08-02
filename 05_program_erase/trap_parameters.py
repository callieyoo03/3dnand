# ============================================================
# Static charge-trap parameters
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
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
# State labels
# ============================================================

TRAP_STATE_LABELS = (
    "State_0_Empty",
    "State_1",
    "State_2",
    "State_3",
    "State_4",
)


# ============================================================
# Conversion functions
# ============================================================

def volume_to_sheet_density(
    volume_density_cm3,
):

    return (
        float(volume_density_cm3)
        * charge_trap_thickness
    )


def electron_density_to_charge_density(
    electron_density_cm3,
):

    return (
        -q
        * float(electron_density_cm3)
    )


# ============================================================
# Threshold-voltage criterion
# ============================================================

THRESHOLD_CURRENT = 1.0e-10
# A


# ============================================================
# I_D-V_G bias settings
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
# Adaptive voltage-ramp settings
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
# Solver settings
# ============================================================

SOLVER_ABSOLUTE_ERROR = 1.0e10

SOLVER_RELATIVE_ERROR = 1.0e-9

SOLVER_MAXIMUM_ITERATIONS = 150


# ============================================================
# Output files
# ============================================================

MEMORY_WINDOW_CSV = (
    "memory_window_id_vg.csv"
)

THRESHOLD_SUMMARY_CSV = (
    "memory_window_vth.csv"
)

FINAL_VTK_NAME = (
    "mos2_gaa_charge_trap_final"
)


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
            "The number of trap-density states must "
            "match the number of state labels."
        )

    if charge_trap_thickness <= 0.0:

        raise ValueError(
            "Charge-trap thickness must be positive."
        )

    if THRESHOLD_CURRENT <= 0.0:

        raise ValueError(
            "Threshold current must be positive."
        )

    if DRAIN_VOLTAGE < 0.0:

        raise ValueError(
            "Drain voltage must not be negative."
        )

    if GATE_OUTPUT_STEP <= 0.0:

        raise ValueError(
            "Gate output step must be positive."
        )

    if (
        GATE_STOP_VOLTAGE
        <= GATE_START_VOLTAGE
    ):

        raise ValueError(
            "Gate stop voltage must be greater "
            "than gate start voltage."
        )

    if DRAIN_INITIAL_STEP <= 0.0:

        raise ValueError(
            "Drain initial step must be positive."
        )

    if DRAIN_MINIMUM_STEP <= 0.0:

        raise ValueError(
            "Drain minimum step must be positive."
        )

    if GATE_INITIAL_STEP <= 0.0:

        raise ValueError(
            "Gate initial step must be positive."
        )

    if GATE_MINIMUM_STEP <= 0.0:

        raise ValueError(
            "Gate minimum step must be positive."
        )

    if SOLVER_MAXIMUM_ITERATIONS <= 0:

        raise ValueError(
            "Maximum solver iterations must be positive."
        )

    previous_density = None

    for density in (
        TRAPPED_ELECTRON_DENSITY_STATES
    ):

        if density < 0.0:

            raise ValueError(
                "Trapped-electron density must not "
                "be negative."
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