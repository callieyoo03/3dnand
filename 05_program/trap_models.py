# ============================================================
# Static charge-trap electrostatic models
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
#
# Purpose:
#   Add prescribed trapped-electron charge to the HfO2
#   ChargeTrap region.
#
# Physical definition:
#
#   Ntrap > 0
#       = number density of trapped electrons
#
#   rho_trap
#       = -q * Ntrap
#
# DEVSIM Poisson residual convention:
#
#   PotentialNodeCharge = -rho
#
# Therefore, for negatively charged trapped electrons:
#
#   PotentialNodeCharge = +q * Ntrap
#
# Important:
#   This is a static fixed-charge model.
#   Capture, emission, tunneling, program time, and retention
#   are not yet calculated.
# ============================================================

from devsim import (
    equation,
    get_equation_list,
    get_node_model_list,
    node_model,
    set_parameter,
)

import trap_parameters as tp


# ============================================================
# Device and region names
# ============================================================

device = "MoS2_GAA"

CHARGE_TRAP_REGION = tp.CHARGE_TRAP_REGION


# ============================================================
# Model names
# ============================================================

TRAPPED_ELECTRON_PARAMETER = (
    "TrappedElectronDensityParameter"
)

TRAPPED_ELECTRON_MODEL = (
    "TrappedElectronDensity"
)

TRAPPED_CHARGE_MODEL = (
    "TrappedChargeDensity"
)

TRAP_POISSON_NODE_MODEL = (
    "ChargeTrapPotentialNodeCharge"
)


# ============================================================
# Initialize the trapped-electron-density parameter
# ============================================================

def initialize_trap_parameter():

    set_parameter(
        device=device,
        region=CHARGE_TRAP_REGION,
        name=TRAPPED_ELECTRON_PARAMETER,
        value=0.0,
    )

    print(
        "Initial trapped-electron density set to "
        "0.0 cm^-3."
    )


# ============================================================
# Create static trapped-charge node models
# ============================================================

def create_trap_charge_models():

    region = CHARGE_TRAP_REGION

    # --------------------------------------------------------
    # Number density of occupied electron traps
    #
    # This is positive as a particle-number density.
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name=TRAPPED_ELECTRON_MODEL,
        equation=TRAPPED_ELECTRON_PARAMETER,
    )

    # --------------------------------------------------------
    # Physical space-charge density
    #
    # Electrons carry negative charge:
    #
    # rho_trap = -q * Ntrap
    #
    # Unit:
    #   C/cm^3
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name=TRAPPED_CHARGE_MODEL,
        equation=(
            f"-{tp.q}*"
            f"{TRAPPED_ELECTRON_MODEL}"
        ),
    )

    # --------------------------------------------------------
    # Node-charge term used in DEVSIM Poisson residual
    #
    # DEVSIM model convention used in this project:
    #
    # PotentialNodeCharge = -rho
    #
    # Since rho_trap = -q*Ntrap:
    #
    # PotentialNodeCharge = +q*Ntrap
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name=TRAP_POISSON_NODE_MODEL,
        equation=(
            f"{tp.q}*"
            f"{TRAPPED_ELECTRON_MODEL}"
        ),
    )

    print(
        "Static trapped-electron and trapped-charge "
        "models created in HfO2."
    )


# ============================================================
# Connect trapped charge to the ChargeTrap Poisson equation
# ============================================================

def connect_trap_charge_to_poisson():

    region = CHARGE_TRAP_REGION

    node_models = get_node_model_list(
        device=device,
        region=region,
    )

    required_models = (
        "Potential",
        "PotentialEdgeFlux",
        TRAP_POISSON_NODE_MODEL,
    )

    for required_model in required_models:

        if required_model not in node_models:

            # PotentialEdgeFlux is an edge model, so it is
            # checked separately through the equation call.
            if required_model == "PotentialEdgeFlux":
                continue

            raise RuntimeError(
                f'Required model "{required_model}" '
                f'is missing in region "{region}".'
            )

    # --------------------------------------------------------
    # Replace the dielectric-only PotentialEquation in
    # ChargeTrap with a Poisson equation containing fixed
    # trapped charge.
    #
    # Calling equation() again with the same equation name
    # updates the region equation definition.
    # --------------------------------------------------------

    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        node_model=TRAP_POISSON_NODE_MODEL,
        edge_model="PotentialEdgeFlux",
        variable_update="log_damp",
    )

    print(
        "HfO2 trapped charge connected to "
        "PotentialEquation."
    )


# ============================================================
# Set one static trapped-electron state
# ============================================================

def set_trapped_electron_density(
    density_cm3,
):

    density_cm3 = float(
        density_cm3
    )

    if density_cm3 < 0.0:

        raise ValueError(
            "Trapped-electron density must not "
            "be negative."
        )

    set_parameter(
        device=device,
        region=CHARGE_TRAP_REGION,
        name=TRAPPED_ELECTRON_PARAMETER,
        value=density_cm3,
    )

    sheet_density_cm2 = (
        tp.volume_to_sheet_density(
            density_cm3
        )
    )

    charge_density_c_cm3 = (
        tp.electron_density_to_charge_density(
            density_cm3
        )
    )

    print(
        "Trap state updated:"
    )

    print(
        f"  Volume density = "
        f"{density_cm3:.6e} cm^-3"
    )

    print(
        f"  Sheet density  = "
        f"{sheet_density_cm2:.6e} cm^-2"
    )

    print(
        f"  Charge density = "
        f"{charge_density_c_cm3:.6e} C/cm^3"
    )


# ============================================================
# Read trap-state information
# ============================================================

def get_trap_state_information(
    density_cm3,
):

    density_cm3 = float(
        density_cm3
    )

    sheet_density_cm2 = (
        tp.volume_to_sheet_density(
            density_cm3
        )
    )

    charge_density_c_cm3 = (
        tp.electron_density_to_charge_density(
            density_cm3
        )
    )

    return {
        "trap_density_cm3": density_cm3,
        "trap_sheet_density_cm2": (
            sheet_density_cm2
        ),
        "trap_charge_density_C_cm3": (
            charge_density_c_cm3
        ),
    }


# ============================================================
# Verification
# ============================================================

def verify_trap_models():

    region = CHARGE_TRAP_REGION

    node_models = get_node_model_list(
        device=device,
        region=region,
    )

    required_node_models = (
        TRAPPED_ELECTRON_MODEL,
        TRAPPED_CHARGE_MODEL,
        TRAP_POISSON_NODE_MODEL,
    )

    for model_name in required_node_models:

        if model_name not in node_models:

            raise RuntimeError(
                f'Trap model "{model_name}" '
                f'was not created in region "{region}".'
            )

    equations = get_equation_list(
        device=device,
        region=region,
    )

    if "PotentialEquation" not in equations:

        raise RuntimeError(
            "PotentialEquation is missing in "
            "the ChargeTrap region."
        )

    print(
        "Static HfO2 charge-trap model "
        "verification completed."
    )


# ============================================================
# Complete setup function
# ============================================================

def create_static_trap_framework():

    initialize_trap_parameter()

    create_trap_charge_models()

    connect_trap_charge_to_poisson()

    verify_trap_models()

    print(
        "Static HfO2 trapped-charge framework created."
    )