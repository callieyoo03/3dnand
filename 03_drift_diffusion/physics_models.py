# ============================================================
# Physics models
# MoS2 cylindrical GAA FET
#
# Electron-only drift-diffusion baseline
#
# Included:
#   1. Material parameters
#   2. Uniform n-type channel doping
#   3. Potential and electron solution variables
#   4. Equilibrium electron/hole reference concentrations
#   5. Poisson equation
#   6. Scharfetter-Gummel electron current
#   7. Electron continuity equation
#   8. Electrostatic interface conditions
#   9. Source / drain / gate boundary conditions
#
# Notes:
#   - Electrons are solved dynamically.
#   - Holes are fixed to EquilibriumHoles.
#   - ChargeTrap currently behaves only as a dielectric.
# ============================================================

from devsim import (
    contact_equation,
    contact_node_model,
    edge_from_node_model,
    edge_model,
    equation,
    interface_equation,
    interface_model,
    node_model,
    node_solution,
    set_parameter,
)

from devsim.python_packages.simple_dd import (
    CreateBernoulli,
    CreateElectronCurrent,
)

import material_parameters as mp


# ============================================================
# Device names
# ============================================================

device = "MoS2_GAA"

SEMICONDUCTOR_REGION = "MoS2"

REGIONS = (
    "CoreOxide",
    "MoS2",
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
)

INTERFACES = (
    "CoreOxide_MoS2",
    "MoS2_TunnelOxide",
    "TunnelOxide_ChargeTrap",
    "ChargeTrap_BlockingOxide",
)


# ============================================================
# Material parameters
# ============================================================

def set_material_parameters():

    # --------------------------------------------------------
    # Core oxide
    # --------------------------------------------------------

    set_parameter(
        device=device,
        region="CoreOxide",
        name="Permittivity",
        value=mp.eps_core_oxide,
    )

    # --------------------------------------------------------
    # MoS2 semiconductor
    # --------------------------------------------------------

    set_parameter(
        device=device,
        region="MoS2",
        name="Permittivity",
        value=mp.eps_mos2,
    )

    set_parameter(
        device=device,
        region="MoS2",
        name="ElectronCharge",
        value=mp.q,
    )

    # Alias used by DEVSIM simple_dd.py
    set_parameter(
        device=device,
        region="MoS2",
        name="q",
        value=mp.q,
    )

    set_parameter(
        device=device,
        region="MoS2",
        name="ThermalVoltage",
        value=mp.Vt,
    )

    # Alias used by DEVSIM simple_dd.py
    set_parameter(
        device=device,
        region="MoS2",
        name="V_t",
        value=mp.Vt,
    )

    set_parameter(
        device=device,
        region="MoS2",
        name="IntrinsicDensity",
        value=mp.ni_MoS2,
    )

    set_parameter(
        device=device,
        region="MoS2",
        name="ElectronMobility",
        value=mp.mu_n_MoS2,
    )

    # --------------------------------------------------------
    # Tunnel oxide
    # --------------------------------------------------------

    set_parameter(
        device=device,
        region="TunnelOxide",
        name="Permittivity",
        value=mp.eps_tunnel_oxide,
    )

    # --------------------------------------------------------
    # Charge-trap dielectric
    # --------------------------------------------------------

    set_parameter(
        device=device,
        region="ChargeTrap",
        name="Permittivity",
        value=mp.eps_charge_trap,
    )

    # --------------------------------------------------------
    # Blocking oxide
    # --------------------------------------------------------

    set_parameter(
        device=device,
        region="BlockingOxide",
        name="Permittivity",
        value=mp.eps_blocking_oxide,
    )

    print("Material parameters assigned.")


# ============================================================
# Doping models
# ============================================================

def create_doping():

    region = SEMICONDUCTOR_REGION

    node_model(
        device=device,
        region=region,
        name="Donors",
        equation=str(mp.channel_doping),
    )

    node_model(
        device=device,
        region=region,
        name="Acceptors",
        equation="0",
    )

    node_model(
        device=device,
        region=region,
        name="NetDoping",
        equation="Donors-Acceptors",
    )

    print("Uniform n-type MoS2 channel doping created.")


# ============================================================
# Solution variables
# ============================================================

def create_solution_variables():

    # --------------------------------------------------------
    # Potential is solved in every physical region
    # --------------------------------------------------------

    for region in REGIONS:

        node_solution(
            device=device,
            region=region,
            name="Potential",
        )

        edge_from_node_model(
            device=device,
            region=region,
            node_model="Potential",
        )

        print(
            f'Potential solution created in region "{region}".'
        )

    # --------------------------------------------------------
    # Only electrons are solved dynamically in MoS2
    # --------------------------------------------------------

    node_solution(
        device=device,
        region=SEMICONDUCTOR_REGION,
        name="Electrons",
    )

    edge_from_node_model(
        device=device,
        region=SEMICONDUCTOR_REGION,
        node_model="Electrons",
    )

    print("Electron solution variable created in MoS2.")


# ============================================================
# Equilibrium carrier reference models
# ============================================================

def create_equilibrium_carrier_models():

    region = SEMICONDUCTOR_REGION

    # --------------------------------------------------------
    # Charge-neutral equilibrium electron concentration
    #
    # n0 = 0.5 * (
    #     NetDoping
    #     + sqrt(NetDoping^2 + 4*ni^2)
    # )
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name="EquilibriumElectrons",
        equation=(
            "0.5*("
            "NetDoping"
            "+(NetDoping^2+4*IntrinsicDensity^2)^(0.5)"
            ")"
        ),
    )

    # --------------------------------------------------------
    # Minority-hole concentration
    #
    # This remains a fixed node model.
    # It is not solved with a continuity equation.
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name="EquilibriumHoles",
        equation=(
            "IntrinsicDensity^2"
            "/EquilibriumElectrons"
        ),
    )

    print("Equilibrium electron and hole models created.")
    
      # --------------------------------------------------------
    # Fixed hole model required by DEVSIM simple_dd.py
    # --------------------------------------------------------

    node_model(
        device=device,
        region=region,
        name="Holes",
        equation="EquilibriumHoles",
    )

    edge_from_node_model(
        device=device,
        region=region,
        node_model="Holes",
    )

    print(
        "Equilibrium electron and fixed-hole "
        "models created."
    )


# ============================================================
# Poisson equation
# ============================================================

def create_poisson_model():

    for region in REGIONS:

        # ----------------------------------------------------
        # Electric field
        #
        # E = -grad(Potential)
        # ----------------------------------------------------

        edge_model(
            device=device,
            region=region,
            name="ElectricField",
            equation=(
                "(Potential@n0-Potential@n1)"
                "*EdgeInverseLength"
            ),
        )

        edge_model(
            device=device,
            region=region,
            name="ElectricField:Potential@n0",
            equation="EdgeInverseLength",
        )

        edge_model(
            device=device,
            region=region,
            name="ElectricField:Potential@n1",
            equation="-EdgeInverseLength",
        )

        # ----------------------------------------------------
        # Electric-displacement flux
        # ----------------------------------------------------

        edge_model(
            device=device,
            region=region,
            name="PotentialEdgeFlux",
            equation="Permittivity*ElectricField",
        )

        edge_model(
            device=device,
            region=region,
            name="PotentialEdgeFlux:Potential@n0",
            equation="Permittivity*EdgeInverseLength",
        )

        edge_model(
            device=device,
            region=region,
            name="PotentialEdgeFlux:Potential@n1",
            equation="-Permittivity*EdgeInverseLength",
        )

        # ----------------------------------------------------
        # Semiconductor region
        #
        # rho = q * (
        #     EquilibriumHoles
        #     - Electrons
        #     + NetDoping
        # )
        #
        # Holes are held at their equilibrium concentration.
        # ----------------------------------------------------

        if region == SEMICONDUCTOR_REGION:

            node_model(
                device=device,
                region=region,
                name="PotentialNodeCharge",
                equation=(
                    "-ElectronCharge*("
                    "EquilibriumHoles"
                    "-Electrons"
                    "+NetDoping"
                    ")"
                ),
            )

            node_model(
                device=device,
                region=region,
                name="PotentialNodeCharge:Electrons",
                equation="ElectronCharge",
            )

            equation(
                device=device,
                region=region,
                name="PotentialEquation",
                variable_name="Potential",
                node_model="PotentialNodeCharge",
                edge_model="PotentialEdgeFlux",
                variable_update="log_damp",
            )

        # ----------------------------------------------------
        # Dielectric regions
        # ----------------------------------------------------

        else:

            equation(
                device=device,
                region=region,
                name="PotentialEquation",
                variable_name="Potential",
                edge_model="PotentialEdgeFlux",
                variable_update="log_damp",
            )

    print("Poisson models created.")


# ============================================================
# Scharfetter-Gummel electron-current model
# ============================================================

def create_electron_current_model():

    region = SEMICONDUCTOR_REGION

    CreateBernoulli(
        device=device,
        region=region,
    )

    CreateElectronCurrent(
        device=device,
        region=region,
        mu_n="ElectronMobility",
    )

    print("Electron current model created.")


# ============================================================
# Electron continuity equation
# ============================================================

def create_continuity_equation():

    equation(
        device=device,
        region=SEMICONDUCTOR_REGION,
        name="ElectronContinuityEquation",
        variable_name="Electrons",
        edge_model="ElectronCurrent",
        variable_update="positive",
    )

    print("Electron continuity equation created.")


# ============================================================
# Electrostatic interface conditions
# ============================================================

def create_interface_models():

    for interface in INTERFACES:

        model_name = (
            f"{interface}_potential_continuity"
        )

        interface_model(
            device=device,
            interface=interface,
            name=model_name,
            equation="Potential@r0-Potential@r1",
        )

        interface_model(
            device=device,
            interface=interface,
            name=f"{model_name}:Potential@r0",
            equation="1",
        )

        interface_model(
            device=device,
            interface=interface,
            name=f"{model_name}:Potential@r1",
            equation="-1",
        )

        interface_equation(
            device=device,
            interface=interface,
            name="PotentialEquation",
            interface_model=model_name,
            type="continuous",
        )

        print(
            f'Potential continuity created at "{interface}".'
        )


# ============================================================
# Contact boundary conditions
# ============================================================

def create_contact_models():

    # --------------------------------------------------------
    # Applied terminal voltages
    # --------------------------------------------------------

    set_parameter(
        device=device,
        name="source_bias",
        value=0.0,
    )

    set_parameter(
        device=device,
        name="drain_bias",
        value=0.0,
    )

    set_parameter(
        device=device,
        name="gate_bias",
        value=0.0,
    )

    # --------------------------------------------------------
    # Ideal ohmic source and drain
    # --------------------------------------------------------

    for contact in ("source", "drain"):

        potential_model = (
            f"{contact}_potential_bc"
        )

        contact_node_model(
            device=device,
            contact=contact,
            name=potential_model,
            equation=(
                f"Potential-{contact}_bias"
                "-ThermalVoltage*"
                "log(EquilibriumElectrons/IntrinsicDensity)"
            ),
        )

        contact_node_model(
            device=device,
            contact=contact,
            name=f"{potential_model}:Potential",
            equation="1",
        )

        contact_equation(
            device=device,
            contact=contact,
            name="PotentialEquation",
            node_model=potential_model,
            edge_charge_model="PotentialEdgeFlux",
        )

        electron_model = (
            f"{contact}_electron_bc"
        )

        contact_node_model(
            device=device,
            contact=contact,
            name=electron_model,
            equation=(
                "Electrons-EquilibriumElectrons"
            ),
        )

        contact_node_model(
            device=device,
            contact=contact,
            name=f"{electron_model}:Electrons",
            equation="1",
        )

        contact_equation(
            device=device,
            contact=contact,
            name="ElectronContinuityEquation",
            node_model=electron_model,
            edge_current_model="ElectronCurrent",
        )

        print(
            f'Electron-only ohmic boundary condition '
            f'created at "{contact}".'
        )

    # --------------------------------------------------------
    # Ideal gate
    #
    # Work-function difference is temporarily zero.
    # --------------------------------------------------------

    contact_node_model(
        device=device,
        contact="gate",
        name="gate_potential_bc",
        equation="Potential-gate_bias",
    )

    contact_node_model(
        device=device,
        contact="gate",
        name="gate_potential_bc:Potential",
        equation="1",
    )

    contact_equation(
        device=device,
        contact="gate",
        name="PotentialEquation",
        node_model="gate_potential_bc",
        edge_charge_model="PotentialEdgeFlux",
    )

    print("Gate electrostatic boundary condition created.")