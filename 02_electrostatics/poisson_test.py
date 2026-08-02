# ============================================================
# MoS2 Cylindrical GAA Charge-Trap Memory
# Stage 2 : Electrostatic Poisson Test
#
# Goal:
#   - Recreate the 2D cylindrical structure
#   - Solve only Potential
#   - Apply Source / Drain / Gate bias
#   - Export Potential and ElectricField to VTK
#
# Coordinate:
#   x = radial direction r
#   y = channel direction z
#
# Unit:
#   length = cm
# ============================================================

from devsim import (
    create_2d_mesh,
    add_2d_mesh_line,
    add_2d_region,
    add_2d_interface,
    add_2d_contact,
    finalize_mesh,
    create_device,
    set_parameter,
    node_solution,
    edge_from_node_model,
    edge_model,
    equation,
    contact_node_model,
    contact_equation,
    interface_model,
    interface_equation,
    solve,
    get_contact_list,
    get_interface_list,
    get_region_list,
    get_node_model_values,
    cylindrical_edge_couple,
    cylindrical_node_volume,
    cylindrical_surface_area,
    write_devices,
)


# ============================================================
# 1. Device / mesh name
# ============================================================

device = "MoS2_GAA"
mesh = "gaa_mesh"


# ============================================================
# 2. Geometry
# ============================================================

r_axis = 0.0

r_core = 1.0e-6
# 10 nm

r_mos2 = 1.2e-6
# 12 nm

r_tox = 1.6e-6
# 16 nm

r_trap = 2.1e-6
# 21 nm

r_block = 2.9e-6
# 29 nm

r_gate_outer = 3.1e-6
# 31 nm

r_air_outer = 3.3e-6
# 33 nm

z_source = 0.0
z_drain = 1.0e-5
# 100 nm


# ============================================================
# 3. Mesh
# ============================================================

create_2d_mesh(mesh=mesh)


# radial direction
add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_axis,
    ps=1.0e-7,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_core,
    ps=5.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_mos2,
    ps=2.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_tox,
    ps=5.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_trap,
    ps=5.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_block,
    ps=5.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_gate_outer,
    ps=5.0e-8,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="x",
    pos=r_air_outer,
    ps=1.0e-7,
)


# channel direction
add_2d_mesh_line(
    mesh=mesh,
    dir="y",
    pos=z_source,
    ps=2.0e-7,
)

add_2d_mesh_line(
    mesh=mesh,
    dir="y",
    pos=z_drain,
    ps=2.0e-7,
)


# ============================================================
# 4. Regions
# ============================================================

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="CoreOxide",
    xl=r_axis,
    xh=r_core,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="MoS2",
    region="MoS2",
    xl=r_core,
    xh=r_mos2,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="TunnelOxide",
    xl=r_mos2,
    xh=r_tox,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="ChargeTrap",
    region="ChargeTrap",
    xl=r_tox,
    xh=r_trap,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="BlockingOxide",
    xl=r_trap,
    xh=r_block,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="Metal",
    region="GateMetal",
    xl=r_block,
    xh=r_gate_outer,
    yl=z_source,
    yh=z_drain,
)

add_2d_region(
    mesh=mesh,
    material="Air",
    region="Air",
    xl=r_gate_outer,
    xh=r_air_outer,
    yl=z_source,
    yh=z_drain,
)


# ============================================================
# 5. Interfaces
# ============================================================

interfaces = (
    (
        "CoreOxide_MoS2",
        "CoreOxide",
        "MoS2",
        r_core,
    ),
    (
        "MoS2_TunnelOxide",
        "MoS2",
        "TunnelOxide",
        r_mos2,
    ),
    (
        "TunnelOxide_ChargeTrap",
        "TunnelOxide",
        "ChargeTrap",
        r_tox,
    ),
    (
        "ChargeTrap_BlockingOxide",
        "ChargeTrap",
        "BlockingOxide",
        r_trap,
    ),
    (
        "BlockingOxide_GateMetal",
        "BlockingOxide",
        "GateMetal",
        r_block,
    ),
)


for name, region0, region1, xpos in interfaces:

    add_2d_interface(
        mesh=mesh,
        name=name,
        region0=region0,
        region1=region1,
        xl=xpos,
        xh=xpos,
        yl=z_source,
        yh=z_drain,
        bloat=1.0e-10,
    )


# ============================================================
# 6. Contacts
# ============================================================

add_2d_contact(
    mesh=mesh,
    name="source",
    region="MoS2",
    material="Metal",
    xl=r_core,
    xh=r_mos2,
    yl=z_source,
    yh=z_source,
    bloat=1.0e-10,
)

add_2d_contact(
    mesh=mesh,
    name="drain",
    region="MoS2",
    material="Metal",
    xl=r_core,
    xh=r_mos2,
    yl=z_drain,
    yh=z_drain,
    bloat=1.0e-10,
)

add_2d_contact(
    mesh=mesh,
    name="gate",
    region="GateMetal",
    material="Metal",
    xl=r_gate_outer,
    xh=r_gate_outer,
    yl=z_source,
    yh=z_drain,
    bloat=1.0e-10,
)


# ============================================================
# 7. Finalize mesh / create device
# ============================================================

finalize_mesh(mesh=mesh)

create_device(
    mesh=mesh,
    device=device,
)


# ============================================================
# 8. Cylindrical coordinate setting
# ============================================================

set_parameter(
    device=device,
    name="raxis_variable",
    value="x",
)

set_parameter(
    device=device,
    name="raxis_zero",
    value=0.0,
)


# ============================================================
# 9. Cylindrical geometry models
# ============================================================

all_regions = (
    "CoreOxide",
    "MoS2",
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
    "GateMetal",
    "Air",
)


for region in all_regions:

    cylindrical_edge_couple(
        device=device,
        region=region,
    )

    cylindrical_node_volume(
        device=device,
        region=region,
    )

    cylindrical_surface_area(
        device=device,
        region=region,
    )


# Tell DEVSIM to use cylindrical integration
set_parameter(
    device=device,
    name="node_volume_model",
    value="CylindricalNodeVolume",
)

set_parameter(
    device=device,
    name="edge_couple_model",
    value="CylindricalEdgeCouple",
)

set_parameter(
    device=device,
    name="edge_node0_volume_model",
    value="CylindricalEdgeNodeVolume@n0",
)

set_parameter(
    device=device,
    name="edge_node1_volume_model",
    value="CylindricalEdgeNodeVolume@n1",
)


# ============================================================
# 10. Material permittivities
# ============================================================
#
# epsilon0 = 8.854e-14 F/cm
#
# For this first electrostatic test, we use simple representative
# relative permittivities.
#
# These are NOT yet the final research-calibrated material values.
# ============================================================

eps0 = 8.854e-14


# Core oxide: SiO2
set_parameter(
    device=device,
    region="CoreOxide",
    name="Permittivity",
    value=3.9 * eps0,
)


# MoS2
# Temporary isotropic value for first Poisson test
set_parameter(
    device=device,
    region="MoS2",
    name="Permittivity",
    value=6.2 * eps0,
)


# Tunnel oxide: SiO2
set_parameter(
    device=device,
    region="TunnelOxide",
    name="Permittivity",
    value=3.9 * eps0,
)


# Charge trap layer
# Temporary Si3N4-like dielectric constant
set_parameter(
    device=device,
    region="ChargeTrap",
    name="Permittivity",
    value=7.5 * eps0,
)


# Blocking oxide
set_parameter(
    device=device,
    region="BlockingOxide",
    name="Permittivity",
    value=3.9 * eps0,
)


# Gate metal
#
# We solve a simple electrostatic Potential equation in this
# geometry-only stage. A numerical permittivity is assigned so
# Potential can exist in the region.
set_parameter(
    device=device,
    region="GateMetal",
    name="Permittivity",
    value=1.0 * eps0,
)


# Air
set_parameter(
    device=device,
    region="Air",
    name="Permittivity",
    value=1.0 * eps0,
)


# ============================================================
# 11. Create Potential / ElectricField / DField
# ============================================================


def create_poisson_region(region):

    # --------------------------------------------------------
    # Potential unknown
    # --------------------------------------------------------

    node_solution(
        device=device,
        region=region,
        name="Potential",
    )


    # --------------------------------------------------------
    # Create Potential@n0 and Potential@n1
    # --------------------------------------------------------

    edge_from_node_model(
        device=device,
        region=region,
        node_model="Potential",
    )


    # --------------------------------------------------------
    # Electric field
    #
    # E = -grad(V)
    #
    # On DEVSIM edge:
    #
    # (Potential@n0 - Potential@n1) / L
    # --------------------------------------------------------

    edge_model(
        device=device,
        region=region,
        name="ElectricField",
        equation=(
            "(Potential@n0 - Potential@n1)"
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


    # --------------------------------------------------------
    # Electric displacement
    #
    # D = epsilon * E
    # --------------------------------------------------------

    edge_model(
        device=device,
        region=region,
        name="DField",
        equation="Permittivity*ElectricField",
    )


    edge_model(
        device=device,
        region=region,
        name="DField:Potential@n0",
        equation=(
            "diff(Permittivity*ElectricField,"
            "Potential@n0)"
        ),
    )


    edge_model(
        device=device,
        region=region,
        name="DField:Potential@n1",
        equation="-DField:Potential@n0",
    )


    # --------------------------------------------------------
    # Poisson / Laplace equation
    #
    # At this stage:
    #
    # rho = 0
    #
    # therefore:
    #
    # div(epsilon grad V) = 0
    #
    # --------------------------------------------------------

    equation(
        device=device,
        region=region,
        name="PotentialEquation",
        variable_name="Potential",
        edge_model="DField",
        variable_update="default",
    )


for region in all_regions:
    create_poisson_region(region)


# ============================================================
# 12. Potential continuity at interfaces
# ============================================================
#
# Potential@r0 = Potential@r1
#
# DEVSIM requires the interface model and its derivatives.
# ============================================================


for interface_name, _, _, _ in interfaces:

    model_name = interface_name + "_PotentialContinuity"

    interface_model(
        device=device,
        interface=interface_name,
        name=model_name,
        equation="Potential@r0-Potential@r1",
    )


    interface_model(
        device=device,
        interface=interface_name,
        name=model_name + ":Potential@r0",
        equation="1",
    )


    interface_model(
        device=device,
        interface=interface_name,
        name=model_name + ":Potential@r1",
        equation="-1",
    )


    interface_equation(
        device=device,
        interface=interface_name,
        name="PotentialEquation",
        interface_model=model_name,
        type="continuous",
    )


# ============================================================
# 13. Contact boundary conditions
# ============================================================
#
# source = 0 V
# drain  = 0 V
# gate   = VG
#
# Potential - contact_bias = 0
# ============================================================


contact_regions = {
    "source": "MoS2",
    "drain": "MoS2",
    "gate": "GateMetal",
}


for contact_name, region in contact_regions.items():

    bc_name = contact_name + "_bc"
    bias_name = contact_name + "_bias"

    contact_node_model(
        device=device,
        contact=contact_name,
        name=bc_name,
        equation="Potential - %s" % bias_name,
    )


    contact_node_model(
        device=device,
        contact=contact_name,
        name=bc_name + ":Potential",
        equation="1",
    )


    contact_equation(
        device=device,
        contact=contact_name,
        name="PotentialEquation",
        node_model=bc_name,
        edge_charge_model="DField",
    )


# ============================================================
# 14. Biases
# ============================================================

VS = 0.0
VD = 0.0
VG = 1.0


set_parameter(
    device=device,
    region="MoS2",
    name="source_bias",
    value=VS,
)

set_parameter(
    device=device,
    region="MoS2",
    name="drain_bias",
    value=VD,
)

set_parameter(
    device=device,
    region="GateMetal",
    name="gate_bias",
    value=VG,
)


# ============================================================
# 15. Print structure
# ============================================================

print()
print("============================================================")
print("ELECTROSTATIC DEVICE READY")
print("============================================================")

print("Regions:")
print(get_region_list(device=device))

print()

print("Interfaces:")
print(get_interface_list(device=device))

print()

print("Contacts:")
print(get_contact_list(device=device))

print()

print("Bias:")
print("  Source =", VS, "V")
print("  Drain  =", VD, "V")
print("  Gate   =", VG, "V")

print()


# ============================================================
# 16. Solve Poisson equation
# ============================================================

solve(
    type="dc",
    absolute_error=1.0,
    relative_error=1.0e-10,
    maximum_iterations=50,
)


print()
print("============================================================")
print("POISSON SOLVE FINISHED")
print("============================================================")


# ============================================================
# 17. Inspect MoS2 potential
# ============================================================

potential_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="Potential",
)


print()
print("MoS2 Potential:")

print(
    "  minimum =",
    min(potential_values),
    "V",
)

print(
    "  maximum =",
    max(potential_values),
    "V",
)


# ============================================================
# 18. Save result
# ============================================================

write_devices(
    file="mos2_gaa_poisson",
    device=device,
    type="vtk",
)

write_devices(
    file="mos2_gaa_poisson.msh",
    device=device,
    type="devsim",
)


print()
print("============================================================")
print("RESULT FILES WRITTEN")
print("============================================================")

print("  mos2_gaa_poisson.vtm")
print("  *.vtu")
print("  mos2_gaa_poisson.msh")
print()