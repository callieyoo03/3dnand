# ============================================================
# Device Structure
# MoS2 Cylindrical GAA Charge-Trap Memory
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
    cylindrical_edge_couple,
    cylindrical_node_volume,
    cylindrical_surface_area,
)


device = "MoS2_GAA"
mesh = "gaa_mesh"


# ============================================================
# Geometry
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
# Create device
# ============================================================

def create_structure():

    create_2d_mesh(mesh=mesh)

    # --------------------------------------------------------
    # radial mesh
    # --------------------------------------------------------

    radial_lines = (
        (r_axis,       1.0e-7),
        (r_core,       5.0e-8),
        (r_mos2,       2.0e-8),
        (r_tox,        5.0e-8),
        (r_trap,       5.0e-8),
        (r_block,      5.0e-8),
        (r_gate_outer, 5.0e-8),
        (r_air_outer,  1.0e-7),
    )

    for position, spacing in radial_lines:

        add_2d_mesh_line(
            mesh=mesh,
            dir="x",
            pos=position,
            ps=spacing,
        )

    # --------------------------------------------------------
    # channel direction
    # --------------------------------------------------------

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

    # ========================================================
    # Regions
    # ========================================================

    regions = (
        ("CoreOxide",     "Oxide",      r_axis,       r_core),
        ("MoS2",          "MoS2",       r_core,       r_mos2),
        ("TunnelOxide",   "Oxide",      r_mos2,       r_tox),
        ("ChargeTrap",    "ChargeTrap", r_tox,        r_trap),
        ("BlockingOxide", "Oxide",      r_trap,       r_block),
        ("GateMetal",     "Metal",      r_block,      r_gate_outer),
        ("Air",           "Air",        r_gate_outer, r_air_outer),
    )

    for region_name, material, xl, xh in regions:

        add_2d_region(
            mesh=mesh,
            material=material,
            region=region_name,
            xl=xl,
            xh=xh,
            yl=z_source,
            yh=z_drain,
        )

    # ========================================================
    # Interfaces
    # ========================================================

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

    # ========================================================
    # Contacts
    # ========================================================

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
    region="BlockingOxide",
    material="Metal",
    xl=r_block,
    xh=r_block,
    yl=z_source,
    yh=z_drain,
    bloat=1.0e-10,
)

    # ========================================================
    # Finalize
    # ========================================================

    finalize_mesh(mesh=mesh)

    create_device(
        mesh=mesh,
        device=device,
    )

    # ========================================================
    # Cylindrical geometry
    # ========================================================

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

    physics_regions = (
        "CoreOxide",
        "MoS2",
        "TunnelOxide",
        "ChargeTrap",
        "BlockingOxide",
    )

    for region in physics_regions:

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

    # ========================================================
    # Select cylindrical integration
    # ========================================================

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


    print("Device structure created.")