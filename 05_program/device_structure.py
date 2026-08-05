# ============================================================
# Device Structure
# MoS2 Cylindrical GAA Charge-Trap Memory
#
# Geometry parameters can be supplied in nm.
# Internal DEVSIM geometry units are cm.
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
# Unit conversion
# ============================================================

NM_TO_CM = 1.0e-7


def nm_to_cm(
    value_nm,
):
    return float(value_nm) * NM_TO_CM


# ============================================================
# Default geometry parameters
# ============================================================

DEFAULT_CORE_RADIUS_NM = 10.0

DEFAULT_MOS2_THICKNESS_NM = 2.0

DEFAULT_TUNNEL_OXIDE_THICKNESS_NM = 4.0

DEFAULT_CHARGE_TRAP_THICKNESS_NM = 5.0

DEFAULT_BLOCKING_OXIDE_THICKNESS_NM = 8.0

DEFAULT_GATE_METAL_THICKNESS_NM = 2.0

DEFAULT_AIR_THICKNESS_NM = 2.0

DEFAULT_CHANNEL_LENGTH_NM = 100.0


# ============================================================
# Default geometry
#
# These module-level values are preserved for compatibility
# with existing scripts that may import r_mos2, r_tox, etc.
# ============================================================

r_axis = 0.0

r_core = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
)

r_mos2 = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
)

r_tox = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
    + DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
)

r_trap = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
    + DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    + DEFAULT_CHARGE_TRAP_THICKNESS_NM
)

r_block = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
    + DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    + DEFAULT_CHARGE_TRAP_THICKNESS_NM
    + DEFAULT_BLOCKING_OXIDE_THICKNESS_NM
)

r_gate_outer = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
    + DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    + DEFAULT_CHARGE_TRAP_THICKNESS_NM
    + DEFAULT_BLOCKING_OXIDE_THICKNESS_NM
    + DEFAULT_GATE_METAL_THICKNESS_NM
)

r_air_outer = nm_to_cm(
    DEFAULT_CORE_RADIUS_NM
    + DEFAULT_MOS2_THICKNESS_NM
    + DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    + DEFAULT_CHARGE_TRAP_THICKNESS_NM
    + DEFAULT_BLOCKING_OXIDE_THICKNESS_NM
    + DEFAULT_GATE_METAL_THICKNESS_NM
    + DEFAULT_AIR_THICKNESS_NM
)

z_source = 0.0

z_drain = nm_to_cm(
    DEFAULT_CHANNEL_LENGTH_NM
)


# ============================================================
# Geometry calculation
# ============================================================

def validate_positive_geometry_parameter(
    name,
    value,
):
    numeric_value = float(value)

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def calculate_geometry(
    core_radius_nm=DEFAULT_CORE_RADIUS_NM,
    mos2_thickness_nm=DEFAULT_MOS2_THICKNESS_NM,
    tunnel_oxide_thickness_nm=(
        DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    ),
    charge_trap_thickness_nm=(
        DEFAULT_CHARGE_TRAP_THICKNESS_NM
    ),
    blocking_oxide_thickness_nm=(
        DEFAULT_BLOCKING_OXIDE_THICKNESS_NM
    ),
    gate_metal_thickness_nm=(
        DEFAULT_GATE_METAL_THICKNESS_NM
    ),
    air_thickness_nm=DEFAULT_AIR_THICKNESS_NM,
    channel_length_nm=DEFAULT_CHANNEL_LENGTH_NM,
):
    core_radius_nm = (
        validate_positive_geometry_parameter(
            "core_radius_nm",
            core_radius_nm,
        )
    )

    mos2_thickness_nm = (
        validate_positive_geometry_parameter(
            "mos2_thickness_nm",
            mos2_thickness_nm,
        )
    )

    tunnel_oxide_thickness_nm = (
        validate_positive_geometry_parameter(
            "tunnel_oxide_thickness_nm",
            tunnel_oxide_thickness_nm,
        )
    )

    charge_trap_thickness_nm = (
        validate_positive_geometry_parameter(
            "charge_trap_thickness_nm",
            charge_trap_thickness_nm,
        )
    )

    blocking_oxide_thickness_nm = (
        validate_positive_geometry_parameter(
            "blocking_oxide_thickness_nm",
            blocking_oxide_thickness_nm,
        )
    )

    gate_metal_thickness_nm = (
        validate_positive_geometry_parameter(
            "gate_metal_thickness_nm",
            gate_metal_thickness_nm,
        )
    )

    air_thickness_nm = (
        validate_positive_geometry_parameter(
            "air_thickness_nm",
            air_thickness_nm,
        )
    )

    channel_length_nm = (
        validate_positive_geometry_parameter(
            "channel_length_nm",
            channel_length_nm,
        )
    )

    core_outer_radius_nm = (
        core_radius_nm
    )

    mos2_outer_radius_nm = (
        core_outer_radius_nm
        + mos2_thickness_nm
    )

    tunnel_oxide_outer_radius_nm = (
        mos2_outer_radius_nm
        + tunnel_oxide_thickness_nm
    )

    charge_trap_outer_radius_nm = (
        tunnel_oxide_outer_radius_nm
        + charge_trap_thickness_nm
    )

    blocking_oxide_outer_radius_nm = (
        charge_trap_outer_radius_nm
        + blocking_oxide_thickness_nm
    )

    gate_outer_radius_nm = (
        blocking_oxide_outer_radius_nm
        + gate_metal_thickness_nm
    )

    air_outer_radius_nm = (
        gate_outer_radius_nm
        + air_thickness_nm
    )

    return {
        # ----------------------------------------------------
        # Thickness parameters in nm
        # ----------------------------------------------------
        "core_radius_nm": (
            core_radius_nm
        ),

        "mos2_thickness_nm": (
            mos2_thickness_nm
        ),

        "tunnel_oxide_thickness_nm": (
            tunnel_oxide_thickness_nm
        ),

        "charge_trap_thickness_nm": (
            charge_trap_thickness_nm
        ),

        "blocking_oxide_thickness_nm": (
            blocking_oxide_thickness_nm
        ),

        "gate_metal_thickness_nm": (
            gate_metal_thickness_nm
        ),

        "air_thickness_nm": (
            air_thickness_nm
        ),

        "channel_length_nm": (
            channel_length_nm
        ),

        # ----------------------------------------------------
        # Outer radii in nm
        # ----------------------------------------------------
        "core_outer_radius_nm": (
            core_outer_radius_nm
        ),

        "mos2_outer_radius_nm": (
            mos2_outer_radius_nm
        ),

        "tunnel_oxide_outer_radius_nm": (
            tunnel_oxide_outer_radius_nm
        ),

        "charge_trap_outer_radius_nm": (
            charge_trap_outer_radius_nm
        ),

        "blocking_oxide_outer_radius_nm": (
            blocking_oxide_outer_radius_nm
        ),

        "gate_outer_radius_nm": (
            gate_outer_radius_nm
        ),

        "air_outer_radius_nm": (
            air_outer_radius_nm
        ),

        # ----------------------------------------------------
        # DEVSIM coordinates in cm
        # ----------------------------------------------------
        "r_axis": (
            0.0
        ),

        "r_core": (
            nm_to_cm(
                core_outer_radius_nm
            )
        ),

        "r_mos2": (
            nm_to_cm(
                mos2_outer_radius_nm
            )
        ),

        "r_tox": (
            nm_to_cm(
                tunnel_oxide_outer_radius_nm
            )
        ),

        "r_trap": (
            nm_to_cm(
                charge_trap_outer_radius_nm
            )
        ),

        "r_block": (
            nm_to_cm(
                blocking_oxide_outer_radius_nm
            )
        ),

        "r_gate_outer": (
            nm_to_cm(
                gate_outer_radius_nm
            )
        ),

        "r_air_outer": (
            nm_to_cm(
                air_outer_radius_nm
            )
        ),

        "z_source": (
            0.0
        ),

        "z_drain": (
            nm_to_cm(
                channel_length_nm
            )
        ),
    }


# ============================================================
# Geometry information
# ============================================================

def print_geometry_summary(
    geometry,
):
    print()
    print("=" * 70)
    print("DEVICE GEOMETRY SUMMARY")
    print("=" * 70)

    print(
        f'Core radius              = '
        f'{geometry["core_radius_nm"]:.3f} nm'
    )

    print(
        f'MoS2 thickness           = '
        f'{geometry["mos2_thickness_nm"]:.3f} nm'
    )

    print(
        f'TunnelOxide thickness    = '
        f'{geometry["tunnel_oxide_thickness_nm"]:.3f} nm'
    )

    print(
        f'ChargeTrap thickness     = '
        f'{geometry["charge_trap_thickness_nm"]:.3f} nm'
    )

    print(
        f'BlockingOxide thickness  = '
        f'{geometry["blocking_oxide_thickness_nm"]:.3f} nm'
    )

    print(
        f'GateMetal thickness      = '
        f'{geometry["gate_metal_thickness_nm"]:.3f} nm'
    )

    print(
        f'Channel length           = '
        f'{geometry["channel_length_nm"]:.3f} nm'
    )

    print()
    print(
        f'MoS2 outer radius        = '
        f'{geometry["mos2_outer_radius_nm"]:.3f} nm'
    )

    print(
        f'TunnelOxide outer radius = '
        f'{geometry["tunnel_oxide_outer_radius_nm"]:.3f} nm'
    )

    print(
        f'ChargeTrap outer radius  = '
        f'{geometry["charge_trap_outer_radius_nm"]:.3f} nm'
    )

    print(
        f'BlockingOxide outer      = '
        f'{geometry["blocking_oxide_outer_radius_nm"]:.3f} nm'
    )

    print(
        f'Gate outer radius        = '
        f'{geometry["gate_outer_radius_nm"]:.3f} nm'
    )


# ============================================================
# Create device
# ============================================================

def create_structure(
    core_radius_nm=DEFAULT_CORE_RADIUS_NM,
    mos2_thickness_nm=DEFAULT_MOS2_THICKNESS_NM,
    tunnel_oxide_thickness_nm=(
        DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
    ),
    charge_trap_thickness_nm=(
        DEFAULT_CHARGE_TRAP_THICKNESS_NM
    ),
    blocking_oxide_thickness_nm=(
        DEFAULT_BLOCKING_OXIDE_THICKNESS_NM
    ),
    gate_metal_thickness_nm=(
        DEFAULT_GATE_METAL_THICKNESS_NM
    ),
    air_thickness_nm=DEFAULT_AIR_THICKNESS_NM,
    channel_length_nm=DEFAULT_CHANNEL_LENGTH_NM,
):
    geometry = calculate_geometry(
        core_radius_nm=core_radius_nm,
        mos2_thickness_nm=mos2_thickness_nm,
        tunnel_oxide_thickness_nm=(
            tunnel_oxide_thickness_nm
        ),
        charge_trap_thickness_nm=(
            charge_trap_thickness_nm
        ),
        blocking_oxide_thickness_nm=(
            blocking_oxide_thickness_nm
        ),
        gate_metal_thickness_nm=(
            gate_metal_thickness_nm
        ),
        air_thickness_nm=air_thickness_nm,
        channel_length_nm=channel_length_nm,
    )

    local_r_axis = geometry[
        "r_axis"
    ]

    local_r_core = geometry[
        "r_core"
    ]

    local_r_mos2 = geometry[
        "r_mos2"
    ]

    local_r_tox = geometry[
        "r_tox"
    ]

    local_r_trap = geometry[
        "r_trap"
    ]

    local_r_block = geometry[
        "r_block"
    ]

    local_r_gate_outer = geometry[
        "r_gate_outer"
    ]

    local_r_air_outer = geometry[
        "r_air_outer"
    ]

    local_z_source = geometry[
        "z_source"
    ]

    local_z_drain = geometry[
        "z_drain"
    ]

    print_geometry_summary(
        geometry
    )

    create_2d_mesh(
        mesh=mesh
    )

    # --------------------------------------------------------
    # Radial mesh
    # --------------------------------------------------------

    radial_lines = (
        (
            local_r_axis,
            1.0e-7,
        ),
        (
            local_r_core,
            5.0e-8,
        ),
        (
            local_r_mos2,
            2.0e-8,
        ),
        (
            local_r_tox,
            5.0e-8,
        ),
        (
            local_r_trap,
            5.0e-8,
        ),
        (
            local_r_block,
            5.0e-8,
        ),
        (
            local_r_gate_outer,
            5.0e-8,
        ),
        (
            local_r_air_outer,
            1.0e-7,
        ),
    )

    for position, spacing in radial_lines:
        add_2d_mesh_line(
            mesh=mesh,
            dir="x",
            pos=position,
            ps=spacing,
        )

    # --------------------------------------------------------
    # Channel-direction mesh
    # --------------------------------------------------------

    add_2d_mesh_line(
        mesh=mesh,
        dir="y",
        pos=local_z_source,
        ps=2.0e-7,
    )

    add_2d_mesh_line(
        mesh=mesh,
        dir="y",
        pos=local_z_drain,
        ps=2.0e-7,
    )

    # ========================================================
    # Regions
    # ========================================================

    regions = (
        (
            "CoreOxide",
            "Oxide",
            local_r_axis,
            local_r_core,
        ),
        (
            "MoS2",
            "MoS2",
            local_r_core,
            local_r_mos2,
        ),
        (
            "TunnelOxide",
            "Oxide",
            local_r_mos2,
            local_r_tox,
        ),
        (
            "ChargeTrap",
            "ChargeTrap",
            local_r_tox,
            local_r_trap,
        ),
        (
            "BlockingOxide",
            "Oxide",
            local_r_trap,
            local_r_block,
        ),
        (
            "GateMetal",
            "Metal",
            local_r_block,
            local_r_gate_outer,
        ),
        (
            "Air",
            "Air",
            local_r_gate_outer,
            local_r_air_outer,
        ),
    )

    for (
        region_name,
        material,
        xl,
        xh,
    ) in regions:
        add_2d_region(
            mesh=mesh,
            material=material,
            region=region_name,
            xl=xl,
            xh=xh,
            yl=local_z_source,
            yh=local_z_drain,
        )

    # ========================================================
    # Interfaces
    # ========================================================

    interfaces = (
        (
            "CoreOxide_MoS2",
            "CoreOxide",
            "MoS2",
            local_r_core,
        ),
        (
            "MoS2_TunnelOxide",
            "MoS2",
            "TunnelOxide",
            local_r_mos2,
        ),
        (
            "TunnelOxide_ChargeTrap",
            "TunnelOxide",
            "ChargeTrap",
            local_r_tox,
        ),
        (
            "ChargeTrap_BlockingOxide",
            "ChargeTrap",
            "BlockingOxide",
            local_r_trap,
        ),
    )

    for (
        name,
        region0,
        region1,
        xpos,
    ) in interfaces:
        add_2d_interface(
            mesh=mesh,
            name=name,
            region0=region0,
            region1=region1,
            xl=xpos,
            xh=xpos,
            yl=local_z_source,
            yh=local_z_drain,
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
        xl=local_r_core,
        xh=local_r_mos2,
        yl=local_z_source,
        yh=local_z_source,
        bloat=1.0e-10,
    )

    add_2d_contact(
        mesh=mesh,
        name="drain",
        region="MoS2",
        material="Metal",
        xl=local_r_core,
        xh=local_r_mos2,
        yl=local_z_drain,
        yh=local_z_drain,
        bloat=1.0e-10,
    )

    add_2d_contact(
        mesh=mesh,
        name="gate",
        region="BlockingOxide",
        material="Metal",
        xl=local_r_block,
        xh=local_r_block,
        yl=local_z_source,
        yh=local_z_drain,
        bloat=1.0e-10,
    )

    # ========================================================
    # Finalize mesh and create device
    # ========================================================

    finalize_mesh(
        mesh=mesh
    )

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

    # Store geometry values as device parameters.
    set_parameter(
        device=device,
        name="core_radius_nm",
        value=geometry[
            "core_radius_nm"
        ],
    )

    set_parameter(
        device=device,
        name="mos2_thickness_nm",
        value=geometry[
            "mos2_thickness_nm"
        ],
    )

    set_parameter(
        device=device,
        name="tunnel_oxide_thickness_nm",
        value=geometry[
            "tunnel_oxide_thickness_nm"
        ],
    )

    set_parameter(
        device=device,
        name="charge_trap_thickness_nm",
        value=geometry[
            "charge_trap_thickness_nm"
        ],
    )

    set_parameter(
        device=device,
        name="blocking_oxide_thickness_nm",
        value=geometry[
            "blocking_oxide_thickness_nm"
        ],
    )

    set_parameter(
        device=device,
        name="channel_length_nm",
        value=geometry[
            "channel_length_nm"
        ],
    )

    print()
    print("Device structure created.")

    return geometry