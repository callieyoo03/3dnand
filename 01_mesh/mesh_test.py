# ============================================================
# MoS2 Cylindrical GAA Charge-Trap Memory
# Stage 1 : Final 2D Geometry / Mesh
#
# Coordinate system
#   x = radial direction (r)
#   y = channel direction (z)
#
# Unit
#   length = cm
#
# 1 nm = 1e-7 cm
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
    get_region_list,
    get_interface_list,
    get_contact_list,
    write_devices,
)


# ============================================================
# 1. Device / Mesh name
# ============================================================

device = "MoS2_GAA"
mesh = "gaa_mesh"


# ============================================================
# 2. Geometry
# ============================================================
#
# Radial direction (x = r)
#
# 0 nm
# │
# │ CoreOxide
# │
# 10 nm
# │
# │ MoS2
# │
# 12 nm
# │
# │ TunnelOxide
# │
# 16 nm
# │
# │ ChargeTrap
# │
# 21 nm
# │
# │ BlockingOxide
# │
# 29 nm
# │
# │ GateMetal
# │
# 31 nm   <-- gate contact
# │
# │ Air
# │
# 33 nm
#
#
# y = z direction
#
# 0 nm     : Source
# 100 nm   : Drain
#
# ============================================================


# ------------------------------------------------------------
# Radial coordinates
# ------------------------------------------------------------

r_axis = 0.0

r_core = 1.0e-6
# 10 nm

r_mos2 = 1.2e-6
# 12 nm
# MoS2 thickness = 2.0 nm, approximately 3 layers

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


# ------------------------------------------------------------
# Channel direction
# ------------------------------------------------------------

z_source = 0.0

z_drain = 1.0e-5
# 100 nm


# ============================================================
# 3. Create Mesh
# ============================================================

create_2d_mesh(mesh=mesh)


# ============================================================
# 4. Radial mesh lines
# ============================================================

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

# MoS2 is very thin -> fine mesh
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


# ============================================================
# 5. Vertical mesh lines
# ============================================================

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
# 6. Regions
# ============================================================


# ------------------------------------------------------------
# Core oxide
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="CoreOxide",

    xl=r_axis,
    xh=r_core,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# MoS2 channel
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="MoS2",
    region="MoS2",

    xl=r_core,
    xh=r_mos2,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# Tunnel oxide
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="TunnelOxide",

    xl=r_mos2,
    xh=r_tox,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# Charge trap layer
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="ChargeTrap",
    region="ChargeTrap",

    xl=r_tox,
    xh=r_trap,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# Blocking oxide
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="Oxide",
    region="BlockingOxide",

    xl=r_trap,
    xh=r_block,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# Gate metal
# ------------------------------------------------------------

add_2d_region(
    mesh=mesh,
    material="Metal",
    region="GateMetal",

    xl=r_block,
    xh=r_gate_outer,

    yl=z_source,
    yh=z_drain,
)


# ------------------------------------------------------------
# Air
#
# GateMetal 바깥쪽에 region을 하나 더 만들어
# Gate contact가 생성될 수 있도록 합니다.
# ------------------------------------------------------------

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
# 7. Interfaces
# ============================================================


# ------------------------------------------------------------
# CoreOxide / MoS2
# ------------------------------------------------------------

add_2d_interface(
    mesh=mesh,
    name="CoreOxide_MoS2",

    region0="CoreOxide",
    region1="MoS2",

    xl=r_core,
    xh=r_core,

    yl=z_source,
    yh=z_drain,

    bloat=1.0e-10,
)


# ------------------------------------------------------------
# MoS2 / TunnelOxide
# ------------------------------------------------------------

add_2d_interface(
    mesh=mesh,
    name="MoS2_TunnelOxide",

    region0="MoS2",
    region1="TunnelOxide",

    xl=r_mos2,
    xh=r_mos2,

    yl=z_source,
    yh=z_drain,

    bloat=1.0e-10,
)


# ------------------------------------------------------------
# TunnelOxide / ChargeTrap
# ------------------------------------------------------------

add_2d_interface(
    mesh=mesh,
    name="TunnelOxide_ChargeTrap",

    region0="TunnelOxide",
    region1="ChargeTrap",

    xl=r_tox,
    xh=r_tox,

    yl=z_source,
    yh=z_drain,

    bloat=1.0e-10,
)


# ------------------------------------------------------------
# ChargeTrap / BlockingOxide
# ------------------------------------------------------------

add_2d_interface(
    mesh=mesh,
    name="ChargeTrap_BlockingOxide",

    region0="ChargeTrap",
    region1="BlockingOxide",

    xl=r_trap,
    xh=r_trap,

    yl=z_source,
    yh=z_drain,

    bloat=1.0e-10,
)


# ------------------------------------------------------------
# BlockingOxide / GateMetal
# ------------------------------------------------------------

add_2d_interface(
    mesh=mesh,
    name="BlockingOxide_GateMetal",

    region0="BlockingOxide",
    region1="GateMetal",

    xl=r_block,
    xh=r_block,

    yl=z_source,
    yh=z_drain,

    bloat=1.0e-10,
)


# ============================================================
# 8. Contacts
# ============================================================
#
# source
# drain
# gate
#
# Gate contact는 GateMetal / Air 경계에서 생성합니다.
# ============================================================


# ------------------------------------------------------------
# Source
#
# MoS2 channel bottom
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Drain
#
# MoS2 channel top
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Gate
#
# GateMetal / Air boundary
#
# x = 31 nm
# ------------------------------------------------------------

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
# 9. Finalize mesh
# ============================================================

finalize_mesh(mesh=mesh)


# ============================================================
# 10. Create Device
# ============================================================

create_device(
    mesh=mesh,
    device=device,
)


# ============================================================
# 11. Cylindrical Coordinate Settings
# ============================================================
#
# x = radial coordinate
#
# cylindrical axis = x = 0
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
# 12. Cylindrical Models
# ============================================================
#
# 실제 Poisson / drift-diffusion 해석에 사용할 region들입니다.
#
# GateMetal과 Air는 현재 geometry를 위한 auxiliary region이므로
# cylindrical semiconductor physics model에서는 제외합니다.
# ============================================================

physics_regions = (
    "CoreOxide",
    "MoS2",
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
)


for region in physics_regions:

    print(
        "Creating cylindrical models for region:",
        region
    )

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


# ============================================================
# 13. Select cylindrical integration models
# ============================================================

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

set_parameter(
    device=device,
    name="element_edge_couple_model",
    value="ElementCylindricalEdgeCouple",
)

set_parameter(
    device=device,
    name="element_node0_volume_model",
    value="ElementCylindricalNodeVolume@en0",
)

set_parameter(
    device=device,
    name="element_node1_volume_model",
    value="ElementCylindricalNodeVolume@en1",
)


# ============================================================
# 14. Device information
# ============================================================

print()
print("============================================================")
print("DEVICE CREATED SUCCESSFULLY")
print("============================================================")
print()

print("Device:")
print(device)

print()

print("Regions:")
print(get_region_list(device=device))

print()

print("Interfaces:")
print(get_interface_list(device=device))

print()

print("Contacts:")
print(get_contact_list(device=device))

print()


# ============================================================
# 15. Check gate contact explicitly
# ============================================================

contacts = get_contact_list(device=device)

if "gate" in contacts:

    print("============================================================")
    print("GATE CONTACT CREATED SUCCESSFULLY")
    print("============================================================")

else:

    print("============================================================")
    print("WARNING: GATE CONTACT WAS NOT CREATED")
    print("============================================================")


# ============================================================
# 16. Write DEVSIM mesh
# ============================================================

write_devices(
    file="mos2_gaa_mesh.msh",
    device=device,
    type="devsim",
)


# ============================================================
# 17. Write VTK mesh
# ============================================================

write_devices(
    file="mos2_gaa_mesh",
    device=device,
    type="vtk",
)


print()
print("============================================================")
print("MESH GENERATION FINISHED")
print("============================================================")

print()
print("Output files:")
print("  mos2_gaa_mesh.msh")
print("  mos2_gaa_mesh.vtm")
print("  *.vtu region files")
print()