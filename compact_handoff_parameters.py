"""Canonical geometry and dielectric parameters for compact-model handoff.

This module deliberately depends only on the Python standard library.  It is
safe to import in metadata generators and unit tests that do not have DEVSIM
installed.  Runtime-facing modules in ``04_charge_trap`` and
``05_program_erase`` retain their historical public names, but derive the
final nominal geometry and dielectric values from this file.
"""

from __future__ import annotations

import math
from types import MappingProxyType


GEOMETRY_VERSION = "mos2_gaa_final_3_5_16nm_v1"
MATERIAL_VERSION = "literature_nominal_al2o3_8p9_hfo2_19p65_v1"

DEVICE_TYPE = "MoS2 cylindrical GAA charge-trap memory"
COORDINATE_SYSTEM = "2D axisymmetric cylindrical (x=r, y=z)"
MESH_COORDINATE_UNIT = "cm"

NM_TO_CM = 1.0e-7

CORE_RADIUS_NM = 10.0
MOS2_THICKNESS_NM = 2.0
TUNNEL_OXIDE_THICKNESS_NM = 3.0
CHARGE_TRAP_THICKNESS_NM = 5.0
BLOCKING_OXIDE_THICKNESS_NM = 16.0
GATE_METAL_THICKNESS_NM = 2.0
AIR_THICKNESS_NM = 2.0
CHANNEL_LENGTH_NM = 100.0

AL2O3_RELATIVE_PERMITTIVITY = 8.9
HFO2_RELATIVE_PERMITTIVITY = 19.65
MOS2_RELATIVE_PERMITTIVITY = 6.7
CORE_OXIDE_RELATIVE_PERMITTIVITY = 3.9
AIR_RELATIVE_PERMITTIVITY = 1.0

# Descriptive aliases retained for callers that prefer the quantity first.
RELATIVE_PERMITTIVITY_AL2O3 = AL2O3_RELATIVE_PERMITTIVITY
RELATIVE_PERMITTIVITY_HFO2 = HFO2_RELATIVE_PERMITTIVITY
RELATIVE_PERMITTIVITY_MOS2 = MOS2_RELATIVE_PERMITTIVITY
RELATIVE_PERMITTIVITY_CORE_OXIDE = CORE_OXIDE_RELATIVE_PERMITTIVITY
RELATIVE_PERMITTIVITY_AIR = AIR_RELATIVE_PERMITTIVITY


NOMINAL_GEOMETRY_NM = MappingProxyType(
    {
        "core_radius_nm": CORE_RADIUS_NM,
        "mos2_thickness_nm": MOS2_THICKNESS_NM,
        "tunnel_oxide_thickness_nm": TUNNEL_OXIDE_THICKNESS_NM,
        "charge_trap_thickness_nm": CHARGE_TRAP_THICKNESS_NM,
        "blocking_oxide_thickness_nm": BLOCKING_OXIDE_THICKNESS_NM,
        "gate_metal_thickness_nm": GATE_METAL_THICKNESS_NM,
        "air_thickness_nm": AIR_THICKNESS_NM,
        "channel_length_nm": CHANNEL_LENGTH_NM,
    }
)

NOMINAL_RELATIVE_PERMITTIVITY = MappingProxyType(
    {
        "CoreOxide": RELATIVE_PERMITTIVITY_CORE_OXIDE,
        "MoS2": RELATIVE_PERMITTIVITY_MOS2,
        "TunnelOxide": RELATIVE_PERMITTIVITY_AL2O3,
        "ChargeTrap": RELATIVE_PERMITTIVITY_HFO2,
        "BlockingOxide": RELATIVE_PERMITTIVITY_AL2O3,
    }
)


def nominal_geometry_kwargs() -> dict[str, float]:
    """Return a mutable copy suitable for ``create_structure(**kwargs)``."""

    return dict(NOMINAL_GEOMETRY_NM)


def geometry_dict() -> dict[str, float]:
    """Return the public canonical geometry mapping in nanometres."""

    return nominal_geometry_kwargs()


def material_dict() -> dict[str, float]:
    """Return canonical region relative permittivities."""

    return dict(NOMINAL_RELATIVE_PERMITTIVITY)


def expected_outer_radii_nm() -> dict[str, float]:
    """Return the analytically accumulated nominal radial coordinates."""

    mos2_outer_radius_nm = CORE_RADIUS_NM + MOS2_THICKNESS_NM
    tunnel_outer_radius_nm = (
        mos2_outer_radius_nm + TUNNEL_OXIDE_THICKNESS_NM
    )
    trap_outer_radius_nm = (
        tunnel_outer_radius_nm + CHARGE_TRAP_THICKNESS_NM
    )
    blocking_outer_radius_nm = (
        trap_outer_radius_nm + BLOCKING_OXIDE_THICKNESS_NM
    )
    gate_outer_radius_nm = (
        blocking_outer_radius_nm + GATE_METAL_THICKNESS_NM
    )
    air_outer_radius_nm = gate_outer_radius_nm + AIR_THICKNESS_NM

    return {
        "core_outer_radius_nm": CORE_RADIUS_NM,
        "mos2_outer_radius_nm": mos2_outer_radius_nm,
        "tunnel_oxide_outer_radius_nm": tunnel_outer_radius_nm,
        "charge_trap_outer_radius_nm": trap_outer_radius_nm,
        "blocking_oxide_outer_radius_nm": blocking_outer_radius_nm,
        "gate_outer_radius_nm": gate_outer_radius_nm,
        "air_outer_radius_nm": air_outer_radius_nm,
    }


def validate_canonical_parameters() -> None:
    """Fail fast if the nominal handoff constants become inconsistent."""

    for name, value in NOMINAL_GEOMETRY_NM.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    for region, value in NOMINAL_RELATIVE_PERMITTIVITY.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"Relative permittivity for {region} must be finite and positive."
            )

    if not GEOMETRY_VERSION or not MATERIAL_VERSION:
        raise ValueError("Geometry and material version tags must not be empty.")


validate_canonical_parameters()
