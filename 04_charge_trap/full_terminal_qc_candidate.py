"""Candidate-only helpers for full terminal charge and quasi-static Cij.

This module deliberately does not change the established ``terminal_charge``
API or any ``shared_data`` contract.  It exposes the direct Poisson-contact
fluxes as *candidate* terminal charges, together with the validation machinery
needed before they may be promoted.  A Ward-Dutton result is kept separate and
is explicitly a mobile-channel partition, not a replacement full-electrode
charge.

The capacitance convention used throughout is::

    Cij = dQi / dVj

There is no SPICE-style sign conversion.  Direct Poisson-contact charge is the
DEVSIM ``PotentialEquation`` contact charge using ``PotentialEdgeFlux`` and the
region's ``CylindricalEdgeCouple``.  Volume charge uses
``CylindricalNodeVolume``; no planar-area multiplier is introduced here.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEVICE_NAME = "MoS2_GAA"
SEMICONDUCTOR_REGION = "MoS2"
CHARGE_TRAP_REGION = "ChargeTrap"
POTENTIAL_EQUATION = "PotentialEquation"
POTENTIAL_EDGE_FLUX_MODEL = "PotentialEdgeFlux"

TERMINALS = ("gate", "drain", "source")
EXPECTED_CONTACT_REGIONS = {
    "gate": "BlockingOxide",
    "drain": SEMICONDUCTOR_REGION,
    "source": SEMICONDUCTOR_REGION,
}
CONTACT_INWARD_NORMALS_RZ = {
    # Normal is from the ideal metal contact into the simulated region.
    "gate": (-1.0, 0.0),
    "drain": (0.0, -1.0),
    "source": (0.0, 1.0),
}
CONTACT_REGION_OUTWARD_NORMALS_RZ = {
    terminal: (-normal_r, -normal_z)
    for terminal, (normal_r, normal_z) in CONTACT_INWARD_NORMALS_RZ.items()
}

DIRECT_CONTACT_METHOD = "direct_poisson_contact_flux"
WARD_DUTTON_METHOD = "ward_dutton_mobile_channel"
TERMINAL_STATUS_SUPPORTED = "supported"
TERMINAL_STATUS_PROVISIONAL = "provisional"
TERMINAL_STATUS_UNSUPPORTED = "unsupported"
WARD_DUTTON_STATUS_MOBILE_ONLY = "mobile_partition_only"

DEFAULT_CHARGE_FLOOR_C = 1.0e-30
DEFAULT_CAPACITANCE_FLOOR_F = 1.0e-30
DEFAULT_MATRIX_ABSOLUTE_TOLERANCE_F = 1.0e-24
DEFAULT_MATRIX_RELATIVE_TOLERANCE = 1.0e-6

DIRECT_CONTACT_LIMITATION = (
    "Poisson-contact flux is an electrostatic boundary reaction of the "
    "modeled ideal contact. It is not a Ward-Dutton partition of distributed "
    "mobile-channel charge and is promoted only after every candidate "
    "validation criterion passes."
)
DIRECT_CONTACT_COLUMN_SUM_LIMITATION = (
    "For raw Poisson-contact fluxes, Gauss law gives sum_i(Cij) = "
    "-d(Qmobile+Qtrap+Qfixed)/dVj. With fixed trap and doping this is "
    "generally -dQmobile/dVj, not zero; a zero three-terminal column sum is "
    "therefore a substantive promotion test rather than an identity."
)
WARD_DUTTON_LIMITATION = (
    "Ward-Dutton values partition signed MoS2 mobile charge only. They exclude "
    "trap, fixed-dopant, and dielectric/electrode charge and must not be "
    "reported as full terminal charges."
)


TERMINAL_CHARGE_FULL_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "Qg_C",
    "Qd_C",
    "Qs_C",
    "Qg_contact_C",
    "Qd_contact_C",
    "Qs_contact_C",
    "Qmobile_C",
    "Qtrap_C",
    "Qfixed_C",
    "global_residual_C",
    "absolute_residual_C",
    "charge_scale_C",
    "relative_residual",
    "method",
    "supported_quantities",
    "terminal_charge_status",
    "limitation",
    "mesh_level",
    "converged",
    "error_message",
)

CAPACITANCE_MATRIX_FULL_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "measured_terminal",
    "perturbed_terminal",
    "capacitance_F",
    "raw_contact_flux_capacitance_F",
    "delta_voltage_V",
    "method",
    "supported",
    "terminal_charge_status",
    "limitation",
    "mesh_level",
    "converged",
    "error_message",
)

CAPACITANCE_SUMMARY_FULL_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "Cgg_F",
    "Cgd_F",
    "Cgs_F",
    "Cdg_F",
    "Cdd_F",
    "Cds_F",
    "Csg_F",
    "Csd_F",
    "Css_F",
    "nominal_delta_voltage_V",
    "sensitivity_delta_voltages_V",
    "maximum_delta_relative_sensitivity",
    "maximum_absolute_row_sum_F",
    "maximum_absolute_column_sum_F",
    "row_sums_passed",
    "column_sums_passed",
    "method",
    "terminal_charge_status",
    "limitation",
    "mesh_level",
    "converged",
    "error_message",
)

DIRECT_CONTACT_VALIDATION_FIELDNAMES = (
    "state_index",
    "state",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "mesh_level",
    "all_quantities_finite",
    "contacts_complete",
    "contact_topology_passed",
    "contact_sign_passed",
    "global_gauss_passed",
    "repeatability_passed",
    "mesh_refinement_passed",
    "continuity_passed",
    "gauge_invariance_passed",
    "matrix_row_sums_passed",
    "matrix_column_sums_passed",
    "global_residual_C",
    "absolute_residual_C",
    "charge_scale_C",
    "relative_residual",
    "terminal_charge_status",
    "limitation",
    "error_message",
)

WARD_DUTTON_FIELDNAMES = (
    "state_index",
    "state",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "Qmobile_channel_C",
    "Qs_mobile_C",
    "Qd_mobile_C",
    "partition_residual_C",
    "absolute_partition_residual_C",
    "relative_partition_residual",
    "minimum_source_weight",
    "maximum_source_weight",
    "minimum_drain_weight",
    "maximum_drain_weight",
    "method",
    "terminal_charge_status",
    "limitation",
    "mesh_level",
    "converged",
    "error_message",
)

METHOD_COMPARISON_FIELDNAMES = (
    "state_index",
    "state",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "Qg_direct_contact_C",
    "Qd_direct_contact_C",
    "Qs_direct_contact_C",
    "Qmobile_channel_C",
    "Qd_ward_dutton_mobile_C",
    "Qs_ward_dutton_mobile_C",
    "direct_method",
    "ward_dutton_method",
    "comparison_scope",
    "mesh_level",
)

MESH_REFINEMENT_QC_FIELDNAMES = (
    "state_index",
    "state",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "comparison",
    "reference_mesh_level",
    "candidate_mesh_level",
    "quantity",
    "reference_value",
    "candidate_value",
    "absolute_difference",
    "relative_difference",
    "comparison_floor",
    "relative_tolerance",
    "passed",
)


_GAUSS_COMPONENT_ALIASES = {
    "Qg_contact_C": ("Qg_contact_C", "Qg_C"),
    "Qd_contact_C": (
        "Qd_contact_C",
        "Qd_C",
        "Qd_poisson_boundary_flux_C",
    ),
    "Qs_contact_C": (
        "Qs_contact_C",
        "Qs_C",
        "Qs_poisson_boundary_flux_C",
    ),
    "Qmobile_C": ("Qmobile_C",),
    "Qtrap_C": ("Qtrap_C",),
    "Qfixed_C": ("Qfixed_C",),
}


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return result


def _first_present(components: Mapping[str, Any], aliases: Sequence[str]) -> float:
    for name in aliases:
        if name in components:
            return _finite_float(components[name], name)
    raise KeyError("missing charge component; accepted names: " + ", ".join(aliases))


def calculate_scaled_gauss_residual(
    components: Mapping[str, Any],
    *,
    configured_charge_floor_C: float = DEFAULT_CHARGE_FLOOR_C,
) -> dict[str, float]:
    """Return signed/absolute Gauss residual using a max-component scale.

    The scale is exactly ``max(abs(each component), configured floor)``.  This
    avoids an unstable relative residual when the physical charges are near
    zero and differs intentionally from the legacy sum-of-magnitudes check.
    """

    floor_C = _finite_float(configured_charge_floor_C, "configured_charge_floor_C")
    if floor_C <= 0.0:
        raise ValueError("configured_charge_floor_C must be positive.")

    resolved = {
        canonical: _first_present(components, aliases)
        for canonical, aliases in _GAUSS_COMPONENT_ALIASES.items()
    }
    signed_C = math.fsum(resolved.values())
    absolute_C = abs(signed_C)
    scale_C = max(floor_C, *(abs(value) for value in resolved.values()))
    return {
        "global_residual_C": signed_C,
        "absolute_residual_C": absolute_C,
        "charge_scale_C": scale_C,
        "relative_residual": absolute_C / scale_C,
    }


def extract_direct_contact_charge_components(
    *,
    device: str = DEVICE_NAME,
    semiconductor_region: str = SEMICONDUCTOR_REGION,
    charge_trap_region: str = CHARGE_TRAP_REGION,
    elementary_charge_C: float | None = None,
    configured_charge_floor_C: float = DEFAULT_CHARGE_FLOOR_C,
) -> dict[str, Any]:
    """Extract all three Poisson-contact charges and physical volume charge."""

    from terminal_charge import extract_terminal_charge_components

    legacy = extract_terminal_charge_components(
        device=device,
        semiconductor_region=semiconductor_region,
        charge_trap_region=charge_trap_region,
        elementary_charge_C=elementary_charge_C,
    )
    components = {
        "Qg_contact_C": _finite_float(legacy["Qg_C"], "Qg_C"),
        "Qd_contact_C": _finite_float(
            legacy["Qd_poisson_boundary_flux_C"],
            "Qd_poisson_boundary_flux_C",
        ),
        "Qs_contact_C": _finite_float(
            legacy["Qs_poisson_boundary_flux_C"],
            "Qs_poisson_boundary_flux_C",
        ),
        "Qmobile_C": _finite_float(legacy["Qmobile_C"], "Qmobile_C"),
        "Qtrap_C": _finite_float(legacy["Qtrap_C"], "Qtrap_C"),
        "Qfixed_C": _finite_float(legacy["Qfixed_C"], "Qfixed_C"),
    }
    residual = calculate_scaled_gauss_residual(
        components,
        configured_charge_floor_C=configured_charge_floor_C,
    )
    return {
        **components,
        **residual,
        "all_finite": True,
        "method": DIRECT_CONTACT_METHOD,
    }


def _normalize_contact_elements(raw_elements: Any) -> tuple[tuple[int, int], ...]:
    values = tuple(raw_elements)
    if not values:
        return ()
    if all(isinstance(value, (int, float)) for value in values):
        if len(values) % 2:
            raise RuntimeError("flat 2-D contact node list must contain node pairs.")
        values = tuple(values[index : index + 2] for index in range(0, len(values), 2))

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, element in enumerate(values):
        nodes = tuple(int(node) for node in element)
        if len(nodes) != 2 or nodes[0] == nodes[1]:
            raise RuntimeError(
                f"contact element {index} must contain two distinct node indices."
            )
        canonical = tuple(sorted(nodes))
        if canonical not in seen:
            edges.append((nodes[0], nodes[1]))
            seen.add(canonical)
    return tuple(edges)


def _cylindrical_contact_surface_area(
    edges: Sequence[tuple[int, int]],
    radial_coordinates_cm: Sequence[float],
    axial_coordinates_cm: Sequence[float],
) -> float:
    """Sweep each r-z contact edge through 2*pi around the r=x axis."""

    areas: list[float] = []
    for node0, node1 in edges:
        try:
            r0 = _nonnegative_float(radial_coordinates_cm[node0], f"x[{node0}]")
            r1 = _nonnegative_float(radial_coordinates_cm[node1], f"x[{node1}]")
            z0 = _finite_float(axial_coordinates_cm[node0], f"y[{node0}]")
            z1 = _finite_float(axial_coordinates_cm[node1], f"y[{node1}]")
        except IndexError as error:
            raise RuntimeError("contact node index is outside its region arrays.") from error
        meridional_length_cm = math.hypot(r1 - r0, z1 - z0)
        areas.append(math.pi * abs(r0 + r1) * meridional_length_cm)
    return math.fsum(areas)


def audit_runtime_contact_topology(
    *,
    device: str = DEVICE_NAME,
    geometry: Mapping[str, Any],
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = 1.0e-12,
    surface_relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Audit terminal regions, contact planes, normals, and swept areas.

    DEVSIM does not accept an explicit contact normal for ``edge_charge_model``.
    Its contact equation gives an incident edge a ``+`` sign at n0/head and a
    ``-`` sign at n1/tail; the resulting charge normal is from the ideal metal
    into the simulated region.  Here that means source ``+z``, drain ``-z``,
    and gate ``-r``.  The coordinate audit makes those inferred directions
    explicit and rejects misplaced or incomplete contacts.
    """

    tolerance_cm = _nonnegative_float(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    surface_rtol = _nonnegative_float(
        surface_relative_tolerance, "surface_relative_tolerance"
    )
    if runtime_api is None:
        import devsim as runtime_api

    required_geometry = ("r_core", "r_mos2", "r_block", "z_source", "z_drain")
    missing_geometry = [name for name in required_geometry if name not in geometry]
    if missing_geometry:
        raise KeyError("geometry is missing: " + ", ".join(missing_geometry))
    geo = {name: _finite_float(geometry[name], name) for name in required_geometry}
    if not geo["r_core"] < geo["r_mos2"] < geo["r_block"]:
        raise ValueError("expected r_core < r_mos2 < r_block.")
    if not geo["z_source"] < geo["z_drain"]:
        raise ValueError("expected z_source < z_drain.")

    contacts = tuple(str(name) for name in runtime_api.get_contact_list(device=device))
    contacts_complete = set(contacts) == set(TERMINALS)
    axis = str(runtime_api.get_parameter(device=device, name="raxis_variable"))
    axis_zero = _finite_float(
        runtime_api.get_parameter(device=device, name="raxis_zero"),
        "raxis_zero",
    )
    node_volume_model = str(
        runtime_api.get_parameter(device=device, name="node_volume_model")
    )
    edge_couple_model = str(
        runtime_api.get_parameter(device=device, name="edge_couple_model")
    )
    axisymmetric_models_ok = (
        axis == "x"
        and math.isclose(axis_zero, 0.0, rel_tol=0.0, abs_tol=1.0e-30)
        and node_volume_model == "CylindricalNodeVolume"
        and edge_couple_model == "CylindricalEdgeCouple"
    )

    expected_plane = {
        "source": ("y", geo["z_source"]),
        "drain": ("y", geo["z_drain"]),
        "gate": ("x", geo["r_block"]),
    }
    expected_surface = {
        "source": math.pi * (geo["r_mos2"] ** 2 - geo["r_core"] ** 2),
        "drain": math.pi * (geo["r_mos2"] ** 2 - geo["r_core"] ** 2),
        "gate": 2.0 * math.pi * geo["r_block"] * (
            geo["z_drain"] - geo["z_source"]
        ),
    }

    contact_reports: dict[str, dict[str, Any]] = {}
    for contact in TERMINALS:
        if contact not in contacts:
            contact_reports[contact] = {
                "region": "",
                "edge_count": 0,
                "node_count": 0,
                "surface_area_cm2": math.nan,
                "expected_surface_area_cm2": expected_surface[contact],
                "surface_relative_error": math.nan,
                "normal_r": math.nan,
                "normal_z": math.nan,
                "normal_magnitude": math.nan,
                "expected_region_outward_normal_r": (
                    CONTACT_REGION_OUTWARD_NORMALS_RZ[contact][0]
                ),
                "expected_region_outward_normal_z": (
                    CONTACT_REGION_OUTWARD_NORMALS_RZ[contact][1]
                ),
                "charge_normal_r": CONTACT_INWARD_NORMALS_RZ[contact][0],
                "charge_normal_z": CONTACT_INWARD_NORMALS_RZ[contact][1],
                "normal_consistent": False,
                "plane_consistent": False,
                "surface_consistent": False,
                "equation_consistent": False,
                "passed": False,
            }
            continue

        regions = tuple(
            str(name)
            for name in runtime_api.get_region_list(device=device, contact=contact)
        )
        region = regions[0] if len(regions) == 1 else ""
        region_ok = region == EXPECTED_CONTACT_REGIONS[contact]
        edges: tuple[tuple[int, int], ...] = ()
        radial: tuple[float, ...] = ()
        axial: tuple[float, ...] = ()
        at_contact: tuple[float, ...] = ()
        normal_r_values: tuple[float, ...] = ()
        normal_z_values: tuple[float, ...] = ()
        planar_surface_values: tuple[float, ...] = ()
        cylindrical_surface_values: tuple[float, ...] = ()
        if region:
            edges = _normalize_contact_elements(
                runtime_api.get_element_node_list(
                    device=device,
                    region=region,
                    contact=contact,
                )
            )
            radial = tuple(
                float(value)
                for value in runtime_api.get_node_model_values(
                    device=device, region=region, name="x"
                )
            )
            axial = tuple(
                float(value)
                for value in runtime_api.get_node_model_values(
                    device=device, region=region, name="y"
                )
            )
            def read_node_model(name: str) -> tuple[float, ...]:
                return tuple(
                    float(value)
                    for value in runtime_api.get_node_model_values(
                        device=device, region=region, name=name
                    )
                )

            at_contact = read_node_model("AtContactNode")
            normal_r_values = read_node_model("ContactNSurfaceNormal_x")
            normal_z_values = read_node_model("ContactNSurfaceNormal_y")
            planar_surface_values = read_node_model("ContactSurfaceArea")
            cylindrical_surface_values = read_node_model("CylindricalSurfaceArea")

        plane_axis, plane_value = expected_plane[contact]
        coordinate_values = radial if plane_axis == "x" else axial
        contact_node_indices = {
            index
            for index, marker in enumerate(at_contact)
            if marker > 0.5
            and math.isclose(
                coordinate_values[index],
                plane_value,
                rel_tol=0.0,
                abs_tol=tolerance_cm,
            )
        }
        # The source/drain contacts in this mesh are two-node boundary
        # segments.  DEVSIM currently returns no elements for those contacts,
        # so use the contact-node mask plus their unique axial plane rather
        # than manufacturing contact edges.
        node_indices = tuple(sorted(contact_node_indices))
        plane_ok = bool(node_indices) and all(
            math.isclose(
                _finite_float(coordinate_values[node], f"{plane_axis}[{node}]"),
                plane_value,
                rel_tol=0.0,
                abs_tol=tolerance_cm,
            )
            for node in node_indices
        )
        edge_swept_surface_area_cm2 = (
            _cylindrical_contact_surface_area(edges, radial, axial)
            if edges
            else math.nan
        )
        planar_surface_area_cm = math.fsum(
            planar_surface_values[node] for node in node_indices
        )
        surface_area_cm2 = math.fsum(
            cylindrical_surface_values[node] for node in node_indices
        )
        expected_area_cm2 = expected_surface[contact]
        surface_relative_error = (
            abs(surface_area_cm2 - expected_area_cm2)
            / max(abs(expected_area_cm2), DEFAULT_CHARGE_FLOOR_C)
            if math.isfinite(surface_area_cm2)
            else math.nan
        )
        surface_ok = (
            math.isfinite(surface_relative_error)
            and surface_relative_error <= surface_rtol
        )

        actual_normal_r = math.fsum(
            normal_r_values[node] for node in node_indices
        ) / len(node_indices) if node_indices else math.nan
        actual_normal_z = math.fsum(
            normal_z_values[node] for node in node_indices
        ) / len(node_indices) if node_indices else math.nan
        normal_magnitude = (
            math.hypot(actual_normal_r, actual_normal_z)
            if node_indices
            else math.nan
        )
        expected_normal_r, expected_normal_z = (
            CONTACT_REGION_OUTWARD_NORMALS_RZ[contact]
        )
        normal_ok = (
            math.isfinite(normal_magnitude)
            and math.isclose(normal_magnitude, 1.0, rel_tol=surface_rtol, abs_tol=1e-12)
            and math.isclose(
                actual_normal_r,
                expected_normal_r,
                rel_tol=surface_rtol,
                abs_tol=1e-12,
            )
            and math.isclose(
                actual_normal_z,
                expected_normal_z,
                rel_tol=surface_rtol,
                abs_tol=1e-12,
            )
        )

        equation_names = tuple(
            str(name)
            for name in runtime_api.get_contact_equation_list(
                device=device, contact=contact
            )
        )
        command: Mapping[str, Any] = {}
        if POTENTIAL_EQUATION in equation_names:
            command = runtime_api.get_contact_equation_command(
                device=device,
                contact=contact,
                name=POTENTIAL_EQUATION,
            )
        equation_ok = (
            POTENTIAL_EQUATION in equation_names
            and str(command.get("edge_charge_model", ""))
            == POTENTIAL_EDGE_FLUX_MODEL
            and not str(command.get("node_charge_model", ""))
            and not str(command.get("element_charge_model", ""))
        )
        passed = (
            region_ok
            and bool(node_indices)
            and plane_ok
            and surface_ok
            and normal_ok
            and equation_ok
            and axisymmetric_models_ok
        )
        contact_reports[contact] = {
            "region": region,
            "edge_count": len(edges),
            "node_count": len(node_indices),
            "surface_area_cm2": surface_area_cm2,
            "planar_contact_surface_area_cm": planar_surface_area_cm,
            "edge_swept_surface_area_cm2": edge_swept_surface_area_cm2,
            "expected_surface_area_cm2": expected_area_cm2,
            "surface_relative_error": surface_relative_error,
            "normal_r": actual_normal_r,
            "normal_z": actual_normal_z,
            "normal_magnitude": normal_magnitude,
            "expected_region_outward_normal_r": expected_normal_r,
            "expected_region_outward_normal_z": expected_normal_z,
            "charge_normal_r": CONTACT_INWARD_NORMALS_RZ[contact][0],
            "charge_normal_z": CONTACT_INWARD_NORMALS_RZ[contact][1],
            "normal_consistent": normal_ok,
            "plane_consistent": plane_ok,
            "surface_consistent": surface_ok,
            "equation_consistent": equation_ok,
            "region_consistent": region_ok,
            "passed": passed,
        }

    sign_consistent = all(
        bool(contact_reports[contact].get("normal_consistent"))
        for contact in TERMINALS
    )
    passed = (
        contacts_complete
        and axisymmetric_models_ok
        and sign_consistent
        and all(bool(report["passed"]) for report in contact_reports.values())
    )
    return {
        "contacts_complete": contacts_complete,
        "axisymmetric_models_ok": axisymmetric_models_ok,
        "node_volume_model": node_volume_model,
        "edge_couple_model": edge_couple_model,
        "raxis_variable": axis,
        "raxis_zero": axis_zero,
        "contact_charge_sign_definition": (
            "+PotentialEdgeFlux*CylindricalEdgeCouple at n0/head and the "
            "negative at n1/tail. ContactNSurfaceNormal is the region-outward "
            "normal; the charge normal used here is its negative, from contact "
            "into the simulated region."
        ),
        "sign_consistent": sign_consistent,
        "contacts": contact_reports,
        "passed": passed,
        "limitations": DIRECT_CONTACT_LIMITATION,
    }


def calculate_ward_dutton_mobile_partition(
    signed_mobile_charge_density_C_cm3: Sequence[float],
    cylindrical_node_volumes_cm3: Sequence[float],
    axial_coordinates_cm: Sequence[float],
    source_position_cm: float,
    drain_position_cm: float,
    *,
    coordinate_tolerance_cm: float = 1.0e-15,
    charge_floor_C: float = DEFAULT_CHARGE_FLOOR_C,
) -> dict[str, float | bool]:
    """Partition signed MoS2 mobile charge with linear Ward-Dutton weights."""

    count = len(cylindrical_node_volumes_cm3)
    if not (
        len(signed_mobile_charge_density_C_cm3)
        == len(axial_coordinates_cm)
        == count
    ):
        raise ValueError("density, volume, and axial-coordinate arrays must match.")
    if count == 0:
        raise ValueError("at least one semiconductor node is required.")
    zs = _finite_float(source_position_cm, "source_position_cm")
    zd = _finite_float(drain_position_cm, "drain_position_cm")
    length_cm = zd - zs
    if length_cm <= 0.0:
        raise ValueError("drain_position_cm must exceed source_position_cm.")
    coordinate_tolerance = _nonnegative_float(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    floor_C = _finite_float(charge_floor_C, "charge_floor_C")
    if floor_C <= 0.0:
        raise ValueError("charge_floor_C must be positive.")

    source_weights: list[float] = []
    drain_weights: list[float] = []
    mobile_elements_C: list[float] = []
    for index, (density, volume, axial) in enumerate(
        zip(
            signed_mobile_charge_density_C_cm3,
            cylindrical_node_volumes_cm3,
            axial_coordinates_cm,
        )
    ):
        rho = _finite_float(density, f"signed_mobile_charge_density_C_cm3[{index}]")
        node_volume = _nonnegative_float(
            volume, f"cylindrical_node_volumes_cm3[{index}]"
        )
        z = _finite_float(axial, f"axial_coordinates_cm[{index}]")
        if z < zs - coordinate_tolerance or z > zd + coordinate_tolerance:
            raise ValueError(f"axial coordinate {index} lies outside source/drain bounds.")
        fraction = (z - zs) / length_cm
        if fraction < 0.0 and abs(z - zs) <= coordinate_tolerance:
            fraction = 0.0
        if fraction > 1.0 and abs(z - zd) <= coordinate_tolerance:
            fraction = 1.0
        drain_weight = fraction
        source_weight = 1.0 - fraction
        if not 0.0 <= source_weight <= 1.0 or not 0.0 <= drain_weight <= 1.0:
            raise RuntimeError("Ward-Dutton weights must lie in [0, 1].")
        source_weights.append(source_weight)
        drain_weights.append(drain_weight)
        mobile_elements_C.append(rho * node_volume)

    qmobile_C = math.fsum(mobile_elements_C)
    qs_mobile_C = math.fsum(
        weight * charge
        for weight, charge in zip(source_weights, mobile_elements_C)
    )
    qd_mobile_C = math.fsum(
        weight * charge
        for weight, charge in zip(drain_weights, mobile_elements_C)
    )
    residual_C = math.fsum((qs_mobile_C, qd_mobile_C, -qmobile_C))
    absolute_residual_C = abs(residual_C)
    scale_C = max(
        floor_C,
        abs(qmobile_C),
        abs(qs_mobile_C),
        abs(qd_mobile_C),
    )
    max_weight_sum_error = max(
        abs(source + drain - 1.0)
        for source, drain in zip(source_weights, drain_weights)
    )
    return {
        "Qmobile_channel_C": qmobile_C,
        "Qs_mobile_C": qs_mobile_C,
        "Qd_mobile_C": qd_mobile_C,
        "partition_residual_C": residual_C,
        "absolute_partition_residual_C": absolute_residual_C,
        "relative_partition_residual": absolute_residual_C / scale_C,
        "minimum_source_weight": min(source_weights),
        "maximum_source_weight": max(source_weights),
        "minimum_drain_weight": min(drain_weights),
        "maximum_drain_weight": max(drain_weights),
        "maximum_weight_sum_error": max_weight_sum_error,
        "weights_valid": max_weight_sum_error <= 1.0e-15,
    }


def extract_ward_dutton_mobile_partition(
    *,
    device: str = DEVICE_NAME,
    semiconductor_region: str = SEMICONDUCTOR_REGION,
    source_position_cm: float,
    drain_position_cm: float,
    elementary_charge_C: float | None = None,
) -> dict[str, float | bool]:
    """Read the MoS2 nodal arrays and apply the mobile-only partition."""

    from devsim import get_node_model_values

    if elementary_charge_C is None:
        import trap_parameters

        elementary_charge_C = float(trap_parameters.q)
    q_C = _finite_float(elementary_charge_C, "elementary_charge_C")
    if q_C <= 0.0:
        raise ValueError("elementary_charge_C must be positive.")

    def values(model: str) -> tuple[float, ...]:
        return tuple(
            float(value)
            for value in get_node_model_values(
                device=device,
                region=semiconductor_region,
                name=model,
            )
        )

    electrons = values("Electrons")
    holes = values("EquilibriumHoles")
    if len(electrons) != len(holes):
        raise RuntimeError("electron and hole arrays differ in length.")
    signed_density = tuple(
        q_C * (hole - electron)
        for electron, hole in zip(electrons, holes)
    )
    result = calculate_ward_dutton_mobile_partition(
        signed_density,
        values("CylindricalNodeVolume"),
        values("y"),
        source_position_cm,
        drain_position_cm,
    )
    return {
        **result,
        "method": WARD_DUTTON_METHOD,
        "terminal_charge_status": WARD_DUTTON_STATUS_MOBILE_ONLY,
        "limitation": WARD_DUTTON_LIMITATION,
        "converged": True,
        "error_message": "",
    }


def vector_central_difference(
    plus: Mapping[Hashable, Any],
    minus: Mapping[Hashable, Any],
    delta_voltage_V: float,
) -> dict[Hashable, float]:
    """Return componentwise ``(Q+ - Q-)/(2*deltaV)``."""

    if set(plus) != set(minus):
        raise ValueError("plus and minus charge vectors must have identical keys.")
    if not plus:
        raise ValueError("charge vectors must not be empty.")
    delta = _finite_float(delta_voltage_V, "delta_voltage_V")
    if delta <= 0.0:
        raise ValueError("delta_voltage_V must be positive.")
    return {
        key: (
            _finite_float(plus[key], f"plus[{key!r}]")
            - _finite_float(minus[key], f"minus[{key!r}]")
        )
        / (2.0 * delta)
        for key in plus
    }


def _validated_matrix(
    matrix_F: Mapping[tuple[str, str], Any],
    terminals: Sequence[str],
) -> dict[tuple[str, str], float]:
    ordered_terminals = tuple(str(name) for name in terminals)
    expected = {
        (measured, perturbed)
        for measured in ordered_terminals
        for perturbed in ordered_terminals
    }
    if set(matrix_F) != expected:
        missing = expected.difference(matrix_F)
        extra = set(matrix_F).difference(expected)
        raise ValueError(
            "capacitance matrix must contain every terminal pair; "
            f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
        )
    return {
        key: _finite_float(matrix_F[key], f"matrix_F[{key!r}]")
        for key in expected
    }


def calculate_matrix_sums(
    matrix_F: Mapping[tuple[str, str], Any],
    *,
    terminals: Sequence[str] = TERMINALS,
    capacitance_floor_F: float = DEFAULT_CAPACITANCE_FLOOR_F,
) -> dict[str, Any]:
    """Calculate measured-terminal rows and perturbed-terminal columns."""

    ordered_terminals = tuple(str(name) for name in terminals)
    matrix = _validated_matrix(matrix_F, ordered_terminals)
    floor_F = _finite_float(capacitance_floor_F, "capacitance_floor_F")
    if floor_F <= 0.0:
        raise ValueError("capacitance_floor_F must be positive.")
    row_sums = {
        measured: math.fsum(
            matrix[(measured, perturbed)] for perturbed in ordered_terminals
        )
        for measured in ordered_terminals
    }
    column_sums = {
        perturbed: math.fsum(
            matrix[(measured, perturbed)] for measured in ordered_terminals
        )
        for perturbed in ordered_terminals
    }
    row_scales = {
        measured: max(
            floor_F,
            *(abs(matrix[(measured, perturbed)]) for perturbed in ordered_terminals),
        )
        for measured in ordered_terminals
    }
    column_scales = {
        perturbed: max(
            floor_F,
            *(abs(matrix[(measured, perturbed)]) for measured in ordered_terminals),
        )
        for perturbed in ordered_terminals
    }
    return {
        "row_sums_F": row_sums,
        "column_sums_F": column_sums,
        "row_scales_F": row_scales,
        "column_scales_F": column_scales,
        "maximum_absolute_row_sum_F": max(abs(value) for value in row_sums.values()),
        "maximum_absolute_column_sum_F": max(
            abs(value) for value in column_sums.values()
        ),
    }


def evaluate_matrix_sum_tolerances(
    matrix_F: Mapping[tuple[str, str], Any],
    *,
    terminals: Sequence[str] = TERMINALS,
    absolute_tolerance_F: float = DEFAULT_MATRIX_ABSOLUTE_TOLERANCE_F,
    relative_tolerance: float = DEFAULT_MATRIX_RELATIVE_TOLERANCE,
    capacitance_floor_F: float = DEFAULT_CAPACITANCE_FLOOR_F,
) -> dict[str, Any]:
    """Check every row/column with ``atol + rtol*max(component,floor)``."""

    absolute = _nonnegative_float(absolute_tolerance_F, "absolute_tolerance_F")
    relative = _nonnegative_float(relative_tolerance, "relative_tolerance")
    diagnostics = calculate_matrix_sums(
        matrix_F,
        terminals=terminals,
        capacitance_floor_F=capacitance_floor_F,
    )
    row_tolerances = {
        terminal: absolute + relative * diagnostics["row_scales_F"][terminal]
        for terminal in terminals
    }
    column_tolerances = {
        terminal: absolute + relative * diagnostics["column_scales_F"][terminal]
        for terminal in terminals
    }
    row_pass = {
        terminal: abs(diagnostics["row_sums_F"][terminal])
        <= row_tolerances[terminal]
        for terminal in terminals
    }
    column_pass = {
        terminal: abs(diagnostics["column_sums_F"][terminal])
        <= column_tolerances[terminal]
        for terminal in terminals
    }
    return {
        **diagnostics,
        "row_tolerances_F": row_tolerances,
        "column_tolerances_F": column_tolerances,
        "row_pass": row_pass,
        "column_pass": column_pass,
        "row_sums_passed": all(row_pass.values()),
        "column_sums_passed": all(column_pass.values()),
        "passed": all(row_pass.values()) and all(column_pass.values()),
    }


def calculate_delta_sensitivity(
    capacitances_by_delta_V: Mapping[float, Mapping[Hashable, Any] | Any],
    *,
    nominal_delta_voltage_V: float,
    capacitance_floor_F: float = DEFAULT_CAPACITANCE_FLOOR_F,
) -> dict[str, Any]:
    """Return maximum relative deviation from a nominal full-matrix result."""

    nominal_delta = _finite_float(
        nominal_delta_voltage_V, "nominal_delta_voltage_V"
    )
    if nominal_delta not in capacitances_by_delta_V:
        raise KeyError("nominal delta is absent from capacitance results.")
    floor_F = _finite_float(capacitance_floor_F, "capacitance_floor_F")
    if floor_F <= 0.0:
        raise ValueError("capacitance_floor_F must be positive.")
    if not capacitances_by_delta_V:
        raise ValueError("at least one delta-voltage result is required.")

    nominal_raw = capacitances_by_delta_V[nominal_delta]
    scalar_mode = not isinstance(nominal_raw, Mapping)
    normalized: dict[float, Mapping[Hashable, Any]] = {}
    for delta_raw, values in capacitances_by_delta_V.items():
        delta = _finite_float(delta_raw, "delta voltage")
        if delta <= 0.0:
            raise ValueError("delta voltages must be positive.")
        if scalar_mode:
            if isinstance(values, Mapping):
                raise ValueError("delta results must be uniformly scalar or mapping.")
            normalized[delta] = {"value": values}
        else:
            if not isinstance(values, Mapping):
                raise ValueError("delta results must be uniformly scalar or mapping.")
            normalized[delta] = values

    nominal = normalized[nominal_delta]
    keys = set(nominal)
    if not keys:
        raise ValueError("capacitance result vectors must not be empty.")
    if any(set(values) != keys for values in normalized.values()):
        raise ValueError("every delta result must contain identical matrix keys.")
    sensitivities: dict[Hashable, float] = {}
    for key in nominal:
        reference = _finite_float(nominal[key], f"nominal[{key!r}]")
        maximum_deviation = max(
            abs(_finite_float(values[key], f"capacitance[{key!r}]") - reference)
            for values in normalized.values()
        )
        sensitivities[key] = maximum_deviation / max(abs(reference), floor_F)
    maximum = max(sensitivities.values())
    result: dict[str, Any] = {
        "nominal_delta_voltage_V": nominal_delta,
        "relative_sensitivity_by_entry": sensitivities,
        "maximum_relative_sensitivity": maximum,
    }
    if scalar_mode:
        result["relative_sensitivity"] = sensitivities["value"]
    return result


def determine_terminal_charge_status(
    criteria: Mapping[str, bool | None],
) -> str:
    """Promote only all-true evidence; false fails and unknown stays provisional."""

    if not criteria or any(value is None for value in criteria.values()):
        return TERMINAL_STATUS_PROVISIONAL
    if all(value is True for value in criteria.values()):
        return TERMINAL_STATUS_SUPPORTED
    return TERMINAL_STATUS_UNSUPPORTED


def apply_terminal_charge_status(
    components: Mapping[str, Any],
    status: str,
    *,
    limitation: str = DIRECT_CONTACT_LIMITATION,
) -> dict[str, Any]:
    """Expose Qd/Qs only for a supported direct-contact candidate."""

    normalized_status = str(status).strip().lower()
    if normalized_status not in {
        TERMINAL_STATUS_SUPPORTED,
        TERMINAL_STATUS_PROVISIONAL,
        TERMINAL_STATUS_UNSUPPORTED,
    }:
        raise ValueError(f"unknown terminal charge status {status!r}.")
    qg = _first_present(components, ("Qg_contact_C", "Qg_C"))
    qd_contact = _first_present(
        components,
        ("Qd_contact_C", "Qd_C", "Qd_poisson_boundary_flux_C"),
    )
    qs_contact = _first_present(
        components,
        ("Qs_contact_C", "Qs_C", "Qs_poisson_boundary_flux_C"),
    )
    promoted = normalized_status == TERMINAL_STATUS_SUPPORTED
    return {
        **dict(components),
        "Qg_C": qg,
        "Qd_C": qd_contact if promoted else math.nan,
        "Qs_C": qs_contact if promoted else math.nan,
        "terminal_charge_status": normalized_status,
        "supported_quantities": "Qg,Qd,Qs" if promoted else "Qg",
        "limitation": "" if promoted else str(limitation),
    }


def _state_values(state: Mapping[str, Any]) -> tuple[int, str, float, float]:
    required = ("state_index", "state", "ntrap_cm3")
    missing = [name for name in required if name not in state]
    if missing:
        raise KeyError("state is missing: " + ", ".join(missing))
    ntrap = _nonnegative_float(state["ntrap_cm3"], "ntrap_cm3")
    if "nsheet_cm2" in state:
        nsheet = _nonnegative_float(state["nsheet_cm2"], "nsheet_cm2")
    else:
        import state_characterization_config as state_config

        nsheet = ntrap * float(state_config.CHARGE_TRAP_THICKNESS_CM)
    return int(state["state_index"]), str(state["state"]), ntrap, nsheet


def build_terminal_charge_full_row(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    VS_V: float,
    components: Mapping[str, Any],
    terminal_charge_status: str,
    mesh_level: str,
    converged: bool = True,
    error_message: str = "",
) -> dict[str, Any]:
    """Build one candidate charge row while retaining raw contact diagnostics."""

    index, name, ntrap, nsheet = _state_values(state)
    if converged:
        promoted = apply_terminal_charge_status(components, terminal_charge_status)
        residual = calculate_scaled_gauss_residual(promoted)
    else:
        status = str(terminal_charge_status).strip().lower()
        raw_qg = _candidate_float(
            components.get("Qg_contact_C", components.get("Qg_C"))
        )
        raw_qd = _candidate_float(
            components.get(
                "Qd_contact_C",
                components.get("Qd_poisson_boundary_flux_C", components.get("Qd_C")),
            )
        )
        raw_qs = _candidate_float(
            components.get(
                "Qs_contact_C",
                components.get("Qs_poisson_boundary_flux_C", components.get("Qs_C")),
            )
        )
        is_supported = status == TERMINAL_STATUS_SUPPORTED
        promoted = {
            **dict(components),
            "Qg_contact_C": raw_qg,
            "Qd_contact_C": raw_qd,
            "Qs_contact_C": raw_qs,
            "Qg_C": raw_qg,
            "Qd_C": raw_qd if is_supported else math.nan,
            "Qs_C": raw_qs if is_supported else math.nan,
            "Qmobile_C": _candidate_float(components.get("Qmobile_C")),
            "Qtrap_C": _candidate_float(components.get("Qtrap_C")),
            "Qfixed_C": _candidate_float(components.get("Qfixed_C")),
            "terminal_charge_status": status,
            "supported_quantities": "Qg,Qd,Qs" if is_supported else "Qg",
            "limitation": "" if is_supported else DIRECT_CONTACT_LIMITATION,
        }
        residual = {
            name: _candidate_float(components.get(name))
            for name in (
                "global_residual_C",
                "absolute_residual_C",
                "charge_scale_C",
                "relative_residual",
            )
        }
    row = {
        "state_index": index,
        "state": name,
        "ntrap_cm3": ntrap,
        "nsheet_cm2": nsheet,
        "VGS_V": _finite_float(VGS_V, "VGS_V"),
        "VDS_V": _finite_float(VDS_V, "VDS_V"),
        "VS_V": _finite_float(VS_V, "VS_V"),
        "Qg_C": promoted["Qg_C"],
        "Qd_C": promoted["Qd_C"],
        "Qs_C": promoted["Qs_C"],
        "Qg_contact_C": (
            _first_present(promoted, ("Qg_contact_C", "Qg_C"))
            if converged
            else _candidate_float(promoted.get("Qg_contact_C"))
        ),
        "Qd_contact_C": (
            _first_present(
                promoted,
                ("Qd_contact_C", "Qd_poisson_boundary_flux_C", "Qd_C"),
            )
            if converged
            else _candidate_float(promoted.get("Qd_contact_C"))
        ),
        "Qs_contact_C": (
            _first_present(
                promoted,
                ("Qs_contact_C", "Qs_poisson_boundary_flux_C", "Qs_C"),
            )
            if converged
            else _candidate_float(promoted.get("Qs_contact_C"))
        ),
        "Qmobile_C": (
            _first_present(promoted, ("Qmobile_C",))
            if converged
            else _candidate_float(promoted.get("Qmobile_C"))
        ),
        "Qtrap_C": (
            _first_present(promoted, ("Qtrap_C",))
            if converged
            else _candidate_float(promoted.get("Qtrap_C"))
        ),
        "Qfixed_C": (
            _first_present(promoted, ("Qfixed_C",))
            if converged
            else _candidate_float(promoted.get("Qfixed_C"))
        ),
        **residual,
        "method": DIRECT_CONTACT_METHOD,
        "supported_quantities": promoted["supported_quantities"],
        "terminal_charge_status": promoted["terminal_charge_status"],
        "limitation": promoted["limitation"],
        "mesh_level": str(mesh_level),
        "converged": bool(converged),
        "error_message": str(error_message),
    }
    if tuple(row) != TERMINAL_CHARGE_FULL_FIELDNAMES:
        raise RuntimeError("terminal charge candidate row does not match its schema.")
    return row


def build_capacitance_matrix_full_row(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    VS_V: float,
    measured_terminal: str,
    perturbed_terminal: str,
    capacitance_F: float,
    delta_voltage_V: float,
    terminal_charge_status: str,
    mesh_level: str,
    converged: bool = True,
    error_message: str = "",
) -> dict[str, Any]:
    """Build one Cij row and NaN unsupported Qd/Qs-derived matrix entries."""

    index, name, ntrap, nsheet = _state_values(state)
    measured = str(measured_terminal).strip().lower()
    perturbed = str(perturbed_terminal).strip().lower()
    if measured not in TERMINALS or perturbed not in TERMINALS:
        raise ValueError("measured and perturbed terminals must be gate/drain/source.")
    status = str(terminal_charge_status).strip().lower()
    if status not in {
        TERMINAL_STATUS_SUPPORTED,
        TERMINAL_STATUS_PROVISIONAL,
        TERMINAL_STATUS_UNSUPPORTED,
    }:
        raise ValueError(f"unknown terminal charge status {status!r}.")
    supported = status == TERMINAL_STATUS_SUPPORTED or measured == "gate"
    numeric_capacitance = (
        _finite_float(capacitance_F, "capacitance_F")
        if converged
        else _candidate_float(capacitance_F)
    )
    row = {
        "state_index": index,
        "state": name,
        "ntrap_cm3": ntrap,
        "nsheet_cm2": nsheet,
        "VGS_V": _finite_float(VGS_V, "VGS_V"),
        "VDS_V": _finite_float(VDS_V, "VDS_V"),
        "VS_V": _finite_float(VS_V, "VS_V"),
        "measured_terminal": measured,
        "perturbed_terminal": perturbed,
        "capacitance_F": numeric_capacitance if supported else math.nan,
        "raw_contact_flux_capacitance_F": numeric_capacitance,
        "delta_voltage_V": _finite_float(delta_voltage_V, "delta_voltage_V"),
        "method": DIRECT_CONTACT_METHOD,
        "supported": supported,
        "terminal_charge_status": status,
        "limitation": "" if supported else DIRECT_CONTACT_LIMITATION,
        "mesh_level": str(mesh_level),
        "converged": bool(converged),
        "error_message": str(error_message),
    }
    if tuple(row) != CAPACITANCE_MATRIX_FULL_FIELDNAMES:
        raise RuntimeError("capacitance candidate row does not match its schema.")
    return row


def _candidate_float(value: Any) -> float:
    """Return a finite CSV number, or a deliberate NaN for failed evidence."""

    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _stored_matrix(
    point: Mapping[str, Any], delta_voltage_V: float
) -> dict[tuple[str, str], float]:
    matrix_data = point.get("matrices", {}).get(f"{float(delta_voltage_V):.12g}", {})
    stored = matrix_data.get("values", {})
    return {
        (measured, perturbed): _candidate_float(
            stored.get(f"{measured}:{perturbed}")
        )
        for measured in TERMINALS
        for perturbed in TERMINALS
    }


def _comparison_passed_for_point(
    comparison: Mapping[str, Any], point: Mapping[str, Any]
) -> bool:
    key = (
        int(point["state_index"]),
        round(float(point["VGS_V"]), 12),
        round(float(point["VDS_V"]), 12),
        round(float(point["VS_V"]), 12),
    )
    rows = [
        row
        for row in comparison.get("rows", ())
        if (
            int(row["state_index"]),
            round(float(row["VGS_V"]), 12),
            round(float(row["VDS_V"]), 12),
            round(float(row["VS_V"]), 12),
        )
        == key
    ]
    return bool(rows) and all(bool(row.get("passed")) for row in rows)


def _candidate_limitation(acceptance: Mapping[str, Any]) -> str:
    status = str(acceptance.get("status", TERMINAL_STATUS_PROVISIONAL))
    if status == TERMINAL_STATUS_SUPPORTED:
        return ""
    failed = sorted(
        str(name)
        for name, passed in acceptance.get("criteria", {}).items()
        if not bool(passed)
    )
    suffix = f" Failed promotion criteria: {', '.join(failed)}." if failed else ""
    return DIRECT_CONTACT_LIMITATION + " " + DIRECT_CONTACT_COLUMN_SUM_LIMITATION + suffix


def build_candidate_output_rows(
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    nominal_delta_voltage_V: float,
    delta_voltages_V: Sequence[float],
) -> dict[str, list[dict[str, Any]]]:
    """Materialize the seven isolated candidate CSV tables.

    Raw direct-contact diagnostics are retained even when promotion fails.
    Public Qg-derived quantities remain numeric, while Qd/Qs and their matrix
    rows become NaN unless the aggregate status is ``supported``.
    """

    if "base" not in bundles:
        raise KeyError("candidate bundles must contain a base mesh result.")
    nominal_delta = _finite_float(
        nominal_delta_voltage_V, "nominal_delta_voltage_V"
    )
    deltas = tuple(
        _finite_float(value, "delta_voltages_V") for value in delta_voltages_V
    )
    if nominal_delta not in deltas:
        raise ValueError("nominal delta must be included in delta_voltages_V.")
    status = str(acceptance.get("status", TERMINAL_STATUS_PROVISIONAL)).lower()
    if status not in {
        TERMINAL_STATUS_SUPPORTED,
        TERMINAL_STATUS_PROVISIONAL,
        TERMINAL_STATUS_UNSUPPORTED,
    }:
        raise ValueError(f"invalid aggregate terminal status {status!r}.")
    limitation = _candidate_limitation(acceptance)
    base = bundles["base"]
    mesh_level = str(base.get("mesh_level", "base"))
    topology = base.get("topology", {})
    points = sorted(
        base.get("points", ()),
        key=lambda row: (
            int(row["state_index"]),
            float(row["VGS_V"]),
            float(row["VDS_V"]),
            float(row["VS_V"]),
        ),
    )
    output: dict[str, list[dict[str, Any]]] = {
        name: []
        for name in (
            "terminal",
            "matrix",
            "summary",
            "validation",
            "ward",
            "comparison",
            "mesh",
        )
    }
    reproducibility = acceptance.get("reproducibility", {})
    mesh_comparison = acceptance.get("mesh", {})

    for point in points:
        state = {
            "state_index": point["state_index"],
            "state": point["state"],
            "ntrap_cm3": point["ntrap_cm3"],
        }
        _, _, ntrap_cm3, nsheet_cm2 = _state_values(state)
        direct = dict(point.get("direct", {}))
        point_error = str(point.get("error_message", ""))
        converged = all(
            math.isfinite(_candidate_float(direct.get(name)))
            for name in _GAUSS_COMPONENT_ALIASES
        )
        terminal_row = build_terminal_charge_full_row(
            state,
            VGS_V=float(point["VGS_V"]),
            VDS_V=float(point["VDS_V"]),
            VS_V=float(point["VS_V"]),
            components=direct,
            terminal_charge_status=status,
            mesh_level=mesh_level,
            converged=converged,
            error_message=point_error,
        )
        if status != TERMINAL_STATUS_SUPPORTED:
            terminal_row["limitation"] = limitation
        output["terminal"].append(terminal_row)

        for delta in deltas:
            delta_matrix = _stored_matrix(point, delta)
            for measured in TERMINALS:
                for perturbed in TERMINALS:
                    matrix_row = build_capacitance_matrix_full_row(
                        state,
                        VGS_V=float(point["VGS_V"]),
                        VDS_V=float(point["VDS_V"]),
                        VS_V=float(point["VS_V"]),
                        measured_terminal=measured,
                        perturbed_terminal=perturbed,
                        capacitance_F=delta_matrix[(measured, perturbed)],
                        delta_voltage_V=delta,
                        terminal_charge_status=status,
                        mesh_level=mesh_level,
                        converged=math.isfinite(
                            delta_matrix[(measured, perturbed)]
                        ),
                        error_message=point_error,
                    )
                    if not bool(matrix_row["supported"]):
                        matrix_row["limitation"] = limitation
                    output["matrix"].append(matrix_row)

        raw_matrix = _stored_matrix(point, nominal_delta)

        nominal_data = point.get("matrices", {}).get(
            f"{nominal_delta:.12g}", {}
        )
        nominal_validation = nominal_data.get("sum_validation", {})
        raw_sensitivity_values = [
            _candidate_float(item.get("value"))
            for item in point.get("sensitivities", {}).values()
        ]
        finite_sensitivity_values = [
            value for value in raw_sensitivity_values if math.isfinite(value)
        ]
        promoted_matrix = {
            key: (
                value
                if status == TERMINAL_STATUS_SUPPORTED or key[0] == "gate"
                else math.nan
            )
            for key, value in raw_matrix.items()
        }
        summary_row = {
            "state_index": int(point["state_index"]),
            "state": str(point["state"]),
            "ntrap_cm3": ntrap_cm3,
            "nsheet_cm2": nsheet_cm2,
            "VGS_V": float(point["VGS_V"]),
            "VDS_V": float(point["VDS_V"]),
            "VS_V": float(point["VS_V"]),
            "Cgg_F": promoted_matrix[("gate", "gate")],
            "Cgd_F": promoted_matrix[("gate", "drain")],
            "Cgs_F": promoted_matrix[("gate", "source")],
            "Cdg_F": promoted_matrix[("drain", "gate")],
            "Cdd_F": promoted_matrix[("drain", "drain")],
            "Cds_F": promoted_matrix[("drain", "source")],
            "Csg_F": promoted_matrix[("source", "gate")],
            "Csd_F": promoted_matrix[("source", "drain")],
            "Css_F": promoted_matrix[("source", "source")],
            "nominal_delta_voltage_V": nominal_delta,
            "sensitivity_delta_voltages_V": ";".join(
                f"{value:.12g}" for value in deltas
            ),
            "maximum_delta_relative_sensitivity": (
                max(finite_sensitivity_values)
                if finite_sensitivity_values
                else math.nan
            ),
            "maximum_absolute_row_sum_F": _candidate_float(
                nominal_validation.get("maximum_absolute_row_sum_F")
            ),
            "maximum_absolute_column_sum_F": _candidate_float(
                nominal_validation.get("maximum_absolute_column_sum_F")
            ),
            "row_sums_passed": bool(nominal_validation.get("row_sums_passed")),
            "column_sums_passed": bool(
                nominal_validation.get("column_sums_passed")
            ),
            "method": DIRECT_CONTACT_METHOD,
            "terminal_charge_status": status,
            "limitation": limitation,
            "mesh_level": mesh_level,
            "converged": all(math.isfinite(value) for value in raw_matrix.values()),
            "error_message": point_error,
        }
        if tuple(summary_row) != CAPACITANCE_SUMMARY_FULL_FIELDNAMES:
            raise RuntimeError("capacitance summary row does not match its schema.")
        output["summary"].append(summary_row)

        direct_values = [
            _candidate_float(direct.get(name)) for name in _GAUSS_COMPONENT_ALIASES
        ]
        matrix_validations = [
            data.get("sum_validation", {})
            for data in point.get("matrices", {}).values()
        ]
        validation_row = {
            "state_index": int(point["state_index"]),
            "state": str(point["state"]),
            "VGS_V": float(point["VGS_V"]),
            "VDS_V": float(point["VDS_V"]),
            "VS_V": float(point["VS_V"]),
            "mesh_level": mesh_level,
            "all_quantities_finite": all(
                math.isfinite(value) for value in direct_values
            ),
            "contacts_complete": bool(topology.get("contacts_complete")),
            "contact_topology_passed": bool(topology.get("passed")),
            "contact_sign_passed": bool(topology.get("sign_consistent")),
            "global_gauss_passed": bool(point.get("residual_passed")),
            "repeatability_passed": _comparison_passed_for_point(
                reproducibility, point
            ),
            "mesh_refinement_passed": _comparison_passed_for_point(
                mesh_comparison, point
            ),
            "continuity_passed": bool(
                acceptance.get("continuity", {}).get("passed")
            ),
            "gauge_invariance_passed": bool(
                point.get("gauge", {}).get("passed")
            ),
            "matrix_row_sums_passed": bool(matrix_validations)
            and all(bool(item.get("row_sums_passed")) for item in matrix_validations),
            "matrix_column_sums_passed": bool(matrix_validations)
            and all(
                bool(item.get("column_sums_passed"))
                for item in matrix_validations
            ),
            "global_residual_C": _candidate_float(
                direct.get("global_residual_C")
            ),
            "absolute_residual_C": _candidate_float(
                direct.get("absolute_residual_C")
            ),
            "charge_scale_C": _candidate_float(direct.get("charge_scale_C")),
            "relative_residual": _candidate_float(
                direct.get("relative_residual")
            ),
            "terminal_charge_status": status,
            "limitation": limitation,
            "error_message": point_error,
        }
        if tuple(validation_row) != DIRECT_CONTACT_VALIDATION_FIELDNAMES:
            raise RuntimeError("direct-contact validation row schema mismatch.")
        output["validation"].append(validation_row)

        ward = point.get("ward", {})
        ward_row = {
            "state_index": int(point["state_index"]),
            "state": str(point["state"]),
            "VGS_V": float(point["VGS_V"]),
            "VDS_V": float(point["VDS_V"]),
            "VS_V": float(point["VS_V"]),
            "Qmobile_channel_C": _candidate_float(
                ward.get("Qmobile_channel_C")
            ),
            "Qs_mobile_C": _candidate_float(ward.get("Qs_mobile_C")),
            "Qd_mobile_C": _candidate_float(ward.get("Qd_mobile_C")),
            "partition_residual_C": _candidate_float(
                ward.get("partition_residual_C")
            ),
            "absolute_partition_residual_C": _candidate_float(
                ward.get("absolute_partition_residual_C")
            ),
            "relative_partition_residual": _candidate_float(
                ward.get("relative_partition_residual")
            ),
            "minimum_source_weight": _candidate_float(
                ward.get("minimum_source_weight")
            ),
            "maximum_source_weight": _candidate_float(
                ward.get("maximum_source_weight")
            ),
            "minimum_drain_weight": _candidate_float(
                ward.get("minimum_drain_weight")
            ),
            "maximum_drain_weight": _candidate_float(
                ward.get("maximum_drain_weight")
            ),
            "method": WARD_DUTTON_METHOD,
            "terminal_charge_status": WARD_DUTTON_STATUS_MOBILE_ONLY,
            "limitation": WARD_DUTTON_LIMITATION,
            "mesh_level": mesh_level,
            "converged": bool(ward.get("converged")),
            "error_message": str(ward.get("error_message", "")),
        }
        if tuple(ward_row) != WARD_DUTTON_FIELDNAMES:
            raise RuntimeError("Ward-Dutton row schema mismatch.")
        output["ward"].append(ward_row)

        comparison_row = {
            "state_index": int(point["state_index"]),
            "state": str(point["state"]),
            "VGS_V": float(point["VGS_V"]),
            "VDS_V": float(point["VDS_V"]),
            "VS_V": float(point["VS_V"]),
            "Qg_direct_contact_C": _candidate_float(direct.get("Qg_contact_C")),
            "Qd_direct_contact_C": _candidate_float(direct.get("Qd_contact_C")),
            "Qs_direct_contact_C": _candidate_float(direct.get("Qs_contact_C")),
            "Qmobile_channel_C": _candidate_float(
                ward.get("Qmobile_channel_C")
            ),
            "Qd_ward_dutton_mobile_C": _candidate_float(
                ward.get("Qd_mobile_C")
            ),
            "Qs_ward_dutton_mobile_C": _candidate_float(
                ward.get("Qs_mobile_C")
            ),
            "direct_method": DIRECT_CONTACT_METHOD,
            "ward_dutton_method": WARD_DUTTON_METHOD,
            "comparison_scope": (
                "Different physical scopes: full electrostatic electrode-flux "
                "candidate versus mobile-channel partition; not interchangeable."
            ),
            "mesh_level": mesh_level,
        }
        if tuple(comparison_row) != METHOD_COMPARISON_FIELDNAMES:
            raise RuntimeError("method-comparison row schema mismatch.")
        output["comparison"].append(comparison_row)

    point_names = {
        (
            int(point["state_index"]),
            round(float(point["VGS_V"]), 12),
            round(float(point["VDS_V"]), 12),
            round(float(point["VS_V"]), 12),
        ): str(point["state"])
        for point in points
    }
    for source in mesh_comparison.get("rows", ()):
        base_value = _candidate_float(source.get("reference_value"))
        fine_value = _candidate_float(source.get("candidate_value"))
        scale = max(abs(base_value), abs(fine_value))
        comparison_floor = 1.0e-28
        tolerance = _candidate_float(source.get("tolerance"))
        relative_tolerance = (
            max(0.0, tolerance - comparison_floor) / scale
            if scale > 0.0 and math.isfinite(tolerance)
            else 0.0
        )
        key = (
            int(source["state_index"]),
            round(float(source["VGS_V"]), 12),
            round(float(source["VDS_V"]), 12),
            round(float(source["VS_V"]), 12),
        )
        mesh_row = {
            "state_index": key[0],
            "state": point_names.get(key, ""),
            "VGS_V": float(source["VGS_V"]),
            "VDS_V": float(source["VDS_V"]),
            "VS_V": float(source["VS_V"]),
            "comparison": str(source.get("comparison", "base_vs_fine")),
            "reference_mesh_level": "base",
            "candidate_mesh_level": "fine",
            "quantity": str(source["quantity"]),
            "reference_value": base_value,
            "candidate_value": fine_value,
            "absolute_difference": _candidate_float(
                source.get("absolute_difference")
            ),
            "relative_difference": _candidate_float(
                source.get("relative_difference")
            ),
            "comparison_floor": comparison_floor,
            "relative_tolerance": relative_tolerance,
            "passed": bool(source.get("passed")),
        }
        if tuple(mesh_row) != MESH_REFINEMENT_QC_FIELDNAMES:
            raise RuntimeError("mesh-refinement row schema mismatch.")
        output["mesh"].append(mesh_row)
    return output


def build_candidate_readme(
    *,
    bundles: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    constants: Mapping[str, Any],
) -> str:
    """Build the deterministic candidate-method and limitation report."""

    status = str(acceptance.get("status", TERMINAL_STATUS_PROVISIONAL))
    base = bundles.get("base", {})
    topology = base.get("topology", {})
    lines = [
        "# Full-terminal Q/C candidate",
        "",
        f"Aggregate direct-contact status: `{status}`",
        "",
        "This isolated candidate does not replace any established shared_data "
        "CSV or public Qg/Cgg/Cgd/Cgs API.",
        "",
        "## Definitions",
        "",
        "- Direct contact: DEVSIM PotentialEquation contact charge with "
        "PotentialEdgeFlux and CylindricalEdgeCouple.",
        "- Internal charge: signed MoS2 mobile charge, ChargeTrap trapped "
        "charge, and MoS2 fixed doping integrated with CylindricalNodeVolume.",
        "- Gauss residual: Qg_contact + Qd_contact + Qs_contact + Qmobile + "
        "Qtrap + Qfixed, scaled by the maximum absolute component and a floor.",
        "- Capacitance sign: Cij = dQi/dVj; no SPICE sign conversion and no "
        "reciprocity constraint are imposed.",
        "- Ward-Dutton: linear runtime-z partition of signed MoS2 mobile charge "
        "only. It is not a full-electrode terminal-charge replacement.",
        "",
        "## Promotion criteria",
        "",
    ]
    for name, passed in sorted(acceptance.get("criteria", {}).items()):
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(["", "## Runtime contact topology", ""])
    for terminal in TERMINALS:
        report = topology.get("contacts", {}).get(terminal, {})
        lines.append(
            f"- {terminal}: nodes={report.get('node_count', 0)}, "
            f"elements={report.get('edge_count', 0)}, "
            f"normal=({report.get('normal_r', math.nan)}, "
            f"{report.get('normal_z', math.nan)}), "
            f"CylindricalSurfaceArea={report.get('surface_area_cm2', math.nan)} "
            f"cm^2, passed={bool(report.get('passed'))}"
        )
    lines.extend(["", "## Numerical settings", ""])
    for name, value in sorted(constants.items()):
        lines.append(f"- {name}: {value}")
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            f"- {DIRECT_CONTACT_LIMITATION}",
            f"- {DIRECT_CONTACT_COLUMN_SUM_LIMITATION}",
            f"- {WARD_DUTTON_LIMITATION}",
            "- When promotion is not supported, Qd, Qs, and capacitance rows "
            "measured at drain/source are NaN. Raw contact-flux diagnostics "
            "remain in the candidate validation data; Qg and the gate-measured "
            "Cgg/Cgd/Cgs values remain available.",
            "- No 50:50 partition, planar-area multiplier, current-times-time "
            "charge, fitted scaling factor, Schottky/contact-resistance model, "
            "ERASE simulation, or retention simulation is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_deterministic_csv(
    output_path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    sort_key: Callable[[Mapping[str, Any]], Any] | None = None,
) -> Path:
    """Write strict, UTF-8, LF-only CSV with caller-controlled row ordering."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_rows = list(rows)
    if sort_key is not None:
        ordered_rows.sort(key=sort_key)
    expected = tuple(str(name) for name in fieldnames)
    for index, row in enumerate(ordered_rows):
        if set(row) != set(expected):
            missing = set(expected).difference(row)
            extra = set(row).difference(expected)
            raise ValueError(
                f"CSV row {index} schema mismatch: "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}."
            )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=expected,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in ordered_rows:
            writer.writerow({name: row[name] for name in expected})
    return path


__all__ = (
    "DEVICE_NAME",
    "SEMICONDUCTOR_REGION",
    "CHARGE_TRAP_REGION",
    "POTENTIAL_EQUATION",
    "TERMINALS",
    "EXPECTED_CONTACT_REGIONS",
    "CONTACT_INWARD_NORMALS_RZ",
    "CONTACT_REGION_OUTWARD_NORMALS_RZ",
    "DIRECT_CONTACT_METHOD",
    "WARD_DUTTON_METHOD",
    "TERMINAL_STATUS_SUPPORTED",
    "TERMINAL_STATUS_PROVISIONAL",
    "TERMINAL_STATUS_UNSUPPORTED",
    "WARD_DUTTON_STATUS_MOBILE_ONLY",
    "DIRECT_CONTACT_LIMITATION",
    "DIRECT_CONTACT_COLUMN_SUM_LIMITATION",
    "WARD_DUTTON_LIMITATION",
    "TERMINAL_CHARGE_FULL_FIELDNAMES",
    "CAPACITANCE_MATRIX_FULL_FIELDNAMES",
    "CAPACITANCE_SUMMARY_FULL_FIELDNAMES",
    "DIRECT_CONTACT_VALIDATION_FIELDNAMES",
    "WARD_DUTTON_FIELDNAMES",
    "METHOD_COMPARISON_FIELDNAMES",
    "MESH_REFINEMENT_QC_FIELDNAMES",
    "calculate_scaled_gauss_residual",
    "extract_direct_contact_charge_components",
    "audit_runtime_contact_topology",
    "calculate_ward_dutton_mobile_partition",
    "extract_ward_dutton_mobile_partition",
    "vector_central_difference",
    "calculate_matrix_sums",
    "evaluate_matrix_sum_tolerances",
    "calculate_delta_sensitivity",
    "determine_terminal_charge_status",
    "apply_terminal_charge_status",
    "build_terminal_charge_full_row",
    "build_capacitance_matrix_full_row",
    "build_candidate_output_rows",
    "build_candidate_readme",
    "write_deterministic_csv",
)
