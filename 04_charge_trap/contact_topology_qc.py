"""Pure validation helpers for physical source/drain contact-flux QC.

This module deliberately contains no device construction, bias orchestration,
or output-path policy.  Runtime inspection is performed through an injected
DEVSIM-compatible API, while all numerical checks are pure functions.  The
capacitance convention is ``Cij = dQi/dVj`` without a SPICE sign conversion.

Two distinctions are intentional:

* A contact-only capacitance column sum is diagnostic, not an acceptance
  condition.  Bias-dependent mobile charge generally makes that sum nonzero.
* Non-contact exterior boundaries have no separately assembled charge term in
  the present model.  Their contribution is therefore declared zero by the
  homogeneous natural boundary condition; an unexplained Gauss residual is
  never back-filled as a synthetic boundary charge.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from typing import Any


DEVICE_NAME = "MoS2_GAA"
TERMINALS = ("gate", "drain", "source")
CONTACT_CHARGE_NAMES = {
    "gate": "Qg_contact_C",
    "drain": "Qd_contact_C",
    "source": "Qs_contact_C",
}
EXPECTED_CONTACT_REGIONS = {
    "gate": "BlockingOxide",
    "drain": "MoS2",
    "source": "MoS2",
}

PHYSICAL_CONTACT_FLUX_METHOD = "physical_electrode_contact_flux"
WARD_DUTTON_METHOD = "ward_dutton_mobile_channel"
WARD_DUTTON_STATUS = "mobile_partition_only"
NATURAL_NONCONTACT_METHOD = "model_declared_homogeneous_natural_neumann"

STATUS_SUPPORTED = "supported"
STATUS_PROVISIONAL = "provisional"
STATUS_UNSUPPORTED = "unsupported"

DEFAULT_AREA_RELATIVE_TOLERANCE = 1.0e-6
DEFAULT_COORDINATE_ABSOLUTE_TOLERANCE_CM = 1.0e-12
DEFAULT_NORMAL_ABSOLUTE_TOLERANCE = 1.0e-8
DEFAULT_DERIVATIVE_ABSOLUTE_TOLERANCE_F = 1.0e-24
DEFAULT_DERIVATIVE_RELATIVE_TOLERANCE = 1.0e-6
DEFAULT_CAPACITANCE_FLOOR_F = 1.0e-30
DEFAULT_CHARGE_FLOOR_C = 1.0e-30

WARD_DUTTON_LIMITATION = (
    "Ward-Dutton values partition signed MoS2 mobile-channel charge only. "
    "They exclude trap, fixed-dopant, dielectric, and electrode charge and "
    "are not full physical electrode charges."
)

REQUIRED_SUPPORT_CRITERIA = (
    "contact_topology",
    "all_reference_points_finite",
    "repeat_reproducibility",
    "mesh_convergence",
    "global_gauss_balance",
    "derivative_gauss_balance",
    "gauge_invariance",
    "dc_regression_acceptable",
)

_PROHIBITED_COLUMN_SUM_CRITERIA = frozenset(
    {
        "contact_column_sum_zero",
        "contact_column_sums_zero",
        "matrix_column_sums",
        "matrix_row_column_sums",
        "zero_capacitance_column_sum",
    }
)

_CHARGE_ALIASES = {
    "Qg_contact_C": ("Qg_contact_C", "Qg_C"),
    "Qd_contact_C": (
        "Qd_contact_C",
        "Qd_poisson_boundary_flux_C",
        "Qd_C",
    ),
    "Qs_contact_C": (
        "Qs_contact_C",
        "Qs_poisson_boundary_flux_C",
        "Qs_C",
    ),
    "Qmobile_C": ("Qmobile_C",),
    "Qtrap_C": ("Qtrap_C",),
    "Qfixed_C": ("Qfixed_C",),
}


CONTACT_TOPOLOGY_FIELDNAMES = (
    "mesh_level",
    "contact",
    "region",
    "edge_count",
    "node_count",
    "plane_axis",
    "plane_coordinate_cm",
    "expected_plane_cm",
    "measured_plane_min_cm",
    "measured_plane_max_cm",
    "plane_max_error_cm",
    "radial_min_cm",
    "radial_max_cm",
    "expected_radial_min_cm",
    "expected_radial_max_cm",
    "axial_min_cm",
    "axial_max_cm",
    "expected_axial_min_cm",
    "expected_axial_max_cm",
    "edge_lengths_cm",
    "edge_outward_normals_rz",
    "maximum_edge_outward_normal_error",
    "geometric_area_cm2",
    "geometric_swept_area_cm2",
    "runtime_area_cm2",
    "cylindrical_surface_area_cm2",
    "expected_area_cm2",
    "analytic_area_cm2",
    "area_relative_error",
    "geometric_area_relative_error",
    "surface_model_relative_error",
    "normal_r",
    "normal_z",
    "normal_method",
    "runtime_normal_r",
    "runtime_normal_z",
    "runtime_normal_matches_geometry",
    "expected_normal_r",
    "expected_normal_z",
    "maximum_node_normal_error",
    "region_consistent",
    "plane_consistent",
    "span_consistent",
    "geometric_area_consistent",
    "surface_model_consistent",
    "normal_consistent",
    "potential_equation_consistent",
    "current_equation_consistent",
    "axisymmetric_models_consistent",
    "plane_passed",
    "radial_span_passed",
    "normal_passed",
    "area_passed",
    "equation_passed",
    "passed",
    "error_message",
)

DERIVATIVE_GAUSS_BALANCE_FIELDNAMES = (
    "state_index",
    "state",
    "mesh_level",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "perturbed_terminal",
    "delta_voltage_V",
    "dQg_contact_dV_F",
    "dQd_contact_dV_F",
    "dQs_contact_dV_F",
    "contact_derivative_sum_F",
    "dQmobile_dV_F",
    "dQtrap_dV_F",
    "dQfixed_dV_F",
    "dQnoncontact_boundary_dV_F",
    "internal_derivative_sum_F",
    "derivative_gauss_residual_F",
    "endpoint_residual_derivative_F",
    "derivative_gauss_absolute_error_F",
    "derivative_gauss_scale_F",
    "derivative_gauss_tolerance_F",
    "derivative_gauss_relative_error",
    "noncontact_boundary_method",
    "passed",
    "error_message",
)

GAUGE_INVARIANCE_FIELDNAMES = (
    "state_index",
    "state",
    "mesh_level",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "measured_terminal",
    "common_mode_delta_voltage_V",
    "matrix_row_sum_F",
    "direct_common_mode_derivative_F",
    "row_direct_difference_F",
    "row_scale_F",
    "row_tolerance_F",
    "direct_tolerance_F",
    "agreement_tolerance_F",
    "row_sum_passed",
    "direct_common_mode_passed",
    "row_direct_agreement_passed",
    "passed",
    "error_message",
)

NUMERIC_COMPARISON_FIELDNAMES = (
    "comparison",
    "quantity",
    "reference_value",
    "candidate_value",
    "absolute_difference",
    "scale",
    "absolute_tolerance",
    "relative_tolerance",
    "combined_tolerance",
    "scaled_relative_difference",
    "passed",
    "error_message",
)

WARD_DUTTON_FIELDNAMES = (
    "state_index",
    "state",
    "mesh_level",
    "VGS_V",
    "VDS_V",
    "VS_V",
    "Qs_mobile_C",
    "Qd_mobile_C",
    "Qmobile_channel_C",
    "partition_residual_C",
    "absolute_partition_residual_C",
    "partition_scale_C",
    "partition_tolerance_C",
    "relative_partition_residual",
    "method",
    "status",
    "limitation",
    "passed",
    "error_message",
)


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = _finite_float(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _single_line_error(error: object | None) -> str:
    if error is None:
        return ""
    message = " ".join(str(error).split())
    if isinstance(error, BaseException):
        name = type(error).__name__
        return f"{name}: {message}" if message else name
    return message


def analytic_annular_area_cm2(
    inner_radius_cm: float,
    outer_radius_cm: float,
) -> float:
    """Return ``pi*(r_outer**2-r_inner**2)`` after strict validation."""

    inner = _nonnegative_float(inner_radius_cm, "inner_radius_cm")
    outer = _positive_float(outer_radius_cm, "outer_radius_cm")
    if not outer > inner:
        raise ValueError("outer_radius_cm must exceed inner_radius_cm")
    return math.pi * (outer * outer - inner * inner)


def _normalize_contact_edges(raw_elements: Any) -> tuple[tuple[int, int], ...]:
    values = tuple(raw_elements)
    if not values:
        return ()
    if all(isinstance(value, (int, float)) for value in values):
        if len(values) % 2:
            raise ValueError("flat 2-D contact element list must contain pairs")
        values = tuple(values[index : index + 2] for index in range(0, len(values), 2))

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for index, element in enumerate(values):
        nodes = tuple(int(node) for node in element)
        if len(nodes) != 2 or nodes[0] == nodes[1] or min(nodes) < 0:
            raise ValueError(
                f"contact element {index} must contain two distinct nonnegative nodes"
            )
        canonical = tuple(sorted(nodes))
        if canonical not in seen:
            edges.append((nodes[0], nodes[1]))
            seen.add(canonical)
    return tuple(edges)


def _cylindrical_swept_edge_area_cm2(
    edges: Sequence[tuple[int, int]],
    radial_coordinates_cm: Sequence[float],
    axial_coordinates_cm: Sequence[float],
) -> float:
    areas: list[float] = []
    for node0, node1 in edges:
        try:
            r0 = _nonnegative_float(radial_coordinates_cm[node0], f"x[{node0}]")
            r1 = _nonnegative_float(radial_coordinates_cm[node1], f"x[{node1}]")
            z0 = _finite_float(axial_coordinates_cm[node0], f"y[{node0}]")
            z1 = _finite_float(axial_coordinates_cm[node1], f"y[{node1}]")
        except IndexError as error:
            raise ValueError("contact node index is outside the region arrays") from error
        meridional_length_cm = math.hypot(r1 - r0, z1 - z0)
        if meridional_length_cm <= 0.0:
            raise ValueError("contact edge has zero meridional length")
        areas.append(math.pi * (r0 + r1) * meridional_length_cm)
    return math.fsum(areas)


def _relative_error(actual: float, expected: float) -> float:
    expected_value = _positive_float(abs(expected), "expected magnitude")
    return abs(_finite_float(actual, "actual") - expected) / expected_value


def _expected_contact_geometry(geometry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    required = ("r_core", "r_mos2", "r_block", "z_source", "z_drain")
    missing = [name for name in required if name not in geometry]
    if missing:
        raise KeyError("geometry is missing: " + ", ".join(missing))
    values = {name: _finite_float(geometry[name], name) for name in required}
    if not 0.0 <= values["r_core"] < values["r_mos2"] < values["r_block"]:
        raise ValueError("geometry must satisfy 0 <= r_core < r_mos2 < r_block")
    if values["z_source"] == values["z_drain"]:
        raise ValueError("source and drain planes must differ")

    axial_direction = math.copysign(1.0, values["z_drain"] - values["z_source"])
    radial_direction = math.copysign(1.0, values["r_block"] - values["r_core"])
    z_min = min(values["z_source"], values["z_drain"])
    z_max = max(values["z_source"], values["z_drain"])
    annular_area = analytic_annular_area_cm2(values["r_core"], values["r_mos2"])
    gate_area = 2.0 * math.pi * values["r_block"] * (z_max - z_min)
    return {
        "source": {
            "region": EXPECTED_CONTACT_REGIONS["source"],
            "plane_axis": "y",
            "plane_cm": values["z_source"],
            "radial_span_cm": (values["r_core"], values["r_mos2"]),
            "axial_span_cm": (values["z_source"], values["z_source"]),
            "outward_normal_rz": (0.0, -axial_direction),
            "area_cm2": annular_area,
        },
        "drain": {
            "region": EXPECTED_CONTACT_REGIONS["drain"],
            "plane_axis": "y",
            "plane_cm": values["z_drain"],
            "radial_span_cm": (values["r_core"], values["r_mos2"]),
            "axial_span_cm": (values["z_drain"], values["z_drain"]),
            "outward_normal_rz": (0.0, axial_direction),
            "area_cm2": annular_area,
        },
        "gate": {
            "region": EXPECTED_CONTACT_REGIONS["gate"],
            "plane_axis": "x",
            "plane_cm": values["r_block"],
            "radial_span_cm": (values["r_block"], values["r_block"]),
            "axial_span_cm": (z_min, z_max),
            "outward_normal_rz": (radial_direction, 0.0),
            "area_cm2": gate_area,
        },
    }


def _all_close_span(
    actual_min: float,
    actual_max: float,
    expected_span: tuple[float, float],
    tolerance_cm: float,
) -> bool:
    return math.isclose(actual_min, expected_span[0], rel_tol=0.0, abs_tol=tolerance_cm) and math.isclose(
        actual_max, expected_span[1], rel_tol=0.0, abs_tol=tolerance_cm
    )


def _failed_contact_report(contact: str, expected: Mapping[str, Any], error: object) -> dict[str, Any]:
    normal_r, normal_z = expected["outward_normal_rz"]
    radial_min, radial_max = expected["radial_span_cm"]
    axial_min, axial_max = expected["axial_span_cm"]
    return {
        "contact": contact,
        "region": "",
        "edge_count": 0,
        "node_count": 0,
        "plane_axis": expected["plane_axis"],
        "plane_coordinate_cm": math.nan,
        "expected_plane_cm": expected["plane_cm"],
        "measured_plane_min_cm": math.nan,
        "measured_plane_max_cm": math.nan,
        "plane_max_error_cm": math.nan,
        "radial_min_cm": math.nan,
        "radial_max_cm": math.nan,
        "expected_radial_min_cm": radial_min,
        "expected_radial_max_cm": radial_max,
        "axial_min_cm": math.nan,
        "axial_max_cm": math.nan,
        "expected_axial_min_cm": axial_min,
        "expected_axial_max_cm": axial_max,
        "edge_lengths_cm": (),
        "edge_outward_normals_rz": (),
        "maximum_edge_outward_normal_error": math.nan,
        "geometric_area_cm2": math.nan,
        "geometric_swept_area_cm2": math.nan,
        "runtime_area_cm2": math.nan,
        "cylindrical_surface_area_cm2": math.nan,
        "expected_area_cm2": expected["area_cm2"],
        "analytic_area_cm2": expected["area_cm2"],
        "area_relative_error": math.nan,
        "geometric_area_relative_error": math.nan,
        "surface_model_relative_error": math.nan,
        "normal_r": normal_r,
        "normal_z": normal_z,
        "normal_method": "geometry_boundary_outward",
        "runtime_normal_r": math.nan,
        "runtime_normal_z": math.nan,
        "runtime_normal_matches_geometry": False,
        "expected_normal_r": normal_r,
        "expected_normal_z": normal_z,
        "maximum_node_normal_error": math.nan,
        "region_consistent": False,
        "plane_consistent": False,
        "span_consistent": False,
        "geometric_area_consistent": False,
        "surface_model_consistent": False,
        "normal_consistent": False,
        "potential_equation_consistent": False,
        "current_equation_consistent": False,
        "axisymmetric_models_consistent": False,
        "plane_passed": False,
        "radial_span_passed": False,
        "normal_passed": False,
        "area_passed": False,
        "equation_passed": False,
        "passed": False,
        "error_message": _single_line_error(error),
    }


def _normalize_region_triangles(raw_elements: Any) -> tuple[tuple[int, int, int], ...]:
    values = tuple(raw_elements)
    if not values:
        return ()
    if all(isinstance(value, (int, float)) for value in values):
        if len(values) % 3:
            raise ValueError("flat 2-D region element list must contain triples")
        values = tuple(values[index : index + 3] for index in range(0, len(values), 3))
    triangles: list[tuple[int, int, int]] = []
    for index, element in enumerate(values):
        nodes = tuple(int(node) for node in element)
        if len(nodes) != 3 or len(set(nodes)) != 3 or min(nodes) < 0:
            raise ValueError(
                f"region element {index} must contain three distinct nonnegative nodes"
            )
        triangles.append(nodes)
    return tuple(triangles)


def _geometry_outward_contact_normal(
    edges: Sequence[tuple[int, int]],
    region_triangles: Sequence[tuple[int, int, int]],
    radial_coordinates_cm: Sequence[float],
    axial_coordinates_cm: Sequence[float],
    expected_normal_rz: tuple[float, float],
) -> dict[str, Any]:
    """Derive an outward normal from each boundary edge and its incident cell."""

    expected_r, expected_z = expected_normal_rz
    weighted_r: list[float] = []
    weighted_z: list[float] = []
    weights: list[float] = []
    edge_normals: list[tuple[float, float]] = []
    edge_errors: list[float] = []
    for edge_index, (node0, node1) in enumerate(edges):
        edge_nodes = {node0, node1}
        incident = [
            triangle for triangle in region_triangles if edge_nodes.issubset(triangle)
        ]
        if len(incident) != 1:
            raise RuntimeError(
                f"contact edge {edge_index} must have exactly one incident region triangle; "
                f"found {len(incident)}"
            )
        triangle = incident[0]
        third_node = next(node for node in triangle if node not in edge_nodes)
        try:
            r0 = radial_coordinates_cm[node0]
            z0 = axial_coordinates_cm[node0]
            r1 = radial_coordinates_cm[node1]
            z1 = axial_coordinates_cm[node1]
            r2 = radial_coordinates_cm[third_node]
            z2 = axial_coordinates_cm[third_node]
        except IndexError as error:
            raise RuntimeError("region triangle node is outside coordinate arrays") from error
        tangent_r = r1 - r0
        tangent_z = z1 - z0
        length = math.hypot(tangent_r, tangent_z)
        if length <= 0.0:
            raise RuntimeError(f"contact edge {edge_index} has zero length")
        candidate0 = (tangent_z / length, -tangent_r / length)
        candidate1 = (-candidate0[0], -candidate0[1])
        midpoint_r = 0.5 * (r0 + r1)
        midpoint_z = 0.5 * (z0 + z1)
        interior_vector = (r2 - midpoint_r, z2 - midpoint_z)
        if math.hypot(*interior_vector) <= 0.0:
            raise RuntimeError(f"contact edge {edge_index} has ambiguous interior")
        dot0 = candidate0[0] * interior_vector[0] + candidate0[1] * interior_vector[1]
        dot1 = candidate1[0] * interior_vector[0] + candidate1[1] * interior_vector[1]
        if math.isclose(dot0, dot1, rel_tol=0.0, abs_tol=1.0e-30):
            raise RuntimeError(f"contact edge {edge_index} outward direction is ambiguous")
        outward_r, outward_z = candidate0 if dot0 < dot1 else candidate1
        if outward_r * interior_vector[0] + outward_z * interior_vector[1] >= 0.0:
            raise RuntimeError(f"contact edge {edge_index} normal does not point outward")
        swept_area = math.pi * (r0 + r1) * length
        if swept_area <= 0.0:
            raise RuntimeError(f"contact edge {edge_index} has nonpositive swept area")
        edge_normals.append((outward_r, outward_z))
        edge_errors.append(math.hypot(outward_r - expected_r, outward_z - expected_z))
        weights.append(swept_area)
        weighted_r.append(swept_area * outward_r)
        weighted_z.append(swept_area * outward_z)
    total_weight = math.fsum(weights)
    average_r = math.fsum(weighted_r) / total_weight
    average_z = math.fsum(weighted_z) / total_weight
    magnitude = math.hypot(average_r, average_z)
    if magnitude <= 0.0:
        raise RuntimeError("area-weighted contact normal cancels to zero")
    return {
        "normal_r": average_r / magnitude,
        "normal_z": average_z / magnitude,
        "edge_outward_normals_rz": tuple(edge_normals),
        "maximum_edge_outward_normal_error": max(edge_errors),
    }


def audit_runtime_contact_topology(
    geometry: Mapping[str, Any],
    *,
    runtime_api: Any | None = None,
    device: str = DEVICE_NAME,
    coordinate_absolute_tolerance_cm: float = DEFAULT_COORDINATE_ABSOLUTE_TOLERANCE_CM,
    area_relative_tolerance: float = DEFAULT_AREA_RELATIVE_TOLERANCE,
    normal_absolute_tolerance: float = DEFAULT_NORMAL_ABSOLUTE_TOLERANCE,
) -> dict[str, Any]:
    """Audit physical contact edges, swept areas, spans, normals, and equations.

    Contact nodes are derived strictly from ``get_element_node_list(...,
    contact=...)``.  A node mask without at least one boundary edge cannot pass.
    """

    if runtime_api is None:
        import devsim as runtime_api

    coordinate_atol = _nonnegative_float(
        coordinate_absolute_tolerance_cm, "coordinate_absolute_tolerance_cm"
    )
    area_rtol = _nonnegative_float(area_relative_tolerance, "area_relative_tolerance")
    normal_atol = _nonnegative_float(normal_absolute_tolerance, "normal_absolute_tolerance")
    expected_by_contact = _expected_contact_geometry(geometry)

    contacts = tuple(str(name) for name in runtime_api.get_contact_list(device=device))
    contacts_complete = set(contacts) == set(TERMINALS)
    try:
        axisymmetric_ok = (
            str(runtime_api.get_parameter(device=device, name="raxis_variable")) == "x"
            and math.isclose(
                _finite_float(
                    runtime_api.get_parameter(device=device, name="raxis_zero"),
                    "raxis_zero",
                ),
                0.0,
                rel_tol=0.0,
                abs_tol=1.0e-30,
            )
            and str(runtime_api.get_parameter(device=device, name="node_volume_model"))
            == "CylindricalNodeVolume"
            and str(runtime_api.get_parameter(device=device, name="edge_couple_model"))
            == "CylindricalEdgeCouple"
        )
    except Exception:
        axisymmetric_ok = False

    reports: dict[str, dict[str, Any]] = {}
    for contact in TERMINALS:
        expected = expected_by_contact[contact]
        if contact not in contacts:
            reports[contact] = _failed_contact_report(
                contact, expected, RuntimeError("contact is absent")
            )
            continue
        try:
            contact_regions = tuple(
                str(name)
                for name in runtime_api.get_region_list(device=device, contact=contact)
            )
            if len(contact_regions) != 1:
                raise RuntimeError("contact must resolve to exactly one region")
            region = contact_regions[0]
            edges = _normalize_contact_edges(
                runtime_api.get_element_node_list(
                    device=device,
                    region=region,
                    contact=contact,
                )
            )
            if not edges:
                raise RuntimeError("contact has zero boundary edges")
            region_triangles = _normalize_region_triangles(
                runtime_api.get_element_node_list(device=device, region=region)
            )
            if not region_triangles:
                raise RuntimeError("contact region has zero triangles")
            node_indices = tuple(sorted({node for edge in edges for node in edge}))
            radial = tuple(
                _finite_float(value, "x")
                for value in runtime_api.get_node_model_values(
                    device=device, region=region, name="x"
                )
            )
            axial = tuple(
                _finite_float(value, "y")
                for value in runtime_api.get_node_model_values(
                    device=device, region=region, name="y"
                )
            )
            surface = tuple(
                _finite_float(value, "CylindricalSurfaceArea")
                for value in runtime_api.get_node_model_values(
                    device=device, region=region, name="CylindricalSurfaceArea"
                )
            )
            array_count = len(radial)
            if not array_count or not all(
                len(values) == array_count
                for values in (axial, surface)
            ):
                raise RuntimeError("contact audit node-model arrays have unequal lengths")
            try:
                normal_r_values = tuple(
                    float(value)
                    for value in runtime_api.get_node_model_values(
                        device=device, region=region, name="ContactNSurfaceNormal_x"
                    )
                )
                normal_z_values = tuple(
                    float(value)
                    for value in runtime_api.get_node_model_values(
                        device=device, region=region, name="ContactNSurfaceNormal_y"
                    )
                )
                if not all(
                    len(values) == array_count
                    for values in (normal_r_values, normal_z_values)
                ):
                    raise ValueError("runtime normal arrays have unequal lengths")
            except Exception:
                # Runtime normals are explicitly diagnostic; their absence or
                # ambiguity cannot invalidate a normal derived from topology.
                normal_r_values = (math.nan,) * array_count
                normal_z_values = (math.nan,) * array_count
            if max(node_indices) >= array_count:
                raise RuntimeError("contact node index is outside region node models")

            radial_at_contact = tuple(radial[index] for index in node_indices)
            axial_at_contact = tuple(axial[index] for index in node_indices)
            plane_values = (
                radial_at_contact if expected["plane_axis"] == "x" else axial_at_contact
            )
            measured_plane_min = min(plane_values)
            measured_plane_max = max(plane_values)
            plane_max_error = max(
                abs(value - float(expected["plane_cm"])) for value in plane_values
            )
            radial_min, radial_max = min(radial_at_contact), max(radial_at_contact)
            axial_min, axial_max = min(axial_at_contact), max(axial_at_contact)
            plane_ok = plane_max_error <= coordinate_atol
            radial_span_ok = _all_close_span(
                radial_min,
                radial_max,
                expected["radial_span_cm"],
                coordinate_atol,
            )
            axial_span_ok = _all_close_span(
                axial_min,
                axial_max,
                expected["axial_span_cm"],
                coordinate_atol,
            )
            span_ok = radial_span_ok and axial_span_ok

            geometric_area = _cylindrical_swept_edge_area_cm2(edges, radial, axial)
            edge_lengths = tuple(
                math.hypot(
                    radial[node1] - radial[node0], axial[node1] - axial[node0]
                )
                for node0, node1 in edges
            )
            surface_area = math.fsum(surface[index] for index in node_indices)
            analytic_area = _positive_float(expected["area_cm2"], "analytic area")
            geometric_error = _relative_error(geometric_area, analytic_area)
            surface_error = _relative_error(surface_area, analytic_area)
            geometric_area_ok = geometric_error <= area_rtol
            surface_area_ok = surface_error <= area_rtol
            area_ok = geometric_area_ok and surface_area_ok

            expected_normal_r, expected_normal_z = expected["outward_normal_rz"]
            weights = tuple(abs(surface[index]) for index in node_indices)
            weight_sum = math.fsum(weights)
            if weight_sum <= 0.0:
                raise RuntimeError("contact CylindricalSurfaceArea is not positive")
            runtime_normals_finite = all(
                math.isfinite(normal_r_values[index])
                and math.isfinite(normal_z_values[index])
                for index in node_indices
            )
            if runtime_normals_finite:
                runtime_normal_r = math.fsum(
                    weight * normal_r_values[index]
                    for weight, index in zip(weights, node_indices)
                ) / weight_sum
                runtime_normal_z = math.fsum(
                    weight * normal_z_values[index]
                    for weight, index in zip(weights, node_indices)
                ) / weight_sum
                node_normal_errors = tuple(
                    math.hypot(
                        normal_r_values[index] - expected_normal_r,
                        normal_z_values[index] - expected_normal_z,
                    )
                    for index in node_indices
                )
                maximum_normal_error = max(node_normal_errors)
                runtime_normal_matches_geometry = (
                    maximum_normal_error <= normal_atol
                    and math.isclose(
                        math.hypot(runtime_normal_r, runtime_normal_z),
                        1.0,
                        rel_tol=0.0,
                        abs_tol=normal_atol,
                    )
                )
            else:
                runtime_normal_r = math.nan
                runtime_normal_z = math.nan
                maximum_normal_error = math.nan
                runtime_normal_matches_geometry = False
            # ContactNSurfaceNormal is retained only as a diagnostic because
            # its sign convention is not a stable electrode-charge definition.
            # The accepted outward normal comes from the unique incident
            # region triangle: the vector from the edge midpoint to its third
            # node points inward, so the opposite perpendicular points outward.
            geometric_normal = _geometry_outward_contact_normal(
                edges,
                region_triangles,
                radial,
                axial,
                (expected_normal_r, expected_normal_z),
            )
            normal_r = float(geometric_normal["normal_r"])
            normal_z = float(geometric_normal["normal_z"])
            maximum_edge_normal_error = float(
                geometric_normal["maximum_edge_outward_normal_error"]
            )
            normal_ok = (
                plane_ok
                and span_ok
                and maximum_edge_normal_error <= normal_atol
                and math.isclose(
                    math.hypot(normal_r, normal_z),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=normal_atol,
                )
            )

            potential_equations = tuple(
                str(name)
                for name in runtime_api.get_contact_equation_list(
                    device=device, contact=contact
                )
            )
            potential_command: Mapping[str, Any] = {}
            if "PotentialEquation" in potential_equations:
                potential_command = runtime_api.get_contact_equation_command(
                    device=device, contact=contact, name="PotentialEquation"
                )
            potential_ok = (
                "PotentialEquation" in potential_equations
                and str(potential_command.get("edge_charge_model", ""))
                == "PotentialEdgeFlux"
            )
            if contact in {"source", "drain"}:
                current_command: Mapping[str, Any] = {}
                if "ElectronContinuityEquation" in potential_equations:
                    current_command = runtime_api.get_contact_equation_command(
                        device=device,
                        contact=contact,
                        name="ElectronContinuityEquation",
                    )
                current_ok = (
                    "ElectronContinuityEquation" in potential_equations
                    and str(current_command.get("edge_current_model", ""))
                    == "ElectronCurrent"
                )
            else:
                current_ok = True

            region_ok = region == expected["region"]
            passed = all(
                (
                    axisymmetric_ok,
                    region_ok,
                    plane_ok,
                    span_ok,
                    area_ok,
                    normal_ok,
                    potential_ok,
                    current_ok,
                )
            )
            expected_radial_min, expected_radial_max = expected["radial_span_cm"]
            expected_axial_min, expected_axial_max = expected["axial_span_cm"]
            reports[contact] = {
                "contact": contact,
                "region": region,
                "edge_count": len(edges),
                "node_count": len(node_indices),
                "plane_axis": expected["plane_axis"],
                "plane_coordinate_cm": 0.5 * (
                    measured_plane_min + measured_plane_max
                ),
                "expected_plane_cm": expected["plane_cm"],
                "measured_plane_min_cm": measured_plane_min,
                "measured_plane_max_cm": measured_plane_max,
                "plane_max_error_cm": plane_max_error,
                "radial_min_cm": radial_min,
                "radial_max_cm": radial_max,
                "expected_radial_min_cm": expected_radial_min,
                "expected_radial_max_cm": expected_radial_max,
                "axial_min_cm": axial_min,
                "axial_max_cm": axial_max,
                "expected_axial_min_cm": expected_axial_min,
                "expected_axial_max_cm": expected_axial_max,
                "edge_lengths_cm": edge_lengths,
                "edge_outward_normals_rz": geometric_normal[
                    "edge_outward_normals_rz"
                ],
                "maximum_edge_outward_normal_error": maximum_edge_normal_error,
                "geometric_area_cm2": geometric_area,
                "geometric_swept_area_cm2": geometric_area,
                "runtime_area_cm2": surface_area,
                "cylindrical_surface_area_cm2": surface_area,
                "expected_area_cm2": analytic_area,
                "analytic_area_cm2": analytic_area,
                "area_relative_error": max(geometric_error, surface_error),
                "geometric_area_relative_error": geometric_error,
                "surface_model_relative_error": surface_error,
                "normal_r": normal_r,
                "normal_z": normal_z,
                "normal_method": "geometry_boundary_outward",
                "runtime_normal_r": runtime_normal_r,
                "runtime_normal_z": runtime_normal_z,
                "runtime_normal_matches_geometry": runtime_normal_matches_geometry,
                "expected_normal_r": expected_normal_r,
                "expected_normal_z": expected_normal_z,
                "maximum_node_normal_error": maximum_normal_error,
                "region_consistent": region_ok,
                "plane_consistent": plane_ok,
                "span_consistent": span_ok,
                "geometric_area_consistent": geometric_area_ok,
                "surface_model_consistent": surface_area_ok,
                "normal_consistent": normal_ok,
                "potential_equation_consistent": potential_ok,
                "current_equation_consistent": current_ok,
                "axisymmetric_models_consistent": axisymmetric_ok,
                "plane_passed": plane_ok,
                "radial_span_passed": radial_span_ok,
                "normal_passed": normal_ok,
                "area_passed": area_ok,
                "equation_passed": potential_ok and current_ok,
                "passed": passed,
                "error_message": "",
            }
        except Exception as error:
            reports[contact] = _failed_contact_report(contact, expected, error)
            reports[contact]["axisymmetric_models_consistent"] = axisymmetric_ok

    passed = contacts_complete and axisymmetric_ok and all(
        bool(reports[contact]["passed"]) for contact in TERMINALS
    )
    return {
        "device": str(device),
        "contacts_complete": contacts_complete,
        "axisymmetric_models_consistent": axisymmetric_ok,
        "contacts": reports,
        "passed": passed,
        "error_message": "" if passed else "one or more contact topology checks failed",
    }


def contact_topology_rows(
    audit: Mapping[str, Any],
    *,
    mesh_level: str,
) -> list[dict[str, Any]]:
    """Flatten one topology audit to strict CSV-schema rows."""

    rows: list[dict[str, Any]] = []
    for contact in TERMINALS:
        report = dict(audit["contacts"][contact])
        report.pop("contact", None)
        row = {"mesh_level": str(mesh_level), "contact": contact, **report}
        if set(row) != set(CONTACT_TOPOLOGY_FIELDNAMES):
            missing = sorted(set(CONTACT_TOPOLOGY_FIELDNAMES) - set(row))
            extra = sorted(set(row) - set(CONTACT_TOPOLOGY_FIELDNAMES))
            raise ValueError(f"topology row schema mismatch; missing={missing}, extra={extra}")
        rows.append(row)
    return rows


def _resolved_charge_components(components: Mapping[str, Any]) -> dict[str, float]:
    resolved: dict[str, float] = {}
    for canonical, aliases in _CHARGE_ALIASES.items():
        for alias in aliases:
            if alias in components:
                resolved[canonical] = _finite_float(components[alias], alias)
                break
        else:
            raise KeyError(
                f"missing {canonical}; accepted aliases are {', '.join(aliases)}"
            )
    return resolved


def _central_difference(plus: float, minus: float, delta_voltage_V: float) -> float:
    delta = _positive_float(delta_voltage_V, "delta_voltage_V")
    return (_finite_float(plus, "plus") - _finite_float(minus, "minus")) / (
        2.0 * delta
    )


def calculate_derivative_gauss_balance(
    plus_components: Mapping[str, Any],
    minus_components: Mapping[str, Any],
    delta_voltage_V: float,
    *,
    absolute_tolerance_F: float = DEFAULT_DERIVATIVE_ABSOLUTE_TOLERANCE_F,
    relative_tolerance: float = DEFAULT_DERIVATIVE_RELATIVE_TOLERANCE,
    capacitance_floor_F: float = DEFAULT_CAPACITANCE_FLOOR_F,
) -> dict[str, Any]:
    """Evaluate the differentiated global Gauss law for one voltage column."""

    delta = _positive_float(delta_voltage_V, "delta_voltage_V")
    absolute = _nonnegative_float(absolute_tolerance_F, "absolute_tolerance_F")
    relative = _nonnegative_float(relative_tolerance, "relative_tolerance")
    floor = _positive_float(capacitance_floor_F, "capacitance_floor_F")
    plus = _resolved_charge_components(plus_components)
    minus = _resolved_charge_components(minus_components)
    derivatives = {
        name: _central_difference(plus[name], minus[name], delta)
        for name in _CHARGE_ALIASES
    }
    contact_sum = math.fsum(
        derivatives[name]
        for name in ("Qg_contact_C", "Qd_contact_C", "Qs_contact_C")
    )
    internal_sum = math.fsum(
        derivatives[name] for name in ("Qmobile_C", "Qtrap_C", "Qfixed_C")
    )
    noncontact_derivative = 0.0
    residual = math.fsum((contact_sum, internal_sum, noncontact_derivative))
    plus_residual = math.fsum(plus.values())
    minus_residual = math.fsum(minus.values())
    endpoint_residual_derivative = _central_difference(
        plus_residual, minus_residual, delta
    )
    scale = max(floor, *(abs(value) for value in derivatives.values()))
    tolerance = absolute + relative * scale
    absolute_error = abs(residual)
    return {
        "dQg_contact_dV_F": derivatives["Qg_contact_C"],
        "dQd_contact_dV_F": derivatives["Qd_contact_C"],
        "dQs_contact_dV_F": derivatives["Qs_contact_C"],
        "contact_derivative_sum_F": contact_sum,
        "dQmobile_dV_F": derivatives["Qmobile_C"],
        "dQtrap_dV_F": derivatives["Qtrap_C"],
        "dQfixed_dV_F": derivatives["Qfixed_C"],
        "dQnoncontact_boundary_dV_F": noncontact_derivative,
        "internal_derivative_sum_F": internal_sum,
        "derivative_gauss_residual_F": residual,
        "endpoint_residual_derivative_F": endpoint_residual_derivative,
        "derivative_gauss_absolute_error_F": absolute_error,
        "derivative_gauss_scale_F": scale,
        "derivative_gauss_tolerance_F": tolerance,
        "derivative_gauss_relative_error": absolute_error / scale,
        "noncontact_boundary_method": NATURAL_NONCONTACT_METHOD,
        "passed": absolute_error <= tolerance,
        "error_message": "" if absolute_error <= tolerance else "derivative Gauss balance exceeds tolerance",
    }


def build_derivative_gauss_row(
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    row = {**dict(metadata), **dict(result)}
    expected = set(DERIVATIVE_GAUSS_BALANCE_FIELDNAMES)
    if set(row) != expected:
        raise ValueError(
            "derivative Gauss row schema mismatch; "
            f"missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}"
        )
    return row


def _validated_matrix(
    matrix_F: Mapping[tuple[str, str], Any],
) -> dict[tuple[str, str], float]:
    expected = {(measured, perturbed) for measured in TERMINALS for perturbed in TERMINALS}
    if set(matrix_F) != expected:
        raise ValueError(
            "capacitance matrix must contain exactly all nine terminal pairs"
        )
    return {
        key: _finite_float(matrix_F[key], f"matrix_F[{key!r}]") for key in expected
    }


def _resolved_direct_terminal_charges(components: Mapping[str, Any]) -> dict[str, float]:
    resolved = _resolved_charge_components(components)
    return {
        terminal: resolved[charge_name]
        for terminal, charge_name in CONTACT_CHARGE_NAMES.items()
    }


def evaluate_gauge_invariance(
    matrix_F: Mapping[tuple[str, str], Any],
    common_mode_plus_components: Mapping[str, Any],
    common_mode_minus_components: Mapping[str, Any],
    common_mode_delta_voltage_V: float,
    *,
    absolute_tolerance_F: float = DEFAULT_DERIVATIVE_ABSOLUTE_TOLERANCE_F,
    relative_tolerance: float = DEFAULT_DERIVATIVE_RELATIVE_TOLERANCE,
    capacitance_floor_F: float = DEFAULT_CAPACITANCE_FLOOR_F,
) -> dict[str, Any]:
    """Check matrix row sums and an independently measured common-mode slope.

    Contact-only column sums are returned as diagnostics and are never tested.
    """

    matrix = _validated_matrix(matrix_F)
    delta = _positive_float(common_mode_delta_voltage_V, "common_mode_delta_voltage_V")
    absolute = _nonnegative_float(absolute_tolerance_F, "absolute_tolerance_F")
    relative = _nonnegative_float(relative_tolerance, "relative_tolerance")
    floor = _positive_float(capacitance_floor_F, "capacitance_floor_F")
    plus = _resolved_direct_terminal_charges(common_mode_plus_components)
    minus = _resolved_direct_terminal_charges(common_mode_minus_components)
    column_sums = {
        perturbed: math.fsum(matrix[(measured, perturbed)] for measured in TERMINALS)
        for perturbed in TERMINALS
    }
    rows: dict[str, dict[str, Any]] = {}
    for measured in TERMINALS:
        row_values = tuple(matrix[(measured, perturbed)] for perturbed in TERMINALS)
        row_sum = math.fsum(row_values)
        direct = _central_difference(plus[measured], minus[measured], delta)
        difference = row_sum - direct
        row_scale = max(floor, *(abs(value) for value in row_values))
        direct_scale = max(row_scale, abs(direct), floor)
        row_tolerance = absolute + relative * row_scale
        direct_tolerance = absolute + relative * direct_scale
        agreement_tolerance = absolute + relative * direct_scale
        row_passed = abs(row_sum) <= row_tolerance
        direct_passed = abs(direct) <= direct_tolerance
        agreement_passed = abs(difference) <= agreement_tolerance
        passed = row_passed and direct_passed and agreement_passed
        rows[measured] = {
            "measured_terminal": measured,
            "common_mode_delta_voltage_V": delta,
            "matrix_row_sum_F": row_sum,
            "direct_common_mode_derivative_F": direct,
            "row_direct_difference_F": difference,
            "row_scale_F": row_scale,
            "row_tolerance_F": row_tolerance,
            "direct_tolerance_F": direct_tolerance,
            "agreement_tolerance_F": agreement_tolerance,
            "row_sum_passed": row_passed,
            "direct_common_mode_passed": direct_passed,
            "row_direct_agreement_passed": agreement_passed,
            "passed": passed,
            "error_message": "" if passed else "gauge-invariance row check failed",
        }
    return {
        "rows": rows,
        "contact_column_sums_F_diagnostic_only": column_sums,
        "column_sums_are_acceptance_criteria": False,
        "passed": all(bool(row["passed"]) for row in rows.values()),
    }


def build_gauge_invariance_rows(
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    expected = set(GAUGE_INVARIANCE_FIELDNAMES)
    for terminal in TERMINALS:
        row = {**dict(metadata), **dict(result["rows"][terminal])}
        if set(row) != expected:
            raise ValueError(
                "gauge row schema mismatch; "
                f"missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}"
            )
        rows.append(row)
    return rows


def compare_numeric_values(
    reference_value: Any,
    candidate_value: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    scale_floor: float = 0.0,
) -> dict[str, Any]:
    """Compare values with ``atol + rtol*max(abs(values), floor)``.

    Non-finite values are explicit failures rather than exceptions, which lets
    a batch comparison retain evidence for every quantity.
    """

    absolute = _nonnegative_float(absolute_tolerance, "absolute_tolerance")
    relative = _nonnegative_float(relative_tolerance, "relative_tolerance")
    floor = _nonnegative_float(scale_floor, "scale_floor")
    try:
        reference = _finite_float(reference_value, "reference_value")
        candidate = _finite_float(candidate_value, "candidate_value")
    except Exception as error:
        return {
            "reference_value": reference_value,
            "candidate_value": candidate_value,
            "absolute_difference": math.nan,
            "scale": math.nan,
            "absolute_tolerance": absolute,
            "relative_tolerance": relative,
            "combined_tolerance": math.nan,
            "scaled_relative_difference": math.nan,
            "passed": False,
            "error_message": _single_line_error(error),
        }
    difference = abs(candidate - reference)
    scale = max(abs(reference), abs(candidate), floor)
    tolerance = absolute + relative * scale
    if scale > 0.0:
        scaled_difference = difference / scale
    else:
        scaled_difference = 0.0 if difference == 0.0 else math.inf
    return {
        "reference_value": reference,
        "candidate_value": candidate,
        "absolute_difference": difference,
        "scale": scale,
        "absolute_tolerance": absolute,
        "relative_tolerance": relative,
        "combined_tolerance": tolerance,
        "scaled_relative_difference": scaled_difference,
        "passed": difference <= tolerance,
        "error_message": "" if difference <= tolerance else "difference exceeds tolerance",
    }


def compare_quantity_maps(
    reference: Mapping[Hashable, Any],
    candidate: Mapping[Hashable, Any],
    *,
    comparison: str,
    absolute_tolerance: float,
    relative_tolerance: float,
    scale_floor: float = 0.0,
    tolerance_overrides: Mapping[Hashable, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare like-keyed repeat or mesh quantities without near-zero blow-up."""

    keys = set(reference) | set(candidate)
    overrides = tolerance_overrides or {}
    rows: list[dict[str, Any]] = []
    for key in sorted(keys, key=str):
        settings = dict(overrides.get(key, {}))
        result = compare_numeric_values(
            reference.get(key, math.nan),
            candidate.get(key, math.nan),
            absolute_tolerance=settings.get("absolute_tolerance", absolute_tolerance),
            relative_tolerance=settings.get("relative_tolerance", relative_tolerance),
            scale_floor=settings.get("scale_floor", scale_floor),
        )
        rows.append(
            {
                "comparison": str(comparison),
                "quantity": str(key),
                **result,
            }
        )
    return {"passed": bool(rows) and all(bool(row["passed"]) for row in rows), "rows": rows}


def calculate_ward_dutton_identity(
    qs_mobile_C: Any,
    qd_mobile_C: Any,
    qmobile_channel_C: Any,
    *,
    absolute_tolerance_C: float = 1.0e-24,
    relative_tolerance: float = 1.0e-8,
    charge_floor_C: float = DEFAULT_CHARGE_FLOOR_C,
) -> dict[str, Any]:
    """Validate ``Qs_mobile + Qd_mobile = Qmobile_channel`` only."""

    qs = _finite_float(qs_mobile_C, "Qs_mobile_C")
    qd = _finite_float(qd_mobile_C, "Qd_mobile_C")
    qmobile = _finite_float(qmobile_channel_C, "Qmobile_channel_C")
    absolute = _nonnegative_float(absolute_tolerance_C, "absolute_tolerance_C")
    relative = _nonnegative_float(relative_tolerance, "relative_tolerance")
    floor = _positive_float(charge_floor_C, "charge_floor_C")
    residual = math.fsum((qs, qd, -qmobile))
    scale = max(floor, abs(qs), abs(qd), abs(qmobile))
    tolerance = absolute + relative * scale
    passed = abs(residual) <= tolerance
    return {
        "Qs_mobile_C": qs,
        "Qd_mobile_C": qd,
        "Qmobile_channel_C": qmobile,
        "partition_residual_C": residual,
        "absolute_partition_residual_C": abs(residual),
        "partition_scale_C": scale,
        "partition_tolerance_C": tolerance,
        "relative_partition_residual": abs(residual) / scale,
        "method": WARD_DUTTON_METHOD,
        "status": WARD_DUTTON_STATUS,
        "limitation": WARD_DUTTON_LIMITATION,
        "passed": passed,
        "error_message": "" if passed else "Ward-Dutton mobile partition identity failed",
    }


def build_ward_dutton_row(
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
) -> dict[str, Any]:
    row = {**dict(metadata), **dict(result)}
    expected = set(WARD_DUTTON_FIELDNAMES)
    if set(row) != expected:
        raise ValueError(
            "Ward-Dutton row schema mismatch; "
            f"missing={sorted(expected-set(row))}, extra={sorted(set(row)-expected)}"
        )
    return row


def determine_physical_contact_flux_status(
    criteria: Mapping[str, bool | None],
    *,
    required_criteria: Sequence[str] = REQUIRED_SUPPORT_CRITERIA,
) -> dict[str, Any]:
    """Return supported/provisional/unsupported without a column-sum gate."""

    forbidden = _PROHIBITED_COLUMN_SUM_CRITERIA.intersection(criteria)
    if forbidden:
        raise ValueError(
            "contact-only column sums cannot be support criteria: "
            + ", ".join(sorted(forbidden))
        )
    required = tuple(str(name) for name in required_criteria)
    if not required or len(set(required)) != len(required):
        raise ValueError("required_criteria must be nonempty and unique")
    resolved = {name: criteria.get(name) for name in required}
    invalid = {
        name: value
        for name, value in resolved.items()
        if value not in {True, False, None}
    }
    if invalid:
        raise TypeError(f"support criteria must be bool or None: {invalid}")
    failed = tuple(name for name, value in resolved.items() if value is False)
    unknown = tuple(name for name, value in resolved.items() if value is None)
    if failed:
        status = STATUS_UNSUPPORTED
    elif unknown:
        status = STATUS_PROVISIONAL
    else:
        status = STATUS_SUPPORTED
    return {
        "method": PHYSICAL_CONTACT_FLUX_METHOD,
        "status": status,
        "supported": status == STATUS_SUPPORTED,
        "criteria": resolved,
        "failed_criteria": failed,
        "unknown_criteria": unknown,
        "contact_column_sums_used_for_acceptance": False,
    }


__all__ = [
    "DEVICE_NAME",
    "TERMINALS",
    "CONTACT_CHARGE_NAMES",
    "EXPECTED_CONTACT_REGIONS",
    "PHYSICAL_CONTACT_FLUX_METHOD",
    "WARD_DUTTON_METHOD",
    "WARD_DUTTON_STATUS",
    "NATURAL_NONCONTACT_METHOD",
    "STATUS_SUPPORTED",
    "STATUS_PROVISIONAL",
    "STATUS_UNSUPPORTED",
    "REQUIRED_SUPPORT_CRITERIA",
    "CONTACT_TOPOLOGY_FIELDNAMES",
    "DERIVATIVE_GAUSS_BALANCE_FIELDNAMES",
    "GAUGE_INVARIANCE_FIELDNAMES",
    "NUMERIC_COMPARISON_FIELDNAMES",
    "WARD_DUTTON_FIELDNAMES",
    "analytic_annular_area_cm2",
    "audit_runtime_contact_topology",
    "contact_topology_rows",
    "calculate_derivative_gauss_balance",
    "build_derivative_gauss_row",
    "evaluate_gauge_invariance",
    "build_gauge_invariance_rows",
    "compare_numeric_values",
    "compare_quantity_maps",
    "calculate_ward_dutton_identity",
    "build_ward_dutton_row",
    "determine_physical_contact_flux_status",
]
