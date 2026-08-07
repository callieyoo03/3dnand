"""Read-only diagnostics for contact-current and mesh root-cause analysis.

The helpers in this module do not construct a device, change a bias, or write
candidate/public data.  Runtime functions accept an injected DEVSIM-compatible
API so their numerical parts can be unit tested without DEVSIM.

DEVSIM contact-equation semantics are reproduced deliberately: for every
region edge incident on exactly one active contact node, the edge model times
the configured edge-couple model is added at ``n0`` and subtracted at ``n1``.
An edge whose two endpoints are contact nodes is skipped.  In the present
axisymmetric model this is

``ElectronCurrent * CylindricalEdgeCouple``.

``CylindricalEdgeCouple`` already contains the finite-volume cylindrical
weight.  No annular area is multiplied into the result and the sum is never
divided by the number of contact edges.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


NM_TO_CM = 1.0e-7
CURRENT_EQUATION = "ElectronContinuityEquation"
CURRENT_EDGE_MODEL = "ElectronCurrent"
CYLINDRICAL_EDGE_COUPLE_MODEL = "CylindricalEdgeCouple"
DEFAULT_PROFILE_DISTANCES_NM = (0.0, 1.0, 2.0, 5.0, 10.0)
DEFAULT_COORDINATE_TOLERANCE_CM = 1.0e-15


def _runtime_api(runtime_api: Any | None) -> Any:
    if runtime_api is not None:
        return runtime_api
    import devsim

    return devsim


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonnegative(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _normalize_elements(
    values: Sequence[Any],
    *,
    nodes_per_element: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    raw_values = tuple(values)
    if raw_values and all(isinstance(value, (int, float)) for value in raw_values):
        if nodes_per_element is None:
            raise ValueError(
                "a flat element-node list requires nodes_per_element"
            )
        if len(raw_values) % nodes_per_element:
            raise ValueError(
                "flat element-node list length is not divisible by "
                f"nodes_per_element={nodes_per_element}"
            )
        raw_values = tuple(
            raw_values[index : index + nodes_per_element]
            for index in range(0, len(raw_values), nodes_per_element)
        )
    elements: list[tuple[int, ...]] = []
    for element_index, element in enumerate(raw_values):
        try:
            nodes = tuple(int(node) for node in element)
        except TypeError as error:
            raise ValueError(
                f"element {element_index} must be a node-index sequence"
            ) from error
        if not nodes:
            raise ValueError(f"element {element_index} is empty")
        if nodes_per_element is not None and len(nodes) != nodes_per_element:
            raise ValueError(
                f"element {element_index} contains {len(nodes)} nodes; "
                f"expected {nodes_per_element}"
            )
        elements.append(nodes)
    return tuple(elements)


def _read_node_values(
    runtime_api: Any,
    *,
    device: str,
    region: str,
    name: str,
) -> tuple[float, ...]:
    return tuple(
        _finite(value, f'{region} node model "{name}"')
        for value in runtime_api.get_node_model_values(
            device=device, region=region, name=name
        )
    )


def _read_edge_values(
    runtime_api: Any,
    *,
    device: str,
    region: str,
    name: str,
) -> tuple[float, ...]:
    return tuple(
        _finite(value, f'{region} edge model "{name}"')
        for value in runtime_api.get_edge_model_values(
            device=device, region=region, name=name
        )
    )


def _ensure_coordinate_edge_models(
    runtime_api: Any,
    *,
    device: str,
    region: str,
) -> None:
    """Ensure endpoint coordinates exist without changing solved quantities."""

    edge_models = set(
        str(name)
        for name in runtime_api.get_edge_model_list(device=device, region=region)
    )
    for coordinate in ("x", "y"):
        required = {f"{coordinate}@n0", f"{coordinate}@n1"}
        if required.issubset(edge_models):
            continue
        edge_from_node_model = getattr(runtime_api, "edge_from_node_model", None)
        if not callable(edge_from_node_model):
            raise RuntimeError(
                f"missing {sorted(required - edge_models)} in region {region!r}; "
                "the runtime API cannot create coordinate endpoint models"
            )
        edge_from_node_model(
            device=device,
            region=region,
            node_model=coordinate,
        )
        edge_models = set(
            str(name)
            for name in runtime_api.get_edge_model_list(
                device=device, region=region
            )
        )
        if not required.issubset(edge_models):
            raise RuntimeError(
                f"coordinate endpoint models {sorted(required)} were not created "
                f"in region {region!r}"
            )


def _coordinate_lookup(
    radial_cm: Sequence[float],
    axial_cm: Sequence[float],
    *,
    tolerance_cm: float,
) -> tuple[dict[tuple[float, float], int], Any]:
    if len(radial_cm) != len(axial_cm):
        raise ValueError("radial and axial node-coordinate arrays must match")
    exact: dict[tuple[float, float], int] = {}
    for node, (radial, axial) in enumerate(zip(radial_cm, axial_cm)):
        key = (radial, axial)
        if key in exact:
            raise ValueError(
                "region contains duplicate node coordinates; endpoint-to-node "
                "mapping would be ambiguous"
            )
        exact[key] = node

    def resolve(radial: float, axial: float) -> int:
        key = (radial, axial)
        if key in exact:
            return exact[key]
        matches = [
            node
            for node, (node_r, node_z) in enumerate(zip(radial_cm, axial_cm))
            if math.isclose(node_r, radial, rel_tol=0.0, abs_tol=tolerance_cm)
            and math.isclose(node_z, axial, rel_tol=0.0, abs_tol=tolerance_cm)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "could not map an edge endpoint to exactly one region node: "
                f"r={radial:.17g} cm, z={axial:.17g} cm, matches={matches}"
            )
        return matches[0]

    return exact, resolve


def _runtime_region_edges(
    *,
    device: str,
    region: str,
    runtime_api: Any,
    model_names: Sequence[str] = (),
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> tuple[tuple[float, ...], tuple[float, ...], list[dict[str, Any]]]:
    """Return DEVSIM edges in edge-model order with endpoint node indices."""

    tolerance = _nonnegative(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    _ensure_coordinate_edge_models(
        runtime_api, device=device, region=region
    )
    radial = _read_node_values(
        runtime_api, device=device, region=region, name="x"
    )
    axial = _read_node_values(
        runtime_api, device=device, region=region, name="y"
    )
    _, resolve = _coordinate_lookup(radial, axial, tolerance_cm=tolerance)

    endpoint_values = {
        name: _read_edge_values(
            runtime_api, device=device, region=region, name=name
        )
        for name in ("x@n0", "x@n1", "y@n0", "y@n1")
    }
    model_values = {
        name: _read_edge_values(
            runtime_api, device=device, region=region, name=name
        )
        for name in model_names
    }
    lengths = {len(values) for values in (*endpoint_values.values(), *model_values.values())}
    if len(lengths) != 1:
        raise RuntimeError(
            f"edge-model array lengths do not match in region {region!r}: "
            f"{sorted(lengths)}"
        )
    edge_count = lengths.pop() if lengths else 0
    records: list[dict[str, Any]] = []
    for edge_index in range(edge_count):
        r0 = endpoint_values["x@n0"][edge_index]
        r1 = endpoint_values["x@n1"][edge_index]
        z0 = endpoint_values["y@n0"][edge_index]
        z1 = endpoint_values["y@n1"][edge_index]
        node0 = resolve(r0, z0)
        node1 = resolve(r1, z1)
        if node0 == node1:
            raise RuntimeError(f"region edge {edge_index} has identical endpoints")
        record: dict[str, Any] = {
            "edge_index": edge_index,
            "node0": node0,
            "node1": node1,
            "r0_cm": r0,
            "z0_cm": z0,
            "r1_cm": r1,
            "z1_cm": z1,
            "midpoint_r_cm": 0.5 * (r0 + r1),
            "midpoint_z_cm": 0.5 * (z0 + z1),
            "delta_r_cm": r1 - r0,
            "delta_z_cm": z1 - z0,
            "length_cm": math.hypot(r1 - r0, z1 - z0),
        }
        for name, values in model_values.items():
            record[name] = values[edge_index]
        records.append(record)
    return radial, axial, records


def _contact_context(
    *,
    device: str,
    contact: str,
    region: str | None,
    runtime_api: Any,
    coordinate_tolerance_cm: float,
) -> dict[str, Any]:
    regions = tuple(
        str(name)
        for name in runtime_api.get_region_list(device=device, contact=contact)
    )
    if len(regions) != 1:
        raise RuntimeError(
            f"contact {contact!r} must belong to exactly one region; got {regions}"
        )
    actual_region = regions[0]
    if region is not None and str(region) != actual_region:
        raise ValueError(
            f"contact {contact!r} belongs to {actual_region!r}, not {region!r}"
        )
    radial = _read_node_values(
        runtime_api, device=device, region=actual_region, name="x"
    )
    axial = _read_node_values(
        runtime_api, device=device, region=actual_region, name="y"
    )
    boundary_elements = _normalize_elements(
        runtime_api.get_element_node_list(
            device=device, region=actual_region, contact=contact
        ),
        nodes_per_element=2,
    )
    contact_nodes = tuple(sorted({node for edge in boundary_elements for node in edge}))
    if not contact_nodes:
        raise RuntimeError(f"contact {contact!r} has no boundary nodes")
    if any(node < 0 or node >= len(radial) for node in contact_nodes):
        raise RuntimeError(f"contact {contact!r} contains an invalid region node index")

    contact_r = tuple(radial[node] for node in contact_nodes)
    contact_z = tuple(axial[node] for node in contact_nodes)
    radial_span = max(contact_r) - min(contact_r)
    axial_span = max(contact_z) - min(contact_z)
    tolerance = _nonnegative(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    radial_constant = radial_span <= tolerance
    axial_constant = axial_span <= tolerance
    if axial_constant and not radial_constant:
        plane_axis = "z"
        plane_coordinate_cm = math.fsum(contact_z) / len(contact_z)
    elif radial_constant and not axial_constant:
        plane_axis = "r"
        plane_coordinate_cm = math.fsum(contact_r) / len(contact_r)
    else:
        raise RuntimeError(
            f"contact {contact!r} is not a unique axis-aligned boundary plane"
        )
    return {
        "contact": contact,
        "region": actual_region,
        "radial_coordinates_cm": radial,
        "axial_coordinates_cm": axial,
        "boundary_elements": boundary_elements,
        "contact_nodes": contact_nodes,
        "plane_axis": plane_axis,
        "plane_coordinate_cm": plane_coordinate_cm,
        "radial_min_cm": min(contact_r),
        "radial_max_cm": max(contact_r),
        "axial_min_cm": min(contact_z),
        "axial_max_cm": max(contact_z),
    }


def manual_contact_current_integral(
    device: str,
    contact: str,
    *,
    region: str | None = None,
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> dict[str, Any]:
    """Reproduce DEVSIM's axisymmetric contact-current orientation sum.

    This diagnostic is intentionally restricted to annular end contacts
    (constant ``z``).  Each returned record locates the contributing internal
    region edge and its distance from the inner and outer end-cap corners.
    """

    api = _runtime_api(runtime_api)
    context = _contact_context(
        device=device,
        contact=contact,
        region=region,
        runtime_api=api,
        coordinate_tolerance_cm=coordinate_tolerance_cm,
    )
    actual_region = str(context["region"])
    if context["plane_axis"] != "z":
        raise ValueError(
            "manual_contact_current_integral requires a constant-z annular "
            "source/drain end contact"
        )
    edge_couple_model = str(
        api.get_parameter(device=device, name="edge_couple_model")
    )
    if edge_couple_model != CYLINDRICAL_EDGE_COUPLE_MODEL:
        raise RuntimeError(
            "axisymmetric current audit requires edge_couple_model="
            f"{CYLINDRICAL_EDGE_COUPLE_MODEL!r}; got {edge_couple_model!r}"
        )
    equations = set(
        str(name)
        for name in api.get_contact_equation_list(
            device=device, contact=contact
        )
    )
    if CURRENT_EQUATION not in equations:
        raise RuntimeError(
            f"contact {contact!r} has no {CURRENT_EQUATION!r}"
        )
    command: Mapping[str, Any] = api.get_contact_equation_command(
        device=device,
        contact=contact,
        name=CURRENT_EQUATION,
    )
    configured_model = str(command.get("edge_current_model", ""))
    if configured_model != CURRENT_EDGE_MODEL:
        raise RuntimeError(
            f"{CURRENT_EQUATION} uses edge_current_model={configured_model!r}, "
            f"not {CURRENT_EDGE_MODEL!r}"
        )

    _, _, region_edges = _runtime_region_edges(
        device=device,
        region=actual_region,
        runtime_api=api,
        model_names=(CURRENT_EDGE_MODEL, CYLINDRICAL_EDGE_COUPLE_MODEL),
        coordinate_tolerance_cm=coordinate_tolerance_cm,
    )
    contact_nodes = set(int(node) for node in context["contact_nodes"])
    inner_corner = (
        float(context["radial_min_cm"]),
        float(context["plane_coordinate_cm"]),
    )
    outer_corner = (
        float(context["radial_max_cm"]),
        float(context["plane_coordinate_cm"]),
    )
    contributions: list[dict[str, Any]] = []
    tangent_edge_count = 0
    for edge in region_edges:
        node0_is_contact = int(edge["node0"]) in contact_nodes
        node1_is_contact = int(edge["node1"]) in contact_nodes
        if node0_is_contact and node1_is_contact:
            tangent_edge_count += 1
            continue
        if node0_is_contact == node1_is_contact:
            continue
        if node0_is_contact:
            contact_node = int(edge["node0"])
            region_node = int(edge["node1"])
            contact_endpoint = "n0"
            sign = 1.0
        else:
            contact_node = int(edge["node1"])
            region_node = int(edge["node0"])
            contact_endpoint = "n1"
            sign = -1.0
        current = float(edge[CURRENT_EDGE_MODEL])
        coupling = float(edge[CYLINDRICAL_EDGE_COUPLE_MODEL])
        contribution = sign * current * coupling
        midpoint_r = float(edge["midpoint_r_cm"])
        midpoint_z = float(edge["midpoint_z_cm"])
        contributions.append(
            {
                "edge_index": int(edge["edge_index"]),
                "contact": contact,
                "region": actual_region,
                "contact_node_index": contact_node,
                "region_node_index": region_node,
                "contact_endpoint": contact_endpoint,
                "orientation_sign": sign,
                "r_cm": midpoint_r,
                "z_cm": midpoint_z,
                "contact_r_cm": float(context["radial_coordinates_cm"][contact_node]),
                "contact_z_cm": float(context["axial_coordinates_cm"][contact_node]),
                "region_r_cm": float(context["radial_coordinates_cm"][region_node]),
                "region_z_cm": float(context["axial_coordinates_cm"][region_node]),
                "inner_radial_corner_distance_cm": math.hypot(
                    midpoint_r - inner_corner[0], midpoint_z - inner_corner[1]
                ),
                "outer_radial_corner_distance_cm": math.hypot(
                    midpoint_r - outer_corner[0], midpoint_z - outer_corner[1]
                ),
                "ElectronCurrent": current,
                "CylindricalEdgeCouple": coupling,
                "contribution_A": contribution,
            }
        )
    return {
        "device": device,
        "contact": contact,
        "region": actual_region,
        "equation": CURRENT_EQUATION,
        "edge_current_model": CURRENT_EDGE_MODEL,
        "edge_couple_model": CYLINDRICAL_EDGE_COUPLE_MODEL,
        "orientation_definition": (
            "+model*couple when the active contact node is n0; "
            "-model*couple when it is n1; skip both-contact-node edges"
        ),
        "contact_node_count": len(contact_nodes),
        "contact_boundary_element_count": len(context["boundary_elements"]),
        "active_incident_edge_count": len(contributions),
        "skipped_both_contact_node_edge_count": tangent_edge_count,
        "manual_current_A": math.fsum(
            float(row["contribution_A"]) for row in contributions
        ),
        "edge_contributions": contributions,
    }


def summarize_contact_current_crosscheck(
    *,
    api_current_A: Any,
    manual_current_A: Any,
    absolute_tolerance_A: Any = 1.0e-24,
    relative_tolerance: Any = 1.0e-10,
) -> dict[str, Any]:
    """Compare API and manual sums without rescaling either quantity."""

    api_value = _finite(api_current_A, "api_current_A")
    manual_value = _finite(manual_current_A, "manual_current_A")
    atol = _nonnegative(absolute_tolerance_A, "absolute_tolerance_A")
    rtol = _nonnegative(relative_tolerance, "relative_tolerance")
    difference = manual_value - api_value
    scale = max(abs(api_value), abs(manual_value))
    tolerance = atol + rtol * scale
    return {
        "api_current_A": api_value,
        "manual_current_A": manual_value,
        "manual_minus_api_A": difference,
        "absolute_difference_A": abs(difference),
        "scale_A": scale,
        "absolute_tolerance_A": atol,
        "relative_tolerance": rtol,
        "combined_tolerance_A": tolerance,
        "relative_error": abs(difference) / max(scale, atol)
        if max(scale, atol) > 0.0
        else 0.0,
        "passed": abs(difference) <= tolerance,
    }


def crosscheck_contact_current(
    device: str,
    contact: str,
    *,
    region: str | None = None,
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
    absolute_tolerance_A: float = 1.0e-24,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Run the manual integral and compare it with ``get_contact_current``."""

    api = _runtime_api(runtime_api)
    manual = manual_contact_current_integral(
        device,
        contact,
        region=region,
        runtime_api=api,
        coordinate_tolerance_cm=coordinate_tolerance_cm,
    )
    api_current = api.get_contact_current(
        device=device,
        contact=contact,
        equation=CURRENT_EQUATION,
    )
    summary = summarize_contact_current_crosscheck(
        api_current_A=api_current,
        manual_current_A=manual["manual_current_A"],
        absolute_tolerance_A=absolute_tolerance_A,
        relative_tolerance=relative_tolerance,
    )
    return {"summary": summary, "manual_integral": manual}


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        return math.nan
    position = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return (
        (1.0 - weight) * float(sorted_values[lower])
        + weight * float(sorted_values[upper])
    )


def distribution_summary(values: Sequence[Any]) -> dict[str, float | int]:
    """Return deterministic scalar distribution diagnostics."""

    finite_values = sorted(
        _finite(value, f"values[{index}]") for index, value in enumerate(values)
    )
    if not finite_values:
        return {
            "count": 0,
            "minimum": math.nan,
            "maximum": math.nan,
            "mean": math.nan,
            "median": math.nan,
            "p05": math.nan,
            "p95": math.nan,
            "sum": 0.0,
        }
    total = math.fsum(finite_values)
    return {
        "count": len(finite_values),
        "minimum": finite_values[0],
        "maximum": finite_values[-1],
        "mean": total / len(finite_values),
        "median": _percentile(finite_values, 0.5),
        "p05": _percentile(finite_values, 0.05),
        "p95": _percentile(finite_values, 0.95),
        "sum": total,
    }


def triangle_quality_metrics(
    radial_coordinates_cm: Sequence[Any],
    axial_coordinates_cm: Sequence[Any],
    triangles: Sequence[Sequence[Any]],
    *,
    degenerate_area_tolerance_cm2: Any = 0.0,
) -> dict[str, Any]:
    """Calculate angle and aspect diagnostics for 2-D triangular elements.

    Aspect ratio is ``longest edge / shortest altitude``.  It equals 2 for a
    right isosceles triangle and ``2/sqrt(3)`` for an equilateral triangle.
    """

    radial = tuple(
        _finite(value, f"radial_coordinates_cm[{index}]")
        for index, value in enumerate(radial_coordinates_cm)
    )
    axial = tuple(
        _finite(value, f"axial_coordinates_cm[{index}]")
        for index, value in enumerate(axial_coordinates_cm)
    )
    if len(radial) != len(axial):
        raise ValueError("radial and axial node-coordinate arrays must match")
    area_tolerance = _nonnegative(
        degenerate_area_tolerance_cm2, "degenerate_area_tolerance_cm2"
    )
    normalized = _normalize_elements(triangles, nodes_per_element=3)
    minimum_angles: list[float] = []
    maximum_angles: list[float] = []
    aspect_ratios: list[float] = []
    areas: list[float] = []
    obtuse_count = 0
    degenerate_count = 0
    element_rows: list[dict[str, Any]] = []
    for triangle_index, nodes in enumerate(normalized):
        if len(nodes) != 3:
            raise ValueError(
                f"element {triangle_index} is not a triangle: {nodes}"
            )
        if any(node < 0 or node >= len(radial) for node in nodes):
            raise ValueError(f"triangle {triangle_index} has an invalid node index")
        points = tuple((radial[node], axial[node]) for node in nodes)
        twice_area = abs(
            (points[1][0] - points[0][0]) * (points[2][1] - points[0][1])
            - (points[1][1] - points[0][1]) * (points[2][0] - points[0][0])
        )
        area = 0.5 * twice_area
        side_lengths = (
            math.hypot(points[1][0] - points[0][0], points[1][1] - points[0][1]),
            math.hypot(points[2][0] - points[1][0], points[2][1] - points[1][1]),
            math.hypot(points[0][0] - points[2][0], points[0][1] - points[2][1]),
        )
        degenerate = area <= area_tolerance or min(side_lengths) <= 0.0
        if degenerate:
            degenerate_count += 1
            element_rows.append(
                {
                    "triangle_index": triangle_index,
                    "nodes": nodes,
                    "area_cm2": area,
                    "minimum_angle_deg": math.nan,
                    "maximum_angle_deg": math.nan,
                    "aspect_ratio": math.inf,
                    "obtuse": False,
                    "degenerate": True,
                }
            )
            continue
        angles: list[float] = []
        for local_node in range(3):
            vertex = points[local_node]
            first = points[(local_node + 1) % 3]
            second = points[(local_node + 2) % 3]
            vector1 = (first[0] - vertex[0], first[1] - vertex[1])
            vector2 = (second[0] - vertex[0], second[1] - vertex[1])
            cross = abs(vector1[0] * vector2[1] - vector1[1] * vector2[0])
            dot = vector1[0] * vector2[0] + vector1[1] * vector2[1]
            angles.append(math.degrees(math.atan2(cross, dot)))
        minimum_angle = min(angles)
        maximum_angle = max(angles)
        longest_edge = max(side_lengths)
        shortest_altitude = twice_area / longest_edge
        aspect_ratio = longest_edge / shortest_altitude
        obtuse = maximum_angle > 90.0 + 1.0e-12
        obtuse_count += int(obtuse)
        minimum_angles.append(minimum_angle)
        maximum_angles.append(maximum_angle)
        aspect_ratios.append(aspect_ratio)
        areas.append(area)
        element_rows.append(
            {
                "triangle_index": triangle_index,
                "nodes": nodes,
                "area_cm2": area,
                "minimum_angle_deg": minimum_angle,
                "maximum_angle_deg": maximum_angle,
                "aspect_ratio": aspect_ratio,
                "obtuse": obtuse,
                "degenerate": False,
            }
        )
    valid_count = len(normalized) - degenerate_count
    return {
        "triangle_count": len(normalized),
        "valid_triangle_count": valid_count,
        "degenerate_triangle_count": degenerate_count,
        "minimum_angle_deg": min(minimum_angles) if minimum_angles else math.nan,
        "maximum_angle_deg": max(maximum_angles) if maximum_angles else math.nan,
        "maximum_aspect_ratio": (
            math.inf
            if degenerate_count
            else max(aspect_ratios) if aspect_ratios else math.inf
        ),
        "obtuse_triangle_count": obtuse_count,
        "obtuse_fraction": obtuse_count / valid_count if valid_count else math.nan,
        "area_cm2": distribution_summary(areas),
        "minimum_angle_distribution_deg": distribution_summary(minimum_angles),
        "aspect_ratio_distribution": distribution_summary(aspect_ratios),
        "elements": element_rows,
    }


def runtime_mesh_quality_metrics(
    device: str,
    region: str,
    *,
    contacts: Sequence[str] = ("source", "drain"),
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> dict[str, Any]:
    """Audit actual triangle and edge geometry near selected contacts."""

    api = _runtime_api(runtime_api)
    tolerance = _nonnegative(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    radial, axial, edges = _runtime_region_edges(
        device=device,
        region=region,
        runtime_api=api,
        model_names=(CYLINDRICAL_EDGE_COUPLE_MODEL,),
        coordinate_tolerance_cm=tolerance,
    )
    triangles = _normalize_elements(
        api.get_element_node_list(device=device, region=region),
        nodes_per_element=3,
    )
    triangle_metrics = triangle_quality_metrics(radial, axial, triangles)
    radial_spacing = [
        abs(float(edge["delta_r_cm"]))
        for edge in edges
        if abs(float(edge["delta_r_cm"])) > tolerance
    ]
    axial_spacing = [
        abs(float(edge["delta_z_cm"]))
        for edge in edges
        if abs(float(edge["delta_z_cm"])) > tolerance
    ]
    edge_lengths = [float(edge["length_cm"]) for edge in edges]

    contact_metrics: dict[str, Any] = {}
    for contact in contacts:
        context = _contact_context(
            device=device,
            contact=str(contact),
            region=region,
            runtime_api=api,
            coordinate_tolerance_cm=tolerance,
        )
        contact_nodes = set(int(node) for node in context["contact_nodes"])
        total_valence = {node: 0 for node in contact_nodes}
        active_valence = {node: 0 for node in contact_nodes}
        incident_couplings: list[float] = []
        incident_lengths: list[float] = []
        incident_radial_spacing: list[float] = []
        incident_axial_spacing: list[float] = []
        active_edge_indices: list[int] = []
        for edge in edges:
            node0 = int(edge["node0"])
            node1 = int(edge["node1"])
            node0_contact = node0 in contact_nodes
            node1_contact = node1 in contact_nodes
            if node0_contact:
                total_valence[node0] += 1
            if node1_contact:
                total_valence[node1] += 1
            if node0_contact == node1_contact:
                continue
            contact_node = node0 if node0_contact else node1
            active_valence[contact_node] += 1
            active_edge_indices.append(int(edge["edge_index"]))
            incident_couplings.append(
                float(edge[CYLINDRICAL_EDGE_COUPLE_MODEL])
            )
            incident_lengths.append(float(edge["length_cm"]))
            delta_r = abs(float(edge["delta_r_cm"]))
            delta_z = abs(float(edge["delta_z_cm"]))
            if delta_r > tolerance:
                incident_radial_spacing.append(delta_r)
            if delta_z > tolerance:
                incident_axial_spacing.append(delta_z)
        valence_rows = [
            {
                "node_index": node,
                "r_cm": radial[node],
                "z_cm": axial[node],
                "total_region_edge_valence": total_valence[node],
                "active_incident_edge_valence": active_valence[node],
            }
            for node in sorted(contact_nodes)
        ]
        contact_metrics[str(contact)] = {
            "contact_node_count": len(contact_nodes),
            "active_incident_edge_count": len(active_edge_indices),
            "active_incident_edge_indices": active_edge_indices,
            "node_valence": valence_rows,
            "total_region_edge_valence": distribution_summary(
                list(total_valence.values())
            ),
            "active_incident_edge_valence": distribution_summary(
                list(active_valence.values())
            ),
            "incident_CylindricalEdgeCouple": distribution_summary(
                incident_couplings
            ),
            "incident_edge_length_cm": distribution_summary(incident_lengths),
            "incident_radial_spacing_cm": distribution_summary(
                incident_radial_spacing
            ),
            "incident_axial_spacing_cm": distribution_summary(
                incident_axial_spacing
            ),
        }
    return {
        "device": device,
        "region": region,
        "region_node_count": len(radial),
        "region_edge_count": len(edges),
        "edge_couple_model": str(
            api.get_parameter(device=device, name="edge_couple_model")
        ),
        "triangle_quality": triangle_metrics,
        "actual_region_edge_spacing": {
            "radial_cm": distribution_summary(radial_spacing),
            "axial_cm": distribution_summary(axial_spacing),
            "edge_length_cm": distribution_summary(edge_lengths),
        },
        "contacts": contact_metrics,
    }


def _inward_direction(
    coordinates_cm: Sequence[float],
    plane_coordinate_cm: float,
    *,
    tolerance_cm: float,
) -> tuple[float, float]:
    offsets = [coordinate - plane_coordinate_cm for coordinate in coordinates_cm]
    has_positive = any(offset > tolerance_cm for offset in offsets)
    has_negative = any(offset < -tolerance_cm for offset in offsets)
    if has_positive and has_negative:
        raise RuntimeError(
            "contact plane has region nodes on both sides; inward profile "
            "direction is ambiguous"
        )
    if has_positive:
        sign = 1.0
    elif has_negative:
        sign = -1.0
    else:
        raise RuntimeError("region has no interior depth away from the contact plane")
    maximum_depth = max(sign * offset for offset in offsets)
    return sign, maximum_depth


def extract_contact_profiles(
    device: str,
    contact: str,
    *,
    region: str | None = None,
    distances_nm: Sequence[Any] = DEFAULT_PROFILE_DISTANCES_NM,
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> dict[str, Any]:
    """Extract raw cross-plane samples at distances inward from a contact.

    Node quantities are linearly interpolated on every region edge crossing a
    requested plane.  Edge quantities retain their DEVSIM edge orientation;
    endpoint coordinates and ``shallow_side_orientation_sign`` are returned so
    downstream diagnostics need not guess that orientation.  No planar or
    annular area is introduced.
    """

    api = _runtime_api(runtime_api)
    tolerance = _nonnegative(
        coordinate_tolerance_cm, "coordinate_tolerance_cm"
    )
    requested_distances = tuple(
        _nonnegative(value, f"distances_nm[{index}]")
        for index, value in enumerate(distances_nm)
    )
    context = _contact_context(
        device=device,
        contact=contact,
        region=region,
        runtime_api=api,
        coordinate_tolerance_cm=tolerance,
    )
    actual_region = str(context["region"])
    radial, axial, edges = _runtime_region_edges(
        device=device,
        region=actual_region,
        runtime_api=api,
        model_names=(
            "ElectricField",
            CURRENT_EDGE_MODEL,
            CYLINDRICAL_EDGE_COUPLE_MODEL,
        ),
        coordinate_tolerance_cm=tolerance,
    )
    potential = _read_node_values(
        api, device=device, region=actual_region, name="Potential"
    )
    electrons = _read_node_values(
        api, device=device, region=actual_region, name="Electrons"
    )
    if not (len(potential) == len(electrons) == len(radial)):
        raise RuntimeError("node model lengths do not match region coordinates")

    plane_axis = str(context["plane_axis"])
    plane_coordinate = float(context["plane_coordinate_cm"])
    depth_coordinates = axial if plane_axis == "z" else radial
    inward_sign, maximum_depth_cm = _inward_direction(
        depth_coordinates,
        plane_coordinate,
        tolerance_cm=tolerance,
    )
    profiles: list[dict[str, Any]] = []
    for target_nm in requested_distances:
        target_cm = target_nm * NM_TO_CM
        samples: list[dict[str, Any]] = []
        if target_cm <= maximum_depth_cm + tolerance:
            for edge in edges:
                node0 = int(edge["node0"])
                node1 = int(edge["node1"])
                coordinate0 = axial[node0] if plane_axis == "z" else radial[node0]
                coordinate1 = axial[node1] if plane_axis == "z" else radial[node1]
                depth0 = inward_sign * (coordinate0 - plane_coordinate)
                depth1 = inward_sign * (coordinate1 - plane_coordinate)
                depth_span = depth1 - depth0
                if abs(depth_span) <= tolerance:
                    continue
                if target_cm < min(depth0, depth1) - tolerance:
                    continue
                if target_cm > max(depth0, depth1) + tolerance:
                    continue
                fraction = (target_cm - depth0) / depth_span
                fraction = min(1.0, max(0.0, fraction))
                if math.isclose(fraction, 0.0, rel_tol=0.0, abs_tol=1.0e-12):
                    plane_intersection_relation = "at_n0_endpoint"
                elif math.isclose(
                    fraction, 1.0, rel_tol=0.0, abs_tol=1.0e-12
                ):
                    plane_intersection_relation = "at_n1_endpoint"
                else:
                    plane_intersection_relation = "edge_interior"
                shallow_sign = 1.0 if depth0 < depth1 else -1.0
                current = float(edge[CURRENT_EDGE_MODEL])
                coupling = float(edge[CYLINDRICAL_EDGE_COUPLE_MODEL])
                samples.append(
                    {
                        "edge_index": int(edge["edge_index"]),
                        "node0": node0,
                        "node1": node1,
                        "interpolation_fraction_n0_to_n1": fraction,
                        "plane_intersection_relation": plane_intersection_relation,
                        "r_cm": radial[node0]
                        + fraction * (radial[node1] - radial[node0]),
                        "z_cm": axial[node0]
                        + fraction * (axial[node1] - axial[node0]),
                        "distance_from_contact_nm": target_nm,
                        "Potential_V": potential[node0]
                        + fraction * (potential[node1] - potential[node0]),
                        "Electrons_cm3": electrons[node0]
                        + fraction * (electrons[node1] - electrons[node0]),
                        "ElectricField_V_per_cm": float(edge["ElectricField"]),
                        "ElectronCurrent": current,
                        "CylindricalEdgeCouple": coupling,
                        "ElectronCurrent_times_CylindricalEdgeCouple_A": (
                            current * coupling
                        ),
                        "shallow_side_orientation_sign": shallow_sign,
                        "shallow_side_oriented_current_A": (
                            shallow_sign * current * coupling
                        ),
                        "edge_r0_cm": radial[node0],
                        "edge_z0_cm": axial[node0],
                        "edge_r1_cm": radial[node1],
                        "edge_z1_cm": axial[node1],
                    }
                )
        samples.sort(key=lambda row: (row["r_cm"], row["z_cm"], row["edge_index"]))
        profiles.append(
            {
                "requested_distance_nm": target_nm,
                "requested_distance_cm": target_cm,
                "within_region_depth": target_cm <= maximum_depth_cm + tolerance,
                "sample_count": len(samples),
                "samples": samples,
            }
        )
    return {
        "device": device,
        "contact": contact,
        "region": actual_region,
        "plane_axis": plane_axis,
        "plane_coordinate_cm": plane_coordinate,
        "inward_coordinate_sign": inward_sign,
        "maximum_inward_depth_cm": maximum_depth_cm,
        "maximum_inward_depth_nm": maximum_depth_cm / NM_TO_CM,
        "distance_definition": (
            "signed inward distance normal to the contact plane; node models "
            "are linearly interpolated on crossing edges and edge models keep "
            "their native DEVSIM n0/n1 orientation. Rows are raw crossing-edge "
            "samples; coincident one-sided endpoint rows must not be summed as "
            "a unique cross-section current"
        ),
        "profiles": profiles,
    }


def manual_contact_current_audit(
    device: str,
    contact: str,
    *,
    region: str | None = None,
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
    absolute_tolerance_A: float = 1.0e-24,
    relative_tolerance: float = 1.0e-10,
) -> dict[str, Any]:
    """Public API for the DEVSIM/manual contact-current cross-check."""

    return crosscheck_contact_current(
        device,
        contact,
        region=region,
        runtime_api=runtime_api,
        coordinate_tolerance_cm=coordinate_tolerance_cm,
        absolute_tolerance_A=absolute_tolerance_A,
        relative_tolerance=relative_tolerance,
    )


def mesh_quality_diagnostic_rows(
    device: str,
    region: str,
    *,
    contacts: Sequence[str] = ("source", "drain"),
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> list[dict[str, Any]]:
    """Return flat, CSV-friendly rows from :func:`runtime_mesh_quality_metrics`."""

    report = runtime_mesh_quality_metrics(
        device,
        region,
        contacts=contacts,
        runtime_api=runtime_api,
        coordinate_tolerance_cm=coordinate_tolerance_cm,
    )
    triangle = report["triangle_quality"]
    spacing = report["actual_region_edge_spacing"]
    rows: list[dict[str, Any]] = [
        {
            "row_type": "region_summary",
            "device": device,
            "region": region,
            "contact": "",
            "node_index": -1,
            "triangle_index": -1,
            "region_node_count": report["region_node_count"],
            "region_edge_count": report["region_edge_count"],
            "triangle_count": triangle["triangle_count"],
            "degenerate_triangle_count": triangle["degenerate_triangle_count"],
            "minimum_angle_deg": triangle["minimum_angle_deg"],
            "maximum_aspect_ratio": triangle["maximum_aspect_ratio"],
            "obtuse_fraction": triangle["obtuse_fraction"],
            "radial_spacing_min_cm": spacing["radial_cm"]["minimum"],
            "radial_spacing_max_cm": spacing["radial_cm"]["maximum"],
            "axial_spacing_min_cm": spacing["axial_cm"]["minimum"],
            "axial_spacing_max_cm": spacing["axial_cm"]["maximum"],
            "edge_length_min_cm": spacing["edge_length_cm"]["minimum"],
            "edge_length_max_cm": spacing["edge_length_cm"]["maximum"],
        }
    ]
    for element in triangle["elements"]:
        rows.append(
            {
                "row_type": "triangle",
                "device": device,
                "region": region,
                "contact": "",
                **element,
            }
        )
    for contact, metrics in report["contacts"].items():
        coupling = metrics["incident_CylindricalEdgeCouple"]
        rows.append(
            {
                "row_type": "contact_summary",
                "device": device,
                "region": region,
                "contact": contact,
                "node_index": -1,
                "triangle_index": -1,
                "contact_node_count": metrics["contact_node_count"],
                "active_incident_edge_count": metrics[
                    "active_incident_edge_count"
                ],
                "incident_couple_minimum": coupling["minimum"],
                "incident_couple_maximum": coupling["maximum"],
                "incident_couple_mean": coupling["mean"],
                "incident_couple_median": coupling["median"],
                "incident_couple_sum": coupling["sum"],
            }
        )
        for valence in metrics["node_valence"]:
            rows.append(
                {
                    "row_type": "contact_node",
                    "device": device,
                    "region": region,
                    "contact": contact,
                    "triangle_index": -1,
                    **valence,
                }
            )
    return rows


def extract_contact_spatial_profile_rows(
    device: str,
    contact: str,
    *,
    region: str | None = None,
    distances_nm: Sequence[Any] = DEFAULT_PROFILE_DISTANCES_NM,
    runtime_api: Any | None = None,
    coordinate_tolerance_cm: float = DEFAULT_COORDINATE_TOLERANCE_CM,
) -> list[dict[str, Any]]:
    """Return flat rows for the requested inward contact-profile planes."""

    report = extract_contact_profiles(
        device,
        contact,
        region=region,
        distances_nm=distances_nm,
        runtime_api=runtime_api,
        coordinate_tolerance_cm=coordinate_tolerance_cm,
    )
    rows: list[dict[str, Any]] = []
    common = {
        "device": device,
        "contact": contact,
        "region": report["region"],
        "plane_axis": report["plane_axis"],
        "plane_coordinate_cm": report["plane_coordinate_cm"],
        "inward_coordinate_sign": report["inward_coordinate_sign"],
    }
    for profile in report["profiles"]:
        if profile["samples"]:
            for sample in profile["samples"]:
                rows.append(
                    {
                        **common,
                        "requested_distance_nm": profile[
                            "requested_distance_nm"
                        ],
                        "within_region_depth": profile["within_region_depth"],
                        "sample_available": True,
                        **sample,
                    }
                )
        else:
            rows.append(
                {
                    **common,
                    "requested_distance_nm": profile["requested_distance_nm"],
                    "within_region_depth": profile["within_region_depth"],
                    "sample_available": False,
                    "edge_index": -1,
                }
            )
    return rows
