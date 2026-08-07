"""Electrostatic field and voltage-drop audits for the ERASE baseline.

The helpers in this module do not alter a DEVSIM solution.  They extract
active-window boundary potentials and first-cell radial fields from the
canonical axisymmetric device so that voltage, length, field, and dielectric
displacement conventions are explicit in the generated CSV files.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT_DIRECTORY = MODULE_DIRECTORY.parent
for search_path in (str(MODULE_DIRECTORY), str(PROJECT_ROOT_DIRECTORY)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

import compact_handoff_parameters as compact_parameters
import material_parameters as material

if Path(compact_parameters.__file__).resolve() != (
    PROJECT_ROOT_DIRECTORY / "compact_handoff_parameters.py"
).resolve():
    raise ImportError("compact_handoff_parameters resolved outside this repository.")
if Path(material.__file__).resolve() != (
    MODULE_DIRECTORY / "material_parameters.py"
).resolve():
    raise ImportError("material_parameters resolved outside 05_program_erase.")


CM_TO_NM = 1.0e7
NM_TO_CM = compact_parameters.NM_TO_CM

ACTIVE_AXIAL_MINIMUM_NM = 10.0
ACTIVE_AXIAL_MAXIMUM_NM = 90.0

LAYER_DEFINITIONS = (
    (
        "TunnelOxide",
        "tunnel_oxide_thickness_nm",
        material.eps_tunnel_oxide,
    ),
    (
        "ChargeTrap",
        "charge_trap_thickness_nm",
        material.eps_charge_trap,
    ),
    (
        "BlockingOxide",
        "blocking_oxide_thickness_nm",
        material.eps_blocking_oxide,
    ),
)


def _finite_values(values: Iterable[float], name: str) -> list[float]:
    converted = [float(value) for value in values]
    if not converted:
        raise ValueError(f"{name} must not be empty.")
    if not all(math.isfinite(value) for value in converted):
        raise ValueError(f"{name} must contain only finite values.")
    return converted


def boundary_potential_statistics(
    x_values_cm: Sequence[float],
    y_values_cm: Sequence[float],
    potential_values_V: Sequence[float],
    boundary: str,
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
    radial_tolerance_cm: float = 1.0e-12,
) -> dict:
    """Return potential statistics on one radial region boundary.

    DEVSIM stores duplicate interface nodes in adjacent regions.  Keeping each
    region's boundary value separate makes any interface-continuity residual
    visible instead of forcing a telescoping voltage sum by construction.
    """

    x_values = _finite_values(x_values_cm, "x_values_cm")
    y_values = _finite_values(y_values_cm, "y_values_cm")
    potentials = _finite_values(potential_values_V, "potential_values_V")
    if not (len(x_values) == len(y_values) == len(potentials)):
        raise ValueError("Node coordinate and potential arrays must match.")
    if boundary not in {"inner", "outer"}:
        raise ValueError("boundary must be 'inner' or 'outer'.")

    minimum_axial_cm = float(minimum_axial_nm) * NM_TO_CM
    maximum_axial_cm = float(maximum_axial_nm) * NM_TO_CM
    if maximum_axial_cm <= minimum_axial_cm:
        raise ValueError("Active axial window must have positive length.")

    boundary_radius_cm = (
        min(x_values) if boundary == "inner" else max(x_values)
    )
    selected = [
        potential
        for x_value, y_value, potential in zip(
            x_values,
            y_values,
            potentials,
        )
        if abs(x_value - boundary_radius_cm) <= radial_tolerance_cm
        and minimum_axial_cm <= y_value <= maximum_axial_cm
    ]
    if not selected:
        raise RuntimeError(
            f"No {boundary} boundary nodes were found in the active window."
        )

    mean_value = sum(selected) / len(selected)
    variance = sum(
        (value - mean_value) ** 2 for value in selected
    ) / len(selected)
    return {
        "boundary": boundary,
        "boundary_radius_cm": boundary_radius_cm,
        "boundary_radius_nm": boundary_radius_cm * CM_TO_NM,
        "node_count": len(selected),
        "potential_mean_V": mean_value,
        "potential_min_V": min(selected),
        "potential_max_V": max(selected),
        "potential_std_V": math.sqrt(variance),
        "active_axial_minimum_nm": float(minimum_axial_nm),
        "active_axial_maximum_nm": float(maximum_axial_nm),
    }


def get_region_boundary_potential_statistics(
    device: str,
    region: str,
    boundary: str,
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
) -> dict:
    """Read DEVSIM node values and evaluate one boundary potential."""

    from devsim import get_node_model_values

    result = boundary_potential_statistics(
        x_values_cm=get_node_model_values(
            device=device,
            region=region,
            name="x",
        ),
        y_values_cm=get_node_model_values(
            device=device,
            region=region,
            name="y",
        ),
        potential_values_V=get_node_model_values(
            device=device,
            region=region,
            name="Potential",
        ),
        boundary=boundary,
        minimum_axial_nm=minimum_axial_nm,
        maximum_axial_nm=maximum_axial_nm,
    )
    result["region"] = region
    return result


def calculate_layer_voltage_drop_rows(
    boundary_statistics: Mapping[str, Mapping[str, Mapping[str, float]]],
    geometry: Mapping[str, float],
) -> tuple[list[dict], dict]:
    """Calculate signed layer drops and a non-forced stack closure audit.

    ``voltage_drop_outer_minus_inner_V`` follows increasing radius.  Because
    electric field is ``-grad(Potential)``, the corresponding outward field is
    the negative voltage drop divided by the layer thickness in centimetres.
    """

    rows = []
    for region, thickness_key, permittivity_F_cm in LAYER_DEFINITIONS:
        region_boundaries = boundary_statistics[region]
        inner = region_boundaries["inner"]
        outer = region_boundaries["outer"]
        thickness_nm = float(geometry[thickness_key])
        thickness_cm = thickness_nm * NM_TO_CM
        if thickness_cm <= 0.0:
            raise ValueError(f"{thickness_key} must be positive.")
        voltage_drop = (
            float(outer["potential_mean_V"])
            - float(inner["potential_mean_V"])
        )
        outward_field = -voltage_drop / thickness_cm
        rows.append(
            {
                "region": region,
                "thickness_nm": thickness_nm,
                "thickness_cm": thickness_cm,
                "inner_radius_nm": inner["boundary_radius_nm"],
                "outer_radius_nm": outer["boundary_radius_nm"],
                "inner_boundary_node_count": inner["node_count"],
                "outer_boundary_node_count": outer["node_count"],
                "inner_potential_mean_V": inner["potential_mean_V"],
                "outer_potential_mean_V": outer["potential_mean_V"],
                "voltage_drop_outer_minus_inner_V": voltage_drop,
                "average_field_from_voltage_drop_outward_V_cm": (
                    outward_field
                ),
                "permittivity_F_cm": float(permittivity_F_cm),
                "average_displacement_from_drop_C_cm2": (
                    float(permittivity_F_cm) * outward_field
                ),
                "mesh_coordinate_unit": "cm",
                "electric_field_unit": "V/cm",
            }
        )

    component_sum = sum(
        row["voltage_drop_outer_minus_inner_V"] for row in rows
    )
    total_stack_drop = (
        float(
            boundary_statistics["BlockingOxide"]["outer"][
                "potential_mean_V"
            ]
        )
        - float(
            boundary_statistics["TunnelOxide"]["inner"][
                "potential_mean_V"
            ]
        )
    )
    tunnel_trap_jump = (
        float(
            boundary_statistics["ChargeTrap"]["inner"]["potential_mean_V"]
        )
        - float(
            boundary_statistics["TunnelOxide"]["outer"]["potential_mean_V"]
        )
    )
    trap_blocking_jump = (
        float(
            boundary_statistics["BlockingOxide"]["inner"][
                "potential_mean_V"
            ]
        )
        - float(
            boundary_statistics["ChargeTrap"]["outer"]["potential_mean_V"]
        )
    )
    closure_residual = component_sum - total_stack_drop
    closure_scale = max(1.0, abs(total_stack_drop))
    summary = {
        "component_voltage_drop_sum_V": component_sum,
        "total_stack_voltage_drop_V": total_stack_drop,
        "layer_voltage_sum_residual_V": closure_residual,
        "layer_voltage_sum_relative_residual": (
            abs(closure_residual) / closure_scale
        ),
        "tunnel_trap_interface_potential_jump_V": tunnel_trap_jump,
        "trap_blocking_interface_potential_jump_V": trap_blocking_jump,
        "component_plus_interface_jump_sum_V": (
            component_sum + tunnel_trap_jump + trap_blocking_jump
        ),
    }
    return rows, summary


def extract_layer_voltage_drop_rows(
    device: str,
    geometry: Mapping[str, float],
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
) -> tuple[list[dict], dict]:
    """Extract all layer boundary potentials and calculate stack drops."""

    boundary_statistics = {}
    for region, _, _ in LAYER_DEFINITIONS:
        boundary_statistics[region] = {
            boundary: get_region_boundary_potential_statistics(
                device=device,
                region=region,
                boundary=boundary,
                minimum_axial_nm=minimum_axial_nm,
                maximum_axial_nm=maximum_axial_nm,
            )
            for boundary in ("inner", "outer")
        }
    return calculate_layer_voltage_drop_rows(
        boundary_statistics=boundary_statistics,
        geometry=geometry,
    )


def _radial_edge_position(edge: Mapping) -> dict:
    enriched = dict(edge)
    enriched["inner_radius_cm"] = min(
        float(edge["x_n0_cm"]),
        float(edge["x_n1_cm"]),
    )
    enriched["outer_radius_cm"] = max(
        float(edge["x_n0_cm"]),
        float(edge["x_n1_cm"]),
    )
    enriched["midpoint_radius_cm"] = 0.5 * (
        float(edge["x_n0_cm"]) + float(edge["x_n1_cm"])
    )
    enriched["midpoint_axial_cm"] = 0.5 * (
        float(edge["y_n0_cm"]) + float(edge["y_n1_cm"])
    )
    return enriched


def select_active_boundary_radial_edges(
    classified_edges: Sequence[Mapping],
    boundary: str,
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
    radius_tolerance_cm: float = 1.0e-12,
) -> list[dict]:
    """Select the innermost or outermost first-cell radial edge layer."""

    if boundary not in {"inner", "outer"}:
        raise ValueError("boundary must be 'inner' or 'outer'.")
    positioned = [
        _radial_edge_position(edge)
        for edge in classified_edges
        if edge.get("direction") == "radial"
    ]
    if not positioned:
        raise RuntimeError("No radial edges were available.")
    key = "inner_radius_cm" if boundary == "inner" else "outer_radius_cm"
    target_radius = (
        min(edge[key] for edge in positioned)
        if boundary == "inner"
        else max(edge[key] for edge in positioned)
    )
    minimum_axial_cm = float(minimum_axial_nm) * NM_TO_CM
    maximum_axial_cm = float(maximum_axial_nm) * NM_TO_CM
    selected = [
        edge
        for edge in positioned
        if abs(edge[key] - target_radius) <= radius_tolerance_cm
        and minimum_axial_cm
        <= edge["midpoint_axial_cm"]
        <= maximum_axial_cm
    ]
    if not selected:
        raise RuntimeError(
            f"No {boundary} first-cell radial edges were found in the active window."
        )
    return sorted(selected, key=lambda edge: edge["midpoint_axial_cm"])


def area_weighted_radial_field(
    edges: Sequence[Mapping],
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
) -> dict:
    """Area-weight a radial first-cell edge layer over the active window."""

    from field_extraction import add_cylindrical_interface_area_weights

    weighted = add_cylindrical_interface_area_weights(
        interface_edges=edges,
        minimum_axial_cm=float(minimum_axial_nm) * NM_TO_CM,
        maximum_axial_cm=float(maximum_axial_nm) * NM_TO_CM,
    )
    total_area = sum(
        float(edge["cylindrical_interface_area_cm2"]) for edge in weighted
    )
    signed_mean = sum(
        float(edge["radial_field_outward_V_cm"])
        * float(edge["cylindrical_interface_area_cm2"])
        for edge in weighted
    ) / total_area
    return {
        "edge_count": len(weighted),
        "total_area_cm2": total_area,
        "field_mean_signed_outward_V_cm": signed_mean,
        "field_mean_abs_V_cm": sum(
            abs(float(edge["radial_field_outward_V_cm"]))
            * float(edge["cylindrical_interface_area_cm2"])
            for edge in weighted
        )
        / total_area,
        "field_max_abs_V_cm": max(
            abs(float(edge["radial_field_outward_V_cm"])) for edge in weighted
        ),
        "edge_data": weighted,
    }


def get_region_boundary_radial_field(
    device: str,
    region: str,
    boundary: str,
    minimum_axial_nm: float = ACTIVE_AXIAL_MINIMUM_NM,
    maximum_axial_nm: float = ACTIVE_AXIAL_MAXIMUM_NM,
) -> dict:
    """Extract one active first-cell radial-field layer from DEVSIM."""

    from field_extraction import (
        add_edge_direction_information,
        get_region_edge_data,
    )

    classified = add_edge_direction_information(
        get_region_edge_data(
            device=device,
            region=region,
            field_model="ElectricField",
        )
    )
    selected = select_active_boundary_radial_edges(
        classified_edges=classified,
        boundary=boundary,
        minimum_axial_nm=minimum_axial_nm,
        maximum_axial_nm=maximum_axial_nm,
    )
    result = area_weighted_radial_field(
        edges=selected,
        minimum_axial_nm=minimum_axial_nm,
        maximum_axial_nm=maximum_axial_nm,
    )
    result.update({"region": region, "boundary": boundary})
    return result


def calculate_displacement_consistency(
    inner_region_field: Mapping[str, float],
    inner_region_permittivity_F_cm: float,
    outer_region_field: Mapping[str, float],
    outer_region_permittivity_F_cm: float,
) -> dict:
    """Compare outward ``epsilon*E`` on the two first-cell sides."""

    inner_D = float(inner_region_permittivity_F_cm) * float(
        inner_region_field["field_mean_signed_outward_V_cm"]
    )
    outer_D = float(outer_region_permittivity_F_cm) * float(
        outer_region_field["field_mean_signed_outward_V_cm"]
    )
    residual = inner_D - outer_D
    scale = max(abs(inner_D), abs(outer_D), 1.0e-300)
    return {
        "inner_epsilon_E_C_cm2": inner_D,
        "outer_epsilon_E_C_cm2": outer_D,
        "epsilon_E_residual_C_cm2": residual,
        "epsilon_E_relative_residual": abs(residual) / scale,
        "extraction_method": "adjacent_region_first_cell_area_weighted_proxy",
    }


def validate_monotonic_field(rows: Sequence[Mapping[str, float]]) -> None:
    """Require mean and maximum tunnel fields to rise with ``abs(VG)``."""

    if not rows:
        raise ValueError("At least one field row is required.")
    ordered = sorted(rows, key=lambda row: abs(float(row["actual_gate_voltage_V"])))
    for key in (
        "devsim_tunnel_field_mean_abs_V_cm",
        "devsim_tunnel_field_max_abs_V_cm",
    ):
        previous = None
        for row in ordered:
            current = float(row[key])
            if previous is not None:
                tolerance = 1.0e-9 * max(1.0, abs(previous), abs(current))
                if current + tolerance < previous:
                    raise RuntimeError(
                        f"{key} is not monotonic with increasing |VG|."
                    )
            previous = current


def validate_layer_voltage_closure(
    summaries: Sequence[Mapping[str, float]],
    relative_tolerance: float = 1.0e-6,
) -> None:
    """Fail when independently sampled layer drops do not close the stack."""

    for summary in summaries:
        residual = float(summary["layer_voltage_sum_relative_residual"])
        if not math.isfinite(residual) or residual > relative_tolerance:
            raise RuntimeError(
                "Layer voltage-drop sum failed the interface-continuity audit: "
                f"relative residual={residual:.6e}."
            )
