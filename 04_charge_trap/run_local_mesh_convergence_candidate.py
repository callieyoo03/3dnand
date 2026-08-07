"""Run the quality-controlled local-mesh convergence *candidate*.

This module is deliberately isolated from every public data writer.  Four
fixed-position local meshes are built in fresh processes, and two reference
points are solved on each mesh.  Independent repeats are run only for the
ultra-fine mesh.  The published CSV schemas are:

``local_mesh_family_definition.csv``
    Fixed refinement positions and level-dependent spacings.
``local_mesh_quality.csv``
    Region triangle/spacing statistics, contact valence/couple statistics,
    and runtime contact topology.
``local_mesh_reference_points.csv``
    Biases, currents, physical contact charges, the raw 3x3 matrix, Gauss,
    derivative-Gauss, gauge, and finite-difference sensitivity diagnostics.
``local_mesh_reference_convergence.csv``
    Pairwise comparisons and the fail-closed final gate.
``local_mesh_contact_normal_profiles.csv``
    Raw crossing-edge samples.  These rows are never summed as terminal
    current and their local axial spacing is the sampled edge's axial span.

NaN is serialized literally as ``NaN``.  The candidate directory is created
by an exclusive atomic rename only after protected inputs have been rehashed.
No ERASE/retention module and no public characterization writer is imported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
RESULTS_DIRECTORY = MODULE_DIRECTORY / "results"
CANDIDATE_DIRECTORY = RESULTS_DIRECTORY / "local_mesh_convergence_candidate"

SCHEMA_VERSION = "local_mesh_convergence_candidate_v1"
REFERENCE_POINTS = (
    {"state_index": 0, "bias_name": "on", "target_VGS_V": 3.0,
     "target_VDS_V": 0.05, "target_VS_V": 0.0, "target_ntrap_cm3": 0.0},
    {"state_index": 4, "bias_name": "off", "target_VGS_V": -1.0,
     "target_VDS_V": 0.05, "target_VS_V": 0.0,
     "target_ntrap_cm3": 2.0e18},
)
PROFILE_DISTANCES_NM = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
DELTA_VOLTAGES_V = (0.0005, 0.001, 0.002)
DELTA_LABELS = ((0.0005, "0p0005"), (0.001, "0p001"), (0.002, "0p002"))
NOMINAL_DELTA_V = 0.001
SUPPORTED_CAPACITANCES = ("Cgg_F", "Cgd_F", "Cgs_F")
TERMINALS = ("gate", "drain", "source")
CONTACT_CHARGE_KEYS = {
    "gate": "Qg_contact_C", "drain": "Qd_contact_C", "source": "Qs_contact_C",
}
CURRENT_CONTINUITY_ATOL_A = 1.0e-24
CURRENT_CONTINUITY_RTOL = 1.0e-6
CAPACITANCE_SENSITIVITY_FLOOR_F = 1.0e-24
BIAS_ATOL_V = 1.0e-12

OUTPUT_FILENAMES = (
    "local_mesh_family_definition.csv",
    "local_mesh_quality.csv",
    "local_mesh_reference_points.csv",
    "local_mesh_reference_convergence.csv",
    "local_mesh_contact_normal_profiles.csv",
)
FULL_DC_OUTPUT_FILENAMES = (
    "local_mesh_full_dc_metrics.csv",
    "local_mesh_full_dc_convergence.csv",
)

POINT_FIELDS = (
    "schema_version", "mesh_level", "level_order", "run_role",
    "state_index", "state", "bias_name", "ntrap_cm3", "target_ntrap_cm3",
    "target_VGS_V", "target_VDS_V", "target_VS_V",
    "actual_VGS_V", "actual_VDS_V", "actual_VS_V", "bias_target_reached",
    "trap_target_reached", "target_reached",
    "ID_A", "drain_current_A", "source_current_A",
    "source_drain_continuity_residual_A", "continuity_scale_A",
    "continuity_tolerance_A", "continuity_passed",
    "Qg_C", "raw_Qd_C", "raw_Qs_C", "Qmobile_C", "Qtrap_C", "Qfixed_C",
    "Qnoncontact_boundary_C",
    "Cgg_F", "Cgd_F", "Cgs_F", "Cdg_F", "Cdd_F", "Cds_F",
    "Csg_F", "Csd_F", "Css_F",
    "Cgg_delta_sensitivity", "Cgd_delta_sensitivity", "Cgs_delta_sensitivity",
    "supported_delta_sensitivity_max", "supported_delta_sensitivity_passed",
    "global_gauss_residual_C", "global_gauss_scale_C",
    "global_gauss_tolerance_C", "global_gauss_passed",
    "derivative_gauss_gate_residual_F", "derivative_gauss_gate_tolerance_F",
    "derivative_gauss_gate_passed", "derivative_gauss_drain_residual_F",
    "derivative_gauss_drain_tolerance_F", "derivative_gauss_drain_passed",
    "derivative_gauss_source_residual_F", "derivative_gauss_source_tolerance_F",
    "derivative_gauss_source_passed", "derivative_gauss_row_count",
    "derivative_gauss_max_normalized_residual", "derivative_gauss_passed",
    *tuple(
        f"derivative_gauss_{terminal}_dv_{label}_{suffix}"
        for _, label in DELTA_LABELS
        for terminal in TERMINALS
        for suffix in ("residual_F", "tolerance_F", "passed")
    ),
    "gauge_gate_row_sum_F", "gauge_gate_tolerance_F", "gauge_gate_passed",
    "gauge_drain_row_sum_F", "gauge_drain_tolerance_F", "gauge_drain_passed",
    "gauge_source_row_sum_F", "gauge_source_tolerance_F", "gauge_source_passed",
    "gauge_row_sum_passed", "source_edge_count", "drain_edge_count",
    "source_area_cm2", "drain_area_cm2", "topology_passed",
    "global_coordinate_count", "triangle_count", "runtime_seconds",
    "worker_process_id", "mesh_position_hash", "mesh_line_validation_passed",
    "active_geometry_sha256", "material_runtime_sha256",
    "state_definition_sha256", "solver_settings_sha256",
    "family_position_hash_sha256", "runtime_invariants_passed",
    "final_reference_validation_passed", "qc_point_passed", "converged",
    "error_message",
)

QUALITY_FIELDS = (
    "schema_version", "mesh_level", "level_order", "row_type", "region",
    "contact", "node_index", "region_node_count", "region_edge_count",
    "triangle_count", "valid_triangle_count", "degenerate_triangle_count",
    "minimum_angle_deg", "minimum_angle_p01_deg", "minimum_angle_median_deg",
    "minimum_angle_p05_deg", "minimum_angle_p99_deg", "maximum_angle_deg",
    "aspect_ratio_minimum", "aspect_ratio_p01", "aspect_ratio_median",
    "aspect_ratio_p95", "aspect_ratio_p99",
    "maximum_aspect_ratio", "obtuse_fraction", "radial_spacing_min_cm",
    "radial_spacing_p01_cm", "radial_spacing_p05_cm", "radial_spacing_median_cm",
    "radial_spacing_p95_cm", "radial_spacing_p99_cm", "radial_spacing_max_cm",
    "axial_spacing_min_cm", "axial_spacing_p01_cm", "axial_spacing_p05_cm",
    "axial_spacing_median_cm", "axial_spacing_p95_cm", "axial_spacing_p99_cm",
    "axial_spacing_max_cm", "distance_from_contact_nm",
    "target_coordinate_cm", "actual_coordinate_cm", "coordinate_error_cm",
    "interval_start_cm", "interval_end_cm", "contact_plane_matches_geometry",
    "contact_node_count", "active_incident_edge_count", "valence_minimum",
    "valence_p05", "valence_median", "valence_p95", "valence_maximum",
    "incident_couple_minimum_cm2", "incident_couple_p05_cm2",
    "incident_couple_median_cm2", "incident_couple_p95_cm2",
    "incident_couple_maximum_cm2",
    "incident_couple_sum_cm2", "edge_count", "geometric_area_cm2",
    "runtime_area_cm2", "expected_area_cm2", "normal_r", "normal_z",
    "contact_area_relative_error", "contact_area_passed", "plane_coordinate_cm",
    "radial_min_cm", "radial_max_cm",
    "global_coordinate_count", "total_element_count", "runtime_seconds",
    "worker_process_id", "mesh_position_hash", "mesh_line_validation_passed",
    "minimum_angle_not_worse", "maximum_aspect_improved_vs_legacy",
    "spacing_monotonic_passed", "quality_passed", "error_message",
)

PROFILE_BASE_FIELDS = (
    "schema_version", "mesh_level", "level_order", "run_role", "state_index",
    "bias_name", "contact", "region", "requested_distance_nm",
    "state", "ntrap_cm3", "target_ntrap_cm3",
    "target_VGS_V", "target_VDS_V", "target_VS_V",
    "actual_VGS_V", "actual_VDS_V", "actual_VS_V", "target_reached",
    "sample_available", "within_region_depth", "edge_index", "node0", "node1",
    "interpolation_fraction_n0_to_n1", "plane_intersection_relation",
    "profile_sample_semantics", "sample_r_cm", "sample_r_nm", "sample_z_cm",
    "sample_z_nm", "potential_V", "electrons_cm3", "electric_field_V_per_cm",
    "electron_current_A_per_cm2", "cylindrical_edge_couple_cm2",
    "electron_current_times_couple_A",
    "shallow_side_orientation_sign", "shallow_side_oriented_current_A",
    "edge_r0_cm", "edge_z0_cm", "edge_r1_cm", "edge_z1_cm",
    "local_axial_spacing_cm", "local_axial_spacing_nm", "plane_axis",
    "plane_coordinate_cm", "inward_coordinate_sign", "error_message",
)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _boolean(value: Any) -> bool:
    return bool(value) and value not in ("False", "false", "0")


def _error(error: object | None) -> str:
    if error is None:
        return ""
    text = " ".join(str(error).split())
    return f"{type(error).__name__}: {text}" if isinstance(error, BaseException) else text


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(_json_safe(value), stream, ensure_ascii=False, sort_keys=True,
                  separators=(",", ":"), allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def _csv_value(value: Any) -> Any:
    if value is None:
        return "NaN"
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN"
    return value


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one deterministic CSV without overwriting an existing file."""

    if path.exists():
        raise FileExistsError(f"refusing to overwrite candidate output: {path}")
    field_tuple = tuple(fields)
    unknown = sorted({key for row in rows for key in row if key not in field_tuple})
    if unknown:
        raise ValueError(f"CSV schema for {path.name} omits fields: {unknown}")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=field_tuple, lineterminator="\n")
        writer.writeheader()
        for source in rows:
            writer.writerow({field: _csv_value(source.get(field, "")) for field in field_tuple})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_payload(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_file_hash(paths: Sequence[Path]) -> str:
    payload = [
        (str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())), _sha256(path))
        for path in paths
    ]
    return _sha256_payload(payload)


def runtime_invariant_hashes(geometry: Mapping[str, Any]) -> dict[str, str]:
    active_keys = (
        "r_axis", "r_core", "r_mos2", "r_tox", "r_trap", "r_block",
        "r_gate_outer", "r_air_outer", "z_source", "z_drain",
    )
    active_geometry = {
        key: format(float(geometry[key]), ".17g") for key in active_keys
    }
    material_files = (
        REPOSITORY_ROOT / "05_program_erase" / "device_structure.py",
        REPOSITORY_ROOT / "compact_handoff_parameters.py",
        MODULE_DIRECTORY / "material_parameters.py",
        MODULE_DIRECTORY / "trap_parameters.py",
        MODULE_DIRECTORY / "trap_models.py",
        MODULE_DIRECTORY / "physics_models.py",
    )
    state_files = (
        MODULE_DIRECTORY / "state_characterization_config.py",
        MODULE_DIRECTORY / "trap_parameters.py",
        MODULE_DIRECTORY / "trap_models.py",
    )
    solver_files = (
        MODULE_DIRECTORY / "local_mesh_family.py",
        MODULE_DIRECTORY / "trap_parameters.py",
        MODULE_DIRECTORY / "state_sweep_helpers.py",
        MODULE_DIRECTORY / "run_memory_window.py",
        MODULE_DIRECTORY / "full_terminal_qc_candidate.py",
        MODULE_DIRECTORY / "run_full_terminal_qc_candidate.py",
        MODULE_DIRECTORY / "run_contact_topology_qc_candidate.py",
    )
    return {
        "active_geometry_sha256": _sha256_payload(active_geometry),
        "material_runtime_sha256": _aggregate_file_hash(material_files),
        "state_definition_sha256": _aggregate_file_hash(state_files),
        "solver_settings_sha256": _aggregate_file_hash(solver_files),
    }


def _contact_topology_signature(topology: Mapping[str, Any]) -> str:
    contacts: dict[str, dict[str, Any]] = {}
    for terminal in TERMINALS:
        report = topology.get("contacts", {}).get(terminal, {})
        contacts[terminal] = {
            "region": str(report.get("region", "")),
            "edge_count": int(report.get("edge_count", 0)),
            **{
                key: format(_number(report.get(key)), ".17g")
                for key in (
                    "runtime_area_cm2", "expected_area_cm2", "normal_r", "normal_z",
                    "plane_coordinate_cm", "radial_min_cm", "radial_max_cm",
                )
            },
        }
    return _sha256_payload(contacts)


def _protected_files() -> list[Path]:
    paths: list[Path] = []
    for root in (REPOSITORY_ROOT / "shared_data", REPOSITORY_ROOT / "05_program_erase",
                 REPOSITORY_ROOT / "_doc_review"):
        if root.is_dir():
            paths.extend(path for path in sorted(root.rglob("*")) if path.is_file())
    memory_window = MODULE_DIRECTORY / "run_memory_window.py"
    if memory_window.is_file():
        paths.append(memory_window)
    if RESULTS_DIRECTORY.is_dir():
        for child in sorted(RESULTS_DIRECTORY.iterdir()):
            if child == CANDIDATE_DIRECTORY or child.name.startswith(".local_mesh_"):
                continue
            if child.is_file():
                paths.append(child)
            elif child.is_dir():
                paths.extend(path for path in sorted(child.rglob("*")) if path.is_file())
    return sorted(set(path.resolve() for path in paths))


def capture_protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT.resolve())): _sha256(path)
        for path in _protected_files()
    }


def assert_protected_hashes(expected: Mapping[str, str]) -> None:
    actual = capture_protected_hashes()
    if dict(expected) != actual:
        changed = sorted(name for name in set(expected) | set(actual)
                         if expected.get(name) != actual.get(name))
        raise RuntimeError("protected input changed: " + ", ".join(changed))


def _percentile(values: Sequence[Any], fraction: float) -> float:
    ordered = sorted(value for value in (_number(item) for item in values) if math.isfinite(value))
    if not ordered:
        return math.nan
    position = fraction * (len(ordered) - 1)
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _runtime_counts(device: str) -> dict[str, int]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    coordinates: set[int] = set()
    elements = 0
    for region in get_region_list(device=device):
        coordinates.update(int(value) for value in get_node_model_values(
            device=device, region=region, name="coordinate_index"))
        elements += len(get_element_node_list(device=device, region=region))
    return {"global_coordinate_count": len(coordinates), "total_element_count": elements}


def _spacing_distribution(values: Sequence[Any], prefix: str) -> dict[str, float]:
    finite = sorted(value for value in (_number(item) for item in values)
                    if math.isfinite(value) and value > 0.0)
    return {
        f"{prefix}_min_cm": min(finite) if finite else math.nan,
        f"{prefix}_p01_cm": _percentile(finite, 0.01),
        f"{prefix}_p05_cm": _percentile(finite, 0.05),
        f"{prefix}_median_cm": _percentile(finite, 0.50),
        f"{prefix}_p95_cm": _percentile(finite, 0.95),
        f"{prefix}_p99_cm": _percentile(finite, 0.99),
        f"{prefix}_max_cm": max(finite) if finite else math.nan,
    }


def _mos2_spacing_rows(
    mesh_level: str, level_order: int, device: str,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Report actual shell spacing and inward contact intervals.

    The requested contact distances are fixed mesh anchors except 0.1 nm,
    which is used only by the solved profile audit.  Contact-distance quality
    rows use the exact inward interval at 0, 0.25, 0.5, 1, 2, and 5 nm.
    """

    from devsim import get_node_model_values

    radial = sorted(set(float(value) for value in get_node_model_values(
        device=device, region="MoS2", name="x")))
    axial = sorted(set(float(value) for value in get_node_model_values(
        device=device, region="MoS2", name="y")))
    radial_deltas = [right - left for left, right in zip(radial, radial[1:])]
    axial_deltas = [right - left for left, right in zip(axial, axial[1:])]
    common = {
        "schema_version": SCHEMA_VERSION,
        "mesh_level": mesh_level,
        "level_order": level_order,
        "region": "MoS2",
        "node_index": -1,
        "error_message": "",
    }
    rows: list[dict[str, Any]] = [{
        **common,
        "row_type": "mos2_radial_spacing",
        "contact": "",
        **_spacing_distribution(radial_deltas, "radial_spacing"),
        **_spacing_distribution(axial_deltas, "axial_spacing"),
        "quality_passed": bool(radial_deltas and axial_deltas),
    }]
    tolerance_cm = 1.0e-14
    for contact, endpoint, expected_endpoint, inward_sign in (
        ("source", axial[0], _number(geometry.get("z_source")), 1),
        ("drain", axial[-1], _number(geometry.get("z_drain")), -1),
    ):
        plane_matches = (
            math.isfinite(expected_endpoint)
            and abs(endpoint - expected_endpoint) <= tolerance_cm
        )
        for distance_nm in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
            target = expected_endpoint + inward_sign * distance_nm * 1.0e-7
            index = min(range(len(axial)), key=lambda item: abs(axial[item] - target))
            neighbour = index + inward_sign
            exact = abs(axial[index] - target) <= tolerance_cm
            spacing = (
                abs(axial[neighbour] - axial[index])
                if 0 <= neighbour < len(axial)
                else math.nan
            )
            rows.append({
                **common,
                "row_type": "contact_distance_spacing",
                "contact": contact,
                "distance_from_contact_nm": distance_nm,
                "target_coordinate_cm": target,
                "actual_coordinate_cm": axial[index],
                "coordinate_error_cm": axial[index] - target,
                "interval_start_cm": axial[index],
                "interval_end_cm": (
                    axial[neighbour] if 0 <= neighbour < len(axial) else math.nan
                ),
                "contact_plane_matches_geometry": plane_matches,
                "axial_spacing_min_cm": spacing,
                "axial_spacing_median_cm": spacing,
                "axial_spacing_max_cm": spacing,
                "quality_passed": (
                    plane_matches and exact and math.isfinite(spacing) and spacing > 0.0
                ),
                "error_message": "" if plane_matches and exact else (
                    "contact plane differs from runtime geometry"
                    if not plane_matches else "requested anchor is absent"
                ),
            })
    return rows


def _preview_geometry(memory_window: Any) -> dict[str, Any]:
    geometry = dict(memory_window.parameterized_device_structure.calculate_geometry(
        tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM))
    from state_sweep_helpers import validate_canonical_handoff_geometry
    validate_canonical_handoff_geometry(geometry)
    return geometry


def _quality_rows_for_runtime(
    mesh_level: str, level_order: int, device: str,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import devsim
    import qc_root_cause_metrics as metrics

    rows: list[dict[str, Any]] = []
    for region in sorted(devsim.get_region_list(device=device)):
        if region in {"Air", "GateMetal"}:
            continue
        contacts = ("source", "drain") if region == "MoS2" else ()
        report = metrics.runtime_mesh_quality_metrics(device, region, contacts=contacts)
        triangle = report["triangle_quality"]
        elements = triangle.get("elements", ())
        minimum_angles = [_number(row.get("minimum_angle_deg")) for row in elements]
        aspects = [_number(row.get("aspect_ratio")) for row in elements]
        spacing = report["actual_region_edge_spacing"]
        row = {
            "schema_version": SCHEMA_VERSION, "mesh_level": mesh_level,
            "level_order": level_order, "row_type": "region_summary", "region": region,
            "contact": "", "node_index": -1,
            "region_node_count": report["region_node_count"],
            "region_edge_count": report["region_edge_count"],
            "triangle_count": triangle["triangle_count"],
            "valid_triangle_count": triangle["valid_triangle_count"],
            "degenerate_triangle_count": triangle["degenerate_triangle_count"],
            "minimum_angle_deg": triangle["minimum_angle_deg"],
            "minimum_angle_p01_deg": _percentile(minimum_angles, 0.01),
            "minimum_angle_p05_deg": _percentile(minimum_angles, 0.05),
            "minimum_angle_median_deg": _percentile(minimum_angles, 0.5),
            "minimum_angle_p99_deg": _percentile(minimum_angles, 0.99),
            "maximum_angle_deg": triangle["maximum_angle_deg"],
            "aspect_ratio_minimum": _percentile(aspects, 0.0),
            "aspect_ratio_p01": _percentile(aspects, 0.01),
            "aspect_ratio_median": _percentile(aspects, 0.5),
            "aspect_ratio_p95": _percentile(aspects, 0.95),
            "aspect_ratio_p99": _percentile(aspects, 0.99),
            "maximum_aspect_ratio": triangle["maximum_aspect_ratio"],
            "obtuse_fraction": triangle["obtuse_fraction"],
            "radial_spacing_min_cm": spacing["radial_cm"]["minimum"],
            "radial_spacing_max_cm": spacing["radial_cm"]["maximum"],
            "axial_spacing_min_cm": spacing["axial_cm"]["minimum"],
            "axial_spacing_max_cm": spacing["axial_cm"]["maximum"],
            "quality_passed": (triangle["degenerate_triangle_count"] == 0
                               and math.isfinite(_number(triangle["minimum_angle_deg"]))),
            "error_message": "",
        }
        rows.append(row)
        for contact, data in report.get("contacts", {}).items():
            couple = data["incident_CylindricalEdgeCouple"]
            valence = data["active_incident_edge_valence"]
            valence_values = [
                _number(item.get("active_incident_edge_valence"))
                for item in data.get("node_valence", ())
            ]
            rows.append({
                "schema_version": SCHEMA_VERSION, "mesh_level": mesh_level,
                "level_order": level_order, "row_type": "contact_summary",
                "region": region, "contact": contact, "node_index": -1,
                "contact_node_count": data["contact_node_count"],
                "active_incident_edge_count": data["active_incident_edge_count"],
                "valence_minimum": valence["minimum"], "valence_maximum": valence["maximum"],
                "valence_p05": _percentile(valence_values, 0.05),
                "valence_median": _percentile(valence_values, 0.50),
                "valence_p95": _percentile(valence_values, 0.95),
                "incident_couple_minimum_cm2": couple["minimum"],
                "incident_couple_p05_cm2": couple["p05"],
                "incident_couple_median_cm2": couple["median"],
                "incident_couple_p95_cm2": couple["p95"],
                "incident_couple_maximum_cm2": couple["maximum"],
                "incident_couple_sum_cm2": couple["sum"],
                "quality_passed": data["active_incident_edge_count"] > 0,
                "error_message": "",
            })
            for node in data["node_valence"]:
                rows.append({
                    "schema_version": SCHEMA_VERSION, "mesh_level": mesh_level,
                    "level_order": level_order, "row_type": "contact_node", "region": region,
                    "contact": contact, "node_index": node["node_index"],
                    "valence_minimum": node["active_incident_edge_valence"],
                    "valence_maximum": node["active_incident_edge_valence"],
                    "radial_min_cm": node["r_cm"], "plane_coordinate_cm": node["z_cm"],
                    "quality_passed": node["active_incident_edge_valence"] > 0,
                    "error_message": "",
                })
        if region == "MoS2":
            rows.extend(_mos2_spacing_rows(
                mesh_level, level_order, device, geometry
            ))
    return rows


def _append_topology_rows(rows: list[dict[str, Any]], topology: Mapping[str, Any],
                          mesh_level: str, level_order: int) -> None:
    for contact, report in sorted(topology.get("contacts", {}).items()):
        geometry_passed = all((
            int(report.get("edge_count", 0)) > 0,
            bool(report.get("axisymmetric_models_consistent")),
            bool(report.get("region_consistent")),
            bool(report.get("plane_consistent")),
            bool(report.get("span_consistent")),
            bool(report.get("geometric_area_consistent")),
            bool(report.get("surface_model_consistent")),
            bool(report.get("normal_consistent")),
        ))
        rows.append({
            "schema_version": SCHEMA_VERSION, "mesh_level": mesh_level,
            "level_order": level_order, "row_type": "topology", "region": report.get("region", ""),
            "contact": contact, "node_index": -1, "edge_count": report.get("edge_count", 0),
            "geometric_area_cm2": report.get("geometric_area_cm2", math.nan),
            "runtime_area_cm2": report.get("runtime_area_cm2", math.nan),
            "expected_area_cm2": report.get("expected_area_cm2", math.nan),
            "contact_area_relative_error": report.get("area_relative_error", math.nan),
            "contact_area_passed": (
                math.isfinite(_number(report.get("area_relative_error")))
                and abs(_number(report.get("area_relative_error"))) < 1.0e-6
            ),
            "normal_r": report.get("normal_r", math.nan), "normal_z": report.get("normal_z", math.nan),
            "plane_coordinate_cm": report.get("plane_coordinate_cm", math.nan),
            "radial_min_cm": report.get("radial_min_cm", math.nan),
            "radial_max_cm": report.get("radial_max_cm", math.nan),
            # A mesh-only probe has no contact equations.  This row audits
            # topology/area/normal only; equation consistency is validated by
            # each solved reference-point worker.
            "quality_passed": geometry_passed and (
                math.isfinite(_number(report.get("area_relative_error")))
                and abs(_number(report.get("area_relative_error"))) < 1.0e-6
            ),
            "error_message": report.get("error_message", ""),
        })


def _legacy_fine_max_aspect() -> float:
    path = RESULTS_DIRECTORY / "qc_root_cause_diagnostics" / "mesh_quality_diagnostic.csv"
    if not path.is_file():
        return math.nan
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        values = [
            _number(row.get("triangle_maximum_aspect_ratio"))
            for row in rows
            if row.get("mesh_level") == "fine"
            and row.get("row_type") == "region_summary"
        ]
    finite = [value for value in values if math.isfinite(value)]
    return max(finite) if finite else math.nan


def evaluate_quality_family(
    quality_rows: list[dict[str, Any]], levels: Sequence[str]
) -> dict[str, dict[str, bool]]:
    """Apply the mesh-quality criteria and append auditable summary rows."""

    legacy_fine_aspect = _legacy_fine_max_aspect()
    summaries: dict[str, dict[str, bool]] = {}
    previous_minimum_angle = math.nan
    previous_spacing: dict[tuple[str, float], float] = {}
    previous_radial_median = math.nan
    for level_order, level in enumerate(levels):
        relevant = [row for row in quality_rows if row.get("mesh_level") == level]
        region_rows = [row for row in relevant if row.get("row_type") == "region_summary"]
        topology_rows = [
            row for row in relevant
            if row.get("row_type") == "topology"
            and row.get("contact") in {"source", "drain"}
        ]
        distance_rows = [
            row for row in relevant if row.get("row_type") == "contact_distance_spacing"
        ]
        radial_row = next(
            (row for row in relevant if row.get("row_type") == "mos2_radial_spacing"),
            {},
        )
        region_minimum_angles = [
            _number(row.get("minimum_angle_deg")) for row in region_rows
        ]
        region_maximum_aspects = [
            _number(row.get("maximum_aspect_ratio")) for row in region_rows
        ]
        minimum_angle = (
            min(region_minimum_angles)
            if region_minimum_angles and all(map(math.isfinite, region_minimum_angles))
            else math.nan
        )
        maximum_aspect = (
            max(region_maximum_aspects)
            if region_maximum_aspects and all(map(math.isfinite, region_maximum_aspects))
            else math.nan
        )
        minimum_angle_not_worse = (
            math.isfinite(minimum_angle)
            and (
                not math.isfinite(previous_minimum_angle)
                or minimum_angle >= previous_minimum_angle - 1.0e-9
            )
        )
        maximum_aspect_improved = (
            math.isfinite(maximum_aspect)
            and (
                math.isfinite(legacy_fine_aspect)
                and maximum_aspect <= 0.5 * legacy_fine_aspect
            )
        )
        expected_distance_keys = {
            (contact, distance)
            for contact in ("source", "drain")
            for distance in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0)
        }
        distance_keys = {
            (str(row.get("contact")), _number(row.get("distance_from_contact_nm")))
            for row in distance_rows
        }
        spacing_monotonic = (
            len(distance_rows) == len(expected_distance_keys)
            and distance_keys == expected_distance_keys
            and bool(radial_row)
        )
        current_spacing: dict[tuple[str, float], float] = {}
        for row in distance_rows:
            key = (str(row.get("contact")), _number(row.get("distance_from_contact_nm")))
            value = _number(row.get("axial_spacing_median_cm"))
            current_spacing[key] = value
            spacing_monotonic = spacing_monotonic and math.isfinite(value) and value > 0.0
            prior = previous_spacing.get(key)
            if prior is not None:
                if key[1] <= 2.0:
                    spacing_monotonic = spacing_monotonic and math.isclose(
                        value, 0.5 * prior, rel_tol=1.0e-6, abs_tol=1.0e-18
                    )
                else:
                    spacing_monotonic = (
                        spacing_monotonic and value <= prior + 1.0e-18
                    )
        radial_median = _number(radial_row.get("radial_spacing_median_cm"))
        spacing_monotonic = (
            spacing_monotonic and math.isfinite(radial_median) and radial_median > 0.0
        )
        if math.isfinite(previous_radial_median):
            spacing_monotonic = spacing_monotonic and math.isclose(
                radial_median, 0.5 * previous_radial_median,
                rel_tol=1.0e-6, abs_tol=1.0e-18,
            )
        base_quality = (
            bool(region_rows)
            and all(bool(row.get("quality_passed")) for row in region_rows)
            and len(topology_rows) == 2
            and all(bool(row.get("quality_passed")) for row in topology_rows)
            and all(bool(row.get("quality_passed")) for row in distance_rows)
            and bool(radial_row.get("quality_passed"))
        )
        passed = all((
            base_quality,
            minimum_angle_not_worse,
            maximum_aspect_improved,
            spacing_monotonic,
        ))
        summary = {
            "schema_version": SCHEMA_VERSION,
            "mesh_level": level,
            "level_order": level_order,
            "row_type": "family_quality_summary",
            "region": "all_active_regions",
            "contact": "",
            "node_index": -1,
            "minimum_angle_deg": minimum_angle,
            "maximum_aspect_ratio": maximum_aspect,
            "minimum_angle_not_worse": minimum_angle_not_worse,
            "maximum_aspect_improved_vs_legacy": maximum_aspect_improved,
            "spacing_monotonic_passed": spacing_monotonic,
            "quality_passed": passed,
            "error_message": "" if passed else "one or more mesh-quality criteria failed",
        }
        quality_rows.append(summary)
        summaries[level] = {"passed": passed}
        previous_minimum_angle = minimum_angle
        previous_spacing = current_spacing
        previous_radial_median = radial_median
    return summaries


@contextmanager
def _custom_legacy_validator(family: Any, level: str,
                             local_line_log: list[dict[str, Any]]) -> Iterator[None]:
    import run_contact_topology_qc_candidate as contact_runner

    original = contact_runner.validate_topology_mesh_line_log

    def validate(rows: Sequence[Mapping[str, Any]], geometry: Mapping[str, Any],
                 scale: float) -> dict[str, Any]:
        base = original(rows, geometry, scale)
        local = family.validate_local_mesh_line_log(local_line_log, geometry, level)
        if not bool(local.get("passed")):
            raise RuntimeError("local mesh-line validation failed: " + ", ".join(local.get("errors", ())))
        return {**dict(base), "local_mesh_validation": local, "passed": True}

    contact_runner.validate_topology_mesh_line_log = validate
    try:
        yield
    finally:
        contact_runner.validate_topology_mesh_line_log = original


def run_probe_worker(mesh_level: str) -> dict[str, Any]:
    import contact_topology_qc
    import local_mesh_family as family
    import run_memory_window as memory_window

    level_order = family.LEVELS.index(mesh_level)
    preview = _preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure, preview, mesh_level, line_log
    ):
        geometry = memory_window.create_structure(
            tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM)
    validation = family.validate_local_mesh_line_log(line_log, geometry, mesh_level)
    if not bool(validation.get("passed")):
        raise RuntimeError("local mesh validation failed: " + ", ".join(validation.get("errors", ())))
    rows = _quality_rows_for_runtime(
        mesh_level, level_order, memory_window.device, geometry
    )
    topology = contact_topology_qc.audit_runtime_contact_topology(geometry, device=memory_window.device)
    _append_topology_rows(rows, topology, mesh_level, level_order)
    counts = _runtime_counts(memory_window.device)
    elapsed = time.perf_counter() - started
    for row in rows:
        row.update({
            "global_coordinate_count": counts["global_coordinate_count"],
            "total_element_count": counts["total_element_count"],
            "runtime_seconds": elapsed, "worker_process_id": os.getpid(),
            "mesh_position_hash": validation.get("position_hash", ""),
            "mesh_line_validation_passed": True,
        })
    return {
        "mode": "probe", "mesh_level": mesh_level, "level_order": level_order,
        "geometry": dict(geometry), "mesh_line_log": line_log,
        "mesh_line_validation": validation, "quality_rows": rows,
        "topology": topology,
        "runtime_invariant_hashes": runtime_invariant_hashes(geometry),
        "mesh_counts": counts, "runtime_seconds": elapsed,
        "worker_process_id": os.getpid(), "error_message": "",
    }


def _supported_sensitivities(point: Mapping[str, Any]) -> dict[str, float | bool]:
    matrices = point.get("matrices", {})
    result: dict[str, float | bool] = {}
    values: list[float] = []
    for perturbed, field in zip(TERMINALS, SUPPORTED_CAPACITANCES):
        samples_by_delta: dict[float, float] = {}
        nominal = math.nan
        for delta_text, data in matrices.items():
            delta = _number(delta_text)
            value = _number(data.get("values", {}).get(f"gate:{perturbed}"))
            if math.isfinite(value):
                samples_by_delta[delta] = value
            if math.isclose(delta, NOMINAL_DELTA_V, rel_tol=0.0, abs_tol=1.0e-15):
                nominal = value
        complete = all(any(math.isclose(delta, required, rel_tol=0.0, abs_tol=1.0e-15)
                           for delta in samples_by_delta)
                       for required in DELTA_VOLTAGES_V)
        sensitivity = (
            max((abs(value - nominal) for value in samples_by_delta.values()), default=math.inf)
            / max(abs(nominal), CAPACITANCE_SENSITIVITY_FLOOR_F)
            if complete and math.isfinite(nominal)
            else math.inf
        )
        key = field.replace("_F", "_delta_sensitivity")
        result[key] = sensitivity
        values.append(sensitivity)
    maximum = max(values, default=math.inf)
    result["supported_delta_sensitivity_max"] = maximum
    result["supported_delta_sensitivity_passed"] = math.isfinite(maximum) and maximum <= 0.01
    return result


def _point_row(bundle: Mapping[str, Any]) -> dict[str, Any]:
    point = next(iter(bundle.get("points", ())), {})
    identity = bundle.get("worker_identity", {})
    level = str(bundle.get("mesh_level", identity.get("mesh_level", "")))
    run_role = str(bundle.get("run_role", "primary"))
    target = next((item for item in REFERENCE_POINTS
                   if int(item["state_index"]) == int(bundle.get("state_index", point.get("state_index", -1)))
                   and item["bias_name"] == str(bundle.get("bias_name", point.get("bias_name", "")))), {})
    actual = bundle.get("actual_biases_after_worker", {})
    source_v = _number(actual.get("source"))
    actual_vgs = _number(actual.get("gate")) - source_v
    actual_vds = _number(actual.get("drain")) - source_v
    actual_vs = source_v
    bias_target_reached = all(
        math.isclose(value, _number(target.get(name)), rel_tol=0.0,
                     abs_tol=BIAS_ATOL_V)
        for value, name in (
            (actual_vgs, "target_VGS_V"),
            (actual_vds, "target_VDS_V"),
            (actual_vs, "target_VS_V"),
        )
    )
    actual_ntrap = _number(point.get("ntrap_cm3"))
    target_ntrap = _number(target.get("target_ntrap_cm3"))
    trap_target_reached = (
        math.isfinite(actual_ntrap)
        and math.isfinite(target_ntrap)
        and math.isclose(actual_ntrap, target_ntrap, rel_tol=1.0e-12, abs_tol=0.0)
    )
    target_reached = bias_target_reached and trap_target_reached
    currents = bundle.get("contact_currents_A", {})
    source_current = _number(currents.get("source"))
    drain_current = _number(currents.get("drain"))
    residual = source_current + drain_current
    scale = max(abs(source_current), abs(drain_current))
    tolerance = CURRENT_CONTINUITY_ATOL_A + CURRENT_CONTINUITY_RTOL * scale
    direct = point.get("direct", {})
    matrix = point.get("nominal_matrix_F", {})
    row: dict[str, Any] = {field: "" for field in POINT_FIELDS}
    row.update({
        "schema_version": SCHEMA_VERSION, "mesh_level": level,
        "level_order": bundle.get("level_order", ""), "run_role": run_role,
        "state_index": point.get("state_index", bundle.get("state_index", -1)),
        "state": point.get("state", ""), "bias_name": point.get("bias_name", bundle.get("bias_name", "")),
        "ntrap_cm3": actual_ntrap, **target,
        "actual_VGS_V": actual_vgs, "actual_VDS_V": actual_vds, "actual_VS_V": actual_vs,
        "bias_target_reached": bias_target_reached,
        "trap_target_reached": trap_target_reached,
        "target_reached": target_reached, "ID_A": drain_current,
        "drain_current_A": drain_current, "source_current_A": source_current,
        "source_drain_continuity_residual_A": residual, "continuity_scale_A": scale,
        "continuity_tolerance_A": tolerance,
        "continuity_passed": math.isfinite(residual) and abs(residual) <= tolerance,
        "Qg_C": _number(direct.get("Qg_contact_C")),
        "raw_Qd_C": _number(direct.get("Qd_contact_C")),
        "raw_Qs_C": _number(direct.get("Qs_contact_C")),
        "Qmobile_C": _number(direct.get("Qmobile_C")),
        "Qtrap_C": _number(direct.get("Qtrap_C")),
        "Qfixed_C": _number(direct.get("Qfixed_C")),
        "Qnoncontact_boundary_C": 0.0,
        "global_coordinate_count": bundle.get("diagnostic_mesh_counts", {}).get("global_coordinate_count", ""),
        "triangle_count": bundle.get("diagnostic_mesh_counts", {}).get("triangle_count", ""),
        "runtime_seconds": bundle.get("runtime_seconds", math.nan),
        "worker_process_id": bundle.get("worker_process_id", identity.get("process_id", "")),
        "mesh_position_hash": bundle.get("mesh_line_validation", {}).get("position_hash", ""),
        "mesh_line_validation_passed": bundle.get("mesh_line_validation", {}).get("passed", False),
        **dict(bundle.get("runtime_invariant_hashes", {})),
        "family_position_hash_sha256": bundle.get("family_position_hash_sha256", ""),
        "runtime_invariants_passed": bundle.get("runtime_invariants_passed", False),
        "final_reference_validation_passed": bundle.get("final_reference_validation", {}).get("passed", False),
        "qc_point_passed": point.get("passed", False),
    })
    for measured in TERMINALS:
        for perturbed in TERMINALS:
            row[f"C{measured[0]}{perturbed[0]}_F"] = _number(matrix.get(f"{measured}:{perturbed}"))
    row.update(_supported_sensitivities(point))
    gauss = point.get("global_gauss", {})
    row.update({
        "global_gauss_residual_C": gauss.get("residual_C", math.nan),
        "global_gauss_scale_C": gauss.get("scale_C", math.nan),
        "global_gauss_tolerance_C": gauss.get("tolerance_C", math.nan),
        "global_gauss_passed": gauss.get("passed", False),
    })
    derivative_rows = tuple(point.get("derivative_gauss_rows", ()))
    derivative_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    duplicate_derivative_key = False
    for item in derivative_rows:
        terminal = str(item.get("perturbed_terminal", ""))
        delta = _number(item.get("delta_voltage_V"))
        label = next((text for expected, text in DELTA_LABELS
                      if math.isclose(delta, expected, rel_tol=0.0,
                                      abs_tol=1.0e-15)), "")
        key = (terminal, label)
        if not label or terminal not in TERMINALS or key in derivative_by_key:
            duplicate_derivative_key = True
            continue
        derivative_by_key[key] = item
    expected_derivative_keys = {
        (terminal, label) for _, label in DELTA_LABELS for terminal in TERMINALS
    }
    derivative_all = (
        not duplicate_derivative_key
        and set(derivative_by_key) == expected_derivative_keys
        and all(bool(item.get("passed", False))
                for item in derivative_by_key.values())
    )
    normalized_derivative_residuals = [
        abs(_number(item.get("residual_F")))
        / max(abs(_number(item.get("tolerance_F"))), 1.0e-300)
        for item in derivative_rows
        if math.isfinite(_number(item.get("residual_F")))
        and math.isfinite(_number(item.get("tolerance_F")))
    ]
    nominal_derivatives = {
        terminal: derivative_by_key.get((terminal, "0p001"), {})
        for terminal in TERMINALS
    }
    for _, label in DELTA_LABELS:
        for terminal in TERMINALS:
            item = derivative_by_key.get((terminal, label), {})
            prefix = f"derivative_gauss_{terminal}_dv_{label}"
            row[f"{prefix}_residual_F"] = item.get("residual_F", math.nan)
            row[f"{prefix}_tolerance_F"] = item.get("tolerance_F", math.nan)
            row[f"{prefix}_passed"] = bool(item.get("passed", False))
    for terminal in TERMINALS:
        item = nominal_derivatives.get(terminal, {})
        passed = bool(item.get("passed", False))
        row[f"derivative_gauss_{terminal}_residual_F"] = item.get("residual_F", math.nan)
        row[f"derivative_gauss_{terminal}_tolerance_F"] = item.get("tolerance_F", math.nan)
        row[f"derivative_gauss_{terminal}_passed"] = passed
    row["derivative_gauss_row_count"] = len(derivative_rows)
    row["derivative_gauss_max_normalized_residual"] = max(
        normalized_derivative_residuals, default=math.inf
    )
    row["derivative_gauss_passed"] = derivative_all
    gauge = point.get("gauge_row_validation", {})
    gauge_all = bool(gauge.get("passed", False))
    for terminal in TERMINALS:
        item = gauge.get("rows", {}).get(terminal, {})
        zero = item.get("row_vs_zero", {})
        row[f"gauge_{terminal}_row_sum_F"] = item.get("row_sum_F", math.nan)
        row[f"gauge_{terminal}_tolerance_F"] = zero.get("tolerance", math.nan)
        row[f"gauge_{terminal}_passed"] = item.get("passed", False)
    row["gauge_row_sum_passed"] = gauge_all
    topology = bundle.get("topology", {})
    for terminal in ("source", "drain"):
        report = topology.get("contacts", {}).get(terminal, {})
        row[f"{terminal}_edge_count"] = report.get("edge_count", 0)
        row[f"{terminal}_area_cm2"] = report.get("runtime_area_cm2", math.nan)
    row["topology_passed"] = topology.get("passed", False)
    errors = [str(point.get("error_message", "")), str(bundle.get("error_message", ""))]
    row["error_message"] = "; ".join(dict.fromkeys(item for item in errors if item))
    row["converged"] = all((
        target_reached,
        row["continuity_passed"],
        bool(row["final_reference_validation_passed"]),
        bool(row["topology_passed"]),
        bool(row["global_gauss_passed"]),
        bool(row["derivative_gauss_passed"]),
        bool(row["gauge_row_sum_passed"]),
        bool(row["supported_delta_sensitivity_passed"]),
        bool(row["runtime_invariants_passed"]),
        math.isfinite(_number(row["ID_A"])),
        all(math.isfinite(_number(row[name])) for name in (
            "Qg_C", "raw_Qd_C", "raw_Qs_C", "Cgg_F", "Cgd_F", "Cgs_F",
        )),
    ))
    return row


def _profile_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    point = next(iter(bundle.get("points", ())), {})
    actual = bundle.get("actual_biases_after_worker", {})
    source_voltage = _number(actual.get("source"))
    actual_vgs = _number(actual.get("gate")) - source_voltage
    actual_vds = _number(actual.get("drain")) - source_voltage
    reference = next((
        item for item in REFERENCE_POINTS
        if int(item["state_index"]) == int(bundle.get("state_index", -1))
        and item["bias_name"] == str(bundle.get("bias_name", ""))
    ), {})
    target_reached = all(
        math.isclose(actual_value, _number(reference.get(target_name)),
                     rel_tol=0.0, abs_tol=BIAS_ATOL_V)
        for actual_value, target_name in (
            (actual_vgs, "target_VGS_V"),
            (actual_vds, "target_VDS_V"),
            (source_voltage, "target_VS_V"),
        )
    )
    for source in bundle.get("profiles", ()):
        row = {field: "" for field in PROFILE_BASE_FIELDS}
        sample_r_cm = _number(source.get("r_cm"))
        sample_z_cm = _number(source.get("z_cm"))
        row.update({
            "schema_version": SCHEMA_VERSION, "mesh_level": bundle.get("mesh_level", ""),
            "level_order": bundle.get("level_order", ""), "run_role": bundle.get("run_role", "primary"),
            "state_index": bundle.get("state_index", -1), "bias_name": bundle.get("bias_name", ""),
            "state": point.get("state", ""),
            "ntrap_cm3": point.get("ntrap_cm3", math.nan),
            **reference,
            "actual_VGS_V": actual_vgs, "actual_VDS_V": actual_vds,
            "actual_VS_V": source_voltage, "target_reached": target_reached,
            **{field: source.get(field, "") for field in PROFILE_BASE_FIELDS if field in source},
            "profile_sample_semantics": (
                "Potential/Electrons are linearly interpolated at the contact-normal "
                "plane crossing; ElectricField/ElectronCurrent/CylindricalEdgeCouple "
                "are incident-edge model values"
            ),
            "sample_r_cm": sample_r_cm,
            "sample_r_nm": sample_r_cm / 1.0e-7,
            "sample_z_cm": sample_z_cm,
            "sample_z_nm": sample_z_cm / 1.0e-7,
            "potential_V": _number(source.get("Potential_V")),
            "electrons_cm3": _number(source.get("Electrons_cm3")),
            "electric_field_V_per_cm": _number(source.get("ElectricField_V_per_cm")),
            "electron_current_A_per_cm2": _number(source.get("ElectronCurrent")),
            "cylindrical_edge_couple_cm2": _number(source.get("CylindricalEdgeCouple")),
            "electron_current_times_couple_A": _number(
                source.get("ElectronCurrent_times_CylindricalEdgeCouple_A")
            ),
        })
        z0, z1 = _number(source.get("edge_z0_cm")), _number(source.get("edge_z1_cm"))
        spacing = abs(z1 - z0) if math.isfinite(z0) and math.isfinite(z1) else math.nan
        row["local_axial_spacing_cm"] = spacing
        row["local_axial_spacing_nm"] = spacing / 1.0e-7
        result.append(row)
    return result


def run_point_worker(mesh_level: str, state_index: int, bias_name: str,
                     run_role: str = "primary") -> dict[str, Any]:
    import devsim
    import local_mesh_family as family
    import qc_root_cause_metrics as root_metrics
    import run_contact_topology_qc_candidate as contact_runner
    import run_memory_window as memory_window

    level_order = family.LEVELS.index(mesh_level)
    preview = _preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure, preview, mesh_level, line_log
    ):
        with _custom_legacy_validator(family, mesh_level, line_log):
            bundle = contact_runner.run_point_worker(
                mesh_level=mesh_level, mesh_scale=1.0,
                state_index=int(state_index), bias_name=str(bias_name))
    validation = family.validate_local_mesh_line_log(line_log, bundle.get("geometry", preview), mesh_level)
    if not bool(validation.get("passed")):
        raise RuntimeError("local mesh validation failed: " + ", ".join(validation.get("errors", ())))
    try:
        absolute_biases = contact_runner._read_biases(contact_runner.legacy_qc.DEVICE_NAME)
    except Exception:
        absolute_biases = {terminal: math.nan for terminal in TERMINALS}
    currents: dict[str, float] = {}
    for terminal in ("source", "drain"):
        try:
            currents[terminal] = float(devsim.get_contact_current(
                device=memory_window.device, contact=terminal,
                equation="ElectronContinuityEquation"))
        except Exception:
            currents[terminal] = math.nan
    try:
        quality = _quality_rows_for_runtime(
            mesh_level, level_order, memory_window.device,
            bundle.get("geometry", preview),
        )
    except Exception as error:
        quality = [{"schema_version": SCHEMA_VERSION, "mesh_level": mesh_level,
                    "level_order": level_order, "row_type": "error", "region": "MoS2",
                    "contact": "", "node_index": -1, "quality_passed": False,
                    "error_message": _error(error)}]
    profiles: list[dict[str, Any]] = []
    if int(state_index) == 0 and str(bias_name) == "on" and str(run_role) == "primary":
        for contact in ("source", "drain"):
            try:
                profiles.extend(root_metrics.extract_contact_spatial_profile_rows(
                    memory_window.device, contact, region="MoS2",
                    distances_nm=PROFILE_DISTANCES_NM))
            except Exception as error:
                profiles.append({"contact": contact, "region": "MoS2",
                                 "sample_available": False, "edge_index": -1,
                                 "error_message": _error(error)})
    counts = contact_runner._runtime_mesh_counts(memory_window.device, bundle.get("topology", {}))
    bundle.update({
        "mode": "point", "mesh_level": mesh_level, "level_order": level_order,
        "state_index": int(state_index), "bias_name": str(bias_name), "run_role": str(run_role),
        "actual_biases_after_worker": absolute_biases, "contact_currents_A": currents,
        "diagnostic_mesh_counts": counts, "quality_rows": quality, "profiles": profiles,
        "mesh_line_log": line_log, "mesh_line_validation": validation,
        "runtime_invariant_hashes": runtime_invariant_hashes(
            bundle.get("geometry", preview)
        ),
        "runtime_seconds": time.perf_counter() - started, "worker_process_id": os.getpid(),
    })
    return bundle


def build_tasks() -> list[dict[str, Any]]:
    import local_mesh_family as family

    tasks = [{"mode": "probe", "mesh_level": level, "run_role": "probe"}
             for level in family.LEVELS]
    for level in family.LEVELS:
        for reference in REFERENCE_POINTS:
            tasks.append({"mode": "point", "mesh_level": level, "run_role": "primary",
                          "state_index": reference["state_index"], "bias_name": reference["bias_name"]})
    for reference in REFERENCE_POINTS:
        tasks.append({"mode": "point", "mesh_level": "local_ultra_fine", "run_role": "repeat",
                      "state_index": reference["state_index"], "bias_name": reference["bias_name"]})
    return tasks


def _task_stem(task: Mapping[str, Any]) -> str:
    if task["mode"] == "probe":
        return f"probe_{task['mesh_level']}"
    return (f"point_{task['mesh_level']}_{task['run_role']}_"
            f"s{int(task['state_index'])}_{task['bias_name']}")


def build_worker_command(task: Mapping[str, Any], output_path: Path,
                         python_executable: str = sys.executable) -> list[str]:
    command = [str(python_executable), "-B", str(Path(__file__).resolve()),
               "--worker-mode", str(task["mode"]), "--mesh-level", str(task["mesh_level"]),
               "--run-role", str(task.get("run_role", "primary")),
               "--worker-output", str(output_path.resolve())]
    if task["mode"] == "point":
        command.extend(["--state-index", str(int(task["state_index"])),
                        "--bias-name", str(task["bias_name"])])
    return command


def _failed_bundle(task: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {**dict(task), "points": [{"state_index": task.get("state_index", -1),
                                      "bias_name": task.get("bias_name", ""),
                                      "error_message": error}],
            "quality_rows": [], "profiles": [], "error_message": error,
            "runtime_seconds": math.nan, "worker_process_id": -1}


def run_fresh_workers(worker_directory: Path, *, jobs: int = 1,
                      python_executable: str = sys.executable,
                      subprocess_run: Callable[..., Any] = subprocess.run) -> list[dict[str, Any]]:
    worker_directory.mkdir(parents=True, exist_ok=False)
    tasks = build_tasks()

    def execute(task: Mapping[str, Any]) -> dict[str, Any]:
        stem = _task_stem(task)
        output = worker_directory / f"{stem}.json"
        log_path = worker_directory / f"{stem}.log"
        command = build_worker_command(task, output, python_executable)
        if subprocess_run is subprocess.run:
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                completed = subprocess_run(command, cwd=REPOSITORY_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT, check=False)
        else:
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        code = int(getattr(completed, "returncode", 0))
        if code or not output.is_file():
            tail = ""
            if log_path.is_file():
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
            return _failed_bundle(task, f"worker return code {code}: {tail}".strip())
        with output.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    if int(jobs) <= 1:
        bundles = []
        for index, task in enumerate(tasks, start=1):
            bundle = execute(task)
            bundles.append(bundle)
            print(f"local-mesh worker {index}/{len(tasks)} {_task_stem(task)}", flush=True)
        return bundles
    bundles_by_stem: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            bundles_by_stem[_task_stem(task)] = future.result()
            print(f"local-mesh worker {index}/{len(tasks)} {_task_stem(task)}", flush=True)
    return [bundles_by_stem[_task_stem(task)] for task in tasks]


def build_full_dc_worker_command(
    task: Mapping[str, Any], output_path: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    return [
        str(python_executable), "-B", str(Path(__file__).resolve()),
        "--worker-mode", "full_dc",
        "--mesh-level", str(task["mesh_level"]),
        "--sweep-kind", str(task["sweep_kind"]),
        "--worker-output", str(output_path.resolve()),
    ]


def run_fresh_full_dc_workers(
    worker_directory: Path, gate: Mapping[str, Any], *,
    python_executable: str = sys.executable,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    """Run only gate-authorized extra/ultra full sweeps in fresh processes."""

    import local_mesh_full_dc as full_dc

    tasks = full_dc.build_full_dc_tasks(gate)
    if not tasks:
        return []
    worker_directory.mkdir(parents=True, exist_ok=False)
    bundles: list[dict[str, Any]] = []
    for index, task in enumerate(tasks, start=1):
        stem = f"full_dc_{task['mesh_level']}_{task['sweep_kind']}"
        output = worker_directory / f"{stem}.json"
        log_path = worker_directory / f"{stem}.log"
        command = build_full_dc_worker_command(task, output, python_executable)
        if subprocess_run is subprocess.run:
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                completed = subprocess_run(
                    command, cwd=REPOSITORY_ROOT, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
        else:
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        code = int(getattr(completed, "returncode", 0))
        if code or not output.is_file():
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
                )
            bundles.append({
                **dict(task), "mode": "full_dc", "passed": False,
                "raw_rows": [], "metrics_rows": [],
                "error_message": f"worker return code {code}: {tail}".strip(),
            })
        else:
            with output.open("r", encoding="utf-8") as stream:
                bundles.append(json.load(stream))
        print(f"local-mesh full-DC worker {index}/{len(tasks)} {stem}", flush=True)
    return bundles


def dispatch_full_dc_if_eligible(gate: Mapping[str, Any], *,
                                 dispatcher: Callable[[], Any] | None = None) -> dict[str, Any]:
    """Dispatch only an explicitly candidate-scoped full sweep after a pass.

    There is intentionally no default public-writer fallback.  A passing gate
    without a candidate dispatcher fails loudly instead of claiming a full DC
    result that was never generated.
    """

    if not (
        bool(gate.get("passed"))
        and bool(gate.get("full_sweep_allowed"))
    ):
        return {"executed": False, "reason": "final reference gate did not pass"}
    if dispatcher is None:
        raise RuntimeError("final gate passed, but no candidate-only full-DC dispatcher is installed")
    return {"executed": True, "result": dispatcher()}


def _dynamic_fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = ()) -> tuple[str, ...]:
    keys = {key for row in rows for key in row}
    return tuple(dict.fromkeys((*preferred, *sorted(keys - set(preferred)))))


def validate_profile_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import local_mesh_family as family

    expected = {
        (level, contact, distance)
        for level in family.LEVELS
        for contact in ("source", "drain")
        for distance in PROFILE_DISTANCES_NM
    }
    available: set[tuple[str, str, float]] = set()
    nonfinite_sample_count = 0
    for row in rows:
        if not _boolean(row.get("sample_available")):
            continue
        key = (
            str(row.get("mesh_level", "")),
            str(row.get("contact", "")),
            _number(row.get("requested_distance_nm")),
        )
        available.add(key)
        required_values = (
            row.get("potential_V"), row.get("electrons_cm3"),
            row.get("electric_field_V_per_cm"),
            row.get("electron_current_A_per_cm2"),
            row.get("local_axial_spacing_cm"),
        )
        if not all(math.isfinite(_number(value)) for value in required_values):
            nonfinite_sample_count += 1
    missing = sorted(expected - available)
    unexpected = sorted(available - expected)
    passed = not missing and not unexpected and nonfinite_sample_count == 0
    return {
        "expected_profile_count": len(expected),
        "available_profile_count": len(available & expected),
        "missing_profile_count": len(missing),
        "unexpected_profile_count": len(unexpected),
        "nonfinite_sample_count": nonfinite_sample_count,
        "passed": passed,
        "error_message": "; ".join(
            item for item in (
                f"missing profile planes={len(missing)}" if missing else "",
                f"unexpected profile planes={len(unexpected)}" if unexpected else "",
                f"nonfinite profile samples={nonfinite_sample_count}"
                if nonfinite_sample_count else "",
            ) if item
        ),
    }


def generate_full_dc_outputs(
    staging: Path, gate: Mapping[str, Any], *,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run and write the two candidate-only full-DC artifacts."""

    import local_mesh_full_dc as full_dc

    bundles = run_fresh_full_dc_workers(
        staging / "full_dc_workers", gate,
        python_executable=python_executable,
    )
    assembled = full_dc.assemble_full_dc_outputs(bundles)
    metric_output_rows: list[dict[str, Any]] = []
    for source in assembled.get("raw_rows", ()):
        metric_output_rows.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": f"{source.get('sweep_kind', '')}_point",
            **dict(source),
        })
    for source in assembled.get("metrics_rows", ()):
        metric_output_rows.append({
            "schema_version": SCHEMA_VERSION,
            "record_type": "extracted_metric",
            **dict(source),
        })

    convergence_output_rows: list[dict[str, Any]] = [
        {"schema_version": SCHEMA_VERSION, "record_type": "dc_convergence", **dict(source)}
        for source in assembled.get("convergence_rows", ())
    ]
    convergence_output_rows.extend(
        {"schema_version": SCHEMA_VERSION, "record_type": "worker_validation", **dict(source)}
        for source in assembled.get("validation_rows", ())
    )
    convergence_output_rows.append({
        "schema_version": SCHEMA_VERSION,
        "record_type": "full_dc_summary",
        "reference_mesh": "local_extra_fine",
        "candidate_mesh": "local_ultra_fine",
        "passed": bool(assembled.get("passed")),
        "status": assembled.get("status", ""),
        "raw_row_count": len(assembled.get("raw_rows", ())),
        "metric_row_count": len(assembled.get("metrics_rows", ())),
        "error_message": "; ".join(map(str, assembled.get("errors", ()))),
    })
    write_csv(
        staging / FULL_DC_OUTPUT_FILENAMES[0],
        _dynamic_fields(metric_output_rows, (
            "schema_version", "record_type", "mesh_level", "sweep_kind",
            "state_index", "state", "VGS_V", "VDS_V", "ID_A",
        )),
        metric_output_rows,
    )
    write_csv(
        staging / FULL_DC_OUTPUT_FILENAMES[1],
        _dynamic_fields(convergence_output_rows, (
            "schema_version", "record_type", "reference_mesh", "candidate_mesh",
            "state_index", "state", "VGS_V", "VDS_V", "metric", "passed",
            "status", "error_message",
        )),
        convergence_output_rows,
    )
    return {
        "passed": bool(assembled.get("passed")),
        "status": assembled.get("status", ""),
        "raw_row_count": len(assembled.get("raw_rows", ())),
        "metric_row_count": len(assembled.get("metrics_rows", ())),
        "convergence_row_count": len(assembled.get("convergence_rows", ())),
        "errors": list(assembled.get("errors", ())),
    }


def generate_candidate(staging: Path, *, jobs: int = 1,
                       python_executable: str = sys.executable) -> dict[str, Any]:
    import local_mesh_convergence as convergence
    import local_mesh_family as family

    staging.mkdir(parents=True, exist_ok=False)
    bundles = run_fresh_workers(staging / "workers", jobs=jobs,
                                python_executable=python_executable)
    probes = [bundle for bundle in bundles if bundle.get("mode") == "probe"]
    points = [bundle for bundle in bundles if bundle.get("mode") == "point"]
    if len(probes) != len(family.LEVELS):
        raise RuntimeError(
            f"expected {len(family.LEVELS)} mesh probes, received {len(probes)}"
        )
    geometry = next((bundle.get("geometry") for bundle in probes if bundle.get("geometry")), None)
    if geometry is None:
        raise RuntimeError("all local-mesh probes failed; family definition unavailable")
    invariant_hashes = runtime_invariant_hashes(geometry)
    probe_position_hashes: set[str] = set()
    for bundle in probes:
        level = str(bundle.get("mesh_level", ""))
        probe_geometry = bundle.get("geometry")
        if not isinstance(probe_geometry, Mapping):
            raise RuntimeError(f"{level}: mesh probe did not return runtime geometry")
        if runtime_invariant_hashes(probe_geometry) != invariant_hashes:
            raise RuntimeError(f"{level}: geometry/material/state/solver invariant mismatch")
        validation = bundle.get("mesh_line_validation", {})
        if not bool(validation.get("passed")):
            raise RuntimeError(f"{level}: mesh-line validation failed")
        probe_position_hashes.add(str(validation.get("position_hash", "")))
    if "" in probe_position_hashes or len(probe_position_hashes) != 1:
        raise RuntimeError("mesh refinement positions differ between family members")
    family_position_hash = next(iter(probe_position_hashes))
    probe_by_level = {str(bundle.get("mesh_level")): bundle for bundle in probes}
    for bundle in bundles:
        level = str(bundle.get("mesh_level", ""))
        worker_errors: list[str] = []
        bundle_geometry = bundle.get("geometry")
        if not isinstance(bundle_geometry, Mapping):
            worker_errors.append("runtime geometry unavailable")
        elif runtime_invariant_hashes(bundle_geometry) != invariant_hashes:
            worker_errors.append("runtime geometry/material/state/solver hashes differ")
        if dict(bundle.get("runtime_invariant_hashes", {})) != invariant_hashes:
            worker_errors.append("worker-recorded runtime invariant hashes differ")
        validation = bundle.get("mesh_line_validation", {})
        if (
            not bool(validation.get("passed"))
            or str(validation.get("position_hash", "")) != family_position_hash
        ):
            worker_errors.append("mesh-line validation/position hash differs")
        expected_probe = probe_by_level.get(level, {})
        expected_topology = expected_probe.get("topology", {})
        actual_topology = bundle.get("topology", {})
        if (
            not expected_topology
            or not actual_topology
            or _contact_topology_signature(actual_topology)
            != _contact_topology_signature(expected_topology)
        ):
            worker_errors.append("contact topology differs from level probe")
        bundle["runtime_invariants_passed"] = not worker_errors
        bundle["family_position_hash_sha256"] = family_position_hash
        if worker_errors:
            message = "; ".join(worker_errors)
            prior = str(bundle.get("error_message", "")).strip()
            bundle["error_message"] = "; ".join(item for item in (prior, message) if item)
    family_rows = family.local_mesh_family_definition_rows(geometry)
    for row in family_rows:
        row.update({
            "schema_version": SCHEMA_VERSION,
            **invariant_hashes,
            "family_position_hash_sha256": family_position_hash,
            "spacing_only_difference_passed": True,
        })
    quality_rows = [dict(row) for bundle in probes for row in bundle.get("quality_rows", ())]
    point_rows = [_point_row(bundle) for bundle in points]
    profile_rows = [row for bundle in points for row in _profile_rows(bundle)]
    convergence_rows = convergence.build_convergence_rows(point_rows)
    for row in convergence_rows:
        row.setdefault("row_type", "pairwise_convergence")
    quality_by_level = evaluate_quality_family(quality_rows, family.LEVELS)
    profile_validation = validate_profile_coverage(profile_rows)
    convergence_rows.append({
        "row_type": "profile_coverage",
        "reference_mesh": "local_base",
        "candidate_mesh": "local_ultra_fine",
        "quantity": "state0_on_contact_normal_profiles",
        **profile_validation,
    })
    gate = convergence.evaluate_final_gate(point_rows, convergence_rows, quality_by_level)
    gate["criteria"]["contact_normal_profile_coverage"] = bool(
        profile_validation["passed"]
    )
    gate["passed"] = all(bool(value) for value in gate["criteria"].values())
    gate["full_sweep_allowed"] = bool(gate["passed"])
    gate["status"] = (
        convergence.STATUS_CONVERGED
        if gate["passed"] else convergence.STATUS_NOT_CONVERGED
    )
    gate_failures = [name for name, passed in gate.get("criteria", {}).items() if not passed]
    full_dc = dispatch_full_dc_if_eligible(
        gate,
        dispatcher=lambda: generate_full_dc_outputs(
            staging, gate, python_executable=python_executable
        ),
    )
    convergence_rows.append({
        "row_type": "final_gate", "reference_mesh": "local_extra_fine",
        "candidate_mesh": "local_ultra_fine", "quantity": "all_required_criteria",
        "passed": bool(gate.get("passed")), "status": gate.get("status", ""),
        "error_message": "; ".join(gate_failures),
        "full_dc_executed": bool(full_dc.get("executed")),
        "full_dc_passed": (
            bool(full_dc.get("result", {}).get("passed"))
            if full_dc.get("executed") else False
        ),
    })
    write_csv(staging / OUTPUT_FILENAMES[0], _dynamic_fields(family_rows, ("schema_version", "mesh_level")), family_rows)
    write_csv(staging / OUTPUT_FILENAMES[1], QUALITY_FIELDS, quality_rows)
    write_csv(staging / OUTPUT_FILENAMES[2], POINT_FIELDS, point_rows)
    write_csv(staging / OUTPUT_FILENAMES[3], _dynamic_fields(convergence_rows,
              ("row_type", "reference_mesh", "candidate_mesh", "state_index", "bias_name",
               "quantity", "passed", "status", "error_message", "full_dc_executed")), convergence_rows)
    write_csv(staging / OUTPUT_FILENAMES[4], PROFILE_BASE_FIELDS, profile_rows)
    return {"gate": gate, "full_dc": full_dc, "point_count": len(point_rows),
            "quality_row_count": len(quality_rows), "profile_row_count": len(profile_rows)}


def run_candidate(*, jobs: int = 1, python_executable: str = sys.executable) -> dict[str, Any]:
    if CANDIDATE_DIRECTORY.exists():
        raise FileExistsError(f"candidate directory already exists: {CANDIDATE_DIRECTORY}")
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    protected = capture_protected_hashes()
    with tempfile.TemporaryDirectory(prefix=".local_mesh_", dir=RESULTS_DIRECTORY) as temporary:
        staging = Path(temporary) / "candidate"
        result = generate_candidate(staging, jobs=jobs, python_executable=python_executable)
        assert_protected_hashes(protected)
        if CANDIDATE_DIRECTORY.exists():
            raise FileExistsError(f"candidate directory appeared during run: {CANDIDATE_DIRECTORY}")
        # The destination must not exist.  ``rename`` is intentionally used
        # instead of replacement semantics so candidate publication cannot
        # overwrite an independently-created directory.
        staging.rename(CANDIDATE_DIRECTORY)
    assert_protected_hashes(protected)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-mode", choices=("probe", "point", "full_dc"))
    parser.add_argument("--mesh-level")
    parser.add_argument("--state-index", type=int)
    parser.add_argument("--bias-name", choices=("on", "off"))
    parser.add_argument("--sweep-kind", choices=("idvg", "idvd"))
    parser.add_argument("--run-role", default="primary")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.worker_mode:
        if not arguments.mesh_level or arguments.worker_output is None:
            raise SystemExit("worker mode requires --mesh-level and --worker-output")
        try:
            if arguments.worker_mode == "probe":
                bundle = run_probe_worker(arguments.mesh_level)
            elif arguments.worker_mode == "point":
                if arguments.state_index is None or arguments.bias_name is None:
                    raise ValueError("point worker requires state index and bias name")
                bundle = run_point_worker(arguments.mesh_level, arguments.state_index,
                                          arguments.bias_name, arguments.run_role)
            else:
                if arguments.sweep_kind is None:
                    raise ValueError("full-DC worker requires sweep kind")
                import local_mesh_full_dc as full_dc
                bundle = full_dc.run_full_dc_worker(
                    arguments.mesh_level, arguments.sweep_kind
                )
                bundle["mode"] = "full_dc"
        except Exception as error:
            traceback.print_exc()
            bundle = _failed_bundle({"mode": arguments.worker_mode,
                                     "mesh_level": arguments.mesh_level,
                                     "state_index": arguments.state_index,
                                     "bias_name": arguments.bias_name,
                                     "sweep_kind": arguments.sweep_kind,
                                     "run_role": arguments.run_role}, _error(error))
            _write_json(arguments.worker_output, bundle)
            return 1
        _write_json(arguments.worker_output, bundle)
        return 0
    result = run_candidate(jobs=arguments.jobs)
    print(json.dumps(_json_safe(result), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_DIRECTORY", "OUTPUT_FILENAMES", "POINT_FIELDS", "QUALITY_FIELDS",
    "PROFILE_BASE_FIELDS", "build_tasks", "build_worker_command",
    "capture_protected_hashes", "assert_protected_hashes", "write_csv",
    "run_probe_worker", "run_point_worker", "run_fresh_workers",
    "dispatch_full_dc_if_eligible", "generate_candidate", "run_candidate", "main",
)
