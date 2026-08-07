"""Generate contact-topology and physical electrode-flux QC candidates.

The established public Q/C tables and the earlier full-terminal candidate are
immutable inputs.  Every ``(mesh, repeat, memory state, reference bias)`` is
evaluated in a fresh process so a failed voltage restore cannot contaminate a
later reference point.

The capacitance convention is ``Cij = dQi/dVj``.  Contact-only column sums are
reported as diagnostics, never required to vanish.  Acceptance uses the
derivative form of global Gauss balance and gauge-invariance row sums.
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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import contact_topology_qc as topology_qc
import full_terminal_qc_candidate as legacy_qc
import run_full_terminal_qc_candidate as legacy_runner


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
PRIOR_CANDIDATE_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "full_terminal_qc_candidate"
)
CANDIDATE_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "contact_topology_qc_candidate"
)
DC_CANDIDATE_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "contact_topology_dc_candidate"
)

MESH_SPECS = (
    ("base", 1.0),
    ("base_repeat", 1.0),
    ("fine", 0.5),
    ("extra_fine", 0.25),
)
REFERENCE_BIASES = (("off", -1.0), ("on", 3.0))
REFERENCE_VDS_V = 0.05
REFERENCE_VS_V = 0.0
DELTA_VOLTAGES_V = (0.0005, 0.001, 0.002)
NOMINAL_DELTA_VOLTAGE_V = 0.001
GAUGE_DELTA_V = 0.001

CHARGE_ATOL_C = 1.0e-24
CHARGE_RTOL = 1.0e-8
DERIVATIVE_ATOL_F = 1.0e-22
DERIVATIVE_RTOL = 1.0e-5
REPEAT_CHARGE_ATOL_C = 1.0e-28
REPEAT_CAPACITANCE_ATOL_F = 1.0e-22
REPEAT_RTOL = 1.0e-4
MESH_CHARGE_ATOL_C = 1.0e-28
MESH_CHARGE_RTOL = 2.0e-2
MESH_CAPACITANCE_ATOL_F = 1.0e-22
MESH_CAPACITANCE_RTOL = 5.0e-2
DELTA_SENSITIVITY_LIMIT = 1.0e-2

CONTACT_CHARGES = ("Qg_contact_C", "Qd_contact_C", "Qs_contact_C")
INTERNAL_CHARGES = ("Qmobile_C", "Qtrap_C", "Qfixed_C")
ALL_CHARGES = (*CONTACT_CHARGES, *INTERNAL_CHARGES)
TERMINALS = ("gate", "drain", "source")
CONTACT_CHARGE_BY_TERMINAL = {
    "gate": "Qg_contact_C",
    "drain": "Qd_contact_C",
    "source": "Qs_contact_C",
}

OUTPUT_FILENAMES = (
    "contact_topology_audit_before.csv",
    "contact_topology_audit_after.csv",
    "contact_area_validation.csv",
    "contact_topology_dc_regression.csv",
    "reference_reset_reproducibility.csv",
    "mesh_levels.csv",
    "mesh_convergence_dc.csv",
    "mesh_convergence_charge.csv",
    "mesh_convergence_capacitance.csv",
    "direct_contact_flux_after_contact_fix.csv",
    "electrode_capacitance_derivative_balance.csv",
    "ward_dutton_mobile_partition_after_fix.csv",
    "method_comparison_after_contact_fix.csv",
    "weighting_potential_feasibility.md",
    "contact_topology_qc_README.md",
)

IMMUTABLE_PUBLIC_QC = (
    SHARED_DATA_DIRECTORY / "terminal_charge_by_state.csv",
    SHARED_DATA_DIRECTORY / "capacitance_matrix_by_state.csv",
    SHARED_DATA_DIRECTORY / "capacitance_summary_by_state.csv",
)

TOPOLOGY_FIELDS = (
    "phase", "mesh_level", "mesh_scale", "contact", "region",
    "edge_count", "node_count", "plane_axis", "plane_coordinate_cm",
    "radial_min_cm", "radial_max_cm", "normal_r", "normal_z",
    "normal_method", "runtime_normal_r", "runtime_normal_z",
    "runtime_normal_matches_geometry",
    "geometric_area_cm2", "runtime_area_cm2", "expected_area_cm2",
    "area_relative_error", "plane_passed", "radial_span_passed",
    "normal_passed", "area_passed", "equation_passed", "passed",
    "provenance", "error_message",
)
AREA_FIELDS = (
    "mesh_level", "mesh_scale", "contact", "edge_count",
    "analytic_area_cm2", "geometric_area_cm2", "runtime_area_cm2",
    "geometric_relative_error", "runtime_relative_error",
    "geometric_passed", "runtime_passed", "passed",
)
RESET_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "quantity", "base_value", "repeat_value", "absolute_difference",
    "scale", "tolerance", "passed", "error_message",
)
MESH_LEVEL_FIELDS = (
    "mesh_level", "mesh_scale", "global_coordinate_count",
    "region_node_count_with_duplicates", "triangle_count",
    "source_edge_count", "drain_edge_count", "gate_edge_count",
    "source_area_cm2", "drain_area_cm2", "gate_area_cm2",
    "mos2_radial_spacing_min_cm", "mos2_radial_spacing_max_cm",
    "mos2_axial_spacing_min_cm", "mos2_axial_spacing_max_cm",
    "source_contact_spacing_min_cm", "source_contact_spacing_max_cm",
    "drain_contact_spacing_min_cm", "drain_contact_spacing_max_cm",
    "mesh_line_count", "passed", "error_message",
)
MESH_CHARGE_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "reference_mesh", "candidate_mesh", "quantity", "reference_value_C",
    "candidate_value_C", "absolute_difference_C", "scale_C",
    "tolerance_C", "passed", "error_message",
)
MESH_CAP_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "reference_mesh", "candidate_mesh", "measured_terminal",
    "perturbed_terminal", "delta_voltage_V", "reference_value_F",
    "candidate_value_F", "absolute_difference_F", "scale_F",
    "tolerance_F", "passed", "error_message",
)
DIRECT_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "mesh_level", "Qg_contact_C", "Qd_contact_C", "Qs_contact_C",
    "Qmobile_C", "Qtrap_C", "Qfixed_C", "Qnoncontact_boundary_C",
    "noncontact_boundary_definition", "global_gauss_residual_C",
    "global_gauss_scale_C", "global_gauss_tolerance_C",
    "global_gauss_passed", "method", "status", "converged",
    "error_message",
)
DERIVATIVE_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "mesh_level", "perturbed_terminal", "delta_voltage_V", "Cgj_F",
    "Cdj_F", "Csj_F", "contact_column_sum_F", "dQmobile_dV_F",
    "dQtrap_dV_F", "dQfixed_dV_F", "dQnoncontact_boundary_dV_F",
    "derivative_gauss_residual_F", "derivative_gauss_scale_F",
    "derivative_gauss_tolerance_F", "derivative_gauss_passed",
    "gauge_row_sum_gate_F", "gauge_row_sum_drain_F",
    "gauge_row_sum_source_F", "direct_common_dQg_dV_F",
    "direct_common_dQd_dV_F", "direct_common_dQs_dV_F",
    "gauge_row_sum_passed", "endpoint_restore_passed", "converged",
    "error_message",
)
WARD_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "mesh_level", "Qs_mobile_C", "Qd_mobile_C", "Qmobile_channel_C",
    "partition_residual_C", "relative_partition_residual", "method",
    "status", "passed", "error_message",
)
METHOD_FIELDS = (
    "state_index", "state", "bias_name", "VGS_V", "VDS_V", "VS_V",
    "method", "scope", "Qg_C", "Qd_C", "Qs_C", "Qmobile_C",
    "support_status", "limitation",
)


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _error(error: object | None) -> str:
    return legacy_runner.normalize_error_message(error)


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
        json.dump(
            _json_safe(value), stream, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_immutable_hashes() -> dict[str, str]:
    paths = list(IMMUTABLE_PUBLIC_QC)
    if PRIOR_CANDIDATE_DIRECTORY.is_dir():
        paths.extend(
            path for path in sorted(PRIOR_CANDIDATE_DIRECTORY.rglob("*"))
            if path.is_file()
        )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("immutable input missing: " + ", ".join(missing))
    return {
        str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())): _sha256(path)
        for path in paths
    }


def assert_immutable_hashes(expected: Mapping[str, str]) -> None:
    actual = capture_immutable_hashes()
    if dict(expected) != actual:
        changed = sorted(
            name for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError("immutable public/prior candidate changed: " + ", ".join(changed))


def _compare(
    reference: Any,
    candidate: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> dict[str, Any]:
    try:
        left = _finite(reference, "reference")
        right = _finite(candidate, "candidate")
        scale = max(abs(left), abs(right))
        tolerance = absolute_tolerance + relative_tolerance * scale
        difference = abs(right - left)
        return {
            "reference": left, "candidate": right,
            "absolute_difference": difference, "scale": scale,
            "tolerance": tolerance, "passed": difference <= tolerance,
            "error_message": "",
        }
    except Exception as error:
        return {
            "reference": math.nan, "candidate": math.nan,
            "absolute_difference": math.nan, "scale": math.nan,
            "tolerance": math.nan, "passed": False,
            "error_message": _error(error),
        }


def _read_biases(device: str) -> dict[str, float]:
    return legacy_runner._read_runtime_biases(device)


def _assert_reference(
    *, device: str, base_biases: Mapping[str, float], ntrap_cm3: float,
) -> None:
    legacy_runner._assert_biases(_read_biases(device), base_biases)
    legacy_runner._assert_trap_density(device, ntrap_cm3)
    from devsim import get_node_model_values

    for region in (
        "CoreOxide", "MoS2", "TunnelOxide", "ChargeTrap", "BlockingOxide"
    ):
        values = tuple(
            float(value) for value in get_node_model_values(
                device=device, region=region, name="Potential"
            )
        )
        if not values or not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"{region} Potential is empty or non-finite")
    electrons = tuple(
        float(value) for value in get_node_model_values(
            device=device, region="MoS2", name="Electrons"
        )
    )
    if not electrons or not all(math.isfinite(value) for value in electrons):
        raise RuntimeError("MoS2 Electrons is empty or non-finite")


def _charge_components(device: str) -> dict[str, float]:
    raw = legacy_qc.extract_direct_contact_charge_components(device=device)
    return {name: _finite(raw[name], name) for name in ALL_CHARGES}


def _restore_terminal(
    terminal: str,
    base_biases: Mapping[str, float],
    delta: float,
    *,
    device: str,
    ntrap_cm3: float,
    baseline_components: Mapping[str, Any],
) -> dict[str, Any]:
    legacy_runner._ramp_absolute_terminal(
        terminal,
        float(base_biases[terminal]),
        delta,
        device=device,
        label=f"contact-topology-QC {terminal} baseline restore",
    )
    _assert_reference(device=device, base_biases=base_biases, ntrap_cm3=ntrap_cm3)
    restored = _charge_components(device)
    comparison = {
        name: _compare(
            baseline_components[name], restored[name],
            absolute_tolerance=REPEAT_CHARGE_ATOL_C,
            relative_tolerance=REPEAT_RTOL,
        )
        for name in ALL_CHARGES
    }
    if not all(row["passed"] for row in comparison.values()):
        failed = ",".join(name for name, row in comparison.items() if not row["passed"])
        raise RuntimeError("baseline charge restore failed: " + failed)
    return {"components": restored, "comparisons": comparison}


def measure_independent_capacitance_column(
    *,
    perturbed_terminal: str,
    base_biases: Mapping[str, float],
    baseline_components: Mapping[str, Any],
    delta_voltage_V: float,
    expected_trap_density_cm3: float,
    device: str,
) -> dict[str, Any]:
    """Measure one column using baseline→+→baseline→−→baseline."""

    terminal = str(perturbed_terminal)
    delta = _finite(delta_voltage_V, "delta_voltage_V")
    result: dict[str, Any] = {
        "perturbed_terminal": terminal,
        "delta_voltage_V": delta,
        "capacitances_F": {name: math.nan for name in CONTACT_CHARGES},
        "all_charge_derivatives_F": {name: math.nan for name in ALL_CHARGES},
        "endpoint_charges": {},
        "endpoint_restore_passed": False,
        "nonperturbed_biases_held": False,
        "trap_density_held": False,
        "baseline_bias_restored": False,
        "baseline_charge_restored": False,
        "converged": False,
        "error_message": "",
    }
    if terminal not in TERMINALS or delta <= 0.0:
        result["error_message"] = "invalid terminal or delta"
        return result
    endpoints: dict[str, dict[str, float]] = {}
    restores: dict[str, Any] = {}
    try:
        _assert_reference(
            device=device, base_biases=base_biases,
            ntrap_cm3=expected_trap_density_cm3,
        )
        for label, sign in (("plus", 1.0), ("minus", -1.0)):
            legacy_runner._ramp_absolute_terminal(
                terminal,
                float(base_biases[terminal]) + sign * delta,
                delta,
                device=device,
                label=f"contact-topology-QC {terminal} {label}",
            )
            actual = _read_biases(device)
            legacy_runner._assert_biases(
                actual, base_biases, excluded_terminal=terminal
            )
            expected_terminal = float(base_biases[terminal]) + sign * delta
            if not math.isclose(
                actual[terminal], expected_terminal,
                rel_tol=0.0, abs_tol=1.0e-12,
            ):
                raise RuntimeError(f"{terminal} {label} endpoint mismatch")
            legacy_runner._assert_trap_density(device, expected_trap_density_cm3)
            endpoints[label] = _charge_components(device)
            restores[label] = _restore_terminal(
                terminal, base_biases, delta, device=device,
                ntrap_cm3=expected_trap_density_cm3,
                baseline_components=baseline_components,
            )
        derivatives = {
            name: (endpoints["plus"][name] - endpoints["minus"][name])
            / (2.0 * delta)
            for name in ALL_CHARGES
        }
        result.update(
            {
                "capacitances_F": {
                    name: derivatives[name] for name in CONTACT_CHARGES
                },
                "all_charge_derivatives_F": derivatives,
                "endpoint_charges": endpoints,
                "endpoint_restores": restores,
                "endpoint_restore_passed": True,
                "nonperturbed_biases_held": True,
                "trap_density_held": True,
                "baseline_bias_restored": True,
                "baseline_charge_restored": True,
                "converged": True,
            }
        )
    except Exception as error:
        result["error_message"] = _error(error)
        try:
            _restore_terminal(
                terminal, base_biases, delta, device=device,
                ntrap_cm3=expected_trap_density_cm3,
                baseline_components=baseline_components,
            )
            result["baseline_bias_restored"] = True
            result["baseline_charge_restored"] = True
        except Exception as restore_error:
            result["error_message"] = (
                str(result["error_message"])
                + "; final restore failed: " + _error(restore_error)
            ).strip("; ")
    return result


def validate_independent_gauge_invariance(
    *,
    base_biases: Mapping[str, float],
    baseline_components: Mapping[str, Any],
    expected_trap_density_cm3: float,
    device: str,
) -> dict[str, Any]:
    """Measure a direct common-mode derivative with independent restores."""

    endpoints: dict[str, dict[str, float]] = {}
    errors: list[str] = []
    comparisons: dict[str, Any] = {}
    for label, shift in (("plus", GAUGE_DELTA_V), ("minus", -GAUGE_DELTA_V)):
        try:
            _assert_reference(
                device=device, base_biases=base_biases,
                ntrap_cm3=expected_trap_density_cm3,
            )
            legacy_runner.ramp_common_mode(base_biases, shift, device=device)
            shifted_biases = {
                terminal: float(base_biases[terminal]) + shift
                for terminal in TERMINALS
            }
            _assert_reference(
                device=device, base_biases=shifted_biases,
                ntrap_cm3=expected_trap_density_cm3,
            )
            endpoints[label] = _charge_components(device)
            comparisons[label] = {
                name: _compare(
                    baseline_components[name], endpoints[label][name],
                    absolute_tolerance=CHARGE_ATOL_C,
                    relative_tolerance=CHARGE_RTOL,
                )
                for name in ALL_CHARGES
            }
        except Exception as error:
            errors.append(f"{label}: {_error(error)}")
        finally:
            try:
                current = _read_biases(device)
                if all(
                    math.isclose(
                        current[terminal],
                        float(base_biases[terminal]) + shift,
                        rel_tol=0.0,
                        abs_tol=1.0e-12,
                    )
                    for terminal in TERMINALS
                ):
                    shifted_biases = {
                        terminal: float(base_biases[terminal]) + shift
                        for terminal in TERMINALS
                    }
                    legacy_runner.ramp_common_mode(
                        shifted_biases, -shift, device=device
                    )
                elif not all(
                    math.isclose(
                        current[terminal], float(base_biases[terminal]),
                        rel_tol=0.0, abs_tol=1.0e-12,
                    )
                    for terminal in TERMINALS
                ):
                    # A failed adaptive common-mode step may leave unequal
                    # terminal offsets.  Restore all three parameters as one
                    # atomic reference and solve before this worker can exit.
                    legacy_runner._set_all_terminal_biases(
                        base_biases, device=device
                    )
                    legacy_runner._solve_dc()
                _assert_reference(
                    device=device, base_biases=base_biases,
                    ntrap_cm3=expected_trap_density_cm3,
                )
                restored = _charge_components(device)
                for name in ALL_CHARGES:
                    row = _compare(
                        baseline_components[name], restored[name],
                        absolute_tolerance=REPEAT_CHARGE_ATOL_C,
                        relative_tolerance=REPEAT_RTOL,
                    )
                    if not row["passed"]:
                        raise RuntimeError(f"gauge restore changed {name}")
            except Exception as restore_error:
                errors.append(f"{label} restore: {_error(restore_error)}")
        if errors:
            break
    derivatives = {name: math.nan for name in ALL_CHARGES}
    if len(endpoints) == 2:
        derivatives = {
            name: (endpoints["plus"][name] - endpoints["minus"][name])
            / (2.0 * GAUGE_DELTA_V)
            for name in ALL_CHARGES
        }
    direct_invariance = bool(comparisons) and all(
        row["passed"]
        for endpoint in comparisons.values()
        for row in endpoint.values()
    )
    return {
        "passed": not errors and direct_invariance,
        "delta_voltage_V": GAUGE_DELTA_V,
        "endpoint_charges": endpoints,
        "comparisons": comparisons,
        "common_mode_derivatives_F": derivatives,
        "baseline_restore": {"passed": not errors},
        "error_message": "; ".join(errors),
    }


def validate_topology_mesh_line_log(
    line_log: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
) -> dict[str, Any]:
    """Validate the eight radial, two physical, and two guard mesh lines."""

    scale_value = _finite(scale, "mesh scale")
    errors: list[str] = []
    if len(line_log) != 12:
        errors.append(f"line_count={len(line_log)} expected=12")
    for index, row in enumerate(line_log):
        base = _finite(row.get("base_spacing_cm"), f"line {index} base spacing")
        scaled = _finite(
            row.get("scaled_spacing_cm"), f"line {index} scaled spacing"
        )
        if not math.isclose(scaled, base * scale_value, rel_tol=0.0, abs_tol=1e-30):
            errors.append(f"line {index} scale")
    positions = {
        (str(row.get("direction")), round(float(row.get("position_cm")), 18))
        for row in line_log
    }
    if len(positions) != len(line_log):
        errors.append("duplicate mesh-line position")
    required = {
        ("x", round(float(geometry[name]), 18))
        for name in (
            "r_axis", "r_core", "r_mos2", "r_tox", "r_trap", "r_block",
            "r_gate_outer", "r_air_outer",
        )
    }
    required.update(
        {
            ("y", round(float(geometry["z_source"]), 18)),
            ("y", round(float(geometry["z_drain"]), 18)),
        }
    )
    if not required.issubset(positions):
        errors.append("physical boundary position missing")
    guard = float(geometry["r_mos2"]) - float(geometry["r_core"])
    expected_guards = (
        float(geometry["z_source"]) - guard,
        float(geometry["z_drain"]) + guard,
    )
    guard_rows = [
        row for row in line_log
        if str(row.get("direction")) == "y"
        and (
            float(row.get("position_cm")) < float(geometry["z_source"])
            or float(row.get("position_cm")) > float(geometry["z_drain"])
        )
    ]
    if len(guard_rows) != 2:
        errors.append(f"guard_line_count={len(guard_rows)} expected=2")
    else:
        actual_guards = sorted(float(row["position_cm"]) for row in guard_rows)
        for index, (actual, expected) in enumerate(
            zip(actual_guards, sorted(expected_guards))
        ):
            if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-30):
                errors.append(f"guard {index} position")
        for index, row in enumerate(guard_rows):
            if not math.isclose(
                float(row["base_spacing_cm"]), guard,
                rel_tol=0.0, abs_tol=1e-30,
            ):
                errors.append(f"guard {index} base spacing")
    if errors:
        raise RuntimeError("topology mesh-line validation failed: " + ", ".join(errors))
    return {
        "passed": True,
        "line_count": len(line_log),
        "scale": scale_value,
        "guard_lines": [dict(row) for row in guard_rows],
        "lines": [dict(row) for row in line_log],
    }


def _runtime_mesh_counts(device: str, topology: Mapping[str, Any]) -> dict[str, Any]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    global_coordinates: set[int] = set()
    region_node_count = 0
    triangle_count = 0
    for region in get_region_list(device=device):
        coordinate_indices = tuple(
            int(value) for value in get_node_model_values(
                device=device, region=region, name="coordinate_index"
            )
        )
        global_coordinates.update(coordinate_indices)
        region_node_count += len(coordinate_indices)
        triangle_count += len(get_element_node_list(device=device, region=region))

    xs = sorted(set(float(value) for value in get_node_model_values(
        device=device, region="MoS2", name="x"
    )))
    ys = sorted(set(float(value) for value in get_node_model_values(
        device=device, region="MoS2", name="y"
    )))

    def spacings(values: Sequence[float]) -> tuple[float, float]:
        deltas = [right - left for left, right in zip(values, values[1:]) if right > left]
        return (min(deltas), max(deltas)) if deltas else (math.nan, math.nan)

    radial_min, radial_max = spacings(xs)
    axial_min, axial_max = spacings(ys)
    reports = topology.get("contacts", {})
    result: dict[str, Any] = {
        "global_coordinate_count": len(global_coordinates),
        "region_node_count_with_duplicates": region_node_count,
        "triangle_count": triangle_count,
        "mos2_radial_spacing_min_cm": radial_min,
        "mos2_radial_spacing_max_cm": radial_max,
        "mos2_axial_spacing_min_cm": axial_min,
        "mos2_axial_spacing_max_cm": axial_max,
    }
    for terminal in TERMINALS:
        report = reports.get(terminal, {})
        result[f"{terminal}_edge_count"] = int(report.get("edge_count", 0))
        result[f"{terminal}_area_cm2"] = float(
            report.get("runtime_area_cm2", report.get("surface_area_cm2", math.nan))
        )
        edge_spacings = tuple(float(value) for value in report.get("edge_lengths_cm", ()))
        result[f"{terminal}_contact_spacing_min_cm"] = (
            min(edge_spacings) if edge_spacings else math.nan
        )
        result[f"{terminal}_contact_spacing_max_cm"] = (
            max(edge_spacings) if edge_spacings else math.nan
        )
    return result


def _point_key(row: Mapping[str, Any]) -> tuple[int, str]:
    return int(row["state_index"]), str(row["bias_name"])


def _matrix_from_point(
    point: Mapping[str, Any], delta: float,
) -> dict[tuple[str, str], float]:
    rows = {
        (str(row["perturbed_terminal"]), float(row["delta_voltage_V"])): row
        for row in point.get("measurements", ())
    }
    matrix: dict[tuple[str, str], float] = {}
    for measured in TERMINALS:
        charge_name = CONTACT_CHARGE_BY_TERMINAL[measured]
        for perturbed in TERMINALS:
            row = rows.get((perturbed, float(delta)), {})
            value = row.get("capacitances_F", {}).get(charge_name)
            matrix[(measured, perturbed)] = (
                float(value) if value is not None else math.nan
            )
    return matrix


def _derivative_balance(derivatives: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        name: _finite(derivatives[name], name) for name in ALL_CHARGES
    }
    values["Qnoncontact_boundary_C"] = 0.0
    residual = math.fsum(values.values())
    scale = max((abs(value) for value in values.values()), default=0.0)
    tolerance = DERIVATIVE_ATOL_F + DERIVATIVE_RTOL * scale
    return {
        "components_F": values,
        "residual_F": residual,
        "scale_F": scale,
        "tolerance_F": tolerance,
        "passed": abs(residual) <= tolerance,
    }


def _global_balance(components: Mapping[str, Any]) -> dict[str, Any]:
    values = {name: _finite(components[name], name) for name in ALL_CHARGES}
    values["Qnoncontact_boundary_C"] = 0.0
    residual = math.fsum(values.values())
    scale = max((abs(value) for value in values.values()), default=0.0)
    tolerance = CHARGE_ATOL_C + CHARGE_RTOL * scale
    return {
        "components_C": values,
        "residual_C": residual,
        "scale_C": scale,
        "tolerance_C": tolerance,
        "passed": abs(residual) <= tolerance,
    }


def _enhance_point(point: dict[str, Any], topology_passed: bool) -> None:
    errors: list[str] = []
    direct = point.get("direct", {})
    try:
        global_balance = _global_balance(direct)
    except Exception as error:
        global_balance = {
            "residual_C": math.nan, "scale_C": math.nan,
            "tolerance_C": math.nan, "passed": False,
        }
        errors.append(_error(error))
    derivative_rows: list[dict[str, Any]] = []
    measurements_ok = True
    for measurement in point.get("measurements", ()):
        try:
            balance = _derivative_balance(
                measurement["all_charge_derivatives_F"]
            )
        except Exception as error:
            balance = {
                "components_F": {}, "residual_F": math.nan,
                "scale_F": math.nan, "tolerance_F": math.nan,
                "passed": False,
            }
            errors.append(_error(error))
        measurement["derivative_gauss"] = balance
        row_ok = bool(measurement.get("converged")) and bool(balance["passed"])
        measurements_ok = measurements_ok and row_ok
        derivative_rows.append({
            "perturbed_terminal": measurement.get("perturbed_terminal"),
            "delta_voltage_V": measurement.get("delta_voltage_V"),
            **balance,
            "passed": row_ok,
        })

    nominal_matrix = _matrix_from_point(point, NOMINAL_DELTA_VOLTAGE_V)
    row_sums = {
        measured: math.fsum(
            nominal_matrix[(measured, perturbed)] for perturbed in TERMINALS
        )
        for measured in TERMINALS
    }
    gauge = point.get("gauge", {})
    common = gauge.get("common_mode_derivatives_F", {})
    gauge_rows: dict[str, Any] = {}
    for measured in TERMINALS:
        charge_name = CONTACT_CHARGE_BY_TERMINAL[measured]
        comparison = _compare(
            row_sums[measured], common.get(charge_name),
            absolute_tolerance=DERIVATIVE_ATOL_F,
            relative_tolerance=DERIVATIVE_RTOL,
        )
        zero_check = _compare(
            row_sums[measured], 0.0,
            absolute_tolerance=DERIVATIVE_ATOL_F,
            relative_tolerance=DERIVATIVE_RTOL,
        )
        gauge_rows[measured] = {
            "row_sum_F": row_sums[measured],
            "direct_common_derivative_F": common.get(charge_name, math.nan),
            "row_vs_direct": comparison,
            "row_vs_zero": zero_check,
            "passed": bool(comparison["passed"] and zero_check["passed"]),
        }
    gauge_row_passed = bool(gauge.get("passed")) and all(
        row["passed"] for row in gauge_rows.values()
    )
    delta_passed = all(
        bool(row.get("passed"))
        for row in point.get("sensitivities", {}).values()
    )
    ward_passed = bool(point.get("ward", {}).get("converged"))
    point_passed = all(
        (
            topology_passed,
            bool(global_balance["passed"]),
            measurements_ok,
            gauge_row_passed,
            delta_passed,
            ward_passed,
        )
    )
    if not topology_passed:
        errors.append("contact topology failed")
    if not global_balance["passed"]:
        errors.append("global Gauss balance failed")
    if not measurements_ok:
        errors.append("endpoint restore or derivative Gauss balance failed")
    if not gauge_row_passed:
        errors.append("gauge row-sum validation failed")
    if not delta_passed:
        errors.append("finite-difference sensitivity failed")
    if not ward_passed:
        errors.append("Ward-Dutton identity failed")
    point.update(
        {
            "global_gauss": global_balance,
            "derivative_gauss_rows": derivative_rows,
            "nominal_matrix_F": {
                f"{measured}:{perturbed}": value
                for (measured, perturbed), value in nominal_matrix.items()
            },
            "gauge_row_validation": {
                "rows": gauge_rows,
                "passed": gauge_row_passed,
            },
            "passed": point_passed,
            "legacy_error_message": str(point.get("error_message", "")),
            "error_message": "; ".join(dict.fromkeys(errors)),
        }
    )


def run_point_worker(
    *, mesh_level: str, mesh_scale: float, state_index: int, bias_name: str,
) -> dict[str, Any]:
    import state_characterization_config as config

    states = [state for state in config.MEMORY_STATES if int(state["state_index"]) == state_index]
    biases = [row for row in REFERENCE_BIASES if row[0] == bias_name]
    if len(states) != 1 or len(biases) != 1:
        raise ValueError("unknown state_index or bias_name")
    config.MEMORY_STATES = (states[0],)
    legacy_runner.REFERENCE_BIASES = (biases[0],)
    legacy_runner.DELTA_VOLTAGES_V = DELTA_VOLTAGES_V
    legacy_runner.NOMINAL_DELTA_VOLTAGE_V = NOMINAL_DELTA_VOLTAGE_V
    legacy_runner.measure_full_capacitance_column = (
        measure_independent_capacitance_column
    )
    legacy_runner.validate_gauge_invariance = validate_independent_gauge_invariance
    legacy_runner.validate_mesh_line_log = validate_topology_mesh_line_log

    bundle = legacy_runner.run_worker(mesh_level, mesh_scale)
    if len(bundle.get("points", ())) != 1:
        raise RuntimeError("isolated worker did not produce exactly one point")
    geometry = bundle["geometry"]
    topology = topology_qc.audit_runtime_contact_topology(
        device=legacy_qc.DEVICE_NAME, geometry=geometry
    )
    bundle["topology"] = topology
    bundle["mesh_counts"] = _runtime_mesh_counts(legacy_qc.DEVICE_NAME, topology)
    point = bundle["points"][0]
    _enhance_point(point, bool(topology.get("passed")))
    base_biases = {
        "gate": float(point["VGS_V"]) + float(point["VS_V"]),
        "drain": float(point["VDS_V"]) + float(point["VS_V"]),
        "source": float(point["VS_V"]),
    }
    try:
        _assert_reference(
            device=legacy_qc.DEVICE_NAME,
            base_biases=base_biases,
            ntrap_cm3=float(point["ntrap_cm3"]),
        )
        bundle["final_reference_validation"] = {"passed": True, "error_message": ""}
    except Exception as error:
        bundle["final_reference_validation"] = {
            "passed": False, "error_message": _error(error)
        }
        point["passed"] = False
        point["error_message"] = (
            str(point.get("error_message", ""))
            + "; final reference validation failed: " + _error(error)
        ).strip("; ")
    bundle["worker_identity"] = {
        "mesh_level": mesh_level, "mesh_scale": float(mesh_scale),
        "state_index": state_index, "bias_name": bias_name,
        "process_id": os.getpid(),
    }
    return bundle


def build_worker_command(
    *, python_executable: str, mesh_level: str, mesh_scale: float,
    state_index: int, bias_name: str, output_path: Path,
) -> list[str]:
    return [
        str(python_executable), "-B", str(Path(__file__).resolve()), "--worker",
        "--mesh-level", mesh_level, "--mesh-scale", f"{mesh_scale:.17g}",
        "--state-index", str(state_index), "--bias-name", bias_name,
        "--worker-output", str(output_path.resolve()),
    ]


def _run_one_subprocess(
    command: Sequence[str], *, log_path: Path,
) -> tuple[int, str]:
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        completed = subprocess.run(
            list(command), cwd=REPOSITORY_ROOT, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    if completed.returncode == 0:
        return 0, ""
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
    return int(completed.returncode), "\n".join(tail)


def run_isolated_workers(
    temporary_directory: Path,
    *, jobs: int = 1, python_executable: str = sys.executable,
) -> dict[str, dict[str, Any]]:
    import state_characterization_config as config

    tasks: list[dict[str, Any]] = []
    for mesh_level, mesh_scale in MESH_SPECS:
        for state in config.MEMORY_STATES:
            for bias_name, _ in REFERENCE_BIASES:
                stem = f"{mesh_level}_s{int(state['state_index'])}_{bias_name}"
                output = temporary_directory / f"{stem}.json"
                tasks.append(
                    {
                        "mesh_level": mesh_level, "mesh_scale": mesh_scale,
                        "state_index": int(state["state_index"]),
                        "bias_name": bias_name, "output": output,
                        "log": temporary_directory / f"{stem}.log",
                    }
                )
    failures: list[str] = []

    def execute(task: Mapping[str, Any]) -> Mapping[str, Any]:
        command = build_worker_command(
            python_executable=python_executable,
            mesh_level=str(task["mesh_level"]),
            mesh_scale=float(task["mesh_scale"]),
            state_index=int(task["state_index"]),
            bias_name=str(task["bias_name"]),
            output_path=Path(task["output"]),
        )
        code, tail = _run_one_subprocess(command, log_path=Path(task["log"]))
        return {**task, "return_code": code, "tail": tail}

    maximum_workers = max(1, int(jobs))
    with ThreadPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        completed_count = 0
        for future in as_completed(futures):
            result = future.result()
            completed_count += 1
            print(
                f"QC worker {completed_count}/{len(tasks)}: "
                f"{result['mesh_level']} state={result['state_index']} "
                f"bias={result['bias_name']} rc={result['return_code']}",
                flush=True,
            )
            if int(result["return_code"]) != 0:
                failures.append(
                    f"{result['mesh_level']}/s{result['state_index']}/"
                    f"{result['bias_name']}:\n{result['tail']}"
                )
    if failures:
        raise RuntimeError("isolated worker failure(s):\n" + "\n---\n".join(failures))

    merged: dict[str, dict[str, Any]] = {}
    for task in tasks:
        with Path(task["output"]).open("r", encoding="utf-8") as stream:
            bundle = json.load(stream)
        level = str(task["mesh_level"])
        if level not in merged:
            merged[level] = {
                "mesh_level": level,
                "mesh_scale": float(task["mesh_scale"]),
                "geometry": bundle["geometry"],
                "mesh_line_validation": bundle["mesh_line_validation"],
                "mesh_counts": bundle["mesh_counts"],
                "topology": bundle["topology"],
                "points": [],
                "worker_identities": [],
            }
        else:
            for invariant in (
                "geometry", "mesh_line_validation", "mesh_counts", "topology"
            ):
                reference = json.dumps(
                    _json_safe(merged[level][invariant]),
                    sort_keys=True, separators=(",", ":"),
                )
                candidate = json.dumps(
                    _json_safe(bundle[invariant]),
                    sort_keys=True, separators=(",", ":"),
                )
                if reference != candidate:
                    raise RuntimeError(
                        f"{level} fresh workers disagree on {invariant}"
                    )
        merged[level]["points"].extend(bundle["points"])
        merged[level]["worker_identities"].append(bundle["worker_identity"])
        if not bundle.get("final_reference_validation", {}).get("passed"):
            merged[level].setdefault("errors", []).append(
                bundle["final_reference_validation"].get("error_message", "")
            )
    for bundle in merged.values():
        bundle["points"].sort(key=lambda row: _point_key(row))
    return merged


def _points_by_key(bundle: Mapping[str, Any]) -> dict[tuple[int, str], Mapping[str, Any]]:
    return {_point_key(point): point for point in bundle.get("points", ())}


def _topology_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, bundle in bundles.items():
        topology = bundle.get("topology", {})
        for contact in TERMINALS:
            report = topology.get("contacts", {}).get(contact, {})
            rows.append(
                {
                    "phase": "after", "mesh_level": level,
                    "mesh_scale": bundle["mesh_scale"], "contact": contact,
                    "region": report.get("region", ""),
                    "edge_count": report.get("edge_count", 0),
                    "node_count": report.get("node_count", 0),
                    "plane_axis": report.get("plane_axis", ""),
                    "plane_coordinate_cm": report.get("plane_coordinate_cm", math.nan),
                    "radial_min_cm": report.get("radial_min_cm", math.nan),
                    "radial_max_cm": report.get("radial_max_cm", math.nan),
                    "normal_r": report.get("normal_r", math.nan),
                    "normal_z": report.get("normal_z", math.nan),
                    "normal_method": report.get("normal_method", ""),
                    "runtime_normal_r": report.get("runtime_normal_r", math.nan),
                    "runtime_normal_z": report.get("runtime_normal_z", math.nan),
                    "runtime_normal_matches_geometry": report.get(
                        "runtime_normal_matches_geometry", False
                    ),
                    "geometric_area_cm2": report.get("geometric_area_cm2", math.nan),
                    "runtime_area_cm2": report.get("runtime_area_cm2", math.nan),
                    "expected_area_cm2": report.get("expected_area_cm2", math.nan),
                    "area_relative_error": report.get("area_relative_error", math.nan),
                    "plane_passed": report.get("plane_passed", False),
                    "radial_span_passed": report.get("radial_span_passed", False),
                    "normal_passed": report.get("normal_passed", False),
                    "area_passed": report.get("area_passed", False),
                    "equation_passed": report.get("equation_passed", False),
                    "passed": report.get("passed", False),
                    "provenance": "fresh DEVSIM runtime API and contact-edge geometry",
                    "error_message": report.get("error_message", ""),
                }
            )
    return rows


def _before_topology_rows() -> list[dict[str, Any]]:
    expected_gate = 2.0 * math.pi * 3.6e-6 * 1.0e-5
    expected_sd = math.pi * ((1.2e-6) ** 2 - (1.0e-6) ** 2)
    raw = (
        ("source", "MoS2", 0, 2, 0.0, 0.0, 0.0, expected_sd),
        ("drain", "MoS2", 0, 2, 0.0, 0.0, 0.0, expected_sd),
        ("gate", "BlockingOxide", 50, 51, 1.0, 0.0, expected_gate, expected_gate),
    )
    rows = []
    for contact, region, edges, nodes, nr, nz, area, expected in raw:
        rows.append(
            {
                "phase": "before", "mesh_level": "committed_baseline",
                "mesh_scale": 1.0, "contact": contact, "region": region,
                "edge_count": edges, "node_count": nodes,
                "plane_axis": "x" if contact == "gate" else "y",
                "plane_coordinate_cm": 3.6e-6 if contact == "gate" else (
                    0.0 if contact == "source" else 1.0e-5
                ),
                "radial_min_cm": math.nan, "radial_max_cm": math.nan,
                "normal_r": nr, "normal_z": nz,
                "normal_method": (
                    "preserved runtime model" if contact == "gate" else "degenerate"
                ),
                "runtime_normal_r": nr, "runtime_normal_z": nz,
                "runtime_normal_matches_geometry": contact == "gate",
                "geometric_area_cm2": area, "runtime_area_cm2": area,
                "expected_area_cm2": expected,
                "area_relative_error": abs(area - expected) / expected,
                "plane_passed": True, "radial_span_passed": contact == "gate",
                "normal_passed": contact == "gate", "area_passed": contact == "gate",
                "equation_passed": True, "passed": contact == "gate",
                "provenance": (
                    "preserved pre-fix shared_data mesh and prior candidate runtime audit"
                ),
                "error_message": "" if contact == "gate" else (
                    "contact node set exists but boundary edge set is empty"
                ),
            }
        )
    return rows


def _area_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for level, bundle in bundles.items():
        for contact in TERMINALS:
            report = bundle["topology"]["contacts"][contact]
            analytic = float(report["expected_area_cm2"])
            geometric = float(report["geometric_area_cm2"])
            runtime = float(report["runtime_area_cm2"])
            ge = abs(geometric - analytic) / analytic
            re = abs(runtime - analytic) / analytic
            rows.append(
                {
                    "mesh_level": level, "mesh_scale": bundle["mesh_scale"],
                    "contact": contact, "edge_count": report["edge_count"],
                    "analytic_area_cm2": analytic,
                    "geometric_area_cm2": geometric,
                    "runtime_area_cm2": runtime,
                    "geometric_relative_error": ge,
                    "runtime_relative_error": re,
                    "geometric_passed": ge <= 1.0e-6,
                    "runtime_passed": re <= 1.0e-6,
                    "passed": ge <= 1.0e-6 and re <= 1.0e-6,
                }
            )
    return rows


def _repeat_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    base = _points_by_key(bundles["base"])
    repeat = _points_by_key(bundles["base_repeat"])
    rows: list[dict[str, Any]] = []
    for key in sorted(base):
        left, right = base[key], repeat[key]
        quantities: dict[str, tuple[float, float, float]] = {}
        for name in ALL_CHARGES:
            quantities[name] = (
                left["direct"][name], right["direct"][name],
                REPEAT_CHARGE_ATOL_C,
            )
        left_matrix = _matrix_from_point(left, NOMINAL_DELTA_VOLTAGE_V)
        right_matrix = _matrix_from_point(right, NOMINAL_DELTA_VOLTAGE_V)
        for measured in TERMINALS:
            for perturbed in TERMINALS:
                name = f"C{measured[0]}{perturbed[0]}_F"
                quantities[name] = (
                    left_matrix[(measured, perturbed)],
                    right_matrix[(measured, perturbed)],
                    REPEAT_CAPACITANCE_ATOL_F,
                )
        for quantity, (left_value, right_value, atol) in quantities.items():
            comparison = _compare(
                left_value, right_value,
                absolute_tolerance=atol, relative_tolerance=REPEAT_RTOL,
            )
            rows.append(
                {
                    "state_index": left["state_index"], "state": left["state"],
                    "bias_name": left["bias_name"], "VGS_V": left["VGS_V"],
                    "VDS_V": left["VDS_V"], "VS_V": left["VS_V"],
                    "quantity": quantity,
                    "base_value": comparison["reference"],
                    "repeat_value": comparison["candidate"],
                    "absolute_difference": comparison["absolute_difference"],
                    "scale": comparison["scale"],
                    "tolerance": comparison["tolerance"],
                    "passed": comparison["passed"],
                    "error_message": comparison["error_message"],
                }
            )
    return rows


def _mesh_level_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level, bundle in bundles.items():
        counts = bundle["mesh_counts"]
        rows.append(
            {
                "mesh_level": level, "mesh_scale": bundle["mesh_scale"],
                "global_coordinate_count": counts["global_coordinate_count"],
                "region_node_count_with_duplicates": counts[
                    "region_node_count_with_duplicates"
                ],
                "triangle_count": counts["triangle_count"],
                "source_edge_count": counts["source_edge_count"],
                "drain_edge_count": counts["drain_edge_count"],
                "gate_edge_count": counts["gate_edge_count"],
                "source_area_cm2": counts["source_area_cm2"],
                "drain_area_cm2": counts["drain_area_cm2"],
                "gate_area_cm2": counts["gate_area_cm2"],
                "mos2_radial_spacing_min_cm": counts["mos2_radial_spacing_min_cm"],
                "mos2_radial_spacing_max_cm": counts["mos2_radial_spacing_max_cm"],
                "mos2_axial_spacing_min_cm": counts["mos2_axial_spacing_min_cm"],
                "mos2_axial_spacing_max_cm": counts["mos2_axial_spacing_max_cm"],
                "source_contact_spacing_min_cm": counts[
                    "source_contact_spacing_min_cm"
                ],
                "source_contact_spacing_max_cm": counts[
                    "source_contact_spacing_max_cm"
                ],
                "drain_contact_spacing_min_cm": counts[
                    "drain_contact_spacing_min_cm"
                ],
                "drain_contact_spacing_max_cm": counts[
                    "drain_contact_spacing_max_cm"
                ],
                "mesh_line_count": bundle["mesh_line_validation"]["line_count"],
                "passed": bool(bundle["topology"]["passed"]),
                "error_message": "" if bundle["topology"]["passed"] else (
                    "contact topology validation failed"
                ),
            }
        )
    return rows


def _mesh_convergence_rows(
    bundles: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fine = _points_by_key(bundles["fine"])
    extra = _points_by_key(bundles["extra_fine"])
    charge_rows: list[dict[str, Any]] = []
    cap_rows: list[dict[str, Any]] = []
    for key in sorted(fine):
        left, right = fine[key], extra[key]
        for quantity in ALL_CHARGES:
            comparison = _compare(
                left["direct"][quantity], right["direct"][quantity],
                absolute_tolerance=MESH_CHARGE_ATOL_C,
                relative_tolerance=MESH_CHARGE_RTOL,
            )
            charge_rows.append(
                {
                    "state_index": left["state_index"], "state": left["state"],
                    "bias_name": left["bias_name"], "VGS_V": left["VGS_V"],
                    "VDS_V": left["VDS_V"], "VS_V": left["VS_V"],
                    "reference_mesh": "fine", "candidate_mesh": "extra_fine",
                    "quantity": quantity,
                    "reference_value_C": comparison["reference"],
                    "candidate_value_C": comparison["candidate"],
                    "absolute_difference_C": comparison["absolute_difference"],
                    "scale_C": comparison["scale"],
                    "tolerance_C": comparison["tolerance"],
                    "passed": comparison["passed"],
                    "error_message": comparison["error_message"],
                }
            )
        for delta in DELTA_VOLTAGES_V:
            lm = _matrix_from_point(left, delta)
            rm = _matrix_from_point(right, delta)
            for measured in TERMINALS:
                for perturbed in TERMINALS:
                    comparison = _compare(
                        lm[(measured, perturbed)], rm[(measured, perturbed)],
                        absolute_tolerance=MESH_CAPACITANCE_ATOL_F,
                        relative_tolerance=MESH_CAPACITANCE_RTOL,
                    )
                    cap_rows.append(
                        {
                            "state_index": left["state_index"],
                            "state": left["state"], "bias_name": left["bias_name"],
                            "VGS_V": left["VGS_V"], "VDS_V": left["VDS_V"],
                            "VS_V": left["VS_V"], "reference_mesh": "fine",
                            "candidate_mesh": "extra_fine",
                            "measured_terminal": measured,
                            "perturbed_terminal": perturbed,
                            "delta_voltage_V": delta,
                            "reference_value_F": comparison["reference"],
                            "candidate_value_F": comparison["candidate"],
                            "absolute_difference_F": comparison["absolute_difference"],
                            "scale_F": comparison["scale"],
                            "tolerance_F": comparison["tolerance"],
                            "passed": comparison["passed"],
                            "error_message": comparison["error_message"],
                        }
                    )
    return charge_rows, cap_rows


def _direct_rows(
    bundles: Mapping[str, Mapping[str, Any]], status: str,
) -> list[dict[str, Any]]:
    rows = []
    for level, bundle in bundles.items():
        for point in bundle["points"]:
            direct = point["direct"]
            gauss = point["global_gauss"]
            rows.append(
                {
                    "state_index": point["state_index"], "state": point["state"],
                    "bias_name": point["bias_name"], "VGS_V": point["VGS_V"],
                    "VDS_V": point["VDS_V"], "VS_V": point["VS_V"],
                    "mesh_level": level,
                    **{name: direct[name] for name in ALL_CHARGES},
                    "Qnoncontact_boundary_C": 0.0,
                    "noncontact_boundary_definition": (
                        "no separately assembled term; homogeneous natural boundary "
                        "in the audited discrete PotentialEquation"
                    ),
                    "global_gauss_residual_C": gauss["residual_C"],
                    "global_gauss_scale_C": gauss["scale_C"],
                    "global_gauss_tolerance_C": gauss["tolerance_C"],
                    "global_gauss_passed": gauss["passed"],
                    "method": "physical_electrode_contact_flux",
                    "status": status, "converged": point["passed"],
                    "error_message": point["error_message"],
                }
            )
    return rows


def _derivative_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    def numeric_or_nan(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return math.nan

    rows = []
    for level, bundle in bundles.items():
        for point in bundle["points"]:
            gauge_rows = point["gauge_row_validation"]["rows"]
            gauge_common = point.get("gauge", {}).get(
                "common_mode_derivatives_F", {}
            )
            for measurement in point.get("measurements", ()):
                derivatives = measurement.get("all_charge_derivatives_F", {})
                balance = measurement.get("derivative_gauss", {})
                perturbed = str(measurement["perturbed_terminal"])
                rows.append(
                    {
                        "state_index": point["state_index"], "state": point["state"],
                        "bias_name": point["bias_name"], "VGS_V": point["VGS_V"],
                        "VDS_V": point["VDS_V"], "VS_V": point["VS_V"],
                        "mesh_level": level, "perturbed_terminal": perturbed,
                        "delta_voltage_V": measurement["delta_voltage_V"],
                        "Cgj_F": derivatives.get("Qg_contact_C", math.nan),
                        "Cdj_F": derivatives.get("Qd_contact_C", math.nan),
                        "Csj_F": derivatives.get("Qs_contact_C", math.nan),
                        "contact_column_sum_F": math.fsum(
                            numeric_or_nan(derivatives.get(name, math.nan))
                            for name in CONTACT_CHARGES
                        ),
                        "dQmobile_dV_F": derivatives.get("Qmobile_C", math.nan),
                        "dQtrap_dV_F": derivatives.get("Qtrap_C", math.nan),
                        "dQfixed_dV_F": derivatives.get("Qfixed_C", math.nan),
                        "dQnoncontact_boundary_dV_F": 0.0,
                        "derivative_gauss_residual_F": balance.get("residual_F", math.nan),
                        "derivative_gauss_scale_F": balance.get("scale_F", math.nan),
                        "derivative_gauss_tolerance_F": balance.get(
                            "tolerance_F", math.nan
                        ),
                        "derivative_gauss_passed": balance.get("passed", False),
                        "gauge_row_sum_gate_F": gauge_rows["gate"]["row_sum_F"],
                        "gauge_row_sum_drain_F": gauge_rows["drain"]["row_sum_F"],
                        "gauge_row_sum_source_F": gauge_rows["source"]["row_sum_F"],
                        "direct_common_dQg_dV_F": gauge_common.get(
                            "Qg_contact_C", math.nan
                        ),
                        "direct_common_dQd_dV_F": gauge_common.get(
                            "Qd_contact_C", math.nan
                        ),
                        "direct_common_dQs_dV_F": gauge_common.get(
                            "Qs_contact_C", math.nan
                        ),
                        "gauge_row_sum_passed": point[
                            "gauge_row_validation"
                        ]["passed"],
                        "endpoint_restore_passed": measurement.get(
                            "endpoint_restore_passed", False
                        ),
                        "converged": measurement.get("converged", False),
                        "error_message": measurement.get("error_message", ""),
                    }
                )
    return rows


def _ward_rows(bundles: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for level, bundle in bundles.items():
        for point in bundle["points"]:
            ward = point["ward"]
            rows.append(
                {
                    "state_index": point["state_index"], "state": point["state"],
                    "bias_name": point["bias_name"], "VGS_V": point["VGS_V"],
                    "VDS_V": point["VDS_V"], "VS_V": point["VS_V"],
                    "mesh_level": level,
                    "Qs_mobile_C": ward.get("Qs_mobile_C", math.nan),
                    "Qd_mobile_C": ward.get("Qd_mobile_C", math.nan),
                    "Qmobile_channel_C": ward.get("Qmobile_channel_C", math.nan),
                    "partition_residual_C": ward.get("partition_residual_C", math.nan),
                    "relative_partition_residual": ward.get(
                        "relative_partition_residual", math.nan
                    ),
                    "method": "ward_dutton_mobile_channel",
                    "status": "mobile_partition_only",
                    "passed": ward.get("converged", False),
                    "error_message": ward.get("error_message", ""),
                }
            )
    return rows


def _method_rows(
    bundles: Mapping[str, Mapping[str, Any]], status: str,
) -> list[dict[str, Any]]:
    rows = []
    for point in bundles["base"]["points"]:
        direct = point["direct"]
        ward = point["ward"]
        common = {
            "state_index": point["state_index"], "state": point["state"],
            "bias_name": point["bias_name"], "VGS_V": point["VGS_V"],
            "VDS_V": point["VDS_V"], "VS_V": point["VS_V"],
        }
        rows.append(
            {
                **common, "method": "physical_electrode_contact_flux",
                "scope": "electrostatic reaction charge at modeled electrodes",
                "Qg_C": direct["Qg_contact_C"], "Qd_C": direct["Qd_contact_C"],
                "Qs_C": direct["Qs_contact_C"], "Qmobile_C": direct["Qmobile_C"],
                "support_status": status,
                "limitation": (
                    "not automatically a charge-conserving compact-model partition"
                ),
            }
        )
        rows.append(
            {
                **common, "method": "ward_dutton_mobile_channel",
                "scope": "linear partition of signed MoS2 mobile charge only",
                "Qg_C": math.nan, "Qd_C": ward.get("Qd_mobile_C", math.nan),
                "Qs_C": ward.get("Qs_mobile_C", math.nan),
                "Qmobile_C": ward.get("Qmobile_channel_C", math.nan),
                "support_status": "mobile_partition_only",
                "limitation": "excludes trap, fixed, dielectric, and electrode charge",
            }
        )
    return rows


def _read_bool_csv_all_passed(path: Path, column: str = "passed") -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return bool(rows) and all(str(row.get(column, "")).strip().lower() == "true" for row in rows)


def _dc_regression_valid(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return bool(rows) and all(
        str(row.get("candidate_present", "")).strip().lower() == "true"
        and str(row.get("candidate_finite", "")).strip().lower() == "true"
        and str(row.get("candidate_converged", "")).strip().lower() == "true"
        for row in rows
    )


def determine_acceptance(
    bundles: Mapping[str, Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    charge_rows: Sequence[Mapping[str, Any]],
    cap_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    derivative_rows = _derivative_rows(bundles)
    criteria = {
        "nondegenerate_contact_topology": all(
            bool(bundle["topology"]["passed"]) for bundle in bundles.values()
        ),
        "all_reference_points_finite_and_restored": all(
            bool(point["passed"])
            for bundle in bundles.values() for point in bundle["points"]
        ),
        "repeat_reproducibility": bool(repeat_rows) and all(
            bool(row["passed"]) for row in repeat_rows
        ),
        "fine_extra_fine_charge_convergence": bool(charge_rows) and all(
            bool(row["passed"]) for row in charge_rows
        ),
        "fine_extra_fine_capacitance_convergence": bool(cap_rows) and all(
            bool(row["passed"]) for row in cap_rows
        ),
        "global_gauss_balance": all(
            bool(point["global_gauss"]["passed"])
            for bundle in bundles.values() for point in bundle["points"]
        ),
        "derivative_gauss_balance": bool(derivative_rows) and all(
            bool(row["derivative_gauss_passed"]) for row in derivative_rows
        ),
        "gauge_invariance_row_sums": all(
            bool(point["gauge_row_validation"]["passed"])
            for bundle in bundles.values() for point in bundle["points"]
        ),
        "ward_dutton_mobile_identity": all(
            bool(point["ward"].get("converged"))
            for bundle in bundles.values() for point in bundle["points"]
        ),
        "dc_regression_complete_finite": _dc_regression_valid(
            DC_CANDIDATE_DIRECTORY / "contact_topology_dc_regression.csv"
        ),
        "dc_mesh_convergence": _read_bool_csv_all_passed(
            DC_CANDIDATE_DIRECTORY / "mesh_convergence_dc.csv",
            column="within_warning_threshold",
        ),
    }
    if all(criteria.values()):
        status = "supported"
    elif criteria["nondegenerate_contact_topology"]:
        status = "provisional"
    else:
        status = "unsupported"
    return {"status": status, "criteria": criteria}


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(fieldnames), extrasaction="raise")
        writer.writeheader()
        expected = set(fieldnames)
        for index, row in enumerate(rows):
            actual = set(row)
            if actual != expected:
                raise ValueError(
                    f"CSV row {index} schema mismatch for {path.name}: "
                    f"missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)}"
                )
            writer.writerow(dict(row))
    temporary.replace(path)


def _copy_csv(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(
            f"DC candidate must be generated before Q/C outputs: {source}"
        )
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = tuple(reader.fieldnames or ())
        rows = list(reader)
    if not fieldnames:
        raise RuntimeError(f"CSV has no header: {source}")
    _write_csv(destination, fieldnames, rows)


def build_weighting_feasibility() -> str:
    return """# Weighting-potential / Green-reciprocity feasibility

Status: **feasible in principle; not implemented in this candidate**.

- The corrected gate/drain/source contact sets are nondegenerate Dirichlet sets,
  so three auxiliary Laplace solves can be posed on the identical dielectric
  geometry: one terminal at 1 and the other two at 0.
- A defensible implementation must specify every remaining boundary as natural
  Neumann or an explicit fourth boundary. The present Air-supported internal
  mesh must not silently become a fourth electrical terminal.
- Before using a weighting solution, verify `psi_g + psi_d + psi_s ~= 1`,
  Green reciprocity, mesh convergence, and agreement between induced electrode
  charge and direct contact reaction flux.
- Such solutions could partition mobile/trap/fixed charge and form a Maxwell
  geometric capacitance matrix. They do not follow automatically from the
  present nonlinear drift-diffusion reference solve.

No weighting-potential solve, charge partition, scaling factor, planar-area
multiplier, or fabricated Qd/Qs value is introduced here.
"""


def build_readme(
    bundles: Mapping[str, Mapping[str, Any]], acceptance: Mapping[str, Any],
) -> str:
    base_topology = bundles["base"]["topology"]["contacts"]
    lines = [
        "# Source/Drain contact-topology and Q/C candidate",
        "",
        f"Physical electrode contact-flux status: **{acceptance['status']}**.",
        "Full compact-model terminal-charge status: **unsupported in this work**.",
        "",
        "## Corrected topology",
        "",
    ]
    for contact in TERMINALS:
        row = base_topology[contact]
        lines.append(
            f"- {contact}: edges={row.get('edge_count')}, "
            f"area={float(row.get('runtime_area_cm2', math.nan)):.12e} cm^2, "
            f"normal=({float(row.get('normal_r', math.nan)):+.1f},"
            f"{float(row.get('normal_z', math.nan)):+.1f}), "
            f"passed={row.get('passed')}"
        )
    lines.extend(["", "## Acceptance gates", ""])
    for name, passed in acceptance["criteria"].items():
        lines.append(f"- {name}: {'PASS' if passed else 'FAIL'}")
    lines.extend(
        [
            "",
            "## Definitions and limitations",
            "",
            "- Qg/Qd/Qs are DEVSIM PotentialEquation contact reaction fluxes "
            "with CylindricalEdgeCouple; no manual sign flip is used.",
            "- Qmobile/Qtrap/Qfixed use CylindricalNodeVolume.",
            "- Noncontact boundary charge is not back-solved from the residual. "
            "The audited discrete model has no separately assembled term and "
            "uses homogeneous natural boundaries there.",
            "- Contact-only capacitance column sums are diagnostics. Acceptance "
            "uses derivative Gauss balance including internal charge derivatives.",
            "- Gauge invariance is checked with measured-terminal row sums and "
            "independent common-mode finite differences.",
            "- Ward-Dutton remains a mobile-channel-only partition.",
            "- Weighting-potential/Green reciprocity is feasibility-only.",
            "- No ERASE or retention simulation is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundles: Mapping[str, Mapping[str, Any]], acceptance: Mapping[str, Any],
    *, directory: Path = CANDIDATE_DIRECTORY,
) -> tuple[Path, ...]:
    if directory.resolve() != CANDIDATE_DIRECTORY.resolve():
        raise ValueError("candidate output directory is fixed")
    directory.mkdir(parents=True, exist_ok=True)
    after_rows = _topology_rows(bundles)
    area_rows = _area_rows(bundles)
    repeat_rows = _repeat_rows(bundles)
    mesh_level_rows = _mesh_level_rows(bundles)
    charge_rows, cap_rows = _mesh_convergence_rows(bundles)
    direct_rows = _direct_rows(bundles, str(acceptance["status"]))
    derivative_rows = _derivative_rows(bundles)
    ward_rows = _ward_rows(bundles)
    method_rows = _method_rows(bundles, str(acceptance["status"]))
    specs = (
        ("contact_topology_audit_before.csv", TOPOLOGY_FIELDS, _before_topology_rows()),
        ("contact_topology_audit_after.csv", TOPOLOGY_FIELDS, after_rows),
        ("contact_area_validation.csv", AREA_FIELDS, area_rows),
        ("reference_reset_reproducibility.csv", RESET_FIELDS, repeat_rows),
        ("mesh_levels.csv", MESH_LEVEL_FIELDS, mesh_level_rows),
        ("mesh_convergence_charge.csv", MESH_CHARGE_FIELDS, charge_rows),
        ("mesh_convergence_capacitance.csv", MESH_CAP_FIELDS, cap_rows),
        ("direct_contact_flux_after_contact_fix.csv", DIRECT_FIELDS, direct_rows),
        ("electrode_capacitance_derivative_balance.csv", DERIVATIVE_FIELDS, derivative_rows),
        ("ward_dutton_mobile_partition_after_fix.csv", WARD_FIELDS, ward_rows),
        ("method_comparison_after_contact_fix.csv", METHOD_FIELDS, method_rows),
    )
    written: list[Path] = []
    for filename, fields, rows in specs:
        path = directory / filename
        _write_csv(path, fields, rows)
        written.append(path)
    for filename in ("contact_topology_dc_regression.csv", "mesh_convergence_dc.csv"):
        path = directory / filename
        _copy_csv(DC_CANDIDATE_DIRECTORY / filename, path)
        written.append(path)
    feasibility = directory / "weighting_potential_feasibility.md"
    feasibility.write_text(build_weighting_feasibility(), encoding="utf-8")
    written.append(feasibility)
    readme = directory / "contact_topology_qc_README.md"
    readme.write_text(build_readme(bundles, acceptance), encoding="utf-8")
    written.append(readme)
    if {path.name for path in written} != set(OUTPUT_FILENAMES):
        raise RuntimeError("candidate output file set mismatch")
    return tuple(written)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate isolated contact-topology and electrode Q/C candidates."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-level", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--state-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bias-name", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.worker:
        required = (
            arguments.mesh_level, arguments.mesh_scale, arguments.state_index,
            arguments.bias_name, arguments.worker_output,
        )
        if any(value is None for value in required):
            raise SystemExit("worker mode requires mesh/state/bias/output arguments")
        bundle = run_point_worker(
            mesh_level=str(arguments.mesh_level),
            mesh_scale=float(arguments.mesh_scale),
            state_index=int(arguments.state_index),
            bias_name=str(arguments.bias_name),
        )
        _write_json(Path(arguments.worker_output), bundle)
        return 0

    immutable_hashes = capture_immutable_hashes()
    with tempfile.TemporaryDirectory(prefix="contact_topology_qc_") as name:
        bundles = run_isolated_workers(
            Path(name), jobs=max(1, int(arguments.jobs))
        )
    assert_immutable_hashes(immutable_hashes)
    repeat_rows = _repeat_rows(bundles)
    charge_rows, cap_rows = _mesh_convergence_rows(bundles)
    acceptance = determine_acceptance(bundles, repeat_rows, charge_rows, cap_rows)
    outputs = write_outputs(bundles, acceptance)
    assert_immutable_hashes(immutable_hashes)
    print(f"Physical electrode contact-flux status: {acceptance['status']}")
    for criterion, passed in acceptance["criteria"].items():
        print(f"  {criterion}: {passed}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
