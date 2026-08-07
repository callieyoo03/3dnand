"""Run isolated current-integration, mesh-quality, and profile diagnostics.

This is a characterization-only runner.  It never imports ERASE or retention
code, and it writes only the four root-cause diagnostic CSV files below.  Each
electrical operating point is constructed and solved in a fresh subprocess so
that a failed State_4/off ramp cannot contaminate another point.

The manual contact-current calculation is delegated to
``qc_root_cause_metrics``.  In particular, neither an annular-area multiplier
nor a contact-edge-count divisor is applied here.
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
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import qc_root_cause_metrics as root_metrics


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "qc_root_cause_diagnostics"
)
SCHEMA_VERSION = "qc_root_cause_current_mesh_profile_v1"

CURRENT_SUMMARY_FILENAME = "current_api_manual_crosscheck.csv"
CURRENT_EDGE_FILENAME = "contact_edge_current_contributions.csv"
MESH_QUALITY_FILENAME = "mesh_quality_diagnostic.csv"
PROFILE_FILENAME = "contact_spatial_profiles.csv"

PROTECTED_DIRECTORIES = (
    REPOSITORY_ROOT / "shared_data",
    MODULE_DIRECTORY / "results" / "contact_topology_qc_candidate",
    MODULE_DIRECTORY / "results" / "contact_topology_dc_candidate",
    MODULE_DIRECTORY / "results" / "full_terminal_qc_candidate",
)

MESH_SPECS = (
    ("base", 1.0),
    ("fine", 0.5),
    ("extra_fine", 0.25),
)
PROFILE_DISTANCES_NM = (0.0, 1.0, 2.0, 5.0, 10.0)
REFERENCE_VDS_V = 0.05
REFERENCE_VS_V = 0.0


CURRENT_SUMMARY_FIELDS = (
    "schema_version",
    "mesh_level",
    "mesh_scale",
    "state_index",
    "state",
    "bias_name",
    "target_VGS_V",
    "target_VDS_V",
    "target_VS_V",
    "target_Ntrap_cm3",
    "actual_VGS_V",
    "actual_VDS_V",
    "actual_VS_V",
    "actual_Ntrap_cm3",
    "target_reached",
    "profile_solution_valid",
    "last_converged_revalidated",
    "source_api_current_A",
    "source_manual_current_A",
    "source_manual_minus_api_A",
    "source_absolute_difference_A",
    "source_relative_error",
    "source_api_manual_passed",
    "drain_api_current_A",
    "drain_manual_current_A",
    "drain_manual_minus_api_A",
    "drain_absolute_difference_A",
    "drain_relative_error",
    "drain_api_manual_passed",
    "api_source_plus_drain_continuity_residual_A",
    "manual_source_plus_drain_continuity_residual_A",
    "api_manual_residual_difference_A",
    "worker_process_id",
    "point_error_message",
    "extraction_error_message",
)

CURRENT_EDGE_FIELDS = (
    "schema_version",
    "mesh_level",
    "mesh_scale",
    "state_index",
    "state",
    "bias_name",
    "target_VGS_V",
    "target_VDS_V",
    "target_VS_V",
    "target_Ntrap_cm3",
    "actual_VGS_V",
    "actual_VDS_V",
    "actual_VS_V",
    "actual_Ntrap_cm3",
    "target_reached",
    "profile_solution_valid",
    "last_converged_revalidated",
    "contact",
    "region",
    "edge_available",
    "edge_index",
    "contact_node_index",
    "region_node_index",
    "contact_endpoint",
    "orientation_sign",
    "midpoint_r_cm",
    "midpoint_r_nm",
    "midpoint_z_cm",
    "midpoint_z_nm",
    "contact_r_cm",
    "contact_z_cm",
    "region_r_cm",
    "region_z_cm",
    "inner_corner_distance_cm",
    "inner_corner_distance_nm",
    "outer_corner_distance_cm",
    "outer_corner_distance_nm",
    "nearest_corner_distance_cm",
    "nearest_corner_distance_nm",
    "electron_current_A_per_cm2",
    "cylindrical_edge_couple_cm2",
    "signed_current_contribution_A",
    "absolute_current_contribution_A",
    "absolute_contribution_fraction",
    "rank_by_absolute_contribution",
    "cumulative_absolute_contribution_fraction",
    "contact_api_current_A",
    "contact_manual_current_A",
    "worker_process_id",
    "error_message",
)

MESH_QUALITY_FIELDS = (
    "schema_version",
    "mesh_level",
    "mesh_scale",
    "row_type",
    "region",
    "contact",
    "node_index",
    "node_r_cm",
    "node_z_cm",
    "region_node_count",
    "region_edge_count",
    "triangle_count",
    "valid_triangle_count",
    "degenerate_triangle_count",
    "triangle_minimum_angle_deg",
    "triangle_minimum_angle_p05_deg",
    "triangle_minimum_angle_median_deg",
    "triangle_maximum_angle_deg",
    "triangle_maximum_aspect_ratio",
    "triangle_aspect_ratio_p95",
    "obtuse_triangle_count",
    "obtuse_triangle_fraction",
    "radial_spacing_min_cm",
    "radial_spacing_max_cm",
    "axial_spacing_min_cm",
    "axial_spacing_max_cm",
    "edge_length_min_cm",
    "edge_length_max_cm",
    "contact_node_count",
    "active_incident_edge_count",
    "total_region_edge_valence",
    "active_incident_edge_valence",
    "total_valence_minimum",
    "total_valence_maximum",
    "active_valence_minimum",
    "active_valence_maximum",
    "incident_couple_count",
    "incident_couple_minimum_cm2",
    "incident_couple_p05_cm2",
    "incident_couple_median_cm2",
    "incident_couple_p95_cm2",
    "incident_couple_maximum_cm2",
    "incident_couple_mean_cm2",
    "incident_couple_sum_cm2",
    "edge_couple_model",
    "worker_process_id",
    "error_message",
)

PROFILE_FIELDS = (
    "schema_version",
    "mesh_level",
    "mesh_scale",
    "state_index",
    "state",
    "bias_name",
    "path_id",
    "target_VGS_V",
    "target_VDS_V",
    "target_VS_V",
    "target_Ntrap_cm3",
    "actual_VGS_V",
    "actual_VDS_V",
    "actual_VS_V",
    "actual_Ntrap_cm3",
    "target_reached",
    "profile_solution_valid",
    "last_converged_revalidated",
    "solution_status",
    "contact",
    "region",
    "plane_axis",
    "plane_coordinate_cm",
    "inward_coordinate_sign",
    "requested_distance_nm",
    "within_region_depth",
    "sample_available",
    "edge_index",
    "node0",
    "node1",
    "interpolation_fraction_n0_to_n1",
    "plane_intersection_relation",
    "profile_sample_semantics",
    "sample_r_cm",
    "sample_r_nm",
    "sample_z_cm",
    "sample_z_nm",
    "potential_V",
    "electrons_cm3",
    "electric_field_V_per_cm",
    "electron_current_A_per_cm2",
    "cylindrical_edge_couple_cm2",
    "electron_current_times_couple_A",
    "shallow_side_orientation_sign",
    "shallow_side_oriented_current_A",
    "edge_r0_cm",
    "edge_z0_cm",
    "edge_r1_cm",
    "edge_z1_cm",
    "worker_process_id",
    "point_error_message",
    "profile_error_message",
)


def build_tasks() -> tuple[dict[str, Any], ...]:
    """Return the minimal set of unique fresh-process electrical points."""

    tasks: list[dict[str, Any]] = []
    for mesh_level, mesh_scale in MESH_SPECS:
        if mesh_level != "extra_fine":
            tasks.append(
                {
                    "mesh_level": mesh_level,
                    "mesh_scale": mesh_scale,
                    "state_index": 0,
                    "bias_name": "off",
                    "target_VGS_V": -1.0,
                    "do_current": True,
                    "do_profile": False,
                    "do_mesh_quality": False,
                }
            )
        tasks.append(
            {
                "mesh_level": mesh_level,
                "mesh_scale": mesh_scale,
                "state_index": 0,
                "bias_name": "on",
                "target_VGS_V": 3.0,
                "do_current": mesh_level != "extra_fine",
                "do_profile": True,
                "do_mesh_quality": True,
            }
        )
        tasks.append(
            {
                "mesh_level": mesh_level,
                "mesh_scale": mesh_scale,
                "state_index": 4,
                "bias_name": "off",
                "target_VGS_V": -1.0,
                "do_current": False,
                "do_profile": True,
                "do_mesh_quality": False,
            }
        )
    return tuple(tasks)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(data),
            stream,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
    temporary.replace(path)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(path: str | Path) -> Path:
    destination = Path(path).resolve()
    for protected in PROTECTED_DIRECTORIES:
        protected_resolved = protected.resolve()
        if destination == protected_resolved or _is_relative_to(
            destination, protected_resolved
        ):
            raise ValueError(
                f"diagnostic output may not be written under {protected_resolved}"
            )
    return destination


def _protected_file_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for directory in PROTECTED_DIRECTORIES:
        if not directory.exists():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            hashes[str(path.resolve())] = digest.hexdigest()
    return hashes


def _runtime_state(device: str) -> dict[str, float]:
    from devsim import get_parameter
    import trap_parameters as trap_parameters
    from trap_models import TRAPPED_ELECTRON_PARAMETER

    source = float(get_parameter(device=device, name="source_bias"))
    gate = float(get_parameter(device=device, name="gate_bias"))
    drain = float(get_parameter(device=device, name="drain_bias"))
    ntrap = float(
        get_parameter(
            device=device,
            region=trap_parameters.CHARGE_TRAP_REGION,
            name=TRAPPED_ELECTRON_PARAMETER,
        )
    )
    return {
        "actual_VGS_V": gate - source,
        "actual_VDS_V": drain - source,
        "actual_VS_V": source,
        "actual_Ntrap_cm3": ntrap,
    }


def _target_matches(actual: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    comparisons = (
        ("actual_VGS_V", "target_VGS_V", 1.0e-12),
        ("actual_VDS_V", "target_VDS_V", 1.0e-12),
        ("actual_VS_V", "target_VS_V", 1.0e-12),
        ("actual_Ntrap_cm3", "target_Ntrap_cm3", 1.0),
    )
    return all(
        math.isclose(
            float(actual[actual_name]),
            float(target[target_name]),
            rel_tol=0.0,
            abs_tol=tolerance,
        )
        for actual_name, target_name, tolerance in comparisons
    )


def _point_common(bundle: Mapping[str, Any]) -> dict[str, Any]:
    status = bundle["point_status"]
    return {
        "schema_version": SCHEMA_VERSION,
        "mesh_level": bundle["mesh_level"],
        "mesh_scale": bundle["mesh_scale"],
        "state_index": status["state_index"],
        "state": status["state"],
        "bias_name": status["bias_name"],
        "target_VGS_V": status["target_VGS_V"],
        "target_VDS_V": status["target_VDS_V"],
        "target_VS_V": status["target_VS_V"],
        "target_Ntrap_cm3": status["target_Ntrap_cm3"],
        "actual_VGS_V": status["actual_VGS_V"],
        "actual_VDS_V": status["actual_VDS_V"],
        "actual_VS_V": status["actual_VS_V"],
        "actual_Ntrap_cm3": status["actual_Ntrap_cm3"],
        "target_reached": status["target_reached"],
        "profile_solution_valid": status["profile_solution_valid"],
        "last_converged_revalidated": status["last_converged_revalidated"],
        "worker_process_id": bundle.get("worker_process_id", ""),
    }


def _failed_point_bundle(task: Mapping[str, Any], error_message: str) -> dict[str, Any]:
    state_index = int(task["state_index"])
    return {
        **dict(task),
        "worker_process_id": "",
        "point_status": {
            "state_index": state_index,
            "state": "State_0_Empty" if state_index == 0 else "State_4_Programmed",
            "bias_name": str(task["bias_name"]),
            "target_VGS_V": float(task["target_VGS_V"]),
            "target_VDS_V": REFERENCE_VDS_V,
            "target_VS_V": REFERENCE_VS_V,
            "target_Ntrap_cm3": 0.0 if state_index == 0 else 2.0e18,
            "actual_VGS_V": math.nan,
            "actual_VDS_V": math.nan,
            "actual_VS_V": math.nan,
            "actual_Ntrap_cm3": math.nan,
            "target_reached": False,
            "profile_solution_valid": False,
            "last_converged_revalidated": False,
            "solution_status": "worker_failed",
            "point_error_message": error_message,
        },
        "current_crosschecks": {},
        "current_error_message": error_message,
        "mesh_quality": None,
        "mesh_quality_error_message": error_message,
        "profiles": {},
        "profile_errors": {"source": error_message, "drain": error_message},
    }


def run_point_worker(task: Mapping[str, Any]) -> dict[str, Any]:
    """Build and evaluate exactly one requested operating point."""

    import run_memory_window as memory_window
    import state_characterization_config as state_config
    from run_contact_topology_dc_candidate import scaled_runtime_mesh_lines
    from state_sweep_helpers import (
        initialize_characterization_device,
        ramp_terminal,
        ramp_trap_density,
    )

    mesh_level = str(task["mesh_level"])
    mesh_scale = float(task["mesh_scale"])
    state_index = int(task["state_index"])
    states = tuple(
        state
        for state in state_config.MEMORY_STATES
        if int(state["state_index"]) == state_index
    )
    if len(states) != 1:
        raise ValueError(f"unknown state_index={state_index}")
    state = states[0]
    target = {
        "state_index": state_index,
        "state": str(state["state"]),
        "bias_name": str(task["bias_name"]),
        "target_VGS_V": float(task["target_VGS_V"]),
        "target_VDS_V": REFERENCE_VDS_V,
        "target_VS_V": REFERENCE_VS_V,
        "target_Ntrap_cm3": float(state["ntrap_cm3"]),
    }
    line_log: list[dict[str, Any]] = []
    with scaled_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        mesh_scale,
        line_log,
    ):
        geometry, point = initialize_characterization_device()

    quality_report: Mapping[str, Any] | None = None
    quality_error = ""
    if bool(task["do_mesh_quality"]):
        try:
            quality_report = dict(root_metrics.runtime_mesh_quality_metrics(
                memory_window.device,
                "MoS2",
                contacts=("source", "drain"),
            ))
            # Per-element rows are not part of the requested CSV and would
            # make the fresh-worker JSON unnecessarily large.
            quality_report["triangle_quality"] = dict(
                quality_report["triangle_quality"]
            )
            quality_report["triangle_quality"].pop("elements", None)
            additional_reports: dict[str, Any] = {}
            additional_errors: dict[str, str] = {}
            for region in (
                "CoreOxide",
                "TunnelOxide",
                "ChargeTrap",
                "BlockingOxide",
            ):
                try:
                    report = dict(root_metrics.runtime_mesh_quality_metrics(
                        memory_window.device, region, contacts=()
                    ))
                    report["triangle_quality"] = dict(report["triangle_quality"])
                    report["triangle_quality"].pop("elements", None)
                    additional_reports[region] = report
                except Exception as error:
                    additional_errors[region] = f"{type(error).__name__}: {error}"
            quality_report["additional_region_reports"] = additional_reports
            quality_report["additional_region_errors"] = additional_errors
        except Exception as error:
            quality_error = f"{type(error).__name__}: {error}"

    point_error = ""
    point_traceback = ""
    try:
        # Path A: Ntrap -> VDS=0.05 V -> VGS target.
        ramp_trap_density(
            point,
            target["target_Ntrap_cm3"],
            label=f"root-cause {mesh_level} Path A trap",
        )
        ramp_terminal(
            point,
            "drain",
            target["target_VDS_V"],
            label=f"root-cause {mesh_level} Path A drain",
        )
        ramp_terminal(
            point,
            "gate",
            target["target_VGS_V"],
            label=f"root-cause {mesh_level} Path A gate",
        )
    except Exception as error:
        point_error = f"{type(error).__name__}: {error}"
        point_traceback = traceback.format_exc()

    actual = _runtime_state(memory_window.device)
    target_reached = not point_error and _target_matches(actual, target)
    profile_solution_valid = target_reached
    last_converged_revalidated = False
    revalidation_error = ""
    if point_error:
        try:
            # The established adaptive ramp restores the last converged bias
            # before raising.  Re-solve that *actual* coordinate with the
            # unchanged default solver settings before exporting a profile.
            memory_window.solve_dc()
        except Exception as error:
            revalidation_error = f"{type(error).__name__}: {error}"
        else:
            profile_solution_valid = True
            last_converged_revalidated = True
            actual = _runtime_state(memory_window.device)

    if target_reached:
        solution_status = "target_converged"
    elif last_converged_revalidated:
        solution_status = "last_converged_revalidated_after_target_failure"
    else:
        solution_status = "no_verified_solution"
    combined_error = point_error
    if revalidation_error:
        combined_error = (
            combined_error + "; last-coordinate revalidation failed: " + revalidation_error
        ).strip("; ")

    point_status = {
        **target,
        **actual,
        "target_reached": target_reached,
        "profile_solution_valid": profile_solution_valid,
        "last_converged_revalidated": last_converged_revalidated,
        "solution_status": solution_status,
        "point_error_message": combined_error,
        "point_exception_traceback": point_traceback,
    }

    current_reports: dict[str, Any] = {}
    current_error = ""
    if bool(task["do_current"]):
        if not target_reached:
            current_error = "target bias was not reached; current crosscheck not evaluated"
        else:
            try:
                for contact in ("source", "drain"):
                    current_reports[contact] = root_metrics.manual_contact_current_audit(
                        memory_window.device,
                        contact,
                        region="MoS2",
                    )
            except Exception as error:
                current_error = f"{type(error).__name__}: {error}"

    profiles: dict[str, Any] = {}
    profile_errors: dict[str, str] = {}
    if bool(task["do_profile"]):
        for contact in ("source", "drain"):
            if not profile_solution_valid:
                profile_errors[contact] = "no verified solution is available"
                continue
            try:
                profiles[contact] = root_metrics.extract_contact_spatial_profile_rows(
                    memory_window.device,
                    contact,
                    region="MoS2",
                    distances_nm=PROFILE_DISTANCES_NM,
                )
            except Exception as error:
                profile_errors[contact] = f"{type(error).__name__}: {error}"

    return {
        **dict(task),
        "geometry": dict(geometry),
        "mesh_line_log": line_log,
        "worker_process_id": os.getpid(),
        "point_status": point_status,
        "current_crosschecks": current_reports,
        "current_error_message": current_error,
        "mesh_quality": quality_report,
        "mesh_quality_error_message": quality_error,
        "profiles": profiles,
        "profile_errors": profile_errors,
    }


def build_worker_command(
    python_executable: str,
    *,
    task: Mapping[str, Any],
    worker_output: str | Path,
) -> list[str]:
    command = [
        str(python_executable),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--mesh-level",
        str(task["mesh_level"]),
        "--mesh-scale",
        f"{float(task['mesh_scale']):.17g}",
        "--state-index",
        str(int(task["state_index"])),
        "--bias-name",
        str(task["bias_name"]),
        "--target-vgs",
        f"{float(task['target_VGS_V']):.17g}",
        "--worker-output",
        str(Path(worker_output).resolve()),
    ]
    for flag, key in (
        ("--do-current", "do_current"),
        ("--do-profile", "do_profile"),
        ("--do-mesh-quality", "do_mesh_quality"),
    ):
        if bool(task[key]):
            command.append(flag)
    return command


def _task_stem(task: Mapping[str, Any]) -> str:
    return (
        f"{task['mesh_level']}_s{int(task['state_index'])}_"
        f"{task['bias_name']}"
    )


def run_fresh_workers(
    temporary_directory: str | Path,
    *,
    python_executable: str = sys.executable,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    """Run one fresh process per point and retain explicit failure bundles."""

    directory = Path(temporary_directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundles: list[dict[str, Any]] = []
    for task in build_tasks():
        stem = _task_stem(task)
        output = directory / f"{stem}.json"
        log_path = directory / f"{stem}.log"
        command = build_worker_command(
            python_executable,
            task=task,
            worker_output=output,
        )
        if subprocess_run is subprocess.run:
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                completed = subprocess_run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        else:
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        return_code = int(getattr(completed, "returncode", 0))
        if return_code or not output.exists():
            tail = ""
            if log_path.exists():
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            error = (
                f"fresh worker failed with return code {return_code}"
                + (f": {tail}" if tail else "")
            )
            bundles.append(_failed_point_bundle(task, error))
            continue
        with output.open("r", encoding="utf-8") as stream:
            bundles.append(json.load(stream))
    return bundles


def _current_rows(
    bundle: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = _point_common(bundle)
    point_error = str(bundle["point_status"].get("point_error_message", ""))
    extraction_error = str(bundle.get("current_error_message", ""))
    reports = bundle.get("current_crosschecks", {})
    summary_row = {field: "" for field in CURRENT_SUMMARY_FIELDS}
    summary_row.update(common)
    summary_row.update(
        {
            "point_error_message": point_error,
            "extraction_error_message": extraction_error,
        }
    )
    edge_rows: list[dict[str, Any]] = []
    api_values: dict[str, float] = {}
    manual_values: dict[str, float] = {}
    for contact in ("source", "drain"):
        report = reports.get(contact)
        if not isinstance(report, Mapping):
            placeholder = {field: "" for field in CURRENT_EDGE_FIELDS}
            placeholder.update(common)
            placeholder.update(
                {
                    "contact": contact,
                    "edge_available": False,
                    "edge_index": -1,
                    "error_message": extraction_error or "contact report unavailable",
                }
            )
            edge_rows.append(placeholder)
            continue
        comparison = report["summary"]
        api_value = float(comparison["api_current_A"])
        manual_value = float(comparison["manual_current_A"])
        api_values[contact] = api_value
        manual_values[contact] = manual_value
        prefix = contact
        summary_row.update(
            {
                f"{prefix}_api_current_A": api_value,
                f"{prefix}_manual_current_A": manual_value,
                f"{prefix}_manual_minus_api_A": comparison["manual_minus_api_A"],
                f"{prefix}_absolute_difference_A": comparison["absolute_difference_A"],
                f"{prefix}_relative_error": comparison["relative_error"],
                f"{prefix}_api_manual_passed": comparison["passed"],
            }
        )
        integral = report["manual_integral"]
        contributions = list(integral.get("edge_contributions", ()))
        ordered = sorted(
            contributions,
            key=lambda row: (-abs(float(row["contribution_A"])), int(row["edge_index"])),
        )
        absolute_total = math.fsum(abs(float(row["contribution_A"])) for row in ordered)
        cumulative = 0.0
        for rank, contribution in enumerate(ordered, start=1):
            absolute_value = abs(float(contribution["contribution_A"]))
            fraction = absolute_value / absolute_total if absolute_total else 0.0
            cumulative += fraction
            inner = float(contribution["inner_radial_corner_distance_cm"])
            outer = float(contribution["outer_radial_corner_distance_cm"])
            midpoint_r = float(contribution["r_cm"])
            midpoint_z = float(contribution["z_cm"])
            row = {field: "" for field in CURRENT_EDGE_FIELDS}
            row.update(common)
            row.update(
                {
                    "contact": contact,
                    "region": contribution["region"],
                    "edge_available": True,
                    "edge_index": contribution["edge_index"],
                    "contact_node_index": contribution["contact_node_index"],
                    "region_node_index": contribution["region_node_index"],
                    "contact_endpoint": contribution["contact_endpoint"],
                    "orientation_sign": contribution["orientation_sign"],
                    "midpoint_r_cm": midpoint_r,
                    "midpoint_r_nm": midpoint_r / root_metrics.NM_TO_CM,
                    "midpoint_z_cm": midpoint_z,
                    "midpoint_z_nm": midpoint_z / root_metrics.NM_TO_CM,
                    "contact_r_cm": contribution["contact_r_cm"],
                    "contact_z_cm": contribution["contact_z_cm"],
                    "region_r_cm": contribution["region_r_cm"],
                    "region_z_cm": contribution["region_z_cm"],
                    "inner_corner_distance_cm": inner,
                    "inner_corner_distance_nm": inner / root_metrics.NM_TO_CM,
                    "outer_corner_distance_cm": outer,
                    "outer_corner_distance_nm": outer / root_metrics.NM_TO_CM,
                    "nearest_corner_distance_cm": min(inner, outer),
                    "nearest_corner_distance_nm": min(inner, outer) / root_metrics.NM_TO_CM,
                    "electron_current_A_per_cm2": contribution["ElectronCurrent"],
                    "cylindrical_edge_couple_cm2": contribution["CylindricalEdgeCouple"],
                    "signed_current_contribution_A": contribution["contribution_A"],
                    "absolute_current_contribution_A": absolute_value,
                    "absolute_contribution_fraction": fraction,
                    "rank_by_absolute_contribution": rank,
                    "cumulative_absolute_contribution_fraction": cumulative,
                    "contact_api_current_A": api_value,
                    "contact_manual_current_A": manual_value,
                    "error_message": "",
                }
            )
            edge_rows.append(row)
    if set(api_values) == {"source", "drain"}:
        api_residual = api_values["source"] + api_values["drain"]
        manual_residual = manual_values["source"] + manual_values["drain"]
        summary_row.update(
            {
                "api_source_plus_drain_continuity_residual_A": api_residual,
                "manual_source_plus_drain_continuity_residual_A": manual_residual,
                "api_manual_residual_difference_A": manual_residual - api_residual,
            }
        )
    return summary_row, edge_rows


def _mesh_quality_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = {
        "schema_version": SCHEMA_VERSION,
        "mesh_level": bundle["mesh_level"],
        "mesh_scale": bundle["mesh_scale"],
        "worker_process_id": bundle.get("worker_process_id", ""),
    }
    report = bundle.get("mesh_quality")
    error_message = str(bundle.get("mesh_quality_error_message", ""))
    if not isinstance(report, Mapping):
        row = {field: "" for field in MESH_QUALITY_FIELDS}
        row.update(common)
        row.update({"row_type": "error", "region": "MoS2", "error_message": error_message})
        return [row]
    triangle = report["triangle_quality"]
    spacing = report["actual_region_edge_spacing"]
    minimum_distribution = triangle["minimum_angle_distribution_deg"]
    aspect_distribution = triangle["aspect_ratio_distribution"]
    region_row = {field: "" for field in MESH_QUALITY_FIELDS}
    region_row.update(common)
    region_row.update(
        {
            "row_type": "region_summary",
            "region": report["region"],
            "region_node_count": report["region_node_count"],
            "region_edge_count": report["region_edge_count"],
            "triangle_count": triangle["triangle_count"],
            "valid_triangle_count": triangle["valid_triangle_count"],
            "degenerate_triangle_count": triangle["degenerate_triangle_count"],
            "triangle_minimum_angle_deg": triangle["minimum_angle_deg"],
            "triangle_minimum_angle_p05_deg": minimum_distribution["p05"],
            "triangle_minimum_angle_median_deg": minimum_distribution["median"],
            "triangle_maximum_angle_deg": triangle["maximum_angle_deg"],
            "triangle_maximum_aspect_ratio": triangle["maximum_aspect_ratio"],
            "triangle_aspect_ratio_p95": aspect_distribution["p95"],
            "obtuse_triangle_count": triangle["obtuse_triangle_count"],
            "obtuse_triangle_fraction": triangle["obtuse_fraction"],
            "radial_spacing_min_cm": spacing["radial_cm"]["minimum"],
            "radial_spacing_max_cm": spacing["radial_cm"]["maximum"],
            "axial_spacing_min_cm": spacing["axial_cm"]["minimum"],
            "axial_spacing_max_cm": spacing["axial_cm"]["maximum"],
            "edge_length_min_cm": spacing["edge_length_cm"]["minimum"],
            "edge_length_max_cm": spacing["edge_length_cm"]["maximum"],
            "edge_couple_model": report["edge_couple_model"],
            "error_message": error_message,
        }
    )
    rows = [region_row]
    for contact in ("source", "drain"):
        data = report["contacts"][contact]
        coupling = data["incident_CylindricalEdgeCouple"]
        total_valence = data["total_region_edge_valence"]
        active_valence = data["active_incident_edge_valence"]
        contact_row = {field: "" for field in MESH_QUALITY_FIELDS}
        contact_row.update(common)
        contact_row.update(
            {
                "row_type": "contact_summary",
                "region": report["region"],
                "contact": contact,
                "contact_node_count": data["contact_node_count"],
                "active_incident_edge_count": data["active_incident_edge_count"],
                "total_valence_minimum": total_valence["minimum"],
                "total_valence_maximum": total_valence["maximum"],
                "active_valence_minimum": active_valence["minimum"],
                "active_valence_maximum": active_valence["maximum"],
                "incident_couple_count": coupling["count"],
                "incident_couple_minimum_cm2": coupling["minimum"],
                "incident_couple_p05_cm2": coupling["p05"],
                "incident_couple_median_cm2": coupling["median"],
                "incident_couple_p95_cm2": coupling["p95"],
                "incident_couple_maximum_cm2": coupling["maximum"],
                "incident_couple_mean_cm2": coupling["mean"],
                "incident_couple_sum_cm2": coupling["sum"],
                "edge_couple_model": report["edge_couple_model"],
                "error_message": error_message,
            }
        )
        rows.append(contact_row)
        for node in data["node_valence"]:
            node_row = {field: "" for field in MESH_QUALITY_FIELDS}
            node_row.update(common)
            node_row.update(
                {
                    "row_type": "contact_node",
                    "region": report["region"],
                    "contact": contact,
                    "node_index": node["node_index"],
                    "node_r_cm": node["r_cm"],
                    "node_z_cm": node["z_cm"],
                    "total_region_edge_valence": node["total_region_edge_valence"],
                    "active_incident_edge_valence": node["active_incident_edge_valence"],
                    "edge_couple_model": report["edge_couple_model"],
                    "error_message": error_message,
                }
            )
            rows.append(node_row)
    for region_name, region_report in sorted(
        report.get("additional_region_reports", {}).items()
    ):
        triangle = region_report["triangle_quality"]
        spacing = region_report["actual_region_edge_spacing"]
        minimum_distribution = triangle["minimum_angle_distribution_deg"]
        aspect_distribution = triangle["aspect_ratio_distribution"]
        row = {field: "" for field in MESH_QUALITY_FIELDS}
        row.update(common)
        row.update(
            {
                "row_type": "region_summary",
                "region": region_name,
                "region_node_count": region_report["region_node_count"],
                "region_edge_count": region_report["region_edge_count"],
                "triangle_count": triangle["triangle_count"],
                "valid_triangle_count": triangle["valid_triangle_count"],
                "degenerate_triangle_count": triangle["degenerate_triangle_count"],
                "triangle_minimum_angle_deg": triangle["minimum_angle_deg"],
                "triangle_minimum_angle_p05_deg": minimum_distribution["p05"],
                "triangle_minimum_angle_median_deg": minimum_distribution["median"],
                "triangle_maximum_angle_deg": triangle["maximum_angle_deg"],
                "triangle_maximum_aspect_ratio": triangle["maximum_aspect_ratio"],
                "triangle_aspect_ratio_p95": aspect_distribution["p95"],
                "obtuse_triangle_count": triangle["obtuse_triangle_count"],
                "obtuse_triangle_fraction": triangle["obtuse_fraction"],
                "radial_spacing_min_cm": spacing["radial_cm"]["minimum"],
                "radial_spacing_max_cm": spacing["radial_cm"]["maximum"],
                "axial_spacing_min_cm": spacing["axial_cm"]["minimum"],
                "axial_spacing_max_cm": spacing["axial_cm"]["maximum"],
                "edge_length_min_cm": spacing["edge_length_cm"]["minimum"],
                "edge_length_max_cm": spacing["edge_length_cm"]["maximum"],
                "edge_couple_model": region_report["edge_couple_model"],
                "error_message": "",
            }
        )
        rows.append(row)
    for region_name, region_error in sorted(
        report.get("additional_region_errors", {}).items()
    ):
        row = {field: "" for field in MESH_QUALITY_FIELDS}
        row.update(common)
        row.update(
            {
                "row_type": "error",
                "region": region_name,
                "error_message": region_error,
            }
        )
        rows.append(row)
    return rows


def _profile_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    common = _point_common(bundle)
    status = bundle["point_status"]
    common.update(
        {
            "path_id": "A",
            "solution_status": status["solution_status"],
            "point_error_message": status.get("point_error_message", ""),
        }
    )
    profiles = bundle.get("profiles", {})
    errors = bundle.get("profile_errors", {})
    rows: list[dict[str, Any]] = []
    for contact in ("source", "drain"):
        raw_rows = profiles.get(contact)
        if not isinstance(raw_rows, list) or not raw_rows:
            for distance in PROFILE_DISTANCES_NM:
                row = {field: "" for field in PROFILE_FIELDS}
                row.update(common)
                row.update(
                    {
                        "contact": contact,
                        "region": "MoS2",
                        "requested_distance_nm": distance,
                        "within_region_depth": "",
                        "sample_available": False,
                        "edge_index": -1,
                        "profile_error_message": errors.get(
                            contact, "profile data unavailable"
                        ),
                    }
                )
                rows.append(row)
            continue
        for sample in raw_rows:
            row = {field: "" for field in PROFILE_FIELDS}
            row.update(common)
            r_cm = sample.get("r_cm", "")
            z_cm = sample.get("z_cm", "")
            row.update(
                {
                    "contact": contact,
                    "region": sample.get("region", "MoS2"),
                    "plane_axis": sample.get("plane_axis", ""),
                    "plane_coordinate_cm": sample.get("plane_coordinate_cm", ""),
                    "inward_coordinate_sign": sample.get("inward_coordinate_sign", ""),
                    "requested_distance_nm": sample.get("requested_distance_nm", ""),
                    "within_region_depth": sample.get("within_region_depth", ""),
                    "sample_available": sample.get("sample_available", False),
                    "edge_index": sample.get("edge_index", -1),
                    "node0": sample.get("node0", ""),
                    "node1": sample.get("node1", ""),
                    "interpolation_fraction_n0_to_n1": sample.get(
                        "interpolation_fraction_n0_to_n1", ""
                    ),
                    "plane_intersection_relation": sample.get(
                        "plane_intersection_relation", ""
                    ),
                    "profile_sample_semantics": (
                        "raw_crossing_edge_sample_not_unique_cross_section;do_not_sum"
                    ),
                    "sample_r_cm": r_cm,
                    "sample_r_nm": (
                        float(r_cm) / root_metrics.NM_TO_CM if r_cm != "" else ""
                    ),
                    "sample_z_cm": z_cm,
                    "sample_z_nm": (
                        float(z_cm) / root_metrics.NM_TO_CM if z_cm != "" else ""
                    ),
                    "potential_V": sample.get("Potential_V", ""),
                    "electrons_cm3": sample.get("Electrons_cm3", ""),
                    "electric_field_V_per_cm": sample.get("ElectricField_V_per_cm", ""),
                    "electron_current_A_per_cm2": sample.get("ElectronCurrent", ""),
                    "cylindrical_edge_couple_cm2": sample.get(
                        "CylindricalEdgeCouple", ""
                    ),
                    "electron_current_times_couple_A": sample.get(
                        "ElectronCurrent_times_CylindricalEdgeCouple_A", ""
                    ),
                    "shallow_side_orientation_sign": sample.get(
                        "shallow_side_orientation_sign", ""
                    ),
                    "shallow_side_oriented_current_A": sample.get(
                        "shallow_side_oriented_current_A", ""
                    ),
                    "edge_r0_cm": sample.get("edge_r0_cm", ""),
                    "edge_z0_cm": sample.get("edge_z0_cm", ""),
                    "edge_r1_cm": sample.get("edge_r1_cm", ""),
                    "edge_z1_cm": sample.get("edge_z1_cm", ""),
                    "profile_error_message": errors.get(contact, ""),
                }
            )
            rows.append(row)
    return rows


def build_output_tables(
    bundles: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    current_summaries: list[dict[str, Any]] = []
    current_edges: list[dict[str, Any]] = []
    mesh_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    for bundle in bundles:
        if bool(bundle.get("do_current")):
            summary, edges = _current_rows(bundle)
            current_summaries.append(summary)
            current_edges.extend(edges)
        if bool(bundle.get("do_mesh_quality")):
            mesh_rows.extend(_mesh_quality_rows(bundle))
        if bool(bundle.get("do_profile")):
            profile_rows.extend(_profile_rows(bundle))
    mesh_order = {name: index for index, (name, _) in enumerate(MESH_SPECS)}
    contact_order = {"source": 0, "drain": 1}
    current_summaries.sort(
        key=lambda row: (
            mesh_order[str(row["mesh_level"])],
            int(row["state_index"]),
            str(row["bias_name"]),
        )
    )
    current_edges.sort(
        key=lambda row: (
            mesh_order[str(row["mesh_level"])],
            int(row["state_index"]),
            str(row["bias_name"]),
            contact_order.get(str(row["contact"]), 99),
            int(row["rank_by_absolute_contribution"] or 10**9),
            int(row["edge_index"]),
        )
    )
    row_type_order = {"region_summary": 0, "contact_summary": 1, "contact_node": 2, "error": 3}
    mesh_rows.sort(
        key=lambda row: (
            mesh_order[str(row["mesh_level"])],
            row_type_order.get(str(row["row_type"]), 99),
            contact_order.get(str(row["contact"]), 99),
            int(row["node_index"] or -1),
        )
    )
    profile_rows.sort(
        key=lambda row: (
            mesh_order[str(row["mesh_level"])],
            int(row["state_index"]),
            contact_order.get(str(row["contact"]), 99),
            float(row["requested_distance_nm"]),
            float(row["sample_r_cm"] or -1.0),
            int(row["edge_index"]),
        )
    )
    return {
        CURRENT_SUMMARY_FILENAME: current_summaries,
        CURRENT_EDGE_FILENAME: current_edges,
        MESH_QUALITY_FILENAME: mesh_rows,
        PROFILE_FILENAME: profile_rows,
    }


def _write_csv(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace stale temporary output: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.rename(path)


def write_outputs(
    output_directory: str | Path,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[Path, ...]:
    directory = validate_output_directory(output_directory)
    schemas = {
        CURRENT_SUMMARY_FILENAME: CURRENT_SUMMARY_FIELDS,
        CURRENT_EDGE_FILENAME: CURRENT_EDGE_FIELDS,
        MESH_QUALITY_FILENAME: MESH_QUALITY_FIELDS,
        PROFILE_FILENAME: PROFILE_FIELDS,
    }
    if set(tables) != set(schemas):
        raise ValueError("diagnostic output table set does not match the fixed schema")
    existing = [directory / filename for filename in schemas if (directory / filename).exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing diagnostic result(s): "
            + ", ".join(str(path) for path in existing)
        )
    written: list[Path] = []
    for filename in (
        CURRENT_SUMMARY_FILENAME,
        CURRENT_EDGE_FILENAME,
        MESH_QUALITY_FILENAME,
        PROFILE_FILENAME,
    ):
        path = directory / filename
        _write_csv(path, schemas[filename], tables[filename])
        written.append(path)
    return tuple(written)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mesh-level")
    parser.add_argument("--mesh-scale", type=float)
    parser.add_argument("--state-index", type=int)
    parser.add_argument("--bias-name")
    parser.add_argument("--target-vgs", type=float)
    parser.add_argument("--do-current", action="store_true")
    parser.add_argument("--do-profile", action="store_true")
    parser.add_argument("--do-mesh-quality", action="store_true")
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate schemas/task isolation without running DEVSIM",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.worker:
        required = (
            args.mesh_level,
            args.mesh_scale,
            args.state_index,
            args.bias_name,
            args.target_vgs,
            args.worker_output,
        )
        if any(value is None for value in required):
            raise ValueError("worker invocation is missing required arguments")
        task = {
            "mesh_level": args.mesh_level,
            "mesh_scale": args.mesh_scale,
            "state_index": args.state_index,
            "bias_name": args.bias_name,
            "target_VGS_V": args.target_vgs,
            "do_current": args.do_current,
            "do_profile": args.do_profile,
            "do_mesh_quality": args.do_mesh_quality,
        }
        try:
            bundle = run_point_worker(task)
        except Exception as error:
            bundle = _failed_point_bundle(
                task,
                f"{type(error).__name__}: {error}\n{traceback.format_exc()}",
            )
        _write_json(args.worker_output, bundle)
        return 0

    validate_output_directory(args.output_directory)
    if args.validate_only:
        tasks = build_tasks()
        if len(tasks) != len({_task_stem(task) for task in tasks}):
            raise RuntimeError("fresh-process task identifiers are not unique")
        print(f"Validated {len(tasks)} isolated point tasks and four CSV schemas.")
        return 0

    protected_before = _protected_file_hashes()
    with tempfile.TemporaryDirectory(prefix="qc-current-mesh-profile-") as temporary:
        bundles = run_fresh_workers(temporary)
    tables = build_output_tables(bundles)
    written = write_outputs(args.output_directory, tables)
    protected_after = _protected_file_hashes()
    if protected_after != protected_before:
        raise RuntimeError("a protected public/candidate file changed during diagnostics")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
