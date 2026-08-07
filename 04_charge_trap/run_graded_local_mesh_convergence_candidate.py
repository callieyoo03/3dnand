"""Run the quality-preserving graded local-mesh convergence candidate.

This is a candidate-only orchestrator.  It never calls a public data writer,
never replaces an existing result directory, and hashes every pre-existing
result plus ``shared_data``, ``05_program_erase``, ``_doc_review`` and the
intentional ``run_memory_window.py`` worktree change before and after the run.

The reference gate is deliberately split into DC-transport and electrostatic
Q/C domains.  Deep-off current below the candidate quantification floor is
preserved as raw diagnostic data but cannot fail the Q/C gate through its sign
or source/drain relative-continuity noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import run_local_mesh_convergence_candidate as base


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
RESULTS_DIRECTORY = MODULE_DIRECTORY / "results"
CANDIDATE_DIRECTORY = RESULTS_DIRECTORY / "graded_local_mesh_convergence_candidate"
SCHEMA_VERSION = "graded_local_mesh_convergence_candidate_v1"

REFERENCE_POINTS = base.REFERENCE_POINTS
TERMINALS = base.TERMINALS
DELTA_VOLTAGES_V = base.DELTA_VOLTAGES_V
POINT_BASE_FIELDS = base.POINT_FIELDS
QUALITY_BASE_FIELDS = base.QUALITY_FIELDS
write_csv = base.write_csv
OUTPUT_FILENAMES = (
    "graded_mesh_family_definition.csv",
    "graded_mesh_quality.csv",
    "graded_mesh_reference_points.csv",
    "graded_mesh_reference_convergence.csv",
)
FULL_DC_OUTPUT_FILENAMES = (
    "graded_mesh_full_dc_metrics.csv",
    "graded_mesh_full_dc_convergence.csv",
)

POINT_EXTRA_FIELDS = (
    "raw_ID_A",
    "raw_drain_current_A",
    "raw_source_current_A",
    "reported_ID_A",
    "dc_quantification_floor_A",
    "dc_current_scale_A",
    "dc_current_status",
    "raw_continuity_residual_A",
    "raw_continuity_tolerance_A",
    "raw_continuity_passed",
    "reported_Ioff_A",
    "reported_on_off_ratio",
    "ioff_status",
    "on_off_status",
    "continuity_gate_applicable",
    "dc_continuity_acceptance_status",
    "dc_transport_point_passed",
    "dc_transport_point_status",
    "electrostatic_qc_point_passed",
    "electrostatic_qc_point_status",
)
POINT_FIELDS = tuple(dict.fromkeys((*base.POINT_FIELDS, *POINT_EXTRA_FIELDS)))


def _dynamic_fields(
    rows: Sequence[Mapping[str, Any]], preferred: Sequence[str] = (),
) -> tuple[str, ...]:
    return base._dynamic_fields(rows, preferred)


def _number(value: Any) -> float:
    return base._number(value)


def _error(error: object | None) -> str:
    return base._error(error)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_staging_path(path: Path) -> bool:
    try:
        relative = path.resolve().relative_to(RESULTS_DIRECTORY.resolve())
    except ValueError:
        return False
    return bool(relative.parts) and relative.parts[0].startswith(
        ".graded_local_mesh_"
    )


def _protected_files() -> list[Path]:
    """Return every pre-existing file that this candidate must not mutate."""

    roots = (
        RESULTS_DIRECTORY,
        REPOSITORY_ROOT / "shared_data",
        REPOSITORY_ROOT / "05_program_erase",
        REPOSITORY_ROOT / "_doc_review",
    )
    files: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or _is_staging_path(path):
                continue
            try:
                path.resolve().relative_to(CANDIDATE_DIRECTORY.resolve())
            except ValueError:
                files.add(path.resolve())
    memory_window = MODULE_DIRECTORY / "run_memory_window.py"
    if memory_window.is_file():
        files.add(memory_window.resolve())
    return sorted(files, key=lambda path: str(path).lower())


def capture_protected_hashes() -> dict[str, str]:
    return {
        str(path.relative_to(REPOSITORY_ROOT.resolve())): _sha256(path)
        for path in _protected_files()
    }


def assert_protected_hashes(expected: Mapping[str, str]) -> None:
    actual = capture_protected_hashes()
    if dict(expected) != actual:
        changed = sorted(
            name
            for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError("protected input changed: " + ", ".join(changed))


def build_tasks() -> list[dict[str, Any]]:
    import graded_local_mesh_family as family

    tasks: list[dict[str, Any]] = [
        {"mode": "probe", "mesh_level": level, "run_role": "probe"}
        for level in family.LEVELS
    ]
    for level in family.LEVELS:
        for reference in REFERENCE_POINTS:
            tasks.append(
                {
                    "mode": "point",
                    "mesh_level": level,
                    "run_role": "primary",
                    "state_index": reference["state_index"],
                    "bias_name": reference["bias_name"],
                }
            )
    for reference in REFERENCE_POINTS:
        tasks.append(
            {
                "mode": "point",
                "mesh_level": "graded_ultra_fine",
                "run_role": "repeat",
                "state_index": reference["state_index"],
                "bias_name": reference["bias_name"],
            }
        )
    return tasks


def _task_stem(task: Mapping[str, Any]) -> str:
    if task["mode"] == "probe":
        return f"probe_{task['mesh_level']}"
    return (
        f"point_{task['mesh_level']}_{task['run_role']}_"
        f"s{int(task['state_index'])}_{task['bias_name']}"
    )


def build_worker_command(
    task: Mapping[str, Any], output_path: Path,
    python_executable: str = sys.executable,
) -> list[str]:
    command = [
        str(python_executable), "-B", str(Path(__file__).resolve()),
        "--worker-mode", str(task["mode"]),
        "--mesh-level", str(task["mesh_level"]),
        "--run-role", str(task.get("run_role", "primary")),
        "--worker-output", str(output_path.resolve()),
    ]
    if task["mode"] == "point":
        command.extend(
            [
                "--state-index", str(int(task["state_index"])),
                "--bias-name", str(task["bias_name"]),
            ]
        )
    return command


def _failed_bundle(task: Mapping[str, Any], error: str) -> dict[str, Any]:
    return {
        **dict(task),
        "points": [
            {
                "state_index": task.get("state_index", -1),
                "bias_name": task.get("bias_name", ""),
                "error_message": error,
            }
        ],
        "quality_rows": [],
        "error_message": error,
        "runtime_seconds": math.nan,
        "worker_process_id": -1,
    }


def run_fresh_workers(
    worker_directory: Path, *, jobs: int = 1,
    python_executable: str = sys.executable,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> list[dict[str, Any]]:
    worker_directory.mkdir(parents=True, exist_ok=False)
    tasks = build_tasks()

    def execute(task: Mapping[str, Any]) -> dict[str, Any]:
        stem = _task_stem(task)
        output = worker_directory / f"{stem}.json"
        log_path = worker_directory / f"{stem}.log"
        command = build_worker_command(task, output, python_executable)
        if subprocess_run is subprocess.run:
            with log_path.open("x", encoding="utf-8", newline="\n") as log:
                completed = subprocess_run(
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        else:
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        code = int(getattr(completed, "returncode", 0))
        if code or not output.is_file():
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            return _failed_bundle(
                task, f"worker return code {code}: {tail}".strip()
            )
        with output.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    if int(jobs) <= 1:
        bundles: list[dict[str, Any]] = []
        for index, task in enumerate(tasks, start=1):
            bundles.append(execute(task))
            print(
                f"graded-mesh worker {index}/{len(tasks)} {_task_stem(task)}",
                flush=True,
            )
        return bundles

    by_stem: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for index, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            by_stem[_task_stem(task)] = future.result()
            print(
                f"graded-mesh worker {index}/{len(tasks)} {_task_stem(task)}",
                flush=True,
            )
    return [by_stem[_task_stem(task)] for task in tasks]


def dispatch_full_dc_if_eligible(
    gate: Mapping[str, Any], *, dispatcher: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    if not (
        bool(gate.get("passed"))
        and bool(gate.get("full_sweep_allowed"))
    ):
        return {"executed": False, "reason": "combined reference gate did not pass"}
    if dispatcher is None:
        raise RuntimeError(
            "combined gate passed, but no candidate-only full-DC dispatcher is installed"
        )
    return {"executed": True, "result": dispatcher()}


COORDINATE_CLUSTER_TOLERANCE_CM = 1.0e-15


def _cluster_coordinates(coordinates: Sequence[Any]) -> list[float]:
    finite = sorted(
        value
        for value in (_number(item) for item in coordinates)
        if math.isfinite(value)
    )
    ordered: list[float] = []
    for value in finite:
        if not ordered or value - ordered[-1] > COORDINATE_CLUSTER_TOLERANCE_CM:
            ordered.append(value)
    return ordered


def _adjacent_ratio_summary(coordinates: Sequence[Any]) -> dict[str, Any]:
    ordered = _cluster_coordinates(coordinates)
    spacings = [
        right - left
        for left, right in zip(ordered, ordered[1:])
    ]
    ratios: list[tuple[float, int]] = []
    for index, (left, right) in enumerate(zip(spacings, spacings[1:])):
        if left > 0.0 and right > 0.0:
            ratios.append((max(left / right, right / left), index))
    worst_ratio, worst_index = (
        max(ratios, key=lambda item: item[0]) if ratios else (math.nan, -1)
    )
    return {
        "coordinate_count": len(ordered),
        "spacing_count": len(spacings),
        "spacing_min_cm": min(spacings) if spacings else math.nan,
        "spacing_max_cm": max(spacings) if spacings else math.nan,
        "adjacent_spacing_ratio_count": len(ratios),
        "adjacent_spacing_ratio_max": worst_ratio,
        "adjacent_spacing_ratio_p95": (
            base._percentile([item[0] for item in ratios], 0.95)
            if ratios else math.nan
        ),
        "worst_ratio_left_coordinate_cm": (
            ordered[worst_index] if worst_index >= 0 else math.nan
        ),
        "worst_ratio_middle_coordinate_cm": (
            ordered[worst_index + 1] if worst_index >= 0 else math.nan
        ),
        "worst_ratio_right_coordinate_cm": (
            ordered[worst_index + 2] if worst_index >= 0 else math.nan
        ),
    }


def _actual_spacing_ratio_rows(
    mesh_level: str, level_order: int, device: str,
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    from devsim import get_node_model_values

    import graded_local_mesh_family as family

    rows: list[dict[str, Any]] = []
    coordinates_by_direction: dict[str, list[float]] = {}
    for direction, model in (("radial", "x"), ("axial", "y")):
        coordinates = _cluster_coordinates(
            get_node_model_values(device=device, region="MoS2", name=model)
        )
        coordinates_by_direction[direction] = coordinates
        summary = _adjacent_ratio_summary(
            coordinates
        )
        nonempty = (
            int(summary["coordinate_count"]) >= 3
            and int(summary["spacing_count"]) >= 2
            and int(summary["adjacent_spacing_ratio_count"]) >= 1
        )
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "mesh_level": mesh_level,
                "level_order": level_order,
                "row_type": "adjacent_spacing_ratio",
                "region": "MoS2",
                "contact": "",
                "node_index": -1,
                "spacing_direction": direction,
                **summary,
                "adjacent_spacing_ratio_limit": 2.0,
                "adjacent_spacing_ratio_passed": bool(
                    nonempty
                    and math.isfinite(
                        _number(summary["adjacent_spacing_ratio_max"])
                    )
                    and _number(summary["adjacent_spacing_ratio_max"]) <= 2.0 + 1.0e-9
                ),
                "quality_passed": True,
                "error_message": "",
            }
        )
    for row in rows:
        row["quality_passed"] = row["adjacent_spacing_ratio_passed"]
        if not row["quality_passed"]:
            row["error_message"] = "adjacent spacing ratio exceeds 2"

    tolerance_cm = 1.0e-14
    radial = coordinates_by_direction["radial"]
    axial = coordinates_by_direction["axial"]
    interval_spacings = family.axial_interval_spacings_cm(mesh_level)
    for index in range(family.RADIAL_ANCHOR_COUNT):
        expected = (
            _number(geometry["r_core"])
            + index
            * (_number(geometry["r_mos2"]) - _number(geometry["r_core"]))
            / (family.RADIAL_ANCHOR_COUNT - 1)
        )
        actual = min(radial, key=lambda value: abs(value - expected)) if radial else math.nan
        error = actual - expected
        passed = math.isfinite(actual) and abs(error) <= tolerance_cm
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "mesh_level": mesh_level,
                "level_order": level_order,
                "row_type": "graded_actual_radial_anchor",
                "region": "MoS2",
                "contact": "",
                "node_index": -1,
                "anchor_index": index,
                "target_coordinate_cm": expected,
                "actual_coordinate_cm": actual,
                "coordinate_error_cm": error,
                "quality_passed": passed,
                "error_message": "" if passed else "graded radial anchor absent",
            }
        )
    for contact, endpoint, inward_sign in (
        ("source", _number(geometry["z_source"]), 1),
        ("drain", _number(geometry["z_drain"]), -1),
    ):
        for index, distance_nm in enumerate(family.AXIAL_DISTANCE_NM):
            expected = endpoint + inward_sign * distance_nm * family.NM_TO_CM
            actual = min(axial, key=lambda value: abs(value - expected)) if axial else math.nan
            coordinate_error = actual - expected
            anchor_passed = (
                math.isfinite(actual) and abs(coordinate_error) <= tolerance_cm
            )
            actual_index = (
                min(range(len(axial)), key=lambda item: abs(axial[item] - expected))
                if axial else -1
            )
            neighbour = actual_index + inward_sign
            actual_spacing = (
                abs(axial[neighbour] - axial[actual_index])
                if 0 <= actual_index < len(axial) and 0 <= neighbour < len(axial)
                else math.nan
            )
            requested = interval_spacings[index]
            spacing_passed = bool(
                math.isfinite(actual_spacing)
                and actual_spacing > COORDINATE_CLUSTER_TOLERANCE_CM
                and actual_spacing <= requested * (1.0 + 1.0e-6)
            )
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "mesh_level": mesh_level,
                    "level_order": level_order,
                    "row_type": "graded_actual_axial_anchor",
                    "region": "MoS2",
                    "contact": contact,
                    "node_index": -1,
                    "anchor_index": index,
                    "distance_from_contact_nm": distance_nm,
                    "target_coordinate_cm": expected,
                    "actual_coordinate_cm": actual,
                    "coordinate_error_cm": coordinate_error,
                    "actual_inward_spacing_cm": actual_spacing,
                    "requested_maximum_inward_spacing_cm": requested,
                    "coordinate_anchor_passed": anchor_passed,
                    "inward_spacing_bound_passed": spacing_passed,
                    "quality_passed": anchor_passed and spacing_passed,
                    "error_message": (
                        "" if anchor_passed and spacing_passed
                        else "graded axial anchor/spacing absent or invalid"
                    ),
                }
            )
    return rows


def run_probe_worker(mesh_level: str) -> dict[str, Any]:
    import contact_topology_qc
    import graded_local_mesh_family as family
    import run_memory_window as memory_window

    level_order = family.LEVELS.index(mesh_level)
    preview = base._preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        preview,
        mesh_level,
        line_log,
    ):
        geometry = memory_window.create_structure(
            tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM
        )
    validation = family.validate_local_mesh_line_log(
        line_log, geometry, mesh_level
    )
    if not bool(validation.get("passed")):
        raise RuntimeError(
            "graded mesh validation failed: "
            + ", ".join(map(str, validation.get("errors", ())))
        )
    rows = base._quality_rows_for_runtime(
        mesh_level, level_order, memory_window.device, geometry
    )
    rows.extend(
        _actual_spacing_ratio_rows(
            mesh_level, level_order, memory_window.device, geometry
        )
    )
    topology = contact_topology_qc.audit_runtime_contact_topology(
        geometry, device=memory_window.device
    )
    base._append_topology_rows(rows, topology, mesh_level, level_order)
    counts = base._runtime_counts(memory_window.device)
    elapsed = time.perf_counter() - started
    for row in rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                "global_coordinate_count": counts["global_coordinate_count"],
                "total_element_count": counts["total_element_count"],
                "runtime_seconds": elapsed,
                "worker_process_id": os.getpid(),
                "mesh_position_hash": validation.get("position_hash", ""),
                "mesh_line_validation_passed": True,
            }
        )
    return {
        "mode": "probe",
        "mesh_level": mesh_level,
        "level_order": level_order,
        "geometry": dict(geometry),
        "mesh_line_log": line_log,
        "mesh_line_validation": validation,
        "quality_rows": rows,
        "topology": topology,
        "runtime_invariant_hashes": base.runtime_invariant_hashes(geometry),
        "mesh_counts": counts,
        "runtime_seconds": elapsed,
        "worker_process_id": os.getpid(),
        "error_message": "",
    }


def run_point_worker(
    mesh_level: str, state_index: int, bias_name: str,
    run_role: str = "primary",
) -> dict[str, Any]:
    import devsim
    import graded_local_mesh_family as family
    import run_contact_topology_qc_candidate as contact_runner
    import run_memory_window as memory_window

    level_order = family.LEVELS.index(mesh_level)
    preview = base._preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        preview,
        mesh_level,
        line_log,
    ):
        with base._custom_legacy_validator(family, mesh_level, line_log):
            bundle = contact_runner.run_point_worker(
                mesh_level=mesh_level,
                mesh_scale=1.0,
                state_index=int(state_index),
                bias_name=str(bias_name),
            )
    validation = family.validate_local_mesh_line_log(
        line_log, bundle.get("geometry", preview), mesh_level
    )
    if not bool(validation.get("passed")):
        raise RuntimeError(
            "graded mesh validation failed: "
            + ", ".join(map(str, validation.get("errors", ())))
        )
    try:
        absolute_biases = contact_runner._read_biases(
            contact_runner.legacy_qc.DEVICE_NAME
        )
    except Exception:
        absolute_biases = {
            terminal: math.nan for terminal in base.TERMINALS
        }
    currents: dict[str, float] = {}
    for terminal in ("source", "drain"):
        try:
            currents[terminal] = float(
                devsim.get_contact_current(
                    device=memory_window.device,
                    contact=terminal,
                    equation="ElectronContinuityEquation",
                )
            )
        except Exception:
            currents[terminal] = math.nan
    counts = contact_runner._runtime_mesh_counts(
        memory_window.device, bundle.get("topology", {})
    )
    bundle.update(
        {
            "mode": "point",
            "mesh_level": mesh_level,
            "level_order": level_order,
            "state_index": int(state_index),
            "bias_name": str(bias_name),
            "run_role": str(run_role),
            "actual_biases_after_worker": absolute_biases,
            "contact_currents_A": currents,
            "diagnostic_mesh_counts": counts,
            "mesh_line_log": line_log,
            "mesh_line_validation": validation,
            "runtime_invariant_hashes": base.runtime_invariant_hashes(
                bundle.get("geometry", preview)
            ),
            "runtime_seconds": time.perf_counter() - started,
            "worker_process_id": os.getpid(),
        }
    )
    return bundle


MOS2_REFERENCE_MINIMUM_ANGLE_DEG = math.degrees(math.atan(1.0 / 8.0))
MOS2_REFERENCE_MAXIMUM_ASPECT_RATIO = 8.0 + 1.0 / 8.0
QUALITY_ABS_TOLERANCE = 1.0e-8


def evaluate_graded_quality_family(
    quality_rows: list[dict[str, Any]], levels: Sequence[str],
) -> dict[str, dict[str, Any]]:
    """Gate only the requested MoS2/contact quality and keep other regions diagnostic."""

    import graded_local_mesh_family as family

    summaries: dict[str, dict[str, Any]] = {}
    prior_angle = math.nan
    prior_aspect = math.nan
    prior_radial = math.nan
    prior_axial = math.nan
    base_angle = math.nan
    base_aspect = math.nan
    for level_order, level in enumerate(levels):
        relevant = [row for row in quality_rows if row.get("mesh_level") == level]
        mos2 = next(
            (
                row
                for row in relevant
                if row.get("row_type") == "region_summary"
                and row.get("region") == "MoS2"
            ),
            {},
        )
        radial = next(
            (
                row
                for row in relevant
                if row.get("row_type") == "adjacent_spacing_ratio"
                and row.get("spacing_direction") == "radial"
            ),
            {},
        )
        axial = next(
            (
                row
                for row in relevant
                if row.get("row_type") == "adjacent_spacing_ratio"
                and row.get("spacing_direction") == "axial"
            ),
            {},
        )
        topology = [
            row
            for row in relevant
            if row.get("row_type") == "topology"
            and row.get("contact") in {"source", "drain"}
        ]
        contact_summaries = [
            row
            for row in relevant
            if row.get("row_type") == "contact_summary"
            and row.get("contact") in {"source", "drain"}
        ]
        actual_anchors = [
            row
            for row in relevant
            if row.get("row_type") in {
                "graded_actual_radial_anchor", "graded_actual_axial_anchor"
            }
        ]
        angle = _number(mos2.get("minimum_angle_deg"))
        aspect = _number(mos2.get("maximum_aspect_ratio"))
        radial_min = _number(mos2.get("radial_spacing_min_cm"))
        radial_max = _number(mos2.get("radial_spacing_max_cm"))
        axial_min = _number(mos2.get("axial_spacing_min_cm"))
        axial_max = _number(mos2.get("axial_spacing_max_cm"))
        level_spec = family.get_level_spec(level)
        expected_radial = level_spec.radial_spacing_cm
        expected_axial_min = level_spec.axial_spacing_cm
        expected_axial_max = family.axial_interval_spacings_cm(level)[-1]
        angle_target_passed = (
            math.isfinite(angle)
            and angle
            >= MOS2_REFERENCE_MINIMUM_ANGLE_DEG - QUALITY_ABS_TOLERANCE
        )
        aspect_target_passed = (
            math.isfinite(aspect)
            and aspect
            <= MOS2_REFERENCE_MAXIMUM_ASPECT_RATIO + QUALITY_ABS_TOLERANCE
        )
        not_worse = (
            math.isfinite(angle)
            and math.isfinite(aspect)
            and (
                not math.isfinite(prior_angle)
                or angle >= prior_angle - QUALITY_ABS_TOLERANCE
            )
            and (
                not math.isfinite(prior_aspect)
                or aspect <= prior_aspect + QUALITY_ABS_TOLERANCE
            )
            and (
                not math.isfinite(base_angle)
                or angle >= base_angle - QUALITY_ABS_TOLERANCE
            )
            and (
                not math.isfinite(base_aspect)
                or aspect <= base_aspect + QUALITY_ABS_TOLERANCE
            )
        )
        absolute_spacing_passed = all(
            (
                math.isfinite(radial_min), math.isfinite(radial_max),
                math.isfinite(axial_min), math.isfinite(axial_max),
                math.isclose(
                    radial_min, expected_radial,
                    rel_tol=1.0e-6, abs_tol=1.0e-18,
                ),
                math.isclose(
                    radial_max, expected_radial,
                    rel_tol=1.0e-6, abs_tol=1.0e-18,
                ),
                math.isclose(
                    axial_min, expected_axial_min,
                    rel_tol=1.0e-6, abs_tol=1.0e-18,
                ),
                math.isclose(
                    axial_max, expected_axial_max,
                    rel_tol=1.0e-6, abs_tol=1.0e-18,
                ),
            )
        )
        spacing_scaled = absolute_spacing_passed
        if level_order > 0:
            spacing_scaled = spacing_scaled and all(
                math.isfinite(value)
                for value in (prior_radial, prior_axial)
            )
            spacing_scaled = spacing_scaled and math.isclose(
                radial_min,
                0.5 * prior_radial,
                rel_tol=1.0e-6,
                abs_tol=1.0e-18,
            )
            spacing_scaled = spacing_scaled and math.isclose(
                axial_max,
                0.5 * prior_axial,
                rel_tol=1.0e-6,
                abs_tol=1.0e-18,
            )
        adjacent_passed = all(
            bool(row.get("adjacent_spacing_ratio_passed"))
            for row in (radial, axial)
        ) and bool(radial) and bool(axial)
        topology_by_contact = {
            str(row.get("contact")): row for row in topology
        }
        contact_summary_by_contact = {
            str(row.get("contact")): row for row in contact_summaries
        }
        unique_contacts = (
            len(topology) == 2
            and set(topology_by_contact) == {"source", "drain"}
            and len(contact_summaries) == 2
            and set(contact_summary_by_contact) == {"source", "drain"}
        )
        expected_edge_count = int(round(
            2.0e-7 / level_spec.radial_spacing_cm
        ))
        expected_node_count = expected_edge_count + 1
        contact_passed = (
            unique_contacts
            and all(bool(row.get("quality_passed")) for row in topology)
            and all(bool(row.get("contact_area_passed")) for row in topology)
            and all(
                int(_number(row.get("edge_count"))) == expected_edge_count
                for row in topology
            )
            and all(
                int(_number(row.get("contact_node_count")))
                == expected_node_count
                and int(_number(row.get("active_incident_edge_count"))) > 0
                and _number(row.get("valence_minimum")) > 0.0
                and math.isclose(
                    _number(row.get("incident_couple_sum_cm2")),
                    _number(topology_by_contact[str(row.get("contact"))].get(
                        "expected_area_cm2"
                    )),
                    rel_tol=1.0e-6,
                    abs_tol=1.0e-24,
                )
                for row in contact_summaries
            )
        )
        anchor_identities = {
            (
                str(row.get("row_type")), str(row.get("contact", "")),
                int(_number(row.get("anchor_index"))),
            )
            for row in actual_anchors
            if math.isfinite(_number(row.get("anchor_index")))
        }
        expected_anchor_identities = {
            ("graded_actual_radial_anchor", "", index)
            for index in range(family.RADIAL_ANCHOR_COUNT)
        } | {
            ("graded_actual_axial_anchor", contact, index)
            for contact in ("source", "drain")
            for index in range(len(family.AXIAL_DISTANCE_NM))
        }
        actual_anchor_passed = (
            len(actual_anchors) == len(expected_anchor_identities)
            and anchor_identities == expected_anchor_identities
            and all(bool(row.get("quality_passed")) for row in actual_anchors)
        )
        base_mesh_valid = bool(mos2) and bool(mos2.get("quality_passed")) and (
            int(float(mos2.get("degenerate_triangle_count", 1))) == 0
        )
        passed = all(
            (
                base_mesh_valid,
                angle_target_passed,
                aspect_target_passed,
                not_worse,
                absolute_spacing_passed,
                spacing_scaled,
                adjacent_passed,
                actual_anchor_passed,
                contact_passed,
            )
        )
        summary = {
            "schema_version": SCHEMA_VERSION,
            "mesh_level": level,
            "level_order": level_order,
            "row_type": "graded_quality_summary",
            "region": "MoS2/contact",
            "contact": "",
            "node_index": -1,
            "minimum_angle_deg": angle,
            "minimum_angle_target_deg": MOS2_REFERENCE_MINIMUM_ANGLE_DEG,
            "minimum_angle_target_passed": angle_target_passed,
            "maximum_aspect_ratio": aspect,
            "maximum_aspect_target": MOS2_REFERENCE_MAXIMUM_ASPECT_RATIO,
            "maximum_aspect_target_passed": aspect_target_passed,
            "minimum_angle_and_aspect_not_worse": not_worse,
            "radial_spacing_min_cm": radial_min,
            "radial_spacing_max_cm": radial_max,
            "expected_radial_spacing_cm": expected_radial,
            "axial_spacing_min_cm": axial_min,
            "axial_spacing_max_cm": axial_max,
            "expected_axial_spacing_min_cm": expected_axial_min,
            "expected_axial_spacing_max_cm": expected_axial_max,
            "absolute_spacing_target_passed": absolute_spacing_passed,
            "spacing_scaled_by_half": spacing_scaled,
            "radial_adjacent_spacing_ratio_max": radial.get(
                "adjacent_spacing_ratio_max", math.nan
            ),
            "axial_adjacent_spacing_ratio_max": axial.get(
                "adjacent_spacing_ratio_max", math.nan
            ),
            "adjacent_spacing_ratio_passed": adjacent_passed,
            "actual_anchor_count": len(actual_anchors),
            "actual_anchor_coverage_passed": actual_anchor_passed,
            "expected_contact_edge_count": expected_edge_count,
            "expected_contact_node_count": expected_node_count,
            "contact_topology_quality_passed": contact_passed,
            "quality_passed": passed,
            "error_message": "" if passed else "graded MoS2/contact quality criterion failed",
        }
        quality_rows.append(summary)
        summaries[level] = {"passed": passed, **summary}
        if math.isfinite(angle) and math.isfinite(aspect):
            if not math.isfinite(base_angle):
                base_angle, base_aspect = angle, aspect
            prior_angle, prior_aspect = angle, aspect
        if math.isfinite(radial_min) and math.isfinite(axial_max):
            prior_radial, prior_axial = radial_min, axial_max
    return summaries


def _annotated_point_row(bundle: Mapping[str, Any]) -> dict[str, Any]:
    import graded_mesh_acceptance as acceptance

    row = base._point_row(bundle)
    row["schema_version"] = SCHEMA_VERSION
    annotated = acceptance.annotate_reference_point(row)
    annotated["ioff_status"] = (
        annotated["dc_current_status"]
        if int(annotated.get("state_index", -1)) == 4
        and str(annotated.get("bias_name", "")) == "off"
        else "not_ioff_reference"
    )
    if annotated["ioff_status"] == "not_ioff_reference":
        annotated["reported_Ioff_A"] = math.nan
    annotated["reported_on_off_ratio"] = math.nan
    annotated["on_off_status"] = (
        "not_available_below_numerical_floor"
        if annotated["ioff_status"] == acceptance.CURRENT_STATUS_BELOW_FLOOR
        else "not_computed_at_reference_point"
    )
    annotated["dc_transport_point_status"] = (
        "passed" if annotated["dc_transport_point_passed"] else "failed"
    )
    annotated["electrostatic_qc_point_status"] = (
        "passed" if annotated["electrostatic_qc_point_passed"] else "failed"
    )
    row.update(annotated)
    return row


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
    """Run only gate-authorized graded extra/ultra full sweeps."""

    import graded_local_mesh_full_dc as full_dc

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
                    command,
                    cwd=REPOSITORY_ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
        else:
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        code = int(getattr(completed, "returncode", 0))
        if code or not output.is_file():
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            bundles.append(
                {
                    **dict(task),
                    "mode": "full_dc",
                    "passed": False,
                    "raw_rows": [],
                    "metrics_rows": [],
                    "error_message": f"worker return code {code}: {tail}".strip(),
                }
            )
        else:
            with output.open("r", encoding="utf-8") as stream:
                bundles.append(json.load(stream))
        print(
            f"graded-mesh full-DC worker {index}/{len(tasks)} {stem}",
            flush=True,
        )
    return bundles


def generate_full_dc_outputs(
    staging: Path, gate: Mapping[str, Any], *,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Generate candidate-only DC rows after the combined reference gate."""

    import graded_local_mesh_full_dc as full_dc

    bundles = run_fresh_full_dc_workers(
        staging / "full_dc_workers",
        gate,
        python_executable=python_executable,
    )
    assembled = full_dc.assemble_full_dc_outputs(bundles)
    metric_output_rows: list[dict[str, Any]] = []
    for source in assembled.get("raw_rows", ()):
        metric_output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": f"{source.get('sweep_kind', '')}_point",
                **dict(source),
            }
        )
    for source in assembled.get("metrics_rows", ()):
        metric_output_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_type": "extracted_metric",
                **dict(source),
            }
        )

    convergence_output_rows: list[dict[str, Any]] = [
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "dc_convergence",
            **dict(source),
        }
        for source in assembled.get("convergence_rows", ())
    ]
    convergence_output_rows.extend(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "worker_validation",
            **dict(source),
        }
        for source in assembled.get("validation_rows", ())
    )
    convergence_output_rows.append(
        {
            "schema_version": SCHEMA_VERSION,
            "record_type": "full_dc_summary",
            "reference_mesh": "graded_extra_fine",
            "candidate_mesh": "graded_ultra_fine",
            "passed": bool(assembled.get("passed")),
            "status": assembled.get("status", ""),
            "raw_row_count": len(assembled.get("raw_rows", ())),
            "metric_row_count": len(assembled.get("metrics_rows", ())),
            "error_message": "; ".join(map(str, assembled.get("errors", ()))),
        }
    )
    write_csv(
        staging / FULL_DC_OUTPUT_FILENAMES[0],
        _dynamic_fields(
            metric_output_rows,
            (
                "schema_version", "record_type", "mesh_level", "sweep_kind",
                "state_index", "state", "VGS_V", "VDS_V", "ID_A",
            ),
        ),
        metric_output_rows,
    )
    write_csv(
        staging / FULL_DC_OUTPUT_FILENAMES[1],
        _dynamic_fields(
            convergence_output_rows,
            (
                "schema_version", "record_type", "reference_mesh",
                "candidate_mesh", "state_index", "state", "VGS_V",
                "VDS_V", "quantity", "acceptance_applies", "passed",
                "status", "error_message",
            ),
        ),
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


def _validate_runtime_invariants(
    bundles: Sequence[dict[str, Any]],
    probes: Sequence[Mapping[str, Any]],
    invariant_hashes: Mapping[str, str],
    family_position_hash: str,
) -> None:
    probe_by_level = {
        str(bundle.get("mesh_level")): bundle for bundle in probes
    }
    for bundle in bundles:
        level = str(bundle.get("mesh_level", ""))
        errors: list[str] = []
        geometry = bundle.get("geometry")
        if not isinstance(geometry, Mapping):
            errors.append("runtime geometry unavailable")
        elif base.runtime_invariant_hashes(geometry) != dict(invariant_hashes):
            errors.append("runtime geometry/material/state/solver hashes differ")
        if dict(bundle.get("runtime_invariant_hashes", {})) != dict(
            invariant_hashes
        ):
            errors.append("worker-recorded runtime invariant hashes differ")
        validation = bundle.get("mesh_line_validation", {})
        if (
            not bool(validation.get("passed"))
            or str(validation.get("position_hash", ""))
            != family_position_hash
        ):
            errors.append("mesh-line validation/position hash differs")
        expected_topology = probe_by_level.get(level, {}).get("topology", {})
        actual_topology = bundle.get("topology", {})
        if (
            not expected_topology
            or not actual_topology
            or base._contact_topology_signature(actual_topology)
            != base._contact_topology_signature(expected_topology)
        ):
            errors.append("contact topology differs from level probe")
        bundle["runtime_invariants_passed"] = not errors
        bundle["family_position_hash_sha256"] = family_position_hash
        if errors:
            previous = str(bundle.get("error_message", "")).strip()
            bundle["error_message"] = "; ".join(
                item for item in (previous, "; ".join(errors)) if item
            )


def generate_candidate(
    staging: Path, *, jobs: int = 1,
    python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Run reference probes/points, gate, and conditionally full DC."""

    import graded_local_mesh_family as family
    import graded_mesh_acceptance as acceptance

    staging.mkdir(parents=True, exist_ok=False)
    bundles = run_fresh_workers(
        staging / "workers", jobs=jobs, python_executable=python_executable
    )
    probes = [bundle for bundle in bundles if bundle.get("mode") == "probe"]
    points = [bundle for bundle in bundles if bundle.get("mode") == "point"]
    if len(probes) != len(family.LEVELS):
        raise RuntimeError(
            f"expected {len(family.LEVELS)} probes, received {len(probes)}"
        )
    geometry = next(
        (
            bundle.get("geometry")
            for bundle in probes
            if isinstance(bundle.get("geometry"), Mapping)
        ),
        None,
    )
    if geometry is None:
        raise RuntimeError("all graded mesh probes failed")
    invariant_hashes = base.runtime_invariant_hashes(geometry)
    position_hashes: set[str] = set()
    for bundle in probes:
        level = str(bundle.get("mesh_level", ""))
        probe_geometry = bundle.get("geometry")
        if not isinstance(probe_geometry, Mapping):
            raise RuntimeError(f"{level}: mesh probe has no runtime geometry")
        if base.runtime_invariant_hashes(probe_geometry) != invariant_hashes:
            raise RuntimeError(f"{level}: runtime invariant mismatch")
        validation = bundle.get("mesh_line_validation", {})
        if not bool(validation.get("passed")):
            raise RuntimeError(f"{level}: mesh-line validation failed")
        position_hashes.add(str(validation.get("position_hash", "")))
    if "" in position_hashes or len(position_hashes) != 1:
        raise RuntimeError("graded family mesh positions differ")
    family_position_hash = next(iter(position_hashes))
    _validate_runtime_invariants(
        bundles, probes, invariant_hashes, family_position_hash
    )

    family_rows = family.local_mesh_family_definition_rows(geometry)
    for row in family_rows:
        row.update(
            {
                "schema_version": SCHEMA_VERSION,
                **invariant_hashes,
                "family_position_hash_sha256": family_position_hash,
                "same_refinement_pattern_passed": True,
                "geometry_material_contact_solver_fixed": True,
            }
        )
    quality_rows = [
        dict(row)
        for bundle in probes
        for row in bundle.get("quality_rows", ())
    ]
    quality_by_level = evaluate_graded_quality_family(
        quality_rows, family.LEVELS
    )
    point_rows = [_annotated_point_row(bundle) for bundle in points]
    convergence_rows = acceptance.build_graded_convergence_rows(point_rows)
    gate_report = acceptance.evaluate_reference_gates(
        point_rows,
        convergence_rows,
        quality_by_level,
        profile_passed=True,
    )
    for row in convergence_rows:
        row["schema_version"] = SCHEMA_VERSION
    criterion_rows = acceptance.criterion_rows(gate_report)
    for row in criterion_rows:
        row["schema_version"] = SCHEMA_VERSION
    combined_gate = dict(gate_report["combined_reference"])
    full_dc = dispatch_full_dc_if_eligible(
        combined_gate,
        dispatcher=lambda: generate_full_dc_outputs(
            staging, combined_gate, python_executable=python_executable
        ),
    )
    summary_row = {
        "schema_version": SCHEMA_VERSION,
        "row_type": "final_gate",
        "acceptance_domain": "combined_reference",
        "reference_mesh": "graded_extra_fine",
        "candidate_mesh": "graded_ultra_fine",
        "quantity": "all_required_reference_criteria",
        "passed": bool(combined_gate.get("passed")),
        "status": combined_gate.get("physical_contact_flux_status", ""),
        "full_sweep_executed": bool(full_dc.get("executed")),
        "full_sweep_passed": (
            bool(full_dc.get("result", {}).get("passed"))
            if full_dc.get("executed")
            else False
        ),
        "error_message": "",
    }
    all_convergence_rows = [
        *convergence_rows, *criterion_rows, summary_row
    ]
    write_csv(
        staging / OUTPUT_FILENAMES[0],
        _dynamic_fields(
            family_rows, ("schema_version", "mesh_level", "direction", "target")
        ),
        family_rows,
    )
    write_csv(
        staging / OUTPUT_FILENAMES[1],
        _dynamic_fields(
            quality_rows,
            (
                "schema_version", "mesh_level", "level_order", "row_type",
                "region", "contact", "minimum_angle_deg",
                "maximum_aspect_ratio", "quality_passed", "error_message",
            ),
        ),
        quality_rows,
    )
    write_csv(
        staging / OUTPUT_FILENAMES[2],
        _dynamic_fields(point_rows, POINT_FIELDS),
        point_rows,
    )
    write_csv(
        staging / OUTPUT_FILENAMES[3],
        _dynamic_fields(
            all_convergence_rows,
            (
                "schema_version", "row_type", "acceptance_domain",
                "reference_mesh", "candidate_mesh", "state_index",
                "bias_name", "quantity", "criterion", "gate_applicable",
                "passed", "status", "error_message",
            ),
        ),
        all_convergence_rows,
    )
    return {
        "gate": gate_report,
        "full_dc": full_dc,
        "point_count": len(point_rows),
        "quality_row_count": len(quality_rows),
    }


def run_candidate(
    *, jobs: int = 1, python_executable: str = sys.executable,
) -> dict[str, Any]:
    """Publish a new candidate by exclusive rename after hash verification."""

    if CANDIDATE_DIRECTORY.exists():
        raise FileExistsError(
            f"candidate directory already exists: {CANDIDATE_DIRECTORY}"
        )
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    protected = capture_protected_hashes()
    with tempfile.TemporaryDirectory(
        prefix=".graded_local_mesh_", dir=RESULTS_DIRECTORY
    ) as temporary:
        staging = Path(temporary) / "candidate"
        result = generate_candidate(
            staging, jobs=jobs, python_executable=python_executable
        )
        assert_protected_hashes(protected)
        if CANDIDATE_DIRECTORY.exists():
            raise FileExistsError(
                f"candidate directory appeared during run: {CANDIDATE_DIRECTORY}"
            )
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
            raise SystemExit(
                "worker mode requires --mesh-level and --worker-output"
            )
        try:
            if arguments.worker_mode == "probe":
                bundle = run_probe_worker(arguments.mesh_level)
            elif arguments.worker_mode == "point":
                if arguments.state_index is None or arguments.bias_name is None:
                    raise ValueError("point worker requires state index and bias name")
                bundle = run_point_worker(
                    arguments.mesh_level,
                    arguments.state_index,
                    arguments.bias_name,
                    arguments.run_role,
                )
            else:
                if arguments.sweep_kind is None:
                    raise ValueError("full-DC worker requires sweep kind")
                import graded_local_mesh_full_dc as full_dc

                bundle = full_dc.run_full_dc_worker(
                    arguments.mesh_level, arguments.sweep_kind
                )
                bundle["mode"] = "full_dc"
        except Exception as error:
            traceback.print_exc()
            bundle = _failed_bundle(
                {
                    "mode": arguments.worker_mode,
                    "mesh_level": arguments.mesh_level,
                    "state_index": arguments.state_index,
                    "bias_name": arguments.bias_name,
                    "sweep_kind": arguments.sweep_kind,
                    "run_role": arguments.run_role,
                },
                _error(error),
            )
            base._write_json(arguments.worker_output, bundle)
            return 1
        base._write_json(arguments.worker_output, bundle)
        return 0
    result = run_candidate(jobs=arguments.jobs)
    print(
        json.dumps(base._json_safe(result), ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CANDIDATE_DIRECTORY", "OUTPUT_FILENAMES", "FULL_DC_OUTPUT_FILENAMES",
    "POINT_FIELDS", "build_tasks", "build_worker_command",
    "build_full_dc_worker_command", "capture_protected_hashes",
    "assert_protected_hashes", "write_csv", "run_probe_worker",
    "run_point_worker", "run_fresh_workers", "run_fresh_full_dc_workers",
    "dispatch_full_dc_if_eligible", "evaluate_graded_quality_family",
    "generate_candidate", "run_candidate", "main",
)
