"""Candidate-only full DC helpers for a passed local-mesh reference gate.

The public characterization command and every public CSV writer are
intentionally absent.  A caller must first obtain four task specifications
from :func:`build_full_dc_tasks`; that function returns no work unless both
``passed`` and ``full_sweep_allowed`` are explicitly true in the reference
gate.  Each task is intended for a fresh process.

``run_full_dc_worker(mesh_level, sweep_kind)`` builds one local mesh and runs
exactly one canonical five-state ID-VG or ID-VD sweep.  ID-VG workers also
reuse :func:`run_state_characterization.build_metrics_rows` to extract Vth,
SS, Ion, Ioff, ON/OFF and gm.  ``assemble_full_dc_outputs`` combines the four
expected bundles and returns rows only; it never writes a file.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any


MESH_LEVELS = ("local_extra_fine", "local_ultra_fine")
SWEEP_KINDS = ("idvg", "idvd")
FULL_DC_STATUS_COMPLETE = "candidate_full_dc_complete"
FULL_DC_STATUS_FAILED = "candidate_full_dc_failed_closed"
BIAS_ATOL_V = 1.0e-12

RAW_NUMERIC_FIELDS = (
    "ntrap_cm3", "nsheet_cm2", "trap_charge_density_C_cm3",
    "VGS_V", "VDS_V", "ID_A", "abs_ID_A",
)
METRIC_FINITE_FIELDS = (
    "ntrap_cm3", "nsheet_cm2", "VDS_V", "Vth_V",
    "threshold_current_A", "SS_mV_dec", "ss_r_squared", "Ion_A",
    "ion_vgs_V", "ion_vds_V", "Ioff_A", "ioff_vgs_V", "ioff_vds_V",
    "on_off_ratio", "gm_max_S", "VGS_at_gm_max_V",
    "ss_current_min_A", "ss_current_max_A", "current_floor_A",
)


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", "", "none", "nan"}:
        return False
    raise ValueError(f"invalid Boolean value {value!r}")


def _error(error: object | None) -> str:
    if error is None:
        return ""
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message}" if isinstance(error, BaseException) else message


def reference_gate_allows_full_dc(reference_gate: Mapping[str, Any]) -> bool:
    """Require two explicit independent authorization bits from the gate."""

    return bool(reference_gate.get("passed")) and bool(
        reference_gate.get("full_sweep_allowed")
    )


def build_full_dc_tasks(reference_gate: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Return the four fresh-worker tasks, or no tasks for a failed gate."""

    if not reference_gate_allows_full_dc(reference_gate):
        return ()
    return tuple(
        {"mesh_level": mesh_level, "sweep_kind": sweep_kind}
        for mesh_level in MESH_LEVELS
        for sweep_kind in SWEEP_KINDS
    )


def _preview_geometry(memory_window: Any) -> dict[str, Any]:
    geometry = dict(
        memory_window.parameterized_device_structure.calculate_geometry(
            tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM
        )
    )
    from state_sweep_helpers import validate_canonical_handoff_geometry

    validate_canonical_handoff_geometry(geometry)
    return geometry


def _runtime_mesh_counts(device: str) -> dict[str, int]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    coordinates: set[int] = set()
    region_nodes = 0
    elements = 0
    for region in get_region_list(device=device):
        coordinate_indices = tuple(
            int(value)
            for value in get_node_model_values(
                device=device, region=region, name="coordinate_index"
            )
        )
        coordinates.update(coordinate_indices)
        region_nodes += len(coordinate_indices)
        elements += len(get_element_node_list(device=device, region=region))
    return {
        "global_coordinate_count": len(coordinates),
        "total_region_node_count": region_nodes,
        "total_element_count": elements,
    }


def _expected_coordinates(
    sweep_kind: str,
    *,
    states: Sequence[Mapping[str, Any]],
    config: Any,
    characterization: Any,
) -> set[tuple[int, float, float]]:
    if sweep_kind == "idvg":
        return characterization.expected_idvg_coordinates(
            states, config.IDVG_VDS_VALUES_V, config.IDVG_VGS_STEP_V
        )
    if sweep_kind == "idvd":
        return characterization.expected_idvd_coordinates(
            states, config.IDVD_VGS_VALUES_V, config.IDVD_VDS_STEP_V
        )
    raise ValueError(f"unsupported sweep_kind {sweep_kind!r}")


def _canonical_expected_coordinates(
    sweep_kind: str, *, states: Sequence[Mapping[str, Any]], config: Any
) -> set[tuple[int, float, float]]:
    """Rebuild the exact configured grid without importing a runtime runner."""

    if sweep_kind == "idvg":
        count = int(round(
            (float(config.IDVG_VGS_STOP_V) - float(config.IDVG_VGS_START_V))
            / float(config.IDVG_VGS_STEP_V)
        ))
        vgs_values = tuple(
            round(float(config.IDVG_VGS_START_V) + index * float(config.IDVG_VGS_STEP_V), 12)
            for index in range(count + 1)
        )
        return {
            (int(state["state_index"]), vgs, round(float(vds), 12))
            for state in states for vds in config.IDVG_VDS_VALUES_V for vgs in vgs_values
        }
    if sweep_kind == "idvd":
        count = int(round(
            (float(config.IDVD_VDS_STOP_V) - float(config.IDVD_VDS_START_V))
            / float(config.IDVD_VDS_STEP_V)
        ))
        vds_values = tuple(
            round(float(config.IDVD_VDS_START_V) + index * float(config.IDVD_VDS_STEP_V), 12)
            for index in range(count + 1)
        )
        return {
            (int(state["state_index"]), round(float(vgs), 12), vds)
            for state in states for vgs in config.IDVD_VGS_VALUES_V for vds in vds_values
        }
    raise ValueError(f"unsupported sweep_kind {sweep_kind!r}")


def validate_full_sweep_rows(
    sweep_kind: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    states: Sequence[Mapping[str, Any]] | None = None,
    expected_coordinates: set[tuple[int, float, float]] | None = None,
    orchestration_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Fail closed on grid, metadata, convergence, errors and finite currents.

    ``states`` and ``expected_coordinates`` are injectable so this validator
    remains unit-testable without DEVSIM.  Production calls always provide the
    canonical five-state configuration and exact characterization grid.
    """

    if sweep_kind not in SWEEP_KINDS:
        raise ValueError(f"unsupported sweep_kind {sweep_kind!r}")
    if states is None or expected_coordinates is None:
        import state_characterization_config as config
        import run_state_characterization as characterization

        states = config.MEMORY_STATES if states is None else states
        expected_coordinates = (
            _expected_coordinates(
                sweep_kind,
                states=states,
                config=config,
                characterization=characterization,
            )
            if expected_coordinates is None
            else expected_coordinates
        )
    state_by_index = {int(state["state_index"]): state for state in states}
    errors: list[str] = []
    if len(state_by_index) != 5 or set(state_by_index) != {0, 1, 2, 3, 4}:
        errors.append("canonical five-state set is incomplete")

    coordinates: list[tuple[int, float, float]] = []
    failed_count = 0
    nonfinite_count = 0
    metadata_error_count = 0
    nonempty_error_count = 0
    for row_index, row in enumerate(rows):
        try:
            state_index = int(row["state_index"])
            coordinate = (
                state_index,
                round(float(row["VGS_V"]), 12),
                round(float(row["VDS_V"]), 12),
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"row {row_index} has an invalid coordinate")
            continue
        coordinates.append(coordinate)
        expected_state = state_by_index.get(state_index)
        if expected_state is None or str(row.get("state", "")) != str(expected_state["state"]):
            metadata_error_count += 1
        elif not math.isclose(
            _number(row.get("ntrap_cm3")), float(expected_state["ntrap_cm3"]),
            rel_tol=1.0e-12, abs_tol=0.0,
        ):
            metadata_error_count += 1
        try:
            converged = _strict_bool(row.get("converged"))
        except ValueError:
            converged = False
        if not converged:
            failed_count += 1
        if str(row.get("error_message", "")).strip():
            nonempty_error_count += 1
        numeric = [_number(row.get(field)) for field in RAW_NUMERIC_FIELDS]
        if not all(math.isfinite(value) for value in numeric):
            nonfinite_count += 1
        else:
            signed = _number(row.get("ID_A"))
            magnitude = _number(row.get("abs_ID_A"))
            if magnitude < 0.0 or not math.isclose(
                magnitude, abs(signed), rel_tol=1.0e-12, abs_tol=1.0e-30
            ):
                nonfinite_count += 1

    coordinate_set = set(coordinates)
    duplicates = len(coordinates) - len(coordinate_set)
    missing = set(expected_coordinates) - coordinate_set
    unexpected = coordinate_set - set(expected_coordinates)
    if orchestration_errors:
        errors.append("orchestration errors: " + "; ".join(map(str, orchestration_errors)))
    if duplicates:
        errors.append(f"duplicate coordinates={duplicates}")
    if missing:
        errors.append(f"missing coordinates={len(missing)}")
    if unexpected:
        errors.append(f"unexpected coordinates={len(unexpected)}")
    if failed_count:
        errors.append(f"unconverged rows={failed_count}")
    if nonfinite_count:
        errors.append(f"nonfinite/invalid-current rows={nonfinite_count}")
    if metadata_error_count:
        errors.append(f"state metadata mismatches={metadata_error_count}")
    if nonempty_error_count:
        errors.append(f"rows with error messages={nonempty_error_count}")
    return {
        "sweep_kind": sweep_kind,
        "expected_row_count": len(expected_coordinates),
        "actual_row_count": len(rows),
        "duplicate_count": duplicates,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "failed_count": failed_count,
        "nonfinite_count": nonfinite_count,
        "metadata_error_count": metadata_error_count,
        "nonempty_error_count": nonempty_error_count,
        "orchestration_error_count": len(orchestration_errors),
        "passed": not errors,
        "errors": errors,
    }


def validate_metrics_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    states: Sequence[Mapping[str, Any]] | None = None,
    vds_values_V: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Require one finite, successful metric row per state and ID-VG VDS."""

    if states is None or vds_values_V is None:
        import state_characterization_config as config

        states = config.MEMORY_STATES if states is None else states
        vds_values_V = config.IDVG_VDS_VALUES_V if vds_values_V is None else vds_values_V
    state_by_index = {int(state["state_index"]): state for state in states}
    expected = {
        (state_index, round(float(vds), 12))
        for state_index in state_by_index
        for vds in vds_values_V
    }
    actual: list[tuple[int, float]] = []
    errors: list[str] = []
    failed_count = 0
    nonfinite_count = 0
    metadata_error_count = 0
    for row_index, row in enumerate(rows):
        try:
            state_index = int(row["state_index"])
            coordinate = (state_index, round(float(row["VDS_V"]), 12))
        except (KeyError, TypeError, ValueError, OverflowError):
            errors.append(f"metric row {row_index} has an invalid coordinate")
            continue
        actual.append(coordinate)
        expected_state = state_by_index.get(state_index)
        if expected_state is None or str(row.get("state", "")) != str(expected_state["state"]):
            metadata_error_count += 1
        elif not math.isclose(
            _number(row.get("ntrap_cm3")), float(expected_state["ntrap_cm3"]),
            rel_tol=1.0e-12, abs_tol=0.0,
        ):
            metadata_error_count += 1
        try:
            successful = _strict_bool(row.get("vth_success")) and _strict_bool(
                row.get("ss_success")
            )
        except ValueError:
            successful = False
        if not successful or str(row.get("vth_error", "")).strip() or str(
            row.get("ss_error", "")
        ).strip():
            failed_count += 1
        values = [_number(row.get(field)) for field in METRIC_FINITE_FIELDS]
        try:
            ss_points = int(row.get("ss_point_count"))
            ss_minimum = int(row.get("ss_minimum_point_count"))
        except (TypeError, ValueError, OverflowError):
            ss_points, ss_minimum = -1, 0
        if (
            not all(math.isfinite(value) for value in values)
            or _number(row.get("Ion_A")) < 0.0
            or _number(row.get("Ioff_A")) < 0.0
            or _number(row.get("on_off_ratio")) < 0.0
            or _number(row.get("gm_max_S")) < 0.0
            or ss_points < ss_minimum
        ):
            nonfinite_count += 1
    actual_set = set(actual)
    duplicates = len(actual) - len(actual_set)
    missing = expected - actual_set
    unexpected = actual_set - expected
    if len(state_by_index) != 5 or set(state_by_index) != {0, 1, 2, 3, 4}:
        errors.append("canonical five-state set is incomplete")
    if duplicates:
        errors.append(f"duplicate metric coordinates={duplicates}")
    if missing:
        errors.append(f"missing metric coordinates={len(missing)}")
    if unexpected:
        errors.append(f"unexpected metric coordinates={len(unexpected)}")
    if failed_count:
        errors.append(f"failed metric rows={failed_count}")
    if nonfinite_count:
        errors.append(f"nonfinite/invalid metric rows={nonfinite_count}")
    if metadata_error_count:
        errors.append(f"metric metadata mismatches={metadata_error_count}")
    return {
        "expected_row_count": len(expected), "actual_row_count": len(rows),
        "duplicate_count": duplicates, "missing_count": len(missing),
        "unexpected_count": len(unexpected), "failed_count": failed_count,
        "nonfinite_count": nonfinite_count,
        "metadata_error_count": metadata_error_count,
        "passed": not errors, "errors": errors,
    }


def run_full_dc_worker(mesh_level: str, sweep_kind: str) -> dict[str, Any]:
    """Run one canonical sweep on one local mesh without writing any file."""

    if mesh_level not in MESH_LEVELS:
        raise ValueError(f"mesh_level must be one of {MESH_LEVELS}")
    if sweep_kind not in SWEEP_KINDS:
        raise ValueError(f"sweep_kind must be one of {SWEEP_KINDS}")

    import local_mesh_family as family
    import run_memory_window as memory_window
    import run_state_characterization as characterization
    import state_characterization_config as config
    from run_idvd_by_state import run_idvd_by_state
    from run_idvg_by_state import run_idvg_by_state
    from state_sweep_helpers import initialize_characterization_device

    config.validate_state_characterization_config()
    preview = _preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        preview,
        mesh_level,
        line_log,
    ):
        geometry, operating_point = initialize_characterization_device()
    mesh_validation = family.validate_local_mesh_line_log(
        line_log, geometry, mesh_level
    )
    if not bool(mesh_validation.get("passed")):
        raise RuntimeError(
            "local mesh-line validation failed: "
            + "; ".join(map(str, mesh_validation.get("errors", ())))
        )

    metrics_rows: list[dict[str, Any]] = []
    metric_validation: dict[str, Any] = {
        "passed": sweep_kind != "idvg", "errors": []
    }
    if sweep_kind == "idvg":
        raw_rows, operating_point = run_idvg_by_state(
            states=config.MEMORY_STATES,
            vds_values_V=config.IDVG_VDS_VALUES_V,
            vgs_start_V=config.IDVG_VGS_START_V,
            vgs_stop_V=config.IDVG_VGS_STOP_V,
            vgs_step_V=config.IDVG_VGS_STEP_V,
            operating_point=operating_point,
        )
        try:
            metrics_rows = characterization.build_metrics_rows(
                raw_rows,
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
        except Exception as error:
            metric_validation = {"passed": False, "errors": [_error(error)]}
        else:
            metric_validation = validate_metrics_rows(
                metrics_rows,
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
    else:
        raw_rows, operating_point = run_idvd_by_state(
            states=config.MEMORY_STATES,
            vgs_values_V=config.IDVD_VGS_VALUES_V,
            vds_start_V=config.IDVD_VDS_START_V,
            vds_stop_V=config.IDVD_VDS_STOP_V,
            vds_step_V=config.IDVD_VDS_STEP_V,
            operating_point=operating_point,
        )

    grid_validation = validate_full_sweep_rows(
        sweep_kind,
        raw_rows,
        states=config.MEMORY_STATES,
        expected_coordinates=_expected_coordinates(
            sweep_kind,
            states=config.MEMORY_STATES,
            config=config,
            characterization=characterization,
        ),
        orchestration_errors=operating_point.orchestration_errors,
    )
    passed = all(
        (
            bool(mesh_validation.get("passed")),
            bool(grid_validation.get("passed")),
            bool(metric_validation.get("passed")),
        )
    )
    errors = [
        *map(str, mesh_validation.get("errors", ())),
        *map(str, grid_validation.get("errors", ())),
        *map(str, metric_validation.get("errors", ())),
    ]
    return {
        "mesh_level": mesh_level,
        "sweep_kind": sweep_kind,
        "geometry": dict(geometry),
        "mesh_line_log": line_log,
        "mesh_line_validation": mesh_validation,
        "mesh_counts": _runtime_mesh_counts(memory_window.device),
        "raw_rows": list(raw_rows),
        "metrics_rows": metrics_rows,
        "grid_validation": grid_validation,
        "metric_validation": metric_validation,
        "orchestration_errors": list(operating_point.orchestration_errors),
        "runtime_seconds": time.perf_counter() - started,
        "worker_process_id": os.getpid(),
        "passed": passed,
        "error_message": "; ".join(dict.fromkeys(item for item in errors if item)),
    }


def _bundle_key(bundle: Mapping[str, Any]) -> tuple[str, str]:
    return str(bundle.get("mesh_level", "")), str(bundle.get("sweep_kind", ""))


def _geometry_signature(geometry: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), format(float(value), ".17g")) for key, value in geometry.items()))


def assemble_full_dc_outputs(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine four worker bundles and build extra-to-ultra convergence rows."""

    import state_characterization_config as config

    expected = {(mesh, sweep) for mesh in MESH_LEVELS for sweep in SWEEP_KINDS}
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    for bundle in bundles:
        key = _bundle_key(bundle)
        if key not in expected:
            errors.append(f"unexpected bundle {key}")
        elif key in indexed:
            errors.append(f"duplicate bundle {key}")
        else:
            indexed[key] = bundle
    missing = sorted(expected - set(indexed))
    if missing:
        errors.append(f"missing bundles={missing}")

    raw_rows: list[dict[str, Any]] = []
    idvg_rows: list[dict[str, Any]] = []
    idvd_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for key in sorted(indexed):
        mesh_level, sweep_kind = key
        bundle = indexed[key]
        if not bool(bundle.get("passed")):
            errors.append(f"worker failed {key}: {bundle.get('error_message', '')}")
        assembled_grid_validation = validate_full_sweep_rows(
            sweep_kind,
            bundle.get("raw_rows", ()),
            states=config.MEMORY_STATES,
            expected_coordinates=_canonical_expected_coordinates(
                sweep_kind, states=config.MEMORY_STATES, config=config
            ),
            orchestration_errors=bundle.get("orchestration_errors", ()),
        )
        assembled_metric_validation = (
            validate_metrics_rows(
                bundle.get("metrics_rows", ()),
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
            if sweep_kind == "idvg"
            else {"passed": True, "errors": []}
        )
        if not assembled_grid_validation["passed"]:
            errors.append(
                f"assembler grid validation failed {key}: "
                + "; ".join(assembled_grid_validation["errors"])
            )
        if not assembled_metric_validation["passed"]:
            errors.append(
                f"assembler metric validation failed {key}: "
                + "; ".join(assembled_metric_validation["errors"])
            )
        validation_rows.append(
            {
                "mesh_level": mesh_level,
                "sweep_kind": sweep_kind,
                "grid_passed": bool(bundle.get("grid_validation", {}).get("passed")),
                "metrics_passed": bool(bundle.get("metric_validation", {}).get("passed")),
                "mesh_line_passed": bool(bundle.get("mesh_line_validation", {}).get("passed")),
                "assembler_grid_passed": bool(assembled_grid_validation["passed"]),
                "assembler_metrics_passed": bool(assembled_metric_validation["passed"]),
                "passed": bool(bundle.get("passed")),
                "runtime_seconds": bundle.get("runtime_seconds", math.nan),
                "error_message": str(bundle.get("error_message", "")),
            }
        )
        for source in bundle.get("raw_rows", ()):
            row = {"mesh_level": mesh_level, "sweep_kind": sweep_kind, **dict(source)}
            raw_rows.append(row)
            (idvg_rows if sweep_kind == "idvg" else idvd_rows).append(row)
        if sweep_kind == "idvg":
            metrics_rows.extend(
                {"mesh_level": mesh_level, **dict(source)}
                for source in bundle.get("metrics_rows", ())
            )

    if len(indexed) == len(expected):
        if any(not bundle.get("geometry") for bundle in indexed.values()):
            errors.append("runtime geometry is missing")
        signatures = {
            _geometry_signature(bundle.get("geometry", {})) for bundle in indexed.values()
        }
        if len(signatures) != 1:
            errors.append("runtime geometry differs across full-DC workers")
        position_hashes = {
            str(bundle.get("mesh_line_validation", {}).get("position_hash", ""))
            for bundle in indexed.values()
        }
        if len(position_hashes) != 1 or "" in position_hashes:
            errors.append("fixed local-mesh position hash differs or is missing")
        for mesh_level in MESH_LEVELS:
            counts = [
                indexed[(mesh_level, sweep)]["mesh_counts"] for sweep in SWEEP_KINDS
            ]
            if counts[0] != counts[1]:
                errors.append(f"mesh counts differ between sweeps for {mesh_level}")
        extra_count = _number(
            indexed[("local_extra_fine", "idvg")].get("mesh_counts", {}).get(
                "global_coordinate_count"
            )
        )
        ultra_count = _number(
            indexed[("local_ultra_fine", "idvg")].get("mesh_counts", {}).get(
                "global_coordinate_count"
            )
        )
        if not (math.isfinite(extra_count) and math.isfinite(ultra_count) and ultra_count > extra_count):
            errors.append("ultra-fine mesh is not strictly refined")

    convergence_rows: list[dict[str, Any]] = []
    if not missing:
        import run_contact_topology_dc_candidate as dc_comparison

        extra_idvg = indexed[("local_extra_fine", "idvg")].get("raw_rows", ())
        ultra_idvg = indexed[("local_ultra_fine", "idvg")].get("raw_rows", ())
        extra_idvd = indexed[("local_extra_fine", "idvd")].get("raw_rows", ())
        ultra_idvd = indexed[("local_ultra_fine", "idvd")].get("raw_rows", ())
        extra_metrics = indexed[("local_extra_fine", "idvg")].get("metrics_rows", ())
        ultra_metrics = indexed[("local_ultra_fine", "idvg")].get("metrics_rows", ())
        try:
            comparisons = dc_comparison.build_regression_rows(
                reference_idvg_rows=extra_idvg,
                candidate_idvg_rows=ultra_idvg,
                reference_idvd_rows=extra_idvd,
                candidate_idvd_rows=ultra_idvd,
                reference_metrics_rows=extra_metrics,
                candidate_metrics_rows=ultra_metrics,
            )
        except Exception as error:
            errors.append("convergence assembly failed: " + _error(error))
        else:
            for source in comparisons:
                convergence_rows.append(
                    {
                        "reference_mesh": "local_extra_fine",
                        "candidate_mesh": "local_ultra_fine",
                        **dict(source),
                        "passed": not bool(source.get("warning")),
                    }
                )
            failed_comparisons = sum(not bool(row["passed"]) for row in convergence_rows)
            if failed_comparisons:
                errors.append(f"extra-to-ultra convergence warnings={failed_comparisons}")

    raw_rows.sort(
        key=lambda row: (
            MESH_LEVELS.index(str(row["mesh_level"])),
            SWEEP_KINDS.index(str(row["sweep_kind"])),
            int(row["state_index"]), float(row["VDS_V"]), float(row["VGS_V"]),
        )
    )
    metrics_rows.sort(
        key=lambda row: (
            MESH_LEVELS.index(str(row["mesh_level"])),
            int(row["state_index"]), float(row["VDS_V"]),
        )
    )
    passed = not errors
    return {
        "status": FULL_DC_STATUS_COMPLETE if passed else FULL_DC_STATUS_FAILED,
        "passed": passed,
        "raw_rows": raw_rows,
        "idvg_rows": idvg_rows,
        "idvd_rows": idvd_rows,
        "metrics_rows": metrics_rows,
        "convergence_rows": convergence_rows,
        "validation_rows": validation_rows,
        "errors": errors,
    }


__all__ = (
    "MESH_LEVELS", "SWEEP_KINDS", "FULL_DC_STATUS_COMPLETE",
    "FULL_DC_STATUS_FAILED", "reference_gate_allows_full_dc",
    "build_full_dc_tasks", "validate_full_sweep_rows", "validate_metrics_rows",
    "run_full_dc_worker", "assemble_full_dc_outputs",
)
