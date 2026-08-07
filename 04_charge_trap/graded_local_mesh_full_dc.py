"""Candidate-only full-DC support for the graded local-mesh family.

This module writes no files and exposes no command-line ``main``.  It reuses
the canonical five-state ID-VG/ID-VD sweep and metric-extraction functions,
but installs :mod:`graded_local_mesh_family` only while the device mesh is
created.  Public ``shared_data`` writers are never called.

Ioff below :data:`GRADED_IOFF_ACCEPTANCE_FLOOR_A` is numerically
floor-limited.  The extracted value is preserved in ``raw_Ioff_A`` while the
reported ``Ioff_A`` and ``on_off_ratio`` become NaN with explicit status tags.
Ioff and ON/OFF are informational at every magnitude and never hard-gate the
split DC acceptance.  Untagged metric NaNs fail validation.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Mapping, Sequence
from typing import Any

import local_mesh_full_dc as base_full_dc


MESH_LEVELS = ("graded_extra_fine", "graded_ultra_fine")
SWEEP_KINDS = base_full_dc.SWEEP_KINDS
FULL_DC_STATUS_COMPLETE = "graded_candidate_full_dc_complete"
FULL_DC_STATUS_FAILED = "graded_candidate_full_dc_failed_closed"

GRADED_IOFF_ACCEPTANCE_FLOOR_A = 5.0e-23
VTH_ABSOLUTE_TOLERANCE_V = 2.0e-3
SS_ION_RELATIVE_TOLERANCE = 2.0e-2
GM_RELATIVE_TOLERANCE = 5.0e-2
CURVE_ABSOLUTE_TOLERANCE_A = 1.0e-18
CURVE_RELATIVE_TOLERANCE = 5.0e-2

IOFF_STATUS_REPORTED = "reported_above_graded_acceptance_floor"
IOFF_STATUS_FLOOR_LIMITED = "floor_limited_below_graded_acceptance_floor"
IOFF_STATUS_INVALID = "invalid_or_untagged"
ON_OFF_STATUS_REPORTED = "reported_from_above_floor_ioff"
ON_OFF_STATUS_FLOOR_LIMITED = "not_reported_floor_limited_ioff"
ON_OFF_STATUS_INVALID = "invalid_or_untagged"

METRIC_REQUIRED_FINITE_FIELDS = (
    "ntrap_cm3", "nsheet_cm2", "VDS_V", "Vth_V",
    "threshold_current_A", "SS_mV_dec", "ss_r_squared", "Ion_A",
    "ion_vgs_V", "ion_vds_V", "ioff_vgs_V", "ioff_vds_V",
    "gm_max_S", "VGS_at_gm_max_V", "ss_current_min_A",
    "ss_current_max_A", "current_floor_A", "ioff_acceptance_floor_A",
    "raw_Ioff_A",
)


def _number(value: Any) -> float:
    return base_full_dc._number(value)


def _strict_bool(value: Any) -> bool:
    return base_full_dc._strict_bool(value)


def _error(error: object | None) -> str:
    return base_full_dc._error(error)


def reference_gate_allows_full_dc(reference_gate: Mapping[str, Any]) -> bool:
    """Require both gate bits; neither an implicit truth nor status is enough."""

    return bool(reference_gate.get("passed")) and bool(
        reference_gate.get("full_sweep_allowed")
    )


def build_full_dc_tasks(reference_gate: Mapping[str, Any]) -> tuple[dict[str, str], ...]:
    """Return four fresh-process tasks only for an explicitly passed gate."""

    if not reference_gate_allows_full_dc(reference_gate):
        return ()
    return tuple(
        {"mesh_level": mesh_level, "sweep_kind": sweep_kind}
        for mesh_level in MESH_LEVELS
        for sweep_kind in SWEEP_KINDS
    )


def postprocess_metrics_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    acceptance_floor_A: float = GRADED_IOFF_ACCEPTANCE_FLOOR_A,
) -> list[dict[str, Any]]:
    """Preserve raw Ioff and explicitly mask floor-limited reported metrics."""

    floor = float(acceptance_floor_A)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("acceptance_floor_A must be finite and positive")
    processed: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        raw_ioff = _number(source.get("Ioff_A"))
        raw_ratio = _number(source.get("on_off_ratio"))
        floor_limited = math.isfinite(raw_ioff) and 0.0 <= raw_ioff < floor
        row.update(
            {
                "raw_Ioff_A": raw_ioff,
                "raw_on_off_ratio": raw_ratio,
                "ioff_acceptance_floor_A": floor,
                "ioff_hard_acceptance_applies": False,
                "on_off_hard_acceptance_applies": False,
                "ioff_floor_limited": floor_limited,
            }
        )
        if floor_limited:
            row.update(
                {
                    "Ioff_A": math.nan,
                    "on_off_ratio": math.nan,
                    "ioff_status": IOFF_STATUS_FLOOR_LIMITED,
                    "on_off_ratio_status": ON_OFF_STATUS_FLOOR_LIMITED,
                }
            )
        elif math.isfinite(raw_ioff) and raw_ioff >= floor:
            row.update(
                {
                    "Ioff_A": raw_ioff,
                    "on_off_ratio": raw_ratio,
                    "ioff_status": IOFF_STATUS_REPORTED,
                    "on_off_ratio_status": (
                        ON_OFF_STATUS_REPORTED
                        if math.isfinite(raw_ratio)
                        else ON_OFF_STATUS_INVALID
                    ),
                }
            )
        else:
            row.update(
                {
                    "Ioff_A": math.nan,
                    "on_off_ratio": math.nan,
                    "ioff_status": IOFF_STATUS_INVALID,
                    "on_off_ratio_status": ON_OFF_STATUS_INVALID,
                }
            )
        processed.append(row)
    return processed


def validate_graded_metrics_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    states: Sequence[Mapping[str, Any]] | None = None,
    vds_values_V: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Validate metric grid and reject every untagged or unexplained NaN."""

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
    coordinates: list[tuple[int, float]] = []
    failures: list[str] = []
    failed_count = 0
    nonfinite_count = 0
    status_error_count = 0
    metadata_error_count = 0
    for index, row in enumerate(rows):
        try:
            state_index = int(row["state_index"])
            coordinate = (state_index, round(float(row["VDS_V"]), 12))
        except (KeyError, TypeError, ValueError, OverflowError):
            failures.append(f"metric row {index} has an invalid coordinate")
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
            extracted = _strict_bool(row.get("vth_success")) and _strict_bool(
                row.get("ss_success")
            )
        except ValueError:
            extracted = False
        if (
            not extracted
            or str(row.get("vth_error", "")).strip()
            or str(row.get("ss_error", "")).strip()
        ):
            failed_count += 1
        finite_values = [_number(row.get(field)) for field in METRIC_REQUIRED_FINITE_FIELDS]
        try:
            ss_points = int(row.get("ss_point_count"))
            ss_required = int(row.get("ss_minimum_point_count"))
        except (TypeError, ValueError, OverflowError):
            ss_points, ss_required = -1, 0
        if (
            not all(math.isfinite(value) for value in finite_values)
            or _number(row.get("Ion_A")) < 0.0
            or _number(row.get("gm_max_S")) < 0.0
            or _number(row.get("raw_Ioff_A")) < 0.0
            or ss_points < ss_required
        ):
            nonfinite_count += 1

        raw_ioff = _number(row.get("raw_Ioff_A"))
        floor = _number(row.get("ioff_acceptance_floor_A"))
        public_ioff = _number(row.get("Ioff_A"))
        public_ratio = _number(row.get("on_off_ratio"))
        raw_ratio = _number(row.get("raw_on_off_ratio"))
        try:
            floor_flag = _strict_bool(row.get("ioff_floor_limited"))
            ioff_hard = _strict_bool(row.get("ioff_hard_acceptance_applies"))
            ratio_hard = _strict_bool(row.get("on_off_hard_acceptance_applies"))
        except ValueError:
            floor_flag, ioff_hard, ratio_hard = False, True, True
        expected_floor_limited = (
            math.isfinite(raw_ioff) and math.isfinite(floor)
            and 0.0 <= raw_ioff < floor
        )
        if expected_floor_limited:
            status_ok = all(
                (
                    floor_flag,
                    not ioff_hard,
                    not ratio_hard,
                    str(row.get("ioff_status")) == IOFF_STATUS_FLOOR_LIMITED,
                    str(row.get("on_off_ratio_status")) == ON_OFF_STATUS_FLOOR_LIMITED,
                    not math.isfinite(public_ioff),
                    not math.isfinite(public_ratio),
                )
            )
            # raw_on_off_ratio may itself be NaN when the underlying extractor
            # reached its lower numerical floor; the explicit floor tag covers it.
        else:
            status_ok = all(
                (
                    math.isfinite(raw_ioff),
                    math.isfinite(floor),
                    raw_ioff >= floor,
                    not floor_flag,
                    not ioff_hard,
                    not ratio_hard,
                    str(row.get("ioff_status")) == IOFF_STATUS_REPORTED,
                    str(row.get("on_off_ratio_status")) == ON_OFF_STATUS_REPORTED,
                    math.isfinite(public_ioff),
                    math.isclose(public_ioff, raw_ioff, rel_tol=0.0, abs_tol=0.0),
                    math.isfinite(public_ratio),
                    math.isfinite(raw_ratio),
                    math.isclose(public_ratio, raw_ratio, rel_tol=0.0, abs_tol=0.0),
                )
            )
        if not status_ok:
            status_error_count += 1

    actual = set(coordinates)
    duplicate_count = len(coordinates) - len(actual)
    missing = expected - actual
    unexpected = actual - expected
    if len(state_by_index) != 5 or set(state_by_index) != {0, 1, 2, 3, 4}:
        failures.append("canonical five-state set is incomplete")
    if duplicate_count:
        failures.append(f"duplicate metric coordinates={duplicate_count}")
    if missing:
        failures.append(f"missing metric coordinates={len(missing)}")
    if unexpected:
        failures.append(f"unexpected metric coordinates={len(unexpected)}")
    if failed_count:
        failures.append(f"failed metric rows={failed_count}")
    if nonfinite_count:
        failures.append(f"nonfinite/invalid metric rows={nonfinite_count}")
    if status_error_count:
        failures.append(f"untagged/invalid Ioff or ON-OFF rows={status_error_count}")
    if metadata_error_count:
        failures.append(f"metric metadata mismatches={metadata_error_count}")
    return {
        "expected_row_count": len(expected),
        "actual_row_count": len(rows),
        "duplicate_count": duplicate_count,
        "missing_count": len(missing),
        "unexpected_count": len(unexpected),
        "failed_count": failed_count,
        "nonfinite_count": nonfinite_count,
        "status_error_count": status_error_count,
        "metadata_error_count": metadata_error_count,
        "floor_limited_count": sum(
            str(row.get("ioff_status")) == IOFF_STATUS_FLOOR_LIMITED for row in rows
        ),
        "passed": not failures,
        "errors": failures,
    }


def _comparison(
    reference: Any,
    candidate: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    scale_floor: float,
) -> dict[str, Any]:
    left = _number(reference)
    right = _number(candidate)
    finite = math.isfinite(left) and math.isfinite(right)
    scale = max(abs(left), abs(right), float(scale_floor)) if finite else math.nan
    difference = abs(right - left) if finite else math.nan
    allowed = (
        float(absolute_tolerance) + float(relative_tolerance) * scale
        if finite else math.nan
    )
    return {
        "reference_value": left,
        "candidate_value": right,
        "absolute_difference": difference,
        "relative_difference": difference / scale if finite else math.nan,
        "comparison_scale": scale,
        "absolute_tolerance": float(absolute_tolerance),
        "relative_tolerance": float(relative_tolerance),
        "allowed_difference": allowed,
        "finite": finite,
        "within_tolerance": bool(finite and difference <= allowed),
    }


def _unique_rows(
    rows: Sequence[Mapping[str, Any]],
    key_function: Any,
    *,
    label: str,
) -> tuple[dict[Any, Mapping[str, Any]], list[str]]:
    result: dict[Any, Mapping[str, Any]] = {}
    errors: list[str] = []
    for row in rows:
        try:
            key = key_function(row)
        except Exception as error:
            errors.append(f"{label} invalid key: {_error(error)}")
            continue
        if key in result:
            errors.append(f"{label} duplicate key {key!r}")
        else:
            result[key] = row
    return result, errors


def _curve_key(row: Mapping[str, Any]) -> tuple[int, float, float]:
    return (
        int(row["state_index"]),
        round(float(row["VGS_V"]), 12),
        round(float(row["VDS_V"]), 12),
    )


def _metric_key(row: Mapping[str, Any]) -> tuple[int, float]:
    return int(row["state_index"]), round(float(row["VDS_V"]), 12)


def _metadata_matches(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        str(left.get("state", "")) == str(right.get("state", ""))
        and math.isclose(
            _number(left.get("ntrap_cm3")), _number(right.get("ntrap_cm3")),
            rel_tol=1.0e-12, abs_tol=0.0,
        )
    )


def build_split_dc_convergence_rows(
    *,
    extra_idvg_rows: Sequence[Mapping[str, Any]],
    ultra_idvg_rows: Sequence[Mapping[str, Any]],
    extra_idvd_rows: Sequence[Mapping[str, Any]],
    ultra_idvd_rows: Sequence[Mapping[str, Any]],
    extra_metrics_rows: Sequence[Mapping[str, Any]],
    ultra_metrics_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build explicit graded extra-to-ultra curve and metric acceptance rows."""

    output: list[dict[str, Any]] = []
    for domain, reference_rows, candidate_rows in (
        ("idvg", extra_idvg_rows, ultra_idvg_rows),
        ("idvd", extra_idvd_rows, ultra_idvd_rows),
    ):
        reference, reference_errors = _unique_rows(
            reference_rows, _curve_key, label=f"{domain} graded-extra"
        )
        candidate, candidate_errors = _unique_rows(
            candidate_rows, _curve_key, label=f"{domain} graded-ultra"
        )
        index_errors = [*reference_errors, *candidate_errors]
        for key in sorted(set(reference) | set(candidate)):
            left = reference.get(key)
            right = candidate.get(key)
            identity = right or left or {}
            comparison = _comparison(
                left.get("ID_A") if left else math.nan,
                right.get("ID_A") if right else math.nan,
                absolute_tolerance=CURVE_ABSOLUTE_TOLERANCE_A,
                relative_tolerance=CURVE_RELATIVE_TOLERANCE,
                scale_floor=CURVE_ABSOLUTE_TOLERANCE_A,
            )
            try:
                left_converged = bool(left) and _strict_bool(left.get("converged"))
                right_converged = bool(right) and _strict_bool(right.get("converged"))
            except ValueError:
                left_converged = right_converged = False
            reasons = list(index_errors)
            if left is None:
                reasons.append("missing_graded_extra_point")
            if right is None:
                reasons.append("missing_graded_ultra_point")
            if left is not None and right is not None and not _metadata_matches(left, right):
                reasons.append("metadata_mismatch")
            if not left_converged:
                reasons.append("graded_extra_unconverged")
            if not right_converged:
                reasons.append("graded_ultra_unconverged")
            if not comparison["within_tolerance"]:
                reasons.append("curve_difference_exceeds_combined_tolerance")
            output.append(
                {
                    "comparison_domain": domain,
                    "quantity": "ID_A",
                    "reference_mesh": "graded_extra_fine",
                    "candidate_mesh": "graded_ultra_fine",
                    "state_index": key[0],
                    "state": str(identity.get("state", "")),
                    "VGS_V": key[1],
                    "VDS_V": key[2],
                    "acceptance_policy": "combined_abs_1e-18_A_plus_rel_5pct",
                    "acceptance_applies": True,
                    **comparison,
                    "passed": not reasons,
                    "error_message": ";".join(dict.fromkeys(reasons)),
                }
            )

    extra_metrics, extra_errors = _unique_rows(
        extra_metrics_rows, _metric_key, label="metrics graded-extra"
    )
    ultra_metrics, ultra_errors = _unique_rows(
        ultra_metrics_rows, _metric_key, label="metrics graded-ultra"
    )
    metric_index_errors = [*extra_errors, *ultra_errors]
    policies = (
        ("Vth_V", VTH_ABSOLUTE_TOLERANCE_V, 0.0, 1.0e-30,
         "absolute_2mV", "vth_success"),
        ("SS_mV_dec", 0.0, SS_ION_RELATIVE_TOLERANCE, 1.0e-12,
         "relative_2pct", "ss_success"),
        ("Ion_A", 0.0, SS_ION_RELATIVE_TOLERANCE, 1.0e-30,
         "relative_2pct", ""),
        ("gm_max_S", 0.0, GM_RELATIVE_TOLERANCE, 1.0e-30,
         "relative_5pct", ""),
    )
    for key in sorted(set(extra_metrics) | set(ultra_metrics)):
        left = extra_metrics.get(key)
        right = ultra_metrics.get(key)
        identity = right or left or {}
        for quantity, atol, rtol, floor, policy, success_field in policies:
            comparison = _comparison(
                left.get(quantity) if left else math.nan,
                right.get(quantity) if right else math.nan,
                absolute_tolerance=atol,
                relative_tolerance=rtol,
                scale_floor=floor,
            )
            reasons = list(metric_index_errors)
            if left is None:
                reasons.append("missing_graded_extra_metric")
            if right is None:
                reasons.append("missing_graded_ultra_metric")
            if left is not None and right is not None and not _metadata_matches(left, right):
                reasons.append("metadata_mismatch")
            if success_field:
                try:
                    if left is None or not _strict_bool(left.get(success_field)):
                        reasons.append("graded_extra_extraction_failed")
                    if right is None or not _strict_bool(right.get(success_field)):
                        reasons.append("graded_ultra_extraction_failed")
                except ValueError:
                    reasons.append("invalid_extraction_status")
            if not comparison["within_tolerance"]:
                reasons.append("metric_difference_exceeds_tolerance")
            output.append(
                {
                    "comparison_domain": "metrics",
                    "quantity": quantity,
                    "reference_mesh": "graded_extra_fine",
                    "candidate_mesh": "graded_ultra_fine",
                    "state_index": key[0],
                    "state": str(identity.get("state", "")),
                    "VGS_V": math.nan,
                    "VDS_V": key[1],
                    "acceptance_policy": policy,
                    "acceptance_applies": True,
                    **comparison,
                    "passed": not reasons,
                    "error_message": ";".join(dict.fromkeys(reasons)),
                }
            )

        for quantity, raw_name, status_name in (
            ("Ioff_A", "raw_Ioff_A", "ioff_status"),
            ("on_off_ratio", "raw_on_off_ratio", "on_off_ratio_status"),
        ):
            left_value = left.get(raw_name) if left else math.nan
            right_value = right.get(raw_name) if right else math.nan
            comparison = _comparison(
                left_value,
                right_value,
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                scale_floor=(GRADED_IOFF_ACCEPTANCE_FLOOR_A if quantity == "Ioff_A" else 1.0),
            )
            left_status = str(left.get(status_name, "")) if left else "missing"
            right_status = str(right.get(status_name, "")) if right else "missing"
            valid_statuses = (
                {IOFF_STATUS_REPORTED, IOFF_STATUS_FLOOR_LIMITED}
                if quantity == "Ioff_A"
                else {ON_OFF_STATUS_REPORTED, ON_OFF_STATUS_FLOOR_LIMITED}
            )
            valid = (
                left is not None and right is not None
                and left_status in valid_statuses and right_status in valid_statuses
                and _metadata_matches(left, right)
                and (
                    math.isfinite(_number(left_value))
                    and math.isfinite(_number(right_value))
                    if quantity == "Ioff_A"
                    else True
                )
            )
            output.append(
                {
                    "comparison_domain": "metrics",
                    "quantity": quantity,
                    "reference_mesh": "graded_extra_fine",
                    "candidate_mesh": "graded_ultra_fine",
                    "state_index": key[0],
                    "state": str(identity.get("state", "")),
                    "VGS_V": math.nan,
                    "VDS_V": key[1],
                    "acceptance_policy": "informational_no_hard_gate",
                    "acceptance_applies": False,
                    **comparison,
                    "reference_status": left_status,
                    "candidate_status": right_status,
                    "passed": valid,
                    "error_message": "" if valid else "invalid_or_untagged_informational_metric",
                }
            )

    domain_order = {"idvg": 0, "idvd": 1, "metrics": 2}
    output.sort(
        key=lambda row: (
            domain_order[str(row["comparison_domain"])],
            int(row["state_index"]),
            _number(row.get("VDS_V")),
            _number(row.get("VGS_V")) if math.isfinite(_number(row.get("VGS_V"))) else -math.inf,
            str(row["quantity"]),
        )
    )
    return output


def run_full_dc_worker(mesh_level: str, sweep_kind: str) -> dict[str, Any]:
    """Run one graded-mesh canonical sweep without writing candidate/public data."""

    if mesh_level not in MESH_LEVELS:
        raise ValueError(f"mesh_level must be one of {MESH_LEVELS}")
    if sweep_kind not in SWEEP_KINDS:
        raise ValueError(f"sweep_kind must be one of {SWEEP_KINDS}")

    import graded_local_mesh_family as family
    import run_memory_window as memory_window
    import run_state_characterization as characterization
    import state_characterization_config as config
    from run_idvd_by_state import run_idvd_by_state
    from run_idvg_by_state import run_idvg_by_state
    from state_sweep_helpers import initialize_characterization_device

    config.validate_state_characterization_config()
    preview = base_full_dc._preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with family.local_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        preview,
        mesh_level,
        line_log,
    ):
        geometry, operating_point = initialize_characterization_device()
    mesh_validation = family.validate_local_mesh_line_log(line_log, geometry, mesh_level)
    if not bool(mesh_validation.get("passed")):
        raise RuntimeError(
            "graded mesh-line validation failed: "
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
            extracted = characterization.build_metrics_rows(
                raw_rows,
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
            metrics_rows = postprocess_metrics_rows(extracted)
            metric_validation = validate_graded_metrics_rows(
                metrics_rows,
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
        except Exception as error:
            metric_validation = {"passed": False, "errors": [_error(error)]}
    else:
        raw_rows, operating_point = run_idvd_by_state(
            states=config.MEMORY_STATES,
            vgs_values_V=config.IDVD_VGS_VALUES_V,
            vds_start_V=config.IDVD_VDS_START_V,
            vds_stop_V=config.IDVD_VDS_STOP_V,
            vds_step_V=config.IDVD_VDS_STEP_V,
            operating_point=operating_point,
        )

    grid_validation = base_full_dc.validate_full_sweep_rows(
        sweep_kind,
        raw_rows,
        states=config.MEMORY_STATES,
        expected_coordinates=base_full_dc._expected_coordinates(
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
        "mesh_counts": base_full_dc._runtime_mesh_counts(memory_window.device),
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


def assemble_full_dc_outputs(bundles: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Revalidate four graded workers and apply the split DC acceptance."""

    import state_characterization_config as config

    expected = {(mesh, sweep) for mesh in MESH_LEVELS for sweep in SWEEP_KINDS}
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    errors: list[str] = []
    for bundle in bundles:
        key = (str(bundle.get("mesh_level", "")), str(bundle.get("sweep_kind", "")))
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
    metrics_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for key in sorted(indexed):
        mesh_level, sweep_kind = key
        bundle = indexed[key]
        raw_validation = base_full_dc.validate_full_sweep_rows(
            sweep_kind,
            bundle.get("raw_rows", ()),
            states=config.MEMORY_STATES,
            expected_coordinates=base_full_dc._canonical_expected_coordinates(
                sweep_kind, states=config.MEMORY_STATES, config=config
            ),
            orchestration_errors=bundle.get("orchestration_errors", ()),
        )
        metric_validation = (
            validate_graded_metrics_rows(
                bundle.get("metrics_rows", ()),
                states=config.MEMORY_STATES,
                vds_values_V=config.IDVG_VDS_VALUES_V,
            )
            if sweep_kind == "idvg"
            else {"passed": True, "errors": [], "floor_limited_count": 0}
        )
        mesh_passed = bool(bundle.get("mesh_line_validation", {}).get("passed"))
        worker_passed = bool(bundle.get("passed"))
        if not worker_passed:
            errors.append(f"worker failed {key}: {bundle.get('error_message', '')}")
        if not raw_validation["passed"]:
            errors.append(
                f"assembler raw-grid validation failed {key}: "
                + "; ".join(raw_validation["errors"])
            )
        if not metric_validation["passed"]:
            errors.append(
                f"assembler metric validation failed {key}: "
                + "; ".join(metric_validation["errors"])
            )
        if not mesh_passed:
            errors.append(f"graded mesh-line validation failed {key}")
        validation_rows.append(
            {
                "mesh_level": mesh_level,
                "sweep_kind": sweep_kind,
                "raw_grid_passed": bool(raw_validation["passed"]),
                "metrics_passed": bool(metric_validation["passed"]),
                "mesh_line_passed": mesh_passed,
                "worker_passed": worker_passed,
                "floor_limited_metric_count": metric_validation.get(
                    "floor_limited_count", 0
                ),
                "passed": all((worker_passed, raw_validation["passed"],
                               metric_validation["passed"], mesh_passed)),
                "runtime_seconds": bundle.get("runtime_seconds", math.nan),
                "error_message": str(bundle.get("error_message", "")),
            }
        )
        raw_rows.extend(
            {"mesh_level": mesh_level, "sweep_kind": sweep_kind, **dict(row)}
            for row in bundle.get("raw_rows", ())
        )
        if sweep_kind == "idvg":
            metrics_rows.extend(
                {"mesh_level": mesh_level, **dict(row)}
                for row in bundle.get("metrics_rows", ())
            )

    if len(indexed) == len(expected):
        if any(not bundle.get("geometry") for bundle in indexed.values()):
            errors.append("runtime geometry is missing")
        try:
            geometry_signatures = {
                base_full_dc._geometry_signature(bundle.get("geometry", {}))
                for bundle in indexed.values()
            }
        except Exception as error:
            geometry_signatures = set()
            errors.append("invalid runtime geometry: " + _error(error))
        if len(geometry_signatures) != 1:
            errors.append("runtime geometry differs across graded full-DC workers")
        position_hashes = {
            str(bundle.get("mesh_line_validation", {}).get("position_hash", ""))
            for bundle in indexed.values()
        }
        if len(position_hashes) != 1 or "" in position_hashes:
            errors.append("fixed graded-mesh position hash differs or is missing")
        for mesh_level in MESH_LEVELS:
            counts = [indexed[(mesh_level, sweep)].get("mesh_counts", {})
                      for sweep in SWEEP_KINDS]
            if counts[0] != counts[1]:
                errors.append(f"mesh counts differ between sweeps for {mesh_level}")
        extra_count = _number(
            indexed[("graded_extra_fine", "idvg")].get("mesh_counts", {}).get(
                "global_coordinate_count"
            )
        )
        ultra_count = _number(
            indexed[("graded_ultra_fine", "idvg")].get("mesh_counts", {}).get(
                "global_coordinate_count"
            )
        )
        if not (
            math.isfinite(extra_count) and math.isfinite(ultra_count)
            and ultra_count > extra_count
        ):
            errors.append("graded ultra-fine mesh is not strictly refined")

    convergence_rows: list[dict[str, Any]] = []
    if not missing:
        try:
            convergence_rows = build_split_dc_convergence_rows(
                extra_idvg_rows=indexed[("graded_extra_fine", "idvg")].get("raw_rows", ()),
                ultra_idvg_rows=indexed[("graded_ultra_fine", "idvg")].get("raw_rows", ()),
                extra_idvd_rows=indexed[("graded_extra_fine", "idvd")].get("raw_rows", ()),
                ultra_idvd_rows=indexed[("graded_ultra_fine", "idvd")].get("raw_rows", ()),
                extra_metrics_rows=indexed[("graded_extra_fine", "idvg")].get("metrics_rows", ()),
                ultra_metrics_rows=indexed[("graded_ultra_fine", "idvg")].get("metrics_rows", ()),
            )
        except Exception as error:
            errors.append("split-acceptance assembly failed: " + _error(error))
        else:
            hard_failures = [
                row for row in convergence_rows
                if bool(row.get("acceptance_applies")) and not bool(row.get("passed"))
            ]
            invalid_informational = [
                row for row in convergence_rows
                if not bool(row.get("acceptance_applies")) and not bool(row.get("passed"))
            ]
            if hard_failures:
                errors.append(f"split DC hard-acceptance failures={len(hard_failures)}")
            if invalid_informational:
                errors.append(
                    f"invalid/untagged informational metrics={len(invalid_informational)}"
                )

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
        "idvg_rows": [row for row in raw_rows if row["sweep_kind"] == "idvg"],
        "idvd_rows": [row for row in raw_rows if row["sweep_kind"] == "idvd"],
        "metrics_rows": metrics_rows,
        "convergence_rows": convergence_rows,
        "validation_rows": validation_rows,
        "errors": errors,
    }


__all__ = (
    "MESH_LEVELS", "SWEEP_KINDS", "GRADED_IOFF_ACCEPTANCE_FLOOR_A",
    "VTH_ABSOLUTE_TOLERANCE_V", "SS_ION_RELATIVE_TOLERANCE",
    "GM_RELATIVE_TOLERANCE", "CURVE_ABSOLUTE_TOLERANCE_A",
    "CURVE_RELATIVE_TOLERANCE", "IOFF_STATUS_REPORTED",
    "IOFF_STATUS_FLOOR_LIMITED", "ON_OFF_STATUS_REPORTED",
    "ON_OFF_STATUS_FLOOR_LIMITED", "reference_gate_allows_full_dc",
    "build_full_dc_tasks", "postprocess_metrics_rows",
    "validate_graded_metrics_rows", "build_split_dc_convergence_rows",
    "run_full_dc_worker", "assemble_full_dc_outputs",
)
