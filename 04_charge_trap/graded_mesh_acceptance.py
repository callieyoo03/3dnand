"""Candidate-only split acceptance for the graded local-mesh study.

The existing characterization current floor (``1e-30 A``) is a logarithmic
floor, not a measurement-resolution claim.  This module therefore derives a
separate DC quantification floor from the already selected absolute and
relative mesh tolerances::

    1e-24 A / 0.02 = 5e-23 A

Raw signed terminal currents are always retained.  A current below that
quantification floor is reported as NaN with the explicit status
``below_numerical_floor``.  Its sign and relative source/drain continuity do
not participate in the electrostatic Q/C gate.

There are no DEVSIM imports and no file writers in this module.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence


GRADED_LEVELS = (
    "graded_base", "graded_fine", "graded_extra_fine", "graded_ultra_fine",
)
EXTRA_FINE = "graded_extra_fine"
ULTRA_FINE = "graded_ultra_fine"
FINAL_PAIR = (EXTRA_FINE, ULTRA_FINE)
STATE0_ON = (0, "on")
STATE4_OFF = (4, "off")
REFERENCE_STATES = (STATE0_ON, STATE4_OFF)

ID_ATOL_A = 1.0e-24
ION_MESH_RTOL = 2.0e-2
DC_CURRENT_QUANTIFICATION_FLOOR_A = ID_ATOL_A / ION_MESH_RTOL
CURRENT_CONTINUITY_ATOL_A = 1.0e-24
CURRENT_CONTINUITY_RTOL = 1.0e-6

CHARGE_FLOOR_C = 1.0e-30
CHARGE_ATOL_C = 1.0e-28
CHARGE_MESH_RTOL = 2.0e-2
CAPACITANCE_FLOOR_F = 1.0e-24
CAPACITANCE_ATOL_F = 1.0e-22
CAPACITANCE_MESH_RTOL = 5.0e-2
REPEAT_RTOL = 1.0e-4
DELTA_SENSITIVITY_LIMIT = 1.0e-2

PHYSICAL_STATUS_CONVERGED = "numerically_converged_provisional"
PHYSICAL_STATUS_NOT_CONVERGED = "provisional_not_mesh_converged"
CURRENT_STATUS_QUANTIFIED = "quantified"
CURRENT_STATUS_BELOW_FLOOR = "below_numerical_floor"
CURRENT_STATUS_NONFINITE = "nonfinite"

POINT_COMMON_FIELDS = (
    "target_reached",
    "final_reference_validation_passed",
    "topology_passed",
    "mesh_line_validation_passed",
    "runtime_invariants_passed",
)
ELECTROSTATIC_POINT_FIELDS = (
    "global_gauss_passed",
    "derivative_gauss_passed",
    "gauge_row_sum_passed",
)
Q_QUANTITIES = ("Qg_C", "raw_Qd_C", "raw_Qs_C")
C_QUANTITIES = ("Cgg_F", "Cgd_F", "Cgs_F")

_LEVEL_ALIASES = {
    "local_base": "graded_base",
    "local_fine": "graded_fine",
    "local_extra_fine": EXTRA_FINE,
    "local_ultra_fine": ULTRA_FINE,
    "local_ultra_fine_repeat": f"{ULTRA_FINE}_repeat",
}


def _canonical_level(value: Any) -> str:
    name = str(value)
    return _LEVEL_ALIASES.get(name, name)


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
    return False


def _state_key(row: Mapping[str, Any]) -> tuple[int, str]:
    try:
        state_index = int(row["state_index"])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("point/comparison row has invalid state_index") from error
    bias_name = str(row.get("bias_name", "")).strip().lower()
    if not bias_name:
        raise ValueError("point/comparison row has no bias_name")
    return state_index, bias_name


def combined_tolerance(
    reference: Any,
    candidate: Any,
    *,
    floor: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Return a fail-closed combined absolute/relative comparison."""

    if not all(math.isfinite(float(value)) for value in (floor, atol, rtol)):
        raise ValueError("comparison tolerances must be finite")
    if floor <= 0.0 or atol < 0.0 or rtol < 0.0:
        raise ValueError("comparison floor must be positive and tolerances nonnegative")
    left = _number(reference)
    right = _number(candidate)
    finite = math.isfinite(left) and math.isfinite(right)
    if finite:
        scale = max(abs(left), abs(right), float(floor))
        difference = abs(right - left)
        allowed = float(atol) + float(rtol) * scale
        relative = difference / scale
    else:
        scale = difference = allowed = relative = math.nan
    return {
        "reference_value": left,
        "candidate_value": right,
        "absolute_difference": difference,
        "comparison_scale": scale,
        "relative_difference": relative,
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "floor": float(floor),
        "allowed_difference": allowed,
        "finite": finite,
        "passed": bool(finite and difference <= allowed),
    }


def classify_current_point(
    point: Mapping[str, Any],
    *,
    quantification_floor_A: float = DC_CURRENT_QUANTIFICATION_FLOOR_A,
) -> dict[str, Any]:
    """Return a copy with raw-current provenance and reporting status.

    The input mapping is never modified.  ``reported_ID_A`` is NaN below the
    floor while ``raw_ID_A``, ``raw_source_current_A`` and
    ``raw_drain_current_A`` retain the solver/API values.
    """

    floor = _number(quantification_floor_A)
    if not math.isfinite(floor) or floor <= 0.0:
        raise ValueError("quantification_floor_A must be finite and positive")
    raw_id = _number(point.get("ID_A", point.get("drain_current_A")))
    raw_drain = _number(point.get("drain_current_A", point.get("ID_A")))
    raw_source = _number(point.get("source_current_A"))
    finite = all(math.isfinite(value) for value in (raw_id, raw_drain, raw_source))
    scale = max(abs(raw_drain), abs(raw_source)) if finite else math.nan
    residual = raw_source + raw_drain if finite else math.nan
    strict_tolerance = (
        CURRENT_CONTINUITY_ATOL_A + CURRENT_CONTINUITY_RTOL * scale
        if finite else math.nan
    )
    strict_continuity = bool(
        finite and abs(residual) <= strict_tolerance
    )
    if not finite:
        status = CURRENT_STATUS_NONFINITE
    elif scale < floor:
        status = CURRENT_STATUS_BELOW_FLOOR
    else:
        status = CURRENT_STATUS_QUANTIFIED
    gate_applicable = status == CURRENT_STATUS_QUANTIFIED
    if status == CURRENT_STATUS_BELOW_FLOOR:
        continuity_status = "not_applicable_below_numerical_floor"
    elif status == CURRENT_STATUS_NONFINITE:
        continuity_status = "failed_nonfinite"
    else:
        continuity_status = "passed" if strict_continuity else "failed"
    result = dict(point)
    result.update(
        {
            "raw_ID_A": raw_id,
            "raw_drain_current_A": raw_drain,
            "raw_source_current_A": raw_source,
            "reported_ID_A": raw_id if status == CURRENT_STATUS_QUANTIFIED else math.nan,
            "reported_Ioff_A": (
                raw_id if status == CURRENT_STATUS_QUANTIFIED else math.nan
            ),
            "dc_quantification_floor_A": floor,
            "dc_current_scale_A": scale,
            "dc_current_status": status,
            "raw_continuity_residual_A": residual,
            "raw_continuity_tolerance_A": strict_tolerance,
            "raw_continuity_passed": strict_continuity,
            "continuity_gate_applicable": gate_applicable,
            "dc_continuity_acceptance_status": continuity_status,
        }
    )
    return result


def classify_current(
    point_or_ID_A: Mapping[str, Any] | float,
    source_current_A: float | None = None,
    drain_current_A: float | None = None,
    *,
    state_index: int = 0,
    bias_name: str = "on",
    quantification_floor_A: float = DC_CURRENT_QUANTIFICATION_FLOOR_A,
) -> dict[str, Any]:
    """Stable classification API accepting a point row or three currents."""

    if isinstance(point_or_ID_A, Mapping):
        point = point_or_ID_A
    else:
        point = {
            "state_index": state_index,
            "bias_name": bias_name,
            "ID_A": point_or_ID_A,
            "drain_current_A": (
                point_or_ID_A if drain_current_A is None else drain_current_A
            ),
            "source_current_A": source_current_A,
        }
    return classify_current_point(
        point, quantification_floor_A=quantification_floor_A
    )


def annotate_reference_point(row: Mapping[str, Any]) -> dict[str, Any]:
    """Return a CSV-ready copy annotated with graded-current acceptance fields."""

    result = classify_current_point(row)
    result["mesh_level"] = _canonical_level(result.get("mesh_level", ""))
    result["dc_transport_point_passed"] = bool(
        _common_point_pass(result)
        and result["dc_current_status"] != CURRENT_STATUS_NONFINITE
        and (
            result["dc_current_status"] == CURRENT_STATUS_BELOW_FLOOR
            or result["raw_continuity_passed"]
        )
    )
    result["electrostatic_qc_point_passed"] = _electrostatic_point_pass(result)
    return result


def _point_index(
    points: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, tuple[int, str], str], Mapping[str, Any]]:
    indexed: dict[tuple[str, tuple[int, str], str], Mapping[str, Any]] = {}
    for point in points:
        level = _canonical_level(point.get("mesh_level", ""))
        role = str(point.get("run_role", "primary")).strip().lower() or "primary"
        key = level, _state_key(point), role
        if key in indexed:
            raise ValueError(f"duplicate point {key!r}")
        indexed[key] = point
    return indexed


def _comparison_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str, str, tuple[int, str], str], Mapping[str, Any]]:
    indexed: dict[
        tuple[str, str, str, tuple[int, str], str], Mapping[str, Any]
    ] = {}
    for row in rows:
        comparison_type = str(row.get("comparison_type", ""))
        if comparison_type not in {"mesh", "repeat"}:
            continue
        key = (
            comparison_type,
            _canonical_level(row.get("reference_mesh", "")),
            _canonical_level(row.get("candidate_mesh", "")),
            _state_key(row),
            str(row.get("quantity", "")),
        )
        if key in indexed:
            raise ValueError(f"duplicate comparison {key!r}")
        indexed[key] = row
    return indexed


def quality_mapping_from_rows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, bool]:
    """Extract fail-closed family-quality decisions from quality CSV rows."""

    result: dict[str, bool] = {}
    for row in rows:
        if str(row.get("row_type", "")) not in {
            "family_quality_summary", "graded_quality_summary"
        }:
            continue
        level = _canonical_level(row.get("mesh_level", ""))
        if level in result:
            raise ValueError(f"duplicate family quality summary for {level!r}")
        result[level] = _strict_bool(row.get("quality_passed"))
    return result


def _quality_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return _strict_bool(value.get("passed", value.get("quality_passed")))
    return _strict_bool(value)


def _quality_for_level(quality_by_level: Mapping[str, Any], level: str) -> bool:
    for raw_level, value in quality_by_level.items():
        if _canonical_level(raw_level) == level:
            return _quality_value(value)
    return False


def _criterion(
    rows: list[dict[str, Any]],
    *,
    domain: str,
    name: str,
    scope: str,
    passed: bool,
    status: str | None = None,
    gate_applicable: bool = True,
    observed_value: Any = "",
    limit_value: Any = "",
    units: str = "",
    reason: str = "",
) -> bool:
    rows.append(
        {
            "row_type": "acceptance_criterion",
            "criterion_order": len(rows),
            "acceptance_domain": domain,
            "criterion": name,
            "scope": scope,
            "gate_applicable": bool(gate_applicable),
            "passed": bool(passed),
            "status": status or ("passed" if passed else "failed"),
            "observed_value": observed_value,
            "limit_value": limit_value,
            "units": units,
            "reason": reason,
        }
    )
    return bool(passed)


def _common_point_pass(point: Mapping[str, Any]) -> bool:
    return all(_strict_bool(point.get(field)) for field in POINT_COMMON_FIELDS)


def _electrostatic_point_pass(point: Mapping[str, Any]) -> bool:
    sensitivity = _number(point.get("supported_delta_sensitivity_max"))
    return all(
        (
            _common_point_pass(point),
            *(_strict_bool(point.get(field)) for field in ELECTROSTATIC_POINT_FIELDS),
            math.isfinite(sensitivity)
            and 0.0 <= sensitivity <= DELTA_SENSITIVITY_LIMIT,
        )
    )


def _comparison(
    indexed: Mapping[
        tuple[str, str, str, tuple[int, str], str], Mapping[str, Any]
    ],
    *,
    comparison_type: str,
    state: tuple[int, str],
    quantity: str,
) -> tuple[dict[str, Any], str]:
    if comparison_type == "mesh":
        reference_mesh, candidate_mesh = FINAL_PAIR
        rtol = (
            ION_MESH_RTOL if quantity == "ID_A"
            else CHARGE_MESH_RTOL if quantity in Q_QUANTITIES
            else CAPACITANCE_MESH_RTOL
        )
    else:
        reference_mesh, candidate_mesh = ULTRA_FINE, f"{ULTRA_FINE}_repeat"
        rtol = REPEAT_RTOL
    if quantity == "ID_A":
        floor, atol, units = 1.0e-30, ID_ATOL_A, "A"
    elif quantity in Q_QUANTITIES:
        floor, atol, units = CHARGE_FLOOR_C, CHARGE_ATOL_C, "C"
    elif quantity in C_QUANTITIES:
        floor, atol, units = CAPACITANCE_FLOOR_F, CAPACITANCE_ATOL_F, "F"
    else:
        raise ValueError(f"unsupported acceptance quantity {quantity!r}")
    key = comparison_type, reference_mesh, candidate_mesh, state, quantity
    row = indexed.get(key, {})
    result = combined_tolerance(
        row.get("reference_value"), row.get("candidate_value"),
        floor=floor, atol=atol, rtol=rtol,
    )
    return result, units


def _quantity_value(point: Mapping[str, Any], quantity: str) -> float:
    return _number(point.get(quantity))


def build_graded_convergence_rows(
    point_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build final-pair and repeat comparisons using graded mesh names.

    State-4/off ID is retained as a diagnostic row.  If both endpoints are
    below the current quantification floor it is explicitly N/A and cannot be
    mistaken for a transport or Q/C acceptance criterion.
    """

    annotated = [annotate_reference_point(row) for row in point_rows]
    indexed = _point_index(annotated)
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("mesh", left, right, "primary")
        for left, right in zip(GRADED_LEVELS, GRADED_LEVELS[1:])
    ]
    comparisons.append(("repeat", ULTRA_FINE, ULTRA_FINE, "repeat"))
    for comparison_type, left_level, right_level, right_role in comparisons:
        final_pair = (
            comparison_type == "repeat"
            or (left_level, right_level) == FINAL_PAIR
        )
        for state in REFERENCE_STATES:
            left = indexed.get((left_level, state, "primary"), {})
            right = indexed.get((right_level, state, right_role), {})
            for quantity in ("ID_A", *Q_QUANTITIES, *C_QUANTITIES):
                if quantity == "ID_A":
                    floor, atol, mesh_rtol, units = (
                        1.0e-30, ID_ATOL_A, ION_MESH_RTOL, "A"
                    )
                    domain = (
                        "dc_transport"
                        if final_pair and state == STATE0_ON
                        else "diagnostic"
                    )
                    gate_applicable = final_pair and state == STATE0_ON
                elif quantity in Q_QUANTITIES:
                    floor, atol, mesh_rtol, units = (
                        CHARGE_FLOOR_C, CHARGE_ATOL_C, CHARGE_MESH_RTOL, "C"
                    )
                    domain, gate_applicable = (
                        ("electrostatic_qc", True)
                        if final_pair else ("diagnostic", False)
                    )
                else:
                    floor, atol, mesh_rtol, units = (
                        CAPACITANCE_FLOOR_F, CAPACITANCE_ATOL_F,
                        CAPACITANCE_MESH_RTOL, "F",
                    )
                    domain, gate_applicable = (
                        ("electrostatic_qc", True)
                        if final_pair else ("diagnostic", False)
                    )
                comparison = combined_tolerance(
                    _quantity_value(left, quantity),
                    _quantity_value(right, quantity),
                    floor=floor, atol=atol,
                    rtol=REPEAT_RTOL if comparison_type == "repeat" else mesh_rtol,
                )
                below_floor = bool(
                    quantity == "ID_A" and state == STATE4_OFF
                    and left.get("dc_current_status") == CURRENT_STATUS_BELOW_FLOOR
                    and right.get("dc_current_status") == CURRENT_STATUS_BELOW_FLOOR
                )
                status = (
                    "not_applicable_below_numerical_floor" if below_floor
                    else "passed" if comparison["passed"] else "failed"
                )
                rows.append(
                    {
                        "row_type": "graded_convergence",
                        "comparison_type": comparison_type,
                        "reference_mesh": left_level,
                        "candidate_mesh": (
                            right_level if comparison_type == "mesh"
                            else f"{right_level}_repeat"
                        ),
                        "state_index": state[0],
                        "bias_name": state[1],
                        "quantity": quantity,
                        "acceptance_domain": domain,
                        "gate_applicable": gate_applicable,
                        "comparison_status": status,
                        "units": units,
                        **comparison,
                    }
                )
    return rows


def evaluate_split_acceptance(
    point_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    quality_by_level: Mapping[str, Any],
    *,
    profile_coverage_passed: bool = True,
) -> dict[str, Any]:
    """Evaluate independent DC, electrostatic-Q/C, and combined gates."""

    point_index = _point_index(point_rows)
    comparison_index = _comparison_index(convergence_rows)
    classified = [classify_current_point(point) for point in point_rows]
    classified_index = _point_index(classified)
    criteria: list[dict[str, Any]] = []

    quality_passed = all(
        _quality_for_level(quality_by_level, level) for level in GRADED_LEVELS
    )

    dc_results: list[bool] = []
    dc_results.append(_criterion(
        criteria, domain="dc_transport", name="final_mesh_quality",
        scope="all_graded_mesh_levels", passed=quality_passed,
        reason="all four graded family-quality summaries must pass",
    ))
    on_points: list[tuple[str, Mapping[str, Any]]] = []
    for level, role in ((EXTRA_FINE, "primary"), (ULTRA_FINE, "primary"),
                        (ULTRA_FINE, "repeat")):
        point = classified_index.get((level, STATE0_ON, role), {})
        scope = f"{level}/{role}/State_0_Empty/on"
        common = bool(point) and _common_point_pass(point)
        dc_results.append(_criterion(
            criteria, domain="dc_transport", name="common_reference_state",
            scope=scope, passed=common,
        ))
        quantified = point.get("dc_current_status") == CURRENT_STATUS_QUANTIFIED
        dc_results.append(_criterion(
            criteria, domain="dc_transport", name="on_current_quantified",
            scope=scope, passed=quantified,
            status=str(point.get("dc_current_status", CURRENT_STATUS_NONFINITE)),
            observed_value=point.get("dc_current_scale_A", math.nan),
            limit_value=DC_CURRENT_QUANTIFICATION_FLOOR_A, units="A",
        ))
        continuity = bool(quantified and point.get("raw_continuity_passed"))
        dc_results.append(_criterion(
            criteria, domain="dc_transport", name="on_current_continuity",
            scope=scope, passed=continuity,
            status=str(point.get("dc_continuity_acceptance_status", "failed")),
            observed_value=abs(_number(point.get("raw_continuity_residual_A"))),
            limit_value=point.get("raw_continuity_tolerance_A", math.nan), units="A",
        ))
        on_points.append((scope, point))

    for comparison_type, name in (
        ("mesh", "state0_on_Ion_extra_to_ultra"),
        ("repeat", "state0_on_Ion_ultra_repeat"),
    ):
        result, units = _comparison(
            comparison_index, comparison_type=comparison_type,
            state=STATE0_ON, quantity="ID_A",
        )
        dc_results.append(_criterion(
            criteria, domain="dc_transport", name=name,
            scope=comparison_type, passed=result["passed"],
            observed_value=result["relative_difference"],
            limit_value=(ION_MESH_RTOL if comparison_type == "mesh" else REPEAT_RTOL),
            units="relative",
            reason=(
                f"combined tolerance allowed_difference={result['allowed_difference']:.17g} {units}"
                if math.isfinite(result["allowed_difference"]) else "missing/nonfinite comparison"
            ),
        ))

    off_statuses: list[str] = []
    for level, role in ((EXTRA_FINE, "primary"), (ULTRA_FINE, "primary"),
                        (ULTRA_FINE, "repeat")):
        point = classified_index.get((level, STATE4_OFF, role), {})
        status = str(point.get("dc_current_status", CURRENT_STATUS_NONFINITE))
        off_statuses.append(status)
        _criterion(
            criteria, domain="diagnostic", name="state4_off_current_quantification",
            scope=f"{level}/{role}/State_4_Programmed/off",
            passed=status != CURRENT_STATUS_NONFINITE,
            status=status, gate_applicable=False,
            observed_value=point.get("dc_current_scale_A", math.nan),
            limit_value=DC_CURRENT_QUANTIFICATION_FLOOR_A, units="A",
            reason="off-current sign and relative continuity do not gate electrostatic Q/C",
        )

    electro_results: list[bool] = []
    electro_results.append(_criterion(
        criteria, domain="electrostatic_qc", name="final_mesh_quality",
        scope="all_graded_mesh_levels", passed=quality_passed,
    ))
    for level, role in ((EXTRA_FINE, "primary"), (ULTRA_FINE, "primary"),
                        (ULTRA_FINE, "repeat")):
        for state in REFERENCE_STATES:
            point = point_index.get((level, state, role), {})
            scope = f"{level}/{role}/state{state[0]}/{state[1]}"
            common = bool(point) and _common_point_pass(point)
            electro_results.append(_criterion(
                criteria, domain="electrostatic_qc",
                name="common_reference_state", scope=scope, passed=common,
            ))
            electro = bool(point) and _electrostatic_point_pass(point)
            sensitivity = _number(point.get("supported_delta_sensitivity_max"))
            electro_results.append(_criterion(
                criteria, domain="electrostatic_qc",
                name="gauss_derivative_gauge_delta", scope=scope,
                passed=electro,
                observed_value=sensitivity,
                limit_value=DELTA_SENSITIVITY_LIMIT, units="relative",
                reason="continuity and signed ID are intentionally excluded",
            ))

    for comparison_type in ("mesh", "repeat"):
        for state in REFERENCE_STATES:
            for quantity in (*Q_QUANTITIES, *C_QUANTITIES):
                result, units = _comparison(
                    comparison_index, comparison_type=comparison_type,
                    state=state, quantity=quantity,
                )
                category = "Q" if quantity in Q_QUANTITIES else "supported_C"
                name = f"{category}_{comparison_type}_{quantity}"
                scope = f"state{state[0]}/{state[1]}"
                electro_results.append(_criterion(
                    criteria, domain="electrostatic_qc", name=name,
                    scope=scope, passed=result["passed"],
                    observed_value=result["relative_difference"],
                    limit_value=(
                        REPEAT_RTOL if comparison_type == "repeat"
                        else CHARGE_MESH_RTOL if quantity in Q_QUANTITIES
                        else CAPACITANCE_MESH_RTOL
                    ),
                    units="relative",
                    reason=(
                        f"combined tolerance allowed_difference={result['allowed_difference']:.17g} {units}"
                        if math.isfinite(result["allowed_difference"])
                        else "missing/nonfinite comparison"
                    ),
                ))

    dc_passed = all(dc_results)
    electro_passed = all(electro_results)
    # Contact-normal profile coverage was part of an earlier diagnostic, but it
    # is not an acceptance item for this graded-mesh task.  Keep the input for
    # API compatibility without letting it authorize or block the full sweep.
    combined_passed = bool(dc_passed and electro_passed)
    all_off_below = bool(off_statuses) and all(
        status == CURRENT_STATUS_BELOW_FLOOR for status in off_statuses
    )
    dc_status = (
        "dc_transport_reference_passed_with_ioff_below_numerical_floor"
        if dc_passed and all_off_below
        else "dc_transport_reference_passed" if dc_passed
        else "dc_transport_reference_failed"
    )
    electro_status = (
        "electrostatic_qc_reference_passed"
        if electro_passed else "electrostatic_qc_reference_failed"
    )
    _criterion(
        criteria, domain="combined_reference", name="dc_transport_gate",
        scope="summary", passed=dc_passed, status=dc_status,
    )
    _criterion(
        criteria, domain="combined_reference", name="electrostatic_qc_gate",
        scope="summary", passed=electro_passed, status=electro_status,
    )
    _criterion(
        criteria, domain="combined_reference",
        name="profile_coverage_not_in_scope", scope="summary", passed=True,
        status="not_requested", gate_applicable=False,
        reason="contact-normal profiles are not an acceptance item in this task",
    )
    _criterion(
        criteria, domain="combined_reference", name="conditional_full_sweep_gate",
        scope="summary", passed=combined_passed,
        status="authorized" if combined_passed else "not_authorized",
    )
    physical_status = (
        PHYSICAL_STATUS_CONVERGED if combined_passed
        else PHYSICAL_STATUS_NOT_CONVERGED
    )
    _criterion(
        criteria, domain="combined_reference", name="physical_contact_flux_status",
        scope="summary", passed=combined_passed, status=physical_status,
    )

    return {
        "dc_transport": {"passed": dc_passed, "status": dc_status},
        "electrostatic_qc": {
            "passed": electro_passed, "status": electro_status,
        },
        "combined_reference": {
            "passed": combined_passed,
            "full_sweep_allowed": combined_passed,
            "physical_contact_flux_status": physical_status,
        },
        "classified_points": classified,
        "criterion_rows": criteria,
    }


def evaluate_reference_gates(
    point_rows: Sequence[Mapping[str, Any]],
    convergence_rows: Sequence[Mapping[str, Any]],
    quality_by_level: Mapping[str, Any],
    profile_passed: bool = True,
) -> dict[str, Any]:
    """Stable runner-facing alias for :func:`evaluate_split_acceptance`."""

    return evaluate_split_acceptance(
        point_rows,
        convergence_rows,
        quality_by_level,
        profile_coverage_passed=profile_passed,
    )


def criterion_rows(gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return defensive copies of the flat, CSV-ready criterion rows."""

    rows = gate.get("criterion_rows", ())
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError("gate criterion_rows must be a sequence")
    return [dict(row) for row in rows]


__all__ = [
    "CAPACITANCE_ATOL_F",
    "CAPACITANCE_FLOOR_F",
    "CAPACITANCE_MESH_RTOL",
    "CHARGE_ATOL_C",
    "CHARGE_FLOOR_C",
    "CHARGE_MESH_RTOL",
    "CURRENT_STATUS_BELOW_FLOOR",
    "CURRENT_STATUS_NONFINITE",
    "CURRENT_STATUS_QUANTIFIED",
    "DC_CURRENT_QUANTIFICATION_FLOOR_A",
    "DELTA_SENSITIVITY_LIMIT",
    "ID_ATOL_A",
    "ION_MESH_RTOL",
    "PHYSICAL_STATUS_CONVERGED",
    "PHYSICAL_STATUS_NOT_CONVERGED",
    "REPEAT_RTOL",
    "GRADED_LEVELS",
    "annotate_reference_point",
    "build_graded_convergence_rows",
    "classify_current",
    "classify_current_point",
    "combined_tolerance",
    "criterion_rows",
    "evaluate_reference_gates",
    "evaluate_split_acceptance",
    "quality_mapping_from_rows",
]
