"""Pure-Python convergence policy for the local contact-mesh ladder.

This module deliberately contains no DEVSIM imports.  It turns flattened
reference-point records into auditable, long-form comparisons and evaluates
only the numerical gates requested for the local-mesh candidate.  In
particular, contact-flux column sums and Ward--Dutton quantities are not
promotion criteria.

The expected reference points are ``State_0_Empty/on`` and
``State_4_Programmed/off``.  A point record must identify ``mesh_level``,
``state_index`` and ``bias_name``.  Primary records may omit ``run_role``;
ultra-fine repeats use ``run_role='repeat'``.  Quantities may be flattened or
kept in the existing ``direct`` and ``nominal_matrix_F`` mappings.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence


MESH_LEVELS = (
    "local_base",
    "local_fine",
    "local_extra_fine",
    "local_ultra_fine",
)
MESH_PAIRS = tuple(zip(MESH_LEVELS[:-1], MESH_LEVELS[1:]))
FINAL_MESH_PAIR = ("local_extra_fine", "local_ultra_fine")

REFERENCE_STATES = ((0, "on"), (4, "off"))
STATE0_ON = (0, "on")

ID_FLOOR_A = 1.0e-30
ID_ATOL_A = 1.0e-24
MESH_ID_RTOL = 2.0e-2

CHARGE_FLOOR_C = 1.0e-30
CHARGE_ATOL_C = 1.0e-28
MESH_CHARGE_RTOL = 2.0e-2

SUPPORTED_CAPACITANCE_FLOOR_F = 1.0e-24
SUPPORTED_CAPACITANCE_ATOL_F = 1.0e-22
MESH_SUPPORTED_CAPACITANCE_RTOL = 5.0e-2

REPEAT_RTOL = 1.0e-4
DELTA_SENSITIVITY_LIMIT = 1.0e-2
BIAS_ATOL_V = 1.0e-12

STATUS_CONVERGED = "numerically_converged_provisional"
STATUS_NOT_CONVERGED = "provisional_not_mesh_converged"


@dataclass(frozen=True)
class QuantitySpec:
    """Tolerance and acceptance scope for one compared quantity."""

    name: str
    category: str
    units: str
    floor: float
    atol: float
    mesh_rtol: float
    final_gate_states: tuple[tuple[int, str], ...]


QUANTITY_SPECS = (
    QuantitySpec(
        "ID_A", "drain_current", "A", ID_FLOOR_A, ID_ATOL_A,
        MESH_ID_RTOL, (STATE0_ON,),
    ),
    QuantitySpec(
        "Qg_C", "physical_contact_charge", "C", CHARGE_FLOOR_C,
        CHARGE_ATOL_C, MESH_CHARGE_RTOL, REFERENCE_STATES,
    ),
    QuantitySpec(
        "raw_Qd_C", "physical_contact_charge", "C", CHARGE_FLOOR_C,
        CHARGE_ATOL_C, MESH_CHARGE_RTOL, REFERENCE_STATES,
    ),
    QuantitySpec(
        "raw_Qs_C", "physical_contact_charge", "C", CHARGE_FLOOR_C,
        CHARGE_ATOL_C, MESH_CHARGE_RTOL, REFERENCE_STATES,
    ),
    QuantitySpec(
        "Cgg_F", "supported_capacitance", "F",
        SUPPORTED_CAPACITANCE_FLOOR_F, SUPPORTED_CAPACITANCE_ATOL_F,
        MESH_SUPPORTED_CAPACITANCE_RTOL, REFERENCE_STATES,
    ),
    QuantitySpec(
        "Cgd_F", "supported_capacitance", "F",
        SUPPORTED_CAPACITANCE_FLOOR_F, SUPPORTED_CAPACITANCE_ATOL_F,
        MESH_SUPPORTED_CAPACITANCE_RTOL, REFERENCE_STATES,
    ),
    QuantitySpec(
        "Cgs_F", "supported_capacitance", "F",
        SUPPORTED_CAPACITANCE_FLOOR_F, SUPPORTED_CAPACITANCE_ATOL_F,
        MESH_SUPPORTED_CAPACITANCE_RTOL, REFERENCE_STATES,
    ),
)
QUANTITY_SPEC_BY_NAME = {spec.name: spec for spec in QUANTITY_SPECS}


_ALIASES = {
    "ID_A": ("ID_A", "Id_A", "drain_current_A", "I_D_A"),
    "Qg_C": ("Qg_C", "Qg_contact_C", "Qg_raw_C"),
    "raw_Qd_C": ("raw_Qd_C", "Qd_contact_C", "Qd_C"),
    "raw_Qs_C": ("raw_Qs_C", "Qs_contact_C", "Qs_C"),
    "Cgg_F": ("Cgg_F",),
    "Cgd_F": ("Cgd_F",),
    "Cgs_F": ("Cgs_F",),
}
_MATRIX_KEYS = {
    "Cgg_F": "gate:gate",
    "Cgd_F": "gate:drain",
    "Cgs_F": "gate:source",
}


def _finite_float(value: Any) -> float:
    """Convert a value to float; return NaN for absent/non-finite input."""

    try:
        converted = float(value)
    except (TypeError, ValueError):
        return math.nan
    return converted if math.isfinite(converted) else math.nan


def combined_tolerance_comparison(
    reference: Any,
    candidate: Any,
    *,
    floor: float,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare using ``abs(diff) <= atol + rtol * max(abs(a), abs(b), floor)``.

    Non-finite values always fail.  ``relative_difference`` is a diagnostic;
    acceptance is based on the combined tolerance so that near-zero values
    are handled explicitly and reproducibly.
    """

    for name, value, positive in (
        ("floor", floor, True),
        ("atol", atol, False),
        ("rtol", rtol, False),
    ):
        numeric = _finite_float(value)
        if not math.isfinite(numeric) or numeric < 0.0 or (
            positive and numeric <= 0.0
        ):
            qualifier = "positive" if positive else "non-negative"
            raise ValueError(f"{name} must be finite and {qualifier}")

    left = _finite_float(reference)
    right = _finite_float(candidate)
    finite = math.isfinite(left) and math.isfinite(right)
    if finite:
        difference = abs(right - left)
        scale = max(abs(left), abs(right), float(floor))
        tolerance = float(atol) + float(rtol) * scale
        relative_difference = difference / scale
        passed = difference <= tolerance
    else:
        difference = math.nan
        scale = math.nan
        tolerance = math.nan
        relative_difference = math.nan
        passed = False
    return {
        "reference_value": left,
        "candidate_value": right,
        "absolute_difference": difference,
        "comparison_scale": scale,
        "relative_difference": relative_difference,
        "floor": float(floor),
        "absolute_tolerance": float(atol),
        "relative_tolerance": float(rtol),
        "allowed_difference": tolerance,
        "near_zero_floor_used": bool(finite and scale == float(floor)),
        "finite": finite,
        "passed": passed,
    }


# Short, convenient alias for callers and tests.
compare_values = combined_tolerance_comparison


def diagnose_monotonicity(
    values: Sequence[Any], *, floor: float
) -> dict[str, Any]:
    """Describe value monotonicity and shrinking successive differences.

    This result is diagnostic only.  Neither property is used by
    :func:`evaluate_final_gate`.
    """

    numeric = tuple(_finite_float(value) for value in values)
    if len(numeric) < 2 or any(not math.isfinite(value) for value in numeric):
        return {
            "value_monotonicity": "indeterminate",
            "successive_difference_nonincreasing": False,
            "successive_differences": (),
        }
    if not math.isfinite(float(floor)) or floor <= 0.0:
        raise ValueError("floor must be finite and positive")
    deltas = tuple(right - left for left, right in zip(numeric[:-1], numeric[1:]))
    epsilon = float(floor)
    nondecreasing = all(delta >= -epsilon for delta in deltas)
    nonincreasing = all(delta <= epsilon for delta in deltas)
    if nondecreasing and nonincreasing:
        direction = "constant_within_floor"
    elif nondecreasing:
        direction = "monotonic_increasing"
    elif nonincreasing:
        direction = "monotonic_decreasing"
    else:
        direction = "non_monotonic"
    differences = tuple(abs(delta) for delta in deltas)
    shrinking = all(
        later <= earlier + epsilon
        for earlier, later in zip(differences[:-1], differences[1:])
    )
    return {
        "value_monotonicity": direction,
        "successive_difference_nonincreasing": shrinking,
        "successive_differences": differences,
    }


def _state_key(point: Mapping[str, Any]) -> tuple[int, str]:
    try:
        state_index = int(point["state_index"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("every point must contain an integer state_index") from error
    bias_name = str(point.get("bias_name", "")).strip().lower()
    if not bias_name:
        raise ValueError("every point must contain bias_name")
    return state_index, bias_name


def _flatten_points(
    points: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    forced_role: str | None = None,
) -> list[dict[str, Any]]:
    """Accept flat rows or ``mesh_level -> rows/points`` bundle mappings."""

    flattened: list[dict[str, Any]] = []
    if isinstance(points, Mapping):
        if "mesh_level" in points and "state_index" in points:
            candidates = [points]
        else:
            candidates = []
            for raw_level, payload in points.items():
                level = str(raw_level)
                role = forced_role
                if level.endswith("_repeat"):
                    level = level[:-7]
                    role = "repeat"
                if isinstance(payload, Mapping) and "points" in payload:
                    payload = payload["points"]
                if isinstance(payload, Mapping):
                    payload = payload.values()
                for point in payload:
                    row = dict(point)
                    row.setdefault("mesh_level", level)
                    if role is not None:
                        row["run_role"] = role
                    candidates.append(row)
    else:
        candidates = list(points)
    for point in candidates:
        row = dict(point)
        if forced_role is not None:
            row["run_role"] = forced_role
        flattened.append(row)
    return flattened


def _extract_quantity(point: Mapping[str, Any], name: str) -> float:
    for alias in _ALIASES[name]:
        if alias in point:
            return _finite_float(point[alias])
    direct = point.get("direct", {})
    if isinstance(direct, Mapping):
        for alias in _ALIASES[name]:
            if alias in direct:
                return _finite_float(direct[alias])
    matrix_key = _MATRIX_KEYS.get(name)
    if matrix_key is not None:
        for container_name in ("nominal_matrix_F", "capacitances_F"):
            container = point.get(container_name, {})
            if isinstance(container, Mapping):
                if name in container:
                    return _finite_float(container[name])
                if matrix_key in container:
                    return _finite_float(container[matrix_key])
        matrices = point.get("matrices", {})
        if isinstance(matrices, Mapping):
            for matrix in matrices.values():
                if isinstance(matrix, Mapping):
                    values = matrix.get("values", matrix)
                    if isinstance(values, Mapping) and matrix_key in values:
                        return _finite_float(values[matrix_key])
    return math.nan


def _point_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, tuple[int, str], str], Mapping[str, Any]]:
    index: dict[tuple[str, tuple[int, str], str], Mapping[str, Any]] = {}
    for point in rows:
        level = str(point.get("mesh_level", ""))
        role = str(point.get("run_role", "primary")).strip().lower() or "primary"
        key = level, _state_key(point), role
        if key in index:
            raise ValueError(f"duplicate reference point {key!r}")
        index[key] = point
    return index


def build_convergence_rows(
    points: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    repeat_points: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    required_states: Sequence[tuple[int, str]] = REFERENCE_STATES,
) -> list[dict[str, Any]]:
    """Build long-form rows for three adjacent pairs and the ultra repeat."""

    flattened = _flatten_points(points)
    if repeat_points is not None:
        flattened.extend(_flatten_points(repeat_points, forced_role="repeat"))
    index = _point_index(flattened)
    rows: list[dict[str, Any]] = []
    normalized_states = tuple(
        (int(state_index), str(bias).lower())
        for state_index, bias in required_states
    )
    monotonic: dict[tuple[tuple[int, str], str], dict[str, Any]] = {}
    for state_key in normalized_states:
        for spec in QUANTITY_SPECS:
            sequence = [
                _extract_quantity(index.get((level, state_key, "primary"), {}), spec.name)
                for level in MESH_LEVELS
            ]
            monotonic[(state_key, spec.name)] = diagnose_monotonicity(
                sequence, floor=spec.floor
            )

    pair_definitions = [
        ("mesh", left, right, "primary") for left, right in MESH_PAIRS
    ] + [("repeat", "local_ultra_fine", "local_ultra_fine", "repeat")]
    for comparison_type, left_level, right_level, right_role in pair_definitions:
        for state_key in normalized_states:
            left = index.get((left_level, state_key, "primary"), {})
            right = index.get((right_level, state_key, right_role), {})
            for spec in QUANTITY_SPECS:
                rtol = REPEAT_RTOL if comparison_type == "repeat" else spec.mesh_rtol
                comparison = combined_tolerance_comparison(
                    _extract_quantity(left, spec.name),
                    _extract_quantity(right, spec.name),
                    floor=spec.floor,
                    atol=spec.atol,
                    rtol=rtol,
                )
                diagnostic = monotonic[(state_key, spec.name)]
                state_index, bias_name = state_key
                gate_applies = (
                    state_key in spec.final_gate_states
                    and (
                        comparison_type == "repeat"
                        or (left_level, right_level) == FINAL_MESH_PAIR
                    )
                )
                rows.append(
                    {
                        "comparison_type": comparison_type,
                        "pair": (
                            f"{left_level}->{right_level}"
                            if comparison_type == "mesh"
                            else "local_ultra_fine->local_ultra_fine_repeat"
                        ),
                        "reference_mesh": left_level,
                        "candidate_mesh": (
                            right_level if comparison_type == "mesh"
                            else "local_ultra_fine_repeat"
                        ),
                        "state_index": state_index,
                        "state": str(left.get("state", right.get("state", ""))),
                        "bias_name": bias_name,
                        "quantity": spec.name,
                        "category": spec.category,
                        "units": spec.units,
                        **comparison,
                        "value_monotonicity": diagnostic["value_monotonicity"],
                        "successive_difference_nonincreasing": diagnostic[
                            "successive_difference_nonincreasing"
                        ],
                        "acceptance_gate_applies": gate_applies,
                    }
                )
    return rows


def _explicit_bool(point: Mapping[str, Any], *paths: tuple[str, ...]) -> bool:
    for path in paths:
        value: Any = point
        for component in path:
            if not isinstance(value, Mapping) or component not in value:
                break
            value = value[component]
        else:
            return bool(value)
    return False


def bias_target_reached(point: Mapping[str, Any]) -> bool:
    """Validate explicit target/actual terminal biases to ``BIAS_ATOL_V``."""

    for key in ("target_reached", "target_bias_reached", "bias_target_passed"):
        if key in point:
            return bool(point[key])
    aliases = {
        "VGS_V": ("actual_VGS_V", "reached_VGS_V", "VGS_V"),
        "VDS_V": ("actual_VDS_V", "reached_VDS_V", "VDS_V"),
        "VS_V": ("actual_VS_V", "reached_VS_V", "VS_V"),
    }
    checked = 0
    for name, actual_aliases in aliases.items():
        target_key = f"target_{name}"
        if target_key not in point:
            continue
        actual = next(
            (_finite_float(point[key]) for key in actual_aliases if key in point),
            math.nan,
        )
        target = _finite_float(point[target_key])
        if not (math.isfinite(actual) and math.isfinite(target)):
            return False
        if abs(actual - target) > BIAS_ATOL_V:
            return False
        checked += 1
    return checked == 3


def supported_delta_sensitivity(point: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate delta sensitivity for Cgg/Cgd/Cgs only."""

    if "supported_delta_sensitivity_passed" in point:
        maximum = math.nan
        for key in (
            "max_supported_C_delta_sensitivity",
            "maximum_supported_capacitance_delta_sensitivity",
            "supported_delta_sensitivity_max",
        ):
            if key in point:
                maximum = _finite_float(point[key])
                break
        return {
            "maximum": maximum,
            "limit": DELTA_SENSITIVITY_LIMIT,
            "passed": bool(point["supported_delta_sensitivity_passed"]),
        }
    for key in (
        "max_supported_C_delta_sensitivity",
        "maximum_supported_capacitance_delta_sensitivity",
        "supported_delta_sensitivity_max",
    ):
        if key in point:
            maximum = _finite_float(point[key])
            return {
                "maximum": maximum,
                "limit": DELTA_SENSITIVITY_LIMIT,
                "passed": math.isfinite(maximum)
                and maximum <= DELTA_SENSITIVITY_LIMIT,
            }
    sensitivity = point.get("sensitivities", {})
    values: list[float] = []
    if isinstance(sensitivity, Mapping):
        for matrix_key in ("gate:gate", "gate:drain", "gate:source"):
            entry = sensitivity.get(matrix_key, {})
            if isinstance(entry, Mapping):
                value = entry.get("value", entry.get("relative_sensitivity"))
            else:
                value = entry
            values.append(_finite_float(value))
    maximum = max(values) if values and all(map(math.isfinite, values)) else math.nan
    return {
        "maximum": maximum,
        "limit": DELTA_SENSITIVITY_LIMIT,
        "passed": math.isfinite(maximum) and maximum <= DELTA_SENSITIVITY_LIMIT,
    }


def reference_point_validation(point: Mapping[str, Any]) -> dict[str, Any]:
    """Return the six required per-point criteria; no unrelated physics gate."""

    derivative_rows = point.get("derivative_gauss_rows", ())
    if "derivative_gauss_passed" in point:
        derivative_passed = bool(point["derivative_gauss_passed"])
    elif derivative_rows:
        derivative_passed = all(bool(row.get("passed")) for row in derivative_rows)
    else:
        derivative_passed = False
    sensitivity = supported_delta_sensitivity(point)
    criteria = {
        "target_bias_reached": bias_target_reached(point),
        "source_drain_continuity": _explicit_bool(
            point,
            ("continuity_passed",),
            ("continuity", "passed"),
        ),
        "global_gauss_balance": _explicit_bool(
            point,
            ("global_gauss_passed",),
            ("global_gauss", "passed"),
        ),
        "derivative_gauss_balance": derivative_passed,
        "gauge_invariance_row_sums": _explicit_bool(
            point,
            ("gauge_row_sum_passed",),
            ("gauge_row_validation", "passed"),
        ),
        "final_reference_state_restored": _explicit_bool(
            point,
            ("final_reference_validation_passed",),
            ("final_reference_validation", "passed"),
        ),
        "contact_topology": _explicit_bool(
            point,
            ("topology_passed",),
            ("topology", "passed"),
        ),
        "mesh_line_validation": _explicit_bool(
            point,
            ("mesh_line_validation_passed",),
            ("mesh_line_validation", "passed"),
        ),
        "runtime_invariants": _explicit_bool(
            point,
            ("runtime_invariants_passed",),
        ),
        "supported_capacitance_delta_sensitivity": bool(sensitivity["passed"]),
    }
    return {
        "criteria": criteria,
        "maximum_supported_capacitance_delta_sensitivity": sensitivity["maximum"],
        "passed": all(criteria.values()),
    }


def _quality_passed(value: Any) -> bool:
    if isinstance(value, Mapping):
        return bool(value.get("passed", value.get("quality_passed", False)))
    return bool(value)


def evaluate_final_gate(
    points: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    convergence_rows: Sequence[Mapping[str, Any]],
    quality_by_level: Mapping[str, Any],
    repeat_points: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    *,
    required_states: Sequence[tuple[int, str]] = REFERENCE_STATES,
) -> dict[str, Any]:
    """Evaluate the final extra-fine/ultra-fine promotion gate.

    The strongest possible result remains provisional: this numerical gate
    cannot establish a supported full terminal-charge/capacitance model.
    """

    flattened = _flatten_points(points)
    if repeat_points is not None:
        flattened.extend(_flatten_points(repeat_points, forced_role="repeat"))
    index = _point_index(flattened)
    states = tuple((int(index_), str(bias).lower()) for index_, bias in required_states)

    primary_point_results = {
        f"{state_index}:{bias}": reference_point_validation(
            index.get(("local_ultra_fine", (state_index, bias), "primary"), {})
        )
        for state_index, bias in states
    }
    repeat_point_results = {
        f"{state_index}:{bias}": reference_point_validation(
            index.get(("local_ultra_fine", (state_index, bias), "repeat"), {})
        )
        for state_index, bias in states
    }

    final_rows = [
        row for row in convergence_rows
        if row.get("comparison_type") == "mesh"
        and row.get("reference_mesh") == FINAL_MESH_PAIR[0]
        and row.get("candidate_mesh") == FINAL_MESH_PAIR[1]
    ]
    repeat_rows = [
        row for row in convergence_rows
        if row.get("comparison_type") == "repeat"
    ]

    def rows_pass(quantity_names: set[str], state_keys: set[tuple[int, str]]) -> bool:
        selected = [
            row for row in final_rows
            if row.get("quantity") in quantity_names
            and (int(row.get("state_index", -1)), str(row.get("bias_name", "")))
            in state_keys
        ]
        expected = len(quantity_names) * len(state_keys)
        return len(selected) == expected and all(bool(row.get("passed")) for row in selected)

    repeat_expected = len(QUANTITY_SPECS) * len(states)
    repeat_numeric_pass = (
        len(repeat_rows) == repeat_expected
        and all(bool(row.get("passed")) for row in repeat_rows)
    )
    criteria = {
        "both_ultra_reference_points_valid": all(
            result["passed"] for result in primary_point_results.values()
        ),
        "state0_on_id_extra_to_ultra": rows_pass({"ID_A"}, {STATE0_ON}),
        "both_states_Q_extra_to_ultra": rows_pass(
            {"Qg_C", "raw_Qd_C", "raw_Qs_C"}, set(states)
        ),
        "both_states_supported_C_extra_to_ultra": rows_pass(
            {"Cgg_F", "Cgd_F", "Cgs_F"}, set(states)
        ),
        "ultra_repeat_reproducibility": repeat_numeric_pass
        and all(result["passed"] for result in repeat_point_results.values()),
        "extra_and_ultra_mesh_quality": all(
            _quality_passed(quality_by_level.get(level, False))
            for level in FINAL_MESH_PAIR
        ),
    }
    passed = all(criteria.values())
    return {
        "status": STATUS_CONVERGED if passed else STATUS_NOT_CONVERGED,
        "full_sweep_allowed": passed,
        "passed": passed,
        "criteria": criteria,
        "primary_point_validation": primary_point_results,
        "repeat_point_validation": repeat_point_results,
        "notes": (
            "Monotonicity is diagnostic only; contact-flux column sums and "
            "Ward-Dutton results are intentionally excluded from this gate."
        ),
    }


__all__ = [
    "BIAS_ATOL_V",
    "CHARGE_ATOL_C",
    "CHARGE_FLOOR_C",
    "DELTA_SENSITIVITY_LIMIT",
    "FINAL_MESH_PAIR",
    "ID_ATOL_A",
    "ID_FLOOR_A",
    "MESH_CHARGE_RTOL",
    "MESH_ID_RTOL",
    "MESH_LEVELS",
    "MESH_PAIRS",
    "MESH_SUPPORTED_CAPACITANCE_RTOL",
    "QUANTITY_SPECS",
    "REFERENCE_STATES",
    "REPEAT_RTOL",
    "STATUS_CONVERGED",
    "STATUS_NOT_CONVERGED",
    "SUPPORTED_CAPACITANCE_ATOL_F",
    "SUPPORTED_CAPACITANCE_FLOOR_F",
    "bias_target_reached",
    "build_convergence_rows",
    "combined_tolerance_comparison",
    "compare_values",
    "diagnose_monotonicity",
    "evaluate_final_gate",
    "reference_point_validation",
    "supported_delta_sensitivity",
]
