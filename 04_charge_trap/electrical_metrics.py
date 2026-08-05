"""DEVSIM-independent extraction of static transistor read metrics.

All public functions accept either parallel ``VGS``/``ID`` sequences or, when
``id_values`` is omitted, a sequence of mappings containing ``VGS_V`` and
``ID_A``.  Current magnitudes are used for Vth, SS, Ion, and Ioff; signed
current is retained for transconductance.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


NAN = math.nan
_VOLTAGE_TOLERANCE_V = 1.0e-12


def _failure(error_key: str, message: str, **values: Any) -> dict[str, Any]:
    """Return a compact failure result with a consistent error field."""

    result = dict(values)
    result[error_key] = message
    return result


def _row_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Return the first available named value from a row."""

    for name in names:
        if name in row:
            return row[name]
    raise KeyError(f"row is missing all supported columns: {tuple(names)}")


def _is_converged(row: Mapping[str, Any]) -> bool:
    """Interpret an optional in-memory or CSV convergence flag."""

    if "converged" not in row:
        return True

    value = row["converged"]
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _prepare_points(
    vgs_values: Sequence[Any],
    id_values: Sequence[Any] | None,
) -> tuple[list[tuple[float, float]], str]:
    """Coerce, filter, sort, and duplicate-check an IV data series."""

    raw_points: list[tuple[Any, Any]] = []

    if id_values is None:
        for raw_row in vgs_values:
            if not isinstance(raw_row, Mapping):
                return [], (
                    "id_values is required when vgs_values is not a row list."
                )
            if not _is_converged(raw_row):
                continue
            try:
                raw_points.append(
                    (
                        _row_value(raw_row, ("VGS_V", "gate_voltage_V")),
                        _row_value(
                            raw_row,
                            ("ID_A", "drain_current_A", "electron_current_A"),
                        ),
                    )
                )
            except KeyError as error:
                return [], str(error)
    else:
        if len(vgs_values) != len(id_values):
            return [], "VGS and ID sequences must have the same length."
        raw_points.extend(zip(vgs_values, id_values))

    points: list[tuple[float, float]] = []
    for raw_vgs, raw_id in raw_points:
        try:
            vgs_V = float(raw_vgs)
            drain_current_A = float(raw_id)
        except (TypeError, ValueError):
            continue
        if math.isfinite(vgs_V) and math.isfinite(drain_current_A):
            points.append((vgs_V, drain_current_A))

    points.sort(key=lambda point: point[0])

    for left, right in zip(points, points[1:]):
        if math.isclose(
            left[0],
            right[0],
            rel_tol=0.0,
            abs_tol=_VOLTAGE_TOLERANCE_V,
        ):
            return [], f"duplicate VGS value: {right[0]:.12g} V"

    return points, ""


def extract_threshold_voltage(
    vgs_values: Sequence[Any],
    id_values: Sequence[Any] | None = None,
    *,
    threshold_current_A: float,
    current_floor_A: float = 1.0e-30,
) -> dict[str, Any]:
    """Extract constant-current Vth using adjacent log-current interpolation."""

    base = {
        "Vth_V": NAN,
        "threshold_current_A": NAN,
        "vth_success": False,
        "vth_error": "",
    }

    try:
        threshold_current_A = float(threshold_current_A)
        current_floor_A = float(current_floor_A)
    except (TypeError, ValueError):
        return _failure("vth_error", "current criteria must be numeric", **base)

    base["threshold_current_A"] = threshold_current_A

    if (
        not math.isfinite(threshold_current_A)
        or threshold_current_A <= 0.0
        or not math.isfinite(current_floor_A)
        or current_floor_A <= 0.0
        or current_floor_A >= threshold_current_A
    ):
        return _failure(
            "vth_error",
            "criteria must satisfy 0 < current floor < threshold current",
            **base,
        )

    points, preparation_error = _prepare_points(vgs_values, id_values)
    if preparation_error:
        return _failure("vth_error", preparation_error, **base)
    if len(points) < 2:
        return _failure(
            "vth_error", "at least two finite IV points are required", **base
        )

    target_log_current = math.log10(
        max(threshold_current_A, current_floor_A)
    )

    for (vgs_1, signed_id_1), (vgs_2, signed_id_2) in zip(
        points, points[1:]
    ):
        current_1 = max(abs(signed_id_1), current_floor_A)
        current_2 = max(abs(signed_id_2), current_floor_A)

        if not (
            current_1 <= threshold_current_A <= current_2
            or current_2 <= threshold_current_A <= current_1
        ):
            continue

        log_current_1 = math.log10(current_1)
        log_current_2 = math.log10(current_2)

        if math.isclose(
            log_current_1,
            target_log_current,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            threshold_voltage_V = vgs_1
        elif math.isclose(
            log_current_2,
            target_log_current,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            threshold_voltage_V = vgs_2
        elif math.isclose(
            log_current_1,
            log_current_2,
            rel_tol=0.0,
            abs_tol=1.0e-30,
        ):
            threshold_voltage_V = 0.5 * (vgs_1 + vgs_2)
        else:
            fraction = (
                (target_log_current - log_current_1)
                / (log_current_2 - log_current_1)
            )
            threshold_voltage_V = vgs_1 + fraction * (vgs_2 - vgs_1)

        return {
            "Vth_V": threshold_voltage_V,
            "threshold_current_A": threshold_current_A,
            "vth_success": True,
            "vth_error": "",
        }

    return _failure(
        "vth_error",
        "threshold current is not crossed within the VGS range",
        **base,
    )


def extract_subthreshold_swing(
    vgs_values: Sequence[Any],
    id_values: Sequence[Any] | None = None,
    *,
    current_min_A: float,
    current_max_A: float,
    minimum_point_count: int = 4,
) -> dict[str, Any]:
    """Fit log10(abs(ID)) versus VGS and return SS in mV/decade."""

    base = {
        "SS_mV_dec": NAN,
        "ss_r_squared": NAN,
        "ss_point_count": 0,
        "ss_success": False,
        "ss_error": "",
    }

    try:
        current_min_A = float(current_min_A)
        current_max_A = float(current_max_A)
    except (TypeError, ValueError):
        return _failure("ss_error", "SS current bounds must be numeric", **base)

    if (
        not math.isfinite(current_min_A)
        or not math.isfinite(current_max_A)
        or not 0.0 < current_min_A < current_max_A
    ):
        return _failure(
            "ss_error",
            "SS current bounds must be finite, positive, and increasing",
            **base,
        )
    if (
        isinstance(minimum_point_count, bool)
        or not isinstance(minimum_point_count, int)
        or minimum_point_count < 2
    ):
        return _failure(
            "ss_error", "minimum_point_count must be an integer of at least 2", **base
        )

    points, preparation_error = _prepare_points(vgs_values, id_values)
    if preparation_error:
        return _failure("ss_error", preparation_error, **base)

    fitting_points = [
        (vgs_V, math.log10(abs(drain_current_A)))
        for vgs_V, drain_current_A in points
        if drain_current_A != 0.0
        and current_min_A <= abs(drain_current_A) <= current_max_A
    ]
    point_count = len(fitting_points)
    base["ss_point_count"] = point_count

    if point_count < minimum_point_count:
        return _failure(
            "ss_error",
            f"SS fit requires at least {minimum_point_count} points; got {point_count}",
            **base,
        )

    mean_vgs = sum(point[0] for point in fitting_points) / point_count
    mean_log_current = sum(point[1] for point in fitting_points) / point_count
    denominator = sum(
        (point[0] - mean_vgs) ** 2 for point in fitting_points
    )
    if denominator <= 0.0 or not math.isfinite(denominator):
        return _failure("ss_error", "SS fit has zero VGS variance", **base)

    slope_decades_per_V = sum(
        (vgs_V - mean_vgs) * (log_current - mean_log_current)
        for vgs_V, log_current in fitting_points
    ) / denominator

    if (
        not math.isfinite(slope_decades_per_V)
        or slope_decades_per_V <= 0.0
    ):
        return _failure(
            "ss_error", "SS regression slope must be finite and positive", **base
        )

    predicted_values = [
        mean_log_current + slope_decades_per_V * (vgs_V - mean_vgs)
        for vgs_V, _ in fitting_points
    ]
    residual_sum = sum(
        (observed[1] - predicted) ** 2
        for observed, predicted in zip(fitting_points, predicted_values)
    )
    total_sum = sum(
        (point[1] - mean_log_current) ** 2 for point in fitting_points
    )
    r_squared = 1.0 if total_sum <= 1.0e-30 else 1.0 - residual_sum / total_sum
    ss_mV_dec = 1000.0 / slope_decades_per_V

    if not (math.isfinite(ss_mV_dec) and math.isfinite(r_squared)):
        return _failure("ss_error", "SS regression produced non-finite output", **base)

    return {
        "SS_mV_dec": ss_mV_dec,
        "ss_r_squared": r_squared,
        "ss_point_count": point_count,
        "ss_success": True,
        "ss_error": "",
    }


def calculate_transconductance(
    vgs_values: Sequence[Any],
    id_values: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Calculate signed gm with one-sided ends and central interior slopes.

    ``gm_max_S`` is the maximum magnitude of the signed derivative and is
    therefore non-negative.  ``gm_S`` retains the current-sign convention.
    """

    base = {
        "VGS_V": [],
        "ID_A": [],
        "gm_S": [],
        "abs_gm_S": [],
        "gm_max_S": NAN,
        "VGS_at_gm_max_V": NAN,
        "gm_success": False,
        "gm_error": "",
    }

    points, preparation_error = _prepare_points(vgs_values, id_values)
    if preparation_error:
        return _failure("gm_error", preparation_error, **base)
    if len(points) < 2:
        return _failure(
            "gm_error", "at least two finite IV points are required", **base
        )

    sorted_vgs = [point[0] for point in points]
    sorted_id = [point[1] for point in points]
    gm_values: list[float] = []

    for index in range(len(points)):
        if index == 0:
            left_index, right_index = 0, 1
            voltage_span = sorted_vgs[right_index] - sorted_vgs[left_index]
            derivative = (
                (sorted_id[right_index] - sorted_id[left_index])
                / voltage_span
            )
        elif index == len(points) - 1:
            left_index, right_index = len(points) - 2, len(points) - 1
            voltage_span = sorted_vgs[right_index] - sorted_vgs[left_index]
            derivative = (
                (sorted_id[right_index] - sorted_id[left_index])
                / voltage_span
            )
        else:
            x_0 = sorted_vgs[index - 1]
            x_1 = sorted_vgs[index]
            x_2 = sorted_vgs[index + 1]
            y_0 = sorted_id[index - 1]
            y_1 = sorted_id[index]
            y_2 = sorted_id[index + 1]

            # Three-point Lagrange derivative at x_1.  Unlike a simple
            # neighbor secant, this remains centered when a failed IV point
            # leaves unequal VGS spacing in the surviving curve.
            derivative = (
                y_0
                * (x_1 - x_2)
                / ((x_0 - x_1) * (x_0 - x_2))
                + y_1
                * (2.0 * x_1 - x_0 - x_2)
                / ((x_1 - x_0) * (x_1 - x_2))
                + y_2
                * (x_1 - x_0)
                / ((x_2 - x_0) * (x_2 - x_1))
            )

        gm_values.append(derivative)

    absolute_gm_values = [abs(value) for value in gm_values]
    maximum_index = max(
        range(len(absolute_gm_values)),
        key=absolute_gm_values.__getitem__,
    )

    return {
        "VGS_V": sorted_vgs,
        "ID_A": sorted_id,
        "gm_S": gm_values,
        "abs_gm_S": absolute_gm_values,
        "gm_max_S": absolute_gm_values[maximum_index],
        "VGS_at_gm_max_V": sorted_vgs[maximum_index],
        "gm_success": True,
        "gm_error": "",
    }


def interpolate_current_at_bias(
    vgs_values: Sequence[Any],
    id_values: Sequence[Any] | None = None,
    *,
    target_vgs_V: float,
) -> dict[str, Any]:
    """Return abs(ID) at an exact VGS or by linear adjacent-point interpolation."""

    base = {
        "current_A": NAN,
        "target_vgs_V": target_vgs_V,
        "interpolation_success": False,
        "used_exact_point": False,
        "interpolation_error": "",
    }

    try:
        target_vgs_V = float(target_vgs_V)
    except (TypeError, ValueError):
        return _failure(
            "interpolation_error", "target VGS must be numeric", **base
        )
    base["target_vgs_V"] = target_vgs_V
    if not math.isfinite(target_vgs_V):
        return _failure(
            "interpolation_error", "target VGS must be finite", **base
        )

    points, preparation_error = _prepare_points(vgs_values, id_values)
    if preparation_error:
        return _failure("interpolation_error", preparation_error, **base)
    if not points:
        return _failure(
            "interpolation_error", "no finite IV points are available", **base
        )

    for vgs_V, signed_id_A in points:
        if math.isclose(
            vgs_V,
            target_vgs_V,
            rel_tol=0.0,
            abs_tol=_VOLTAGE_TOLERANCE_V,
        ):
            return {
                "current_A": abs(signed_id_A),
                "target_vgs_V": target_vgs_V,
                "interpolation_success": True,
                "used_exact_point": True,
                "interpolation_error": "",
            }

    for (vgs_1, signed_id_1), (vgs_2, signed_id_2) in zip(
        points, points[1:]
    ):
        if vgs_1 < target_vgs_V < vgs_2:
            fraction = (target_vgs_V - vgs_1) / (vgs_2 - vgs_1)
            interpolated_current_A = abs(signed_id_1) + fraction * (
                abs(signed_id_2) - abs(signed_id_1)
            )
            return {
                "current_A": interpolated_current_A,
                "target_vgs_V": target_vgs_V,
                "interpolation_success": True,
                "used_exact_point": False,
                "interpolation_error": "",
            }

    return _failure(
        "interpolation_error",
        "target VGS lies outside the available IV range",
        **base,
    )


def _rows_at_vds(
    rows: Sequence[Mapping[str, Any]],
    target_vds_V: float,
) -> list[Mapping[str, Any]]:
    """Select converged rows at a drain bias without altering their currents."""

    selected: list[Mapping[str, Any]] = []
    for row in rows:
        try:
            vds_V = float(row["VDS_V"])
        except (KeyError, TypeError, ValueError):
            continue
        if (
            math.isfinite(vds_V)
            and math.isclose(
                vds_V,
                float(target_vds_V),
                rel_tol=0.0,
                abs_tol=_VOLTAGE_TOLERANCE_V,
            )
            and _is_converged(row)
        ):
            selected.append(row)
    return selected


def extract_state_metrics(
    idvg_rows: Sequence[Mapping[str, Any]],
    *,
    vds_V: float,
    threshold_current_A: float,
    ss_current_min_A: float,
    ss_current_max_A: float,
    ss_minimum_point_count: int,
    ion_vgs_V: float,
    ioff_vgs_V: float,
    ion_vds_V: float = 0.05,
    ioff_vds_V: float = 0.05,
    current_floor_A: float = 1.0e-30,
) -> dict[str, Any]:
    """Extract one standard metrics row for a state and summary VDS."""

    if not idvg_rows:
        raise ValueError("idvg_rows must contain at least one state row")

    first_row = idvg_rows[0]
    required_metadata = ("state_index", "state", "ntrap_cm3", "nsheet_cm2")
    missing_metadata = [name for name in required_metadata if name not in first_row]
    if missing_metadata:
        raise KeyError(f"ID-VG rows are missing metadata: {missing_metadata}")

    curve_rows = _rows_at_vds(idvg_rows, vds_V)
    threshold = extract_threshold_voltage(
        curve_rows,
        threshold_current_A=threshold_current_A,
        current_floor_A=current_floor_A,
    )
    subthreshold = extract_subthreshold_swing(
        curve_rows,
        current_min_A=ss_current_min_A,
        current_max_A=ss_current_max_A,
        minimum_point_count=ss_minimum_point_count,
    )
    transconductance = calculate_transconductance(curve_rows)

    ion_rows = _rows_at_vds(idvg_rows, ion_vds_V)
    ioff_rows = _rows_at_vds(idvg_rows, ioff_vds_V)
    ion_result = interpolate_current_at_bias(
        ion_rows,
        target_vgs_V=ion_vgs_V,
    )
    ioff_result = interpolate_current_at_bias(
        ioff_rows,
        target_vgs_V=ioff_vgs_V,
    )

    ion_A = (
        float(ion_result["current_A"])
        if ion_result["interpolation_success"]
        else NAN
    )
    ioff_A = (
        float(ioff_result["current_A"])
        if ioff_result["interpolation_success"]
        else NAN
    )

    if (
        math.isfinite(ion_A)
        and math.isfinite(ioff_A)
        and ioff_A > float(current_floor_A)
    ):
        on_off_ratio = ion_A / ioff_A
    else:
        on_off_ratio = NAN

    return {
        "state_index": int(first_row["state_index"]),
        "state": str(first_row["state"]),
        "ntrap_cm3": float(first_row["ntrap_cm3"]),
        "nsheet_cm2": float(first_row["nsheet_cm2"]),
        "VDS_V": float(vds_V),
        "Vth_V": float(threshold["Vth_V"]),
        "threshold_current_A": float(threshold_current_A),
        "vth_success": bool(threshold["vth_success"]),
        "vth_error": str(threshold["vth_error"]),
        "SS_mV_dec": float(subthreshold["SS_mV_dec"]),
        "ss_r_squared": float(subthreshold["ss_r_squared"]),
        "ss_point_count": int(subthreshold["ss_point_count"]),
        "ss_success": bool(subthreshold["ss_success"]),
        "ss_error": str(subthreshold["ss_error"]),
        "Ion_A": ion_A,
        "ion_vgs_V": float(ion_vgs_V),
        "ion_vds_V": float(ion_vds_V),
        "Ioff_A": ioff_A,
        "ioff_vgs_V": float(ioff_vgs_V),
        "ioff_vds_V": float(ioff_vds_V),
        "on_off_ratio": on_off_ratio,
        "gm_max_S": float(transconductance["gm_max_S"]),
        "VGS_at_gm_max_V": float(
            transconductance["VGS_at_gm_max_V"]
        ),
        "ss_current_min_A": float(ss_current_min_A),
        "ss_current_max_A": float(ss_current_max_A),
        "ss_minimum_point_count": int(ss_minimum_point_count),
        "current_floor_A": float(current_floor_A),
    }


__all__ = [
    "calculate_transconductance",
    "extract_state_metrics",
    "extract_subthreshold_swing",
    "extract_threshold_voltage",
    "interpolate_current_at_bias",
]
