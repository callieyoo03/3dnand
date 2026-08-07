"""Regenerate isolated DC curves and compare them with the public baseline.

This runner deliberately writes only below::

    04_charge_trap/results/contact_topology_dc_candidate/

It reuses the established five memory states, ID-VG/ID-VD sweep functions,
and electrical-metric extractor.  ``shared_data`` is read-only reference data.
The public invocation runs base, fine, and extra-fine meshes in fresh Python
processes.  Private worker mode exists so a topology-QC orchestrator can reuse
the same result-bundle contract without importing DEVSIM in its parent process.

No program/erase or retention simulation is imported or executed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import state_characterization_config as config


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
CANDIDATE_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "contact_topology_dc_candidate"
)
FORBIDDEN_WRITE_DIRECTORIES = (
    config.SHARED_DATA_DIRECTORY.resolve(),
    (REPOSITORY_ROOT / "_doc_review").resolve(),
    (REPOSITORY_ROOT / "05_program_erase").resolve(),
)

REFERENCE_PATHS = {
    "idvg": config.IDVG_BY_STATE_CSV_PATH,
    "idvd": config.IDVD_BY_STATE_CSV_PATH,
    "metrics": config.METRICS_BY_STATE_CSV_PATH,
}
OUTPUT_FILENAMES = {
    "idvg": "idvg_by_state.csv",
    "idvd": "idvd_by_state.csv",
    "metrics": "metrics_by_state.csv",
    "regression": "contact_topology_dc_regression.csv",
    "mesh": "mesh_convergence_dc.csv",
}

WORKER_SPECS = (
    ("base", 1.0),
    ("fine", 0.5),
    ("extra_fine", 0.25),
)

VTH_WARNING_ABSOLUTE_V = 2.0e-3
ION_SS_WARNING_RELATIVE = 2.0e-2
CURVE_WARNING_RELATIVE = 5.0e-2
# The public model already identifies 1e-18 A as the lower defensible SS
# analysis current.  It is therefore used as an absolute curve-comparison
# tolerance so zero-VDS and deep-off numerical noise is not treated as a
# meaningful percentage regression.
CURVE_WARNING_ABSOLUTE_A = float(config.SS_CURRENT_MIN_A)
NUMERIC_SCALE_FLOOR = 1.0e-30

REGRESSION_FIELDNAMES = (
    "comparison_domain",
    "quantity",
    "state_index",
    "state",
    "VGS_V",
    "VDS_V",
    "reference_value",
    "candidate_value",
    "reference_abs_ID_A",
    "candidate_abs_ID_A",
    "absolute_difference",
    "relative_difference",
    "relative_scale",
    "absolute_tolerance",
    "relative_tolerance",
    "combined_tolerance",
    "reference_present",
    "candidate_present",
    "reference_finite",
    "candidate_finite",
    "reference_converged",
    "candidate_converged",
    "convergence_match",
    "reference_monotonic",
    "candidate_monotonic",
    "within_warning_threshold",
    "warning",
    "warning_reason",
)

MESH_CONVERGENCE_FIELDNAMES = (
    "state_index",
    "state",
    "VDS_V",
    "quantity",
    "fine_value",
    "extra_fine_value",
    "absolute_difference",
    "relative_difference",
    "relative_scale",
    "absolute_tolerance",
    "relative_tolerance",
    "combined_tolerance",
    "fine_success",
    "extra_fine_success",
    "within_warning_threshold",
    "warning",
    "warning_reason",
)

PRIMARY_METRIC_FIELDS = ("Vth_V", "Ion_A", "SS_mV_dec")
SECONDARY_METRIC_FIELDS = (
    "Ioff_A",
    "on_off_ratio",
    "gm_max_S",
    "VGS_at_gm_max_V",
)


def _finite_or_nan(value: Any) -> float:
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


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_reference_hashes() -> dict[str, str]:
    missing = [str(path) for path in REFERENCE_PATHS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing DC reference CSV(s): " + ", ".join(missing))
    return {name: sha256_file(path) for name, path in REFERENCE_PATHS.items()}


def assert_reference_hashes_unchanged(expected: Mapping[str, str]) -> None:
    actual = capture_reference_hashes()
    if dict(expected) != actual:
        raise RuntimeError(
            "contact-topology DC candidate modified shared_data references: "
            f"expected={dict(expected)}, actual={actual}"
        )


def read_strict_csv(
    path: str | Path, fieldnames: Sequence[str]
) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        actual = tuple(reader.fieldnames or ())
        expected = tuple(fieldnames)
        if actual != expected:
            raise RuntimeError(
                f"CSV schema mismatch for {Path(path)}: {actual} != {expected}"
            )
        rows = list(reader)
    if any(None in row for row in rows):
        raise RuntimeError(f"CSV contains extra columns: {Path(path)}")
    return rows


def _validate_write_path(path: str | Path) -> Path:
    destination = Path(path)
    resolved = destination.resolve()
    for forbidden in FORBIDDEN_WRITE_DIRECTORIES:
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError(f"candidate output may not write under {forbidden}")
    return destination


def write_deterministic_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    destination = _validate_write_path(path)
    expected = tuple(fieldnames)
    if not expected or len(expected) != len(set(expected)):
        raise ValueError("CSV fieldnames must be nonempty and unique")
    materialized = [dict(row) for row in rows]
    for index, row in enumerate(materialized):
        if set(row) != set(expected):
            raise ValueError(
                f"CSV row {index} schema mismatch: "
                f"missing={sorted(set(expected) - set(row))}, "
                f"extra={sorted(set(row) - set(expected))}"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=expected,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows({name: row[name] for name in expected} for row in materialized)
    return destination


def compare_numeric_values(
    reference_value: Any,
    candidate_value: Any,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    relative_scale_floor: float = NUMERIC_SCALE_FLOOR,
) -> dict[str, Any]:
    """Compare finite values with absolute plus baseline-scaled tolerance."""

    absolute_limit = float(absolute_tolerance)
    relative_limit = float(relative_tolerance)
    scale_floor = float(relative_scale_floor)
    if (
        not math.isfinite(absolute_limit)
        or not math.isfinite(relative_limit)
        or not math.isfinite(scale_floor)
        or absolute_limit < 0.0
        or relative_limit < 0.0
        or scale_floor <= 0.0
    ):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    reference = _finite_or_nan(reference_value)
    candidate = _finite_or_nan(candidate_value)
    reference_finite = math.isfinite(reference)
    candidate_finite = math.isfinite(candidate)
    scale = max(abs(reference), scale_floor) if reference_finite else math.nan
    if reference_finite and candidate_finite:
        absolute_difference = abs(candidate - reference)
        relative_difference = absolute_difference / scale
        combined_tolerance = absolute_limit + relative_limit * scale
        within = absolute_difference <= combined_tolerance
    else:
        absolute_difference = math.nan
        relative_difference = math.nan
        combined_tolerance = (
            absolute_limit + relative_limit * scale
            if math.isfinite(scale)
            else math.nan
        )
        within = False
    return {
        "reference_value": reference,
        "candidate_value": candidate,
        "absolute_difference": absolute_difference,
        "relative_difference": relative_difference,
        "relative_scale": scale,
        "absolute_tolerance": absolute_limit,
        "relative_tolerance": relative_limit,
        "combined_tolerance": combined_tolerance,
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "within_warning_threshold": within,
    }


def _unique_map(
    rows: Sequence[Mapping[str, Any]],
    key_builder: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    *,
    label: str,
) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    result: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = key_builder(row)
        if key in result:
            raise ValueError(f"duplicate {label} coordinate {key!r}")
        result[key] = row
    return result


def _iv_key(row: Mapping[str, Any]) -> tuple[int, float, float]:
    return (
        int(row["state_index"]),
        round(float(row["VGS_V"]), 12),
        round(float(row["VDS_V"]), 12),
    )


def _metric_key(row: Mapping[str, Any]) -> tuple[int, float]:
    return int(row["state_index"]), round(float(row["VDS_V"]), 12)


def _curve_group_key(
    row: Mapping[str, Any], domain: str
) -> tuple[int, float]:
    if domain == "idvg":
        return int(row["state_index"]), round(float(row["VDS_V"]), 12)
    if domain == "idvd":
        return int(row["state_index"]), round(float(row["VGS_V"]), 12)
    raise ValueError("curve domain must be idvg or idvd")


def evaluate_curve_monotonicity(
    rows: Sequence[Mapping[str, Any]], domain: str
) -> dict[tuple[int, float], bool]:
    """Return tolerant nondecreasing |ID| status for each complete curve."""

    groups: dict[tuple[int, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(_curve_group_key(row, domain), []).append(row)
    result: dict[tuple[int, float], bool] = {}
    coordinate_name = "VGS_V" if domain == "idvg" else "VDS_V"
    if domain == "idvg":
        expected_count = int(
            round(
                (config.IDVG_VGS_STOP_V - config.IDVG_VGS_START_V)
                / config.IDVG_VGS_STEP_V
            )
        ) + 1
    else:
        expected_count = int(
            round(
                (config.IDVD_VDS_STOP_V - config.IDVD_VDS_START_V)
                / config.IDVD_VDS_STEP_V
            )
        ) + 1
    for key, group in groups.items():
        ordered = sorted(group, key=lambda row: float(row[coordinate_name]))
        converged = all(_strict_bool(row.get("converged")) for row in ordered)
        currents = [_finite_or_nan(row.get("abs_ID_A")) for row in ordered]
        finite = all(math.isfinite(value) for value in currents)
        monotonic = len(ordered) == expected_count and converged and finite and all(
            right + CURVE_WARNING_ABSOLUTE_A >= left
            for left, right in zip(currents, currents[1:])
        )
        result[key] = monotonic
    return result


def _warning_reason(
    *,
    reference_present: bool,
    candidate_present: bool,
    reference_finite: bool,
    candidate_finite: bool,
    reference_converged: bool,
    candidate_converged: bool,
    within: bool,
    metadata_match: bool,
) -> str:
    reasons: list[str] = []
    if not reference_present:
        reasons.append("missing_reference_point")
    if not candidate_present:
        reasons.append("missing_candidate_point")
    if reference_present and not reference_finite:
        reasons.append("nonfinite_reference")
    if candidate_present and not candidate_finite:
        reasons.append("nonfinite_candidate")
    if reference_present and not reference_converged:
        reasons.append("reference_not_converged")
    if candidate_present and not candidate_converged:
        reasons.append("candidate_not_converged")
    if reference_present and candidate_present and reference_converged != candidate_converged:
        reasons.append("convergence_mismatch")
    if reference_present and candidate_present and not metadata_match:
        reasons.append("state_metadata_mismatch")
    if (
        reference_present
        and candidate_present
        and reference_finite
        and candidate_finite
        and reference_converged
        and candidate_converged
        and not within
    ):
        reasons.append("difference_exceeds_warning_threshold")
    return ";".join(reasons)


def build_curve_regression_rows(
    domain: str,
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if domain not in {"idvg", "idvd"}:
        raise ValueError("domain must be idvg or idvd")
    reference = _unique_map(reference_rows, _iv_key, label=f"{domain} reference")
    candidate = _unique_map(candidate_rows, _iv_key, label=f"{domain} candidate")
    reference_monotonic = evaluate_curve_monotonicity(reference_rows, domain)
    candidate_monotonic = evaluate_curve_monotonicity(candidate_rows, domain)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(reference) | set(candidate)):
        reference_row = reference.get(key)
        candidate_row = candidate.get(key)
        reference_present = reference_row is not None
        candidate_present = candidate_row is not None
        identity_row = candidate_row or reference_row
        if identity_row is None:
            raise AssertionError("union key must have an identity row")
        reference_converged = (
            _strict_bool(reference_row.get("converged")) if reference_row else False
        )
        candidate_converged = (
            _strict_bool(candidate_row.get("converged")) if candidate_row else False
        )
        comparison = compare_numeric_values(
            reference_row.get("ID_A") if reference_row else math.nan,
            candidate_row.get("ID_A") if candidate_row else math.nan,
            absolute_tolerance=CURVE_WARNING_ABSOLUTE_A,
            relative_tolerance=CURVE_WARNING_RELATIVE,
            relative_scale_floor=CURVE_WARNING_ABSOLUTE_A,
        )
        metadata_match = bool(reference_row and candidate_row) and (
            str(reference_row["state"]) == str(candidate_row["state"])
            and math.isclose(
                float(reference_row["ntrap_cm3"]),
                float(candidate_row["ntrap_cm3"]),
                rel_tol=1.0e-12,
                abs_tol=0.0,
            )
        )
        reason = _warning_reason(
            reference_present=reference_present,
            candidate_present=candidate_present,
            reference_finite=comparison["reference_finite"],
            candidate_finite=comparison["candidate_finite"],
            reference_converged=reference_converged,
            candidate_converged=candidate_converged,
            within=comparison["within_warning_threshold"],
            metadata_match=metadata_match,
        )
        group_key = _curve_group_key(identity_row, domain)
        row = {
            "comparison_domain": domain,
            "quantity": "ID_A",
            "state_index": key[0],
            "state": str(identity_row["state"]),
            "VGS_V": key[1],
            "VDS_V": key[2],
            **{
                name: comparison[name]
                for name in (
                    "reference_value",
                    "candidate_value",
                )
            },
            "reference_abs_ID_A": _finite_or_nan(
                reference_row.get("abs_ID_A") if reference_row else math.nan
            ),
            "candidate_abs_ID_A": _finite_or_nan(
                candidate_row.get("abs_ID_A") if candidate_row else math.nan
            ),
            **{
                name: comparison[name]
                for name in (
                    "absolute_difference",
                    "relative_difference",
                    "relative_scale",
                    "absolute_tolerance",
                    "relative_tolerance",
                    "combined_tolerance",
                )
            },
            "reference_present": reference_present,
            "candidate_present": candidate_present,
            "reference_finite": comparison["reference_finite"],
            "candidate_finite": comparison["candidate_finite"],
            "reference_converged": reference_converged,
            "candidate_converged": candidate_converged,
            "convergence_match": (
                reference_present
                and candidate_present
                and reference_converged == candidate_converged
            ),
            "reference_monotonic": reference_monotonic.get(group_key, False),
            "candidate_monotonic": candidate_monotonic.get(group_key, False),
            "within_warning_threshold": comparison["within_warning_threshold"],
            "warning": bool(reason),
            "warning_reason": reason,
        }
        if tuple(row) != REGRESSION_FIELDNAMES:
            raise RuntimeError("curve regression row schema mismatch")
        rows.append(row)
    return rows


def _metric_policy(quantity: str) -> tuple[float, float, float]:
    if quantity == "Vth_V":
        return VTH_WARNING_ABSOLUTE_V, 0.0, NUMERIC_SCALE_FLOOR
    if quantity == "SS_mV_dec":
        return 0.0, ION_SS_WARNING_RELATIVE, 1.0e-12
    if quantity == "Ion_A":
        return 0.0, ION_SS_WARNING_RELATIVE, CURVE_WARNING_ABSOLUTE_A
    if quantity in set(SECONDARY_METRIC_FIELDS):
        # Retained for regression visibility; the task defines warning limits
        # only for Vth, Ion, SS, and pointwise IV curves.  Do not silently
        # repurpose the curve limit for secondary extracted metrics.
        return math.nan, math.nan, NUMERIC_SCALE_FLOOR
    raise KeyError(f"unknown metric quantity {quantity!r}")


def _metric_success(row: Mapping[str, Any] | None, quantity: str) -> bool:
    if row is None:
        return False
    if quantity == "Vth_V":
        return _strict_bool(row.get("vth_success"))
    if quantity == "SS_mV_dec":
        return _strict_bool(row.get("ss_success"))
    return math.isfinite(_finite_or_nan(row.get(quantity)))


def _vth_monotonicity(
    rows: Sequence[Mapping[str, Any]],
) -> dict[float, bool]:
    by_vds: dict[float, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_vds.setdefault(round(float(row["VDS_V"]), 12), []).append(row)
    result: dict[float, bool] = {}
    for vds, group in by_vds.items():
        ordered = sorted(group, key=lambda row: int(row["state_index"]))
        values = [_finite_or_nan(row.get("Vth_V")) for row in ordered]
        result[vds] = (
            len(ordered) == len(config.MEMORY_STATES)
            and all(_metric_success(row, "Vth_V") for row in ordered)
            and all(right > left for left, right in zip(values, values[1:]))
        )
    return result


def build_metrics_regression_rows(
    reference_rows: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    reference = _unique_map(reference_rows, _metric_key, label="metrics reference")
    candidate = _unique_map(candidate_rows, _metric_key, label="metrics candidate")
    reference_vth_monotonic = _vth_monotonicity(reference_rows)
    candidate_vth_monotonic = _vth_monotonicity(candidate_rows)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(reference) | set(candidate)):
        reference_row = reference.get(key)
        candidate_row = candidate.get(key)
        identity_row = candidate_row or reference_row
        if identity_row is None:
            raise AssertionError("union key must have an identity row")
        for quantity in (*PRIMARY_METRIC_FIELDS, *SECONDARY_METRIC_FIELDS):
            absolute_tolerance, relative_tolerance, scale_floor = _metric_policy(
                quantity
            )
            informational = not math.isfinite(absolute_tolerance)
            comparison = (
                compare_numeric_values(
                    reference_row.get(quantity) if reference_row else math.nan,
                    candidate_row.get(quantity) if candidate_row else math.nan,
                    absolute_tolerance=absolute_tolerance,
                    relative_tolerance=relative_tolerance,
                    relative_scale_floor=scale_floor,
                )
                if not informational
                else compare_numeric_values(
                    reference_row.get(quantity) if reference_row else math.nan,
                    candidate_row.get(quantity) if candidate_row else math.nan,
                    absolute_tolerance=0.0,
                    relative_tolerance=0.0,
                    relative_scale_floor=scale_floor,
                )
            )
            if informational and comparison["reference_finite"] and comparison["candidate_finite"]:
                comparison["absolute_tolerance"] = math.nan
                comparison["relative_tolerance"] = math.nan
                comparison["combined_tolerance"] = math.nan
                comparison["within_warning_threshold"] = True
            reference_success = _metric_success(reference_row, quantity)
            candidate_success = _metric_success(candidate_row, quantity)
            metadata_match = bool(reference_row and candidate_row) and (
                str(reference_row["state"]) == str(candidate_row["state"])
                and math.isclose(
                    float(reference_row["ntrap_cm3"]),
                    float(candidate_row["ntrap_cm3"]),
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
            )
            reason = _warning_reason(
                reference_present=reference_row is not None,
                candidate_present=candidate_row is not None,
                reference_finite=comparison["reference_finite"],
                candidate_finite=comparison["candidate_finite"],
                reference_converged=reference_success,
                candidate_converged=candidate_success,
                within=comparison["within_warning_threshold"],
                metadata_match=metadata_match,
            )
            row = {
                "comparison_domain": "metrics",
                "quantity": quantity,
                "state_index": key[0],
                "state": str(identity_row["state"]),
                "VGS_V": math.nan,
                "VDS_V": key[1],
                "reference_value": comparison["reference_value"],
                "candidate_value": comparison["candidate_value"],
                "reference_abs_ID_A": math.nan,
                "candidate_abs_ID_A": math.nan,
                "absolute_difference": comparison["absolute_difference"],
                "relative_difference": comparison["relative_difference"],
                "relative_scale": comparison["relative_scale"],
                "absolute_tolerance": comparison["absolute_tolerance"],
                "relative_tolerance": comparison["relative_tolerance"],
                "combined_tolerance": comparison["combined_tolerance"],
                "reference_present": reference_row is not None,
                "candidate_present": candidate_row is not None,
                "reference_finite": comparison["reference_finite"],
                "candidate_finite": comparison["candidate_finite"],
                "reference_converged": reference_success,
                "candidate_converged": candidate_success,
                "convergence_match": (
                    reference_row is not None
                    and candidate_row is not None
                    and reference_success == candidate_success
                ),
                "reference_monotonic": (
                    reference_vth_monotonic.get(key[1], False)
                    if quantity == "Vth_V"
                    else ""
                ),
                "candidate_monotonic": (
                    candidate_vth_monotonic.get(key[1], False)
                    if quantity == "Vth_V"
                    else ""
                ),
                "within_warning_threshold": comparison[
                    "within_warning_threshold"
                ],
                "warning": bool(reason),
                "warning_reason": reason,
            }
            if tuple(row) != REGRESSION_FIELDNAMES:
                raise RuntimeError("metric regression row schema mismatch")
            rows.append(row)
    return rows


def build_regression_rows(
    *,
    reference_idvg_rows: Sequence[Mapping[str, Any]],
    candidate_idvg_rows: Sequence[Mapping[str, Any]],
    reference_idvd_rows: Sequence[Mapping[str, Any]],
    candidate_idvd_rows: Sequence[Mapping[str, Any]],
    reference_metrics_rows: Sequence[Mapping[str, Any]],
    candidate_metrics_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = [
        *build_curve_regression_rows(
            "idvg", reference_idvg_rows, candidate_idvg_rows
        ),
        *build_curve_regression_rows(
            "idvd", reference_idvd_rows, candidate_idvd_rows
        ),
        *build_metrics_regression_rows(
            reference_metrics_rows, candidate_metrics_rows
        ),
    ]
    domain_order = {"idvg": 0, "idvd": 1, "metrics": 2}
    rows.sort(
        key=lambda row: (
            domain_order[str(row["comparison_domain"])],
            int(row["state_index"]),
            float(row["VGS_V"])
            if math.isfinite(_finite_or_nan(row["VGS_V"]))
            else -math.inf,
            float(row["VDS_V"]),
            str(row["quantity"]),
        )
    )
    return rows


def build_mesh_convergence_rows(
    fine_metrics_rows: Sequence[Mapping[str, Any]],
    extra_fine_metrics_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    fine = _unique_map(fine_metrics_rows, _metric_key, label="fine metrics")
    extra = _unique_map(
        extra_fine_metrics_rows, _metric_key, label="extra-fine metrics"
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(set(fine) | set(extra)):
        fine_row = fine.get(key)
        extra_row = extra.get(key)
        identity = extra_row or fine_row
        if identity is None:
            raise AssertionError("union key must have an identity row")
        for quantity in PRIMARY_METRIC_FIELDS:
            absolute_tolerance, relative_tolerance, scale_floor = _metric_policy(
                quantity
            )
            comparison = compare_numeric_values(
                fine_row.get(quantity) if fine_row else math.nan,
                extra_row.get(quantity) if extra_row else math.nan,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                relative_scale_floor=scale_floor,
            )
            fine_success = _metric_success(fine_row, quantity)
            extra_success = _metric_success(extra_row, quantity)
            reasons: list[str] = []
            if fine_row is None:
                reasons.append("missing_fine_metric")
            if extra_row is None:
                reasons.append("missing_extra_fine_metric")
            if fine_row is not None and not fine_success:
                reasons.append("fine_metric_failed")
            if extra_row is not None and not extra_success:
                reasons.append("extra_fine_metric_failed")
            if (
                fine_success
                and extra_success
                and not comparison["within_warning_threshold"]
            ):
                reasons.append("mesh_difference_exceeds_warning_threshold")
            row = {
                "state_index": key[0],
                "state": str(identity["state"]),
                "VDS_V": key[1],
                "quantity": quantity,
                "fine_value": comparison["reference_value"],
                "extra_fine_value": comparison["candidate_value"],
                "absolute_difference": comparison["absolute_difference"],
                "relative_difference": comparison["relative_difference"],
                "relative_scale": comparison["relative_scale"],
                "absolute_tolerance": comparison["absolute_tolerance"],
                "relative_tolerance": comparison["relative_tolerance"],
                "combined_tolerance": comparison["combined_tolerance"],
                "fine_success": fine_success,
                "extra_fine_success": extra_success,
                "within_warning_threshold": comparison[
                    "within_warning_threshold"
                ],
                "warning": bool(reasons),
                "warning_reason": ";".join(reasons),
            }
            if tuple(row) != MESH_CONVERGENCE_FIELDNAMES:
                raise RuntimeError("mesh-convergence row schema mismatch")
            rows.append(row)
    return rows


@contextmanager
def scaled_runtime_mesh_lines(
    structure_module: Any,
    mesh_scale: float,
    line_log: list[dict[str, Any]],
) -> Iterator[None]:
    """Scale only mesh spacing ``ps`` and restore the runtime function."""

    scale = float(mesh_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("mesh_scale must be finite and positive")
    original = structure_module.add_2d_mesh_line

    def scaled_add_2d_mesh_line(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("mesh audit requires keyword mesh-line calls")
        base_spacing = float(kwargs["ps"])
        forwarded = dict(kwargs)
        forwarded["ps"] = base_spacing * scale
        line_log.append(
            {
                "direction": str(kwargs["dir"]),
                "position_cm": float(kwargs["pos"]),
                "base_spacing_cm": base_spacing,
                "scaled_spacing_cm": forwarded["ps"],
                "mesh_scale": scale,
            }
        )
        return original(**forwarded)

    structure_module.add_2d_mesh_line = scaled_add_2d_mesh_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = original


def validate_mesh_line_bundles(
    bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    base_lines = bundles["base"].get("mesh_line_log", [])
    errors: list[str] = []
    for level, expected_scale in WORKER_SPECS:
        rows = bundles[level].get("mesh_line_log", [])
        if len(rows) != len(base_lines) or not rows:
            errors.append(f"{level}: mesh-line count mismatch")
            continue
        for index, (base, row) in enumerate(zip(base_lines, rows)):
            same_fixed_data = (
                str(base["direction"]) == str(row["direction"])
                and math.isclose(
                    float(base["position_cm"]),
                    float(row["position_cm"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-30,
                )
                and math.isclose(
                    float(base["base_spacing_cm"]),
                    float(row["base_spacing_cm"]),
                    rel_tol=0.0,
                    abs_tol=1.0e-30,
                )
            )
            scaled_ok = math.isclose(
                float(row["scaled_spacing_cm"]),
                float(row["base_spacing_cm"]) * expected_scale,
                rel_tol=1.0e-14,
                abs_tol=1.0e-30,
            )
            if not same_fixed_data or not scaled_ok:
                errors.append(f"{level}: mesh line {index} mismatch")
    return {"passed": not errors, "errors": errors}


def validate_mesh_refinement_bundles(
    bundles: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Require fixed geometry and strictly refined runtime mesh counts."""

    errors: list[str] = []
    base_geometry = bundles["base"].get("geometry")
    if any(bundle.get("geometry") != base_geometry for bundle in bundles.values()):
        errors.append("runtime geometry changed across mesh levels")
    for count_name in ("total_region_node_count", "total_element_count"):
        counts = [
            _finite_or_nan(bundles[level].get("mesh_counts", {}).get(count_name))
            for level, _ in WORKER_SPECS
        ]
        if not all(math.isfinite(value) for value in counts):
            errors.append(f"{count_name} is missing or non-finite")
        elif not counts[0] < counts[1] < counts[2]:
            errors.append(
                f"{count_name} is not strictly refined: "
                + " < ".join(f"{value:g}" for value in counts)
            )
    return {"passed": not errors, "errors": errors}


def _runtime_mesh_counts(device: str) -> dict[str, int]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    nodes = 0
    elements = 0
    for region in get_region_list(device=device):
        nodes += len(get_node_model_values(device=device, region=region, name="x"))
        elements += len(get_element_node_list(device=device, region=region))
    return {"total_region_node_count": nodes, "total_element_count": elements}


def run_worker(mesh_level: str, mesh_scale: float) -> dict[str, Any]:
    """Run the canonical full DC characterization for one mesh spacing."""

    import run_memory_window as memory_window
    from run_idvd_by_state import run_idvd_by_state
    from run_idvg_by_state import run_idvg_by_state
    from run_state_characterization import (
        build_metrics_rows,
        evaluate_physical_sanity,
        expected_idvd_coordinates,
        expected_idvg_coordinates,
        validate_bias_grid,
    )
    from state_sweep_helpers import initialize_characterization_device

    config.validate_state_characterization_config()
    line_log: list[dict[str, Any]] = []
    with scaled_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        mesh_scale,
        line_log,
    ):
        geometry, operating_point = initialize_characterization_device()

    idvg_rows, operating_point = run_idvg_by_state(
        states=config.MEMORY_STATES,
        vds_values_V=config.IDVG_VDS_VALUES_V,
        vgs_start_V=config.IDVG_VGS_START_V,
        vgs_stop_V=config.IDVG_VGS_STOP_V,
        vgs_step_V=config.IDVG_VGS_STEP_V,
        operating_point=operating_point,
    )
    metrics_rows = build_metrics_rows(
        idvg_rows,
        states=config.MEMORY_STATES,
        vds_values_V=config.IDVG_VDS_VALUES_V,
    )
    idvd_rows, operating_point = run_idvd_by_state(
        states=config.MEMORY_STATES,
        vgs_values_V=config.IDVD_VGS_VALUES_V,
        vds_start_V=config.IDVD_VDS_START_V,
        vds_stop_V=config.IDVD_VDS_STOP_V,
        vds_step_V=config.IDVD_VDS_STEP_V,
        operating_point=operating_point,
    )
    sanity = evaluate_physical_sanity(
        idvg_rows=idvg_rows,
        idvd_rows=idvd_rows,
        metrics_rows=metrics_rows,
        programmed_state_index=int(config.MEMORY_STATES[-1]["state_index"]),
        orchestration_errors=operating_point.orchestration_errors,
    )
    sanity["idvg_grid"] = validate_bias_grid(
        idvg_rows,
        expected_coordinates=expected_idvg_coordinates(
            config.MEMORY_STATES,
            config.IDVG_VDS_VALUES_V,
            config.IDVG_VGS_STEP_V,
        ),
    )
    sanity["idvd_grid"] = validate_bias_grid(
        idvd_rows,
        expected_coordinates=expected_idvd_coordinates(
            config.MEMORY_STATES,
            config.IDVD_VGS_VALUES_V,
            config.IDVD_VDS_STEP_V,
        ),
    )
    return {
        "mesh_level": str(mesh_level),
        "mesh_scale": float(mesh_scale),
        "geometry": dict(geometry),
        "mesh_line_log": line_log,
        "mesh_counts": _runtime_mesh_counts(memory_window.device),
        "idvg_rows": idvg_rows,
        "idvd_rows": idvd_rows,
        "metrics_rows": metrics_rows,
        "sanity": sanity,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(name): _json_safe(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        # Preserve an explicit failed numeric point through strict JSON.  A
        # JSON null would later become an empty CSV field, whereas the existing
        # IV contract intentionally records the parseable token ``nan``.
        return "nan"
    if isinstance(value, Path):
        return str(value)
    return value


def write_worker_bundle(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    destination = _validate_write_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", newline="\n", encoding="utf-8") as stream:
        json.dump(
            _json_safe(bundle),
            stream,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        stream.write("\n")
    temporary.replace(destination)
    return destination


def build_worker_command(
    python_executable: str,
    *,
    mesh_level: str,
    mesh_scale: float,
    worker_output: str | Path,
) -> list[str]:
    return [
        str(python_executable),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--mesh-level",
        str(mesh_level),
        "--mesh-scale",
        f"{float(mesh_scale):.17g}",
        "--worker-output",
        str(Path(worker_output).resolve()),
    ]


def run_fresh_workers(
    temporary_directory: str | Path,
    *,
    python_executable: str = sys.executable,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> dict[str, dict[str, Any]]:
    directory = Path(temporary_directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict[str, Any]] = {}
    for mesh_level, mesh_scale in WORKER_SPECS:
        worker_output = directory / f"{mesh_level}.json"
        command = build_worker_command(
            python_executable,
            mesh_level=mesh_level,
            mesh_scale=mesh_scale,
            worker_output=worker_output,
        )
        print("Running fresh DC candidate worker:", " ".join(command), flush=True)
        log_path = directory / f"{mesh_level}.log"
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
            # Preserve the narrow dependency-injection contract used by unit
            # tests while keeping real DEVSIM output out of the orchestrator.
            completed = subprocess_run(command, cwd=REPOSITORY_ROOT, check=False)
        return_code = int(getattr(completed, "returncode", 0))
        if return_code != 0:
            tail = ""
            if log_path.is_file():
                tail = "\n".join(
                    log_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).splitlines()[-40:]
                )
            raise RuntimeError(
                f"DC candidate worker {mesh_level!r} exited with {return_code}"
                + (f"\n{tail}" if tail else "")
            )
        with worker_output.open("r", encoding="utf-8") as stream:
            bundle = json.load(stream)
        if str(bundle.get("mesh_level")) != mesh_level or not math.isclose(
            float(bundle.get("mesh_scale")),
            mesh_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"worker bundle identity mismatch: {worker_output}")
        bundles[mesh_level] = bundle
    return bundles


def read_reference_bundle() -> dict[str, list[dict[str, str]]]:
    return {
        "idvg": read_strict_csv(REFERENCE_PATHS["idvg"], config.IDVG_FIELDNAMES),
        "idvd": read_strict_csv(REFERENCE_PATHS["idvd"], config.IDVD_FIELDNAMES),
        "metrics": read_strict_csv(
            REFERENCE_PATHS["metrics"], config.METRICS_FIELDNAMES
        ),
    }


def build_output_tables(
    bundles: Mapping[str, Mapping[str, Any]],
    reference_bundle: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    required = {name for name, _ in WORKER_SPECS}
    if set(bundles) != required:
        raise ValueError(f"worker bundle set must be {sorted(required)}")
    mesh_audit = validate_mesh_line_bundles(bundles)
    if not mesh_audit["passed"]:
        raise RuntimeError("mesh-line audit failed: " + "; ".join(mesh_audit["errors"]))
    refinement_audit = validate_mesh_refinement_bundles(bundles)
    if not refinement_audit["passed"]:
        raise RuntimeError(
            "mesh-refinement audit failed: "
            + "; ".join(refinement_audit["errors"])
        )
    base = bundles["base"]
    regression = build_regression_rows(
        reference_idvg_rows=reference_bundle["idvg"],
        candidate_idvg_rows=base["idvg_rows"],
        reference_idvd_rows=reference_bundle["idvd"],
        candidate_idvd_rows=base["idvd_rows"],
        reference_metrics_rows=reference_bundle["metrics"],
        candidate_metrics_rows=base["metrics_rows"],
    )
    mesh = build_mesh_convergence_rows(
        bundles["fine"]["metrics_rows"],
        bundles["extra_fine"]["metrics_rows"],
    )
    return {
        "idvg": [dict(row) for row in base["idvg_rows"]],
        "idvd": [dict(row) for row in base["idvd_rows"]],
        "metrics": [dict(row) for row in base["metrics_rows"]],
        "regression": regression,
        "mesh": mesh,
    }


def write_candidate_outputs(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_directory: str | Path = CANDIDATE_DIRECTORY,
) -> tuple[Path, ...]:
    directory = Path(output_directory)
    specs = (
        ("idvg", config.IDVG_FIELDNAMES),
        ("idvd", config.IDVD_FIELDNAMES),
        ("metrics", config.METRICS_FIELDNAMES),
        ("regression", REGRESSION_FIELDNAMES),
        ("mesh", MESH_CONVERGENCE_FIELDNAMES),
    )
    return tuple(
        write_deterministic_csv(
            directory / OUTPUT_FILENAMES[name],
            fieldnames,
            list(tables[name]),
        )
        for name, fieldnames in specs
    )


def emit_regression_for_qc(
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    qc_output_directory: str | Path,
) -> tuple[Path, Path]:
    """Emit only DC regression artifacts when the QC orchestrator requests."""

    directory = Path(qc_output_directory)
    regression = write_deterministic_csv(
        directory / OUTPUT_FILENAMES["regression"],
        REGRESSION_FIELDNAMES,
        list(tables["regression"]),
    )
    mesh = write_deterministic_csv(
        directory / OUTPUT_FILENAMES["mesh"],
        MESH_CONVERGENCE_FIELDNAMES,
        list(tables["mesh"]),
    )
    return regression, mesh


def summarize_tables(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    regression = list(tables["regression"])
    mesh = list(tables["mesh"])
    return {
        "idvg_point_count": len(tables["idvg"]),
        "idvd_point_count": len(tables["idvd"]),
        "metrics_row_count": len(tables["metrics"]),
        "regression_warning_count": sum(bool(row["warning"]) for row in regression),
        "curve_warning_count": sum(
            bool(row["warning"])
            for row in regression
            if row["comparison_domain"] in {"idvg", "idvd"}
        ),
        "metric_warning_count": sum(
            bool(row["warning"])
            for row in regression
            if row["comparison_domain"] == "metrics"
        ),
        "mesh_warning_count": sum(bool(row["warning"]) for row in mesh),
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate isolated five-state DC data on base/fine/extra-fine "
            "meshes and compare it with read-only shared_data references."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-level", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--qc-output-directory",
        type=Path,
        help=(
            "Additionally emit contact_topology_dc_regression.csv and "
            "mesh_convergence_dc.csv into an orchestrator-owned QC directory."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.worker:
        if (
            not arguments.mesh_level
            or arguments.mesh_scale is None
            or arguments.worker_output is None
            or arguments.qc_output_directory is not None
        ):
            raise SystemExit("private worker mode requires level/scale/output only")
        bundle = run_worker(arguments.mesh_level, arguments.mesh_scale)
        write_worker_bundle(arguments.worker_output, bundle)
        return 0
    if any(
        value is not None
        for value in (
            arguments.mesh_level,
            arguments.mesh_scale,
            arguments.worker_output,
        )
    ):
        raise SystemExit("worker-only arguments require --worker")

    reference_hashes = capture_reference_hashes()
    reference_bundle = read_reference_bundle()
    with tempfile.TemporaryDirectory(prefix="contact_topology_dc_") as temporary:
        bundles = run_fresh_workers(temporary)
    assert_reference_hashes_unchanged(reference_hashes)
    tables = build_output_tables(bundles, reference_bundle)
    outputs = write_candidate_outputs(tables)
    if arguments.qc_output_directory is not None:
        outputs += emit_regression_for_qc(tables, arguments.qc_output_directory)
    assert_reference_hashes_unchanged(reference_hashes)

    summary = summarize_tables(tables)
    print("Contact-topology DC candidate summary")
    for name, value in summary.items():
        print(f"  {name}: {value}")
    for path in outputs:
        print(f"  {path.resolve()}")
    # Regression thresholds are warnings, not post-hoc physics-fit gates.
    # The process fails only on orchestration/schema/reference-integrity errors.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
