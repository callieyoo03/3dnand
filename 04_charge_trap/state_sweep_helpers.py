"""Shared orchestration helpers for trapped-charge state IV sweeps.

This module deliberately contains no device physics.  Device construction,
solving, bias ramps, and trap-density ramps are delegated to the established
``run_memory_window`` implementation.  The helpers here only keep track of the
last converged operating point and enforce the standard IV CSV schema.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import run_memory_window as memory_window
import state_characterization_config as config
import trap_parameters as tp
from trap_models import get_trap_state_information


# Re-export the established solver and ramp implementations.  Keeping these as
# direct aliases also makes it explicit that this module does not duplicate the
# underlying device physics.
solve_dc = memory_window.solve_dc
set_terminal_bias = memory_window.set_terminal_bias
get_drain_current = memory_window.get_drain_current_A
create_output_voltage_list = memory_window.create_output_voltage_list
adaptive_voltage_ramp = memory_window.adaptive_voltage_ramp
adaptive_trap_density_ramp = memory_window.adaptive_trap_density_ramp
initialize_device = memory_window.initialize_device
solve_empty_state = memory_window.solve_empty_state


def _ramp_constant(
    config_name: str,
    legacy_name: str,
    fallback: float,
) -> float:
    """Read a ramp constant while accepting the existing legacy spelling."""

    value = getattr(
        config,
        config_name,
        getattr(config, legacy_name, fallback),
    )
    return float(value)


DRAIN_INITIAL_STEP_V = _ramp_constant(
    "DRAIN_INITIAL_STEP_V",
    "DRAIN_INITIAL_STEP",
    getattr(tp, "DRAIN_INITIAL_STEP_V", tp.DRAIN_INITIAL_STEP),
)
DRAIN_MINIMUM_STEP_V = _ramp_constant(
    "DRAIN_MINIMUM_STEP_V",
    "DRAIN_MINIMUM_STEP",
    getattr(tp, "DRAIN_MINIMUM_STEP_V", tp.DRAIN_MINIMUM_STEP),
)
GATE_INITIAL_STEP_V = _ramp_constant(
    "GATE_INITIAL_STEP_V",
    "GATE_INITIAL_STEP",
    getattr(tp, "GATE_INITIAL_STEP_V", tp.GATE_INITIAL_STEP),
)
GATE_MINIMUM_STEP_V = _ramp_constant(
    "GATE_MINIMUM_STEP_V",
    "GATE_MINIMUM_STEP",
    getattr(tp, "GATE_MINIMUM_STEP_V", tp.GATE_MINIMUM_STEP),
)
TRAP_INITIAL_STEP_CM3 = _ramp_constant(
    "TRAP_INITIAL_STEP_CM3",
    "TRAP_INITIAL_STEP",
    getattr(
        tp,
        "TRAP_INITIAL_STEP_CM3",
        getattr(
            tp,
            "TRAP_INITIAL_STEP",
            memory_window.DEFAULT_TRAP_INITIAL_STEP_CM3,
        ),
    ),
)
TRAP_MINIMUM_STEP_CM3 = _ramp_constant(
    "TRAP_MINIMUM_STEP_CM3",
    "TRAP_MINIMUM_STEP",
    getattr(
        tp,
        "TRAP_MINIMUM_STEP_CM3",
        getattr(
            tp,
            "TRAP_MINIMUM_STEP",
            memory_window.DEFAULT_TRAP_MINIMUM_STEP_CM3,
        ),
    ),
)


IV_FIELDNAMES = tuple(config.IDVG_FIELDNAMES)

if IV_FIELDNAMES != tuple(config.IDVD_FIELDNAMES):
    raise ValueError("ID-VG and ID-VD must use the same standard IV schema.")


@dataclass
class OperatingPoint:
    """Last successfully solved gate, drain, and trap-density coordinates."""

    gate_voltage_V: float = 0.0
    drain_voltage_V: float = 0.0
    trap_density_cm3: float = 0.0
    solution_valid: bool = True
    orchestration_errors: list[str] = field(default_factory=list)


def _read_terminal_bias(terminal: str) -> float:
    """Read the actual DEVSIM terminal parameter lazily.

    The lazy import keeps the strict CSV writer importable in unit tests that
    intentionally run without DEVSIM.
    """

    from devsim import get_parameter

    return float(
        get_parameter(
            device=memory_window.device,
            name=f"{terminal}_bias",
        )
    )


def _read_trap_density() -> float:
    """Read the actual region-scoped trapped-electron parameter."""

    from devsim import get_parameter
    from trap_models import TRAPPED_ELECTRON_PARAMETER

    return float(
        get_parameter(
            device=memory_window.device,
            region=tp.CHARGE_TRAP_REGION,
            name=TRAPPED_ELECTRON_PARAMETER,
        )
    )


def initialize_characterization_device() -> tuple[dict[str, Any], OperatingPoint]:
    """Build the existing device and solve its empty, zero-bias state once.

    Returns
    -------
    tuple
        The geometry dictionary returned by ``initialize_device`` and a fresh
        tracker representing the converged zero-bias, empty-trap solution.
    """

    geometry = initialize_device()
    validate_canonical_handoff_geometry(geometry)
    solve_empty_state()
    return geometry, OperatingPoint()


def validate_canonical_handoff_geometry(
    geometry: Mapping[str, Any],
) -> dict[str, float]:
    """Reject environment or caller overrides in final handoff runners.

    ``run_memory_window`` retains its historical environment-driven tunnel-
    oxide override for backward compatibility with exploratory scripts.  All
    state-characterization, Q/C, metadata, and mesh runners enter through this
    helper and must use the canonical 3/5/16-nm geometry instead.
    """

    canonical = memory_window.compact_parameters
    expected = {
        **canonical.geometry_dict(),
        **canonical.expected_outer_radii_nm(),
    }
    verified: dict[str, float] = {}
    for name, expected_value in expected.items():
        if name not in geometry:
            raise RuntimeError(
                f"Final handoff geometry is missing {name!r}."
            )
        actual_value = float(geometry[name])
        if not math.isfinite(actual_value) or not math.isclose(
            actual_value,
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                "Final handoff geometry must match the canonical source: "
                f"{name}={actual_value:.12e} nm, expected "
                f"{float(expected_value):.12e} nm."
            )
        verified[name] = actual_value
    return verified


def ramp_terminal(
    point: OperatingPoint,
    terminal: str,
    target_voltage: float,
    label: str | None = None,
) -> float:
    """Ramp gate or drain with the configured steps and update on success.

    The tracker is synchronized with the actual DEVSIM bias even after a
    partially completed ramp.  ``solution_valid`` separately records whether
    that coordinate has a converged solution.  A same-bias request is a no-op
    only when a preceding solve explicitly established a valid solution.
    """

    normalized_terminal = str(terminal).strip().lower()

    if normalized_terminal == "gate":
        tracker_attribute = "gate_voltage_V"
        initial_step = GATE_INITIAL_STEP_V
        minimum_step = GATE_MINIMUM_STEP_V
    elif normalized_terminal == "drain":
        tracker_attribute = "drain_voltage_V"
        initial_step = DRAIN_INITIAL_STEP_V
        minimum_step = DRAIN_MINIMUM_STEP_V
    else:
        raise ValueError(
            'terminal must be either "gate" or "drain". '
            f"Received: {terminal!r}"
        )

    target_voltage = float(target_voltage)
    if not math.isfinite(target_voltage):
        raise ValueError("target terminal voltage must be finite.")

    # Synchronize before every ramp.  The established adaptive routine can
    # converge intermediate steps and then raise after reaching its minimum
    # step; in that case the device parameter is no longer the original start.
    actual_start_voltage = _read_terminal_bias(normalized_terminal)
    setattr(point, tracker_attribute, actual_start_voltage)

    ramp_label = label or (
        f"{normalized_terminal} ramp to {target_voltage:+.6f} V"
    )

    if math.isclose(
        actual_start_voltage,
        target_voltage,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ) and point.solution_valid:
        converged_voltage = target_voltage
    elif math.isclose(
        actual_start_voltage,
        target_voltage,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        try:
            set_terminal_bias(normalized_terminal, target_voltage)
            solve_dc()
        except Exception:
            point.solution_valid = False
            raise
        converged_voltage = target_voltage
    else:
        try:
            converged_voltage = adaptive_voltage_ramp(
                terminal=normalized_terminal,
                start_voltage=actual_start_voltage,
                target_voltage=target_voltage,
                initial_step=initial_step,
                minimum_step=minimum_step,
                label=ramp_label,
            )
        except Exception:
            # Even on failure, retain the actual partially ramped/recovered
            # coordinate so the next reset starts from the real device bias.
            point.solution_valid = False
            setattr(
                point,
                tracker_attribute,
                _read_terminal_bias(normalized_terminal),
            )
            raise

    actual_converged_voltage = _read_terminal_bias(normalized_terminal)
    setattr(point, tracker_attribute, actual_converged_voltage)
    point.solution_valid = True
    return actual_converged_voltage


def ramp_trap_density(
    point: OperatingPoint,
    target_density_cm3: float,
    label: str | None = None,
) -> float:
    """Ramp trap density while tracking its coordinate and solution validity."""

    target_density_cm3 = float(target_density_cm3)
    if not math.isfinite(target_density_cm3) or target_density_cm3 < 0.0:
        raise ValueError(
            "target trapped-electron density must be finite and nonnegative."
        )

    actual_start_density_cm3 = _read_trap_density()
    point.trap_density_cm3 = actual_start_density_cm3

    ramp_label = label or (
        f"trap-density ramp to {target_density_cm3:.6e} cm^-3"
    )

    if (
        abs(target_density_cm3 - actual_start_density_cm3) <= 1.0
        and point.solution_valid
    ):
        converged_density_cm3 = target_density_cm3
    elif abs(target_density_cm3 - actual_start_density_cm3) <= 1.0:
        try:
            solve_dc()
        except Exception:
            point.solution_valid = False
            raise
        converged_density_cm3 = target_density_cm3
    else:
        try:
            converged_density_cm3 = adaptive_trap_density_ramp(
                start_density_cm3=actual_start_density_cm3,
                target_density_cm3=target_density_cm3,
                initial_step_cm3=TRAP_INITIAL_STEP_CM3,
                minimum_step_cm3=TRAP_MINIMUM_STEP_CM3,
                label=ramp_label,
            )
        except Exception:
            point.trap_density_cm3 = _read_trap_density()
            point.solution_valid = False
            raise

    actual_converged_density_cm3 = _read_trap_density()
    point.trap_density_cm3 = actual_converged_density_cm3
    point.solution_valid = True
    return actual_converged_density_cm3


def normalize_error_message(error: object | None) -> str:
    """Convert an exception or message to one compact, CSV-safe line."""

    if error is None:
        return ""

    message = " ".join(str(error).split())

    if isinstance(error, BaseException):
        error_type = type(error).__name__
        return f"{error_type}: {message}" if message else error_type

    return message


def build_iv_row(
    state: Mapping[str, Any],
    VGS: float,
    VDS: float,
    ID: float | None,
    converged: bool,
    error_message: object | None,
) -> dict[str, Any]:
    """Build one row using the required shared ID-VG/ID-VD CSV schema.

    A failed bias point always receives ``nan`` for both signed and absolute
    drain current.  Successful points preserve the original drain-current sign.
    Trap sheet density and charge density are obtained from the existing trap
    model conversion helper.
    """

    required_state_keys = {"state_index", "state", "ntrap_cm3"}
    missing_keys = required_state_keys.difference(state)
    if missing_keys:
        raise KeyError(
            "State definition is missing required key(s): "
            + ", ".join(sorted(missing_keys))
        )

    trap_information = get_trap_state_information(state["ntrap_cm3"])
    point_converged = bool(converged)
    normalized_error = normalize_error_message(error_message)

    if point_converged:
        try:
            drain_current_A = float(ID)
        except (TypeError, ValueError, OverflowError):
            point_converged = False
            drain_current_A = math.nan
            if not normalized_error:
                normalized_error = "invalid drain current"
        else:
            if not math.isfinite(drain_current_A):
                point_converged = False
                drain_current_A = math.nan
                if not normalized_error:
                    normalized_error = "non-finite drain current"
    else:
        drain_current_A = math.nan
        if not normalized_error:
            normalized_error = "bias point did not converge"

    absolute_drain_current_A = (
        abs(drain_current_A) if point_converged else math.nan
    )

    return {
        "state_index": int(state["state_index"]),
        "state": str(state["state"]),
        "ntrap_cm3": trap_information["trap_density_cm3"],
        "nsheet_cm2": trap_information["trap_sheet_density_cm2"],
        "trap_charge_density_C_cm3": trap_information[
            "trap_charge_density_C_cm3"
        ],
        "VGS_V": float(VGS),
        "VDS_V": float(VDS),
        "ID_A": drain_current_A,
        "abs_ID_A": absolute_drain_current_A,
        "converged": point_converged,
        "error_message": normalized_error,
    }


def write_standard_csv(
    output_path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Validate and write rows as a strict UTF-8 CSV file.

    Every row is validated before the destination is opened, preventing a late
    schema error from leaving a partially written interface file.
    """

    output_path = Path(output_path)
    ordered_fieldnames = tuple(fieldnames)

    if not ordered_fieldnames:
        raise ValueError("fieldnames must contain at least one column.")
    if len(set(ordered_fieldnames)) != len(ordered_fieldnames):
        raise ValueError("fieldnames must not contain duplicate columns.")

    expected_keys = set(ordered_fieldnames)
    validated_rows: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        actual_keys = set(row.keys())
        if actual_keys != expected_keys:
            missing_keys = sorted(expected_keys - actual_keys)
            extra_keys = sorted(actual_keys - expected_keys)
            raise ValueError(
                f"CSV row {row_index} does not match fieldnames; "
                f"missing={missing_keys}, extra={extra_keys}."
            )
        validated_rows.append(dict(row))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(mode="w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=ordered_fieldnames,
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(validated_rows)

    return output_path


# Familiar name for callers migrating from run_memory_window.write_csv.
write_csv = write_standard_csv


__all__ = [
    "IV_FIELDNAMES",
    "OperatingPoint",
    "adaptive_trap_density_ramp",
    "adaptive_voltage_ramp",
    "build_iv_row",
    "create_output_voltage_list",
    "get_drain_current",
    "initialize_characterization_device",
    "normalize_error_message",
    "ramp_terminal",
    "ramp_trap_density",
    "set_terminal_bias",
    "solve_dc",
    "validate_canonical_handoff_geometry",
    "write_csv",
    "write_standard_csv",
]
