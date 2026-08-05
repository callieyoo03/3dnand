"""Configuration for static memory-state electrical characterization.

This module deliberately has no DEVSIM dependency.  It is the single source of
truth for the state definitions, bias sweeps, metric extraction settings, CSV
schemas, and output paths used by the state-characterization runners.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence, TypedDict

import trap_parameters as trap_parameters


class MemoryState(TypedDict):
    """A fixed trapped-electron-density state used for read characterization."""

    state_index: int
    state: str
    ntrap_cm3: float


# ---------------------------------------------------------------------------
# Repository and output paths
# ---------------------------------------------------------------------------

MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
STATE_CHARACTERIZATION_RESULTS_DIRECTORY = (
    MODULE_DIRECTORY
    / "results"
    / "state_characterization_final_3_5_16nm"
)

LEGACY_SNAPSHOT_DIRECTORY = (
    STATE_CHARACTERIZATION_RESULTS_DIRECTORY / "legacy_4_5_8nm_k9_20"
)

IDVG_BY_STATE_CSV_PATH = SHARED_DATA_DIRECTORY / "idvg_by_state.csv"
IDVD_BY_STATE_CSV_PATH = SHARED_DATA_DIRECTORY / "idvd_by_state.csv"
METRICS_BY_STATE_CSV_PATH = SHARED_DATA_DIRECTORY / "metrics_by_state.csv"
MEMORY_STATE_MAP_CSV_PATH = SHARED_DATA_DIRECTORY / "memory_state_map.csv"
BASELINE_COMPARISON_CSV_PATH = (
    SHARED_DATA_DIRECTORY / "baseline_comparison_legacy_vs_final.csv"
)

# These hashes identify the verified legacy bundle committed at 0d7d917.  They
# are used only to protect the old data before a validated candidate replaces
# the four canonical files; they are not fitting or physics parameters.
LEGACY_SHARED_DATA_SHA256 = {
    "idvg_by_state.csv": (
        "c5ba12034c4cde9f6835f94712527367d6972c2e8b641f5a8740bdd8b06fc808"
    ),
    "idvd_by_state.csv": (
        "107e3c115c1f2ec818b8f0afdfb1dab80a3b6e8ec454766402ada649c80603f5"
    ),
    "metrics_by_state.csv": (
        "9d7aaa4097703f832fd97aaf92abd848768d105e9631ca6d19aa300d01ebca22"
    ),
    "memory_state_map.csv": (
        "fc0ef0f30d5f3a490ef858346a8070632e4c96061e5b95f6eb3ea169ce2224a2"
    ),
}


# ---------------------------------------------------------------------------
# Static memory states
#
# The density values and density conversions remain owned by trap_parameters.
# Only the final state label is made more explicit for this characterization
# interface, as required by the standard CSV contract.
# ---------------------------------------------------------------------------

CHARGE_TRAP_THICKNESS_CM = float(trap_parameters.charge_trap_thickness)
ELEMENTARY_CHARGE_C = float(trap_parameters.q)

_STATE_NAMES = (
    "State_0_Empty",
    "State_1",
    "State_2",
    "State_3",
    "State_4_Programmed",
)

MEMORY_STATES: tuple[MemoryState, ...] = tuple(
    {
        "state_index": state_index,
        "state": state_name,
        "ntrap_cm3": float(ntrap_cm3),
    }
    for state_index, (state_name, ntrap_cm3) in enumerate(
        zip(
            _STATE_NAMES,
            trap_parameters.TRAPPED_ELECTRON_DENSITY_STATES,
        )
    )
)


# ---------------------------------------------------------------------------
# Full ID-VG sweep
# ---------------------------------------------------------------------------

IDVG_VDS_VALUES_V = (
    0.01,
    float(trap_parameters.DRAIN_VOLTAGE),
    0.10,
)
IDVG_VGS_START_V = float(trap_parameters.GATE_START_VOLTAGE)
IDVG_VGS_STOP_V = float(trap_parameters.GATE_STOP_VOLTAGE)
IDVG_VGS_STEP_V = float(trap_parameters.GATE_OUTPUT_STEP)

# A failed Newton recovery may need several deterministic attempts before the
# output point can be solved again.  The runner records a failed row after this
# bounded limit instead of looping indefinitely or inventing a current value.
IDVG_POINT_MAX_ATTEMPTS = 8


# ---------------------------------------------------------------------------
# Full ID-VD sweep
# ---------------------------------------------------------------------------

IDVD_VGS_VALUES_V = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
IDVD_VDS_START_V = 0.0
IDVD_VDS_STOP_V = 0.10
IDVD_VDS_STEP_V = 0.01


# ---------------------------------------------------------------------------
# Metric extraction settings
#
# The SS window is an explicit initial analysis setting.  It may be adjusted
# after inspecting simulated ID-VG data, but extraction code must not silently
# tune it to make a fit succeed.
# ---------------------------------------------------------------------------

VTH_TARGET_CURRENT_A = float(trap_parameters.THRESHOLD_CURRENT)

ION_VGS_V = 3.0
ION_VDS_V = float(trap_parameters.DRAIN_VOLTAGE)

IOFF_VGS_V = -1.0
IOFF_VDS_V = float(trap_parameters.DRAIN_VOLTAGE)

CURRENT_FLOOR_A = 1.0e-30

# The final handoff contract fixes the regression window at 1.0e-18 A through
# 1.0e-10 A.  The lower limit remains well above the 1.0e-30 numeric floor.
SS_CURRENT_MIN_A = 1.0e-18
SS_CURRENT_MAX_A = 1.0e-10
SS_MINIMUM_POINT_COUNT = 4


# ---------------------------------------------------------------------------
# Smoke-test subset and coarse sweeps
# ---------------------------------------------------------------------------

SMOKE_STATE_INDICES = (0, 4)
SMOKE_MEMORY_STATES: tuple[MemoryState, ...] = tuple(
    MEMORY_STATES[state_index] for state_index in SMOKE_STATE_INDICES
)

SMOKE_IDVG_VDS_VALUES_V = (float(trap_parameters.DRAIN_VOLTAGE),)
SMOKE_IDVG_VGS_STEP_V = 0.125

SMOKE_IDVD_VGS_VALUES_V = (0.0, 1.5, 3.0)
SMOKE_IDVD_VDS_STEP_V = 0.05


# ---------------------------------------------------------------------------
# Stable CSV schemas.  Tuple order is the public interface.
# ---------------------------------------------------------------------------

IDVG_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "trap_charge_density_C_cm3",
    "VGS_V",
    "VDS_V",
    "ID_A",
    "abs_ID_A",
    "converged",
    "error_message",
)

IDVD_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "trap_charge_density_C_cm3",
    "VGS_V",
    "VDS_V",
    "ID_A",
    "abs_ID_A",
    "converged",
    "error_message",
)

METRICS_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VDS_V",
    "Vth_V",
    "threshold_current_A",
    "vth_success",
    "vth_error",
    "SS_mV_dec",
    "ss_r_squared",
    "ss_point_count",
    "ss_success",
    "ss_error",
    "Ion_A",
    "ion_vgs_V",
    "ion_vds_V",
    "Ioff_A",
    "ioff_vgs_V",
    "ioff_vds_V",
    "on_off_ratio",
    "gm_max_S",
    "VGS_at_gm_max_V",
    "ss_current_min_A",
    "ss_current_max_A",
    "ss_minimum_point_count",
    "current_floor_A",
)

MEMORY_STATE_MAP_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "qsheet_C_cm2",
    "VDS_V",
    "Vth_V",
    "delta_Vth_from_empty_V",
)

BASELINE_COMPARISON_FIELDNAMES = (
    "state_index",
    "state",
    "VDS_V",
    "legacy_Vth_V",
    "final_Vth_V",
    "Vth_difference_V",
    "legacy_SS_mV_dec",
    "final_SS_mV_dec",
    "legacy_Ion_A",
    "final_Ion_A",
    "legacy_Ioff_A",
    "final_Ioff_A",
    "legacy_on_off_ratio",
    "final_on_off_ratio",
    "legacy_gm_max_S",
    "final_gm_max_S",
    "legacy_condition",
    "final_condition",
    "ioff_on_off_warning",
)


def volume_to_sheet_density(ntrap_cm3: float) -> float:
    """Convert trapped-electron volume density from cm^-3 to cm^-2."""

    return float(trap_parameters.volume_to_sheet_density(ntrap_cm3))


def electron_density_to_charge_density(ntrap_cm3: float) -> float:
    """Convert electron number density in cm^-3 to signed C/cm^3."""

    return float(
        trap_parameters.electron_density_to_charge_density(ntrap_cm3)
    )


def electron_sheet_density_to_charge_density(nsheet_cm2: float) -> float:
    """Convert electron sheet density in cm^-2 to signed C/cm^2."""

    return -ELEMENTARY_CHARGE_C * float(nsheet_cm2)


def _require_finite(name: str, value: object) -> float:
    """Return *value* as float, rejecting booleans and non-finite values."""

    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number, not a boolean.")

    try:
        numeric_value = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number.") from error

    if not math.isfinite(numeric_value):
        raise ValueError(f"{name} must be finite.")

    return numeric_value


def _validate_strictly_increasing(
    name: str,
    values: Sequence[float],
) -> tuple[float, ...]:
    """Validate a non-empty, finite, strictly increasing numeric sequence."""

    if not values:
        raise ValueError(f"{name} must not be empty.")

    finite_values = tuple(
        _require_finite(f"{name}[{index}]", value)
        for index, value in enumerate(values)
    )

    if any(
        right <= left
        for left, right in zip(finite_values, finite_values[1:])
    ):
        raise ValueError(f"{name} must be strictly increasing.")

    return finite_values


def _validate_uniform_sweep(
    name: str,
    start: float,
    stop: float,
    step: float,
) -> None:
    """Validate finite ordered bounds and a positive endpoint-aligned step."""

    finite_start = _require_finite(f"{name} start", start)
    finite_stop = _require_finite(f"{name} stop", stop)
    finite_step = _require_finite(f"{name} step", step)

    if finite_stop <= finite_start:
        raise ValueError(f"{name} stop must be greater than its start.")
    if finite_step <= 0.0:
        raise ValueError(f"{name} step must be positive.")

    interval_count = (finite_stop - finite_start) / finite_step
    if not math.isclose(
        interval_count,
        round(interval_count),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{name} step must land exactly on the stop value.")


def _contains_voltage(values: Iterable[float], target: float) -> bool:
    """Return whether a bias tuple contains *target* within float tolerance."""

    return any(
        math.isclose(float(value), target, rel_tol=0.0, abs_tol=1.0e-12)
        for value in values
    )


def validate_state_characterization_config() -> None:
    """Validate state, sweep, extraction, path, and CSV-schema invariants."""

    expected_state_names = (
        "State_0_Empty",
        "State_1",
        "State_2",
        "State_3",
        "State_4_Programmed",
    )
    expected_densities_cm3 = (0.0, 2.0e17, 5.0e17, 1.0e18, 2.0e18)
    expected_indices = tuple(range(5))

    if len(MEMORY_STATES) != 5:
        raise ValueError("MEMORY_STATES must contain exactly five states.")

    for state in MEMORY_STATES:
        if tuple(state) != ("state_index", "state", "ntrap_cm3"):
            raise ValueError(
                "Each memory state must contain only state_index, state, "
                "and ntrap_cm3 in that order."
            )

    actual_indices = tuple(state["state_index"] for state in MEMORY_STATES)
    if actual_indices != expected_indices or any(
        isinstance(index, bool) or not isinstance(index, int)
        for index in actual_indices
    ):
        raise ValueError("Memory-state indices must be exactly 0, 1, 2, 3, 4.")

    actual_names = tuple(state["state"] for state in MEMORY_STATES)
    if actual_names != expected_state_names:
        raise ValueError(
            "Memory-state names must match the standard state labels exactly."
        )
    if MEMORY_STATES[0]["state"] != "State_0_Empty":
        raise ValueError("The first memory state must be State_0_Empty.")

    actual_densities_cm3 = _validate_strictly_increasing(
        "memory-state ntrap_cm3",
        tuple(state["ntrap_cm3"] for state in MEMORY_STATES),
    )
    if actual_densities_cm3 != expected_densities_cm3:
        raise ValueError(
            "Memory-state trap densities must match the five specified values."
        )
    if actual_densities_cm3[0] != 0.0:
        raise ValueError("The first (empty) memory state must have zero charge.")

    if not math.isclose(
        CHARGE_TRAP_THICKNESS_CM,
        float(trap_parameters.charge_trap_thickness),
        rel_tol=0.0,
        abs_tol=0.0,
    ) or CHARGE_TRAP_THICKNESS_CM <= 0.0:
        raise ValueError("Charge-trap thickness must be finite and positive.")
    _require_finite("CHARGE_TRAP_THICKNESS_CM", CHARGE_TRAP_THICKNESS_CM)
    if _require_finite("ELEMENTARY_CHARGE_C", ELEMENTARY_CHARGE_C) <= 0.0:
        raise ValueError("ELEMENTARY_CHARGE_C must be positive.")

    idvg_vds_values = _validate_strictly_increasing(
        "IDVG_VDS_VALUES_V", IDVG_VDS_VALUES_V
    )
    if idvg_vds_values[0] < 0.0:
        raise ValueError("IDVG drain biases must not be negative.")
    _validate_uniform_sweep(
        "IDVG VGS sweep",
        IDVG_VGS_START_V,
        IDVG_VGS_STOP_V,
        IDVG_VGS_STEP_V,
    )
    if (
        isinstance(IDVG_POINT_MAX_ATTEMPTS, bool)
        or not isinstance(IDVG_POINT_MAX_ATTEMPTS, int)
        or IDVG_POINT_MAX_ATTEMPTS < 1
    ):
        raise ValueError("IDVG_POINT_MAX_ATTEMPTS must be a positive integer.")

    idvd_vgs_values = _validate_strictly_increasing(
        "IDVD_VGS_VALUES_V", IDVD_VGS_VALUES_V
    )
    if (
        idvd_vgs_values[0] < IDVG_VGS_START_V
        or idvd_vgs_values[-1] > IDVG_VGS_STOP_V
    ):
        raise ValueError("IDVD gate biases must lie within the IDVG gate range.")
    _validate_uniform_sweep(
        "IDVD VDS sweep",
        IDVD_VDS_START_V,
        IDVD_VDS_STOP_V,
        IDVD_VDS_STEP_V,
    )
    if IDVD_VDS_START_V < 0.0:
        raise ValueError("IDVD_VDS_START_V must not be negative.")

    threshold_current = _require_finite(
        "VTH_TARGET_CURRENT_A", VTH_TARGET_CURRENT_A
    )
    current_floor = _require_finite("CURRENT_FLOOR_A", CURRENT_FLOOR_A)
    ss_current_min = _require_finite("SS_CURRENT_MIN_A", SS_CURRENT_MIN_A)
    ss_current_max = _require_finite("SS_CURRENT_MAX_A", SS_CURRENT_MAX_A)
    if not 0.0 < current_floor < ss_current_min < ss_current_max:
        raise ValueError(
            "Current settings must satisfy 0 < floor < SS minimum < SS maximum."
        )
    if threshold_current <= 0.0:
        raise ValueError("VTH_TARGET_CURRENT_A must be positive.")
    if ss_current_max > threshold_current:
        raise ValueError(
            "SS_CURRENT_MAX_A must not exceed VTH_TARGET_CURRENT_A."
        )
    if (
        isinstance(SS_MINIMUM_POINT_COUNT, bool)
        or not isinstance(SS_MINIMUM_POINT_COUNT, int)
        or SS_MINIMUM_POINT_COUNT < 2
    ):
        raise ValueError("SS_MINIMUM_POINT_COUNT must be an integer of at least 2.")

    for name, gate_bias in (
        ("ION_VGS_V", ION_VGS_V),
        ("IOFF_VGS_V", IOFF_VGS_V),
    ):
        finite_gate_bias = _require_finite(name, gate_bias)
        if not IDVG_VGS_START_V <= finite_gate_bias <= IDVG_VGS_STOP_V:
            raise ValueError(f"{name} must lie within the IDVG gate sweep.")
    if ION_VGS_V <= IOFF_VGS_V:
        raise ValueError("ION_VGS_V must be greater than IOFF_VGS_V.")

    for name, drain_bias in (
        ("ION_VDS_V", ION_VDS_V),
        ("IOFF_VDS_V", IOFF_VDS_V),
    ):
        finite_drain_bias = _require_finite(name, drain_bias)
        if not _contains_voltage(idvg_vds_values, finite_drain_bias):
            raise ValueError(f"{name} must be one of the IDVG drain biases.")

    if SMOKE_STATE_INDICES != (0, 4):
        raise ValueError("Smoke states must be exactly empty and fully programmed.")
    if tuple(state["state_index"] for state in SMOKE_MEMORY_STATES) != (0, 4):
        raise ValueError("SMOKE_MEMORY_STATES must match SMOKE_STATE_INDICES.")

    smoke_idvg_vds_values = _validate_strictly_increasing(
        "SMOKE_IDVG_VDS_VALUES_V", SMOKE_IDVG_VDS_VALUES_V
    )
    if any(
        not _contains_voltage(idvg_vds_values, value)
        for value in smoke_idvg_vds_values
    ):
        raise ValueError("Smoke IDVG drain biases must be a full-sweep subset.")
    _validate_uniform_sweep(
        "smoke IDVG VGS sweep",
        IDVG_VGS_START_V,
        IDVG_VGS_STOP_V,
        SMOKE_IDVG_VGS_STEP_V,
    )
    if SMOKE_IDVG_VGS_STEP_V < IDVG_VGS_STEP_V:
        raise ValueError("Smoke IDVG step must be at least the full-sweep step.")

    smoke_idvd_vgs_values = _validate_strictly_increasing(
        "SMOKE_IDVD_VGS_VALUES_V", SMOKE_IDVD_VGS_VALUES_V
    )
    if any(
        not _contains_voltage(idvd_vgs_values, value)
        for value in smoke_idvd_vgs_values
    ):
        raise ValueError("Smoke IDVD gate biases must be a full-sweep subset.")
    _validate_uniform_sweep(
        "smoke IDVD VDS sweep",
        IDVD_VDS_START_V,
        IDVD_VDS_STOP_V,
        SMOKE_IDVD_VDS_STEP_V,
    )
    if SMOKE_IDVD_VDS_STEP_V < IDVD_VDS_STEP_V:
        raise ValueError("Smoke IDVD step must be at least the full-sweep step.")

    expected_paths = {
        IDVG_BY_STATE_CSV_PATH: "idvg_by_state.csv",
        IDVD_BY_STATE_CSV_PATH: "idvd_by_state.csv",
        METRICS_BY_STATE_CSV_PATH: "metrics_by_state.csv",
        MEMORY_STATE_MAP_CSV_PATH: "memory_state_map.csv",
        BASELINE_COMPARISON_CSV_PATH: (
            "baseline_comparison_legacy_vs_final.csv"
        ),
    }
    for csv_path, expected_name in expected_paths.items():
        if csv_path.parent != SHARED_DATA_DIRECTORY or csv_path.name != expected_name:
            raise ValueError(f"Invalid shared-data CSV path: {csv_path}")

    for schema_name, fieldnames in (
        ("IDVG_FIELDNAMES", IDVG_FIELDNAMES),
        ("IDVD_FIELDNAMES", IDVD_FIELDNAMES),
        ("METRICS_FIELDNAMES", METRICS_FIELDNAMES),
        ("MEMORY_STATE_MAP_FIELDNAMES", MEMORY_STATE_MAP_FIELDNAMES),
        (
            "BASELINE_COMPARISON_FIELDNAMES",
            BASELINE_COMPARISON_FIELDNAMES,
        ),
    ):
        if not isinstance(fieldnames, tuple) or not fieldnames:
            raise ValueError(f"{schema_name} must be a non-empty tuple.")
        if any(not isinstance(field, str) or not field for field in fieldnames):
            raise ValueError(f"{schema_name} must contain non-empty strings.")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{schema_name} must not contain duplicate fields.")


# Fail fast if this public configuration is edited into an inconsistent state.
validate_state_characterization_config()
