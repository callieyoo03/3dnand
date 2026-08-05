"""Generate fixed-trap quasi-static gate-capacitance reference data.

Only derivatives of the insulated gate charge are supported: Cgg, Cgd, and
Cgs.  Every derivative uses converged central-difference endpoints and an
adaptive baseline restore for all five memory states at off/on bias.
Drain/source measured matrix entries are explicit NaNs rather than fabricated
channel-charge partitions.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import state_characterization_config as config
from state_sweep_helpers import (
    OperatingPoint,
    initialize_characterization_device,
    ramp_terminal,
    ramp_trap_density,
    solve_dc,
    write_standard_csv,
)
from terminal_charge import (
    CAPACITANCE_MATRIX_FIELDNAMES,
    CAPACITANCE_SUMMARY_FIELDNAMES,
    DEVICE_NAME,
    TERMINALS,
    adaptive_micro_ramp,
    build_capacitance_matrix_rows,
    build_capacitance_summary_row,
    measure_gate_capacitance,
    normalize_error_message,
    validate_capacitance_summary_diagnostics,
    validate_devsim_charge_runtime,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
DEFAULT_MATRIX_OUTPUT_PATH = (
    SHARED_DATA_DIRECTORY / "capacitance_matrix_by_state.csv"
)
DEFAULT_SUMMARY_OUTPUT_PATH = (
    SHARED_DATA_DIRECTORY / "capacitance_summary_by_state.csv"
)

REFERENCE_DRAIN_VOLTAGE_V = float(config.ION_VDS_V)
REFERENCE_BIASES = (
    ("off", float(config.IOFF_VGS_V)),
    ("on", float(config.ION_VGS_V)),
)
DEFAULT_DELTA_VOLTAGES_V = (0.0005, 0.001, 0.002)
DEFAULT_NOMINAL_DELTA_VOLTAGE_V = 0.001


def reference_states() -> tuple[Mapping[str, Any], ...]:
    states = tuple(config.MEMORY_STATES)
    if len(states) != 5:
        raise RuntimeError(
            f"exactly five memory states are required; received {len(states)}."
        )
    return states


def parse_delta_voltages(text: str) -> tuple[float, ...]:
    """Parse a comma-separated, unique set of positive perturbation voltages."""

    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "delta voltages must be comma-separated numbers"
        ) from error
    if not values:
        raise argparse.ArgumentTypeError("at least one delta voltage is required")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("delta voltages must be finite and positive")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("delta voltages must be unique")
    return tuple(sorted(values))


def _prepare_state_at_read_drain(
    point: OperatingPoint,
    state: Mapping[str, Any],
) -> None:
    adaptive_micro_ramp(
        "source",
        0.0,
        DEFAULT_NOMINAL_DELTA_VOLTAGE_V,
        label="capacitance source reset",
    )
    ramp_terminal(point, "gate", 0.0, label="capacitance gate reset")
    ramp_terminal(point, "drain", 0.0, label="capacitance drain reset")
    ramp_trap_density(
        point,
        float(state["ntrap_cm3"]),
        label=f"capacitance {state['state']} trap ramp",
    )
    ramp_terminal(
        point,
        "drain",
        REFERENCE_DRAIN_VOLTAGE_V,
        label=f"capacitance {state['state']} drain read ramp",
    )


def _restore_reference_base(
    point: OperatingPoint,
    *,
    gate_voltage_V: float,
    delta_voltage_V: float,
) -> None:
    """Recover all three absolute terminal biases after a failed perturbation."""

    adaptive_micro_ramp(
        "source",
        0.0,
        delta_voltage_V,
        label="capacitance source recovery",
    )
    ramp_terminal(
        point,
        "drain",
        REFERENCE_DRAIN_VOLTAGE_V,
        label="capacitance drain recovery",
    )
    ramp_terminal(
        point,
        "gate",
        gate_voltage_V,
        label="capacitance gate recovery",
    )
    if not point.solution_valid:
        solve_dc()
        point.solution_valid = True


def _failed_gate_results(
    error: object,
    delta_voltages_V: Sequence[float],
) -> dict[tuple[str, float], dict[str, Any]]:
    message = normalize_error_message(error)
    return {
        (terminal, float(delta)): {
            "perturbed_terminal": terminal,
            "delta_voltage_V": float(delta),
            "capacitance_F": math.nan,
            "converged": False,
            "error_message": message,
        }
        for delta in delta_voltages_V
        for terminal in TERMINALS
    }


def _measure_reference_capacitances(
    point: OperatingPoint,
    state: Mapping[str, Any],
    *,
    gate_voltage_V: float,
    delta_voltages_V: Sequence[float],
) -> dict[tuple[str, float], dict[str, Any]]:
    results: dict[tuple[str, float], dict[str, Any]] = {}
    for delta_voltage_V in delta_voltages_V:
        for terminal in TERMINALS:
            result = measure_gate_capacitance(
                perturbed_terminal=terminal,
                base_voltage_V={
                    "gate": gate_voltage_V,
                    "drain": REFERENCE_DRAIN_VOLTAGE_V,
                    "source": 0.0,
                }[terminal],
                delta_voltage_V=float(delta_voltage_V),
                expected_trap_density_cm3=float(state["ntrap_cm3"]),
                device=DEVICE_NAME,
            )
            results[(terminal, float(delta_voltage_V))] = result
            if not bool(result["converged"]):
                point.solution_valid = False
                try:
                    _restore_reference_base(
                        point,
                        gate_voltage_V=gate_voltage_V,
                        delta_voltage_V=float(delta_voltage_V),
                    )
                except Exception as recovery_error:
                    recovery_message = normalize_error_message(recovery_error)
                    result["error_message"] = (
                        str(result["error_message"])
                        + "; reference recovery failed: "
                        + recovery_message
                    ).strip("; ")
                    return {
                        **results,
                        **{
                            key: value
                            for key, value in _failed_gate_results(
                                result["error_message"], delta_voltages_V
                            ).items()
                            if key not in results
                        },
                    }
    return results


def run_capacitance_references(
    point: OperatingPoint,
    *,
    delta_voltages_V: Sequence[float] = DEFAULT_DELTA_VOLTAGES_V,
    nominal_delta_voltage_V: float = DEFAULT_NOMINAL_DELTA_VOLTAGE_V,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all five fixed-trap states at off/on reference biases."""

    deltas = tuple(float(value) for value in delta_voltages_V)
    nominal = float(nominal_delta_voltage_V)
    if nominal not in deltas:
        raise ValueError("the nominal delta voltage must be in the sensitivity set.")

    matrix_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for state in reference_states():
        try:
            _prepare_state_at_read_drain(point, state)
            state_setup_error: Exception | None = None
        except Exception as error:
            state_setup_error = error

        for bias_name, gate_voltage_V in REFERENCE_BIASES:
            if state_setup_error is not None:
                gate_results = _failed_gate_results(state_setup_error, deltas)
            else:
                try:
                    ramp_terminal(
                        point,
                        "gate",
                        gate_voltage_V,
                        label=f"capacitance {state['state']} {bias_name} gate ramp",
                    )
                    gate_results = _measure_reference_capacitances(
                        point,
                        state,
                        gate_voltage_V=gate_voltage_V,
                        delta_voltages_V=deltas,
                    )
                except Exception as error:
                    point.solution_valid = False
                    gate_results = _failed_gate_results(error, deltas)

            matrix_rows.extend(
                build_capacitance_matrix_rows(
                    state,
                    VGS_V=gate_voltage_V,
                    VDS_V=REFERENCE_DRAIN_VOLTAGE_V,
                    gate_results=gate_results,
                )
            )
            summary_rows.append(
                build_capacitance_summary_row(
                    state,
                    VGS_V=gate_voltage_V,
                    VDS_V=REFERENCE_DRAIN_VOLTAGE_V,
                    gate_results=gate_results,
                    nominal_delta_voltage_V=nominal,
                    sensitivity_delta_voltages_V=deltas,
                )
            )

    matrix_rows.sort(
        key=lambda row: (
            int(row["state_index"]),
            float(row["VGS_V"]),
            float(row["delta_voltage_V"]),
            TERMINALS.index(str(row["perturbed_terminal"])),
            TERMINALS.index(str(row["measured_terminal"])),
        )
    )
    summary_rows.sort(
        key=lambda row: (int(row["state_index"]), float(row["VGS_V"]))
    )
    return matrix_rows, summary_rows


def _best_effort_zero_reset(point: OperatingPoint) -> list[str]:
    errors: list[str] = []
    try:
        adaptive_micro_ramp(
            "source",
            0.0,
            DEFAULT_NOMINAL_DELTA_VOLTAGE_V,
            label="capacitance final source reset",
        )
    except Exception as error:
        errors.append(normalize_error_message(error))
    for terminal in ("gate", "drain"):
        try:
            ramp_terminal(
                point,
                terminal,
                0.0,
                label=f"capacitance final {terminal} reset",
            )
        except Exception as error:
            errors.append(normalize_error_message(error))
    try:
        ramp_trap_density(point, 0.0, label="capacitance final trap reset")
    except Exception as error:
        errors.append(normalize_error_message(error))
    return errors


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate five-state fixed-Ntrap Cgg/Cgd/Cgs central-difference "
            "data at off/on reference biases."
        )
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=DEFAULT_MATRIX_OUTPUT_PATH,
        help=f"matrix CSV path (default: {DEFAULT_MATRIX_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help=f"summary CSV path (default: {DEFAULT_SUMMARY_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--delta-voltages",
        type=parse_delta_voltages,
        default=DEFAULT_DELTA_VOLTAGES_V,
        metavar="V1,V2,...",
        help="central-difference sensitivity voltages (default: 0.0005,0.001,0.002)",
    )
    parser.add_argument(
        "--nominal-delta",
        type=float,
        default=DEFAULT_NOMINAL_DELTA_VOLTAGE_V,
        metavar="V",
        help="nominal central-difference voltage (default: 0.001)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.nominal_delta not in tuple(args.delta_voltages):
        raise SystemExit("--nominal-delta must appear in --delta-voltages")

    _, point = initialize_characterization_device()
    runtime = validate_devsim_charge_runtime(device=DEVICE_NAME)
    print(
        "Capacitance runtime: "
        f"r={runtime['raxis_variable']}, "
        f"node_volume={runtime['node_volume_model']}, "
        f"edge_couple={runtime['edge_couple_model']}"
    )
    matrix_rows, summary_rows = run_capacitance_references(
        point,
        delta_voltages_V=args.delta_voltages,
        nominal_delta_voltage_V=args.nominal_delta,
    )
    reset_errors = _best_effort_zero_reset(point)

    write_standard_csv(
        args.matrix_output,
        CAPACITANCE_MATRIX_FIELDNAMES,
        matrix_rows,
    )
    write_standard_csv(
        args.summary_output,
        CAPACITANCE_SUMMARY_FIELDNAMES,
        summary_rows,
    )
    diagnostic_report = validate_capacitance_summary_diagnostics(summary_rows)
    failed = [row for row in summary_rows if not bool(row["converged"])]
    print(
        f"Capacitance references: {len(summary_rows) - len(failed)} converged, "
        f"{len(failed)} failed"
    )
    print(f"Capacitance matrix CSV: {Path(args.matrix_output).resolve()}")
    print(f"Capacitance summary CSV: {Path(args.summary_output).resolve()}")
    print(
        "Capacitance diagnostics: "
        f"max sensitivity={diagnostic_report['maximum_relative_sensitivity']:.6e}, "
        f"max gate-row sum={diagnostic_report['maximum_gate_row_sum_F']:.6e} F"
    )
    for error in reset_errors:
        print(f"Reset warning: {error}")
    return 0 if not failed and not reset_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
