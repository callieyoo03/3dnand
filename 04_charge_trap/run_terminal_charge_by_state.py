"""Generate physically supported terminal-charge data at reference biases.

All five configured memory states are evaluated at off/on bias.  Source and
drain compact terminal charges intentionally remain NaN; their raw Poisson
boundary fluxes are retained only so the discrete cylindrical Gauss law can be
audited.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import state_characterization_config as config
from state_sweep_helpers import (
    OperatingPoint,
    initialize_characterization_device,
    ramp_terminal,
    ramp_trap_density,
    write_standard_csv,
)
from terminal_charge import (
    DEVICE_NAME,
    TERMINAL_CHARGE_FIELDNAMES,
    build_failed_terminal_charge_row,
    build_terminal_charge_row,
    charge_conservation_within_tolerance,
    extract_terminal_charge_components,
    normalize_error_message,
    validate_devsim_charge_runtime,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
DEFAULT_OUTPUT_PATH = REPOSITORY_ROOT / "shared_data" / "terminal_charge_by_state.csv"

REFERENCE_DRAIN_VOLTAGE_V = float(config.ION_VDS_V)
REFERENCE_BIASES = (
    ("off", float(config.IOFF_VGS_V)),
    ("on", float(config.ION_VGS_V)),
)


def reference_states() -> tuple[Mapping[str, Any], ...]:
    """Return all five configured static memory-state definitions."""

    states = tuple(config.MEMORY_STATES)
    if len(states) != 5:
        raise RuntimeError(
            f"exactly five memory states are required; received {len(states)}."
        )
    return states


def _prepare_state_at_read_drain(
    point: OperatingPoint,
    state: Mapping[str, Any],
) -> None:
    """Reach one fixed-Ntrap state at VGS=0 and the read drain voltage."""

    ramp_terminal(point, "gate", 0.0, label="terminal-charge gate reset")
    ramp_terminal(point, "drain", 0.0, label="terminal-charge drain reset")
    ramp_trap_density(
        point,
        float(state["ntrap_cm3"]),
        label=f"terminal-charge {state['state']} trap ramp",
    )
    ramp_terminal(
        point,
        "drain",
        REFERENCE_DRAIN_VOLTAGE_V,
        label=f"terminal-charge {state['state']} drain read ramp",
    )


def run_terminal_charge_references(
    point: OperatingPoint,
    *,
    absolute_tolerance_C: float = 1.0e-24,
    relative_tolerance: float = 1.0e-8,
) -> list[dict[str, Any]]:
    """Evaluate all five memory states at off/on reference biases."""

    rows: list[dict[str, Any]] = []
    for state in reference_states():
        try:
            _prepare_state_at_read_drain(point, state)
        except Exception as error:
            for _, gate_voltage_V in REFERENCE_BIASES:
                rows.append(
                    build_failed_terminal_charge_row(
                        state,
                        VGS_V=gate_voltage_V,
                        VDS_V=REFERENCE_DRAIN_VOLTAGE_V,
                        error=error,
                    )
                )
            continue

        for bias_name, gate_voltage_V in REFERENCE_BIASES:
            try:
                ramp_terminal(
                    point,
                    "gate",
                    gate_voltage_V,
                    label=(
                        f"terminal-charge {state['state']} {bias_name} gate ramp"
                    ),
                )
                components = extract_terminal_charge_components(device=DEVICE_NAME)
                if not charge_conservation_within_tolerance(
                    components,
                    absolute_tolerance_C=absolute_tolerance_C,
                    relative_tolerance=relative_tolerance,
                ):
                    raise RuntimeError(
                        "discrete charge-conservation residual exceeds tolerance: "
                        f"{components['charge_sum_C']:.12e} C"
                    )
                row = build_terminal_charge_row(
                    state,
                    VGS_V=gate_voltage_V,
                    VDS_V=REFERENCE_DRAIN_VOLTAGE_V,
                    components=components,
                )
            except Exception as error:
                point.solution_valid = False
                row = build_failed_terminal_charge_row(
                    state,
                    VGS_V=gate_voltage_V,
                    VDS_V=REFERENCE_DRAIN_VOLTAGE_V,
                    error=error,
                )
            rows.append(row)

    return sorted(
        rows,
        key=lambda row: (int(row["state_index"]), float(row["VGS_V"])),
    )


def _best_effort_zero_reset(point: OperatingPoint) -> list[str]:
    errors: list[str] = []
    for terminal in ("gate", "drain"):
        try:
            ramp_terminal(
                point,
                terminal,
                0.0,
                label=f"terminal-charge final {terminal} reset",
            )
        except Exception as error:
            errors.append(normalize_error_message(error))
    try:
        ramp_trap_density(point, 0.0, label="terminal-charge final trap reset")
    except Exception as error:
        errors.append(normalize_error_message(error))
    return errors


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate five-state off/on gate charge and cylindrical "
            "volume-charge diagnostics."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--charge-absolute-tolerance",
        type=float,
        default=1.0e-24,
        metavar="C",
        help="absolute discrete Gauss-law tolerance in coulombs",
    )
    parser.add_argument(
        "--charge-relative-tolerance",
        type=float,
        default=1.0e-8,
        help="relative discrete Gauss-law tolerance",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    _, point = initialize_characterization_device()
    runtime = validate_devsim_charge_runtime(device=DEVICE_NAME)
    print(
        "Terminal-charge runtime: "
        f"r={runtime['raxis_variable']}, "
        f"node_volume={runtime['node_volume_model']}, "
        f"edge_couple={runtime['edge_couple_model']}"
    )

    rows = run_terminal_charge_references(
        point,
        absolute_tolerance_C=args.charge_absolute_tolerance,
        relative_tolerance=args.charge_relative_tolerance,
    )
    reset_errors = _best_effort_zero_reset(point)
    write_standard_csv(args.output, TERMINAL_CHARGE_FIELDNAMES, rows)

    failed_rows = [row for row in rows if not bool(row["converged"])]
    print(
        f"Terminal-charge points: {len(rows) - len(failed_rows)} converged, "
        f"{len(failed_rows)} failed"
    )
    print(f"Terminal-charge CSV: {Path(args.output).resolve()}")
    for error in reset_errors:
        print(f"Reset warning: {error}")
    return 0 if not failed_rows and not reset_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
