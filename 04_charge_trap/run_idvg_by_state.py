"""Run static trapped-charge ID-VG sweeps.

Run the full standalone sweep from the repository root with::

    python 04_charge_trap/run_idvg_by_state.py

The integrated CLI in ``run_state_characterization.py`` reuses
``run_idvg_by_state`` so that the DEVSIM device is initialized only once.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import state_characterization_config as config
from state_sweep_helpers import (
    OperatingPoint,
    build_iv_row,
    create_output_voltage_list,
    get_drain_current,
    initialize_characterization_device,
    normalize_error_message,
    ramp_terminal,
    ramp_trap_density,
    set_terminal_bias,
    solve_dc,
    write_standard_csv,
)


def _reset_zero_bias(
    operating_point: OperatingPoint,
    *,
    label: str,
) -> str:
    """Return the device to zero gate/drain bias and report any failure."""

    errors: list[str] = []

    for terminal in ("gate", "drain"):
        try:
            ramp_terminal(
                operating_point,
                terminal,
                0.0,
                label=f"{label} {terminal} reset",
            )
        except Exception as error:
            errors.append(
                f"{terminal} reset failed: "
                f"{normalize_error_message(error)}"
            )

    return "; ".join(errors)


def _append_failed_curve(
    rows: list[dict[str, Any]],
    *,
    state: Mapping[str, Any],
    vds_V: float,
    vgs_values_V: Iterable[float],
    error_message: str,
) -> None:
    """Append one explicit failed row for every requested ID-VG point."""

    for vgs_V in vgs_values_V:
        rows.append(
            build_iv_row(
                state=state,
                VGS=float(vgs_V),
                VDS=float(vds_V),
                ID=math.nan,
                converged=False,
                error_message=error_message,
            )
        )


def _ramp_idvg_point(
    operating_point: OperatingPoint,
    *,
    target_vgs_V: float,
    label: str,
) -> None:
    """Ramp one output point with bounded zero-gate reseeding on retry."""

    last_error: Exception | None = None

    for attempt in range(1, config.IDVG_POINT_MAX_ATTEMPTS + 1):
        if attempt > 1:
            # A failed Newton recovery can leave the solution vector invalid
            # even though the gate parameter was restored.  Gate=0 V is the
            # already-established, strongly convergent state-preparation
            # anchor.  Re-solve there before retrying the target so the next
            # adaptive ramp never starts from an invalid solution vector.
            try:
                set_terminal_bias("gate", 0.0)
                operating_point.gate_voltage_V = 0.0
                operating_point.solution_valid = False
                solve_dc(maximum_iterations=200)
                operating_point.solution_valid = True
            except Exception as error:
                last_error = error
                print(
                    f"{label}: output-point attempt {attempt} could not "
                    "reseed the solution at VGS=0 V."
                )
                continue

        try:
            ramp_terminal(
                operating_point,
                "gate",
                target_vgs_V,
                label=label,
            )
            return
        except Exception as error:
            last_error = error
            if attempt < config.IDVG_POINT_MAX_ATTEMPTS:
                print(
                    f"{label}: output-point attempt {attempt} failed; "
                    "retrying from the synchronized device bias."
                )

    raise RuntimeError(
        f"{label}: failed after {config.IDVG_POINT_MAX_ATTEMPTS} attempts; "
        f"last error: {normalize_error_message(last_error)}"
    ) from last_error


def run_idvg_by_state(
    *,
    states: Sequence[Mapping[str, Any]] = config.MEMORY_STATES,
    vds_values_V: Sequence[float] = config.IDVG_VDS_VALUES_V,
    vgs_start_V: float = config.IDVG_VGS_START_V,
    vgs_stop_V: float = config.IDVG_VGS_STOP_V,
    vgs_step_V: float = config.IDVG_VGS_STEP_V,
    operating_point: OperatingPoint | None = None,
) -> tuple[list[dict[str, Any]], OperatingPoint]:
    """Sweep gate voltage for every requested trapped-charge state and VDS.

    A failed bias point is represented by a row with ``converged=False`` and
    ``ID_A=nan``.  The adaptive ramp restores its last converged solution;
    the following requested point is then attempted from that tracked state.
    """

    config.validate_state_characterization_config()

    if operating_point is None:
        _, operating_point = initialize_characterization_device()

    gate_voltages_V = create_output_voltage_list(
        start_voltage=float(vgs_start_V),
        stop_voltage=float(vgs_stop_V),
        step_voltage=float(vgs_step_V),
    )

    rows: list[dict[str, Any]] = []

    # Keep the existing memory-window continuation order for each drain bias:
    # drain ramp -> gate zero -> trap ramp -> gate start -> ID-VG sweep.
    for requested_vds_V in vds_values_V:
        requested_vds_V = float(requested_vds_V)

        drain_preparation_error = _reset_zero_bias(
            operating_point,
            label=f"VDS={requested_vds_V:+.3f} V preparation",
        )
        if not drain_preparation_error:
            try:
                ramp_terminal(
                    operating_point,
                    "drain",
                    requested_vds_V,
                    label=f"drain ramp to {requested_vds_V:+.3f} V",
                )
            except Exception as error:
                drain_preparation_error = (
                    "drain-bias preparation failed: "
                    f"{normalize_error_message(error)}"
                )

        if drain_preparation_error:
            for state in states:
                _append_failed_curve(
                    rows,
                    state=state,
                    vds_V=requested_vds_V,
                    vgs_values_V=gate_voltages_V,
                    error_message=drain_preparation_error,
                )
            continue

        for state in states:
            state_label = str(state["state"])
            target_density_cm3 = float(state["ntrap_cm3"])
            state_error = ""

            try:
                ramp_terminal(
                    operating_point,
                    "gate",
                    0.0,
                    label=f"{state_label} gate reset",
                )
                ramp_trap_density(
                    operating_point,
                    target_density_cm3,
                    label=f"{state_label} trap ramp",
                )
                ramp_terminal(
                    operating_point,
                    "gate",
                    float(vgs_start_V),
                    label=(
                        f"{state_label} gate pre-ramp at "
                        f"VDS={requested_vds_V:+.3f} V"
                    ),
                )
            except Exception as error:
                state_error = (
                    "curve preparation failed: "
                    f"{normalize_error_message(error)}"
                )

            if state_error:
                _append_failed_curve(
                    rows,
                    state=state,
                    vds_V=requested_vds_V,
                    vgs_values_V=gate_voltages_V,
                    error_message=state_error,
                )
            else:
                for requested_vgs_V in gate_voltages_V:
                    requested_vgs_V = float(requested_vgs_V)

                    try:
                        _ramp_idvg_point(
                            operating_point,
                            target_vgs_V=requested_vgs_V,
                            label=(
                                f"{state_label} ID-VG target "
                                f"VGS={requested_vgs_V:+.3f} V, "
                                f"VDS={requested_vds_V:+.3f} V"
                            ),
                        )
                        drain_current_A = float(get_drain_current())

                        if not math.isfinite(drain_current_A):
                            raise RuntimeError(
                                "DEVSIM returned a non-finite drain current."
                            )

                        rows.append(
                            build_iv_row(
                                state=state,
                                VGS=requested_vgs_V,
                                VDS=requested_vds_V,
                                ID=drain_current_A,
                                converged=True,
                                error_message="",
                            )
                        )
                    except Exception as error:
                        rows.append(
                            build_iv_row(
                                state=state,
                                VGS=requested_vgs_V,
                                VDS=requested_vds_V,
                                ID=math.nan,
                                converged=False,
                                error_message=normalize_error_message(error),
                            )
                        )

            try:
                ramp_terminal(
                    operating_point,
                    "gate",
                    0.0,
                    label=f"{state_label} post-sweep gate reset",
                )
            except Exception as error:
                reset_message = (
                    f"{state_label} post-sweep gate reset failed: "
                    f"{normalize_error_message(error)}"
                )
                operating_point.orchestration_errors.append(reset_message)
                print(
                    f"WARNING: {reset_message}"
                )

        reset_error = _reset_zero_bias(
            operating_point,
            label=f"VDS={requested_vds_V:+.3f} V completion",
        )
        if reset_error:
            operating_point.orchestration_errors.append(reset_error)
            print(f"WARNING: {reset_error}")
        try:
            ramp_trap_density(
                operating_point,
                0.0,
                label=(
                    f"VDS={requested_vds_V:+.3f} V empty-state reset"
                ),
            )
        except Exception as error:
            reset_message = (
                "empty-state reset failed after "
                f"VDS={requested_vds_V:+.3f} V: "
                f"{normalize_error_message(error)}"
            )
            operating_point.orchestration_errors.append(reset_message)
            print(
                f"WARNING: {reset_message}"
            )

    rows.sort(
        key=lambda row: (
            int(row["state_index"]),
            float(row["VDS_V"]),
            float(row["VGS_V"]),
        )
    )

    return rows, operating_point


def summarize_convergence(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Return the successful and failed ID-VG bias-point counts."""

    successful = sum(bool(row["converged"]) for row in rows)
    return successful, len(rows) - successful


def main() -> int:
    """Run the complete standalone ID-VG characterization."""

    rows, operating_point = run_idvg_by_state()
    output_path = (
        Path(config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY)
        / "idvg_by_state_standalone.csv"
    )
    write_standard_csv(
        output_path,
        config.IDVG_FIELDNAMES,
        rows,
    )

    successful, failed = summarize_convergence(rows)
    print(f"ID-VG bias points: {successful} converged, {failed} failed")
    print(f"Standalone ID-VG CSV: {output_path.resolve()}")
    return 0 if failed == 0 and not operating_point.orchestration_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
