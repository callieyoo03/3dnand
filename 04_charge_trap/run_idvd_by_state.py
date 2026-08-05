"""Run static trapped-charge ID-VD sweeps.

Run the full standalone sweep from the repository root with::

    python 04_charge_trap/run_idvd_by_state.py

``run_state_characterization.py`` imports the sweep function and shares one
initialized DEVSIM device with the ID-VG sweep.
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
    write_standard_csv,
)


def _reset_zero_bias(
    operating_point: OperatingPoint,
    *,
    label: str,
) -> str:
    """Return gate and drain to zero bias, preserving all error details."""

    errors: list[str] = []

    for terminal in ("drain", "gate"):
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
    vgs_V: float,
    vds_values_V: Iterable[float],
    error_message: str,
) -> None:
    """Append one explicit failed row for every requested drain bias."""

    for vds_V in vds_values_V:
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


def run_idvd_by_state(
    *,
    states: Sequence[Mapping[str, Any]] = config.MEMORY_STATES,
    vgs_values_V: Sequence[float] = config.IDVD_VGS_VALUES_V,
    vds_start_V: float = config.IDVD_VDS_START_V,
    vds_stop_V: float = config.IDVD_VDS_STOP_V,
    vds_step_V: float = config.IDVD_VDS_STEP_V,
    operating_point: OperatingPoint | None = None,
) -> tuple[list[dict[str, Any]], OperatingPoint]:
    """Sweep drain voltage for every requested state and fixed VGS.

    The signed numerical current at VDS=0 is retained exactly as returned by
    DEVSIM.  Failed points are represented explicitly rather than replaced by
    a current from a different bias.
    """

    config.validate_state_characterization_config()

    if operating_point is None:
        _, operating_point = initialize_characterization_device()

    drain_voltages_V = create_output_voltage_list(
        start_voltage=float(vds_start_V),
        stop_voltage=float(vds_stop_V),
        step_voltage=float(vds_step_V),
    )

    rows: list[dict[str, Any]] = []

    for state in states:
        state_label = str(state["state"])
        target_density_cm3 = float(state["ntrap_cm3"])

        state_error = _reset_zero_bias(
            operating_point,
            label=state_label,
        )

        if not state_error:
            try:
                ramp_trap_density(
                    operating_point,
                    target_density_cm3,
                    label=f"{state_label} trap ramp",
                )
            except Exception as error:
                state_error = (
                    "trap-state preparation failed: "
                    f"{normalize_error_message(error)}"
                )

        for requested_vgs_V in vgs_values_V:
            requested_vgs_V = float(requested_vgs_V)

            if state_error:
                _append_failed_curve(
                    rows,
                    state=state,
                    vgs_V=requested_vgs_V,
                    vds_values_V=drain_voltages_V,
                    error_message=state_error,
                )
                continue

            preparation_errors: list[str] = []

            try:
                ramp_terminal(
                    operating_point,
                    "drain",
                    0.0,
                    label=f"{state_label} drain reset",
                )
                ramp_terminal(
                    operating_point,
                    "gate",
                    requested_vgs_V,
                    label=(
                        f"{state_label} gate ramp to "
                        f"{requested_vgs_V:+.3f} V"
                    ),
                )
            except Exception as error:
                preparation_errors.append(
                    "curve preparation failed: "
                    f"{normalize_error_message(error)}"
                )

            if preparation_errors:
                _append_failed_curve(
                    rows,
                    state=state,
                    vgs_V=requested_vgs_V,
                    vds_values_V=drain_voltages_V,
                    error_message="; ".join(preparation_errors),
                )
            else:
                for requested_vds_V in drain_voltages_V:
                    requested_vds_V = float(requested_vds_V)

                    try:
                        ramp_terminal(
                            operating_point,
                            "drain",
                            requested_vds_V,
                            label=(
                                f"{state_label} ID-VD target "
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

            reset_error = _reset_zero_bias(
                operating_point,
                label=(
                    f"{state_label} VGS={requested_vgs_V:+.3f} V"
                ),
            )
            if reset_error:
                operating_point.orchestration_errors.append(reset_error)
                print(f"WARNING: {reset_error}")

    rows.sort(
        key=lambda row: (
            int(row["state_index"]),
            float(row["VGS_V"]),
            float(row["VDS_V"]),
        )
    )

    return rows, operating_point


def summarize_convergence(rows: Sequence[Mapping[str, Any]]) -> tuple[int, int]:
    """Return the successful and failed ID-VD bias-point counts."""

    successful = sum(bool(row["converged"]) for row in rows)
    return successful, len(rows) - successful


def main() -> int:
    """Run the complete standalone ID-VD characterization."""

    rows, operating_point = run_idvd_by_state()
    output_path = (
        Path(config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY)
        / "idvd_by_state_standalone.csv"
    )
    write_standard_csv(
        output_path,
        config.IDVD_FIELDNAMES,
        rows,
    )

    successful, failed = summarize_convergence(rows)
    print(f"ID-VD bias points: {successful} converged, {failed} failed")
    print(f"Standalone ID-VD CSV: {output_path.resolve()}")
    return 0 if failed == 0 and not operating_point.orchestration_errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
