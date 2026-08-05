"""Integrated static state-characterization command line runner.

Examples from the repository root::

    python 04_charge_trap/run_state_characterization.py --smoke
    python 04_charge_trap/run_state_characterization.py --idvg-only
    python 04_charge_trap/run_state_characterization.py --idvd-only
    python 04_charge_trap/run_state_characterization.py --all

The implementation intentionally reuses the established static trapped-charge
and electron-only drift-diffusion models.  It does not calculate or alter
program/erase dynamics, tunneling, retention, or endurance.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import state_characterization_config as config
from electrical_metrics import extract_state_metrics
from run_idvd_by_state import (
    run_idvd_by_state,
    summarize_convergence as summarize_idvd_convergence,
)
from run_idvg_by_state import (
    run_idvg_by_state,
    summarize_convergence as summarize_idvg_convergence,
)
from state_sweep_helpers import (
    initialize_characterization_device,
    write_standard_csv,
)


def build_metrics_rows(
    idvg_rows: Sequence[Mapping[str, Any]],
    *,
    states: Sequence[Mapping[str, Any]],
    vds_values_V: Sequence[float],
) -> list[dict[str, Any]]:
    """Extract one strict metrics row per state and requested ID-VG VDS."""

    metrics_rows: list[dict[str, Any]] = []

    for state in states:
        state_index = int(state["state_index"])
        state_rows = [
            row
            for row in idvg_rows
            if int(row["state_index"]) == state_index
        ]
        if not state_rows:
            raise RuntimeError(
                f"No ID-VG rows were produced for state index {state_index}."
            )

        for vds_V in vds_values_V:
            extracted = extract_state_metrics(
                state_rows,
                vds_V=float(vds_V),
                threshold_current_A=config.VTH_TARGET_CURRENT_A,
                ss_current_min_A=config.SS_CURRENT_MIN_A,
                ss_current_max_A=config.SS_CURRENT_MAX_A,
                ss_minimum_point_count=config.SS_MINIMUM_POINT_COUNT,
                ion_vgs_V=config.ION_VGS_V,
                ion_vds_V=config.ION_VDS_V,
                ioff_vgs_V=config.IOFF_VGS_V,
                ioff_vds_V=config.IOFF_VDS_V,
                current_floor_A=config.CURRENT_FLOOR_A,
            )

            missing_fields = [
                field
                for field in config.METRICS_FIELDNAMES
                if field not in extracted
            ]
            if missing_fields:
                raise RuntimeError(
                    "Metric extraction omitted required fields: "
                    f"{missing_fields}"
                )

            metrics_rows.append(
                {
                    field: extracted[field]
                    for field in config.METRICS_FIELDNAMES
                }
            )

    metrics_rows.sort(
        key=lambda row: (
            int(row["state_index"]),
            float(row["VDS_V"]),
        )
    )
    return metrics_rows


def _find_metric_at_vds(
    metrics_rows: Sequence[Mapping[str, Any]],
    *,
    state_index: int,
    vds_V: float,
) -> Mapping[str, Any] | None:
    """Find one state/VDS metric row using strict voltage tolerance."""

    for row in metrics_rows:
        if (
            int(row["state_index"]) == int(state_index)
            and math.isclose(
                float(row["VDS_V"]),
                float(vds_V),
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            return row
    return None


def build_memory_state_map(
    metrics_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build Vth shifts relative to the empty state at each drain bias."""

    map_rows: list[dict[str, Any]] = []

    for metric_row in metrics_rows:
        vds_V = float(metric_row["VDS_V"])
        empty_row = _find_metric_at_vds(
            metrics_rows,
            state_index=0,
            vds_V=vds_V,
        )

        threshold_voltage_V = float(metric_row["Vth_V"])
        empty_threshold_voltage_V = (
            float(empty_row["Vth_V"])
            if empty_row is not None
            else math.nan
        )
        threshold_shift_V = (
            threshold_voltage_V - empty_threshold_voltage_V
            if math.isfinite(threshold_voltage_V)
            and math.isfinite(empty_threshold_voltage_V)
            else math.nan
        )
        sheet_density_cm2 = float(metric_row["nsheet_cm2"])

        map_rows.append(
            {
                "state_index": int(metric_row["state_index"]),
                "state": str(metric_row["state"]),
                "ntrap_cm3": float(metric_row["ntrap_cm3"]),
                "nsheet_cm2": sheet_density_cm2,
                "qsheet_C_cm2": (
                    config.electron_sheet_density_to_charge_density(
                        sheet_density_cm2
                    )
                ),
                "VDS_V": vds_V,
                "Vth_V": threshold_voltage_V,
                "delta_Vth_from_empty_V": threshold_shift_V,
            }
        )

    map_rows.sort(
        key=lambda row: (
            int(row["state_index"]),
            float(row["VDS_V"]),
        )
    )
    return map_rows


def evaluate_physical_sanity(
    *,
    idvg_rows: Sequence[Mapping[str, Any]],
    idvd_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    programmed_state_index: int,
    orchestration_errors: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate convergence, metric completeness, and physical trends."""

    all_iv_rows = list(idvg_rows) + list(idvd_rows)
    failed_bias_points = sum(not bool(row["converged"]) for row in all_iv_rows)

    finite_current_ok = True
    for row in all_iv_rows:
        if not bool(row["converged"]):
            continue
        signed_current_A = float(row["ID_A"])
        absolute_current_A = float(row["abs_ID_A"])
        if (
            not math.isfinite(signed_current_A)
            or not math.isfinite(absolute_current_A)
            or not math.isclose(
                absolute_current_A,
                abs(signed_current_A),
                rel_tol=1.0e-12,
                abs_tol=config.CURRENT_FLOOR_A,
            )
        ):
            finite_current_ok = False
            break

    metric_errors: list[str] = []
    finite_metric_fields = (
        "Vth_V",
        "SS_mV_dec",
        "ss_r_squared",
        "Ion_A",
        "Ioff_A",
        "gm_max_S",
        "VGS_at_gm_max_V",
    )
    for row in metrics_rows:
        metric_identity = (
            f"state={row['state']}, VDS={float(row['VDS_V']):.3f} V"
        )
        if not bool(row["vth_success"]):
            metric_errors.append(
                f"{metric_identity}: Vth failed: {row['vth_error']}"
            )
        if not bool(row["ss_success"]):
            metric_errors.append(
                f"{metric_identity}: SS failed: {row['ss_error']}"
            )
        for field_name in finite_metric_fields:
            try:
                field_value = float(row[field_name])
            except (KeyError, TypeError, ValueError):
                metric_errors.append(
                    f"{metric_identity}: {field_name} is not numeric"
                )
                continue
            if not math.isfinite(field_value):
                metric_errors.append(
                    f"{metric_identity}: {field_name} is not finite"
                )

        if int(row["ss_point_count"]) < config.SS_MINIMUM_POINT_COUNT:
            metric_errors.append(
                f"{metric_identity}: insufficient SS point count"
            )
        if float(row["Ion_A"]) < 0.0 or float(row["Ioff_A"]) < 0.0:
            metric_errors.append(
                f"{metric_identity}: Ion/Ioff magnitudes must be nonnegative"
            )
        ion_A = float(row["Ion_A"])
        ioff_A = float(row["Ioff_A"])
        on_off_ratio = float(row["on_off_ratio"])
        if ioff_A > config.CURRENT_FLOOR_A:
            expected_ratio = ion_A / ioff_A
            if not (
                math.isfinite(on_off_ratio)
                and math.isclose(
                    on_off_ratio,
                    expected_ratio,
                    rel_tol=1.0e-12,
                    abs_tol=0.0,
                )
            ):
                metric_errors.append(
                    f"{metric_identity}: ON/OFF ratio is invalid"
                )
        elif not math.isnan(on_off_ratio):
            metric_errors.append(
                f"{metric_identity}: floor-limited ON/OFF must be nan"
            )
        if float(row["gm_max_S"]) < 0.0:
            metric_errors.append(
                f"{metric_identity}: gm_max_S must be a magnitude"
            )

    metrics_complete_ok = not metric_errors

    vth_order_ok: bool | None = None
    vth_monotonic_ok: bool | None = None
    empty_vth_V = math.nan
    programmed_vth_V = math.nan
    if metrics_rows:
        empty_metric = _find_metric_at_vds(
            metrics_rows,
            state_index=0,
            vds_V=config.ION_VDS_V,
        )
        programmed_metric = _find_metric_at_vds(
            metrics_rows,
            state_index=programmed_state_index,
            vds_V=config.ION_VDS_V,
        )
        if empty_metric is not None:
            empty_vth_V = float(empty_metric["Vth_V"])
        if programmed_metric is not None:
            programmed_vth_V = float(programmed_metric["Vth_V"])
        vth_order_ok = (
            math.isfinite(empty_vth_V)
            and math.isfinite(programmed_vth_V)
            and programmed_vth_V > empty_vth_V
        )

        reference_metrics = sorted(
            (
                row
                for row in metrics_rows
                if math.isclose(
                    float(row["VDS_V"]),
                    config.ION_VDS_V,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            ),
            key=lambda row: int(row["state_index"]),
        )
        reference_vths = [float(row["Vth_V"]) for row in reference_metrics]
        vth_monotonic_ok = (
            len(reference_vths) >= 2
            and all(math.isfinite(value) for value in reference_vths)
            and all(
                right > left
                for left, right in zip(reference_vths, reference_vths[1:])
            )
        )

    right_shift_ok: bool | None = None
    right_shift_fraction = math.nan
    if idvg_rows:
        empty_curve = {
            round(float(row["VGS_V"]), 12): float(row["abs_ID_A"])
            for row in idvg_rows
            if int(row["state_index"]) == 0
            and bool(row["converged"])
            and math.isclose(
                float(row["VDS_V"]),
                config.ION_VDS_V,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        }
        state_indices = sorted(
            {
                int(row["state_index"])
                for row in idvg_rows
                if int(row["state_index"]) != 0
            }
        )
        state_shift_fractions: list[float] = []
        for state_index in state_indices:
            state_curve = {
                round(float(row["VGS_V"]), 12): float(row["abs_ID_A"])
                for row in idvg_rows
                if int(row["state_index"]) == state_index
                and bool(row["converged"])
                and math.isclose(
                    float(row["VDS_V"]),
                    config.ION_VDS_V,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
            }
            common_voltages = sorted(set(empty_curve) & set(state_curve))
            if not common_voltages:
                state_shift_fractions.append(0.0)
                continue
            right_shift_count = sum(
                state_curve[voltage]
                <= empty_curve[voltage] * (1.0 + 1.0e-9)
                for voltage in common_voltages
            )
            state_shift_fractions.append(
                right_shift_count / len(common_voltages)
            )

        if state_shift_fractions:
            right_shift_fraction = min(state_shift_fractions)
            right_shift_ok = all(
                fraction >= 0.6 for fraction in state_shift_fractions
            )
        else:
            right_shift_ok = False

    return {
        "failed_bias_points": failed_bias_points,
        "finite_current_ok": finite_current_ok,
        "metrics_complete_ok": metrics_complete_ok,
        "metric_errors": metric_errors,
        "empty_vth_V": empty_vth_V,
        "programmed_vth_V": programmed_vth_V,
        "vth_order_ok": vth_order_ok,
        "vth_monotonic_ok": vth_monotonic_ok,
        "right_shift_fraction": right_shift_fraction,
        "right_shift_ok": right_shift_ok,
        "orchestration_errors": list(orchestration_errors),
        "orchestration_ok": not orchestration_errors,
    }


def _format_metric(value: Any) -> str:
    """Format a numeric summary value without hiding NaN failures."""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{numeric_value:.6e}" if math.isfinite(numeric_value) else "nan"


def print_summary(
    *,
    mode: str,
    states: Sequence[Mapping[str, Any]],
    idvg_rows: Sequence[Mapping[str, Any]],
    idvd_rows: Sequence[Mapping[str, Any]],
    metrics_rows: Sequence[Mapping[str, Any]],
    memory_map_rows: Sequence[Mapping[str, Any]],
    sanity: Mapping[str, Any],
    generated_paths: Sequence[Path],
) -> None:
    """Print the required convergence, metric, sanity, and file summary."""

    print()
    print("=" * 76)
    print(f"STATE CHARACTERIZATION SUMMARY ({mode})")
    print("=" * 76)

    if idvg_rows:
        successful, failed = summarize_idvg_convergence(idvg_rows)
        print(f"ID-VG bias points: {successful} converged, {failed} failed")
    if idvd_rows:
        successful, failed = summarize_idvd_convergence(idvd_rows)
        print(f"ID-VD bias points: {successful} converged, {failed} failed")

    print(
        "Extraction criteria: "
        f"Vth@{config.VTH_TARGET_CURRENT_A:.3e} A, "
        f"SS window={config.SS_CURRENT_MIN_A:.3e}.."
        f"{config.SS_CURRENT_MAX_A:.3e} A, "
        f"Ion=ID({config.ION_VGS_V:+.2f} V, {config.ION_VDS_V:+.2f} V), "
        f"Ioff=ID({config.IOFF_VGS_V:+.2f} V, {config.IOFF_VDS_V:+.2f} V)"
    )

    for state in states:
        metric_row = _find_metric_at_vds(
            metrics_rows,
            state_index=int(state["state_index"]),
            vds_V=config.ION_VDS_V,
        )
        if metric_row is None:
            continue
        map_row = _find_metric_at_vds(
            memory_map_rows,
            state_index=int(state["state_index"]),
            vds_V=config.ION_VDS_V,
        )
        delta_vth = (
            map_row["delta_Vth_from_empty_V"]
            if map_row is not None
            else math.nan
        )
        print()
        print(f"{metric_row['state']}:")
        print(
            "  Vth="
            f"{_format_metric(metric_row['Vth_V'])} V, "
            "DeltaVth="
            f"{_format_metric(delta_vth)} V, "
            "SS="
            f"{_format_metric(metric_row['SS_mV_dec'])} mV/dec"
        )
        print(
            "  Ion="
            f"{_format_metric(metric_row['Ion_A'])} A, "
            "Ioff="
            f"{_format_metric(metric_row['Ioff_A'])} A, "
            "ON/OFF="
            f"{_format_metric(metric_row['on_off_ratio'])}, "
            "gm,max="
            f"{_format_metric(metric_row['gm_max_S'])} S"
        )

    print()
    print("Sanity checks:")
    print(f"  finite converged currents: {sanity['finite_current_ok']}")
    print(f"  orchestration/reset health: {sanity['orchestration_ok']}")
    if metrics_rows:
        print(f"  complete finite metrics: {sanity['metrics_complete_ok']}")
    if sanity["vth_order_ok"] is not None:
        print(
            "  programmed Vth > empty Vth: "
            f"{sanity['vth_order_ok']} "
            f"({_format_metric(sanity['programmed_vth_V'])} > "
            f"{_format_metric(sanity['empty_vth_V'])} V)"
        )
        print(
            "  Vth strictly increases across states: "
            f"{sanity['vth_monotonic_ok']}"
        )
    if sanity["right_shift_ok"] is not None:
        print(
            "  programmed curve generally right-shifted: "
            f"{sanity['right_shift_ok']} "
            f"(fraction={_format_metric(sanity['right_shift_fraction'])})"
        )

    for error_message in sanity["orchestration_errors"]:
        print(f"  orchestration error: {error_message}")
    for error_message in sanity["metric_errors"]:
        print(f"  metric error: {error_message}")

    print()
    print("Generated files:")
    for path in generated_paths:
        print(f"  {path.resolve()}")


def _parse_args() -> argparse.Namespace:
    """Parse one mutually exclusive characterization mode."""

    parser = argparse.ArgumentParser(
        description="Extract static ID-VG, ID-VD, Vth, SS, Ion/Ioff, and gm."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--smoke", action="store_true")
    mode_group.add_argument("--idvg-only", action="store_true")
    mode_group.add_argument("--idvd-only", action="store_true")
    mode_group.add_argument("--all", action="store_true")
    return parser.parse_args()


def _selected_mode(arguments: argparse.Namespace) -> str:
    """Return the explicitly requested characterization mode."""

    if arguments.smoke:
        return "smoke"
    if arguments.idvg_only:
        return "idvg-only"
    if arguments.idvd_only:
        return "idvd-only"
    if arguments.all:
        return "all"
    raise RuntimeError("No characterization mode was selected.")


def main() -> int:
    """Run the selected characterization and write raw plus standard CSVs."""

    arguments = _parse_args()
    mode = _selected_mode(arguments)
    config.validate_state_characterization_config()

    smoke = mode == "smoke"
    run_idvg = mode in {"smoke", "idvg-only", "all"}
    run_idvd = mode in {"smoke", "idvd-only", "all"}
    states = config.SMOKE_MEMORY_STATES if smoke else config.MEMORY_STATES

    _, operating_point = initialize_characterization_device()

    idvg_rows: list[dict[str, Any]] = []
    idvd_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    memory_map_rows: list[dict[str, Any]] = []
    generated_paths: list[Path] = []

    if run_idvg:
        idvg_vds_values = (
            config.SMOKE_IDVG_VDS_VALUES_V
            if smoke
            else config.IDVG_VDS_VALUES_V
        )
        idvg_step_V = (
            config.SMOKE_IDVG_VGS_STEP_V
            if smoke
            else config.IDVG_VGS_STEP_V
        )
        idvg_rows, operating_point = run_idvg_by_state(
            states=states,
            vds_values_V=idvg_vds_values,
            vgs_start_V=config.IDVG_VGS_START_V,
            vgs_stop_V=config.IDVG_VGS_STOP_V,
            vgs_step_V=idvg_step_V,
            operating_point=operating_point,
        )
        metrics_rows = build_metrics_rows(
            idvg_rows,
            states=states,
            vds_values_V=idvg_vds_values,
        )
        memory_map_rows = build_memory_state_map(metrics_rows)

        raw_idvg_path = (
            config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY
            / f"idvg_by_state_{mode}.csv"
        )
        raw_metrics_path = (
            config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY
            / f"metrics_by_state_{mode}.csv"
        )
        raw_map_path = (
            config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY
            / f"memory_state_map_{mode}.csv"
        )
        write_standard_csv(raw_idvg_path, config.IDVG_FIELDNAMES, idvg_rows)
        write_standard_csv(
            raw_metrics_path,
            config.METRICS_FIELDNAMES,
            metrics_rows,
        )
        write_standard_csv(
            raw_map_path,
            config.MEMORY_STATE_MAP_FIELDNAMES,
            memory_map_rows,
        )
        generated_paths.extend(
            (
                raw_idvg_path,
                raw_metrics_path,
                raw_map_path,
            )
        )

    if run_idvd:
        idvd_vgs_values = (
            config.SMOKE_IDVD_VGS_VALUES_V
            if smoke
            else config.IDVD_VGS_VALUES_V
        )
        idvd_step_V = (
            config.SMOKE_IDVD_VDS_STEP_V
            if smoke
            else config.IDVD_VDS_STEP_V
        )
        idvd_rows, operating_point = run_idvd_by_state(
            states=states,
            vgs_values_V=idvd_vgs_values,
            vds_start_V=config.IDVD_VDS_START_V,
            vds_stop_V=config.IDVD_VDS_STOP_V,
            vds_step_V=idvd_step_V,
            operating_point=operating_point,
        )

        raw_idvd_path = (
            config.STATE_CHARACTERIZATION_RESULTS_DIRECTORY
            / f"idvd_by_state_{mode}.csv"
        )
        write_standard_csv(raw_idvd_path, config.IDVD_FIELDNAMES, idvd_rows)
        generated_paths.append(raw_idvd_path)

    sanity = evaluate_physical_sanity(
        idvg_rows=idvg_rows,
        idvd_rows=idvd_rows,
        metrics_rows=metrics_rows,
        programmed_state_index=int(states[-1]["state_index"]),
        orchestration_errors=operating_point.orchestration_errors,
    )

    checks = [
        sanity["failed_bias_points"] == 0,
        bool(sanity["finite_current_ok"]),
        bool(sanity["orchestration_ok"]),
    ]
    if run_idvg:
        checks.extend(
            (
                bool(sanity["metrics_complete_ok"]),
                bool(sanity["vth_order_ok"]),
                bool(sanity["vth_monotonic_ok"]),
                bool(sanity["right_shift_ok"]),
            )
        )

    run_valid = all(checks)

    # Only a complete, validated --all run publishes the four canonical CSVs
    # as one coherent dataset.  Smoke and partial modes remain in results/ so
    # they cannot clobber or mix a previously validated shared_data bundle.
    if mode == "all" and run_valid:
        write_standard_csv(
            config.IDVG_BY_STATE_CSV_PATH,
            config.IDVG_FIELDNAMES,
            idvg_rows,
        )
        write_standard_csv(
            config.IDVD_BY_STATE_CSV_PATH,
            config.IDVD_FIELDNAMES,
            idvd_rows,
        )
        write_standard_csv(
            config.METRICS_BY_STATE_CSV_PATH,
            config.METRICS_FIELDNAMES,
            metrics_rows,
        )
        write_standard_csv(
            config.MEMORY_STATE_MAP_CSV_PATH,
            config.MEMORY_STATE_MAP_FIELDNAMES,
            memory_map_rows,
        )
        generated_paths.extend(
            (
                config.IDVG_BY_STATE_CSV_PATH,
                config.IDVD_BY_STATE_CSV_PATH,
                config.METRICS_BY_STATE_CSV_PATH,
                config.MEMORY_STATE_MAP_CSV_PATH,
            )
        )

    print_summary(
        mode=mode,
        states=states,
        idvg_rows=idvg_rows,
        idvd_rows=idvd_rows,
        metrics_rows=metrics_rows,
        memory_map_rows=memory_map_rows,
        sanity=sanity,
        generated_paths=generated_paths,
    )

    if mode != "all":
        print(
            "Canonical shared_data CSVs are published only by a successful "
            "--all run."
        )
    return 0 if run_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
