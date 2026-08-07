"""Active-window sensitivity for the preserved electron-FN negative control.

This runner reuses the Stage-1 staged solver and all-edge log-domain FN
diagnostic.  It writes a new candidate only; it never reads or overwrites the
baseline result directory.  The values remain non-predictive diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import edge_resolved_erase as edge_fn
import erase_parameters as ep
from erase_state_solver import EraseStateSolver
from field_extraction import get_active_inner_interface_radial_field_statistics


MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_PATH = (
    MODULE_DIRECTORY
    / "results"
    / "electron_trap_emission_framework"
    / "active_window_sensitivity.csv"
)

ACTIVE_WINDOWS_NM = (
    (10.0, 90.0),
    (11.0, 89.0),
    (12.0, 88.0),
    (20.0, 80.0),
)

MODEL_FIELDS = (
    "model_class",
    "physical_role",
    "model_status",
    "calibration_status",
    "predictive_erase_model",
)

CSV_FIELDS = MODEL_FIELDS + (
    "requested_gate_voltage_V",
    "actual_gate_voltage_V",
    "source_voltage_V",
    "drain_voltage_V",
    "trap_density_cm3",
    "window_minimum_nm",
    "window_maximum_nm",
    "window_length_nm",
    "exact_target_reached",
    "converged",
    "error_message",
    "active_interface_edge_count",
    "active_interface_area_cm2",
    "area_ratio_vs_10_90",
    "field_mean_signed_outward_V_cm",
    "field_mean_abs_V_cm",
    "field_max_abs_V_cm",
    "field_mean_abs_ratio_vs_10_90",
    "field_max_abs_ratio_vs_10_90",
    "log10_effective_current_density_A_cm2",
    "log10_effective_current_density_delta_vs_10_90",
    "log10_total_tunneling_current_A",
    "log10_total_current_delta_vs_10_90",
    "endpoint_pair_current_fraction",
    "top_two_edge_current_fraction",
    "top_10_percent_edge_current_fraction",
    "maximum_field_edge_at_selected_window_endpoint",
    "primary_integration",
    "window_role",
)


def _model_metadata() -> dict:
    expected = {
        "model_class": "electron_FN_diagnostic",
        "physical_role": "negative_control",
        "model_status": "baseline_negative_result",
        "calibration_status": "uncalibrated",
        "predictive_erase_model": False,
    }
    actual = dict(ep.MODEL_METADATA)
    if actual != expected:
        raise RuntimeError("Electron-FN negative-control metadata changed.")
    return actual


def _safe_log10_delta(value: float, reference: float) -> float:
    value = float(value)
    reference = float(reference)
    if value == reference:
        return 0.0
    if not math.isfinite(value) or not math.isfinite(reference):
        return math.nan
    return value - reference


def summarize_window(
    statistics: Mapping,
    edge_result: Mapping,
) -> dict:
    """Return field/FN metrics and explicit end-edge dominance measures."""

    rows = list(edge_result["edge_results"])
    if not rows:
        raise ValueError("A window sensitivity result requires at least one edge.")
    rows.sort(key=lambda row: float(row["midpoint_axial_nm"]))
    fractions = [
        float(row["fraction_of_total_tunneling_current"]) for row in rows
    ]
    descending = sorted(fractions, reverse=True)
    top_ten_count = max(1, int(math.ceil(0.10 * len(rows))))

    maximum_field_index = max(
        range(len(rows)),
        key=lambda index: abs(float(rows[index]["electric_field_signed_V_cm"])),
    )
    return {
        "active_interface_edge_count": int(
            statistics["active_interface_edge_count"]
        ),
        "active_interface_area_cm2": float(
            statistics["total_interface_area_cm2"]
        ),
        "field_mean_signed_outward_V_cm": float(
            statistics["area_weighted_field_mean_signed_V_cm"]
        ),
        "field_mean_abs_V_cm": float(
            statistics["area_weighted_field_mean_abs_V_cm"]
        ),
        "field_max_abs_V_cm": float(statistics["field_max_abs_V_cm"]),
        "log10_effective_current_density_A_cm2": float(
            edge_result["log10_effective_current_density_A_cm2"]
        ),
        "log10_total_tunneling_current_A": float(
            edge_result["log10_total_tunneling_current_A"]
        ),
        "endpoint_pair_current_fraction": fractions[0]
        + (fractions[-1] if len(fractions) > 1 else 0.0),
        "top_two_edge_current_fraction": sum(descending[:2]),
        "top_10_percent_edge_current_fraction": sum(
            descending[:top_ten_count]
        ),
        "maximum_field_edge_at_selected_window_endpoint": (
            maximum_field_index in {0, len(rows) - 1}
        ),
    }


def add_reference_comparison(row: Mapping, reference: Mapping) -> dict:
    """Add ratios/deltas relative to the 10--90 nm row at the same bias."""

    output = dict(row)
    output.update(
        {
            "area_ratio_vs_10_90": float(row["active_interface_area_cm2"])
            / float(reference["active_interface_area_cm2"]),
            "field_mean_abs_ratio_vs_10_90": float(
                row["field_mean_abs_V_cm"]
            )
            / max(float(reference["field_mean_abs_V_cm"]), 1.0e-300),
            "field_max_abs_ratio_vs_10_90": float(row["field_max_abs_V_cm"])
            / max(float(reference["field_max_abs_V_cm"]), 1.0e-300),
            "log10_effective_current_density_delta_vs_10_90": (
                _safe_log10_delta(
                    row["log10_effective_current_density_A_cm2"],
                    reference["log10_effective_current_density_A_cm2"],
                )
            ),
            "log10_total_current_delta_vs_10_90": _safe_log10_delta(
                row["log10_total_tunneling_current_A"],
                reference["log10_total_tunneling_current_A"],
            ),
        }
    )
    return output


def collect_at_bias(
    solver: EraseStateSolver,
    requested_gate_voltage_V: float,
) -> list[dict]:
    state = solver.read_state()
    raw_rows = []
    for minimum_nm, maximum_nm in ACTIVE_WINDOWS_NM:
        statistics = get_active_inner_interface_radial_field_statistics(
            device=solver.runtime.device_name,
            region="TunnelOxide",
            minimum_axial_nm=minimum_nm,
            maximum_axial_nm=maximum_nm,
        )
        edge_result = edge_fn.integrate_all_edges_log_domain(
            statistics["edge_data"]
        )
        summary = summarize_window(statistics, edge_result)
        raw_rows.append(
            {
                **_model_metadata(),
                "requested_gate_voltage_V": float(requested_gate_voltage_V),
                "actual_gate_voltage_V": float(state.gate_voltage_V),
                "source_voltage_V": float(state.source_voltage_V),
                "drain_voltage_V": float(state.drain_voltage_V),
                "trap_density_cm3": float(state.trap_density_cm3),
                "window_minimum_nm": minimum_nm,
                "window_maximum_nm": maximum_nm,
                "window_length_nm": maximum_nm - minimum_nm,
                "exact_target_reached": math.isclose(
                    state.gate_voltage_V,
                    requested_gate_voltage_V,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                ),
                "converged": True,
                "error_message": "",
                **summary,
                "primary_integration": "all_edge_logsumexp_no_field_cutoff",
                "window_role": (
                    "reference" if (minimum_nm, maximum_nm) == ACTIVE_WINDOWS_NM[0]
                    else "endpoint_exclusion_sensitivity"
                ),
            }
        )

    reference = raw_rows[0]
    return [add_reference_comparison(row, reference) for row in raw_rows]


def _csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_rows(path: Path, rows: Sequence[Mapping]) -> None:
    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite active-window candidate: {path}"
        )
    expected = set(CSV_FIELDS)
    for index, row in enumerate(rows):
        if set(row) != expected:
            raise ValueError(
                f"Active-window row {index} schema mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: _csv_value(row[name]) for name in CSV_FIELDS})


def run(output_path: Path = DEFAULT_OUTPUT_PATH) -> list[dict]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite active-window candidate: {output_path}"
        )
    ep.validate_parameters()
    edge_fn.validate_proxy_parameters()

    solver = EraseStateSolver()
    solver.initialize_staged()
    solver.set_terminal_bias("source", ep.SOURCE_VOLTAGE_V)
    solver.set_terminal_bias("drain", ep.DRAIN_VOLTAGE_V)
    solver.set_terminal_bias("gate", 0.0)
    solver.ramp_trap_density(
        ep.INITIAL_TRAP_VOLUME_DENSITY_CM3,
        label="active-window trapped-electron continuation",
    )

    rows = []
    for target_V in ep.ERASE_GATE_TARGETS_V:
        if not math.isclose(
            solver.read_state().gate_voltage_V,
            target_V,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ):
            solver.ramp_gate(
                target_V,
                label=f"active-window target {target_V:+.3f} V",
            )
        rows.extend(collect_at_bias(solver, target_V))

    write_rows(Path(output_path), rows)
    return rows


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    rows = run(arguments.output)
    print(f"Wrote {len(rows)} active-window rows to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
