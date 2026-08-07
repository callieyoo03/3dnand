"""Independent report-based ERASE electron-FN negative-control runner.

This runner stops at the requested negative baseline.  It does not solve or
parameterize holes, update trapped charge in time, create an erased state, or
claim predictive ERASE behavior.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Mapping, Sequence

import edge_resolved_erase as edge_fn
import erase_field_diagnostics as field_audit
import erase_parameters as ep
import material_parameters as material
from erase_state_solver import AdaptiveRampError, EraseStateSolver
from field_extraction import (
    get_active_inner_interface_radial_field_statistics,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "erase_reimplementation_baseline"
)

OUTPUT_FILENAMES = (
    "erase_voltage_sweep.csv",
    "erase_layer_voltage_drop.csv",
    "erase_field_consistency.csv",
    "erase_edge_distribution.csv",
    "erase_fn_log_domain.csv",
    "erase_baseline_README.md",
)

MODEL_FIELDS = (
    "model_class",
    "physical_role",
    "model_status",
    "calibration_status",
    "predictive_erase_model",
)

BIAS_FIELDS = (
    "requested_gate_voltage_V",
    "actual_gate_voltage_V",
    "source_voltage_V",
    "drain_voltage_V",
    "trap_density_cm3",
    "trap_sheet_density_cm2",
)

VOLTAGE_SWEEP_FIELDS = MODEL_FIELDS + BIAS_FIELDS + (
    "target_residual_sheet_density_cm2",
    "exact_target_reached",
    "converged",
    "error_message",
    "drain_current_A",
    "gate_ramp_attempt_count",
    "gate_ramp_failed_attempt_count",
    "trap_ramp_attempt_count",
    "trap_ramp_failed_attempt_count",
    "last_successful_step_kind",
    "last_successful_step_label",
    "last_successful_step_value",
    "last_successful_step_size",
    "devsim_tunnel_field_mean_signed_outward_V_cm",
    "devsim_tunnel_field_mean_abs_V_cm",
    "devsim_tunnel_field_max_abs_V_cm",
    "average_field_FN_current_density_A_cm2",
    "average_field_FN_log10_current_density_A_cm2",
    "edge_logsum_effective_current_density_A_cm2",
    "edge_logsum_log10_effective_current_density_A_cm2",
    "negative_control_time_diagnostic_log10_s_mean_field",
    "negative_control_time_diagnostic_log10_s_edge_resolved",
)

LAYER_VOLTAGE_FIELDS = MODEL_FIELDS + BIAS_FIELDS + (
    "region",
    "thickness_nm",
    "thickness_cm",
    "inner_radius_nm",
    "outer_radius_nm",
    "inner_boundary_node_count",
    "outer_boundary_node_count",
    "inner_potential_mean_V",
    "outer_potential_mean_V",
    "voltage_drop_outer_minus_inner_V",
    "average_field_from_voltage_drop_outward_V_cm",
    "permittivity_F_cm",
    "average_displacement_from_drop_C_cm2",
    "component_voltage_drop_sum_V",
    "total_stack_voltage_drop_V",
    "layer_voltage_sum_residual_V",
    "layer_voltage_sum_relative_residual",
    "tunnel_trap_interface_potential_jump_V",
    "trap_blocking_interface_potential_jump_V",
    "component_plus_interface_jump_sum_V",
    "mesh_coordinate_unit",
    "electric_field_unit",
)

FIELD_CONSISTENCY_FIELDS = MODEL_FIELDS + BIAS_FIELDS + (
    "electric_field_model_definition",
    "mesh_coordinate_unit",
    "potential_unit",
    "electric_field_unit",
    "field_unit_audit",
    "radial_positive_direction",
    "electron_force_direction_for_positive_radial_field",
    "tunnel_radial_field_sign",
    "devsim_tunnel_field_mean_signed_outward_V_cm",
    "devsim_tunnel_field_mean_abs_V_cm",
    "devsim_tunnel_field_max_abs_V_cm",
    "tunnel_field_from_voltage_drop_outward_V_cm",
    "tunnel_field_definition_relative_difference",
    "active_interface_edge_count",
    "active_interface_area_cm2",
    "tunnel_trap_inner_epsilon_E_C_cm2",
    "tunnel_trap_outer_epsilon_E_C_cm2",
    "tunnel_trap_epsilon_E_residual_C_cm2",
    "tunnel_trap_epsilon_E_relative_residual",
    "tunnel_trap_extraction_method",
    "trap_blocking_inner_epsilon_E_C_cm2",
    "trap_blocking_outer_epsilon_E_C_cm2",
    "trap_blocking_epsilon_E_residual_C_cm2",
    "trap_blocking_epsilon_E_relative_residual",
    "trap_blocking_extraction_method",
    "layer_voltage_sum_relative_residual",
)

EDGE_DISTRIBUTION_FIELDS = MODEL_FIELDS + BIAS_FIELDS + (
    "edge_sequence_index",
    "edge_index",
    "midpoint_axial_nm",
    "axial_segment_lower_nm",
    "axial_segment_upper_nm",
    "axial_segment_length_nm",
    "interface_radius_nm",
    "interface_radius_cm",
    "axial_segment_length_cm",
    "interface_area_cm2",
    "electric_field_signed_outward_V_cm",
    "electric_field_abs_V_cm",
    "fowler_nordheim_exponent",
    "fowler_nordheim_current_density_A_cm2",
    "log_fowler_nordheim_current_density_A_cm2",
    "log10_fowler_nordheim_current_density_A_cm2",
    "local_tunneling_current_A",
    "log_local_tunneling_current_A",
    "log10_local_tunneling_current_A",
    "fraction_of_total_tunneling_current",
    "diagnostic_field_threshold_V_cm",
    "above_diagnostic_field_threshold",
    "included_in_primary_integration",
)

FN_LOG_DOMAIN_FIELDS = MODEL_FIELDS + BIAS_FIELDS + (
    "aggregation_method",
    "parameter_role",
    "barrier_height_eV",
    "effective_mass_ratio",
    "representative_field_abs_V_cm",
    "total_interface_area_cm2",
    "edge_count",
    "current_density_A_cm2",
    "log_current_density_A_cm2",
    "log10_current_density_A_cm2",
    "total_tunneling_current_A",
    "log_total_tunneling_current_A",
    "log10_total_tunneling_current_A",
    "diagnostic_field_threshold_V_cm",
    "diagnostic_above_threshold_edge_count",
    "diagnostic_above_threshold_area_fraction",
    "charge_to_remove_for_target_C_cm2",
    "negative_control_time_diagnostic_s",
    "negative_control_time_diagnostic_log10_s",
    "time_result_role",
)


def _model_metadata() -> dict:
    metadata = dict(ep.MODEL_METADATA)
    if metadata != {
        "model_class": "electron_FN_diagnostic",
        "physical_role": "negative_control",
        "model_status": "baseline_negative_result",
        "calibration_status": "uncalibrated",
        "predictive_erase_model": False,
    }:
        raise RuntimeError("ERASE negative-control metadata is inconsistent.")
    return metadata


def _bias_metadata(requested_gate_voltage_V: float, state) -> dict:
    return {
        "requested_gate_voltage_V": float(requested_gate_voltage_V),
        "actual_gate_voltage_V": float(state.gate_voltage_V),
        "source_voltage_V": float(state.source_voltage_V),
        "drain_voltage_V": float(state.drain_voltage_V),
        "trap_density_cm3": float(state.trap_density_cm3),
        "trap_sheet_density_cm2": ep.volume_to_sheet_density(
            state.trap_density_cm3
        ),
    }


def _csv_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def validate_csv_rows(
    fieldnames: Sequence[str],
    rows: Sequence[Mapping],
) -> None:
    """Reject missing/extra columns before writing a scientific result."""

    if len(set(fieldnames)) != len(fieldnames):
        raise ValueError("CSV field names must be unique.")
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        actual = set(row)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"CSV row {index} schema mismatch; missing={missing}, extra={extra}."
            )


def write_csv_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping],
) -> None:
    validate_csv_rows(fieldnames, rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: _csv_value(row[name]) for name in fieldnames}
            )


def _relative_difference(first: float, second: float) -> float:
    scale = max(abs(float(first)), abs(float(second)), 1.0e-300)
    return abs(float(first) - float(second)) / scale


def _radial_sign(field_V_cm: float) -> str:
    value = float(field_V_cm)
    if value > 0.0:
        return "outward_center_to_gate"
    if value < 0.0:
        return "inward_gate_to_center"
    return "zero"


def _negative_control_time_diagnostic(log_current_density: float) -> dict:
    """Return constant-field charge/J time as a non-predictive diagnostic."""

    sheet_density_change = (
        ep.INITIAL_TRAP_SHEET_DENSITY_CM2
        - ep.TARGET_RESIDUAL_SHEET_DENSITY_CM2
    )
    charge_to_remove = ep.ELEMENTARY_CHARGE_C * sheet_density_change
    log_charge = math.log(charge_to_remove)
    if log_current_density == -math.inf:
        log_time = math.inf
    else:
        log_time = log_charge - float(log_current_density)
    if log_time == math.inf or log_time > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        time_s = math.inf
    else:
        time_s = math.exp(log_time)
    return {
        "charge_to_remove_for_target_C_cm2": charge_to_remove,
        "negative_control_time_diagnostic_s": time_s,
        "negative_control_time_diagnostic_log10_s": (
            math.inf if log_time == math.inf else log_time / math.log(10.0)
        ),
        "time_result_role": (
            "constant_field_FN_negative_control_diagnostic_not_erase_prediction"
        ),
    }


def _get_drain_current_A(device: str) -> float:
    from devsim import get_contact_current

    return float(
        get_contact_current(
            device=device,
            contact="drain",
            equation="ElectronContinuityEquation",
        )
    )


def _collect_bias_diagnostics(
    solver: EraseStateSolver,
    geometry: Mapping[str, float],
    requested_gate_voltage_V: float,
    gate_ramp_result,
    trap_ramp_result,
) -> dict:
    state = solver.read_state()
    common = {
        **_model_metadata(),
        **_bias_metadata(requested_gate_voltage_V, state),
    }

    layer_rows_raw, layer_summary = field_audit.extract_layer_voltage_drop_rows(
        device=solver.runtime.device_name,
        geometry=geometry,
        minimum_axial_nm=ep.ACTIVE_AXIAL_START_NM,
        maximum_axial_nm=ep.ACTIVE_AXIAL_STOP_NM,
    )
    layer_by_region = {row["region"]: row for row in layer_rows_raw}

    tunnel_statistics = get_active_inner_interface_radial_field_statistics(
        device=solver.runtime.device_name,
        region="TunnelOxide",
        minimum_axial_nm=ep.ACTIVE_AXIAL_START_NM,
        maximum_axial_nm=ep.ACTIVE_AXIAL_STOP_NM,
    )
    mean_signed = float(
        tunnel_statistics["area_weighted_field_mean_signed_V_cm"]
    )
    mean_abs = float(tunnel_statistics["area_weighted_field_mean_abs_V_cm"])
    maximum_abs = float(tunnel_statistics["field_max_abs_V_cm"])
    total_area = float(tunnel_statistics["total_interface_area_cm2"])

    mean_fn = edge_fn.evaluate_mean_field_fn(
        mean_field_V_cm=mean_abs,
        total_interface_area_cm2=total_area,
    )
    edge_result = edge_fn.integrate_all_edges_log_domain(
        tunnel_statistics["edge_data"]
    )

    tunnel_outer = field_audit.get_region_boundary_radial_field(
        device=solver.runtime.device_name,
        region="TunnelOxide",
        boundary="outer",
    )
    trap_inner = field_audit.get_region_boundary_radial_field(
        device=solver.runtime.device_name,
        region="ChargeTrap",
        boundary="inner",
    )
    trap_outer = field_audit.get_region_boundary_radial_field(
        device=solver.runtime.device_name,
        region="ChargeTrap",
        boundary="outer",
    )
    blocking_inner = field_audit.get_region_boundary_radial_field(
        device=solver.runtime.device_name,
        region="BlockingOxide",
        boundary="inner",
    )
    tunnel_trap_D = field_audit.calculate_displacement_consistency(
        tunnel_outer,
        material.eps_tunnel_oxide,
        trap_inner,
        material.eps_charge_trap,
    )
    trap_blocking_D = field_audit.calculate_displacement_consistency(
        trap_outer,
        material.eps_charge_trap,
        blocking_inner,
        material.eps_blocking_oxide,
    )

    layer_rows = []
    for layer_row in layer_rows_raw:
        layer_rows.append(
            {
                **common,
                **layer_row,
                **layer_summary,
            }
        )

    tunnel_drop_field = float(
        layer_by_region["TunnelOxide"][
            "average_field_from_voltage_drop_outward_V_cm"
        ]
    )
    field_row = {
        **common,
        "electric_field_model_definition": (
            "(Potential@n0-Potential@n1)*EdgeInverseLength"
        ),
        "mesh_coordinate_unit": "cm",
        "potential_unit": "V",
        "electric_field_unit": "V/cm",
        "field_unit_audit": (
            "verified_from_cm_coordinates_and_Potential_in_V"
        ),
        "radial_positive_direction": "cylinder_center_to_gate",
        "electron_force_direction_for_positive_radial_field": (
            "gate_to_channel"
        ),
        "tunnel_radial_field_sign": _radial_sign(mean_signed),
        "devsim_tunnel_field_mean_signed_outward_V_cm": mean_signed,
        "devsim_tunnel_field_mean_abs_V_cm": mean_abs,
        "devsim_tunnel_field_max_abs_V_cm": maximum_abs,
        "tunnel_field_from_voltage_drop_outward_V_cm": tunnel_drop_field,
        "tunnel_field_definition_relative_difference": _relative_difference(
            mean_signed,
            tunnel_drop_field,
        ),
        "active_interface_edge_count": int(
            tunnel_statistics["active_interface_edge_count"]
        ),
        "active_interface_area_cm2": total_area,
        "tunnel_trap_inner_epsilon_E_C_cm2": tunnel_trap_D[
            "inner_epsilon_E_C_cm2"
        ],
        "tunnel_trap_outer_epsilon_E_C_cm2": tunnel_trap_D[
            "outer_epsilon_E_C_cm2"
        ],
        "tunnel_trap_epsilon_E_residual_C_cm2": tunnel_trap_D[
            "epsilon_E_residual_C_cm2"
        ],
        "tunnel_trap_epsilon_E_relative_residual": tunnel_trap_D[
            "epsilon_E_relative_residual"
        ],
        "tunnel_trap_extraction_method": tunnel_trap_D["extraction_method"],
        "trap_blocking_inner_epsilon_E_C_cm2": trap_blocking_D[
            "inner_epsilon_E_C_cm2"
        ],
        "trap_blocking_outer_epsilon_E_C_cm2": trap_blocking_D[
            "outer_epsilon_E_C_cm2"
        ],
        "trap_blocking_epsilon_E_residual_C_cm2": trap_blocking_D[
            "epsilon_E_residual_C_cm2"
        ],
        "trap_blocking_epsilon_E_relative_residual": trap_blocking_D[
            "epsilon_E_relative_residual"
        ],
        "trap_blocking_extraction_method": trap_blocking_D[
            "extraction_method"
        ],
        "layer_voltage_sum_relative_residual": layer_summary[
            "layer_voltage_sum_relative_residual"
        ],
    }

    edge_rows = []
    for raw_edge in edge_result["edge_results"]:
        edge_rows.append(
            {
                **common,
                "edge_sequence_index": raw_edge["edge_sequence_index"],
                "edge_index": raw_edge["edge_index"],
                "midpoint_axial_nm": raw_edge["midpoint_axial_nm"],
                "axial_segment_lower_nm": raw_edge[
                    "axial_segment_lower_nm"
                ],
                "axial_segment_upper_nm": raw_edge[
                    "axial_segment_upper_nm"
                ],
                "axial_segment_length_nm": raw_edge[
                    "axial_segment_length_nm"
                ],
                "interface_radius_nm": raw_edge["interface_radius_nm"],
                "interface_radius_cm": raw_edge["interface_radius_cm"],
                "axial_segment_length_cm": raw_edge[
                    "axial_segment_length_cm"
                ],
                "interface_area_cm2": raw_edge["interface_area_cm2"],
                "electric_field_signed_outward_V_cm": raw_edge[
                    "electric_field_signed_V_cm"
                ],
                "electric_field_abs_V_cm": raw_edge[
                    "electric_field_abs_V_cm"
                ],
                "fowler_nordheim_exponent": raw_edge[
                    "fowler_nordheim_exponent"
                ],
                "fowler_nordheim_current_density_A_cm2": raw_edge[
                    "fowler_nordheim_current_density_A_cm2"
                ],
                "log_fowler_nordheim_current_density_A_cm2": raw_edge[
                    "log_fowler_nordheim_current_density_A_cm2"
                ],
                "log10_fowler_nordheim_current_density_A_cm2": raw_edge[
                    "log10_fowler_nordheim_current_density_A_cm2"
                ],
                "local_tunneling_current_A": raw_edge[
                    "local_tunneling_current_A"
                ],
                "log_local_tunneling_current_A": raw_edge[
                    "log_local_tunneling_current_A"
                ],
                "log10_local_tunneling_current_A": raw_edge[
                    "log10_local_tunneling_current_A"
                ],
                "fraction_of_total_tunneling_current": raw_edge[
                    "fraction_of_total_tunneling_current"
                ],
                "diagnostic_field_threshold_V_cm": raw_edge[
                    "diagnostic_field_threshold_V_cm"
                ],
                "above_diagnostic_field_threshold": raw_edge[
                    "above_diagnostic_field_threshold"
                ],
                "included_in_primary_integration": raw_edge[
                    "included_in_primary_integration"
                ],
            }
        )

    mean_time = _negative_control_time_diagnostic(
        mean_fn["log_fowler_nordheim_current_density_A_cm2"]
    )
    edge_time = _negative_control_time_diagnostic(
        edge_result["log_effective_current_density_A_cm2"]
    )
    fn_rows = [
        {
            **common,
            "aggregation_method": mean_fn["aggregation_method"],
            "parameter_role": mean_fn["parameter_role"],
            "barrier_height_eV": mean_fn["barrier_height_eV"],
            "effective_mass_ratio": mean_fn["effective_mass_ratio"],
            "representative_field_abs_V_cm": mean_abs,
            "total_interface_area_cm2": total_area,
            "edge_count": 1,
            "current_density_A_cm2": mean_fn[
                "fowler_nordheim_current_density_A_cm2"
            ],
            "log_current_density_A_cm2": mean_fn[
                "log_fowler_nordheim_current_density_A_cm2"
            ],
            "log10_current_density_A_cm2": mean_fn[
                "log10_fowler_nordheim_current_density_A_cm2"
            ],
            "total_tunneling_current_A": mean_fn[
                "total_tunneling_current_A"
            ],
            "log_total_tunneling_current_A": mean_fn[
                "log_total_tunneling_current_A"
            ],
            "log10_total_tunneling_current_A": mean_fn[
                "log10_total_tunneling_current_A"
            ],
            "diagnostic_field_threshold_V_cm": "",
            "diagnostic_above_threshold_edge_count": "",
            "diagnostic_above_threshold_area_fraction": "",
            **mean_time,
        },
        {
            **common,
            "aggregation_method": edge_result["aggregation_method"],
            "parameter_role": edge_result["parameter_role"],
            "barrier_height_eV": edge_result["barrier_height_eV"],
            "effective_mass_ratio": edge_result["effective_mass_ratio"],
            "representative_field_abs_V_cm": mean_abs,
            "total_interface_area_cm2": edge_result[
                "total_interface_area_cm2"
            ],
            "edge_count": edge_result["edge_count"],
            "current_density_A_cm2": edge_result[
                "effective_current_density_A_cm2"
            ],
            "log_current_density_A_cm2": edge_result[
                "log_effective_current_density_A_cm2"
            ],
            "log10_current_density_A_cm2": edge_result[
                "log10_effective_current_density_A_cm2"
            ],
            "total_tunneling_current_A": edge_result[
                "total_tunneling_current_A"
            ],
            "log_total_tunneling_current_A": edge_result[
                "log_total_tunneling_current_A"
            ],
            "log10_total_tunneling_current_A": edge_result[
                "log10_total_tunneling_current_A"
            ],
            "diagnostic_field_threshold_V_cm": edge_result[
                "diagnostic_field_threshold_V_cm"
            ],
            "diagnostic_above_threshold_edge_count": edge_result[
                "diagnostic_above_threshold_edge_count"
            ],
            "diagnostic_above_threshold_area_fraction": edge_result[
                "diagnostic_above_threshold_area_fraction"
            ],
            **edge_time,
        },
    ]

    last_step = state.last_successful_step
    gate_attempts = () if gate_ramp_result is None else gate_ramp_result.attempts
    trap_attempts = () if trap_ramp_result is None else trap_ramp_result.attempts
    voltage_row = {
        **common,
        "target_residual_sheet_density_cm2": (
            ep.TARGET_RESIDUAL_SHEET_DENSITY_CM2
        ),
        "exact_target_reached": math.isclose(
            state.gate_voltage_V,
            float(requested_gate_voltage_V),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ),
        "converged": True,
        "error_message": "",
        "drain_current_A": _get_drain_current_A(solver.runtime.device_name),
        "gate_ramp_attempt_count": len(gate_attempts),
        "gate_ramp_failed_attempt_count": sum(
            not attempt.success for attempt in gate_attempts
        ),
        "trap_ramp_attempt_count": len(trap_attempts),
        "trap_ramp_failed_attempt_count": sum(
            not attempt.success for attempt in trap_attempts
        ),
        "last_successful_step_kind": "" if last_step is None else last_step.kind,
        "last_successful_step_label": "" if last_step is None else last_step.label,
        "last_successful_step_value": "" if last_step is None else last_step.value,
        "last_successful_step_size": "" if last_step is None else last_step.step_size,
        "devsim_tunnel_field_mean_signed_outward_V_cm": mean_signed,
        "devsim_tunnel_field_mean_abs_V_cm": mean_abs,
        "devsim_tunnel_field_max_abs_V_cm": maximum_abs,
        "average_field_FN_current_density_A_cm2": mean_fn[
            "fowler_nordheim_current_density_A_cm2"
        ],
        "average_field_FN_log10_current_density_A_cm2": mean_fn[
            "log10_fowler_nordheim_current_density_A_cm2"
        ],
        "edge_logsum_effective_current_density_A_cm2": edge_result[
            "effective_current_density_A_cm2"
        ],
        "edge_logsum_log10_effective_current_density_A_cm2": edge_result[
            "log10_effective_current_density_A_cm2"
        ],
        "negative_control_time_diagnostic_log10_s_mean_field": mean_time[
            "negative_control_time_diagnostic_log10_s"
        ],
        "negative_control_time_diagnostic_log10_s_edge_resolved": edge_time[
            "negative_control_time_diagnostic_log10_s"
        ],
    }
    return {
        "voltage_row": voltage_row,
        "layer_rows": layer_rows,
        "layer_summary": layer_summary,
        "field_row": field_row,
        "edge_rows": edge_rows,
        "fn_rows": fn_rows,
    }


def _failed_voltage_row(
    solver: EraseStateSolver,
    requested_gate_voltage_V: float,
    error: Exception,
    trap_ramp_result,
) -> dict:
    state = solver.read_state()
    common = {
        **_model_metadata(),
        **_bias_metadata(requested_gate_voltage_V, state),
    }
    attempts = tuple(getattr(error, "attempts", ()))
    trap_attempts = () if trap_ramp_result is None else trap_ramp_result.attempts
    last_step = state.last_successful_step
    empty_diagnostics = {
        "devsim_tunnel_field_mean_signed_outward_V_cm": "",
        "devsim_tunnel_field_mean_abs_V_cm": "",
        "devsim_tunnel_field_max_abs_V_cm": "",
        "average_field_FN_current_density_A_cm2": "",
        "average_field_FN_log10_current_density_A_cm2": "",
        "edge_logsum_effective_current_density_A_cm2": "",
        "edge_logsum_log10_effective_current_density_A_cm2": "",
        "negative_control_time_diagnostic_log10_s_mean_field": "",
        "negative_control_time_diagnostic_log10_s_edge_resolved": "",
    }
    return {
        **common,
        "target_residual_sheet_density_cm2": ep.TARGET_RESIDUAL_SHEET_DENSITY_CM2,
        "exact_target_reached": False,
        "converged": False,
        "error_message": str(error),
        "drain_current_A": _get_drain_current_A(solver.runtime.device_name),
        "gate_ramp_attempt_count": len(attempts),
        "gate_ramp_failed_attempt_count": sum(
            not attempt.success for attempt in attempts
        ),
        "trap_ramp_attempt_count": len(trap_attempts),
        "trap_ramp_failed_attempt_count": sum(
            not attempt.success for attempt in trap_attempts
        ),
        "last_successful_step_kind": "" if last_step is None else last_step.kind,
        "last_successful_step_label": "" if last_step is None else last_step.label,
        "last_successful_step_value": "" if last_step is None else last_step.value,
        "last_successful_step_size": "" if last_step is None else last_step.step_size,
        **empty_diagnostics,
    }


def _output_paths(output_directory: Path) -> dict[str, Path]:
    return {
        filename: output_directory / filename for filename in OUTPUT_FILENAMES
    }


def _ensure_new_candidate_paths(output_directory: Path) -> dict[str, Path]:
    paths = _output_paths(output_directory)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "Refusing to overwrite an existing ERASE candidate artifact: "
            f"{formatted}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    return paths


def _format_scientific(value) -> str:
    if value == "":
        return ""
    numeric = float(value)
    if math.isinf(numeric):
        return "inf" if numeric > 0.0 else "-inf"
    return f"{numeric:.6e}"


def build_readme(
    geometry: Mapping[str, float],
    voltage_rows: Sequence[Mapping],
    field_rows: Sequence[Mapping],
    trap_ramp_result,
) -> str:
    """Build the auditable, non-predictive candidate README."""

    field_by_gate = {
        float(row["requested_gate_voltage_V"]): row for row in field_rows
    }
    lines = [
        "# ERASE reimplementation: electron-FN negative control",
        "",
        "This directory is an independent, report-based Stage-1 reimplementation. "
        "It is not a reproduction of missing teammate source code.",
        "",
        "## Mandatory model classification",
        "",
        "- `model_class = electron_FN_diagnostic`",
        "- `physical_role = negative_control`",
        "- `model_status = baseline_negative_result`",
        "- `calibration_status = uncalibrated`",
        "- `predictive_erase_model = false`",
        "",
        "The 3.56 eV barrier and 0.28 m0 tunneling mass come from the existing "
        "program-electron model. They are used only to demonstrate the negative "
        "control and are not physical hole-ERASE parameters. Any charge/J time "
        "reported in the FN CSV is a constant-field diagnostic, not an ERASE "
        "prediction.",
        "",
        "## Runtime import and geometry/material audit",
        "",
        "- Geometry source of truth: `compact_handoff_parameters.py`, imported "
        "through `05_program_erase/device_structure.py`.",
        "- Material source of truth: `compact_handoff_parameters.py`, imported "
        "through `05_program_erase/material_parameters.py`.",
        "- Coordinate system: 2D axisymmetric (`x=r`, `y=z`), mesh coordinates "
        "in cm.",
        f"- Core / MoS2 / tunnel / trap / blocking: "
        f"{geometry['core_radius_nm']:.1f} / "
        f"{geometry['mos2_thickness_nm']:.1f} / "
        f"{geometry['tunnel_oxide_thickness_nm']:.1f} / "
        f"{geometry['charge_trap_thickness_nm']:.1f} / "
        f"{geometry['blocking_oxide_thickness_nm']:.1f} nm.",
        f"- Channel length: {geometry['channel_length_nm']:.1f} nm; active "
        f"diagnostic window: {ep.ACTIVE_AXIAL_START_NM:.1f}--"
        f"{ep.ACTIVE_AXIAL_STOP_NM:.1f} nm.",
        f"- Relative permittivity: Al2O3 = "
        f"{ep.compact_parameters.AL2O3_RELATIVE_PERMITTIVITY:.2f}; HfO2 = "
        f"{ep.compact_parameters.HFO2_RELATIVE_PERMITTIVITY:.2f}.",
        "- The stale `TUNNEL_OXIDE_THICKNESS_NM = 4.0` in the legacy "
        "tunneling-parameter module is not used for runtime geometry or voltage "
        "drop. Runtime tunnel thickness is 3 nm.",
        "",
        "## Numerical sequence and rollback",
        "",
        "1. Build the empty-trap device and solve Poisson only at VS=VD=VG=0 V.",
        "2. Reinitialize MoS2 Electrons from the equilibrium model, add electron "
        "current/continuity and ideal-ohmic electron contacts, then solve coupled "
        "electron drift-diffusion.",
        "3. Continue Ntrap from 0 to 2e18 cm^-3 using adaptive steps.",
        "4. Continue VG sequentially through 0, -4, -6, -8, -10, -12 V at "
        "VS=VD=0 V.",
        "5. Before every continuation trial, save Potential in all five physical "
        "regions, MoS2 Electrons, three contact biases, Ntrap, and last-success "
        "metadata. A failed trial restores those arrays/scalars directly before "
        "halving the step.",
        "",
        f"Trap continuation trials: {len(trap_ramp_result.attempts)} total, "
        f"{sum(not attempt.success for attempt in trap_ramp_result.attempts)} "
        "failed-and-restored.",
        "",
        "## Field, sign, and unit conventions",
        "",
        "`ElectricField = (Potential@n0 - Potential@n1) * EdgeInverseLength`, "
        "so Potential in V and coordinates in cm give V/cm. Edge orientation is "
        "corrected so positive radial field points from the cylinder center to "
        "the gate; electron force then points toward the channel. Layer fields "
        "are independently checked as `-(V_outer - V_inner)/thickness_cm`. "
        "Layer voltage closure retains independently sampled interface jumps.",
        "",
        "The epsilon*E comparison uses the adjacent first radial cell in each "
        "material. It is a mesh-dependent consistency proxy, not an exact "
        "surface-field evaluation.",
        "",
        "## Converged voltage results",
        "",
        "| requested VG (V) | actual VG (V) | mean |Etox| (V/cm) | max |Etox| "
        "(V/cm) | log10 J mean | log10 J edge |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in voltage_rows:
        requested = float(row["requested_gate_voltage_V"])
        field = field_by_gate.get(requested)
        if not row["converged"] or field is None:
            lines.append(
                f"| {requested:.1f} | {_format_scientific(row['actual_gate_voltage_V'])} "
                "| failed | failed | failed | failed |"
            )
            continue
        lines.append(
            f"| {requested:.1f} | {float(row['actual_gate_voltage_V']):.12g} | "
            f"{float(field['devsim_tunnel_field_mean_abs_V_cm']):.6e} | "
            f"{float(field['devsim_tunnel_field_max_abs_V_cm']):.6e} | "
            f"{_format_scientific(row['average_field_FN_log10_current_density_A_cm2'])} | "
            f"{_format_scientific(row['edge_logsum_log10_effective_current_density_A_cm2'])} |"
        )
    lines.extend(
        [
            "",
            "The sanity criterion is qualitative: mean and maximum tunnel fields "
            "must rise with |VG|, while both FN diagnostics remain far too small "
            "to explain practical erase. Differences from report numbers are "
            "attributed to the canonical 3 nm runtime mesh and the explicitly "
            "documented active-interface/area definitions rather than fitted.",
            "",
            "## Explicit limitations",
            "",
            "- No hole continuity, generation, transport, or tunneling equation.",
            "- No BTBT, GIDL, impact ionization, or body hot-hole injection.",
            "- No trapped-charge time feedback and no residual-charge erased state.",
            "- No retention model and no contact-model change.",
            "- Ntrap=0 is only an empty reference; this run never creates or labels "
            "it as an actual erased state.",
            "- Missing for a future hole-assisted stage: hole barrier provenance, "
            "hole tunneling mass, injection prefactor, neutralization efficiency, "
            "hole-generation law/source, occupancy dependence, and calibration "
            "data. No values are invented here.",
            "",
            "## Files",
            "",
            "- `erase_voltage_sweep.csv`: convergence, state, field, and FN summary.",
            "- `erase_layer_voltage_drop.csv`: independently sampled layer drops.",
            "- `erase_field_consistency.csv`: units/sign, field, and epsilon*E audits.",
            "- `erase_edge_distribution.csv`: every active cylindrical interface "
            "edge; all are included in the primary integration.",
            "- `erase_fn_log_domain.csv`: mean-field and all-edge log-domain FN "
            "negative-control diagnostics.",
            "",
        ]
    )
    return "\n".join(lines)


def write_candidate_outputs(
    output_directory: Path,
    geometry: Mapping[str, float],
    voltage_rows: Sequence[Mapping],
    layer_rows: Sequence[Mapping],
    field_rows: Sequence[Mapping],
    edge_rows: Sequence[Mapping],
    fn_rows: Sequence[Mapping],
    trap_ramp_result,
) -> dict[str, Path]:
    paths = _ensure_new_candidate_paths(output_directory)
    write_csv_rows(
        paths["erase_voltage_sweep.csv"],
        VOLTAGE_SWEEP_FIELDS,
        voltage_rows,
    )
    write_csv_rows(
        paths["erase_layer_voltage_drop.csv"],
        LAYER_VOLTAGE_FIELDS,
        layer_rows,
    )
    write_csv_rows(
        paths["erase_field_consistency.csv"],
        FIELD_CONSISTENCY_FIELDS,
        field_rows,
    )
    write_csv_rows(
        paths["erase_edge_distribution.csv"],
        EDGE_DISTRIBUTION_FIELDS,
        edge_rows,
    )
    write_csv_rows(
        paths["erase_fn_log_domain.csv"],
        FN_LOG_DOMAIN_FIELDS,
        fn_rows,
    )
    paths["erase_baseline_README.md"].write_text(
        build_readme(
            geometry=geometry,
            voltage_rows=voltage_rows,
            field_rows=field_rows,
            trap_ramp_result=trap_ramp_result,
        ),
        encoding="utf-8",
    )
    return paths


def run_baseline(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    write_outputs: bool = True,
) -> dict:
    """Run the complete Stage-1 baseline and optionally emit its candidate."""

    ep.validate_parameters()
    edge_fn.validate_proxy_parameters()
    solver = EraseStateSolver()

    print("[ERASE baseline] Poisson-only -> electron-DD staged initialization")
    geometry = solver.initialize_staged()
    solver.set_terminal_bias("source", ep.SOURCE_VOLTAGE_V)
    solver.set_terminal_bias("drain", ep.DRAIN_VOLTAGE_V)
    solver.set_terminal_bias("gate", 0.0)

    print("[ERASE baseline] trapped-electron continuation to 2e18 cm^-3")
    trap_ramp_result = solver.ramp_trap_density(
        ep.INITIAL_TRAP_VOLUME_DENSITY_CM3,
        label="initial trapped-electron continuation",
    )

    voltage_rows = []
    layer_rows = []
    layer_summaries = []
    field_rows = []
    edge_rows = []
    fn_rows = []
    terminal_error = None

    for target_gate_voltage in ep.ERASE_GATE_TARGETS_V:
        print(f"[ERASE baseline] target VG={target_gate_voltage:+.3f} V")
        gate_ramp_result = None
        try:
            if not math.isclose(
                solver.read_state().gate_voltage_V,
                target_gate_voltage,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                gate_ramp_result = solver.ramp_gate(
                    target_gate_voltage,
                    label=f"negative gate target {target_gate_voltage:+.3f} V",
                )
            diagnostic = _collect_bias_diagnostics(
                solver=solver,
                geometry=geometry,
                requested_gate_voltage_V=target_gate_voltage,
                gate_ramp_result=gate_ramp_result,
                trap_ramp_result=(
                    trap_ramp_result if target_gate_voltage == 0.0 else None
                ),
            )
        except AdaptiveRampError as error:
            voltage_rows.append(
                _failed_voltage_row(
                    solver,
                    target_gate_voltage,
                    error,
                    trap_ramp_result if target_gate_voltage == 0.0 else None,
                )
            )
            terminal_error = error
            break

        voltage_rows.append(diagnostic["voltage_row"])
        layer_rows.extend(diagnostic["layer_rows"])
        layer_summaries.append(diagnostic["layer_summary"])
        field_rows.append(diagnostic["field_row"])
        edge_rows.extend(diagnostic["edge_rows"])
        fn_rows.extend(diagnostic["fn_rows"])

        print(
            "  converged: mean/max |Etox|="
            f"{diagnostic['field_row']['devsim_tunnel_field_mean_abs_V_cm']:.6e}/"
            f"{diagnostic['field_row']['devsim_tunnel_field_max_abs_V_cm']:.6e} V/cm; "
            "edge log10 Jeff="
            f"{diagnostic['voltage_row']['edge_logsum_log10_effective_current_density_A_cm2']:.6e}"
        )

    if terminal_error is None:
        if len(voltage_rows) != len(ep.ERASE_GATE_TARGETS_V):
            raise RuntimeError("The requested negative gate sweep is incomplete.")
        if not all(row["exact_target_reached"] for row in voltage_rows):
            raise RuntimeError("At least one gate target was not reached exactly.")
        field_audit.validate_monotonic_field(field_rows)
        field_audit.validate_layer_voltage_closure(layer_summaries)

    paths = {}
    if write_outputs:
        paths = write_candidate_outputs(
            output_directory=Path(output_directory),
            geometry=geometry,
            voltage_rows=voltage_rows,
            layer_rows=layer_rows,
            field_rows=field_rows,
            edge_rows=edge_rows,
            fn_rows=fn_rows,
            trap_ramp_result=trap_ramp_result,
        )

    result = {
        "geometry": geometry,
        "voltage_rows": voltage_rows,
        "layer_rows": layer_rows,
        "field_rows": field_rows,
        "edge_rows": edge_rows,
        "fn_rows": fn_rows,
        "trap_ramp_result": trap_ramp_result,
        "output_paths": paths,
    }
    if terminal_error is not None:
        raise RuntimeError(
            "Negative gate sweep stopped after rollback at the last converged state."
        ) from terminal_error
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="New candidate output directory (existing artifacts are never overwritten).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Run and validate the simulation without writing candidate files.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    result = run_baseline(
        output_directory=arguments.output_dir,
        write_outputs=not arguments.no_write,
    )
    print("[ERASE baseline] completed")
    if result["output_paths"]:
        for path in result["output_paths"].values():
            print(f"  {path}")
    else:
        print("  no files written (--no-write)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
