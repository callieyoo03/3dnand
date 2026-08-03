# ============================================================
# MoS2 cylindrical GAA charge-trap memory
# Positive-gate-bias electric-field and tunneling sweep
#
# Main functions
# --------------
# 1. Build a parameterized device structure
# 2. Solve the empty-trap electron drift-diffusion state
# 3. Sweep positive gate voltage
# 4. Extract dielectric radial fields
# 5. Extract full-channel interface field
# 6. Extract active-window interface field
# 7. Calculate FN tunneling from the mean field
# 8. Calculate edge-resolved FN tunneling
# 9. Integrate tunneling current over cylindrical interface area
# 10. Save all results to CSV
# ============================================================

import csv
import math
import os
from pathlib import Path

from devsim import (
    get_contact_current,
    get_device_list,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
)

from device_structure import (
    create_structure,
    device,
)

from physics_models import (
    REGIONS,
    create_contact_models,
    create_continuity_equation,
    create_doping,
    create_electron_current_model,
    create_equilibrium_carrier_models,
    create_interface_models,
    create_poisson_model,
    create_solution_variables,
    set_material_parameters,
)

from trap_models import (
    create_static_trap_framework,
    set_trapped_electron_density,
)

from field_extraction import (
    DIELECTRIC_REGIONS,
    DEFAULT_ACTIVE_AXIAL_MINIMUM_NM,
    DEFAULT_ACTIVE_AXIAL_MAXIMUM_NM,
    get_active_inner_interface_radial_field_statistics,
    get_dielectric_field_statistics,
    get_inner_interface_radial_field_statistics,
    print_active_interface_field_statistics,
    print_inner_interface_field_statistics,
    print_region_field_statistics,
)

from tunneling_models import (
    evaluate_tunneling_current_density,
    print_tunneling_result,
)

import trap_parameters as tp
import tunneling_parameters as tunnel_params


# ============================================================
# Geometry input
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = float(
    os.environ.get(
        "TUNNEL_OXIDE_THICKNESS_NM",
        "4.0",
    )
)


# ============================================================
# Active interface window
# ============================================================

ACTIVE_AXIAL_MINIMUM_NM = float(
    os.environ.get(
        "ACTIVE_AXIAL_MINIMUM_NM",
        str(
            DEFAULT_ACTIVE_AXIAL_MINIMUM_NM
        ),
    )
)

ACTIVE_AXIAL_MAXIMUM_NM = float(
    os.environ.get(
        "ACTIVE_AXIAL_MAXIMUM_NM",
        str(
            DEFAULT_ACTIVE_AXIAL_MAXIMUM_NM
        ),
    )
)


# ============================================================
# Sweep settings
# ============================================================

GATE_VOLTAGES = (
    0.0,
    2.0,
    4.0,
    6.0,
)

SOURCE_VOLTAGE = 0.0

DRAIN_VOLTAGE = 0.05

TRAPPED_ELECTRON_DENSITY = 0.0
# cm^-3

GATE_INITIAL_STEP = 0.10

GATE_MINIMUM_STEP = 1.0e-3

DRAIN_INITIAL_STEP = 0.01

DRAIN_MINIMUM_STEP = 5.0e-4


# ============================================================
# Output path
# ============================================================

OUTPUT_DIRECTORY = Path(
    "results"
)

default_output_filename = (
    f"positive_field_sweep_"
    f"tox_{TUNNEL_OXIDE_THICKNESS_NM:.1f}nm.csv"
)

OUTPUT_CSV = Path(
    os.environ.get(
        "POSITIVE_FIELD_SWEEP_OUTPUT",
        str(
            OUTPUT_DIRECTORY
            / default_output_filename
        ),
    )
)


# ============================================================
# Console utility
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


# ============================================================
# Solver utilities
# ============================================================

def solve_dc(
    maximum_iterations=None,
):
    if maximum_iterations is None:
        maximum_iterations = (
            tp.SOLVER_MAXIMUM_ITERATIONS
        )

    solve(
        type="dc",
        solver_type="direct",
        absolute_error=(
            tp.SOLVER_ABSOLUTE_ERROR
        ),
        relative_error=(
            tp.SOLVER_RELATIVE_ERROR
        ),
        maximum_iterations=maximum_iterations,
    )


def set_terminal_bias(
    terminal,
    voltage,
):
    set_parameter(
        device=device,
        name=f"{terminal}_bias",
        value=float(
            voltage
        ),
    )


def get_drain_current():
    return get_contact_current(
        device=device,
        contact="drain",
        equation="ElectronContinuityEquation",
    )


def verify_device_structure():
    devices = get_device_list()

    if device not in devices:
        raise RuntimeError(
            f'Device "{device}" was not created.'
        )

    regions = get_region_list(
        device=device,
    )

    for region in REGIONS:
        if region not in regions:
            raise RuntimeError(
                f'Region "{region}" is missing.'
            )

    print(
        "Device structure verification completed."
    )


# ============================================================
# Tunneling-result validation
# ============================================================

def validate_tunneling_result(
    tunneling_result,
):
    required_keys = (
        "electric_field_signed_V_cm",
        "electric_field_abs_V_cm",
        "fowler_nordheim_exponent",
        "fowler_nordheim_current_density_A_cm2",
        "signed_fowler_nordheim_current_density_A_cm2",
        "direct_tunneling_current_density_A_cm2",
        "trap_assisted_tunneling_current_density_A_cm2",
        "total_tunneling_current_density_A_cm2",
    )

    for key in required_keys:
        if key not in tunneling_result:
            raise RuntimeError(
                f'Missing tunneling result key: "{key}"'
            )

    finite_keys = (
        "electric_field_signed_V_cm",
        "electric_field_abs_V_cm",
        "fowler_nordheim_exponent",
        "fowler_nordheim_current_density_A_cm2",
        "signed_fowler_nordheim_current_density_A_cm2",
        "direct_tunneling_current_density_A_cm2",
        "trap_assisted_tunneling_current_density_A_cm2",
        "total_tunneling_current_density_A_cm2",
    )

    for key in finite_keys:
        value = float(
            tunneling_result[key]
        )

        if not math.isfinite(
            value
        ):
            raise RuntimeError(
                f'Non-finite tunneling value for "{key}".'
            )

    nonnegative_keys = (
        "electric_field_abs_V_cm",
        "fowler_nordheim_current_density_A_cm2",
        "direct_tunneling_current_density_A_cm2",
        "trap_assisted_tunneling_current_density_A_cm2",
        "total_tunneling_current_density_A_cm2",
    )

    for key in nonnegative_keys:
        value = float(
            tunneling_result[key]
        )

        if value < 0.0:
            raise RuntimeError(
                f'Negative tunneling magnitude for "{key}".'
            )


# ============================================================
# Adaptive voltage ramp
# ============================================================

def adaptive_voltage_ramp(
    terminal,
    start_voltage,
    target_voltage,
    initial_step,
    minimum_step,
    label,
):
    current_voltage = float(
        start_voltage
    )

    target_voltage = float(
        target_voltage
    )

    if abs(
        target_voltage
        - current_voltage
    ) <= 1.0e-15:
        return current_voltage

    direction = (
        1.0
        if target_voltage > current_voltage
        else -1.0
    )

    step = abs(
        float(
            initial_step
        )
    )

    maximum_step = step

    while (
        direction
        * (
            target_voltage
            - current_voltage
        )
        > 1.0e-12
    ):
        remaining_voltage = abs(
            target_voltage
            - current_voltage
        )

        trial_step = min(
            step,
            remaining_voltage,
        )

        trial_voltage = (
            current_voltage
            + direction * trial_step
        )

        print(
            f"{label}: attempting "
            f"{trial_voltage:+.6f} V "
            f"(step = {trial_step:.6f} V)"
        )

        set_terminal_bias(
            terminal=terminal,
            voltage=trial_voltage,
        )

        try:
            solve_dc()

        except Exception as solve_error:
            print(
                f"{label}: convergence failed at "
                f"{trial_voltage:+.6f} V."
            )

            set_terminal_bias(
                terminal=terminal,
                voltage=current_voltage,
            )

            try:
                solve_dc(
                    maximum_iterations=200,
                )

            except Exception as recovery_error:
                raise RuntimeError(
                    f"{label}: could not recover "
                    f"the converged solution at "
                    f"{current_voltage:+.6f} V."
                ) from recovery_error

            step *= 0.5

            print(
                f"{label}: step reduced to "
                f"{step:.6f} V."
            )

            if step < minimum_step:
                raise RuntimeError(
                    f"{label}: voltage step became smaller "
                    f"than the minimum "
                    f"{minimum_step:.6f} V."
                ) from solve_error

            continue

        current_voltage = trial_voltage

        drain_current = get_drain_current()

        print(
            f"{label}: converged at "
            f"{current_voltage:+.6f} V, "
            f"ID = {drain_current:+.6e} A"
        )

        step = min(
            maximum_step,
            step * 1.5,
        )

    return current_voltage


# ============================================================
# Mean-field tunneling evaluation
# ============================================================

def evaluate_interface_tunneling(
    tunnel_interface_statistics,
):
    """
    Calculate tunneling using the full-channel signed mean
    interface field.

    This preserves the earlier J(mean E) calculation so that it
    can be compared with edge-resolved tunneling.
    """

    interface_field_signed_V_cm = float(
        tunnel_interface_statistics[
            "field_mean_signed_V_cm"
        ]
    )

    tunneling_result = (
        evaluate_tunneling_current_density(
            electric_field_V_cm=(
                interface_field_signed_V_cm
            )
        )
    )

    validate_tunneling_result(
        tunneling_result
    )

    program_time_s = float(
        tunnel_params.PROGRAM_TIME_S
    )

    total_current_density_A_cm2 = float(
        tunneling_result[
            "total_tunneling_current_density_A_cm2"
        ]
    )

    injected_charge_density_C_cm2 = (
        total_current_density_A_cm2
        * program_time_s
    )

    injected_electron_sheet_density_cm2 = (
        injected_charge_density_C_cm2
        / tunnel_params.q
    )

    return {
        "tunneling_result": (
            tunneling_result
        ),

        "program_time_s": (
            program_time_s
        ),

        "total_current_density_A_cm2": (
            total_current_density_A_cm2
        ),

        "injected_charge_density_C_cm2": (
            injected_charge_density_C_cm2
        ),

        "injected_electron_sheet_density_cm2": (
            injected_electron_sheet_density_cm2
        ),
    }


# ============================================================
# Edge-resolved tunneling evaluation
# ============================================================

def evaluate_edge_resolved_interface_tunneling(
    active_interface_statistics,
):
    """
    Calculate tunneling independently at every active interface
    edge.

    For each edge:

        I_i = J(E_i) * A_i

    Integrated current:

        I_total = sum_i I_i

    Area-averaged effective current density:

        J_effective = I_total / A_total
    """

    edge_data = (
        active_interface_statistics[
            "edge_data"
        ]
    )

    if not edge_data:
        raise RuntimeError(
            "Active interface edge data are empty."
        )

    total_interface_area_cm2 = float(
        active_interface_statistics[
            "total_interface_area_cm2"
        ]
    )

    if total_interface_area_cm2 <= 0.0:
        raise RuntimeError(
            "Total active interface area must be positive."
        )

    program_time_s = float(
        tunnel_params.PROGRAM_TIME_S
    )

    total_tunneling_current_A = 0.0

    minimum_local_current_density_A_cm2 = None

    maximum_local_current_density_A_cm2 = 0.0

    maximum_local_field_abs_V_cm = 0.0

    edge_results = []

    for edge in edge_data:
        edge_field_signed_V_cm = float(
            edge[
                "radial_field_outward_V_cm"
            ]
        )

        edge_area_cm2 = float(
            edge[
                "cylindrical_interface_area_cm2"
            ]
        )

        if edge_area_cm2 <= 0.0:
            raise RuntimeError(
                "An interface edge has a non-positive area."
            )

        tunneling_result = (
            evaluate_tunneling_current_density(
                electric_field_V_cm=(
                    edge_field_signed_V_cm
                )
            )
        )

        validate_tunneling_result(
            tunneling_result
        )

        local_current_density_A_cm2 = float(
            tunneling_result[
                "total_tunneling_current_density_A_cm2"
            ]
        )

        local_tunneling_current_A = (
            local_current_density_A_cm2
            * edge_area_cm2
        )

        total_tunneling_current_A += (
            local_tunneling_current_A
        )

        maximum_local_current_density_A_cm2 = max(
            maximum_local_current_density_A_cm2,
            local_current_density_A_cm2,
        )

        if (
            minimum_local_current_density_A_cm2
            is None
        ):
            minimum_local_current_density_A_cm2 = (
                local_current_density_A_cm2
            )

        else:
            minimum_local_current_density_A_cm2 = min(
                minimum_local_current_density_A_cm2,
                local_current_density_A_cm2,
            )

        maximum_local_field_abs_V_cm = max(
            maximum_local_field_abs_V_cm,
            abs(
                edge_field_signed_V_cm
            ),
        )

        edge_results.append(
            {
                "edge_index": (
                    edge[
                        "edge_index"
                    ]
                ),

                "midpoint_axial_nm": (
                    edge[
                        "midpoint_axial_nm"
                    ]
                ),

                "axial_segment_lower_nm": (
                    edge[
                        "axial_segment_lower_nm"
                    ]
                ),

                "axial_segment_upper_nm": (
                    edge[
                        "axial_segment_upper_nm"
                    ]
                ),

                "axial_segment_length_nm": (
                    edge[
                        "axial_segment_length_nm"
                    ]
                ),

                "interface_radius_nm": (
                    edge[
                        "interface_radius_nm"
                    ]
                ),

                "interface_area_cm2": (
                    edge_area_cm2
                ),

                "electric_field_signed_V_cm": (
                    edge_field_signed_V_cm
                ),

                "electric_field_abs_V_cm": (
                    abs(
                        edge_field_signed_V_cm
                    )
                ),

                "fowler_nordheim_exponent": (
                    tunneling_result[
                        "fowler_nordheim_exponent"
                    ]
                ),

                "fowler_nordheim_current_density_A_cm2": (
                    tunneling_result[
                        "fowler_nordheim_current_density_A_cm2"
                    ]
                ),

                "total_tunneling_current_density_A_cm2": (
                    local_current_density_A_cm2
                ),

                "local_tunneling_current_A": (
                    local_tunneling_current_A
                ),
            }
        )

    effective_current_density_A_cm2 = (
        total_tunneling_current_A
        / total_interface_area_cm2
    )

    total_injected_charge_C = (
        total_tunneling_current_A
        * program_time_s
    )

    total_injected_electron_count = (
        total_injected_charge_C
        / tunnel_params.q
    )

    area_averaged_injected_charge_density_C_cm2 = (
        effective_current_density_A_cm2
        * program_time_s
    )

    area_averaged_injected_electron_sheet_density_cm2 = (
        area_averaged_injected_charge_density_C_cm2
        / tunnel_params.q
    )

    return {
        "edge_count": (
            len(edge_results)
        ),

        "edge_results": (
            edge_results
        ),

        "active_axial_minimum_nm": (
            active_interface_statistics[
                "active_axial_minimum_nm"
            ]
        ),

        "active_axial_maximum_nm": (
            active_interface_statistics[
                "active_axial_maximum_nm"
            ]
        ),

        "total_interface_area_cm2": (
            total_interface_area_cm2
        ),

        "total_tunneling_current_A": (
            total_tunneling_current_A
        ),

        "effective_current_density_A_cm2": (
            effective_current_density_A_cm2
        ),

        "minimum_local_current_density_A_cm2": (
            minimum_local_current_density_A_cm2
        ),

        "maximum_local_current_density_A_cm2": (
            maximum_local_current_density_A_cm2
        ),

        "maximum_local_field_abs_V_cm": (
            maximum_local_field_abs_V_cm
        ),

        "program_time_s": (
            program_time_s
        ),

        "total_injected_charge_C": (
            total_injected_charge_C
        ),

        "total_injected_electron_count": (
            total_injected_electron_count
        ),

        "area_averaged_injected_charge_density_C_cm2": (
            area_averaged_injected_charge_density_C_cm2
        ),

        "area_averaged_injected_electron_sheet_density_cm2": (
            area_averaged_injected_electron_sheet_density_cm2
        ),
    }


def print_edge_resolved_tunneling_result(
    result,
):
    print()
    print(
        "EDGE-RESOLVED ACTIVE-INTERFACE TUNNELING"
    )

    print(
        f'  selected edge count       = '
        f'{result["edge_count"]}'
    )

    print(
        f'  active axial range        = '
        f'{result["active_axial_minimum_nm"]:.3f} '
        f'to '
        f'{result["active_axial_maximum_nm"]:.3f} nm'
    )

    print(
        f'  total interface area      = '
        f'{result["total_interface_area_cm2"]:.6e} cm^2'
    )

    print(
        f'  maximum local field       = '
        f'{result["maximum_local_field_abs_V_cm"]:.6e} '
        f'V/cm'
    )

    print(
        f'  minimum local J           = '
        f'{result["minimum_local_current_density_A_cm2"]:.6e} '
        f'A/cm^2'
    )

    print(
        f'  maximum local J           = '
        f'{result["maximum_local_current_density_A_cm2"]:.6e} '
        f'A/cm^2'
    )

    print(
        f'  effective current density = '
        f'{result["effective_current_density_A_cm2"]:.6e} '
        f'A/cm^2'
    )

    print(
        f'  total tunneling current   = '
        f'{result["total_tunneling_current_A"]:.6e} A'
    )

    print(
        f'  total injected charge     = '
        f'{result["total_injected_charge_C"]:.6e} C'
    )

    print(
        f'  injected electron count   = '
        f'{result["total_injected_electron_count"]:.6e}'
    )

    print(
        f'  average injected sheet N  = '
        f'{result["area_averaged_injected_electron_sheet_density_cm2"]:.6e} '
        f'cm^-2'
    )


# ============================================================
# Main simulation
# ============================================================

def main():
    print_section(
        "STEP 1 : CREATE DEVICE STRUCTURE"
    )

    geometry = create_structure(
        tunnel_oxide_thickness_nm=(
            TUNNEL_OXIDE_THICKNESS_NM
        ),
    )

    verify_device_structure()


    print_section(
        "STEP 2 : MATERIAL PARAMETERS"
    )

    set_material_parameters()


    print_section(
        "STEP 3 : DOPING"
    )

    create_doping()


    print_section(
        "STEP 4 : SOLUTION VARIABLES"
    )

    create_solution_variables()


    print_section(
        "STEP 5 : EQUILIBRIUM CARRIER MODELS"
    )

    create_equilibrium_carrier_models()


    print_section(
        "STEP 6 : INITIALIZE ELECTRONS"
    )

    set_node_values(
        device=device,
        region="MoS2",
        name="Electrons",
        init_from="EquilibriumElectrons",
    )

    print(
        "Electrons initialized from "
        "EquilibriumElectrons."
    )


    print_section(
        "STEP 7 : POISSON EQUATION"
    )

    create_poisson_model()


    print_section(
        "STEP 8 : STATIC TRAP FRAMEWORK"
    )

    create_static_trap_framework()


    print_section(
        "STEP 9 : ELECTRON CURRENT MODEL"
    )

    create_electron_current_model()


    print_section(
        "STEP 10 : ELECTRON CONTINUITY EQUATION"
    )

    create_continuity_equation()


    print_section(
        "STEP 11 : INTERFACE EQUATIONS"
    )

    create_interface_models()


    print_section(
        "STEP 12 : CONTACT EQUATIONS"
    )

    create_contact_models()


    print_section(
        "STEP 13 : EMPTY-TRAP ZERO-BIAS EQUILIBRIUM"
    )

    set_terminal_bias(
        terminal="source",
        voltage=SOURCE_VOLTAGE,
    )

    set_terminal_bias(
        terminal="drain",
        voltage=0.0,
    )

    set_terminal_bias(
        terminal="gate",
        voltage=0.0,
    )

    set_trapped_electron_density(
        TRAPPED_ELECTRON_DENSITY
    )

    solve_dc(
        maximum_iterations=200,
    )

    print(
        "Empty-trap zero-bias equilibrium completed."
    )


    print_section(
        "STEP 14 : DRAIN READ-BIAS RAMP"
    )

    current_drain_voltage = (
        adaptive_voltage_ramp(
            terminal="drain",
            start_voltage=0.0,
            target_voltage=DRAIN_VOLTAGE,
            initial_step=DRAIN_INITIAL_STEP,
            minimum_step=DRAIN_MINIMUM_STEP,
            label="Drain ramp",
        )
    )


    print_section(
        "STEP 15 : POSITIVE GATE FIELD AND TUNNELING SWEEP"
    )

    OUTPUT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    current_gate_voltage = 0.0

    for target_gate_voltage in GATE_VOLTAGES:
        current_gate_voltage = (
            adaptive_voltage_ramp(
                terminal="gate",
                start_voltage=current_gate_voltage,
                target_voltage=target_gate_voltage,
                initial_step=GATE_INITIAL_STEP,
                minimum_step=GATE_MINIMUM_STEP,
                label=(
                    f"Gate target "
                    f"{target_gate_voltage:+.3f} V"
                ),
            )
        )

        drain_current = get_drain_current()

        dielectric_statistics = (
            get_dielectric_field_statistics(
                device=device,
            )
        )

        tunnel_interface_statistics = (
            get_inner_interface_radial_field_statistics(
                device=device,
                region="TunnelOxide",
            )
        )

        active_tunnel_interface_statistics = (
            get_active_inner_interface_radial_field_statistics(
                device=device,
                region="TunnelOxide",
                minimum_axial_nm=(
                    ACTIVE_AXIAL_MINIMUM_NM
                ),
                maximum_axial_nm=(
                    ACTIVE_AXIAL_MAXIMUM_NM
                ),
            )
        )

        interface_tunneling = (
            evaluate_interface_tunneling(
                tunnel_interface_statistics
            )
        )

        edge_resolved_tunneling = (
            evaluate_edge_resolved_interface_tunneling(
                active_tunnel_interface_statistics
            )
        )

        tunneling_result = (
            interface_tunneling[
                "tunneling_result"
            ]
        )

        program_time_s = (
            interface_tunneling[
                "program_time_s"
            ]
        )

        injected_charge_density_C_cm2 = (
            interface_tunneling[
                "injected_charge_density_C_cm2"
            ]
        )

        injected_electron_sheet_density_cm2 = (
            interface_tunneling[
                "injected_electron_sheet_density_cm2"
            ]
        )

        print()
        print(
            f"FIELD RESULT: "
            f"TOX = "
            f"{geometry['tunnel_oxide_thickness_nm']:.3f} nm, "
            f"VG = {current_gate_voltage:+.3f} V, "
            f"VD = {current_drain_voltage:+.3f} V, "
            f"ID = {drain_current:+.6e} A"
        )

        for region in DIELECTRIC_REGIONS:
            print_region_field_statistics(
                statistics=(
                    dielectric_statistics[
                        region
                    ]
                ),
            )

        print()
        print(
            "Full-channel MoS2 / TunnelOxide interface"
        )

        print_inner_interface_field_statistics(
            statistics=(
                tunnel_interface_statistics
            ),
        )

        print()
        print(
            "Active MoS2 / TunnelOxide interface"
        )

        print_active_interface_field_statistics(
            statistics=(
                active_tunnel_interface_statistics
            ),
        )

        print()
        print(
            "MEAN-FIELD TUNNELING APPROXIMATION"
        )

        print_tunneling_result(
            result=tunneling_result,
        )

        print()
        print(
            f"Program time = "
            f"{program_time_s:.6e} s"
        )

        print(
            f"Mean-field injected charge density = "
            f"{injected_charge_density_C_cm2:.6e} "
            f"C/cm^2"
        )

        print(
            f"Mean-field injected electron sheet density = "
            f"{injected_electron_sheet_density_cm2:.6e} "
            f"cm^-2"
        )

        print_edge_resolved_tunneling_result(
            result=edge_resolved_tunneling,
        )

        for region in DIELECTRIC_REGIONS:
            statistics = (
                dielectric_statistics[
                    region
                ]
            )

            result_row = {
                "tunnel_oxide_thickness_nm": (
                    geometry[
                        "tunnel_oxide_thickness_nm"
                    ]
                ),

                "charge_trap_thickness_nm": (
                    geometry[
                        "charge_trap_thickness_nm"
                    ]
                ),

                "blocking_oxide_thickness_nm": (
                    geometry[
                        "blocking_oxide_thickness_nm"
                    ]
                ),

                "mos2_outer_radius_nm": (
                    geometry[
                        "mos2_outer_radius_nm"
                    ]
                ),

                "tunnel_oxide_outer_radius_nm": (
                    geometry[
                        "tunnel_oxide_outer_radius_nm"
                    ]
                ),

                "gate_outer_radius_nm": (
                    geometry[
                        "gate_outer_radius_nm"
                    ]
                ),

                "gate_voltage_V": (
                    current_gate_voltage
                ),

                "drain_voltage_V": (
                    current_drain_voltage
                ),

                "trap_density_cm3": (
                    TRAPPED_ELECTRON_DENSITY
                ),

                "drain_current_A": (
                    drain_current
                ),

                "region": (
                    region
                ),

                "edge_count": (
                    statistics[
                        "edge_count"
                    ]
                ),

                "total_edge_count": (
                    statistics[
                        "total_edge_count"
                    ]
                ),

                "radial_edge_count": (
                    statistics[
                        "radial_edge_count"
                    ]
                ),

                "axial_edge_count": (
                    statistics[
                        "axial_edge_count"
                    ]
                ),

                "diagonal_edge_count": (
                    statistics[
                        "diagonal_edge_count"
                    ]
                ),

                "degenerate_edge_count": (
                    statistics[
                        "degenerate_edge_count"
                    ]
                ),

                "field_min_V_cm": (
                    statistics[
                        "field_min_V_cm"
                    ]
                ),

                "field_max_V_cm": (
                    statistics[
                        "field_max_V_cm"
                    ]
                ),

                "field_mean_signed_V_cm": (
                    statistics[
                        "field_mean_signed_V_cm"
                    ]
                ),

                "field_mean_abs_V_cm": (
                    statistics[
                        "field_mean_abs_V_cm"
                    ]
                ),

                "field_max_abs_V_cm": (
                    statistics[
                        "field_max_abs_V_cm"
                    ]
                ),

                "tunnel_interface_edge_count": (
                    tunnel_interface_statistics[
                        "inner_interface_edge_count"
                    ]
                ),

                "tunnel_interface_inner_radius_nm": (
                    tunnel_interface_statistics[
                        "inner_radius_nm"
                    ]
                ),

                "tunnel_interface_outer_radius_nm": (
                    tunnel_interface_statistics[
                        "outer_radius_nm"
                    ]
                ),

                "tunnel_interface_midpoint_radius_nm": (
                    tunnel_interface_statistics[
                        "midpoint_radius_nm"
                    ]
                ),

                "tunnel_interface_minimum_axial_nm": (
                    tunnel_interface_statistics[
                        "minimum_axial_nm"
                    ]
                ),

                "tunnel_interface_maximum_axial_nm": (
                    tunnel_interface_statistics[
                        "maximum_axial_nm"
                    ]
                ),

                "tunnel_interface_field_min_V_cm": (
                    tunnel_interface_statistics[
                        "field_min_V_cm"
                    ]
                ),

                "tunnel_interface_field_max_V_cm": (
                    tunnel_interface_statistics[
                        "field_max_V_cm"
                    ]
                ),

                "tunnel_interface_field_mean_signed_V_cm": (
                    tunnel_interface_statistics[
                        "field_mean_signed_V_cm"
                    ]
                ),

                "tunnel_interface_field_mean_abs_V_cm": (
                    tunnel_interface_statistics[
                        "field_mean_abs_V_cm"
                    ]
                ),

                "tunnel_interface_field_max_abs_V_cm": (
                    tunnel_interface_statistics[
                        "field_max_abs_V_cm"
                    ]
                ),

                "active_axial_minimum_nm": (
                    edge_resolved_tunneling[
                        "active_axial_minimum_nm"
                    ]
                ),

                "active_axial_maximum_nm": (
                    edge_resolved_tunneling[
                        "active_axial_maximum_nm"
                    ]
                ),

                "active_interface_edge_count": (
                    edge_resolved_tunneling[
                        "edge_count"
                    ]
                ),

                "active_interface_area_cm2": (
                    edge_resolved_tunneling[
                        "total_interface_area_cm2"
                    ]
                ),

                "active_interface_field_mean_signed_V_cm": (
                    active_tunnel_interface_statistics[
                        "field_mean_signed_V_cm"
                    ]
                ),

                "active_interface_field_mean_abs_V_cm": (
                    active_tunnel_interface_statistics[
                        "field_mean_abs_V_cm"
                    ]
                ),

                "active_interface_field_area_weighted_signed_V_cm": (
                    active_tunnel_interface_statistics[
                        "area_weighted_field_mean_signed_V_cm"
                    ]
                ),

                "active_interface_field_area_weighted_abs_V_cm": (
                    active_tunnel_interface_statistics[
                        "area_weighted_field_mean_abs_V_cm"
                    ]
                ),

                "active_interface_field_max_abs_V_cm": (
                    active_tunnel_interface_statistics[
                        "field_max_abs_V_cm"
                    ]
                ),

                "tunneling_barrier_height_eV": (
                    tunnel_params.BARRIER_HEIGHT_EV
                ),

                "tunneling_effective_mass_ratio": (
                    tunnel_params.TUNNEL_EFFECTIVE_MASS_RATIO
                ),

                "program_time_s": (
                    program_time_s
                ),

                "fowler_nordheim_exponent": (
                    tunneling_result[
                        "fowler_nordheim_exponent"
                    ]
                ),

                "fowler_nordheim_current_density_A_cm2": (
                    tunneling_result[
                        "fowler_nordheim_current_density_A_cm2"
                    ]
                ),

                "signed_fowler_nordheim_current_density_A_cm2": (
                    tunneling_result[
                        "signed_fowler_nordheim_current_density_A_cm2"
                    ]
                ),

                "direct_tunneling_current_density_A_cm2": (
                    tunneling_result[
                        "direct_tunneling_current_density_A_cm2"
                    ]
                ),

                "trap_assisted_tunneling_current_density_A_cm2": (
                    tunneling_result[
                        "trap_assisted_tunneling_current_density_A_cm2"
                    ]
                ),

                "total_tunneling_current_density_A_cm2": (
                    tunneling_result[
                        "total_tunneling_current_density_A_cm2"
                    ]
                ),

                "injected_charge_density_C_cm2": (
                    injected_charge_density_C_cm2
                ),

                "injected_electron_sheet_density_cm2": (
                    injected_electron_sheet_density_cm2
                ),

                "edge_resolved_effective_current_density_A_cm2": (
                    edge_resolved_tunneling[
                        "effective_current_density_A_cm2"
                    ]
                ),

                "edge_resolved_total_tunneling_current_A": (
                    edge_resolved_tunneling[
                        "total_tunneling_current_A"
                    ]
                ),

                "edge_resolved_minimum_local_current_density_A_cm2": (
                    edge_resolved_tunneling[
                        "minimum_local_current_density_A_cm2"
                    ]
                ),

                "edge_resolved_maximum_local_current_density_A_cm2": (
                    edge_resolved_tunneling[
                        "maximum_local_current_density_A_cm2"
                    ]
                ),

                "edge_resolved_maximum_local_field_abs_V_cm": (
                    edge_resolved_tunneling[
                        "maximum_local_field_abs_V_cm"
                    ]
                ),

                "edge_resolved_total_injected_charge_C": (
                    edge_resolved_tunneling[
                        "total_injected_charge_C"
                    ]
                ),

                "edge_resolved_total_injected_electron_count": (
                    edge_resolved_tunneling[
                        "total_injected_electron_count"
                    ]
                ),

                "edge_resolved_average_injected_charge_density_C_cm2": (
                    edge_resolved_tunneling[
                        "area_averaged_injected_charge_density_C_cm2"
                    ]
                ),

                "edge_resolved_average_injected_electron_sheet_density_cm2": (
                    edge_resolved_tunneling[
                        "area_averaged_injected_electron_sheet_density_cm2"
                    ]
                ),
            }

            results.append(
                result_row
            )


    print_section(
        "STEP 16 : SAVE FIELD AND TUNNELING RESULTS"
    )

    field_names = (
        "tunnel_oxide_thickness_nm",
        "charge_trap_thickness_nm",
        "blocking_oxide_thickness_nm",
        "mos2_outer_radius_nm",
        "tunnel_oxide_outer_radius_nm",
        "gate_outer_radius_nm",
        "gate_voltage_V",
        "drain_voltage_V",
        "trap_density_cm3",
        "drain_current_A",
        "region",
        "edge_count",
        "total_edge_count",
        "radial_edge_count",
        "axial_edge_count",
        "diagonal_edge_count",
        "degenerate_edge_count",
        "field_min_V_cm",
        "field_max_V_cm",
        "field_mean_signed_V_cm",
        "field_mean_abs_V_cm",
        "field_max_abs_V_cm",
        "tunnel_interface_edge_count",
        "tunnel_interface_inner_radius_nm",
        "tunnel_interface_outer_radius_nm",
        "tunnel_interface_midpoint_radius_nm",
        "tunnel_interface_minimum_axial_nm",
        "tunnel_interface_maximum_axial_nm",
        "tunnel_interface_field_min_V_cm",
        "tunnel_interface_field_max_V_cm",
        "tunnel_interface_field_mean_signed_V_cm",
        "tunnel_interface_field_mean_abs_V_cm",
        "tunnel_interface_field_max_abs_V_cm",
        "active_axial_minimum_nm",
        "active_axial_maximum_nm",
        "active_interface_edge_count",
        "active_interface_area_cm2",
        "active_interface_field_mean_signed_V_cm",
        "active_interface_field_mean_abs_V_cm",
        "active_interface_field_area_weighted_signed_V_cm",
        "active_interface_field_area_weighted_abs_V_cm",
        "active_interface_field_max_abs_V_cm",
        "tunneling_barrier_height_eV",
        "tunneling_effective_mass_ratio",
        "program_time_s",
        "fowler_nordheim_exponent",
        "fowler_nordheim_current_density_A_cm2",
        "signed_fowler_nordheim_current_density_A_cm2",
        "direct_tunneling_current_density_A_cm2",
        "trap_assisted_tunneling_current_density_A_cm2",
        "total_tunneling_current_density_A_cm2",
        "injected_charge_density_C_cm2",
        "injected_electron_sheet_density_cm2",
        "edge_resolved_effective_current_density_A_cm2",
        "edge_resolved_total_tunneling_current_A",
        "edge_resolved_minimum_local_current_density_A_cm2",
        "edge_resolved_maximum_local_current_density_A_cm2",
        "edge_resolved_maximum_local_field_abs_V_cm",
        "edge_resolved_total_injected_charge_C",
        "edge_resolved_total_injected_electron_count",
        "edge_resolved_average_injected_charge_density_C_cm2",
        "edge_resolved_average_injected_electron_sheet_density_cm2",
    )

    with OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    print(
        f'Field and tunneling results written to '
        f'"{OUTPUT_CSV}".'
    )


    print_section(
        "POSITIVE FIELD AND TUNNELING SWEEP SUCCESSFUL"
    )

    print(
        f"TunnelOxide thickness: "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        f"Active axial window: "
        f"{ACTIVE_AXIAL_MINIMUM_NM:.3f} "
        f"to "
        f"{ACTIVE_AXIAL_MAXIMUM_NM:.3f} nm"
    )

    print(
        f"Gate voltages: "
        f"{GATE_VOLTAGES}"
    )

    print(
        f"Drain voltage: "
        f"{current_drain_voltage:+.3f} V"
    )

    print(
        f"Trap density: "
        f"{TRAPPED_ELECTRON_DENSITY:.3e} cm^-3"
    )

    print(
        f"Program time: "
        f"{tunnel_params.PROGRAM_TIME_S:.3e} s"
    )

    print(
        f"Barrier height: "
        f"{tunnel_params.BARRIER_HEIGHT_EV:.3f} eV"
    )

    print(
        f"Effective mass ratio: "
        f"{tunnel_params.TUNNEL_EFFECTIVE_MASS_RATIO:.3f}"
    )

    print(
        f'Output CSV: "{OUTPUT_CSV}"'
    )


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_POSITIVE_FIELD_SWEEP SCRIPT STARTED"
    )

    main()