# ============================================================
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
# Positive-gate-bias dielectric field sweep
#
# Purpose:
#   1. Build the electron-only drift-diffusion device
#   2. Use the empty trapped-charge state
#   3. Ramp drain to the read voltage
#   4. Sweep only positive gate voltages
#   5. Extract dielectric electric-field statistics
#   6. Save results to results/positive_field_sweep.csv
#
# Important:
#   - No negative gate-voltage sweep is performed.
#   - No tunneling current is calculated yet.
#   - Both radial and axial edges are still included.
# ============================================================

import csv
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
    get_dielectric_field_statistics,
    print_region_field_statistics,
)

import trap_parameters as tp


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

OUTPUT_DIRECTORY = Path("results")

OUTPUT_CSV = (
    OUTPUT_DIRECTORY
    / "positive_field_sweep.csv"
)


# ============================================================
# Utility functions
# ============================================================

def print_section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


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
        value=float(voltage),
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
        target_voltage - current_voltage
    ) <= 1.0e-15:
        return current_voltage

    direction = (
        1.0
        if target_voltage > current_voltage
        else -1.0
    )

    step = abs(
        initial_step
    )

    maximum_step = abs(
        initial_step
    )

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
                    f"{label}: voltage step became "
                    f"smaller than the minimum "
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
# Build and initialize device
# ============================================================

print_section(
    "STEP 1 : CREATE DEVICE STRUCTURE"
)

create_structure()
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


# ============================================================
# Empty-trap equilibrium
# ============================================================

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


# ============================================================
# Drain voltage ramp
# ============================================================

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


# ============================================================
# Positive gate sweep
# ============================================================

print_section(
    "STEP 15 : POSITIVE GATE FIELD SWEEP"
)

OUTPUT_DIRECTORY.mkdir(
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

    print()
    print(
        f"FIELD RESULT: "
        f"VG = {current_gate_voltage:+.3f} V, "
        f"VD = {current_drain_voltage:+.3f} V, "
        f"ID = {drain_current:+.6e} A"
    )

    for region in DIELECTRIC_REGIONS:
        statistics = (
            dielectric_statistics[region]
        )

        print_region_field_statistics(
            statistics=statistics,
        )

        result_row = {
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
            "region": region,
            "edge_count": (
                statistics["edge_count"]
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
        }

        results.append(
            result_row
        )


# ============================================================
# Save CSV
# ============================================================

print_section(
    "STEP 16 : SAVE FIELD RESULTS"
)

field_names = (
    "gate_voltage_V",
    "drain_voltage_V",
    "trap_density_cm3",
    "drain_current_A",
    "region",
    "edge_count",
    "field_min_V_cm",
    "field_max_V_cm",
    "field_mean_signed_V_cm",
    "field_mean_abs_V_cm",
    "field_max_abs_V_cm",
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
    f'Field-sweep results written to '
    f'"{OUTPUT_CSV}".'
)


# ============================================================
# Final status
# ============================================================

print_section(
    "POSITIVE FIELD SWEEP SUCCESSFUL"
)

print(
    f"Gate voltages: {GATE_VOLTAGES}"
)

print(
    f"Drain voltage: "
    f"{current_drain_voltage:+.3f} V"
)

print(
    f"Trap density: "
    f"{TRAPPED_ELECTRON_DENSITY:.3e} cm^-3"
)