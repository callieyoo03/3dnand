# ============================================================
# MoS2 cylindrical GAA FET
# Electron-only drift-diffusion I_D-V_G simulation
#
# Simulation sequence:
#   1. Build device and physical models
#   2. Solve zero-bias equilibrium
#   3. Ramp drain voltage to 0.05 V
#   4. Ramp gate voltage to -1.0 V
#   5. Sweep gate voltage from -1.0 V to 2.0 V
#   6. Extract drain electron current
#   7. Save I_D-V_G data to CSV
#
# An adaptive voltage ramp is used:
#   - If a solve fails, the voltage step is reduced by half.
#   - The previous converged bias is restored.
# ============================================================

import csv
from pathlib import Path

from devsim import (
    get_contact_current,
    get_contact_list,
    get_device_list,
    get_interface_list,
    get_node_model_list,
    get_node_model_values,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
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


# ============================================================
# Simulation settings
# ============================================================

SOURCE_VOLTAGE = 0.0

DRAIN_TARGET_VOLTAGE = 0.05
DRAIN_INITIAL_STEP = 0.01
DRAIN_MINIMUM_STEP = 0.0005

GATE_START_VOLTAGE = -1.0
GATE_STOP_VOLTAGE = 2.0
GATE_OUTPUT_STEP = 0.1

GATE_RAMP_INITIAL_STEP = 0.05
GATE_RAMP_MINIMUM_STEP = 0.0025

OUTPUT_CSV = Path("id_vg.csv")
OUTPUT_VTK = "mos2_gaa_id_vg_final"


# ============================================================
# Utility functions
# ============================================================

def print_section(title):

    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def solve_dc(
    maximum_iterations=100,
):

    solve(
        type="dc",
        solver_type="direct",
        absolute_error=1.0e10,
        relative_error=1.0e-9,
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


def print_min_max(
    model_name,
    values,
    unit,
):

    if not values:

        print(
            f"{model_name}: no values found."
        )

        return

    print(
        f"{model_name:20s}: "
        f"min = {min(values):.6e}, "
        f"max = {max(values):.6e} "
        f"{unit}"
    )


def verify_structure():

    created_devices = get_device_list()

    if device not in created_devices:

        raise RuntimeError(
            f'Device "{device}" was not created.'
        )

    created_regions = get_region_list(
        device=device,
    )

    for region in REGIONS:

        if region not in created_regions:

            raise RuntimeError(
                f'Region "{region}" is missing.'
            )

    print("Device structure verification completed.")


def verify_solution_variables():

    for region in REGIONS:

        node_models = get_node_model_list(
            device=device,
            region=region,
        )

        if "Potential" not in node_models:

            raise RuntimeError(
                f'Potential is missing in "{region}".'
            )

    mos2_models = get_node_model_list(
        device=device,
        region="MoS2",
    )

    if "Electrons" not in mos2_models:

        raise RuntimeError(
            "Electrons solution is missing in MoS2."
        )

    print("Solution-variable verification completed.")


def create_output_voltage_list(
    start,
    stop,
    step,
):

    voltages = []

    count = int(
        round(
            (stop - start) / step
        )
    )

    for index in range(count + 1):

        voltage = start + index * step

        voltages.append(
            round(voltage, 12)
        )

    return voltages


# ============================================================
# Adaptive bias ramp
# ============================================================

def adaptive_ramp(
    terminal,
    start_voltage,
    target_voltage,
    initial_step,
    minimum_step,
    label,
):

    current_voltage = float(start_voltage)
    target_voltage = float(target_voltage)

    if abs(
        target_voltage - current_voltage
    ) < 1.0e-15:

        return current_voltage

    direction = (
        1.0
        if target_voltage > current_voltage
        else -1.0
    )

    step = abs(initial_step)

    while (
        direction
        * (target_voltage - current_voltage)
        > 1.0e-12
    ):

        remaining = abs(
            target_voltage - current_voltage
        )

        trial_step = min(
            step,
            remaining,
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

        except Exception as error:

            print(
                f"{label}: convergence failed at "
                f"{trial_voltage:+.6f} V."
            )

            print(
                f"{label}: restoring previous "
                f"converged bias "
                f"{current_voltage:+.6f} V."
            )

            set_terminal_bias(
                terminal=terminal,
                voltage=current_voltage,
            )

            # Recover the previous converged operating point.
            try:

                solve_dc(
                    maximum_iterations=150,
                )

            except Exception as recovery_error:

                raise RuntimeError(
                    f"{label}: failed to recover the "
                    f"previous converged solution at "
                    f"{current_voltage:+.6f} V."
                ) from recovery_error

            step *= 0.5

            print(
                f"{label}: reducing step to "
                f"{step:.6f} V."
            )

            if step < minimum_step:

                raise RuntimeError(
                    f"{label}: required voltage step "
                    f"is smaller than the allowed "
                    f"minimum step "
                    f"{minimum_step:.6f} V."
                ) from error

            continue

        current_voltage = trial_voltage

        drain_current = get_drain_current()

        print(
            f"{label}: converged at "
            f"{current_voltage:+.6f} V, "
            f"ID = {drain_current:+.6e} A"
        )

        # After a successful small step, gradually restore
        # the step size toward its initial value.
        step = min(
            abs(initial_step),
            step * 1.5,
        )

    return current_voltage


# ============================================================
# Step 1: Create device structure
# ============================================================

print_section(
    "STEP 1 : CREATE DEVICE STRUCTURE"
)

create_structure()

verify_structure()


# ============================================================
# Step 2: Assign material parameters
# ============================================================

print_section(
    "STEP 2 : MATERIAL PARAMETERS"
)

set_material_parameters()


# ============================================================
# Step 3: Create doping
# ============================================================

print_section(
    "STEP 3 : DOPING"
)

create_doping()


# ============================================================
# Step 4: Create solution variables
# ============================================================

print_section(
    "STEP 4 : SOLUTION VARIABLES"
)

create_solution_variables()

verify_solution_variables()


# ============================================================
# Step 5: Create equilibrium carrier models
# ============================================================

print_section(
    "STEP 5 : EQUILIBRIUM CARRIER MODELS"
)

create_equilibrium_carrier_models()


# ============================================================
# Step 6: Initialize electron solution
# ============================================================

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


# ============================================================
# Step 7: Create Poisson equation
# ============================================================

print_section(
    "STEP 7 : POISSON EQUATION"
)

create_poisson_model()


# ============================================================
# Step 8: Create electron-current model
# ============================================================

print_section(
    "STEP 8 : ELECTRON CURRENT MODEL"
)

create_electron_current_model()


# ============================================================
# Step 9: Create electron continuity equation
# ============================================================

print_section(
    "STEP 9 : ELECTRON CONTINUITY EQUATION"
)

create_continuity_equation()


# ============================================================
# Step 10: Create interface equations
# ============================================================

print_section(
    "STEP 10 : INTERFACE EQUATIONS"
)

create_interface_models()


# ============================================================
# Step 11: Create contact equations
# ============================================================

print_section(
    "STEP 11 : CONTACT EQUATIONS"
)

create_contact_models()


# ============================================================
# Step 12: Print device information
# ============================================================

print_section(
    "STEP 12 : DEVICE INFORMATION"
)

print("Devices:")
print(get_device_list())

print()
print("Regions:")
print(
    get_region_list(
        device=device,
    )
)

print()
print("Interfaces:")
print(
    get_interface_list(
        device=device,
    )
)

print()
print("Contacts:")
print(
    get_contact_list(
        device=device,
    )
)


# ============================================================
# Step 13: Zero-bias equilibrium
# ============================================================

print_section(
    "STEP 13 : ZERO-BIAS EQUILIBRIUM"
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

solve_dc(
    maximum_iterations=150,
)

print(
    "Zero-bias electron-only "
    "drift-diffusion solution completed."
)


# ============================================================
# Step 14: Check equilibrium results
# ============================================================

print_section(
    "STEP 14 : EQUILIBRIUM RESULTS"
)

potential_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="Potential",
)

electron_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="Electrons",
)

equilibrium_hole_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="EquilibriumHoles",
)

print_min_max(
    model_name="Potential",
    values=potential_values,
    unit="V",
)

print_min_max(
    model_name="Electrons",
    values=electron_values,
    unit="cm^-3",
)

print_min_max(
    model_name="Fixed holes",
    values=equilibrium_hole_values,
    unit="cm^-3",
)


# ============================================================
# Step 15: Ramp drain voltage
# ============================================================

print_section(
    "STEP 15 : ADAPTIVE DRAIN RAMP"
)

current_drain_voltage = adaptive_ramp(
    terminal="drain",
    start_voltage=0.0,
    target_voltage=DRAIN_TARGET_VOLTAGE,
    initial_step=DRAIN_INITIAL_STEP,
    minimum_step=DRAIN_MINIMUM_STEP,
    label="Drain ramp",
)

print()
print(
    f"Drain bias reached "
    f"{current_drain_voltage:+.6f} V."
)


# ============================================================
# Step 16: Ramp gate to sweep starting point
# ============================================================

print_section(
    "STEP 16 : ADAPTIVE GATE PRE-RAMP"
)

current_gate_voltage = adaptive_ramp(
    terminal="gate",
    start_voltage=0.0,
    target_voltage=GATE_START_VOLTAGE,
    initial_step=GATE_RAMP_INITIAL_STEP,
    minimum_step=GATE_RAMP_MINIMUM_STEP,
    label="Gate pre-ramp",
)

print()
print(
    f"Gate bias reached "
    f"{current_gate_voltage:+.6f} V."
)


# ============================================================
# Step 17: I_D-V_G sweep
# ============================================================

print_section(
    "STEP 17 : I_D-V_G SWEEP"
)

gate_output_voltages = (
    create_output_voltage_list(
        start=GATE_START_VOLTAGE,
        stop=GATE_STOP_VOLTAGE,
        step=GATE_OUTPUT_STEP,
    )
)

iv_results = []

for target_gate_voltage in gate_output_voltages:

    current_gate_voltage = adaptive_ramp(
        terminal="gate",
        start_voltage=current_gate_voltage,
        target_voltage=target_gate_voltage,
        initial_step=GATE_RAMP_INITIAL_STEP,
        minimum_step=GATE_RAMP_MINIMUM_STEP,
        label=(
            f"Gate sweep target "
            f"{target_gate_voltage:+.3f} V"
        ),
    )

    electron_current = get_drain_current()

    absolute_current = abs(
        electron_current
    )

    iv_results.append(
        {
            "gate_voltage_V": current_gate_voltage,
            "drain_voltage_V": current_drain_voltage,
            "electron_current_A": electron_current,
            "abs_electron_current_A": absolute_current,
        }
    )

    print(
        f"OUTPUT: "
        f"VG = {current_gate_voltage:+.3f} V, "
        f"VD = {current_drain_voltage:+.3f} V, "
        f"ID = {electron_current:+.6e} A, "
        f"|ID| = {absolute_current:.6e} A"
    )


# ============================================================
# Step 18: Save CSV
# ============================================================

print_section(
    "STEP 18 : SAVE I_D-V_G RESULTS"
)

with OUTPUT_CSV.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = (
        "gate_voltage_V",
        "drain_voltage_V",
        "electron_current_A",
        "abs_electron_current_A",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        iv_results
    )

print(
    f'I_D-V_G results saved to "{OUTPUT_CSV}".'
)


# ============================================================
# Step 19: Final solution inspection
# ============================================================

print_section(
    "STEP 19 : FINAL SOLUTION RESULTS"
)

final_potential_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="Potential",
)

final_electron_values = get_node_model_values(
    device=device,
    region="MoS2",
    name="Electrons",
)

print_min_max(
    model_name="Potential",
    values=final_potential_values,
    unit="V",
)

print_min_max(
    model_name="Electrons",
    values=final_electron_values,
    unit="cm^-3",
)


# ============================================================
# Step 20: Export final solution
# ============================================================

print_section(
    "STEP 20 : EXPORT FINAL SOLUTION"
)

write_devices(
    file=OUTPUT_VTK,
    type="vtk",
)

print(
    f'Final VTK solution written to '
    f'"{OUTPUT_VTK}".'
)


# ============================================================
# Final status
# ============================================================

print()
print("=" * 60)
print(
    "ELECTRON-ONLY I_D-V_G "
    "SIMULATION SUCCESSFUL"
)
print("=" * 60)