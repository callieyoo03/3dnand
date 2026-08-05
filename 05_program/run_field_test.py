# ============================================================
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
# Static trapped-charge memory-window simulation
#
# Simulation sequence:
#
#   1. Build the electron-only drift-diffusion FET
#   2. Add fixed trapped-electron charge to HfO2
#   3. Solve the empty-trap equilibrium state
#   4. Ramp drain bias to the read voltage
#   5. For each trapped-electron-density state:
#        a. Ramp gate to 0 V
#        b. Ramp trapped-electron density
#        c. Ramp gate to the sweep start voltage
#        d. Sweep I_D-V_G
#        e. Extract threshold voltage
#   6. Save:
#        - memory_window_id_vg.csv
#        - memory_window_vth.csv
#
# Important:
#
#   This is a static charge-state simulation.
#
#   It does not yet calculate:
#       - tunneling current
#       - capture and emission
#       - program/erase time
#       - retention
#       - endurance
# ============================================================

import csv
import math
from pathlib import Path

from devsim import (
    get_contact_current,
    get_contact_list,
    get_device_list,
    get_interface_list,
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

from trap_models import (
    create_static_trap_framework,
    get_trap_state_information,
    set_trapped_electron_density,
)

from field_extraction import (
    print_dielectric_field_statistics,
)


import trap_parameters as tp

# ============================================================
# Output paths
# ============================================================

MEMORY_WINDOW_CSV = Path(
    tp.MEMORY_WINDOW_CSV
)

THRESHOLD_SUMMARY_CSV = Path(
    tp.THRESHOLD_SUMMARY_CSV
)


# ============================================================
# Additional numerical settings
# ============================================================

TRAP_INITIAL_STEP = 2.0e17
# cm^-3

TRAP_MINIMUM_STEP = 1.0e15
# cm^-3

CURRENT_FLOOR = 1.0e-30
# A
# Used only for logarithmic interpolation.


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
        f"{model_name:24s}: "
        f"min = {min(values):.6e}, "
        f"max = {max(values):.6e} "
        f"{unit}"
    )


def create_output_voltage_list(
    start,
    stop,
    step,
):

    if step <= 0.0:

        raise ValueError(
            "Gate output step must be positive."
        )

    number_of_steps = int(
        round(
            (stop - start) / step
        )
    )

    voltages = []

    for index in range(
        number_of_steps + 1
    ):

        voltage = (
            start
            + index * step
        )

        voltages.append(
            round(voltage, 12)
        )

    return voltages


# ============================================================
# Verification
# ============================================================

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
# Adaptive terminal-voltage ramp
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

        current_voltage = (
            trial_voltage
        )

        drain_current = (
            get_drain_current()
        )

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
# Adaptive trapped-electron-density ramp
# ============================================================

def adaptive_trap_density_ramp(
    start_density,
    target_density,
    initial_step,
    minimum_step,
    label,
):

    current_density = float(
        start_density
    )

    target_density = float(
        target_density
    )

    if abs(
        target_density
        - current_density
    ) <= 1.0:

        return current_density

    direction = (
        1.0
        if target_density > current_density
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
            target_density
            - current_density
        )
        > 1.0
    ):

        remaining_density = abs(
            target_density
            - current_density
        )

        trial_step = min(
            step,
            remaining_density,
        )

        trial_density = (
            current_density
            + direction * trial_step
        )

        print(
            f"{label}: attempting "
            f"Ntrap = {trial_density:.6e} cm^-3 "
            f"(step = {trial_step:.6e} cm^-3)"
        )

        set_trapped_electron_density(
            trial_density
        )

        try:

            solve_dc()

        except Exception as solve_error:

            print(
                f"{label}: convergence failed at "
                f"Ntrap = {trial_density:.6e} cm^-3."
            )

            set_trapped_electron_density(
                current_density
            )

            try:

                solve_dc(
                    maximum_iterations=200,
                )

            except Exception as recovery_error:

                raise RuntimeError(
                    f"{label}: could not recover "
                    f"Ntrap = "
                    f"{current_density:.6e} cm^-3."
                ) from recovery_error

            step *= 0.5

            print(
                f"{label}: trap-density step "
                f"reduced to {step:.6e} cm^-3."
            )

            if step < minimum_step:

                raise RuntimeError(
                    f"{label}: trap-density step "
                    f"became smaller than minimum "
                    f"{minimum_step:.6e} cm^-3."
                ) from solve_error

            continue

        current_density = (
            trial_density
        )

        print(
            f"{label}: converged at "
            f"Ntrap = "
            f"{current_density:.6e} cm^-3."
        )

        step = min(
            maximum_step,
            step * 1.5,
        )

    return current_density


# ============================================================
# Threshold-voltage extraction
# ============================================================

def extract_threshold_voltage(
    state_results,
    threshold_current,
):

    if len(state_results) < 2:

        return None

    threshold_log_current = math.log10(
        max(
            threshold_current,
            CURRENT_FLOOR,
        )
    )

    for index in range(
        len(state_results) - 1
    ):

        point_1 = (
            state_results[index]
        )

        point_2 = (
            state_results[index + 1]
        )

        vg_1 = float(
            point_1["gate_voltage_V"]
        )

        vg_2 = float(
            point_2["gate_voltage_V"]
        )

        current_1 = max(
            float(
                point_1[
                    "abs_drain_current_A"
                ]
            ),
            CURRENT_FLOOR,
        )

        current_2 = max(
            float(
                point_2[
                    "abs_drain_current_A"
                ]
            ),
            CURRENT_FLOOR,
        )

        crossed_threshold = (
            (
                current_1
                <= threshold_current
                <= current_2
            )
            or
            (
                current_2
                <= threshold_current
                <= current_1
            )
        )

        if not crossed_threshold:

            continue

        log_current_1 = math.log10(
            current_1
        )

        log_current_2 = math.log10(
            current_2
        )

        denominator = (
            log_current_2
            - log_current_1
        )

        if abs(denominator) < 1.0e-30:

            return (
                0.5
                * (
                    vg_1
                    + vg_2
                )
            )

        interpolation_fraction = (
            threshold_log_current
            - log_current_1
        ) / denominator

        threshold_voltage = (
            vg_1
            + interpolation_fraction
            * (
                vg_2
                - vg_1
            )
        )

        return threshold_voltage

    return None


# ============================================================
# Initial parameter validation
# ============================================================

print_section(
    "STEP 1 : VALIDATE TRAP PARAMETERS"
)

tp.validate_trap_parameters()

print(
    "Trap parameters validated."
)


# ============================================================
# Build device structure
# ============================================================

print_section(
    "STEP 2 : CREATE DEVICE STRUCTURE"
)

create_structure()

verify_device_structure()


# ============================================================
# Assign material parameters
# ============================================================

print_section(
    "STEP 3 : MATERIAL PARAMETERS"
)

set_material_parameters()


# ============================================================
# Create doping
# ============================================================

print_section(
    "STEP 4 : DOPING"
)

create_doping()


# ============================================================
# Create solution variables
# ============================================================

print_section(
    "STEP 5 : SOLUTION VARIABLES"
)

create_solution_variables()


# ============================================================
# Create equilibrium carrier models
# ============================================================

print_section(
    "STEP 6 : EQUILIBRIUM CARRIER MODELS"
)

create_equilibrium_carrier_models()


# ============================================================
# Initialize mobile electrons
# ============================================================

print_section(
    "STEP 7 : INITIALIZE ELECTRONS"
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
# Create Poisson equation
# ============================================================

print_section(
    "STEP 8 : POISSON EQUATION"
)

create_poisson_model()


# ============================================================
# Add fixed trapped charge to HfO2
# ============================================================

print_section(
    "STEP 9 : STATIC HfO2 TRAP FRAMEWORK"
)

create_static_trap_framework()


# ============================================================
# Create electron-current model
# ============================================================

print_section(
    "STEP 10 : ELECTRON CURRENT MODEL"
)

create_electron_current_model()


# ============================================================
# Create electron continuity equation
# ============================================================

print_section(
    "STEP 11 : ELECTRON CONTINUITY EQUATION"
)

create_continuity_equation()


# ============================================================
# Create interfaces
# ============================================================

print_section(
    "STEP 12 : INTERFACE EQUATIONS"
)

create_interface_models()


# ============================================================
# Create terminal contacts
# ============================================================

print_section(
    "STEP 13 : CONTACT EQUATIONS"
)

create_contact_models()


# ============================================================
# Device information
# ============================================================

print_section(
    "STEP 14 : DEVICE INFORMATION"
)

print("Devices:")
print(
    get_device_list()
)

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
# Empty-trap zero-bias equilibrium
# ============================================================

print_section(
    "STEP 15 : EMPTY-TRAP ZERO-BIAS EQUILIBRIUM"
)

set_terminal_bias(
    terminal="source",
    voltage=tp.SOURCE_VOLTAGE,
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
    0.0
)

solve_dc(
    maximum_iterations=200,
)

print(
    "Empty-trap zero-bias equilibrium completed."
)


# ============================================================
# Equilibrium-result inspection
# ============================================================

print_section(
    "STEP 16 : EMPTY-TRAP EQUILIBRIUM RESULTS"
)

equilibrium_potential = (
    get_node_model_values(
        device=device,
        region="MoS2",
        name="Potential",
    )
)

equilibrium_electrons = (
    get_node_model_values(
        device=device,
        region="MoS2",
        name="Electrons",
    )
)

print_min_max(
    model_name="Potential",
    values=equilibrium_potential,
    unit="V",
)

print_min_max(
    model_name="Electrons",
    values=equilibrium_electrons,
    unit="cm^-3",
)


# ============================================================
# Ramp drain to read voltage
# ============================================================

print_section(
    "STEP 17 : DRAIN READ-BIAS RAMP"
)

current_drain_voltage = (
    adaptive_voltage_ramp(
        terminal="drain",
        start_voltage=0.0,
        target_voltage=tp.DRAIN_VOLTAGE,
        initial_step=tp.DRAIN_INITIAL_STEP,
        minimum_step=tp.DRAIN_MINIMUM_STEP,
        label="Drain ramp",
    )
)

print(
    f"Drain read bias reached "
    f"{current_drain_voltage:+.6f} V."
)


# ============================================================
# Prepare voltage points
# ============================================================

gate_output_voltages = (
    create_output_voltage_list(
        start=tp.GATE_START_VOLTAGE,
        stop=tp.GATE_STOP_VOLTAGE,
        step=tp.GATE_OUTPUT_STEP,
    )
)


# ============================================================
# Trap-state and I_D-V_G sweeps
# ============================================================

print_section(
    "STEP 18 : MULTI-STATE MEMORY-WINDOW SWEEP"
)

all_iv_results = []

threshold_summary = []

current_gate_voltage = 0.0

current_trap_density = 0.0

empty_state_threshold_voltage = None


for (
    state_index,
    (
        state_label,
        target_trap_density,
    ),
) in enumerate(
    zip(
        tp.TRAP_STATE_LABELS,
        tp.TRAPPED_ELECTRON_DENSITY_STATES,
    )
):

    print_section(
        f"TRAP STATE {state_index}: {state_label}"
    )

    # --------------------------------------------------------
    # Return gate to 0 V before changing trap density
    # --------------------------------------------------------

    current_gate_voltage = (
        adaptive_voltage_ramp(
            terminal="gate",
            start_voltage=current_gate_voltage,
            target_voltage=0.0,
            initial_step=tp.GATE_INITIAL_STEP,
            minimum_step=tp.GATE_MINIMUM_STEP,
            label=(
                f"{state_label} gate reset"
            ),
        )
    )

    # --------------------------------------------------------
    # Change HfO2 trapped-electron density
    # --------------------------------------------------------

    current_trap_density = (
        adaptive_trap_density_ramp(
            start_density=current_trap_density,
            target_density=target_trap_density,
            initial_step=TRAP_INITIAL_STEP,
            minimum_step=TRAP_MINIMUM_STEP,
            label=(
                f"{state_label} trap ramp"
            ),
        )
    )

    trap_information = (
        get_trap_state_information(
            current_trap_density
        )
    )

    # --------------------------------------------------------
    # Ramp gate to sweep starting voltage
    # --------------------------------------------------------

    current_gate_voltage = (
        adaptive_voltage_ramp(
            terminal="gate",
            start_voltage=current_gate_voltage,
            target_voltage=(
                tp.GATE_START_VOLTAGE
            ),
            initial_step=tp.GATE_INITIAL_STEP,
            minimum_step=tp.GATE_MINIMUM_STEP,
            label=(
                f"{state_label} gate pre-ramp"
            ),
        )
    )

    state_iv_results = []

    # --------------------------------------------------------
    # I_D-V_G sweep
    # --------------------------------------------------------

    for target_gate_voltage in (
        gate_output_voltages
    ):

        current_gate_voltage = (
            adaptive_voltage_ramp(
                terminal="gate",
                start_voltage=current_gate_voltage,
                target_voltage=target_gate_voltage,
                initial_step=tp.GATE_INITIAL_STEP,
                minimum_step=tp.GATE_MINIMUM_STEP,
                label=(
                    f"{state_label} "
                    f"VG target "
                    f"{target_gate_voltage:+.3f} V"
                ),
            )
        )

        drain_current = (
            get_drain_current()
        )

        absolute_drain_current = abs(
            drain_current
        )

        result_row = {
            "state_index": state_index,
            "state_label": state_label,
            "trap_density_cm3": (
                trap_information[
                    "trap_density_cm3"
                ]
            ),
            "trap_sheet_density_cm2": (
                trap_information[
                    "trap_sheet_density_cm2"
                ]
            ),
            "trap_charge_density_C_cm3": (
                trap_information[
                    "trap_charge_density_C_cm3"
                ]
            ),
            "gate_voltage_V": (
                current_gate_voltage
            ),
            "drain_voltage_V": (
                current_drain_voltage
            ),
            "drain_current_A": (
                drain_current
            ),
            "abs_drain_current_A": (
                absolute_drain_current
            ),
        }

        state_iv_results.append(
            result_row
        )

        all_iv_results.append(
            result_row
        )

        print(
            f"OUTPUT: "
            f"state = {state_label}, "
            f"Ntrap = "
            f"{current_trap_density:.3e} cm^-3, "
            f"VG = "
            f"{current_gate_voltage:+.3f} V, "
            f"ID = "
            f"{drain_current:+.6e} A"
        )

    # --------------------------------------------------------
    # Extract constant-current threshold voltage
    # --------------------------------------------------------

    threshold_voltage = (
        extract_threshold_voltage(
            state_results=state_iv_results,
            threshold_current=(
                tp.THRESHOLD_CURRENT
            ),
        )
    )

    if state_index == 0:

        empty_state_threshold_voltage = (
            threshold_voltage
        )

    if (
        threshold_voltage is not None
        and empty_state_threshold_voltage
        is not None
    ):

        threshold_shift = (
            threshold_voltage
            - empty_state_threshold_voltage
        )

    else:

        threshold_shift = None

    threshold_summary.append(
        {
            "state_index": state_index,
            "state_label": state_label,
            "trap_density_cm3": (
                trap_information[
                    "trap_density_cm3"
                ]
            ),
            "trap_sheet_density_cm2": (
                trap_information[
                    "trap_sheet_density_cm2"
                ]
            ),
            "threshold_current_A": (
                tp.THRESHOLD_CURRENT
            ),
            "threshold_voltage_V": (
                threshold_voltage
            ),
            "threshold_shift_from_empty_V": (
                threshold_shift
            ),
        }
    )

    if threshold_voltage is None:

        print(
            f"{state_label}: threshold current "
            f"{tp.THRESHOLD_CURRENT:.3e} A "
            f"was not crossed within the "
            f"gate sweep range."
        )

    else:

        print(
            f"{state_label}: "
            f"Vth = "
            f"{threshold_voltage:.6f} V"
        )

        print(
            f"{state_label}: "
            f"Delta Vth from empty state = "
            f"{threshold_shift:.6f} V"
        )


# ============================================================
# Save full I_D-V_G data
# ============================================================

print_section(
    "STEP 19 : SAVE MEMORY-WINDOW I_D-V_G DATA"
)

with MEMORY_WINDOW_CSV.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = (
        "state_index",
        "state_label",
        "trap_density_cm3",
        "trap_sheet_density_cm2",
        "trap_charge_density_C_cm3",
        "gate_voltage_V",
        "drain_voltage_V",
        "drain_current_A",
        "abs_drain_current_A",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        all_iv_results
    )

print(
    f'Full memory-window data saved to '
    f'"{MEMORY_WINDOW_CSV}".'
)


# ============================================================
# Save threshold-voltage summary
# ============================================================

print_section(
    "STEP 20 : SAVE THRESHOLD-VOLTAGE SUMMARY"
)

with THRESHOLD_SUMMARY_CSV.open(
    mode="w",
    newline="",
    encoding="utf-8",
) as csv_file:

    fieldnames = (
        "state_index",
        "state_label",
        "trap_density_cm3",
        "trap_sheet_density_cm2",
        "threshold_current_A",
        "threshold_voltage_V",
        "threshold_shift_from_empty_V",
    )

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        threshold_summary
    )

print(
    f'Threshold-voltage summary saved to '
    f'"{THRESHOLD_SUMMARY_CSV}".'
)


# ============================================================
# Print threshold summary
# ============================================================

print_section(
    "STEP 21 : THRESHOLD-VOLTAGE RESULTS"
)

for summary_row in threshold_summary:

    state_label = (
        summary_row["state_label"]
    )

    trap_density = (
        summary_row[
            "trap_density_cm3"
        ]
    )

    threshold_voltage = (
        summary_row[
            "threshold_voltage_V"
        ]
    )

    threshold_shift = (
        summary_row[
            "threshold_shift_from_empty_V"
        ]
    )

    if threshold_voltage is None:

        print(
            f"{state_label:20s} "
            f"Ntrap = {trap_density:.3e} cm^-3, "
            f"Vth = not extracted"
        )

    else:

        print(
            f"{state_label:20s} "
            f"Ntrap = {trap_density:.3e} cm^-3, "
            f"Vth = {threshold_voltage:.6f} V, "
            f"Delta Vth = "
            f"{threshold_shift:.6f} V"
        )


# ============================================================
# Final solution inspection
# ============================================================

print_section(
    "STEP 22 : FINAL SOLUTION RESULTS"
)

final_potential = (
    get_node_model_values(
        device=device,
        region="MoS2",
        name="Potential",
    )
)

final_electrons = (
    get_node_model_values(
        device=device,
        region="MoS2",
        name="Electrons",
    )
)

final_trapped_electrons = (
    get_node_model_values(
        device=device,
        region=tp.CHARGE_TRAP_REGION,
        name="TrappedElectronDensity",
    )
)

print_min_max(
    model_name="MoS2 Potential",
    values=final_potential,
    unit="V",
)

print_min_max(
    model_name="MoS2 Electrons",
    values=final_electrons,
    unit="cm^-3",
)

print_min_max(
    model_name="HfO2 trapped e-",
    values=final_trapped_electrons,
    unit="cm^-3",
)


# ============================================================
# Export final device state
# ============================================================

print_section(
    "STEP 23 : EXPORT FINAL DEVICE STATE"
)

write_devices(
    file=tp.FINAL_VTK_NAME,
    type="vtk",
)

print(
    f'Final device state written to '
    f'"{tp.FINAL_VTK_NAME}".'
)


# ============================================================
# Final status
# ============================================================

print()
print("=" * 70)
print(
    "STATIC CHARGE-TRAP MEMORY-WINDOW "
    "SIMULATION SUCCESSFUL"
)
print("=" * 70)


from field_extraction import (
    print_dielectric_field_statistics,
)


# ============================================================
# Dielectric electric-field inspection
# ============================================================

print_section(
    "STEP 24 : DIELECTRIC ELECTRIC-FIELD ANALYSIS"
)

dielectric_field_results = (
    print_dielectric_field_statistics(
        device=device,
    )
)

print()
print(
    "Dielectric electric-field extraction completed."
)