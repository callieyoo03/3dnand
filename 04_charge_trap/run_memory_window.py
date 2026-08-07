# ============================================================
# MoS2 / Al2O3 / HfO2 / Al2O3 cylindrical GAA memory
# Static trapped-charge memory-window simulation
#
# Main functions
# --------------
# 1. Build the electron-only drift-diffusion device
# 2. Apply multiple fixed trapped-electron-density states
# 3. Sweep read gate voltage for every trap state
# 4. Save all ID-VG curves
# 5. Extract threshold voltage by constant-current criterion
# 6. Calculate threshold-voltage shift from the empty state
#
# Environment-variable inputs
# ---------------------------
# TUNNEL_OXIDE_THICKNESS_NM
#
# TRAPPED_ELECTRON_DENSITY_STATES_CM3
#   Example:
#   0,2.0e17,5.0e17,1.0e18,2.0e18
#
# TRAP_STATE_LABELS
#   Example:
#   Empty,State_1,State_2,State_3,State_4
#
# READ_SOURCE_VOLTAGE_V
# READ_DRAIN_VOLTAGE_V
# READ_GATE_START_V
# READ_GATE_STOP_V
# READ_GATE_STEP_V
# THRESHOLD_CURRENT_A
#
# MEMORY_WINDOW_OUTPUT_CSV
# THRESHOLD_SUMMARY_OUTPUT_CSV
#
# Important
# ---------
# This is a static charge-state simulation.
#
# It does not directly calculate:
#   - tunneling current
#   - capture or emission rate
#   - program or erase time
#   - retention
#   - endurance
#
# Program-time simulations can calculate trap-density states
# first and pass them into this script through environment
# variables.
# ============================================================

import csv
import math
import os
from pathlib import Path

from devsim import (
    get_contact_current,
    get_contact_list,
    get_device_list,
    get_interface_list,
    get_node_model_values,
    get_parameter,
    get_region_list,
    set_node_values,
    set_parameter,
    solve,
    write_devices,
)

# ============================================================
# Parameterized device-structure module loading
#
# The original 04_charge_trap/device_structure.py uses a fixed
# geometry and defines:
#
#     create_structure()
#
# The parameterized structure implementation is maintained in:
#
#     05_program_erase/device_structure.py
#
# It supports:
#
#     create_structure(tunnel_oxide_thickness_nm=...)
#
# This explicit import avoids creating another duplicate
# device_structure.py file.
# ============================================================

import importlib.util
import sys


PROJECT_ROOT_DIRECTORY = (
    Path(__file__).resolve().parent.parent
)

PROJECT_ROOT_TEXT = str(PROJECT_ROOT_DIRECTORY)

if PROJECT_ROOT_TEXT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_TEXT)

import compact_handoff_parameters as compact_parameters

if Path(compact_parameters.__file__).resolve() != (
    PROJECT_ROOT_DIRECTORY / "compact_handoff_parameters.py"
).resolve():
    raise ImportError("compact_handoff_parameters resolved outside this repository.")

PARAMETERIZED_DEVICE_STRUCTURE_PATH = (
    PROJECT_ROOT_DIRECTORY
    / "05_program_erase"
    / "device_structure.py"
)


def load_parameterized_device_structure_module():
    if not PARAMETERIZED_DEVICE_STRUCTURE_PATH.exists():
        raise FileNotFoundError(
            "Parameterized device_structure.py was not found:\n"
            f"{PARAMETERIZED_DEVICE_STRUCTURE_PATH.resolve()}"
        )

    module_name = (
        "program_erase_parameterized_device_structure"
    )

    existing_module = sys.modules.get(
        module_name
    )

    if existing_module is not None:
        return existing_module

    module_specification = (
        importlib.util.spec_from_file_location(
            module_name,
            PARAMETERIZED_DEVICE_STRUCTURE_PATH,
        )
    )

    if (
        module_specification is None
        or module_specification.loader is None
    ):
        raise ImportError(
            "Could not create an import specification for:\n"
            f"{PARAMETERIZED_DEVICE_STRUCTURE_PATH.resolve()}"
        )

    module = (
        importlib.util.module_from_spec(
            module_specification
        )
    )

    sys.modules[
        module_name
    ] = module

    module_specification.loader.exec_module(
        module
    )

    required_attributes = (
        "create_structure",
        "device",
    )

    for attribute_name in required_attributes:
        if not hasattr(
            module,
            attribute_name,
        ):
            raise ImportError(
                "The parameterized device-structure module "
                f'does not define "{attribute_name}".'
            )

    return module


parameterized_device_structure = (
    load_parameterized_device_structure_module()
)

from contact_topology_structure import (
    bind_characterization_create_structure,
)

create_structure = (
    bind_characterization_create_structure(
        parameterized_device_structure
    )
)

device = (
    parameterized_device_structure.device
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

import trap_parameters as tp
import material_parameters as mp


# ============================================================
# General constants
# ============================================================

CURRENT_FLOOR_A = 1.0e-30

DEFAULT_TUNNEL_OXIDE_THICKNESS_NM = (
    compact_parameters.TUNNEL_OXIDE_THICKNESS_NM
)

DEFAULT_TRAP_INITIAL_STEP_CM3 = 2.0e17
DEFAULT_TRAP_MINIMUM_STEP_CM3 = 1.0e15


# ============================================================
# Script path
# ============================================================

SCRIPT_DIRECTORY = Path(
    __file__
).resolve().parent


# ============================================================
# Input parsing utilities
# ============================================================

def parse_float(
    variable_name,
    default_value,
):
    value_text = os.environ.get(
        variable_name,
        str(default_value),
    ).strip()

    try:
        value = float(
            value_text
        )

    except ValueError as error:
        raise ValueError(
            f"{variable_name} must be a number. "
            f'Received: "{value_text}"'
        ) from error

    if not math.isfinite(
        value
    ):
        raise ValueError(
            f"{variable_name} must be finite. "
            f"Received: {value}"
        )

    return value


def parse_positive_float(
    variable_name,
    default_value,
):
    value = parse_float(
        variable_name=variable_name,
        default_value=default_value,
    )

    if value <= 0.0:
        raise ValueError(
            f"{variable_name} must be greater than zero. "
            f"Received: {value}"
        )

    return value


def parse_nonnegative_float(
    variable_name,
    default_value,
):
    value = parse_float(
        variable_name=variable_name,
        default_value=default_value,
    )

    if value < 0.0:
        raise ValueError(
            f"{variable_name} must not be negative. "
            f"Received: {value}"
        )

    return value


def parse_float_list(
    variable_name,
    default_values,
    nonnegative=False,
    strictly_increasing=False,
):
    default_text = ",".join(
        str(value)
        for value in default_values
    )

    value_text = os.environ.get(
        variable_name,
        default_text,
    )

    values = []

    for token in value_text.split(","):
        stripped_token = token.strip()

        if not stripped_token:
            continue

        try:
            value = float(
                stripped_token
            )

        except ValueError as error:
            raise ValueError(
                f"{variable_name} contains a nonnumeric value: "
                f'"{stripped_token}"'
            ) from error

        if not math.isfinite(
            value
        ):
            raise ValueError(
                f"{variable_name} values must be finite. "
                f"Received: {value}"
            )

        if (
            nonnegative
            and value < 0.0
        ):
            raise ValueError(
                f"{variable_name} values must not be negative. "
                f"Received: {value}"
            )

        values.append(
            value
        )

    if not values:
        raise ValueError(
            f"{variable_name} must contain at least one value."
        )

    if strictly_increasing:
        for previous_value, current_value in zip(
            values,
            values[1:],
        ):
            if current_value <= previous_value:
                raise ValueError(
                    f"{variable_name} must be strictly increasing. "
                    f"Received: {values}"
                )

    return tuple(
        values
    )


def parse_label_list(
    variable_name,
    default_labels,
):
    default_text = ",".join(
        str(label)
        for label in default_labels
    )

    value_text = os.environ.get(
        variable_name,
        default_text,
    )

    labels = [
        token.strip()
        for token in value_text.split(",")
        if token.strip()
    ]

    if not labels:
        raise ValueError(
            f"{variable_name} must contain at least one label."
        )

    return tuple(
        labels
    )


def parse_output_path(
    variable_name,
    default_filename,
):
    default_path = (
        SCRIPT_DIRECTORY
        / default_filename
    )

    output_text = os.environ.get(
        variable_name,
        str(default_path),
    ).strip()

    if not output_text:
        raise ValueError(
            f"{variable_name} must not be empty."
        )

    return Path(
        output_text
    ).expanduser()


# ============================================================
# Simulation inputs
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = (
    parse_positive_float(
        variable_name=(
            "TUNNEL_OXIDE_THICKNESS_NM"
        ),
        default_value=(
            DEFAULT_TUNNEL_OXIDE_THICKNESS_NM
        ),
    )
)

TRAPPED_ELECTRON_DENSITY_STATES_CM3 = (
    parse_float_list(
        variable_name=(
            "TRAPPED_ELECTRON_DENSITY_STATES_CM3"
        ),
        default_values=(
            tp.TRAPPED_ELECTRON_DENSITY_STATES
        ),
        nonnegative=True,
        strictly_increasing=True,
    )
)

TRAP_STATE_LABELS = parse_label_list(
    variable_name="TRAP_STATE_LABELS",
    default_labels=tp.TRAP_STATE_LABELS,
)

SOURCE_VOLTAGE_V = (
    parse_float(
        variable_name=(
            "READ_SOURCE_VOLTAGE_V"
        ),
        default_value=(
            tp.SOURCE_VOLTAGE
        ),
    )
)

DRAIN_VOLTAGE_V = (
    parse_nonnegative_float(
        variable_name=(
            "READ_DRAIN_VOLTAGE_V"
        ),
        default_value=(
            tp.DRAIN_VOLTAGE
        ),
    )
)

GATE_START_VOLTAGE_V = (
    parse_float(
        variable_name=(
            "READ_GATE_START_V"
        ),
        default_value=(
            tp.GATE_START_VOLTAGE
        ),
    )
)

GATE_STOP_VOLTAGE_V = (
    parse_float(
        variable_name=(
            "READ_GATE_STOP_V"
        ),
        default_value=(
            tp.GATE_STOP_VOLTAGE
        ),
    )
)

GATE_OUTPUT_STEP_V = (
    parse_positive_float(
        variable_name=(
            "READ_GATE_STEP_V"
        ),
        default_value=(
            tp.GATE_OUTPUT_STEP
        ),
    )
)

THRESHOLD_CURRENT_A = (
    parse_positive_float(
        variable_name=(
            "THRESHOLD_CURRENT_A"
        ),
        default_value=(
            tp.THRESHOLD_CURRENT
        ),
    )
)

DRAIN_INITIAL_STEP_V = (
    parse_positive_float(
        variable_name=(
            "DRAIN_INITIAL_STEP_V"
        ),
        default_value=(
            tp.DRAIN_INITIAL_STEP
        ),
    )
)

DRAIN_MINIMUM_STEP_V = (
    parse_positive_float(
        variable_name=(
            "DRAIN_MINIMUM_STEP_V"
        ),
        default_value=(
            tp.DRAIN_MINIMUM_STEP
        ),
    )
)

GATE_INITIAL_STEP_V = (
    parse_positive_float(
        variable_name=(
            "GATE_INITIAL_STEP_V"
        ),
        default_value=(
            tp.GATE_INITIAL_STEP
        ),
    )
)

GATE_MINIMUM_STEP_V = (
    parse_positive_float(
        variable_name=(
            "GATE_MINIMUM_STEP_V"
        ),
        default_value=(
            tp.GATE_MINIMUM_STEP
        ),
    )
)

TRAP_INITIAL_STEP_CM3 = (
    parse_positive_float(
        variable_name=(
            "TRAP_INITIAL_STEP_CM3"
        ),
        default_value=(
            DEFAULT_TRAP_INITIAL_STEP_CM3
        ),
    )
)

TRAP_MINIMUM_STEP_CM3 = (
    parse_positive_float(
        variable_name=(
            "TRAP_MINIMUM_STEP_CM3"
        ),
        default_value=(
            DEFAULT_TRAP_MINIMUM_STEP_CM3
        ),
    )
)

MEMORY_WINDOW_CSV = parse_output_path(
    variable_name=(
        "MEMORY_WINDOW_OUTPUT_CSV"
    ),
    default_filename=(
        tp.MEMORY_WINDOW_CSV
    ),
)

THRESHOLD_SUMMARY_CSV = (
    parse_output_path(
        variable_name=(
            "THRESHOLD_SUMMARY_OUTPUT_CSV"
        ),
        default_filename=(
            tp.THRESHOLD_SUMMARY_CSV
        ),
    )
)

FINAL_VTK_OUTPUT = os.environ.get(
    "MEMORY_WINDOW_FINAL_VTK",
    "",
).strip()


# ============================================================
# Utility functions
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 76)
    print(title)
    print("=" * 76)


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


def format_optional_float(
    value,
):
    if value is None:
        return "not_found"

    return f"{float(value):.12e}"


# ============================================================
# Input validation
# ============================================================

def validate_inputs():
    if (
        len(
            TRAPPED_ELECTRON_DENSITY_STATES_CM3
        )
        != len(
            TRAP_STATE_LABELS
        )
    ):
        raise ValueError(
            "The number of trap-density states must match "
            "the number of trap-state labels.\n"
            "Trap-density count: "
            f"{len(TRAPPED_ELECTRON_DENSITY_STATES_CM3)}\n"
            "Label count: "
            f"{len(TRAP_STATE_LABELS)}"
        )

    if (
        GATE_STOP_VOLTAGE_V
        <= GATE_START_VOLTAGE_V
    ):
        raise ValueError(
            "READ_GATE_STOP_V must be greater than "
            "READ_GATE_START_V."
        )

    if (
        TRAP_MINIMUM_STEP_CM3
        > TRAP_INITIAL_STEP_CM3
    ):
        raise ValueError(
            "TRAP_MINIMUM_STEP_CM3 must not exceed "
            "TRAP_INITIAL_STEP_CM3."
        )

    if (
        GATE_MINIMUM_STEP_V
        > GATE_INITIAL_STEP_V
    ):
        raise ValueError(
            "GATE_MINIMUM_STEP_V must not exceed "
            "GATE_INITIAL_STEP_V."
        )

    if (
        DRAIN_MINIMUM_STEP_V
        > DRAIN_INITIAL_STEP_V
    ):
        raise ValueError(
            "DRAIN_MINIMUM_STEP_V must not exceed "
            "DRAIN_INITIAL_STEP_V."
        )

    if (
        TRAPPED_ELECTRON_DENSITY_STATES_CM3[0]
        != 0.0
    ):
        raise ValueError(
            "The first trap-density state must be 0.0 cm^-3 "
            "so that threshold-voltage shifts can be referenced "
            "to the empty-trap state."
        )


# ============================================================
# DEVSIM utilities
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


def get_drain_current_A():
    return float(
        get_contact_current(
            device=device,
            contact="drain",
            equation=(
                "ElectronContinuityEquation"
            ),
        )
    )


def verify_device_structure():
    existing_devices = (
        get_device_list()
    )

    if device not in existing_devices:
        raise RuntimeError(
            f'Device "{device}" was not created.'
        )

    existing_regions = (
        get_region_list(
            device=device,
        )
    )

    for region in REGIONS:
        if region not in existing_regions:
            raise RuntimeError(
                f'Region "{region}" is missing.'
            )

    print(
        "Device structure verification completed."
    )


# ============================================================
# Voltage-list generation
# ============================================================

def create_output_voltage_list(
    start_voltage,
    stop_voltage,
    step_voltage,
):
    if step_voltage <= 0.0:
        raise ValueError(
            "Gate output step must be positive."
        )

    voltage_span = (
        stop_voltage
        - start_voltage
    )

    if voltage_span <= 0.0:
        raise ValueError(
            "Gate voltage span must be positive."
        )

    number_of_full_steps = int(
        math.floor(
            voltage_span
            / step_voltage
            + 1.0e-12
        )
    )

    voltages = [
        round(
            start_voltage
            + index * step_voltage,
            12,
        )
        for index in range(
            number_of_full_steps + 1
        )
    ]

    if (
        not voltages
        or abs(
            voltages[-1]
            - stop_voltage
        )
        > 1.0e-10
    ):
        voltages.append(
            round(
                stop_voltage,
                12,
            )
        )

    for previous_voltage, current_voltage in zip(
        voltages,
        voltages[1:],
    ):
        if current_voltage <= previous_voltage:
            raise RuntimeError(
                "Generated gate-voltage values are not "
                "strictly increasing."
            )

    return tuple(
        voltages
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
                    f"{label}: could not recover the solution "
                    f"at {current_voltage:+.6f} V."
                ) from recovery_error

            step *= 0.5

            print(
                f"{label}: step reduced to "
                f"{step:.6e} V."
            )

            if step < minimum_step:
                raise RuntimeError(
                    f"{label}: voltage step became smaller "
                    f"than the minimum step "
                    f"{minimum_step:.6e} V."
                ) from solve_error

            continue

        current_voltage = (
            trial_voltage
        )

        drain_current_A = (
            get_drain_current_A()
        )

        print(
            f"{label}: converged at "
            f"{current_voltage:+.6f} V, "
            f"ID = {drain_current_A:+.6e} A"
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
    start_density_cm3,
    target_density_cm3,
    initial_step_cm3,
    minimum_step_cm3,
    label,
):
    current_density_cm3 = float(
        start_density_cm3
    )

    target_density_cm3 = float(
        target_density_cm3
    )

    if abs(
        target_density_cm3
        - current_density_cm3
    ) <= 1.0:
        return current_density_cm3

    direction = (
        1.0
        if (
            target_density_cm3
            > current_density_cm3
        )
        else -1.0
    )

    step_cm3 = abs(
        float(
            initial_step_cm3
        )
    )

    maximum_step_cm3 = (
        step_cm3
    )

    while (
        direction
        * (
            target_density_cm3
            - current_density_cm3
        )
        > 1.0
    ):
        remaining_density_cm3 = abs(
            target_density_cm3
            - current_density_cm3
        )

        trial_step_cm3 = min(
            step_cm3,
            remaining_density_cm3,
        )

        trial_density_cm3 = (
            current_density_cm3
            + direction * trial_step_cm3
        )

        print(
            f"{label}: attempting "
            f"Ntrap = {trial_density_cm3:.6e} cm^-3 "
            f"(step = {trial_step_cm3:.6e} cm^-3)"
        )

        set_trapped_electron_density(
            trial_density_cm3
        )

        try:
            solve_dc()

        except Exception as solve_error:
            print(
                f"{label}: convergence failed at "
                f"Ntrap = {trial_density_cm3:.6e} cm^-3."
            )

            set_trapped_electron_density(
                current_density_cm3
            )

            try:
                solve_dc(
                    maximum_iterations=200,
                )

            except Exception as recovery_error:
                raise RuntimeError(
                    f"{label}: could not recover "
                    f"Ntrap = "
                    f"{current_density_cm3:.6e} cm^-3."
                ) from recovery_error

            step_cm3 *= 0.5

            print(
                f"{label}: trap-density step reduced to "
                f"{step_cm3:.6e} cm^-3."
            )

            if (
                step_cm3
                < minimum_step_cm3
            ):
                raise RuntimeError(
                    f"{label}: trap-density step became "
                    f"smaller than the minimum step "
                    f"{minimum_step_cm3:.6e} cm^-3."
                ) from solve_error

            continue

        current_density_cm3 = (
            trial_density_cm3
        )

        print(
            f"{label}: converged at "
            f"Ntrap = "
            f"{current_density_cm3:.6e} cm^-3."
        )

        step_cm3 = min(
            maximum_step_cm3,
            step_cm3 * 1.5,
        )

    return current_density_cm3


# ============================================================
# Threshold-voltage extraction
# ============================================================

def extract_threshold_voltage(
    state_results,
    threshold_current_A,
):
    if len(
        state_results
    ) < 2:
        return None

    if threshold_current_A <= 0.0:
        raise ValueError(
            "Threshold current must be positive."
        )

    threshold_log_current = (
        math.log10(
            max(
                threshold_current_A,
                CURRENT_FLOOR_A,
            )
        )
    )

    for point_1, point_2 in zip(
        state_results,
        state_results[1:],
    ):
        gate_voltage_1_V = float(
            point_1[
                "gate_voltage_V"
            ]
        )

        gate_voltage_2_V = float(
            point_2[
                "gate_voltage_V"
            ]
        )

        current_1_A = max(
            float(
                point_1[
                    "abs_drain_current_A"
                ]
            ),
            CURRENT_FLOOR_A,
        )

        current_2_A = max(
            float(
                point_2[
                    "abs_drain_current_A"
                ]
            ),
            CURRENT_FLOOR_A,
        )

        threshold_crossed = (
            current_1_A
            <= threshold_current_A
            <= current_2_A
        ) or (
            current_2_A
            <= threshold_current_A
            <= current_1_A
        )

        if not threshold_crossed:
            continue

        log_current_1 = (
            math.log10(
                current_1_A
            )
        )

        log_current_2 = (
            math.log10(
                current_2_A
            )
        )

        denominator = (
            log_current_2
            - log_current_1
        )

        if abs(
            denominator
        ) < 1.0e-30:
            return 0.5 * (
                gate_voltage_1_V
                + gate_voltage_2_V
            )

        interpolation_fraction = (
            threshold_log_current
            - log_current_1
        ) / denominator

        threshold_voltage_V = (
            gate_voltage_1_V
            + interpolation_fraction
            * (
                gate_voltage_2_V
                - gate_voltage_1_V
            )
        )

        return (
            threshold_voltage_V
        )

    return None


# ============================================================
# Device initialization
# ============================================================

def expected_runtime_geometry_nm():
    """Return the geometry values that this state runner must create."""

    geometry = compact_parameters.geometry_dict()
    geometry["tunnel_oxide_thickness_nm"] = float(
        TUNNEL_OXIDE_THICKNESS_NM
    )

    core_outer_radius_nm = geometry["core_radius_nm"]
    mos2_outer_radius_nm = (
        core_outer_radius_nm + geometry["mos2_thickness_nm"]
    )
    tunnel_outer_radius_nm = (
        mos2_outer_radius_nm + geometry["tunnel_oxide_thickness_nm"]
    )
    trap_outer_radius_nm = (
        tunnel_outer_radius_nm + geometry["charge_trap_thickness_nm"]
    )
    blocking_outer_radius_nm = (
        trap_outer_radius_nm + geometry["blocking_oxide_thickness_nm"]
    )
    gate_outer_radius_nm = (
        blocking_outer_radius_nm + geometry["gate_metal_thickness_nm"]
    )
    air_outer_radius_nm = (
        gate_outer_radius_nm + geometry["air_thickness_nm"]
    )

    geometry.update(
        {
            "core_outer_radius_nm": core_outer_radius_nm,
            "mos2_outer_radius_nm": mos2_outer_radius_nm,
            "tunnel_oxide_outer_radius_nm": tunnel_outer_radius_nm,
            "charge_trap_outer_radius_nm": trap_outer_radius_nm,
            "blocking_oxide_outer_radius_nm": blocking_outer_radius_nm,
            "gate_outer_radius_nm": gate_outer_radius_nm,
            "air_outer_radius_nm": air_outer_radius_nm,
        }
    )
    return geometry


def verify_runtime_geometry(geometry):
    """Verify every supplied layer dimension and accumulated radius."""

    expected_geometry = expected_runtime_geometry_nm()

    for geometry_key, expected_value in expected_geometry.items():
        if geometry_key not in geometry:
            raise RuntimeError(
                "Geometry output is missing the key "
                f'"{geometry_key}".'
            )

        actual_value = float(geometry[geometry_key])
        if not math.isclose(
            actual_value,
            float(expected_value),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise RuntimeError(
                f"Runtime geometry mismatch for {geometry_key}.\n"
                f"Expected: {float(expected_value):.12e} nm\n"
                f"Created:  {actual_value:.12e} nm"
            )

    return expected_geometry


def verify_runtime_material_parameters():
    """Verify the DEVSIM permittivities and print their relative values."""

    expected_absolute_permittivity = {
        "CoreOxide": mp.eps_core_oxide,
        "MoS2": mp.eps_mos2,
        "TunnelOxide": mp.eps_tunnel_oxide,
        "ChargeTrap": mp.eps_charge_trap,
        "BlockingOxide": mp.eps_blocking_oxide,
    }
    expected_relative_permittivity = (
        compact_parameters.material_dict()
    )
    actual_relative_permittivity = {}

    for region, expected_absolute in expected_absolute_permittivity.items():
        actual_absolute = float(
            get_parameter(
                device=device,
                region=region,
                name="Permittivity",
            )
        )
        if not math.isclose(
            actual_absolute,
            float(expected_absolute),
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ):
            raise RuntimeError(
                f"Runtime permittivity mismatch in {region}.\n"
                f"Expected: {float(expected_absolute):.12e} F/cm\n"
                f"Assigned: {actual_absolute:.12e} F/cm"
            )

        actual_relative = actual_absolute / mp.eps0
        expected_relative = float(expected_relative_permittivity[region])
        if not math.isclose(
            actual_relative,
            expected_relative,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"Relative-permittivity mismatch in {region}: "
                f"{actual_relative:.12e} != {expected_relative:.12e}."
            )
        actual_relative_permittivity[region] = actual_relative

    print("Runtime material verification:")
    print(
        "  Material version       = "
        f"{compact_parameters.MATERIAL_VERSION}"
    )
    print(
        "  Al2O3 relative k       = "
        f"{actual_relative_permittivity['TunnelOxide']:.6f}"
    )
    print(
        "  HfO2 relative k        = "
        f"{actual_relative_permittivity['ChargeTrap']:.6f}"
    )

    return actual_relative_permittivity

def initialize_device():
    print_section(
        "STEP 1 : VALIDATE INPUTS"
    )

    validate_inputs()

    print(
        "Input parameters validated."
    )

    print_section(
        "STEP 2 : CREATE DEVICE STRUCTURE"
    )

    geometry = create_structure(
        core_radius_nm=compact_parameters.CORE_RADIUS_NM,
        mos2_thickness_nm=compact_parameters.MOS2_THICKNESS_NM,
        tunnel_oxide_thickness_nm=TUNNEL_OXIDE_THICKNESS_NM,
        charge_trap_thickness_nm=(
            compact_parameters.CHARGE_TRAP_THICKNESS_NM
        ),
        blocking_oxide_thickness_nm=(
            compact_parameters.BLOCKING_OXIDE_THICKNESS_NM
        ),
        gate_metal_thickness_nm=(
            compact_parameters.GATE_METAL_THICKNESS_NM
        ),
        air_thickness_nm=compact_parameters.AIR_THICKNESS_NM,
        channel_length_nm=compact_parameters.CHANNEL_LENGTH_NM,
    )

    if geometry is None:
        raise RuntimeError(
            "Parameterized create_structure() returned None. "
            "A geometry dictionary was expected."
        )

    verify_runtime_geometry(geometry)

    verify_device_structure()

    print(
        "Created geometry:"
    )

    print(
        "  Geometry version       = "
        f"{compact_parameters.GEOMETRY_VERSION}"
    )

    print(
        "  Core radius            = "
        f"{geometry['core_radius_nm']:.6f} nm"
    )

    print(
        "  MoS2 thickness         = "
        f"{geometry['mos2_thickness_nm']:.6f} nm"
    )

    print(
        "  TunnelOxide thickness = "
        f"{geometry['tunnel_oxide_thickness_nm']:.6f} nm"
    )

    print(
        "  ChargeTrap thickness  = "
        f"{geometry['charge_trap_thickness_nm']:.6f} nm"
    )

    print(
        "  BlockingOxide thickness = "
        f"{geometry['blocking_oxide_thickness_nm']:.6f} nm"
    )

    print(
        "  Gate metal thickness  = "
        f"{geometry['gate_metal_thickness_nm']:.6f} nm"
    )

    print(
        "  Air thickness         = "
        f"{geometry['air_thickness_nm']:.6f} nm"
    )

    print(
        "  Channel length        = "
        f"{geometry['channel_length_nm']:.6f} nm"
    )

    print(
        "  MoS2 outer radius      = "
        f"{geometry['mos2_outer_radius_nm']:.6f} nm"
    )

    print(
        "  TunnelOxide outer      = "
        f"{geometry['tunnel_oxide_outer_radius_nm']:.6f} nm"
    )

    print(
        "  ChargeTrap outer       = "
        f"{geometry['charge_trap_outer_radius_nm']:.6f} nm"
    )

    print(
        "  BlockingOxide outer    = "
        f"{geometry['blocking_oxide_outer_radius_nm']:.6f} nm"
    )

    print(
        "  Gate outer radius      = "
        f"{geometry['gate_outer_radius_nm']:.6f} nm"
    )

    print_section(
        "STEP 3 : MATERIAL PARAMETERS"
    )

    set_material_parameters()
    verify_runtime_material_parameters()

    print_section(
        "STEP 4 : DOPING"
    )

    create_doping()

    print_section(
        "STEP 5 : SOLUTION VARIABLES"
    )

    create_solution_variables()

    print_section(
        "STEP 6 : EQUILIBRIUM CARRIER MODELS"
    )

    create_equilibrium_carrier_models()

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

    print_section(
        "STEP 8 : POISSON EQUATION"
    )

    create_poisson_model()

    print_section(
        "STEP 9 : STATIC TRAP FRAMEWORK"
    )

    create_static_trap_framework()

    print_section(
        "STEP 10 : ELECTRON CURRENT MODEL"
    )

    create_electron_current_model()

    print_section(
        "STEP 11 : ELECTRON CONTINUITY EQUATION"
    )

    create_continuity_equation()

    print_section(
        "STEP 12 : INTERFACE EQUATIONS"
    )

    create_interface_models()

    print_section(
        "STEP 13 : CONTACT EQUATIONS"
    )

    create_contact_models()

    return geometry

# ============================================================
# Device-information output
# ============================================================

def print_device_information():
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
# Empty-state initialization
# ============================================================

def solve_empty_state():
    print_section(
        "STEP 15 : EMPTY-TRAP ZERO-BIAS EQUILIBRIUM"
    )

    set_terminal_bias(
        terminal="source",
        voltage=SOURCE_VOLTAGE_V,
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
# Memory-window sweep
# ============================================================

def run_memory_window_sweep(
    geometry,
):
    print_section(
        "STEP 17 : DRAIN READ-BIAS RAMP"
    )

    current_drain_voltage_V = (
        adaptive_voltage_ramp(
            terminal="drain",
            start_voltage=0.0,
            target_voltage=DRAIN_VOLTAGE_V,
            initial_step=DRAIN_INITIAL_STEP_V,
            minimum_step=DRAIN_MINIMUM_STEP_V,
            label="Drain ramp",
        )
    )

    print(
        f"Drain read bias reached "
        f"{current_drain_voltage_V:+.6f} V."
    )

    gate_output_voltages_V = (
        create_output_voltage_list(
            start_voltage=(
                GATE_START_VOLTAGE_V
            ),
            stop_voltage=(
                GATE_STOP_VOLTAGE_V
            ),
            step_voltage=(
                GATE_OUTPUT_STEP_V
            ),
        )
    )

    print_section(
        "STEP 18 : MULTI-STATE MEMORY-WINDOW SWEEP"
    )

    print(
        "TunnelOxide thickness: "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.6f} nm"
    )

    print(
        "Trap-density states:"
    )

    for state_index, (
        state_label,
        trap_density_cm3,
    ) in enumerate(
        zip(
            TRAP_STATE_LABELS,
            TRAPPED_ELECTRON_DENSITY_STATES_CM3,
        )
    ):
        print(
            f"  {state_index:2d}: "
            f"{state_label}, "
            f"Ntrap = {trap_density_cm3:.6e} cm^-3"
        )

    print(
        "Gate sweep: "
        f"{GATE_START_VOLTAGE_V:+.3f} "
        f"to {GATE_STOP_VOLTAGE_V:+.3f} V, "
        f"step = {GATE_OUTPUT_STEP_V:.3f} V"
    )

    print(
        "Threshold current: "
        f"{THRESHOLD_CURRENT_A:.6e} A"
    )

    all_iv_results = []
    threshold_summary = []

    current_gate_voltage_V = 0.0
    current_trap_density_cm3 = 0.0

    empty_state_threshold_voltage_V = (
        None
    )

    for state_index, (
        state_label,
        target_trap_density_cm3,
    ) in enumerate(
        zip(
            TRAP_STATE_LABELS,
            TRAPPED_ELECTRON_DENSITY_STATES_CM3,
        )
    ):
        print_section(
            f"TRAP STATE {state_index}: "
            f"{state_label}"
        )

        current_gate_voltage_V = (
            adaptive_voltage_ramp(
                terminal="gate",
                start_voltage=(
                    current_gate_voltage_V
                ),
                target_voltage=0.0,
                initial_step=(
                    GATE_INITIAL_STEP_V
                ),
                minimum_step=(
                    GATE_MINIMUM_STEP_V
                ),
                label=(
                    f"{state_label} gate reset"
                ),
            )
        )

        current_trap_density_cm3 = (
            adaptive_trap_density_ramp(
                start_density_cm3=(
                    current_trap_density_cm3
                ),
                target_density_cm3=(
                    target_trap_density_cm3
                ),
                initial_step_cm3=(
                    TRAP_INITIAL_STEP_CM3
                ),
                minimum_step_cm3=(
                    TRAP_MINIMUM_STEP_CM3
                ),
                label=(
                    f"{state_label} trap ramp"
                ),
            )
        )

        trap_information = (
            get_trap_state_information(
                current_trap_density_cm3
            )
        )

        current_gate_voltage_V = (
            adaptive_voltage_ramp(
                terminal="gate",
                start_voltage=(
                    current_gate_voltage_V
                ),
                target_voltage=(
                    GATE_START_VOLTAGE_V
                ),
                initial_step=(
                    GATE_INITIAL_STEP_V
                ),
                minimum_step=(
                    GATE_MINIMUM_STEP_V
                ),
                label=(
                    f"{state_label} gate pre-ramp"
                ),
            )
        )

        state_iv_results = []

        for target_gate_voltage_V in (
            gate_output_voltages_V
        ):
            current_gate_voltage_V = (
                adaptive_voltage_ramp(
                    terminal="gate",
                    start_voltage=(
                        current_gate_voltage_V
                    ),
                    target_voltage=(
                        target_gate_voltage_V
                    ),
                    initial_step=(
                        GATE_INITIAL_STEP_V
                    ),
                    minimum_step=(
                        GATE_MINIMUM_STEP_V
                    ),
                    label=(
                        f"{state_label} "
                        f"VG target "
                        f"{target_gate_voltage_V:+.3f} V"
                    ),
                )
            )

            drain_current_A = (
                get_drain_current_A()
            )

            absolute_drain_current_A = abs(
                drain_current_A
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
                "state_index": (
                    state_index
                ),
                "state_label": (
                    state_label
                ),
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
                    current_gate_voltage_V
                ),
                "drain_voltage_V": (
                    current_drain_voltage_V
                ),
                "drain_current_A": (
                    drain_current_A
                ),
                "abs_drain_current_A": (
                    absolute_drain_current_A
                ),
                "log10_abs_drain_current_A": (
                    math.log10(
                        max(
                            absolute_drain_current_A,
                            CURRENT_FLOOR_A,
                        )
                    )
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
                f"{current_trap_density_cm3:.3e} cm^-3, "
                f"VG = "
                f"{current_gate_voltage_V:+.3f} V, "
                f"ID = "
                f"{drain_current_A:+.6e} A"
            )

        threshold_voltage_V = (
            extract_threshold_voltage(
                state_results=(
                    state_iv_results
                ),
                threshold_current_A=(
                    THRESHOLD_CURRENT_A
                ),
            )
        )

        if state_index == 0:
            empty_state_threshold_voltage_V = (
                threshold_voltage_V
            )

        if (
            threshold_voltage_V
            is not None
            and empty_state_threshold_voltage_V
            is not None
        ):
            threshold_shift_V = (
                threshold_voltage_V
                - empty_state_threshold_voltage_V
            )

        else:
            threshold_shift_V = (
                None
            )

        threshold_summary.append(
            {
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
                "state_index": (
                    state_index
                ),
                "state_label": (
                    state_label
                ),
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
                    THRESHOLD_CURRENT_A
                ),
                "threshold_voltage_V": (
                    threshold_voltage_V
                ),
                "empty_state_threshold_voltage_V": (
                    empty_state_threshold_voltage_V
                ),
                "threshold_shift_from_empty_V": (
                    threshold_shift_V
                ),
                "threshold_found": (
                    threshold_voltage_V
                    is not None
                ),
            }
        )

        if threshold_voltage_V is None:
            print(
                f"{state_label}: threshold current "
                f"{THRESHOLD_CURRENT_A:.3e} A "
                f"was not crossed within the "
                f"gate sweep range."
            )

        else:
            print(
                f"{state_label}: "
                f"Vth = "
                f"{threshold_voltage_V:.6f} V"
            )

            print(
                f"{state_label}: "
                f"Delta Vth from empty state = "
                f"{threshold_shift_V:.6f} V"
            )

    return (
        all_iv_results,
        threshold_summary,
    )


# ============================================================
# CSV output
# ============================================================

MEMORY_WINDOW_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "charge_trap_thickness_nm",
    "state_index",
    "state_label",
    "trap_density_cm3",
    "trap_sheet_density_cm2",
    "trap_charge_density_C_cm3",
    "gate_voltage_V",
    "drain_voltage_V",
    "drain_current_A",
    "abs_drain_current_A",
    "log10_abs_drain_current_A",
)

THRESHOLD_SUMMARY_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "charge_trap_thickness_nm",
    "state_index",
    "state_label",
    "trap_density_cm3",
    "trap_sheet_density_cm2",
    "threshold_current_A",
    "threshold_voltage_V",
    "empty_state_threshold_voltage_V",
    "threshold_shift_from_empty_V",
    "threshold_found",
)


def write_csv(
    output_path,
    fieldnames,
    rows,
):
    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="raise",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# Console summary
# ============================================================

def print_threshold_summary(
    threshold_summary,
):
    print_section(
        "STEP 21 : THRESHOLD-VOLTAGE RESULTS"
    )

    for summary_row in (
        threshold_summary
    ):
        state_index = int(
            summary_row[
                "state_index"
            ]
        )

        state_label = (
            summary_row[
                "state_label"
            ]
        )

        trap_density_cm3 = float(
            summary_row[
                "trap_density_cm3"
            ]
        )

        trap_sheet_density_cm2 = float(
            summary_row[
                "trap_sheet_density_cm2"
            ]
        )

        threshold_voltage_V = (
            summary_row[
                "threshold_voltage_V"
            ]
        )

        threshold_shift_V = (
            summary_row[
                "threshold_shift_from_empty_V"
            ]
        )

        print()
        print(
            f"State {state_index}: "
            f"{state_label}"
        )

        print(
            f"  Ntrap volume = "
            f"{trap_density_cm3:.6e} cm^-3"
        )

        print(
            f"  Ntrap sheet  = "
            f"{trap_sheet_density_cm2:.6e} cm^-2"
        )

        print(
            f"  Vth          = "
            f"{format_optional_float(threshold_voltage_V)} V"
        )

        print(
            f"  Delta Vth    = "
            f"{format_optional_float(threshold_shift_V)} V"
        )


# ============================================================
# Optional VTK output
# ============================================================

def write_optional_vtk():
    if not FINAL_VTK_OUTPUT:
        return

    output_path = Path(
        FINAL_VTK_OUTPUT
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_devices(
        file=str(
            output_path
        ),
        type="vtk",
    )

    print(
        f'Final device data written to '
        f'"{output_path.resolve()}".'
    )


# ============================================================
# Main
# ============================================================

def main():
    print(
        "RUN_MEMORY_WINDOW SCRIPT STARTED"
    )

    geometry = (
        initialize_device()
    )

    print_device_information()

    solve_empty_state()

    (
        all_iv_results,
        threshold_summary,
    ) = run_memory_window_sweep(
        geometry=geometry,
    )

    print_section(
        "STEP 19 : SAVE MEMORY-WINDOW I_D-V_G DATA"
    )

    write_csv(
        output_path=(
            MEMORY_WINDOW_CSV
        ),
        fieldnames=(
            MEMORY_WINDOW_FIELDNAMES
        ),
        rows=(
            all_iv_results
        ),
    )

    print(
        f'Full memory-window data saved to '
        f'"{MEMORY_WINDOW_CSV.resolve()}".'
    )

    print_section(
        "STEP 20 : SAVE THRESHOLD-VOLTAGE SUMMARY"
    )

    write_csv(
        output_path=(
            THRESHOLD_SUMMARY_CSV
        ),
        fieldnames=(
            THRESHOLD_SUMMARY_FIELDNAMES
        ),
        rows=(
            threshold_summary
        ),
    )

    print(
        f'Threshold-voltage summary saved to '
        f'"{THRESHOLD_SUMMARY_CSV.resolve()}".'
    )

    print_threshold_summary(
        threshold_summary=(
            threshold_summary
        ),
    )

    write_optional_vtk()

    print_section(
        "MEMORY-WINDOW SIMULATION COMPLETED"
    )

    print(
        "TunnelOxide thickness: "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.6f} nm"
    )

    print(
        "Trap-state count: "
        f"{len(TRAPPED_ELECTRON_DENSITY_STATES_CM3)}"
    )

    print(
        "Threshold current: "
        f"{THRESHOLD_CURRENT_A:.6e} A"
    )

    print(
        "ID-VG output:\n"
        f"{MEMORY_WINDOW_CSV.resolve()}"
    )

    print(
        "Vth output:\n"
        f"{THRESHOLD_SUMMARY_CSV.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
