# ============================================================
# Quasi-static trapped-charge feedback program simulation
#
# Comparison conditions
# ---------------------
# TunnelOxide thickness:
#   3.0 nm
#
# Program gate voltages:
#   11.0 V
#   12.0 V
#
# Method
# ------
# 1. Select a trapped-electron sheet-density grid.
# 2. Convert sheet density to volume density:
#
#       Ntrap_volume = Ntrap_sheet / trap_thickness
#
# 3. Re-run the electrostatic / drift-diffusion solution
#    at every trapped-charge point.
# 4. Recalculate the tunnel-oxide field and FN current.
# 5. Integrate:
#
#       dNsheet/dt = eta_capture * J / q
#
#       dt = q / eta_capture * dNsheet / J
#
# 6. Compare the accumulated program time at 11 V and 12 V.
#
# Important
# ---------
# This is a quasi-static feedback calculation, not a fully
# coupled transient DEVSIM trap-rate equation.
# ============================================================

import csv
import math
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# Physical constant
# ============================================================

ELEMENTARY_CHARGE_C = 1.602176634e-19


# ============================================================
# Simulation conditions
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = 3.0

PROGRAM_GATE_VOLTAGES_V = (
    11.0,
    12.0,
)

DRAIN_VOLTAGE_V = 0.05

CHARGE_TRAP_THICKNESS_NM = 5.0

TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2 = (
    1.0e12
)

NUMBER_OF_SHEET_DENSITY_INTERVALS = 20

CAPTURE_EFFICIENCY = 1.0


# ============================================================
# Paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

STATIC_SWEEP_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "program_feedback_3p0nm_11V_12V"
)

STATIC_CASE_DIRECTORY = (
    RESULTS_DIRECTORY
    / "static_cases"
)

STATIC_POINTS_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "program_feedback_static_points.csv"
)

INTEGRATION_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "program_feedback_time_integration.csv"
)

COMPARISON_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "program_feedback_comparison.csv"
)


# ============================================================
# Result-column names
# ============================================================

TARGET_REGION = "TunnelOxide"

CURRENT_DENSITY_COLUMNS = (
    "edge_resolved_effective_current_density_A_cm2",
    "total_tunneling_current_density_A_cm2",
)

AVERAGE_FIELD_COLUMNS = (
    "active_interface_field_area_weighted_abs_V_cm",
    "active_interface_field_mean_abs_V_cm",
    "tunnel_interface_field_mean_abs_V_cm",
)

MAXIMUM_FIELD_COLUMNS = (
    "edge_resolved_maximum_local_field_abs_V_cm",
    "active_interface_field_max_abs_V_cm",
    "tunnel_interface_field_max_abs_V_cm",
)


# ============================================================
# Console utility
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


# ============================================================
# Validation
# ============================================================

def validate_finite(
    name,
    value,
):
    numeric_value = float(value)

    if not math.isfinite(
        numeric_value
    ):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_positive(
    name,
    value,
):
    numeric_value = validate_finite(
        name=name,
        value=value,
    )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_nonnegative(
    name,
    value,
):
    numeric_value = validate_finite(
        name=name,
        value=value,
    )

    if numeric_value < 0.0:
        raise ValueError(
            f"{name} must not be negative. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_settings():
    if not STATIC_SWEEP_SCRIPT.exists():
        raise FileNotFoundError(
            "run_positive_field_sweep.py "
            "was not found:\n"
            f"{STATIC_SWEEP_SCRIPT.resolve()}"
        )

    validate_positive(
        name="TUNNEL_OXIDE_THICKNESS_NM",
        value=TUNNEL_OXIDE_THICKNESS_NM,
    )

    validate_positive(
        name="CHARGE_TRAP_THICKNESS_NM",
        value=CHARGE_TRAP_THICKNESS_NM,
    )

    validate_positive(
        name=(
            "TARGET_TRAPPED_ELECTRON_"
            "SHEET_DENSITY_CM2"
        ),
        value=(
            TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
        ),
    )

    if (
        not isinstance(
            NUMBER_OF_SHEET_DENSITY_INTERVALS,
            int,
        )
        or NUMBER_OF_SHEET_DENSITY_INTERVALS <= 0
    ):
        raise ValueError(
            "NUMBER_OF_SHEET_DENSITY_INTERVALS "
            "must be a positive integer."
        )

    capture_efficiency = validate_positive(
        name="CAPTURE_EFFICIENCY",
        value=CAPTURE_EFFICIENCY,
    )

    if capture_efficiency > 1.0:
        raise ValueError(
            "CAPTURE_EFFICIENCY must not exceed 1.0."
        )

    if not PROGRAM_GATE_VOLTAGES_V:
        raise ValueError(
            "At least one program gate voltage "
            "must be provided."
        )

    previous_voltage = None

    for gate_voltage_V in (
        PROGRAM_GATE_VOLTAGES_V
    ):
        validate_nonnegative(
            name="program gate voltage",
            value=gate_voltage_V,
        )

        if (
            previous_voltage is not None
            and gate_voltage_V <= previous_voltage
        ):
            raise ValueError(
                "PROGRAM_GATE_VOLTAGES_V must be "
                "strictly increasing."
            )

        previous_voltage = gate_voltage_V


# ============================================================
# Unit conversion
# ============================================================

def nm_to_cm(
    thickness_nm,
):
    return float(thickness_nm) * 1.0e-7


def sheet_to_volume_density_cm3(
    sheet_density_cm2,
):
    trap_thickness_cm = nm_to_cm(
        CHARGE_TRAP_THICKNESS_NM
    )

    if trap_thickness_cm <= 0.0:
        raise RuntimeError(
            "Charge-trap thickness must be positive."
        )

    return (
        float(sheet_density_cm2)
        / trap_thickness_cm
    )


# ============================================================
# Sheet-density grid
# ============================================================

def build_sheet_density_grid():
    interval_count = (
        NUMBER_OF_SHEET_DENSITY_INTERVALS
    )

    target_density_cm2 = (
        TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
    )

    step_density_cm2 = (
        target_density_cm2
        / interval_count
    )

    return tuple(
        index * step_density_cm2
        for index in range(
            interval_count + 1
        )
    )


# ============================================================
# Filename utilities
# ============================================================

def voltage_token(
    voltage_V,
):
    return (
        f"{float(voltage_V):.1f}"
        .replace(".", "p")
    )


def density_token(
    density_cm2,
):
    return f"{float(density_cm2):.3e}".replace(
        "+",
        "",
    ).replace(
        ".",
        "p",
    )


def get_static_case_output_path(
    gate_voltage_V,
    sheet_density_cm2,
):
    return (
        STATIC_CASE_DIRECTORY
        / (
            "feedback_static_"
            f"vg_{voltage_token(gate_voltage_V)}V_"
            f"nsheet_{density_token(sheet_density_cm2)}"
            ".csv"
        )
    )


# ============================================================
# CSV utilities
# ============================================================

def read_csv_rows(
    csv_path,
):
    with Path(csv_path).open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = reader.fieldnames
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(
            "CSV header was not found:\n"
            f"{Path(csv_path).resolve()}"
        )

    if not rows:
        raise RuntimeError(
            "CSV contains no rows:\n"
            f"{Path(csv_path).resolve()}"
        )

    return rows


def get_first_available_float(
    row,
    candidate_columns,
):
    for column_name in candidate_columns:
        value = row.get(
            column_name,
            "",
        )

        if value not in (
            "",
            None,
        ):
            numeric_value = float(value)

            if not math.isfinite(
                numeric_value
            ):
                raise RuntimeError(
                    "Non-finite CSV value found in "
                    f'"{column_name}".'
                )

            return numeric_value

    raise RuntimeError(
        "None of the requested CSV columns "
        "was found:\n"
        f"{candidate_columns}"
    )


def select_tunnel_oxide_row(
    rows,
):
    matching_rows = [
        row
        for row in rows
        if row.get(
            "region",
            "",
        ).strip() == TARGET_REGION
    ]

    if len(matching_rows) != 1:
        raise RuntimeError(
            "Expected exactly one TunnelOxide row, "
            f"but found {len(matching_rows)}."
        )

    return matching_rows[0]


# ============================================================
# Static feedback point
# ============================================================

def run_static_feedback_point(
    gate_voltage_V,
    sheet_density_cm2,
):
    volume_density_cm3 = (
        sheet_to_volume_density_cm3(
            sheet_density_cm2
        )
    )

    output_csv = (
        get_static_case_output_path(
            gate_voltage_V=gate_voltage_V,
            sheet_density_cm2=sheet_density_cm2,
        )
    )

    environment = os.environ.copy()

    environment[
        "TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        TUNNEL_OXIDE_THICKNESS_NM
    )

    environment[
        "GATE_VOLTAGES_V"
    ] = str(
        gate_voltage_V
    )

    environment[
        "TRAPPED_ELECTRON_DENSITY_CM3"
    ] = str(
        volume_density_cm3
    )

    environment[
        "POSITIVE_FIELD_SWEEP_OUTPUT"
    ] = str(
        output_csv.resolve()
    )

    print()
    print("-" * 96)

    print(
        "STATIC FEEDBACK POINT"
    )

    print(
        f"  Gate voltage          : "
        f"{gate_voltage_V:.3f} V"
    )

    print(
        f"  Trapped sheet density : "
        f"{sheet_density_cm2:.6e} cm^-2"
    )

    print(
        f"  Trapped volume density: "
        f"{volume_density_cm3:.6e} cm^-3"
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            str(STATIC_SWEEP_SCRIPT),
        ],
        cwd=str(
            SCRIPT_DIRECTORY
        ),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Static feedback simulation failed.\n"
            f"Gate voltage: {gate_voltage_V:.3f} V\n"
            "Trapped sheet density: "
            f"{sheet_density_cm2:.6e} cm^-2\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "Static output CSV was not created:\n"
            f"{output_csv.resolve()}"
        )

    rows = read_csv_rows(
        output_csv
    )

    tunnel_oxide_row = (
        select_tunnel_oxide_row(
            rows
        )
    )

    current_density_A_cm2 = (
        get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                CURRENT_DENSITY_COLUMNS
            ),
        )
    )

    average_field_V_cm = (
        get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                AVERAGE_FIELD_COLUMNS
            ),
        )
    )

    maximum_field_V_cm = (
        get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                MAXIMUM_FIELD_COLUMNS
            ),
        )
    )

    actual_trap_thickness_nm = (
        get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                "charge_trap_thickness_nm",
            ),
        )
    )

    thickness_difference_nm = abs(
        actual_trap_thickness_nm
        - CHARGE_TRAP_THICKNESS_NM
    )

    if thickness_difference_nm > 1.0e-9:
        raise RuntimeError(
            "Charge-trap thickness mismatch.\n"
            "Controller value: "
            f"{CHARGE_TRAP_THICKNESS_NM:.6f} nm\n"
            "Simulation output value: "
            f"{actual_trap_thickness_nm:.6f} nm"
        )

    if current_density_A_cm2 < 0.0:
        raise RuntimeError(
            "Tunneling current-density magnitude "
            "must not be negative."
        )

    return {
        "tunnel_oxide_thickness_nm": (
            TUNNEL_OXIDE_THICKNESS_NM
        ),
        "charge_trap_thickness_nm": (
            CHARGE_TRAP_THICKNESS_NM
        ),
        "gate_voltage_V": (
            gate_voltage_V
        ),
        "drain_voltage_V": (
            tunnel_oxide_row.get(
                "drain_voltage_V",
                DRAIN_VOLTAGE_V,
            )
        ),
        "trapped_electron_sheet_density_cm2": (
            sheet_density_cm2
        ),
        "trapped_electron_volume_density_cm3": (
            volume_density_cm3
        ),
        "active_interface_average_field_abs_V_cm": (
            average_field_V_cm
        ),
        "active_interface_maximum_field_abs_V_cm": (
            maximum_field_V_cm
        ),
        "tunneling_current_density_A_cm2": (
            current_density_A_cm2
        ),
        "static_output_csv": (
            str(
                output_csv.resolve()
            )
        ),
    }


# ============================================================
# Time integration
# ============================================================

def integrate_program_time(
    static_points,
):
    if len(static_points) < 2:
        raise ValueError(
            "At least two static points are required."
        )

    integration_rows = []

    cumulative_time_s = 0.0

    for interval_index in range(
        len(static_points) - 1
    ):
        lower_point = (
            static_points[
                interval_index
            ]
        )

        upper_point = (
            static_points[
                interval_index + 1
            ]
        )

        lower_density_cm2 = float(
            lower_point[
                "trapped_electron_sheet_density_cm2"
            ]
        )

        upper_density_cm2 = float(
            upper_point[
                "trapped_electron_sheet_density_cm2"
            ]
        )

        density_increment_cm2 = (
            upper_density_cm2
            - lower_density_cm2
        )

        if density_increment_cm2 <= 0.0:
            raise RuntimeError(
                "Sheet-density grid must be increasing."
            )

        lower_current_density_A_cm2 = float(
            lower_point[
                "tunneling_current_density_A_cm2"
            ]
        )

        upper_current_density_A_cm2 = float(
            upper_point[
                "tunneling_current_density_A_cm2"
            ]
        )

        if (
            lower_current_density_A_cm2 <= 0.0
            or upper_current_density_A_cm2 <= 0.0
        ):
            interval_time_s = math.inf

        else:
            inverse_current_average = 0.5 * (
                1.0
                / lower_current_density_A_cm2
                + 1.0
                / upper_current_density_A_cm2
            )

            interval_time_s = (
                ELEMENTARY_CHARGE_C
                * density_increment_cm2
                * inverse_current_average
                / CAPTURE_EFFICIENCY
            )

        if math.isinf(
            cumulative_time_s
        ) or math.isinf(
            interval_time_s
        ):
            cumulative_time_s = math.inf

        else:
            cumulative_time_s += (
                interval_time_s
            )

        current_ratio = (
            upper_current_density_A_cm2
            / lower_current_density_A_cm2
            if lower_current_density_A_cm2 > 0.0
            else math.nan
        )

        integration_rows.append(
            {
                "gate_voltage_V": (
                    lower_point[
                        "gate_voltage_V"
                    ]
                ),
                "interval_index": (
                    interval_index
                ),
                "sheet_density_lower_cm2": (
                    lower_density_cm2
                ),
                "sheet_density_upper_cm2": (
                    upper_density_cm2
                ),
                "sheet_density_increment_cm2": (
                    density_increment_cm2
                ),
                "current_density_lower_A_cm2": (
                    lower_current_density_A_cm2
                ),
                "current_density_upper_A_cm2": (
                    upper_current_density_A_cm2
                ),
                "upper_to_lower_current_ratio": (
                    current_ratio
                ),
                "average_field_lower_V_cm": (
                    lower_point[
                        "active_interface_average_field_abs_V_cm"
                    ]
                ),
                "average_field_upper_V_cm": (
                    upper_point[
                        "active_interface_average_field_abs_V_cm"
                    ]
                ),
                "maximum_field_lower_V_cm": (
                    lower_point[
                        "active_interface_maximum_field_abs_V_cm"
                    ]
                ),
                "maximum_field_upper_V_cm": (
                    upper_point[
                        "active_interface_maximum_field_abs_V_cm"
                    ]
                ),
                "capture_efficiency": (
                    CAPTURE_EFFICIENCY
                ),
                "interval_program_time_s": (
                    interval_time_s
                ),
                "cumulative_program_time_s": (
                    cumulative_time_s
                ),
            }
        )

    return integration_rows


# ============================================================
# CSV writing
# ============================================================

STATIC_POINT_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "charge_trap_thickness_nm",
    "gate_voltage_V",
    "drain_voltage_V",
    "trapped_electron_sheet_density_cm2",
    "trapped_electron_volume_density_cm3",
    "active_interface_average_field_abs_V_cm",
    "active_interface_maximum_field_abs_V_cm",
    "tunneling_current_density_A_cm2",
    "static_output_csv",
)

INTEGRATION_FIELDNAMES = (
    "gate_voltage_V",
    "interval_index",
    "sheet_density_lower_cm2",
    "sheet_density_upper_cm2",
    "sheet_density_increment_cm2",
    "current_density_lower_A_cm2",
    "current_density_upper_A_cm2",
    "upper_to_lower_current_ratio",
    "average_field_lower_V_cm",
    "average_field_upper_V_cm",
    "maximum_field_lower_V_cm",
    "maximum_field_upper_V_cm",
    "capture_efficiency",
    "interval_program_time_s",
    "cumulative_program_time_s",
)

COMPARISON_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "gate_voltage_V",
    "target_sheet_density_cm2",
    "capture_efficiency",
    "initial_current_density_A_cm2",
    "final_current_density_A_cm2",
    "final_to_initial_current_ratio",
    "initial_average_field_V_cm",
    "final_average_field_V_cm",
    "final_to_initial_average_field_ratio",
    "initial_maximum_field_V_cm",
    "final_maximum_field_V_cm",
    "feedback_program_time_s",
    "fixed_initial_current_program_time_s",
    "feedback_to_fixed_time_ratio",
)


def write_csv(
    output_path,
    fieldnames,
    rows,
):
    with Path(output_path).open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# Comparison summary
# ============================================================

def build_comparison_row(
    gate_voltage_V,
    static_points,
    integration_rows,
):
    initial_point = static_points[0]
    final_point = static_points[-1]

    initial_current_density_A_cm2 = float(
        initial_point[
            "tunneling_current_density_A_cm2"
        ]
    )

    final_current_density_A_cm2 = float(
        final_point[
            "tunneling_current_density_A_cm2"
        ]
    )

    initial_average_field_V_cm = float(
        initial_point[
            "active_interface_average_field_abs_V_cm"
        ]
    )

    final_average_field_V_cm = float(
        final_point[
            "active_interface_average_field_abs_V_cm"
        ]
    )

    initial_maximum_field_V_cm = float(
        initial_point[
            "active_interface_maximum_field_abs_V_cm"
        ]
    )

    final_maximum_field_V_cm = float(
        final_point[
            "active_interface_maximum_field_abs_V_cm"
        ]
    )

    feedback_program_time_s = float(
        integration_rows[-1][
            "cumulative_program_time_s"
        ]
    )

    fixed_initial_current_program_time_s = (
        ELEMENTARY_CHARGE_C
        * TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
        / (
            CAPTURE_EFFICIENCY
            * initial_current_density_A_cm2
        )
    )

    return {
        "tunnel_oxide_thickness_nm": (
            TUNNEL_OXIDE_THICKNESS_NM
        ),
        "gate_voltage_V": (
            gate_voltage_V
        ),
        "target_sheet_density_cm2": (
            TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
        ),
        "capture_efficiency": (
            CAPTURE_EFFICIENCY
        ),
        "initial_current_density_A_cm2": (
            initial_current_density_A_cm2
        ),
        "final_current_density_A_cm2": (
            final_current_density_A_cm2
        ),
        "final_to_initial_current_ratio": (
            final_current_density_A_cm2
            / initial_current_density_A_cm2
        ),
        "initial_average_field_V_cm": (
            initial_average_field_V_cm
        ),
        "final_average_field_V_cm": (
            final_average_field_V_cm
        ),
        "final_to_initial_average_field_ratio": (
            final_average_field_V_cm
            / initial_average_field_V_cm
        ),
        "initial_maximum_field_V_cm": (
            initial_maximum_field_V_cm
        ),
        "final_maximum_field_V_cm": (
            final_maximum_field_V_cm
        ),
        "feedback_program_time_s": (
            feedback_program_time_s
        ),
        "fixed_initial_current_program_time_s": (
            fixed_initial_current_program_time_s
        ),
        "feedback_to_fixed_time_ratio": (
            feedback_program_time_s
            / fixed_initial_current_program_time_s
        ),
    }


# ============================================================
# Console result
# ============================================================

def print_voltage_result(
    comparison_row,
):
    print()
    print(
        f"VG = "
        f"{comparison_row['gate_voltage_V']:.3f} V"
    )

    print(
        "  Initial J             = "
        f"{comparison_row['initial_current_density_A_cm2']:.6e} "
        "A/cm^2"
    )

    print(
        "  Final J               = "
        f"{comparison_row['final_current_density_A_cm2']:.6e} "
        "A/cm^2"
    )

    print(
        "  Final / initial J     = "
        f"{comparison_row['final_to_initial_current_ratio']:.6e}"
    )

    print(
        "  Initial average field = "
        f"{comparison_row['initial_average_field_V_cm']:.6e} "
        "V/cm"
    )

    print(
        "  Final average field   = "
        f"{comparison_row['final_average_field_V_cm']:.6e} "
        "V/cm"
    )

    print(
        "  Fixed-field time      = "
        f"{comparison_row['fixed_initial_current_program_time_s']:.6e} "
        "s"
    )

    print(
        "  Feedback time         = "
        f"{comparison_row['feedback_program_time_s']:.6e} "
        "s"
    )

    print(
        "  Feedback / fixed time = "
        f"{comparison_row['feedback_to_fixed_time_ratio']:.6e}"
    )


# ============================================================
# Main
# ============================================================

def main():
    validate_settings()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATIC_CASE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet_density_grid = (
        build_sheet_density_grid()
    )

    print_section(
        "QUASI-STATIC PROGRAM FEEDBACK COMPARISON"
    )

    print(
        "TunnelOxide thickness: "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "ChargeTrap thickness: "
        f"{CHARGE_TRAP_THICKNESS_NM:.3f} nm"
    )

    print(
        "Gate voltages: "
        f"{PROGRAM_GATE_VOLTAGES_V}"
    )

    print(
        "Target sheet density: "
        f"{TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2:.6e} "
        "cm^-2"
    )

    print(
        "Sheet-density intervals: "
        f"{NUMBER_OF_SHEET_DENSITY_INTERVALS}"
    )

    print(
        "Capture efficiency: "
        f"{CAPTURE_EFFICIENCY:.6f}"
    )

    all_static_points = []
    all_integration_rows = []
    comparison_rows = []

    for gate_voltage_V in (
        PROGRAM_GATE_VOLTAGES_V
    ):
        print_section(
            f"PROGRAM FEEDBACK AT VG = "
            f"{gate_voltage_V:.3f} V"
        )

        voltage_static_points = []

        for sheet_density_cm2 in (
            sheet_density_grid
        ):
            static_point = (
                run_static_feedback_point(
                    gate_voltage_V=gate_voltage_V,
                    sheet_density_cm2=(
                        sheet_density_cm2
                    ),
                )
            )

            voltage_static_points.append(
                static_point
            )

            all_static_points.append(
                static_point
            )

        voltage_integration_rows = (
            integrate_program_time(
                static_points=(
                    voltage_static_points
                ),
            )
        )

        all_integration_rows.extend(
            voltage_integration_rows
        )

        comparison_row = (
            build_comparison_row(
                gate_voltage_V=gate_voltage_V,
                static_points=(
                    voltage_static_points
                ),
                integration_rows=(
                    voltage_integration_rows
                ),
            )
        )

        comparison_rows.append(
            comparison_row
        )

        print_voltage_result(
            comparison_row
        )

    write_csv(
        output_path=STATIC_POINTS_OUTPUT_CSV,
        fieldnames=STATIC_POINT_FIELDNAMES,
        rows=all_static_points,
    )

    write_csv(
        output_path=INTEGRATION_OUTPUT_CSV,
        fieldnames=INTEGRATION_FIELDNAMES,
        rows=all_integration_rows,
    )

    write_csv(
        output_path=COMPARISON_OUTPUT_CSV,
        fieldnames=COMPARISON_FIELDNAMES,
        rows=comparison_rows,
    )

    print_section(
        "PROGRAM FEEDBACK COMPARISON COMPLETED"
    )

    for comparison_row in comparison_rows:
        print_voltage_result(
            comparison_row
        )

    print()
    print(
        "Static-point CSV:\n"
        f"{STATIC_POINTS_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Time-integration CSV:\n"
        f"{INTEGRATION_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Comparison CSV:\n"
        f"{COMPARISON_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Interpretation note:"
    )

    print(
        "A feedback-to-fixed time ratio greater than 1 "
        "means that accumulated trapped charge reduced "
        "the tunneling current and increased the program time."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_PROGRAM_FEEDBACK_COMPARE SCRIPT STARTED"
    )

    main()