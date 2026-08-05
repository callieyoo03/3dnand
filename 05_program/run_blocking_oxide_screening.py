# ============================================================
# Blocking-oxide thickness screening
#
# Fixed conditions
# ----------------
# TunnelOxide thickness = 3.0 nm
# ChargeTrap thickness  = 5.0 nm
# Gate voltage          = 12.0 V
#
# Sweep condition
# ---------------
# BlockingOxide thickness:
#   8, 12, 16, 20, 30 nm
#
# Purpose
# -------
# 1. Reuse the existing run_positive_field_sweep.py
# 2. Apply charge-trap and blocking-oxide thicknesses without
#    modifying the existing positive-field sweep file
# 3. Run each geometry in an independent subprocess
# 4. Combine detailed results
# 5. Extract TunnelOxide and BlockingOxide field statistics
# 6. Estimate fixed-field time to reach the target sheet charge
#
# Important
# ---------
# Program-time estimates in this screening are based on:
#
#     t = q * N_target / J
#
# They do not yet include:
#   - trapped-charge feedback
#   - trap saturation
#   - capture efficiency below 100%
#   - gate-side injection
#   - oxide degradation or breakdown
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
# Fixed structure and bias conditions
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = 3.0

CHARGE_TRAP_THICKNESS_NM = 5.0

BLOCKING_OXIDE_THICKNESSES_NM = (
    8.0,
    12.0,
    16.0,
    20.0,
    30.0,
)

GATE_VOLTAGE_V = 12.0

TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2 = (
    1.0e12
)


# ============================================================
# Result regions
# ============================================================

TUNNEL_OXIDE_REGION = "TunnelOxide"

BLOCKING_OXIDE_REGION = "BlockingOxide"


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIRECTORY = Path(
    __file__
).resolve().parent

POSITIVE_FIELD_SWEEP_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "blocking_oxide_screening_12V"
)

INDIVIDUAL_CASE_DIRECTORY = (
    RESULTS_DIRECTORY
    / "individual_cases"
)

COMBINED_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_screening_combined.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_screening_summary.csv"
)

RANKING_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_screening_ranking.csv"
)


# ============================================================
# CSV columns used for fallback lookup
# ============================================================

TUNNEL_CURRENT_DENSITY_COLUMNS = (
    "edge_resolved_effective_current_density_A_cm2",
    "total_tunneling_current_density_A_cm2",
    "fowler_nordheim_current_density_A_cm2",
)

TUNNEL_AVERAGE_FIELD_COLUMNS = (
    "active_interface_field_area_weighted_abs_V_cm",
    "active_interface_field_mean_abs_V_cm",
    "tunnel_interface_field_mean_abs_V_cm",
    "field_mean_abs_V_cm",
)

TUNNEL_MAXIMUM_FIELD_COLUMNS = (
    "edge_resolved_maximum_local_field_abs_V_cm",
    "active_interface_field_max_abs_V_cm",
    "tunnel_interface_field_max_abs_V_cm",
    "field_max_abs_V_cm",
)

BLOCKING_AVERAGE_FIELD_COLUMNS = (
    "field_mean_abs_V_cm",
)

BLOCKING_MAXIMUM_FIELD_COLUMNS = (
    "field_max_abs_V_cm",
)


# ============================================================
# Console utilities
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 96)
    print(title)
    print("=" * 96)


def print_case_separator():
    print()
    print("-" * 96)


# ============================================================
# Validation utilities
# ============================================================

def validate_finite(
    name,
    value,
):
    numeric_value = float(
        value
    )

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
    if not POSITIVE_FIELD_SWEEP_SCRIPT.exists():
        raise FileNotFoundError(
            "run_positive_field_sweep.py was not found:\n"
            f"{POSITIVE_FIELD_SWEEP_SCRIPT.resolve()}"
        )

    validate_positive(
        name="TUNNEL_OXIDE_THICKNESS_NM",
        value=TUNNEL_OXIDE_THICKNESS_NM,
    )

    validate_positive(
        name="CHARGE_TRAP_THICKNESS_NM",
        value=CHARGE_TRAP_THICKNESS_NM,
    )

    validate_nonnegative(
        name="GATE_VOLTAGE_V",
        value=GATE_VOLTAGE_V,
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

    if not BLOCKING_OXIDE_THICKNESSES_NM:
        raise ValueError(
            "At least one blocking-oxide thickness "
            "must be provided."
        )

    previous_thickness_nm = None

    for thickness_nm in (
        BLOCKING_OXIDE_THICKNESSES_NM
    ):
        validate_positive(
            name="BlockingOxide thickness",
            value=thickness_nm,
        )

        if (
            previous_thickness_nm is not None
            and thickness_nm <= previous_thickness_nm
        ):
            raise ValueError(
                "BLOCKING_OXIDE_THICKNESSES_NM must be "
                "strictly increasing."
            )

        previous_thickness_nm = (
            thickness_nm
        )


# ============================================================
# Filename utilities
# ============================================================

def number_to_filename_token(
    value,
):
    return (
        f"{float(value):.1f}"
        .replace(
            ".",
            "p",
        )
        .replace(
            "-",
            "m",
        )
    )


def get_case_output_csv(
    blocking_oxide_thickness_nm,
):
    tunnel_token = number_to_filename_token(
        TUNNEL_OXIDE_THICKNESS_NM
    )

    trap_token = number_to_filename_token(
        CHARGE_TRAP_THICKNESS_NM
    )

    blocking_token = number_to_filename_token(
        blocking_oxide_thickness_nm
    )

    gate_token = number_to_filename_token(
        GATE_VOLTAGE_V
    )

    return (
        INDIVIDUAL_CASE_DIRECTORY
        / (
            "field_screening_"
            f"tox_{tunnel_token}nm_"
            f"trap_{trap_token}nm_"
            f"block_{blocking_token}nm_"
            f"vg_{gate_token}V.csv"
        )
    )


# ============================================================
# Wrapper code
#
# The existing run_positive_field_sweep.py imports:
#
#     create_structure
#
# directly from device_structure.py. Its main() only supplies
# tunnel_oxide_thickness_nm.
#
# The wrapper below imports the existing sweep as a module,
# replaces its create_structure reference with a configured
# wrapper, and then calls its existing main().
#
# This allows charge-trap and blocking-oxide geometry sweeps
# without modifying run_positive_field_sweep.py.
# ============================================================

def build_subprocess_wrapper_code():
    return r'''
import os

import run_positive_field_sweep as positive_sweep


_original_create_structure = (
    positive_sweep.create_structure
)


def configured_create_structure(
    *args,
    **kwargs,
):
    kwargs[
        "tunnel_oxide_thickness_nm"
    ] = float(
        os.environ[
            "TUNNEL_OXIDE_THICKNESS_NM"
        ]
    )

    kwargs[
        "charge_trap_thickness_nm"
    ] = float(
        os.environ[
            "CHARGE_TRAP_THICKNESS_NM"
        ]
    )

    kwargs[
        "blocking_oxide_thickness_nm"
    ] = float(
        os.environ[
            "BLOCKING_OXIDE_THICKNESS_NM"
        ]
    )

    return _original_create_structure(
        *args,
        **kwargs,
    )


positive_sweep.create_structure = (
    configured_create_structure
)

positive_sweep.main()
'''


# ============================================================
# Individual blocking-oxide case execution
# ============================================================

def run_single_blocking_oxide_case(
    blocking_oxide_thickness_nm,
):
    output_csv = get_case_output_csv(
        blocking_oxide_thickness_nm
    )

    environment = os.environ.copy()

    environment[
        "TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        float(
            TUNNEL_OXIDE_THICKNESS_NM
        )
    )

    environment[
        "CHARGE_TRAP_THICKNESS_NM"
    ] = str(
        float(
            CHARGE_TRAP_THICKNESS_NM
        )
    )

    environment[
        "BLOCKING_OXIDE_THICKNESS_NM"
    ] = str(
        float(
            blocking_oxide_thickness_nm
        )
    )

    environment[
        "GATE_VOLTAGES_V"
    ] = str(
        float(
            GATE_VOLTAGE_V
        )
    )

    environment[
        "POSITIVE_FIELD_SWEEP_OUTPUT"
    ] = str(
        output_csv.resolve()
    )

    wrapper_code = (
        build_subprocess_wrapper_code()
    )

    print_case_separator()

    print(
        "STARTING BLOCKING-OXIDE CASE"
    )

    print(
        "  TunnelOxide thickness : "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "  ChargeTrap thickness  : "
        f"{CHARGE_TRAP_THICKNESS_NM:.3f} nm"
    )

    print(
        "  BlockingOxide thickness: "
        f"{blocking_oxide_thickness_nm:.3f} nm"
    )

    print(
        "  Gate voltage          : "
        f"{GATE_VOLTAGE_V:.3f} V"
    )

    print(
        "  Output CSV:\n"
        f"  {output_csv.resolve()}"
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            wrapper_code,
        ],
        cwd=str(
            SCRIPT_DIRECTORY
        ),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Blocking-oxide screening case failed.\n"
            "BlockingOxide thickness: "
            f"{blocking_oxide_thickness_nm:.3f} nm\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "The subprocess completed, but the expected "
            "output CSV was not created:\n"
            f"{output_csv.resolve()}"
        )

    print(
        "Blocking-oxide case completed: "
        f"{blocking_oxide_thickness_nm:.3f} nm"
    )

    return output_csv


# ============================================================
# CSV reading
# ============================================================

def read_csv_rows(
    csv_path,
):
    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():
        raise FileNotFoundError(
            "CSV file was not found:\n"
            f"{csv_path.resolve()}"
        )

    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = reader.fieldnames
        rows = list(
            reader
        )

    if not fieldnames:
        raise RuntimeError(
            "CSV header was not found:\n"
            f"{csv_path.resolve()}"
        )

    if not rows:
        raise RuntimeError(
            "CSV contains no result rows:\n"
            f"{csv_path.resolve()}"
        )

    return (
        list(
            fieldnames
        ),
        rows,
    )


# ============================================================
# Detailed-result combination
# ============================================================

def combine_case_results(
    case_output_paths,
):
    combined_fieldnames = None
    combined_rows = []

    for case_output_path in (
        case_output_paths
    ):
        fieldnames, rows = read_csv_rows(
            case_output_path
        )

        if combined_fieldnames is None:
            combined_fieldnames = list(
                fieldnames
            )

        elif list(
            fieldnames
        ) != combined_fieldnames:
            raise RuntimeError(
                "CSV columns do not match between "
                "blocking-oxide cases:\n"
                f"{case_output_path.resolve()}"
            )

        combined_rows.extend(
            rows
        )

    if combined_fieldnames is None:
        raise RuntimeError(
            "No CSV column names were collected."
        )

    if not combined_rows:
        raise RuntimeError(
            "No detailed result rows were collected."
        )

    with COMBINED_OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=combined_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            combined_rows
        )

    return (
        combined_fieldnames,
        combined_rows,
    )


# ============================================================
# CSV-value utilities
# ============================================================

def get_first_available_value(
    row,
    candidate_column_names,
    default_value="",
):
    for column_name in (
        candidate_column_names
    ):
        value = row.get(
            column_name,
            "",
        )

        if value not in (
            None,
            "",
        ):
            return value

    return default_value


def get_required_float(
    row,
    candidate_column_names,
    quantity_name,
):
    value = get_first_available_value(
        row=row,
        candidate_column_names=(
            candidate_column_names
        ),
        default_value="",
    )

    if value == "":
        raise RuntimeError(
            f"Could not find {quantity_name}. "
            "Candidate CSV columns:\n"
            f"{candidate_column_names}"
        )

    numeric_value = float(
        value
    )

    if not math.isfinite(
        numeric_value
    ):
        raise RuntimeError(
            f"{quantity_name} is not finite. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def select_single_region_row(
    rows,
    region_name,
):
    matching_rows = [
        row
        for row in rows
        if row.get(
            "region",
            "",
        ).strip() == region_name
    ]

    if len(
        matching_rows
    ) != 1:
        raise RuntimeError(
            f'Expected exactly one "{region_name}" row, '
            f"but found {len(matching_rows)}."
        )

    return matching_rows[0]


# ============================================================
# Program-time estimate
# ============================================================

def calculate_fixed_field_program_time_s(
    current_density_A_cm2,
):
    current_density_A_cm2 = (
        validate_nonnegative(
            name="Tunneling current density",
            value=current_density_A_cm2,
        )
    )

    if current_density_A_cm2 == 0.0:
        return math.inf

    target_charge_density_C_cm2 = (
        ELEMENTARY_CHARGE_C
        * TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
    )

    return (
        target_charge_density_C_cm2
        / current_density_A_cm2
    )


def format_finite_or_infinity(
    value,
):
    if math.isinf(
        value
    ):
        return "inf"

    return f"{float(value):.12e}"


# ============================================================
# Summary generation
# ============================================================

def build_summary_rows(
    case_output_paths,
):
    summary_rows = []

    for case_output_path in (
        case_output_paths
    ):
        _, rows = read_csv_rows(
            case_output_path
        )

        tunnel_row = (
            select_single_region_row(
                rows=rows,
                region_name=(
                    TUNNEL_OXIDE_REGION
                ),
            )
        )

        blocking_row = (
            select_single_region_row(
                rows=rows,
                region_name=(
                    BLOCKING_OXIDE_REGION
                ),
            )
        )

        tunnel_current_density_A_cm2 = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    TUNNEL_CURRENT_DENSITY_COLUMNS
                ),
                quantity_name=(
                    "TunnelOxide tunneling "
                    "current density"
                ),
            )
        )

        tunnel_average_field_V_cm = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    TUNNEL_AVERAGE_FIELD_COLUMNS
                ),
                quantity_name=(
                    "TunnelOxide average field"
                ),
            )
        )

        tunnel_maximum_field_V_cm = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    TUNNEL_MAXIMUM_FIELD_COLUMNS
                ),
                quantity_name=(
                    "TunnelOxide maximum field"
                ),
            )
        )

        blocking_average_field_V_cm = (
            get_required_float(
                row=blocking_row,
                candidate_column_names=(
                    BLOCKING_AVERAGE_FIELD_COLUMNS
                ),
                quantity_name=(
                    "BlockingOxide average field"
                ),
            )
        )

        blocking_maximum_field_V_cm = (
            get_required_float(
                row=blocking_row,
                candidate_column_names=(
                    BLOCKING_MAXIMUM_FIELD_COLUMNS
                ),
                quantity_name=(
                    "BlockingOxide maximum field"
                ),
            )
        )

        program_time_s = (
            calculate_fixed_field_program_time_s(
                current_density_A_cm2=(
                    tunnel_current_density_A_cm2
                ),
            )
        )

        blocking_oxide_thickness_nm = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    (
                        "blocking_oxide_"
                        "thickness_nm"
                    ),
                ),
                quantity_name=(
                    "BlockingOxide thickness"
                ),
            )
        )

        charge_trap_thickness_nm = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    "charge_trap_thickness_nm",
                ),
                quantity_name=(
                    "ChargeTrap thickness"
                ),
            )
        )

        tunnel_oxide_thickness_nm = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    "tunnel_oxide_thickness_nm",
                ),
                quantity_name=(
                    "TunnelOxide thickness"
                ),
            )
        )

        gate_voltage_V = (
            get_required_float(
                row=tunnel_row,
                candidate_column_names=(
                    "gate_voltage_V",
                ),
                quantity_name=(
                    "Gate voltage"
                ),
            )
        )

        summary_rows.append(
            {
                "tunnel_oxide_thickness_nm": (
                    tunnel_oxide_thickness_nm
                ),
                "charge_trap_thickness_nm": (
                    charge_trap_thickness_nm
                ),
                "blocking_oxide_thickness_nm": (
                    blocking_oxide_thickness_nm
                ),
                "gate_voltage_V": (
                    gate_voltage_V
                ),
                "target_sheet_density_cm2": (
                    TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
                ),
                "tunnel_average_field_abs_V_cm": (
                    tunnel_average_field_V_cm
                ),
                "tunnel_maximum_field_abs_V_cm": (
                    tunnel_maximum_field_V_cm
                ),
                "blocking_average_field_abs_V_cm": (
                    blocking_average_field_V_cm
                ),
                "blocking_maximum_field_abs_V_cm": (
                    blocking_maximum_field_V_cm
                ),
                "tunnel_to_blocking_average_field_ratio": (
                    tunnel_average_field_V_cm
                    / blocking_average_field_V_cm
                    if blocking_average_field_V_cm > 0.0
                    else math.inf
                ),
                "tunnel_to_blocking_maximum_field_ratio": (
                    tunnel_maximum_field_V_cm
                    / blocking_maximum_field_V_cm
                    if blocking_maximum_field_V_cm > 0.0
                    else math.inf
                ),
                "tunneling_current_density_A_cm2": (
                    tunnel_current_density_A_cm2
                ),
                "fixed_field_program_time_s": (
                    format_finite_or_infinity(
                        program_time_s
                    )
                ),
                "fixed_field_assumption": (
                    True
                ),
                "capture_efficiency_assumed": (
                    1.0
                ),
                "source_csv": (
                    str(
                        Path(
                            case_output_path
                        ).resolve()
                    )
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: float(
            row[
                "blocking_oxide_thickness_nm"
            ]
        )
    )

    return summary_rows


# ============================================================
# Summary and ranking CSV output
# ============================================================

SUMMARY_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "charge_trap_thickness_nm",
    "blocking_oxide_thickness_nm",
    "gate_voltage_V",
    "target_sheet_density_cm2",
    "tunnel_average_field_abs_V_cm",
    "tunnel_maximum_field_abs_V_cm",
    "blocking_average_field_abs_V_cm",
    "blocking_maximum_field_abs_V_cm",
    "tunnel_to_blocking_average_field_ratio",
    "tunnel_to_blocking_maximum_field_ratio",
    "tunneling_current_density_A_cm2",
    "fixed_field_program_time_s",
    "fixed_field_assumption",
    "capture_efficiency_assumed",
    "source_csv",
)


def write_summary_csv(
    summary_rows,
):
    with SUMMARY_OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                SUMMARY_FIELDNAMES
            ),
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )


def program_time_sort_value(
    row,
):
    value = row[
        "fixed_field_program_time_s"
    ]

    if value == "inf":
        return math.inf

    return float(
        value
    )


def write_ranking_csv(
    summary_rows,
):
    ranked_rows = sorted(
        summary_rows,
        key=program_time_sort_value,
    )

    ranking_rows = []

    for rank, row in enumerate(
        ranked_rows,
        start=1,
    ):
        ranking_rows.append(
            {
                "program_speed_rank": (
                    rank
                ),
                **row,
            }
        )

    with RANKING_OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                (
                    "program_speed_rank",
                )
                + SUMMARY_FIELDNAMES
            ),
        )

        writer.writeheader()

        writer.writerows(
            ranking_rows
        )

    return ranking_rows


# ============================================================
# Console reporting
# ============================================================

def float_or_nan(
    value,
):
    if value in (
        None,
        "",
    ):
        return float(
            "nan"
        )

    return float(
        value
    )


def print_summary(
    summary_rows,
):
    print_section(
        "BLOCKING-OXIDE SCREENING SUMMARY"
    )

    for row in (
        summary_rows
    ):
        program_time_text = row[
            "fixed_field_program_time_s"
        ]

        if program_time_text == "inf":
            program_time_display = (
                "inf"
            )

        else:
            program_time_display = (
                f"{float(program_time_text):.6e}"
            )

        print(
            "tblock="
            f"{float_or_nan(row['blocking_oxide_thickness_nm']):5.1f} nm, "
            "Etunnel_avg="
            f"{float_or_nan(row['tunnel_average_field_abs_V_cm']):.6e} V/cm, "
            "Etunnel_max="
            f"{float_or_nan(row['tunnel_maximum_field_abs_V_cm']):.6e} V/cm, "
            "Eblock_avg="
            f"{float_or_nan(row['blocking_average_field_abs_V_cm']):.6e} V/cm, "
            "Eblock_max="
            f"{float_or_nan(row['blocking_maximum_field_abs_V_cm']):.6e} V/cm, "
            "J="
            f"{float_or_nan(row['tunneling_current_density_A_cm2']):.6e} A/cm^2, "
            "t_target="
            f"{program_time_display} s"
        )


def print_program_speed_ranking(
    ranking_rows,
):
    print_section(
        "FIXED-FIELD PROGRAM-SPEED RANKING"
    )

    for row in (
        ranking_rows
    ):
        print(
            f"{int(row['program_speed_rank']):2d}. "
            "BlockingOxide="
            f"{float(row['blocking_oxide_thickness_nm']):.3f} nm, "
            "J="
            f"{float(row['tunneling_current_density_A_cm2']):.6e} A/cm^2, "
            "time="
            f"{row['fixed_field_program_time_s']} s"
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

    INDIVIDUAL_CASE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section(
        "BLOCKING-OXIDE THICKNESS SCREENING AT 12 V"
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
        "BlockingOxide thicknesses: "
        f"{BLOCKING_OXIDE_THICKNESSES_NM}"
    )

    print(
        "Gate voltage: "
        f"{GATE_VOLTAGE_V:.3f} V"
    )

    print(
        "Target trapped-electron sheet density: "
        f"{TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2:.6e} "
        "cm^-2"
    )

    print(
        "Existing positive-field script:\n"
        f"{POSITIVE_FIELD_SWEEP_SCRIPT.resolve()}"
    )

    print(
        "Result directory:\n"
        f"{RESULTS_DIRECTORY.resolve()}"
    )

    case_output_paths = []

    for blocking_oxide_thickness_nm in (
        BLOCKING_OXIDE_THICKNESSES_NM
    ):
        case_output_path = (
            run_single_blocking_oxide_case(
                blocking_oxide_thickness_nm=(
                    blocking_oxide_thickness_nm
                ),
            )
        )

        case_output_paths.append(
            case_output_path
        )

    print_section(
        "COMBINE DETAILED SCREENING RESULTS"
    )

    combine_case_results(
        case_output_paths=(
            case_output_paths
        ),
    )

    print(
        "Detailed case results combined."
    )

    summary_rows = build_summary_rows(
        case_output_paths=(
            case_output_paths
        ),
    )

    write_summary_csv(
        summary_rows=(
            summary_rows
        ),
    )

    ranking_rows = write_ranking_csv(
        summary_rows=(
            summary_rows
        ),
    )

    print_summary(
        summary_rows=(
            summary_rows
        ),
    )

    print_program_speed_ranking(
        ranking_rows=(
            ranking_rows
        ),
    )

    print_section(
        "BLOCKING-OXIDE SCREENING COMPLETED"
    )

    print(
        "Individual result count: "
        f"{len(case_output_paths)}"
    )

    print(
        "Combined detailed CSV:\n"
        f"{COMBINED_OUTPUT_CSV.resolve()}"
    )

    print(
        "Summary CSV:\n"
        f"{SUMMARY_OUTPUT_CSV.resolve()}"
    )

    print(
        "Program-speed ranking CSV:\n"
        f"{RANKING_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Interpretation guidance:"
    )

    print(
        "1. This first screening compares every blocking-oxide "
        "thickness at the same 12 V gate bias."
    )

    print(
        "2. A longer blocking oxide can reduce the tunnel-oxide "
        "field and therefore reduce channel-side FN tunneling."
    )

    print(
        "3. The fastest condition is not automatically the best "
        "memory structure."
    )

    print(
        "4. After this screening, gate voltage should be adjusted "
        "only for blocking-oxide candidates that remain viable."
    )

    print(
        "5. Final candidates must be re-evaluated with trapped-"
        "charge feedback and threshold-voltage shift."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_BLOCKING_OXIDE_SCREENING SCRIPT STARTED"
    )

    main()