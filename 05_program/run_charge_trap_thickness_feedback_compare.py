# ============================================================
# Charge-trap thickness program-feedback comparison
#
# Fixed conditions
# ----------------
# TunnelOxide thickness   = 3.0 nm
# BlockingOxide thickness = 16.0 nm
# Program gate voltage    = 16.0 V
# Drain voltage            = 0.05 V
#
# ChargeTrap thickness sweep
# --------------------------
# 3, 5, 7, 10 nm
#
# Comparison condition
# --------------------
# All structures are compared at the same final trapped-electron
# sheet density:
#
#     N_sheet,target = 1.0e12 cm^-2
#
# For each ChargeTrap thickness:
#
#     N_volume = N_sheet / t_trap
#
# Existing scripts reused
# -----------------------
# run_program_feedback_compare.py
# run_positive_field_sweep.py
#
# Important correction
# --------------------
# The static-point result dictionary returned to the existing
# run_program_feedback_compare.py does NOT contain
# blocking_oxide_thickness_nm.
#
# This prevents the CSV error:
#
# ValueError: dict contains fields not in fieldnames:
# 'blocking_oxide_thickness_nm'
#
# BlockingOxide thickness is added later by this outer script
# when combined result CSV files are generated.
# ============================================================

import csv
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# Fixed device conditions
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = 3.0

BLOCKING_OXIDE_THICKNESS_NM = 16.0

PROGRAM_GATE_VOLTAGE_V = 16.0

DRAIN_VOLTAGE_V = 0.05


# ============================================================
# Charge-trap thickness sweep
# ============================================================

CHARGE_TRAP_THICKNESSES_NM = (
    3.0,
    5.0,
    7.0,
    10.0,
)


# ============================================================
# Program-feedback conditions
# ============================================================

TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2 = 1.0e12

NUMBER_OF_SHEET_DENSITY_INTERVALS = 20

CAPTURE_EFFICIENCY = 1.0


# ============================================================
# Execution options
# ============================================================

# True:
#   Existing result folder is deleted before simulation.
#
# False:
#   Existing files are overwritten where possible.
#
# Using True is recommended after a failed execution.
DELETE_EXISTING_RESULTS = True


# ============================================================
# Paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

FEEDBACK_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_program_feedback_compare.py"
)

POSITIVE_FIELD_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "charge_trap_thickness_feedback_compare"
)

CASE_RESULTS_DIRECTORY = (
    RESULTS_DIRECTORY
    / "case_results"
)

COMBINED_COMPARISON_CSV = (
    RESULTS_DIRECTORY
    / "charge_trap_feedback_comparison.csv"
)

COMBINED_STATIC_POINTS_CSV = (
    RESULTS_DIRECTORY
    / "charge_trap_feedback_static_points.csv"
)

COMBINED_TIME_INTEGRATION_CSV = (
    RESULTS_DIRECTORY
    / "charge_trap_feedback_time_integration.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "charge_trap_feedback_summary.csv"
)

RANKING_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "charge_trap_feedback_ranking.csv"
)


# ============================================================
# Console utilities
# ============================================================

def print_section(title):
    print()
    print("=" * 108)
    print(title)
    print("=" * 108)


def print_separator():
    print()
    print("-" * 108)


# ============================================================
# Validation utilities
# ============================================================

def validate_finite(name, value):
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_positive(name, value):
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


def validate_nonnegative(name, value):
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
    if not FEEDBACK_SCRIPT.exists():
        raise FileNotFoundError(
            "run_program_feedback_compare.py was not found:\n"
            f"{FEEDBACK_SCRIPT.resolve()}"
        )

    if not POSITIVE_FIELD_SCRIPT.exists():
        raise FileNotFoundError(
            "run_positive_field_sweep.py was not found:\n"
            f"{POSITIVE_FIELD_SCRIPT.resolve()}"
        )

    validate_positive(
        name="TUNNEL_OXIDE_THICKNESS_NM",
        value=TUNNEL_OXIDE_THICKNESS_NM,
    )

    validate_positive(
        name="BLOCKING_OXIDE_THICKNESS_NM",
        value=BLOCKING_OXIDE_THICKNESS_NM,
    )

    validate_nonnegative(
        name="PROGRAM_GATE_VOLTAGE_V",
        value=PROGRAM_GATE_VOLTAGE_V,
    )

    validate_nonnegative(
        name="DRAIN_VOLTAGE_V",
        value=DRAIN_VOLTAGE_V,
    )

    validate_positive(
        name="TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2",
        value=TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2,
    )

    validate_positive(
        name="NUMBER_OF_SHEET_DENSITY_INTERVALS",
        value=NUMBER_OF_SHEET_DENSITY_INTERVALS,
    )

    validate_positive(
        name="CAPTURE_EFFICIENCY",
        value=CAPTURE_EFFICIENCY,
    )

    if CAPTURE_EFFICIENCY > 1.0:
        raise ValueError(
            "CAPTURE_EFFICIENCY must not exceed 1.0."
        )

    if not CHARGE_TRAP_THICKNESSES_NM:
        raise ValueError(
            "CHARGE_TRAP_THICKNESSES_NM must not be empty."
        )

    previous_thickness_nm = None

    for thickness_nm in CHARGE_TRAP_THICKNESSES_NM:
        validate_positive(
            name="ChargeTrap thickness",
            value=thickness_nm,
        )

        if (
            previous_thickness_nm is not None
            and thickness_nm <= previous_thickness_nm
        ):
            raise ValueError(
                "CHARGE_TRAP_THICKNESSES_NM must be "
                "strictly increasing."
            )

        previous_thickness_nm = thickness_nm


# ============================================================
# Density conversion
# ============================================================

def charge_trap_thickness_cm(
    charge_trap_thickness_nm,
):
    return (
        float(charge_trap_thickness_nm)
        * 1.0e-7
    )


def sheet_to_volume_density_cm3(
    sheet_density_cm2,
    charge_trap_thickness_nm,
):
    thickness_cm = charge_trap_thickness_cm(
        charge_trap_thickness_nm
    )

    if thickness_cm <= 0.0:
        raise ValueError(
            "ChargeTrap thickness must be positive."
        )

    return (
        float(sheet_density_cm2)
        / thickness_cm
    )


# ============================================================
# Case definitions
# ============================================================

def number_to_token(value):
    return (
        f"{float(value):.1f}"
        .replace(".", "p")
        .replace("-", "m")
    )


def build_cases():
    cases = []

    for charge_trap_thickness_nm in (
        CHARGE_TRAP_THICKNESSES_NM
    ):
        target_volume_density_cm3 = (
            sheet_to_volume_density_cm3(
                sheet_density_cm2=(
                    TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
                ),
                charge_trap_thickness_nm=(
                    charge_trap_thickness_nm
                ),
            )
        )

        cases.append(
            {
                "case_name": (
                    "charge_trap_"
                    f"{number_to_token(charge_trap_thickness_nm)}"
                    "nm"
                ),
                "case_label": (
                    f"ChargeTrap "
                    f"{charge_trap_thickness_nm:.1f} nm"
                ),
                "charge_trap_thickness_nm": (
                    charge_trap_thickness_nm
                ),
                "target_volume_density_cm3": (
                    target_volume_density_cm3
                ),
            }
        )

    return tuple(cases)


CHARGE_TRAP_CASES = build_cases()


# ============================================================
# Case output paths
# ============================================================

def get_case_directory(case):
    return (
        CASE_RESULTS_DIRECTORY
        / case["case_name"]
    )


def get_case_static_points_csv(case):
    return (
        get_case_directory(case)
        / "program_feedback_static_points.csv"
    )


def get_case_integration_csv(case):
    return (
        get_case_directory(case)
        / "program_feedback_time_integration.csv"
    )


def get_case_comparison_csv(case):
    return (
        get_case_directory(case)
        / "program_feedback_comparison.csv"
    )


# ============================================================
# Subprocess wrapper
#
# This string is executed in a separate Python process for
# each ChargeTrap thickness.
# ============================================================

def build_case_wrapper_code():
    return r'''
import os
import subprocess
import sys
from pathlib import Path

import run_program_feedback_compare as feedback


# ============================================================
# Read settings from environment variables
# ============================================================

tunnel_oxide_thickness_nm = float(
    os.environ[
        "CASE_TUNNEL_OXIDE_THICKNESS_NM"
    ]
)

charge_trap_thickness_nm = float(
    os.environ[
        "CASE_CHARGE_TRAP_THICKNESS_NM"
    ]
)

blocking_oxide_thickness_nm = float(
    os.environ[
        "CASE_BLOCKING_OXIDE_THICKNESS_NM"
    ]
)

program_gate_voltage_V = float(
    os.environ[
        "CASE_PROGRAM_GATE_VOLTAGE_V"
    ]
)

drain_voltage_V = float(
    os.environ[
        "CASE_DRAIN_VOLTAGE_V"
    ]
)

target_sheet_density_cm2 = float(
    os.environ[
        "CASE_TARGET_SHEET_DENSITY_CM2"
    ]
)

number_of_intervals = int(
    os.environ[
        "CASE_NUMBER_OF_INTERVALS"
    ]
)

capture_efficiency = float(
    os.environ[
        "CASE_CAPTURE_EFFICIENCY"
    ]
)

case_result_directory = Path(
    os.environ[
        "CASE_RESULT_DIRECTORY"
    ]
).resolve()


# ============================================================
# Override existing feedback-script conditions
# ============================================================

feedback.TUNNEL_OXIDE_THICKNESS_NM = (
    tunnel_oxide_thickness_nm
)

feedback.CHARGE_TRAP_THICKNESS_NM = (
    charge_trap_thickness_nm
)

feedback.PROGRAM_GATE_VOLTAGES_V = (
    program_gate_voltage_V,
)

feedback.TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2 = (
    target_sheet_density_cm2
)

feedback.NUMBER_OF_SHEET_DENSITY_INTERVALS = (
    number_of_intervals
)

feedback.CAPTURE_EFFICIENCY = (
    capture_efficiency
)

feedback.DRAIN_VOLTAGE_V = (
    drain_voltage_V
)

feedback.RESULTS_DIRECTORY = (
    case_result_directory
)

feedback.STATIC_CASE_DIRECTORY = (
    case_result_directory
    / "static_cases"
)

feedback.STATIC_POINTS_OUTPUT_CSV = (
    case_result_directory
    / "program_feedback_static_points.csv"
)

feedback.INTEGRATION_OUTPUT_CSV = (
    case_result_directory
    / "program_feedback_time_integration.csv"
)

feedback.COMPARISON_OUTPUT_CSV = (
    case_result_directory
    / "program_feedback_comparison.csv"
)


# ============================================================
# Positive-field solver wrapper
#
# This wrapper injects all three dielectric thicknesses into
# the parameterized create_structure() function.
# ============================================================

positive_field_wrapper_code = r"""
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
"""


# ============================================================
# Charge-density conversion
#
# Target sheet density remains identical for every
# ChargeTrap thickness.
# ============================================================

def configured_sheet_to_volume_density_cm3(
    sheet_density_cm2,
):
    thickness_cm = (
        charge_trap_thickness_nm
        * 1.0e-7
    )

    if thickness_cm <= 0.0:
        raise ValueError(
            "ChargeTrap thickness must be positive."
        )

    return (
        float(sheet_density_cm2)
        / thickness_cm
    )


feedback.sheet_to_volume_density_cm3 = (
    configured_sheet_to_volume_density_cm3
)


# ============================================================
# Replacement static feedback-point simulation
# ============================================================

def configured_run_static_feedback_point(
    gate_voltage_V,
    sheet_density_cm2,
):
    volume_density_cm3 = (
        configured_sheet_to_volume_density_cm3(
            sheet_density_cm2
        )
    )

    output_csv = (
        feedback.get_static_case_output_path(
            gate_voltage_V=gate_voltage_V,
            sheet_density_cm2=sheet_density_cm2,
        )
    )

    environment = os.environ.copy()

    environment[
        "TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        tunnel_oxide_thickness_nm
    )

    environment[
        "CHARGE_TRAP_THICKNESS_NM"
    ] = str(
        charge_trap_thickness_nm
    )

    environment[
        "BLOCKING_OXIDE_THICKNESS_NM"
    ] = str(
        blocking_oxide_thickness_nm
    )

    environment[
        "GATE_VOLTAGES_V"
    ] = str(
        gate_voltage_V
    )

    environment[
        "DRAIN_VOLTAGE_V"
    ] = str(
        drain_voltage_V
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
    print("-" * 100)
    print("STATIC CHARGE-TRAP FEEDBACK POINT")

    print(
        "  TunnelOxide thickness   : "
        f"{tunnel_oxide_thickness_nm:.3f} nm"
    )

    print(
        "  ChargeTrap thickness    : "
        f"{charge_trap_thickness_nm:.3f} nm"
    )

    print(
        "  BlockingOxide thickness : "
        f"{blocking_oxide_thickness_nm:.3f} nm"
    )

    print(
        "  Gate voltage            : "
        f"{gate_voltage_V:.3f} V"
    )

    print(
        "  Trapped sheet density   : "
        f"{sheet_density_cm2:.6e} cm^-2"
    )

    print(
        "  Trapped volume density  : "
        f"{volume_density_cm3:.6e} cm^-3"
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            positive_field_wrapper_code,
        ],
        cwd=str(
            feedback.SCRIPT_DIRECTORY
        ),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Configured static feedback simulation failed.\n"
            "TunnelOxide thickness: "
            f"{tunnel_oxide_thickness_nm:.3f} nm\n"
            "ChargeTrap thickness: "
            f"{charge_trap_thickness_nm:.3f} nm\n"
            "BlockingOxide thickness: "
            f"{blocking_oxide_thickness_nm:.3f} nm\n"
            "Gate voltage: "
            f"{gate_voltage_V:.3f} V\n"
            "Trapped sheet density: "
            f"{sheet_density_cm2:.6e} cm^-2\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "Static-point output CSV was not created:\n"
            f"{output_csv.resolve()}"
        )

    rows = feedback.read_csv_rows(
        output_csv
    )

    tunnel_oxide_row = (
        feedback.select_tunnel_oxide_row(
            rows
        )
    )

    current_density_A_cm2 = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                feedback.CURRENT_DENSITY_COLUMNS
            ),
        )
    )

    average_field_V_cm = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                feedback.AVERAGE_FIELD_COLUMNS
            ),
        )
    )

    maximum_field_V_cm = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                feedback.MAXIMUM_FIELD_COLUMNS
            ),
        )
    )

    actual_tunnel_thickness_nm = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                "tunnel_oxide_thickness_nm",
            ),
        )
    )

    actual_trap_thickness_nm = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                "charge_trap_thickness_nm",
            ),
        )
    )

    actual_blocking_thickness_nm = (
        feedback.get_first_available_float(
            row=tunnel_oxide_row,
            candidate_columns=(
                "blocking_oxide_thickness_nm",
            ),
        )
    )

    if abs(
        actual_tunnel_thickness_nm
        - tunnel_oxide_thickness_nm
    ) > 1.0e-9:
        raise RuntimeError(
            "TunnelOxide thickness mismatch.\n"
            "Requested: "
            f"{tunnel_oxide_thickness_nm:.6f} nm\n"
            "Simulation output: "
            f"{actual_tunnel_thickness_nm:.6f} nm"
        )

    if abs(
        actual_trap_thickness_nm
        - charge_trap_thickness_nm
    ) > 1.0e-9:
        raise RuntimeError(
            "ChargeTrap thickness mismatch.\n"
            "Requested: "
            f"{charge_trap_thickness_nm:.6f} nm\n"
            "Simulation output: "
            f"{actual_trap_thickness_nm:.6f} nm"
        )

    if abs(
        actual_blocking_thickness_nm
        - blocking_oxide_thickness_nm
    ) > 1.0e-9:
        raise RuntimeError(
            "BlockingOxide thickness mismatch.\n"
            "Requested: "
            f"{blocking_oxide_thickness_nm:.6f} nm\n"
            "Simulation output: "
            f"{actual_blocking_thickness_nm:.6f} nm"
        )

    if current_density_A_cm2 < 0.0:
        raise RuntimeError(
            "Tunneling-current density must not be negative."
        )

    # ========================================================
    # IMPORTANT
    #
    # Do not add the following key here:
    #
    #     blocking_oxide_thickness_nm
    #
    # The original run_program_feedback_compare.py static-point
    # CSV schema does not include that field.
    #
    # BlockingOxide thickness is added by the outer script when
    # final combined CSV files are written.
    # ========================================================

    return {
        "tunnel_oxide_thickness_nm": (
            tunnel_oxide_thickness_nm
        ),
        "charge_trap_thickness_nm": (
            charge_trap_thickness_nm
        ),
        "gate_voltage_V": (
            gate_voltage_V
        ),
        "drain_voltage_V": (
            tunnel_oxide_row.get(
                "drain_voltage_V",
                drain_voltage_V,
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


feedback.run_static_feedback_point = (
    configured_run_static_feedback_point
)


# ============================================================
# Execute configured feedback simulation
# ============================================================

feedback.main()
'''


# ============================================================
# Run one ChargeTrap thickness case
# ============================================================

def run_case(case):
    case_directory = get_case_directory(
        case
    )

    case_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    environment = os.environ.copy()

    environment[
        "CASE_TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        float(TUNNEL_OXIDE_THICKNESS_NM)
    )

    environment[
        "CASE_CHARGE_TRAP_THICKNESS_NM"
    ] = str(
        float(
            case[
                "charge_trap_thickness_nm"
            ]
        )
    )

    environment[
        "CASE_BLOCKING_OXIDE_THICKNESS_NM"
    ] = str(
        float(BLOCKING_OXIDE_THICKNESS_NM)
    )

    environment[
        "CASE_PROGRAM_GATE_VOLTAGE_V"
    ] = str(
        float(PROGRAM_GATE_VOLTAGE_V)
    )

    environment[
        "CASE_DRAIN_VOLTAGE_V"
    ] = str(
        float(DRAIN_VOLTAGE_V)
    )

    environment[
        "CASE_TARGET_SHEET_DENSITY_CM2"
    ] = str(
        float(
            TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
        )
    )

    environment[
        "CASE_NUMBER_OF_INTERVALS"
    ] = str(
        int(NUMBER_OF_SHEET_DENSITY_INTERVALS)
    )

    environment[
        "CASE_CAPTURE_EFFICIENCY"
    ] = str(
        float(CAPTURE_EFFICIENCY)
    )

    environment[
        "CASE_RESULT_DIRECTORY"
    ] = str(
        case_directory.resolve()
    )

    print_separator()

    print(
        "STARTING CHARGE-TRAP THICKNESS CASE"
    )

    print(
        "  Case                    : "
        f"{case['case_label']}"
    )

    print(
        "  TunnelOxide thickness   : "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "  ChargeTrap thickness    : "
        f"{case['charge_trap_thickness_nm']:.3f} nm"
    )

    print(
        "  BlockingOxide thickness : "
        f"{BLOCKING_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "  Program gate voltage    : "
        f"{PROGRAM_GATE_VOLTAGE_V:.3f} V"
    )

    print(
        "  Target sheet density    : "
        f"{TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2:.6e} "
        "cm^-2"
    )

    print(
        "  Target volume density   : "
        f"{case['target_volume_density_cm3']:.6e} "
        "cm^-3"
    )

    print(
        "  Result directory:\n"
        f"  {case_directory.resolve()}"
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            "-c",
            build_case_wrapper_code(),
        ],
        cwd=str(
            SCRIPT_DIRECTORY
        ),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "ChargeTrap thickness feedback case failed.\n"
            f"Case: {case['case_label']}\n"
            "ChargeTrap thickness: "
            f"{case['charge_trap_thickness_nm']:.3f} nm\n"
            "BlockingOxide thickness: "
            f"{BLOCKING_OXIDE_THICKNESS_NM:.3f} nm\n"
            "Program voltage: "
            f"{PROGRAM_GATE_VOLTAGE_V:.3f} V\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    required_outputs = (
        get_case_static_points_csv(case),
        get_case_integration_csv(case),
        get_case_comparison_csv(case),
    )

    for output_path in required_outputs:
        if not output_path.exists():
            raise RuntimeError(
                "Case completed, but an expected output "
                "was not created:\n"
                f"{output_path.resolve()}"
            )

    print(
        "ChargeTrap case completed: "
        f"{case['case_label']}"
    )


# ============================================================
# CSV utilities
# ============================================================

def read_csv(csv_path):
    csv_path = Path(csv_path)

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
        rows = list(reader)

    if not fieldnames:
        raise RuntimeError(
            "CSV header was not found:\n"
            f"{csv_path.resolve()}"
        )

    if not rows:
        raise RuntimeError(
            "CSV contains no rows:\n"
            f"{csv_path.resolve()}"
        )

    return list(fieldnames), rows


def write_csv(
    output_path,
    fieldnames,
    rows,
):
    output_path = Path(output_path)

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
        writer.writerows(rows)


def ordered_union_fieldnames(rows):
    fieldnames = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    return fieldnames


# ============================================================
# Combine static-point and integration CSV files
# ============================================================

def combine_case_csv_files(
    source_path_function,
    output_path,
):
    combined_rows = []
    original_fieldnames = None

    metadata_fieldnames = (
        "case_name",
        "case_label",
        "blocking_oxide_thickness_nm",
        "program_gate_voltage_V",
        "target_sheet_density_cm2",
        "target_volume_density_cm3",
    )

    for case in CHARGE_TRAP_CASES:
        source_path = source_path_function(
            case
        )

        fieldnames, rows = read_csv(
            source_path
        )

        if original_fieldnames is None:
            original_fieldnames = fieldnames

        elif fieldnames != original_fieldnames:
            raise RuntimeError(
                "Case CSV columns do not match:\n"
                f"{source_path.resolve()}"
            )

        for row in rows:
            combined_rows.append(
                {
                    "case_name": (
                        case["case_name"]
                    ),
                    "case_label": (
                        case["case_label"]
                    ),
                    "blocking_oxide_thickness_nm": (
                        BLOCKING_OXIDE_THICKNESS_NM
                    ),
                    "program_gate_voltage_V": (
                        PROGRAM_GATE_VOLTAGE_V
                    ),
                    "target_sheet_density_cm2": (
                        TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
                    ),
                    "target_volume_density_cm3": (
                        case[
                            "target_volume_density_cm3"
                        ]
                    ),
                    **row,
                }
            )

    if original_fieldnames is None:
        raise RuntimeError(
            "No case CSV fields were collected."
        )

    duplicate_fields = set(
        metadata_fieldnames
    ).intersection(
        original_fieldnames
    )

    combined_fieldnames = list(
        metadata_fieldnames
    )

    for fieldname in original_fieldnames:
        if fieldname not in combined_fieldnames:
            combined_fieldnames.append(
                fieldname
            )

    write_csv(
        output_path=output_path,
        fieldnames=combined_fieldnames,
        rows=combined_rows,
    )

    return combined_rows


# ============================================================
# Build combined comparison rows
# ============================================================

def build_combined_comparison_rows():
    combined_rows = []

    for case in CHARGE_TRAP_CASES:
        comparison_path = (
            get_case_comparison_csv(case)
        )

        _, rows = read_csv(
            comparison_path
        )

        if len(rows) != 1:
            raise RuntimeError(
                "Expected exactly one comparison row for "
                f"{case['case_label']}, "
                f"but found {len(rows)}."
            )

        source_row = rows[0]

        combined_row = {
            "case_name": (
                case["case_name"]
            ),
            "case_label": (
                case["case_label"]
            ),
            "tunnel_oxide_thickness_nm": (
                TUNNEL_OXIDE_THICKNESS_NM
            ),
            "charge_trap_thickness_nm": (
                case[
                    "charge_trap_thickness_nm"
                ]
            ),
            "blocking_oxide_thickness_nm": (
                BLOCKING_OXIDE_THICKNESS_NM
            ),
            "program_gate_voltage_V": (
                PROGRAM_GATE_VOLTAGE_V
            ),
            "target_sheet_density_cm2": (
                TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
            ),
            "target_volume_density_cm3": (
                case[
                    "target_volume_density_cm3"
                ]
            ),
            "number_of_sheet_density_intervals": (
                NUMBER_OF_SHEET_DENSITY_INTERVALS
            ),
            "capture_efficiency": (
                CAPTURE_EFFICIENCY
            ),
        }

        for key, value in source_row.items():
            if key == "gate_voltage_V":
                continue

            if key not in combined_row:
                combined_row[key] = value

        combined_rows.append(
            combined_row
        )

    return combined_rows


# ============================================================
# Result extraction utilities
# ============================================================

def get_first_optional_float(
    row,
    candidate_columns,
):
    for column_name in candidate_columns:
        value = row.get(
            column_name,
            "",
        )

        if value not in (
            None,
            "",
        ):
            return float(value)

    return None


def get_required_feedback_time_s(row):
    value = get_first_optional_float(
        row=row,
        candidate_columns=(
            "feedback_program_time_s",
            "feedback_total_program_time_s",
            "feedback_time_s",
            "total_program_time_s",
        ),
    )

    if value is None:
        raise RuntimeError(
            "Could not find feedback program-time column.\n"
            f"Available columns: {tuple(row.keys())}"
        )

    if not math.isfinite(value):
        return math.inf

    return value


def get_optional_fixed_time_s(row):
    return get_first_optional_float(
        row=row,
        candidate_columns=(
            "fixed_field_program_time_s",
            "fixed_program_time_s",
        ),
    )


def get_optional_initial_current_density(row):
    return get_first_optional_float(
        row=row,
        candidate_columns=(
            "initial_tunneling_current_density_A_cm2",
            "initial_current_density_A_cm2",
        ),
    )


def get_optional_final_current_density(row):
    return get_first_optional_float(
        row=row,
        candidate_columns=(
            "final_tunneling_current_density_A_cm2",
            "final_current_density_A_cm2",
        ),
    )


def get_optional_initial_field(row):
    return get_first_optional_float(
        row=row,
        candidate_columns=(
            "initial_average_field_abs_V_cm",
            "initial_field_abs_V_cm",
        ),
    )


def get_optional_final_field(row):
    return get_first_optional_float(
        row=row,
        candidate_columns=(
            "final_average_field_abs_V_cm",
            "final_field_abs_V_cm",
        ),
    )


# ============================================================
# Derived metric utilities
# ============================================================

def safe_ratio(
    numerator,
    denominator,
):
    if numerator is None:
        return None

    if denominator is None:
        return None

    if denominator == 0.0:
        return math.inf

    return numerator / denominator


def relative_reduction(
    initial_value,
    final_value,
):
    if initial_value is None:
        return None

    if final_value is None:
        return None

    if initial_value == 0.0:
        return None

    return (
        initial_value
        - final_value
    ) / initial_value


# ============================================================
# Summary generation
# ============================================================

def build_summary_rows(
    comparison_rows,
):
    summary_rows = []

    for row in comparison_rows:
        feedback_time_s = (
            get_required_feedback_time_s(row)
        )

        fixed_time_s = (
            get_optional_fixed_time_s(row)
        )

        initial_current_density = (
            get_optional_initial_current_density(
                row
            )
        )

        final_current_density = (
            get_optional_final_current_density(
                row
            )
        )

        initial_field = (
            get_optional_initial_field(row)
        )

        final_field = (
            get_optional_final_field(row)
        )

        summary_rows.append(
            {
                "case_name": (
                    row["case_name"]
                ),
                "case_label": (
                    row["case_label"]
                ),
                "tunnel_oxide_thickness_nm": (
                    row[
                        "tunnel_oxide_thickness_nm"
                    ]
                ),
                "charge_trap_thickness_nm": (
                    row[
                        "charge_trap_thickness_nm"
                    ]
                ),
                "blocking_oxide_thickness_nm": (
                    row[
                        "blocking_oxide_thickness_nm"
                    ]
                ),
                "program_gate_voltage_V": (
                    row[
                        "program_gate_voltage_V"
                    ]
                ),
                "target_sheet_density_cm2": (
                    row[
                        "target_sheet_density_cm2"
                    ]
                ),
                "target_volume_density_cm3": (
                    row[
                        "target_volume_density_cm3"
                    ]
                ),
                "initial_average_field_abs_V_cm": (
                    initial_field
                ),
                "final_average_field_abs_V_cm": (
                    final_field
                ),
                "field_reduction_fraction": (
                    relative_reduction(
                        initial_value=initial_field,
                        final_value=final_field,
                    )
                ),
                "initial_tunneling_current_density_A_cm2": (
                    initial_current_density
                ),
                "final_tunneling_current_density_A_cm2": (
                    final_current_density
                ),
                "current_reduction_fraction": (
                    relative_reduction(
                        initial_value=(
                            initial_current_density
                        ),
                        final_value=(
                            final_current_density
                        ),
                    )
                ),
                "fixed_field_program_time_s": (
                    fixed_time_s
                ),
                "feedback_program_time_s": (
                    feedback_time_s
                ),
                "feedback_to_fixed_time_ratio": (
                    safe_ratio(
                        numerator=feedback_time_s,
                        denominator=fixed_time_s,
                    )
                ),
            }
        )

    summary_rows.sort(
        key=lambda row: float(
            row[
                "charge_trap_thickness_nm"
            ]
        )
    )

    return summary_rows


SUMMARY_FIELDNAMES = (
    "case_name",
    "case_label",
    "tunnel_oxide_thickness_nm",
    "charge_trap_thickness_nm",
    "blocking_oxide_thickness_nm",
    "program_gate_voltage_V",
    "target_sheet_density_cm2",
    "target_volume_density_cm3",
    "initial_average_field_abs_V_cm",
    "final_average_field_abs_V_cm",
    "field_reduction_fraction",
    "initial_tunneling_current_density_A_cm2",
    "final_tunneling_current_density_A_cm2",
    "current_reduction_fraction",
    "fixed_field_program_time_s",
    "feedback_program_time_s",
    "feedback_to_fixed_time_ratio",
)


# ============================================================
# Ranking generation
# ============================================================

def build_ranking_rows(
    comparison_rows,
):
    sorted_rows = sorted(
        comparison_rows,
        key=get_required_feedback_time_s,
    )

    ranking_rows = []

    for rank, row in enumerate(
        sorted_rows,
        start=1,
    ):
        ranking_rows.append(
            {
                "feedback_program_speed_rank": rank,
                **row,
            }
        )

    return ranking_rows


# ============================================================
# Console formatting
# ============================================================

def format_optional_scientific(value):
    if value is None:
        return "not_available"

    if math.isinf(value):
        return "inf"

    return f"{value:.6e}"


def format_optional_percent(value):
    if value is None:
        return "not_available"

    if math.isinf(value):
        return "inf"

    return f"{100.0 * value:.3f} %"


# ============================================================
# Console reporting
# ============================================================

def print_summary(summary_rows):
    print_section(
        "CHARGE-TRAP THICKNESS FEEDBACK SUMMARY"
    )

    for row in summary_rows:
        print()

        print(
            f"{row['case_label']}"
        )

        print(
            "  ChargeTrap thickness     : "
            f"{float(row['charge_trap_thickness_nm']):.3f} nm"
        )

        print(
            "  Target volume density    : "
            f"{float(row['target_volume_density_cm3']):.6e} "
            "cm^-3"
        )

        print(
            "  Initial tunnel field     : "
            f"{format_optional_scientific(row['initial_average_field_abs_V_cm'])} "
            "V/cm"
        )

        print(
            "  Final tunnel field       : "
            f"{format_optional_scientific(row['final_average_field_abs_V_cm'])} "
            "V/cm"
        )

        print(
            "  Field reduction          : "
            f"{format_optional_percent(row['field_reduction_fraction'])}"
        )

        print(
            "  Initial current density  : "
            f"{format_optional_scientific(row['initial_tunneling_current_density_A_cm2'])} "
            "A/cm^2"
        )

        print(
            "  Final current density    : "
            f"{format_optional_scientific(row['final_tunneling_current_density_A_cm2'])} "
            "A/cm^2"
        )

        print(
            "  Current reduction        : "
            f"{format_optional_percent(row['current_reduction_fraction'])}"
        )

        print(
            "  Fixed-field time         : "
            f"{format_optional_scientific(row['fixed_field_program_time_s'])} "
            "s"
        )

        print(
            "  Feedback program time    : "
            f"{format_optional_scientific(row['feedback_program_time_s'])} "
            "s"
        )

        print(
            "  Feedback / fixed ratio   : "
            f"{format_optional_scientific(row['feedback_to_fixed_time_ratio'])}"
        )


def print_ranking(ranking_rows):
    print_section(
        "CHARGE-TRAP FEEDBACK PROGRAM-SPEED RANKING"
    )

    for row in ranking_rows:
        feedback_time_s = (
            get_required_feedback_time_s(row)
        )

        print(
            f"{int(row['feedback_program_speed_rank']):2d}. "
            "ChargeTrap="
            f"{float(row['charge_trap_thickness_nm']):.1f} nm, "
            "target Nvolume="
            f"{float(row['target_volume_density_cm3']):.6e} cm^-3, "
            "feedback time="
            f"{feedback_time_s:.6e} s"
        )


# ============================================================
# Result-directory preparation
# ============================================================

def prepare_result_directory():
    if (
        DELETE_EXISTING_RESULTS
        and RESULTS_DIRECTORY.exists()
    ):
        print(
            "Deleting existing result directory:\n"
            f"{RESULTS_DIRECTORY.resolve()}"
        )

        shutil.rmtree(
            RESULTS_DIRECTORY
        )

    CASE_RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Main
# ============================================================

def main():
    validate_settings()

    prepare_result_directory()

    print_section(
        "CHARGE-TRAP THICKNESS PROGRAM-FEEDBACK COMPARISON"
    )

    print(
        "Fixed TunnelOxide thickness   : "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "Fixed BlockingOxide thickness : "
        f"{BLOCKING_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "Fixed program gate voltage    : "
        f"{PROGRAM_GATE_VOLTAGE_V:.3f} V"
    )

    print(
        "Drain voltage                 : "
        f"{DRAIN_VOLTAGE_V:.3f} V"
    )

    print(
        "Target trapped sheet density  : "
        f"{TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2:.6e} "
        "cm^-2"
    )

    print(
        "Sheet-density intervals       : "
        f"{NUMBER_OF_SHEET_DENSITY_INTERVALS}"
    )

    print(
        "Capture efficiency            : "
        f"{CAPTURE_EFFICIENCY:.6f}"
    )

    print()
    print("ChargeTrap cases:")

    for case in CHARGE_TRAP_CASES:
        print(
            "  ChargeTrap="
            f"{case['charge_trap_thickness_nm']:.1f} nm, "
            "target Nvolume="
            f"{case['target_volume_density_cm3']:.6e} cm^-3"
        )

    for case in CHARGE_TRAP_CASES:
        run_case(case)

    print_section(
        "COMBINING CHARGE-TRAP CASE RESULTS"
    )

    combine_case_csv_files(
        source_path_function=(
            get_case_static_points_csv
        ),
        output_path=(
            COMBINED_STATIC_POINTS_CSV
        ),
    )

    combine_case_csv_files(
        source_path_function=(
            get_case_integration_csv
        ),
        output_path=(
            COMBINED_TIME_INTEGRATION_CSV
        ),
    )

    comparison_rows = (
        build_combined_comparison_rows()
    )

    write_csv(
        output_path=COMBINED_COMPARISON_CSV,
        fieldnames=ordered_union_fieldnames(
            comparison_rows
        ),
        rows=comparison_rows,
    )

    summary_rows = build_summary_rows(
        comparison_rows
    )

    write_csv(
        output_path=SUMMARY_OUTPUT_CSV,
        fieldnames=SUMMARY_FIELDNAMES,
        rows=summary_rows,
    )

    ranking_rows = build_ranking_rows(
        comparison_rows
    )

    write_csv(
        output_path=RANKING_OUTPUT_CSV,
        fieldnames=ordered_union_fieldnames(
            ranking_rows
        ),
        rows=ranking_rows,
    )

    print_summary(
        summary_rows
    )

    print_ranking(
        ranking_rows
    )

    print_section(
        "CHARGE-TRAP THICKNESS FEEDBACK COMPARISON COMPLETED"
    )

    print(
        "Combined comparison CSV:\n"
        f"{COMBINED_COMPARISON_CSV.resolve()}"
    )

    print()
    print(
        "Summary CSV:\n"
        f"{SUMMARY_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Combined static-point CSV:\n"
        f"{COMBINED_STATIC_POINTS_CSV.resolve()}"
    )

    print()
    print(
        "Combined time-integration CSV:\n"
        f"{COMBINED_TIME_INTEGRATION_CSV.resolve()}"
    )

    print()
    print(
        "Program-speed ranking CSV:\n"
        f"{RANKING_OUTPUT_CSV.resolve()}"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_CHARGE_TRAP_THICKNESS_FEEDBACK_COMPARE "
        "SCRIPT STARTED"
    )

    main()