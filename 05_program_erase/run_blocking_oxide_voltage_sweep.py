# ============================================================
# Blocking-oxide thickness dependent gate-voltage sweep
#
# Fixed structure
# ---------------
# TunnelOxide thickness = 3.0 nm
# ChargeTrap thickness  = 5.0 nm
#
# Sweep conditions
# ----------------
# BlockingOxide = 8 nm
#   VG = 11, 12 V
#
# BlockingOxide = 12 nm
#   VG = 12, 13, 14 V
#
# BlockingOxide = 16 nm
#   VG = 14, 16, 18 V
#
# BlockingOxide = 20 nm
#   VG = 16, 18, 20, 22 V
#
# BlockingOxide = 30 nm
#   VG = 20, 22, 24, 26 V
#
# Purpose
# -------
# 1. Reuse run_positive_field_sweep.py
# 2. Apply geometry-specific gate-voltage lists
# 3. Run each blocking-oxide thickness independently
# 4. Combine detailed result CSV files
# 5. Extract field, tunneling-current, and program-time metrics
# 6. Rank conditions by fixed-field program time
#
# Important
# ---------
# Program time is estimated using:
#
#     t = q * N_target / J
#
# This screening does not yet include:
#   - trapped-charge feedback
#   - trap saturation
#   - capture efficiency below 100%
#   - retention
#   - gate-side tunneling
#   - oxide degradation
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
# Fixed structure
# ============================================================

TUNNEL_OXIDE_THICKNESS_NM = 3.0

CHARGE_TRAP_THICKNESS_NM = 5.0


# ============================================================
# Blocking-oxide-dependent voltage sweep
# ============================================================

BLOCKING_OXIDE_VOLTAGE_SWEEP = {
    8.0: (
        11.0,
        12.0,
    ),
    12.0: (
        12.0,
        13.0,
        14.0,
    ),
    16.0: (
        14.0,
        16.0,
        18.0,
    ),
    20.0: (
        16.0,
        18.0,
        20.0,
        22.0,
    ),
    30.0: (
        20.0,
        22.0,
        24.0,
        26.0,
    ),
}


# ============================================================
# Target charge
# ============================================================

TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2 = (
    1.0e12
)


# ============================================================
# Region names
# ============================================================

TUNNEL_OXIDE_REGION = "TunnelOxide"
BLOCKING_OXIDE_REGION = "BlockingOxide"


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIRECTORY = (
    Path(__file__)
    .resolve()
    .parent
)

POSITIVE_FIELD_SWEEP_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "blocking_oxide_voltage_sweep"
)

INDIVIDUAL_CASE_DIRECTORY = (
    RESULTS_DIRECTORY
    / "individual_cases"
)

COMBINED_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_voltage_sweep_combined.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_voltage_sweep_summary.csv"
)

RANKING_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_voltage_sweep_ranking.csv"
)

BEST_CONDITION_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "blocking_oxide_best_conditions.csv"
)


# ============================================================
# Candidate CSV columns
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
    print("=" * 104)
    print(title)
    print("=" * 104)


def print_case_separator():
    print()
    print("-" * 104)


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

    validate_positive(
        name=(
            "TARGET_TRAPPED_ELECTRON_"
            "SHEET_DENSITY_CM2"
        ),
        value=(
            TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2
        ),
    )

    if not BLOCKING_OXIDE_VOLTAGE_SWEEP:
        raise ValueError(
            "BLOCKING_OXIDE_VOLTAGE_SWEEP "
            "must not be empty."
        )

    previous_blocking_thickness_nm = None

    for (
        blocking_oxide_thickness_nm,
        gate_voltages_V,
    ) in BLOCKING_OXIDE_VOLTAGE_SWEEP.items():
        validate_positive(
            name="BlockingOxide thickness",
            value=blocking_oxide_thickness_nm,
        )

        if (
            previous_blocking_thickness_nm
            is not None
            and blocking_oxide_thickness_nm
            <= previous_blocking_thickness_nm
        ):
            raise ValueError(
                "BlockingOxide thickness keys must be "
                "strictly increasing."
            )

        previous_blocking_thickness_nm = (
            blocking_oxide_thickness_nm
        )

        if not gate_voltages_V:
            raise ValueError(
                "Each blocking-oxide thickness must "
                "have at least one gate voltage."
            )

        previous_gate_voltage_V = None

        for gate_voltage_V in gate_voltages_V:
            validate_nonnegative(
                name="Gate voltage",
                value=gate_voltage_V,
            )

            if (
                previous_gate_voltage_V
                is not None
                and gate_voltage_V
                <= previous_gate_voltage_V
            ):
                raise ValueError(
                    "Gate voltages for each blocking oxide "
                    "must be strictly increasing."
                )

            previous_gate_voltage_V = (
                gate_voltage_V
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

    return (
        INDIVIDUAL_CASE_DIRECTORY
        / (
            "blocking_voltage_sweep_"
            f"tox_{tunnel_token}nm_"
            f"trap_{trap_token}nm_"
            f"block_{blocking_token}nm.csv"
        )
    )


def gate_voltages_to_environment_text(
    gate_voltages_V,
):
    return ",".join(
        f"{float(voltage_V):g}"
        for voltage_V in gate_voltages_V
    )


# ============================================================
# Subprocess wrapper
#
# run_positive_field_sweep.py calls create_structure() with only
# tunnel_oxide_thickness_nm.
#
# This wrapper replaces that reference so that charge-trap and
# blocking-oxide thicknesses are also passed to the existing
# parameterized device_structure.py.
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
# Individual blocking-oxide sweep execution
# ============================================================

def run_single_blocking_oxide_case(
    blocking_oxide_thickness_nm,
    gate_voltages_V,
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
    ] = gate_voltages_to_environment_text(
        gate_voltages_V
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
        "STARTING BLOCKING-OXIDE VOLTAGE CASE"
    )

    print(
        "  TunnelOxide thickness  : "
        f"{TUNNEL_OXIDE_THICKNESS_NM:.3f} nm"
    )

    print(
        "  ChargeTrap thickness   : "
        f"{CHARGE_TRAP_THICKNESS_NM:.3f} nm"
    )

    print(
        "  BlockingOxide thickness: "
        f"{blocking_oxide_thickness_nm:.3f} nm"
    )

    print(
        "  Gate voltages          : "
        f"{gate_voltages_V}"
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
            "Blocking-oxide voltage sweep failed.\n"
            "BlockingOxide thickness: "
            f"{blocking_oxide_thickness_nm:.3f} nm\n"
            f"Gate voltages: {gate_voltages_V}\n"
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
        "Blocking-oxide voltage case completed: "
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

    for case_output_path in case_output_paths:
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
            "No CSV fieldnames were collected."
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

    return combined_rows


# ============================================================
# CSV-value utilities
# ============================================================

def get_first_available_value(
    row,
    candidate_column_names,
    default_value="",
):
    for column_name in candidate_column_names:
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
    gate_voltage_V,
):
    matching_rows = []

    for row in rows:
        row_region = row.get(
            "region",
            "",
        ).strip()

        if row_region != region_name:
            continue

        row_gate_voltage_V = float(
            row[
                "gate_voltage_V"
            ]
        )

        if abs(
            row_gate_voltage_V
            - gate_voltage_V
        ) <= 1.0e-9:
            matching_rows.append(
                row
            )

    if len(
        matching_rows
    ) != 1:
        raise RuntimeError(
            f'Expected exactly one "{region_name}" row '
            f"at VG={gate_voltage_V:.6f} V, "
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
# Program-time classification
# ============================================================

def classify_program_time(
    program_time_s,
):
    if math.isinf(
        program_time_s
    ):
        return "unreachable"

    if program_time_s <= 1.0e-7:
        return "below_100_ns"

    if program_time_s <= 1.0e-6:
        return "below_1_us"

    if program_time_s <= 1.0e-5:
        return "below_10_us"

    if program_time_s <= 1.0e-4:
        return "below_100_us"

    if program_time_s <= 1.0e-3:
        return "below_1_ms"

    if program_time_s <= 1.0:
        return "below_1_s"

    return "above_1_s"


# ============================================================
# Summary generation
# ============================================================

def build_summary_rows(
    case_output_paths,
):
    summary_rows = []

    for case_output_path in case_output_paths:
        _, rows = read_csv_rows(
            case_output_path
        )

        blocking_thicknesses_in_file = sorted(
            {
                float(
                    row[
                        "blocking_oxide_thickness_nm"
                    ]
                )
                for row in rows
            }
        )

        if len(
            blocking_thicknesses_in_file
        ) != 1:
            raise RuntimeError(
                "Expected one blocking-oxide thickness "
                "per case CSV."
            )

        blocking_oxide_thickness_nm = (
            blocking_thicknesses_in_file[0]
        )

        gate_voltages_in_file = sorted(
            {
                float(
                    row[
                        "gate_voltage_V"
                    ]
                )
                for row in rows
            }
        )

        for gate_voltage_V in gate_voltages_in_file:
            tunnel_row = (
                select_single_region_row(
                    rows=rows,
                    region_name=(
                        TUNNEL_OXIDE_REGION
                    ),
                    gate_voltage_V=(
                        gate_voltage_V
                    ),
                )
            )

            blocking_row = (
                select_single_region_row(
                    rows=rows,
                    region_name=(
                        BLOCKING_OXIDE_REGION
                    ),
                    gate_voltage_V=(
                        gate_voltage_V
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
                        if blocking_average_field_V_cm
                        > 0.0
                        else math.inf
                    ),
                    "tunnel_to_blocking_maximum_field_ratio": (
                        tunnel_maximum_field_V_cm
                        / blocking_maximum_field_V_cm
                        if blocking_maximum_field_V_cm
                        > 0.0
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
                    "program_time_classification": (
                        classify_program_time(
                            program_time_s
                        )
                    ),
                    "fixed_field_assumption": True,
                    "capture_efficiency_assumed": 1.0,
                    "source_csv": str(
                        Path(
                            case_output_path
                        ).resolve()
                    ),
                }
            )

    summary_rows.sort(
        key=lambda row: (
            float(
                row[
                    "blocking_oxide_thickness_nm"
                ]
            ),
            float(
                row[
                    "gate_voltage_V"
                ]
            ),
        )
    )

    return summary_rows


# ============================================================
# Output fieldnames
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
    "program_time_classification",
    "fixed_field_assumption",
    "capture_efficiency_assumed",
    "source_csv",
)


# ============================================================
# CSV writing
# ============================================================

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
# Best condition for each blocking oxide
# ============================================================

def select_best_conditions(
    summary_rows,
):
    rows_by_blocking_thickness = {}

    for row in summary_rows:
        blocking_thickness_nm = float(
            row[
                "blocking_oxide_thickness_nm"
            ]
        )

        rows_by_blocking_thickness.setdefault(
            blocking_thickness_nm,
            [],
        ).append(
            row
        )

    best_rows = []

    for (
        blocking_thickness_nm,
        thickness_rows,
    ) in sorted(
        rows_by_blocking_thickness.items()
    ):
        best_row = min(
            thickness_rows,
            key=program_time_sort_value,
        )

        best_rows.append(
            {
                "blocking_oxide_thickness_nm": (
                    blocking_thickness_nm
                ),
                "selected_gate_voltage_V": (
                    best_row[
                        "gate_voltage_V"
                    ]
                ),
                "tunnel_average_field_abs_V_cm": (
                    best_row[
                        "tunnel_average_field_abs_V_cm"
                    ]
                ),
                "tunnel_maximum_field_abs_V_cm": (
                    best_row[
                        "tunnel_maximum_field_abs_V_cm"
                    ]
                ),
                "blocking_average_field_abs_V_cm": (
                    best_row[
                        "blocking_average_field_abs_V_cm"
                    ]
                ),
                "blocking_maximum_field_abs_V_cm": (
                    best_row[
                        "blocking_maximum_field_abs_V_cm"
                    ]
                ),
                "tunneling_current_density_A_cm2": (
                    best_row[
                        "tunneling_current_density_A_cm2"
                    ]
                ),
                "fixed_field_program_time_s": (
                    best_row[
                        "fixed_field_program_time_s"
                    ]
                ),
                "program_time_classification": (
                    best_row[
                        "program_time_classification"
                    ]
                ),
                "selection_basis": (
                    "minimum_fixed_field_program_time_"
                    "within_tested_voltage_range"
                ),
            }
        )

    return best_rows


BEST_CONDITION_FIELDNAMES = (
    "blocking_oxide_thickness_nm",
    "selected_gate_voltage_V",
    "tunnel_average_field_abs_V_cm",
    "tunnel_maximum_field_abs_V_cm",
    "blocking_average_field_abs_V_cm",
    "blocking_maximum_field_abs_V_cm",
    "tunneling_current_density_A_cm2",
    "fixed_field_program_time_s",
    "program_time_classification",
    "selection_basis",
)


def write_best_condition_csv(
    best_rows,
):
    with BEST_CONDITION_OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=(
                BEST_CONDITION_FIELDNAMES
            ),
        )

        writer.writeheader()

        writer.writerows(
            best_rows
        )


# ============================================================
# Console reporting
# ============================================================

def print_summary(
    summary_rows,
):
    print_section(
        "BLOCKING-OXIDE VOLTAGE-SWEEP SUMMARY"
    )

    for row in summary_rows:
        print(
            "tblock="
            f"{float(row['blocking_oxide_thickness_nm']):5.1f} nm, "
            "VG="
            f"{float(row['gate_voltage_V']):5.1f} V, "
            "Etun_avg="
            f"{float(row['tunnel_average_field_abs_V_cm']):.6e} V/cm, "
            "Etun_max="
            f"{float(row['tunnel_maximum_field_abs_V_cm']):.6e} V/cm, "
            "Eblock_avg="
            f"{float(row['blocking_average_field_abs_V_cm']):.6e} V/cm, "
            "Eblock_max="
            f"{float(row['blocking_maximum_field_abs_V_cm']):.6e} V/cm, "
            "J="
            f"{float(row['tunneling_current_density_A_cm2']):.6e} A/cm^2, "
            "time="
            f"{row['fixed_field_program_time_s']} s"
        )


def print_program_speed_ranking(
    ranking_rows,
    maximum_rows=15,
):
    print_section(
        "FASTEST TESTED CONDITIONS"
    )

    for row in ranking_rows[
        :maximum_rows
    ]:
        print(
            f"{int(row['program_speed_rank']):2d}. "
            "BlockingOxide="
            f"{float(row['blocking_oxide_thickness_nm']):.1f} nm, "
            "VG="
            f"{float(row['gate_voltage_V']):.1f} V, "
            "J="
            f"{float(row['tunneling_current_density_A_cm2']):.6e} A/cm^2, "
            "time="
            f"{row['fixed_field_program_time_s']} s"
        )


def print_best_conditions(
    best_rows,
):
    print_section(
        "BEST TESTED VOLTAGE FOR EACH BLOCKING-OXIDE THICKNESS"
    )

    for row in best_rows:
        print(
            "BlockingOxide="
            f"{float(row['blocking_oxide_thickness_nm']):5.1f} nm, "
            "selected VG="
            f"{float(row['selected_gate_voltage_V']):5.1f} V, "
            "Etun_avg="
            f"{float(row['tunnel_average_field_abs_V_cm']):.6e} V/cm, "
            "Eblock_max="
            f"{float(row['blocking_maximum_field_abs_V_cm']):.6e} V/cm, "
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
        "BLOCKING-OXIDE DEPENDENT GATE-VOLTAGE SWEEP"
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
        "Target trapped-electron sheet density: "
        f"{TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2:.6e} "
        "cm^-2"
    )

    print()
    print(
        "Sweep conditions:"
    )

    for (
        blocking_oxide_thickness_nm,
        gate_voltages_V,
    ) in BLOCKING_OXIDE_VOLTAGE_SWEEP.items():
        print(
            "  BlockingOxide "
            f"{blocking_oxide_thickness_nm:.1f} nm "
            "-> VG "
            f"{gate_voltages_V}"
        )

    print()
    print(
        "Result directory:\n"
        f"{RESULTS_DIRECTORY.resolve()}"
    )

    case_output_paths = []

    for (
        blocking_oxide_thickness_nm,
        gate_voltages_V,
    ) in BLOCKING_OXIDE_VOLTAGE_SWEEP.items():
        case_output_path = (
            run_single_blocking_oxide_case(
                blocking_oxide_thickness_nm=(
                    blocking_oxide_thickness_nm
                ),
                gate_voltages_V=(
                    gate_voltages_V
                ),
            )
        )

        case_output_paths.append(
            case_output_path
        )

    print_section(
        "COMBINE DETAILED VOLTAGE-SWEEP RESULTS"
    )

    combine_case_results(
        case_output_paths=(
            case_output_paths
        ),
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

    best_rows = select_best_conditions(
        summary_rows=(
            summary_rows
        ),
    )

    write_best_condition_csv(
        best_rows=(
            best_rows
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
        maximum_rows=15,
    )

    print_best_conditions(
        best_rows=(
            best_rows
        ),
    )

    print_section(
        "BLOCKING-OXIDE VOLTAGE SWEEP COMPLETED"
    )

    print(
        "Individual case count: "
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

    print(
        "Best tested condition CSV:\n"
        f"{BEST_CONDITION_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "Important interpretation:"
    )

    print(
        "The highest tested voltage will usually produce the "
        "shortest fixed-field program time."
    )

    print(
        "Therefore, the best-condition CSV is only a numerical "
        "speed ranking within the tested ranges, not the final "
        "device-optimization decision."
    )

    print(
        "Final selection must also consider tunnel-oxide field, "
        "blocking-oxide field, reliability, retention, gate-side "
        "injection, trapped-charge feedback, and threshold shift."
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_BLOCKING_OXIDE_VOLTAGE_SWEEP SCRIPT STARTED"
    )

    main()