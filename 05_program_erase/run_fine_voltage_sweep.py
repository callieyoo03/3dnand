# ============================================================
# Fine voltage sweep controller
#
# Candidate TunnelOxide thicknesses:
#   2.5, 3.0, 3.5 nm
#
# Gate-voltage sweep:
#   8, 9, 10, 11, 12, 13, 14 V
#
# Output:
#   1. Individual detailed CSV for each thickness
#   2. Combined detailed CSV
#   3. TunnelOxide-only summary CSV
#   4. Program-time estimate for target sheet density
#
# Assumptions for program-time estimate:
#   - Constant tunneling current density
#   - 100% electron capture efficiency
#   - No trapped-charge feedback
#   - No trap saturation or detrapping
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
# Sweep conditions
# ============================================================

TUNNEL_OXIDE_THICKNESSES_NM = (
    2.5,
    3.0,
    3.5,
)

GATE_VOLTAGES_V = (
    8.0,
    9.0,
    10.0,
    11.0,
    12.0,
    13.0,
    14.0,
)

TARGET_ELECTRON_SHEET_DENSITY_CM2 = (
    1.0e12
)


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

SINGLE_CASE_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "fine_voltage_sweep_8to14V"
)

COMBINED_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "fine_voltage_sweep_combined.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "fine_voltage_sweep_summary.csv"
)

RANKING_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "fine_voltage_sweep_ranking.csv"
)


# ============================================================
# Target region
# ============================================================

TARGET_REGION = "TunnelOxide"


# ============================================================
# Console utility
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 92)
    print(title)
    print("=" * 92)


# ============================================================
# Validation
# ============================================================

def validate_finite(
    name,
    value,
):
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_nonnegative_finite(
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


def validate_positive_finite(
    name,
    value,
):
    numeric_value = (
        validate_nonnegative_finite(
            name=name,
            value=value,
        )
    )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_sweep_settings():
    if not SINGLE_CASE_SCRIPT.exists():
        raise FileNotFoundError(
            "run_positive_field_sweep.py was not found:\n"
            f"{SINGLE_CASE_SCRIPT.resolve()}"
        )

    if not TUNNEL_OXIDE_THICKNESSES_NM:
        raise ValueError(
            "At least one TunnelOxide thickness "
            "must be provided."
        )

    if not GATE_VOLTAGES_V:
        raise ValueError(
            "At least one gate voltage "
            "must be provided."
        )

    previous_thickness_nm = None

    for thickness_nm in (
        TUNNEL_OXIDE_THICKNESSES_NM
    ):
        validate_positive_finite(
            name="TunnelOxide thickness",
            value=thickness_nm,
        )

        if (
            previous_thickness_nm is not None
            and thickness_nm
            <= previous_thickness_nm
        ):
            raise ValueError(
                "TunnelOxide thicknesses must be "
                "strictly increasing."
            )

        previous_thickness_nm = (
            thickness_nm
        )

    previous_voltage_V = None

    for voltage_V in GATE_VOLTAGES_V:
        validate_nonnegative_finite(
            name="gate voltage",
            value=voltage_V,
        )

        if (
            previous_voltage_V is not None
            and voltage_V
            <= previous_voltage_V
        ):
            raise ValueError(
                "Gate voltages must be "
                "strictly increasing."
            )

        previous_voltage_V = voltage_V

    validate_positive_finite(
        name=(
            "TARGET_ELECTRON_SHEET_DENSITY_CM2"
        ),
        value=(
            TARGET_ELECTRON_SHEET_DENSITY_CM2
        ),
    )


# ============================================================
# Filename and environment utilities
# ============================================================

def thickness_to_filename_token(
    thickness_nm,
):
    return (
        f"{float(thickness_nm):.1f}"
        .replace(".", "p")
    )


def voltage_environment_text():
    return ",".join(
        f"{float(voltage_V):g}"
        for voltage_V in GATE_VOLTAGES_V
    )


def get_case_output_csv(
    thickness_nm,
):
    thickness_token = (
        thickness_to_filename_token(
            thickness_nm
        )
    )

    return (
        RESULTS_DIRECTORY
        / (
            "positive_field_sweep_"
            f"tox_{thickness_token}nm_"
            "vg_8to14V.csv"
        )
    )


# ============================================================
# Single-thickness execution
# ============================================================

def run_single_thickness_case(
    thickness_nm,
):
    output_csv = get_case_output_csv(
        thickness_nm
    )

    environment = os.environ.copy()

    environment[
        "TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        float(thickness_nm)
    )

    environment[
        "GATE_VOLTAGES_V"
    ] = voltage_environment_text()

    environment[
        "POSITIVE_FIELD_SWEEP_OUTPUT"
    ] = str(
        output_csv.resolve()
    )

    print()
    print("-" * 92)
    print("STARTING FINE-SWEEP THICKNESS CASE")

    print(
        "TunnelOxide thickness: "
        f"{thickness_nm:.3f} nm"
    )

    print(
        "Gate voltages: "
        f"{GATE_VOLTAGES_V}"
    )

    print(
        "Output CSV:\n"
        f"{output_csv.resolve()}"
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            str(SINGLE_CASE_SCRIPT),
        ],
        cwd=str(SCRIPT_DIRECTORY),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            "Fine voltage sweep failed.\n"
            f"TunnelOxide thickness: "
            f"{thickness_nm:.3f} nm\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "Expected result CSV was not created:\n"
            f"{output_csv.resolve()}"
        )

    print(
        "Thickness case completed: "
        f"{thickness_nm:.3f} nm"
    )

    return output_csv


# ============================================================
# CSV reading
# ============================================================

def read_csv_rows(
    csv_path,
):
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


# ============================================================
# Combine detailed results
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
            combined_fieldnames = fieldnames

        elif fieldnames != combined_fieldnames:
            raise RuntimeError(
                "CSV columns do not match between "
                "thickness cases:\n"
                f"{case_output_path.resolve()}"
            )

        combined_rows.extend(rows)

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
        writer.writerows(combined_rows)

    return combined_rows


# ============================================================
# Data handling
# ============================================================

def get_first_available_value(
    row,
    candidate_columns,
    default_value="",
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
            return value

    return default_value


def calculate_required_program_time_s(
    current_density_A_cm2,
):
    current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="current_density_A_cm2",
            value=current_density_A_cm2,
        )
    )

    if current_density_A_cm2 == 0.0:
        return math.inf

    target_charge_density_C_cm2 = (
        TARGET_ELECTRON_SHEET_DENSITY_CM2
        * ELEMENTARY_CHARGE_C
    )

    return (
        target_charge_density_C_cm2
        / current_density_A_cm2
    )


def classify_required_program_time(
    required_time_s,
):
    if math.isinf(required_time_s):
        return "unreachable_at_zero_current"

    if required_time_s <= 1.0e-9:
        return "within_1_ns"

    if required_time_s <= 1.0e-8:
        return "within_10_ns"

    if required_time_s <= 1.0e-7:
        return "within_100_ns"

    if required_time_s <= 1.0e-6:
        return "within_1_us"

    if required_time_s <= 1.0e-5:
        return "within_10_us"

    if required_time_s <= 1.0e-4:
        return "within_100_us"

    if required_time_s <= 1.0e-3:
        return "within_1_ms"

    if required_time_s <= 1.0:
        return "within_1_s"

    return "longer_than_1_s"


def format_finite_or_infinity(
    value,
):
    if math.isinf(value):
        return "inf"

    return f"{value:.12e}"


# ============================================================
# Summary generation
# ============================================================

def build_summary_rows(
    combined_rows,
):
    summary_rows = []

    for row in combined_rows:
        region = row.get(
            "region",
            "",
        ).strip()

        if region != TARGET_REGION:
            continue

        average_field_V_cm = (
            get_first_available_value(
                row=row,
                candidate_columns=(
                    "active_interface_field_area_weighted_abs_V_cm",
                    "active_interface_field_mean_abs_V_cm",
                    "tunnel_interface_field_mean_abs_V_cm",
                ),
            )
        )

        maximum_field_V_cm = (
            get_first_available_value(
                row=row,
                candidate_columns=(
                    "edge_resolved_maximum_local_field_abs_V_cm",
                    "active_interface_field_max_abs_V_cm",
                    "tunnel_interface_field_max_abs_V_cm",
                ),
            )
        )

        edge_current_density_A_cm2 = (
            get_first_available_value(
                row=row,
                candidate_columns=(
                    "edge_resolved_effective_current_density_A_cm2",
                    "total_tunneling_current_density_A_cm2",
                ),
            )
        )

        if edge_current_density_A_cm2 == "":
            raise RuntimeError(
                "Could not find an edge-resolved "
                "current-density column."
            )

        edge_current_density_A_cm2 = float(
            edge_current_density_A_cm2
        )

        required_program_time_s = (
            calculate_required_program_time_s(
                current_density_A_cm2=(
                    edge_current_density_A_cm2
                ),
            )
        )

        summary_rows.append(
            {
                "tunnel_oxide_thickness_nm": (
                    row.get(
                        "tunnel_oxide_thickness_nm",
                        "",
                    )
                ),
                "gate_voltage_V": (
                    row.get(
                        "gate_voltage_V",
                        "",
                    )
                ),
                "drain_voltage_V": (
                    row.get(
                        "drain_voltage_V",
                        "",
                    )
                ),
                "drain_current_A": (
                    row.get(
                        "drain_current_A",
                        "",
                    )
                ),
                "active_interface_average_field_abs_V_cm": (
                    average_field_V_cm
                ),
                "active_interface_maximum_field_abs_V_cm": (
                    maximum_field_V_cm
                ),
                "fowler_nordheim_exponent": (
                    row.get(
                        "fowler_nordheim_exponent",
                        "",
                    )
                ),
                "mean_field_current_density_A_cm2": (
                    row.get(
                        "total_tunneling_current_density_A_cm2",
                        "",
                    )
                ),
                "edge_resolved_current_density_A_cm2": (
                    edge_current_density_A_cm2
                ),
                "target_electron_sheet_density_cm2": (
                    TARGET_ELECTRON_SHEET_DENSITY_CM2
                ),
                "required_program_time_s": (
                    format_finite_or_infinity(
                        required_program_time_s
                    )
                ),
                "required_program_time_classification": (
                    classify_required_program_time(
                        required_program_time_s
                    )
                ),
                "edge_resolved_total_tunneling_current_A": (
                    row.get(
                        "edge_resolved_total_tunneling_current_A",
                        "",
                    )
                ),
                "edge_resolved_minimum_local_current_density_A_cm2": (
                    row.get(
                        "edge_resolved_minimum_local_current_density_A_cm2",
                        "",
                    )
                ),
                "edge_resolved_maximum_local_current_density_A_cm2": (
                    row.get(
                        "edge_resolved_maximum_local_current_density_A_cm2",
                        "",
                    )
                ),
                "tunneling_barrier_height_eV": (
                    row.get(
                        "tunneling_barrier_height_eV",
                        "",
                    )
                ),
                "tunneling_effective_mass_ratio": (
                    row.get(
                        "tunneling_effective_mass_ratio",
                        "",
                    )
                ),
                "fixed_field_approximation": True,
                "capture_efficiency_assumed": 1.0,
            }
        )

    if not summary_rows:
        raise RuntimeError(
            f'No "{TARGET_REGION}" rows were found.'
        )

    summary_rows.sort(
        key=lambda result_row: (
            float(
                result_row[
                    "tunnel_oxide_thickness_nm"
                ]
            ),
            float(
                result_row[
                    "gate_voltage_V"
                ]
            ),
        )
    )

    return summary_rows


# ============================================================
# CSV output
# ============================================================

SUMMARY_FIELDNAMES = (
    "tunnel_oxide_thickness_nm",
    "gate_voltage_V",
    "drain_voltage_V",
    "drain_current_A",
    "active_interface_average_field_abs_V_cm",
    "active_interface_maximum_field_abs_V_cm",
    "fowler_nordheim_exponent",
    "mean_field_current_density_A_cm2",
    "edge_resolved_current_density_A_cm2",
    "target_electron_sheet_density_cm2",
    "required_program_time_s",
    "required_program_time_classification",
    "edge_resolved_total_tunneling_current_A",
    "edge_resolved_minimum_local_current_density_A_cm2",
    "edge_resolved_maximum_local_current_density_A_cm2",
    "tunneling_barrier_height_eV",
    "tunneling_effective_mass_ratio",
    "fixed_field_approximation",
    "capture_efficiency_assumed",
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
            fieldnames=SUMMARY_FIELDNAMES,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


def write_ranking_csv(
    summary_rows,
):
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            math.inf
            if row[
                "required_program_time_s"
            ] == "inf"
            else float(
                row[
                    "required_program_time_s"
                ]
            )
        ),
    )

    ranking_rows = []

    for rank, row in enumerate(
        ranked_rows,
        start=1,
    ):
        ranking_rows.append(
            {
                "rank": rank,
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
                ("rank",)
                + SUMMARY_FIELDNAMES
            ),
        )

        writer.writeheader()
        writer.writerows(ranking_rows)

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
        return float("nan")

    return float(value)


def print_summary(
    summary_rows,
):
    print_section(
        "FINE VOLTAGE SWEEP SUMMARY"
    )

    for row in summary_rows:
        required_time_text = (
            row[
                "required_program_time_s"
            ]
        )

        if required_time_text == "inf":
            required_time_display = "inf"
        else:
            required_time_display = (
                f"{float(required_time_text):.6e}"
            )

        print(
            "tox="
            f"{float_or_nan(row['tunnel_oxide_thickness_nm']):4.1f} nm, "
            "VG="
            f"{float_or_nan(row['gate_voltage_V']):4.1f} V, "
            "Eavg="
            f"{float_or_nan(row['active_interface_average_field_abs_V_cm']):.6e} "
            "V/cm, "
            "Emax="
            f"{float_or_nan(row['active_interface_maximum_field_abs_V_cm']):.6e} "
            "V/cm, "
            "Jedge="
            f"{float_or_nan(row['edge_resolved_current_density_A_cm2']):.6e} "
            "A/cm^2, "
            "t_target="
            f"{required_time_display} s"
        )


def print_top_conditions(
    ranking_rows,
    number_of_conditions=10,
):
    print_section(
        "FASTEST FIXED-FIELD PROGRAM CONDITIONS"
    )

    for row in ranking_rows[
        :number_of_conditions
    ]:
        print(
            f"{row['rank']:2d}. "
            "tox="
            f"{float(row['tunnel_oxide_thickness_nm']):.3f} nm, "
            "VG="
            f"{float(row['gate_voltage_V']):+.3f} V, "
            "J="
            f"{float(row['edge_resolved_current_density_A_cm2']):.6e} "
            "A/cm^2, "
            "t="
            f"{row['required_program_time_s']} s, "
            "class="
            f"{row['required_program_time_classification']}"
        )


# ============================================================
# Main
# ============================================================

def main():
    validate_sweep_settings()

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print_section(
        "FINE VOLTAGE SWEEP: 8 TO 14 V"
    )

    print(
        "TunnelOxide thicknesses: "
        f"{TUNNEL_OXIDE_THICKNESSES_NM}"
    )

    print(
        "Gate voltages: "
        f"{GATE_VOLTAGES_V}"
    )

    print(
        "Target electron sheet density: "
        f"{TARGET_ELECTRON_SHEET_DENSITY_CM2:.3e} cm^-2"
    )

    print(
        "Single-case script:\n"
        f"{SINGLE_CASE_SCRIPT.resolve()}"
    )

    print(
        "Output directory:\n"
        f"{RESULTS_DIRECTORY.resolve()}"
    )

    case_output_paths = []

    for thickness_nm in (
        TUNNEL_OXIDE_THICKNESSES_NM
    ):
        case_output_csv = (
            run_single_thickness_case(
                thickness_nm=thickness_nm,
            )
        )

        case_output_paths.append(
            case_output_csv
        )

    print_section(
        "COMBINING FINE-SWEEP RESULTS"
    )

    combined_rows = combine_case_results(
        case_output_paths=case_output_paths,
    )

    summary_rows = build_summary_rows(
        combined_rows=combined_rows,
    )

    write_summary_csv(
        summary_rows=summary_rows,
    )

    ranking_rows = write_ranking_csv(
        summary_rows=summary_rows,
    )

    print_summary(
        summary_rows=summary_rows,
    )

    print_top_conditions(
        ranking_rows=ranking_rows,
        number_of_conditions=10,
    )

    print_section(
        "FINE VOLTAGE SWEEP COMPLETED"
    )

    print(
        "Detailed combined CSV:\n"
        f"{COMBINED_OUTPUT_CSV.resolve()}"
    )

    print(
        "Summary CSV:\n"
        f"{SUMMARY_OUTPUT_CSV.resolve()}"
    )

    print(
        "Program-time ranking CSV:\n"
        f"{RANKING_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "CAUTION: Required program times are calculated "
        "using a fixed-field model with 100% capture "
        "efficiency. Trapped-charge feedback, trap "
        "saturation, oxide reliability, and retention "
        "are not included."
    )


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    print(
        "RUN_FINE_VOLTAGE_SWEEP SCRIPT STARTED"
    )

    main()