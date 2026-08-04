# ============================================================
# Coarse high-voltage TunnelOxide sweep controller
#
# Candidate TunnelOxide thicknesses:
#   2.5, 3.0, 3.5 nm
#
# Gate-voltage sweep:
#   6, 10, 14, 18, 22, 26, 30 V
#
# Each thickness case is executed in an independent Python
# process to prevent DEVSIM device and mesh name collisions.
#
# Existing low-voltage results are preserved because all
# outputs are stored in a separate directory.
# ============================================================

import csv
import math
import os
import subprocess
import sys
from pathlib import Path


# ============================================================
# Sweep settings
# ============================================================

TUNNEL_OXIDE_THICKNESSES_NM = (
    2.5,
    3.0,
    3.5,
)

GATE_VOLTAGES_V = (
    6.0,
    10.0,
    14.0,
    18.0,
    22.0,
    26.0,
    30.0,
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
    / "coarse_voltage_sweep_6to30V"
)

COMBINED_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "coarse_voltage_sweep_combined.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "coarse_voltage_sweep_summary.csv"
)


# ============================================================
# Required CSV information
# ============================================================

TARGET_REGION = "TunnelOxide"

SUMMARY_COLUMNS = (
    "tunnel_oxide_thickness_nm",
    "gate_voltage_V",
    "drain_voltage_V",
    "drain_current_A",
    "active_interface_field_area_weighted_abs_V_cm",
    "active_interface_field_max_abs_V_cm",
    "edge_resolved_maximum_local_field_abs_V_cm",
    "fowler_nordheim_exponent",
    "fowler_nordheim_current_density_A_cm2",
    "total_tunneling_current_density_A_cm2",
    "edge_resolved_effective_current_density_A_cm2",
    "edge_resolved_total_tunneling_current_A",
    "edge_resolved_minimum_local_current_density_A_cm2",
    "edge_resolved_maximum_local_current_density_A_cm2",
)


# ============================================================
# Utility functions
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def validate_positive_finite(
    name,
    value,
):
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_nonnegative_finite(
    name,
    value,
):
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    if numeric_value < 0.0:
        raise ValueError(
            f"{name} must not be negative. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def thickness_label(
    thickness_nm,
):
    return (
        f"{float(thickness_nm):.1f}"
        .replace(".", "p")
    )


def voltage_list_text():
    return ",".join(
        f"{float(voltage):g}"
        for voltage in GATE_VOLTAGES_V
    )


def case_output_path(
    thickness_nm,
):
    label = thickness_label(
        thickness_nm
    )

    return (
        RESULTS_DIRECTORY
        / (
            "positive_field_sweep_"
            f"tox_{label}nm_"
            "vg_6to30V.csv"
        )
    )


# ============================================================
# Input validation
# ============================================================

def validate_settings():
    if not SINGLE_CASE_SCRIPT.exists():
        raise FileNotFoundError(
            "Single-case sweep script was not found:\n"
            f"{SINGLE_CASE_SCRIPT.resolve()}"
        )

    for thickness_nm in (
        TUNNEL_OXIDE_THICKNESSES_NM
    ):
        validate_positive_finite(
            name="TunnelOxide thickness",
            value=thickness_nm,
        )

    previous_voltage = None

    for voltage_V in GATE_VOLTAGES_V:
        validate_nonnegative_finite(
            name="gate voltage",
            value=voltage_V,
        )

        if (
            previous_voltage is not None
            and voltage_V <= previous_voltage
        ):
            raise ValueError(
                "Gate voltages must be arranged "
                "in strictly increasing order."
            )

        previous_voltage = voltage_V


# ============================================================
# Single-thickness subprocess execution
# ============================================================

def run_single_thickness_case(
    thickness_nm,
):
    output_csv = case_output_path(
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
    ] = voltage_list_text()

    environment[
        "POSITIVE_FIELD_SWEEP_OUTPUT"
    ] = str(
        output_csv.resolve()
    )

    print()
    print("-" * 88)

    print(
        "Starting thickness case"
    )

    print(
        "  TunnelOxide thickness: "
        f"{thickness_nm:.3f} nm"
    )

    print(
        "  Gate voltages: "
        f"{GATE_VOLTAGES_V}"
    )

    print(
        "  Output CSV:\n"
        f"  {output_csv.resolve()}"
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
            "Coarse voltage sweep failed.\n"
            f"TunnelOxide thickness: {thickness_nm:.3f} nm\n"
            "Subprocess return code: "
            f"{completed_process.returncode}"
        )

    if not output_csv.exists():
        raise RuntimeError(
            "The subprocess completed, but the expected "
            "result CSV was not created:\n"
            f"{output_csv.resolve()}"
        )

    print(
        f"Thickness case {thickness_nm:.3f} nm completed."
    )

    return output_csv


# ============================================================
# CSV input
# ============================================================

def read_csv_rows(
    csv_path,
):
    csv_path = Path(csv_path)

    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        fieldnames = reader.fieldnames
        rows = list(reader)

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

    return fieldnames, rows


# ============================================================
# Detailed-result combination
# ============================================================

def combine_case_results(
    case_paths,
):
    combined_fieldnames = None
    combined_rows = []

    for case_path in case_paths:
        fieldnames, rows = read_csv_rows(
            case_path
        )

        if combined_fieldnames is None:
            combined_fieldnames = fieldnames

        elif fieldnames != combined_fieldnames:
            raise RuntimeError(
                "CSV columns do not match between "
                "thickness cases:\n"
                f"{case_path.resolve()}"
            )

        combined_rows.extend(rows)

    if combined_fieldnames is None:
        raise RuntimeError(
            "No CSV field names were collected."
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
# Summary generation
# ============================================================

def check_summary_columns(
    fieldnames,
):
    required_columns = {
        "region",
        "tunnel_oxide_thickness_nm",
        "gate_voltage_V",
        "edge_resolved_effective_current_density_A_cm2",
        "edge_resolved_maximum_local_field_abs_V_cm",
    }

    missing_columns = (
        required_columns
        - set(fieldnames)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise KeyError(
            "The detailed CSV is missing columns "
            "required for the coarse-sweep summary: "
            f"{missing_text}"
        )


def build_summary_rows(
    combined_rows,
):
    if not combined_rows:
        raise RuntimeError(
            "There are no combined result rows."
        )

    check_summary_columns(
        combined_rows[0].keys()
    )

    summary_rows = []

    for row in combined_rows:
        if row.get("region") != TARGET_REGION:
            continue

        summary_row = {}

        for column_name in SUMMARY_COLUMNS:
            summary_row[column_name] = row.get(
                column_name,
                "",
            )

        summary_rows.append(summary_row)

    if not summary_rows:
        raise RuntimeError(
            f'No "{TARGET_REGION}" rows were found '
            "in the detailed sweep results."
        )

    summary_rows.sort(
        key=lambda row: (
            float(
                row[
                    "tunnel_oxide_thickness_nm"
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
            fieldnames=SUMMARY_COLUMNS,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# Console output
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


def print_summary_rows(
    summary_rows,
):
    print_section(
        "COARSE VOLTAGE SWEEP SUMMARY"
    )

    for row in summary_rows:
        thickness_nm = float_or_nan(
            row[
                "tunnel_oxide_thickness_nm"
            ]
        )

        gate_voltage_V = float_or_nan(
            row[
                "gate_voltage_V"
            ]
        )

        average_field_V_cm = float_or_nan(
            row[
                "active_interface_field_area_weighted_abs_V_cm"
            ]
        )

        maximum_field_V_cm = float_or_nan(
            row[
                "edge_resolved_maximum_local_field_abs_V_cm"
            ]
        )

        current_density_A_cm2 = float_or_nan(
            row[
                "edge_resolved_effective_current_density_A_cm2"
            ]
        )

        print(
            f"tox={thickness_nm:4.1f} nm, "
            f"VG={gate_voltage_V:5.1f} V, "
            f"Eavg={average_field_V_cm:.6e} V/cm, "
            f"Emax={maximum_field_V_cm:.6e} V/cm, "
            f"Jedge={current_density_A_cm2:.6e} A/cm^2"
        )


# ============================================================
# Highest-current ranking
# ============================================================

def print_highest_current_conditions(
    summary_rows,
    number_of_rows=10,
):
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: float_or_nan(
            row[
                "edge_resolved_effective_current_density_A_cm2"
            ]
        ),
        reverse=True,
    )

    print_section(
        "HIGHEST EDGE-RESOLVED CURRENT-DENSITY CONDITIONS"
    )

    for rank, row in enumerate(
        ranked_rows[:number_of_rows],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            "tox="
            f"{float_or_nan(row['tunnel_oxide_thickness_nm']):.3f} nm, "
            "VG="
            f"{float_or_nan(row['gate_voltage_V']):+.3f} V, "
            "J="
            f"{float_or_nan(row['edge_resolved_effective_current_density_A_cm2']):.6e} "
            "A/cm^2, "
            "Emax="
            f"{float_or_nan(row['edge_resolved_maximum_local_field_abs_V_cm']):.6e} "
            "V/cm"
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

    print_section(
        "COARSE HIGH-VOLTAGE SWEEP: 6 TO 30 V"
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
        "Single-case script:\n"
        f"{SINGLE_CASE_SCRIPT.resolve()}"
    )

    print(
        "Result directory:\n"
        f"{RESULTS_DIRECTORY.resolve()}"
    )

    case_paths = []

    for thickness_nm in (
        TUNNEL_OXIDE_THICKNESSES_NM
    ):
        case_path = (
            run_single_thickness_case(
                thickness_nm=thickness_nm,
            )
        )

        case_paths.append(case_path)

    print_section(
        "COMBINING COARSE-SWEEP RESULTS"
    )

    combined_rows = combine_case_results(
        case_paths=case_paths,
    )

    summary_rows = build_summary_rows(
        combined_rows=combined_rows,
    )

    write_summary_csv(
        summary_rows=summary_rows,
    )

    print_summary_rows(
        summary_rows=summary_rows,
    )

    print_highest_current_conditions(
        summary_rows=summary_rows,
    )

    print_section(
        "COARSE VOLTAGE SWEEP COMPLETED"
    )

    print(
        "Completed thickness cases: "
        f"{len(case_paths)}"
    )

    print(
        "Detailed combined CSV:\n"
        f"{COMBINED_OUTPUT_CSV.resolve()}"
    )

    print(
        "Summary CSV:\n"
        f"{SUMMARY_OUTPUT_CSV.resolve()}"
    )

    print()
    print(
        "CAUTION: The largest tunneling current is not "
        "automatically the optimal operating condition. "
        "Maximum local field, oxide reliability, retention, "
        "and trapped-charge feedback must also be evaluated."
    )


if __name__ == "__main__":
    main()