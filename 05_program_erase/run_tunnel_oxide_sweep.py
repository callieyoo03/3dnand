# ============================================================
# TunnelOxide thickness sweep controller
#
# Each geometry is executed in an independent Python process.
# This avoids DEVSIM device/mesh name collisions.
# ============================================================

import csv
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
    4.0,
    4.5,
)

SCRIPT_DIRECTORY = Path(
    __file__
).resolve().parent

SINGLE_CASE_SCRIPT = (
    SCRIPT_DIRECTORY
    / "run_positive_field_sweep.py"
)

RESULTS_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "tunnel_oxide_sweep"
)

COMBINED_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "tunnel_oxide_sweep_combined.csv"
)

SUMMARY_OUTPUT_CSV = (
    RESULTS_DIRECTORY
    / "tunnel_oxide_sweep_summary.csv"
)


# ============================================================
# Utility functions
# ============================================================

def print_section(
    title,
):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def thickness_label(
    thickness_nm,
):
    return (
        f"{float(thickness_nm):.1f}"
        .replace(".", "p")
    )


def case_output_path(
    thickness_nm,
):
    label = thickness_label(
        thickness_nm
    )

    return (
        RESULTS_DIRECTORY
        / f"positive_field_sweep_tox_{label}nm.csv"
    )


def run_single_case(
    thickness_nm,
):
    output_path = case_output_path(
        thickness_nm
    )

    environment = os.environ.copy()

    environment[
        "TUNNEL_OXIDE_THICKNESS_NM"
    ] = str(
        float(thickness_nm)
    )

    environment[
        "POSITIVE_FIELD_SWEEP_OUTPUT"
    ] = str(
        output_path
    )

    print()
    print(
        f"Running TunnelOxide thickness "
        f"= {thickness_nm:.3f} nm"
    )

    print(
        f'Output file: "{output_path}"'
    )

    completed_process = subprocess.run(
        [
            sys.executable,
            str(
                SINGLE_CASE_SCRIPT
            ),
        ],
        cwd=str(
            SCRIPT_DIRECTORY
        ),
        env=environment,
        check=False,
    )

    if completed_process.returncode != 0:
        raise RuntimeError(
            f"TunnelOxide thickness case "
            f"{thickness_nm:.3f} nm failed "
            f"with return code "
            f"{completed_process.returncode}."
        )

    if not output_path.exists():
        raise RuntimeError(
            f'Expected result file was not created: '
            f'"{output_path}"'
        )

    print(
        f"Case {thickness_nm:.3f} nm completed."
    )

    return output_path


def read_csv_rows(
    csv_path,
):
    with csv_path.open(
        mode="r",
        newline="",
        encoding="utf-8",
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
            f'No CSV header found in "{csv_path}".'
        )

    if not rows:
        raise RuntimeError(
            f'No result rows found in "{csv_path}".'
        )

    return fieldnames, rows


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
                f'CSV columns do not match in '
                f'"{case_path}".'
            )

        combined_rows.extend(
            rows
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

    print(
        f'Combined result written to '
        f'"{COMBINED_OUTPUT_CSV}".'
    )

    return combined_rows


def create_summary(
    combined_rows,
):
    summary_fieldnames = (
        "tunnel_oxide_thickness_nm",
        "gate_voltage_V",
        "drain_voltage_V",
        "drain_current_A",
        "tunnel_interface_field_mean_abs_V_cm",
        "tunnel_interface_field_max_abs_V_cm",
        "fowler_nordheim_exponent",
        "fowler_nordheim_current_density_A_cm2",
        "total_tunneling_current_density_A_cm2",
        "program_time_s",
        "injected_charge_density_C_cm2",
        "injected_electron_sheet_density_cm2",
    )

    summary_rows = []

    for row in combined_rows:
        # Interface and tunneling values are repeated for
        # each dielectric region in the detailed CSV.
        # Keep only the TunnelOxide row for the summary.
        if row[
            "region"
        ] != "TunnelOxide":
            continue

        summary_row = {
            field: row[field]
            for field in summary_fieldnames
        }

        summary_rows.append(
            summary_row
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

    with SUMMARY_OUTPUT_CSV.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=summary_fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            summary_rows
        )

    print(
        f'Summary result written to '
        f'"{SUMMARY_OUTPUT_CSV}".'
    )


# ============================================================
# Main
# ============================================================

def main():
    print_section(
        "TUNNEL OXIDE THICKNESS SWEEP"
    )

    print(
        f"Thickness cases: "
        f"{TUNNEL_OXIDE_THICKNESSES_NM}"
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    case_paths = []

    for thickness_nm in (
        TUNNEL_OXIDE_THICKNESSES_NM
    ):
        case_path = run_single_case(
            thickness_nm=thickness_nm,
        )

        case_paths.append(
            case_path
        )

    print_section(
        "COMBINE CASE RESULTS"
    )

    combined_rows = (
        combine_case_results(
            case_paths=case_paths,
        )
    )

    create_summary(
        combined_rows=combined_rows,
    )

    print_section(
        "TUNNEL OXIDE THICKNESS SWEEP SUCCESSFUL"
    )

    print(
        f"Completed cases: "
        f"{len(case_paths)}"
    )

    print(
        f'Combined CSV: '
        f'"{COMBINED_OUTPUT_CSV}"'
    )

    print(
        f'Summary CSV: '
        f'"{SUMMARY_OUTPUT_CSV}"'
    )


if __name__ == "__main__":
    main()