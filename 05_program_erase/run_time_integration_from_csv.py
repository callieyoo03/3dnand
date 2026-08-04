# ============================================================
# Fixed-field time integration from positive-field sweep CSV
#
# Purpose
# -------
# 1. Read positive-field sweep results
# 2. Select TunnelOxide rows
# 3. Integrate tunneling current density over time
# 4. Save time-resolved and summary CSV files
#
# Limitation
# ----------
# Trapped charge is not yet fed back into the Poisson solution.
# Therefore, current density is assumed constant during each run.
# ============================================================

import csv
import math
import os
from pathlib import Path

from time_integration import integrate_constant_current_density


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = (
    SCRIPT_DIRECTORY
    / "results"
    / "positive_field_sweep_tox_4.0nm.csv"
)

INPUT_CSV = Path(
    os.environ.get(
        "TIME_INTEGRATION_INPUT_CSV",
        str(DEFAULT_INPUT_CSV),
    )
)

DEFAULT_OUTPUT_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "time_integration"
)

OUTPUT_DIRECTORY = Path(
    os.environ.get(
        "TIME_INTEGRATION_OUTPUT_DIRECTORY",
        str(DEFAULT_OUTPUT_DIRECTORY),
    )
)


# ============================================================
# Integration settings
# ============================================================

TOTAL_TIME_S = float(
    os.environ.get(
        "PROGRAM_TOTAL_TIME_S",
        "1.0e-6",
    )
)

TIME_STEP_S = float(
    os.environ.get(
        "PROGRAM_TIME_STEP_S",
        "1.0e-8",
    )
)


# ============================================================
# CSV column names
# ============================================================

REGION_COLUMN = "region"

TARGET_REGION = "TunnelOxide"

CURRENT_DENSITY_COLUMN = (
    "total_tunneling_current_density_A_cm2"
)

GATE_VOLTAGE_COLUMN = "gate_voltage_V"

TUNNEL_OXIDE_THICKNESS_COLUMN = (
    "tunnel_oxide_thickness_nm"
)


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
    numeric_value = validate_nonnegative_finite(
        name=name,
        value=value,
    )

    if numeric_value <= 0.0:
        raise ValueError(
            f"{name} must be greater than zero. "
            f"Received: {numeric_value}"
        )

    return numeric_value


# ============================================================
# CSV input
# ============================================================

def check_required_columns(
    fieldnames,
):
    required_columns = {
        REGION_COLUMN,
        CURRENT_DENSITY_COLUMN,
        GATE_VOLTAGE_COLUMN,
        TUNNEL_OXIDE_THICKNESS_COLUMN,
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
            "Input CSV is missing required columns: "
            f"{missing_text}"
        )


def read_sweep_rows(
    input_csv,
):
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            "Input sweep CSV was not found:\n"
            f"{input_csv.resolve()}"
        )

    selected_rows = []

    with input_csv.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                "Input CSV has no header."
            )

        check_required_columns(
            reader.fieldnames
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            region = row[REGION_COLUMN].strip()

            # The CSV contains TunnelOxide, ChargeTrap,
            # and BlockingOxide rows for each voltage.
            # Only the TunnelOxide row is used here.
            if region != TARGET_REGION:
                continue

            current_density_A_cm2 = (
                validate_nonnegative_finite(
                    name=(
                        f"row {row_number} "
                        f"{CURRENT_DENSITY_COLUMN}"
                    ),
                    value=row[
                        CURRENT_DENSITY_COLUMN
                    ],
                )
            )

            gate_voltage_V = validate_finite(
                name=(
                    f"row {row_number} "
                    f"{GATE_VOLTAGE_COLUMN}"
                ),
                value=row[
                    GATE_VOLTAGE_COLUMN
                ],
            )

            tunnel_oxide_thickness_nm = (
                validate_positive_finite(
                    name=(
                        f"row {row_number} "
                        f"{TUNNEL_OXIDE_THICKNESS_COLUMN}"
                    ),
                    value=row[
                        TUNNEL_OXIDE_THICKNESS_COLUMN
                    ],
                )
            )

            selected_rows.append(
                {
                    "region": region,
                    "current_density_A_cm2": (
                        current_density_A_cm2
                    ),
                    "gate_voltage_V": (
                        gate_voltage_V
                    ),
                    "tunnel_oxide_thickness_nm": (
                        tunnel_oxide_thickness_nm
                    ),
                }
            )

    if not selected_rows:
        raise RuntimeError(
            f'No rows with region "{TARGET_REGION}" '
            "were found in the input CSV."
        )

    return selected_rows


# ============================================================
# Filename utilities
# ============================================================

def voltage_to_filename_token(
    voltage_V,
):
    sign_token = (
        "p"
        if voltage_V >= 0.0
        else "m"
    )

    magnitude_token = (
        f"{abs(voltage_V):.3f}"
        .replace(".", "p")
    )

    return (
        f"{sign_token}{magnitude_token}V"
    )


def thickness_to_filename_token(
    thickness_nm,
):
    return (
        f"{thickness_nm:.3f}"
        .replace(".", "p")
        + "nm"
    )


# ============================================================
# Time-resolved CSV output
# ============================================================

def write_time_resolved_csv(
    output_csv,
    sweep_row,
    integration_result,
):
    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "region",
        "tunnel_oxide_thickness_nm",
        "gate_voltage_V",
        "time_s",
        "time_step_s",
        "current_density_A_cm2",
        "incremental_charge_density_C_cm2",
        "cumulative_charge_density_C_cm2",
        "cumulative_electron_sheet_density_cm2",
        "fixed_field_approximation",
    ]

    with output_csv.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for integration_row in integration_result[
            "rows"
        ]:
            writer.writerow(
                {
                    "region": (
                        sweep_row["region"]
                    ),
                    "tunnel_oxide_thickness_nm": (
                        sweep_row[
                            "tunnel_oxide_thickness_nm"
                        ]
                    ),
                    "gate_voltage_V": (
                        sweep_row[
                            "gate_voltage_V"
                        ]
                    ),
                    "time_s": (
                        integration_row[
                            "time_s"
                        ]
                    ),
                    "time_step_s": (
                        integration_row[
                            "time_step_s"
                        ]
                    ),
                    "current_density_A_cm2": (
                        integration_row[
                            "current_density_A_cm2"
                        ]
                    ),
                    "incremental_charge_density_C_cm2": (
                        integration_row[
                            "incremental_charge_density_C_cm2"
                        ]
                    ),
                    "cumulative_charge_density_C_cm2": (
                        integration_row[
                            "cumulative_charge_density_C_cm2"
                        ]
                    ),
                    "cumulative_electron_sheet_density_cm2": (
                        integration_row[
                            "cumulative_electron_sheet_density_cm2"
                        ]
                    ),
                    "fixed_field_approximation": True,
                }
            )


# ============================================================
# Summary CSV output
# ============================================================

def write_summary_csv(
    output_csv,
    summary_rows,
):
    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "region",
        "tunnel_oxide_thickness_nm",
        "gate_voltage_V",
        "current_density_A_cm2",
        "total_time_s",
        "time_step_s",
        "number_of_steps",
        "final_charge_density_C_cm2",
        "final_electron_sheet_density_cm2",
        "numerical_error_C_cm2",
        "time_resolved_output_csv",
        "fixed_field_approximation",
    ]

    with output_csv.open(
        mode="w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(summary_rows)


# ============================================================
# Main execution
# ============================================================

def main():
    total_time_s = validate_nonnegative_finite(
        name="TOTAL_TIME_S",
        value=TOTAL_TIME_S,
    )

    time_step_s = validate_positive_finite(
        name="TIME_STEP_S",
        value=TIME_STEP_S,
    )

    sweep_rows = read_sweep_rows(
        input_csv=INPUT_CSV,
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_rows = []

    print()
    print("=" * 72)
    print("FIXED-FIELD TIME INTEGRATION FROM SWEEP CSV")
    print("=" * 72)

    print(
        "Input CSV:\n"
        f"{INPUT_CSV.resolve()}"
    )

    print(
        "Output directory:\n"
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print(
        "Selected region: "
        f"{TARGET_REGION}"
    )

    print(
        "Current-density column: "
        f"{CURRENT_DENSITY_COLUMN}"
    )

    print(
        "Total integration time: "
        f"{total_time_s:.6e} s"
    )

    print(
        "Time step: "
        f"{time_step_s:.6e} s"
    )

    print(
        "Number of voltage conditions: "
        f"{len(sweep_rows)}"
    )

    print()

    for sweep_row in sweep_rows:
        current_density_A_cm2 = (
            sweep_row[
                "current_density_A_cm2"
            ]
        )

        gate_voltage_V = (
            sweep_row[
                "gate_voltage_V"
            ]
        )

        tunnel_oxide_thickness_nm = (
            sweep_row[
                "tunnel_oxide_thickness_nm"
            ]
        )

        integration_result = (
            integrate_constant_current_density(
                current_density_A_cm2=(
                    current_density_A_cm2
                ),
                total_time_s=total_time_s,
                time_step_s=time_step_s,
            )
        )

        voltage_token = (
            voltage_to_filename_token(
                gate_voltage_V
            )
        )

        thickness_token = (
            thickness_to_filename_token(
                tunnel_oxide_thickness_nm
            )
        )

        output_filename = (
            "time_integration_"
            f"tox_{thickness_token}_"
            f"vg_{voltage_token}.csv"
        )

        output_csv = (
            OUTPUT_DIRECTORY
            / output_filename
        )

        write_time_resolved_csv(
            output_csv=output_csv,
            sweep_row=sweep_row,
            integration_result=(
                integration_result
            ),
        )

        summary_rows.append(
            {
                "region": (
                    sweep_row["region"]
                ),
                "tunnel_oxide_thickness_nm": (
                    tunnel_oxide_thickness_nm
                ),
                "gate_voltage_V": (
                    gate_voltage_V
                ),
                "current_density_A_cm2": (
                    current_density_A_cm2
                ),
                "total_time_s": (
                    total_time_s
                ),
                "time_step_s": (
                    time_step_s
                ),
                "number_of_steps": (
                    integration_result[
                        "number_of_steps"
                    ]
                ),
                "final_charge_density_C_cm2": (
                    integration_result[
                        "final_charge_density_C_cm2"
                    ]
                ),
                "final_electron_sheet_density_cm2": (
                    integration_result[
                        "final_electron_sheet_density_cm2"
                    ]
                ),
                "numerical_error_C_cm2": (
                    integration_result[
                        "numerical_error_C_cm2"
                    ]
                ),
                "time_resolved_output_csv": (
                    str(output_csv.resolve())
                ),
                "fixed_field_approximation": (
                    True
                ),
            }
        )

        print(
            f"tox = {tunnel_oxide_thickness_nm:.3f} nm, "
            f"VG = {gate_voltage_V:+.3f} V, "
            f"J = {current_density_A_cm2:.6e} A/cm^2, "
            "final electron density = "
            f"{integration_result['final_electron_sheet_density_cm2']:.6e} "
            "cm^-2"
        )

    summary_csv = (
        OUTPUT_DIRECTORY
        / "time_integration_summary.csv"
    )

    write_summary_csv(
        output_csv=summary_csv,
        summary_rows=summary_rows,
    )

    print()
    print("=" * 72)
    print("TIME INTEGRATION COMPLETED")
    print("=" * 72)

    print(
        "Time-resolved CSV directory:\n"
        f"{OUTPUT_DIRECTORY.resolve()}"
    )

    print(
        "Summary CSV:\n"
        f"{summary_csv.resolve()}"
    )

    print()
    print(
        "NOTE: Results use a fixed-field approximation. "
        "Trapped-charge feedback is not included yet."
    )


if __name__ == "__main__":
    main()