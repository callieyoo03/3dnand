# ============================================================
# Program-feasibility analysis
#
# Purpose
# -------
# Read the fixed-field time-integration summary and estimate:
#
# 1. Time required to reach target electron sheet densities
# 2. Current density required to reach each target within
#    selected program-pulse times
# 3. Ratio between required and simulated current densities
#
# Important limitation
# --------------------
# This analysis assumes a constant tunneling current density.
# Trapped-charge feedback, trap saturation, capture efficiency,
# detrapping, and field redistribution are not included.
# ============================================================

import csv
import math
import os
from pathlib import Path


# ============================================================
# Fundamental constant
# ============================================================

ELEMENTARY_CHARGE_C = 1.602176634e-19


# ============================================================
# Project paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_INPUT_CSV = (
    SCRIPT_DIRECTORY
    / "results"
    / "time_integration"
    / "time_integration_summary.csv"
)

INPUT_CSV = Path(
    os.environ.get(
        "PROGRAM_FEASIBILITY_INPUT_CSV",
        str(DEFAULT_INPUT_CSV),
    )
)

DEFAULT_OUTPUT_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "program_feasibility"
)

OUTPUT_DIRECTORY = Path(
    os.environ.get(
        "PROGRAM_FEASIBILITY_OUTPUT_DIRECTORY",
        str(DEFAULT_OUTPUT_DIRECTORY),
    )
)


# ============================================================
# Analysis conditions
# ============================================================

# Candidate target trapped-electron sheet densities.
#
# These values are currently analysis targets only.
# They must later be replaced or narrowed using literature
# values and the selected trap model.
TARGET_ELECTRON_SHEET_DENSITIES_CM2 = (
    1.0e10,
    1.0e11,
    1.0e12,
    1.0e13,
)

# Candidate program-pulse durations.
PROGRAM_PULSE_TIMES_S = (
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0,
)


# ============================================================
# CSV columns
# ============================================================

REGION_COLUMN = "region"

THICKNESS_COLUMN = (
    "tunnel_oxide_thickness_nm"
)

GATE_VOLTAGE_COLUMN = (
    "gate_voltage_V"
)

CURRENT_DENSITY_COLUMN = (
    "current_density_A_cm2"
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
# Physical calculations
# ============================================================

def sheet_density_to_charge_density(
    electron_sheet_density_cm2,
):
    """
    Convert electron sheet density [cm^-2]
    to charge-density magnitude [C/cm^2].
    """

    electron_sheet_density_cm2 = (
        validate_nonnegative_finite(
            name="electron_sheet_density_cm2",
            value=electron_sheet_density_cm2,
        )
    )

    return (
        electron_sheet_density_cm2
        * ELEMENTARY_CHARGE_C
    )


def calculate_required_time(
    target_electron_sheet_density_cm2,
    current_density_A_cm2,
):
    """
    Calculate time needed to reach the target sheet density.

    Q = J * t
    t = q * N_sheet / J
    """

    target_charge_density_C_cm2 = (
        sheet_density_to_charge_density(
            target_electron_sheet_density_cm2
        )
    )

    current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="current_density_A_cm2",
            value=current_density_A_cm2,
        )
    )

    if current_density_A_cm2 == 0.0:
        return math.inf

    return (
        target_charge_density_C_cm2
        / current_density_A_cm2
    )


def calculate_required_current_density(
    target_electron_sheet_density_cm2,
    program_time_s,
):
    """
    Calculate current density required to reach a target
    sheet density within the selected program time.

    J_required = q * N_sheet / t_program
    """

    target_charge_density_C_cm2 = (
        sheet_density_to_charge_density(
            target_electron_sheet_density_cm2
        )
    )

    program_time_s = validate_positive_finite(
        name="program_time_s",
        value=program_time_s,
    )

    return (
        target_charge_density_C_cm2
        / program_time_s
    )


def calculate_current_shortfall_ratio(
    required_current_density_A_cm2,
    simulated_current_density_A_cm2,
):
    """
    Return J_required / J_simulated.

    A value greater than 1 means the simulated current is
    smaller than the required current.
    """

    required_current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="required_current_density_A_cm2",
            value=required_current_density_A_cm2,
        )
    )

    simulated_current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="simulated_current_density_A_cm2",
            value=simulated_current_density_A_cm2,
        )
    )

    if simulated_current_density_A_cm2 == 0.0:
        if required_current_density_A_cm2 == 0.0:
            return 1.0

        return math.inf

    return (
        required_current_density_A_cm2
        / simulated_current_density_A_cm2
    )


# ============================================================
# Formatting
# ============================================================

def format_scientific_or_infinity(
    value,
):
    if math.isinf(value):
        return "inf"

    return f"{value:.12e}"


def classify_required_time(
    required_time_s,
):
    """
    Descriptive classification only.

    This does not determine whether the device is acceptable.
    Final criteria must be defined from target specifications
    and literature.
    """

    if math.isinf(required_time_s):
        return "unreachable_at_zero_current"

    if required_time_s <= 1.0e-6:
        return "within_1_us"

    if required_time_s <= 1.0e-3:
        return "within_1_ms"

    if required_time_s <= 1.0:
        return "within_1_s"

    if required_time_s <= 3600.0:
        return "within_1_hour"

    if required_time_s <= 86400.0:
        return "within_1_day"

    if required_time_s <= 31557600.0:
        return "within_1_year"

    return "longer_than_1_year"


# ============================================================
# Input
# ============================================================

def check_required_columns(
    fieldnames,
):
    required_columns = {
        REGION_COLUMN,
        THICKNESS_COLUMN,
        GATE_VOLTAGE_COLUMN,
        CURRENT_DENSITY_COLUMN,
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


def read_summary_rows(
    input_csv,
):
    input_csv = Path(input_csv)

    if not input_csv.exists():
        raise FileNotFoundError(
            "Time-integration summary CSV was not found:\n"
            f"{input_csv.resolve()}"
        )

    rows = []

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

            thickness_nm = (
                validate_positive_finite(
                    name=(
                        f"row {row_number} "
                        f"{THICKNESS_COLUMN}"
                    ),
                    value=row[THICKNESS_COLUMN],
                )
            )

            gate_voltage_V = validate_finite(
                name=(
                    f"row {row_number} "
                    f"{GATE_VOLTAGE_COLUMN}"
                ),
                value=row[GATE_VOLTAGE_COLUMN],
            )

            current_density_A_cm2 = (
                validate_nonnegative_finite(
                    name=(
                        f"row {row_number} "
                        f"{CURRENT_DENSITY_COLUMN}"
                    ),
                    value=row[CURRENT_DENSITY_COLUMN],
                )
            )

            rows.append(
                {
                    "region": region,
                    "tunnel_oxide_thickness_nm": (
                        thickness_nm
                    ),
                    "gate_voltage_V": (
                        gate_voltage_V
                    ),
                    "current_density_A_cm2": (
                        current_density_A_cm2
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "Input CSV contains no data rows."
        )

    return rows


# ============================================================
# Analysis
# ============================================================

def build_required_time_rows(
    summary_rows,
):
    output_rows = []

    for summary_row in summary_rows:
        for target_density_cm2 in (
            TARGET_ELECTRON_SHEET_DENSITIES_CM2
        ):
            required_time_s = (
                calculate_required_time(
                    target_electron_sheet_density_cm2=(
                        target_density_cm2
                    ),
                    current_density_A_cm2=(
                        summary_row[
                            "current_density_A_cm2"
                        ]
                    ),
                )
            )

            target_charge_density_C_cm2 = (
                sheet_density_to_charge_density(
                    target_density_cm2
                )
            )

            output_rows.append(
                {
                    "region": (
                        summary_row["region"]
                    ),
                    "tunnel_oxide_thickness_nm": (
                        summary_row[
                            "tunnel_oxide_thickness_nm"
                        ]
                    ),
                    "gate_voltage_V": (
                        summary_row[
                            "gate_voltage_V"
                        ]
                    ),
                    "simulated_current_density_A_cm2": (
                        summary_row[
                            "current_density_A_cm2"
                        ]
                    ),
                    "target_electron_sheet_density_cm2": (
                        target_density_cm2
                    ),
                    "target_charge_density_C_cm2": (
                        target_charge_density_C_cm2
                    ),
                    "required_program_time_s": (
                        format_scientific_or_infinity(
                            required_time_s
                        )
                    ),
                    "required_time_classification": (
                        classify_required_time(
                            required_time_s
                        )
                    ),
                    "fixed_field_approximation": True,
                }
            )

    return output_rows


def build_required_current_rows(
    summary_rows,
):
    output_rows = []

    for summary_row in summary_rows:
        simulated_current_density_A_cm2 = (
            summary_row[
                "current_density_A_cm2"
            ]
        )

        for target_density_cm2 in (
            TARGET_ELECTRON_SHEET_DENSITIES_CM2
        ):
            for program_time_s in (
                PROGRAM_PULSE_TIMES_S
            ):
                required_current_density_A_cm2 = (
                    calculate_required_current_density(
                        target_electron_sheet_density_cm2=(
                            target_density_cm2
                        ),
                        program_time_s=(
                            program_time_s
                        ),
                    )
                )

                shortfall_ratio = (
                    calculate_current_shortfall_ratio(
                        required_current_density_A_cm2=(
                            required_current_density_A_cm2
                        ),
                        simulated_current_density_A_cm2=(
                            simulated_current_density_A_cm2
                        ),
                    )
                )

                output_rows.append(
                    {
                        "region": (
                            summary_row["region"]
                        ),
                        "tunnel_oxide_thickness_nm": (
                            summary_row[
                                "tunnel_oxide_thickness_nm"
                            ]
                        ),
                        "gate_voltage_V": (
                            summary_row[
                                "gate_voltage_V"
                            ]
                        ),
                        "target_electron_sheet_density_cm2": (
                            target_density_cm2
                        ),
                        "program_time_s": (
                            program_time_s
                        ),
                        "simulated_current_density_A_cm2": (
                            simulated_current_density_A_cm2
                        ),
                        "required_current_density_A_cm2": (
                            required_current_density_A_cm2
                        ),
                        "required_to_simulated_current_ratio": (
                            format_scientific_or_infinity(
                                shortfall_ratio
                            )
                        ),
                        "simulated_current_meets_requirement": (
                            simulated_current_density_A_cm2
                            >= required_current_density_A_cm2
                        ),
                        "fixed_field_approximation": True,
                    }
                )

    return output_rows


# ============================================================
# Output
# ============================================================

def write_csv(
    output_csv,
    rows,
    fieldnames,
):
    output_csv = Path(output_csv)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        writer.writerows(rows)


# ============================================================
# Console report
# ============================================================

def print_highest_voltage_summary(
    summary_rows,
):
    highest_voltage_row = max(
        summary_rows,
        key=lambda row: row["gate_voltage_V"],
    )

    gate_voltage_V = highest_voltage_row[
        "gate_voltage_V"
    ]

    thickness_nm = highest_voltage_row[
        "tunnel_oxide_thickness_nm"
    ]

    current_density_A_cm2 = (
        highest_voltage_row[
            "current_density_A_cm2"
        ]
    )

    print()
    print("=" * 72)
    print("HIGHEST-VOLTAGE CONDITION")
    print("=" * 72)

    print(
        f"TunnelOxide thickness: "
        f"{thickness_nm:.3f} nm"
    )

    print(
        f"Gate voltage: "
        f"{gate_voltage_V:+.3f} V"
    )

    print(
        "Simulated current density: "
        f"{current_density_A_cm2:.6e} A/cm^2"
    )

    print()
    print(
        "Required fixed-field program time:"
    )

    for target_density_cm2 in (
        TARGET_ELECTRON_SHEET_DENSITIES_CM2
    ):
        required_time_s = calculate_required_time(
            target_electron_sheet_density_cm2=(
                target_density_cm2
            ),
            current_density_A_cm2=(
                current_density_A_cm2
            ),
        )

        print(
            f"  Nsheet = {target_density_cm2:.3e} cm^-2"
            f" -> t = "
            f"{format_scientific_or_infinity(required_time_s)} s"
            f" [{classify_required_time(required_time_s)}]"
        )


# ============================================================
# Main
# ============================================================

def main():
    summary_rows = read_summary_rows(
        input_csv=INPUT_CSV,
    )

    required_time_rows = (
        build_required_time_rows(
            summary_rows=summary_rows,
        )
    )

    required_current_rows = (
        build_required_current_rows(
            summary_rows=summary_rows,
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_time_csv = (
        OUTPUT_DIRECTORY
        / "required_program_time.csv"
    )

    required_current_csv = (
        OUTPUT_DIRECTORY
        / "required_current_density.csv"
    )

    write_csv(
        output_csv=required_time_csv,
        rows=required_time_rows,
        fieldnames=[
            "region",
            "tunnel_oxide_thickness_nm",
            "gate_voltage_V",
            "simulated_current_density_A_cm2",
            "target_electron_sheet_density_cm2",
            "target_charge_density_C_cm2",
            "required_program_time_s",
            "required_time_classification",
            "fixed_field_approximation",
        ],
    )

    write_csv(
        output_csv=required_current_csv,
        rows=required_current_rows,
        fieldnames=[
            "region",
            "tunnel_oxide_thickness_nm",
            "gate_voltage_V",
            "target_electron_sheet_density_cm2",
            "program_time_s",
            "simulated_current_density_A_cm2",
            "required_current_density_A_cm2",
            "required_to_simulated_current_ratio",
            "simulated_current_meets_requirement",
            "fixed_field_approximation",
        ],
    )

    print()
    print("=" * 72)
    print("PROGRAM-FEASIBILITY ANALYSIS")
    print("=" * 72)

    print(
        "Input CSV:\n"
        f"{INPUT_CSV.resolve()}"
    )

    print(
        "Number of device conditions: "
        f"{len(summary_rows)}"
    )

    print(
        "Target sheet-density conditions: "
        f"{len(TARGET_ELECTRON_SHEET_DENSITIES_CM2)}"
    )

    print(
        "Program-pulse conditions: "
        f"{len(PROGRAM_PULSE_TIMES_S)}"
    )

    print_highest_voltage_summary(
        summary_rows=summary_rows,
    )

    print()
    print("=" * 72)
    print("ANALYSIS COMPLETED")
    print("=" * 72)

    print(
        "Required-program-time CSV:\n"
        f"{required_time_csv.resolve()}"
    )

    print(
        "Required-current-density CSV:\n"
        f"{required_current_csv.resolve()}"
    )

    print()
    print(
        "NOTE: Target sheet densities are provisional analysis "
        "conditions, not finalized device specifications."
    )

    print(
        "NOTE: Results assume constant tunneling current and "
        "100% electron capture efficiency."
    )


if __name__ == "__main__":
    main()