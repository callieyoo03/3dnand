# ============================================================
# Multi-thickness program-feasibility analysis
#
# Reads all positive-field sweep CSV files in:
#   results/tunnel_oxide_sweep/
#
# Uses:
#   edge_resolved_effective_current_density_A_cm2
#
# Calculates:
#   1. Required program time for target sheet charge
#   2. Injected electron sheet density at selected pulse times
#   3. Best thickness-voltage conditions
#
# Limitations:
#   - Fixed electric field
#   - Constant tunneling current
#   - 100% capture efficiency
#   - No trapped-charge feedback
#   - No trap saturation or detrapping
# ============================================================

import csv
import math
import os
from pathlib import Path


# ============================================================
# Constants
# ============================================================

ELEMENTARY_CHARGE_C = 1.602176634e-19


# ============================================================
# Paths
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

DEFAULT_INPUT_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "tunnel_oxide_sweep"
)

INPUT_DIRECTORY = Path(
    os.environ.get(
        "MULTITHICKNESS_INPUT_DIRECTORY",
        str(DEFAULT_INPUT_DIRECTORY),
    )
)

DEFAULT_OUTPUT_DIRECTORY = (
    SCRIPT_DIRECTORY
    / "results"
    / "multithickness_program_feasibility"
)

OUTPUT_DIRECTORY = Path(
    os.environ.get(
        "MULTITHICKNESS_OUTPUT_DIRECTORY",
        str(DEFAULT_OUTPUT_DIRECTORY),
    )
)


# ============================================================
# Analysis settings
# ============================================================

TARGET_REGION = "TunnelOxide"

TARGET_ELECTRON_SHEET_DENSITIES_CM2 = (
    1.0e10,
    1.0e11,
    1.0e12,
    1.0e13,
)

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

THICKNESS_COLUMN = "tunnel_oxide_thickness_nm"

GATE_VOLTAGE_COLUMN = "gate_voltage_V"

CURRENT_DENSITY_COLUMN = (
    "edge_resolved_effective_current_density_A_cm2"
)

INTERFACE_FIELD_COLUMN = (
    "active_interface_field_area_weighted_abs_V_cm"
)

MAXIMUM_FIELD_COLUMN = (
    "edge_resolved_maximum_local_field_abs_V_cm"
)

BARRIER_HEIGHT_COLUMN = (
    "tunneling_barrier_height_eV"
)

EFFECTIVE_MASS_COLUMN = (
    "tunneling_effective_mass_ratio"
)


# ============================================================
# Validation
# ============================================================

def validate_finite(name, value):
    numeric_value = float(value)

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{name} must be finite. "
            f"Received: {numeric_value}"
        )

    return numeric_value


def validate_nonnegative_finite(name, value):
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


def validate_positive_finite(name, value):
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

def calculate_injected_sheet_density(
    current_density_A_cm2,
    program_time_s,
):
    current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="current_density_A_cm2",
            value=current_density_A_cm2,
        )
    )

    program_time_s = validate_positive_finite(
        name="program_time_s",
        value=program_time_s,
    )

    charge_density_C_cm2 = (
        current_density_A_cm2
        * program_time_s
    )

    return (
        charge_density_C_cm2
        / ELEMENTARY_CHARGE_C
    )


def calculate_required_program_time(
    target_sheet_density_cm2,
    current_density_A_cm2,
):
    target_sheet_density_cm2 = (
        validate_positive_finite(
            name="target_sheet_density_cm2",
            value=target_sheet_density_cm2,
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

    target_charge_density_C_cm2 = (
        target_sheet_density_cm2
        * ELEMENTARY_CHARGE_C
    )

    return (
        target_charge_density_C_cm2
        / current_density_A_cm2
    )


def classify_required_time(required_time_s):
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


def format_number(value):
    if math.isinf(value):
        return "inf"

    return f"{value:.12e}"


# ============================================================
# Input
# ============================================================

def check_required_columns(fieldnames):
    required_columns = {
        REGION_COLUMN,
        THICKNESS_COLUMN,
        GATE_VOLTAGE_COLUMN,
        CURRENT_DENSITY_COLUMN,
        INTERFACE_FIELD_COLUMN,
        MAXIMUM_FIELD_COLUMN,
        BARRIER_HEIGHT_COLUMN,
        EFFECTIVE_MASS_COLUMN,
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


def read_one_sweep_csv(input_csv):
    rows = []

    with input_csv.open(
        mode="r",
        newline="",
        encoding="utf-8-sig",
    ) as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames is None:
            raise RuntimeError(
                f"CSV has no header: {input_csv}"
            )

        check_required_columns(
            reader.fieldnames
        )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            region = row[REGION_COLUMN].strip()

            if region != TARGET_REGION:
                continue

            thickness_nm = (
                validate_positive_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{THICKNESS_COLUMN}"
                    ),
                    value=row[THICKNESS_COLUMN],
                )
            )

            gate_voltage_V = validate_finite(
                name=(
                    f"{input_csv.name} row {row_number} "
                    f"{GATE_VOLTAGE_COLUMN}"
                ),
                value=row[GATE_VOLTAGE_COLUMN],
            )

            current_density_A_cm2 = (
                validate_nonnegative_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{CURRENT_DENSITY_COLUMN}"
                    ),
                    value=row[CURRENT_DENSITY_COLUMN],
                )
            )

            interface_field_V_cm = (
                validate_nonnegative_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{INTERFACE_FIELD_COLUMN}"
                    ),
                    value=row[INTERFACE_FIELD_COLUMN],
                )
            )

            maximum_field_V_cm = (
                validate_nonnegative_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{MAXIMUM_FIELD_COLUMN}"
                    ),
                    value=row[MAXIMUM_FIELD_COLUMN],
                )
            )

            barrier_height_eV = (
                validate_positive_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{BARRIER_HEIGHT_COLUMN}"
                    ),
                    value=row[BARRIER_HEIGHT_COLUMN],
                )
            )

            effective_mass_ratio = (
                validate_positive_finite(
                    name=(
                        f"{input_csv.name} row {row_number} "
                        f"{EFFECTIVE_MASS_COLUMN}"
                    ),
                    value=row[EFFECTIVE_MASS_COLUMN],
                )
            )

            rows.append(
                {
                    "source_csv": input_csv.name,
                    "region": region,
                    "tunnel_oxide_thickness_nm": (
                        thickness_nm
                    ),
                    "gate_voltage_V": (
                        gate_voltage_V
                    ),
                    "edge_resolved_current_density_A_cm2": (
                        current_density_A_cm2
                    ),
                    "interface_field_V_cm": (
                        interface_field_V_cm
                    ),
                    "maximum_local_field_V_cm": (
                        maximum_field_V_cm
                    ),
                    "barrier_height_eV": (
                        barrier_height_eV
                    ),
                    "effective_mass_ratio": (
                        effective_mass_ratio
                    ),
                }
            )

    return rows


def read_all_sweep_csv_files(input_directory):
    input_directory = Path(input_directory)

    if not input_directory.exists():
        raise FileNotFoundError(
            "Tunnel-oxide sweep directory was not found:\n"
            f"{input_directory.resolve()}"
        )

    input_files = sorted(
        input_directory.glob(
            "positive_field_sweep_tox_*.csv"
        )
    )

    if not input_files:
        raise FileNotFoundError(
            "No tunnel-oxide sweep CSV files were found in:\n"
            f"{input_directory.resolve()}"
        )

    all_rows = []

    for input_csv in input_files:
        file_rows = read_one_sweep_csv(
            input_csv=input_csv,
        )

        print(
            f"Loaded {len(file_rows)} TunnelOxide rows "
            f"from {input_csv.name}"
        )

        all_rows.extend(file_rows)

    if not all_rows:
        raise RuntimeError(
            "No TunnelOxide rows were found."
        )

    return all_rows


# ============================================================
# Analysis tables
# ============================================================

def build_required_time_rows(device_rows):
    output_rows = []

    for device_row in device_rows:
        current_density_A_cm2 = (
            device_row[
                "edge_resolved_current_density_A_cm2"
            ]
        )

        for target_density_cm2 in (
            TARGET_ELECTRON_SHEET_DENSITIES_CM2
        ):
            required_time_s = (
                calculate_required_program_time(
                    target_sheet_density_cm2=(
                        target_density_cm2
                    ),
                    current_density_A_cm2=(
                        current_density_A_cm2
                    ),
                )
            )

            output_rows.append(
                {
                    **device_row,
                    "target_electron_sheet_density_cm2": (
                        target_density_cm2
                    ),
                    "required_program_time_s": (
                        format_number(
                            required_time_s
                        )
                    ),
                    "required_time_classification": (
                        classify_required_time(
                            required_time_s
                        )
                    ),
                    "fixed_field_approximation": True,
                    "capture_efficiency_assumed": 1.0,
                }
            )

    return output_rows


def build_injected_density_rows(device_rows):
    output_rows = []

    for device_row in device_rows:
        current_density_A_cm2 = (
            device_row[
                "edge_resolved_current_density_A_cm2"
            ]
        )

        for program_time_s in PROGRAM_PULSE_TIMES_S:
            injected_sheet_density_cm2 = (
                calculate_injected_sheet_density(
                    current_density_A_cm2=(
                        current_density_A_cm2
                    ),
                    program_time_s=program_time_s,
                )
            )

            output_rows.append(
                {
                    **device_row,
                    "program_time_s": (
                        program_time_s
                    ),
                    "injected_charge_density_C_cm2": (
                        current_density_A_cm2
                        * program_time_s
                    ),
                    "injected_electron_sheet_density_cm2": (
                        injected_sheet_density_cm2
                    ),
                    "fixed_field_approximation": True,
                    "capture_efficiency_assumed": 1.0,
                }
            )

    return output_rows


def build_best_condition_rows(
    required_time_rows,
):
    output_rows = []

    target_densities = sorted(
        {
            row[
                "target_electron_sheet_density_cm2"
            ]
            for row in required_time_rows
        }
    )

    for target_density_cm2 in target_densities:
        candidates = [
            row
            for row in required_time_rows
            if row[
                "target_electron_sheet_density_cm2"
            ]
            == target_density_cm2
            and row[
                "required_program_time_s"
            ]
            != "inf"
        ]

        candidates.sort(
            key=lambda row: float(
                row["required_program_time_s"]
            )
        )

        for rank, candidate in enumerate(
            candidates[:10],
            start=1,
        ):
            output_rows.append(
                {
                    "rank": rank,
                    **candidate,
                }
            )

    return output_rows


# ============================================================
# CSV output
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
# Console summary
# ============================================================

def print_condition_summary(device_rows):
    positive_rows = [
        row
        for row in device_rows
        if row[
            "edge_resolved_current_density_A_cm2"
        ] > 0.0
    ]

    positive_rows.sort(
        key=lambda row: row[
            "edge_resolved_current_density_A_cm2"
        ],
        reverse=True,
    )

    print()
    print("=" * 84)
    print("TOP CURRENT-DENSITY CONDITIONS")
    print("=" * 84)

    for rank, row in enumerate(
        positive_rows[:10],
        start=1,
    ):
        print(
            f"{rank:2d}. "
            f"tox={row['tunnel_oxide_thickness_nm']:.3f} nm, "
            f"VG={row['gate_voltage_V']:+.3f} V, "
            f"J={row['edge_resolved_current_density_A_cm2']:.6e} A/cm^2, "
            f"Eavg={row['interface_field_V_cm']:.6e} V/cm"
        )


def print_target_summary(
    required_time_rows,
    target_density_cm2,
):
    selected_rows = [
        row
        for row in required_time_rows
        if row[
            "target_electron_sheet_density_cm2"
        ]
        == target_density_cm2
        and row[
            "required_program_time_s"
        ]
        != "inf"
    ]

    selected_rows.sort(
        key=lambda row: float(
            row["required_program_time_s"]
        )
    )

    print()
    print("=" * 84)
    print(
        "FASTEST CONDITIONS FOR "
        f"Nsheet={target_density_cm2:.3e} cm^-2"
    )
    print("=" * 84)

    for rank, row in enumerate(
        selected_rows[:10],
        start=1,
    ):
        required_time_s = float(
            row[
                "required_program_time_s"
            ]
        )

        print(
            f"{rank:2d}. "
            f"tox={row['tunnel_oxide_thickness_nm']:.3f} nm, "
            f"VG={row['gate_voltage_V']:+.3f} V, "
            f"J={row['edge_resolved_current_density_A_cm2']:.6e} A/cm^2, "
            f"t={required_time_s:.6e} s "
            f"[{row['required_time_classification']}]"
        )


# ============================================================
# Main
# ============================================================

def main():
    device_rows = read_all_sweep_csv_files(
        input_directory=INPUT_DIRECTORY,
    )

    required_time_rows = (
        build_required_time_rows(
            device_rows=device_rows,
        )
    )

    injected_density_rows = (
        build_injected_density_rows(
            device_rows=device_rows,
        )
    )

    best_condition_rows = (
        build_best_condition_rows(
            required_time_rows=(
                required_time_rows
            ),
        )
    )

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    required_time_csv = (
        OUTPUT_DIRECTORY
        / "multithickness_required_program_time.csv"
    )

    injected_density_csv = (
        OUTPUT_DIRECTORY
        / "multithickness_injected_sheet_density.csv"
    )

    best_condition_csv = (
        OUTPUT_DIRECTORY
        / "multithickness_best_conditions.csv"
    )

    common_fields = [
        "source_csv",
        "region",
        "tunnel_oxide_thickness_nm",
        "gate_voltage_V",
        "edge_resolved_current_density_A_cm2",
        "interface_field_V_cm",
        "maximum_local_field_V_cm",
        "barrier_height_eV",
        "effective_mass_ratio",
    ]

    write_csv(
        output_csv=required_time_csv,
        rows=required_time_rows,
        fieldnames=(
            common_fields
            + [
                "target_electron_sheet_density_cm2",
                "required_program_time_s",
                "required_time_classification",
                "fixed_field_approximation",
                "capture_efficiency_assumed",
            ]
        ),
    )

    write_csv(
        output_csv=injected_density_csv,
        rows=injected_density_rows,
        fieldnames=(
            common_fields
            + [
                "program_time_s",
                "injected_charge_density_C_cm2",
                "injected_electron_sheet_density_cm2",
                "fixed_field_approximation",
                "capture_efficiency_assumed",
            ]
        ),
    )

    write_csv(
        output_csv=best_condition_csv,
        rows=best_condition_rows,
        fieldnames=(
            ["rank"]
            + common_fields
            + [
                "target_electron_sheet_density_cm2",
                "required_program_time_s",
                "required_time_classification",
                "fixed_field_approximation",
                "capture_efficiency_assumed",
            ]
        ),
    )

    print()
    print("=" * 84)
    print("MULTI-THICKNESS PROGRAM-FEASIBILITY ANALYSIS")
    print("=" * 84)

    print(
        "Input directory:\n"
        f"{INPUT_DIRECTORY.resolve()}"
    )

    print(
        "Number of thickness-voltage conditions: "
        f"{len(device_rows)}"
    )

    print_condition_summary(
        device_rows=device_rows,
    )

    print_target_summary(
        required_time_rows=required_time_rows,
        target_density_cm2=1.0e12,
    )

    print()
    print("=" * 84)
    print("ANALYSIS COMPLETED")
    print("=" * 84)

    print(
        "Required-time results:\n"
        f"{required_time_csv.resolve()}"
    )

    print(
        "Injected-density results:\n"
        f"{injected_density_csv.resolve()}"
    )

    print(
        "Best-condition ranking:\n"
        f"{best_condition_csv.resolve()}"
    )

    print()
    print(
        "NOTE: These are comparative fixed-field estimates. "
        "They are not final program-performance predictions."
    )


if __name__ == "__main__":
    main()