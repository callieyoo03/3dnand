# ============================================================
# Time-integration utilities
# MoS2 cylindrical GAA charge-trap memory
#
# Scope
# -----
# This module performs fixed-field time integration using a
# supplied tunneling current density.
#
# Current limitation
# ------------------
# Trapped charge is not yet fed back into the Poisson solution.
# Therefore, the tunneling current density remains constant
# during one integration run.
# ============================================================

import math


# ============================================================
# Fundamental constant
# ============================================================

ELEMENTARY_CHARGE_C = 1.602176634e-19


# ============================================================
# Input validation
# ============================================================

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
# Time-grid generation
# ============================================================

def generate_time_points(
    total_time_s,
    time_step_s,
):
    total_time_s = validate_nonnegative_finite(
        name="total_time_s",
        value=total_time_s,
    )

    time_step_s = validate_positive_finite(
        name="time_step_s",
        value=time_step_s,
    )

    time_points_s = [0.0]

    if total_time_s == 0.0:
        return time_points_s

    current_time_s = 0.0

    while current_time_s < total_time_s:
        remaining_time_s = (
            total_time_s
            - current_time_s
        )

        actual_step_s = min(
            time_step_s,
            remaining_time_s,
        )

        current_time_s += actual_step_s

        time_points_s.append(
            current_time_s
        )

    return time_points_s


# ============================================================
# Fixed-current-density integration
# ============================================================

def integrate_constant_current_density(
    current_density_A_cm2,
    total_time_s,
    time_step_s,
):
    """
    Integrate a constant tunneling current density over time.

    Parameters
    ----------
    current_density_A_cm2 : float
        Tunneling current-density magnitude in A/cm^2.

    total_time_s : float
        Total integration time in seconds.

    time_step_s : float
        Nominal time step in seconds.

    Returns
    -------
    dict
        Summary and time-resolved integration results.
    """

    current_density_A_cm2 = (
        validate_nonnegative_finite(
            name="current_density_A_cm2",
            value=current_density_A_cm2,
        )
    )

    total_time_s = validate_nonnegative_finite(
        name="total_time_s",
        value=total_time_s,
    )

    time_step_s = validate_positive_finite(
        name="time_step_s",
        value=time_step_s,
    )

    time_points_s = generate_time_points(
        total_time_s=total_time_s,
        time_step_s=time_step_s,
    )

    cumulative_charge_density_C_cm2 = 0.0

    cumulative_electron_sheet_density_cm2 = 0.0

    rows = []

    rows.append(
        {
            "time_s": 0.0,
            "time_step_s": 0.0,
            "current_density_A_cm2": (
                current_density_A_cm2
            ),
            "incremental_charge_density_C_cm2": 0.0,
            "cumulative_charge_density_C_cm2": 0.0,
            "cumulative_electron_sheet_density_cm2": 0.0,
        }
    )

    previous_time_s = 0.0

    for current_time_s in time_points_s[1:]:
        actual_step_s = (
            current_time_s
            - previous_time_s
        )

        incremental_charge_density_C_cm2 = (
            current_density_A_cm2
            * actual_step_s
        )

        cumulative_charge_density_C_cm2 += (
            incremental_charge_density_C_cm2
        )

        cumulative_electron_sheet_density_cm2 = (
            cumulative_charge_density_C_cm2
            / ELEMENTARY_CHARGE_C
        )

        rows.append(
            {
                "time_s": current_time_s,
                "time_step_s": actual_step_s,
                "current_density_A_cm2": (
                    current_density_A_cm2
                ),
                "incremental_charge_density_C_cm2": (
                    incremental_charge_density_C_cm2
                ),
                "cumulative_charge_density_C_cm2": (
                    cumulative_charge_density_C_cm2
                ),
                "cumulative_electron_sheet_density_cm2": (
                    cumulative_electron_sheet_density_cm2
                ),
            }
        )

        previous_time_s = current_time_s

    expected_charge_density_C_cm2 = (
        current_density_A_cm2
        * total_time_s
    )

    numerical_error_C_cm2 = abs(
        cumulative_charge_density_C_cm2
        - expected_charge_density_C_cm2
    )

    return {
        "current_density_A_cm2": (
            current_density_A_cm2
        ),
        "total_time_s": total_time_s,
        "requested_time_step_s": time_step_s,
        "number_of_steps": len(rows) - 1,
        "final_charge_density_C_cm2": (
            cumulative_charge_density_C_cm2
        ),
        "final_electron_sheet_density_cm2": (
            cumulative_electron_sheet_density_cm2
        ),
        "expected_charge_density_C_cm2": (
            expected_charge_density_C_cm2
        ),
        "numerical_error_C_cm2": (
            numerical_error_C_cm2
        ),
        "rows": rows,
    }


# ============================================================
# Console reporting
# ============================================================

def print_integration_summary(
    integration_result,
):
    print()
    print("=" * 72)
    print("FIXED-FIELD TIME-INTEGRATION SUMMARY")
    print("=" * 72)

    print(
        "Current density: "
        f"{integration_result['current_density_A_cm2']:.6e} "
        "A/cm^2"
    )

    print(
        "Total time: "
        f"{integration_result['total_time_s']:.6e} s"
    )

    print(
        "Number of steps: "
        f"{integration_result['number_of_steps']}"
    )

    print(
        "Final charge density: "
        f"{integration_result['final_charge_density_C_cm2']:.6e} "
        "C/cm^2"
    )

    print(
        "Final electron sheet density: "
        f"{integration_result['final_electron_sheet_density_cm2']:.6e} "
        "cm^-2"
    )

    print(
        "Integration error: "
        f"{integration_result['numerical_error_C_cm2']:.6e} "
        "C/cm^2"
    )