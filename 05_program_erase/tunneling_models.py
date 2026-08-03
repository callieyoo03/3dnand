# ============================================================
# Tunneling models
# MoS2 / Al2O3 / HfO2 charge-trap memory
#
# Implemented:
#   - Fowler-Nordheim tunneling
#
# Future models:
#   - Direct tunneling
#   - Trap-assisted tunneling
#
# Unit convention
# ---------------
# Input electric field:
#     V/cm
#
# Returned current density:
#     A/cm^2
# ============================================================

import math

import tunneling_parameters as tp


# ============================================================
# Utility functions
# ============================================================

def validate_electric_field(
    electric_field_V_cm,
):
    """
    Validate and convert an electric-field input.

    Parameters
    ----------
    electric_field_V_cm : float
        Signed or unsigned electric field in V/cm.

    Returns
    -------
    float
        Validated electric field in V/cm.
    """

    electric_field_V_cm = float(
        electric_field_V_cm
    )

    if not math.isfinite(
        electric_field_V_cm
    ):
        raise ValueError(
            "Electric field must be finite."
        )

    return electric_field_V_cm


def electric_field_V_cm_to_V_m(
    electric_field_V_cm,
):
    """
    Convert electric field from V/cm to V/m.
    """

    electric_field_V_cm = (
        validate_electric_field(
            electric_field_V_cm
        )
    )

    return (
        electric_field_V_cm
        * tp.V_CM_TO_V_M
    )


# ============================================================
# Fowler-Nordheim tunneling
# ============================================================

def fowler_nordheim_exponent(
    electric_field_V_cm,
):
    """
    Calculate the positive Fowler-Nordheim exponent magnitude.

    The tunneling expression uses:

        exp(-exponent)

    Parameters
    ----------
    electric_field_V_cm : float
        Electric-field magnitude or signed field in V/cm.

    Returns
    -------
    float
        Dimensionless positive exponent magnitude.
    """

    electric_field_V_cm = abs(
        validate_electric_field(
            electric_field_V_cm
        )
    )

    if (
        electric_field_V_cm
        < tp.MINIMUM_FIELD_V_CM
    ):
        return math.inf

    electric_field_V_m = (
        electric_field_V_cm_to_V_m(
            electric_field_V_cm
        )
    )

    return (
        tp.FN_B_SI
        / electric_field_V_m
    )


def fowler_nordheim_current_density(
    electric_field_V_cm,
):
    """
    Calculate Fowler-Nordheim current density.

    Model
    -----
    J = A * E^2 * exp(-B/E)

    Parameters
    ----------
    electric_field_V_cm : float
        Signed or unsigned tunnel-oxide electric field in V/cm.

        The current-density magnitude depends on |E|.

    Returns
    -------
    float
        Fowler-Nordheim current-density magnitude in A/cm^2.
    """

    electric_field_V_cm = abs(
        validate_electric_field(
            electric_field_V_cm
        )
    )

    if (
        electric_field_V_cm
        < tp.MINIMUM_FIELD_V_CM
    ):
        return 0.0

    electric_field_V_m = (
        electric_field_V_cm_to_V_m(
            electric_field_V_cm
        )
    )

    exponent_magnitude = (
        tp.FN_B_SI
        / electric_field_V_m
    )

    if (
        exponent_magnitude
        >= tp.MAXIMUM_EXPONENT_MAGNITUDE
    ):
        return 0.0

    current_density_A_m2 = (
        tp.FN_A_SI
        * electric_field_V_m**2
        * math.exp(
            -exponent_magnitude
        )
    )

    current_density_A_cm2 = (
        current_density_A_m2
        * tp.A_M2_TO_A_CM2
    )

    return current_density_A_cm2


def signed_fowler_nordheim_current_density(
    electric_field_V_cm,
):
    """
    Return a signed Fowler-Nordheim current density.

    Sign convention
    ---------------
    The returned current-density sign follows the supplied
    electric-field sign.

    In the current cylindrical coordinate convention:

        positive field:
            cylinder center -> gate

        negative field:
            gate -> cylinder center

    The magnitude is calculated from |E|.
    """

    electric_field_V_cm = (
        validate_electric_field(
            electric_field_V_cm
        )
    )

    if electric_field_V_cm == 0.0:
        return 0.0

    current_density_magnitude = (
        fowler_nordheim_current_density(
            electric_field_V_cm
        )
    )

    direction_sign = (
        1.0
        if electric_field_V_cm > 0.0
        else -1.0
    )

    return (
        direction_sign
        * current_density_magnitude
    )


# ============================================================
# Backward-compatible Fowler-Nordheim function
# ============================================================

def fowler_nordheim_current(
    Eox,
):
    """
    Backward-compatible wrapper.

    Parameters
    ----------
    Eox : float
        Electric field in V/cm.

    Returns
    -------
    float
        Current density in A/cm^2.

    Notes
    -----
    Despite the historical function name, this function returns
    current density rather than total current.
    """

    return fowler_nordheim_current_density(
        electric_field_V_cm=Eox,
    )


# ============================================================
# Direct tunneling
# ============================================================

def direct_tunneling_current_density(
    electric_field_V_cm,
):
    """
    Direct-tunneling placeholder.

    Returns
    -------
    float
        Zero until a direct-tunneling model is implemented.
    """

    validate_electric_field(
        electric_field_V_cm
    )

    return 0.0


def direct_tunneling_current(
    Eox,
):
    """
    Backward-compatible direct-tunneling wrapper.
    """

    return direct_tunneling_current_density(
        electric_field_V_cm=Eox,
    )


# ============================================================
# Trap-assisted tunneling
# ============================================================

def trap_assisted_tunneling_current_density(
    electric_field_V_cm,
):
    """
    Trap-assisted-tunneling placeholder.

    Returns
    -------
    float
        Zero until a TAT model is implemented.
    """

    validate_electric_field(
        electric_field_V_cm
    )

    return 0.0


def tat_current(
    Eox,
):
    """
    Backward-compatible TAT wrapper.
    """

    return (
        trap_assisted_tunneling_current_density(
            electric_field_V_cm=Eox,
        )
    )


# ============================================================
# Total tunneling current density
# ============================================================

def total_tunneling_current_density(
    electric_field_V_cm,
):
    """
    Calculate total tunneling current-density magnitude.

    At the present stage, only Fowler-Nordheim tunneling
    contributes.
    """

    return (
        fowler_nordheim_current_density(
            electric_field_V_cm
        )
        + direct_tunneling_current_density(
            electric_field_V_cm
        )
        + trap_assisted_tunneling_current_density(
            electric_field_V_cm
        )
    )


def total_tunneling_current(
    Eox,
):
    """
    Backward-compatible total-current wrapper.

    Despite the historical name, the return value is current
    density in A/cm^2.
    """

    return total_tunneling_current_density(
        electric_field_V_cm=Eox,
    )


# ============================================================
# Result packaging
# ============================================================

def evaluate_tunneling_current_density(
    electric_field_V_cm,
):
    """
    Evaluate all currently defined tunneling components.

    Returns
    -------
    dict
        Electric field, exponent, and current densities.
    """

    electric_field_V_cm = (
        validate_electric_field(
            electric_field_V_cm
        )
    )

    field_magnitude_V_cm = abs(
        electric_field_V_cm
    )

    fn_exponent = (
        fowler_nordheim_exponent(
            field_magnitude_V_cm
        )
    )

    fn_current_density = (
        fowler_nordheim_current_density(
            field_magnitude_V_cm
        )
    )

    fn_signed_current_density = (
        signed_fowler_nordheim_current_density(
            electric_field_V_cm
        )
    )

    direct_current_density = (
        direct_tunneling_current_density(
            field_magnitude_V_cm
        )
    )

    tat_current_density_value = (
        trap_assisted_tunneling_current_density(
            field_magnitude_V_cm
        )
    )

    total_current_density = (
        fn_current_density
        + direct_current_density
        + tat_current_density_value
    )

    return {
        "electric_field_signed_V_cm": (
            electric_field_V_cm
        ),

        "electric_field_abs_V_cm": (
            field_magnitude_V_cm
        ),

        "fowler_nordheim_exponent": (
            fn_exponent
        ),

        "fowler_nordheim_current_density_A_cm2": (
            fn_current_density
        ),

        "signed_fowler_nordheim_current_density_A_cm2": (
            fn_signed_current_density
        ),

        "direct_tunneling_current_density_A_cm2": (
            direct_current_density
        ),

        "trap_assisted_tunneling_current_density_A_cm2": (
            tat_current_density_value
        ),

        "total_tunneling_current_density_A_cm2": (
            total_current_density
        ),
    }


def print_tunneling_result(
    result,
):
    """
    Print one tunneling-current-density result.
    """

    print()
    print("=" * 70)
    print("TUNNELING CURRENT-DENSITY RESULT")
    print("=" * 70)

    print(
        f'Signed oxide field = '
        f'{result["electric_field_signed_V_cm"]:+.6e} V/cm'
    )

    print(
        f'Absolute oxide field = '
        f'{result["electric_field_abs_V_cm"]:.6e} V/cm'
    )

    exponent_value = (
        result["fowler_nordheim_exponent"]
    )

    if math.isinf(
        exponent_value
    ):
        exponent_text = "infinity"
    else:
        exponent_text = (
            f"{exponent_value:.6e}"
        )

    print(
        f'FN exponent = {exponent_text}'
    )

    print(
        f'FN current density = '
        f'{result["fowler_nordheim_current_density_A_cm2"]:.6e} '
        f'A/cm^2'
    )

    print(
        f'Signed FN current density = '
        f'{result["signed_fowler_nordheim_current_density_A_cm2"]:+.6e} '
        f'A/cm^2'
    )

    print(
        f'Total tunneling current density = '
        f'{result["total_tunneling_current_density_A_cm2"]:.6e} '
        f'A/cm^2'
    )