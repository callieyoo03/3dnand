# ============================================================
# Electric-field extraction utilities
# MoS2 cylindrical GAA charge-trap memory
#
# Purpose:
#   - Read DEVSIM ElectricField edge-model values
#   - Calculate field statistics for dielectric regions
#
# Important:
#   The present version evaluates all edges in each region.
#   A later step will separate radial and axial edges.
# ============================================================

from devsim import (
    get_edge_model_list,
    get_edge_model_values,
)


# ============================================================
# Regions used for dielectric field analysis
# ============================================================

DIELECTRIC_REGIONS = (
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
)


# ============================================================
# Basic utility functions
# ============================================================

def validate_region_field_model(
    device,
    region,
    field_model="ElectricField",
):
    """
    Confirm that the requested edge model exists.

    Parameters
    ----------
    device : str
        DEVSIM device name.

    region : str
        DEVSIM region name.

    field_model : str
        Edge model to inspect.

    Raises
    ------
    RuntimeError
        If the requested edge model does not exist.
    """

    edge_models = get_edge_model_list(
        device=device,
        region=region,
    )

    if field_model not in edge_models:
        raise RuntimeError(
            f'Edge model "{field_model}" is missing '
            f'in region "{region}".'
        )


def get_region_field_values(
    device,
    region,
    field_model="ElectricField",
):
    """
    Return electric-field values from one DEVSIM region.

    Returns
    -------
    list[float]
        Edge-model values in V/cm.
    """

    validate_region_field_model(
        device=device,
        region=region,
        field_model=field_model,
    )

    values = get_edge_model_values(
        device=device,
        region=region,
        name=field_model,
    )

    values = [
        float(value)
        for value in values
    ]

    if not values:
        raise RuntimeError(
            f'No values were returned for "{field_model}" '
            f'in region "{region}".'
        )

    return values


def calculate_field_statistics(
    values,
):
    """
    Calculate signed and absolute electric-field statistics.

    Parameters
    ----------
    values : iterable[float]
        Electric-field values in V/cm.

    Returns
    -------
    dict
        Minimum, maximum, mean, absolute mean,
        and maximum absolute field.
    """

    values = [
        float(value)
        for value in values
    ]

    if not values:
        raise ValueError(
            "Electric-field value list is empty."
        )

    absolute_values = [
        abs(value)
        for value in values
    ]

    number_of_values = len(values)

    return {
        "edge_count": number_of_values,
        "field_min_V_cm": min(values),
        "field_max_V_cm": max(values),
        "field_mean_signed_V_cm": (
            sum(values) / number_of_values
        ),
        "field_mean_abs_V_cm": (
            sum(absolute_values) / number_of_values
        ),
        "field_max_abs_V_cm": max(
            absolute_values
        ),
    }


def get_region_field_statistics(
    device,
    region,
    field_model="ElectricField",
):
    """
    Read and summarize one region's electric field.
    """

    values = get_region_field_values(
        device=device,
        region=region,
        field_model=field_model,
    )

    statistics = calculate_field_statistics(
        values=values,
    )

    statistics["region"] = region
    statistics["field_model"] = field_model

    return statistics


def get_dielectric_field_statistics(
    device,
):
    """
    Return electric-field statistics for all dielectric layers.

    Regions:
        - TunnelOxide
        - ChargeTrap
        - BlockingOxide
    """

    results = {}

    for region in DIELECTRIC_REGIONS:
        results[region] = (
            get_region_field_statistics(
                device=device,
                region=region,
            )
        )

    return results


def print_region_field_statistics(
    statistics,
):
    """
    Print one region's electric-field statistics.
    """

    print(
        f'Region: {statistics["region"]}'
    )

    print(
        f'  edge count       = '
        f'{statistics["edge_count"]}'
    )

    print(
        f'  minimum field    = '
        f'{statistics["field_min_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  maximum field    = '
        f'{statistics["field_max_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  signed mean      = '
        f'{statistics["field_mean_signed_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  absolute mean    = '
        f'{statistics["field_mean_abs_V_cm"]:.6e} V/cm'
    )

    print(
        f'  maximum absolute = '
        f'{statistics["field_max_abs_V_cm"]:.6e} V/cm'
    )


def print_dielectric_field_statistics(
    device,
):
    """
    Read and print all dielectric-region field statistics.
    """

    results = get_dielectric_field_statistics(
        device=device,
    )

    print()
    print("=" * 70)
    print("DIELECTRIC ELECTRIC-FIELD STATISTICS")
    print("=" * 70)

    for region in DIELECTRIC_REGIONS:
        print()
        print_region_field_statistics(
            statistics=results[region],
        )

    return results