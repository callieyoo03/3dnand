# ============================================================
# Electric-field extraction utilities
# MoS2 cylindrical GAA charge-trap memory
#
# Coordinate interpretation:
#   x -> radial direction, r
#   y -> channel direction, z
#
# This module:
#   1. Reads ElectricField edge-model values
#   2. Separates radial, axial, and diagonal edges
#   3. Converts the radial field to an outward-directed value
#   4. Calculates field statistics
# ============================================================

from devsim import (
    edge_from_node_model,
    get_edge_model_list,
    get_edge_model_values,
    get_node_model_list,
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
# Numerical settings
# ============================================================

EDGE_DIRECTION_TOLERANCE_CM = 1.0e-15


# ============================================================
# Model validation
# ============================================================

def validate_region_field_model(
    device,
    region,
    field_model="ElectricField",
):
    """
    Confirm that an electric-field edge model exists.
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


def ensure_coordinate_edge_models(
    device,
    region,
):
    """
    Ensure that edge-endpoint coordinate models exist.

    Required models:
        x@n0, x@n1
        y@n0, y@n1

    In the cylindrical 2D mesh:
        x = radial coordinate r
        y = axial coordinate z
    """

    node_models = get_node_model_list(
        device=device,
        region=region,
    )

    for coordinate_name in ("x", "y"):
        if coordinate_name not in node_models:
            raise RuntimeError(
                f'Coordinate node model "{coordinate_name}" '
                f'is missing in region "{region}".'
            )

    edge_models = get_edge_model_list(
        device=device,
        region=region,
    )

    x_models_exist = (
        "x@n0" in edge_models
        and "x@n1" in edge_models
    )

    if not x_models_exist:
        edge_from_node_model(
            device=device,
            region=region,
            node_model="x",
        )

    edge_models = get_edge_model_list(
        device=device,
        region=region,
    )

    y_models_exist = (
        "y@n0" in edge_models
        and "y@n1" in edge_models
    )

    if not y_models_exist:
        edge_from_node_model(
            device=device,
            region=region,
            node_model="y",
        )


# ============================================================
# Raw edge-data extraction
# ============================================================

def get_edge_model_values_as_float(
    device,
    region,
    model_name,
):
    """
    Read one DEVSIM edge model as a float list.
    """

    values = get_edge_model_values(
        device=device,
        region=region,
        name=model_name,
    )

    return [
        float(value)
        for value in values
    ]


def get_region_edge_data(
    device,
    region,
    field_model="ElectricField",
):
    """
    Return field and coordinate data for all edges.

    Returns
    -------
    list[dict]
        One dictionary per edge.
    """

    validate_region_field_model(
        device=device,
        region=region,
        field_model=field_model,
    )

    ensure_coordinate_edge_models(
        device=device,
        region=region,
    )

    field_values = (
        get_edge_model_values_as_float(
            device=device,
            region=region,
            model_name=field_model,
        )
    )

    x_n0_values = (
        get_edge_model_values_as_float(
            device=device,
            region=region,
            model_name="x@n0",
        )
    )

    x_n1_values = (
        get_edge_model_values_as_float(
            device=device,
            region=region,
            model_name="x@n1",
        )
    )

    y_n0_values = (
        get_edge_model_values_as_float(
            device=device,
            region=region,
            model_name="y@n0",
        )
    )

    y_n1_values = (
        get_edge_model_values_as_float(
            device=device,
            region=region,
            model_name="y@n1",
        )
    )

    value_counts = {
        len(field_values),
        len(x_n0_values),
        len(x_n1_values),
        len(y_n0_values),
        len(y_n1_values),
    }

    if len(value_counts) != 1:
        raise RuntimeError(
            f"Edge-model array lengths do not match "
            f'in region "{region}".'
        )

    edge_data = []

    for edge_index in range(
        len(field_values)
    ):
        x_n0 = x_n0_values[edge_index]
        x_n1 = x_n1_values[edge_index]

        y_n0 = y_n0_values[edge_index]
        y_n1 = y_n1_values[edge_index]

        delta_x = x_n1 - x_n0
        delta_y = y_n1 - y_n0

        edge_data.append(
            {
                "edge_index": edge_index,
                "field_V_cm": (
                    field_values[edge_index]
                ),
                "x_n0_cm": x_n0,
                "x_n1_cm": x_n1,
                "y_n0_cm": y_n0,
                "y_n1_cm": y_n1,
                "delta_x_cm": delta_x,
                "delta_y_cm": delta_y,
            }
        )

    if not edge_data:
        raise RuntimeError(
            f'No edge data were returned '
            f'for region "{region}".'
        )

    return edge_data


# ============================================================
# Edge-direction classification
# ============================================================

def classify_edge_direction(
    delta_x_cm,
    delta_y_cm,
    tolerance_cm=EDGE_DIRECTION_TOLERANCE_CM,
):
    """
    Classify one edge.

    radial:
        delta_x != 0 and delta_y approximately 0

    axial:
        delta_x approximately 0 and delta_y != 0

    diagonal:
        both delta_x and delta_y are nonzero

    degenerate:
        both coordinate changes are approximately zero
    """

    delta_x_abs = abs(
        delta_x_cm
    )

    delta_y_abs = abs(
        delta_y_cm
    )

    x_changes = (
        delta_x_abs > tolerance_cm
    )

    y_changes = (
        delta_y_abs > tolerance_cm
    )

    if x_changes and not y_changes:
        return "radial"

    if y_changes and not x_changes:
        return "axial"

    if x_changes and y_changes:
        return "diagonal"

    return "degenerate"


def add_edge_direction_information(
    edge_data,
    tolerance_cm=EDGE_DIRECTION_TOLERANCE_CM,
):
    """
    Add direction labels and outward radial fields.

    DEVSIM's ElectricField value is referenced to each edge's
    n0 -> n1 orientation. Since mesh edge orientation may differ,
    the radial field is converted to an outward-directed sign.

    Positive radial field:
        directed from the cylinder axis toward the gate.

    Negative radial field:
        directed from the gate toward the cylinder axis.
    """

    classified_data = []

    for edge in edge_data:
        delta_x = edge["delta_x_cm"]
        delta_y = edge["delta_y_cm"]

        direction = classify_edge_direction(
            delta_x_cm=delta_x,
            delta_y_cm=delta_y,
            tolerance_cm=tolerance_cm,
        )

        radial_field_outward = None

        if direction == "radial":
            if delta_x > 0.0:
                radial_orientation_sign = 1.0
            else:
                radial_orientation_sign = -1.0

            radial_field_outward = (
                edge["field_V_cm"]
                * radial_orientation_sign
            )

        enriched_edge = dict(
            edge
        )

        enriched_edge["direction"] = (
            direction
        )

        enriched_edge[
            "radial_field_outward_V_cm"
        ] = radial_field_outward

        classified_data.append(
            enriched_edge
        )

    return classified_data


def get_region_direction_counts(
    classified_edge_data,
):
    """
    Count radial, axial, diagonal, and degenerate edges.
    """

    counts = {
        "radial": 0,
        "axial": 0,
        "diagonal": 0,
        "degenerate": 0,
    }

    for edge in classified_edge_data:
        direction = edge["direction"]

        counts[direction] += 1

    return counts


# ============================================================
# Field statistics
# ============================================================

def calculate_field_statistics(
    values,
):
    """
    Calculate signed and absolute field statistics.
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

    value_count = len(values)

    return {
        "edge_count": value_count,
        "field_min_V_cm": min(values),
        "field_max_V_cm": max(values),
        "field_mean_signed_V_cm": (
            sum(values) / value_count
        ),
        "field_mean_abs_V_cm": (
            sum(absolute_values) / value_count
        ),
        "field_max_abs_V_cm": max(
            absolute_values
        ),
    }


def get_region_radial_field_statistics(
    device,
    region,
    field_model="ElectricField",
):
    """
    Extract only radial-edge electric-field statistics.
    """

    raw_edge_data = get_region_edge_data(
        device=device,
        region=region,
        field_model=field_model,
    )

    classified_edge_data = (
        add_edge_direction_information(
            edge_data=raw_edge_data,
        )
    )

    direction_counts = (
        get_region_direction_counts(
            classified_edge_data=(
                classified_edge_data
            ),
        )
    )

    radial_field_values = []

    for edge in classified_edge_data:
        if edge["direction"] != "radial":
            continue

        radial_field_values.append(
            edge[
                "radial_field_outward_V_cm"
            ]
        )

    if not radial_field_values:
        raise RuntimeError(
            f'No radial edges were found '
            f'in region "{region}".'
        )

    statistics = calculate_field_statistics(
        values=radial_field_values,
    )

    statistics["region"] = region
    statistics["field_model"] = field_model
    statistics["field_direction"] = "radial"
    statistics["total_edge_count"] = len(
        classified_edge_data
    )
    statistics["radial_edge_count"] = (
        direction_counts["radial"]
    )
    statistics["axial_edge_count"] = (
        direction_counts["axial"]
    )
    statistics["diagonal_edge_count"] = (
        direction_counts["diagonal"]
    )
    statistics["degenerate_edge_count"] = (
        direction_counts["degenerate"]
    )

    return statistics


def get_dielectric_radial_field_statistics(
    device,
):
    """
    Return radial-field statistics for all dielectric layers.
    """

    results = {}

    for region in DIELECTRIC_REGIONS:
        results[region] = (
            get_region_radial_field_statistics(
                device=device,
                region=region,
            )
        )

    return results


# ============================================================
# Compatibility functions
# ============================================================

def get_region_field_statistics(
    device,
    region,
    field_model="ElectricField",
):
    """
    Compatibility wrapper.

    The default statistics now represent radial edges only.
    """

    return get_region_radial_field_statistics(
        device=device,
        region=region,
        field_model=field_model,
    )


def get_dielectric_field_statistics(
    device,
):
    """
    Compatibility wrapper.

    The returned results now contain radial fields only.
    """

    return get_dielectric_radial_field_statistics(
        device=device,
    )


# ============================================================
# Console output
# ============================================================

def print_region_field_statistics(
    statistics,
):
    """
    Print one region's radial-field statistics.
    """

    print()
    print(
        f'Region: {statistics["region"]}'
    )

    print(
        "  field direction        = radial"
    )

    print(
        f'  total edge count       = '
        f'{statistics["total_edge_count"]}'
    )

    print(
        f'  radial edge count      = '
        f'{statistics["radial_edge_count"]}'
    )

    print(
        f'  axial edge count       = '
        f'{statistics["axial_edge_count"]}'
    )

    print(
        f'  diagonal edge count    = '
        f'{statistics["diagonal_edge_count"]}'
    )

    print(
        f'  radial minimum         = '
        f'{statistics["field_min_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial maximum         = '
        f'{statistics["field_max_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial signed mean     = '
        f'{statistics["field_mean_signed_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial absolute mean   = '
        f'{statistics["field_mean_abs_V_cm"]:.6e} V/cm'
    )

    print(
        f'  radial maximum abs.    = '
        f'{statistics["field_max_abs_V_cm"]:.6e} V/cm'
    )


def print_dielectric_field_statistics(
    device,
):
    """
    Read and print radial dielectric-field statistics.
    """

    results = (
        get_dielectric_radial_field_statistics(
            device=device,
        )
    )

    print()
    print("=" * 70)
    print(
        "RADIAL DIELECTRIC ELECTRIC-FIELD STATISTICS"
    )
    print("=" * 70)

    for region in DIELECTRIC_REGIONS:
        print_region_field_statistics(
            statistics=results[region],
        )

    return results