# ============================================================
# Electric-field extraction utilities
# MoS2 cylindrical GAA charge-trap memory
#
# Coordinate interpretation:
#   x -> radial direction, r
#   y -> channel direction, z
#
# Main functions:
#   1. Read DEVSIM ElectricField edge-model values
#   2. Separate radial, axial, and diagonal edges
#   3. Convert radial fields to a consistent outward direction
#   4. Calculate radial-field statistics
#   5. Extract the innermost radial edge layer
#      near the MoS2 / TunnelOxide interface
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

INNER_INTERFACE_RADIUS_TOLERANCE_CM = 1.0e-15


# ============================================================
# Model validation
# ============================================================

def validate_region_field_model(
    device,
    region,
    field_model="ElectricField",
):
    """
    Confirm that the requested electric-field edge model exists.

    Parameters
    ----------
    device : str
        DEVSIM device name.

    region : str
        DEVSIM region name.

    field_model : str
        Name of the electric-field edge model.

    Raises
    ------
    RuntimeError
        If the edge model is not present in the region.
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

    Required models
    ---------------
    x@n0, x@n1
        Radial-coordinate values at both edge endpoints.

    y@n0, y@n1
        Axial-coordinate values at both edge endpoints.

    Coordinate convention
    ---------------------
    x = cylindrical radial coordinate r
    y = channel-direction coordinate z
    """

    node_models = get_node_model_list(
        device=device,
        region=region,
    )

    for coordinate_name in (
        "x",
        "y",
    ):
        if coordinate_name not in node_models:
            raise RuntimeError(
                f'Coordinate node model "{coordinate_name}" '
                f'is missing in region "{region}".'
            )

    edge_models = get_edge_model_list(
        device=device,
        region=region,
    )

    x_edge_models_exist = (
        "x@n0" in edge_models
        and "x@n1" in edge_models
    )

    if not x_edge_models_exist:
        edge_from_node_model(
            device=device,
            region=region,
            node_model="x",
        )

    edge_models = get_edge_model_list(
        device=device,
        region=region,
    )

    y_edge_models_exist = (
        "y@n0" in edge_models
        and "y@n1" in edge_models
    )

    if not y_edge_models_exist:
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
    Read one DEVSIM edge model and return float values.

    Returns
    -------
    list[float]
        Edge-model values.
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
    Return electric-field and coordinate data for all edges.

    Returns
    -------
    list[dict]
        One dictionary for each region edge.
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
    Classify one mesh edge according to its coordinate change.

    radial
        delta_x is nonzero and delta_y is approximately zero.

    axial
        delta_x is approximately zero and delta_y is nonzero.

    diagonal
        Both delta_x and delta_y are nonzero.

    degenerate
        Both coordinate changes are approximately zero.
    """

    delta_x_absolute = abs(
        delta_x_cm
    )

    delta_y_absolute = abs(
        delta_y_cm
    )

    x_changes = (
        delta_x_absolute > tolerance_cm
    )

    y_changes = (
        delta_y_absolute > tolerance_cm
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
    Add an edge-direction label and outward radial-field value.

    DEVSIM's ElectricField edge value follows the mesh edge's
    n0-to-n1 orientation. Mesh edge orientation can vary.

    To compare all radial edges consistently, this function
    converts the field sign to the following convention:

        positive radial field:
            cylinder center -> gate

        negative radial field:
            gate -> cylinder center
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
    Calculate signed and absolute electric-field statistics.

    Parameters
    ----------
    values : iterable[float]
        Electric-field values in V/cm.

    Returns
    -------
    dict
        Minimum, maximum, signed mean, absolute mean,
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
    Extract electric-field statistics from radial edges only.
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
    Return radial electric-field statistics for all dielectrics.
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
# Inner-interface radial-field extraction
# ============================================================

def get_inner_interface_radial_edges(
    device,
    region,
    field_model="ElectricField",
    radius_tolerance_cm=(
        INNER_INTERFACE_RADIUS_TOLERANCE_CM
    ),
):
    """
    Select the innermost radial-edge layer of a region.

    For TunnelOxide, this corresponds to the radial mesh layer
    closest to the MoS2 / TunnelOxide interface.

    Selection procedure
    -------------------
    1. Extract every edge in the requested region.
    2. Retain radial edges only.
    3. Calculate the inner endpoint radius of every radial edge.
    4. Find the minimum inner radius.
    5. Select radial edges located at that radius.

    Returns
    -------
    list[dict]
        Radial edges belonging to the innermost mesh layer.
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

    radial_edges = [
        edge
        for edge in classified_edge_data
        if edge["direction"] == "radial"
    ]

    if not radial_edges:
        raise RuntimeError(
            f'No radial edges were found '
            f'in region "{region}".'
        )

    radial_edges_with_position = []

    for edge in radial_edges:
        enriched_edge = dict(
            edge
        )

        enriched_edge["inner_radius_cm"] = min(
            edge["x_n0_cm"],
            edge["x_n1_cm"],
        )

        enriched_edge["outer_radius_cm"] = max(
            edge["x_n0_cm"],
            edge["x_n1_cm"],
        )

        enriched_edge["midpoint_radius_cm"] = (
            0.5
            * (
                edge["x_n0_cm"]
                + edge["x_n1_cm"]
            )
        )

        enriched_edge["midpoint_axial_cm"] = (
            0.5
            * (
                edge["y_n0_cm"]
                + edge["y_n1_cm"]
            )
        )

        radial_edges_with_position.append(
            enriched_edge
        )

    minimum_inner_radius_cm = min(
        edge["inner_radius_cm"]
        for edge in radial_edges_with_position
    )

    interface_edges = [
        edge
        for edge in radial_edges_with_position
        if abs(
            edge["inner_radius_cm"]
            - minimum_inner_radius_cm
        ) <= radius_tolerance_cm
    ]

    if not interface_edges:
        raise RuntimeError(
            f'No inner-interface radial edges were selected '
            f'in region "{region}".'
        )

    return interface_edges


def get_inner_interface_radial_field_statistics(
    device,
    region,
    field_model="ElectricField",
):
    """
    Calculate radial electric-field statistics at the innermost
    radial mesh layer of a region.

    For TunnelOxide, the extracted value represents the field
    immediately adjacent to the MoS2 / TunnelOxide interface.

    This field will later be used as an input for tunneling
    current-density calculations.
    """

    interface_edges = (
        get_inner_interface_radial_edges(
            device=device,
            region=region,
            field_model=field_model,
        )
    )

    field_values = [
        edge["radial_field_outward_V_cm"]
        for edge in interface_edges
    ]

    statistics = calculate_field_statistics(
        values=field_values,
    )

    inner_radius_cm = min(
        edge["inner_radius_cm"]
        for edge in interface_edges
    )

    outer_radius_cm = max(
        edge["outer_radius_cm"]
        for edge in interface_edges
    )

    midpoint_radius_cm = (
        sum(
            edge["midpoint_radius_cm"]
            for edge in interface_edges
        )
        / len(interface_edges)
    )

    minimum_axial_cm = min(
        edge["midpoint_axial_cm"]
        for edge in interface_edges
    )

    maximum_axial_cm = max(
        edge["midpoint_axial_cm"]
        for edge in interface_edges
    )

    statistics["region"] = region
    statistics["field_model"] = field_model
    statistics["field_direction"] = "radial"

    statistics["extraction_location"] = (
        "inner_interface"
    )

    statistics["inner_interface_edge_count"] = (
        len(interface_edges)
    )

    statistics["inner_radius_cm"] = (
        inner_radius_cm
    )

    statistics["outer_radius_cm"] = (
        outer_radius_cm
    )

    statistics["midpoint_radius_cm"] = (
        midpoint_radius_cm
    )

    statistics["minimum_axial_cm"] = (
        minimum_axial_cm
    )

    statistics["maximum_axial_cm"] = (
        maximum_axial_cm
    )

    statistics["inner_radius_nm"] = (
        inner_radius_cm * 1.0e7
    )

    statistics["outer_radius_nm"] = (
        outer_radius_cm * 1.0e7
    )

    statistics["midpoint_radius_nm"] = (
        midpoint_radius_cm * 1.0e7
    )

    statistics["minimum_axial_nm"] = (
        minimum_axial_cm * 1.0e7
    )

    statistics["maximum_axial_nm"] = (
        maximum_axial_cm * 1.0e7
    )

    return statistics


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

    The default region statistics now represent radial edges.
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

    Returned dielectric statistics contain radial fields only.
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
    Print one region's radial electric-field statistics.
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
        f'  degenerate edge count  = '
        f'{statistics["degenerate_edge_count"]}'
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


def print_inner_interface_field_statistics(
    statistics,
):
    """
    Print inner-interface radial electric-field statistics.
    """

    print()
    print(
        f'Interface region: {statistics["region"]}'
    )

    print(
        "  extraction location     = inner interface"
    )

    print(
        f'  selected edge count     = '
        f'{statistics["inner_interface_edge_count"]}'
    )

    print(
        f'  inner radius            = '
        f'{statistics["inner_radius_nm"]:.6f} nm'
    )

    print(
        f'  outer radius            = '
        f'{statistics["outer_radius_nm"]:.6f} nm'
    )

    print(
        f'  midpoint radius         = '
        f'{statistics["midpoint_radius_nm"]:.6f} nm'
    )

    print(
        f'  axial range             = '
        f'{statistics["minimum_axial_nm"]:.6f} '
        f'to '
        f'{statistics["maximum_axial_nm"]:.6f} nm'
    )

    print(
        f'  radial minimum          = '
        f'{statistics["field_min_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial maximum          = '
        f'{statistics["field_max_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial signed mean      = '
        f'{statistics["field_mean_signed_V_cm"]:+.6e} V/cm'
    )

    print(
        f'  radial absolute mean    = '
        f'{statistics["field_mean_abs_V_cm"]:.6e} V/cm'
    )

    print(
        f'  radial maximum absolute = '
        f'{statistics["field_max_abs_V_cm"]:.6e} V/cm'
    )


def print_dielectric_field_statistics(
    device,
):
    """
    Read and print radial field statistics for all dielectrics.
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