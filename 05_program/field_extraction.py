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
#   6. Filter the interface edges by an active axial window
#   7. Calculate cylindrical interface-area weights
# ============================================================

import math

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
# Unit conversion
# ============================================================

NM_TO_CM = 1.0e-7
CM_TO_NM = 1.0e7


# ============================================================
# Numerical settings
# ============================================================

EDGE_DIRECTION_TOLERANCE_CM = 1.0e-15

INNER_INTERFACE_RADIUS_TOLERANCE_CM = 1.0e-15


# ============================================================
# Default active interface window
#
# The full simulated channel extends from 0 to 100 nm.
# The initial active window excludes source/drain fringe areas.
# ============================================================

DEFAULT_ACTIVE_AXIAL_MINIMUM_NM = 10.0
DEFAULT_ACTIVE_AXIAL_MAXIMUM_NM = 90.0


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

    Sign convention
    ---------------
    Positive radial field:
        cylinder center -> gate

    Negative radial field:
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


def calculate_area_weighted_field_statistics(
    edge_data,
):
    """
    Calculate area-weighted field statistics.

    Each edge dictionary must contain:
        radial_field_outward_V_cm
        cylindrical_interface_area_cm2
    """

    if not edge_data:
        raise ValueError(
            "Interface edge-data list is empty."
        )

    total_area_cm2 = sum(
        float(
            edge[
                "cylindrical_interface_area_cm2"
            ]
        )
        for edge in edge_data
    )

    if total_area_cm2 <= 0.0:
        raise ValueError(
            "Total cylindrical interface area must be positive."
        )

    weighted_signed_sum = 0.0
    weighted_absolute_sum = 0.0

    for edge in edge_data:
        field_value = float(
            edge[
                "radial_field_outward_V_cm"
            ]
        )

        area_value = float(
            edge[
                "cylindrical_interface_area_cm2"
            ]
        )

        weighted_signed_sum += (
            field_value
            * area_value
        )

        weighted_absolute_sum += (
            abs(field_value)
            * area_value
        )

    return {
        "area_weighted_field_mean_signed_V_cm": (
            weighted_signed_sum
            / total_area_cm2
        ),
        "area_weighted_field_mean_abs_V_cm": (
            weighted_absolute_sum
            / total_area_cm2
        ),
        "total_interface_area_cm2": (
            total_area_cm2
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

    interface_edges.sort(
        key=lambda edge: (
            edge["midpoint_axial_cm"]
        )
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
        inner_radius_cm
        * CM_TO_NM
    )

    statistics["outer_radius_nm"] = (
        outer_radius_cm
        * CM_TO_NM
    )

    statistics["midpoint_radius_nm"] = (
        midpoint_radius_cm
        * CM_TO_NM
    )

    statistics["minimum_axial_nm"] = (
        minimum_axial_cm
        * CM_TO_NM
    )

    statistics["maximum_axial_nm"] = (
        maximum_axial_cm
        * CM_TO_NM
    )

    statistics["edge_data"] = (
        interface_edges
    )

    return statistics


# ============================================================
# Active axial-window validation
# ============================================================

def validate_active_axial_window(
    minimum_axial_nm,
    maximum_axial_nm,
):
    """
    Validate the selected active axial range.
    """

    minimum_axial_nm = float(
        minimum_axial_nm
    )

    maximum_axial_nm = float(
        maximum_axial_nm
    )

    if not math.isfinite(
        minimum_axial_nm
    ):
        raise ValueError(
            "minimum_axial_nm must be finite."
        )

    if not math.isfinite(
        maximum_axial_nm
    ):
        raise ValueError(
            "maximum_axial_nm must be finite."
        )

    if minimum_axial_nm < 0.0:
        raise ValueError(
            "minimum_axial_nm must not be negative."
        )

    if maximum_axial_nm <= minimum_axial_nm:
        raise ValueError(
            "maximum_axial_nm must be greater than "
            "minimum_axial_nm."
        )

    return (
        minimum_axial_nm,
        maximum_axial_nm,
    )


# ============================================================
# Cylindrical interface-area weighting
# ============================================================

def add_cylindrical_interface_area_weights(
    interface_edges,
    minimum_axial_cm,
    maximum_axial_cm,
):
    """
    Assign a representative axial segment and cylindrical
    interface area to every selected interface edge.

    The axial segment boundaries are constructed midway between
    adjacent edge midpoint positions. The first and last
    boundaries are clipped to the selected active window.

    Area for edge i:
        A_i = 2 * pi * r_i * delta_z_i
    """

    if not interface_edges:
        raise ValueError(
            "Interface edge list is empty."
        )

    sorted_edges = sorted(
        interface_edges,
        key=lambda edge: (
            edge["midpoint_axial_cm"]
        ),
    )

    midpoint_values = [
        float(
            edge["midpoint_axial_cm"]
        )
        for edge in sorted_edges
    ]

    weighted_edges = []

    for edge_index, edge in enumerate(
        sorted_edges
    ):
        midpoint_axial_cm = (
            midpoint_values[edge_index]
        )

        if edge_index == 0:
            lower_boundary_cm = (
                minimum_axial_cm
            )
        else:
            lower_boundary_cm = (
                0.5
                * (
                    midpoint_values[
                        edge_index - 1
                    ]
                    + midpoint_axial_cm
                )
            )

        if edge_index == (
            len(sorted_edges) - 1
        ):
            upper_boundary_cm = (
                maximum_axial_cm
            )
        else:
            upper_boundary_cm = (
                0.5
                * (
                    midpoint_axial_cm
                    + midpoint_values[
                        edge_index + 1
                    ]
                )
            )

        lower_boundary_cm = max(
            lower_boundary_cm,
            minimum_axial_cm,
        )

        upper_boundary_cm = min(
            upper_boundary_cm,
            maximum_axial_cm,
        )

        axial_segment_length_cm = (
            upper_boundary_cm
            - lower_boundary_cm
        )

        if axial_segment_length_cm <= 0.0:
            raise RuntimeError(
                "Calculated axial segment length is "
                "not positive."
            )

        interface_radius_cm = float(
            edge["inner_radius_cm"]
        )

        cylindrical_interface_area_cm2 = (
            2.0
            * math.pi
            * interface_radius_cm
            * axial_segment_length_cm
        )

        enriched_edge = dict(
            edge
        )

        enriched_edge[
            "midpoint_axial_nm"
        ] = (
            midpoint_axial_cm
            * CM_TO_NM
        )

        enriched_edge[
            "axial_segment_lower_cm"
        ] = (
            lower_boundary_cm
        )

        enriched_edge[
            "axial_segment_upper_cm"
        ] = (
            upper_boundary_cm
        )

        enriched_edge[
            "axial_segment_lower_nm"
        ] = (
            lower_boundary_cm
            * CM_TO_NM
        )

        enriched_edge[
            "axial_segment_upper_nm"
        ] = (
            upper_boundary_cm
            * CM_TO_NM
        )

        enriched_edge[
            "axial_segment_length_cm"
        ] = (
            axial_segment_length_cm
        )

        enriched_edge[
            "axial_segment_length_nm"
        ] = (
            axial_segment_length_cm
            * CM_TO_NM
        )

        enriched_edge[
            "interface_radius_cm"
        ] = (
            interface_radius_cm
        )

        enriched_edge[
            "interface_radius_nm"
        ] = (
            interface_radius_cm
            * CM_TO_NM
        )

        enriched_edge[
            "cylindrical_interface_area_cm2"
        ] = (
            cylindrical_interface_area_cm2
        )

        weighted_edges.append(
            enriched_edge
        )

    return weighted_edges


# ============================================================
# Active inner-interface extraction
# ============================================================

def get_active_inner_interface_radial_edges(
    device,
    region,
    field_model="ElectricField",
    minimum_axial_nm=(
        DEFAULT_ACTIVE_AXIAL_MINIMUM_NM
    ),
    maximum_axial_nm=(
        DEFAULT_ACTIVE_AXIAL_MAXIMUM_NM
    ),
):
    """
    Select inner-interface radial edges inside an active axial
    window and assign cylindrical interface-area weights.

    For TunnelOxide, these edges represent the field near the
    MoS2 / TunnelOxide interface after excluding the source and
    drain fringe regions.
    """

    (
        minimum_axial_nm,
        maximum_axial_nm,
    ) = validate_active_axial_window(
        minimum_axial_nm=minimum_axial_nm,
        maximum_axial_nm=maximum_axial_nm,
    )

    minimum_axial_cm = (
        minimum_axial_nm
        * NM_TO_CM
    )

    maximum_axial_cm = (
        maximum_axial_nm
        * NM_TO_CM
    )

    interface_edges = (
        get_inner_interface_radial_edges(
            device=device,
            region=region,
            field_model=field_model,
        )
    )

    active_edges = []

    for edge in interface_edges:
        midpoint_axial_cm = float(
            edge["midpoint_axial_cm"]
        )

        if (
            midpoint_axial_cm
            < minimum_axial_cm
        ):
            continue

        if (
            midpoint_axial_cm
            > maximum_axial_cm
        ):
            continue

        active_edges.append(
            dict(edge)
        )

    if not active_edges:
        raise RuntimeError(
            f"No active inner-interface edges were found "
            f'in region "{region}" between '
            f"{minimum_axial_nm:.3f} and "
            f"{maximum_axial_nm:.3f} nm."
        )

    weighted_active_edges = (
        add_cylindrical_interface_area_weights(
            interface_edges=active_edges,
            minimum_axial_cm=minimum_axial_cm,
            maximum_axial_cm=maximum_axial_cm,
        )
    )

    return weighted_active_edges


def get_active_inner_interface_radial_field_statistics(
    device,
    region,
    field_model="ElectricField",
    minimum_axial_nm=(
        DEFAULT_ACTIVE_AXIAL_MINIMUM_NM
    ),
    maximum_axial_nm=(
        DEFAULT_ACTIVE_AXIAL_MAXIMUM_NM
    ),
):
    """
    Calculate active-window interface field statistics.

    Returns both ordinary edge-count averages and cylindrical
    interface-area-weighted averages.

    The returned edge_data can be used for edge-resolved
    tunneling-current calculations.
    """

    (
        minimum_axial_nm,
        maximum_axial_nm,
    ) = validate_active_axial_window(
        minimum_axial_nm=minimum_axial_nm,
        maximum_axial_nm=maximum_axial_nm,
    )

    active_edges = (
        get_active_inner_interface_radial_edges(
            device=device,
            region=region,
            field_model=field_model,
            minimum_axial_nm=minimum_axial_nm,
            maximum_axial_nm=maximum_axial_nm,
        )
    )

    field_values = [
        edge[
            "radial_field_outward_V_cm"
        ]
        for edge in active_edges
    ]

    statistics = calculate_field_statistics(
        values=field_values,
    )

    weighted_statistics = (
        calculate_area_weighted_field_statistics(
            edge_data=active_edges,
        )
    )

    interface_radius_cm = (
        sum(
            edge["interface_radius_cm"]
            for edge in active_edges
        )
        / len(active_edges)
    )

    selected_midpoint_values_nm = [
        edge["midpoint_axial_nm"]
        for edge in active_edges
    ]

    statistics.update(
        weighted_statistics
    )

    statistics["region"] = region
    statistics["field_model"] = field_model
    statistics["field_direction"] = "radial"

    statistics["extraction_location"] = (
        "active_inner_interface"
    )

    statistics["active_interface_edge_count"] = (
        len(active_edges)
    )

    statistics["active_axial_minimum_nm"] = (
        minimum_axial_nm
    )

    statistics["active_axial_maximum_nm"] = (
        maximum_axial_nm
    )

    statistics["active_axial_length_nm"] = (
        maximum_axial_nm
        - minimum_axial_nm
    )

    statistics[
        "selected_midpoint_minimum_axial_nm"
    ] = min(
        selected_midpoint_values_nm
    )

    statistics[
        "selected_midpoint_maximum_axial_nm"
    ] = max(
        selected_midpoint_values_nm
    )

    statistics["interface_radius_cm"] = (
        interface_radius_cm
    )

    statistics["interface_radius_nm"] = (
        interface_radius_cm
        * CM_TO_NM
    )

    statistics["edge_data"] = (
        active_edges
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

    The default region statistics represent radial edges.
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
    Print full-channel inner-interface field statistics.
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


def print_active_interface_field_statistics(
    statistics,
):
    """
    Print active-window inner-interface field statistics.
    """

    print()
    print(
        f'Active interface region: '
        f'{statistics["region"]}'
    )

    print(
        "  extraction location     = "
        "active inner interface"
    )

    print(
        f'  selected edge count     = '
        f'{statistics["active_interface_edge_count"]}'
    )

    print(
        f'  requested axial window  = '
        f'{statistics["active_axial_minimum_nm"]:.3f} '
        f'to '
        f'{statistics["active_axial_maximum_nm"]:.3f} nm'
    )

    print(
        f'  selected midpoint range = '
        f'{statistics["selected_midpoint_minimum_axial_nm"]:.3f} '
        f'to '
        f'{statistics["selected_midpoint_maximum_axial_nm"]:.3f} nm'
    )

    print(
        f'  interface radius        = '
        f'{statistics["interface_radius_nm"]:.6f} nm'
    )

    print(
        f'  total interface area    = '
        f'{statistics["total_interface_area_cm2"]:.6e} '
        f'cm^2'
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
        f'  area-weighted signed    = '
        f'{statistics["area_weighted_field_mean_signed_V_cm"]:+.6e} '
        f'V/cm'
    )

    print(
        f'  area-weighted absolute  = '
        f'{statistics["area_weighted_field_mean_abs_V_cm"]:.6e} '
        f'V/cm'
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