"""Log-domain Fowler--Nordheim diagnostics for the ERASE baseline.

This module deliberately does not implement a physical hole-erase model.  It
applies the repository's existing *electron* FN parameters to the outward
radial tunnel-oxide field as an uncalibrated negative-control diagnostic.  The
linear-domain functions in :mod:`tunneling_models` are retained for regression
and reporting, while the primary edge aggregation is performed in log space so
that very small currents are not discarded by floating-point underflow.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import tunneling_models
import tunneling_parameters


MODEL_CLASS = "electron_FN_diagnostic"
PHYSICAL_ROLE = "negative_control"
MODEL_STATUS = "baseline_negative_result"
CALIBRATION_STATUS = "uncalibrated"
PREDICTIVE_ERASE_MODEL = False

PROXY_BARRIER_HEIGHT_EV = 3.56
PROXY_EFFECTIVE_MASS_RATIO = 0.28

# This threshold is reported only as a diagnostic edge count.  It never
# excludes an edge from the primary log-domain integration.
DEFAULT_DIAGNOSTIC_FIELD_THRESHOLD_V_CM = 1.0e5

LN_10 = math.log(10.0)


def get_model_metadata() -> dict:
    """Return the mandatory provenance/status labels for every result."""

    return {
        "model_class": MODEL_CLASS,
        "physical_role": PHYSICAL_ROLE,
        "model_status": MODEL_STATUS,
        "calibration_status": CALIBRATION_STATUS,
        "predictive_erase_model": PREDICTIVE_ERASE_MODEL,
        "barrier_height_eV": float(
            tunneling_parameters.BARRIER_HEIGHT_EV
        ),
        "effective_mass_ratio": float(
            tunneling_parameters.TUNNEL_EFFECTIVE_MASS_RATIO
        ),
        "parameter_role": (
            "program_electron_FN_used_only_as_negative_control_diagnostic"
        ),
    }


def validate_proxy_parameters() -> None:
    """Fail if the existing FN model no longer matches the negative control."""

    if not math.isclose(
        float(tunneling_parameters.BARRIER_HEIGHT_EV),
        PROXY_BARRIER_HEIGHT_EV,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Electron-FN negative control requires the documented 3.56 eV "
            "program-electron barrier."
        )

    if not math.isclose(
        float(tunneling_parameters.TUNNEL_EFFECTIVE_MASS_RATIO),
        PROXY_EFFECTIVE_MASS_RATIO,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Electron-FN negative control requires the documented 0.28 m0 "
            "program-electron tunneling mass."
        )


def cylindrical_edge_area_cm2(
    radius_cm: float,
    axial_segment_length_cm: float,
) -> float:
    """Return ``2*pi*r_i*delta_z_i`` for one axisymmetric interface edge."""

    radius_cm = float(radius_cm)
    axial_segment_length_cm = float(axial_segment_length_cm)

    if not math.isfinite(radius_cm) or radius_cm <= 0.0:
        raise ValueError("Cylindrical interface radius must be finite and positive.")
    if (
        not math.isfinite(axial_segment_length_cm)
        or axial_segment_length_cm <= 0.0
    ):
        raise ValueError("Axial representative length must be finite and positive.")

    return 2.0 * math.pi * radius_cm * axial_segment_length_cm


def logsumexp(log_values: Iterable[float]) -> float:
    """Accurately return ``log(sum(exp(log_values)))``.

    Negative infinity represents an exact zero contribution.  An empty
    sequence is invalid because silently treating a missing interface as zero
    would hide a mesh/extraction error.
    """

    values = [float(value) for value in log_values]
    if not values:
        raise ValueError("logsumexp requires at least one value.")
    if any(math.isnan(value) or value == math.inf for value in values):
        raise ValueError("logsumexp values must be finite or negative infinity.")

    maximum = max(values)
    if maximum == -math.inf:
        return -math.inf

    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )


def natural_log_to_linear(log_value: float) -> float:
    """Convert a natural-log magnitude to double precision when representable."""

    log_value = float(log_value)
    if log_value == -math.inf:
        return 0.0
    if not math.isfinite(log_value):
        raise ValueError("Logarithmic magnitude must be finite or negative infinity.")

    # math.exp naturally returns subnormal values where available.  Explicitly
    # avoid raising OverflowError; an infinite current is always an error here.
    if log_value > math.log(float.fromhex("0x1.fffffffffffffp+1023")):
        raise OverflowError("FN magnitude exceeds double precision.")
    return math.exp(log_value)


def log10_from_natural_log(log_value: float) -> float:
    """Convert a natural logarithm to base 10, preserving exact zero."""

    log_value = float(log_value)
    if log_value == -math.inf:
        return -math.inf
    if not math.isfinite(log_value):
        raise ValueError("Logarithmic magnitude must be finite or negative infinity.")
    return log_value / LN_10


def log_fowler_nordheim_current_density(
    electric_field_V_cm: float,
) -> float:
    """Return natural log of the FN magnitude in A/cm2.

    Unlike the existing linear-domain reporting helper, this function applies
    no nonzero-field cutoff.  Therefore every finite, nonzero edge field is
    retained in the primary integration even when its linear current rounds to
    zero.  The coefficients and unit conversion are the existing repository FN
    model's coefficients, not independently fitted ERASE parameters.
    """

    validate_proxy_parameters()
    field_V_cm = abs(
        tunneling_models.validate_electric_field(electric_field_V_cm)
    )
    if field_V_cm == 0.0:
        return -math.inf

    field_V_m = tunneling_models.electric_field_V_cm_to_V_m(field_V_cm)
    exponent = tunneling_parameters.FN_B_SI / field_V_m

    return (
        math.log(tunneling_parameters.FN_A_SI)
        + 2.0 * math.log(field_V_m)
        - exponent
        + math.log(tunneling_parameters.A_M2_TO_A_CM2)
    )


def evaluate_mean_field_fn(
    mean_field_V_cm: float,
    total_interface_area_cm2: float,
) -> dict:
    """Evaluate the comparison FN result obtained from one mean field."""

    total_interface_area_cm2 = float(total_interface_area_cm2)
    if (
        not math.isfinite(total_interface_area_cm2)
        or total_interface_area_cm2 <= 0.0
    ):
        raise ValueError("Total interface area must be finite and positive.")

    field_V_cm = tunneling_models.validate_electric_field(mean_field_V_cm)
    log_density = log_fowler_nordheim_current_density(field_V_cm)
    log_current = log_density + math.log(total_interface_area_cm2)

    result = get_model_metadata()
    result.update(
        {
            "aggregation_method": "FN_of_area_weighted_mean_abs_field",
            "electric_field_signed_V_cm": field_V_cm,
            "electric_field_abs_V_cm": abs(field_V_cm),
            "total_interface_area_cm2": total_interface_area_cm2,
            "fowler_nordheim_current_density_A_cm2": (
                tunneling_models.fowler_nordheim_current_density(field_V_cm)
            ),
            "log_fowler_nordheim_current_density_A_cm2": log_density,
            "log10_fowler_nordheim_current_density_A_cm2": (
                log10_from_natural_log(log_density)
            ),
            "total_tunneling_current_A": natural_log_to_linear(log_current),
            "log_total_tunneling_current_A": log_current,
            "log10_total_tunneling_current_A": (
                log10_from_natural_log(log_current)
            ),
        }
    )
    return result


def _area_from_extracted_edge(edge: Mapping) -> float:
    """Recompute and validate the field-extraction cylindrical area."""

    try:
        radius_cm = float(edge["interface_radius_cm"])
        length_cm = float(edge["axial_segment_length_cm"])
        stored_area_cm2 = float(edge["cylindrical_interface_area_cm2"])
    except KeyError as error:
        raise KeyError(
            "Each edge requires interface_radius_cm, axial_segment_length_cm, "
            "and cylindrical_interface_area_cm2 from field_extraction."
        ) from error

    calculated_area_cm2 = cylindrical_edge_area_cm2(radius_cm, length_cm)
    if not math.isclose(
        stored_area_cm2,
        calculated_area_cm2,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Stored interface area is inconsistent with 2*pi*r_i*delta_z_i."
        )
    return calculated_area_cm2


def evaluate_edge_resolved_fn(
    interface_edges: Sequence[Mapping],
    diagnostic_field_threshold_V_cm: float = (
        DEFAULT_DIAGNOSTIC_FIELD_THRESHOLD_V_CM
    ),
) -> dict:
    """Integrate all active-interface FN edge contributions in log space.

    ``diagnostic_field_threshold_V_cm`` affects only the reported count and
    area fraction.  No edge is removed from the primary log-sum-exp result.
    The input is the weighted edge list returned by
    ``field_extraction.get_active_inner_interface_radial_edges``.
    """

    validate_proxy_parameters()
    if not interface_edges:
        raise ValueError("Active interface edge list is empty.")

    threshold = float(diagnostic_field_threshold_V_cm)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("Diagnostic field threshold must be finite and nonnegative.")

    metadata = get_model_metadata()
    edge_results = []
    edge_log_currents = []
    total_area_cm2 = 0.0
    threshold_area_cm2 = 0.0
    threshold_count = 0

    for sequence_index, edge in enumerate(interface_edges):
        try:
            signed_field_V_cm = float(edge["radial_field_outward_V_cm"])
        except KeyError as error:
            raise KeyError(
                "Each edge requires radial_field_outward_V_cm from field_extraction."
            ) from error
        signed_field_V_cm = tunneling_models.validate_electric_field(
            signed_field_V_cm
        )
        field_magnitude_V_cm = abs(signed_field_V_cm)
        area_cm2 = _area_from_extracted_edge(edge)
        total_area_cm2 += area_cm2

        log_density = log_fowler_nordheim_current_density(signed_field_V_cm)
        log_current = (
            -math.inf
            if log_density == -math.inf
            else log_density + math.log(area_cm2)
        )
        edge_log_currents.append(log_current)

        above_threshold = field_magnitude_V_cm >= threshold
        if above_threshold:
            threshold_count += 1
            threshold_area_cm2 += area_cm2

        linear_density = tunneling_models.fowler_nordheim_current_density(
            signed_field_V_cm
        )

        row = dict(metadata)
        row.update(
            {
                "edge_sequence_index": sequence_index,
                "edge_index": edge.get("edge_index", sequence_index),
                "midpoint_axial_nm": edge.get("midpoint_axial_nm"),
                "axial_segment_lower_nm": edge.get("axial_segment_lower_nm"),
                "axial_segment_upper_nm": edge.get("axial_segment_upper_nm"),
                "axial_segment_length_nm": edge.get("axial_segment_length_nm"),
                "interface_radius_nm": edge.get("interface_radius_nm"),
                "interface_radius_cm": float(edge["interface_radius_cm"]),
                "axial_segment_length_cm": float(
                    edge["axial_segment_length_cm"]
                ),
                "interface_area_cm2": area_cm2,
                "electric_field_signed_V_cm": signed_field_V_cm,
                "electric_field_abs_V_cm": field_magnitude_V_cm,
                "fowler_nordheim_exponent": (
                    tunneling_models.fowler_nordheim_exponent(signed_field_V_cm)
                ),
                "fowler_nordheim_current_density_A_cm2": linear_density,
                "log_fowler_nordheim_current_density_A_cm2": log_density,
                "log10_fowler_nordheim_current_density_A_cm2": (
                    log10_from_natural_log(log_density)
                ),
                "local_tunneling_current_A": (
                    natural_log_to_linear(log_current)
                ),
                "log_local_tunneling_current_A": log_current,
                "log10_local_tunneling_current_A": (
                    log10_from_natural_log(log_current)
                ),
                "diagnostic_field_threshold_V_cm": threshold,
                "above_diagnostic_field_threshold": above_threshold,
                "included_in_primary_integration": True,
            }
        )
        edge_results.append(row)

    total_log_current = logsumexp(edge_log_currents)
    effective_log_density = (
        -math.inf
        if total_log_current == -math.inf
        else total_log_current - math.log(total_area_cm2)
    )

    # Add a stable dominance measure after the total is known.
    for row in edge_results:
        edge_log_current = row["log_local_tunneling_current_A"]
        if total_log_current == -math.inf or edge_log_current == -math.inf:
            row["fraction_of_total_tunneling_current"] = 0.0
        else:
            row["fraction_of_total_tunneling_current"] = math.exp(
                edge_log_current - total_log_current
            )

    result = dict(metadata)
    result.update(
        {
            "aggregation_method": "all_edge_logsumexp_no_field_cutoff",
            "edge_count": len(edge_results),
            "total_interface_area_cm2": total_area_cm2,
            "total_tunneling_current_A": natural_log_to_linear(
                total_log_current
            ),
            "log_total_tunneling_current_A": total_log_current,
            "log10_total_tunneling_current_A": (
                log10_from_natural_log(total_log_current)
            ),
            "effective_current_density_A_cm2": natural_log_to_linear(
                effective_log_density
            ),
            "log_effective_current_density_A_cm2": effective_log_density,
            "log10_effective_current_density_A_cm2": (
                log10_from_natural_log(effective_log_density)
            ),
            "diagnostic_field_threshold_V_cm": threshold,
            "diagnostic_above_threshold_edge_count": threshold_count,
            "diagnostic_above_threshold_area_cm2": threshold_area_cm2,
            "diagnostic_above_threshold_area_fraction": (
                threshold_area_cm2 / total_area_cm2
            ),
            "edge_results": edge_results,
        }
    )
    return result


# Explicit alias used by the voltage-sweep runner and CSV nomenclature.
integrate_all_edges_log_domain = evaluate_edge_resolved_fn
