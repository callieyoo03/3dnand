"""Physically scoped terminal-charge and quasi-static capacitance helpers.

Only quantities directly supported by the present DEVSIM model are exposed as
compact-model handoff data.  The insulated-gate charge is obtained from the
Poisson contact equation.  Semiconductor and trapped charges are integrated
with DEVSIM's cylindrical node volumes; no planar area, scale factor, current
integration, or arbitrary source/drain charge partition is used.

The source and drain Poisson contact charges remain useful for a discrete
Gauss-law check.  They are deliberately labelled as boundary-flux diagnostics,
not as compact-model ``Qd``/``Qs``.  The latter remain NaN because the current
electron-only DC model does not define a unique partition of distributed
channel charge between its conducting contacts.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any


DEVICE_NAME = "MoS2_GAA"
SEMICONDUCTOR_REGION = "MoS2"
CHARGE_TRAP_REGION = "ChargeTrap"
POTENTIAL_EQUATION = "PotentialEquation"

SUPPORTED_QUANTITIES = "Qg,Qtrap,Qmobile,Qfixed,Cgg,Cgd,Cgs"
UNSUPPORTED_PARTITION_LIMITATION = (
    "Qd/Qs compact-model channel-charge partition is not defined by the "
    "present conducting-contact DC model; values remain NaN. Source/drain "
    "Poisson boundary fluxes are diagnostics only."
)
CHARGE_METHOD = (
    "DEVSIM PotentialEquation contact displacement charge plus direct "
    "CylindricalNodeVolume integration"
)
CAPACITANCE_METHOD = (
    "central difference of DEVSIM gate contact charge with fixed Ntrap and "
    "adaptive bias micro-ramps"
)

MAX_CAPACITANCE_RELATIVE_SENSITIVITY = 1.0e-2
MAX_GATE_CAPACITANCE_ROW_SUM_F = 1.0e-22

TERMINALS = ("gate", "drain", "source")
MEASURED_TERMINALS = TERMINALS


TERMINAL_CHARGE_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "Qg_C",
    "Qd_C",
    "Qs_C",
    "Qd_poisson_boundary_flux_C",
    "Qs_poisson_boundary_flux_C",
    "Qtrap_C",
    "Qmobile_C",
    "Qfixed_C",
    "charge_sum_C",
    "charge_conservation_error_C",
    "method",
    "supported_quantities",
    "limitation",
    "converged",
    "error_message",
)

CAPACITANCE_MATRIX_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "perturbed_terminal",
    "measured_terminal",
    "capacitance_F",
    "delta_voltage_V",
    "method",
    "supported",
    "limitation",
    "converged",
    "error_message",
)

CAPACITANCE_SUMMARY_FIELDNAMES = (
    "state_index",
    "state",
    "ntrap_cm3",
    "nsheet_cm2",
    "VGS_V",
    "VDS_V",
    "Cgg_F",
    "Cgd_F",
    "Cgs_F",
    "nominal_delta_voltage_V",
    "sensitivity_delta_voltages_V",
    "Cgg_relative_sensitivity",
    "Cgd_relative_sensitivity",
    "Cgs_relative_sensitivity",
    "gate_capacitance_row_sum_F",
    "method",
    "supported_quantities",
    "limitation",
    "converged",
    "error_message",
)


def normalize_error_message(error: object | None) -> str:
    """Return one compact line suitable for a CSV cell."""

    if error is None:
        return ""
    message = " ".join(str(error).split())
    if isinstance(error, BaseException):
        error_name = type(error).__name__
        return f"{error_name}: {message}" if message else error_name
    return message


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def integrate_cylindrical_node_quantity(
    values: Sequence[float],
    node_volumes_cm3: Sequence[float],
) -> float:
    """Integrate a nodal density with already-generated cylindrical volumes."""

    if len(values) != len(node_volumes_cm3):
        raise ValueError("values and node volumes must have equal lengths.")
    if not values:
        raise ValueError("at least one nodal value is required.")

    products: list[float] = []
    for index, (value, volume) in enumerate(zip(values, node_volumes_cm3)):
        numeric_value = _finite_float(value, f"values[{index}]")
        numeric_volume = _finite_float(volume, f"node_volumes_cm3[{index}]")
        if numeric_volume < 0.0:
            raise ValueError("cylindrical node volumes must be nonnegative.")
        products.append(numeric_value * numeric_volume)
    return math.fsum(products)


def calculate_semiconductor_charge_components(
    electrons_cm3: Sequence[float],
    holes_cm3: Sequence[float],
    net_doping_cm3: Sequence[float],
    node_volumes_cm3: Sequence[float],
    elementary_charge_C: float,
) -> tuple[float, float]:
    """Return modeled mobile-carrier and fixed-dopant charges in coulombs."""

    count = len(node_volumes_cm3)
    if not (
        len(electrons_cm3)
        == len(holes_cm3)
        == len(net_doping_cm3)
        == count
    ):
        raise ValueError("all semiconductor nodal arrays must have equal lengths.")

    charge_C = _finite_float(elementary_charge_C, "elementary_charge_C")
    if charge_C <= 0.0:
        raise ValueError("elementary_charge_C must be positive.")

    mobile_density = [
        _finite_float(hole, f"holes_cm3[{index}]")
        - _finite_float(electron, f"electrons_cm3[{index}]")
        for index, (electron, hole) in enumerate(
            zip(electrons_cm3, holes_cm3)
        )
    ]
    mobile_charge_C = charge_C * integrate_cylindrical_node_quantity(
        mobile_density,
        node_volumes_cm3,
    )
    fixed_charge_C = charge_C * integrate_cylindrical_node_quantity(
        net_doping_cm3,
        node_volumes_cm3,
    )
    return mobile_charge_C, fixed_charge_C


def calculate_charge_conservation(
    *,
    gate_charge_C: float,
    drain_boundary_flux_C: float,
    source_boundary_flux_C: float,
    trap_charge_C: float,
    mobile_charge_C: float,
    fixed_charge_C: float,
) -> tuple[float, float]:
    """Return signed discrete Gauss residual and its absolute magnitude."""

    values = (
        gate_charge_C,
        drain_boundary_flux_C,
        source_boundary_flux_C,
        trap_charge_C,
        mobile_charge_C,
        fixed_charge_C,
    )
    finite_values = tuple(
        _finite_float(value, f"charge_component[{index}]")
        for index, value in enumerate(values)
    )
    signed_sum_C = math.fsum(finite_values)
    return signed_sum_C, abs(signed_sum_C)


def charge_conservation_within_tolerance(
    components: Mapping[str, float],
    *,
    absolute_tolerance_C: float = 1.0e-24,
    relative_tolerance: float = 1.0e-8,
) -> bool:
    """Check a Gauss residual against combined absolute/relative tolerances."""

    absolute_tolerance_C = _finite_float(
        absolute_tolerance_C, "absolute_tolerance_C"
    )
    relative_tolerance = _finite_float(
        relative_tolerance, "relative_tolerance"
    )
    if absolute_tolerance_C < 0.0 or relative_tolerance < 0.0:
        raise ValueError("charge-conservation tolerances must be nonnegative.")

    contributing_names = (
        "Qg_C",
        "Qd_poisson_boundary_flux_C",
        "Qs_poisson_boundary_flux_C",
        "Qtrap_C",
        "Qmobile_C",
        "Qfixed_C",
    )
    scale_C = math.fsum(abs(_finite_float(components[name], name)) for name in contributing_names)
    tolerance_C = max(
        absolute_tolerance_C,
        relative_tolerance * scale_C,
    )
    return abs(_finite_float(components["charge_sum_C"], "charge_sum_C")) <= tolerance_C


def validate_devsim_charge_runtime(device: str = DEVICE_NAME) -> dict[str, Any]:
    """Validate the API, contact equations, and cylindrical integration setup."""

    from devsim import (
        get_contact_charge,
        get_contact_equation_command,
        get_contact_equation_list,
        get_parameter,
    )

    if not callable(get_contact_charge):
        raise RuntimeError("DEVSIM get_contact_charge API is unavailable.")

    node_volume_model = str(
        get_parameter(device=device, name="node_volume_model")
    )
    edge_couple_model = str(
        get_parameter(device=device, name="edge_couple_model")
    )
    radial_axis = str(get_parameter(device=device, name="raxis_variable"))
    radial_zero = float(get_parameter(device=device, name="raxis_zero"))

    if node_volume_model != "CylindricalNodeVolume":
        raise RuntimeError(
            "node_volume_model must be CylindricalNodeVolume; received "
            f"{node_volume_model!r}."
        )
    if edge_couple_model != "CylindricalEdgeCouple":
        raise RuntimeError(
            "edge_couple_model must be CylindricalEdgeCouple; received "
            f"{edge_couple_model!r}."
        )
    if radial_axis != "x" or not math.isclose(
        radial_zero, 0.0, rel_tol=0.0, abs_tol=1.0e-30
    ):
        raise RuntimeError(
            "the expected axisymmetric convention is r=x with raxis_zero=0."
        )

    for contact in TERMINALS:
        equations = tuple(
            get_contact_equation_list(device=device, contact=contact)
        )
        if POTENTIAL_EQUATION not in equations:
            raise RuntimeError(
                f"{POTENTIAL_EQUATION} is missing on contact {contact!r}."
            )
        command = get_contact_equation_command(
            device=device,
            contact=contact,
            name=POTENTIAL_EQUATION,
        )
        if str(command.get("edge_charge_model", "")) != "PotentialEdgeFlux":
            raise RuntimeError(
                f"{contact!r} PotentialEquation must integrate "
                "edge_charge_model='PotentialEdgeFlux'."
            )

    return {
        "node_volume_model": node_volume_model,
        "edge_couple_model": edge_couple_model,
        "raxis_variable": radial_axis,
        "raxis_zero": radial_zero,
        "contact_charge_api": "get_contact_charge",
    }


def _get_node_values(device: str, region: str, model: str) -> tuple[float, ...]:
    from devsim import get_node_model_values

    return tuple(
        float(value)
        for value in get_node_model_values(
            device=device,
            region=region,
            name=model,
        )
    )


def _get_poisson_contact_charge(device: str, contact: str) -> float:
    from devsim import get_contact_charge

    value = float(
        get_contact_charge(
            device=device,
            contact=contact,
            equation=POTENTIAL_EQUATION,
        )
    )
    return _finite_float(value, f"{contact} Poisson contact charge")


def extract_terminal_charge_components(
    *,
    device: str = DEVICE_NAME,
    semiconductor_region: str = SEMICONDUCTOR_REGION,
    charge_trap_region: str = CHARGE_TRAP_REGION,
    elementary_charge_C: float | None = None,
) -> dict[str, float]:
    """Extract supported charges and raw S/D boundary-flux diagnostics."""

    if elementary_charge_C is None:
        import trap_parameters as trap_parameters

        elementary_charge_C = float(trap_parameters.q)

    gate_charge_C = _get_poisson_contact_charge(device, "gate")
    drain_boundary_flux_C = _get_poisson_contact_charge(device, "drain")
    source_boundary_flux_C = _get_poisson_contact_charge(device, "source")

    semiconductor_volumes_cm3 = _get_node_values(
        device, semiconductor_region, "CylindricalNodeVolume"
    )
    electrons_cm3 = _get_node_values(device, semiconductor_region, "Electrons")
    holes_cm3 = _get_node_values(
        device, semiconductor_region, "EquilibriumHoles"
    )
    net_doping_cm3 = _get_node_values(
        device, semiconductor_region, "NetDoping"
    )
    mobile_charge_C, fixed_charge_C = calculate_semiconductor_charge_components(
        electrons_cm3,
        holes_cm3,
        net_doping_cm3,
        semiconductor_volumes_cm3,
        elementary_charge_C,
    )

    trap_charge_density_C_cm3 = _get_node_values(
        device, charge_trap_region, "TrappedChargeDensity"
    )
    trap_volumes_cm3 = _get_node_values(
        device, charge_trap_region, "CylindricalNodeVolume"
    )
    trap_charge_C = integrate_cylindrical_node_quantity(
        trap_charge_density_C_cm3,
        trap_volumes_cm3,
    )

    charge_sum_C, conservation_error_C = calculate_charge_conservation(
        gate_charge_C=gate_charge_C,
        drain_boundary_flux_C=drain_boundary_flux_C,
        source_boundary_flux_C=source_boundary_flux_C,
        trap_charge_C=trap_charge_C,
        mobile_charge_C=mobile_charge_C,
        fixed_charge_C=fixed_charge_C,
    )
    return {
        "Qg_C": gate_charge_C,
        "Qd_C": math.nan,
        "Qs_C": math.nan,
        "Qd_poisson_boundary_flux_C": drain_boundary_flux_C,
        "Qs_poisson_boundary_flux_C": source_boundary_flux_C,
        "Qtrap_C": trap_charge_C,
        "Qmobile_C": mobile_charge_C,
        "Qfixed_C": fixed_charge_C,
        "charge_sum_C": charge_sum_C,
        "charge_conservation_error_C": conservation_error_C,
    }


def _state_values(state: Mapping[str, Any]) -> tuple[int, str, float, float]:
    required = {"state_index", "state", "ntrap_cm3"}
    missing = required.difference(state)
    if missing:
        raise KeyError(
            "state is missing required key(s): " + ", ".join(sorted(missing))
        )
    ntrap_cm3 = _finite_float(state["ntrap_cm3"], "ntrap_cm3")
    if ntrap_cm3 < 0.0:
        raise ValueError("ntrap_cm3 must be nonnegative.")
    if "nsheet_cm2" in state:
        nsheet_cm2 = _finite_float(state["nsheet_cm2"], "nsheet_cm2")
    else:
        import state_characterization_config as state_config

        nsheet_cm2 = ntrap_cm3 * float(state_config.CHARGE_TRAP_THICKNESS_CM)
    return int(state["state_index"]), str(state["state"]), ntrap_cm3, nsheet_cm2


def build_terminal_charge_row(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    components: Mapping[str, float],
) -> dict[str, Any]:
    """Build one converged row using the terminal-charge CSV contract."""

    state_index, state_name, ntrap_cm3, nsheet_cm2 = _state_values(state)
    component_names = TERMINAL_CHARGE_FIELDNAMES[6:16]
    missing = set(component_names).difference(components)
    if missing:
        raise KeyError(
            "terminal-charge components are missing key(s): "
            + ", ".join(sorted(missing))
        )
    for unsupported_name in ("Qd_C", "Qs_C"):
        if not math.isnan(float(components[unsupported_name])):
            raise ValueError(f"{unsupported_name} must remain NaN.")
    for supported_name in set(component_names).difference({"Qd_C", "Qs_C"}):
        _finite_float(components[supported_name], supported_name)
    row = {
        "state_index": state_index,
        "state": state_name,
        "ntrap_cm3": ntrap_cm3,
        "nsheet_cm2": nsheet_cm2,
        "VGS_V": _finite_float(VGS_V, "VGS_V"),
        "VDS_V": _finite_float(VDS_V, "VDS_V"),
        **{name: components[name] for name in component_names},
        "method": CHARGE_METHOD,
        "supported_quantities": SUPPORTED_QUANTITIES,
        "limitation": UNSUPPORTED_PARTITION_LIMITATION,
        "converged": True,
        "error_message": "",
    }
    if tuple(row) != TERMINAL_CHARGE_FIELDNAMES:
        raise RuntimeError("terminal-charge row order does not match its schema.")
    return row


def build_failed_terminal_charge_row(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    error: object,
) -> dict[str, Any]:
    """Build an explicit failed terminal-charge operating-point row."""

    state_index, state_name, ntrap_cm3, nsheet_cm2 = _state_values(state)
    row = {
        "state_index": state_index,
        "state": state_name,
        "ntrap_cm3": ntrap_cm3,
        "nsheet_cm2": nsheet_cm2,
        "VGS_V": float(VGS_V),
        "VDS_V": float(VDS_V),
        "Qg_C": math.nan,
        "Qd_C": math.nan,
        "Qs_C": math.nan,
        "Qd_poisson_boundary_flux_C": math.nan,
        "Qs_poisson_boundary_flux_C": math.nan,
        "Qtrap_C": math.nan,
        "Qmobile_C": math.nan,
        "Qfixed_C": math.nan,
        "charge_sum_C": math.nan,
        "charge_conservation_error_C": math.nan,
        "method": CHARGE_METHOD,
        "supported_quantities": SUPPORTED_QUANTITIES,
        "limitation": UNSUPPORTED_PARTITION_LIMITATION,
        "converged": False,
        "error_message": normalize_error_message(error),
    }
    return row


def central_difference(
    value_plus: float,
    value_minus: float,
    delta_voltage_V: float,
) -> float:
    """Return ``(Q(+dV)-Q(-dV))/(2*dV)`` after strict validation."""

    plus = _finite_float(value_plus, "value_plus")
    minus = _finite_float(value_minus, "value_minus")
    delta = _finite_float(delta_voltage_V, "delta_voltage_V")
    if delta <= 0.0:
        raise ValueError("delta_voltage_V must be positive.")
    return (plus - minus) / (2.0 * delta)


def calculate_relative_sensitivity(
    capacitances_F: Mapping[float, float],
    *,
    nominal_delta_voltage_V: float,
    floor_F: float = 1.0e-30,
) -> float:
    """Return maximum relative deviation from the nominal finite difference."""

    nominal_delta = _finite_float(
        nominal_delta_voltage_V, "nominal_delta_voltage_V"
    )
    if nominal_delta not in capacitances_F:
        raise KeyError("nominal delta is absent from capacitance results.")
    nominal_value = _finite_float(
        capacitances_F[nominal_delta], "nominal capacitance"
    )
    floor = _finite_float(floor_F, "floor_F")
    if floor <= 0.0:
        raise ValueError("floor_F must be positive.")
    deviations = [
        abs(_finite_float(value, "capacitance") - nominal_value)
        for value in capacitances_F.values()
    ]
    return max(deviations, default=0.0) / max(abs(nominal_value), floor)


def _read_terminal_bias(device: str, terminal: str) -> float:
    from devsim import get_parameter

    return float(
        get_parameter(device=device, name=f"{terminal}_bias")
    )


def _read_trap_density(device: str) -> float:
    from devsim import get_parameter
    from trap_models import TRAPPED_ELECTRON_PARAMETER

    return float(
        get_parameter(
            device=device,
            region=CHARGE_TRAP_REGION,
            name=TRAPPED_ELECTRON_PARAMETER,
        )
    )


def adaptive_micro_ramp(
    terminal: str,
    target_voltage_V: float,
    delta_voltage_V: float,
    *,
    device: str = DEVICE_NAME,
    label: str | None = None,
) -> float:
    """Reach a perturbation voltage with sub-delta continuation steps."""

    normalized_terminal = str(terminal).strip().lower()
    if normalized_terminal not in TERMINALS:
        raise ValueError(f"unsupported terminal {terminal!r}.")
    target = _finite_float(target_voltage_V, "target_voltage_V")
    delta = _finite_float(delta_voltage_V, "delta_voltage_V")
    if delta <= 0.0:
        raise ValueError("delta_voltage_V must be positive.")

    from run_memory_window import adaptive_voltage_ramp

    start = _read_terminal_bias(device, normalized_terminal)
    if math.isclose(start, target, rel_tol=0.0, abs_tol=1.0e-15):
        return target

    initial_step = delta / 2.0
    minimum_step = delta / 64.0
    reached = float(
        adaptive_voltage_ramp(
            terminal=normalized_terminal,
            start_voltage=start,
            target_voltage=target,
            initial_step=initial_step,
            minimum_step=minimum_step,
            label=label or f"{normalized_terminal} capacitance micro-ramp",
        )
    )
    actual = _read_terminal_bias(device, normalized_terminal)
    if not math.isclose(actual, target, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(
            f"{normalized_terminal} perturbation reached {actual}, expected {target}."
        )
    if not math.isclose(reached, actual, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("adaptive-ramp return value and runtime bias disagree.")
    return actual


def measure_gate_capacitance(
    *,
    perturbed_terminal: str,
    base_voltage_V: float,
    delta_voltage_V: float,
    expected_trap_density_cm3: float,
    device: str = DEVICE_NAME,
    ramp_function: Callable[..., float] | None = None,
    gate_charge_function: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Measure one Qg derivative and always attempt to restore its base bias."""

    terminal = str(perturbed_terminal).strip().lower()
    if terminal not in TERMINALS:
        raise ValueError(f"unsupported terminal {perturbed_terminal!r}.")
    base = _finite_float(base_voltage_V, "base_voltage_V")
    delta = _finite_float(delta_voltage_V, "delta_voltage_V")
    expected_trap = _finite_float(
        expected_trap_density_cm3, "expected_trap_density_cm3"
    )
    if delta <= 0.0 or expected_trap < 0.0:
        raise ValueError("delta must be positive and Ntrap nonnegative.")

    if ramp_function is None:
        ramp_function = adaptive_micro_ramp
    if gate_charge_function is None:
        gate_charge_function = lambda: _get_poisson_contact_charge(device, "gate")

    result: dict[str, Any] = {
        "perturbed_terminal": terminal,
        "delta_voltage_V": delta,
        "capacitance_F": math.nan,
        "converged": False,
        "error_message": "",
    }
    primary_error: BaseException | None = None

    try:
        actual_trap = _read_trap_density(device)
        if abs(actual_trap - expected_trap) > 1.0:
            raise RuntimeError(
                f"Ntrap changed before perturbation: {actual_trap:.12e} cm^-3."
            )

        ramp_function(
            terminal,
            base + delta,
            delta,
            device=device,
            label=f"{terminal} +{delta:.3e} V capacitance perturbation",
        )
        charge_plus_C = _finite_float(gate_charge_function(), "Qg plus")

        ramp_function(
            terminal,
            base - delta,
            delta,
            device=device,
            label=f"{terminal} -{delta:.3e} V capacitance perturbation",
        )
        charge_minus_C = _finite_float(gate_charge_function(), "Qg minus")

        actual_trap = _read_trap_density(device)
        if abs(actual_trap - expected_trap) > 1.0:
            raise RuntimeError(
                f"Ntrap changed during perturbation: {actual_trap:.12e} cm^-3."
            )

        result["capacitance_F"] = central_difference(
            charge_plus_C,
            charge_minus_C,
            delta,
        )
        result["converged"] = True
    except Exception as error:  # preserve solver exception context in CSV
        primary_error = error
        result["error_message"] = normalize_error_message(error)
    finally:
        restored = False
        try:
            ramp_function(
                terminal,
                base,
                delta,
                device=device,
                label=f"{terminal} capacitance baseline restore",
            )
            restored = True
        except Exception as restore_error:
            result["capacitance_F"] = math.nan
            result["converged"] = False
            restore_message = normalize_error_message(restore_error)
            if primary_error is None:
                result["error_message"] = "baseline restore failed: " + restore_message
            else:
                result["error_message"] += "; baseline restore failed: " + restore_message

        if restored:
            actual_trap = _read_trap_density(device)
            if abs(actual_trap - expected_trap) > 1.0:
                result["capacitance_F"] = math.nan
                result["converged"] = False
                message = (
                    "Ntrap changed after baseline restore: "
                    f"{actual_trap:.12e} cm^-3"
                )
                result["error_message"] = (
                    str(result["error_message"]) + "; " + message
                ).strip("; ")

    return result


def build_capacitance_matrix_rows(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    gate_results: Mapping[tuple[str, float], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build supported gate rows plus explicit unsupported drain/source rows."""

    state_index, state_name, ntrap_cm3, nsheet_cm2 = _state_values(state)
    rows: list[dict[str, Any]] = []
    for perturbed_terminal, delta in sorted(
        gate_results,
        key=lambda item: (float(item[1]), TERMINALS.index(item[0])),
    ):
        result = gate_results[(perturbed_terminal, delta)]
        for measured_terminal in MEASURED_TERMINALS:
            supported = measured_terminal == "gate"
            row = {
                "state_index": state_index,
                "state": state_name,
                "ntrap_cm3": ntrap_cm3,
                "nsheet_cm2": nsheet_cm2,
                "VGS_V": float(VGS_V),
                "VDS_V": float(VDS_V),
                "perturbed_terminal": perturbed_terminal,
                "measured_terminal": measured_terminal,
                "capacitance_F": (
                    float(result["capacitance_F"]) if supported else math.nan
                ),
                "delta_voltage_V": float(delta),
                "method": CAPACITANCE_METHOD,
                "supported": supported,
                "limitation": "" if supported else UNSUPPORTED_PARTITION_LIMITATION,
                # Convergence describes the perturbation solves.  `supported`
                # separately distinguishes intentional NaN matrix entries.
                "converged": bool(result["converged"]),
                "error_message": str(result.get("error_message", "")),
            }
            if tuple(row) != CAPACITANCE_MATRIX_FIELDNAMES:
                raise RuntimeError("capacitance matrix row does not match schema.")
            rows.append(row)
    return rows


def build_capacitance_summary_row(
    state: Mapping[str, Any],
    *,
    VGS_V: float,
    VDS_V: float,
    gate_results: Mapping[tuple[str, float], Mapping[str, Any]],
    nominal_delta_voltage_V: float,
    sensitivity_delta_voltages_V: Sequence[float],
) -> dict[str, Any]:
    """Summarize nominal Cgg/Cgd/Cgs and finite-difference sensitivity."""

    state_index, state_name, ntrap_cm3, nsheet_cm2 = _state_values(state)
    nominal = float(nominal_delta_voltage_V)
    deltas = tuple(float(value) for value in sensitivity_delta_voltages_V)
    terminal_columns = {
        "gate": ("Cgg_F", "Cgg_relative_sensitivity"),
        "drain": ("Cgd_F", "Cgd_relative_sensitivity"),
        "source": ("Cgs_F", "Cgs_relative_sensitivity"),
    }
    values: dict[str, float] = {}
    errors: list[str] = []

    for terminal, (value_column, sensitivity_column) in terminal_columns.items():
        by_delta: dict[float, float] = {}
        for delta in deltas:
            result = gate_results.get((terminal, delta))
            if result is None or not bool(result.get("converged")):
                message = "missing or failed capacitance result"
                if result is not None and result.get("error_message"):
                    message = str(result["error_message"])
                errors.append(f"{terminal} at {delta:.6e} V: {message}")
                continue
            capacitance = float(result["capacitance_F"])
            if not math.isfinite(capacitance):
                errors.append(f"{terminal} at {delta:.6e} V is non-finite")
                continue
            by_delta[delta] = capacitance

        if nominal in by_delta:
            values[value_column] = by_delta[nominal]
        else:
            values[value_column] = math.nan

        if len(by_delta) == len(deltas) and nominal in by_delta:
            values[sensitivity_column] = calculate_relative_sensitivity(
                by_delta,
                nominal_delta_voltage_V=nominal,
            )
        else:
            values[sensitivity_column] = math.nan

    converged = not errors
    row = {
        "state_index": state_index,
        "state": state_name,
        "ntrap_cm3": ntrap_cm3,
        "nsheet_cm2": nsheet_cm2,
        "VGS_V": float(VGS_V),
        "VDS_V": float(VDS_V),
        "Cgg_F": values["Cgg_F"],
        "Cgd_F": values["Cgd_F"],
        "Cgs_F": values["Cgs_F"],
        "nominal_delta_voltage_V": nominal,
        "sensitivity_delta_voltages_V": ",".join(
            f"{delta:.12g}" for delta in deltas
        ),
        "Cgg_relative_sensitivity": values["Cgg_relative_sensitivity"],
        "Cgd_relative_sensitivity": values["Cgd_relative_sensitivity"],
        "Cgs_relative_sensitivity": values["Cgs_relative_sensitivity"],
        "gate_capacitance_row_sum_F": math.fsum(
            (
                values["Cgg_F"],
                values["Cgd_F"],
                values["Cgs_F"],
            )
        ) if all(
            math.isfinite(values[name])
            for name in ("Cgg_F", "Cgd_F", "Cgs_F")
        ) else math.nan,
        "method": CAPACITANCE_METHOD,
        "supported_quantities": SUPPORTED_QUANTITIES,
        "limitation": UNSUPPORTED_PARTITION_LIMITATION,
        "converged": converged,
        "error_message": "; ".join(errors),
    }
    if tuple(row) != CAPACITANCE_SUMMARY_FIELDNAMES:
        raise RuntimeError("capacitance summary row does not match schema.")
    return row


def validate_capacitance_summary_diagnostics(
    rows: Sequence[Mapping[str, Any]],
    *,
    maximum_relative_sensitivity: float = (
        MAX_CAPACITANCE_RELATIVE_SENSITIVITY
    ),
    maximum_gate_row_sum_F: float = MAX_GATE_CAPACITANCE_ROW_SUM_F,
) -> dict[str, float]:
    """Enforce explicit stability and gate-row-conservation limits."""

    sensitivity_limit = _finite_float(
        maximum_relative_sensitivity,
        "maximum_relative_sensitivity",
    )
    row_sum_limit = _finite_float(
        maximum_gate_row_sum_F,
        "maximum_gate_row_sum_F",
    )
    if sensitivity_limit <= 0.0 or row_sum_limit <= 0.0:
        raise ValueError("capacitance diagnostic limits must be positive.")
    if not rows:
        raise ValueError("at least one capacitance summary row is required.")

    max_sensitivity = 0.0
    max_row_sum = 0.0
    for index, row in enumerate(rows):
        if not bool(row.get("converged")):
            raise RuntimeError(
                f"capacitance summary row {index} did not converge."
            )
        for field in (
            "Cgg_relative_sensitivity",
            "Cgd_relative_sensitivity",
            "Cgs_relative_sensitivity",
        ):
            value = _finite_float(row.get(field), f"row {index} {field}")
            if value < 0.0:
                raise RuntimeError(f"row {index} {field} is negative.")
            max_sensitivity = max(max_sensitivity, value)
        row_sum = abs(
            _finite_float(
                row.get("gate_capacitance_row_sum_F"),
                f"row {index} gate_capacitance_row_sum_F",
            )
        )
        max_row_sum = max(max_row_sum, row_sum)

    if max_sensitivity > sensitivity_limit:
        raise RuntimeError(
            "capacitance finite-difference sensitivity exceeds limit: "
            f"{max_sensitivity:.12e} > {sensitivity_limit:.12e}."
        )
    if max_row_sum > row_sum_limit:
        raise RuntimeError(
            "gate capacitance row-sum residual exceeds limit: "
            f"{max_row_sum:.12e} F > {row_sum_limit:.12e} F."
        )
    return {
        "maximum_relative_sensitivity": max_sensitivity,
        "maximum_gate_row_sum_F": max_row_sum,
        "relative_sensitivity_limit": sensitivity_limit,
        "gate_row_sum_limit_F": row_sum_limit,
    }


__all__ = [
    "CAPACITANCE_MATRIX_FIELDNAMES",
    "CAPACITANCE_METHOD",
    "CAPACITANCE_SUMMARY_FIELDNAMES",
    "CHARGE_METHOD",
    "DEVICE_NAME",
    "SUPPORTED_QUANTITIES",
    "TERMINALS",
    "TERMINAL_CHARGE_FIELDNAMES",
    "UNSUPPORTED_PARTITION_LIMITATION",
    "adaptive_micro_ramp",
    "build_capacitance_matrix_rows",
    "build_capacitance_summary_row",
    "build_failed_terminal_charge_row",
    "build_terminal_charge_row",
    "calculate_charge_conservation",
    "calculate_relative_sensitivity",
    "calculate_semiconductor_charge_components",
    "central_difference",
    "charge_conservation_within_tolerance",
    "extract_terminal_charge_components",
    "integrate_cylindrical_node_quantity",
    "measure_gate_capacitance",
    "normalize_error_message",
    "validate_capacitance_summary_diagnostics",
    "validate_devsim_charge_runtime",
]
