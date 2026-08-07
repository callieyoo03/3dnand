"""Potential-profile framework for localized electron-trap emission.

This module is deliberately independent of the electron-FN negative-control
baseline.  It does not import DEVSIM at module import time, does not contain a
default trap level, and does not invent a tunnelling mass or attempt frequency.
The default mode is therefore ``disabled``.

Energy convention
-----------------
Conduction-band reference energies are supplied in electron-volts relative to
one explicitly named common reference.  For an electron in electrostatic
potential ``phi`` (volts),

    E_c(r) [eV] = E_c,0(material) [eV] - phi(r) [V].

The currently supported localized level is specified as a positive depth
below the *local charge-trap conduction band*.  Thus

    E_t [eV] = E_c,0(trap material) - phi(r_t) - depth.

The WKB path is constructed from the full radial ``Potential`` profile, not a
first-cell electric field.  With coordinates converted from cm to m,

    S = integral sqrt(2 m*(r) max(E_c(r)-E_t, 0)) / hbar dr,
    T = exp(-2 S).

Rates have units s^-1 and use an explicitly supplied attempt-frequency
partition.  No hole-assisted process is represented here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence


ELEMENTARY_CHARGE_C = 1.602176634e-19
REDUCED_PLANCK_CONSTANT_J_S = 1.054571817e-34
FREE_ELECTRON_MASS_KG = 9.1093837139e-31
CM_TO_M = 1.0e-2

TRAP_EMISSION_MODEL_MODE = "disabled"
PREDICTIVE_ERASE_MODEL = False
HOLE_ASSISTED_MODEL_IMPLEMENTED = False
SUPPORTED_TRAP_ENERGY_REFERENCE = "local_charge_trap_conduction_band"


class TrapEmissionMode(str, Enum):
    """Permitted activation modes for the framework."""

    DISABLED = "disabled"
    SENSITIVITY_ONLY = "sensitivity_only"
    USER_SUPPLIED = "user_supplied"
    LITERATURE_PARAMETER_SET = "literature_parameter_set"


class SpatialDistribution(str, Enum):
    """Labels for extensible radial trap populations."""

    SINGLE_POSITION = "single_position"
    UNIFORM = "uniform"
    INTERFACE_LOCALIZED = "interface_localized"
    EXPLICIT_BINS = "explicit_bins"


class ProvenanceKind(str, Enum):
    SYNTHETIC_TEST = "synthetic_test"
    USER_SUPPLIED = "user_supplied"
    LITERATURE = "literature"


class MissingEmissionParameterError(ValueError):
    """Raised before an active calculation when inputs are incomplete."""

    def __init__(self, missing_parameters: Sequence[str]):
        self.missing_parameters = tuple(sorted(set(missing_parameters)))
        super().__init__(
            "Missing localized electron-emission parameter(s): "
            + ", ".join(self.missing_parameters)
        )


def _finite(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite.")
    return numeric


def _positive(name: str, value: float) -> float:
    numeric = _finite(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return numeric


def _nonnegative(name: str, value: float) -> float:
    numeric = _finite(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative.")
    return numeric


def _frozen_float_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    converted: dict[str, float] = {}
    for key, value in values.items():
        label = str(key).strip()
        if not label:
            raise ValueError("Material labels must not be empty.")
        converted[label] = _finite(f"value for {label}", value)
    return MappingProxyType(converted)


@dataclass(frozen=True)
class ParameterProvenance:
    """Auditable origin of one active parameter set."""

    kind: ProvenanceKind
    reference: str
    notes: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProvenanceKind(self.kind))
        if not self.reference.strip():
            raise ValueError("Parameter provenance requires a reference.")


@dataclass(frozen=True)
class TrapEnergyBin:
    """One localized level, expressed below the local trap-layer E_c."""

    depth_below_local_conduction_band_eV: float
    population_weight: float = 1.0
    label: str = "single_level"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth_below_local_conduction_band_eV",
            _nonnegative(
                "depth_below_local_conduction_band_eV",
                self.depth_below_local_conduction_band_eV,
            ),
        )
        object.__setattr__(
            self,
            "population_weight",
            _nonnegative("population_weight", self.population_weight),
        )
        if not self.label.strip():
            raise ValueError("Trap-energy-bin label must not be empty.")


@dataclass(frozen=True)
class RadialTrapPositionBin:
    """One localized radial population bin; radius is in centimetres."""

    radius_cm: float
    population_weight: float = 1.0
    material: str = "ChargeTrap"
    label: str = "single_position"

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_cm", _positive("radius_cm", self.radius_cm))
        object.__setattr__(
            self,
            "population_weight",
            _nonnegative("population_weight", self.population_weight),
        )
        if not self.material.strip() or not self.label.strip():
            raise ValueError("Trap position material and label must not be empty.")


@dataclass(frozen=True)
class AttemptFrequencyPartition:
    """Pre-transmission partition of localized escape attempts."""

    channel_weight: float
    gate_weight: float
    assumption: str

    def __post_init__(self) -> None:
        channel = _nonnegative("channel_weight", self.channel_weight)
        gate = _nonnegative("gate_weight", self.gate_weight)
        if not math.isclose(channel + gate, 1.0, rel_tol=1.0e-12, abs_tol=1.0e-15):
            raise ValueError("Attempt-frequency channel/gate weights must sum to one.")
        if not self.assumption.strip():
            raise ValueError("Attempt-frequency partition requires an assumption.")
        object.__setattr__(self, "channel_weight", channel)
        object.__setattr__(self, "gate_weight", gate)


@dataclass(frozen=True)
class ElectronTrapEmissionParameters:
    """Possibly incomplete parameter set used by fail-fast activation.

    Optional fields are intentional: an audit may represent missing repository
    data without supplying surrogate values.  ``validate_complete`` is called
    automatically before every active WKB evaluation.
    """

    attempt_frequency_Hz: Optional[float] = None
    trap_energy_reference: Optional[str] = None
    band_edge_reference: Optional[str] = None
    conduction_band_edge_eV_by_material: Mapping[str, float] = field(
        default_factory=dict
    )
    electron_effective_mass_ratio_by_material: Mapping[str, float] = field(
        default_factory=dict
    )
    energy_bins: tuple[TrapEnergyBin, ...] = ()
    radial_position_bins: tuple[RadialTrapPositionBin, ...] = ()
    spatial_distribution: Optional[SpatialDistribution] = None
    stable_trap_fraction: Optional[float] = None
    attempt_partition: Optional[AttemptFrequencyPartition] = None
    provenance: Optional[ParameterProvenance] = None

    def __post_init__(self) -> None:
        if self.attempt_frequency_Hz is not None:
            object.__setattr__(
                self,
                "attempt_frequency_Hz",
                _positive("attempt_frequency_Hz", self.attempt_frequency_Hz),
            )
        if self.stable_trap_fraction is not None:
            stable = _finite("stable_trap_fraction", self.stable_trap_fraction)
            if not 0.0 <= stable <= 1.0:
                raise ValueError("stable_trap_fraction must lie in [0, 1].")
            object.__setattr__(self, "stable_trap_fraction", stable)
        if self.spatial_distribution is not None:
            object.__setattr__(
                self,
                "spatial_distribution",
                SpatialDistribution(self.spatial_distribution),
            )
        object.__setattr__(
            self,
            "conduction_band_edge_eV_by_material",
            _frozen_float_mapping(self.conduction_band_edge_eV_by_material),
        )
        masses = _frozen_float_mapping(self.electron_effective_mass_ratio_by_material)
        if any(value <= 0.0 for value in masses.values()):
            raise ValueError("Every electron tunnelling mass ratio must be positive.")
        object.__setattr__(self, "electron_effective_mass_ratio_by_material", masses)
        object.__setattr__(self, "energy_bins", tuple(self.energy_bins))
        object.__setattr__(self, "radial_position_bins", tuple(self.radial_position_bins))

    def missing_required_parameters(
        self,
        profile_materials: Sequence[str] = (),
    ) -> tuple[str, ...]:
        missing: list[str] = []
        if self.attempt_frequency_Hz is None:
            missing.append("attempt_frequency_Hz")
        if self.trap_energy_reference is None:
            missing.append("trap_energy_reference")
        elif self.trap_energy_reference != SUPPORTED_TRAP_ENERGY_REFERENCE:
            missing.append(
                f"supported trap_energy_reference ({SUPPORTED_TRAP_ENERGY_REFERENCE})"
            )
        if not (self.band_edge_reference or "").strip():
            missing.append("band_edge_reference")
        if not self.energy_bins:
            missing.append("HfO2 electron trap energy/depth")
        if not self.radial_position_bins:
            missing.append("trap spatial distribution/position bins")
        if self.spatial_distribution is None:
            missing.append("spatial_distribution")
        # ``stable_trap_fraction`` is intentionally audit-only at this stage.
        # The requested single-population equation has no residual floor:
        # dN/dt = -Gamma_total*N.  A future floor-bearing population model must
        # require this parameter explicitly rather than changing this equation.
        if self.attempt_partition is None:
            missing.append("channel/gate emission branching assumption")
        if self.provenance is None:
            missing.append("parameter provenance")
        for material in sorted(set(profile_materials)):
            if material not in self.conduction_band_edge_eV_by_material:
                missing.append(f"{material} conduction-band edge/offset")
            if material not in self.electron_effective_mass_ratio_by_material:
                missing.append(f"{material} electron tunnelling effective mass")
        for position in self.radial_position_bins:
            if position.material not in self.conduction_band_edge_eV_by_material:
                missing.append(f"{position.material} conduction-band edge/offset")
        if self.energy_bins and sum(bin.population_weight for bin in self.energy_bins) <= 0.0:
            missing.append("positive trap-energy population weight")
        if self.radial_position_bins and sum(
            bin.population_weight for bin in self.radial_position_bins
        ) <= 0.0:
            missing.append("positive radial-position population weight")
        return tuple(sorted(set(missing)))


@dataclass(frozen=True)
class TrapEmissionConfiguration:
    mode: TrapEmissionMode = TrapEmissionMode.DISABLED
    parameters: Optional[ElectronTrapEmissionParameters] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", TrapEmissionMode(self.mode))


@dataclass(frozen=True)
class RadialPotentialSegment:
    """Linear potential segment within one material."""

    inner_radius_cm: float
    outer_radius_cm: float
    inner_potential_V: float
    outer_potential_V: float
    material: str

    def __post_init__(self) -> None:
        inner = _positive("inner_radius_cm", self.inner_radius_cm)
        outer = _positive("outer_radius_cm", self.outer_radius_cm)
        if outer <= inner:
            raise ValueError("Radial-potential segments require outer > inner radius.")
        object.__setattr__(self, "inner_radius_cm", inner)
        object.__setattr__(self, "outer_radius_cm", outer)
        object.__setattr__(
            self, "inner_potential_V", _finite("inner_potential_V", self.inner_potential_V)
        )
        object.__setattr__(
            self, "outer_potential_V", _finite("outer_potential_V", self.outer_potential_V)
        )
        if not self.material.strip():
            raise ValueError("Segment material must not be empty.")

    def potential_at(self, radius_cm: float) -> float:
        radius = _finite("radius_cm", radius_cm)
        tolerance = 1.0e-15
        if not self.inner_radius_cm - tolerance <= radius <= self.outer_radius_cm + tolerance:
            raise ValueError("Radius lies outside this potential segment.")
        fraction = (radius - self.inner_radius_cm) / (
            self.outer_radius_cm - self.inner_radius_cm
        )
        fraction = min(1.0, max(0.0, fraction))
        return self.inner_potential_V + fraction * (
            self.outer_potential_V - self.inner_potential_V
        )


@dataclass(frozen=True)
class RadialPotentialProfile:
    segments: tuple[RadialPotentialSegment, ...]
    axial_position_cm: float
    source: str = "DEVSIM Potential node model"

    def __post_init__(self) -> None:
        segments = tuple(self.segments)
        if not segments:
            raise ValueError("A radial Potential profile requires at least one segment.")
        for previous, current in zip(segments, segments[1:]):
            if current.inner_radius_cm < previous.outer_radius_cm - 1.0e-15:
                raise ValueError("Radial Potential segments overlap or are unordered.")
            if not math.isclose(
                current.inner_radius_cm,
                previous.outer_radius_cm,
                rel_tol=0.0,
                abs_tol=1.0e-15,
            ):
                raise ValueError("Radial Potential profile contains an unmodelled gap.")
        object.__setattr__(self, "segments", segments)
        object.__setattr__(
            self, "axial_position_cm", _finite("axial_position_cm", self.axial_position_cm)
        )
        if not self.source.strip():
            raise ValueError("Potential-profile source must not be empty.")

    @property
    def inner_radius_cm(self) -> float:
        return self.segments[0].inner_radius_cm

    @property
    def outer_radius_cm(self) -> float:
        return self.segments[-1].outer_radius_cm

    @property
    def materials(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(segment.material for segment in self.segments))

    def locate(self, radius_cm: float, preferred_material: Optional[str] = None) -> RadialPotentialSegment:
        radius = _finite("radius_cm", radius_cm)
        candidates = [
            segment
            for segment in self.segments
            if segment.inner_radius_cm - 1.0e-15
            <= radius
            <= segment.outer_radius_cm + 1.0e-15
        ]
        if preferred_material is not None:
            preferred = [s for s in candidates if s.material == preferred_material]
            if preferred:
                candidates = preferred
        if not candidates:
            raise ValueError("Trap radius is outside the supplied Potential profile.")
        return candidates[0]


@dataclass(frozen=True)
class BandEdgeSegment:
    inner_radius_cm: float
    outer_radius_cm: float
    inner_conduction_band_eV: float
    outer_conduction_band_eV: float
    material: str


@dataclass(frozen=True)
class WKBPathResult:
    destination: str
    endpoint_radius_cm: float
    path_length_cm: float
    action_integral_dimensionless: float
    transmission_exponent: float
    log_transmission: float
    transmission: float
    segment_count: int


@dataclass(frozen=True)
class TrapEmissionResult:
    supported: bool
    mode: str
    model_status: str
    unsupported_reason: Optional[str]
    predictive_erase_model: bool
    hole_assisted_model_implemented: bool
    trap_energy_eV: Optional[float] = None
    gamma_channel_s1: Optional[float] = None
    gamma_gate_s1: Optional[float] = None
    gamma_total_s1: Optional[float] = None
    log_gamma_channel_s1: Optional[float] = None
    log_gamma_gate_s1: Optional[float] = None
    log_gamma_total_s1: Optional[float] = None
    channel_branching_ratio: Optional[float] = None
    gate_branching_ratio: Optional[float] = None
    channel_path: Optional[WKBPathResult] = None
    gate_path: Optional[WKBPathResult] = None


@dataclass(frozen=True)
class TrapEmissionGridResult:
    supported: bool
    mode: str
    results: tuple[TrapEmissionResult, ...]
    population_weights: tuple[float, ...]
    initial_population_average_gamma_total_s1: Optional[float]
    unsupported_reason: Optional[str] = None


def build_radial_potential_profile(
    radii_cm: Sequence[float],
    potentials_V: Sequence[float],
    interval_materials: Sequence[str],
    axial_position_cm: float,
    source: str = "DEVSIM Potential node model",
) -> RadialPotentialProfile:
    """Build a piecewise-linear radial profile from ordered node samples."""

    radii = tuple(_positive("radius_cm", value) for value in radii_cm)
    potentials = tuple(_finite("potential_V", value) for value in potentials_V)
    materials = tuple(str(value) for value in interval_materials)
    if len(radii) < 2 or len(potentials) != len(radii):
        raise ValueError("Radii and potentials must have the same length >= 2.")
    if len(materials) != len(radii) - 1:
        raise ValueError("One interval material is required per radial interval.")
    if any(outer <= inner for inner, outer in zip(radii, radii[1:])):
        raise ValueError("Radial samples must be strictly increasing.")
    return RadialPotentialProfile(
        segments=tuple(
            RadialPotentialSegment(
                inner_radius_cm=radii[index],
                outer_radius_cm=radii[index + 1],
                inner_potential_V=potentials[index],
                outer_potential_V=potentials[index + 1],
                material=materials[index],
            )
            for index in range(len(materials))
        ),
        axial_position_cm=axial_position_cm,
        source=source,
    )


def extract_devsim_radial_potential_profile(
    device: str,
    axial_position_cm: float,
    regions: Sequence[str] = ("TunnelOxide", "ChargeTrap", "BlockingOxide"),
    runtime_api: Any = None,
    maximum_axial_distance_cm: float = 1.0e-12,
) -> RadialPotentialProfile:
    """Extract a radial profile through DEVSIM without a hard import dependency.

    The nearest axial mesh line is selected independently in each requested
    region.  Each material's radial nodes become linear Potential segments.
    """

    if runtime_api is None:
        import devsim as runtime_api  # type: ignore[no-redef]

    target = _finite("axial_position_cm", axial_position_cm)
    tolerance = _nonnegative("maximum_axial_distance_cm", maximum_axial_distance_cm)
    segments: list[RadialPotentialSegment] = []
    selected_axial_positions: list[float] = []
    for region in regions:
        x_values = tuple(
            float(value)
            for value in runtime_api.get_node_model_values(
                device=device, region=region, name="x"
            )
        )
        y_values = tuple(
            float(value)
            for value in runtime_api.get_node_model_values(
                device=device, region=region, name="y"
            )
        )
        potential_values = tuple(
            float(value)
            for value in runtime_api.get_node_model_values(
                device=device, region=region, name="Potential"
            )
        )
        if not x_values or not (len(x_values) == len(y_values) == len(potential_values)):
            raise RuntimeError(f"Invalid DEVSIM node arrays for region {region}.")
        selected_y = min(set(y_values), key=lambda value: abs(value - target))
        if abs(selected_y - target) > tolerance:
            raise RuntimeError(
                f"Nearest axial mesh line in {region} is too far from target: "
                f"{abs(selected_y - target):.6e} cm."
            )
        selected_axial_positions.append(selected_y)
        radial_values: dict[float, list[float]] = {}
        for radius, axial, potential in zip(x_values, y_values, potential_values):
            if math.isclose(axial, selected_y, rel_tol=0.0, abs_tol=1.0e-15):
                radial_values.setdefault(radius, []).append(potential)
        ordered = sorted(
            (radius, sum(values) / len(values))
            for radius, values in radial_values.items()
        )
        if len(ordered) < 2:
            raise RuntimeError(f"Region {region} has fewer than two radial samples.")
        for (inner_r, inner_v), (outer_r, outer_v) in zip(ordered, ordered[1:]):
            segments.append(
                RadialPotentialSegment(
                    inner_radius_cm=inner_r,
                    outer_radius_cm=outer_r,
                    inner_potential_V=inner_v,
                    outer_potential_V=outer_v,
                    material=str(region),
                )
            )
    segments.sort(key=lambda segment: (segment.inner_radius_cm, segment.outer_radius_cm))
    return RadialPotentialProfile(
        segments=tuple(segments),
        axial_position_cm=sum(selected_axial_positions) / len(selected_axial_positions),
        source="DEVSIM Potential node model; nearest axial mesh line",
    )


def build_band_edge_profile(
    profile: RadialPotentialProfile,
    conduction_band_edge_eV_by_material: Mapping[str, float],
) -> tuple[BandEdgeSegment, ...]:
    """Construct E_c(r)=E_c,0(material)-Potential(r), in eV."""

    offsets = _frozen_float_mapping(conduction_band_edge_eV_by_material)
    missing = sorted(set(profile.materials) - set(offsets))
    if missing:
        raise MissingEmissionParameterError(
            [f"{material} conduction-band edge/offset" for material in missing]
        )
    return tuple(
        BandEdgeSegment(
            inner_radius_cm=segment.inner_radius_cm,
            outer_radius_cm=segment.outer_radius_cm,
            inner_conduction_band_eV=(
                offsets[segment.material] - segment.inner_potential_V
            ),
            outer_conduction_band_eV=(
                offsets[segment.material] - segment.outer_potential_V
            ),
            material=segment.material,
        )
        for segment in profile.segments
    )


def _linear_sqrt_positive_integral(
    start_eV: float,
    stop_eV: float,
    length_m: float,
) -> float:
    """Return integral sqrt(max(linear barrier in eV, 0)) dx."""

    if length_m <= 0.0:
        return 0.0
    start = float(start_eV)
    stop = float(stop_eV)
    if start <= 0.0 and stop <= 0.0:
        return 0.0
    if math.isclose(start, stop, rel_tol=1.0e-14, abs_tol=1.0e-15):
        return length_m * math.sqrt(max(start, 0.0))
    if start >= 0.0 and stop >= 0.0:
        return (
            length_m
            * (2.0 / 3.0)
            * (stop ** 1.5 - start ** 1.5)
            / (stop - start)
        )
    if start > 0.0:
        positive_fraction = start / (start - stop)
        return length_m * positive_fraction * (2.0 / 3.0) * math.sqrt(start)
    positive_fraction = stop / (stop - start)
    return length_m * positive_fraction * (2.0 / 3.0) * math.sqrt(stop)


def calculate_wkb_path(
    band_profile: Sequence[BandEdgeSegment],
    trap_radius_cm: float,
    endpoint_radius_cm: float,
    trap_energy_eV: float,
    electron_effective_mass_ratio_by_material: Mapping[str, float],
    destination: str,
) -> WKBPathResult:
    """Integrate a channel- or gate-directed WKB path through U(r)."""

    trap_radius = _positive("trap_radius_cm", trap_radius_cm)
    endpoint = _positive("endpoint_radius_cm", endpoint_radius_cm)
    trap_energy = _finite("trap_energy_eV", trap_energy_eV)
    if math.isclose(trap_radius, endpoint, rel_tol=0.0, abs_tol=1.0e-18):
        raise ValueError("WKB endpoint must differ from the trap radius.")
    masses = _frozen_float_mapping(electron_effective_mass_ratio_by_material)
    lower, upper = sorted((trap_radius, endpoint))
    action = 0.0
    used_segments = 0
    covered_length_cm = 0.0
    for segment in band_profile:
        overlap_inner = max(lower, segment.inner_radius_cm)
        overlap_outer = min(upper, segment.outer_radius_cm)
        if overlap_outer <= overlap_inner:
            continue
        if segment.material not in masses:
            raise MissingEmissionParameterError(
                [f"{segment.material} electron tunnelling effective mass"]
            )
        mass_ratio = masses[segment.material]
        if mass_ratio <= 0.0:
            raise ValueError("Electron tunnelling mass ratios must be positive.")
        width = segment.outer_radius_cm - segment.inner_radius_cm
        inner_fraction = (overlap_inner - segment.inner_radius_cm) / width
        outer_fraction = (overlap_outer - segment.inner_radius_cm) / width
        band_inner = segment.inner_conduction_band_eV + inner_fraction * (
            segment.outer_conduction_band_eV - segment.inner_conduction_band_eV
        )
        band_outer = segment.inner_conduction_band_eV + outer_fraction * (
            segment.outer_conduction_band_eV - segment.inner_conduction_band_eV
        )
        integral_sqrt_eV_m = _linear_sqrt_positive_integral(
            band_inner - trap_energy,
            band_outer - trap_energy,
            (overlap_outer - overlap_inner) * CM_TO_M,
        )
        action += (
            math.sqrt(
                2.0
                * mass_ratio
                * FREE_ELECTRON_MASS_KG
                * ELEMENTARY_CHARGE_C
            )
            / REDUCED_PLANCK_CONSTANT_J_S
            * integral_sqrt_eV_m
        )
        used_segments += 1
        covered_length_cm += overlap_outer - overlap_inner
    required_length_cm = upper - lower
    if not math.isclose(
        covered_length_cm,
        required_length_cm,
        rel_tol=1.0e-10,
        abs_tol=1.0e-15,
    ):
        raise ValueError("The Potential profile does not cover the complete WKB path.")
    log_transmission = -2.0 * action
    transmission = math.exp(log_transmission) if log_transmission > -745.0 else 0.0
    return WKBPathResult(
        destination=str(destination),
        endpoint_radius_cm=endpoint,
        path_length_cm=required_length_cm,
        action_integral_dimensionless=action,
        transmission_exponent=2.0 * action,
        log_transmission=log_transmission,
        transmission=transmission,
        segment_count=used_segments,
    )


def _logaddexp(first: float, second: float) -> float:
    if first == -math.inf:
        return second
    if second == -math.inf:
        return first
    maximum = max(first, second)
    return maximum + math.log(math.exp(first - maximum) + math.exp(second - maximum))


def _validate_active_configuration(
    configuration: TrapEmissionConfiguration,
    profile: RadialPotentialProfile,
) -> ElectronTrapEmissionParameters:
    if configuration.parameters is None:
        raise MissingEmissionParameterError(["electron-trap emission parameter set"])
    parameters = configuration.parameters
    missing = parameters.missing_required_parameters(profile.materials)
    if missing:
        raise MissingEmissionParameterError(missing)
    assert parameters.provenance is not None
    if configuration.mode is TrapEmissionMode.USER_SUPPLIED and (
        parameters.provenance.kind is not ProvenanceKind.USER_SUPPLIED
    ):
        raise ValueError("user_supplied mode requires user_supplied provenance.")
    if configuration.mode is TrapEmissionMode.LITERATURE_PARAMETER_SET and (
        parameters.provenance.kind is not ProvenanceKind.LITERATURE
    ):
        raise ValueError("literature_parameter_set mode requires literature provenance.")
    return parameters


def evaluate_localized_trap_emission(
    configuration: TrapEmissionConfiguration,
    profile: Optional[RadialPotentialProfile] = None,
    energy_bin: Optional[TrapEnergyBin] = None,
    position_bin: Optional[RadialTrapPositionBin] = None,
    channel_endpoint_radius_cm: Optional[float] = None,
    gate_endpoint_radius_cm: Optional[float] = None,
) -> TrapEmissionResult:
    """Evaluate one energy/position bin or return explicit disabled status."""

    if configuration.mode is TrapEmissionMode.DISABLED:
        return TrapEmissionResult(
            supported=False,
            mode=configuration.mode.value,
            model_status="disabled_unsupported",
            unsupported_reason=(
                "Localized electron-trap emission is disabled; no rate or "
                "terminal branching value was fabricated."
            ),
            predictive_erase_model=PREDICTIVE_ERASE_MODEL,
            hole_assisted_model_implemented=HOLE_ASSISTED_MODEL_IMPLEMENTED,
        )
    if profile is None:
        raise MissingEmissionParameterError(["DEVSIM radial Potential profile"])
    parameters = _validate_active_configuration(configuration, profile)
    selected_energy = energy_bin
    if selected_energy is None:
        if len(parameters.energy_bins) != 1:
            raise ValueError("Select one energy_bin when multiple bins are configured.")
        selected_energy = parameters.energy_bins[0]
    selected_position = position_bin
    if selected_position is None:
        if len(parameters.radial_position_bins) != 1:
            raise ValueError("Select one position_bin when multiple bins are configured.")
        selected_position = parameters.radial_position_bins[0]
    trap_segment = profile.locate(
        selected_position.radius_cm,
        preferred_material=selected_position.material,
    )
    if trap_segment.material != selected_position.material:
        raise ValueError("Trap position is not inside its declared material.")
    trap_potential_V = trap_segment.potential_at(selected_position.radius_cm)
    trap_band_reference_eV = parameters.conduction_band_edge_eV_by_material[
        selected_position.material
    ]
    trap_energy_eV = (
        trap_band_reference_eV
        - trap_potential_V
        - selected_energy.depth_below_local_conduction_band_eV
    )
    band_profile = build_band_edge_profile(
        profile, parameters.conduction_band_edge_eV_by_material
    )
    channel_endpoint = (
        profile.inner_radius_cm
        if channel_endpoint_radius_cm is None
        else _positive("channel_endpoint_radius_cm", channel_endpoint_radius_cm)
    )
    gate_endpoint = (
        profile.outer_radius_cm
        if gate_endpoint_radius_cm is None
        else _positive("gate_endpoint_radius_cm", gate_endpoint_radius_cm)
    )
    if not channel_endpoint < selected_position.radius_cm < gate_endpoint:
        raise ValueError("Trap radius must lie between channel and gate endpoints.")
    channel_path = calculate_wkb_path(
        band_profile,
        selected_position.radius_cm,
        channel_endpoint,
        trap_energy_eV,
        parameters.electron_effective_mass_ratio_by_material,
        destination="channel",
    )
    gate_path = calculate_wkb_path(
        band_profile,
        selected_position.radius_cm,
        gate_endpoint,
        trap_energy_eV,
        parameters.electron_effective_mass_ratio_by_material,
        destination="gate",
    )
    assert parameters.attempt_frequency_Hz is not None
    assert parameters.attempt_partition is not None
    log_attempt = math.log(parameters.attempt_frequency_Hz)
    channel_weight = parameters.attempt_partition.channel_weight
    gate_weight = parameters.attempt_partition.gate_weight
    log_gamma_channel = (
        -math.inf
        if channel_weight == 0.0
        else log_attempt + math.log(channel_weight) + channel_path.log_transmission
    )
    log_gamma_gate = (
        -math.inf
        if gate_weight == 0.0
        else log_attempt + math.log(gate_weight) + gate_path.log_transmission
    )
    log_gamma_total = _logaddexp(log_gamma_channel, log_gamma_gate)
    gamma_channel = math.exp(log_gamma_channel) if log_gamma_channel > -745.0 else 0.0
    gamma_gate = math.exp(log_gamma_gate) if log_gamma_gate > -745.0 else 0.0
    gamma_total = math.exp(log_gamma_total) if log_gamma_total > -745.0 else 0.0
    if log_gamma_total == -math.inf:
        # Both rates are exactly absent, so a branching ratio is undefined.
        channel_branching = None
        gate_branching = None
    else:
        channel_branching = (
            0.0
            if log_gamma_channel == -math.inf
            else math.exp(log_gamma_channel - log_gamma_total)
        )
        gate_branching = (
            0.0
            if log_gamma_gate == -math.inf
            else math.exp(log_gamma_gate - log_gamma_total)
        )
    return TrapEmissionResult(
        supported=True,
        mode=configuration.mode.value,
        model_status=(
            "sensitivity_only"
            if configuration.mode is TrapEmissionMode.SENSITIVITY_ONLY
            else "parameterized_framework_unvalidated"
        ),
        unsupported_reason=None,
        predictive_erase_model=PREDICTIVE_ERASE_MODEL,
        hole_assisted_model_implemented=HOLE_ASSISTED_MODEL_IMPLEMENTED,
        trap_energy_eV=trap_energy_eV,
        gamma_channel_s1=gamma_channel,
        gamma_gate_s1=gamma_gate,
        gamma_total_s1=gamma_total,
        log_gamma_channel_s1=log_gamma_channel,
        log_gamma_gate_s1=log_gamma_gate,
        log_gamma_total_s1=log_gamma_total,
        channel_branching_ratio=channel_branching,
        gate_branching_ratio=gate_branching,
        channel_path=channel_path,
        gate_path=gate_path,
    )


def evaluate_trap_emission_grid(
    configuration: TrapEmissionConfiguration,
    profile: Optional[RadialPotentialProfile] = None,
) -> TrapEmissionGridResult:
    """Evaluate all configured energy/radial bins without collapsing detail."""

    if configuration.mode is TrapEmissionMode.DISABLED:
        disabled = evaluate_localized_trap_emission(configuration)
        return TrapEmissionGridResult(
            supported=False,
            mode=configuration.mode.value,
            results=(disabled,),
            population_weights=(),
            initial_population_average_gamma_total_s1=None,
            unsupported_reason=disabled.unsupported_reason,
        )
    if profile is None:
        raise MissingEmissionParameterError(["DEVSIM radial Potential profile"])
    parameters = _validate_active_configuration(configuration, profile)
    energy_total = sum(item.population_weight for item in parameters.energy_bins)
    position_total = sum(
        item.population_weight for item in parameters.radial_position_bins
    )
    results: list[TrapEmissionResult] = []
    weights: list[float] = []
    for energy in parameters.energy_bins:
        for position in parameters.radial_position_bins:
            results.append(
                evaluate_localized_trap_emission(
                    configuration,
                    profile,
                    energy_bin=energy,
                    position_bin=position,
                )
            )
            weights.append(
                energy.population_weight
                / energy_total
                * position.population_weight
                / position_total
            )
    average_gamma = sum(
        weight * float(result.gamma_total_s1)
        for weight, result in zip(weights, results)
    )
    return TrapEmissionGridResult(
        supported=True,
        mode=configuration.mode.value,
        results=tuple(results),
        population_weights=tuple(weights),
        initial_population_average_gamma_total_s1=average_gamma,
    )


def trap_density_derivative_cm3_s(
    trap_density_cm3: float,
    gamma_total_s1: float,
) -> float:
    """Return dNtrap/dt = -Gamma_total*Ntrap in cm^-3 s^-1."""

    density = _nonnegative("trap_density_cm3", trap_density_cm3)
    gamma = _nonnegative("gamma_total_s1", gamma_total_s1)
    return -gamma * density


def uniform_radial_position_bins(
    inner_radius_cm: float,
    outer_radius_cm: float,
    count: int,
    material: str = "ChargeTrap",
) -> tuple[RadialTrapPositionBin, ...]:
    """Create midpoint bins only from explicitly supplied spatial limits."""

    inner = _positive("inner_radius_cm", inner_radius_cm)
    outer = _positive("outer_radius_cm", outer_radius_cm)
    if outer <= inner or int(count) != count or int(count) <= 0:
        raise ValueError("Uniform bins require outer > inner and positive integer count.")
    count = int(count)
    spacing = (outer - inner) / count
    return tuple(
        RadialTrapPositionBin(
            radius_cm=inner + (index + 0.5) * spacing,
            population_weight=1.0 / count,
            material=material,
            label=f"uniform_radial_bin_{index}",
        )
        for index in range(count)
    )


def interface_localized_position_bin(
    radius_cm: float,
    interface_label: str,
    material: str = "ChargeTrap",
) -> RadialTrapPositionBin:
    """Represent a user-specified interface-localized trap position."""

    if not interface_label.strip():
        raise ValueError("interface_label must not be empty.")
    return RadialTrapPositionBin(
        radius_cm=radius_cm,
        population_weight=1.0,
        material=material,
        label=f"interface_localized:{interface_label}",
    )
