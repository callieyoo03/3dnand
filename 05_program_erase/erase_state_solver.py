"""Staged DEVSIM state solver for the ERASE electron-FN negative control.

The adaptive continuations in older exploratory runners restored only the
scalar bias or trap parameter after a failed Newton trial.  This module takes
a true checkpoint immediately before every trial and restores all Potential
vectors, the MoS2 Electrons vector, all three terminal biases, trap density,
and last-success metadata without an extra recovery solve.

DEVSIM imports are lazy.  The orchestration and rollback logic can therefore
be tested with a small in-memory runtime stub.
"""

from __future__ import annotations

import importlib
import math
import sys
import traceback as traceback_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import erase_parameters as ep


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT_DIRECTORY = MODULE_DIRECTORY.parent


@dataclass(frozen=True)
class StepRecord:
    """Metadata for the most recent successfully solved continuation step."""

    kind: str
    label: str
    value: float
    step_size: float
    attempt_index: int


@dataclass(frozen=True)
class OperatingState:
    """Scalar coordinates of the current DEVSIM state."""

    gate_voltage_V: float
    drain_voltage_V: float
    source_voltage_V: float
    trap_density_cm3: float
    last_successful_step: StepRecord | None = None


@dataclass(frozen=True)
class SolutionSnapshot:
    """Complete mutable DEVSIM state required for a lossless trial rollback."""

    potential_by_region: Mapping[str, tuple[float, ...]]
    mos2_electrons_cm3: tuple[float, ...]
    gate_voltage_V: float
    drain_voltage_V: float
    source_voltage_V: float
    trap_density_cm3: float
    last_successful_step: StepRecord | None


@dataclass(frozen=True)
class RampAttempt:
    """One continuation trial, including the original solver failure text."""

    kind: str
    label: str
    attempt_index: int
    start_value: float
    trial_value: float
    target_value: float
    step_size: float
    success: bool
    rolled_back: bool
    exception_type: str = ""
    exception_message: str = ""
    exception_traceback: str = ""


@dataclass(frozen=True)
class RampResult:
    """Successful adaptive-ramp result and its complete trial history."""

    kind: str
    label: str
    start_value: float
    target_value: float
    final_value: float
    attempts: tuple[RampAttempt, ...] = field(default_factory=tuple)


class AdaptiveRampError(RuntimeError):
    """Raised after rollback when the required step falls below its minimum."""

    def __init__(
        self,
        message: str,
        *,
        attempts: Sequence[RampAttempt],
        last_converged_state: OperatingState,
    ) -> None:
        super().__init__(message)
        self.attempts = tuple(attempts)
        self.last_converged_state = last_converged_state


class EraseRuntime(Protocol):
    """Small runtime surface used by :class:`EraseStateSolver`."""

    device_name: str
    regions: Sequence[str]
    semiconductor_region: str

    def build_poisson_only(self) -> Mapping[str, Any]: ...
    def initialize_electrons_from_equilibrium(self) -> None: ...
    def activate_electron_dd(self) -> None: ...
    def solve_dc(self) -> None: ...
    def get_node_values(self, region: str, name: str) -> Sequence[float]: ...
    def set_node_values(
        self, region: str, name: str, values: Sequence[float]
    ) -> None: ...
    def get_terminal_bias(self, terminal: str) -> float: ...
    def set_terminal_bias(self, terminal: str, voltage_V: float) -> None: ...
    def get_trap_density(self) -> float: ...
    def set_trap_density(self, density_cm3: float) -> None: ...


def continuation_direction(start_value: float, target_value: float) -> float:
    """Return +1, -1, or 0 for a finite continuation interval."""

    start = float(start_value)
    target = float(target_value)
    if not math.isfinite(start) or not math.isfinite(target):
        raise ValueError("Continuation endpoints must be finite.")
    if target > start:
        return 1.0
    if target < start:
        return -1.0
    return 0.0


def negative_ramp_trial_values(
    start_voltage_V: float,
    target_voltage_V: float,
    step_voltage_V: float,
) -> tuple[float, ...]:
    """Return deterministic negative-ramp trials for direction unit tests."""

    start = float(start_voltage_V)
    target = float(target_voltage_V)
    step = float(step_voltage_V)
    if target > start:
        raise ValueError("A negative gate ramp target must not exceed its start.")
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("step_voltage_V must be finite and positive.")
    values: list[float] = []
    current = start
    while current - target > 1.0e-12:
        current -= min(step, current - target)
        values.append(round(current, 15))
    return tuple(values)


def _insert_runtime_paths() -> None:
    for path in (str(MODULE_DIRECTORY), str(PROJECT_ROOT_DIRECTORY)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _load_local_module(name: str) -> Any:
    """Import one runtime module and reject same-named modules from elsewhere."""

    _insert_runtime_paths()
    module = importlib.import_module(name)
    expected = (MODULE_DIRECTORY / f"{name}.py").resolve()
    actual = Path(getattr(module, "__file__", "")).resolve()
    if actual != expected:
        raise ImportError(f"{name} resolved to {actual}; expected {expected}.")
    return module


class DevsimRuntimeAdapter:
    """Lazy adapter around the repository's established DEVSIM helpers."""

    device_name = "MoS2_GAA"
    regions = (
        "CoreOxide",
        "MoS2",
        "TunnelOxide",
        "ChargeTrap",
        "BlockingOxide",
    )
    semiconductor_region = "MoS2"
    trap_region = "ChargeTrap"

    def __init__(self) -> None:
        _insert_runtime_paths()
        self.devsim = importlib.import_module("devsim")
        self.structure = _load_local_module("device_structure")
        self.material = _load_local_module("material_parameters")
        self.physics = _load_local_module("physics_models")
        self.traps = _load_local_module("trap_models")
        self.trap_parameters = _load_local_module("trap_parameters")
        self.trap_parameter_name = self.traps.TRAPPED_ELECTRON_PARAMETER
        self._electron_dd_active = False

    @staticmethod
    def _validate_geometry(geometry: Mapping[str, Any]) -> None:
        expected = ep.runtime_geometry_expected_nm()
        for name, expected_value in expected.items():
            if name not in geometry:
                raise RuntimeError(f"Runtime geometry is missing {name!r}.")
            actual = float(geometry[name])
            if not math.isclose(
                actual, float(expected_value), rel_tol=0.0, abs_tol=1.0e-9
            ):
                raise RuntimeError(
                    f"Runtime geometry mismatch: {name}={actual:.12e} nm, "
                    f"expected {float(expected_value):.12e} nm."
                )

    def _validate_materials(self) -> None:
        for region, expected_relative in ep.runtime_material_expected().items():
            permittivity = float(
                self.devsim.get_parameter(
                    device=self.device_name,
                    region=region,
                    name="Permittivity",
                )
            )
            actual_relative = permittivity / float(self.material.eps0)
            if not math.isclose(
                actual_relative,
                float(expected_relative),
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"Runtime material mismatch in {region}: "
                    f"relative permittivity {actual_relative:.12e}, "
                    f"expected {float(expected_relative):.12e}."
                )

    def _create_poisson_only_contacts(self) -> None:
        """Create the potential subset of the established ideal contacts."""

        for terminal in ("source", "drain", "gate"):
            self.set_terminal_bias(terminal, 0.0)

        for contact in ("source", "drain"):
            model = f"{contact}_potential_bc"
            self.devsim.contact_node_model(
                device=self.device_name,
                contact=contact,
                name=model,
                equation=(
                    f"Potential-{contact}_bias-ThermalVoltage*"
                    "log(EquilibriumElectrons/IntrinsicDensity)"
                ),
            )
            self.devsim.contact_node_model(
                device=self.device_name,
                contact=contact,
                name=f"{model}:Potential",
                equation="1",
            )
            self.devsim.contact_equation(
                device=self.device_name,
                contact=contact,
                name="PotentialEquation",
                node_model=model,
                edge_charge_model="PotentialEdgeFlux",
            )

        self.devsim.contact_node_model(
            device=self.device_name,
            contact="gate",
            name="gate_potential_bc",
            equation="Potential-gate_bias",
        )
        self.devsim.contact_node_model(
            device=self.device_name,
            contact="gate",
            name="gate_potential_bc:Potential",
            equation="1",
        )
        self.devsim.contact_equation(
            device=self.device_name,
            contact="gate",
            name="PotentialEquation",
            node_model="gate_potential_bc",
            edge_charge_model="PotentialEdgeFlux",
        )

    def build_poisson_only(self) -> Mapping[str, Any]:
        geometry = self.structure.create_structure(
            **ep.compact_parameters.nominal_geometry_kwargs()
        )
        self._validate_geometry(geometry)
        self.physics.set_material_parameters()
        self._validate_materials()
        self.physics.create_doping()
        self.physics.create_solution_variables()
        self.physics.create_equilibrium_carrier_models()
        self.initialize_electrons_from_equilibrium()
        self.physics.create_poisson_model()
        self.traps.create_static_trap_framework()
        self.physics.create_interface_models()
        self._create_poisson_only_contacts()
        self.set_trap_density(0.0)
        return geometry

    def initialize_electrons_from_equilibrium(self) -> None:
        self.devsim.set_node_values(
            device=self.device_name,
            region=self.semiconductor_region,
            name="Electrons",
            init_from="EquilibriumElectrons",
        )

    def activate_electron_dd(self) -> None:
        if self._electron_dd_active:
            return
        self.physics.create_electron_current_model()
        self.physics.create_continuity_equation()
        # Reuses the established ideal-ohmic/gate model.  It replaces the
        # identical potential contact equations and adds electron contacts.
        self.physics.create_contact_models()
        self._electron_dd_active = True

    def solve_dc(self) -> None:
        self.devsim.solve(
            type="dc",
            solver_type="direct",
            absolute_error=ep.SOLVER_ABSOLUTE_ERROR,
            relative_error=ep.SOLVER_RELATIVE_ERROR,
            maximum_iterations=ep.SOLVER_MAXIMUM_ITERATIONS,
        )

    def get_node_values(self, region: str, name: str) -> Sequence[float]:
        return self.devsim.get_node_model_values(
            device=self.device_name,
            region=region,
            name=name,
        )

    def set_node_values(
        self,
        region: str,
        name: str,
        values: Sequence[float],
    ) -> None:
        self.devsim.set_node_values(
            device=self.device_name,
            region=region,
            name=name,
            values=list(values),
        )

    def get_terminal_bias(self, terminal: str) -> float:
        return float(
            self.devsim.get_parameter(
                device=self.device_name,
                name=f"{terminal}_bias",
            )
        )

    def set_terminal_bias(self, terminal: str, voltage_V: float) -> None:
        terminal_name = str(terminal).strip().lower()
        if terminal_name not in {"gate", "source", "drain"}:
            raise ValueError(f"Unsupported terminal {terminal!r}.")
        self.devsim.set_parameter(
            device=self.device_name,
            name=f"{terminal_name}_bias",
            value=float(voltage_V),
        )

    def get_trap_density(self) -> float:
        return float(
            self.devsim.get_parameter(
                device=self.device_name,
                region=self.trap_region,
                name=self.trap_parameter_name,
            )
        )

    def set_trap_density(self, density_cm3: float) -> None:
        self.traps.set_trapped_electron_density(float(density_cm3))


class EraseStateSolver:
    """Orchestrate staged initialization and rollback-safe continuations."""

    def __init__(self, runtime: EraseRuntime | None = None) -> None:
        self.runtime: EraseRuntime = runtime or DevsimRuntimeAdapter()
        self.last_successful_step: StepRecord | None = None
        self.geometry: dict[str, Any] | None = None

    def read_state(self) -> OperatingState:
        return OperatingState(
            gate_voltage_V=float(self.runtime.get_terminal_bias("gate")),
            drain_voltage_V=float(self.runtime.get_terminal_bias("drain")),
            source_voltage_V=float(self.runtime.get_terminal_bias("source")),
            trap_density_cm3=float(self.runtime.get_trap_density()),
            last_successful_step=self.last_successful_step,
        )

    def checkpoint(
        self,
        last_successful_step: StepRecord | None = None,
    ) -> SolutionSnapshot:
        metadata = (
            self.last_successful_step
            if last_successful_step is None
            else last_successful_step
        )
        return SolutionSnapshot(
            potential_by_region={
                region: tuple(
                    float(value)
                    for value in self.runtime.get_node_values(region, "Potential")
                )
                for region in self.runtime.regions
            },
            mos2_electrons_cm3=tuple(
                float(value)
                for value in self.runtime.get_node_values(
                    self.runtime.semiconductor_region,
                    "Electrons",
                )
            ),
            gate_voltage_V=float(self.runtime.get_terminal_bias("gate")),
            drain_voltage_V=float(self.runtime.get_terminal_bias("drain")),
            source_voltage_V=float(self.runtime.get_terminal_bias("source")),
            trap_density_cm3=float(self.runtime.get_trap_density()),
            last_successful_step=metadata,
        )

    def restore(self, snapshot: SolutionSnapshot) -> None:
        for region in self.runtime.regions:
            if region not in snapshot.potential_by_region:
                raise ValueError(f"Snapshot has no Potential vector for {region}.")
            self.runtime.set_node_values(
                region,
                "Potential",
                snapshot.potential_by_region[region],
            )
        self.runtime.set_node_values(
            self.runtime.semiconductor_region,
            "Electrons",
            snapshot.mos2_electrons_cm3,
        )
        self.runtime.set_terminal_bias("gate", snapshot.gate_voltage_V)
        self.runtime.set_terminal_bias("drain", snapshot.drain_voltage_V)
        self.runtime.set_terminal_bias("source", snapshot.source_voltage_V)
        self.runtime.set_trap_density(snapshot.trap_density_cm3)
        self.last_successful_step = snapshot.last_successful_step

    def solve_dc(self) -> None:
        self.runtime.solve_dc()

    def set_terminal_bias(self, terminal: str, voltage_V: float) -> None:
        self.runtime.set_terminal_bias(terminal, voltage_V)

    def initialize_staged(self) -> dict[str, Any]:
        """Solve empty-trap Poisson, then activate and solve electron DD."""

        ep.validate_parameters()
        geometry = dict(self.runtime.build_poisson_only())
        self.runtime.solve_dc()
        self.last_successful_step = StepRecord(
            kind="initialization",
            label="empty_trap_zero_bias_poisson",
            value=0.0,
            step_size=0.0,
            attempt_index=1,
        )

        # Reinitialize the electron solution explicitly after the electrostatic
        # equilibrium, then activate current/continuity and solve the coupled DD
        # system without changing bias, charge, material, or solver settings.
        self.runtime.initialize_electrons_from_equilibrium()
        self.runtime.activate_electron_dd()
        self.runtime.solve_dc()
        self.last_successful_step = StepRecord(
            kind="initialization",
            label="empty_trap_zero_bias_electron_dd",
            value=0.0,
            step_size=0.0,
            attempt_index=2,
        )
        self.geometry = geometry
        return dict(geometry)

    def _adaptive_ramp(
        self,
        *,
        kind: str,
        target_value: float,
        initial_step: float,
        minimum_step: float,
        label: str,
    ) -> RampResult:
        target = float(target_value)
        step = abs(float(initial_step))
        minimum = abs(float(minimum_step))
        if not math.isfinite(target):
            raise ValueError("Ramp target must be finite.")
        if not math.isfinite(step) or step <= 0.0:
            raise ValueError("Initial step must be finite and positive.")
        if not math.isfinite(minimum) or minimum <= 0.0:
            raise ValueError("Minimum step must be finite and positive.")
        if minimum > step:
            raise ValueError("Minimum step must not exceed the initial step.")

        state = self.read_state()
        if kind == "gate":
            current = state.gate_voltage_V
        elif kind == "trap":
            current = state.trap_density_cm3
        else:
            raise ValueError(f"Unsupported ramp kind {kind!r}.")

        start = current
        direction = continuation_direction(current, target)
        if direction == 0.0:
            return RampResult(kind, label, start, target, current, ())

        maximum_step = step
        tolerance = 1.0e-12 if kind == "gate" else 1.0
        attempts: list[RampAttempt] = []
        attempt_index = 0

        while direction * (target - current) > tolerance:
            trial_step = min(step, abs(target - current))
            trial = current + direction * trial_step
            attempt_index += 1
            snapshot = self.checkpoint()

            if kind == "gate":
                self.runtime.set_terminal_bias("gate", trial)
            else:
                self.runtime.set_trap_density(trial)

            try:
                self.runtime.solve_dc()
            except Exception as solve_error:
                original_traceback = "".join(
                    traceback_module.format_exception(
                        type(solve_error),
                        solve_error,
                        solve_error.__traceback__,
                    )
                )
                self.restore(snapshot)
                attempts.append(
                    RampAttempt(
                        kind=kind,
                        label=label,
                        attempt_index=attempt_index,
                        start_value=current,
                        trial_value=trial,
                        target_value=target,
                        step_size=trial_step,
                        success=False,
                        rolled_back=True,
                        exception_type=type(solve_error).__name__,
                        exception_message=str(solve_error),
                        exception_traceback=original_traceback,
                    )
                )
                step *= 0.5
                if step < minimum:
                    raise AdaptiveRampError(
                        f"{label}: required {kind} step {step:.6e} is below "
                        f"minimum {minimum:.6e}; the last converged state was "
                        "restored.",
                        attempts=attempts,
                        last_converged_state=self.read_state(),
                    ) from solve_error
                continue

            current = trial
            self.last_successful_step = StepRecord(
                kind=kind,
                label=label,
                value=current,
                step_size=direction * trial_step,
                attempt_index=attempt_index,
            )
            attempts.append(
                RampAttempt(
                    kind=kind,
                    label=label,
                    attempt_index=attempt_index,
                    start_value=snapshot.gate_voltage_V
                    if kind == "gate"
                    else snapshot.trap_density_cm3,
                    trial_value=trial,
                    target_value=target,
                    step_size=trial_step,
                    success=True,
                    rolled_back=False,
                )
            )
            step = min(maximum_step, step * ep.STEP_GROWTH_FACTOR)

        return RampResult(
            kind=kind,
            label=label,
            start_value=start,
            target_value=target,
            final_value=current,
            attempts=tuple(attempts),
        )

    def ramp_trap_density(
        self,
        target_density_cm3: float,
        initial_step_cm3: float = ep.TRAP_INITIAL_STEP_CM3,
        minimum_step_cm3: float = ep.TRAP_MINIMUM_STEP_CM3,
        label: str = "trapped-electron continuation",
    ) -> RampResult:
        if float(target_density_cm3) < 0.0:
            raise ValueError("Trapped-electron density must not be negative.")
        return self._adaptive_ramp(
            kind="trap",
            target_value=target_density_cm3,
            initial_step=initial_step_cm3,
            minimum_step=minimum_step_cm3,
            label=label,
        )

    def ramp_gate(
        self,
        target_voltage_V: float,
        initial_step_V: float = ep.GATE_INITIAL_STEP_V,
        minimum_step_V: float = ep.GATE_MINIMUM_STEP_V,
        label: str = "negative gate continuation",
    ) -> RampResult:
        current = self.read_state().gate_voltage_V
        if float(target_voltage_V) > current + 1.0e-15:
            raise ValueError(
                "ERASE ramp_gate only accepts a target at or below the "
                "current gate voltage."
            )
        return self._adaptive_ramp(
            kind="gate",
            target_value=target_voltage_V,
            initial_step=initial_step_V,
            minimum_step=minimum_step_V,
            label=label,
        )
