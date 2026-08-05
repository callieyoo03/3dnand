"""Deterministic metadata and SHA-256 helpers for the compact-model handoff.

The module is deliberately DEVSIM-free at import time.  Runtime verification
uses a lazily supplied/imported DEVSIM API, while ordinary schema, CSV, and
manifest tests can run with the standard library alone.

The handoff manifest intentionally does not list itself.  A file cannot embed
its own stable cryptographic digest; every other delivered artifact is hashed
and the manifest is validated independently after it is written.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.util
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
CANONICAL_PARAMETER_PATH = REPOSITORY_ROOT / "compact_handoff_parameters.py"
PROGRAM_CASE_PATH = REPOSITORY_ROOT / "05_program_erase" / "run_final_program_case.py"
TUNNELING_PARAMETER_PATH = REPOSITORY_ROOT / "05_program_erase" / "tunneling_parameters.py"


DEVICE_STRUCTURE_FIELDNAMES = (
    "parameter",
    "value",
    "unit",
    "description",
    "source_file",
    "runtime_verified",
    "derived",
)

MATERIAL_FIELDNAMES = (
    "region",
    "parameter",
    "value",
    "unit",
    "source_file",
    "provenance",
    "status",
    "notes",
    "runtime_verified",
)

SIMULATION_CONDITION_FIELDNAMES = (
    "analysis",
    "parameter",
    "value",
    "unit",
    "notes",
)

REGION_CONTACT_FIELDNAMES = (
    "entity_type",
    "name",
    "material_or_region",
    "adjacent_region",
    "location",
    "boundary_condition",
    "notes",
)

MANIFEST_FIELDNAMES = (
    "file",
    "category",
    "description",
    "generated_by",
    "geometry_version",
    "material_version",
    "sha256",
)

MANIFEST_SELF_HASH_POLICY = (
    "handoff_manifest.csv is intentionally excluded because a file cannot "
    "contain its own stable SHA-256 digest"
)

EXPECTED_REGIONS = {
    "CoreOxide": "Oxide",
    "MoS2": "MoS2",
    "TunnelOxide": "Oxide",
    "ChargeTrap": "ChargeTrap",
    "BlockingOxide": "Oxide",
    "GateMetal": "Metal",
    "Air": "Air",
}

EXPECTED_INTERFACES = {
    "CoreOxide_MoS2",
    "MoS2_TunnelOxide",
    "TunnelOxide_ChargeTrap",
    "ChargeTrap_BlockingOxide",
}

EXPECTED_CONTACTS = {"source", "drain", "gate"}
PHYSICS_REGIONS = (
    "CoreOxide",
    "MoS2",
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
)

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MODULE_CACHE: dict[Path, ModuleType] = {}


def _load_module_from_path(path: Path, module_name: str) -> ModuleType:
    """Load one repository module under a collision-free private name."""

    resolved_path = path.resolve()
    cached = _MODULE_CACHE.get(resolved_path)
    if cached is not None:
        return cached
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Required module was not found: {resolved_path}")

    specification = importlib.util.spec_from_file_location(module_name, resolved_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not construct an import specification for {resolved_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    _MODULE_CACHE[resolved_path] = module
    return module


def canonical_parameters() -> ModuleType:
    """Return the repository-level compact-handoff source of truth."""

    module = _load_module_from_path(
        CANONICAL_PARAMETER_PATH,
        "compact_handoff_parameters_runtime",
    )
    required_names = (
        "GEOMETRY_VERSION",
        "MATERIAL_VERSION",
        "DEVICE_TYPE",
        "COORDINATE_SYSTEM",
        "CORE_RADIUS_NM",
        "MOS2_THICKNESS_NM",
        "TUNNEL_OXIDE_THICKNESS_NM",
        "CHARGE_TRAP_THICKNESS_NM",
        "BLOCKING_OXIDE_THICKNESS_NM",
        "GATE_METAL_THICKNESS_NM",
        "AIR_THICKNESS_NM",
        "CHANNEL_LENGTH_NM",
        "AL2O3_RELATIVE_PERMITTIVITY",
        "HFO2_RELATIVE_PERMITTIVITY",
        "MOS2_RELATIVE_PERMITTIVITY",
        "CORE_OXIDE_RELATIVE_PERMITTIVITY",
    )
    missing = [name for name in required_names if not hasattr(module, name)]
    if missing:
        raise AttributeError(
            "compact_handoff_parameters.py is missing required names: "
            + ", ".join(missing)
        )
    return module


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be numeric, not boolean.")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric.") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    return number


def expected_geometry() -> dict[str, float]:
    """Return the nominal 3/5/16 nm geometry, including derived radii."""

    parameters = canonical_parameters()
    core = _finite_float("CORE_RADIUS_NM", parameters.CORE_RADIUS_NM)
    mos2 = _finite_float("MOS2_THICKNESS_NM", parameters.MOS2_THICKNESS_NM)
    tunnel = _finite_float(
        "TUNNEL_OXIDE_THICKNESS_NM", parameters.TUNNEL_OXIDE_THICKNESS_NM
    )
    trap = _finite_float(
        "CHARGE_TRAP_THICKNESS_NM", parameters.CHARGE_TRAP_THICKNESS_NM
    )
    blocking = _finite_float(
        "BLOCKING_OXIDE_THICKNESS_NM", parameters.BLOCKING_OXIDE_THICKNESS_NM
    )
    gate = _finite_float(
        "GATE_METAL_THICKNESS_NM", parameters.GATE_METAL_THICKNESS_NM
    )
    air = _finite_float("AIR_THICKNESS_NM", parameters.AIR_THICKNESS_NM)
    length = _finite_float("CHANNEL_LENGTH_NM", parameters.CHANNEL_LENGTH_NM)

    values = (core, mos2, tunnel, trap, blocking, gate, air, length)
    if any(value <= 0.0 for value in values):
        raise ValueError("All canonical geometry dimensions must be positive.")

    mos2_outer = core + mos2
    tunnel_outer = mos2_outer + tunnel
    trap_outer = tunnel_outer + trap
    blocking_outer = trap_outer + blocking
    gate_outer = blocking_outer + gate
    air_outer = gate_outer + air
    nm_to_cm = 1.0e-7

    return {
        "core_radius_nm": core,
        "core_outer_radius_nm": core,
        "mos2_thickness_nm": mos2,
        "tunnel_oxide_thickness_nm": tunnel,
        "charge_trap_thickness_nm": trap,
        "blocking_oxide_thickness_nm": blocking,
        "gate_metal_thickness_nm": gate,
        "air_thickness_nm": air,
        "channel_length_nm": length,
        "mos2_outer_radius_nm": mos2_outer,
        "tunnel_oxide_outer_radius_nm": tunnel_outer,
        "charge_trap_outer_radius_nm": trap_outer,
        "blocking_oxide_outer_radius_nm": blocking_outer,
        "gate_outer_radius_nm": gate_outer,
        "air_outer_radius_nm": air_outer,
        "r_axis": 0.0,
        "r_core": core * nm_to_cm,
        "r_mos2": mos2_outer * nm_to_cm,
        "r_tox": tunnel_outer * nm_to_cm,
        "r_trap": trap_outer * nm_to_cm,
        "r_block": blocking_outer * nm_to_cm,
        "r_gate_outer": gate_outer * nm_to_cm,
        "r_air_outer": air_outer * nm_to_cm,
        "z_source": 0.0,
        "z_drain": length * nm_to_cm,
    }


def validate_geometry_mapping(geometry: Mapping[str, Any]) -> dict[str, float]:
    """Validate a runtime geometry dictionary against the canonical geometry."""

    expected = expected_geometry()
    checked: dict[str, float] = {}
    for name, expected_value in expected.items():
        if name not in geometry:
            raise ValueError(f"Runtime geometry is missing {name!r}.")
        actual_value = _finite_float(f"runtime geometry {name}", geometry[name])
        tolerance = 1.0e-12 if name.startswith("r_") or name.startswith("z_") else 1.0e-9
        if not math.isclose(
            actual_value,
            expected_value,
            rel_tol=1.0e-12,
            abs_tol=tolerance,
        ):
            raise ValueError(
                f"Runtime geometry mismatch for {name}: "
                f"actual={actual_value:.12e}, expected={expected_value:.12e}."
            )
        checked[name] = actual_value
    return checked


def _boolean_text(value: bool) -> str:
    return "true" if value else "false"


def build_device_structure_rows(
    geometry: Mapping[str, Any] | None = None,
    *,
    runtime_verified: bool = False,
) -> list[dict[str, Any]]:
    """Build the required device-structure metadata rows."""

    parameters = canonical_parameters()
    values = validate_geometry_mapping(geometry) if geometry is not None else expected_geometry()
    verified = _boolean_text(runtime_verified)
    source = "compact_handoff_parameters.py"

    definitions = (
        ("device_type", parameters.DEVICE_TYPE, "", "Cylindrical GAA charge-trap memory device", False),
        ("coordinate_system", parameters.COORDINATE_SYSTEM, "", "DEVSIM x=r, y=z two-dimensional axisymmetric coordinates", False),
        ("gate_length_nm", values["channel_length_nm"], "nm", "Gate length along the active channel", False),
        ("core_radius_nm", values["core_radius_nm"], "nm", "Inner core-oxide radius", False),
        ("channel_inner_radius_nm", values["core_radius_nm"], "nm", "MoS2 inner radius", True),
        ("channel_outer_radius_nm", values["mos2_outer_radius_nm"], "nm", "MoS2 outer radius", True),
        ("channel_thickness_nm", values["mos2_thickness_nm"], "nm", "MoS2 channel thickness", False),
        ("tunnel_oxide_thickness_nm", values["tunnel_oxide_thickness_nm"], "nm", "Al2O3 tunnel dielectric thickness", False),
        ("charge_trap_thickness_nm", values["charge_trap_thickness_nm"], "nm", "HfO2 charge-trap-layer thickness", False),
        ("blocking_oxide_thickness_nm", values["blocking_oxide_thickness_nm"], "nm", "Al2O3 blocking dielectric thickness", False),
        ("gate_inner_radius_nm", values["blocking_oxide_outer_radius_nm"], "nm", "Gate-contact inner radius at the blocking-oxide boundary", True),
        ("active_axial_length_nm", values["channel_length_nm"], "nm", "Active source-to-drain axial length", False),
    )

    return [
        {
            "parameter": name,
            "value": value,
            "unit": unit,
            "description": description,
            "source_file": source,
            "runtime_verified": verified,
            "derived": _boolean_text(derived),
        }
        for name, value, unit, description, derived in definitions
    ]


def _import_local_material_parameters() -> ModuleType:
    return _load_module_from_path(
        MODULE_DIRECTORY / "material_parameters.py",
        "charge_trap_material_parameters_metadata",
    )


def _import_tunneling_parameters() -> ModuleType:
    return _load_module_from_path(
        TUNNELING_PARAMETER_PATH,
        "program_tunneling_parameters_metadata",
    )


def build_material_rows(*, runtime_verified: bool = False) -> list[dict[str, Any]]:
    """Build material/transport metadata without executing a program runner."""

    canonical = canonical_parameters()
    material = _import_local_material_parameters()
    tunneling = _import_tunneling_parameters()
    # Only quantities that validate_runtime_metadata reads from the solved
    # DEVSIM device may be labeled runtime-verified.  Repository declarations,
    # unused placeholders, and program/FN references remain false even when a
    # runtime validation report exists.
    runtime_verifiable_parameters = {
        "relative_permittivity",
        "electron_mobility",
        "temperature",
        "channel_doping",
    }

    definitions = (
        ("TunnelOxide;BlockingOxide", "relative_permittivity", canonical.AL2O3_RELATIVE_PERMITTIVITY, "1", "compact_handoff_parameters.py", "literature nominal Al2O3 dielectric constant", "literature_nominal", "Used by both Al2O3 dielectric regions"),
        ("ChargeTrap", "relative_permittivity", canonical.HFO2_RELATIVE_PERMITTIVITY, "1", "compact_handoff_parameters.py", "literature nominal HfO2 dielectric constant", "literature_nominal", "Charge-trap dielectric"),
        ("MoS2", "relative_permittivity", canonical.MOS2_RELATIVE_PERMITTIVITY, "1", "compact_handoff_parameters.py", "repository out-of-plane MoS2 value", "repo_default", "Not process-calibrated"),
        ("CoreOxide", "relative_permittivity", canonical.CORE_OXIDE_RELATIVE_PERMITTIVITY, "1", "compact_handoff_parameters.py", "SiO2-like core approximation", "repo_default", "Core material is an approximation"),
        ("MoS2", "bandgap", material.Eg_MoS2, "eV", "04_charge_trap/material_parameters.py", "repository transport baseline", "repo_default", "Used indirectly in intrinsic-density calculation"),
        ("MoS2", "electron_affinity", material.ElectronAffinity_MoS2, "eV", "04_charge_trap/material_parameters.py", "repository transport baseline", "repo_default", "Recorded for handoff; no explicit heterojunction model"),
        ("MoS2", "electron_mobility", material.mu_n_MoS2, "cm^2/(V*s)", "04_charge_trap/material_parameters.py", "constant mobility assumption", "uncalibrated", "Used by the electron continuity model"),
        ("MoS2", "hole_mobility", material.mu_p_MoS2, "cm^2/(V*s)", "04_charge_trap/material_parameters.py", "future ambipolar placeholder", "placeholder", "No hole continuity equation is solved"),
        ("Gate", "work_function", material.gate_work_function, "eV", "04_charge_trap/material_parameters.py", "repository placeholder", "placeholder", "Current gate boundary uses zero work-function offset"),
        ("Device", "temperature", material.temperature, "K", "04_charge_trap/material_parameters.py", "repository simulation condition", "repo_default", "Isothermal DC simulation"),
        ("MoS2", "channel_doping", material.channel_doping, "cm^-3", "04_charge_trap/material_parameters.py", "uniform channel model", "uncalibrated", "Spatially applied as uniform Donors"),
        ("Source;Drain", "source_drain_doping", material.source_drain_doping, "cm^-3", "04_charge_trap/material_parameters.py", "declared repository value", "placeholder", "Declared but not spatially implemented in the present device"),
        ("TunnelOxide", "FN_barrier_height", tunneling.BARRIER_HEIGHT_EV, "eV", "05_program_erase/tunneling_parameters.py", "CBO-based MoS2/Al2O3 candidate", "literature_nominal", "Uncalibrated; not changed or refit in this handoff"),
        ("TunnelOxide", "tunneling_effective_mass_ratio", tunneling.TUNNEL_EFFECTIVE_MASS_RATIO, "m0", "05_program_erase/tunneling_parameters.py", "literature nominal tunneling mass", "literature_nominal", "Uncalibrated; not changed or refit in this handoff"),
    )

    return [
        {
            "region": region,
            "parameter": parameter,
            "value": value,
            "unit": unit,
            "source_file": source_file,
            "provenance": provenance,
            "status": status,
            "notes": notes,
            "runtime_verified": _boolean_text(
                runtime_verified and parameter in runtime_verifiable_parameters
            ),
        }
        for (
            region,
            parameter,
            value,
            unit,
            source_file,
            provenance,
            status,
            notes,
        ) in definitions
    ]


def _read_top_level_literal_constants(path: Path, names: Sequence[str]) -> dict[str, Any]:
    """Read simple literal runner constants without importing/executing it."""

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    requested = set(names)
    values: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name) or target.id not in requested:
            continue
        try:
            values[target.id] = ast.literal_eval(statement.value)
        except (ValueError, TypeError):
            continue
    missing = requested.difference(values)
    if missing:
        raise ValueError(
            f"Could not read required literal constants from {path}: "
            + ", ".join(sorted(missing))
        )
    return values


def _compact_sequence(values: Iterable[Any]) -> str:
    return ";".join(f"{float(value):.12g}" for value in values)


def build_simulation_condition_rows() -> list[dict[str, Any]]:
    """Build sweep, extraction, solver, program, and Q/C conditions."""

    config = _load_module_from_path(
        MODULE_DIRECTORY / "state_characterization_config.py",
        "state_characterization_config_metadata",
    )
    trap = _load_module_from_path(
        MODULE_DIRECTORY / "trap_parameters.py",
        "charge_trap_parameters_metadata",
    )
    program = _read_top_level_literal_constants(
        PROGRAM_CASE_PATH,
        (
            "PROGRAM_GATE_VOLTAGE_V",
            "DRAIN_VOLTAGE_V",
            "TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2",
        ),
    )

    rows: list[dict[str, Any]] = []

    def add(analysis: str, parameter: str, value: Any, unit: str, notes: str) -> None:
        rows.append(
            {
                "analysis": analysis,
                "parameter": parameter,
                "value": value,
                "unit": unit,
                "notes": notes,
            }
        )

    add("ID-VG", "VDS_values", _compact_sequence(config.IDVG_VDS_VALUES_V), "V", "Fixed drain biases")
    add("ID-VG", "VGS_start", config.IDVG_VGS_START_V, "V", "Inclusive sweep start")
    add("ID-VG", "VGS_stop", config.IDVG_VGS_STOP_V, "V", "Inclusive sweep stop")
    add("ID-VG", "VGS_step", config.IDVG_VGS_STEP_V, "V", "Uniform output step")
    add("ID-VD", "VGS_values", _compact_sequence(config.IDVD_VGS_VALUES_V), "V", "Fixed gate biases")
    add("ID-VD", "VDS_start", config.IDVD_VDS_START_V, "V", "Inclusive sweep start")
    add("ID-VD", "VDS_stop", config.IDVD_VDS_STOP_V, "V", "Inclusive sweep stop")
    add("ID-VD", "VDS_step", config.IDVD_VDS_STEP_V, "V", "Uniform output step")

    for state in config.MEMORY_STATES:
        add(
            "memory_state",
            str(state["state"]),
            state["ntrap_cm3"],
            "cm^-3",
            f"state_index={state['state_index']}; static fixed trapped-electron density",
        )

    add("metric", "Vth_constant_current", config.VTH_TARGET_CURRENT_A, "A", "Log-current interpolation")
    add("metric", "SS_current_min", config.SS_CURRENT_MIN_A, "A", "Lower regression-window bound")
    add("metric", "SS_current_max", config.SS_CURRENT_MAX_A, "A", "Upper regression-window bound")
    add("metric", "SS_minimum_points", config.SS_MINIMUM_POINT_COUNT, "count", "Minimum points for log-current regression")
    add("metric", "Ion_bias", f"VGS={config.ION_VGS_V:.12g};VDS={config.ION_VDS_V:.12g}", "V", "Ion is the absolute drain-current magnitude")
    add("metric", "Ioff_bias", f"VGS={config.IOFF_VGS_V:.12g};VDS={config.IOFF_VDS_V:.12g}", "V", "Ioff is the absolute drain-current magnitude")
    add("metric", "gm_definition", "max(abs(dID/dVGS))", "S", "Three-point derivative on the sampled ID-VG curve")
    add("metric", "current_floor", config.CURRENT_FLOOR_A, "A", "Protects logarithms and floor-limited ON/OFF reporting")

    add("solver", "solver_type", "direct", "", "DEVSIM DC Newton solve")
    add("solver", "absolute_error", trap.SOLVER_ABSOLUTE_ERROR, "", "DEVSIM absolute convergence criterion")
    add("solver", "relative_error", trap.SOLVER_RELATIVE_ERROR, "", "DEVSIM relative convergence criterion")
    add("solver", "maximum_iterations", trap.SOLVER_MAXIMUM_ITERATIONS, "count", "Default DC limit")
    add("solver", "gate_initial_ramp_step", trap.GATE_INITIAL_STEP, "V", "Adaptive continuation")
    add("solver", "gate_minimum_ramp_step", trap.GATE_MINIMUM_STEP, "V", "Adaptive continuation")
    add("solver", "drain_initial_ramp_step", trap.DRAIN_INITIAL_STEP, "V", "Adaptive continuation")
    add("solver", "drain_minimum_ramp_step", trap.DRAIN_MINIMUM_STEP, "V", "Adaptive continuation")

    add("program_reference", "gate_voltage", program["PROGRAM_GATE_VOLTAGE_V"], "V", "Existing nominal program condition; no FN recalibration performed")
    add("program_reference", "drain_voltage", program["DRAIN_VOLTAGE_V"], "V", "Existing nominal program condition")
    add("program_reference", "target_nsheet", program["TARGET_TRAPPED_ELECTRON_SHEET_DENSITY_CM2"], "cm^-2", "Equivalent to 2.0e18 cm^-3 for a 5 nm CTL")
    add("program_reference", "target_nvolume", config.MEMORY_STATES[-1]["ntrap_cm3"], "cm^-3", "Static programmed read state")

    add("units", "mesh_coordinate", "centimeter", "cm", "DEVSIM x=r and y=z coordinates")
    add("units", "drain_current", "cylindrical_contact_integral", "A", "get_contact_current with cylindrical edge coupling; not per-unit-depth")
    add("units", "local_electron_current", "current_density", "A/cm^2", "VTK element-vector components")
    add("geometry", "axisymmetric_weighting", "CylindricalNodeVolume;CylindricalEdgeCouple", "", "2*pi*r rotational weighting; no arbitrary planar area")
    add("capacitance", "definition", "Cij=dQi/dVj", "F", "Direct derivative sign; measured charge index first")
    add("capacitance", "nominal_delta_voltage", 1.0e-3, "V", "Central finite difference with fixed trap density")
    add("capacitance", "sensitivity_delta_voltage", "0.0005;0.001;0.002", "V", "Central-difference sensitivity set")

    return rows


def build_region_contact_rows(geometry: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """Build a complete map of all runtime regions, interfaces, and contacts."""

    values = validate_geometry_mapping(geometry) if geometry is not None else expected_geometry()
    length = values["channel_length_nm"]

    rows: list[dict[str, Any]] = []

    def add(
        entity_type: str,
        name: str,
        material_or_region: str,
        adjacent_region: str,
        location: str,
        boundary_condition: str,
        notes: str,
    ) -> None:
        rows.append(
            {
                "entity_type": entity_type,
                "name": name,
                "material_or_region": material_or_region,
                "adjacent_region": adjacent_region,
                "location": location,
                "boundary_condition": boundary_condition,
                "notes": notes,
            }
        )

    region_definitions = (
        ("CoreOxide", "Oxide", "MoS2", 0.0, values["core_radius_nm"], "Poisson dielectric"),
        ("MoS2", "MoS2", "CoreOxide;TunnelOxide", values["core_radius_nm"], values["mos2_outer_radius_nm"], "Poisson plus electron continuity"),
        ("TunnelOxide", "Oxide", "MoS2;ChargeTrap", values["mos2_outer_radius_nm"], values["tunnel_oxide_outer_radius_nm"], "Poisson dielectric"),
        ("ChargeTrap", "ChargeTrap", "TunnelOxide;BlockingOxide", values["tunnel_oxide_outer_radius_nm"], values["charge_trap_outer_radius_nm"], "Poisson dielectric with prescribed trapped charge"),
        ("BlockingOxide", "Oxide", "ChargeTrap;GateMetal", values["charge_trap_outer_radius_nm"], values["blocking_oxide_outer_radius_nm"], "Poisson dielectric; gate contact at outer boundary"),
        ("GateMetal", "Metal", "BlockingOxide;Air", values["blocking_oxide_outer_radius_nm"], values["gate_outer_radius_nm"], "Mesh-only region; no bulk equation"),
        ("Air", "Air", "GateMetal", values["gate_outer_radius_nm"], values["air_outer_radius_nm"], "Mesh-only region; no bulk equation"),
    )
    for name, material, adjacent, r_min, r_max, notes in region_definitions:
        add(
            "region",
            name,
            material,
            adjacent,
            f"r={r_min:.12g}..{r_max:.12g} nm; z=0..{length:.12g} nm",
            "bulk model" if name in PHYSICS_REGIONS else "none",
            notes,
        )

    interface_definitions = (
        ("CoreOxide_MoS2", "CoreOxide", "MoS2", values["core_radius_nm"]),
        ("MoS2_TunnelOxide", "MoS2", "TunnelOxide", values["mos2_outer_radius_nm"]),
        ("TunnelOxide_ChargeTrap", "TunnelOxide", "ChargeTrap", values["tunnel_oxide_outer_radius_nm"]),
        ("ChargeTrap_BlockingOxide", "ChargeTrap", "BlockingOxide", values["charge_trap_outer_radius_nm"]),
    )
    for name, region0, region1, radius in interface_definitions:
        add(
            "interface",
            name,
            region0,
            region1,
            f"r={radius:.12g} nm; z=0..{length:.12g} nm",
            "Potential continuity and flux conservation",
            "Electrostatic interface",
        )

    add("contact", "source", "Metal", "MoS2", f"z=0 nm; r={values['core_radius_nm']:.12g}..{values['mos2_outer_radius_nm']:.12g} nm", "Ideal electron-only ohmic contact; source_bias", "Potential and electron continuity contact equations")
    add("contact", "drain", "Metal", "MoS2", f"z={length:.12g} nm; r={values['core_radius_nm']:.12g}..{values['mos2_outer_radius_nm']:.12g} nm", "Ideal electron-only ohmic contact; drain_bias", "Potential and electron continuity contact equations")
    add("contact", "gate", "Metal", "BlockingOxide", f"r={values['blocking_oxide_outer_radius_nm']:.12g} nm; z=0..{length:.12g} nm", "Ideal electrostatic gate; gate_bias", "Contact lies on BlockingOxide; GateMetal bulk is mesh-only")
    return rows


def _runtime_get_parameter(runtime_api: Any, *, device: str, name: str, region: str | None = None) -> Any:
    arguments = {"device": device, "name": name}
    if region is not None:
        arguments["region"] = region
    return runtime_api.get_parameter(**arguments)


def validate_runtime_metadata(
    geometry: Mapping[str, Any],
    *,
    device: str = "MoS2_GAA",
    runtime_api: Any | None = None,
) -> dict[str, Any]:
    """Validate actual DEVSIM geometry, materials, topology, and model presence."""

    if runtime_api is None:
        import devsim as runtime_api  # type: ignore[no-redef]

    checked_geometry = validate_geometry_mapping(geometry)
    canonical = canonical_parameters()
    material = _import_local_material_parameters()

    regions = set(runtime_api.get_region_list(device=device))
    contacts = set(runtime_api.get_contact_list(device=device))
    interfaces = set(runtime_api.get_interface_list(device=device))
    if regions != set(EXPECTED_REGIONS):
        raise RuntimeError(f"Runtime region set mismatch: {sorted(regions)}")
    if contacts != EXPECTED_CONTACTS:
        raise RuntimeError(f"Runtime contact set mismatch: {sorted(contacts)}")
    if interfaces != EXPECTED_INTERFACES:
        raise RuntimeError(f"Runtime interface set mismatch: {sorted(interfaces)}")

    for region, expected_material in EXPECTED_REGIONS.items():
        actual_material = runtime_api.get_material(device=device, region=region)
        if actual_material != expected_material:
            raise RuntimeError(
                f"Runtime material mismatch for {region}: "
                f"actual={actual_material!r}, expected={expected_material!r}."
            )

    device_parameter_names = (
        "core_radius_nm",
        "mos2_thickness_nm",
        "tunnel_oxide_thickness_nm",
        "charge_trap_thickness_nm",
        "blocking_oxide_thickness_nm",
        "channel_length_nm",
    )
    for name in device_parameter_names:
        actual = _finite_float(
            f"device parameter {name}",
            _runtime_get_parameter(runtime_api, device=device, name=name),
        )
        expected = checked_geometry[name]
        if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-9):
            raise RuntimeError(
                f"Runtime device parameter mismatch for {name}: "
                f"actual={actual:.12e}, expected={expected:.12e}."
            )

    expected_relative_permittivities = {
        "CoreOxide": float(canonical.CORE_OXIDE_RELATIVE_PERMITTIVITY),
        "MoS2": float(canonical.MOS2_RELATIVE_PERMITTIVITY),
        "TunnelOxide": float(canonical.AL2O3_RELATIVE_PERMITTIVITY),
        "ChargeTrap": float(canonical.HFO2_RELATIVE_PERMITTIVITY),
        "BlockingOxide": float(canonical.AL2O3_RELATIVE_PERMITTIVITY),
    }
    actual_relative_permittivities: dict[str, float] = {}
    for region, expected_relative in expected_relative_permittivities.items():
        absolute = _finite_float(
            f"{region} Permittivity",
            _runtime_get_parameter(
                runtime_api,
                device=device,
                region=region,
                name="Permittivity",
            ),
        )
        relative = absolute / float(material.eps0)
        if not math.isclose(relative, expected_relative, rel_tol=1.0e-12, abs_tol=1.0e-12):
            raise RuntimeError(
                f"Runtime relative permittivity mismatch for {region}: "
                f"actual={relative:.12e}, expected={expected_relative:.12e}."
            )
        actual_relative_permittivities[region] = relative

    actual_electron_mobility = _finite_float(
        "MoS2 ElectronMobility",
        _runtime_get_parameter(
            runtime_api,
            device=device,
            region="MoS2",
            name="ElectronMobility",
        ),
    )
    if not math.isclose(
        actual_electron_mobility,
        float(material.mu_n_MoS2),
        rel_tol=1.0e-12,
        abs_tol=1.0e-12,
    ):
        raise RuntimeError(
            "Runtime ElectronMobility mismatch: "
            f"actual={actual_electron_mobility:.12e}, "
            f"expected={float(material.mu_n_MoS2):.12e}."
        )

    actual_thermal_voltage = _finite_float(
        "MoS2 ThermalVoltage",
        _runtime_get_parameter(
            runtime_api,
            device=device,
            region="MoS2",
            name="ThermalVoltage",
        ),
    )
    if not math.isclose(
        actual_thermal_voltage,
        float(material.Vt),
        rel_tol=1.0e-12,
        abs_tol=1.0e-15,
    ):
        raise RuntimeError(
            "Runtime ThermalVoltage mismatch: "
            f"actual={actual_thermal_voltage:.12e}, "
            f"expected={float(material.Vt):.12e}."
        )

    donor_values = tuple(
        float(value)
        for value in runtime_api.get_node_model_values(
            device=device,
            region="MoS2",
            name="Donors",
        )
    )
    if not donor_values or any(
        not math.isfinite(value)
        or not math.isclose(
            value,
            float(material.channel_doping),
            rel_tol=1.0e-12,
            abs_tol=1.0,
        )
        for value in donor_values
    ):
        raise RuntimeError("Runtime Donors model does not match channel_doping.")

    expected_weighting = {
        "raxis_variable": "x",
        "node_volume_model": "CylindricalNodeVolume",
        "edge_couple_model": "CylindricalEdgeCouple",
        "edge_node0_volume_model": "CylindricalEdgeNodeVolume@n0",
        "edge_node1_volume_model": "CylindricalEdgeNodeVolume@n1",
    }
    for name, expected in expected_weighting.items():
        actual = _runtime_get_parameter(runtime_api, device=device, name=name)
        if str(actual) != expected:
            raise RuntimeError(
                f"Runtime cylindrical-weighting mismatch for {name}: "
                f"actual={actual!r}, expected={expected!r}."
            )

    for region in PHYSICS_REGIONS:
        node_models = set(runtime_api.get_node_model_list(device=device, region=region))
        edge_models = set(runtime_api.get_edge_model_list(device=device, region=region))
        if "Potential" not in node_models or "ElectricField" not in edge_models:
            raise RuntimeError(f"Required electrostatic models are missing in {region}.")
    mos2_nodes = set(runtime_api.get_node_model_list(device=device, region="MoS2"))
    mos2_edges = set(runtime_api.get_edge_model_list(device=device, region="MoS2"))
    if not {"Electrons", "NetDoping"}.issubset(mos2_nodes) or "ElectronCurrent" not in mos2_edges:
        raise RuntimeError("Required MoS2 transport models are missing.")
    trap_nodes = set(runtime_api.get_node_model_list(device=device, region="ChargeTrap"))
    if not {"TrappedElectronDensity", "TrappedChargeDensity"}.issubset(trap_nodes):
        raise RuntimeError("Required charge-trap models are missing.")

    for contact in EXPECTED_CONTACTS:
        equations = set(runtime_api.get_contact_equation_list(device=device, contact=contact))
        if "PotentialEquation" not in equations:
            raise RuntimeError(f"PotentialEquation is missing at contact {contact}.")
    for contact in ("source", "drain"):
        equations = set(runtime_api.get_contact_equation_list(device=device, contact=contact))
        if "ElectronContinuityEquation" not in equations:
            raise RuntimeError(f"ElectronContinuityEquation is missing at contact {contact}.")

    return {
        "runtime_verified": True,
        "device": device,
        "geometry": checked_geometry,
        "regions": sorted(regions),
        "interfaces": sorted(interfaces),
        "contacts": sorted(contacts),
        "relative_permittivities": actual_relative_permittivities,
        "electron_mobility_cm2_Vs": actual_electron_mobility,
        "thermal_voltage_V": actual_thermal_voltage,
        "channel_doping_cm3": float(material.channel_doping),
        "axisymmetric_weighting": dict(expected_weighting),
    }


def _validate_rows(fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected_keys = set(fieldnames)
    validated: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        actual_keys = set(row)
        if actual_keys != expected_keys:
            raise ValueError(
                f"Row {index} does not match schema; "
                f"missing={sorted(expected_keys - actual_keys)}, "
                f"extra={sorted(actual_keys - expected_keys)}."
            )
        validated.append({name: row[name] for name in fieldnames})
    return validated


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, Any]],
) -> Path:
    """Write a strict, deterministic UTF-8 CSV with LF line endings."""

    destination = Path(path)
    validated = _validate_rows(fieldnames, rows)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=tuple(fieldnames),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(validated)
    return destination


def write_metadata_csvs(
    output_directory: str | Path = SHARED_DATA_DIRECTORY,
    *,
    geometry: Mapping[str, Any] | None = None,
    runtime_verified: bool = False,
) -> dict[str, Path]:
    """Write the four required metadata CSVs and return their paths."""

    directory = Path(output_directory)
    outputs = {
        "device_structure": directory / "device_structure_parameters.csv",
        "material": directory / "material_parameters.csv",
        "simulation_conditions": directory / "simulation_conditions.csv",
        "region_contact_map": directory / "region_contact_map.csv",
    }
    write_csv_rows(
        outputs["device_structure"],
        DEVICE_STRUCTURE_FIELDNAMES,
        build_device_structure_rows(geometry, runtime_verified=runtime_verified),
    )
    write_csv_rows(
        outputs["material"],
        MATERIAL_FIELDNAMES,
        build_material_rows(runtime_verified=runtime_verified),
    )
    write_csv_rows(
        outputs["simulation_conditions"],
        SIMULATION_CONDITION_FIELDNAMES,
        build_simulation_condition_rows(),
    )
    write_csv_rows(
        outputs["region_contact_map"],
        REGION_CONTACT_FIELDNAMES,
        build_region_contact_rows(geometry),
    )
    return outputs


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_artifact_path(root: Path, artifact: str | Path) -> tuple[Path, str]:
    candidate = Path(artifact)
    absolute = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Artifact is outside manifest root: {absolute}") from error
    if not absolute.is_file():
        raise FileNotFoundError(f"Manifest artifact is missing or not a file: {absolute}")
    return absolute, relative.as_posix()


def _infer_artifact_category(relative_path: str) -> str:
    path = Path(relative_path)
    suffix = path.suffix.lower()
    if "mesh" in path.parts:
        if suffix == ".msh":
            return "mesh"
        if suffix in {".vtu", ".vtm", ".visit"}:
            return "visualization"
        if suffix == ".md":
            return "documentation"
    if suffix == ".csv":
        if path.name in {
            "idvg_by_state.csv",
            "idvd_by_state.csv",
            "metrics_by_state.csv",
            "memory_state_map.csv",
            "baseline_comparison_legacy_vs_final.csv",
        }:
            return "electrical_data"
        if "capacitance" in path.name or "terminal_charge" in path.name:
            return "charge_capacitance_data"
        return "metadata"
    if suffix == ".md":
        return "documentation"
    return "handoff_artifact"


def build_manifest_rows(
    root_directory: str | Path,
    artifacts: Iterable[str | Path | Mapping[str, Any]],
    *,
    generated_by: str = "04_charge_trap/run_final_handoff.py",
) -> list[dict[str, Any]]:
    """Hash a deterministic artifact list, excluding the manifest itself."""

    root = Path(root_directory).resolve()
    canonical = canonical_parameters()
    rows_by_file: dict[str, dict[str, Any]] = {}

    for artifact in artifacts:
        if isinstance(artifact, Mapping):
            if "file" not in artifact:
                raise KeyError("Manifest artifact mappings require a 'file' key.")
            requested_path = artifact["file"]
            category_override = artifact.get("category")
            description_override = artifact.get("description")
            generator_override = artifact.get("generated_by")
        else:
            requested_path = artifact
            category_override = None
            description_override = None
            generator_override = None

        absolute, relative = _relative_artifact_path(root, requested_path)
        if Path(relative).name == "handoff_manifest.csv":
            continue
        if relative in rows_by_file:
            raise ValueError(f"Duplicate manifest artifact: {relative}")

        rows_by_file[relative] = {
            "file": relative,
            "category": category_override or _infer_artifact_category(relative),
            "description": description_override or f"Compact-model handoff artifact: {Path(relative).name}",
            "generated_by": generator_override or generated_by,
            "geometry_version": canonical.GEOMETRY_VERSION,
            "material_version": canonical.MATERIAL_VERSION,
            "sha256": sha256_file(absolute),
        }

    return [rows_by_file[name] for name in sorted(rows_by_file)]


def write_manifest(
    manifest_path: str | Path,
    root_directory: str | Path,
    artifacts: Iterable[str | Path | Mapping[str, Any]],
    *,
    generated_by: str = "04_charge_trap/run_final_handoff.py",
) -> Path:
    rows = build_manifest_rows(
        root_directory,
        artifacts,
        generated_by=generated_by,
    )
    return write_csv_rows(manifest_path, MANIFEST_FIELDNAMES, rows)


def validate_manifest(
    manifest_path: str | Path,
    *,
    root_directory: str | Path | None = None,
    expected_files: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Recompute every listed digest and validate path/schema invariants."""

    manifest = Path(manifest_path).resolve()
    root = Path(root_directory).resolve() if root_directory is not None else manifest.parent
    with manifest.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_FIELDNAMES:
            raise ValueError("Manifest header does not match the required schema.")
        rows = list(reader)

    listed: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != set(MANIFEST_FIELDNAMES):
            raise ValueError(f"Manifest row {index} does not match its header.")
        relative = row["file"]
        if relative in listed:
            raise ValueError(f"Duplicate manifest path: {relative}")
        listed.add(relative)
        absolute, normalized = _relative_artifact_path(root, relative)
        if normalized != relative.replace("\\", "/"):
            raise ValueError(f"Manifest path is not normalized: {relative}")
        if absolute == manifest or Path(relative).name == "handoff_manifest.csv":
            raise ValueError(MANIFEST_SELF_HASH_POLICY)
        expected_digest = row["sha256"].lower()
        if not _SHA256_PATTERN.fullmatch(expected_digest):
            raise ValueError(f"Invalid SHA-256 text for {relative}: {row['sha256']!r}")
        actual_digest = sha256_file(absolute)
        if actual_digest != expected_digest:
            raise ValueError(
                f"SHA-256 mismatch for {relative}: "
                f"actual={actual_digest}, expected={expected_digest}."
            )

    if expected_files is not None:
        expected: set[str] = set()
        for path in expected_files:
            _, relative = _relative_artifact_path(root, path)
            if Path(relative).name != "handoff_manifest.csv":
                expected.add(relative)
        if listed != expected:
            raise ValueError(
                "Manifest artifact set mismatch; "
                f"missing={sorted(expected - listed)}, extra={sorted(listed - expected)}."
            )

    return {
        "valid": True,
        "manifest": str(manifest),
        "root_directory": str(root),
        "artifact_count": len(rows),
        "self_hash_policy": MANIFEST_SELF_HASH_POLICY,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write compact-handoff metadata CSVs or validate a SHA-256 manifest."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=SHARED_DATA_DIRECTORY,
        help="Directory for the four metadata CSVs (default: shared_data).",
    )
    parser.add_argument(
        "--validate-manifest",
        type=Path,
        help="Validate an existing handoff manifest instead of writing metadata.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments.validate_manifest is not None:
        report = validate_manifest(arguments.validate_manifest)
        print(
            f"Validated {report['artifact_count']} handoff artifact hashes; "
            f"{MANIFEST_SELF_HASH_POLICY}."
        )
        return 0

    paths = write_metadata_csvs(arguments.output_directory)
    for name, path in paths.items():
        print(f"{name}: {path.resolve()}")
    print("runtime_verified=false (standalone metadata-only mode)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEVICE_STRUCTURE_FIELDNAMES",
    "MANIFEST_FIELDNAMES",
    "MANIFEST_SELF_HASH_POLICY",
    "MATERIAL_FIELDNAMES",
    "REGION_CONTACT_FIELDNAMES",
    "SIMULATION_CONDITION_FIELDNAMES",
    "build_device_structure_rows",
    "build_manifest_rows",
    "build_material_rows",
    "build_region_contact_rows",
    "build_simulation_condition_rows",
    "canonical_parameters",
    "expected_geometry",
    "sha256_file",
    "validate_geometry_mapping",
    "validate_manifest",
    "validate_runtime_metadata",
    "write_csv_rows",
    "write_manifest",
    "write_metadata_csvs",
]
