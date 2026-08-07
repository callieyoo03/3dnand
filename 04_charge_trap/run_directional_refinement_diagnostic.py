"""Run isolated directional-mesh and Air-guard diagnostics.

This module is intentionally a *candidate-only* orchestrator.  It reuses the
existing contact-topology Q/C worker and changes only ``add_2d_mesh_line``
spacing during one device build.  Material parameters, active-region
boundaries, contact equations, solver tolerances, memory states, and public
CSV files are not changed.

The public run has two phases:

1. Fresh mesh-only probe processes choose the directional strength whose
   active-equation unknown-count proxy is closest to the current global
   extra-fine mesh (uniform ``ps`` scale 0.25).  Air/GateMetal coordinates
   are excluded from matching and retained only as mesh-size metadata.
2. Fresh point processes evaluate only State_0/on and State_4/off.  Separate
   State_0/on workers vary the characterization-only axial Air guard.

No ERASE or retention module is imported or executed here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import run_contact_topology_qc_candidate as contact_qc_runner


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
OUTPUT_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "qc_root_cause_diagnostics"
)
DIRECTIONAL_OUTPUT = OUTPUT_DIRECTORY / "directional_refinement_comparison.csv"
AIR_GUARD_OUTPUT = OUTPUT_DIRECTORY / "air_guard_invariance.csv"

CURRENT_GLOBAL_STRENGTH = 16.0
DIRECTIONAL_STRENGTH_CANDIDATES_BY_VARIANT = {
    "axial_near_sd": (2.0, 4.0, 8.0, 16.0, 32.0),
    # The two one-dimensional radial variants grow at very different rates
    # because r_core+r_mos2 bounds the full shell whereas r_mos2 alone is a
    # single interface line.  Probe each on its own practical range instead of
    # silently accepting the original shared upper bound of 32.
    "radial_mos2": (2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 64.0, 80.0, 88.0, 96.0, 104.0),
    "contact_corner_only": (2.0, 4.0, 8.0, 16.0, 32.0, 40.0, 48.0, 56.0),
    # Refining only the existing r_mos2 line has a rapidly diminishing DOF
    # return.  The expanded range documents that limitation without creating a
    # new interface band or changing geometry.
    "mos2_tunnel_interface": (
        2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0,
    ),
}
DOF_RELATIVE_TOLERANCE = 0.15
AIR_GUARD_FACTORS = (0.5, 1.0, 2.0)

POINT_SPECS = (
    (0, "on", 3.0),
    (4, "off", -1.0),
)
REFERENCE_VDS_V = 0.05
REFERENCE_VS_V = 0.0

ACTIVE_GEOMETRY_KEYS = (
    "r_axis",
    "r_core",
    "r_mos2",
    "r_tox",
    "r_trap",
    "r_block",
    "r_gate_outer",
    "r_air_outer",
    "z_source",
    "z_drain",
)

IMMUTABLE_DIRECTORIES = (
    MODULE_DIRECTORY / "results" / "full_terminal_qc_candidate",
    MODULE_DIRECTORY / "results" / "contact_topology_dc_candidate",
    MODULE_DIRECTORY / "results" / "contact_topology_qc_candidate",
)


@dataclass(frozen=True)
class DirectionalVariant:
    """One mesh-spacing transformation with an explicit dimensional budget."""

    name: str
    description: str
    geometry_keys: tuple[str, ...]
    effective_dimensions: int
    global_refinement: bool = False

    def spacing_scale(self, strength: float) -> float:
        strength_value = _positive_float(strength, "refinement strength")
        return strength_value ** (-1.0 / float(self.effective_dimensions))


VARIANTS = (
    DirectionalVariant(
        name="current_global",
        description=(
            "Current uniform extra-fine reference: every runtime mesh-line "
            "spacing is multiplied by 0.25."
        ),
        geometry_keys=(),
        effective_dimensions=2,
        global_refinement=True,
    ),
    DirectionalVariant(
        name="axial_near_sd",
        description=(
            "Refine only the existing z_source and z_drain mesh lines; their "
            "spacing controls the near-contact axial grading."
        ),
        geometry_keys=("z_source", "z_drain"),
        effective_dimensions=1,
    ),
    DirectionalVariant(
        name="radial_mos2",
        description=(
            "Refine the two radial mesh lines bounding the MoS2 shell "
            "(r_core and r_mos2)."
        ),
        geometry_keys=("r_core", "r_mos2"),
        effective_dimensions=1,
    ),
    DirectionalVariant(
        name="contact_corner_only",
        description=(
            "Refine the r_core/r_mos2 and z_source/z_drain cross-lines.  The "
            "line mesher cannot create a strictly local 2-D patch, so this is "
            "an intersection-focused corner surrogate, not an unstructured "
            "corner-only remesh."
        ),
        geometry_keys=("r_core", "r_mos2", "z_source", "z_drain"),
        effective_dimensions=2,
    ),
    DirectionalVariant(
        name="mos2_tunnel_interface",
        description=(
            "Refine only the existing r_mos2 line at the MoS2/TunnelOxide "
            "interface."
        ),
        geometry_keys=("r_mos2",),
        effective_dimensions=1,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


DIRECTIONAL_FIELDS = (
    "experiment",
    "variant",
    "variant_description",
    "refinement_strength",
    "applied_spacing_scale",
    "refined_geometry_keys",
    "guard_extension_factor",
    "state_index",
    "state",
    "bias_name",
    "target_VGS_V",
    "target_VDS_V",
    "target_VS_V",
    "actual_VGS_V",
    "actual_VDS_V",
    "actual_VS_V",
    "target_reached",
    "ID_A",
    "Qg_C",
    "raw_Qd_C",
    "raw_Qs_C",
    "Qmobile_C",
    "Cgg_F",
    "Cgd_F",
    "Cgs_F",
    "global_gauss_residual_C",
    "global_gauss_scale_C",
    "global_gauss_tolerance_C",
    "global_gauss_passed",
    "runtime_seconds",
    "global_coordinate_count",
    "active_unknown_count",
    "region_node_count_with_duplicates",
    "triangle_count",
    "source_edge_count",
    "drain_edge_count",
    "gate_edge_count",
    "mesh_line_count",
    "reference_global_coordinate_count",
    "reference_active_unknown_count",
    "dof_relative_difference",
    "dof_comparable",
    "dof_match_note",
    "active_geometry_sha256",
    "worker_process_id",
    "qc_point_passed",
    "converged",
    "error_message",
)

AIR_GUARD_FIELDS = (
    "experiment",
    "guard_extension_factor",
    "guard_extension_cm",
    "baseline_guard_extension_factor",
    "state_index",
    "state",
    "bias_name",
    "target_VGS_V",
    "target_VDS_V",
    "target_VS_V",
    "target_reached",
    "quantity",
    "value",
    "baseline_value",
    "absolute_difference",
    "relative_difference",
    "comparison_floor",
    "runtime_seconds",
    "global_coordinate_count",
    "triangle_count",
    "source_edge_count",
    "drain_edge_count",
    "active_geometry_sha256",
    "active_geometry_unchanged",
    "worker_process_id",
    "converged",
    "error_message",
)

AIR_GUARD_QUANTITIES = (
    "ID_A",
    "Qg_C",
    "raw_Qd_C",
    "raw_Qs_C",
    "Qmobile_C",
    "Cgg_F",
    "Cgd_F",
    "Cgs_F",
    "global_gauss_residual_C",
)

COMPARISON_FLOORS = {
    "ID_A": 1.0e-30,
    "Qg_C": 1.0e-30,
    "raw_Qd_C": 1.0e-30,
    "raw_Qs_C": 1.0e-30,
    "Qmobile_C": 1.0e-30,
    "Cgg_F": 1.0e-24,
    "Cgd_F": 1.0e-24,
    "Cgs_F": 1.0e-24,
    "global_gauss_residual_C": 1.0e-30,
}


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _error(error: object | None) -> str:
    return contact_qc_runner._error(error)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(value),
            stream,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        stream.write("\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_immutable_inputs() -> dict[str, str]:
    """Hash public Q/C and every pre-existing candidate artifact."""

    paths = list(contact_qc_runner.IMMUTABLE_PUBLIC_QC)
    for directory in IMMUTABLE_DIRECTORIES:
        if directory.is_dir():
            paths.extend(
                path for path in sorted(directory.rglob("*")) if path.is_file()
            )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("immutable diagnostic input missing: " + ", ".join(missing))
    return {
        str(path.resolve().relative_to(REPOSITORY_ROOT.resolve())): _sha256_file(path)
        for path in paths
    }


def assert_immutable_inputs(expected: Mapping[str, str]) -> None:
    actual = capture_immutable_inputs()
    if dict(expected) != actual:
        changed = sorted(
            name for name in set(expected) | set(actual)
            if expected.get(name) != actual.get(name)
        )
        raise RuntimeError("immutable candidate/public input changed: " + ", ".join(changed))


def active_geometry_sha256(geometry: Mapping[str, Any]) -> str:
    """Hash only fixed device coordinates, excluding the temporary guard."""

    payload = {
        key: format(float(geometry[key]), ".17g") for key in ACTIVE_GEOMETRY_KEYS
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _coordinate_targets(
    geometry: Mapping[str, Any], variant: DirectionalVariant,
) -> dict[tuple[str, float], str]:
    targets: dict[tuple[str, float], str] = {}
    for key in variant.geometry_keys:
        direction = "y" if key.startswith("z_") else "x"
        targets[(direction, float(geometry[key]))] = key
    return targets


def line_spacing_scale(
    *, direction: str, position_cm: float, geometry: Mapping[str, Any],
    variant: DirectionalVariant, strength: float,
) -> tuple[float, str]:
    """Return the spacing multiplier and the matched geometry label."""

    scale = variant.spacing_scale(strength)
    if variant.global_refinement:
        return scale, "all_runtime_lines"
    targets = _coordinate_targets(geometry, variant)
    for (target_direction, target_position), key in targets.items():
        if direction == target_direction and math.isclose(
            float(position_cm), target_position, rel_tol=0.0, abs_tol=1.0e-30
        ):
            return scale, key
    return 1.0, "unchanged"


@contextmanager
def directional_runtime_mesh_lines(
    structure_module: Any,
    geometry: Mapping[str, Any],
    variant: DirectionalVariant,
    strength: float,
    line_log: list[dict[str, Any]],
) -> Iterator[None]:
    """Temporarily refine selected runtime mesh lines and log forwarded data."""

    _positive_float(strength, "refinement strength")
    original = structure_module.add_2d_mesh_line

    def add_directional_line(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError("directional mesh audit requires keyword mesh-line calls")
        direction = str(kwargs.get("dir", ""))
        if direction not in {"x", "y"}:
            raise RuntimeError(f"unexpected mesh-line direction {direction!r}")
        position = float(kwargs["pos"])
        input_spacing = _positive_float(kwargs["ps"], "input mesh spacing")
        scale, target = line_spacing_scale(
            direction=direction,
            position_cm=position,
            geometry=geometry,
            variant=variant,
            strength=strength,
        )
        output_spacing = input_spacing * scale
        line_log.append(
            {
                "direction": direction,
                "position_cm": position,
                "input_spacing_cm": input_spacing,
                "output_spacing_cm": output_spacing,
                "applied_scale": scale,
                "target": target,
            }
        )
        forwarded = dict(kwargs)
        forwarded["ps"] = output_spacing
        return original(**forwarded)

    structure_module.add_2d_mesh_line = add_directional_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = original


def validate_directional_mesh_line_log(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    variant: DirectionalVariant,
    strength: float,
) -> dict[str, Any]:
    """Verify that only the declared line spacings changed."""

    errors: list[str] = []
    seen_targets: set[str] = set()
    if len(rows) != 12:
        errors.append(f"line_count={len(rows)} expected=12")
    for index, row in enumerate(rows):
        expected_scale, expected_target = line_spacing_scale(
            direction=str(row.get("direction")),
            position_cm=_number(row.get("position_cm")),
            geometry=geometry,
            variant=variant,
            strength=strength,
        )
        actual_scale = _number(row.get("applied_scale"))
        input_spacing = _number(row.get("input_spacing_cm"))
        output_spacing = _number(row.get("output_spacing_cm"))
        if not math.isclose(actual_scale, expected_scale, rel_tol=1.0e-14, abs_tol=1.0e-30):
            errors.append(f"line {index} scale")
        if str(row.get("target")) != expected_target:
            errors.append(f"line {index} target")
        if not math.isclose(
            output_spacing, input_spacing * expected_scale,
            rel_tol=1.0e-14, abs_tol=1.0e-30,
        ):
            errors.append(f"line {index} spacing")
        if expected_target not in {"unchanged", "all_runtime_lines"}:
            seen_targets.add(expected_target)
    if not variant.global_refinement:
        missing = set(variant.geometry_keys) - seen_targets
        if missing:
            errors.append("missing targets=" + ",".join(sorted(missing)))
    return {
        "passed": not errors,
        "errors": errors,
        "line_count": len(rows),
        "applied_spacing_scale": variant.spacing_scale(strength),
        "refined_geometry_keys": list(variant.geometry_keys),
    }


@contextmanager
def scaled_air_guard(contact_structure_module: Any, factor: float) -> Iterator[None]:
    """Scale only the axial Air extension, keeping guard-line ``ps`` fixed."""

    factor_value = _positive_float(factor, "Air guard factor")
    original = contact_structure_module.expanded_air_bounds

    def scaled(geometry: Mapping[str, Any]) -> dict[str, float]:
        bounds = dict(original(geometry))
        baseline_guard = _positive_float(
            bounds["guard_cm"], "baseline Air guard spacing"
        )
        extension = baseline_guard * factor_value
        bounds.update(
            {
                "yl": float(geometry["z_source"]) - extension,
                "yh": float(geometry["z_drain"]) + extension,
                # _temporary_contact_topology uses guard_cm as the guard-line
                # mesh spacing.  Keep it at the canonical 2-nm value so this
                # experiment varies extension only.
                "guard_cm": baseline_guard,
            }
        )
        return bounds

    contact_structure_module.expanded_air_bounds = scaled
    try:
        yield
    finally:
        contact_structure_module.expanded_air_bounds = original


def validate_guard_mesh_line_log(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
    guard_factor: float,
) -> dict[str, Any]:
    """Validate physical lines and a deliberately scaled guard pair."""

    scale_value = _positive_float(scale, "legacy mesh scale")
    baseline_guard = (
        float(geometry["r_mos2"]) - float(geometry["r_core"])
    )
    guard = baseline_guard * _positive_float(guard_factor, "Air guard factor")
    expected_guards = (
        float(geometry["z_source"]) - guard,
        float(geometry["z_drain"]) + guard,
    )
    errors: list[str] = []
    if len(rows) != 12:
        errors.append(f"line_count={len(rows)} expected=12")
    positions = {
        (str(row.get("direction")), round(_number(row.get("position_cm")), 18))
        for row in rows
    }
    required = {
        ("x", round(float(geometry[key]), 18))
        for key in ACTIVE_GEOMETRY_KEYS if key.startswith("r_")
    }
    required.update(
        {
            ("y", round(float(geometry["z_source"]), 18)),
            ("y", round(float(geometry["z_drain"]), 18)),
            *(('y', round(value, 18)) for value in expected_guards),
        }
    )
    if not required.issubset(positions):
        errors.append("required physical or guard mesh line missing")
    guard_rows = [
        row for row in rows
        if str(row.get("direction")) == "y"
        and (
            _number(row.get("position_cm")) < float(geometry["z_source"])
            or _number(row.get("position_cm")) > float(geometry["z_drain"])
        )
    ]
    if len(guard_rows) != 2:
        errors.append(f"guard_line_count={len(guard_rows)} expected=2")
    for index, row in enumerate(guard_rows):
        if not math.isclose(
            _number(row.get("base_spacing_cm")),
            baseline_guard,
            rel_tol=0.0,
            abs_tol=1.0e-30,
        ):
            errors.append(f"guard line {index} base spacing changed")
    for index, row in enumerate(rows):
        base = _number(row.get("base_spacing_cm"))
        scaled_spacing = _number(row.get("scaled_spacing_cm"))
        if not math.isclose(
            scaled_spacing, base * scale_value, rel_tol=0.0, abs_tol=1.0e-30
        ):
            errors.append(f"line {index} legacy scale")
    return {
        "passed": not errors,
        "errors": errors,
        "line_count": len(rows),
        "guard_extension_cm": guard,
        "guard_mesh_spacing_cm": baseline_guard,
        "guard_factor": float(guard_factor),
    }


@contextmanager
def guard_aware_legacy_validation(guard_factor: float) -> Iterator[None]:
    """Let the reused worker accept only the requested non-default guard."""

    original = contact_qc_runner.validate_topology_mesh_line_log

    def validate(
        rows: Sequence[Mapping[str, Any]],
        geometry: Mapping[str, Any],
        scale: float,
    ) -> dict[str, Any]:
        result = validate_guard_mesh_line_log(rows, geometry, scale, guard_factor)
        if not result["passed"]:
            raise RuntimeError(
                "Air-guard mesh-line validation failed: "
                + ", ".join(result["errors"])
            )
        return result

    contact_qc_runner.validate_topology_mesh_line_log = validate
    try:
        yield
    finally:
        contact_qc_runner.validate_topology_mesh_line_log = original


def _runtime_mesh_counts(device: str) -> dict[str, int]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    coordinates: set[int] = set()
    region_nodes = 0
    triangles = 0
    nodes_by_region: dict[str, int] = {}
    for region in get_region_list(device=device):
        indices = tuple(
            int(value) for value in get_node_model_values(
                device=device, region=region, name="coordinate_index"
            )
        )
        coordinates.update(indices)
        nodes_by_region[str(region)] = len(indices)
        region_nodes += len(indices)
        triangles += len(get_element_node_list(device=device, region=region))
    # Equation-DOF proxy: Potential in the four dielectric regions and
    # Potential+Electrons in MoS2.  Air and GateMetal are mesh-only regions.
    active_unknown_count = (
        2 * nodes_by_region.get("MoS2", 0)
        + nodes_by_region.get("CoreOxide", 0)
        + nodes_by_region.get("TunnelOxide", 0)
        + nodes_by_region.get("ChargeTrap", 0)
        + nodes_by_region.get("BlockingOxide", 0)
    )
    return {
        "global_coordinate_count": len(coordinates),
        "active_unknown_count": active_unknown_count,
        "region_node_count_with_duplicates": region_nodes,
        "triangle_count": triangles,
    }


def _preview_geometry(memory_window: Any) -> dict[str, Any]:
    geometry = dict(
        memory_window.parameterized_device_structure.calculate_geometry(
            tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM
        )
    )
    # The same guard used by the established characterization runners rejects
    # environment-driven geometry overrides before either a probe or a solve.
    from state_sweep_helpers import validate_canonical_handoff_geometry

    validate_canonical_handoff_geometry(geometry)
    return geometry


def run_probe_worker(
    *, variant_name: str, strength: float, guard_factor: float = 1.0,
) -> dict[str, Any]:
    """Build and count one mesh without installing or solving device physics."""

    import contact_topology_structure
    import run_memory_window as memory_window

    variant = VARIANT_BY_NAME[variant_name]
    preview = _preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with scaled_air_guard(contact_topology_structure, guard_factor):
        with directional_runtime_mesh_lines(
            memory_window.parameterized_device_structure,
            preview,
            variant,
            strength,
            line_log,
        ):
            geometry = memory_window.create_structure(
                tunnel_oxide_thickness_nm=memory_window.TUNNEL_OXIDE_THICKNESS_NM
            )
    elapsed = time.perf_counter() - started
    if active_geometry_sha256(preview) != active_geometry_sha256(geometry):
        raise RuntimeError("mesh probe changed active geometry")
    validation = validate_directional_mesh_line_log(
        line_log, geometry, variant, strength
    )
    if not validation["passed"]:
        raise RuntimeError("directional mesh validation failed: " + ", ".join(validation["errors"]))
    return {
        "mode": "probe",
        "variant": variant.name,
        "strength": float(strength),
        "guard_factor": float(guard_factor),
        "geometry": dict(geometry),
        "active_geometry_sha256": active_geometry_sha256(geometry),
        "mesh_counts": _runtime_mesh_counts(memory_window.device),
        "mesh_line_log": line_log,
        "mesh_line_validation": validation,
        "runtime_seconds": elapsed,
        "worker_process_id": os.getpid(),
        "error_message": "",
    }


def run_point_worker(
    *, variant_name: str, strength: float, state_index: int, bias_name: str,
    guard_factor: float = 1.0,
) -> dict[str, Any]:
    """Evaluate one Q/C reference point in its already-isolated process."""

    import contact_topology_structure
    import run_memory_window as memory_window

    variant = VARIANT_BY_NAME[variant_name]
    preview = _preview_geometry(memory_window)
    line_log: list[dict[str, Any]] = []
    started = time.perf_counter()
    with scaled_air_guard(contact_topology_structure, guard_factor):
        with guard_aware_legacy_validation(guard_factor):
            with directional_runtime_mesh_lines(
                memory_window.parameterized_device_structure,
                preview,
                variant,
                strength,
                line_log,
            ):
                bundle = contact_qc_runner.run_point_worker(
                    mesh_level=f"directional:{variant.name}",
                    mesh_scale=1.0,
                    state_index=int(state_index),
                    bias_name=str(bias_name),
                )
    elapsed = time.perf_counter() - started
    geometry = bundle.get("geometry", preview)
    if active_geometry_sha256(preview) != active_geometry_sha256(geometry):
        raise RuntimeError("directional point worker changed active geometry")
    validation = validate_directional_mesh_line_log(
        line_log, geometry, variant, strength
    )
    if not validation["passed"]:
        raise RuntimeError("directional mesh validation failed: " + ", ".join(validation["errors"]))

    try:
        actual_biases = contact_qc_runner._read_biases(contact_qc_runner.legacy_qc.DEVICE_NAME)
    except Exception:
        actual_biases = {"gate": math.nan, "drain": math.nan, "source": math.nan}
    try:
        drain_current = float(memory_window.get_drain_current_A())
    except Exception:
        drain_current = math.nan

    bundle.update(
        {
            "mode": "point",
            "variant": variant.name,
            "strength": float(strength),
            "guard_factor": float(guard_factor),
            "active_geometry_sha256": active_geometry_sha256(geometry),
            "directional_mesh_line_log": line_log,
            "directional_mesh_line_validation": validation,
            "actual_biases_after_worker": actual_biases,
            "drain_current_A": drain_current,
            "diagnostic_mesh_counts": _runtime_mesh_counts(
                contact_qc_runner.legacy_qc.DEVICE_NAME
            ),
            "runtime_seconds": elapsed,
            "worker_process_id": os.getpid(),
        }
    )
    return bundle


def select_strength_for_similar_dof(
    probe_counts: Mapping[float, int], reference_count: int,
) -> dict[str, Any]:
    """Choose the probed strength closest to the global-reference DOF."""

    target = int(reference_count)
    if target <= 0 or not probe_counts:
        raise ValueError("positive reference and at least one probe are required")
    candidates: list[tuple[float, float, int]] = []
    for strength, count in probe_counts.items():
        count_value = int(count)
        if count_value <= 0:
            continue
        relative = abs(count_value - target) / float(target)
        candidates.append((relative, float(strength), count_value))
    if not candidates:
        raise ValueError("all mesh probes have invalid coordinate counts")
    relative, strength, count = min(candidates, key=lambda row: (row[0], row[1]))
    return {
        "strength": strength,
        "active_unknown_count": count,
        "reference_active_unknown_count": target,
        "dof_relative_difference": relative,
        "dof_comparable": relative <= DOF_RELATIVE_TOLERANCE,
    }


def build_worker_command(
    *, python_executable: str, mode: str, variant_name: str, strength: float,
    output_path: Path, guard_factor: float = 1.0,
    state_index: int | None = None, bias_name: str | None = None,
) -> list[str]:
    if mode not in {"probe", "point"}:
        raise ValueError("mode must be probe or point")
    command = [
        str(python_executable),
        "-B",
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--variant",
        variant_name,
        "--strength",
        format(float(strength), ".17g"),
        "--guard-factor",
        format(float(guard_factor), ".17g"),
        "--worker-output",
        str(output_path.resolve()),
    ]
    if mode == "point":
        if state_index is None or bias_name is None:
            raise ValueError("point worker requires state_index and bias_name")
        command.extend(("--state-index", str(int(state_index)), "--bias-name", str(bias_name)))
    return command


def _run_subprocess(command: Sequence[str], log_path: Path) -> dict[str, Any]:
    output_path = Path(command[command.index("--worker-output") + 1])
    completed = subprocess.run(
        list(command), capture_output=True, text=True, check=False
    )
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"worker exited {completed.returncode}; inspect {log_path}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"worker did not create {output_path}")
    with output_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _run_commands(
    commands: Sequence[tuple[str, list[str]]], directory: Path, jobs: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}

    def execute(label: str, command: list[str]) -> tuple[str, dict[str, Any]]:
        return label, _run_subprocess(command, directory / f"{label}.log")

    with ThreadPoolExecutor(max_workers=max(1, int(jobs))) as executor:
        futures = {
            executor.submit(execute, label, command): label
            for label, command in commands
        }
        for future in as_completed(futures):
            label, result = future.result()
            results[label] = result
    return results


def _point_label(
    variant_name: str, state_index: int, bias_name: str, guard_factor: float,
) -> str:
    guard_text = format(float(guard_factor), ".12g").replace(".", "p")
    return f"point_{variant_name}_s{state_index}_{bias_name}_g{guard_text}"


def run_isolated_diagnostics(
    temporary_directory: Path, *, jobs: int = 1,
) -> dict[str, Any]:
    """Probe DOF, run ten directional points, and run two extra guard points."""

    temporary_directory.mkdir(parents=True, exist_ok=True)
    python = sys.executable
    probe_commands: list[tuple[str, list[str]]] = []
    probe_specs: list[tuple[str, float]] = [
        ("current_global", CURRENT_GLOBAL_STRENGTH)
    ]
    probe_specs.extend(
        (variant.name, strength)
        for variant in VARIANTS if variant.name != "current_global"
        for strength in DIRECTIONAL_STRENGTH_CANDIDATES_BY_VARIANT[variant.name]
    )
    for variant_name, strength in probe_specs:
        label = f"probe_{variant_name}_{format(strength, '.12g').replace('.', 'p')}"
        output = temporary_directory / f"{label}.json"
        probe_commands.append(
            (
                label,
                build_worker_command(
                    python_executable=python,
                    mode="probe",
                    variant_name=variant_name,
                    strength=strength,
                    output_path=output,
                ),
            )
        )
    probes = _run_commands(probe_commands, temporary_directory, jobs)
    global_label = f"probe_current_global_{format(CURRENT_GLOBAL_STRENGTH, '.12g')}"
    if not isinstance(probes.get(global_label, {}).get("mesh_counts"), Mapping):
        raise RuntimeError(
            "current_global mesh-only reference probe failed: "
            + str(probes.get(global_label, {}).get("error_message", "missing bundle"))
        )
    reference_probe = probes[global_label]["mesh_counts"]
    reference_count = int(reference_probe["active_unknown_count"])
    reference_global_count = int(reference_probe["global_coordinate_count"])
    selections: dict[str, dict[str, Any]] = {
        "current_global": {
            "strength": CURRENT_GLOBAL_STRENGTH,
            "active_unknown_count": reference_count,
            "reference_active_unknown_count": reference_count,
            "global_coordinate_count": reference_global_count,
            "reference_global_coordinate_count": reference_global_count,
            "dof_relative_difference": 0.0,
            "dof_comparable": True,
        }
    }
    for variant in VARIANTS:
        if variant.name == "current_global":
            continue
        counts: dict[float, int] = {}
        failed_candidates: dict[float, str] = {}
        candidate_strengths = DIRECTIONAL_STRENGTH_CANDIDATES_BY_VARIANT[
            variant.name
        ]
        for strength in candidate_strengths:
            label = (
                f"probe_{variant.name}_"
                f"{format(strength, '.12g').replace('.', 'p')}"
            )
            bundle = probes[label]
            mesh_counts = bundle.get("mesh_counts")
            if isinstance(mesh_counts, Mapping):
                counts[strength] = int(mesh_counts["active_unknown_count"])
            else:
                failed_candidates[strength] = str(
                    bundle.get("error_message", "missing mesh counts")
                )
        selection = select_strength_for_similar_dof(counts, reference_count)
        selected_label = (
            f"probe_{variant.name}_"
            f"{format(selection['strength'], '.12g').replace('.', 'p')}"
        )
        selection.update(
            {
                "global_coordinate_count": int(
                    probes[selected_label]["mesh_counts"]["global_coordinate_count"]
                ),
                "reference_global_coordinate_count": reference_global_count,
                "failed_probe_count": len(failed_candidates),
                "failed_probe_messages": failed_candidates,
                "maximum_probed_strength": max(candidate_strengths),
            }
        )
        selections[variant.name] = selection

    point_commands: list[tuple[str, list[str]]] = []
    for variant in VARIANTS:
        strength = float(selections[variant.name]["strength"])
        for state_index, bias_name, _ in POINT_SPECS:
            label = _point_label(variant.name, state_index, bias_name, 1.0)
            point_commands.append(
                (
                    label,
                    build_worker_command(
                        python_executable=python,
                        mode="point",
                        variant_name=variant.name,
                        strength=strength,
                        state_index=state_index,
                        bias_name=bias_name,
                        guard_factor=1.0,
                        output_path=temporary_directory / f"{label}.json",
                    ),
                )
            )
    for guard_factor in AIR_GUARD_FACTORS:
        if math.isclose(guard_factor, 1.0, rel_tol=0.0, abs_tol=0.0):
            continue
        label = _point_label("current_global", 0, "on", guard_factor)
        point_commands.append(
            (
                label,
                build_worker_command(
                    python_executable=python,
                    mode="point",
                    variant_name="current_global",
                    strength=CURRENT_GLOBAL_STRENGTH,
                    state_index=0,
                    bias_name="on",
                    guard_factor=guard_factor,
                    output_path=temporary_directory / f"{label}.json",
                ),
            )
        )
    points = _run_commands(point_commands, temporary_directory, jobs)
    return {"probes": probes, "selections": selections, "points": points}


def _nested(mapping: Mapping[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def point_values(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the requested current, charge, C, Gauss, and mesh quantities."""

    points = bundle.get("points", ())
    point = points[0] if isinstance(points, Sequence) and len(points) == 1 else {}
    direct = point.get("direct", {}) if isinstance(point, Mapping) else {}
    matrix = point.get("nominal_matrix_F", {}) if isinstance(point, Mapping) else {}
    gauss = point.get("global_gauss", {}) if isinstance(point, Mapping) else {}
    mesh = bundle.get("diagnostic_mesh_counts", bundle.get("mesh_counts", {}))
    actual = bundle.get("actual_biases_after_worker", {})
    final_reference = bundle.get("final_reference_validation", {})
    expected_vgs = _number(point.get("VGS_V"))
    expected_vds = _number(point.get("VDS_V"))
    expected_vs = _number(point.get("VS_V"))
    actual_gate = _number(actual.get("gate"))
    actual_drain = _number(actual.get("drain"))
    actual_source = _number(actual.get("source"))
    actual_vgs = actual_gate - actual_source
    actual_vds = actual_drain - actual_source
    target_reached = bool(final_reference.get("passed")) and all(
        math.isfinite(value)
        for value in (actual_vgs, actual_vds, actual_source, expected_vgs, expected_vds, expected_vs)
    ) and all(
        math.isclose(actual_value, expected_value, rel_tol=0.0, abs_tol=1.0e-12)
        for actual_value, expected_value in (
            (actual_vgs, expected_vgs),
            (actual_vds, expected_vds),
            (actual_source, expected_vs),
        )
    )
    topology = bundle.get("topology", {})
    contacts = topology.get("contacts", {}) if isinstance(topology, Mapping) else {}
    errors = [
        str(point.get("error_message", "")) if isinstance(point, Mapping) else "missing point",
        str(final_reference.get("error_message", "")),
        str(bundle.get("error_message", "")),
    ]
    result = {
        "state_index": int(point.get("state_index", -1)) if isinstance(point, Mapping) else -1,
        "state": str(point.get("state", "")) if isinstance(point, Mapping) else "",
        "bias_name": str(point.get("bias_name", "")) if isinstance(point, Mapping) else "",
        "target_VGS_V": expected_vgs,
        "target_VDS_V": expected_vds,
        "target_VS_V": expected_vs,
        "actual_VGS_V": actual_vgs,
        "actual_VDS_V": actual_vds,
        "actual_VS_V": actual_source,
        "target_reached": target_reached,
        "ID_A": _number(bundle.get("drain_current_A")),
        "Qg_C": _number(direct.get("Qg_contact_C")),
        "raw_Qd_C": _number(direct.get("Qd_contact_C")),
        "raw_Qs_C": _number(direct.get("Qs_contact_C")),
        "Qmobile_C": _number(direct.get("Qmobile_C")),
        "Cgg_F": _number(matrix.get("gate:gate")),
        "Cgd_F": _number(matrix.get("gate:drain")),
        "Cgs_F": _number(matrix.get("gate:source")),
        "global_gauss_residual_C": _number(gauss.get("residual_C")),
        "global_gauss_scale_C": _number(gauss.get("scale_C")),
        "global_gauss_tolerance_C": _number(gauss.get("tolerance_C")),
        "global_gauss_passed": bool(gauss.get("passed")),
        "runtime_seconds": _number(bundle.get("runtime_seconds")),
        "global_coordinate_count": int(mesh.get("global_coordinate_count", 0)),
        "active_unknown_count": int(mesh.get("active_unknown_count", 0)),
        "region_node_count_with_duplicates": int(mesh.get("region_node_count_with_duplicates", 0)),
        "triangle_count": int(mesh.get("triangle_count", 0)),
        "source_edge_count": int(_nested(contacts, "source", "edge_count") or 0),
        "drain_edge_count": int(_nested(contacts, "drain", "edge_count") or 0),
        "gate_edge_count": int(_nested(contacts, "gate", "edge_count") or 0),
        "mesh_line_count": int(_nested(bundle, "directional_mesh_line_validation", "line_count") or 0),
        "active_geometry_sha256": str(bundle.get("active_geometry_sha256", "")),
        "worker_process_id": int(bundle.get("worker_process_id", -1)),
        "qc_point_passed": bool(point.get("passed")) if isinstance(point, Mapping) else False,
        "converged": False,
        "error_message": "; ".join(dict.fromkeys(error for error in errors if error)),
    }
    requested_values = (
        result["ID_A"],
        result["Qg_C"],
        result["raw_Qd_C"],
        result["raw_Qs_C"],
        result["Qmobile_C"],
        result["Cgg_F"],
        result["Cgd_F"],
        result["Cgs_F"],
        result["global_gauss_residual_C"],
    )
    result["converged"] = target_reached and all(
        math.isfinite(float(value)) for value in requested_values
    )
    return result


def build_directional_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selections = result["selections"]
    points = result["points"]
    for variant in VARIANTS:
        selection = selections[variant.name]
        for state_index, bias_name, target_vgs in POINT_SPECS:
            bundle = points[_point_label(variant.name, state_index, bias_name, 1.0)]
            values = point_values(bundle)
            # Preserve requested targets even when the reused worker failed before
            # it could construct its point dictionary.
            values["state_index"] = state_index
            values["bias_name"] = bias_name
            values["target_VGS_V"] = target_vgs
            values["target_VDS_V"] = REFERENCE_VDS_V
            values["target_VS_V"] = REFERENCE_VS_V
            reference_count = int(selection["reference_active_unknown_count"])
            actual_count = int(values["active_unknown_count"])
            if reference_count > 0 and actual_count > 0:
                dof_relative_difference = abs(actual_count - reference_count) / float(
                    reference_count
                )
                dof_comparable = dof_relative_difference <= DOF_RELATIVE_TOLERANCE
            else:
                dof_relative_difference = math.nan
                dof_comparable = False
            row = {
                "experiment": "directional_refinement",
                "variant": variant.name,
                "variant_description": variant.description,
                "refinement_strength": selection["strength"],
                "applied_spacing_scale": variant.spacing_scale(selection["strength"]),
                "refined_geometry_keys": (
                    "all_runtime_lines" if variant.global_refinement
                    else ";".join(variant.geometry_keys)
                ),
                "guard_extension_factor": 1.0,
                "state_index": values["state_index"],
                "state": values["state"],
                "bias_name": values["bias_name"],
                "target_VGS_V": values["target_VGS_V"],
                "target_VDS_V": values["target_VDS_V"],
                "target_VS_V": values["target_VS_V"],
                "actual_VGS_V": values["actual_VGS_V"],
                "actual_VDS_V": values["actual_VDS_V"],
                "actual_VS_V": values["actual_VS_V"],
                "target_reached": values["target_reached"],
                "ID_A": values["ID_A"],
                "Qg_C": values["Qg_C"],
                "raw_Qd_C": values["raw_Qd_C"],
                "raw_Qs_C": values["raw_Qs_C"],
                "Qmobile_C": values["Qmobile_C"],
                "Cgg_F": values["Cgg_F"],
                "Cgd_F": values["Cgd_F"],
                "Cgs_F": values["Cgs_F"],
                "global_gauss_residual_C": values["global_gauss_residual_C"],
                "global_gauss_scale_C": values["global_gauss_scale_C"],
                "global_gauss_tolerance_C": values["global_gauss_tolerance_C"],
                "global_gauss_passed": values["global_gauss_passed"],
                "runtime_seconds": values["runtime_seconds"],
                "global_coordinate_count": values["global_coordinate_count"],
                "active_unknown_count": values["active_unknown_count"],
                "region_node_count_with_duplicates": values[
                    "region_node_count_with_duplicates"
                ],
                "triangle_count": values["triangle_count"],
                "source_edge_count": values["source_edge_count"],
                "drain_edge_count": values["drain_edge_count"],
                "gate_edge_count": values["gate_edge_count"],
                "mesh_line_count": values["mesh_line_count"],
                "reference_global_coordinate_count": selection[
                    "reference_global_coordinate_count"
                ],
                "reference_active_unknown_count": selection[
                    "reference_active_unknown_count"
                ],
                "dof_relative_difference": dof_relative_difference,
                "dof_comparable": dof_comparable,
                "dof_match_note": (
                    ""
                    if dof_comparable
                    else (
                        "Existing runtime mesh-line refinement did not reach "
                        "the 15% active-unknown match within the practical "
                        f"probe range (maximum strength "
                        f"{selection['maximum_probed_strength']:.12g}); do not "
                        "rank this row against current_global by equal-DOF."
                    )
                ),
                "active_geometry_sha256": values["active_geometry_sha256"],
                "worker_process_id": values["worker_process_id"],
                "qc_point_passed": values["qc_point_passed"],
                "converged": values["converged"],
                "error_message": values["error_message"],
            }
            if tuple(row) != DIRECTIONAL_FIELDS:
                raise RuntimeError("directional row schema mismatch")
            rows.append(row)
    return rows


def _relative_difference(left: float, right: float, floor: float) -> tuple[float, float]:
    if not math.isfinite(left) or not math.isfinite(right):
        return math.nan, math.nan
    difference = abs(left - right)
    scale = max(abs(left), abs(right), float(floor))
    return difference, difference / scale


def build_air_guard_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    points = result["points"]
    bundles = {
        factor: points[_point_label("current_global", 0, "on", factor)]
        for factor in AIR_GUARD_FACTORS
    }
    values_by_factor = {factor: point_values(bundle) for factor, bundle in bundles.items()}
    baseline = values_by_factor[1.0]
    baseline_geometry_hash = baseline["active_geometry_sha256"]
    rows: list[dict[str, Any]] = []
    for factor in AIR_GUARD_FACTORS:
        values = values_by_factor[factor]
        geometry = bundles[factor].get("geometry", {})
        guard_cm = (
            _number(geometry.get("r_mos2")) - _number(geometry.get("r_core"))
        ) * factor
        for quantity in AIR_GUARD_QUANTITIES:
            floor = COMPARISON_FLOORS[quantity]
            absolute, relative = _relative_difference(
                _number(values[quantity]), _number(baseline[quantity]), floor
            )
            row = {
                "experiment": "air_guard_invariance",
                "guard_extension_factor": factor,
                "guard_extension_cm": guard_cm,
                "baseline_guard_extension_factor": 1.0,
                "state_index": 0,
                "state": values["state"],
                "bias_name": "on",
                "target_VGS_V": 3.0,
                "target_VDS_V": REFERENCE_VDS_V,
                "target_VS_V": REFERENCE_VS_V,
                "target_reached": values["target_reached"],
                "quantity": quantity,
                "value": values[quantity],
                "baseline_value": baseline[quantity],
                "absolute_difference": absolute,
                "relative_difference": relative,
                "comparison_floor": floor,
                "runtime_seconds": values["runtime_seconds"],
                "global_coordinate_count": values["global_coordinate_count"],
                "triangle_count": values["triangle_count"],
                "source_edge_count": values["source_edge_count"],
                "drain_edge_count": values["drain_edge_count"],
                "active_geometry_sha256": values["active_geometry_sha256"],
                "active_geometry_unchanged": (
                    values["active_geometry_sha256"] == baseline_geometry_hash
                ),
                "worker_process_id": values["worker_process_id"],
                "converged": values["converged"],
                "error_message": values["error_message"],
            }
            if tuple(row) != AIR_GUARD_FIELDS:
                raise RuntimeError("Air-guard row schema mismatch")
            rows.append(row)
    return rows


def _write_new_csv(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]],
) -> None:
    if path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing diagnostic result: {path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing to replace a stale temporary file: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            if tuple(row) != tuple(fieldnames):
                raise RuntimeError(f"CSV row schema mismatch for {path.name}")
            writer.writerow(row)
    # ``rename`` fails on Windows if another process created the final path
    # after the preflight check; it therefore cannot silently overwrite it.
    temporary.rename(path)


def write_outputs(result: Mapping[str, Any]) -> tuple[Path, Path]:
    existing = [path for path in (DIRECTIONAL_OUTPUT, AIR_GUARD_OUTPUT) if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite existing diagnostic result(s): "
            + ", ".join(str(path) for path in existing)
        )
    directional_rows = build_directional_rows(result)
    guard_rows = build_air_guard_rows(result)
    _write_new_csv(DIRECTIONAL_OUTPUT, DIRECTIONAL_FIELDS, directional_rows)
    _write_new_csv(AIR_GUARD_OUTPUT, AIR_GUARD_FIELDS, guard_rows)
    return DIRECTIONAL_OUTPUT, AIR_GUARD_OUTPUT


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated directional-refinement and Air-guard diagnostics."
    )
    parser.add_argument("--worker-mode", choices=("probe", "point"), help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=tuple(VARIANT_BY_NAME), help=argparse.SUPPRESS)
    parser.add_argument("--strength", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--guard-factor", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--state-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--bias-name", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--jobs", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.worker_mode:
        if arguments.variant is None or arguments.strength is None or arguments.worker_output is None:
            raise SystemExit("worker mode requires variant, strength, and output")
        try:
            if arguments.worker_mode == "probe":
                result = run_probe_worker(
                    variant_name=arguments.variant,
                    strength=arguments.strength,
                    guard_factor=arguments.guard_factor,
                )
            else:
                if arguments.state_index is None or arguments.bias_name is None:
                    raise ValueError("point worker requires state-index and bias-name")
                result = run_point_worker(
                    variant_name=arguments.variant,
                    strength=arguments.strength,
                    state_index=arguments.state_index,
                    bias_name=arguments.bias_name,
                    guard_factor=arguments.guard_factor,
                )
        except Exception as error:
            result = {
                "mode": arguments.worker_mode,
                "variant": arguments.variant,
                "strength": arguments.strength,
                "guard_factor": arguments.guard_factor,
                "worker_process_id": os.getpid(),
                "error_message": _error(error),
            }
        _write_json(arguments.worker_output, result)
        return 0

    immutable = capture_immutable_inputs()
    with tempfile.TemporaryDirectory(prefix="directional_refinement_") as name:
        result = run_isolated_diagnostics(Path(name), jobs=max(1, arguments.jobs))
    assert_immutable_inputs(immutable)
    outputs = write_outputs(result)
    assert_immutable_inputs(immutable)
    for variant in VARIANTS:
        selection = result["selections"][variant.name]
        print(
            f"{variant.name}: strength={selection['strength']:.12g}, "
            f"coordinates={selection['global_coordinate_count']}, "
            f"active_unknowns={selection['active_unknown_count']}, "
            f"DOF difference={selection['dof_relative_difference']:.3%}, "
            f"comparable={selection['dof_comparable']}"
        )
    for path in outputs:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
