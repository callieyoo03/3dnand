"""Generate isolated full-terminal Q/C *candidate* diagnostics.

This runner intentionally does not publish into :mod:`shared_data`.  It keeps
the established Qg/Cgg/Cgd/Cgs API untouched, evaluates raw Poisson-contact
charges in fresh DEVSIM processes, and only promotes drain/source quantities
when every validation criterion succeeds.  Raw values remain available in the
validation CSV even when promotion is denied.

The public invocation starts three fresh workers (base, an independent base
repeat, and a fine mesh).  The worker mode is private implementation detail;
it temporarily scales only ``add_2d_mesh_line(..., ps=...)`` in the actual
runtime structure module.  Geometry line positions, regions, contacts, state
definitions, material parameters, and solver settings are not changed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import full_terminal_qc_candidate as qc


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
CANDIDATE_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "full_terminal_qc_candidate"
)

PUBLIC_QC_FILES = (
    SHARED_DATA_DIRECTORY / "terminal_charge_by_state.csv",
    SHARED_DATA_DIRECTORY / "capacitance_matrix_by_state.csv",
    SHARED_DATA_DIRECTORY / "capacitance_summary_by_state.csv",
)

OUTPUT_FILENAMES = {
    "terminal": "terminal_charge_full_candidate.csv",
    "matrix": "capacitance_matrix_full_candidate.csv",
    "summary": "capacitance_summary_full_candidate.csv",
    "validation": "direct_contact_flux_validation.csv",
    "ward": "ward_dutton_mobile_partition.csv",
    "comparison": "method_comparison.csv",
    "mesh": "mesh_refinement_qc.csv",
    "readme": "full_terminal_qc_README.md",
}

WORKER_SPECS = (
    ("base", 1.0),
    ("base_repeat", 1.0),
    ("fine", 0.5),
)

REFERENCE_BIASES = (
    ("off", -1.0),
    ("on", 3.0),
)
REFERENCE_VDS_V = 0.05
REFERENCE_VS_V = 0.0
DELTA_VOLTAGES_V = (0.0005, 0.001, 0.002)
NOMINAL_DELTA_VOLTAGE_V = 0.001
GAUGE_SHIFTS_V = (-0.01, 0.01)

CHARGE_FLOOR_C = 1.0e-30
CHARGE_ABSOLUTE_TOLERANCE_C = 1.0e-24
CHARGE_RELATIVE_TOLERANCE = 1.0e-8
PRIMARY_ABSOLUTE_TOLERANCE = 1.0e-28
REPRODUCIBILITY_RELATIVE_TOLERANCE = 1.0e-4
MESH_RELATIVE_TOLERANCE = 5.0e-2
CAPACITANCE_SUM_ABSOLUTE_TOLERANCE_F = 1.0e-22
CAPACITANCE_SUM_RELATIVE_TOLERANCE = 1.0e-4
DELTA_SENSITIVITY_LIMIT = 1.0e-2

EXPECTED_BASE_SPACING_CM = (
    ("x", 1.0e-7),
    ("x", 5.0e-8),
    ("x", 2.0e-8),
    ("x", 5.0e-8),
    ("x", 5.0e-8),
    ("x", 5.0e-8),
    ("x", 5.0e-8),
    ("x", 1.0e-7),
    ("y", 2.0e-7),
    ("y", 2.0e-7),
)

DIRECT_CHARGE_NAMES = (
    "Qg_contact_C",
    "Qd_contact_C",
    "Qs_contact_C",
)
INTERNAL_CHARGE_NAMES = (
    "Qmobile_C",
    "Qtrap_C",
    "Qfixed_C",
)
ALL_CHARGE_NAMES = (*DIRECT_CHARGE_NAMES, *INTERNAL_CHARGE_NAMES)


def sha256_file(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for one existing file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_public_qc_hashes() -> dict[str, str]:
    """Capture the immutable public Q/C bundle before candidate work."""

    missing = [str(path) for path in PUBLIC_QC_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required public Q/C file(s) are missing: " + ", ".join(missing)
        )
    return {path.name: sha256_file(path) for path in PUBLIC_QC_FILES}


def assert_public_qc_hashes_unchanged(expected: Mapping[str, str]) -> None:
    """Fail if any established shared-data Q/C file changed."""

    actual = capture_public_qc_hashes()
    if dict(expected) != actual:
        differences = {
            name: {"before": expected.get(name), "after": actual.get(name)}
            for name in sorted(set(expected) | set(actual))
            if expected.get(name) != actual.get(name)
        }
        raise RuntimeError(
            "Candidate execution changed an established shared_data Q/C file: "
            f"{differences}"
        )


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def normalize_error_message(error: object | None) -> str:
    """Return one deterministic single-line error message."""

    if error is None:
        return ""
    message = " ".join(str(error).split())
    if isinstance(error, BaseException):
        name = type(error).__name__
        return f"{name}: {message}" if message else name
    return message


def values_close(
    left: float,
    right: float,
    *,
    absolute_tolerance: float = PRIMARY_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = REPRODUCIBILITY_RELATIVE_TOLERANCE,
) -> tuple[bool, float, float, float]:
    """Compare two primary Q/C values with an absolute-plus-scaled limit."""

    left_value = _finite_float(left, "left")
    right_value = _finite_float(right, "right")
    scale = max(abs(left_value), abs(right_value))
    tolerance = float(absolute_tolerance) + float(relative_tolerance) * scale
    difference = abs(right_value - left_value)
    return difference <= tolerance, difference, scale, tolerance


def compare_charge_vectors(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    names: Sequence[str] = ALL_CHARGE_NAMES,
    absolute_tolerance: float = PRIMARY_ABSOLUTE_TOLERANCE,
    relative_tolerance: float = REPRODUCIBILITY_RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    """Compare like-for-like charge components without hiding failures."""

    rows: list[dict[str, Any]] = []
    passed = True
    for name in names:
        try:
            within, difference, scale, tolerance = values_close(
                reference[name],
                candidate[name],
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
            )
        except Exception as error:
            within = False
            difference = math.nan
            scale = math.nan
            tolerance = math.nan
            error_message = normalize_error_message(error)
        else:
            error_message = ""
        passed = passed and within
        rows.append(
            {
                "quantity": name,
                "reference": reference.get(name, math.nan),
                "candidate": candidate.get(name, math.nan),
                "absolute_difference": difference,
                "scale": scale,
                "tolerance": tolerance,
                "passed": within,
                "error_message": error_message,
            }
        )
    return {"passed": passed, "comparisons": rows}


@contextmanager
def scaled_runtime_mesh_lines(
    structure_module: Any,
    scale: float,
    line_log: list[dict[str, float | str]],
) -> Iterator[None]:
    """Temporarily scale only runtime mesh ``ps`` values and log every line."""

    scale_value = _finite_float(scale, "mesh scale")
    if scale_value <= 0.0:
        raise ValueError("mesh scale must be positive")
    original = structure_module.add_2d_mesh_line

    def scaled_add_2d_mesh_line(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise RuntimeError(
                "candidate mesh audit requires keyword add_2d_mesh_line calls"
            )
        direction = str(kwargs.get("dir", ""))
        if direction not in {"x", "y"}:
            raise RuntimeError(f"unexpected mesh-line direction {direction!r}")
        position = _finite_float(kwargs.get("pos"), "mesh line position")
        base_spacing = _finite_float(kwargs.get("ps"), "mesh line spacing")
        if base_spacing <= 0.0:
            raise RuntimeError("mesh line spacing must be positive")
        scaled_spacing = base_spacing * scale_value
        line_log.append(
            {
                "direction": direction,
                "position_cm": position,
                "base_spacing_cm": base_spacing,
                "scaled_spacing_cm": scaled_spacing,
                "scale": scale_value,
            }
        )
        forwarded = dict(kwargs)
        forwarded["ps"] = scaled_spacing
        return original(**forwarded)

    structure_module.add_2d_mesh_line = scaled_add_2d_mesh_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = original


def expected_mesh_line_positions(geometry: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    """Return exact runtime geometry boundaries expected in mesh-line calls."""

    return (
        ("x", float(geometry["r_axis"])),
        ("x", float(geometry["r_core"])),
        ("x", float(geometry["r_mos2"])),
        ("x", float(geometry["r_tox"])),
        ("x", float(geometry["r_trap"])),
        ("x", float(geometry["r_block"])),
        ("x", float(geometry["r_gate_outer"])),
        ("x", float(geometry["r_air_outer"])),
        ("y", float(geometry["z_source"])),
        ("y", float(geometry["z_drain"])),
    )


def validate_mesh_line_log(
    line_log: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    scale: float,
) -> dict[str, Any]:
    """Require unchanged line positions and exact base/fine spacing scaling."""

    expected_positions = expected_mesh_line_positions(geometry)
    if len(line_log) != len(expected_positions):
        raise RuntimeError(
            f"mesh line count mismatch: {len(line_log)} != {len(expected_positions)}"
        )
    errors: list[str] = []
    for index, (row, expected_position, expected_spacing) in enumerate(
        zip(line_log, expected_positions, EXPECTED_BASE_SPACING_CM)
    ):
        expected_direction, expected_coordinate = expected_position
        spacing_direction, base_spacing = expected_spacing
        if str(row["direction"]) != expected_direction:
            errors.append(f"line {index} direction")
        if spacing_direction != expected_direction:
            errors.append(f"internal expected spacing direction {index}")
        if not math.isclose(
            float(row["position_cm"]),
            expected_coordinate,
            rel_tol=0.0,
            abs_tol=1.0e-30,
        ):
            errors.append(f"line {index} position")
        if not math.isclose(
            float(row["base_spacing_cm"]),
            base_spacing,
            rel_tol=0.0,
            abs_tol=1.0e-30,
        ):
            errors.append(f"line {index} base spacing")
        if not math.isclose(
            float(row["scaled_spacing_cm"]),
            base_spacing * float(scale),
            rel_tol=0.0,
            abs_tol=1.0e-30,
        ):
            errors.append(f"line {index} scaled spacing")
    if errors:
        raise RuntimeError("mesh-line validation failed: " + ", ".join(errors))
    return {
        "passed": True,
        "line_count": len(line_log),
        "scale": float(scale),
        "lines": [dict(row) for row in line_log],
    }


def build_worker_command(
    python_executable: str,
    *,
    mesh_level: str,
    mesh_scale: float,
    output_path: str | Path,
) -> list[str]:
    """Build one deterministic fresh-process worker command."""

    return [
        str(python_executable),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--mesh-level",
        str(mesh_level),
        "--mesh-scale",
        f"{float(mesh_scale):.17g}",
        "--worker-output",
        str(Path(output_path).resolve()),
    ]


def run_fresh_workers(
    temporary_directory: str | Path,
    *,
    python_executable: str = sys.executable,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> dict[str, dict[str, Any]]:
    """Run base, independent base repeat, and fine workers serially."""

    directory = Path(temporary_directory)
    directory.mkdir(parents=True, exist_ok=True)
    bundles: dict[str, dict[str, Any]] = {}
    for mesh_level, mesh_scale in WORKER_SPECS:
        worker_output = directory / f"{mesh_level}.json"
        command = build_worker_command(
            python_executable,
            mesh_level=mesh_level,
            mesh_scale=mesh_scale,
            output_path=worker_output,
        )
        print("Running fresh candidate worker:", " ".join(command), flush=True)
        completed = subprocess_run(
            command,
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        return_code = int(getattr(completed, "returncode", 0))
        if return_code != 0:
            raise RuntimeError(
                f"candidate worker {mesh_level!r} exited with {return_code}"
            )
        if not worker_output.is_file():
            raise RuntimeError(
                f"candidate worker {mesh_level!r} did not write {worker_output}"
            )
        with worker_output.open("r", encoding="utf-8") as input_file:
            bundle = json.load(input_file)
        if str(bundle.get("mesh_level")) != mesh_level:
            raise RuntimeError(f"worker level mismatch in {worker_output}")
        if not math.isclose(
            float(bundle.get("mesh_scale")),
            mesh_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise RuntimeError(f"worker scale mismatch in {worker_output}")
        bundles[mesh_level] = bundle
    return bundles


def _json_safe(value: Any) -> Any:
    """Convert nested data into strict JSON without non-standard NaN tokens."""

    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_worker_bundle(path: Path, bundle: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            _json_safe(bundle),
            output_file,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        output_file.write("\n")
    temporary.replace(path)


def _read_runtime_biases(device: str) -> dict[str, float]:
    from devsim import get_parameter

    return {
        terminal: float(
            get_parameter(device=device, name=f"{terminal}_bias")
        )
        for terminal in qc.TERMINALS
    }


def _read_runtime_trap_density(device: str) -> float:
    from devsim import get_parameter
    from trap_models import TRAPPED_ELECTRON_PARAMETER

    return float(
        get_parameter(
            device=device,
            region="ChargeTrap",
            name=TRAPPED_ELECTRON_PARAMETER,
        )
    )


def _assert_biases(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
    *,
    excluded_terminal: str | None = None,
) -> None:
    for terminal in qc.TERMINALS:
        if terminal == excluded_terminal:
            continue
        if not math.isclose(
            float(actual[terminal]),
            float(expected[terminal]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"{terminal} bias changed: {actual[terminal]} != {expected[terminal]}"
            )


def _assert_trap_density(device: str, expected_cm3: float) -> None:
    actual = _read_runtime_trap_density(device)
    if abs(actual - float(expected_cm3)) > 1.0:
        raise RuntimeError(
            f"Ntrap changed: {actual:.12e} != {float(expected_cm3):.12e} cm^-3"
        )


def _direct_charge_vector(components: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: _finite_float(components[name], name)
        for name in DIRECT_CHARGE_NAMES
    }


def _ramp_absolute_terminal(
    terminal: str,
    target_voltage_V: float,
    delta_voltage_V: float,
    *,
    device: str,
    label: str,
) -> float:
    from terminal_charge import adaptive_micro_ramp

    return float(
        adaptive_micro_ramp(
            terminal,
            float(target_voltage_V),
            float(delta_voltage_V),
            device=device,
            label=label,
        )
    )


def measure_full_capacitance_column(
    *,
    perturbed_terminal: str,
    base_biases: Mapping[str, float],
    baseline_components: Mapping[str, Any],
    delta_voltage_V: float,
    expected_trap_density_cm3: float,
    device: str,
) -> dict[str, Any]:
    """Measure d(Qg,Qd,Qs)/dVj and prove a full reference restore."""

    terminal = str(perturbed_terminal)
    delta = _finite_float(delta_voltage_V, "delta voltage")
    result: dict[str, Any] = {
        "perturbed_terminal": terminal,
        "delta_voltage_V": delta,
        "capacitances_F": {name: math.nan for name in DIRECT_CHARGE_NAMES},
        "endpoint_charges": {},
        "nonperturbed_biases_held": False,
        "trap_density_held": False,
        "baseline_bias_restored": False,
        "baseline_charge_restored": False,
        "converged": False,
        "error_message": "",
    }
    primary_error: BaseException | None = None
    try:
        _assert_biases(_read_runtime_biases(device), base_biases)
        _assert_trap_density(device, expected_trap_density_cm3)
        endpoint_vectors: dict[str, dict[str, float]] = {}
        for sign_name, sign in (("plus", 1.0), ("minus", -1.0)):
            _ramp_absolute_terminal(
                terminal,
                float(base_biases[terminal]) + sign * delta,
                delta,
                device=device,
                label=f"full-QC {terminal} {sign_name} endpoint",
            )
            actual_biases = _read_runtime_biases(device)
            _assert_biases(
                actual_biases,
                base_biases,
                excluded_terminal=terminal,
            )
            expected_terminal = float(base_biases[terminal]) + sign * delta
            if not math.isclose(
                actual_biases[terminal],
                expected_terminal,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise RuntimeError(
                    f"{terminal} endpoint mismatch at {sign_name}"
                )
            _assert_trap_density(device, expected_trap_density_cm3)
            endpoint = qc.extract_direct_contact_charge_components(device=device)
            endpoint_vectors[sign_name] = _direct_charge_vector(endpoint)
        result["nonperturbed_biases_held"] = True
        result["trap_density_held"] = True
        result["endpoint_charges"] = endpoint_vectors
        result["capacitances_F"] = qc.vector_central_difference(
            endpoint_vectors["plus"],
            endpoint_vectors["minus"],
            delta,
        )
        result["converged"] = True
    except Exception as error:
        primary_error = error
        result["error_message"] = normalize_error_message(error)
    finally:
        try:
            _ramp_absolute_terminal(
                terminal,
                float(base_biases[terminal]),
                delta,
                device=device,
                label=f"full-QC {terminal} baseline restore",
            )
            restored_biases = _read_runtime_biases(device)
            _assert_biases(restored_biases, base_biases)
            result["baseline_bias_restored"] = True
            _assert_trap_density(device, expected_trap_density_cm3)
            restored_components = qc.extract_direct_contact_charge_components(
                device=device
            )
            comparison = compare_charge_vectors(
                baseline_components,
                restored_components,
            )
            result["baseline_charge_restore_comparison"] = comparison
            result["baseline_charge_restored"] = bool(comparison["passed"])
            if not result["baseline_charge_restored"]:
                raise RuntimeError("baseline contact/internal charge did not reproduce")
        except Exception as restore_error:
            result["converged"] = False
            message = "baseline restore failed: " + normalize_error_message(
                restore_error
            )
            if primary_error is None:
                result["error_message"] = message
            else:
                result["error_message"] = (
                    str(result["error_message"]) + "; " + message
                ).strip("; ")
    return result


def _set_all_terminal_biases(
    biases: Mapping[str, float],
    *,
    device: str,
) -> None:
    from devsim import set_parameter

    for terminal in qc.TERMINALS:
        set_parameter(
            device=device,
            name=f"{terminal}_bias",
            value=float(biases[terminal]),
        )


def _solve_dc() -> None:
    from run_memory_window import solve_dc

    solve_dc()


def ramp_common_mode(
    base_biases: Mapping[str, float],
    target_shift_V: float,
    *,
    device: str,
) -> dict[str, float]:
    """Move all absolute terminal voltages together with adaptive steps."""

    target_shift = _finite_float(target_shift_V, "gauge shift")
    current_biases = _read_runtime_biases(device)
    _assert_biases(current_biases, base_biases)
    current_shift = 0.0
    direction = 1.0 if target_shift >= 0.0 else -1.0
    step = min(0.001, abs(target_shift))
    minimum_step = max(abs(target_shift) / 1024.0, 1.0e-6)
    if target_shift == 0.0:
        return current_biases
    while direction * (target_shift - current_shift) > 1.0e-15:
        trial_step = min(step, abs(target_shift - current_shift))
        trial_shift = current_shift + direction * trial_step
        trial_biases = {
            terminal: float(base_biases[terminal]) + trial_shift
            for terminal in qc.TERMINALS
        }
        _set_all_terminal_biases(trial_biases, device=device)
        try:
            _solve_dc()
        except Exception:
            _set_all_terminal_biases(
                {
                    terminal: float(base_biases[terminal]) + current_shift
                    for terminal in qc.TERMINALS
                },
                device=device,
            )
            _solve_dc()
            step *= 0.5
            if step < minimum_step:
                raise
            continue
        current_shift = trial_shift
        step = min(step * 1.5, 0.001)
    actual = _read_runtime_biases(device)
    expected = {
        terminal: float(base_biases[terminal]) + target_shift
        for terminal in qc.TERMINALS
    }
    _assert_biases(actual, expected)
    return actual


def validate_gauge_invariance(
    *,
    base_biases: Mapping[str, float],
    baseline_components: Mapping[str, Any],
    expected_trap_density_cm3: float,
    device: str,
) -> dict[str, Any]:
    """Check two common-mode shifts and a final baseline reproduction."""

    shift_results: list[dict[str, Any]] = []
    errors: list[str] = []
    for shift in GAUGE_SHIFTS_V:
        try:
            ramp_common_mode(base_biases, shift, device=device)
            _assert_trap_density(device, expected_trap_density_cm3)
            shifted = qc.extract_direct_contact_charge_components(device=device)
            comparison = compare_charge_vectors(
                baseline_components,
                shifted,
            )
            shift_results.append(
                {
                    "shift_V": shift,
                    "passed": bool(comparison["passed"]),
                    "comparisons": comparison["comparisons"],
                }
            )
            if not comparison["passed"]:
                errors.append(f"gauge shift {shift:+.6e} V changed charge")
        except Exception as error:
            errors.append(
                f"gauge shift {shift:+.6e} V: {normalize_error_message(error)}"
            )
        finally:
            try:
                current = _read_runtime_biases(device)
                current_shift = current["source"] - float(base_biases["source"])
                shifted_base = {
                    terminal: float(base_biases[terminal]) + current_shift
                    for terminal in qc.TERMINALS
                }
                _assert_biases(current, shifted_base)
                ramp_common_mode(shifted_base, -current_shift, device=device)
                _assert_biases(_read_runtime_biases(device), base_biases)
            except Exception as restore_error:
                errors.append(
                    "gauge baseline restore: "
                    + normalize_error_message(restore_error)
                )
    try:
        restored = qc.extract_direct_contact_charge_components(device=device)
        restore_comparison = compare_charge_vectors(
            baseline_components,
            restored,
        )
        if not restore_comparison["passed"]:
            errors.append("post-gauge baseline charge did not reproduce")
    except Exception as error:
        restore_comparison = {"passed": False, "comparisons": []}
        errors.append(normalize_error_message(error))
    return {
        "passed": not errors,
        "shifts": shift_results,
        "baseline_restore": restore_comparison,
        "error_message": "; ".join(errors),
    }


def _prepare_reference_state(point: Any, state: Mapping[str, Any]) -> None:
    from state_sweep_helpers import ramp_terminal, ramp_trap_density

    _ramp_absolute_terminal(
        "source",
        REFERENCE_VS_V,
        NOMINAL_DELTA_VOLTAGE_V,
        device=qc.DEVICE_NAME,
        label="full-QC source reset",
    )
    ramp_terminal(point, "gate", 0.0, label="full-QC gate reset")
    ramp_terminal(point, "drain", 0.0, label="full-QC drain reset")
    ramp_trap_density(
        point,
        float(state["ntrap_cm3"]),
        label=f"full-QC {state['state']} trap ramp",
    )
    ramp_terminal(
        point,
        "drain",
        REFERENCE_VDS_V,
        label=f"full-QC {state['state']} read drain",
    )


def _matrix_from_measurements(
    measurements: Sequence[Mapping[str, Any]],
    delta_voltage_V: float,
) -> dict[tuple[str, str], float]:
    by_terminal = {
        str(row["perturbed_terminal"]): row
        for row in measurements
        if math.isclose(
            float(row["delta_voltage_V"]),
            float(delta_voltage_V),
            rel_tol=0.0,
            abs_tol=0.0,
        )
    }
    matrix: dict[tuple[str, str], float] = {}
    contact_to_terminal = {
        "Qg_contact_C": "gate",
        "Qd_contact_C": "drain",
        "Qs_contact_C": "source",
    }
    for perturbed in qc.TERMINALS:
        row = by_terminal.get(perturbed)
        for charge_name, measured in contact_to_terminal.items():
            if row is None or not bool(row.get("converged")):
                matrix[(measured, perturbed)] = math.nan
            else:
                value = row["capacitances_F"].get(charge_name)
                matrix[(measured, perturbed)] = (
                    float(value) if value is not None else math.nan
                )
    return matrix


def _run_reference_point(
    *,
    point: Any,
    state: Mapping[str, Any],
    bias_name: str,
    gate_voltage_V: float,
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    from state_sweep_helpers import ramp_terminal

    errors: list[str] = []
    try:
        ramp_terminal(
            point,
            "gate",
            gate_voltage_V,
            label=f"full-QC {state['state']} {bias_name} gate",
        )
        base_biases = _read_runtime_biases(qc.DEVICE_NAME)
        expected_biases = {
            "gate": gate_voltage_V,
            "drain": REFERENCE_VDS_V,
            "source": REFERENCE_VS_V,
        }
        _assert_biases(base_biases, expected_biases)
        _assert_trap_density(qc.DEVICE_NAME, float(state["ntrap_cm3"]))
        direct = qc.extract_direct_contact_charge_components(
            device=qc.DEVICE_NAME,
            configured_charge_floor_C=CHARGE_FLOOR_C,
        )
        residual_passed = (
            abs(float(direct["global_residual_C"]))
            <= CHARGE_ABSOLUTE_TOLERANCE_C
            + CHARGE_RELATIVE_TOLERANCE * float(direct["charge_scale_C"])
        )
        ward = qc.extract_ward_dutton_mobile_partition(
            device=qc.DEVICE_NAME,
            source_position_cm=float(geometry["z_source"]),
            drain_position_cm=float(geometry["z_drain"]),
        )
        measurements = [
            measure_full_capacitance_column(
                perturbed_terminal=terminal,
                base_biases=base_biases,
                baseline_components=direct,
                delta_voltage_V=delta,
                expected_trap_density_cm3=float(state["ntrap_cm3"]),
                device=qc.DEVICE_NAME,
            )
            for delta in DELTA_VOLTAGES_V
            for terminal in qc.TERMINALS
        ]
        for measurement in measurements:
            if not bool(measurement.get("converged")):
                errors.append(
                    "capacitance endpoint failed "
                    f"(terminal={measurement.get('perturbed_terminal')}, "
                    f"delta={_numeric_or_nan(measurement.get('delta_voltage_V')):.12g} V): "
                    + (
                        str(measurement.get("error_message", ""))
                        or "endpoint or baseline-restore validation did not converge"
                    )
                )
        matrices: dict[str, Any] = {}
        for delta in DELTA_VOLTAGES_V:
            matrix = _matrix_from_measurements(measurements, delta)
            matrix_finite = all(math.isfinite(value) for value in matrix.values())
            if matrix_finite:
                sums = qc.calculate_matrix_sums(matrix)
                sum_validation = qc.evaluate_matrix_sum_tolerances(
                    matrix,
                    absolute_tolerance_F=CAPACITANCE_SUM_ABSOLUTE_TOLERANCE_F,
                    relative_tolerance=CAPACITANCE_SUM_RELATIVE_TOLERANCE,
                )
                if not bool(sum_validation["passed"]):
                    errors.append(
                        f"matrix row/column-sum validation failed at delta={delta:.12g} V"
                    )
            else:
                nan_by_terminal = {terminal: math.nan for terminal in qc.TERMINALS}
                sums = {
                    "row_sums_F": dict(nan_by_terminal),
                    "column_sums_F": dict(nan_by_terminal),
                    "row_scales_F": dict(nan_by_terminal),
                    "column_scales_F": dict(nan_by_terminal),
                    "maximum_absolute_row_sum_F": math.nan,
                    "maximum_absolute_column_sum_F": math.nan,
                }
                sum_validation = {
                    **sums,
                    "row_tolerances_F": dict(nan_by_terminal),
                    "column_tolerances_F": dict(nan_by_terminal),
                    "row_pass": {terminal: False for terminal in qc.TERMINALS},
                    "column_pass": {terminal: False for terminal in qc.TERMINALS},
                    "row_sums_passed": False,
                    "column_sums_passed": False,
                    "passed": False,
                }
                errors.append(f"non-finite capacitance matrix at delta={delta:.12g} V")
            matrices[f"{delta:.12g}"] = {
                "values": {
                    f"{measured}:{perturbed}": value
                    for (measured, perturbed), value in matrix.items()
                },
                "sums": sums,
                "sum_validation": sum_validation,
            }
        sensitivities: dict[str, Any] = {}
        for measured in qc.TERMINALS:
            for perturbed in qc.TERMINALS:
                values_by_delta = {
                    delta: _matrix_from_measurements(measurements, delta)[
                        (measured, perturbed)
                    ]
                    for delta in DELTA_VOLTAGES_V
                }
                try:
                    sensitivity_result = qc.calculate_delta_sensitivity(
                        values_by_delta,
                        nominal_delta_voltage_V=NOMINAL_DELTA_VOLTAGE_V,
                    )
                    sensitivity = float(sensitivity_result["relative_sensitivity"])
                    sensitivity_passed = sensitivity <= DELTA_SENSITIVITY_LIMIT
                    if not sensitivity_passed:
                        errors.append(
                            "delta-voltage sensitivity failed "
                            f"for C{measured[0]}{perturbed[0]}: {sensitivity:.12g}"
                        )
                except Exception as error:
                    sensitivity = math.nan
                    sensitivity_passed = False
                    errors.append(normalize_error_message(error))
                sensitivities[f"{measured}:{perturbed}"] = {
                    "value": sensitivity,
                    "passed": sensitivity_passed,
                }
        gauge = validate_gauge_invariance(
            base_biases=base_biases,
            baseline_components=direct,
            expected_trap_density_cm3=float(state["ntrap_cm3"]),
            device=qc.DEVICE_NAME,
        )
        if not bool(gauge["passed"]):
            errors.append(
                "gauge-invariance validation failed: "
                + (str(gauge.get("error_message", "")) or "unspecified gauge failure")
            )
        if not bool(ward.get("converged")):
            errors.append(
                "Ward-Dutton mobile partition failed: "
                + (str(ward.get("error_message", "")) or "unspecified partition failure")
            )
        point_passed = (
            residual_passed
            and all(bool(row["converged"]) for row in measurements)
            and all(
                bool(data["sum_validation"]["passed"])
                for data in matrices.values()
            )
            and all(bool(data["passed"]) for data in sensitivities.values())
            and bool(gauge["passed"])
            and bool(ward["converged"])
        )
    except Exception as error:
        errors.append(normalize_error_message(error))
        base_biases = {
            "gate": gate_voltage_V,
            "drain": REFERENCE_VDS_V,
            "source": REFERENCE_VS_V,
        }
        direct = {name: math.nan for name in ALL_CHARGE_NAMES}
        direct.update(
            {
                "global_residual_C": math.nan,
                "absolute_residual_C": math.nan,
                "charge_scale_C": math.nan,
                "relative_residual": math.nan,
            }
        )
        ward = {
            "Qs_mobile_C": math.nan,
            "Qd_mobile_C": math.nan,
            "Qmobile_channel_C": math.nan,
            "partition_residual_C": math.nan,
            "relative_partition_residual": math.nan,
            "converged": False,
            "error_message": errors[-1],
        }
        measurements = []
        matrices = {}
        sensitivities = {}
        gauge = {"passed": False, "error_message": errors[-1]}
        residual_passed = False
        point_passed = False
    return {
        "state_index": int(state["state_index"]),
        "state": str(state["state"]),
        "ntrap_cm3": float(state["ntrap_cm3"]),
        "bias_name": bias_name,
        "VGS_V": float(gate_voltage_V - base_biases["source"]),
        "VDS_V": float(base_biases["drain"] - base_biases["source"]),
        "VS_V": float(base_biases["source"]),
        "absolute_biases": base_biases,
        "direct": direct,
        "residual_passed": residual_passed,
        "ward": ward,
        "measurements": measurements,
        "matrices": matrices,
        "sensitivities": sensitivities,
        "gauge": gauge,
        "passed": point_passed,
        "error_message": "; ".join(errors),
    }


def _runtime_mesh_counts(device: str) -> dict[str, Any]:
    from devsim import get_element_node_list, get_node_model_values, get_region_list

    regions: dict[str, dict[str, int]] = {}
    total_elements = 0
    total_region_nodes = 0
    for region in sorted(get_region_list(device=device)):
        node_count = len(
            get_node_model_values(device=device, region=region, name="x")
        )
        element_count = len(
            get_element_node_list(device=device, region=region)
        )
        regions[region] = {
            "node_count": node_count,
            "element_count": element_count,
        }
        total_region_nodes += node_count
        total_elements += element_count
    return {
        "regions": regions,
        "total_region_node_count": total_region_nodes,
        "total_element_count": total_elements,
    }


def run_worker(mesh_level: str, mesh_scale: float) -> dict[str, Any]:
    """Build one mesh and evaluate every candidate diagnostic in this process."""

    import run_memory_window as memory_window
    import state_characterization_config as config
    from state_sweep_helpers import initialize_characterization_device

    line_log: list[dict[str, float | str]] = []
    with scaled_runtime_mesh_lines(
        memory_window.parameterized_device_structure,
        mesh_scale,
        line_log,
    ):
        geometry, point = initialize_characterization_device()
    mesh_line_validation = validate_mesh_line_log(
        line_log,
        geometry,
        mesh_scale,
    )
    topology = qc.audit_runtime_contact_topology(
        device=qc.DEVICE_NAME,
        geometry=geometry,
    )
    points: list[dict[str, Any]] = []
    errors: list[str] = []
    for state in config.MEMORY_STATES:
        try:
            _prepare_reference_state(point, state)
        except Exception as error:
            setup_error = normalize_error_message(error)
        else:
            setup_error = ""
        for bias_name, gate_voltage in REFERENCE_BIASES:
            if setup_error:
                points.append(
                    {
                        "state_index": int(state["state_index"]),
                        "state": str(state["state"]),
                        "ntrap_cm3": float(state["ntrap_cm3"]),
                        "bias_name": bias_name,
                        "VGS_V": gate_voltage,
                        "VDS_V": REFERENCE_VDS_V,
                        "VS_V": REFERENCE_VS_V,
                        "passed": False,
                        "error_message": setup_error,
                    }
                )
                continue
            row = _run_reference_point(
                point=point,
                state=state,
                bias_name=bias_name,
                gate_voltage_V=gate_voltage,
                geometry=geometry,
            )
            points.append(row)
            if row["error_message"]:
                errors.append(str(row["error_message"]))
    return {
        "mesh_level": str(mesh_level),
        "mesh_scale": float(mesh_scale),
        "geometry": dict(geometry),
        "mesh_line_validation": mesh_line_validation,
        "mesh_counts": _runtime_mesh_counts(qc.DEVICE_NAME),
        "topology": topology,
        "points": sorted(
            points,
            key=lambda row: (int(row["state_index"]), float(row["VGS_V"])),
        ),
        "errors": errors,
    }


def _point_key(row: Mapping[str, Any]) -> tuple[int, float, float, float]:
    return (
        int(row["state_index"]),
        round(float(row["VGS_V"]), 12),
        round(float(row["VDS_V"]), 12),
        round(float(row["VS_V"]), 12),
    )


def _points_by_key(bundle: Mapping[str, Any]) -> dict[tuple[int, float, float, float], Mapping[str, Any]]:
    return {_point_key(row): row for row in bundle.get("points", [])}


def _matrix_value(
    point: Mapping[str, Any],
    measured: str,
    perturbed: str,
    delta: float,
) -> float:
    try:
        data = point["matrices"][f"{float(delta):.12g}"]["values"]
        value = data[f"{measured}:{perturbed}"]
        return float(value) if value is not None else math.nan
    except (KeyError, TypeError, ValueError, OverflowError):
        return math.nan


def _numeric_or_nan(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def evaluate_discrete_continuity(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Require finite state/bias secant slopes without fitting a jump limit.

    Ten isolated reference points cannot mathematically prove continuity.  The
    defensible discrete audit is therefore limited to requiring finite charges
    and finite secant slopes versus Ntrap at each bias and versus VGS for each
    state.  Magnitude thresholds are deliberately not fitted after the fact.
    """

    points = list(bundle.get("points", []))
    errors: list[str] = []
    slopes: list[dict[str, Any]] = []
    by_bias: dict[tuple[float, float, float], list[Mapping[str, Any]]] = {}
    by_state: dict[int, list[Mapping[str, Any]]] = {}
    for point in points:
        try:
            key = (
                round(float(point["VGS_V"]), 12),
                round(float(point["VDS_V"]), 12),
                round(float(point["VS_V"]), 12),
            )
            by_bias.setdefault(key, []).append(point)
            by_state.setdefault(int(point["state_index"]), []).append(point)
        except Exception as error:
            errors.append(normalize_error_message(error))

    for bias_key, rows in sorted(by_bias.items()):
        ordered = sorted(rows, key=lambda row: float(row.get("ntrap_cm3", math.nan)))
        if len(ordered) != 5:
            errors.append(f"bias {bias_key} does not contain five states")
            continue
        for left, right in zip(ordered, ordered[1:]):
            delta_coordinate = _numeric_or_nan(right.get("ntrap_cm3")) - _numeric_or_nan(
                left.get("ntrap_cm3")
            )
            if not math.isfinite(delta_coordinate) or delta_coordinate <= 0.0:
                errors.append(f"non-increasing Ntrap at bias {bias_key}")
                continue
            for quantity in ALL_CHARGE_NAMES:
                left_value = _numeric_or_nan(left.get("direct", {}).get(quantity))
                right_value = _numeric_or_nan(right.get("direct", {}).get(quantity))
                slope = (right_value - left_value) / delta_coordinate
                finite = math.isfinite(slope)
                if not finite:
                    errors.append(f"non-finite {quantity}/Ntrap slope at {bias_key}")
                slopes.append(
                    {
                        "axis": "ntrap_cm3",
                        "quantity": quantity,
                        "coordinate": bias_key,
                        "slope": slope,
                        "finite": finite,
                    }
                )

    for state_index, rows in sorted(by_state.items()):
        ordered = sorted(rows, key=lambda row: float(row.get("VGS_V", math.nan)))
        if len(ordered) != 2:
            errors.append(f"state {state_index} does not contain off/on points")
            continue
        left, right = ordered
        delta_coordinate = _numeric_or_nan(right.get("VGS_V")) - _numeric_or_nan(
            left.get("VGS_V")
        )
        if not math.isfinite(delta_coordinate) or delta_coordinate <= 0.0:
            errors.append(f"invalid VGS secant for state {state_index}")
            continue
        for quantity in ALL_CHARGE_NAMES:
            left_value = _numeric_or_nan(left.get("direct", {}).get(quantity))
            right_value = _numeric_or_nan(right.get("direct", {}).get(quantity))
            slope = (right_value - left_value) / delta_coordinate
            finite = math.isfinite(slope)
            if not finite:
                errors.append(f"non-finite {quantity}/VGS slope at state {state_index}")
            slopes.append(
                {
                    "axis": "VGS_V",
                    "quantity": quantity,
                    "coordinate": state_index,
                    "slope": slope,
                    "finite": finite,
                }
            )
    return {
        "passed": len(points) == 10 and not errors,
        "slopes": slopes,
        "limitation": (
            "Finite secants on the sampled state/bias grid are a discrete "
            "continuity diagnostic, not a proof between sample points."
        ),
        "error_message": "; ".join(errors),
    }


def compare_worker_points(
    reference_bundle: Mapping[str, Any],
    candidate_bundle: Mapping[str, Any],
    *,
    comparison_name: str,
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare primary raw charges and all nominal matrix entries by key."""

    left = _points_by_key(reference_bundle)
    right = _points_by_key(candidate_bundle)
    rows: list[dict[str, Any]] = []
    passed = set(left) == set(right) and len(left) == 10
    for key in sorted(set(left) | set(right)):
        left_point = left.get(key)
        right_point = right.get(key)
        quantities: list[tuple[str, float, float]] = []
        if left_point is not None and right_point is not None:
            for name in (*ALL_CHARGE_NAMES, "global_residual_C"):
                quantities.append(
                    (
                        name,
                        _numeric_or_nan(left_point.get("direct", {}).get(name)),
                        _numeric_or_nan(right_point.get("direct", {}).get(name)),
                    )
                )
            for measured in qc.TERMINALS:
                for perturbed in qc.TERMINALS:
                    name = f"C{measured[0]}{perturbed[0]}_F"
                    quantities.append(
                        (
                            name,
                            _matrix_value(
                                left_point,
                                measured,
                                perturbed,
                                NOMINAL_DELTA_VOLTAGE_V,
                            ),
                            _matrix_value(
                                right_point,
                                measured,
                                perturbed,
                                NOMINAL_DELTA_VOLTAGE_V,
                            ),
                        )
                    )
            for axis_name, data_key in (
                ("row", "row_sums_F"),
                ("column", "column_sums_F"),
            ):
                left_diagnostics = left_point.get("matrices", {}).get(
                    f"{NOMINAL_DELTA_VOLTAGE_V:.12g}", {}
                ).get("sums", {})
                right_diagnostics = right_point.get("matrices", {}).get(
                    f"{NOMINAL_DELTA_VOLTAGE_V:.12g}", {}
                ).get("sums", {})
                for terminal in qc.TERMINALS:
                    quantities.append(
                        (
                            f"{axis_name}_sum_{terminal}_F",
                            _numeric_or_nan(
                                left_diagnostics.get(data_key, {}).get(terminal)
                            ),
                            _numeric_or_nan(
                                right_diagnostics.get(data_key, {}).get(terminal)
                            ),
                        )
                    )
        if not quantities:
            passed = False
            rows.append(
                {
                    "comparison": comparison_name,
                    "state_index": key[0],
                    "VGS_V": key[1],
                    "VDS_V": key[2],
                    "VS_V": key[3],
                    "quantity": "missing_point",
                    "reference_value": math.nan,
                    "candidate_value": math.nan,
                    "absolute_difference": math.nan,
                    "relative_difference": math.nan,
                    "tolerance": math.nan,
                    "passed": False,
                }
            )
            continue
        for name, left_value, right_value in quantities:
            try:
                within, difference, scale, tolerance = values_close(
                    left_value,
                    right_value,
                    absolute_tolerance=PRIMARY_ABSOLUTE_TOLERANCE,
                    relative_tolerance=relative_tolerance,
                )
                relative_difference = difference / max(
                    scale, PRIMARY_ABSOLUTE_TOLERANCE
                )
            except Exception:
                within = False
                difference = math.nan
                relative_difference = math.nan
                tolerance = math.nan
            passed = passed and within
            rows.append(
                {
                    "comparison": comparison_name,
                    "state_index": key[0],
                    "VGS_V": key[1],
                    "VDS_V": key[2],
                    "VS_V": key[3],
                    "quantity": name,
                    "reference_value": left_value,
                    "candidate_value": right_value,
                    "absolute_difference": difference,
                    "relative_difference": relative_difference,
                    "tolerance": tolerance,
                    "passed": within,
                }
            )
    return {"passed": passed, "rows": rows}


def comparison_rows_for_point(
    comparison: Mapping[str, Any],
    point: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    """Return only repeat/mesh comparison rows for one reference point."""

    key = _point_key(point)
    return [
        row
        for row in comparison.get("rows", ())
        if _point_key(row) == key
    ]


def comparison_passed_for_point(
    comparison: Mapping[str, Any],
    point: Mapping[str, Any],
) -> bool:
    rows = comparison_rows_for_point(comparison, point)
    return bool(rows) and all(bool(row.get("passed")) for row in rows)


def comparison_failure_summary_for_point(
    label: str,
    comparison: Mapping[str, Any],
    point: Mapping[str, Any],
) -> str:
    rows = comparison_rows_for_point(comparison, point)
    failed = [row for row in rows if not bool(row.get("passed"))]
    if not failed:
        return ""
    nonfinite_quantities = sorted(
        str(row.get("quantity", ""))
        for row in failed
        if not math.isfinite(_numeric_or_nan(row.get("reference_value")))
        or not math.isfinite(_numeric_or_nan(row.get("candidate_value")))
    )
    finite_relative = [
        _numeric_or_nan(row.get("relative_difference"))
        for row in failed
        if math.isfinite(_numeric_or_nan(row.get("relative_difference")))
    ]
    details = [f"{label} failed {len(failed)}/{len(rows)} comparison rows"]
    if nonfinite_quantities:
        details.append("non-finite=" + ",".join(nonfinite_quantities))
    if finite_relative:
        details.append(f"max_scaled_relative={max(finite_relative):.12e}")
    return " (".join((details[0], "; ".join(details[1:]) + ")")) if len(details) > 1 else details[0]


def determine_aggregate_acceptance(
    bundles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Evaluate every direct-contact promotion gate deterministically."""

    required = {name for name, _ in WORKER_SPECS}
    if set(bundles) != required:
        raise RuntimeError(
            f"worker bundle set mismatch: {sorted(bundles)} != {sorted(required)}"
        )
    base = bundles["base"]
    repeat = bundles["base_repeat"]
    fine = bundles["fine"]
    reproducibility = compare_worker_points(
        base,
        repeat,
        comparison_name="repeated_run",
        relative_tolerance=REPRODUCIBILITY_RELATIVE_TOLERANCE,
    )
    mesh = compare_worker_points(
        base,
        fine,
        comparison_name="base_vs_fine",
        relative_tolerance=MESH_RELATIVE_TOLERANCE,
    )
    base_points = list(base.get("points", []))
    continuity = evaluate_discrete_continuity(base)
    base_counts = base.get("mesh_counts", {})
    repeat_counts = repeat.get("mesh_counts", {})
    fine_counts = fine.get("mesh_counts", {})
    mesh_structure_refined = (
        base_counts == repeat_counts
        and _numeric_or_nan(fine_counts.get("total_element_count"))
        > _numeric_or_nan(base_counts.get("total_element_count"))
        and _numeric_or_nan(fine_counts.get("total_region_node_count"))
        > _numeric_or_nan(base_counts.get("total_region_node_count"))
    )
    geometry_fixed = all(
        bundle.get("geometry") == base.get("geometry") for bundle in bundles.values()
    )
    mesh_lines_passed = all(
        bool(bundle.get("mesh_line_validation", {}).get("passed"))
        for bundle in bundles.values()
    )
    topology_passed = all(
        bool(bundle.get("topology", {}).get("passed"))
        for bundle in bundles.values()
    )
    criteria = {
        "topology_and_contact_sign": topology_passed,
        "all_10_reference_points": len(base_points) == 10,
        "all_contact_and_internal_charges_finite": all(
            all(
                value is not None and math.isfinite(float(value))
                for name, value in point.get("direct", {}).items()
                if name in ALL_CHARGE_NAMES
            )
            and all(name in point.get("direct", {}) for name in ALL_CHARGE_NAMES)
            for point in base_points
        ),
        "global_gauss_residual": all(
            bool(point.get("residual_passed")) for point in base_points
        ),
        "baseline_restore": all(
            all(
                bool(measurement.get("baseline_bias_restored"))
                and bool(measurement.get("baseline_charge_restored"))
                and bool(measurement.get("nonperturbed_biases_held"))
                and bool(measurement.get("trap_density_held"))
                for measurement in point.get("measurements", [])
            )
            and len(point.get("measurements", [])) == 9
            for point in base_points
        ),
        "gauge_invariance": all(
            bool(point.get("gauge", {}).get("passed")) for point in base_points
        ),
        "state_bias_discrete_continuity": bool(continuity["passed"]),
        "matrix_finite": all(
            all(
                value is not None and math.isfinite(float(value))
                for matrix_data in point.get("matrices", {}).values()
                for value in matrix_data.get("values", {}).values()
            )
            and len(point.get("matrices", {})) == 3
            for point in base_points
        ),
        "matrix_row_column_sums": all(
            all(
                bool(matrix_data.get("sum_validation", {}).get("passed"))
                for matrix_data in point.get("matrices", {}).values()
            )
            and len(point.get("matrices", {})) == 3
            for point in base_points
        ),
        "delta_voltage_sensitivity": all(
            len(point.get("sensitivities", {})) == 9
            and all(
                bool(item.get("passed"))
                for item in point.get("sensitivities", {}).values()
            )
            for point in base_points
        ),
        "repeated_run_reproducibility": bool(reproducibility["passed"]),
        "geometry_fixed_across_meshes": geometry_fixed,
        "mesh_line_positions_fixed": mesh_lines_passed,
        "fine_mesh_is_refined": mesh_structure_refined,
        "mesh_refinement": bool(mesh["passed"]) and mesh_structure_refined,
    }
    status = qc.determine_terminal_charge_status(criteria)
    return {
        "status": status,
        "supported": status == qc.TERMINAL_STATUS_SUPPORTED,
        "criteria": criteria,
        "reproducibility": reproducibility,
        "mesh": mesh,
        "continuity": continuity,
    }


def _state_from_point(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "state_index": int(point["state_index"]),
        "state": str(point["state"]),
        "ntrap_cm3": _numeric_or_nan(point.get("ntrap_cm3")),
    }


def _state_sheet_density(state: Mapping[str, Any]) -> float:
    import state_characterization_config as config

    return float(state["ntrap_cm3"]) * float(config.CHARGE_TRAP_THICKNESS_CM)


def _promotion_error(acceptance: Mapping[str, Any]) -> str:
    failed = [
        name for name, passed in acceptance["criteria"].items() if not bool(passed)
    ]
    return (
        "direct-contact promotion failed: " + ",".join(failed)
        if failed
        else ""
    )


def _failed_terminal_row(
    point: Mapping[str, Any],
    *,
    status: str,
    mesh_level: str,
    error_message: str,
) -> dict[str, Any]:
    state = _state_from_point(point)
    row = {name: math.nan for name in qc.TERMINAL_CHARGE_FULL_FIELDNAMES}
    row.update(
        {
            "state_index": state["state_index"],
            "state": state["state"],
            "ntrap_cm3": state["ntrap_cm3"],
            "nsheet_cm2": _state_sheet_density(state),
            "VGS_V": _numeric_or_nan(point.get("VGS_V")),
            "VDS_V": _numeric_or_nan(point.get("VDS_V")),
            "VS_V": _numeric_or_nan(point.get("VS_V")),
            "method": qc.DIRECT_CONTACT_METHOD,
            "supported_quantities": "",
            "terminal_charge_status": status,
            "limitation": qc.DIRECT_CONTACT_LIMITATION,
            "mesh_level": mesh_level,
            "converged": False,
            "error_message": error_message,
        }
    )
    return row


def _matrix_measurement(
    point: Mapping[str, Any],
    perturbed: str,
    delta: float,
) -> Mapping[str, Any] | None:
    for measurement in point.get("measurements", []):
        if (
            str(measurement.get("perturbed_terminal")) == perturbed
            and math.isclose(
                _numeric_or_nan(measurement.get("delta_voltage_V")),
                float(delta),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            return measurement
    return None


def _failed_matrix_row(
    point: Mapping[str, Any],
    *,
    measured: str,
    perturbed: str,
    delta: float,
    status: str,
    mesh_level: str,
    error_message: str,
) -> dict[str, Any]:
    state = _state_from_point(point)
    supported = status == qc.TERMINAL_STATUS_SUPPORTED or measured == "gate"
    row = {name: math.nan for name in qc.CAPACITANCE_MATRIX_FULL_FIELDNAMES}
    row.update(
        {
            "state_index": state["state_index"],
            "state": state["state"],
            "ntrap_cm3": state["ntrap_cm3"],
            "nsheet_cm2": _state_sheet_density(state),
            "VGS_V": _numeric_or_nan(point.get("VGS_V")),
            "VDS_V": _numeric_or_nan(point.get("VDS_V")),
            "VS_V": _numeric_or_nan(point.get("VS_V")),
            "measured_terminal": measured,
            "perturbed_terminal": perturbed,
            "capacitance_F": math.nan,
            "delta_voltage_V": float(delta),
            "method": qc.DIRECT_CONTACT_METHOD,
            "supported": supported,
            "terminal_charge_status": status,
            "limitation": qc.DIRECT_CONTACT_LIMITATION,
            "mesh_level": mesh_level,
            "converged": False,
            "error_message": error_message,
        }
    )
    return row


def _build_summary_row(
    point: Mapping[str, Any],
    *,
    status: str,
    promotion_error: str,
) -> dict[str, Any]:
    state = _state_from_point(point)
    nominal = point.get("matrices", {}).get(
        f"{NOMINAL_DELTA_VOLTAGE_V:.12g}", {}
    )
    raw_values = nominal.get("values", {})
    names = {
        ("gate", "gate"): "Cgg_F",
        ("gate", "drain"): "Cgd_F",
        ("gate", "source"): "Cgs_F",
        ("drain", "gate"): "Cdg_F",
        ("drain", "drain"): "Cdd_F",
        ("drain", "source"): "Cds_F",
        ("source", "gate"): "Csg_F",
        ("source", "drain"): "Csd_F",
        ("source", "source"): "Css_F",
    }
    promoted: dict[str, float] = {}
    for (measured, perturbed), field in names.items():
        raw = _numeric_or_nan(raw_values.get(f"{measured}:{perturbed}"))
        promoted[field] = (
            raw
            if status == qc.TERMINAL_STATUS_SUPPORTED or measured == "gate"
            else math.nan
        )
    sensitivities = [
        _numeric_or_nan(item.get("value"))
        for item in point.get("sensitivities", {}).values()
    ]
    maximum_sensitivity = (
        max(sensitivities)
        if len(sensitivities) == 9 and all(math.isfinite(v) for v in sensitivities)
        else math.nan
    )
    sums = nominal.get("sums", {})
    validation = nominal.get("sum_validation", {})
    converged = (
        len(point.get("measurements", [])) == 9
        and all(bool(row.get("converged")) for row in point.get("measurements", []))
    )
    row = {
        "state_index": state["state_index"],
        "state": state["state"],
        "ntrap_cm3": state["ntrap_cm3"],
        "nsheet_cm2": _state_sheet_density(state),
        "VGS_V": _numeric_or_nan(point.get("VGS_V")),
        "VDS_V": _numeric_or_nan(point.get("VDS_V")),
        "VS_V": _numeric_or_nan(point.get("VS_V")),
        **promoted,
        "nominal_delta_voltage_V": NOMINAL_DELTA_VOLTAGE_V,
        "sensitivity_delta_voltages_V": ",".join(
            f"{value:.12g}" for value in DELTA_VOLTAGES_V
        ),
        "maximum_delta_relative_sensitivity": maximum_sensitivity,
        "maximum_absolute_row_sum_F": _numeric_or_nan(
            sums.get("maximum_absolute_row_sum_F")
        ),
        "maximum_absolute_column_sum_F": _numeric_or_nan(
            sums.get("maximum_absolute_column_sum_F")
        ),
        "row_sums_passed": bool(validation.get("row_sums_passed")),
        "column_sums_passed": bool(validation.get("column_sums_passed")),
        "method": qc.DIRECT_CONTACT_METHOD,
        "terminal_charge_status": status,
        "limitation": "" if status == qc.TERMINAL_STATUS_SUPPORTED else qc.DIRECT_CONTACT_LIMITATION,
        "mesh_level": "base",
        "converged": converged,
        "error_message": "; ".join(
            value
            for value in (str(point.get("error_message", "")), promotion_error)
            if value
        ),
    }
    if tuple(row) != qc.CAPACITANCE_SUMMARY_FULL_FIELDNAMES:
        raise RuntimeError("candidate summary row schema mismatch")
    return row


def build_candidate_output_rows(
    bundles: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Build the seven strict candidate CSV tables from worker evidence."""

    status = str(acceptance["status"])
    promotion_error = _promotion_error(acceptance)
    base_points = list(bundles["base"].get("points", []))
    terminal_rows: list[dict[str, Any]] = []
    matrix_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    ward_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []

    for point in base_points:
        state = _state_from_point(point)
        direct = point.get("direct", {})
        direct_finite = all(
            math.isfinite(_numeric_or_nan(direct.get(name)))
            for name in ALL_CHARGE_NAMES
        )
        row_error = "; ".join(
            value
            for value in (str(point.get("error_message", "")), promotion_error)
            if value
        )
        if direct_finite:
            terminal_rows.append(
                qc.build_terminal_charge_full_row(
                    state,
                    VGS_V=float(point["VGS_V"]),
                    VDS_V=float(point["VDS_V"]),
                    VS_V=float(point["VS_V"]),
                    components=direct,
                    terminal_charge_status=status,
                    mesh_level="base",
                    converged=True,
                    error_message=row_error,
                )
            )
        else:
            terminal_rows.append(
                _failed_terminal_row(
                    point,
                    status=status,
                    mesh_level="base",
                    error_message=row_error or "non-finite direct charge",
                )
            )

        for delta in DELTA_VOLTAGES_V:
            for perturbed in qc.TERMINALS:
                measurement = _matrix_measurement(point, perturbed, delta)
                for measured, contact_name in (
                    ("gate", "Qg_contact_C"),
                    ("drain", "Qd_contact_C"),
                    ("source", "Qs_contact_C"),
                ):
                    raw = (
                        _numeric_or_nan(
                            measurement.get("capacitances_F", {}).get(contact_name)
                        )
                        if measurement is not None
                        else math.nan
                    )
                    if measurement is not None and math.isfinite(raw):
                        matrix_rows.append(
                            qc.build_capacitance_matrix_full_row(
                                state,
                                VGS_V=float(point["VGS_V"]),
                                VDS_V=float(point["VDS_V"]),
                                VS_V=float(point["VS_V"]),
                                measured_terminal=measured,
                                perturbed_terminal=perturbed,
                                capacitance_F=raw,
                                delta_voltage_V=delta,
                                terminal_charge_status=status,
                                mesh_level="base",
                                converged=bool(measurement.get("converged")),
                                error_message=(
                                    str(measurement.get("error_message", ""))
                                    or (promotion_error if measured != "gate" else "")
                                ),
                            )
                        )
                    else:
                        matrix_rows.append(
                            _failed_matrix_row(
                                point,
                                measured=measured,
                                perturbed=perturbed,
                                delta=delta,
                                status=status,
                                mesh_level="base",
                                error_message=(
                                    str(measurement.get("error_message", ""))
                                    if measurement is not None
                                    else "missing capacitance endpoint result"
                                ),
                            )
                        )
        summary_rows.append(
            _build_summary_row(
                point,
                status=status,
                promotion_error=promotion_error,
            )
        )

        ward = point.get("ward", {})
        ward_rows.append(
            {
                "state_index": state["state_index"],
                "state": state["state"],
                "VGS_V": _numeric_or_nan(point.get("VGS_V")),
                "VDS_V": _numeric_or_nan(point.get("VDS_V")),
                "VS_V": _numeric_or_nan(point.get("VS_V")),
                "Qmobile_channel_C": _numeric_or_nan(ward.get("Qmobile_channel_C")),
                "Qs_mobile_C": _numeric_or_nan(ward.get("Qs_mobile_C")),
                "Qd_mobile_C": _numeric_or_nan(ward.get("Qd_mobile_C")),
                "partition_residual_C": _numeric_or_nan(ward.get("partition_residual_C")),
                "absolute_partition_residual_C": _numeric_or_nan(ward.get("absolute_partition_residual_C")),
                "relative_partition_residual": _numeric_or_nan(ward.get("relative_partition_residual")),
                "minimum_source_weight": _numeric_or_nan(ward.get("minimum_source_weight")),
                "maximum_source_weight": _numeric_or_nan(ward.get("maximum_source_weight")),
                "minimum_drain_weight": _numeric_or_nan(ward.get("minimum_drain_weight")),
                "maximum_drain_weight": _numeric_or_nan(ward.get("maximum_drain_weight")),
                "method": qc.WARD_DUTTON_METHOD,
                "terminal_charge_status": qc.WARD_DUTTON_STATUS_MOBILE_ONLY,
                "limitation": qc.WARD_DUTTON_LIMITATION,
                "mesh_level": "base",
                "converged": bool(ward.get("converged")),
                "error_message": str(ward.get("error_message", "")),
            }
        )
        comparison_rows.append(
            {
                "state_index": state["state_index"],
                "state": state["state"],
                "VGS_V": _numeric_or_nan(point.get("VGS_V")),
                "VDS_V": _numeric_or_nan(point.get("VDS_V")),
                "VS_V": _numeric_or_nan(point.get("VS_V")),
                "Qg_direct_contact_C": _numeric_or_nan(direct.get("Qg_contact_C")),
                "Qd_direct_contact_C": _numeric_or_nan(direct.get("Qd_contact_C")),
                "Qs_direct_contact_C": _numeric_or_nan(direct.get("Qs_contact_C")),
                "Qmobile_channel_C": _numeric_or_nan(ward.get("Qmobile_channel_C")),
                "Qd_ward_dutton_mobile_C": _numeric_or_nan(ward.get("Qd_mobile_C")),
                "Qs_ward_dutton_mobile_C": _numeric_or_nan(ward.get("Qs_mobile_C")),
                "direct_method": qc.DIRECT_CONTACT_METHOD,
                "ward_dutton_method": qc.WARD_DUTTON_METHOD,
                "comparison_scope": (
                    "not_same_quantity: full_electrostatic_electrode_candidate "
                    "versus mobile_channel_partition_only; no difference or ratio"
                ),
                "mesh_level": "base",
            }
        )

    continuity_passed = bool(acceptance["criteria"]["state_bias_discrete_continuity"])
    for mesh_level, bundle in bundles.items():
        topology = bundle.get("topology", {})
        for point in bundle.get("points", []):
            direct = point.get("direct", {})
            matrices = point.get("matrices", {})
            repeat_point_passed = comparison_passed_for_point(
                acceptance["reproducibility"], point
            )
            mesh_point_passed = comparison_passed_for_point(
                acceptance["mesh"], point
            )
            comparison_errors = (
                comparison_failure_summary_for_point(
                    "repeated run", acceptance["reproducibility"], point
                ),
                comparison_failure_summary_for_point(
                    "base/fine mesh", acceptance["mesh"], point
                ),
            )
            validation_rows.append(
                {
                    "state_index": int(point["state_index"]),
                    "state": str(point["state"]),
                    "VGS_V": _numeric_or_nan(point.get("VGS_V")),
                    "VDS_V": _numeric_or_nan(point.get("VDS_V")),
                    "VS_V": _numeric_or_nan(point.get("VS_V")),
                    "mesh_level": mesh_level,
                    "all_quantities_finite": all(
                        math.isfinite(_numeric_or_nan(direct.get(name)))
                        for name in ALL_CHARGE_NAMES
                    ),
                    "contacts_complete": bool(topology.get("contacts_complete")),
                    "contact_topology_passed": bool(topology.get("passed")),
                    "contact_sign_passed": bool(topology.get("sign_consistent")),
                    "global_gauss_passed": bool(point.get("residual_passed")),
                    "repeatability_passed": repeat_point_passed,
                    "mesh_refinement_passed": mesh_point_passed,
                    "continuity_passed": continuity_passed,
                    "gauge_invariance_passed": bool(point.get("gauge", {}).get("passed")),
                    "matrix_row_sums_passed": len(matrices) == 3 and all(
                        bool(value.get("sum_validation", {}).get("row_sums_passed"))
                        for value in matrices.values()
                    ),
                    "matrix_column_sums_passed": len(matrices) == 3 and all(
                        bool(value.get("sum_validation", {}).get("column_sums_passed"))
                        for value in matrices.values()
                    ),
                    "global_residual_C": _numeric_or_nan(direct.get("global_residual_C")),
                    "absolute_residual_C": _numeric_or_nan(direct.get("absolute_residual_C")),
                    "charge_scale_C": _numeric_or_nan(direct.get("charge_scale_C")),
                    "relative_residual": _numeric_or_nan(direct.get("relative_residual")),
                    "terminal_charge_status": status,
                    "limitation": (
                        qc.DIRECT_CONTACT_LIMITATION
                        + " "
                        + qc.DIRECT_CONTACT_COLUMN_SUM_LIMITATION
                    ),
                    "error_message": "; ".join(
                        value
                        for value in (
                            str(point.get("error_message", "")),
                            *comparison_errors,
                            promotion_error,
                        )
                        if value
                    ),
                }
            )

    state_names = {
        int(point["state_index"]): str(point["state"]) for point in base_points
    }
    comparison_specs = (
        (
            "repeated_run",
            "base",
            "base_repeat",
            acceptance["reproducibility"],
            REPRODUCIBILITY_RELATIVE_TOLERANCE,
        ),
        (
            "base_vs_fine",
            "base",
            "fine",
            acceptance["mesh"],
            MESH_RELATIVE_TOLERANCE,
        ),
    )
    mesh_rows = [
        {
            "state_index": int(row["state_index"]),
            "state": state_names.get(int(row["state_index"]), ""),
            "VGS_V": row["VGS_V"],
            "VDS_V": row["VDS_V"],
            "VS_V": row["VS_V"],
            "comparison": comparison_name,
            "reference_mesh_level": reference_level,
            "candidate_mesh_level": candidate_level,
            "quantity": row["quantity"],
            "reference_value": row["reference_value"],
            "candidate_value": row["candidate_value"],
            "absolute_difference": row["absolute_difference"],
            "relative_difference": row["relative_difference"],
            "comparison_floor": PRIMARY_ABSOLUTE_TOLERANCE,
            "relative_tolerance": relative_tolerance,
            "passed": bool(row["passed"]),
        }
        for (
            comparison_name,
            reference_level,
            candidate_level,
            comparison,
            relative_tolerance,
        ) in comparison_specs
        for row in comparison["rows"]
    ]
    return {
        "terminal": terminal_rows,
        "matrix": matrix_rows,
        "summary": summary_rows,
        "validation": validation_rows,
        "ward": ward_rows,
        "comparison": comparison_rows,
        "mesh": mesh_rows,
    }


def build_candidate_readme(
    bundles: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
) -> str:
    def comparison_summary(label: str, comparison: Mapping[str, Any]) -> str:
        rows = list(comparison.get("rows", ()))
        failed = [row for row in rows if not bool(row.get("passed"))]
        nonfinite = [
            row
            for row in rows
            if not math.isfinite(_numeric_or_nan(row.get("reference_value")))
            or not math.isfinite(_numeric_or_nan(row.get("candidate_value")))
        ]
        finite_relative = [
            _numeric_or_nan(row.get("relative_difference"))
            for row in rows
            if math.isfinite(_numeric_or_nan(row.get("relative_difference")))
        ]
        maximum_relative = max(finite_relative, default=math.nan)
        return (
            f"- {label}: {'PASS' if comparison.get('passed') else 'FAIL'}; "
            f"failed rows `{len(failed)}/{len(rows)}`, non-finite pairs "
            f"`{len(nonfinite)}`, maximum finite scaled relative difference "
            f"`{maximum_relative:.12e}`. Per-row absolute differences retain "
            "their charge (C) or capacitance (F) units in the CSV."
        )

    criteria_lines = "\n".join(
        f"| `{name}` | {'PASS' if passed else 'FAIL'} |"
        for name, passed in acceptance["criteria"].items()
    )
    topology_lines: list[str] = []
    for mesh_level, bundle in bundles.items():
        for contact, report in bundle.get("topology", {}).get("contacts", {}).items():
            topology_lines.append(
                "| "
                + " | ".join(
                    (
                        mesh_level,
                        contact,
                        str(report.get("edge_count", "")),
                        str(report.get("node_count", "")),
                        f"{_numeric_or_nan(report.get('surface_area_cm2')):.12e}",
                        f"({_numeric_or_nan(report.get('normal_r')):.6g}, "
                        f"{_numeric_or_nan(report.get('normal_z')):.6g})",
                        "PASS" if report.get("passed") else "FAIL",
                    )
                )
                + " |"
            )
    return f"""# Full-terminal quasi-static Q/C candidate

This directory is an isolated candidate. It does not replace or modify the
established `shared_data/terminal_charge_by_state.csv`,
`capacitance_matrix_by_state.csv`, or `capacitance_summary_by_state.csv`.

## Decision

`terminal_charge_status = {acceptance['status']}`

Raw `Qg_contact`, `Qd_contact`, and `Qs_contact` use the same DEVSIM
`get_contact_charge(..., equation="PotentialEquation")` definition and the
same `PotentialEdgeFlux*CylindricalEdgeCouple` weighting. Qd/Qs and measured
drain/source matrix entries are numeric only when every criterion below
passes; otherwise they remain NaN. Existing gate charge and gate-measured
Cgg/Cgd/Cgs remain independently available.

| Criterion | Result |
|---|---|
{criteria_lines}

## Contact topology and normals

The reported `ContactNSurfaceNormal` vector is audited against the expected
region-boundary geometry, but it is not used to manufacture or flip charge.
Charge sign remains exactly the DEVSIM `PotentialEquation` contact reaction
sign and is checked by Gauss closure and bias response. No manual sign flip or
area multiplier is applied. A zero vector or zero cylindrical surface fails
promotion.

| Mesh | Contact | Edges | Nodes | Cylindrical surface (cm2) | Runtime normal (r,z) | Result |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(topology_lines)}

## Capacitance convention

All entries use `Cij = dQi/dVj`, with independent absolute Vg, Vd, and Vs
perturbations at 0.5, 1, and 2 mV and a nominal 1 mV central difference. Ntrap,
geometry, material parameters, the other two biases, and solver settings stay
fixed. Every endpoint is followed by bias and charge restoration checks.
Matrix symmetry is neither imposed nor used as an acceptance criterion.
For raw Poisson-contact flux, Gauss law gives
`sum_i(Cij) = -d(Qmobile+Qtrap+Qfixed)/dVj`; with fixed trap and doping this is
generally `-dQmobile/dVj`, not identically zero. The requested zero column-sum
test is therefore retained as a substantive promotion criterion and its
failure is not hidden by a SPICE sign conversion.

## Ward-Dutton comparison

`ward_dutton_mobile_channel` partitions only signed MoS2 mobile charge using
runtime `z`, `zs`, and `zd`. It excludes Qtrap and Qfixed, is always labelled
`mobile_partition_only`, and is never promoted to a full-electrode fallback.
No 50:50 partition, planar area, current-times-time, fitted scale, or dielectric
charge partition is used.

## Weighting-potential feasibility

A Green-reciprocity/weighting-potential extension is conceptually possible
only after each physical electrode is represented by a non-degenerate boundary
with validated axisymmetric surface measure. The current source and drain each
have zero contact edges, zero cylindrical surface area, and a zero reported
normal, so a defensible weighting-potential solve cannot repair this topology
in candidate post-processing. No weighting-potential model is implemented here;
that expansion requires an explicit geometry/contact-model decision and user
approval.

## Numerical checks and limitations

- Base, independent base repeat, and `ps*0.5` fine meshes run in fresh Python
  processes; only mesh spacing changes and all line positions are verified.
  `mesh_refinement_qc.csv` retains every repeat and base/fine comparison row.
{comparison_summary('Base versus independent base repeat', acceptance['reproducibility'])}
{comparison_summary('Base versus fine mesh', acceptance['mesh'])}
- The Gauss relative denominator is the maximum of all six charge magnitudes
  and `{CHARGE_FLOOR_C:.1e} C`.
- Charge acceptance uses `{CHARGE_ABSOLUTE_TOLERANCE_C:.1e} C +
  {CHARGE_RELATIVE_TOLERANCE:.1e}*scale`.
- Reproducibility uses `{PRIMARY_ABSOLUTE_TOLERANCE:.1e} +
  {REPRODUCIBILITY_RELATIVE_TOLERANCE:.1e}*scale`; mesh comparison uses the same
  floor and `{MESH_RELATIVE_TOLERANCE:.1e}` relative tolerance.
- The sampled finite-secant continuity check is diagnostic and cannot prove
  continuity between the ten reference points.
- Ward-Dutton and direct contact flux represent different physical quantities;
  `method_comparison.csv` intentionally contains no difference or ratio.
- This remains a quasi-static electron-only, fixed-trap model. It does not add
  Schottky barriers, contact resistance, metal fitting, ERASE, or retention.
"""


def write_candidate_outputs(
    bundles: Mapping[str, Mapping[str, Any]],
    acceptance: Mapping[str, Any],
    *,
    output_directory: str | Path = CANDIDATE_DIRECTORY,
) -> tuple[Path, ...]:
    """Write only the seven requested CSVs and one candidate README."""

    directory = Path(output_directory).resolve()
    if directory != CANDIDATE_DIRECTORY.resolve():
        raise ValueError(
            "candidate output directory is fixed to "
            f"{CANDIDATE_DIRECTORY.resolve()}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    rows = build_candidate_output_rows(bundles, acceptance)
    csv_specs = (
        ("terminal", qc.TERMINAL_CHARGE_FULL_FIELDNAMES),
        ("matrix", qc.CAPACITANCE_MATRIX_FULL_FIELDNAMES),
        ("summary", qc.CAPACITANCE_SUMMARY_FULL_FIELDNAMES),
        ("validation", qc.DIRECT_CONTACT_VALIDATION_FIELDNAMES),
        ("ward", qc.WARD_DUTTON_FIELDNAMES),
        ("comparison", qc.METHOD_COMPARISON_FIELDNAMES),
        ("mesh", qc.MESH_REFINEMENT_QC_FIELDNAMES),
    )
    written: list[Path] = []
    for key, fieldnames in csv_specs:
        path = directory / OUTPUT_FILENAMES[key]
        qc.write_deterministic_csv(path, fieldnames, rows[key])
        written.append(path)
    readme = directory / OUTPUT_FILENAMES["readme"]
    readme_text = build_candidate_readme(bundles, acceptance)
    readme.write_text(readme_text.rstrip() + "\n", encoding="utf-8")
    written.append(readme)
    expected_names = set(OUTPUT_FILENAMES.values())
    if {path.name for path in written} != expected_names:
        raise RuntimeError("candidate output file set mismatch")
    return tuple(written)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate isolated full-terminal quasi-static Q/C candidate data "
            "without changing shared_data."
        )
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-level", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.worker:
        if (
            not arguments.mesh_level
            or arguments.mesh_scale is None
            or arguments.worker_output is None
        ):
            raise SystemExit("private worker mode requires level, scale, and output")
        bundle = run_worker(arguments.mesh_level, arguments.mesh_scale)
        _write_worker_bundle(arguments.worker_output, bundle)
        return 0
    if any(
        value is not None
        for value in (
            arguments.mesh_level,
            arguments.mesh_scale,
            arguments.worker_output,
        )
    ):
        raise SystemExit("worker-only arguments require --worker")

    public_hashes = capture_public_qc_hashes()
    with tempfile.TemporaryDirectory(prefix="full_terminal_qc_") as temporary:
        bundles = run_fresh_workers(temporary)
    assert_public_qc_hashes_unchanged(public_hashes)
    acceptance = determine_aggregate_acceptance(bundles)
    outputs = write_candidate_outputs(bundles, acceptance)
    assert_public_qc_hashes_unchanged(public_hashes)
    print(f"Direct-contact candidate status: {acceptance['status']}")
    for criterion, passed in acceptance["criteria"].items():
        print(f"  {criterion}: {passed}")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
