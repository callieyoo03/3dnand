"""Fresh-process continuation diagnostics for the programmed/off reference.

This runner is intentionally separate from the production characterization and
candidate generators.  It does not alter material parameters, equations,
solver tolerances, maximum iterations, or the public/candidate data bundles.

The public invocation starts one fresh Python process for each combination of
base/fine mesh and continuation path A-D.  Each worker's complete stdout and
stderr stream is retained under ``programmed_off_solver_logs``.  The strict CSV
contains one row per DC solve trial plus one summary row per worker.

No simulation is performed merely by importing this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import subprocess
import sys
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
DEFAULT_OUTPUT_DIRECTORY = (
    MODULE_DIRECTORY / "results" / "qc_root_cause_diagnostics"
)
DIAGNOSTIC_CSV_NAME = "programmed_off_continuation_diagnostic.csv"
SOLVER_LOG_DIRECTORY_NAME = "programmed_off_solver_logs"
SCHEMA_VERSION = "programmed_off_continuation_v1"

TARGET_STATE_INDEX = 4
TARGET_GATE_BIAS_V = -1.0
TARGET_DRAIN_BIAS_V = 0.05
TARGET_SOURCE_BIAS_V = 0.0
LOW_DRAIN_ANCHOR_V = 0.01

MESH_SPECS = (
    ("base", 1.0),
    ("fine", 0.5),
)
PATH_IDS = ("A", "B", "C", "D")

PROTECTED_OUTPUT_DIRECTORIES = (
    REPOSITORY_ROOT / "shared_data",
    MODULE_DIRECTORY / "results" / "contact_topology_qc_candidate",
    MODULE_DIRECTORY / "results" / "contact_topology_dc_candidate",
    MODULE_DIRECTORY / "results" / "full_terminal_qc_candidate",
)


DIAGNOSTIC_FIELDS = (
    "schema_version",
    "record_type",
    "run_id",
    "mesh_level",
    "mesh_scale",
    "path_id",
    "path_description",
    "worker_pid",
    "log_file",
    "stage_index",
    "stage_name",
    "ramp_kind",
    "terminal",
    "trial_index",
    "trial_role",
    "stage_target_bias_V",
    "stage_target_ntrap_cm3",
    "requested_gate_bias_V",
    "requested_drain_bias_V",
    "requested_source_bias_V",
    "requested_ntrap_cm3",
    "trial_voltage_step_V",
    "trial_ntrap_step_cm3",
    "last_converged_gate_bias_V",
    "last_converged_drain_bias_V",
    "last_converged_source_bias_V",
    "last_converged_ntrap_cm3",
    "solve_maximum_iterations_override",
    "solve_success",
    "exception_type",
    "exception_message",
    "exception_cause_chain",
    "exception_traceback",
    "possible_equation_failure",
    "equation_failure_evidence",
    "solver_output_line_count",
    "solver_output_excerpt",
    "path_success",
    "final_gate_bias_V",
    "final_drain_bias_V",
    "final_source_bias_V",
    "final_ntrap_cm3",
    "target_gate_reached",
    "target_drain_reached",
    "target_source_reached",
    "target_ntrap_reached",
    "final_solution_available",
    "final_ID_A",
    "final_abs_ID_A",
    "final_source_current_A",
    "current_conservation_residual_A",
    "Vth_reference_current_A",
    "reference_ID_A",
    "reference_abs_ID_A",
    "reference_current_source",
    "reference_current_sha256",
    "Qg_contact_C",
    "Qd_raw_contact_C",
    "Qs_raw_contact_C",
    "solver_absolute_error",
    "solver_relative_error",
    "solver_maximum_iterations",
    "gate_initial_step_V",
    "gate_minimum_step_V",
    "drain_initial_step_V",
    "drain_minimum_step_V",
    "trap_initial_step_cm3",
    "trap_minimum_step_cm3",
    "error_message",
)


@dataclass(frozen=True)
class RuntimeState:
    """Terminal biases and static trapped-electron density at one instant."""

    gate_bias_V: float
    drain_bias_V: float
    source_bias_V: float
    ntrap_cm3: float


@dataclass(frozen=True)
class RampStage:
    """One high-level continuation stage executed with existing ramp helpers."""

    stage_index: int
    stage_name: str
    ramp_kind: str
    terminal: str
    target_value: float

    @property
    def target_bias_V(self) -> float | None:
        return self.target_value if self.ramp_kind == "terminal" else None

    @property
    def target_ntrap_cm3(self) -> float | None:
        return self.target_value if self.ramp_kind == "trap" else None


PATH_DESCRIPTIONS = {
    "A": "Ntrap(target) -> Vd(0.05 V) -> Vg(-1 V)",
    "B": "Ntrap(target) -> Vg(-1 V at Vd=0) -> Vd(0.05 V)",
    "C": (
        "Ntrap(target) -> Vd(0.01 V) -> Vg(-1 V) -> Vd(0.05 V)"
    ),
    "D": (
        "Ntrap(0,2e17,5e17,1e18,2e18 cm^-3) -> "
        "Vg(-1 V at Vd=0) -> Vd(0.05 V)"
    ),
}


def build_path_stages(
    path_id: str,
    state_densities_cm3: Sequence[float],
) -> tuple[RampStage, ...]:
    """Return the exact, auditable stage sequence for path A-D."""

    normalized = str(path_id).strip().upper()
    if normalized not in PATH_IDS:
        raise ValueError(f"unknown continuation path {path_id!r}")
    densities = tuple(float(value) for value in state_densities_cm3)
    if len(densities) != 5 or any(
        not math.isfinite(value) or value < 0.0 for value in densities
    ):
        raise ValueError("state_densities_cm3 must contain five nonnegative values")
    if any(right < left for left, right in zip(densities, densities[1:])):
        raise ValueError("state densities must be nondecreasing")

    target = densities[-1]
    if normalized == "A":
        raw = (
            ("target trap ramp", "trap", "trap", target),
            ("read drain ramp", "terminal", "drain", TARGET_DRAIN_BIAS_V),
            ("off gate ramp", "terminal", "gate", TARGET_GATE_BIAS_V),
        )
    elif normalized == "B":
        raw = (
            ("target trap ramp", "trap", "trap", target),
            ("off gate ramp at zero drain", "terminal", "gate", TARGET_GATE_BIAS_V),
            ("read drain ramp", "terminal", "drain", TARGET_DRAIN_BIAS_V),
        )
    elif normalized == "C":
        raw = (
            ("target trap ramp", "trap", "trap", target),
            ("low-drain anchor ramp", "terminal", "drain", LOW_DRAIN_ANCHOR_V),
            ("off gate ramp at low drain", "terminal", "gate", TARGET_GATE_BIAS_V),
            ("read drain completion ramp", "terminal", "drain", TARGET_DRAIN_BIAS_V),
        )
    else:
        density_stages = tuple(
            (
                f"state-{index} trap ramp",
                "trap",
                "trap",
                density,
            )
            for index, density in enumerate(densities)
        )
        raw = (
            *density_stages,
            ("off gate ramp after state ladder", "terminal", "gate", TARGET_GATE_BIAS_V),
            ("read drain ramp", "terminal", "drain", TARGET_DRAIN_BIAS_V),
        )

    return tuple(
        RampStage(
            stage_index=index,
            stage_name=name,
            ramp_kind=kind,
            terminal=terminal,
            target_value=float(target_value),
        )
        for index, (name, kind, terminal, target_value) in enumerate(raw, start=1)
    )


def _blank_row() -> dict[str, Any]:
    row = {field: "" for field in DIAGNOSTIC_FIELDS}
    row["schema_version"] = SCHEMA_VERSION
    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_output_directory(path: str | Path) -> Path:
    """Reject public and prior-candidate output trees."""

    destination = Path(path).resolve()
    for protected in PROTECTED_OUTPUT_DIRECTORIES:
        protected_resolved = protected.resolve()
        if destination == protected_resolved or _is_within(
            destination, protected_resolved
        ):
            raise ValueError(
                f"diagnostic output may not be written under {protected_resolved}"
            )
    return destination


def format_exception_cause_chain(error: BaseException | None) -> str:
    """Serialize explicit causes/contexts without losing the DEVSIM exception."""

    if error is None:
        return ""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        message = " ".join(str(current).split())
        parts.append(
            f"{type(current).__module__}.{type(current).__name__}: {message}"
        )
        current = current.__cause__ or current.__context__
    return " <- ".join(parts)


def format_exception_traceback(error: BaseException | None) -> str:
    if error is None:
        return ""
    return "".join(
        traceback.format_exception(type(error), error, error.__traceback__)
    )


_KNOWN_EQUATIONS = (
    "PotentialEquation",
    "ElectronContinuityEquation",
)


def infer_equation_failure(
    error: BaseException | None,
    solver_output: str,
) -> tuple[str, str]:
    """Return conservative equation labels and the text supporting them."""

    error_text = format_exception_cause_chain(error)
    combined = f"{error_text}\n{solver_output}"
    equations = [
        name for name in _KNOWN_EQUATIONS
        if re.search(re.escape(name), combined, flags=re.IGNORECASE)
    ]
    evidence_lines = [
        " ".join(line.split())
        for line in combined.splitlines()
        if any(
            token in line.lower()
            for token in (
                "equation",
                "iteration",
                "converg",
                "relative error",
                "absolute error",
                "factor",
                "singular",
            )
        )
    ]
    evidence = " | ".join(evidence_lines[-20:])
    return (";".join(equations) if equations else "unknown", evidence)


class _Tee(io.TextIOBase):
    """Copy Python-level output to the permanent worker log and a buffer."""

    def __init__(self, primary: TextIO, capture: io.StringIO) -> None:
        self._primary = primary
        self._capture = capture

    def write(self, value: str) -> int:
        self._primary.write(value)
        self._capture.write(value)
        return len(value)

    def flush(self) -> None:
        self._primary.flush()
        self._capture.flush()

    def fileno(self) -> int:
        return self._primary.fileno()

    def isatty(self) -> bool:
        return self._primary.isatty()

    @property
    def encoding(self) -> str | None:
        return getattr(self._primary, "encoding", None)


class SolveTrialRecorder:
    """Instrument existing ``solve_dc`` calls without changing their options."""

    def __init__(
        self,
        *,
        read_state: Callable[[], RuntimeState],
        solve_function: Callable[..., Any],
        run_id: str,
        mesh_level: str,
        mesh_scale: float,
        path_id: str,
        log_file: str,
    ) -> None:
        self.read_state = read_state
        self.solve_function = solve_function
        self.run_id = str(run_id)
        self.mesh_level = str(mesh_level)
        self.mesh_scale = float(mesh_scale)
        self.path_id = str(path_id).upper()
        self.log_file = str(log_file)
        self.rows: list[dict[str, Any]] = []
        self.last_converged = read_state()
        self.current_stage: RampStage | None = None
        self.trial_index = 0
        self._previous_trial_failed = False

    def begin_stage(self, stage: RampStage) -> int:
        if self.current_stage is not None:
            raise RuntimeError("a diagnostic ramp stage is already active")
        self.current_stage = stage
        return len(self.rows)

    def end_stage(self) -> None:
        self.current_stage = None

    def _identity_row(self) -> dict[str, Any]:
        row = _blank_row()
        row.update(
            {
                "record_type": "ramp_trial",
                "run_id": self.run_id,
                "mesh_level": self.mesh_level,
                "mesh_scale": self.mesh_scale,
                "path_id": self.path_id,
                "path_description": PATH_DESCRIPTIONS[self.path_id],
                "worker_pid": os.getpid(),
                "log_file": self.log_file,
            }
        )
        return row

    def _state_fields(self, prefix: str, state: RuntimeState) -> dict[str, float]:
        if prefix == "requested":
            return {
                "requested_gate_bias_V": state.gate_bias_V,
                "requested_drain_bias_V": state.drain_bias_V,
                "requested_source_bias_V": state.source_bias_V,
                "requested_ntrap_cm3": state.ntrap_cm3,
            }
        if prefix == "last_converged":
            return {
                "last_converged_gate_bias_V": state.gate_bias_V,
                "last_converged_drain_bias_V": state.drain_bias_V,
                "last_converged_source_bias_V": state.source_bias_V,
                "last_converged_ntrap_cm3": state.ntrap_cm3,
            }
        raise ValueError(f"unsupported state-field prefix {prefix!r}")

    def _trial_row(
        self,
        *,
        requested: RuntimeState,
        last_converged: RuntimeState,
        maximum_iterations: Any,
    ) -> dict[str, Any]:
        stage = self.current_stage
        if stage is None:
            raise RuntimeError("solve_dc called outside an active diagnostic stage")
        self.trial_index += 1
        if stage.ramp_kind == "terminal":
            voltage_step = (
                getattr(requested, f"{stage.terminal}_bias_V")
                - getattr(last_converged, f"{stage.terminal}_bias_V")
            )
            trap_step = math.nan
        else:
            voltage_step = math.nan
            trap_step = requested.ntrap_cm3 - last_converged.ntrap_cm3
        same_as_last = requested == last_converged
        row = self._identity_row()
        row.update(
            {
                "stage_index": stage.stage_index,
                "stage_name": stage.stage_name,
                "ramp_kind": stage.ramp_kind,
                "terminal": stage.terminal,
                "trial_index": self.trial_index,
                "trial_role": (
                    "recovery" if self._previous_trial_failed and same_as_last
                    else "advance"
                ),
                "stage_target_bias_V": (
                    stage.target_bias_V if stage.target_bias_V is not None else ""
                ),
                "stage_target_ntrap_cm3": (
                    stage.target_ntrap_cm3
                    if stage.target_ntrap_cm3 is not None
                    else ""
                ),
                "trial_voltage_step_V": (
                    voltage_step if math.isfinite(voltage_step) else ""
                ),
                "trial_ntrap_step_cm3": (
                    trap_step if math.isfinite(trap_step) else ""
                ),
                "solve_maximum_iterations_override": (
                    "" if maximum_iterations is None else maximum_iterations
                ),
                **self._state_fields("requested", requested),
                **self._state_fields("last_converged", last_converged),
            }
        )
        return row

    def solve_wrapper(self, *args: Any, **kwargs: Any) -> Any:
        requested = self.read_state()
        previous = self.last_converged
        maximum_iterations = kwargs.get("maximum_iterations")
        row = self._trial_row(
            requested=requested,
            last_converged=previous,
            maximum_iterations=maximum_iterations,
        )
        capture = io.StringIO()
        print(
            "[programmed-off-diagnostic] solve trial "
            f"{row['trial_index']} path={self.path_id} stage={row['stage_name']} "
            f"requested={asdict(requested)}",
            flush=True,
        )
        try:
            with redirect_stdout(_Tee(sys.stdout, capture)), redirect_stderr(
                _Tee(sys.stderr, capture)
            ):
                result = self.solve_function(*args, **kwargs)
        except Exception as error:
            solver_output = capture.getvalue()
            equation, evidence = infer_equation_failure(error, solver_output)
            row.update(
                {
                    "solve_success": False,
                    "exception_type": (
                        f"{type(error).__module__}.{type(error).__name__}"
                    ),
                    "exception_message": " ".join(str(error).split()),
                    "exception_cause_chain": format_exception_cause_chain(error),
                    "exception_traceback": format_exception_traceback(error),
                    "possible_equation_failure": equation,
                    "equation_failure_evidence": evidence,
                    "solver_output_line_count": len(solver_output.splitlines()),
                    "solver_output_excerpt": "\n".join(
                        solver_output.splitlines()[-40:]
                    ),
                    "error_message": "solve_dc failed",
                }
            )
            self.rows.append(row)
            self._previous_trial_failed = True
            print(
                "[programmed-off-diagnostic] original solve exception follows:",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exception(type(error), error, error.__traceback__)
            raise
        else:
            solver_output = capture.getvalue()
            converged = self.read_state()
            row.update(
                {
                    "solve_success": True,
                    "solver_output_line_count": len(solver_output.splitlines()),
                    "solver_output_excerpt": "\n".join(
                        solver_output.splitlines()[-40:]
                    ),
                    "error_message": "",
                }
            )
            self.rows.append(row)
            self.last_converged = converged
            self._previous_trial_failed = False
            return result

    def record_noop(
        self,
        stage: RampStage,
        *,
        success: bool = True,
        error: BaseException | None = None,
    ) -> None:
        """Record a stage that reached no ``solve_dc`` call."""

        state = self.read_state()
        row = self._identity_row()
        self.trial_index += 1
        row.update(
            {
                "stage_index": stage.stage_index,
                "stage_name": stage.stage_name,
                "ramp_kind": stage.ramp_kind,
                "terminal": stage.terminal,
                "trial_index": self.trial_index,
                "trial_role": "no_op" if success else "stage_failure_before_solve",
                "stage_target_bias_V": (
                    stage.target_bias_V if stage.target_bias_V is not None else ""
                ),
                "stage_target_ntrap_cm3": (
                    stage.target_ntrap_cm3
                    if stage.target_ntrap_cm3 is not None
                    else ""
                ),
                "trial_voltage_step_V": 0.0 if stage.ramp_kind == "terminal" else "",
                "trial_ntrap_step_cm3": 0.0 if stage.ramp_kind == "trap" else "",
                "solve_success": bool(success),
                "exception_type": (
                    f"{type(error).__module__}.{type(error).__name__}"
                    if error is not None else ""
                ),
                "exception_message": (
                    " ".join(str(error).split()) if error is not None else ""
                ),
                "exception_cause_chain": format_exception_cause_chain(error),
                "exception_traceback": format_exception_traceback(error),
                "possible_equation_failure": (
                    infer_equation_failure(error, "")[0]
                    if error is not None else ""
                ),
                "error_message": (
                    "ramp stage failed before solve_dc" if error is not None else ""
                ),
                **self._state_fields("requested", state),
                **self._state_fields("last_converged", self.last_converged),
            }
        )
        self.rows.append(row)

    @contextmanager
    def instrument(self, memory_window: Any, helpers: Any) -> Iterator[None]:
        memory_original = memory_window.solve_dc
        helpers_original = helpers.solve_dc
        if memory_original is not self.solve_function:
            raise RuntimeError("solve_function does not match run_memory_window.solve_dc")
        memory_window.solve_dc = self.solve_wrapper
        helpers.solve_dc = self.solve_wrapper
        try:
            yield
        finally:
            memory_window.solve_dc = memory_original
            helpers.solve_dc = helpers_original


@contextmanager
def scaled_runtime_mesh_lines(
    structure_module: Any,
    scale: float,
) -> Iterator[None]:
    """Scale only existing ``ps`` values; geometry and physics remain fixed."""

    mesh_scale = float(scale)
    if not math.isfinite(mesh_scale) or mesh_scale <= 0.0:
        raise ValueError("mesh scale must be finite and positive")
    original = structure_module.add_2d_mesh_line

    def scaled_add_2d_mesh_line(*args: Any, **kwargs: Any) -> Any:
        if args or "ps" not in kwargs:
            raise RuntimeError("diagnostic mesh wrapper requires keyword ps")
        forwarded = dict(kwargs)
        forwarded["ps"] = float(forwarded["ps"]) * mesh_scale
        return original(**forwarded)

    structure_module.add_2d_mesh_line = scaled_add_2d_mesh_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = original


def _read_runtime_state(device: str, charge_trap_region: str) -> RuntimeState:
    from devsim import get_parameter
    from trap_models import TRAPPED_ELECTRON_PARAMETER

    return RuntimeState(
        gate_bias_V=float(get_parameter(device=device, name="gate_bias")),
        drain_bias_V=float(get_parameter(device=device, name="drain_bias")),
        source_bias_V=float(get_parameter(device=device, name="source_bias")),
        ntrap_cm3=float(
            get_parameter(
                device=device,
                region=charge_trap_region,
                name=TRAPPED_ELECTRON_PARAMETER,
            )
        ),
    )


def load_reference_current(
    path: str | Path,
    *,
    state_index: int = TARGET_STATE_INDEX,
    gate_bias_V: float = TARGET_GATE_BIAS_V,
    drain_bias_V: float = TARGET_DRAIN_BIAS_V,
) -> dict[str, Any]:
    """Read exactly one immutable public ID-VG reference row."""

    source = Path(path).resolve()
    with source.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        matches = [
            row for row in reader
            if int(row["state_index"]) == int(state_index)
            and math.isclose(
                float(row["VGS_V"]), gate_bias_V, rel_tol=0.0, abs_tol=1.0e-12
            )
            and math.isclose(
                float(row["VDS_V"]), drain_bias_V, rel_tol=0.0, abs_tol=1.0e-12
            )
        ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one reference-current row in {source}; found {len(matches)}"
        )
    row = matches[0]
    return {
        "reference_ID_A": float(row["ID_A"]),
        "reference_abs_ID_A": float(row["abs_ID_A"]),
        "reference_current_source": str(source),
        "reference_current_sha256": _sha256(source),
    }


def _target_checks(state: RuntimeState, target_ntrap_cm3: float) -> dict[str, bool]:
    return {
        "target_gate_reached": math.isclose(
            state.gate_bias_V, TARGET_GATE_BIAS_V, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "target_drain_reached": math.isclose(
            state.drain_bias_V, TARGET_DRAIN_BIAS_V, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "target_source_reached": math.isclose(
            state.source_bias_V, TARGET_SOURCE_BIAS_V, rel_tol=0.0, abs_tol=1.0e-12
        ),
        "target_ntrap_reached": math.isclose(
            state.ntrap_cm3,
            float(target_ntrap_cm3),
            rel_tol=1.0e-12,
            abs_tol=1.0,
        ),
    }


def _solver_settings(tp: Any, helpers: Any) -> dict[str, Any]:
    return {
        "solver_absolute_error": float(tp.SOLVER_ABSOLUTE_ERROR),
        "solver_relative_error": float(tp.SOLVER_RELATIVE_ERROR),
        "solver_maximum_iterations": int(tp.SOLVER_MAXIMUM_ITERATIONS),
        "gate_initial_step_V": float(helpers.GATE_INITIAL_STEP_V),
        "gate_minimum_step_V": float(helpers.GATE_MINIMUM_STEP_V),
        "drain_initial_step_V": float(helpers.DRAIN_INITIAL_STEP_V),
        "drain_minimum_step_V": float(helpers.DRAIN_MINIMUM_STEP_V),
        "trap_initial_step_cm3": float(helpers.TRAP_INITIAL_STEP_CM3),
        "trap_minimum_step_cm3": float(helpers.TRAP_MINIMUM_STEP_CM3),
    }


def _summary_identity(
    *,
    run_id: str,
    mesh_level: str,
    mesh_scale: float,
    path_id: str,
    log_file: str,
) -> dict[str, Any]:
    row = _blank_row()
    row.update(
        {
            "record_type": "path_summary",
            "run_id": run_id,
            "mesh_level": mesh_level,
            "mesh_scale": mesh_scale,
            "path_id": path_id,
            "path_description": PATH_DESCRIPTIONS[path_id],
            "worker_pid": os.getpid(),
            "log_file": log_file,
        }
    )
    return row


def run_worker(
    *,
    run_id: str,
    mesh_level: str,
    mesh_scale: float,
    path_id: str,
    log_file: str,
) -> dict[str, Any]:
    """Build one fresh device and execute one continuation path."""

    import run_memory_window as memory_window
    import state_characterization_config as config
    import state_sweep_helpers as helpers
    import trap_parameters as tp
    from devsim import get_contact_current
    from terminal_charge import extract_terminal_charge_components

    normalized_path = str(path_id).upper()
    states = tuple(config.MEMORY_STATES)
    densities = tuple(float(state["ntrap_cm3"]) for state in states)
    if len(states) != 5 or int(states[-1]["state_index"]) != TARGET_STATE_INDEX:
        raise RuntimeError("canonical five-state configuration is unavailable")
    stages = build_path_stages(normalized_path, densities)

    with scaled_runtime_mesh_lines(
        memory_window.parameterized_device_structure, mesh_scale
    ):
        _, point = helpers.initialize_characterization_device()

    read_state = lambda: _read_runtime_state(memory_window.device, tp.CHARGE_TRAP_REGION)
    recorder = SolveTrialRecorder(
        read_state=read_state,
        solve_function=memory_window.solve_dc,
        run_id=run_id,
        mesh_level=mesh_level,
        mesh_scale=mesh_scale,
        path_id=normalized_path,
        log_file=log_file,
    )
    path_error: BaseException | None = None

    with recorder.instrument(memory_window, helpers):
        for stage in stages:
            row_count = recorder.begin_stage(stage)
            try:
                if stage.ramp_kind == "trap":
                    helpers.ramp_trap_density(
                        point,
                        stage.target_value,
                        label=f"diagnostic path {normalized_path}: {stage.stage_name}",
                    )
                else:
                    helpers.ramp_terminal(
                        point,
                        stage.terminal,
                        stage.target_value,
                        label=f"diagnostic path {normalized_path}: {stage.stage_name}",
                    )
            except Exception as error:
                path_error = error
                print(
                    "[programmed-off-diagnostic] ramp stage failed; "
                    "high-level exception follows:",
                    file=sys.stderr,
                    flush=True,
                )
                traceback.print_exception(type(error), error, error.__traceback__)
            finally:
                recorder.end_stage()
            if len(recorder.rows) == row_count:
                recorder.record_noop(
                    stage,
                    success=path_error is None,
                    error=path_error,
                )
            if path_error is not None:
                break

    final_state = read_state()
    checks = _target_checks(final_state, densities[-1])
    path_success = path_error is None and all(checks.values())
    final_solution_available = final_state == recorder.last_converged
    measurement_error: BaseException | None = None
    final_values = {
        "final_ID_A": math.nan,
        "final_abs_ID_A": math.nan,
        "final_source_current_A": math.nan,
        "current_conservation_residual_A": math.nan,
        "Qg_contact_C": math.nan,
        "Qd_raw_contact_C": math.nan,
        "Qs_raw_contact_C": math.nan,
    }
    # A failed adaptive ramp normally restores and solves the last converged
    # coordinate before raising.  Preserve ID and raw contact charge there as
    # requested, while keeping target_reached/path_success false.  If recovery
    # itself failed, do not read quantities from an invalid solution.
    if final_solution_available:
        try:
            drain_current = float(memory_window.get_drain_current_A())
            source_current = float(
                get_contact_current(
                    device=memory_window.device,
                    contact="source",
                    equation="ElectronContinuityEquation",
                )
            )
            direct = extract_terminal_charge_components(
                device=memory_window.device
            )
            final_values.update(
                {
                    "final_ID_A": drain_current,
                    "final_abs_ID_A": abs(drain_current),
                    "final_source_current_A": source_current,
                    "current_conservation_residual_A": drain_current + source_current,
                    "Qg_contact_C": float(direct["Qg_C"]),
                    "Qd_raw_contact_C": float(
                        direct["Qd_poisson_boundary_flux_C"]
                    ),
                    "Qs_raw_contact_C": float(
                        direct["Qs_poisson_boundary_flux_C"]
                    ),
                }
            )
        except Exception as error:
            measurement_error = error
            path_success = False

    reference = load_reference_current(REPOSITORY_ROOT / "shared_data" / "idvg_by_state.csv")
    summary = _summary_identity(
        run_id=run_id,
        mesh_level=mesh_level,
        mesh_scale=mesh_scale,
        path_id=normalized_path,
        log_file=log_file,
    )
    report_error = path_error or measurement_error
    summary_equation, summary_evidence = (
        infer_equation_failure(report_error, "")
        if report_error is not None else ("", "")
    )
    summary.update(
        {
            "path_success": path_success,
            "final_gate_bias_V": final_state.gate_bias_V,
            "final_drain_bias_V": final_state.drain_bias_V,
            "final_source_bias_V": final_state.source_bias_V,
            "final_ntrap_cm3": final_state.ntrap_cm3,
            **checks,
            "final_solution_available": final_solution_available,
            **final_values,
            "Vth_reference_current_A": float(config.VTH_TARGET_CURRENT_A),
            **reference,
            **_solver_settings(tp, helpers),
            "exception_type": (
                f"{type(report_error).__module__}.{type(report_error).__name__}"
                if report_error is not None else ""
            ),
            "exception_message": (
                " ".join(str(report_error).split()) if report_error is not None else ""
            ),
            "exception_cause_chain": format_exception_cause_chain(report_error),
            "exception_traceback": format_exception_traceback(report_error),
            "possible_equation_failure": summary_equation,
            "equation_failure_evidence": summary_evidence,
            "error_message": (
                "continuation or final extraction failed"
                if report_error is not None else (
                    "target state mismatch" if not all(checks.values()) else ""
                )
            ),
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "mesh_level": mesh_level,
        "mesh_scale": mesh_scale,
        "path_id": normalized_path,
        "stages": [asdict(stage) for stage in stages],
        "rows": recorder.rows,
        "summary": summary,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "nan"
    if isinstance(value, Path):
        return str(value)
    return value


def write_worker_json(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite worker artifact {destination}")
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            _json_safe(bundle), stream, ensure_ascii=False, allow_nan=False,
            sort_keys=True, separators=(",", ":"),
        )
        stream.write("\n")
    temporary.replace(destination)
    return destination


def write_diagnostic_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    """Validate every row before atomically creating the strict CSV."""

    destination = Path(path).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic CSV {destination}")
    expected = set(DIAGNOSTIC_FIELDS)
    materialized: list[dict[str, Any]] = []
    for index, source_row in enumerate(rows):
        row = dict(source_row)
        if set(row) != expected:
            raise ValueError(
                f"diagnostic row {index} schema mismatch: "
                f"missing={sorted(expected - set(row))}, "
                f"extra={sorted(set(row) - expected)}"
            )
        materialized.append(row)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=DIAGNOSTIC_FIELDS,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)
    temporary.replace(destination)
    return destination


def build_worker_command(
    *,
    python_executable: str,
    run_id: str,
    mesh_level: str,
    mesh_scale: float,
    path_id: str,
    worker_json: Path,
    log_relative_path: str,
) -> list[str]:
    return [
        str(python_executable),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--run-id",
        str(run_id),
        "--mesh-level",
        str(mesh_level),
        "--mesh-scale",
        f"{float(mesh_scale):.17g}",
        "--path-id",
        str(path_id),
        "--worker-json",
        str(worker_json.resolve()),
        "--log-relative-path",
        str(log_relative_path),
    ]


def build_worker_specs(
    output_directory: str | Path,
    *,
    run_id: str,
    meshes: Sequence[str] = tuple(level for level, _ in MESH_SPECS),
    paths: Sequence[str] = PATH_IDS,
) -> list[dict[str, Any]]:
    root = Path(output_directory).resolve()
    log_directory = root / SOLVER_LOG_DIRECTORY_NAME
    scale_by_level = dict(MESH_SPECS)
    specs: list[dict[str, Any]] = []
    for mesh_level in meshes:
        if mesh_level not in scale_by_level:
            raise ValueError(f"unknown mesh level {mesh_level!r}")
        for path_id in paths:
            normalized = str(path_id).upper()
            if normalized not in PATH_IDS:
                raise ValueError(f"unknown path {path_id!r}")
            stem = f"{run_id}_{mesh_level}_path_{normalized}"
            log_path = log_directory / f"{stem}.log"
            specs.append(
                {
                    "mesh_level": mesh_level,
                    "mesh_scale": scale_by_level[mesh_level],
                    "path_id": normalized,
                    "log_path": log_path,
                    "log_relative_path": str(log_path.relative_to(root)),
                    "worker_json": log_directory / f"{stem}.json",
                }
            )
    return specs


def _worker_failure_summary(
    *,
    run_id: str,
    spec: Mapping[str, Any],
    return_code: int,
    message: str,
) -> dict[str, Any]:
    row = _summary_identity(
        run_id=run_id,
        mesh_level=str(spec["mesh_level"]),
        mesh_scale=float(spec["mesh_scale"]),
        path_id=str(spec["path_id"]),
        log_file=str(spec["log_relative_path"]),
    )
    row.update(
        {
            "path_success": False,
            "exception_type": "subprocess.CalledProcessError",
            "exception_message": f"worker return code {return_code}",
            "exception_cause_chain": f"worker return code {return_code}",
            "solver_output_excerpt": message,
            "error_message": "fresh worker failed before producing a bundle",
        }
    )
    return row


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}_p{os.getpid()}"


def run_fresh_workers(
    output_directory: str | Path,
    *,
    python_executable: str = sys.executable,
    run_id: str | None = None,
    meshes: Sequence[str] = tuple(level for level, _ in MESH_SPECS),
    paths: Sequence[str] = PATH_IDS,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Execute every requested worker serially and retain every raw log."""

    root = validate_output_directory(output_directory)
    identifier = str(run_id or _new_run_id())
    csv_path = root / DIAGNOSTIC_CSV_NAME
    if csv_path.exists():
        raise FileExistsError(
            f"refusing to overwrite an existing diagnostic run: {csv_path}"
        )
    log_directory = root / SOLVER_LOG_DIRECTORY_NAME
    log_directory.mkdir(parents=True, exist_ok=True)
    specs = build_worker_specs(
        root, run_id=identifier, meshes=meshes, paths=paths
    )
    rows: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        log_path = Path(spec["log_path"])
        worker_json = Path(spec["worker_json"])
        if log_path.exists() or worker_json.exists():
            raise FileExistsError(
                f"refusing to overwrite permanent worker artifacts for {spec}"
            )
        command = build_worker_command(
            python_executable=python_executable,
            run_id=identifier,
            mesh_level=str(spec["mesh_level"]),
            mesh_scale=float(spec["mesh_scale"]),
            path_id=str(spec["path_id"]),
            worker_json=worker_json,
            log_relative_path=str(spec["log_relative_path"]),
        )
        print(
            f"Programmed/off diagnostic worker {index}/{len(specs)}: "
            f"mesh={spec['mesh_level']} path={spec['path_id']}",
            flush=True,
        )
        with log_path.open("x", encoding="utf-8", newline="\n") as log:
            completed = subprocess_run(
                command,
                cwd=REPOSITORY_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        return_code = int(getattr(completed, "returncode", 0))
        if return_code == 0 and worker_json.is_file():
            with worker_json.open("r", encoding="utf-8") as stream:
                bundle = json.load(stream)
            if (
                str(bundle.get("run_id")) != identifier
                or str(bundle.get("mesh_level")) != str(spec["mesh_level"])
                or str(bundle.get("path_id")) != str(spec["path_id"])
            ):
                raise RuntimeError(f"worker identity mismatch in {worker_json}")
            rows.extend(dict(row) for row in bundle.get("rows", ()))
            rows.append(dict(bundle["summary"]))
        else:
            tail = "\n".join(
                log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
            )
            rows.append(
                _worker_failure_summary(
                    run_id=identifier,
                    spec=spec,
                    return_code=return_code,
                    message=tail,
                )
            )
    return write_diagnostic_csv(csv_path, rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
        help="new diagnostic output directory; existing CSVs are never overwritten",
    )
    parser.add_argument(
        "--meshes",
        nargs="+",
        choices=tuple(level for level, _ in MESH_SPECS),
        default=tuple(level for level, _ in MESH_SPECS),
    )
    parser.add_argument(
        "--paths", nargs="+", choices=PATH_IDS, default=PATH_IDS
    )
    parser.add_argument("--python-executable", default=sys.executable)

    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--run-id", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-level", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-scale", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--path-id", choices=PATH_IDS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-json", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--log-relative-path", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.worker:
        required = (
            arguments.run_id,
            arguments.mesh_level,
            arguments.mesh_scale,
            arguments.path_id,
            arguments.worker_json,
            arguments.log_relative_path,
        )
        if any(value is None for value in required):
            raise SystemExit("worker mode requires run/mesh/path/json/log arguments")
        bundle = run_worker(
            run_id=str(arguments.run_id),
            mesh_level=str(arguments.mesh_level),
            mesh_scale=float(arguments.mesh_scale),
            path_id=str(arguments.path_id),
            log_file=str(arguments.log_relative_path),
        )
        write_worker_json(arguments.worker_json, bundle)
        return 0

    output = run_fresh_workers(
        arguments.output_directory,
        python_executable=str(arguments.python_executable),
        meshes=tuple(arguments.meshes),
        paths=tuple(arguments.paths),
    )
    print(f"Programmed/off continuation diagnostic written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
