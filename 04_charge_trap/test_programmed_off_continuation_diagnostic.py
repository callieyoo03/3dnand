"""DEVSIM-free tests for the programmed/off continuation diagnostic."""

from __future__ import annotations

import csv
import io
import json
import tempfile
import types
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import run_programmed_off_continuation_diagnostic as diagnostic


STATE_DENSITIES = (0.0, 2.0e17, 5.0e17, 1.0e18, 2.0e18)


class ProgrammedOffContinuationDiagnosticTests(unittest.TestCase):
    def test_path_definitions_are_exact(self) -> None:
        stages_a = diagnostic.build_path_stages("A", STATE_DENSITIES)
        self.assertEqual(
            [(stage.ramp_kind, stage.terminal, stage.target_value) for stage in stages_a],
            [
                ("trap", "trap", 2.0e18),
                ("terminal", "drain", 0.05),
                ("terminal", "gate", -1.0),
            ],
        )

        stages_b = diagnostic.build_path_stages("B", STATE_DENSITIES)
        self.assertEqual([stage.terminal for stage in stages_b], ["trap", "gate", "drain"])

        stages_c = diagnostic.build_path_stages("C", STATE_DENSITIES)
        self.assertEqual(
            [stage.target_value for stage in stages_c],
            [2.0e18, 0.01, -1.0, 0.05],
        )

        stages_d = diagnostic.build_path_stages("D", STATE_DENSITIES)
        self.assertEqual(
            [stage.target_value for stage in stages_d[:5]], list(STATE_DENSITIES)
        )
        self.assertEqual(
            [(stage.terminal, stage.target_value) for stage in stages_d[-2:]],
            [("gate", -1.0), ("drain", 0.05)],
        )

    def test_default_specs_cover_eight_fresh_workers(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            specs = diagnostic.build_worker_specs(name, run_id="fixture")

        self.assertEqual(len(specs), 8)
        identities = {
            (spec["mesh_level"], spec["path_id"]) for spec in specs
        }
        self.assertEqual(
            identities,
            {(mesh, path) for mesh in ("base", "fine") for path in "ABCD"},
        )
        self.assertEqual(len({str(spec["worker_json"]) for spec in specs}), 8)
        self.assertEqual(len({str(spec["log_path"]) for spec in specs}), 8)

    def test_worker_command_identifies_one_fresh_path(self) -> None:
        command = diagnostic.build_worker_command(
            python_executable="python",
            run_id="fixture",
            mesh_level="fine",
            mesh_scale=0.5,
            path_id="C",
            worker_json=Path("worker.json"),
            log_relative_path="programmed_off_solver_logs/fine_C.log",
        )

        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--mesh-level") + 1], "fine")
        self.assertEqual(command[command.index("--path-id") + 1], "C")
        self.assertEqual(command[command.index("--mesh-scale") + 1], "0.5")

    def test_exception_chain_and_equation_inference_keep_original_cause(self) -> None:
        try:
            try:
                raise ValueError("linear factorization failed")
            except ValueError as cause:
                raise RuntimeError("PotentialEquation convergence failure") from cause
        except RuntimeError as error:
            chain = diagnostic.format_exception_cause_chain(error)
            formatted = diagnostic.format_exception_traceback(error)
            equation, evidence = diagnostic.infer_equation_failure(
                error,
                "Iteration 12 PotentialEquation relative error did not converge",
            )

        self.assertIn("RuntimeError", chain)
        self.assertIn("ValueError", chain)
        self.assertIn("linear factorization failed", chain)
        self.assertIn("The above exception was the direct cause", formatted)
        self.assertEqual(equation, "PotentialEquation")
        self.assertIn("Iteration 12", evidence)

    def test_recorder_captures_failed_trial_without_changing_solver_arguments(self) -> None:
        state = {
            "value": diagnostic.RuntimeState(0.0, 0.05, 0.0, 2.0e18),
        }
        calls: list[dict[str, object]] = []

        def read_state() -> diagnostic.RuntimeState:
            return state["value"]

        def fake_solve(*args: object, **kwargs: object) -> None:
            calls.append({"args": args, "kwargs": dict(kwargs)})
            print("Iteration 9 ElectronContinuityEquation convergence failure")
            try:
                raise ValueError("matrix factorization failed")
            except ValueError as cause:
                raise RuntimeError("ElectronContinuityEquation failed") from cause

        memory_window = types.SimpleNamespace(solve_dc=fake_solve)
        helpers = types.SimpleNamespace(solve_dc=fake_solve)
        recorder = diagnostic.SolveTrialRecorder(
            read_state=read_state,
            solve_function=fake_solve,
            run_id="fixture",
            mesh_level="base",
            mesh_scale=1.0,
            path_id="A",
            log_file="programmed_off_solver_logs/fixture.log",
        )
        stage = diagnostic.RampStage(
            3, "off gate ramp", "terminal", "gate", -1.0
        )
        state["value"] = diagnostic.RuntimeState(-0.95, 0.05, 0.0, 2.0e18)

        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with recorder.instrument(memory_window, helpers):
                recorder.begin_stage(stage)
                with self.assertRaises(RuntimeError):
                    memory_window.solve_dc(maximum_iterations=150)
                recorder.end_stage()

        self.assertEqual(calls, [{"args": (), "kwargs": {"maximum_iterations": 150}}])
        self.assertEqual(len(recorder.rows), 1)
        row = recorder.rows[0]
        self.assertFalse(row["solve_success"])
        self.assertEqual(row["requested_gate_bias_V"], -0.95)
        self.assertEqual(row["last_converged_gate_bias_V"], 0.0)
        self.assertEqual(row["trial_voltage_step_V"], -0.95)
        self.assertEqual(row["solve_maximum_iterations_override"], 150)
        self.assertIn("RuntimeError", row["exception_cause_chain"])
        self.assertIn("ValueError", row["exception_cause_chain"])
        self.assertEqual(
            row["possible_equation_failure"], "ElectronContinuityEquation"
        )
        self.assertIn("Iteration 9", row["solver_output_excerpt"])

    def test_noop_stage_is_recorded_for_path_d_state_zero(self) -> None:
        state = diagnostic.RuntimeState(0.0, 0.0, 0.0, 0.0)
        recorder = diagnostic.SolveTrialRecorder(
            read_state=lambda: state,
            solve_function=lambda: None,
            run_id="fixture",
            mesh_level="base",
            mesh_scale=1.0,
            path_id="D",
            log_file="programmed_off_solver_logs/fixture.log",
        )
        stage = diagnostic.build_path_stages("D", STATE_DENSITIES)[0]

        recorder.record_noop(stage)

        self.assertEqual(recorder.rows[0]["trial_role"], "no_op")
        self.assertEqual(recorder.rows[0]["trial_ntrap_step_cm3"], 0.0)
        self.assertTrue(recorder.rows[0]["solve_success"])

    def test_strict_csv_round_trip_preserves_multiline_traceback(self) -> None:
        row = diagnostic._blank_row()
        row.update(
            {
                "record_type": "path_summary",
                "run_id": "fixture",
                "mesh_level": "base",
                "mesh_scale": 1.0,
                "path_id": "A",
                "path_description": diagnostic.PATH_DESCRIPTIONS["A"],
                "exception_traceback": "line one\nline two",
                "path_success": False,
            }
        )
        with tempfile.TemporaryDirectory() as name:
            output = Path(name) / diagnostic.DIAGNOSTIC_CSV_NAME
            diagnostic.write_diagnostic_csv(output, [row])
            with output.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                parsed = list(reader)
                header = tuple(reader.fieldnames or ())

        self.assertEqual(header, diagnostic.DIAGNOSTIC_FIELDS)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["exception_traceback"], "line one\nline two")
        self.assertEqual(parsed[0]["path_success"], "False")

    def test_reference_current_requires_one_exact_coordinate(self) -> None:
        fields = ("state_index", "VGS_V", "VDS_V", "ID_A", "abs_ID_A")
        with tempfile.TemporaryDirectory() as name:
            source = Path(name) / "idvg.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "state_index": 4,
                        "VGS_V": -1.0,
                        "VDS_V": 0.05,
                        "ID_A": -4.0e-22,
                        "abs_ID_A": 4.0e-22,
                    }
                )
            result = diagnostic.load_reference_current(source)

        self.assertEqual(result["reference_ID_A"], -4.0e-22)
        self.assertEqual(result["reference_abs_ID_A"], 4.0e-22)
        self.assertEqual(len(result["reference_current_sha256"]), 64)

    def test_coordinator_keeps_log_json_and_merges_strict_csv(self) -> None:
        commands: list[list[str]] = []

        def fake_subprocess(
            command: list[str],
            *,
            cwd: Path,
            stdout: io.TextIOBase,
            stderr: object,
            check: bool,
        ) -> object:
            commands.append(list(command))
            self.assertEqual(cwd, diagnostic.REPOSITORY_ROOT)
            self.assertFalse(check)
            stdout.write("Iteration 1 PotentialEquation converged\n")
            worker_json = Path(command[command.index("--worker-json") + 1])
            path_id = command[command.index("--path-id") + 1]
            mesh_level = command[command.index("--mesh-level") + 1]
            mesh_scale = float(command[command.index("--mesh-scale") + 1])
            run_id = command[command.index("--run-id") + 1]
            log_file = command[command.index("--log-relative-path") + 1]
            trial = diagnostic._blank_row()
            trial.update(
                {
                    "record_type": "ramp_trial",
                    "run_id": run_id,
                    "mesh_level": mesh_level,
                    "mesh_scale": mesh_scale,
                    "path_id": path_id,
                    "path_description": diagnostic.PATH_DESCRIPTIONS[path_id],
                    "log_file": log_file,
                    "solve_success": True,
                }
            )
            summary = diagnostic._blank_row()
            summary.update(
                {
                    "record_type": "path_summary",
                    "run_id": run_id,
                    "mesh_level": mesh_level,
                    "mesh_scale": mesh_scale,
                    "path_id": path_id,
                    "path_description": diagnostic.PATH_DESCRIPTIONS[path_id],
                    "log_file": log_file,
                    "path_success": True,
                }
            )
            worker_json.parent.mkdir(parents=True, exist_ok=True)
            with worker_json.open("w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "run_id": run_id,
                        "mesh_level": mesh_level,
                        "mesh_scale": mesh_scale,
                        "path_id": path_id,
                        "rows": [trial],
                        "summary": summary,
                    },
                    stream,
                )
            return types.SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as name:
            output = diagnostic.run_fresh_workers(
                name,
                run_id="fixture",
                meshes=("base",),
                paths=("A",),
                subprocess_run=fake_subprocess,
            )
            with output.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            logs = list(
                (Path(name) / diagnostic.SOLVER_LOG_DIRECTORY_NAME).glob("*.log")
            )
            bundles = list(
                (Path(name) / diagnostic.SOLVER_LOG_DIRECTORY_NAME).glob("*.json")
            )
            log_text = logs[0].read_text(encoding="utf-8")

        self.assertEqual(len(commands), 1)
        self.assertEqual(
            [row["record_type"] for row in rows],
            ["ramp_trial", "path_summary"],
        )
        self.assertEqual(len(logs), 1)
        self.assertEqual(len(bundles), 1)
        self.assertIn("PotentialEquation converged", log_text)

    def test_protected_candidate_output_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            diagnostic.validate_output_directory(
                diagnostic.MODULE_DIRECTORY
                / "results"
                / "contact_topology_qc_candidate"
                / "nested"
            )


if __name__ == "__main__":
    unittest.main()
