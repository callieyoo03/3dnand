"""Pure unit tests for the candidate-only local mesh runner."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import run_local_mesh_convergence_candidate as runner
import local_mesh_family


class LocalMeshRunnerTests(unittest.TestCase):
    def test_task_matrix_uses_fresh_probe_point_and_repeat_processes(self):
        tasks = runner.build_tasks()
        self.assertEqual(len(tasks), 14)
        self.assertEqual(sum(task["mode"] == "probe" for task in tasks), 4)
        primary = [task for task in tasks
                   if task["mode"] == "point" and task["run_role"] == "primary"]
        repeats = [task for task in tasks if task.get("run_role") == "repeat"]
        self.assertEqual(len(primary), 8)
        self.assertEqual(len(repeats), 2)
        self.assertEqual({task["mesh_level"] for task in repeats}, {"local_ultra_fine"})
        self.assertEqual(
            {(task["state_index"], task["bias_name"]) for task in repeats},
            {(0, "on"), (4, "off")},
        )
        self.assertEqual(len({runner._task_stem(task) for task in tasks}), len(tasks))

    def test_worker_command_is_candidate_only(self):
        task = {"mode": "point", "mesh_level": "local_fine", "run_role": "primary",
                "state_index": 0, "bias_name": "on"}
        command = runner.build_worker_command(task, Path("worker.json"), "python-test")
        self.assertEqual(command[0], "python-test")
        self.assertIn("--worker-mode", command)
        self.assertIn("--worker-output", command)
        self.assertNotIn("--all", command)
        self.assertFalse(any("shared_data" in item for item in command))

    def test_csv_writes_literal_NaN_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            runner.write_csv(path, ("name", "value"),
                             ({"name": "missing", "value": math.nan},))
            self.assertEqual(path.read_text(encoding="utf-8"), "name,value\nmissing,NaN\n")
            with self.assertRaises(FileExistsError):
                runner.write_csv(path, ("name", "value"), ())

    def test_full_dc_dispatch_is_fail_closed(self):
        calls: list[str] = []

        def dispatch():
            calls.append("called")
            return 7

        result = runner.dispatch_full_dc_if_eligible({"passed": False}, dispatcher=dispatch)
        self.assertFalse(result["executed"])
        self.assertEqual(calls, [])
        result = runner.dispatch_full_dc_if_eligible(
            {"passed": True, "full_sweep_allowed": False}, dispatcher=dispatch
        )
        self.assertFalse(result["executed"])
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, "no candidate-only"):
            runner.dispatch_full_dc_if_eligible(
                {"passed": True, "full_sweep_allowed": True}
            )
        result = runner.dispatch_full_dc_if_eligible(
            {"passed": True, "full_sweep_allowed": True}, dispatcher=dispatch
        )
        self.assertTrue(result["executed"])
        self.assertEqual(result["result"], 7)
        self.assertEqual(calls, ["called"])

    def test_point_flattening_keeps_raw_matrix_and_required_validation(self):
        matrix = {
            f"{measured}:{perturbed}": (index + 1) * 1.0e-18
            for index, (measured, perturbed) in enumerate(
                (pair for measured in runner.TERMINALS
                 for perturbed in runner.TERMINALS
                 for pair in ((measured, perturbed),))
            )
        }
        matrices = {
            text: {"values": dict(matrix)} for text in ("0.0005", "0.001", "0.002")
        }
        derivative = [
            {"perturbed_terminal": terminal, "delta_voltage_V": delta,
             "residual_F": 1.0e-25, "tolerance_F": 1.0e-22, "passed": True}
            for delta in runner.DELTA_VOLTAGES_V
            for terminal in runner.TERMINALS
        ]
        gauge_rows = {
            terminal: {"row_sum_F": 0.0, "row_vs_zero": {"tolerance": 1.0e-22},
                       "passed": True}
            for terminal in runner.TERMINALS
        }
        bundle = {
            "mode": "point", "mesh_level": "local_ultra_fine", "level_order": 3,
            "state_index": 0, "bias_name": "on", "run_role": "primary",
            "actual_biases_after_worker": {"gate": 3.0, "drain": 0.05, "source": 0.0},
            "contact_currents_A": {"source": -2.0e-6, "drain": 2.0e-6},
            "points": [{
                "state_index": 0, "state": "State_0_Empty", "bias_name": "on",
                "ntrap_cm3": 0.0,
                "direct": {"Qg_contact_C": 1.0e-17, "Qd_contact_C": -4.0e-18,
                           "Qs_contact_C": -5.0e-18, "Qmobile_C": -2.0e-18,
                           "Qtrap_C": 0.0, "Qfixed_C": 0.0},
                "nominal_matrix_F": matrix, "matrices": matrices,
                "global_gauss": {"residual_C": 0.0, "scale_C": 1.0e-17,
                                 "tolerance_C": 1.0e-24, "passed": True},
                "derivative_gauss_rows": derivative,
                "gauge_row_validation": {"rows": gauge_rows, "passed": True},
                "passed": True, "error_message": "",
            }],
            "topology": {"passed": True, "contacts": {
                "source": {"edge_count": 4, "runtime_area_cm2": 1.38e-12},
                "drain": {"edge_count": 4, "runtime_area_cm2": 1.38e-12},
            }},
            "mesh_line_validation": {"passed": True, "position_hash": "abc"},
            "runtime_invariants_passed": True,
            "family_position_hash_sha256": "abc",
            "final_reference_validation": {"passed": True},
            "runtime_seconds": 1.0, "worker_process_id": 42,
        }
        row = runner._point_row(bundle)
        self.assertTrue(row["target_reached"])
        self.assertTrue(row["continuity_passed"])
        self.assertTrue(row["global_gauss_passed"])
        self.assertTrue(row["derivative_gauss_passed"])
        self.assertEqual(row["derivative_gauss_row_count"], 9)
        self.assertTrue(row["derivative_gauss_gate_dv_0p0005_passed"])
        self.assertTrue(row["derivative_gauss_source_dv_0p002_passed"])
        self.assertTrue(row["gauge_row_sum_passed"])
        self.assertTrue(row["supported_delta_sensitivity_passed"])
        self.assertEqual(row["Cgg_F"], matrix["gate:gate"])
        self.assertEqual(row["Css_F"], matrix["source:source"])
        self.assertEqual(row["raw_Qd_C"], -4.0e-18)
        self.assertTrue(row["converged"])

    def test_profile_local_spacing_comes_from_edge_not_terminal_area(self):
        bundle = {
            "mesh_level": "local_base", "level_order": 0, "run_role": "primary",
            "state_index": 0, "bias_name": "on",
            "profiles": [{"contact": "source", "region": "MoS2",
                          "requested_distance_nm": 0.25, "sample_available": True,
                          "edge_index": 2, "edge_z0_cm": 0.0, "edge_z1_cm": 2.5e-8,
                          "r_cm": 1.1e-6, "z_cm": 2.5e-8,
                          "Potential_V": 0.75, "Electrons_cm3": 2.0e15,
                          "ElectricField_V_per_cm": 3.0e4,
                          "ElectronCurrent": 4.0e5,
                          "CylindricalEdgeCouple": 5.0e-13,
                          "ElectronCurrent_times_CylindricalEdgeCouple_A": 2.0e-7}],
        }
        row = runner._profile_rows(bundle)[0]
        self.assertAlmostEqual(row["local_axial_spacing_cm"], 2.5e-8)
        self.assertAlmostEqual(row["local_axial_spacing_nm"], 0.25)
        self.assertAlmostEqual(row["sample_r_nm"], 11.0)
        self.assertEqual(row["potential_V"], 0.75)
        self.assertEqual(row["electron_current_A_per_cm2"], 4.0e5)
        self.assertEqual(row["electron_current_times_couple_A"], 2.0e-7)
        self.assertNotIn("area", " ".join(row))

    def test_protected_hash_audit_covers_prior_results_and_ignores_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "04_charge_trap"
            results = module / "results"
            for path, contents in (
                (root / "shared_data" / "public.csv", "public"),
                (root / "05_program_erase" / "model.py", "erase-preserved"),
                (root / "_doc_review" / "note.txt", "local"),
                (module / "run_memory_window.py", "tracked"),
                (results / "prior_candidate" / "old.csv", "old"),
                (results / ".local_mesh_scratch" / "new.json", "scratch"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(contents, encoding="utf-8")
            with mock.patch.multiple(
                runner,
                REPOSITORY_ROOT=root,
                MODULE_DIRECTORY=module,
                RESULTS_DIRECTORY=results,
                CANDIDATE_DIRECTORY=results / "local_mesh_convergence_candidate",
            ):
                baseline = runner.capture_protected_hashes()
                self.assertFalse(any("scratch" in name for name in baseline))
                (results / "prior_candidate" / "old.csv").write_text("changed", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "old.csv"):
                    runner.assert_protected_hashes(baseline)

    def test_fake_subprocess_runner_creates_one_output_per_task(self):
        def fake_run(command, **kwargs):
            output = Path(command[command.index("--worker-output") + 1])
            mode = command[command.index("--worker-mode") + 1]
            level = command[command.index("--mesh-level") + 1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"mode": mode, "mesh_level": level}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory) / "workers"
            bundles = runner.run_fresh_workers(
                worker_directory, subprocess_run=fake_run, python_executable="python-test"
            )
            self.assertEqual(len(bundles), 14)
            self.assertEqual(len(list(worker_directory.glob("*.json"))), 14)

    def test_failed_gate_never_creates_full_dc_worker_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            worker_directory = Path(directory) / "full_dc_workers"
            bundles = runner.run_fresh_full_dc_workers(
                worker_directory,
                {"passed": False, "full_sweep_allowed": False},
                python_executable="python-test",
            )
            self.assertEqual(bundles, [])
            self.assertFalse(worker_directory.exists())

    def test_profile_coverage_requires_all_finite_planes(self):
        rows = [
            {
                "mesh_level": level,
                "contact": contact,
                "requested_distance_nm": distance,
                "sample_available": True,
                "potential_V": 0.0,
                "electrons_cm3": 1.0,
                "electric_field_V_per_cm": 2.0,
                "electron_current_A_per_cm2": 3.0,
                "local_axial_spacing_cm": 1.0e-9,
            }
            for level in local_mesh_family.LEVELS
            for contact in ("source", "drain")
            for distance in runner.PROFILE_DISTANCES_NM
        ]
        self.assertTrue(runner.validate_profile_coverage(rows)["passed"])
        self.assertFalse(runner.validate_profile_coverage(rows[:-1])["passed"])
        rows[0]["potential_V"] = math.nan
        self.assertFalse(runner.validate_profile_coverage(rows)["passed"])


if __name__ == "__main__":
    unittest.main()
