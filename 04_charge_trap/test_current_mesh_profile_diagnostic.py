"""Focused mock tests for the root-cause diagnostic orchestrator."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_current_mesh_profile_diagnostic as runner


def _task(
    *,
    mesh_level: str = "base",
    mesh_scale: float = 1.0,
    state_index: int = 0,
    bias_name: str = "on",
    target_vgs: float = 3.0,
    do_current: bool = True,
    do_profile: bool = False,
    do_mesh_quality: bool = False,
) -> dict[str, object]:
    return {
        "mesh_level": mesh_level,
        "mesh_scale": mesh_scale,
        "state_index": state_index,
        "bias_name": bias_name,
        "target_VGS_V": target_vgs,
        "do_current": do_current,
        "do_profile": do_profile,
        "do_mesh_quality": do_mesh_quality,
    }


def _successful_bundle(task: dict[str, object]) -> dict[str, object]:
    state_index = int(task["state_index"])
    ntrap = 0.0 if state_index == 0 else 2.0e18
    return {
        **task,
        "worker_process_id": 1234,
        "point_status": {
            "state_index": state_index,
            "state": "State_0_Empty" if state_index == 0 else "State_4_Programmed",
            "bias_name": task["bias_name"],
            "target_VGS_V": task["target_VGS_V"],
            "target_VDS_V": 0.05,
            "target_VS_V": 0.0,
            "target_Ntrap_cm3": ntrap,
            "actual_VGS_V": task["target_VGS_V"],
            "actual_VDS_V": 0.05,
            "actual_VS_V": 0.0,
            "actual_Ntrap_cm3": ntrap,
            "target_reached": True,
            "profile_solution_valid": True,
            "last_converged_revalidated": False,
            "solution_status": "target_converged",
            "point_error_message": "",
        },
        "current_crosschecks": {},
        "current_error_message": "",
        "mesh_quality": None,
        "mesh_quality_error_message": "",
        "profiles": {},
        "profile_errors": {},
    }


def _contact_report(
    contact: str, api_current: float, contributions: tuple[float, ...]
) -> dict[str, object]:
    manual = sum(contributions)
    return {
        "summary": {
            "api_current_A": api_current,
            "manual_current_A": manual,
            "manual_minus_api_A": manual - api_current,
            "absolute_difference_A": abs(manual - api_current),
            "relative_error": abs(manual - api_current) / max(abs(api_current), 1.0e-24),
            "passed": manual == api_current,
        },
        "manual_integral": {
            "manual_current_A": manual,
            "edge_contributions": [
                {
                    "edge_index": edge_index,
                    "contact": contact,
                    "region": "MoS2",
                    "contact_node_index": edge_index,
                    "region_node_index": edge_index + 10,
                    "contact_endpoint": "n0",
                    "orientation_sign": 1.0,
                    "r_cm": 1.0e-6 + edge_index * 1.0e-8,
                    "z_cm": 0.0,
                    "contact_r_cm": 1.0e-6,
                    "contact_z_cm": 0.0,
                    "region_r_cm": 1.0e-6,
                    "region_z_cm": 1.0e-7,
                    "inner_radial_corner_distance_cm": 1.0e-8 * (edge_index + 1),
                    "outer_radial_corner_distance_cm": 2.0e-8 * (edge_index + 1),
                    "ElectronCurrent": contribution / 2.0,
                    "CylindricalEdgeCouple": 2.0,
                    "contribution_A": contribution,
                }
                for edge_index, contribution in enumerate(contributions)
            ],
        },
    }


class TaskIsolationTests(unittest.TestCase):
    def test_minimal_tasks_are_unique_and_cover_only_requested_points(self) -> None:
        tasks = runner.build_tasks()
        self.assertEqual(len(tasks), 8)
        self.assertEqual(len({runner._task_stem(task) for task in tasks}), 8)

        current_points = {
            (task["mesh_level"], task["state_index"], task["bias_name"])
            for task in tasks
            if task["do_current"]
        }
        self.assertEqual(
            current_points,
            {
                ("base", 0, "off"),
                ("base", 0, "on"),
                ("fine", 0, "off"),
                ("fine", 0, "on"),
            },
        )
        quality_points = [task for task in tasks if task["do_mesh_quality"]]
        self.assertEqual(
            [task["mesh_level"] for task in quality_points],
            ["base", "fine", "extra_fine"],
        )
        self.assertTrue(
            all(task["state_index"] == 0 and task["bias_name"] == "on" for task in quality_points)
        )

    def test_fresh_worker_orchestrator_starts_one_command_per_point(self) -> None:
        calls: list[list[str]] = []

        def fake_run(command, *, cwd, check):
            del cwd, check
            command = list(command)
            calls.append(command)
            output = Path(command[command.index("--worker-output") + 1])
            task = {
                "mesh_level": command[command.index("--mesh-level") + 1],
                "mesh_scale": float(command[command.index("--mesh-scale") + 1]),
                "state_index": int(command[command.index("--state-index") + 1]),
                "bias_name": command[command.index("--bias-name") + 1],
                "target_VGS_V": float(command[command.index("--target-vgs") + 1]),
                "do_current": "--do-current" in command,
                "do_profile": "--do-profile" in command,
                "do_mesh_quality": "--do-mesh-quality" in command,
            }
            runner._write_json(output, runner._failed_point_bundle(task, "synthetic"))
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary:
            bundles = runner.run_fresh_workers(
                temporary,
                python_executable="mock-python",
                subprocess_run=fake_run,
            )
        self.assertEqual(len(calls), 8)
        self.assertEqual(len(bundles), 8)
        self.assertTrue(all("--worker" in command for command in calls))
        worker_outputs = {
            command[command.index("--worker-output") + 1] for command in calls
        }
        self.assertEqual(len(worker_outputs), 8)


class CurrentRowTests(unittest.TestCase):
    def test_manual_api_rows_keep_native_weighting_and_corner_ranking(self) -> None:
        bundle = _successful_bundle(_task())
        bundle["current_crosschecks"] = {
            "source": _contact_report("source", -3.0, (-1.0, -2.0)),
            "drain": _contact_report("drain", 3.0, (2.5, 0.5)),
        }
        summary, rows = runner._current_rows(bundle)
        self.assertEqual(summary["source_api_current_A"], -3.0)
        self.assertEqual(summary["source_manual_current_A"], -3.0)
        self.assertEqual(summary["drain_api_current_A"], 3.0)
        self.assertEqual(summary["api_source_plus_drain_continuity_residual_A"], 0.0)
        self.assertEqual(summary["manual_source_plus_drain_continuity_residual_A"], 0.0)

        source = [row for row in rows if row["contact"] == "source"]
        self.assertEqual([row["rank_by_absolute_contribution"] for row in source], [1, 2])
        self.assertEqual(source[0]["signed_current_contribution_A"], -2.0)
        self.assertAlmostEqual(source[0]["absolute_contribution_fraction"], 2.0 / 3.0)
        self.assertEqual(
            source[0]["signed_current_contribution_A"],
            source[0]["electron_current_A_per_cm2"]
            * source[0]["cylindrical_edge_couple_cm2"]
            * source[0]["orientation_sign"],
        )
        tables = runner.build_output_tables([bundle])
        with tempfile.TemporaryDirectory() as temporary:
            written = runner.write_outputs(temporary, tables)
            self.assertEqual(len(written), 4)


class ProfileAndMeshRowsTests(unittest.TestCase):
    def test_failed_target_profile_preserves_actual_revalidated_coordinate(self) -> None:
        task = _task(
            state_index=4,
            bias_name="off",
            target_vgs=-1.0,
            do_current=False,
            do_profile=True,
        )
        bundle = _successful_bundle(task)
        status = bundle["point_status"]
        status.update(
            {
                "actual_VGS_V": -0.9671875,
                "target_reached": False,
                "profile_solution_valid": True,
                "last_converged_revalidated": True,
                "solution_status": "last_converged_revalidated_after_target_failure",
                "point_error_message": "minimum voltage step reached",
            }
        )
        sample = {
            "contact": "source",
            "region": "MoS2",
            "plane_axis": "z",
            "plane_coordinate_cm": 0.0,
            "inward_coordinate_sign": 1.0,
            "requested_distance_nm": 0.0,
            "within_region_depth": True,
            "sample_available": True,
            "edge_index": 1,
            "r_cm": 1.1e-6,
            "z_cm": 0.0,
            "Potential_V": -0.2,
            "Electrons_cm3": 1.0e10,
            "ElectricField_V_per_cm": 2.0,
            "ElectronCurrent": 3.0,
            "CylindricalEdgeCouple": 4.0,
            "ElectronCurrent_times_CylindricalEdgeCouple_A": 12.0,
        }
        bundle["profiles"] = {"source": [sample], "drain": [dict(sample, contact="drain")]}
        rows = runner._profile_rows(bundle)
        self.assertTrue(rows)
        self.assertTrue(all(not row["target_reached"] for row in rows))
        self.assertTrue(all(row["profile_solution_valid"] for row in rows))
        self.assertTrue(all(row["last_converged_revalidated"] for row in rows))
        self.assertTrue(all(row["actual_VGS_V"] == -0.9671875 for row in rows))

    def test_mesh_quality_aggregate_and_contact_nodes_are_flattened(self) -> None:
        task = _task(do_current=False, do_mesh_quality=True)
        bundle = _successful_bundle(task)
        distribution = {
            "count": 2,
            "minimum": 1.0,
            "maximum": 2.0,
            "mean": 1.5,
            "median": 1.5,
            "p05": 1.05,
            "p95": 1.95,
            "sum": 3.0,
        }
        bundle["mesh_quality"] = {
            "region": "MoS2",
            "region_node_count": 3,
            "region_edge_count": 3,
            "edge_couple_model": "CylindricalEdgeCouple",
            "triangle_quality": {
                "triangle_count": 1,
                "valid_triangle_count": 1,
                "degenerate_triangle_count": 0,
                "minimum_angle_deg": 30.0,
                "maximum_angle_deg": 100.0,
                "maximum_aspect_ratio": 4.0,
                "obtuse_triangle_count": 1,
                "obtuse_fraction": 1.0,
                "minimum_angle_distribution_deg": distribution,
                "aspect_ratio_distribution": distribution,
            },
            "actual_region_edge_spacing": {
                "radial_cm": distribution,
                "axial_cm": distribution,
                "edge_length_cm": distribution,
            },
            "contacts": {
                contact: {
                    "contact_node_count": 1,
                    "active_incident_edge_count": 2,
                    "incident_CylindricalEdgeCouple": distribution,
                    "total_region_edge_valence": distribution,
                    "active_incident_edge_valence": distribution,
                    "node_valence": [
                        {
                            "node_index": 0,
                            "r_cm": 1.0e-6,
                            "z_cm": 0.0,
                            "total_region_edge_valence": 3,
                            "active_incident_edge_valence": 2,
                        }
                    ],
                }
                for contact in ("source", "drain")
            },
        }
        rows = runner._mesh_quality_rows(bundle)
        self.assertEqual([row["row_type"] for row in rows].count("region_summary"), 1)
        self.assertEqual([row["row_type"] for row in rows].count("contact_summary"), 2)
        self.assertEqual([row["row_type"] for row in rows].count("contact_node"), 2)
        self.assertEqual(rows[0]["triangle_minimum_angle_deg"], 30.0)
        self.assertEqual(rows[0]["triangle_maximum_aspect_ratio"], 4.0)


class OutputSafetyTests(unittest.TestCase):
    def test_public_candidate_paths_are_rejected(self) -> None:
        for protected in runner.PROTECTED_DIRECTORIES:
            with self.assertRaises(ValueError):
                runner.validate_output_directory(protected)

    def test_four_csvs_are_written_once_without_overwrite(self) -> None:
        tables = {
            runner.CURRENT_SUMMARY_FILENAME: [],
            runner.CURRENT_EDGE_FILENAME: [],
            runner.MESH_QUALITY_FILENAME: [],
            runner.PROFILE_FILENAME: [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = runner.write_outputs(temporary, tables)
            first = {path.name: path.read_bytes() for path in paths}
            with self.assertRaises(FileExistsError):
                runner.write_outputs(temporary, tables)
            self.assertEqual(set(first), set(tables))
            for path in paths:
                with path.open("r", encoding="utf-8", newline="") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
