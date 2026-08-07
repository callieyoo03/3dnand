"""Pure tests for the graded-mesh candidate orchestrator."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import graded_local_mesh_family as family
import run_graded_local_mesh_convergence_candidate as runner


def _quality_rows() -> list[dict]:
    rows: list[dict] = []
    expected_area = 1.3823007675795089e-12
    for order, level in enumerate(family.LEVELS):
        spec = family.get_level_spec(level)
        axial_max = family.axial_interval_spacings_cm(level)[-1]
        edge_count = int(round(2.0e-7 / spec.radial_spacing_cm))
        rows.append(
            {
                "mesh_level": level,
                "row_type": "region_summary",
                "region": "MoS2",
                "minimum_angle_deg": runner.MOS2_REFERENCE_MINIMUM_ANGLE_DEG,
                "maximum_aspect_ratio": runner.MOS2_REFERENCE_MAXIMUM_ASPECT_RATIO,
                "degenerate_triangle_count": 0,
                "radial_spacing_min_cm": spec.radial_spacing_cm,
                "radial_spacing_max_cm": spec.radial_spacing_cm,
                "axial_spacing_min_cm": spec.axial_spacing_cm,
                "axial_spacing_max_cm": axial_max,
                "quality_passed": True,
            }
        )
        for direction in ("radial", "axial"):
            rows.append(
                {
                    "mesh_level": level,
                    "row_type": "adjacent_spacing_ratio",
                    "spacing_direction": direction,
                    "adjacent_spacing_ratio_max": 2.0,
                    "adjacent_spacing_ratio_passed": True,
                }
            )
        for index in range(family.RADIAL_ANCHOR_COUNT):
            rows.append(
                {
                    "mesh_level": level,
                    "row_type": "graded_actual_radial_anchor",
                    "contact": "",
                    "anchor_index": index,
                    "quality_passed": True,
                }
            )
        for contact in ("source", "drain"):
            normal_z = -1.0 if contact == "source" else 1.0
            rows.append(
                {
                    "mesh_level": level,
                    "row_type": "topology",
                    "contact": contact,
                    "edge_count": edge_count,
                    "expected_area_cm2": expected_area,
                    "normal_z": normal_z,
                    "contact_area_passed": True,
                    "quality_passed": True,
                }
            )
            rows.append(
                {
                    "mesh_level": level,
                    "row_type": "contact_summary",
                    "contact": contact,
                    "contact_node_count": edge_count + 1,
                    "active_incident_edge_count": 2 * edge_count + 1,
                    "valence_minimum": 1,
                    "incident_couple_sum_cm2": expected_area,
                    "quality_passed": True,
                }
            )
            for index, distance in enumerate(family.AXIAL_DISTANCE_NM):
                rows.append(
                    {
                        "mesh_level": level,
                        "row_type": "graded_actual_axial_anchor",
                        "contact": contact,
                        "anchor_index": index,
                        "distance_from_contact_nm": distance,
                        "quality_passed": True,
                    }
                )
    return rows


class GradedRunnerTests(unittest.TestCase):
    def test_task_matrix_is_four_probes_eight_primary_two_repeat(self):
        tasks = runner.build_tasks()
        self.assertEqual(len(tasks), 14)
        self.assertEqual(sum(task["mode"] == "probe" for task in tasks), 4)
        self.assertEqual(sum(task.get("run_role") == "primary" for task in tasks), 8)
        repeats = [task for task in tasks if task.get("run_role") == "repeat"]
        self.assertEqual(len(repeats), 2)
        self.assertEqual(
            {task["mesh_level"] for task in repeats}, {"graded_ultra_fine"}
        )
        self.assertEqual(len({_task_key(task) for task in tasks}), len(tasks))

    def test_worker_command_has_no_public_writer_switch(self):
        task = {
            "mode": "point", "mesh_level": "graded_fine",
            "run_role": "primary", "state_index": 0, "bias_name": "on",
        }
        command = runner.build_worker_command(
            task, Path("point.json"), python_executable="python-test"
        )
        self.assertEqual(command[0], "python-test")
        self.assertNotIn("--all", command)
        self.assertFalse(any("shared_data" in item for item in command))

    def test_empty_and_collapsed_coordinates_fail_closed(self):
        empty = runner._adjacent_ratio_summary([])
        collapsed = runner._adjacent_ratio_summary([1.0, 1.0, 1.0])
        for summary in (empty, collapsed):
            self.assertEqual(summary["adjacent_spacing_ratio_count"], 0)
            self.assertTrue(math.isnan(summary["adjacent_spacing_ratio_max"]))

    def test_quality_family_passes_constant_shape_and_fails_degradation(self):
        rows = _quality_rows()
        summaries = runner.evaluate_graded_quality_family(rows, family.LEVELS)
        self.assertTrue(all(row["passed"] for row in summaries.values()))
        self.assertEqual(
            summaries["graded_ultra_fine"]["expected_contact_edge_count"], 32
        )

        degraded = _quality_rows()
        ultra = next(
            row for row in degraded
            if row.get("mesh_level") == "graded_ultra_fine"
            and row.get("row_type") == "region_summary"
            and row.get("region") == "MoS2"
        )
        ultra["minimum_angle_deg"] -= 0.1
        summaries = runner.evaluate_graded_quality_family(
            degraded, family.LEVELS
        )
        self.assertFalse(summaries["graded_ultra_fine"]["passed"])

    def test_dispatch_and_full_dc_task_creation_are_conditional(self):
        called: list[bool] = []

        def dispatch():
            called.append(True)
            return {"passed": True}

        result = runner.dispatch_full_dc_if_eligible(
            {"passed": False, "full_sweep_allowed": False}, dispatcher=dispatch
        )
        self.assertFalse(result["executed"])
        self.assertEqual(called, [])
        result = runner.dispatch_full_dc_if_eligible(
            {"passed": True, "full_sweep_allowed": True}, dispatcher=dispatch
        )
        self.assertTrue(result["executed"])
        self.assertEqual(called, [True])

    def test_fake_subprocess_creates_exact_reference_worker_matrix(self):
        def fake_run(command, **kwargs):
            output = Path(command[command.index("--worker-output") + 1])
            mode = command[command.index("--worker-mode") + 1]
            level = command[command.index("--mesh-level") + 1]
            output.write_text(
                json.dumps({"mode": mode, "mesh_level": level}),
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            bundles = runner.run_fresh_workers(
                Path(directory) / "workers",
                subprocess_run=fake_run,
                python_executable="python-test",
            )
            self.assertEqual(len(bundles), 14)

    def test_protected_hashes_cover_old_candidates_and_doc_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            module = root / "04_charge_trap"
            results = module / "results"
            for path, text in (
                (root / "shared_data" / "public.csv", "public"),
                (root / "05_program_erase" / "erase.py", "erase"),
                (root / "_doc_review" / "local.docx", "local"),
                (module / "run_memory_window.py", "tracked"),
                (results / "old_candidate" / "old.csv", "old"),
                (results / ".graded_local_mesh_scratch" / "new.json", "scratch"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
            with mock.patch.multiple(
                runner,
                REPOSITORY_ROOT=root,
                MODULE_DIRECTORY=module,
                RESULTS_DIRECTORY=results,
                CANDIDATE_DIRECTORY=(
                    results / "graded_local_mesh_convergence_candidate"
                ),
            ):
                hashes = runner.capture_protected_hashes()
                self.assertTrue(any("_doc_review" in name for name in hashes))
                self.assertFalse(any("scratch" in name for name in hashes))
                (results / "old_candidate" / "old.csv").write_text(
                    "changed", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "old.csv"):
                    runner.assert_protected_hashes(hashes)


def _task_key(task: dict) -> tuple:
    return (
        task.get("mode"), task.get("mesh_level"), task.get("run_role"),
        task.get("state_index"), task.get("bias_name"),
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
