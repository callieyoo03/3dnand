"""DEVSIM-free tests for isolated full-terminal Q/C candidate work."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import full_terminal_qc_candidate as qc
import run_full_terminal_qc_candidate as runner
from terminal_charge import integrate_cylindrical_node_quantity


STATE = {
    "state_index": 4,
    "state": "State_4_Programmed",
    "ntrap_cm3": 2.0e18,
    "nsheet_cm2": 1.0e12,
}


class CandidatePhysicsMathTests(unittest.TestCase):
    def test_synthetic_cylindrical_integration_uses_supplied_volumes(self) -> None:
        self.assertEqual(
            integrate_cylindrical_node_quantity(
                [2.0, -3.0, 4.0],
                [0.5, 2.0, 0.25],
            ),
            -4.0,
        )

    def test_scaled_gauss_residual_uses_max_component_and_floor(self) -> None:
        result = qc.calculate_scaled_gauss_residual(
            {
                "Qg_contact_C": 5.0,
                "Qd_contact_C": -1.0,
                "Qs_contact_C": -2.0,
                "Qmobile_C": 0.5,
                "Qtrap_C": -3.0,
                "Qfixed_C": 0.5,
            },
            configured_charge_floor_C=10.0,
        )
        self.assertEqual(result["global_residual_C"], 0.0)
        self.assertEqual(result["charge_scale_C"], 10.0)
        self.assertEqual(result["relative_residual"], 0.0)

    def test_contact_sign_convention_is_explicit_and_not_manually_flipped(self) -> None:
        self.assertEqual(qc.CONTACT_INWARD_NORMALS_RZ["gate"], (-1.0, 0.0))
        self.assertEqual(qc.CONTACT_INWARD_NORMALS_RZ["drain"], (0.0, -1.0))
        self.assertEqual(qc.CONTACT_INWARD_NORMALS_RZ["source"], (0.0, 1.0))

    def test_ward_dutton_weights_and_mobile_charge_close(self) -> None:
        result = qc.calculate_ward_dutton_mobile_partition(
            signed_mobile_charge_density_C_cm3=[-1.0, -2.0, -4.0],
            cylindrical_node_volumes_cm3=[1.0, 1.0, 1.0],
            axial_coordinates_cm=[2.0, 3.0, 6.0],
            source_position_cm=2.0,
            drain_position_cm=6.0,
        )
        self.assertEqual(result["minimum_source_weight"], 0.0)
        self.assertEqual(result["maximum_source_weight"], 1.0)
        self.assertEqual(result["minimum_drain_weight"], 0.0)
        self.assertEqual(result["maximum_drain_weight"], 1.0)
        self.assertTrue(result["weights_valid"])
        self.assertAlmostEqual(
            result["Qs_mobile_C"] + result["Qd_mobile_C"],
            result["Qmobile_channel_C"],
        )
        self.assertEqual(result["partition_residual_C"], 0.0)
        self.assertNotEqual(result["Qs_mobile_C"], result["Qd_mobile_C"])

    def test_vector_central_difference_preserves_component_identity(self) -> None:
        result = qc.vector_central_difference(
            {"Qg_contact_C": 4.002, "Qd_contact_C": -2.001},
            {"Qg_contact_C": 3.998, "Qd_contact_C": -1.999},
            0.001,
        )
        self.assertAlmostEqual(result["Qg_contact_C"], 2.0)
        self.assertAlmostEqual(result["Qd_contact_C"], -1.0)

    def test_matrix_row_and_column_sums_use_direct_cij_convention(self) -> None:
        matrix = {
            ("gate", "gate"): 2.0,
            ("gate", "drain"): -1.0,
            ("gate", "source"): -1.0,
            ("drain", "gate"): -1.0,
            ("drain", "drain"): 1.0,
            ("drain", "source"): 0.0,
            ("source", "gate"): -1.0,
            ("source", "drain"): 0.0,
            ("source", "source"): 1.0,
        }
        sums = qc.calculate_matrix_sums(matrix)
        self.assertEqual(sums["row_sums_F"], dict.fromkeys(qc.TERMINALS, 0.0))
        self.assertEqual(sums["column_sums_F"], dict.fromkeys(qc.TERMINALS, 0.0))
        validation = qc.evaluate_matrix_sum_tolerances(
            matrix,
            absolute_tolerance_F=1.0e-12,
            relative_tolerance=0.0,
        )
        self.assertTrue(validation["row_sums_passed"])
        self.assertTrue(validation["column_sums_passed"])

    def test_common_mode_gauge_comparison_is_like_for_like(self) -> None:
        reference = {name: float(index + 1) for index, name in enumerate(runner.ALL_CHARGE_NAMES)}
        same = dict(reference)
        report = runner.compare_charge_vectors(reference, same)
        self.assertTrue(report["passed"])
        changed = dict(reference)
        changed["Qg_contact_C"] += 1.0
        self.assertFalse(runner.compare_charge_vectors(reference, changed)["passed"])

    def test_common_mode_gauge_validation_ramps_and_restores_all_biases(self) -> None:
        base_biases = {"gate": 3.0, "drain": 0.05, "source": 0.0}
        runtime_biases = dict(base_biases)
        charges = {
            name: float(index + 1) * 1.0e-18
            for index, name in enumerate(runner.ALL_CHARGE_NAMES)
        }
        ramp_calls: list[float] = []

        def fake_ramp(base, shift, *, device):
            del device
            ramp_calls.append(float(shift))
            runtime_biases.update(
                {terminal: float(base[terminal]) + float(shift) for terminal in qc.TERMINALS}
            )
            return dict(runtime_biases)

        with (
            patch.object(runner, "ramp_common_mode", side_effect=fake_ramp),
            patch.object(runner, "_read_runtime_biases", side_effect=lambda device: dict(runtime_biases)),
            patch.object(runner, "_assert_trap_density"),
            patch.object(qc, "extract_direct_contact_charge_components", return_value=dict(charges)),
        ):
            report = runner.validate_gauge_invariance(
                base_biases=base_biases,
                baseline_components=charges,
                expected_trap_density_cm3=0.0,
                device=qc.DEVICE_NAME,
            )

        self.assertTrue(report["passed"])
        self.assertEqual(ramp_calls, [-0.01, 0.01, 0.01, -0.01])
        self.assertEqual(runtime_biases, base_biases)

    def test_unsupported_status_keeps_qg_and_nan_qd_qs(self) -> None:
        components = {
            "Qg_contact_C": 5.0,
            "Qd_contact_C": -2.0,
            "Qs_contact_C": -3.0,
        }
        unsupported = qc.apply_terminal_charge_status(
            components,
            qc.TERMINAL_STATUS_UNSUPPORTED,
        )
        self.assertEqual(unsupported["Qg_C"], 5.0)
        self.assertTrue(math.isnan(unsupported["Qd_C"]))
        self.assertTrue(math.isnan(unsupported["Qs_C"]))
        self.assertEqual(unsupported["supported_quantities"], "Qg")

        supported = qc.apply_terminal_charge_status(
            components,
            qc.TERMINAL_STATUS_SUPPORTED,
        )
        self.assertEqual(supported["Qd_C"], -2.0)
        self.assertEqual(supported["Qs_C"], -3.0)


class CandidateSchemaTests(unittest.TestCase):
    def test_all_candidate_schemas_are_unique_and_contain_required_columns(self) -> None:
        schemas = (
            qc.TERMINAL_CHARGE_FULL_FIELDNAMES,
            qc.CAPACITANCE_MATRIX_FULL_FIELDNAMES,
            qc.CAPACITANCE_SUMMARY_FULL_FIELDNAMES,
            qc.DIRECT_CONTACT_VALIDATION_FIELDNAMES,
            qc.WARD_DUTTON_FIELDNAMES,
            qc.METHOD_COMPARISON_FIELDNAMES,
            qc.MESH_REFINEMENT_QC_FIELDNAMES,
        )
        for schema in schemas:
            with self.subTest(first=schema[0]):
                self.assertEqual(len(schema), len(set(schema)))
                self.assertIn("state_index", schema)
                self.assertIn("state", schema)

        self.assertTrue(
            {
                "VGS_V",
                "VDS_V",
                "VS_V",
                "Qg_C",
                "Qd_C",
                "Qs_C",
                "global_residual_C",
                "relative_residual",
                "terminal_charge_status",
            }.issubset(qc.TERMINAL_CHARGE_FULL_FIELDNAMES)
        )
        self.assertTrue(
            {
                "measured_terminal",
                "perturbed_terminal",
                "capacitance_F",
                "delta_voltage_V",
            }.issubset(qc.CAPACITANCE_MATRIX_FULL_FIELDNAMES)
        )
        self.assertTrue(
            {
                "comparison",
                "reference_mesh_level",
                "candidate_mesh_level",
                "reference_value",
                "candidate_value",
            }.issubset(qc.MESH_REFINEMENT_QC_FIELDNAMES)
        )

    def test_output_set_is_exact_and_outside_shared_data(self) -> None:
        self.assertEqual(len(runner.OUTPUT_FILENAMES), 8)
        self.assertEqual(len(set(runner.OUTPUT_FILENAMES.values())), 8)
        self.assertEqual(
            set(runner.OUTPUT_FILENAMES.values()),
            {
                "terminal_charge_full_candidate.csv",
                "capacitance_matrix_full_candidate.csv",
                "capacitance_summary_full_candidate.csv",
                "direct_contact_flux_validation.csv",
                "ward_dutton_mobile_partition.csv",
                "method_comparison.csv",
                "mesh_refinement_qc.csv",
                "full_terminal_qc_README.md",
            },
        )
        self.assertNotIn(
            runner.SHARED_DATA_DIRECTORY.resolve(),
            runner.CANDIDATE_DIRECTORY.resolve().parents,
        )

    def test_strict_writer_preserves_schema_and_nan_text(self) -> None:
        components = {
            "Qg_contact_C": 5.0,
            "Qd_contact_C": -2.0,
            "Qs_contact_C": -3.0,
            "Qmobile_C": -1.0,
            "Qtrap_C": 0.0,
            "Qfixed_C": 1.0,
        }
        row = qc.build_terminal_charge_full_row(
            STATE,
            VGS_V=-1.0,
            VDS_V=0.05,
            VS_V=0.0,
            components=components,
            terminal_charge_status=qc.TERMINAL_STATUS_UNSUPPORTED,
            mesh_level="base",
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.csv"
            qc.write_deterministic_csv(
                path,
                qc.TERMINAL_CHARGE_FULL_FIELDNAMES,
                [row],
            )
            with path.open(newline="", encoding="utf-8") as input_file:
                reader = csv.DictReader(input_file)
                rows = list(reader)
        self.assertEqual(tuple(reader.fieldnames or ()), qc.TERMINAL_CHARGE_FULL_FIELDNAMES)
        self.assertEqual(rows[0]["Qd_C"], "nan")
        self.assertEqual(rows[0]["Qs_C"], "nan")


class MeshIsolationTests(unittest.TestCase):
    @staticmethod
    def geometry() -> dict[str, float]:
        return {
            "r_axis": 0.0,
            "r_core": 1.0e-6,
            "r_mos2": 1.2e-6,
            "r_tox": 1.5e-6,
            "r_trap": 2.0e-6,
            "r_block": 3.6e-6,
            "r_gate_outer": 3.8e-6,
            "r_air_outer": 4.0e-6,
            "z_source": 0.0,
            "z_drain": 1.0e-5,
        }

    def test_candidate_monkeypatch_changes_only_ps_and_restores_function(self) -> None:
        calls: list[dict[str, object]] = []

        def add_line(**kwargs):
            calls.append(dict(kwargs))

        module = types.SimpleNamespace(add_2d_mesh_line=add_line)
        original = module.add_2d_mesh_line
        line_log: list[dict[str, float | str]] = []
        geometry = self.geometry()
        with runner.scaled_runtime_mesh_lines(module, 0.5, line_log):
            for (direction, position), (_, spacing) in zip(
                runner.expected_mesh_line_positions(geometry),
                runner.EXPECTED_BASE_SPACING_CM,
            ):
                module.add_2d_mesh_line(
                    mesh="gaa_mesh",
                    dir=direction,
                    pos=position,
                    ps=spacing,
                )
        self.assertIs(module.add_2d_mesh_line, original)
        report = runner.validate_mesh_line_log(line_log, geometry, 0.5)
        self.assertTrue(report["passed"])
        self.assertEqual(len(calls), 10)
        for call, logged in zip(calls, line_log):
            self.assertEqual(call["dir"], logged["direction"])
            self.assertEqual(call["pos"], logged["position_cm"])
            self.assertEqual(call["ps"], logged["scaled_spacing_cm"])


class WorkerOrchestrationTests(unittest.TestCase):
    def test_worker_commands_are_fresh_and_use_expected_mesh_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            calls: list[list[str]] = []

            def fake_run(command, **kwargs):
                del kwargs
                calls.append(list(command))
                output = Path(command[command.index("--worker-output") + 1])
                level = command[command.index("--mesh-level") + 1]
                scale = float(command[command.index("--mesh-scale") + 1])
                output.write_text(
                    json.dumps({"mesh_level": level, "mesh_scale": scale}),
                    encoding="utf-8",
                )
                return types.SimpleNamespace(returncode=0)

            bundles = runner.run_fresh_workers(
                temporary,
                python_executable="candidate-python",
                subprocess_run=fake_run,
            )
        self.assertEqual(tuple(bundles), ("base", "base_repeat", "fine"))
        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [float(call[call.index("--mesh-scale") + 1]) for call in calls],
            [1.0, 1.0, 0.5],
        )
        self.assertEqual(len({call[call.index("--worker-output") + 1] for call in calls}), 3)

    def test_public_qc_hash_guard_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = tuple(Path(temporary) / name for name in ("a.csv", "b.csv", "c.csv"))
            for index, path in enumerate(paths):
                path.write_text(f"value\n{index}\n", encoding="utf-8")
            with patch.object(runner, "PUBLIC_QC_FILES", paths):
                hashes = runner.capture_public_qc_hashes()
                runner.assert_public_qc_hashes_unchanged(hashes)
                paths[1].write_text("value\nchanged\n", encoding="utf-8")
                with self.assertRaisesRegex(RuntimeError, "shared_data Q/C"):
                    runner.assert_public_qc_hashes_unchanged(hashes)

    def test_repeated_run_comparison_is_deterministic(self) -> None:
        def point(state_index: int, gate_voltage: float) -> dict[str, object]:
            matrix = {
                f"{measured}:{perturbed}": 0.0
                for measured in qc.TERMINALS
                for perturbed in qc.TERMINALS
            }
            sums = qc.calculate_matrix_sums(
                {
                    tuple(name.split(":")): value
                    for name, value in matrix.items()
                }
            )
            return {
                "state_index": state_index,
                "state": f"State_{state_index}",
                "VGS_V": gate_voltage,
                "VDS_V": 0.05,
                "VS_V": 0.0,
                "direct": {
                    **{name: 0.0 for name in runner.ALL_CHARGE_NAMES},
                    "global_residual_C": 0.0,
                },
                "matrices": {"0.001": {"values": matrix, "sums": sums}},
            }

        left = {
            "points": [
                point(state_index, gate_voltage)
                for state_index in range(5)
                for gate_voltage in (-1.0, 3.0)
            ]
        }
        right = json.loads(json.dumps(left))
        report = runner.compare_worker_points(
            left,
            right,
            comparison_name="repeat",
            relative_tolerance=1.0e-4,
        )
        self.assertTrue(report["passed"])
        self.assertTrue(all(row["passed"] for row in report["rows"]))
        self.assertEqual(len(report["rows"]), 220)
        self.assertTrue(
            runner.comparison_passed_for_point(report, left["points"][0])
        )

        right["points"][0]["direct"]["Qg_contact_C"] = 1.0e-20
        changed = runner.compare_worker_points(
            left,
            right,
            comparison_name="repeat",
            relative_tolerance=1.0e-4,
        )
        self.assertFalse(changed["passed"])
        self.assertFalse(
            runner.comparison_passed_for_point(changed, left["points"][0])
        )
        self.assertIn(
            "repeated run failed 1/22 comparison rows",
            runner.comparison_failure_summary_for_point(
                "repeated run", changed, left["points"][0]
            ),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
