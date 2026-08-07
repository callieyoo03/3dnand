"""DEVSIM-free tests for the isolated contact-topology DC candidate runner."""

from __future__ import annotations

import csv
import json
import math
import tempfile
import types
import unittest
from pathlib import Path

import run_contact_topology_dc_candidate as candidate
import state_characterization_config as config


def iv_row(
    *,
    state_index: int = 0,
    vgs: float = 0.0,
    vds: float = 0.05,
    current: float = 1.0e-12,
    converged: bool = True,
) -> dict[str, object]:
    row = {
        "state_index": state_index,
        "state": config.MEMORY_STATES[state_index]["state"],
        "ntrap_cm3": config.MEMORY_STATES[state_index]["ntrap_cm3"],
        "nsheet_cm2": config.volume_to_sheet_density(
            config.MEMORY_STATES[state_index]["ntrap_cm3"]
        ),
        "trap_charge_density_C_cm3": config.electron_density_to_charge_density(
            config.MEMORY_STATES[state_index]["ntrap_cm3"]
        ),
        "VGS_V": vgs,
        "VDS_V": vds,
        "ID_A": current if converged else math.nan,
        "abs_ID_A": abs(current) if converged else math.nan,
        "converged": converged,
        "error_message": "" if converged else "fixture failure",
    }
    assert tuple(row) == config.IDVG_FIELDNAMES
    return row


def metric_row(
    *,
    state_index: int = 0,
    vds: float = 0.05,
    vth: float = 0.6,
    ss: float = 65.0,
    ion: float = 1.0e-8,
) -> dict[str, object]:
    row = {
        "state_index": state_index,
        "state": config.MEMORY_STATES[state_index]["state"],
        "ntrap_cm3": config.MEMORY_STATES[state_index]["ntrap_cm3"],
        "nsheet_cm2": config.volume_to_sheet_density(
            config.MEMORY_STATES[state_index]["ntrap_cm3"]
        ),
        "VDS_V": vds,
        "Vth_V": vth,
        "threshold_current_A": config.VTH_TARGET_CURRENT_A,
        "vth_success": True,
        "vth_error": "",
        "SS_mV_dec": ss,
        "ss_r_squared": 0.999,
        "ss_point_count": 5,
        "ss_success": True,
        "ss_error": "",
        "Ion_A": ion,
        "ion_vgs_V": config.ION_VGS_V,
        "ion_vds_V": config.ION_VDS_V,
        "Ioff_A": 1.0e-20,
        "ioff_vgs_V": config.IOFF_VGS_V,
        "ioff_vds_V": config.IOFF_VDS_V,
        "on_off_ratio": ion / 1.0e-20,
        "gm_max_S": 2.0e-8,
        "VGS_at_gm_max_V": 0.7,
        "ss_current_min_A": config.SS_CURRENT_MIN_A,
        "ss_current_max_A": config.SS_CURRENT_MAX_A,
        "ss_minimum_point_count": config.SS_MINIMUM_POINT_COUNT,
        "current_floor_A": config.CURRENT_FLOOR_A,
    }
    assert tuple(row) == config.METRICS_FIELDNAMES
    return row


class NumericComparisonTests(unittest.TestCase):
    def test_curve_relative_threshold_and_absolute_near_zero_handling(self) -> None:
        passing = candidate.compare_numeric_values(
            1.0e-12,
            1.049e-12,
            absolute_tolerance=candidate.CURVE_WARNING_ABSOLUTE_A,
            relative_tolerance=candidate.CURVE_WARNING_RELATIVE,
            relative_scale_floor=candidate.CURVE_WARNING_ABSOLUTE_A,
        )
        failing = candidate.compare_numeric_values(
            1.0e-12,
            1.051e-12,
            absolute_tolerance=candidate.CURVE_WARNING_ABSOLUTE_A,
            relative_tolerance=candidate.CURVE_WARNING_RELATIVE,
            relative_scale_floor=candidate.CURVE_WARNING_ABSOLUTE_A,
        )
        near_zero = candidate.compare_numeric_values(
            0.0,
            5.0e-19,
            absolute_tolerance=candidate.CURVE_WARNING_ABSOLUTE_A,
            relative_tolerance=candidate.CURVE_WARNING_RELATIVE,
            relative_scale_floor=candidate.CURVE_WARNING_ABSOLUTE_A,
        )
        self.assertTrue(passing["within_warning_threshold"])
        self.assertFalse(failing["within_warning_threshold"])
        self.assertTrue(near_zero["within_warning_threshold"])
        self.assertEqual(
            near_zero["relative_scale"], candidate.CURVE_WARNING_ABSOLUTE_A
        )

    def test_nonfinite_input_is_explicit_failure(self) -> None:
        result = candidate.compare_numeric_values(
            1.0,
            math.nan,
            absolute_tolerance=0.0,
            relative_tolerance=0.05,
        )
        self.assertTrue(result["reference_finite"])
        self.assertFalse(result["candidate_finite"])
        self.assertFalse(result["within_warning_threshold"])
        self.assertTrue(math.isnan(result["absolute_difference"]))


class CurveRegressionTests(unittest.TestCase):
    def test_union_preserves_missing_points_and_convergence_columns(self) -> None:
        reference = [
            iv_row(vgs=0.0, current=1.0e-12),
            iv_row(vgs=0.1, current=2.0e-12),
        ]
        candidate_rows = [
            iv_row(vgs=0.0, current=1.0e-12),
            iv_row(vgs=0.2, current=3.0e-12),
        ]
        rows = candidate.build_curve_regression_rows(
            "idvg", reference, candidate_rows
        )
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(tuple(row) == candidate.REGRESSION_FIELDNAMES for row in rows))
        missing_candidate = next(row for row in rows if row["VGS_V"] == 0.1)
        missing_reference = next(row for row in rows if row["VGS_V"] == 0.2)
        self.assertFalse(missing_candidate["candidate_present"])
        self.assertIn("missing_candidate_point", missing_candidate["warning_reason"])
        self.assertFalse(missing_reference["reference_present"])
        self.assertIn("missing_reference_point", missing_reference["warning_reason"])
        self.assertIn("reference_monotonic", missing_candidate)
        self.assertIn("candidate_converged", missing_candidate)

    def test_pointwise_five_percent_warning(self) -> None:
        reference = [iv_row(current=1.0e-12)]
        passing = candidate.build_curve_regression_rows(
            "idvg", reference, [iv_row(current=1.049e-12)]
        )[0]
        failing = candidate.build_curve_regression_rows(
            "idvg", reference, [iv_row(current=1.051e-12)]
        )[0]
        self.assertFalse(passing["warning"])
        self.assertTrue(failing["warning"])
        self.assertEqual(failing["relative_tolerance"], 0.05)

    def test_monotonicity_is_curve_level_and_tolerates_absolute_floor(self) -> None:
        voltages = [
            round(config.IDVG_VGS_START_V + index * config.IDVG_VGS_STEP_V, 12)
            for index in range(41)
        ]
        monotonic = [
            iv_row(vgs=voltage, current=(index + 1) * 1.0e-12)
            for index, voltage in enumerate(voltages)
        ]
        decreasing = [dict(row) for row in monotonic]
        decreasing[-1]["ID_A"] = 1.0e-12
        decreasing[-1]["abs_ID_A"] = 1.0e-12
        self.assertTrue(
            candidate.evaluate_curve_monotonicity(monotonic, "idvg")[(0, 0.05)]
        )
        self.assertFalse(
            candidate.evaluate_curve_monotonicity(decreasing, "idvg")[(0, 0.05)]
        )


class MetricRegressionTests(unittest.TestCase):
    @staticmethod
    def by_quantity(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
        return {str(row["quantity"]): row for row in rows}

    def test_primary_warning_thresholds(self) -> None:
        reference = [metric_row()]
        within = self.by_quantity(
            candidate.build_metrics_regression_rows(
                reference,
                [metric_row(vth=0.6019, ss=65.0 * 1.019, ion=1.0e-8 * 1.019)],
            )
        )
        outside = self.by_quantity(
            candidate.build_metrics_regression_rows(
                reference,
                [metric_row(vth=0.6021, ss=65.0 * 1.021, ion=1.0e-8 * 1.021)],
            )
        )
        for quantity in candidate.PRIMARY_METRIC_FIELDS:
            self.assertFalse(within[quantity]["warning"], quantity)
            self.assertTrue(outside[quantity]["warning"], quantity)
        self.assertEqual(
            outside["Vth_V"]["absolute_tolerance"],
            candidate.VTH_WARNING_ABSOLUTE_V,
        )
        self.assertEqual(
            outside["Ion_A"]["relative_tolerance"],
            candidate.ION_SS_WARNING_RELATIVE,
        )

    def test_vth_monotonicity_column_uses_all_five_states(self) -> None:
        reference = [
            metric_row(state_index=index, vth=0.6 + index * 0.1)
            for index in range(5)
        ]
        candidate_rows = [dict(row) for row in reference]
        rows = candidate.build_metrics_regression_rows(reference, candidate_rows)
        vth_rows = [row for row in rows if row["quantity"] == "Vth_V"]
        self.assertEqual(len(vth_rows), 5)
        self.assertTrue(all(row["reference_monotonic"] for row in vth_rows))
        self.assertTrue(all(row["candidate_monotonic"] for row in vth_rows))


class MeshAndOrchestrationTests(unittest.TestCase):
    def test_mesh_primary_rows_and_schema(self) -> None:
        fine = [metric_row()]
        extra = [metric_row(vth=0.601, ss=65.5, ion=1.01e-8)]
        rows = candidate.build_mesh_convergence_rows(fine, extra)
        self.assertEqual(
            [row["quantity"] for row in rows],
            list(candidate.PRIMARY_METRIC_FIELDS),
        )
        self.assertTrue(
            all(tuple(row) == candidate.MESH_CONVERGENCE_FIELDNAMES for row in rows)
        )

    def test_mesh_scaler_changes_only_ps_and_restores(self) -> None:
        calls: list[dict[str, object]] = []

        def add_line(**kwargs):
            calls.append(dict(kwargs))

        module = types.SimpleNamespace(add_2d_mesh_line=add_line)
        original = module.add_2d_mesh_line
        log: list[dict[str, object]] = []
        with candidate.scaled_runtime_mesh_lines(module, 0.25, log):
            module.add_2d_mesh_line(
                mesh="mesh", dir="x", pos=1.0e-6, ps=2.0e-8
            )
        self.assertIs(module.add_2d_mesh_line, original)
        self.assertEqual(calls[0]["pos"], 1.0e-6)
        self.assertEqual(calls[0]["ps"], 5.0e-9)
        self.assertEqual(log[0]["base_spacing_cm"], 2.0e-8)

    def test_fresh_worker_contract_uses_three_mesh_levels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            commands: list[list[str]] = []

            def fake_run(command, **kwargs):
                del kwargs
                commands.append(list(command))
                level = command[command.index("--mesh-level") + 1]
                scale = float(command[command.index("--mesh-scale") + 1])
                output = Path(command[command.index("--worker-output") + 1])
                output.write_text(
                    json.dumps({"mesh_level": level, "mesh_scale": scale}),
                    encoding="utf-8",
                )
                return types.SimpleNamespace(returncode=0)

            bundles = candidate.run_fresh_workers(
                temporary,
                python_executable="fixture-python",
                subprocess_run=fake_run,
            )
        self.assertEqual(tuple(bundles), ("base", "fine", "extra_fine"))
        self.assertEqual(
            [float(command[command.index("--mesh-scale") + 1]) for command in commands],
            [1.0, 0.5, 0.25],
        )

    def test_worker_json_preserves_explicit_nan_token(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bundle.json"
            candidate.write_worker_bundle(
                path,
                {"mesh_level": "base", "mesh_scale": 1.0, "value": math.nan},
            )
            decoded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(decoded["value"], "nan")

    def test_refinement_requires_fixed_geometry_and_increasing_counts(self) -> None:
        bundles = {
            level: {
                "geometry": {"r_core": 1.0e-6},
                "mesh_counts": {
                    "total_region_node_count": 100 * (index + 1),
                    "total_element_count": 200 * (index + 1),
                },
            }
            for index, (level, _) in enumerate(candidate.WORKER_SPECS)
        }
        self.assertTrue(candidate.validate_mesh_refinement_bundles(bundles)["passed"])
        bundles["extra_fine"]["geometry"] = {"r_core": 2.0e-6}
        self.assertFalse(candidate.validate_mesh_refinement_bundles(bundles)["passed"])


class OutputContractTests(unittest.TestCase):
    def test_expected_full_grid_sizes_and_isolated_directory(self) -> None:
        idvg_count = (
            len(config.MEMORY_STATES)
            * len(config.IDVG_VDS_VALUES_V)
            * int(
                round(
                    (config.IDVG_VGS_STOP_V - config.IDVG_VGS_START_V)
                    / config.IDVG_VGS_STEP_V
                )
                + 1
            )
        )
        idvd_count = (
            len(config.MEMORY_STATES)
            * len(config.IDVD_VGS_VALUES_V)
            * int(
                round(
                    (config.IDVD_VDS_STOP_V - config.IDVD_VDS_START_V)
                    / config.IDVD_VDS_STEP_V
                )
                + 1
            )
        )
        self.assertEqual(idvg_count, 615)
        self.assertEqual(idvd_count, 385)
        self.assertNotIn(
            config.SHARED_DATA_DIRECTORY.resolve(),
            candidate.CANDIDATE_DIRECTORY.resolve().parents,
        )

    def test_schemas_are_unique_and_include_required_audit_columns(self) -> None:
        for schema in (
            candidate.REGRESSION_FIELDNAMES,
            candidate.MESH_CONVERGENCE_FIELDNAMES,
        ):
            self.assertEqual(len(schema), len(set(schema)))
        self.assertTrue(
            {
                "reference_converged",
                "candidate_converged",
                "reference_monotonic",
                "candidate_monotonic",
                "absolute_difference",
                "relative_difference",
                "warning",
            }.issubset(candidate.REGRESSION_FIELDNAMES)
        )

    def test_deterministic_writer_and_optional_qc_emission(self) -> None:
        regression_row = candidate.build_curve_regression_rows(
            "idvg", [iv_row()], [iv_row()]
        )[0]
        mesh_rows = candidate.build_mesh_convergence_rows(
            [metric_row()], [metric_row()]
        )
        tables = {"regression": [regression_row], "mesh": mesh_rows}
        with tempfile.TemporaryDirectory() as temporary:
            first, second = candidate.emit_regression_for_qc(tables, temporary)
            first_bytes = first.read_bytes()
            candidate.emit_regression_for_qc(tables, temporary)
            self.assertEqual(first.read_bytes(), first_bytes)
            with first.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(
                    tuple(reader.fieldnames or ()), candidate.REGRESSION_FIELDNAMES
                )
            self.assertTrue(second.is_file())

    def test_writes_under_shared_data_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "may not write"):
            candidate.write_deterministic_csv(
                config.SHARED_DATA_DIRECTORY / "forbidden_fixture.csv",
                ("value",),
                [{"value": 1}],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
