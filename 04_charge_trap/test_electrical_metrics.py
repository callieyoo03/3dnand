"""Unit tests for DEVSIM-independent electrical metric extraction.

Run directly from the repository root or this directory::

    python -B 04_charge_trap/test_electrical_metrics.py
    python -B test_electrical_metrics.py

The CSV writer test replaces solver-facing imports with inert modules while
loading ``state_sweep_helpers``.  Consequently this suite remains runnable on
a normal Python installation without DEVSIM.
"""

from __future__ import annotations

import csv
import importlib
import math
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from electrical_metrics import (
    calculate_transconductance,
    extract_state_metrics,
    extract_subthreshold_swing,
    extract_threshold_voltage,
    interpolate_current_at_bias,
)
import state_characterization_config as config


METRICS_FIELDNAMES = config.METRICS_FIELDNAMES


def _iv_rows(
    currents_A: list[float] | tuple[float, ...],
    *,
    vds_V: float = 0.05,
) -> list[dict[str, object]]:
    """Build a complete, converged five-point ID-VG curve for one state."""

    gate_voltages_V = (-1.0, 0.0, 1.0, 2.0, 3.0)
    if len(currents_A) != len(gate_voltages_V):
        raise ValueError("The test curve requires exactly five currents.")

    return [
        {
            "state_index": 0,
            "state": "State_0_Empty",
            "ntrap_cm3": 0.0,
            "nsheet_cm2": 0.0,
            "VGS_V": vgs_V,
            "VDS_V": vds_V,
            "ID_A": current_A,
            "converged": True,
            "error_message": "",
        }
        for vgs_V, current_A in zip(gate_voltages_V, currents_A)
    ]


def _extract_complete_metrics(
    rows: list[dict[str, object]],
    *,
    vds_V: float = 0.05,
) -> dict[str, object]:
    """Use stable test criteria to produce one standard metrics row."""

    return extract_state_metrics(
        rows,
        vds_V=vds_V,
        threshold_current_A=1.0e-10,
        ss_current_min_A=config.SS_CURRENT_MIN_A,
        ss_current_max_A=config.SS_CURRENT_MAX_A,
        ss_minimum_point_count=config.SS_MINIMUM_POINT_COUNT,
        ion_vgs_V=3.0,
        ioff_vgs_V=-1.0,
        ion_vds_V=0.05,
        ioff_vds_V=0.05,
        current_floor_A=config.CURRENT_FLOOR_A,
    )


def _load_helpers_without_devsim():
    """Load the real sweep helpers without importing DEVSIM."""

    unused = lambda *args, **kwargs: None  # noqa: E731 - compact inert stub

    memory_window_stub = types.ModuleType("run_memory_window")
    for name in (
        "solve_dc",
        "set_terminal_bias",
        "get_drain_current_A",
        "create_output_voltage_list",
        "adaptive_voltage_ramp",
        "adaptive_trap_density_ramp",
        "initialize_device",
        "solve_empty_state",
    ):
        setattr(memory_window_stub, name, unused)
    memory_window_stub.DEFAULT_TRAP_INITIAL_STEP_CM3 = 1.0
    memory_window_stub.DEFAULT_TRAP_MINIMUM_STEP_CM3 = 1.0

    trap_models_stub = types.ModuleType("trap_models")
    trap_models_stub.get_trap_state_information = unused

    solver_stubs = {
        "run_memory_window": memory_window_stub,
        "trap_models": trap_models_stub,
    }
    sys.modules.pop("state_sweep_helpers", None)
    with patch.dict(sys.modules, solver_stubs):
        helper_module = importlib.import_module("state_sweep_helpers")
    sys.modules.pop("state_sweep_helpers", None)
    return helper_module


def _load_csv_writer_without_devsim():
    """Load the real strict CSV writer without importing solver modules."""

    return _load_helpers_without_devsim().write_standard_csv


def _load_idvg_runner_without_devsim():
    """Load the ID-VG orchestration against the inert helper module."""

    helper_module = _load_helpers_without_devsim()
    sys.modules.pop("run_idvg_by_state", None)
    with patch.dict(sys.modules, {"state_sweep_helpers": helper_module}):
        runner_module = importlib.import_module("run_idvg_by_state")
    sys.modules.pop("run_idvg_by_state", None)
    return runner_module, helper_module


def _load_integrated_runner_without_devsim():
    """Load the integrated sanity evaluator against inert sweep modules."""

    helper_module = _load_helpers_without_devsim()

    idvg_stub = types.ModuleType("run_idvg_by_state")
    idvg_stub.run_idvg_by_state = lambda *args, **kwargs: None
    idvg_stub.summarize_convergence = lambda rows: (
        sum(bool(row["converged"]) for row in rows),
        sum(not bool(row["converged"]) for row in rows),
    )

    idvd_stub = types.ModuleType("run_idvd_by_state")
    idvd_stub.run_idvd_by_state = lambda *args, **kwargs: None
    idvd_stub.summarize_convergence = idvg_stub.summarize_convergence

    sys.modules.pop("run_state_characterization", None)
    with patch.dict(
        sys.modules,
        {
            "state_sweep_helpers": helper_module,
            "run_idvg_by_state": idvg_stub,
            "run_idvd_by_state": idvd_stub,
        },
    ):
        integrated_module = importlib.import_module(
            "run_state_characterization"
        )
    sys.modules.pop("run_state_characterization", None)
    return integrated_module


class ThresholdVoltageTests(unittest.TestCase):
    """Constant-current Vth cases required by the characterization spec."""

    def test_known_exponential_curve_uses_log_interpolation(self) -> None:
        result = extract_threshold_voltage(
            [0.0, 1.0],
            [1.0e-12, 1.0e-8],
            threshold_current_A=1.0e-10,
        )

        self.assertTrue(result["vth_success"])
        self.assertAlmostEqual(result["Vth_V"], 0.5, places=14)
        self.assertEqual(result["vth_error"], "")

    def test_missing_threshold_crossing_returns_nan_and_error(self) -> None:
        result = extract_threshold_voltage(
            [-1.0, 0.0, 1.0],
            [1.0e-14, 2.0e-14, 3.0e-14],
            threshold_current_A=1.0e-10,
        )

        self.assertFalse(result["vth_success"])
        self.assertTrue(math.isnan(result["Vth_V"]))
        self.assertIn("not crossed", result["vth_error"])

    def test_signed_current_is_converted_to_magnitude(self) -> None:
        result = extract_threshold_voltage(
            [0.0, 1.0],
            [-1.0e-12, -1.0e-8],
            threshold_current_A=1.0e-10,
        )

        self.assertTrue(result["vth_success"])
        self.assertAlmostEqual(result["Vth_V"], 0.5, places=14)

    def test_unsorted_gate_voltage_input_is_sorted(self) -> None:
        result = extract_threshold_voltage(
            [1.0, -1.0, 0.0],
            [1.0e-8, 1.0e-12, 1.0e-10],
            threshold_current_A=1.0e-10,
        )

        self.assertTrue(result["vth_success"])
        self.assertAlmostEqual(result["Vth_V"], 0.0, places=14)


class SubthresholdSwingTests(unittest.TestCase):
    """Subthreshold regression success and failure cases."""

    def test_known_exponential_slope(self) -> None:
        vgs_values = [0.0, 0.1, 0.2, 0.3]
        currents_A = [1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10]
        result = extract_subthreshold_swing(
            vgs_values,
            currents_A,
            current_min_A=1.0e-13,
            current_max_A=1.0e-10,
            minimum_point_count=4,
        )

        self.assertTrue(result["ss_success"])
        self.assertAlmostEqual(result["SS_mV_dec"], 100.0, places=12)
        self.assertAlmostEqual(result["ss_r_squared"], 1.0, places=14)
        self.assertEqual(result["ss_point_count"], 4)

    def test_insufficient_fit_points_returns_nan_and_count(self) -> None:
        result = extract_subthreshold_swing(
            [0.0, 0.1, 0.2],
            [1.0e-13, 1.0e-12, 1.0e-11],
            current_min_A=1.0e-13,
            current_max_A=1.0e-10,
            minimum_point_count=4,
        )

        self.assertFalse(result["ss_success"])
        self.assertTrue(math.isnan(result["SS_mV_dec"]))
        self.assertEqual(result["ss_point_count"], 3)
        self.assertIn("at least 4", result["ss_error"])

    def test_duplicate_gate_voltage_is_rejected_clearly(self) -> None:
        result = extract_subthreshold_swing(
            [0.0, 0.1, 0.1, 0.2],
            [1.0e-13, 1.0e-12, 2.0e-12, 1.0e-11],
            current_min_A=1.0e-13,
            current_max_A=1.0e-10,
            minimum_point_count=3,
        )

        self.assertFalse(result["ss_success"])
        self.assertTrue(math.isnan(result["SS_mV_dec"]))
        self.assertIn("duplicate VGS", result["ss_error"])


class BiasInterpolationTests(unittest.TestCase):
    """Exact and interpolated Ion/Ioff sampling cases."""

    def test_exact_bias_point_uses_current_magnitude(self) -> None:
        result = interpolate_current_at_bias(
            [-1.0, 0.0, 1.0],
            [-1.0e-13, -2.0e-12, -3.0e-11],
            target_vgs_V=0.0,
        )

        self.assertTrue(result["interpolation_success"])
        self.assertTrue(result["used_exact_point"])
        self.assertEqual(result["current_A"], 2.0e-12)

    def test_missing_bias_point_uses_linear_interpolation(self) -> None:
        result = interpolate_current_at_bias(
            [0.0, 1.0],
            [-2.0e-12, -6.0e-12],
            target_vgs_V=0.25,
        )

        self.assertTrue(result["interpolation_success"])
        self.assertFalse(result["used_exact_point"])
        self.assertAlmostEqual(result["current_A"], 3.0e-12, places=25)

    def test_zero_ioff_produces_nan_on_off_ratio(self) -> None:
        rows = _iv_rows((0.0, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10))
        metrics = _extract_complete_metrics(rows)

        self.assertEqual(metrics["Ioff_A"], 0.0)
        self.assertTrue(math.isnan(metrics["on_off_ratio"]))


class TransconductanceTests(unittest.TestCase):
    """Signed derivative and maximum-magnitude summary cases."""

    def test_gm_lengths_sign_and_maximum_location(self) -> None:
        result = calculate_transconductance(
            [2.0, 0.0, 3.0, 1.0],
            [-4.0, 0.0, -10.0, -1.0],
        )

        self.assertTrue(result["gm_success"])
        self.assertEqual(result["VGS_V"], [0.0, 1.0, 2.0, 3.0])
        self.assertEqual(len(result["gm_S"]), 4)
        self.assertEqual(len(result["abs_gm_S"]), 4)
        self.assertTrue(all(value < 0.0 for value in result["gm_S"]))
        self.assertEqual(result["gm_max_S"], 6.0)
        self.assertEqual(result["VGS_at_gm_max_V"], 3.0)

    def test_nonuniform_grid_uses_centered_three_point_derivative(self) -> None:
        result = calculate_transconductance(
            [0.0, 1.0, 3.0],
            [0.0, 1.0, 9.0],
        )

        self.assertTrue(result["gm_success"])
        self.assertAlmostEqual(result["gm_S"][1], 2.0, places=14)


class StateMetricsAndCsvTests(unittest.TestCase):
    """Full-state selection and public CSV contract tests."""

    def test_state_metrics_use_requested_vds_and_fixed_ion_ioff_vds(self) -> None:
        read_rows = _iv_rows(
            (1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10),
            vds_V=0.05,
        )
        summary_rows = _iv_rows(
            (2.0e-14, 2.0e-13, 2.0e-12, 2.0e-11, 2.0e-10),
            vds_V=0.10,
        )
        metrics = _extract_complete_metrics(
            read_rows + summary_rows,
            vds_V=0.10,
        )

        self.assertEqual(metrics["VDS_V"], 0.10)
        self.assertEqual(metrics["Ion_A"], 1.0e-10)
        self.assertEqual(metrics["Ioff_A"], 1.0e-14)
        self.assertAlmostEqual(metrics["Vth_V"], 2.6989700043360187)

    def test_metrics_keys_match_schema_and_strict_writer(self) -> None:
        rows = _iv_rows(
            (1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10)
        )
        metrics = _extract_complete_metrics(rows)

        self.assertEqual(tuple(metrics), METRICS_FIELDNAMES)
        self.assertEqual(set(metrics), set(METRICS_FIELDNAMES))

        write_standard_csv = _load_csv_writer_without_devsim()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "metrics.csv"
            returned_path = write_standard_csv(
                output_path,
                METRICS_FIELDNAMES,
                [metrics],
            )
            self.assertEqual(returned_path, output_path)

            with output_path.open(encoding="utf-8", newline="") as csv_file:
                csv_rows = list(csv.reader(csv_file))
            self.assertEqual(tuple(csv_rows[0]), METRICS_FIELDNAMES)
            self.assertEqual(len(csv_rows), 2)

    def test_strict_writer_rejects_missing_and_extra_keys_before_write(self) -> None:
        rows = _iv_rows(
            (1.0e-14, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10)
        )
        metrics = _extract_complete_metrics(rows)
        write_standard_csv = _load_csv_writer_without_devsim()

        with tempfile.TemporaryDirectory() as temporary_directory:
            for case_name, malformed_row, expected_error in (
                (
                    "missing",
                    {key: value for key, value in metrics.items() if key != "Vth_V"},
                    r"missing=",
                ),
                (
                    "extra",
                    {**metrics, "unexpected": 1},
                    r"extra=",
                ),
            ):
                with self.subTest(case=case_name):
                    output_path = Path(temporary_directory) / f"{case_name}.csv"
                    with self.assertRaisesRegex(
                        ValueError,
                        re.escape(expected_error),
                    ):
                        write_standard_csv(
                            output_path,
                            METRICS_FIELDNAMES,
                            [malformed_row],
                        )
                    self.assertFalse(output_path.exists())

    def test_all_public_csv_schemas_accept_exact_row_keys(self) -> None:
        write_standard_csv = _load_csv_writer_without_devsim()
        iv_row = {
            "state_index": 0,
            "state": "State_0_Empty",
            "ntrap_cm3": 0.0,
            "nsheet_cm2": 0.0,
            "trap_charge_density_C_cm3": -0.0,
            "VGS_V": 0.0,
            "VDS_V": 0.05,
            "ID_A": 1.0e-12,
            "abs_ID_A": 1.0e-12,
            "converged": True,
            "error_message": "",
        }
        map_row = {
            "state_index": 0,
            "state": "State_0_Empty",
            "ntrap_cm3": 0.0,
            "nsheet_cm2": 0.0,
            "qsheet_C_cm2": -0.0,
            "VDS_V": 0.05,
            "Vth_V": 0.6,
            "delta_Vth_from_empty_V": 0.0,
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            for filename, fieldnames, row in (
                ("idvg.csv", config.IDVG_FIELDNAMES, iv_row),
                ("idvd.csv", config.IDVD_FIELDNAMES, iv_row),
                ("map.csv", config.MEMORY_STATE_MAP_FIELDNAMES, map_row),
            ):
                with self.subTest(filename=filename):
                    self.assertEqual(tuple(row), fieldnames)
                    write_standard_csv(
                        Path(temporary_directory) / filename,
                        fieldnames,
                        [row],
                    )


class SweepOrchestrationTests(unittest.TestCase):
    """Solver-state tracking and bounded output-point retry cases."""

    def test_valid_same_bias_is_a_no_op(self) -> None:
        helpers = _load_helpers_without_devsim()
        point = helpers.OperatingPoint(gate_voltage_V=-1.0)

        with (
            patch.object(helpers, "_read_terminal_bias", return_value=-1.0),
            patch.object(helpers, "solve_dc") as solve_mock,
        ):
            result = helpers.ramp_terminal(point, "gate", -1.0)

        self.assertEqual(result, -1.0)
        self.assertTrue(point.solution_valid)
        solve_mock.assert_not_called()

    def test_failed_ramp_invalidates_then_same_bias_revalidates(self) -> None:
        helpers = _load_helpers_without_devsim()
        point = helpers.OperatingPoint()

        with (
            patch.object(helpers, "_read_terminal_bias", return_value=0.0),
            patch.object(
                helpers,
                "adaptive_voltage_ramp",
                side_effect=RuntimeError("continuation failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "continuation failed"):
                helpers.ramp_terminal(point, "gate", 0.1)

        self.assertFalse(point.solution_valid)

        with (
            patch.object(helpers, "_read_terminal_bias", return_value=0.0),
            patch.object(helpers, "solve_dc") as solve_mock,
        ):
            result = helpers.ramp_terminal(point, "gate", 0.0)

        self.assertEqual(result, 0.0)
        self.assertTrue(point.solution_valid)
        solve_mock.assert_called_once_with()

    def test_idvg_point_retry_stops_after_success(self) -> None:
        runner, helpers = _load_idvg_runner_without_devsim()
        point = helpers.OperatingPoint()
        ramp_mock = Mock(
            side_effect=(
                RuntimeError("first"),
                RuntimeError("second"),
                None,
            )
        )

        with patch.object(runner, "ramp_terminal", ramp_mock):
            runner._ramp_idvg_point(
                point,
                target_vgs_V=-0.9,
                label="unit-test point",
            )

        self.assertEqual(ramp_mock.call_count, 3)

    def test_idvg_point_retry_is_bounded(self) -> None:
        runner, helpers = _load_idvg_runner_without_devsim()
        point = helpers.OperatingPoint()
        ramp_mock = Mock(side_effect=RuntimeError("still failing"))

        with (
            patch.object(runner, "ramp_terminal", ramp_mock),
            patch.object(runner.config, "IDVG_POINT_MAX_ATTEMPTS", 3),
        ):
            with self.assertRaisesRegex(RuntimeError, "failed after 3 attempts"):
                runner._ramp_idvg_point(
                    point,
                    target_vgs_V=-0.9,
                    label="unit-test point",
                )

        self.assertEqual(ramp_mock.call_count, 3)

    def test_floor_limited_on_off_is_valid_sanity_output(self) -> None:
        rows = _iv_rows((0.0, 1.0e-13, 1.0e-12, 1.0e-11, 1.0e-10))
        metrics = _extract_complete_metrics(rows)
        self.assertTrue(math.isnan(float(metrics["on_off_ratio"])))

        integrated = _load_integrated_runner_without_devsim()
        sanity = integrated.evaluate_physical_sanity(
            idvg_rows=(),
            idvd_rows=(),
            metrics_rows=(metrics,),
            programmed_state_index=0,
        )

        self.assertTrue(sanity["metrics_complete_ok"])
        self.assertEqual(sanity["metric_errors"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
