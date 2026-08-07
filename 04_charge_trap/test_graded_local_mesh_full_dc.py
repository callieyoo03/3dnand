"""Pure policy tests for candidate-only graded local-mesh full-DC helpers."""

from __future__ import annotations

import copy
import math
import unittest

import graded_local_mesh_full_dc as full_dc


STATES = tuple(
    {"state_index": index, "state": f"State_{index}", "ntrap_cm3": index * 5.0e17}
    for index in range(5)
)
VDS_VALUES = (0.05,)


def _metrics(ioff_A: float = 1.0e-22) -> list[dict[str, object]]:
    rows = []
    for state in STATES:
        index = int(state["state_index"])
        ion = 1.0e-6 / (index + 1)
        rows.append(
            {
                **state,
                "nsheet_cm2": float(state["ntrap_cm3"]) * 5.0e-7,
                "VDS_V": 0.05,
                "Vth_V": 0.2 + 0.1 * index,
                "threshold_current_A": 1.0e-8,
                "vth_success": True,
                "vth_error": "",
                "SS_mV_dec": 70.0,
                "ss_r_squared": 0.999,
                "ss_point_count": 8,
                "ss_minimum_point_count": 5,
                "ss_success": True,
                "ss_error": "",
                "Ion_A": ion,
                "ion_vgs_V": 3.0,
                "ion_vds_V": 0.05,
                "Ioff_A": ioff_A,
                "ioff_vgs_V": -1.0,
                "ioff_vds_V": 0.05,
                "on_off_ratio": ion / ioff_A,
                "gm_max_S": 2.0e-6 / (index + 1),
                "VGS_at_gm_max_V": 2.5,
                "ss_current_min_A": 1.0e-13,
                "ss_current_max_A": 1.0e-8,
                "current_floor_A": 1.0e-30,
            }
        )
    return rows


class GradedFullDcPolicyTest(unittest.TestCase):
    def test_gate_and_task_matrix_require_both_explicit_bits(self):
        self.assertEqual(full_dc.build_full_dc_tasks({"passed": True}), ())
        self.assertEqual(
            full_dc.build_full_dc_tasks({"full_sweep_allowed": True}), ()
        )
        tasks = full_dc.build_full_dc_tasks(
            {"passed": True, "full_sweep_allowed": True}
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {(row["mesh_level"], row["sweep_kind"]) for row in tasks},
            {(level, sweep) for level in full_dc.MESH_LEVELS for sweep in full_dc.SWEEP_KINDS},
        )

    def test_floor_limited_ioff_is_preserved_but_explicitly_masked(self):
        processed = full_dc.postprocess_metrics_rows(_metrics(4.9e-23))
        row = processed[0]
        self.assertEqual(row["raw_Ioff_A"], 4.9e-23)
        self.assertTrue(math.isnan(float(row["Ioff_A"])))
        self.assertTrue(math.isnan(float(row["on_off_ratio"])))
        self.assertEqual(row["ioff_status"], full_dc.IOFF_STATUS_FLOOR_LIMITED)
        self.assertEqual(
            row["on_off_ratio_status"], full_dc.ON_OFF_STATUS_FLOOR_LIMITED
        )
        self.assertFalse(row["ioff_hard_acceptance_applies"])
        self.assertFalse(row["on_off_hard_acceptance_applies"])
        validation = full_dc.validate_graded_metrics_rows(
            processed, states=STATES, vds_values_V=VDS_VALUES
        )
        self.assertTrue(validation["passed"], validation["errors"])

    def test_acceptance_floor_is_inclusive_for_reporting(self):
        processed = full_dc.postprocess_metrics_rows(
            _metrics(full_dc.GRADED_IOFF_ACCEPTANCE_FLOOR_A)
        )
        row = processed[0]
        self.assertEqual(row["Ioff_A"], row["raw_Ioff_A"])
        self.assertTrue(math.isfinite(float(row["on_off_ratio"])))
        self.assertEqual(row["ioff_status"], full_dc.IOFF_STATUS_REPORTED)
        validation = full_dc.validate_graded_metrics_rows(
            processed, states=STATES, vds_values_V=VDS_VALUES
        )
        self.assertTrue(validation["passed"], validation["errors"])

    def test_untagged_nan_and_wrong_floor_tag_fail_closed(self):
        processed = full_dc.postprocess_metrics_rows(_metrics(4.9e-23))
        broken = copy.deepcopy(processed)
        broken[0]["ioff_status"] = full_dc.IOFF_STATUS_REPORTED
        result = full_dc.validate_graded_metrics_rows(
            broken, states=STATES, vds_values_V=VDS_VALUES
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["status_error_count"], 1)

        above = full_dc.postprocess_metrics_rows(_metrics(1.0e-22))
        above[0]["on_off_ratio"] = math.nan
        result = full_dc.validate_graded_metrics_rows(
            above, states=STATES, vds_values_V=VDS_VALUES
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["status_error_count"], 1)

    def test_combined_absolute_relative_comparison_boundaries(self):
        curve = full_dc._comparison(
            1.0e-18,
            1.0e-18 + 1.0e-18 + 0.05e-18,
            absolute_tolerance=1.0e-18,
            relative_tolerance=0.05,
            scale_floor=0.0,
        )
        self.assertTrue(curve["within_tolerance"])
        vth = full_dc._comparison(
            0.5,
            0.5020000001,
            absolute_tolerance=full_dc.VTH_ABSOLUTE_TOLERANCE_V,
            relative_tolerance=0.0,
            scale_floor=0.0,
        )
        self.assertFalse(vth["within_tolerance"])

    def test_invalid_worker_inputs_fail_before_runtime_imports(self):
        with self.assertRaisesRegex(ValueError, "mesh_level"):
            full_dc.run_full_dc_worker("not_a_mesh", "idvg")
        with self.assertRaisesRegex(ValueError, "sweep_kind"):
            full_dc.run_full_dc_worker(full_dc.MESH_LEVELS[0], "not_a_sweep")

    def test_module_has_no_writer_or_main_entrypoint(self):
        self.assertFalse(hasattr(full_dc, "main"))
        self.assertFalse(any(name.startswith("write_") for name in vars(full_dc)))


if __name__ == "__main__":
    unittest.main()
