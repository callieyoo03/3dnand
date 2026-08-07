"""Pure unit tests for candidate-only local-mesh full DC helpers."""

from __future__ import annotations

import copy
import math
import unittest

import local_mesh_full_dc as full_dc
import state_characterization_config as config


def _raw_rows(sweep_kind: str, scale: float = 1.0):
    state_by_index = {int(state["state_index"]): state for state in config.MEMORY_STATES}
    coordinates = full_dc._canonical_expected_coordinates(
        sweep_kind, states=config.MEMORY_STATES, config=config
    )
    rows = []
    for state_index, vgs, vds in sorted(coordinates):
        state = state_by_index[state_index]
        current = scale * (state_index + 1) * 1.0e-12 * math.exp(vgs) * vds
        rows.append(
            {
                "state_index": state_index,
                "state": state["state"],
                "ntrap_cm3": state["ntrap_cm3"],
                "nsheet_cm2": state["ntrap_cm3"] * 5.0e-7,
                "trap_charge_density_C_cm3": -state["ntrap_cm3"] * 1.602e-19,
                "VGS_V": vgs,
                "VDS_V": vds,
                "ID_A": current,
                "abs_ID_A": abs(current),
                "converged": True,
                "error_message": "",
            }
        )
    return rows


def _metric_rows(scale: float = 1.0):
    rows = []
    for state in config.MEMORY_STATES:
        state_index = int(state["state_index"])
        for vds in config.IDVG_VDS_VALUES_V:
            ion = scale * 1.0e-6 / (state_index + 1)
            ioff = scale * 1.0e-12 / (state_index + 1)
            rows.append(
                {
                    "state_index": state_index,
                    "state": state["state"],
                    "ntrap_cm3": state["ntrap_cm3"],
                    "nsheet_cm2": state["ntrap_cm3"] * 5.0e-7,
                    "VDS_V": float(vds),
                    "Vth_V": 0.2 + 0.15 * state_index,
                    "threshold_current_A": config.VTH_TARGET_CURRENT_A,
                    "vth_success": True,
                    "vth_error": "",
                    "SS_mV_dec": 70.0,
                    "ss_r_squared": 0.999,
                    "ss_point_count": 8,
                    "ss_success": True,
                    "ss_error": "",
                    "Ion_A": ion,
                    "ion_vgs_V": config.ION_VGS_V,
                    "ion_vds_V": config.ION_VDS_V,
                    "Ioff_A": ioff,
                    "ioff_vgs_V": config.IOFF_VGS_V,
                    "ioff_vds_V": config.IOFF_VDS_V,
                    "on_off_ratio": ion / ioff,
                    "gm_max_S": scale * 2.0e-6 / (state_index + 1),
                    "VGS_at_gm_max_V": 2.5,
                    "ss_current_min_A": config.SS_CURRENT_MIN_A,
                    "ss_current_max_A": config.SS_CURRENT_MAX_A,
                    "ss_minimum_point_count": config.SS_MINIMUM_POINT_COUNT,
                    "current_floor_A": config.CURRENT_FLOOR_A,
                }
            )
    return rows


def _bundle(mesh_level: str, sweep_kind: str, scale: float = 1.0):
    raw = _raw_rows(sweep_kind, scale)
    metrics = _metric_rows(scale) if sweep_kind == "idvg" else []
    expected = full_dc._canonical_expected_coordinates(
        sweep_kind, states=config.MEMORY_STATES, config=config
    )
    grid = full_dc.validate_full_sweep_rows(
        sweep_kind, raw, states=config.MEMORY_STATES,
        expected_coordinates=expected,
    )
    metric_validation = (
        full_dc.validate_metrics_rows(
            metrics, states=config.MEMORY_STATES,
            vds_values_V=config.IDVG_VDS_VALUES_V,
        )
        if sweep_kind == "idvg"
        else {"passed": True, "errors": []}
    )
    count = 100 if mesh_level == "local_extra_fine" else 200
    return {
        "mesh_level": mesh_level,
        "sweep_kind": sweep_kind,
        "geometry": {"r_core": 1.0e-6, "r_mos2": 1.2e-6, "z_source": 0.0,
                     "z_drain": 1.0e-5},
        "mesh_line_validation": {"passed": True, "position_hash": "fixed-positions"},
        "mesh_counts": {"global_coordinate_count": count,
                        "total_region_node_count": 2 * count,
                        "total_element_count": 3 * count},
        "raw_rows": raw,
        "metrics_rows": metrics,
        "grid_validation": grid,
        "metric_validation": metric_validation,
        "orchestration_errors": [],
        "runtime_seconds": 1.0,
        "passed": grid["passed"] and metric_validation["passed"],
        "error_message": "",
    }


class LocalMeshFullDCTests(unittest.TestCase):
    def test_reference_gate_requires_two_explicit_true_bits(self):
        for gate in ({}, {"passed": True}, {"full_sweep_allowed": True},
                     {"passed": False, "full_sweep_allowed": True}):
            self.assertFalse(full_dc.reference_gate_allows_full_dc(gate))
            self.assertEqual(full_dc.build_full_dc_tasks(gate), ())
        tasks = full_dc.build_full_dc_tasks(
            {"passed": True, "full_sweep_allowed": True}
        )
        self.assertEqual(len(tasks), 4)
        self.assertEqual(
            {(task["mesh_level"], task["sweep_kind"]) for task in tasks},
            {(mesh, sweep) for mesh in full_dc.MESH_LEVELS
             for sweep in full_dc.SWEEP_KINDS},
        )

    def test_worker_rejects_invalid_inputs_before_runtime_imports(self):
        with self.assertRaisesRegex(ValueError, "mesh_level"):
            full_dc.run_full_dc_worker("base", "idvg")
        with self.assertRaisesRegex(ValueError, "sweep_kind"):
            full_dc.run_full_dc_worker("local_extra_fine", "all")

    def test_full_sweep_validation_is_fail_closed(self):
        states = tuple(
            {"state_index": index, "state": f"s{index}", "ntrap_cm3": float(index)}
            for index in range(5)
        )
        expected = {(index, 3.0, 0.05) for index in range(5)}
        rows = [
            {
                "state_index": index, "state": f"s{index}", "ntrap_cm3": float(index),
                "nsheet_cm2": float(index), "trap_charge_density_C_cm3": -float(index),
                "VGS_V": 3.0, "VDS_V": 0.05, "ID_A": 1.0e-6,
                "abs_ID_A": 1.0e-6, "converged": True, "error_message": "",
            }
            for index in range(5)
        ]
        valid = full_dc.validate_full_sweep_rows(
            "idvg", rows, states=states, expected_coordinates=expected
        )
        self.assertTrue(valid["passed"])

        missing = full_dc.validate_full_sweep_rows(
            "idvg", rows[:-1], states=states, expected_coordinates=expected
        )
        self.assertFalse(missing["passed"])
        self.assertEqual(missing["missing_count"], 1)

        broken = copy.deepcopy(rows)
        broken[0]["ID_A"] = math.nan
        broken[1]["converged"] = False
        broken[2]["error_message"] = "Newton failed"
        invalid = full_dc.validate_full_sweep_rows(
            "idvg", broken, states=states, expected_coordinates=expected
        )
        self.assertFalse(invalid["passed"])
        self.assertEqual(invalid["nonfinite_count"], 1)
        self.assertEqual(invalid["failed_count"], 1)
        self.assertEqual(invalid["nonempty_error_count"], 1)

    def test_metrics_validation_rejects_nonfinite_or_failed_extraction(self):
        rows = _metric_rows()
        self.assertTrue(full_dc.validate_metrics_rows(
            rows, states=config.MEMORY_STATES,
            vds_values_V=config.IDVG_VDS_VALUES_V,
        )["passed"])
        bad = copy.deepcopy(rows)
        bad[0]["on_off_ratio"] = math.nan
        bad[1]["ss_success"] = False
        validation = full_dc.validate_metrics_rows(
            bad, states=config.MEMORY_STATES,
            vds_values_V=config.IDVG_VDS_VALUES_V,
        )
        self.assertFalse(validation["passed"])
        self.assertEqual(validation["nonfinite_count"], 1)
        self.assertEqual(validation["failed_count"], 1)

    def test_assemble_returns_raw_metrics_and_extra_ultra_convergence(self):
        bundles = [
            _bundle(mesh, sweep)
            for mesh in full_dc.MESH_LEVELS
            for sweep in full_dc.SWEEP_KINDS
        ]
        output = full_dc.assemble_full_dc_outputs(bundles)
        self.assertTrue(output["passed"], output["errors"])
        self.assertEqual(output["status"], full_dc.FULL_DC_STATUS_COMPLETE)
        expected_raw = 2 * (len(_raw_rows("idvg")) + len(_raw_rows("idvd")))
        self.assertEqual(len(output["raw_rows"]), expected_raw)
        self.assertEqual(len(output["metrics_rows"]), 2 * len(_metric_rows()))
        self.assertTrue(output["convergence_rows"])
        self.assertEqual(
            {row["comparison_domain"] for row in output["convergence_rows"]},
            {"idvg", "idvd", "metrics"},
        )
        self.assertTrue(all(row["passed"] for row in output["convergence_rows"]))

    def test_assembler_revalidates_rows_instead_of_trusting_worker_flag(self):
        bundles = [
            _bundle(mesh, sweep)
            for mesh in full_dc.MESH_LEVELS
            for sweep in full_dc.SWEEP_KINDS
        ]
        bundles[0]["raw_rows"] = bundles[0]["raw_rows"][:-1]
        bundles[0]["passed"] = True
        output = full_dc.assemble_full_dc_outputs(bundles)
        self.assertFalse(output["passed"])
        self.assertEqual(output["status"], full_dc.FULL_DC_STATUS_FAILED)
        self.assertTrue(any("assembler grid validation failed" in error
                            for error in output["errors"]))

    def test_missing_or_duplicate_bundle_fails_closed(self):
        bundles = [
            _bundle(mesh, sweep)
            for mesh in full_dc.MESH_LEVELS
            for sweep in full_dc.SWEEP_KINDS
        ]
        missing = full_dc.assemble_full_dc_outputs(bundles[:-1])
        self.assertFalse(missing["passed"])
        self.assertTrue(any("missing bundles" in error for error in missing["errors"]))
        duplicate = full_dc.assemble_full_dc_outputs([*bundles, bundles[0]])
        self.assertFalse(duplicate["passed"])
        self.assertTrue(any("duplicate bundle" in error for error in duplicate["errors"]))


if __name__ == "__main__":
    unittest.main()
