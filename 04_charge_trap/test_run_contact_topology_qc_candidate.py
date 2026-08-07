"""DEVSIM-free tests for the contact-topology QC orchestrator."""

from __future__ import annotations

import math
from pathlib import Path
import unittest

import run_contact_topology_qc_candidate as runner


class ContactTopologyQCRunnerTests(unittest.TestCase):
    def test_derivative_gauss_accepts_nonzero_contact_column_sum(self) -> None:
        result = runner._derivative_balance(
            {
                "Qg_contact_C": 4.0e-18,
                "Qd_contact_C": -1.0e-18,
                "Qs_contact_C": -1.0e-18,
                "Qmobile_C": -2.0e-18,
                "Qtrap_C": 0.0,
                "Qfixed_C": 0.0,
            }
        )

        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["residual_F"], 0.0)
        contact_sum = sum(
            result["components_F"][name] for name in runner.CONTACT_CHARGES
        )
        self.assertAlmostEqual(contact_sum, 2.0e-18)
        self.assertNotEqual(contact_sum, 0.0)

    def test_global_balance_never_backsolves_noncontact_flux(self) -> None:
        result = runner._global_balance(
            {
                "Qg_contact_C": 1.0e-18,
                "Qd_contact_C": 0.0,
                "Qs_contact_C": 0.0,
                "Qmobile_C": 0.0,
                "Qtrap_C": 0.0,
                "Qfixed_C": 0.0,
            }
        )

        self.assertEqual(result["components_C"]["Qnoncontact_boundary_C"], 0.0)
        self.assertEqual(result["residual_C"], 1.0e-18)
        self.assertFalse(result["passed"])

    def test_mesh_line_validator_accepts_scaled_guard_lines(self) -> None:
        geometry = {
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
        positions = [
            ("y", -2.0e-7),
            ("y", 1.02e-5),
            *(("x", geometry[name]) for name in (
                "r_axis", "r_core", "r_mos2", "r_tox", "r_trap",
                "r_block", "r_gate_outer", "r_air_outer",
            )),
            ("y", 0.0),
            ("y", 1.0e-5),
        ]
        rows = [
            {
                "direction": direction,
                "position_cm": position,
                "base_spacing_cm": 2.0e-7,
                "scaled_spacing_cm": 1.0e-7,
                "scale": 0.5,
            }
            for direction, position in positions
        ]

        result = runner.validate_topology_mesh_line_log(rows, geometry, 0.5)

        self.assertTrue(result["passed"])
        self.assertEqual(result["line_count"], 12)
        self.assertEqual(len(result["guard_lines"]), 2)

    def test_worker_command_identifies_one_reference_point(self) -> None:
        command = runner.build_worker_command(
            python_executable="python",
            mesh_level="extra_fine",
            mesh_scale=0.25,
            state_index=4,
            bias_name="on",
            output_path=Path("point.json"),
        )

        self.assertIn("--worker", command)
        self.assertEqual(command[command.index("--state-index") + 1], "4")
        self.assertEqual(command[command.index("--bias-name") + 1], "on")
        self.assertEqual(command[command.index("--mesh-scale") + 1], "0.25")

    def test_historical_before_rows_capture_the_degenerate_contacts(self) -> None:
        rows = {row["contact"]: row for row in runner._before_topology_rows()}

        self.assertEqual(rows["source"]["edge_count"], 0)
        self.assertEqual(rows["drain"]["edge_count"], 0)
        self.assertEqual(rows["gate"]["edge_count"], 50)
        self.assertFalse(rows["source"]["passed"])
        self.assertTrue(rows["gate"]["passed"])
        expected = math.pi * ((1.2e-6) ** 2 - (1.0e-6) ** 2)
        self.assertAlmostEqual(rows["source"]["expected_area_cm2"], expected)

    def test_required_output_set_matches_specification(self) -> None:
        self.assertEqual(len(runner.OUTPUT_FILENAMES), 15)
        self.assertEqual(len(set(runner.OUTPUT_FILENAMES)), 15)
        self.assertIn("electrode_capacitance_derivative_balance.csv", runner.OUTPUT_FILENAMES)
        self.assertIn("weighting_potential_feasibility.md", runner.OUTPUT_FILENAMES)

    def test_derivative_csv_rows_tolerate_json_null_components(self) -> None:
        point = {
            "state_index": 0,
            "state": "State_0_Empty",
            "bias_name": "off",
            "VGS_V": -1.0,
            "VDS_V": 0.05,
            "VS_V": 0.0,
            "gauge_row_validation": {
                "passed": False,
                "rows": {
                    name: {"row_sum_F": math.nan}
                    for name in runner.TERMINALS
                },
            },
            "gauge": {"common_mode_derivatives_F": {}},
            "measurements": [{
                "perturbed_terminal": "gate",
                "delta_voltage_V": 0.001,
                "all_charge_derivatives_F": {
                    "Qg_contact_C": None,
                    "Qd_contact_C": 1.0e-18,
                    "Qs_contact_C": -1.0e-18,
                },
                "derivative_gauss": {},
            }],
        }

        rows = runner._derivative_rows({"base": {"points": [point]}})

        self.assertEqual(len(rows), 1)
        self.assertTrue(math.isnan(rows[0]["contact_column_sum_F"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
