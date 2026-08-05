"""Unit tests for final-package reproducibility orchestration."""

from __future__ import annotations

import copy
import unittest

import run_final_handoff as handoff


def _bundle(value: str = "0") -> dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]:
    return {
        name: (("value", "label"), [{"value": value, "label": "stable"}])
        for name in handoff.REPRODUCIBILITY_FILES
    }


class ReproducibilityTests(unittest.TestCase):
    def test_contract_includes_all_dc_and_charge_capacitance_csvs(self) -> None:
        self.assertEqual(len(handoff.DC_REPRODUCIBILITY_FILES), 5)
        self.assertEqual(
            set(handoff.CHARGE_CAPACITANCE_REPRODUCIBILITY_FILES),
            {
                "terminal_charge_by_state.csv",
                "capacitance_matrix_by_state.csv",
                "capacitance_summary_by_state.csv",
            },
        )
        self.assertEqual(len(handoff.REPRODUCIBILITY_FILES), 8)

    def test_qc_uses_tight_absolute_tolerance_and_exact_nan_placement(self) -> None:
        before = _bundle()
        within = copy.deepcopy(before)
        within["terminal_charge_by_state.csv"][1][0]["value"] = "5e-29"
        reports = handoff.compare_dc_bundles(before, within)
        charge_report = next(
            row
            for row in reports
            if row["file"] == "terminal_charge_by_state.csv"
        )
        self.assertTrue(charge_report["passed"])
        self.assertEqual(
            charge_report["absolute_tolerance"],
            handoff.CHARGE_CAPACITANCE_ABSOLUTE_TOLERANCE,
        )

        outside = copy.deepcopy(before)
        outside["terminal_charge_by_state.csv"][1][0]["value"] = "2e-28"
        with self.assertRaisesRegex(RuntimeError, "DC/QC"):
            handoff.compare_dc_bundles(before, outside)

        nan_mismatch = copy.deepcopy(before)
        nan_mismatch["capacitance_matrix_by_state.csv"][1][0]["value"] = "nan"
        with self.assertRaisesRegex(RuntimeError, "DC/QC"):
            handoff.compare_dc_bundles(before, nan_mismatch)

    def test_only_derived_summary_diagnostics_receive_overrides(self) -> None:
        before = _bundle()
        sensitivity = copy.deepcopy(before)
        sensitivity["capacitance_summary_by_state.csv"] = (
            ("Cgg_relative_sensitivity", "gate_capacitance_row_sum_F"),
            [
                {
                    "Cgg_relative_sensitivity": "0",
                    "gate_capacitance_row_sum_F": "0",
                }
            ],
        )
        after = copy.deepcopy(sensitivity)
        after["capacitance_summary_by_state.csv"][1][0][
            "Cgg_relative_sensitivity"
        ] = "5e-9"
        after["capacitance_summary_by_state.csv"][1][0][
            "gate_capacitance_row_sum_F"
        ] = "5e-24"
        reports = handoff.compare_dc_bundles(sensitivity, after)
        summary = next(
            row
            for row in reports
            if row["file"] == "capacitance_summary_by_state.csv"
        )
        self.assertTrue(summary["passed"])
        self.assertIn("derived-diagnostic", summary["notes"])

        primary = copy.deepcopy(before)
        primary["capacitance_summary_by_state.csv"] = (
            ("Cgg_F",),
            [{"Cgg_F": "0"}],
        )
        primary_changed = copy.deepcopy(primary)
        primary_changed["capacitance_summary_by_state.csv"][1][0][
            "Cgg_F"
        ] = "2e-28"
        with self.assertRaisesRegex(RuntimeError, "DC/QC"):
            handoff.compare_dc_bundles(primary, primary_changed)

    def test_reproducibility_cannot_skip_charge_or_dc(self) -> None:
        with self.assertRaises(SystemExit):
            handoff.main(["--reproducibility-only", "--skip-charge"])
        with self.assertRaises(SystemExit):
            handoff.main(["--reproducibility-check", "--skip-dc"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
