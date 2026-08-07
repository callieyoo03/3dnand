"""Pure unit tests for active-window electron-FN sensitivity reporting."""

import math
import unittest

import run_active_window_sensitivity as runner


class ActiveWindowSensitivityTests(unittest.TestCase):
    def _statistics(self):
        return {
            "active_interface_edge_count": 4,
            "total_interface_area_cm2": 8.0,
            "area_weighted_field_mean_signed_V_cm": 2.0,
            "area_weighted_field_mean_abs_V_cm": 2.0,
            "field_max_abs_V_cm": 5.0,
        }

    def _edge_result(self):
        fields = (5.0, 1.0, 2.0, 4.0)
        fractions = (0.60, 0.05, 0.10, 0.25)
        return {
            "log10_effective_current_density_A_cm2": -100.0,
            "log10_total_tunneling_current_A": -112.0,
            "edge_results": [
                {
                    "midpoint_axial_nm": 10.0 + index,
                    "electric_field_signed_V_cm": fields[index],
                    "fraction_of_total_tunneling_current": fractions[index],
                }
                for index in range(4)
            ],
        }

    def test_window_set_is_exact(self):
        self.assertEqual(
            runner.ACTIVE_WINDOWS_NM,
            ((10.0, 90.0), (11.0, 89.0), (12.0, 88.0), (20.0, 80.0)),
        )

    def test_endpoint_and_top_edge_dominance(self):
        row = runner.summarize_window(self._statistics(), self._edge_result())
        self.assertAlmostEqual(row["endpoint_pair_current_fraction"], 0.85)
        self.assertAlmostEqual(row["top_two_edge_current_fraction"], 0.85)
        self.assertAlmostEqual(row["top_10_percent_edge_current_fraction"], 0.60)
        self.assertTrue(row["maximum_field_edge_at_selected_window_endpoint"])

    def test_log_domain_reference_delta(self):
        reference = runner.summarize_window(
            self._statistics(), self._edge_result()
        )
        row = dict(reference)
        row["active_interface_area_cm2"] = 4.0
        row["field_mean_abs_V_cm"] = 1.0
        row["field_max_abs_V_cm"] = 4.0
        row["log10_effective_current_density_A_cm2"] = -102.0
        row["log10_total_tunneling_current_A"] = -114.5
        compared = runner.add_reference_comparison(row, reference)
        self.assertAlmostEqual(compared["area_ratio_vs_10_90"], 0.5)
        self.assertAlmostEqual(compared["field_mean_abs_ratio_vs_10_90"], 0.5)
        self.assertAlmostEqual(compared["field_max_abs_ratio_vs_10_90"], 0.8)
        self.assertAlmostEqual(
            compared["log10_effective_current_density_delta_vs_10_90"], -2.0
        )
        self.assertAlmostEqual(
            compared["log10_total_current_delta_vs_10_90"], -2.5
        )

    def test_negative_control_metadata_and_schema(self):
        self.assertEqual(runner._model_metadata()["physical_role"], "negative_control")
        self.assertIs(runner._model_metadata()["predictive_erase_model"], False)
        self.assertEqual(len(runner.CSV_FIELDS), len(set(runner.CSV_FIELDS)))

    def test_equal_infinities_have_zero_log_delta(self):
        self.assertEqual(runner._safe_log10_delta(-math.inf, -math.inf), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
