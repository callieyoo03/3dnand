"""Unit tests for log-domain edge-resolved ERASE proxy diagnostics."""

import math
import unittest

import edge_resolved_erase as erase_fn
import tunneling_models


class EdgeResolvedEraseTests(unittest.TestCase):
    def _edge(self, edge_index, field_V_cm, radius_cm, length_cm):
        area_cm2 = erase_fn.cylindrical_edge_area_cm2(radius_cm, length_cm)
        return {
            "edge_index": edge_index,
            "radial_field_outward_V_cm": field_V_cm,
            "interface_radius_cm": radius_cm,
            "interface_radius_nm": radius_cm * 1.0e7,
            "midpoint_axial_nm": 20.0 + edge_index,
            "axial_segment_lower_nm": 19.5 + edge_index,
            "axial_segment_upper_nm": 20.5 + edge_index,
            "axial_segment_length_cm": length_cm,
            "axial_segment_length_nm": length_cm * 1.0e7,
            "cylindrical_interface_area_cm2": area_cm2,
        }

    def test_proxy_labels_and_parameters(self):
        metadata = erase_fn.get_model_metadata()
        self.assertEqual(metadata["model_class"], "electron_FN_diagnostic")
        self.assertEqual(metadata["physical_role"], "negative_control")
        self.assertEqual(metadata["model_status"], "baseline_negative_result")
        self.assertEqual(metadata["calibration_status"], "uncalibrated")
        self.assertIs(metadata["predictive_erase_model"], False)
        self.assertAlmostEqual(metadata["barrier_height_eV"], 3.56)
        self.assertAlmostEqual(metadata["effective_mass_ratio"], 0.28)
        erase_fn.validate_proxy_parameters()

    def test_cylindrical_edge_area(self):
        radius_cm = 12.0e-7
        length_cm = 2.0e-7
        expected = 2.0 * math.pi * radius_cm * length_cm
        self.assertAlmostEqual(
            erase_fn.cylindrical_edge_area_cm2(radius_cm, length_cm),
            expected,
            places=28,
        )

    def test_field_unit_conversion_reuses_tunneling_model(self):
        self.assertEqual(
            tunneling_models.electric_field_V_cm_to_V_m(1.25e6),
            1.25e8,
        )

    def test_zero_field_fn(self):
        self.assertEqual(
            erase_fn.log_fowler_nordheim_current_density(0.0),
            -math.inf,
        )
        mean = erase_fn.evaluate_mean_field_fn(0.0, 1.0e-12)
        self.assertEqual(mean["fowler_nordheim_current_density_A_cm2"], 0.0)
        self.assertEqual(mean["total_tunneling_current_A"], 0.0)
        self.assertEqual(mean["log10_total_tunneling_current_A"], -math.inf)

    def test_nonzero_subthreshold_field_remains_in_log_domain(self):
        # The legacy linear helper intentionally cuts off below 1 V/cm.  The
        # primary log result must nevertheless retain every nonzero edge.
        field_V_cm = 0.5
        self.assertEqual(
            tunneling_models.fowler_nordheim_current_density(field_V_cm),
            0.0,
        )
        self.assertTrue(
            math.isfinite(
                erase_fn.log_fowler_nordheim_current_density(field_V_cm)
            )
        )

    def test_logsumexp_handles_underflow(self):
        actual = erase_fn.logsumexp([-1000.0, -1001.0])
        expected = -1000.0 + math.log1p(math.exp(-1.0))
        self.assertAlmostEqual(actual, expected, places=13)
        self.assertEqual(erase_fn.logsumexp([-math.inf, -math.inf]), -math.inf)

    def test_all_edges_are_integrated_without_threshold_cutoff(self):
        edges = [
            self._edge(1, 4.0e6, 12.0e-7, 1.0e-7),
            self._edge(2, 5.0e6, 12.0e-7, 2.0e-7),
            self._edge(3, 0.0, 12.0e-7, 1.0e-7),
        ]
        result = erase_fn.evaluate_edge_resolved_fn(
            edges,
            diagnostic_field_threshold_V_cm=4.5e6,
        )

        self.assertEqual(result["edge_count"], 3)
        self.assertEqual(result["diagnostic_above_threshold_edge_count"], 1)
        self.assertTrue(
            all(row["included_in_primary_integration"] for row in result["edge_results"])
        )
        self.assertEqual(
            result["aggregation_method"],
            "all_edge_logsumexp_no_field_cutoff",
        )

        expected_logs = []
        for edge in edges:
            log_density = erase_fn.log_fowler_nordheim_current_density(
                edge["radial_field_outward_V_cm"]
            )
            expected_logs.append(
                -math.inf
                if log_density == -math.inf
                else log_density
                + math.log(edge["cylindrical_interface_area_cm2"])
            )
        self.assertAlmostEqual(
            result["log_total_tunneling_current_A"],
            erase_fn.logsumexp(expected_logs),
            places=13,
        )

        fractions = sum(
            row["fraction_of_total_tunneling_current"]
            for row in result["edge_results"]
        )
        self.assertAlmostEqual(fractions, 1.0, places=13)

    def test_rejects_area_not_matching_axisymmetric_formula(self):
        edge = self._edge(1, 5.0e6, 12.0e-7, 1.0e-7)
        edge["cylindrical_interface_area_cm2"] *= 2.0
        with self.assertRaisesRegex(ValueError, "2\\*pi"):
            erase_fn.evaluate_edge_resolved_fn([edge])

    def test_log_and_linear_density_agree_when_representable(self):
        field_V_cm = 5.0e6
        log_density = erase_fn.log_fowler_nordheim_current_density(field_V_cm)
        linear_density = tunneling_models.fowler_nordheim_current_density(
            field_V_cm
        )
        self.assertAlmostEqual(
            math.exp(log_density) / linear_density,
            1.0,
            places=12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
