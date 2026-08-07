"""Unit tests for ERASE voltage-drop, field, and sign audits."""

import math
import unittest

import erase_field_diagnostics as audit


class EraseFieldDiagnosticTests(unittest.TestCase):
    def test_boundary_potential_selection_uses_active_window(self):
        x_values = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
        y_values = [0.0, 5.0e-6, 1.0e-5] * 2
        potentials = [99.0, 1.0, 99.0, 88.0, -2.0, 88.0]
        inner = audit.boundary_potential_statistics(
            x_values,
            y_values,
            potentials,
            boundary="inner",
            minimum_axial_nm=10.0,
            maximum_axial_nm=90.0,
        )
        outer = audit.boundary_potential_statistics(
            x_values,
            y_values,
            potentials,
            boundary="outer",
            minimum_axial_nm=10.0,
            maximum_axial_nm=90.0,
        )
        self.assertEqual(inner["potential_mean_V"], 1.0)
        self.assertEqual(outer["potential_mean_V"], -2.0)

    def test_layer_voltage_sum_and_field_units(self):
        def boundary(radius_nm, potential_V):
            return {
                "boundary_radius_nm": radius_nm,
                "node_count": 3,
                "potential_mean_V": potential_V,
            }

        boundaries = {
            "TunnelOxide": {
                "inner": boundary(12.0, 0.0),
                "outer": boundary(15.0, -1.0),
            },
            "ChargeTrap": {
                "inner": boundary(15.0, -1.0),
                "outer": boundary(20.0, -2.0),
            },
            "BlockingOxide": {
                "inner": boundary(20.0, -2.0),
                "outer": boundary(36.0, -4.0),
            },
        }
        geometry = {
            "tunnel_oxide_thickness_nm": 3.0,
            "charge_trap_thickness_nm": 5.0,
            "blocking_oxide_thickness_nm": 16.0,
        }
        rows, summary = audit.calculate_layer_voltage_drop_rows(
            boundaries,
            geometry,
        )
        self.assertEqual(summary["total_stack_voltage_drop_V"], -4.0)
        self.assertEqual(summary["layer_voltage_sum_residual_V"], 0.0)
        self.assertAlmostEqual(
            rows[0]["average_field_from_voltage_drop_outward_V_cm"],
            1.0 / (3.0e-7),
        )
        self.assertEqual(rows[0]["electric_field_unit"], "V/cm")

    def test_first_cell_boundary_selection(self):
        def edge(index, inner, outer, axial_nm):
            return {
                "edge_index": index,
                "direction": "radial",
                "x_n0_cm": inner,
                "x_n1_cm": outer,
                "y_n0_cm": axial_nm * 1.0e-7,
                "y_n1_cm": axial_nm * 1.0e-7,
                "radial_field_outward_V_cm": float(index),
            }

        edges = [
            edge(1, 12.0e-7, 12.5e-7, 20.0),
            edge(2, 12.0e-7, 12.5e-7, 80.0),
            edge(3, 14.5e-7, 15.0e-7, 20.0),
            edge(4, 14.5e-7, 15.0e-7, 80.0),
        ]
        inner = audit.select_active_boundary_radial_edges(edges, "inner")
        outer = audit.select_active_boundary_radial_edges(edges, "outer")
        self.assertEqual([row["edge_index"] for row in inner], [1, 2])
        self.assertEqual([row["edge_index"] for row in outer], [3, 4])

    def test_displacement_residual(self):
        inner = {"field_mean_signed_outward_V_cm": 2.0}
        outer = {"field_mean_signed_outward_V_cm": 1.0}
        result = audit.calculate_displacement_consistency(
            inner,
            5.0,
            outer,
            10.0,
        )
        self.assertEqual(result["epsilon_E_residual_C_cm2"], 0.0)
        self.assertEqual(result["epsilon_E_relative_residual"], 0.0)

    def test_monotonic_field_sanity(self):
        rows = [
            {
                "actual_gate_voltage_V": voltage,
                "devsim_tunnel_field_mean_abs_V_cm": field,
                "devsim_tunnel_field_max_abs_V_cm": 2.0 * field,
            }
            for voltage, field in ((0.0, 1.0), (-4.0, 2.0), (-8.0, 3.0))
        ]
        audit.validate_monotonic_field(rows)
        rows[-1]["devsim_tunnel_field_mean_abs_V_cm"] = 0.5
        with self.assertRaisesRegex(RuntimeError, "not monotonic"):
            audit.validate_monotonic_field(rows)

    def test_layer_closure_rejects_large_residual(self):
        audit.validate_layer_voltage_closure(
            [{"layer_voltage_sum_relative_residual": 1.0e-8}]
        )
        with self.assertRaisesRegex(RuntimeError, "voltage-drop sum"):
            audit.validate_layer_voltage_closure(
                [{"layer_voltage_sum_relative_residual": 1.0e-3}]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

