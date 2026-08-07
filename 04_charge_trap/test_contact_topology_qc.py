"""Unit tests for contact_topology_qc without a DEVSIM dependency."""

from __future__ import annotations

import math
import unittest

import contact_topology_qc as qc


def _surface_weights(
    radial: list[float],
    axial: list[float],
    edges: list[tuple[int, int]],
) -> list[float]:
    values = [0.0] * len(radial)
    for node0, node1 in edges:
        length = math.hypot(
            radial[node1] - radial[node0], axial[node1] - axial[node0]
        )
        area = math.pi * (radial[node0] + radial[node1]) * length
        values[node0] += 0.5 * area
        values[node1] += 0.5 * area
    return values


class FakeRuntimeAPI:
    def __init__(self) -> None:
        r0 = 1.0e-6
        rmid = 1.1e-6
        r1 = 1.2e-6
        rgate = 3.6e-6
        z0 = 0.0
        zmid = 5.0e-6
        z1 = 1.0e-5
        mos2_x = [r0, rmid, r1, r0, rmid, r1, 1.05e-6, 1.15e-6, 1.05e-6, 1.15e-6]
        mos2_y = [z0, z0, z0, z1, z1, z1, 1.0e-6, 1.0e-6, 9.0e-6, 9.0e-6]
        source_edges = [(0, 1), (1, 2)]
        drain_edges = [(3, 4), (4, 5)]
        mos2_surface = _surface_weights(mos2_x, mos2_y, source_edges + drain_edges)
        gate_x = [rgate, rgate, rgate, 3.5e-6, 3.5e-6]
        gate_y = [z0, zmid, z1, 2.5e-6, 7.5e-6]
        gate_edges = [(0, 1), (1, 2)]
        gate_surface = _surface_weights(gate_x, gate_y, gate_edges)
        self.contact_regions = {
            "source": "MoS2",
            "drain": "MoS2",
            "gate": "BlockingOxide",
        }
        self.edges = {
            "source": source_edges,
            "drain": drain_edges,
            "gate": gate_edges,
        }
        self.triangles = {
            "MoS2": [(0, 1, 6), (1, 2, 7), (3, 4, 8), (4, 5, 9)],
            "BlockingOxide": [(0, 1, 3), (1, 2, 4)],
        }
        self.models = {
            "MoS2": {
                "x": mos2_x,
                "y": mos2_y,
                "CylindricalSurfaceArea": mos2_surface,
                "ContactNSurfaceNormal_x": [0.0] * 10,
                "ContactNSurfaceNormal_y": [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            },
            "BlockingOxide": {
                "x": gate_x,
                "y": gate_y,
                "CylindricalSurfaceArea": gate_surface,
                "ContactNSurfaceNormal_x": [1.0, 1.0, 1.0, 0.0, 0.0],
                "ContactNSurfaceNormal_y": [0.0, 0.0, 0.0, 0.0, 0.0],
            },
        }

    def get_contact_list(self, *, device: str):
        return tuple(self.contact_regions)

    def get_region_list(self, *, device: str, contact: str):
        return (self.contact_regions[contact],)

    def get_element_node_list(
        self, *, device: str, region: str, contact: str | None = None
    ):
        if contact is None:
            return tuple(self.triangles[region])
        return tuple(self.edges[contact])

    def get_node_model_values(self, *, device: str, region: str, name: str):
        return tuple(self.models[region][name])

    def get_parameter(self, *, device: str, name: str):
        return {
            "raxis_variable": "x",
            "raxis_zero": 0.0,
            "node_volume_model": "CylindricalNodeVolume",
            "edge_couple_model": "CylindricalEdgeCouple",
        }[name]

    def get_contact_equation_list(self, *, device: str, contact: str):
        if contact in {"source", "drain"}:
            return ("PotentialEquation", "ElectronContinuityEquation")
        return ("PotentialEquation",)

    def get_contact_equation_command(self, *, device: str, contact: str, name: str):
        if name == "PotentialEquation":
            return {"edge_charge_model": "PotentialEdgeFlux"}
        if name == "ElectronContinuityEquation":
            return {"edge_current_model": "ElectronCurrent"}
        raise KeyError(name)


GEOMETRY = {
    "r_core": 1.0e-6,
    "r_mos2": 1.2e-6,
    "r_block": 3.6e-6,
    "z_source": 0.0,
    "z_drain": 1.0e-5,
}


def _linear_endpoint_components(
    derivatives_F: dict[str, float],
    delta_V: float,
) -> tuple[dict[str, float], dict[str, float]]:
    baselines = {
        "Qg_contact_C": 1.0e-16,
        "Qd_contact_C": -2.0e-17,
        "Qs_contact_C": -3.0e-17,
        "Qmobile_C": -5.0e-17,
        "Qtrap_C": 0.0,
        "Qfixed_C": 0.0,
    }
    plus = {
        name: baselines[name] + derivatives_F[name] * delta_V for name in baselines
    }
    minus = {
        name: baselines[name] - derivatives_F[name] * delta_V for name in baselines
    }
    return plus, minus


class ContactTopologyTests(unittest.TestCase):
    def test_analytic_annular_area(self) -> None:
        expected = math.pi * ((1.2e-6) ** 2 - (1.0e-6) ** 2)
        self.assertAlmostEqual(qc.analytic_annular_area_cm2(1.0e-6, 1.2e-6), expected)
        with self.assertRaises(ValueError):
            qc.analytic_annular_area_cm2(1.2e-6, 1.0e-6)

    def test_runtime_edge_audit_accepts_physical_end_caps(self) -> None:
        audit = qc.audit_runtime_contact_topology(
            GEOMETRY, runtime_api=FakeRuntimeAPI()
        )
        self.assertTrue(audit["passed"])
        expected_area = qc.analytic_annular_area_cm2(1.0e-6, 1.2e-6)
        for contact, expected_normal_z in (("source", -1.0), ("drain", 1.0)):
            report = audit["contacts"][contact]
            self.assertEqual(report["edge_count"], 2)
            self.assertAlmostEqual(report["radial_min_cm"], 1.0e-6)
            self.assertAlmostEqual(report["radial_max_cm"], 1.2e-6)
            self.assertAlmostEqual(report["geometric_swept_area_cm2"], expected_area)
            self.assertAlmostEqual(report["cylindrical_surface_area_cm2"], expected_area)
            self.assertEqual(report["normal_r"], 0.0)
            self.assertEqual(report["normal_z"], expected_normal_z)
        self.assertGreater(audit["contacts"]["gate"]["edge_count"], 0)
        rows = qc.contact_topology_rows(audit, mesh_level="base")
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(set(row) == set(qc.CONTACT_TOPOLOGY_FIELDNAMES) for row in rows))

    def test_zero_edge_contact_is_an_explicit_failure(self) -> None:
        runtime = FakeRuntimeAPI()
        runtime.edges["source"] = []
        audit = qc.audit_runtime_contact_topology(GEOMETRY, runtime_api=runtime)
        self.assertFalse(audit["passed"])
        self.assertEqual(audit["contacts"]["source"]["edge_count"], 0)
        self.assertIn("zero boundary edges", audit["contacts"]["source"]["error_message"])

    def test_runtime_normal_sign_is_diagnostic_not_acceptance(self) -> None:
        runtime = FakeRuntimeAPI()
        runtime.models["MoS2"]["ContactNSurfaceNormal_y"] = [1.0] * 10
        audit = qc.audit_runtime_contact_topology(GEOMETRY, runtime_api=runtime)
        self.assertTrue(audit["passed"])
        source = audit["contacts"]["source"]
        self.assertEqual(source["normal_method"], "geometry_boundary_outward")
        self.assertEqual(source["normal_z"], -1.0)
        self.assertEqual(source["runtime_normal_z"], 1.0)
        self.assertFalse(source["runtime_normal_matches_geometry"])
        self.assertTrue(source["normal_passed"])

    def test_geometry_derived_normals_reverse_with_axial_orientation(self) -> None:
        runtime = FakeRuntimeAPI()
        reversed_geometry = {**GEOMETRY, "z_source": 1.0e-5, "z_drain": 0.0}
        runtime.models["MoS2"]["ContactNSurfaceNormal_y"] = [
            1.0,
            1.0,
            1.0,
            -1.0,
            -1.0,
            -1.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
        runtime.models["MoS2"]["y"] = (
            [1.0e-5] * 3
            + [0.0] * 3
            + [9.0e-6, 9.0e-6, 1.0e-6, 1.0e-6]
        )
        audit = qc.audit_runtime_contact_topology(
            reversed_geometry, runtime_api=runtime
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["contacts"]["source"]["expected_normal_z"], 1.0)
        self.assertEqual(audit["contacts"]["drain"]["expected_normal_z"], -1.0)


class DerivativeGaussTests(unittest.TestCase):
    def test_contact_plus_internal_derivatives_close_gauss_balance(self) -> None:
        delta = 1.0e-3
        derivatives = {
            "Qg_contact_C": 5.0e-17,
            "Qd_contact_C": -2.0e-17,
            "Qs_contact_C": -1.0e-17,
            "Qmobile_C": -2.0e-17,
            "Qtrap_C": 0.0,
            "Qfixed_C": 0.0,
        }
        plus, minus = _linear_endpoint_components(derivatives, delta)
        result = qc.calculate_derivative_gauss_balance(
            plus,
            minus,
            delta,
            absolute_tolerance_F=1.0e-24,
            relative_tolerance=1.0e-8,
        )
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["contact_derivative_sum_F"], 2.0e-17)
        self.assertAlmostEqual(result["dQmobile_dV_F"], -2.0e-17)
        self.assertEqual(result["dQnoncontact_boundary_dV_F"], 0.0)
        self.assertEqual(
            result["noncontact_boundary_method"], qc.NATURAL_NONCONTACT_METHOD
        )
        self.assertAlmostEqual(
            result["derivative_gauss_residual_F"],
            result["endpoint_residual_derivative_F"],
        )

    def test_derivative_gauss_failure_is_not_hidden_as_noncontact_flux(self) -> None:
        delta = 1.0e-3
        derivatives = {
            "Qg_contact_C": 5.0e-17,
            "Qd_contact_C": -2.0e-17,
            "Qs_contact_C": -1.0e-17,
            "Qmobile_C": -1.0e-17,
            "Qtrap_C": 0.0,
            "Qfixed_C": 0.0,
        }
        plus, minus = _linear_endpoint_components(derivatives, delta)
        result = qc.calculate_derivative_gauss_balance(
            plus,
            minus,
            delta,
            absolute_tolerance_F=1.0e-24,
            relative_tolerance=1.0e-8,
        )
        self.assertFalse(result["passed"])
        self.assertGreater(abs(result["derivative_gauss_residual_F"]), 1.0e-18)
        self.assertEqual(result["dQnoncontact_boundary_dV_F"], 0.0)

    def test_derivative_gauss_schema_builder(self) -> None:
        derivatives = {name: 0.0 for name in qc._CHARGE_ALIASES}
        plus, minus = _linear_endpoint_components(derivatives, 1.0e-3)
        result = qc.calculate_derivative_gauss_balance(plus, minus, 1.0e-3)
        metadata = {
            "state_index": 0,
            "state": "Empty",
            "mesh_level": "base",
            "VGS_V": -1.0,
            "VDS_V": 0.05,
            "VS_V": 0.0,
            "perturbed_terminal": "gate",
            "delta_voltage_V": 1.0e-3,
        }
        row = qc.build_derivative_gauss_row(metadata, result)
        self.assertEqual(set(row), set(qc.DERIVATIVE_GAUSS_BALANCE_FIELDNAMES))


class GaugeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Every measured-terminal row sums to zero.  Contact-only columns do not.
        self.matrix = {
            ("gate", "gate"): 5.0e-17,
            ("gate", "drain"): -3.0e-17,
            ("gate", "source"): -2.0e-17,
            ("drain", "gate"): -2.0e-17,
            ("drain", "drain"): 2.0e-17,
            ("drain", "source"): 0.0,
            ("source", "gate"): -1.0e-17,
            ("source", "drain"): -1.0e-17,
            ("source", "source"): 2.0e-17,
        }
        self.baseline = {
            "Qg_contact_C": 1.0e-16,
            "Qd_contact_C": -2.0e-17,
            "Qs_contact_C": -3.0e-17,
            "Qmobile_C": -5.0e-17,
            "Qtrap_C": 0.0,
            "Qfixed_C": 0.0,
        }

    def test_gauge_rows_pass_even_when_contact_columns_are_nonzero(self) -> None:
        result = qc.evaluate_gauge_invariance(
            self.matrix, self.baseline, self.baseline, 1.0e-3
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["column_sums_are_acceptance_criteria"])
        self.assertNotEqual(
            result["contact_column_sums_F_diagnostic_only"]["gate"], 0.0
        )
        metadata = {
            "state_index": 0,
            "state": "Empty",
            "mesh_level": "base",
            "VGS_V": -1.0,
            "VDS_V": 0.05,
            "VS_V": 0.0,
        }
        rows = qc.build_gauge_invariance_rows(metadata, result)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(set(row) == set(qc.GAUGE_INVARIANCE_FIELDNAMES) for row in rows))

    def test_direct_common_mode_change_fails_gauge_check(self) -> None:
        plus = dict(self.baseline)
        minus = dict(self.baseline)
        plus["Qd_contact_C"] += 1.0e-20
        minus["Qd_contact_C"] -= 1.0e-20
        result = qc.evaluate_gauge_invariance(
            self.matrix,
            plus,
            minus,
            1.0e-3,
            absolute_tolerance_F=1.0e-24,
            relative_tolerance=1.0e-8,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["rows"]["drain"]["direct_common_mode_passed"])


class ComparisonAndDecisionTests(unittest.TestCase):
    def test_near_zero_comparison_uses_absolute_tolerance(self) -> None:
        result = qc.compare_numeric_values(
            0.0,
            5.0e-29,
            absolute_tolerance=1.0e-28,
            relative_tolerance=1.0e-4,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["scale"], 5.0e-29)

    def test_quantity_map_comparison_retains_nonfinite_failure(self) -> None:
        result = qc.compare_quantity_maps(
            {"Qg": 1.0e-16, "near_zero": 0.0},
            {"Qg": 1.00001e-16, "near_zero": math.nan},
            comparison="repeat",
            absolute_tolerance=1.0e-28,
            relative_tolerance=1.0e-4,
        )
        self.assertFalse(result["passed"])
        self.assertEqual(len(result["rows"]), 2)
        failed = [row for row in result["rows"] if not row["passed"]]
        self.assertEqual(failed[0]["quantity"], "near_zero")
        self.assertEqual(set(result["rows"][0]), set(qc.NUMERIC_COMPARISON_FIELDNAMES))

    def test_support_decision_has_no_column_sum_gate(self) -> None:
        all_true = {name: True for name in qc.REQUIRED_SUPPORT_CRITERIA}
        decision = qc.determine_physical_contact_flux_status(all_true)
        self.assertEqual(decision["status"], qc.STATUS_SUPPORTED)
        self.assertFalse(decision["contact_column_sums_used_for_acceptance"])
        failed = dict(all_true)
        failed["mesh_convergence"] = False
        self.assertEqual(
            qc.determine_physical_contact_flux_status(failed)["status"],
            qc.STATUS_UNSUPPORTED,
        )
        unknown = dict(all_true)
        unknown["dc_regression_acceptable"] = None
        self.assertEqual(
            qc.determine_physical_contact_flux_status(unknown)["status"],
            qc.STATUS_PROVISIONAL,
        )
        with self.assertRaisesRegex(ValueError, "column sums"):
            qc.determine_physical_contact_flux_status(
                {**all_true, "matrix_column_sums": True}
            )


class WardDuttonTests(unittest.TestCase):
    def test_mobile_partition_identity_and_schema(self) -> None:
        result = qc.calculate_ward_dutton_identity(-4.0e-17, -6.0e-17, -1.0e-16)
        self.assertTrue(result["passed"])
        self.assertEqual(result["status"], "mobile_partition_only")
        self.assertIn("mobile-channel charge only", result["limitation"])
        metadata = {
            "state_index": 0,
            "state": "Empty",
            "mesh_level": "base",
            "VGS_V": -1.0,
            "VDS_V": 0.05,
            "VS_V": 0.0,
        }
        row = qc.build_ward_dutton_row(metadata, result)
        self.assertEqual(set(row), set(qc.WARD_DUTTON_FIELDNAMES))

    def test_mobile_partition_failure(self) -> None:
        result = qc.calculate_ward_dutton_identity(
            -4.0e-17,
            -5.0e-17,
            -1.0e-16,
            absolute_tolerance_C=1.0e-24,
            relative_tolerance=1.0e-8,
        )
        self.assertFalse(result["passed"])


class SchemaTests(unittest.TestCase):
    def test_schemas_are_unique(self) -> None:
        for schema in (
            qc.CONTACT_TOPOLOGY_FIELDNAMES,
            qc.DERIVATIVE_GAUSS_BALANCE_FIELDNAMES,
            qc.GAUGE_INVARIANCE_FIELDNAMES,
            qc.NUMERIC_COMPARISON_FIELDNAMES,
            qc.WARD_DUTTON_FIELDNAMES,
        ):
            self.assertTrue(schema)
            self.assertEqual(len(schema), len(set(schema)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
