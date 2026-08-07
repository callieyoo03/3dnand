"""Unit tests for qc_root_cause_metrics without running DEVSIM."""

from __future__ import annotations

import math
import unittest

import qc_root_cause_metrics as metrics


class FakeRuntimeAPI:
    """Two-by-one-nanometre MoS2 strip with explicit edge orientation."""

    def __init__(self) -> None:
        radial_inner = 1.0e-6
        spacing = 1.0e-7
        self.radial = (
            radial_inner,
            radial_inner + spacing,
            radial_inner,
            radial_inner + spacing,
            radial_inner,
            radial_inner + spacing,
        )
        self.axial = (0.0, 0.0, spacing, spacing, 2.0 * spacing, 2.0 * spacing)
        self.triangles = (
            (0, 1, 2),
            (1, 3, 2),
            (2, 3, 4),
            (3, 5, 4),
        )
        # Edge endpoint order is intentionally mixed.  Source node 0 is n1
        # on edge 1; drain nodes occur as both n0 and n1.
        self.edges = (
            (0, 1),
            (2, 0),
            (1, 2),
            (1, 3),
            (2, 3),
            (4, 2),
            (3, 4),
            (5, 3),
            (4, 5),
        )
        self.contacts = {
            "source": ((0, 1),),
            "drain": ((4, 5),),
        }
        self.current = (999.0, 2.0, -3.0, 4.0, 0.0, 5.0, 6.0, -7.0, 999.0)
        self.coupling = (100.0, 10.0, 2.0, 3.0, 1.0, 4.0, 5.0, 6.0, 100.0)
        self.field = tuple(100.0 + index for index in range(len(self.edges)))
        self.potential = (0.0, 0.0, 1.0, 1.0, 2.0, 2.0)
        self.electrons = tuple(1.0e10 * (1.0 + value) for value in self.potential)

    def get_region_list(self, *, device: str, contact: str):
        return ("MoS2",)

    def get_element_node_list(
        self,
        *,
        device: str,
        region: str,
        contact: str | None = None,
    ):
        if contact is None:
            return self.triangles
        return self.contacts[contact]

    def get_node_model_values(self, *, device: str, region: str, name: str):
        return {
            "x": self.radial,
            "y": self.axial,
            "Potential": self.potential,
            "Electrons": self.electrons,
        }[name]

    def get_edge_model_list(self, *, device: str, region: str):
        return (
            "x@n0",
            "x@n1",
            "y@n0",
            "y@n1",
            "ElectronCurrent",
            "ElectricField",
            "CylindricalEdgeCouple",
        )

    def get_edge_model_values(self, *, device: str, region: str, name: str):
        endpoint_models = {
            "x@n0": tuple(self.radial[node0] for node0, _ in self.edges),
            "x@n1": tuple(self.radial[node1] for _, node1 in self.edges),
            "y@n0": tuple(self.axial[node0] for node0, _ in self.edges),
            "y@n1": tuple(self.axial[node1] for _, node1 in self.edges),
        }
        return {
            **endpoint_models,
            "ElectronCurrent": self.current,
            "ElectricField": self.field,
            "CylindricalEdgeCouple": self.coupling,
        }[name]

    def get_parameter(self, *, device: str, name: str):
        if name == "edge_couple_model":
            return "CylindricalEdgeCouple"
        raise KeyError(name)

    def get_contact_equation_list(self, *, device: str, contact: str):
        return ("PotentialEquation", "ElectronContinuityEquation")

    def get_contact_equation_command(self, *, device: str, contact: str, name: str):
        if name == "ElectronContinuityEquation":
            return {"edge_current_model": "ElectronCurrent"}
        return {"edge_charge_model": "PotentialEdgeFlux"}

    def get_contact_current(self, *, device: str, contact: str, equation: str):
        # Source: -2*10 + (-3)*2 + 4*3 = -14 A.
        # Drain: 5*4 - 6*5 + (-7)*6 = -52 A.
        return {"source": -14.0, "drain": -52.0}[contact]


class ContactCurrentAuditTests(unittest.TestCase):
    def test_manual_sum_matches_devsim_orientation_and_skips_boundary_edge(self) -> None:
        runtime = FakeRuntimeAPI()
        result = metrics.manual_contact_current_audit(
            "device", "source", runtime_api=runtime
        )
        summary = result["summary"]
        manual = result["manual_integral"]

        self.assertTrue(summary["passed"])
        self.assertEqual(summary["api_current_A"], -14.0)
        self.assertEqual(manual["manual_current_A"], -14.0)
        self.assertEqual(manual["active_incident_edge_count"], 3)
        self.assertEqual(manual["skipped_both_contact_node_edge_count"], 1)

        rows = {row["edge_index"]: row for row in manual["edge_contributions"]}
        self.assertEqual(set(rows), {1, 2, 3})
        self.assertEqual(rows[1]["contact_endpoint"], "n1")
        self.assertEqual(rows[1]["orientation_sign"], -1.0)
        self.assertEqual(rows[1]["contribution_A"], -20.0)
        self.assertEqual(rows[2]["orientation_sign"], 1.0)
        self.assertEqual(rows[2]["contribution_A"], -6.0)
        self.assertEqual(rows[3]["contribution_A"], 12.0)

        spacing = 1.0e-7
        self.assertAlmostEqual(
            rows[1]["inner_radial_corner_distance_cm"], 0.5 * spacing
        )
        self.assertAlmostEqual(
            rows[1]["outer_radial_corner_distance_cm"],
            math.hypot(spacing, 0.5 * spacing),
        )

    def test_drain_uses_both_n0_and_n1_signs(self) -> None:
        result = metrics.manual_contact_current_integral(
            "device", "drain", runtime_api=FakeRuntimeAPI()
        )
        self.assertEqual(result["manual_current_A"], -52.0)
        signs = {
            row["edge_index"]: row["orientation_sign"]
            for row in result["edge_contributions"]
        }
        self.assertEqual(signs, {5: 1.0, 6: -1.0, 7: 1.0})

    def test_crosscheck_failure_preserves_unscaled_difference(self) -> None:
        summary = metrics.summarize_contact_current_crosscheck(
            api_current_A=10.0,
            manual_current_A=12.0,
            absolute_tolerance_A=0.0,
            relative_tolerance=0.01,
        )
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["manual_minus_api_A"], 2.0)
        self.assertEqual(summary["scale_A"], 12.0)
        self.assertEqual(summary["combined_tolerance_A"], 0.12)


class MeshQualityTests(unittest.TestCase):
    def test_triangle_quality_and_actual_edge_spacing(self) -> None:
        runtime = FakeRuntimeAPI()
        report = metrics.runtime_mesh_quality_metrics(
            "device",
            "MoS2",
            contacts=("source", "drain"),
            runtime_api=runtime,
        )
        triangle = report["triangle_quality"]
        self.assertEqual(triangle["triangle_count"], 4)
        self.assertEqual(triangle["degenerate_triangle_count"], 0)
        self.assertAlmostEqual(triangle["minimum_angle_deg"], 45.0)
        self.assertAlmostEqual(triangle["maximum_aspect_ratio"], 2.0)
        self.assertEqual(triangle["obtuse_fraction"], 0.0)

        spacing = report["actual_region_edge_spacing"]
        self.assertAlmostEqual(spacing["radial_cm"]["minimum"], 1.0e-7)
        self.assertAlmostEqual(spacing["axial_cm"]["minimum"], 1.0e-7)
        self.assertAlmostEqual(spacing["edge_length_cm"]["minimum"], 1.0e-7)
        self.assertAlmostEqual(
            spacing["edge_length_cm"]["maximum"], math.sqrt(2.0) * 1.0e-7
        )

        source = report["contacts"]["source"]
        self.assertEqual(source["active_incident_edge_count"], 3)
        self.assertEqual(
            [row["total_region_edge_valence"] for row in source["node_valence"]],
            [2, 3],
        )
        self.assertEqual(
            [row["active_incident_edge_valence"] for row in source["node_valence"]],
            [1, 2],
        )
        coupling = source["incident_CylindricalEdgeCouple"]
        self.assertEqual(coupling["count"], 3)
        self.assertEqual(coupling["minimum"], 2.0)
        self.assertEqual(coupling["maximum"], 10.0)
        self.assertEqual(coupling["sum"], 15.0)

    def test_obtuse_triangle_is_counted(self) -> None:
        quality = metrics.triangle_quality_metrics(
            (0.0, 2.0, 0.5),
            (0.0, 0.0, 0.2),
            ((0, 1, 2),),
        )
        self.assertEqual(quality["obtuse_triangle_count"], 1)
        self.assertEqual(quality["obtuse_fraction"], 1.0)
        self.assertGreater(quality["maximum_angle_deg"], 90.0)

    def test_public_rows_include_region_triangle_contact_and_node_records(self) -> None:
        rows = metrics.mesh_quality_diagnostic_rows(
            "device", "MoS2", runtime_api=FakeRuntimeAPI()
        )
        row_types = [row["row_type"] for row in rows]
        self.assertEqual(row_types.count("region_summary"), 1)
        self.assertEqual(row_types.count("triangle"), 4)
        self.assertEqual(row_types.count("contact_summary"), 2)
        self.assertEqual(row_types.count("contact_node"), 4)


class SpatialProfileTests(unittest.TestCase):
    def test_cross_plane_profiles_interpolate_nodes_and_keep_edge_models(self) -> None:
        report = metrics.extract_contact_profiles(
            "device",
            "source",
            distances_nm=(0.0, 1.0, 2.0, 5.0),
            runtime_api=FakeRuntimeAPI(),
        )
        self.assertEqual(report["plane_axis"], "z")
        self.assertEqual(report["inward_coordinate_sign"], 1.0)
        self.assertAlmostEqual(report["maximum_inward_depth_nm"], 2.0)

        profiles = {
            profile["requested_distance_nm"]: profile
            for profile in report["profiles"]
        }
        self.assertEqual(profiles[0.0]["sample_count"], 3)
        self.assertEqual(profiles[2.0]["sample_count"], 3)
        self.assertFalse(profiles[5.0]["within_region_depth"])
        self.assertEqual(profiles[5.0]["sample_count"], 0)
        self.assertTrue(profiles[1.0]["samples"])
        for sample in profiles[1.0]["samples"]:
            self.assertAlmostEqual(sample["Potential_V"], 1.0)
            self.assertAlmostEqual(sample["Electrons_cm3"], 2.0e10)
            edge = sample["edge_index"]
            self.assertEqual(sample["ElectricField_V_per_cm"], 100.0 + edge)
            self.assertEqual(
                sample["ElectronCurrent_times_CylindricalEdgeCouple_A"],
                FakeRuntimeAPI().current[edge] * FakeRuntimeAPI().coupling[edge],
            )

    def test_public_profile_rows_keep_requested_empty_planes(self) -> None:
        rows = metrics.extract_contact_spatial_profile_rows(
            "device",
            "source",
            distances_nm=(0.0, 5.0),
            runtime_api=FakeRuntimeAPI(),
        )
        source_plane_rows = [row for row in rows if row["requested_distance_nm"] == 0.0]
        outside_rows = [row for row in rows if row["requested_distance_nm"] == 5.0]
        self.assertEqual(len(source_plane_rows), 3)
        self.assertTrue(all(row["sample_available"] for row in source_plane_rows))
        self.assertEqual(len(outside_rows), 1)
        self.assertFalse(outside_rows[0]["sample_available"])
        self.assertFalse(outside_rows[0]["within_region_depth"])


if __name__ == "__main__":
    unittest.main()
