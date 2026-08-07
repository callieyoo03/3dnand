"""DEVSIM-free tests for the directional-refinement diagnostic runner."""

from __future__ import annotations

import math
import types
import unittest
from pathlib import Path

import run_directional_refinement_diagnostic as diagnostic


GEOMETRY = {
    "r_axis": 0.0,
    "r_core": 1.0e-6,
    "r_mos2": 1.065e-6,
    "r_tox": 1.365e-6,
    "r_trap": 1.865e-6,
    "r_block": 3.465e-6,
    "r_gate_outer": 3.665e-6,
    "r_air_outer": 3.865e-6,
    "z_source": 0.0,
    "z_drain": 1.0e-5,
}


def fake_point_bundle(
    *, state_index: int, bias_name: str, guard_factor: float = 1.0,
    coordinate_count: int = 1000,
) -> dict[str, object]:
    target_gate = 3.0 if bias_name == "on" else -1.0
    state = "State_0_Empty" if state_index == 0 else "State_4_Programmed"
    return {
        "geometry": dict(GEOMETRY),
        "points": [
            {
                "state_index": state_index,
                "state": state,
                "bias_name": bias_name,
                "VGS_V": target_gate,
                "VDS_V": 0.05,
                "VS_V": 0.0,
                "direct": {
                    "Qg_contact_C": 5.0e-18,
                    "Qd_contact_C": -1.0e-18,
                    "Qs_contact_C": -2.0e-18,
                    "Qmobile_C": -2.0e-18,
                },
                "nominal_matrix_F": {
                    "gate:gate": 4.0e-18,
                    "gate:drain": -1.0e-18,
                    "gate:source": -1.0e-18,
                },
                "global_gauss": {
                    "residual_C": 0.0,
                    "scale_C": 5.0e-18,
                    "tolerance_C": 1.0e-24,
                    "passed": True,
                },
                "passed": False,
                "error_message": "provisional matrix",
            }
        ],
        "mesh_counts": {
            "global_coordinate_count": coordinate_count,
            "region_node_count_with_duplicates": coordinate_count + 100,
            "triangle_count": 2 * coordinate_count,
        },
        "diagnostic_mesh_counts": {
            "global_coordinate_count": coordinate_count,
            "active_unknown_count": coordinate_count + 250,
            "region_node_count_with_duplicates": coordinate_count + 100,
            "triangle_count": 2 * coordinate_count,
        },
        "topology": {
            "contacts": {
                "source": {"edge_count": 4},
                "drain": {"edge_count": 4},
                "gate": {"edge_count": 40},
            }
        },
        "directional_mesh_line_validation": {"line_count": 12},
        "final_reference_validation": {"passed": True, "error_message": ""},
        "actual_biases_after_worker": {
            "gate": target_gate,
            "drain": 0.05,
            "source": 0.0,
        },
        "drain_current_A": 2.0e-8,
        "runtime_seconds": 1.25,
        "worker_process_id": int(100 + 10 * guard_factor),
        "active_geometry_sha256": diagnostic.active_geometry_sha256(GEOMETRY),
        "guard_factor": guard_factor,
        "error_message": "",
    }


class DirectionalVariantTests(unittest.TestCase):
    def test_required_variants_are_unique_and_global_matches_extra_fine(self) -> None:
        self.assertEqual(
            tuple(variant.name for variant in diagnostic.VARIANTS),
            (
                "current_global",
                "axial_near_sd",
                "radial_mos2",
                "contact_corner_only",
                "mos2_tunnel_interface",
            ),
        )
        self.assertEqual(len(diagnostic.VARIANT_BY_NAME), 5)
        global_variant = diagnostic.VARIANT_BY_NAME["current_global"]
        self.assertAlmostEqual(
            global_variant.spacing_scale(diagnostic.CURRENT_GLOBAL_STRENGTH),
            0.25,
        )

    def test_axial_variant_changes_only_source_and_drain_lines(self) -> None:
        forwarded: list[dict[str, object]] = []

        def add_line(**kwargs: object) -> None:
            forwarded.append(dict(kwargs))

        structure = types.SimpleNamespace(add_2d_mesh_line=add_line)
        original = structure.add_2d_mesh_line
        log: list[dict[str, object]] = []
        variant = diagnostic.VARIANT_BY_NAME["axial_near_sd"]
        with diagnostic.directional_runtime_mesh_lines(
            structure, GEOMETRY, variant, 4.0, log
        ):
            structure.add_2d_mesh_line(mesh="m", dir="y", pos=0.0, ps=2.0e-7)
            structure.add_2d_mesh_line(mesh="m", dir="y", pos=5.0e-6, ps=2.0e-7)
            structure.add_2d_mesh_line(
                mesh="m", dir="x", pos=GEOMETRY["r_mos2"], ps=2.0e-8
            )

        self.assertIs(structure.add_2d_mesh_line, original)
        self.assertAlmostEqual(float(forwarded[0]["ps"]), 5.0e-8)
        self.assertAlmostEqual(float(forwarded[1]["ps"]), 2.0e-7)
        self.assertAlmostEqual(float(forwarded[2]["ps"]), 2.0e-8)
        self.assertEqual(log[0]["target"], "z_source")
        self.assertEqual(log[1]["target"], "unchanged")
        self.assertEqual(log[2]["target"], "unchanged")

    def test_corner_surrogate_uses_two_dimensional_strength(self) -> None:
        variant = diagnostic.VARIANT_BY_NAME["contact_corner_only"]
        scale, target = diagnostic.line_spacing_scale(
            direction="x",
            position_cm=GEOMETRY["r_mos2"],
            geometry=GEOMETRY,
            variant=variant,
            strength=16.0,
        )
        unchanged, label = diagnostic.line_spacing_scale(
            direction="x",
            position_cm=GEOMETRY["r_tox"],
            geometry=GEOMETRY,
            variant=variant,
            strength=16.0,
        )
        self.assertAlmostEqual(scale, 0.25)
        self.assertEqual(target, "r_mos2")
        self.assertEqual(unchanged, 1.0)
        self.assertEqual(label, "unchanged")

    def test_air_guard_scaling_is_temporary(self) -> None:
        baseline_guard = GEOMETRY["r_mos2"] - GEOMETRY["r_core"]
        module = types.SimpleNamespace(
            expanded_air_bounds=lambda geometry: {
                "xl": geometry["r_axis"],
                "xh": geometry["r_air_outer"],
                "yl": geometry["z_source"] - baseline_guard,
                "yh": geometry["z_drain"] + baseline_guard,
                "guard_cm": baseline_guard,
            }
        )
        original = module.expanded_air_bounds
        with diagnostic.scaled_air_guard(module, 2.0):
            bounds = module.expanded_air_bounds(GEOMETRY)
            self.assertAlmostEqual(
                bounds["yl"], GEOMETRY["z_source"] - 2.0 * baseline_guard
            )
            self.assertAlmostEqual(
                bounds["yh"], GEOMETRY["z_drain"] + 2.0 * baseline_guard
            )
            self.assertAlmostEqual(bounds["guard_cm"], baseline_guard)
        self.assertIs(module.expanded_air_bounds, original)


class OrchestrationTests(unittest.TestCase):
    def test_dof_selection_chooses_nearest_actual_coordinate_count(self) -> None:
        selected = diagnostic.select_strength_for_similar_dof(
            {2.0: 700, 4.0: 920, 8.0: 1100}, 1000
        )
        self.assertEqual(selected["strength"], 4.0)
        self.assertEqual(selected["active_unknown_count"], 920)
        self.assertAlmostEqual(selected["dof_relative_difference"], 0.08)
        self.assertTrue(selected["dof_comparable"])

    def test_worker_command_requires_exactly_one_point_identity(self) -> None:
        command = diagnostic.build_worker_command(
            python_executable="python",
            mode="point",
            variant_name="radial_mos2",
            strength=8.0,
            state_index=4,
            bias_name="off",
            guard_factor=1.0,
            output_path=Path("worker.json"),
        )
        self.assertEqual(command[command.index("--worker-mode") + 1], "point")
        self.assertEqual(command[command.index("--state-index") + 1], "4")
        self.assertEqual(command[command.index("--bias-name") + 1], "off")
        self.assertNotIn("erase", " ".join(command).lower())
        self.assertNotIn("retention", " ".join(command).lower())

    def test_point_values_keep_target_reach_separate_from_qc_promotion(self) -> None:
        values = diagnostic.point_values(
            fake_point_bundle(state_index=4, bias_name="off")
        )
        self.assertTrue(values["target_reached"])
        self.assertTrue(values["converged"])
        self.assertFalse(values["qc_point_passed"])
        self.assertEqual(values["ID_A"], 2.0e-8)
        self.assertEqual(values["raw_Qd_C"], -1.0e-18)
        self.assertEqual(values["Cgg_F"], 4.0e-18)

    def test_comparison_rows_cover_only_two_requested_points_per_variant(self) -> None:
        selections: dict[str, dict[str, object]] = {}
        points: dict[str, dict[str, object]] = {}
        for variant in diagnostic.VARIANTS:
            selections[variant.name] = {
                "strength": 16.0,
                "active_unknown_count": 1250,
                "reference_active_unknown_count": 1250,
                "global_coordinate_count": 1000,
                "reference_global_coordinate_count": 1000,
                "dof_relative_difference": 0.0,
                "dof_comparable": True,
            }
            for state_index, bias_name, _ in diagnostic.POINT_SPECS:
                points[
                    diagnostic._point_label(variant.name, state_index, bias_name, 1.0)
                ] = fake_point_bundle(state_index=state_index, bias_name=bias_name)
        rows = diagnostic.build_directional_rows(
            {"selections": selections, "points": points}
        )
        self.assertEqual(len(rows), 10)
        self.assertEqual({(row["state_index"], row["bias_name"]) for row in rows}, {(0, "on"), (4, "off")})
        self.assertTrue(all(tuple(row) == diagnostic.DIRECTIONAL_FIELDS for row in rows))

    def test_air_guard_rows_are_long_form_and_geometry_invariant(self) -> None:
        points = {
            diagnostic._point_label("current_global", 0, "on", factor):
                fake_point_bundle(state_index=0, bias_name="on", guard_factor=factor)
            for factor in diagnostic.AIR_GUARD_FACTORS
        }
        rows = diagnostic.build_air_guard_rows({"points": points})
        self.assertEqual(
            len(rows),
            len(diagnostic.AIR_GUARD_FACTORS) * len(diagnostic.AIR_GUARD_QUANTITIES),
        )
        self.assertTrue(all(row["active_geometry_unchanged"] for row in rows))
        baseline_rows = [row for row in rows if row["guard_extension_factor"] == 1.0]
        self.assertTrue(all(row["absolute_difference"] == 0.0 for row in baseline_rows))
        self.assertTrue(all(tuple(row) == diagnostic.AIR_GUARD_FIELDS for row in rows))


if __name__ == "__main__":
    unittest.main(verbosity=2)
