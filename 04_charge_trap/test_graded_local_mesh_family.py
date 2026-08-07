"""DEVSIM-free tests for the quality-preserving graded mesh family."""

from __future__ import annotations

import math
import random
import types
import unittest

import graded_local_mesh_family as family


GEOMETRY = {
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
ORIGINAL_LINES = (
    ("y", -2.0e-7, 2.0e-7),
    ("y", 1.02e-5, 2.0e-7),
    ("x", 0.0, 1.0e-7),
    ("x", 1.0e-6, 5.0e-8),
    ("x", 1.2e-6, 2.0e-8),
    ("x", 1.5e-6, 5.0e-8),
    ("x", 2.0e-6, 5.0e-8),
    ("x", 3.6e-6, 5.0e-8),
    ("x", 3.8e-6, 5.0e-8),
    ("x", 4.0e-6, 1.0e-7),
    ("y", 0.0, 2.0e-7),
    ("y", 1.0e-5, 2.0e-7),
)


def _realize(level: str):
    downstream_rows: list[dict[str, object]] = []

    def downstream(*args, **kwargs):
        if args:
            raise AssertionError("downstream accepts keywords only")
        downstream_rows.append(dict(kwargs))

    module = types.SimpleNamespace(add_2d_mesh_line=downstream)
    log: list[dict[str, object]] = []
    with family.local_runtime_mesh_lines(module, GEOMETRY, level, log):
        for direction, position, spacing in ORIGINAL_LINES:
            module.add_2d_mesh_line(
                mesh="gaa_mesh", dir=direction, pos=position, ps=spacing
            )
    return module, downstream, downstream_rows, log


class GradedLocalMeshFamilyTests(unittest.TestCase):
    def test_level_names_positions_and_complete_line_count(self) -> None:
        self.assertEqual(
            family.LEVELS,
            (
                "graded_base",
                "graded_fine",
                "graded_extra_fine",
                "graded_ultra_fine",
            ),
        )
        hashes = set()
        position_sets = []
        for level in family.LEVELS:
            _, _, rows, log = _realize(level)
            self.assertEqual(len(rows), 33)
            self.assertEqual(len(log), 33)
            report = family.validate_local_mesh_line_log(log, GEOMETRY, level)
            self.assertTrue(report["passed"], report["errors"])
            self.assertTrue(report["adjacent_requested_spacing_ratio_passed"])
            hashes.add(report["position_hash"])
            position_sets.append(
                {(row["dir"], round(float(row["pos"]), 18)) for row in rows}
            )
        self.assertEqual(len(hashes), 1)
        self.assertTrue(
            all(positions == position_sets[0] for positions in position_sets[1:])
        )

    def test_exact_axial_interval_targets_and_ratio_limit(self) -> None:
        expected_nm = {
            "graded_base": (.25, .25, .25, .25, .5, 1, 2, 1, 2, 4),
            "graded_fine": (.125, .125, .125, .125, .25, .5, 1, 1, 2, 2),
            "graded_extra_fine": (
                .0625, .0625, .0625, .0625, .125, .25, .5, 1, 1, 1,
            ),
            "graded_ultra_fine": (
                .03125, .03125, .03125, .03125, .0625, .125, .25, .5, .5, .5,
            ),
        }
        for level, expected in expected_nm.items():
            actual = tuple(
                value / family.NM_TO_CM
                for value in family.axial_interval_spacings_cm(level)
            )
            for observed, target in zip(actual, expected):
                self.assertAlmostEqual(observed, target, places=14)
            ratios = [
                max(left, right) / min(left, right)
                for left, right in zip(actual, actual[1:])
            ]
            self.assertLessEqual(max(ratios), 2.0)

    def test_mos2_anisotropy_bound_is_constant(self) -> None:
        for level in family.LEVELS:
            spec = family.get_level_spec(level)
            central = family.axial_interval_spacings_cm(level)[-1]
            anisotropy = central / spec.radial_spacing_cm
            self.assertAlmostEqual(anisotropy, 8.0)
            self.assertAlmostEqual(math.degrees(math.atan(1.0 / anisotropy)),
                                   7.125016348901798)
            self.assertAlmostEqual(anisotropy + 1.0 / anisotropy, 8.125)

    def test_runtime_radial_and_symmetric_axial_anchors(self) -> None:
        _, _, rows, log = _realize("graded_ultra_fine")
        radial_nm = sorted(
            float(row["position_cm"]) / family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("mos2_radial_anchor_")
        )
        for actual, expected in zip(radial_nm, (10, 10.5, 11, 11.5, 12)):
            self.assertAlmostEqual(actual, expected, places=12)
        source_nm = sorted(
            float(row["position_cm"]) / family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("source_axial_anchor_")
        )
        drain_nm = sorted(
            (GEOMETRY["z_drain"] - float(row["position_cm"])) / family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("drain_axial_anchor_")
        )
        for values in (source_nm, drain_nm):
            for actual, expected in zip(values, family.AXIAL_DISTANCE_NM):
                self.assertAlmostEqual(actual, expected, places=12)
        positions = [(row["dir"], row["pos"]) for row in rows]
        self.assertEqual(len(positions), len(set(positions)))

    def test_side_spacings_guard_and_passthrough_are_explicit(self) -> None:
        _, _, _, log = _realize("graded_extra_fine")
        by_target = {str(row["target"]): row for row in log}
        source = by_target["source_axial_anchor_0"]
        drain = by_target["drain_axial_anchor_0"]
        self.assertAlmostEqual(float(source["output_negative_spacing_cm"]), 2e-7)
        self.assertAlmostEqual(float(source["output_positive_spacing_cm"]), 6.25e-9)
        self.assertAlmostEqual(float(drain["output_negative_spacing_cm"]), 6.25e-9)
        self.assertAlmostEqual(float(drain["output_positive_spacing_cm"]), 2e-7)
        expected_passthrough = {
            "radial_boundary_r_axis": 1.0e-7,
            "radial_boundary_r_tox": 5.0e-8,
            "radial_boundary_r_trap": 5.0e-8,
            "radial_boundary_r_block": 5.0e-8,
            "radial_boundary_r_gate_outer": 5.0e-8,
            "radial_boundary_r_air_outer": 1.0e-7,
        }
        for target, spacing in expected_passthrough.items():
            row = by_target[target]
            self.assertEqual(float(row["output_negative_spacing_cm"]), spacing)
            self.assertEqual(float(row["output_positive_spacing_cm"]), spacing)

    def test_family_rows_record_both_sides_and_adjacent_ratio(self) -> None:
        rows = family.local_mesh_family_definition_rows(GEOMETRY)
        self.assertEqual(len(rows), 33 * len(family.LEVELS))
        for field in (
            "negative_spacing_cm",
            "negative_spacing_nm",
            "positive_spacing_cm",
            "positive_spacing_nm",
            "adjacent_requested_spacing_ratio",
            "adjacent_ratio_scope",
        ):
            self.assertTrue(all(field in row for row in rows))
        active = [
            row for row in rows
            if row["adjacent_ratio_scope"] == "adjacent_active_axial_intervals"
        ]
        self.assertTrue(active)
        self.assertLessEqual(
            max(float(row["adjacent_requested_spacing_ratio"]) for row in active),
            2.0,
        )
        self.assertEqual(len({row["position_hash_sha256"] for row in rows}), 1)

    def test_validator_is_order_independent_and_fail_closed(self) -> None:
        _, _, _, log = _realize("graded_fine")
        shuffled = list(log)
        random.Random(13).shuffle(shuffled)
        valid = family.validate_local_mesh_line_log(
            shuffled, GEOMETRY, "graded_fine"
        )
        self.assertTrue(valid["passed"], valid["errors"])
        self.assertFalse(
            family.validate_local_mesh_line_log(
                shuffled[:-1], GEOMETRY, "graded_fine"
            )["passed"]
        )
        corrupted = [dict(row) for row in shuffled]
        row = next(
            item for item in corrupted
            if item["target"] == "source_axial_anchor_9"
        )
        row["output_negative_spacing_cm"] = 9.0e-7
        self.assertFalse(
            family.validate_local_mesh_line_log(
                corrupted, GEOMETRY, "graded_fine"
            )["passed"]
        )

    def test_restoration_and_invalid_inputs(self) -> None:
        module, downstream, _, _ = _realize("graded_base")
        self.assertIs(module.add_2d_mesh_line, downstream)
        module = types.SimpleNamespace(add_2d_mesh_line=downstream)
        original = module.add_2d_mesh_line
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with family.local_runtime_mesh_lines(
                module, GEOMETRY, "graded_base", []
            ):
                raise RuntimeError("synthetic failure")
        self.assertIs(module.add_2d_mesh_line, original)
        with self.assertRaisesRegex(ValueError, "unknown graded mesh level"):
            family.get_level_spec("graded_imaginary")
        with self.assertRaisesRegex(ValueError, "too short"):
            family.build_local_mesh_policy(
                dict(GEOMETRY, z_drain=1.0e-6), "graded_base"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
