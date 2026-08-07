"""DEVSIM-free tests for the fixed-position local mesh family."""

from __future__ import annotations

import math
import random
import types
import unittest

import local_mesh_family as mesh_family


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
            raise AssertionError("test downstream accepts keywords only")
        downstream_rows.append(dict(kwargs))

    module = types.SimpleNamespace(add_2d_mesh_line=downstream)
    log: list[dict[str, object]] = []
    with mesh_family.local_runtime_mesh_lines(module, GEOMETRY, level, log):
        for direction, position, spacing in ORIGINAL_LINES:
            module.add_2d_mesh_line(
                mesh="gaa_mesh", dir=direction, pos=position, ps=spacing
            )
    return module, downstream, downstream_rows, log


class LocalMeshFamilyTests(unittest.TestCase):
    def test_fixed_positions_and_dyadic_spacing_ladder(self) -> None:
        realized = {}
        hashes = set()
        for level in mesh_family.LEVELS:
            _, _, rows, log = _realize(level)
            self.assertEqual(len(rows), 31)
            self.assertEqual(len(log), 31)
            report = mesh_family.validate_local_mesh_line_log(
                log, GEOMETRY, level
            )
            self.assertTrue(report["passed"], report["errors"])
            keys = {(row["dir"], round(float(row["pos"]), 18)) for row in rows}
            self.assertEqual(len(keys), len(rows))
            realized[level] = {
                (str(row["target"])): (
                    float(row["output_negative_spacing_cm"]),
                    float(row["output_positive_spacing_cm"]),
                )
                for row in log
            }
            hashes.add(report["position_hash"])

        self.assertEqual(len(hashes), 1)
        for earlier, later in zip(mesh_family.LEVELS, mesh_family.LEVELS[1:]):
            for target, spacings in realized[earlier].items():
                later_spacings = realized[later][target]
                if target.startswith("radial_boundary_") or target.endswith("_air_guard"):
                    self.assertEqual(later_spacings, spacings)
                else:
                    for side, (spacing, later_spacing) in enumerate(
                        zip(spacings, later_spacings)
                    ):
                        passthrough_side = (
                            target == "mos2_radial_anchor_0" and side == 0
                        ) or (
                            target == "mos2_radial_anchor_4" and side == 1
                        )
                        if passthrough_side:
                            self.assertEqual(later_spacing, spacing)
                        else:
                            self.assertLessEqual(later_spacing, spacing)

            # The electrically important 0--2 nm interval remains exactly
            # dyadic.  The farther transition/bulk intervals may hit their
            # fixed anti-overrefinement floor.
            for side_name in ("source", "drain"):
                for index in range(5):
                    target = f"{side_name}_axial_anchor_{index}"
                    for side, (spacing, later_spacing) in enumerate(
                        zip(realized[earlier][target], realized[later][target])
                    ):
                        guard_side = index == 0 and (
                            (side_name == "source" and side == 0)
                            or (side_name == "drain" and side == 1)
                        )
                        expected = spacing if guard_side else 0.5 * spacing
                        self.assertTrue(math.isclose(
                            later_spacing, expected,
                            rel_tol=1.0e-14, abs_tol=1.0e-30,
                        ), f"{target} side={side}")

    def test_expected_anchor_positions_and_no_duplicate_endpoints(self) -> None:
        _, _, rows, log = _realize("local_base")
        x_positions_nm = sorted(
            float(row["position_cm"]) / mesh_family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("mos2_radial_anchor_")
        )
        for actual, expected in zip(
            x_positions_nm, (10.0, 10.5, 11.0, 11.5, 12.0)
        ):
            self.assertAlmostEqual(actual, expected, places=12)

        source_nm = sorted(
            float(row["position_cm"]) / mesh_family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("source_axial_anchor_")
        )
        drain_inward_nm = sorted(
            (GEOMETRY["z_drain"] - float(row["position_cm"]))
            / mesh_family.NM_TO_CM
            for row in log
            if str(row["target"]).startswith("drain_axial_anchor_")
        )
        for actual_values in (source_nm, drain_inward_nm):
            for actual, expected in zip(
                actual_values, mesh_family.AXIAL_DISTANCE_NM
            ):
                self.assertAlmostEqual(actual, expected, places=12)
        positions = [(row["dir"], row["pos"]) for row in rows]
        self.assertEqual(len(positions), len(set(positions)))

    def test_dielectric_and_other_radial_lines_are_unchanged(self) -> None:
        _, _, _, log = _realize("local_ultra_fine")
        expected = {
            "radial_boundary_r_axis": 1.0e-7,
            "radial_boundary_r_tox": 5.0e-8,
            "radial_boundary_r_trap": 5.0e-8,
            "radial_boundary_r_block": 5.0e-8,
            "radial_boundary_r_gate_outer": 5.0e-8,
            "radial_boundary_r_air_outer": 1.0e-7,
        }
        actual = {
            str(row["target"]): float(row["output_spacing_cm"])
            for row in log
            if str(row["target"]).startswith("radial_boundary_")
        }
        self.assertEqual(actual, expected)

    def test_validator_is_order_independent_and_rejects_corruption(self) -> None:
        _, _, _, log = _realize("local_fine")
        shuffled = list(log)
        random.Random(17).shuffle(shuffled)
        report = mesh_family.validate_local_mesh_line_log(
            shuffled, GEOMETRY, "local_fine"
        )
        self.assertTrue(report["passed"], report["errors"])

        missing = shuffled[:-1]
        self.assertFalse(
            mesh_family.validate_local_mesh_line_log(
                missing, GEOMETRY, "local_fine"
            )["passed"]
        )
        wrong_spacing = [dict(row) for row in shuffled]
        controlled = next(
            row for row in wrong_spacing if row["target"] == "source_axial_anchor_0"
        )
        controlled["output_spacing_cm"] = float(
            controlled["output_spacing_cm"]
        ) * 2.0
        controlled["output_positive_spacing_cm"] = controlled[
            "output_spacing_cm"
        ]
        controlled["scaled_spacing_cm"] = controlled["output_spacing_cm"]
        self.assertFalse(
            mesh_family.validate_local_mesh_line_log(
                wrong_spacing, GEOMETRY, "local_fine"
            )["passed"]
        )

    def test_context_restores_downstream_on_success_and_failure(self) -> None:
        module, downstream, _, _ = _realize("local_base")
        self.assertIs(module.add_2d_mesh_line, downstream)

        module = types.SimpleNamespace(add_2d_mesh_line=downstream)
        original = module.add_2d_mesh_line
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with mesh_family.local_runtime_mesh_lines(
                module, GEOMETRY, "local_base", []
            ):
                raise RuntimeError("synthetic failure")
        self.assertIs(module.add_2d_mesh_line, original)

    def test_family_definition_hash_excludes_spacing(self) -> None:
        rows = mesh_family.local_mesh_family_definition_rows(GEOMETRY)
        self.assertEqual(len(rows), 31 * len(mesh_family.LEVELS))
        hashes = {row["position_hash_sha256"] for row in rows}
        self.assertEqual(len(hashes), 1)
        position_sets = {
            level: {
                (row["direction"], row["position_cm"])
                for row in rows
                if row["mesh_level"] == level
            }
            for level in mesh_family.LEVELS
        }
        self.assertTrue(
            all(
                position_sets[level] == position_sets[mesh_family.LEVELS[0]]
                for level in mesh_family.LEVELS[1:]
            )
        )

    def test_unknown_level_and_overlapping_anchors_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown local mesh level"):
            mesh_family.get_level_spec("local_imaginary")
        short = dict(GEOMETRY, z_drain=1.0e-6)
        with self.assertRaisesRegex(ValueError, "too short"):
            mesh_family.build_local_mesh_policy(short, "local_base")


if __name__ == "__main__":
    unittest.main(verbosity=2)
