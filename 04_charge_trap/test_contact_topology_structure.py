"""DEVSIM-free tests for the characterization contact-topology adapter."""

from __future__ import annotations

import types
import unittest

import contact_topology_structure as topology


def _geometry() -> dict[str, float]:
    return {
        "r_axis": 0.0,
        "r_core": 1.0e-6,
        "r_mos2": 1.2e-6,
        "r_air_outer": 4.0e-6,
        "z_source": 0.0,
        "z_drain": 1.0e-5,
    }


class FakeStructureModule(types.SimpleNamespace):
    def __init__(self) -> None:
        super().__init__()
        self.mesh_lines: list[dict[str, object]] = []
        self.regions: list[dict[str, object]] = []
        self.contacts: list[dict[str, object]] = []
        self.add_2d_mesh_line = self._add_2d_mesh_line
        self.add_2d_region = self._add_2d_region
        self.add_2d_contact = self._add_2d_contact
        self.calculate_geometry = lambda *args, **kwargs: _geometry()
        self.create_structure = self._create_structure

    def _add_2d_mesh_line(self, *args, **kwargs):
        self.mesh_lines.append(dict(kwargs))

    def _add_2d_region(self, *args, **kwargs):
        self.regions.append(dict(kwargs))

    def _add_2d_contact(self, *args, **kwargs):
        self.contacts.append(dict(kwargs))

    def _create_structure(self, *args, **kwargs):
        self.add_2d_mesh_line(mesh="gaa_mesh", dir="x", pos=0.0, ps=1.0e-7)
        self.add_2d_mesh_line(mesh="gaa_mesh", dir="y", pos=0.0, ps=2.0e-7)
        self.add_2d_mesh_line(mesh="gaa_mesh", dir="y", pos=1.0e-5, ps=2.0e-7)
        for name, material, xl, xh in (
            ("CoreOxide", "Oxide", 0.0, 1.0e-6),
            ("MoS2", "MoS2", 1.0e-6, 1.2e-6),
            ("GateMetal", "Metal", 3.6e-6, 3.8e-6),
            ("Air", "Air", 3.8e-6, 4.0e-6),
        ):
            self.add_2d_region(
                mesh="gaa_mesh",
                material=material,
                region=name,
                xl=xl,
                xh=xh,
                yl=0.0,
                yh=1.0e-5,
            )
        self.add_2d_contact(
            mesh="gaa_mesh",
            name="source",
            region="MoS2",
            material="Metal",
            xl=1.0e-6,
            xh=1.2e-6,
            yl=0.0,
            yh=0.0,
            bloat=1.0e-10,
        )
        self.add_2d_contact(
            mesh="gaa_mesh",
            name="drain",
            region="MoS2",
            material="Metal",
            xl=1.0e-6,
            xh=1.2e-6,
            yl=1.0e-5,
            yh=1.0e-5,
            bloat=1.0e-10,
        )
        return _geometry()


class ContactTopologyStructureTests(unittest.TestCase):
    def test_air_background_and_guard_lines_preserve_active_geometry(self) -> None:
        module = FakeStructureModule()
        create_structure = topology.bind_characterization_create_structure(module)

        returned = create_structure()

        self.assertEqual(returned, _geometry())
        self.assertEqual(
            [(row["dir"], row["pos"]) for row in module.mesh_lines[:2]],
            [("y", -2.0e-7), ("y", 1.02e-5)],
        )
        self.assertEqual(module.regions[0]["region"], "Air")
        self.assertEqual(
            {
                key: module.regions[0][key]
                for key in ("xl", "xh", "yl", "yh")
            },
            {
                "xl": 0.0,
                "xh": 4.0e-6,
                "yl": -2.0e-7,
                "yh": 1.02e-5,
            },
        )
        self.assertEqual(
            [row["region"] for row in module.regions[1:]],
            ["CoreOxide", "MoS2", "GateMetal"],
        )
        mos2 = next(row for row in module.regions if row["region"] == "MoS2")
        self.assertEqual((mos2["yl"], mos2["yh"]), (0.0, 1.0e-5))
        self.assertEqual(
            [(row["name"], row["region"]) for row in module.contacts],
            [("source", "MoS2"), ("drain", "MoS2")],
        )

    def test_composes_with_existing_mesh_spacing_wrapper(self) -> None:
        module = FakeStructureModule()
        raw_add_line = module.add_2d_mesh_line
        scaled_calls: list[dict[str, object]] = []

        def scaled_add_line(*args, **kwargs):
            forwarded = dict(kwargs)
            forwarded["ps"] = float(forwarded["ps"]) * 0.5
            scaled_calls.append(dict(forwarded))
            return raw_add_line(*args, **forwarded)

        module.add_2d_mesh_line = scaled_add_line
        create_structure = topology.bind_characterization_create_structure(module)
        create_structure()

        self.assertIs(module.add_2d_mesh_line, scaled_add_line)
        self.assertEqual(len(scaled_calls), 5)
        self.assertEqual(
            [row["ps"] for row in scaled_calls[:2]],
            [1.0e-7, 1.0e-7],
        )

    def test_restores_module_functions_after_build_exception(self) -> None:
        module = FakeStructureModule()
        original_add_line = module.add_2d_mesh_line
        original_add_region = module.add_2d_region

        def failing_create_structure(*args, **kwargs):
            module.add_2d_mesh_line(
                mesh="gaa_mesh", dir="x", pos=0.0, ps=1.0e-7
            )
            raise RuntimeError("synthetic mesh failure")

        module.create_structure = failing_create_structure
        create_structure = topology.bind_characterization_create_structure(module)

        with self.assertRaisesRegex(RuntimeError, "synthetic mesh failure"):
            create_structure()

        self.assertIs(module.add_2d_mesh_line, original_add_line)
        self.assertIs(module.add_2d_region, original_add_region)

    def test_geometry_helpers_derive_guard_from_runtime_mos2(self) -> None:
        bounds = topology.expanded_air_bounds(_geometry())
        self.assertAlmostEqual(bounds["guard_cm"], 2.0e-7)
        self.assertAlmostEqual(bounds["yl"], -2.0e-7)
        self.assertAlmostEqual(bounds["yh"], 1.02e-5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
