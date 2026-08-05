"""DEVSIM-free tests for the compact-handoff runtime parameter contract.

These tests validate source-of-truth wiring and public compatibility names.
They do not create a device, run PROGRAM/ERASE, or invoke a DEVSIM solve.
"""

from __future__ import annotations

import ast
import importlib.util
import math
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
CANONICAL_PATH = REPOSITORY_ROOT / "compact_handoff_parameters.py"


def load_module(module_name: str, path: Path):
    """Load one source file under a collision-free test module name."""

    specification = importlib.util.spec_from_file_location(module_name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {path}.")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def fake_devsim_module() -> types.ModuleType:
    """Provide import-only placeholders for structure-module dependencies."""

    module = types.ModuleType("devsim")
    for function_name in (
        "create_2d_mesh",
        "add_2d_mesh_line",
        "add_2d_region",
        "add_2d_interface",
        "add_2d_contact",
        "finalize_mesh",
        "create_device",
        "set_parameter",
        "cylindrical_edge_couple",
        "cylindrical_node_volume",
        "cylindrical_surface_area",
    ):
        setattr(module, function_name, lambda *args, **kwargs: None)
    return module


class RuntimeParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.canonical = load_module(
            "compact_handoff_parameters_tested",
            CANONICAL_PATH,
        )

    def test_canonical_public_contract(self) -> None:
        required_names = (
            "GEOMETRY_VERSION",
            "MATERIAL_VERSION",
            "DEVICE_TYPE",
            "COORDINATE_SYSTEM",
            "CORE_RADIUS_NM",
            "MOS2_THICKNESS_NM",
            "TUNNEL_OXIDE_THICKNESS_NM",
            "CHARGE_TRAP_THICKNESS_NM",
            "BLOCKING_OXIDE_THICKNESS_NM",
            "GATE_METAL_THICKNESS_NM",
            "AIR_THICKNESS_NM",
            "CHANNEL_LENGTH_NM",
            "AL2O3_RELATIVE_PERMITTIVITY",
            "HFO2_RELATIVE_PERMITTIVITY",
            "MOS2_RELATIVE_PERMITTIVITY",
            "CORE_OXIDE_RELATIVE_PERMITTIVITY",
            "geometry_dict",
            "material_dict",
        )
        for name in required_names:
            self.assertTrue(hasattr(self.canonical, name), name)

        self.assertEqual(
            self.canonical.geometry_dict(),
            {
                "core_radius_nm": 10.0,
                "mos2_thickness_nm": 2.0,
                "tunnel_oxide_thickness_nm": 3.0,
                "charge_trap_thickness_nm": 5.0,
                "blocking_oxide_thickness_nm": 16.0,
                "gate_metal_thickness_nm": 2.0,
                "air_thickness_nm": 2.0,
                "channel_length_nm": 100.0,
            },
        )
        self.assertEqual(
            self.canonical.material_dict(),
            {
                "CoreOxide": 3.9,
                "MoS2": 6.7,
                "TunnelOxide": 8.9,
                "ChargeTrap": 19.65,
                "BlockingOxide": 8.9,
            },
        )

    def test_parameter_import_is_independent_of_working_directory(self) -> None:
        original_directory = Path.cwd()
        try:
            os.chdir(REPOSITORY_ROOT.parent)
            material = load_module(
                "charge_trap_material_from_other_cwd",
                MODULE_DIRECTORY / "material_parameters.py",
            )
        finally:
            os.chdir(original_directory)

        self.assertEqual(material.relative_permittivity_al2o3, 8.9)
        self.assertEqual(material.relative_permittivity_hfo2, 19.65)
        self.assertEqual(
            Path(material.compact_parameters.__file__).resolve(),
            CANONICAL_PATH.resolve(),
        )

    def test_material_compatibility_modules_match(self) -> None:
        charge_trap_material = load_module(
            "charge_trap_material_parameters_tested",
            MODULE_DIRECTORY / "material_parameters.py",
        )
        program_material = load_module(
            "program_material_parameters_tested",
            REPOSITORY_ROOT / "05_program_erase" / "material_parameters.py",
        )

        for material in (charge_trap_material, program_material):
            self.assertEqual(material.relative_permittivity_al2o3, 8.9)
            self.assertEqual(material.relative_permittivity_hfo2, 19.65)
            self.assertEqual(material.relative_permittivity_mos2, 6.7)
            self.assertEqual(material.relative_permittivity_core_oxide, 3.9)
            self.assertTrue(
                math.isclose(material.tunnel_oxide_thickness, 3.0e-7)
            )
            self.assertTrue(
                math.isclose(material.charge_trap_thickness, 5.0e-7)
            )
            self.assertTrue(
                math.isclose(material.blocking_oxide_thickness, 16.0e-7)
            )

    def test_trap_compatibility_modules_use_five_nm(self) -> None:
        for directory_name, module_name in (
            ("04_charge_trap", "charge_trap_parameters_tested"),
            ("05_program_erase", "program_trap_parameters_tested"),
        ):
            trap_parameters = load_module(
                module_name,
                REPOSITORY_ROOT / directory_name / "trap_parameters.py",
            )
            self.assertTrue(
                math.isclose(trap_parameters.charge_trap_thickness, 5.0e-7)
            )
            self.assertEqual(
                trap_parameters.volume_to_sheet_density(2.0e18),
                1.0e12,
            )

    def test_parameterized_structure_defaults_and_radii(self) -> None:
        with patch.dict(sys.modules, {"devsim": fake_devsim_module()}):
            structure = load_module(
                "program_device_structure_tested",
                REPOSITORY_ROOT / "05_program_erase" / "device_structure.py",
            )

        self.assertEqual(structure.DEFAULT_CORE_RADIUS_NM, 10.0)
        self.assertEqual(structure.DEFAULT_MOS2_THICKNESS_NM, 2.0)
        self.assertEqual(structure.DEFAULT_TUNNEL_OXIDE_THICKNESS_NM, 3.0)
        self.assertEqual(structure.DEFAULT_CHARGE_TRAP_THICKNESS_NM, 5.0)
        self.assertEqual(structure.DEFAULT_BLOCKING_OXIDE_THICKNESS_NM, 16.0)
        self.assertEqual(structure.DEFAULT_GATE_METAL_THICKNESS_NM, 2.0)
        self.assertEqual(structure.DEFAULT_AIR_THICKNESS_NM, 2.0)
        self.assertEqual(structure.DEFAULT_CHANNEL_LENGTH_NM, 100.0)

        geometry = structure.calculate_geometry()
        expected_radii = {
            "core_outer_radius_nm": 10.0,
            "mos2_outer_radius_nm": 12.0,
            "tunnel_oxide_outer_radius_nm": 15.0,
            "charge_trap_outer_radius_nm": 20.0,
            "blocking_oxide_outer_radius_nm": 36.0,
            "gate_outer_radius_nm": 38.0,
            "air_outer_radius_nm": 40.0,
        }
        for name, expected_value in expected_radii.items():
            self.assertEqual(geometry[name], expected_value)

    def test_legacy_structure_public_radii_follow_canonical_geometry(self) -> None:
        with patch.dict(sys.modules, {"devsim": fake_devsim_module()}):
            structure = load_module(
                "charge_trap_device_structure_tested",
                MODULE_DIRECTORY / "device_structure.py",
            )

        scale = self.canonical.NM_TO_CM
        self.assertTrue(math.isclose(structure.r_core, 10.0 * scale))
        self.assertTrue(math.isclose(structure.r_mos2, 12.0 * scale))
        self.assertTrue(math.isclose(structure.r_tox, 15.0 * scale))
        self.assertTrue(math.isclose(structure.r_trap, 20.0 * scale))
        self.assertTrue(math.isclose(structure.r_block, 36.0 * scale))
        self.assertTrue(math.isclose(structure.r_gate_outer, 38.0 * scale))
        self.assertTrue(math.isclose(structure.r_air_outer, 40.0 * scale))
        self.assertTrue(math.isclose(structure.z_drain, 100.0 * scale))

    def test_state_initializer_explicitly_passes_and_verifies_all_geometry(self) -> None:
        source_path = MODULE_DIRECTORY / "run_memory_window.py"
        syntax_tree = ast.parse(
            source_path.read_text(encoding="utf-8"),
            filename=str(source_path),
        )
        initialize_function = next(
            node
            for node in syntax_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "initialize_device"
        )
        structure_call = next(
            node
            for node in ast.walk(initialize_function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "create_structure"
        )
        self.assertEqual(
            {keyword.arg for keyword in structure_call.keywords},
            {
                "core_radius_nm",
                "mos2_thickness_nm",
                "tunnel_oxide_thickness_nm",
                "charge_trap_thickness_nm",
                "blocking_oxide_thickness_nm",
                "gate_metal_thickness_nm",
                "air_thickness_nm",
                "channel_length_nm",
            },
        )

        called_names = {
            node.func.id
            for node in ast.walk(initialize_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("verify_runtime_geometry", called_names)
        self.assertIn("verify_runtime_material_parameters", called_names)

        helper_tree = ast.parse(
            (MODULE_DIRECTORY / "state_sweep_helpers.py").read_text(
                encoding="utf-8"
            )
        )
        helper_initializer = next(
            node
            for node in helper_tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "initialize_characterization_device"
        )
        helper_calls = {
            node.func.id
            for node in ast.walk(helper_initializer)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("validate_canonical_handoff_geometry", helper_calls)

    def test_final_program_geometry_uses_canonical_source(self) -> None:
        final_program = load_module(
            "final_program_geometry_tested",
            REPOSITORY_ROOT / "05_program_erase" / "run_final_program_case.py",
        )
        self.assertEqual(final_program.TUNNEL_OXIDE_THICKNESS_NM, 3.0)
        self.assertEqual(final_program.CHARGE_TRAP_THICKNESSES_NM, (5.0,))
        self.assertEqual(final_program.BLOCKING_OXIDE_THICKNESS_NM, 16.0)
        self.assertEqual(final_program.PROGRAM_GATE_VOLTAGE_V, 16.0)
        self.assertEqual(final_program.DRAIN_VOLTAGE_V, 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
