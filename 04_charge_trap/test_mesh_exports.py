"""Standard-library tests for final mesh/VTK export helpers."""

from __future__ import annotations

import base64
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import export_handoff_mesh as mesh_export
import handoff_metadata


class FakeBiasRuntime:
    def __init__(self, reference: dict[str, object]) -> None:
        self.values = {
            "gate_bias": float(reference["VGS_V"]),
            "drain_bias": float(reference["VDS_V"]),
            "source_bias": float(reference["source_V"]),
            "NtrapParameter": float(reference["ntrap_cm3"]),
        }

    def get_parameter(
        self,
        *,
        device: str,
        name: str,
        region: str | None = None,
    ) -> float:
        return self.values[name]


class FakeModelRuntime:
    def __init__(self) -> None:
        self.nodes = {
            region: {"Potential", "CylindricalNodeVolume", "AtContactNode"}
            for region in mesh_export.PHYSICS_REGIONS
        }
        self.nodes["MoS2"].update({"Electrons", "NetDoping", "Holes"})
        self.nodes["ChargeTrap"].update(
            {"TrappedElectronDensity", "TrappedChargeDensity"}
        )
        self.edges = {
            region: {"ElectricField"}
            for region in mesh_export.PHYSICS_REGIONS
        }
        self.edges["MoS2"].add("ElectronCurrent")
        self.elements = {region: set() for region in mesh_export.PHYSICS_REGIONS}
        self.created_equations: list[str] = []

    def get_node_model_list(self, *, device: str, region: str) -> tuple[str, ...]:
        return tuple(self.nodes[region])

    def get_edge_model_list(self, *, device: str, region: str) -> tuple[str, ...]:
        return tuple(self.edges[region])

    def get_element_model_list(self, *, device: str, region: str) -> tuple[str, ...]:
        return tuple(self.elements[region])

    def element_from_edge_model(
        self,
        *,
        device: str,
        region: str,
        edge_model: str,
    ) -> None:
        self.elements[region].update({f"{edge_model}_x", f"{edge_model}_y"})

    def element_model(
        self,
        *,
        device: str,
        region: str,
        name: str,
        equation: str,
        display_type: str,
    ) -> None:
        self.created_equations.append(equation)
        self.elements[region].add(name)


def _write_synthetic_bundle(
    base: Path,
    *,
    ntrap_cm3: float,
) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    geometry = handoff_metadata.expected_geometry()
    z_max = geometry["channel_length_nm"] * 1.0e-7
    vtu_names: list[str] = []

    model_names_by_region = {
        "CoreOxide": {
            "Potential",
            "ElectricField_x",
            "ElectricField_y",
            "ElectricFieldMagnitude",
        },
        "MoS2": {
            "Electrons",
            "NetDoping",
            "ElectronCurrent_x",
            "ElectronCurrent_y",
            "ElectronCurrentMagnitude",
        },
        "ChargeTrap": {"TrappedElectronDensity", "TrappedChargeDensity"},
    }
    radial_bounds = mesh_export._expected_region_radial_bounds_cm()
    for index, region in enumerate(mesh_export.EXPECTED_VTK_REGION_ORDER):
        filename = f"{base.name}_{region}.vtu"
        vtu_names.append(filename)
        arrays = []
        for name in sorted(model_names_by_region.get(region, ())):
            if name == "TrappedElectronDensity":
                values = f"{ntrap_cm3} {ntrap_cm3} {ntrap_cm3}"
            elif name == "TrappedChargeDensity":
                charge = -1.602176634e-19 * ntrap_cm3
                values = f"{charge} {charge} {charge}"
            else:
                values = "1 1 1"
            arrays.append(
                f'<DataArray type="Float64" Name="{name}" '
                f'format="ascii">{values}</DataArray>'
            )
        point_data = "".join(arrays)
        r_min, r_max = radial_bounds[region]
        xml = (
            '<?xml version="1.0"?>'
            '<VTKFile type="UnstructuredGrid" version="0.1">'
            '<UnstructuredGrid><Piece NumberOfPoints="3" NumberOfCells="1">'
            f'<PointData>{point_data}</PointData><CellData></CellData>'
            '<Points><DataArray type="Float64" NumberOfComponents="3" '
            f'format="ascii">{r_min} 0 0 {r_max} 0 0 '
            f'{r_min} {z_max} 0</DataArray></Points>'
            '<Cells>'
            '<DataArray type="Int32" Name="connectivity" format="ascii">0 1 2</DataArray>'
            '<DataArray type="Int32" Name="offsets" format="ascii">3</DataArray>'
            '<DataArray type="UInt8" Name="types" format="ascii">5</DataArray>'
            '</Cells></Piece></UnstructuredGrid></VTKFile>'
        )
        (base.parent / filename).write_text(xml, encoding="utf-8")

    datasets = "".join(
        f'<DataSet index="{index}" name="{region}" file="{filename}"/>'
        for index, (region, filename) in enumerate(
            zip(mesh_export.EXPECTED_VTK_REGION_ORDER, vtu_names)
        )
    )
    vtm = (
        '<?xml version="1.0"?>'
        '<VTKFile type="vtkMultiBlockDataSet" version="1.0">'
        f'<vtkMultiBlockDataSet>{datasets}</vtkMultiBlockDataSet></VTKFile>'
    )
    base.with_suffix(".vtm").write_text(vtm, encoding="utf-8")
    base.with_suffix(".visit").write_text(
        "!NBLOCKS 7\n" + "\n".join(vtu_names) + "\n",
        encoding="utf-8",
    )


def _write_synthetic_mesh(path: Path) -> None:
    lines = ["begin_device mock"]
    lines.extend(
        f'begin_region "{name}" "{material}"'
        for name, material in handoff_metadata.EXPECTED_REGIONS.items()
    )
    lines.extend(
        f'begin_contact "{name}" "{region}" "{material}"'
        for name, (region, material) in mesh_export.EXPECTED_MESH_CONTACTS.items()
    )
    lines.extend(
        f'begin_interface "{name}" "{region0}" "{region1}"'
        for name, (region0, region1) in mesh_export.EXPECTED_MESH_INTERFACES.items()
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class MeshExportTests(unittest.TestCase):
    def test_generated_trailing_blank_line_removal_is_minimal_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "generated.vtu"
            output.write_bytes(b"<VTKFile>payload</VTKFile>\n\n")
            self.assertTrue(mesh_export._remove_generated_trailing_blank_lines(output))
            self.assertEqual(output.read_bytes(), b"<VTKFile>payload</VTKFile>\n")
            self.assertFalse(mesh_export._remove_generated_trailing_blank_lines(output))

    def test_devsim_inline_zlib_binary_values_are_decoded(self) -> None:
        expected = (0.0, 1.25, -3.5, 2.0e18)
        raw = struct.pack("<4d", *expected)
        compressed = zlib.compress(raw)
        header = struct.pack("<4I", 1, 32768, len(raw), len(compressed))
        encoded = (
            base64.b64encode(header).decode("ascii")
            + base64.b64encode(compressed).decode("ascii")
        )
        root = ET.fromstring(
            '<VTKFile type="UnstructuredGrid" byte_order="LittleEndian" '
            'header_type="UInt32" compressor="vtkZLibDataCompressor">'
            f'<DataArray type="Float64" format="binary">{encoded}</DataArray>'
            "</VTKFile>"
        )
        array = next(iter(root))
        actual = mesh_export._read_data_array_values(
            array,
            root,
            Path("synthetic.vtu"),
        )
        self.assertEqual(actual, expected)

    def test_reference_contract_contains_four_required_biases(self) -> None:
        references = mesh_export.reference_specifications()
        self.assertEqual(
            tuple(reference["slug"] for reference in references),
            (
                "empty_off_reference",
                "empty_on_reference",
                "programmed_off_reference",
                "programmed_on_reference",
            ),
        )
        self.assertEqual(
            {(reference["VGS_V"], reference["VDS_V"]) for reference in references},
            {(-1.0, 0.05), (3.0, 0.05)},
        )
        self.assertEqual(references[0]["ntrap_cm3"], 0.0)
        self.assertEqual(references[-1]["ntrap_cm3"], 2.0e18)

    def test_export_allowlist_omits_fixed_holes_and_parameter_only_permittivity(self) -> None:
        self.assertNotIn("Holes", mesh_export.EXPORT_MODEL_NAMES)
        self.assertNotIn("Permittivity", mesh_export.EXPORT_MODEL_NAMES)
        self.assertIn("ElectricFieldMagnitude", mesh_export.EXPORT_MODEL_NAMES)
        self.assertIn("ElectronCurrentMagnitude", mesh_export.EXPORT_MODEL_NAMES)

    def test_visualization_models_are_derived_from_existing_edges(self) -> None:
        runtime = FakeModelRuntime()
        report = mesh_export.prepare_visualization_models(runtime_api=runtime)
        self.assertIn("ElectricField_x", report["TunnelOxide"])
        self.assertIn("ElectronCurrent_x", report["MoS2"])
        self.assertIn("TrappedChargeDensity", report["ChargeTrap"])
        self.assertTrue(runtime.created_equations)
        self.assertTrue(all("^0.5" in equation for equation in runtime.created_equations))
        self.assertTrue(all(equation.strip() != "0" for equation in runtime.created_equations))

    def test_bias_assertion_reads_actual_parameters(self) -> None:
        reference = dict(mesh_export.reference_specifications()[0])
        runtime = FakeBiasRuntime(reference)
        report = mesh_export.assert_reference_bias_and_trap(
            reference,
            runtime_api=runtime,
            trap_parameter_name="NtrapParameter",
        )
        self.assertEqual(report["gate_bias"], -1.0)
        runtime.values["gate_bias"] = -0.9
        with self.assertRaisesRegex(RuntimeError, "gate_bias"):
            mesh_export.assert_reference_bias_and_trap(
                reference,
                runtime_api=runtime,
                trap_parameter_name="NtrapParameter",
            )

    def test_vtm_and_vtu_smoke_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory) / "reference" / "device"
            _write_synthetic_bundle(base, ntrap_cm3=0.0)
            report = mesh_export.validate_vtk_bundle(base)
            self.assertEqual(report["region_file_count"], 7)
            self.assertGreater(report["point_count"], 0)
            self.assertIn("Potential", report["model_names"])
            coordinates = mesh_export.read_point_coordinates_from_bundle(base)
            self.assertGreater(len(coordinates), 0)
            trapped = mesh_export.read_model_values_from_bundle(
                base,
                "TrappedElectronDensity",
            )
            self.assertTrue(all(value == 0.0 for value in trapped))

    def test_vtm_rejects_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory) / "device"
            base.with_suffix(".vtm").write_text(
                '<?xml version="1.0"?><VTKFile type="vtkMultiBlockDataSet">'
                '<vtkMultiBlockDataSet><DataSet file="missing.vtu"/>'
                '</vtkMultiBlockDataSet></VTKFile>',
                encoding="utf-8",
            )
            with self.assertRaises(FileNotFoundError):
                mesh_export.read_vtm_dataset_paths(base.with_suffix(".vtm"))

    def test_absolute_vtm_and_visit_references_are_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            base = Path(temporary_directory) / "reference" / "device"
            _write_synthetic_bundle(base, ntrap_cm3=0.0)
            vtm = base.with_suffix(".vtm")
            tree = ET.parse(vtm)
            for dataset in tree.getroot().iter("DataSet"):
                dataset.set(
                    "file",
                    str((base.parent / dataset.attrib["file"]).resolve()),
                )
            tree.write(vtm, encoding="utf-8", xml_declaration=True)
            absolute_names = [
                str((base.parent / line).resolve())
                for line in base.with_suffix(".visit")
                .read_text(encoding="utf-8")
                .splitlines()[1:]
                if line
            ]
            base.with_suffix(".visit").write_text(
                "!NBLOCKS 7\n" + "\n".join(absolute_names) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "portable relative basenames"):
                mesh_export.read_vtm_dataset_paths(vtm)
            report = mesh_export.normalize_vtk_bundle_references(base)
            self.assertEqual(report["vtm_files"], report["visit_files"])
            self.assertTrue(
                all(Path(filename).name == filename for filename in report["vtm_files"])
            )
            validated = mesh_export.validate_vtk_bundle(base)
            self.assertEqual(validated["region_file_count"], 7)

    def test_complete_reference_validation_checks_mesh_charge_and_extrema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            _write_synthetic_mesh(root / mesh_export.FINAL_MESH_FILENAME)
            for reference in mesh_export.reference_specifications():
                base = root / str(reference["slug"]) / str(reference["base_name"])
                _write_synthetic_bundle(
                    base,
                    ntrap_cm3=float(reference["ntrap_cm3"]),
                )
            report = mesh_export.validate_reference_exports(root)
            self.assertTrue(report["valid"])
            self.assertEqual(report["mesh_topology"]["region_count"], 7)
            self.assertEqual(report["mesh_topology"]["contact_count"], 3)
            self.assertEqual(report["mesh_topology"]["interface_count"], 4)
            self.assertEqual(len(report["references"]), 4)
            for bundle in report["references"].values():
                self.assertEqual(bundle["region_file_count"], 7)
                self.assertAlmostEqual(
                    bundle["coordinate_extrema_cm"]["r_max"],
                    4.0e-6,
                )

    def test_mesh_topology_rejects_duplicate_region_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            mesh = Path(temporary_directory) / mesh_export.FINAL_MESH_FILENAME
            _write_synthetic_mesh(mesh)
            with mesh.open("a", encoding="utf-8") as stream:
                stream.write('begin_region "Air" "Air"\n')
            with self.assertRaisesRegex(ValueError, "Duplicate begin_region"):
                mesh_export._validate_mesh_file(mesh)


if __name__ == "__main__":
    unittest.main(verbosity=2)
