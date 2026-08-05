"""Unit tests for deterministic compact-handoff metadata and manifests."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import handoff_metadata as metadata


class FakeRuntimeAPI:
    """Small DEVSIM-shaped adapter for runtime-validator unit tests."""

    def __init__(self, geometry: dict[str, float]) -> None:
        canonical = metadata.canonical_parameters()
        self.device_parameters = {
            name: geometry[name]
            for name in (
                "core_radius_nm",
                "mos2_thickness_nm",
                "tunnel_oxide_thickness_nm",
                "charge_trap_thickness_nm",
                "blocking_oxide_thickness_nm",
                "channel_length_nm",
            )
        }
        self.device_parameters.update(
            {
                "raxis_variable": "x",
                "node_volume_model": "CylindricalNodeVolume",
                "edge_couple_model": "CylindricalEdgeCouple",
                "edge_node0_volume_model": "CylindricalEdgeNodeVolume@n0",
                "edge_node1_volume_model": "CylindricalEdgeNodeVolume@n1",
            }
        )
        eps0 = 8.8541878128e-14
        self.permittivities = {
            "CoreOxide": eps0 * float(canonical.CORE_OXIDE_RELATIVE_PERMITTIVITY),
            "MoS2": eps0 * float(canonical.MOS2_RELATIVE_PERMITTIVITY),
            "TunnelOxide": eps0 * float(canonical.AL2O3_RELATIVE_PERMITTIVITY),
            "ChargeTrap": eps0 * float(canonical.HFO2_RELATIVE_PERMITTIVITY),
            "BlockingOxide": eps0 * float(canonical.AL2O3_RELATIVE_PERMITTIVITY),
        }

    def get_region_list(self, *, device: str) -> tuple[str, ...]:
        return tuple(metadata.EXPECTED_REGIONS)

    def get_contact_list(self, *, device: str) -> tuple[str, ...]:
        return tuple(metadata.EXPECTED_CONTACTS)

    def get_interface_list(self, *, device: str) -> tuple[str, ...]:
        return tuple(metadata.EXPECTED_INTERFACES)

    def get_material(self, *, device: str, region: str) -> str:
        return metadata.EXPECTED_REGIONS[region]

    def get_parameter(
        self,
        *,
        device: str,
        name: str,
        region: str | None = None,
    ) -> object:
        if region is not None:
            if name == "Permittivity":
                return self.permittivities[region]
            if region == "MoS2" and name == "ElectronMobility":
                return 50.0
            if region == "MoS2" and name == "ThermalVoltage":
                return 1.380649e-23 * 300.0 / 1.602176634e-19
            raise KeyError(name)
        return self.device_parameters[name]

    def get_node_model_values(
        self,
        *,
        device: str,
        region: str,
        name: str,
    ) -> tuple[float, ...]:
        if region == "MoS2" and name == "Donors":
            return (1.0e15, 1.0e15, 1.0e15)
        raise KeyError(name)

    def get_node_model_list(self, *, device: str, region: str) -> tuple[str, ...]:
        models = {"Potential"}
        if region == "MoS2":
            models.update({"Electrons", "NetDoping"})
        if region == "ChargeTrap":
            models.update({"TrappedElectronDensity", "TrappedChargeDensity"})
        return tuple(models)

    def get_edge_model_list(self, *, device: str, region: str) -> tuple[str, ...]:
        models = {"ElectricField"}
        if region == "MoS2":
            models.add("ElectronCurrent")
        return tuple(models)

    def get_contact_equation_list(self, *, device: str, contact: str) -> tuple[str, ...]:
        if contact in {"source", "drain"}:
            return ("PotentialEquation", "ElectronContinuityEquation")
        return ("PotentialEquation",)


class HandoffMetadataTests(unittest.TestCase):
    def test_canonical_geometry_is_final_stack(self) -> None:
        geometry = metadata.expected_geometry()
        self.assertEqual(geometry["core_radius_nm"], 10.0)
        self.assertEqual(geometry["mos2_thickness_nm"], 2.0)
        self.assertEqual(geometry["tunnel_oxide_thickness_nm"], 3.0)
        self.assertEqual(geometry["charge_trap_thickness_nm"], 5.0)
        self.assertEqual(geometry["blocking_oxide_thickness_nm"], 16.0)
        self.assertEqual(geometry["blocking_oxide_outer_radius_nm"], 36.0)
        self.assertEqual(geometry["channel_length_nm"], 100.0)

    def test_geometry_validator_rejects_mismatch(self) -> None:
        geometry = metadata.expected_geometry()
        geometry["blocking_oxide_thickness_nm"] = 8.0
        with self.assertRaisesRegex(ValueError, "blocking_oxide_thickness_nm"):
            metadata.validate_geometry_mapping(geometry)

    def test_all_metadata_schemas_are_exact(self) -> None:
        geometry = metadata.expected_geometry()
        cases = (
            (
                metadata.DEVICE_STRUCTURE_FIELDNAMES,
                metadata.build_device_structure_rows(
                    geometry,
                    runtime_verified=True,
                ),
            ),
            (metadata.MATERIAL_FIELDNAMES, metadata.build_material_rows()),
            (
                metadata.SIMULATION_CONDITION_FIELDNAMES,
                metadata.build_simulation_condition_rows(),
            ),
            (
                metadata.REGION_CONTACT_FIELDNAMES,
                metadata.build_region_contact_rows(geometry),
            ),
        )
        for schema, rows in cases:
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual(tuple(row), schema)
                self.assertEqual(set(row), set(schema))

    def test_required_material_values_and_limitations_are_recorded(self) -> None:
        rows = metadata.build_material_rows(runtime_verified=True)
        by_parameter = {row["parameter"]: row for row in rows}
        relative_rows = [
            row for row in rows if row["parameter"] == "relative_permittivity"
        ]
        relative_values = {float(row["value"]) for row in relative_rows}
        self.assertIn(8.9, relative_values)
        self.assertIn(19.65, relative_values)
        self.assertIn(6.7, relative_values)
        self.assertIn(3.9, relative_values)
        self.assertEqual(by_parameter["hole_mobility"]["status"], "placeholder")
        self.assertEqual(by_parameter["hole_mobility"]["runtime_verified"], "false")
        self.assertEqual(by_parameter["electron_mobility"]["runtime_verified"], "true")
        self.assertIn(
            "not spatially implemented",
            by_parameter["source_drain_doping"]["notes"],
        )

    def test_region_map_is_complete(self) -> None:
        rows = metadata.build_region_contact_rows()
        counts = {
            entity: sum(row["entity_type"] == entity for row in rows)
            for entity in ("region", "interface", "contact")
        }
        self.assertEqual(counts, {"region": 7, "interface": 4, "contact": 3})
        gate = next(
            row
            for row in rows
            if row["entity_type"] == "contact" and row["name"] == "gate"
        )
        self.assertEqual(gate["adjacent_region"], "BlockingOxide")
        self.assertIn("GateMetal bulk is mesh-only", gate["notes"])

    def test_simulation_metadata_excludes_erase_and_retention(self) -> None:
        rows = metadata.build_simulation_condition_rows()
        serialized = " ".join(str(value) for row in rows for value in row.values()).lower()
        self.assertNotIn("erase", serialized)
        self.assertNotIn("retention", serialized)
        self.assertIn("cij=dqi/dvj", serialized)
        self.assertIn("0.0005;0.001;0.002", serialized)

    def test_runtime_validator_accepts_consistent_device(self) -> None:
        geometry = metadata.expected_geometry()
        report = metadata.validate_runtime_metadata(
            geometry,
            runtime_api=FakeRuntimeAPI(geometry),
        )
        self.assertTrue(report["runtime_verified"])
        self.assertEqual(len(report["regions"]), 7)
        self.assertAlmostEqual(
            report["relative_permittivities"]["TunnelOxide"],
            8.9,
        )

    def test_runtime_validator_rejects_material_mismatch(self) -> None:
        geometry = metadata.expected_geometry()
        runtime = FakeRuntimeAPI(geometry)
        runtime.permittivities["ChargeTrap"] = 20.0 * 8.8541878128e-14
        with self.assertRaisesRegex(RuntimeError, "ChargeTrap"):
            metadata.validate_runtime_metadata(geometry, runtime_api=runtime)

    def test_csv_writer_uses_exact_schema_and_lf(self) -> None:
        rows = metadata.build_device_structure_rows()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "device.csv"
            metadata.write_csv_rows(
                path,
                metadata.DEVICE_STRUCTURE_FIELDNAMES,
                rows,
            )
            raw = path.read_bytes()
            self.assertNotIn(b"\r\n", raw)
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(
                    tuple(reader.fieldnames or ()),
                    metadata.DEVICE_STRUCTURE_FIELDNAMES,
                )
                self.assertEqual(len(list(reader)), len(rows))

    def test_manifest_is_sorted_self_excluding_and_verifiable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            second = root / "zeta.csv"
            first = root / "alpha.md"
            manifest = root / "handoff_manifest.csv"
            second.write_text("a,b\n1,2\n", encoding="utf-8")
            first.write_text("# handoff\n", encoding="utf-8")
            manifest.write_text("placeholder", encoding="utf-8")

            rows = metadata.build_manifest_rows(
                root,
                (second, manifest, first),
                generated_by="unit-test",
            )
            self.assertEqual([row["file"] for row in rows], ["alpha.md", "zeta.csv"])
            self.assertNotIn("handoff_manifest.csv", {row["file"] for row in rows})

            metadata.write_csv_rows(manifest, metadata.MANIFEST_FIELDNAMES, rows)
            report = metadata.validate_manifest(
                manifest,
                root_directory=root,
                expected_files=(first, second, manifest),
            )
            self.assertTrue(report["valid"])
            self.assertEqual(report["artifact_count"], 2)

            second.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                metadata.validate_manifest(manifest, root_directory=root)

    def test_manifest_rejects_artifact_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "root"
            root.mkdir()
            outside = Path(temporary_directory) / "outside.csv"
            outside.write_text("x\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside manifest root"):
                metadata.build_manifest_rows(root, (outside,))


if __name__ == "__main__":
    unittest.main(verbosity=2)
