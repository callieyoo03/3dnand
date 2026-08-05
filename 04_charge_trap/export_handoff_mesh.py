"""Export and validate final mesh plus four static read-reference VTK bundles.

Only models that already exist in the solved DEVSIM device, or exact vector
components/magnitudes derived from those models, are exported.  No missing
field is replaced with a fabricated zero model.  In particular, ``Holes`` is
not exported because the present electron-only model keeps it at a fixed
equilibrium expression rather than solving a hole continuity equation.
"""

from __future__ import annotations

import argparse
import base64
import math
import shlex
import struct
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
DEFAULT_MESH_DIRECTORY = REPOSITORY_ROOT / "shared_data" / "mesh"
FINAL_MESH_FILENAME = "mos2_gaa_final_3_5_16nm.msh"

PHYSICS_REGIONS = (
    "CoreOxide",
    "MoS2",
    "TunnelOxide",
    "ChargeTrap",
    "BlockingOxide",
)

# Exact-name allowlist passed to DEVSIM's include_test callback.  Coordinates
# and mesh connectivity remain part of the VTK structure independent of this
# model allowlist.
EXPORT_MODEL_NAMES = frozenset(
    {
        "AtContactNode",
        "CylindricalNodeVolume",
        "CylindricalSurfaceArea",
        "Potential",
        "Electrons",
        "NetDoping",
        "TrappedElectronDensity",
        "TrappedChargeDensity",
        "ElectricField_x",
        "ElectricField_y",
        "ElectricFieldMagnitude",
        "ElectronCurrent_x",
        "ElectronCurrent_y",
        "ElectronCurrentMagnitude",
    }
)

REQUIRED_VTK_MODELS = frozenset(
    {
        "Potential",
        "Electrons",
        "NetDoping",
        "TrappedElectronDensity",
        "TrappedChargeDensity",
        "ElectricField_x",
        "ElectricField_y",
        "ElectricFieldMagnitude",
        "ElectronCurrent_x",
        "ElectronCurrent_y",
        "ElectronCurrentMagnitude",
    }
)

UNSUPPORTED_VTK_MODELS = frozenset({"Holes", "Permittivity"})

# DEVSIM currently emits one VTK block per region in lexicographic order.
# The block identity is independently verified from its radial coordinate
# bounds before this order is accepted or written into the VTM index.
EXPECTED_VTK_REGION_ORDER = (
    "Air",
    "BlockingOxide",
    "ChargeTrap",
    "CoreOxide",
    "GateMetal",
    "MoS2",
    "TunnelOxide",
)

EXPECTED_MESH_CONTACTS = {
    "source": ("MoS2", "Metal"),
    "drain": ("MoS2", "Metal"),
    "gate": ("BlockingOxide", "Metal"),
}

EXPECTED_MESH_INTERFACES = {
    "CoreOxide_MoS2": ("CoreOxide", "MoS2"),
    "MoS2_TunnelOxide": ("MoS2", "TunnelOxide"),
    "TunnelOxide_ChargeTrap": ("TunnelOxide", "ChargeTrap"),
    "ChargeTrap_BlockingOxide": ("ChargeTrap", "BlockingOxide"),
}


def reference_specifications() -> tuple[dict[str, Any], ...]:
    """Return the four required static-state/bias export specifications."""

    import state_characterization_config as config

    empty = config.MEMORY_STATES[0]
    programmed = config.MEMORY_STATES[-1]
    return (
        {
            "slug": "empty_off_reference",
            "base_name": "mos2_gaa_empty_off_reference",
            "state_index": int(empty["state_index"]),
            "state": str(empty["state"]),
            "ntrap_cm3": float(empty["ntrap_cm3"]),
            "VGS_V": float(config.IOFF_VGS_V),
            "VDS_V": float(config.IOFF_VDS_V),
            "source_V": 0.0,
        },
        {
            "slug": "empty_on_reference",
            "base_name": "mos2_gaa_empty_on_reference",
            "state_index": int(empty["state_index"]),
            "state": str(empty["state"]),
            "ntrap_cm3": float(empty["ntrap_cm3"]),
            "VGS_V": float(config.ION_VGS_V),
            "VDS_V": float(config.ION_VDS_V),
            "source_V": 0.0,
        },
        {
            "slug": "programmed_off_reference",
            "base_name": "mos2_gaa_programmed_off_reference",
            "state_index": int(programmed["state_index"]),
            "state": str(programmed["state"]),
            "ntrap_cm3": float(programmed["ntrap_cm3"]),
            "VGS_V": float(config.IOFF_VGS_V),
            "VDS_V": float(config.IOFF_VDS_V),
            "source_V": 0.0,
        },
        {
            "slug": "programmed_on_reference",
            "base_name": "mos2_gaa_programmed_on_reference",
            "state_index": int(programmed["state_index"]),
            "state": str(programmed["state"]),
            "ntrap_cm3": float(programmed["ntrap_cm3"]),
            "VGS_V": float(config.ION_VGS_V),
            "VDS_V": float(config.ION_VDS_V),
            "source_V": 0.0,
        },
    )


def _runtime_api(runtime_api: Any | None) -> Any:
    if runtime_api is None:
        import devsim

        return devsim
    return runtime_api


def _ensure_element_components(
    runtime_api: Any,
    *,
    device: str,
    region: str,
    edge_model: str,
) -> tuple[str, str]:
    edge_models = set(runtime_api.get_edge_model_list(device=device, region=region))
    if edge_model not in edge_models:
        raise RuntimeError(
            f"Cannot construct visualization components: edge model "
            f"{edge_model!r} is missing in region {region!r}."
        )

    component_names = (f"{edge_model}_x", f"{edge_model}_y")
    element_models = set(
        runtime_api.get_element_model_list(device=device, region=region)
    )
    if not set(component_names).issubset(element_models):
        runtime_api.element_from_edge_model(
            device=device,
            region=region,
            edge_model=edge_model,
        )
        element_models = set(
            runtime_api.get_element_model_list(device=device, region=region)
        )
    if not set(component_names).issubset(element_models):
        raise RuntimeError(
            f"DEVSIM did not create {edge_model} vector components in {region}."
        )
    return component_names


def _ensure_vector_magnitude(
    runtime_api: Any,
    *,
    device: str,
    region: str,
    component_names: tuple[str, str],
    magnitude_name: str,
) -> None:
    element_models = set(
        runtime_api.get_element_model_list(device=device, region=region)
    )
    if magnitude_name in element_models:
        return

    x_name, y_name = component_names
    runtime_api.element_model(
        device=device,
        region=region,
        name=magnitude_name,
        # DEVSIM/SYMDIFF 2.10 exposes the power operator but does not register
        # ``sqrt`` as a built-in function.  This is exactly the Euclidean
        # magnitude and introduces no scaling or fabricated field.
        equation=f"((({x_name})^2 + ({y_name})^2))^0.5",
        display_type="scalar",
    )
    element_models = set(
        runtime_api.get_element_model_list(device=device, region=region)
    )
    if magnitude_name not in element_models:
        raise RuntimeError(
            f"DEVSIM did not create derived magnitude {magnitude_name!r} "
            f"in region {region!r}."
        )


def prepare_visualization_models(
    *,
    device: str = "MoS2_GAA",
    runtime_api: Any | None = None,
) -> dict[str, tuple[str, ...]]:
    """Create only exact field components/magnitudes needed for visualization."""

    api = _runtime_api(runtime_api)
    available: dict[str, tuple[str, ...]] = {}
    for region in PHYSICS_REGIONS:
        node_models = set(api.get_node_model_list(device=device, region=region))
        if "Potential" not in node_models:
            raise RuntimeError(f"Potential is missing in region {region!r}.")
        electric_components = _ensure_element_components(
            api,
            device=device,
            region=region,
            edge_model="ElectricField",
        )
        _ensure_vector_magnitude(
            api,
            device=device,
            region=region,
            component_names=electric_components,
            magnitude_name="ElectricFieldMagnitude",
        )

        if region == "MoS2":
            required_nodes = {"Electrons", "NetDoping"}
            if not required_nodes.issubset(node_models):
                raise RuntimeError(
                    "MoS2 visualization requires existing Electrons and NetDoping models."
                )
            current_components = _ensure_element_components(
                api,
                device=device,
                region=region,
                edge_model="ElectronCurrent",
            )
            _ensure_vector_magnitude(
                api,
                device=device,
                region=region,
                component_names=current_components,
                magnitude_name="ElectronCurrentMagnitude",
            )

        if region == "ChargeTrap":
            required_trap_models = {
                "TrappedElectronDensity",
                "TrappedChargeDensity",
            }
            if not required_trap_models.issubset(node_models):
                raise RuntimeError(
                    "ChargeTrap visualization requires the existing trapped-charge models."
                )

        available[region] = tuple(
            sorted(
                (set(api.get_node_model_list(device=device, region=region))
                 | set(api.get_element_model_list(device=device, region=region)))
                & EXPORT_MODEL_NAMES
            )
        )
    return available


def assert_reference_bias_and_trap(
    reference: Mapping[str, Any],
    *,
    device: str = "MoS2_GAA",
    runtime_api: Any | None = None,
    trap_parameter_name: str | None = None,
) -> dict[str, float]:
    """Assert actual DEVSIM parameters before a reference bundle is written."""

    api = _runtime_api(runtime_api)
    if trap_parameter_name is None:
        from trap_models import TRAPPED_ELECTRON_PARAMETER

        trap_parameter_name = TRAPPED_ELECTRON_PARAMETER

    expected = {
        "gate_bias": float(reference["VGS_V"]),
        "drain_bias": float(reference["VDS_V"]),
        "source_bias": float(reference.get("source_V", 0.0)),
    }
    actual: dict[str, float] = {}
    for parameter_name, expected_value in expected.items():
        value = float(api.get_parameter(device=device, name=parameter_name))
        if not math.isfinite(value) or not math.isclose(
            value,
            expected_value,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise RuntimeError(
                f"Reference bias mismatch for {parameter_name}: "
                f"actual={value:.12e}, expected={expected_value:.12e}."
            )
        actual[parameter_name] = value

    actual_trap = float(
        api.get_parameter(
            device=device,
            region="ChargeTrap",
            name=trap_parameter_name,
        )
    )
    expected_trap = float(reference["ntrap_cm3"])
    trap_tolerance = max(1.0, abs(expected_trap) * 1.0e-12)
    if not math.isfinite(actual_trap) or not math.isclose(
        actual_trap,
        expected_trap,
        rel_tol=1.0e-12,
        abs_tol=trap_tolerance,
    ):
        raise RuntimeError(
            "Reference trap-density mismatch: "
            f"actual={actual_trap:.12e}, expected={expected_trap:.12e}."
        )
    actual["ntrap_cm3"] = actual_trap
    return actual


def _apply_reference(
    reference: Mapping[str, Any],
    operating_point: Any,
) -> None:
    """Reach one reference through the established continuation helpers."""

    from state_sweep_helpers import ramp_terminal, ramp_trap_density

    # Trap changes are performed at the established VGS=0 V anchor.  Drain is
    # held at the read bias, matching state-characterization preparation.
    ramp_terminal(
        operating_point,
        "gate",
        0.0,
        label=f"{reference['slug']} gate preparation",
    )
    ramp_terminal(
        operating_point,
        "drain",
        float(reference["VDS_V"]),
        label=f"{reference['slug']} drain preparation",
    )
    ramp_trap_density(
        operating_point,
        float(reference["ntrap_cm3"]),
        label=f"{reference['slug']} trap preparation",
    )
    ramp_terminal(
        operating_point,
        "gate",
        float(reference["VGS_V"]),
        label=f"{reference['slug']} gate bias",
    )


def _include_export_model(name: str) -> bool:
    return name in EXPORT_MODEL_NAMES


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


_VTK_SCALAR_TYPES = {
    "Float32": "f",
    "Float64": "d",
    "Int8": "b",
    "UInt8": "B",
    "Int16": "h",
    "UInt16": "H",
    "Int32": "i",
    "UInt32": "I",
    "Int64": "q",
    "UInt64": "Q",
}


def _decode_inline_binary_data_array(
    array: ET.Element,
    vtk_root: ET.Element,
    path: Path,
) -> tuple[float, ...]:
    """Decode VTK XML inline binary data, including zlib block headers."""

    encoded = "".join((array.text or "").split())
    if not encoded:
        raise ValueError(f"Empty binary DataArray in {path}.")

    byte_order = vtk_root.attrib.get("byte_order", "LittleEndian")
    if byte_order == "LittleEndian":
        endian = "<"
    elif byte_order == "BigEndian":
        endian = ">"
    else:
        raise ValueError(f"Unsupported VTK byte order {byte_order!r} in {path}.")

    header_type = vtk_root.attrib.get("header_type", "UInt32")
    header_formats = {"UInt32": "I", "UInt64": "Q"}
    if header_type not in header_formats:
        raise ValueError(
            f"Unsupported VTK header type {header_type!r} in {path}."
        )
    header_format = header_formats[header_type]
    header_word_size = struct.calcsize(header_format)
    first_word_char_count = 4 * math.ceil(header_word_size / 3)
    try:
        first_word = base64.b64decode(encoded[:first_word_char_count])
        number_of_blocks = struct.unpack(
            endian + header_format,
            first_word[:header_word_size],
        )[0]
    except (ValueError, struct.error) as error:
        raise ValueError(f"Malformed VTK binary header in {path}.") from error

    compressor = vtk_root.attrib.get("compressor")
    if compressor != "vtkZLibDataCompressor":
        raise ValueError(
            "Only DEVSIM's inline vtkZLibDataCompressor output is supported; "
            f"found {compressor!r} in {path}."
        )
    if number_of_blocks < 1:
        raise ValueError(f"Invalid VTK binary block count in {path}.")

    header_word_count = 3 + number_of_blocks
    header_byte_count = header_word_count * header_word_size
    header_char_count = 4 * math.ceil(header_byte_count / 3)
    try:
        header_bytes = base64.b64decode(encoded[:header_char_count])
        header_values = struct.unpack(
            endian + header_format * header_word_count,
            header_bytes,
        )
        payload = base64.b64decode(encoded[header_char_count:])
    except (ValueError, struct.error) as error:
        raise ValueError(f"Malformed compressed VTK DataArray in {path}.") from error

    _, nominal_block_size, final_block_size, *compressed_sizes = header_values
    if len(payload) != sum(compressed_sizes):
        raise ValueError(f"Compressed VTK payload length mismatch in {path}.")

    raw_parts: list[bytes] = []
    offset = 0
    for block_index, compressed_size in enumerate(compressed_sizes):
        block = payload[offset : offset + compressed_size]
        offset += compressed_size
        try:
            raw = zlib.decompress(block)
        except zlib.error as error:
            raise ValueError(f"Invalid zlib VTK block in {path}.") from error
        expected_size = (
            final_block_size
            if block_index == number_of_blocks - 1
            else nominal_block_size
        )
        if len(raw) != expected_size:
            raise ValueError(
                f"Uncompressed VTK block length mismatch in {path}: "
                f"{len(raw)} != {expected_size}."
            )
        raw_parts.append(raw)
    raw_data = b"".join(raw_parts)

    scalar_type = array.attrib.get("type")
    scalar_format = _VTK_SCALAR_TYPES.get(str(scalar_type))
    if scalar_format is None:
        raise ValueError(f"Unsupported VTK scalar type {scalar_type!r} in {path}.")
    scalar_size = struct.calcsize(scalar_format)
    if len(raw_data) % scalar_size:
        raise ValueError(f"VTK scalar payload is misaligned in {path}.")
    value_count = len(raw_data) // scalar_size
    values = struct.unpack(endian + scalar_format * value_count, raw_data)
    return tuple(float(value) for value in values)


def _read_data_array_values(
    array: ET.Element,
    vtk_root: ET.Element,
    path: Path,
) -> tuple[float, ...]:
    """Read finite values from an ASCII or DEVSIM inline-binary DataArray."""

    data_format = array.attrib.get("format", "ascii").lower()
    if data_format == "ascii":
        values = tuple(float(token) for token in (array.text or "").split())
    elif data_format == "binary":
        values = _decode_inline_binary_data_array(array, vtk_root, path)
    else:
        raise ValueError(
            f"Unsupported VTK DataArray format {data_format!r} in {path}."
        )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"Non-finite VTK DataArray value in {path}.")
    return values


def _parse_nonnegative_integer(value: str | None, field: str, path: Path) -> int:
    try:
        number = int(value or "")
    except ValueError as error:
        raise ValueError(f"Invalid {field} in {path}: {value!r}") from error
    if number < 0:
        raise ValueError(f"Negative {field} in {path}: {number}")
    return number


def _expected_region_radial_bounds_cm() -> dict[str, tuple[float, float]]:
    from handoff_metadata import expected_geometry

    geometry = expected_geometry()
    return {
        "CoreOxide": (geometry["r_axis"], geometry["r_core"]),
        "MoS2": (geometry["r_core"], geometry["r_mos2"]),
        "TunnelOxide": (geometry["r_mos2"], geometry["r_tox"]),
        "ChargeTrap": (geometry["r_tox"], geometry["r_trap"]),
        "BlockingOxide": (geometry["r_trap"], geometry["r_block"]),
        "GateMetal": (geometry["r_block"], geometry["r_gate_outer"]),
        "Air": (geometry["r_gate_outer"], geometry["r_air_outer"]),
    }


def _read_point_coordinates_from_vtu(
    path: str | Path,
) -> tuple[tuple[float, float, float], ...]:
    """Decode every point coordinate in one ASCII or binary VTU."""

    vtu = Path(path).resolve()
    root = ET.parse(vtu).getroot()
    coordinates: list[tuple[float, float, float]] = []
    for points in root.iter():
        if _local_xml_name(points.tag) != "Points":
            continue
        arrays = [
            child for child in points if _local_xml_name(child.tag) == "DataArray"
        ]
        if len(arrays) != 1:
            raise ValueError(f"Expected one Points DataArray in {vtu}.")
        array = arrays[0]
        components = int(array.attrib.get("NumberOfComponents", "3"))
        if components not in {2, 3}:
            raise ValueError(
                f"Unsupported point component count {components} in {vtu}."
            )
        values = list(_read_data_array_values(array, root, vtu))
        if len(values) % components:
            raise ValueError(f"Malformed point coordinate array in {vtu}.")
        for offset in range(0, len(values), components):
            point = values[offset : offset + components]
            if components == 2:
                point.append(0.0)
            coordinates.append((point[0], point[1], point[2]))
    if not coordinates:
        raise ValueError(f"No point coordinates were found in {vtu}.")
    return tuple(coordinates)


def _identify_vtu_region(path: str | Path) -> tuple[str, tuple[float, float]]:
    """Identify a VTU block from its independently decoded radial bounds."""

    vtu = Path(path).resolve()
    coordinates = _read_point_coordinates_from_vtu(vtu)
    actual_bounds = (
        min(point[0] for point in coordinates),
        max(point[0] for point in coordinates),
    )
    matches = [
        region
        for region, expected_bounds in _expected_region_radial_bounds_cm().items()
        if all(
            math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for actual, expected in zip(actual_bounds, expected_bounds)
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            f"VTU radial bounds do not uniquely identify a region in {vtu}: "
            f"bounds={actual_bounds}, matches={matches}."
        )
    return matches[0], actual_bounds


def _read_vtm_dataset_entries(vtm_path: str | Path) -> tuple[tuple[str, Path], ...]:
    """Parse named, portable VTM entries and validate block identity."""

    vtm = Path(vtm_path).resolve()
    tree = ET.parse(vtm)
    root = tree.getroot()
    if root.attrib.get("type") != "vtkMultiBlockDataSet":
        raise ValueError(f"Not a VTK multiblock file: {vtm}")

    referenced: list[tuple[str, Path]] = []
    for element in root.iter():
        if _local_xml_name(element.tag) != "DataSet":
            continue
        filename = element.attrib.get("file")
        if not filename:
            raise ValueError(f"VTM DataSet entry has no file attribute: {vtm}")
        reference = Path(filename)
        if reference.is_absolute() or reference.name != filename:
            raise ValueError(
                "VTM references must be portable relative basenames: "
                f"{filename!r}"
            )
        candidate = (vtm.parent / filename).resolve()
        if candidate.suffix.lower() != ".vtu":
            raise ValueError(f"VTM references a non-VTU file: {filename}")
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(f"Referenced VTU is missing or empty: {candidate}")
        name = element.attrib.get("name")
        actual_region, _ = _identify_vtu_region(candidate)
        if name != actual_region:
            raise ValueError(
                f"VTM region name mismatch for {filename}: "
                f"name={name!r}, coordinates identify {actual_region!r}."
            )
        referenced.append((actual_region, candidate))

    if not referenced:
        raise ValueError(f"VTM contains no VTU datasets: {vtm}")
    paths = [path for _, path in referenced]
    if len(paths) != len(set(paths)):
        raise ValueError(f"VTM contains duplicate VTU references: {vtm}")
    names = tuple(name for name, _ in referenced)
    if names != EXPECTED_VTK_REGION_ORDER:
        raise ValueError(
            f"Unexpected VTM block ordering/names: {names}; "
            f"expected {EXPECTED_VTK_REGION_ORDER}."
        )
    return tuple(referenced)


def read_vtm_dataset_paths(vtm_path: str | Path) -> tuple[Path, ...]:
    """Parse safe relative VTU references from a DEVSIM VTM file."""

    return tuple(path for _, path in _read_vtm_dataset_entries(vtm_path))


def read_visit_dataset_paths(visit_path: str | Path) -> tuple[Path, ...]:
    """Parse a portable ``.visit`` block list and validate every VTU."""

    visit = Path(visit_path).resolve()
    lines = [line.strip() for line in visit.read_text(encoding="utf-8").splitlines()]
    if not lines or not lines[0].startswith("!NBLOCKS "):
        raise ValueError(f"Invalid VISIT block-list header: {visit}")
    try:
        declared_count = int(lines[0].split(maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Invalid VISIT block count: {visit}") from error
    filenames = [line for line in lines[1:] if line]
    if declared_count != len(filenames):
        raise ValueError(
            f"VISIT block count mismatch in {visit}: "
            f"declared={declared_count}, actual={len(filenames)}."
        )

    referenced: list[Path] = []
    for filename in filenames:
        reference = Path(filename)
        if reference.is_absolute() or reference.name != filename:
            raise ValueError(
                "VISIT references must be portable relative basenames: "
                f"{filename!r}"
            )
        candidate = (visit.parent / filename).resolve()
        if candidate.suffix.lower() != ".vtu":
            raise ValueError(f"VISIT references a non-VTU file: {filename}")
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise FileNotFoundError(f"Referenced VTU is missing or empty: {candidate}")
        referenced.append(candidate)
    if len(referenced) != len(set(referenced)):
        raise ValueError(f"VISIT contains duplicate VTU references: {visit}")
    return tuple(referenced)


def normalize_vtk_bundle_references(base_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Replace DEVSIM's absolute VTM/VISIT paths with portable basenames.

    DEVSIM writes the two bundle indexes using host-absolute paths.  Every VTU
    is required to reside next to its index, so retaining only the basename is
    lossless and makes the handoff relocatable.
    """

    base = Path(base_path).resolve()
    vtm = base.with_suffix(".vtm")
    visit = base.with_suffix(".visit")
    tree = ET.parse(vtm)
    root = tree.getroot()
    vtm_names: list[str] = []
    vtm_regions: list[str] = []
    for element in root.iter():
        if _local_xml_name(element.tag) != "DataSet":
            continue
        filename = element.attrib.get("file")
        if not filename:
            raise ValueError(f"VTM DataSet entry has no file attribute: {vtm}")
        source = Path(filename)
        if not source.is_absolute():
            source = vtm.parent / source
        source = source.resolve()
        if source.parent != vtm.parent or source.suffix.lower() != ".vtu":
            raise ValueError(
                f"Cannot normalize out-of-bundle VTM reference: {filename!r}"
            )
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Referenced VTU is missing or empty: {source}")
        _remove_generated_trailing_blank_lines(source)
        region, _ = _identify_vtu_region(source)
        element.set("file", source.name)
        element.set("name", region)
        vtm_names.append(source.name)
        vtm_regions.append(region)
    if not vtm_names:
        raise ValueError(f"VTM contains no VTU datasets: {vtm}")
    if tuple(vtm_regions) != EXPECTED_VTK_REGION_ORDER:
        raise ValueError(
            f"Unexpected DEVSIM VTK block order {tuple(vtm_regions)}; "
            f"expected {EXPECTED_VTK_REGION_ORDER}."
        )
    ET.indent(tree, space="  ")
    tree.write(vtm, encoding="utf-8", xml_declaration=True)

    visit_lines = [
        line.strip() for line in visit.read_text(encoding="utf-8").splitlines()
    ]
    if not visit_lines or not visit_lines[0].startswith("!NBLOCKS "):
        raise ValueError(f"Invalid VISIT block-list header: {visit}")
    normalized_visit_names: list[str] = []
    for filename in (line for line in visit_lines[1:] if line):
        source = Path(filename)
        if not source.is_absolute():
            source = visit.parent / source
        source = source.resolve()
        if source.parent != visit.parent or source.suffix.lower() != ".vtu":
            raise ValueError(
                f"Cannot normalize out-of-bundle VISIT reference: {filename!r}"
            )
        if not source.is_file() or source.stat().st_size == 0:
            raise FileNotFoundError(f"Referenced VTU is missing or empty: {source}")
        normalized_visit_names.append(source.name)
    if tuple(normalized_visit_names) != tuple(vtm_names):
        raise ValueError("VTM and VISIT references differ before normalization.")
    visit.write_text(
        f"!NBLOCKS {len(normalized_visit_names)}\n"
        + "\n".join(normalized_visit_names)
        + "\n",
        encoding="utf-8",
    )
    return {
        "vtm_files": tuple(vtm_names),
        "visit_files": tuple(normalized_visit_names),
        "regions": tuple(vtm_regions),
    }


def inspect_vtu(path: str | Path) -> dict[str, Any]:
    """Validate basic VTU structure and report point/cell data-array names."""

    vtu = Path(path).resolve()
    tree = ET.parse(vtu)
    root = tree.getroot()
    if root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError(f"Not a VTK unstructured-grid file: {vtu}")

    pieces = [element for element in root.iter() if _local_xml_name(element.tag) == "Piece"]
    if not pieces:
        raise ValueError(f"VTU contains no Piece: {vtu}")
    point_count = sum(
        _parse_nonnegative_integer(piece.attrib.get("NumberOfPoints"), "NumberOfPoints", vtu)
        for piece in pieces
    )
    cell_count = sum(
        _parse_nonnegative_integer(piece.attrib.get("NumberOfCells"), "NumberOfCells", vtu)
        for piece in pieces
    )
    if point_count <= 0 or cell_count <= 0:
        raise ValueError(
            f"VTU must contain nonzero points and cells: "
            f"points={point_count}, cells={cell_count}, file={vtu}"
        )

    model_names: set[str] = set()
    for container in root.iter():
        if _local_xml_name(container.tag) not in {"PointData", "CellData"}:
            continue
        for array in container:
            if _local_xml_name(array.tag) != "DataArray":
                continue
            name = array.attrib.get("Name")
            if name:
                # Decode every exported physical model, not just coordinates
                # and trapped charge, so corrupt/non-finite binary payloads
                # cannot pass merely because their XML headers are intact.
                _read_data_array_values(array, root, vtu)
                model_names.add(name)
    coordinates = _read_point_coordinates_from_vtu(vtu)
    radial_values = [point[0] for point in coordinates]
    axial_values = [point[1] for point in coordinates]
    return {
        "file": str(vtu),
        "point_count": point_count,
        "cell_count": cell_count,
        "model_names": tuple(sorted(model_names)),
        "radial_bounds_cm": (min(radial_values), max(radial_values)),
        "axial_bounds_cm": (min(axial_values), max(axial_values)),
    }


def validate_vtk_bundle(
    base_path: str | Path,
    *,
    required_models: Sequence[str] = tuple(REQUIRED_VTK_MODELS),
    forbidden_models: Sequence[str] = tuple(UNSUPPORTED_VTK_MODELS),
) -> dict[str, Any]:
    """Validate one VTM/VISIT/VTU bundle and its exported model allowlist."""

    base = Path(base_path)
    vtm = base.with_suffix(".vtm")
    visit = base.with_suffix(".visit")
    if not vtm.is_file() or vtm.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty VTM file: {vtm}")
    if not visit.is_file() or visit.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty VISIT file: {visit}")

    vtu_paths = read_vtm_dataset_paths(vtm)
    visit_vtu_paths = read_visit_dataset_paths(visit)
    if visit_vtu_paths != vtu_paths:
        raise ValueError("VTM and VISIT reference different VTU files or ordering.")
    entries = _read_vtm_dataset_entries(vtm)
    reports = [inspect_vtu(path) for _, path in entries]
    exported_models = {
        name for report in reports for name in report["model_names"]
    }
    missing = set(required_models).difference(exported_models)
    forbidden = set(forbidden_models).intersection(exported_models)
    if missing:
        raise ValueError(f"VTK bundle is missing required models: {sorted(missing)}")
    if forbidden:
        raise ValueError(
            "VTK bundle unexpectedly exports unsupported models: "
            f"{sorted(forbidden)}"
        )

    expected_bounds = _expected_region_radial_bounds_cm()
    report_by_region = {
        region: report for (region, _), report in zip(entries, reports)
    }
    for region, target_bounds in expected_bounds.items():
        report = report_by_region[region]
        actual_bounds = report["radial_bounds_cm"]
        if not all(
            math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)
            for actual, expected in zip(actual_bounds, target_bounds)
        ):
            raise ValueError(
                f"VTU radial bounds mismatch for {region}: "
                f"actual={actual_bounds}, expected={target_bounds}."
            )
    regional_model_requirements = {
        "MoS2": {"Electrons", "NetDoping", "ElectronCurrentMagnitude"},
        "ChargeTrap": {"TrappedElectronDensity", "TrappedChargeDensity"},
    }
    for region, required in regional_model_requirements.items():
        present = set(report_by_region[region]["model_names"])
        if not required.issubset(present):
            raise ValueError(
                f"VTU block {region} lacks region-specific models: "
                f"{sorted(required.difference(present))}."
            )

    return {
        "base_path": str(base.resolve()),
        "vtm": str(vtm.resolve()),
        "visit": str(visit.resolve()),
        "vtu_files": tuple(str(path) for path in vtu_paths),
        "region_file_count": len(vtu_paths),
        "point_count": sum(report["point_count"] for report in reports),
        "cell_count": sum(report["cell_count"] for report in reports),
        "model_names": tuple(sorted(exported_models)),
        "block_regions": tuple(region for region, _ in entries),
        "radial_bounds_cm_by_region": {
            region: report_by_region[region]["radial_bounds_cm"]
            for region in EXPECTED_VTK_REGION_ORDER
        },
    }


def read_model_values_from_bundle(
    base_path: str | Path,
    model_name: str,
) -> tuple[float, ...]:
    """Read all finite values for one named PointData/CellData model."""

    vtm = Path(base_path).with_suffix(".vtm")
    values: list[float] = []
    for vtu in read_vtm_dataset_paths(vtm):
        root = ET.parse(vtu).getroot()
        for container in root.iter():
            if _local_xml_name(container.tag) not in {"PointData", "CellData"}:
                continue
            for array in container:
                if (
                    _local_xml_name(array.tag) != "DataArray"
                    or array.attrib.get("Name") != model_name
                ):
                    continue
                values.extend(_read_data_array_values(array, root, vtu))
    if not values:
        raise ValueError(f"Model {model_name!r} was not found in bundle {base_path}.")
    return tuple(values)


def read_point_coordinates_from_bundle(
    base_path: str | Path,
) -> tuple[tuple[float, float, float], ...]:
    """Read finite point coordinates from every VTU in one bundle."""

    vtm = Path(base_path).with_suffix(".vtm")
    coordinates: list[tuple[float, float, float]] = []
    for vtu in read_vtm_dataset_paths(vtm):
        coordinates.extend(_read_point_coordinates_from_vtu(vtu))
    if not coordinates:
        raise ValueError(f"No point coordinates were found in bundle {base_path}.")
    return tuple(coordinates)


def _validate_mesh_file(mesh_path: Path) -> dict[str, Any]:
    """Validate exact region/contact/interface declarations in a DEVSIM MSH."""

    if not mesh_path.is_file() or mesh_path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty final DEVSIM mesh: {mesh_path}")
    from handoff_metadata import EXPECTED_REGIONS

    regions: dict[str, str] = {}
    contacts: dict[str, tuple[str, str]] = {}
    interfaces: dict[str, tuple[str, str]] = {}
    for line in mesh_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("begin_region "):
            fields = shlex.split(stripped)
            if len(fields) != 3:
                raise ValueError(f"Malformed begin_region in {mesh_path}: {line}")
            if fields[1] in regions:
                raise ValueError(
                    f"Duplicate begin_region {fields[1]!r} in {mesh_path}."
                )
            regions[fields[1]] = fields[2]
        elif stripped.startswith("begin_contact "):
            fields = shlex.split(stripped)
            if len(fields) != 4:
                raise ValueError(f"Malformed begin_contact in {mesh_path}: {line}")
            if fields[1] in contacts:
                raise ValueError(
                    f"Duplicate begin_contact {fields[1]!r} in {mesh_path}."
                )
            contacts[fields[1]] = (fields[2], fields[3])
        elif stripped.startswith("begin_interface "):
            fields = shlex.split(stripped)
            if len(fields) != 4:
                raise ValueError(f"Malformed begin_interface in {mesh_path}: {line}")
            if fields[1] in interfaces:
                raise ValueError(
                    f"Duplicate begin_interface {fields[1]!r} in {mesh_path}."
                )
            interfaces[fields[1]] = (fields[2], fields[3])

    if regions != EXPECTED_REGIONS:
        raise ValueError(
            f"Final MSH region declarations differ: {regions}; "
            f"expected {EXPECTED_REGIONS}."
        )
    if contacts != EXPECTED_MESH_CONTACTS:
        raise ValueError(
            f"Final MSH contact declarations differ: {contacts}; "
            f"expected {EXPECTED_MESH_CONTACTS}."
        )
    if interfaces != EXPECTED_MESH_INTERFACES:
        raise ValueError(
            f"Final MSH interface declarations differ: {interfaces}; "
            f"expected {EXPECTED_MESH_INTERFACES}."
        )
    return {
        "region_count": len(regions),
        "contact_count": len(contacts),
        "interface_count": len(interfaces),
        "regions": tuple(sorted(regions)),
        "contacts": tuple(sorted(contacts)),
        "interfaces": tuple(sorted(interfaces)),
    }


def _remove_generated_trailing_blank_lines(path: str | Path) -> bool:
    """Remove only redundant final newlines emitted by DEVSIM writers.

    DEVSIM terminates MSH and VTU output with two LF bytes.  Keeping one LF
    preserves the conventional text-file terminator while avoiding a blank
    line at EOF.  No XML, mesh, model, or numeric payload byte is otherwise
    changed.
    """

    output = Path(path)
    data = output.read_bytes()
    normalized = data
    while normalized.endswith(b"\r\n\r\n"):
        normalized = normalized[:-2]
    while normalized.endswith(b"\n\n"):
        normalized = normalized[:-1]
    if normalized == data:
        return False
    output.write_bytes(normalized)
    return True


def validate_reference_exports(
    output_directory: str | Path = DEFAULT_MESH_DIRECTORY,
) -> dict[str, Any]:
    """Validate the final MSH and all four reference VTK bundles."""

    directory = Path(output_directory)
    mesh_path = directory / FINAL_MESH_FILENAME
    mesh_report = _validate_mesh_file(mesh_path)
    bundle_reports: dict[str, dict[str, Any]] = {}
    for reference in reference_specifications():
        base = directory / reference["slug"] / reference["base_name"]
        bundle_report = validate_vtk_bundle(base)
        if bundle_report["region_file_count"] != 7:
            raise ValueError(
                f"Reference {reference['slug']} must contain exactly seven "
                f"region VTUs; found {bundle_report['region_file_count']}."
            )
        coordinates = read_point_coordinates_from_bundle(base)
        from handoff_metadata import expected_geometry

        expected = expected_geometry()
        radial_values = [point[0] for point in coordinates]
        axial_values = [point[1] for point in coordinates]
        expected_r_max_cm = expected["air_outer_radius_nm"] * 1.0e-7
        expected_z_max_cm = expected["channel_length_nm"] * 1.0e-7
        extrema = (
            ("minimum radius", min(radial_values), 0.0),
            ("maximum radius", max(radial_values), expected_r_max_cm),
            ("minimum axial coordinate", min(axial_values), 0.0),
            ("maximum axial coordinate", max(axial_values), expected_z_max_cm),
        )
        for name, actual, target in extrema:
            if not math.isclose(actual, target, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ValueError(
                    f"VTK {name} mismatch for {reference['slug']}: "
                    f"actual={actual:.12e} cm, expected={target:.12e} cm."
                )
        bundle_report["coordinate_extrema_cm"] = {
            "r_min": min(radial_values),
            "r_max": max(radial_values),
            "z_min": min(axial_values),
            "z_max": max(axial_values),
        }
        bundle_reports[str(reference["slug"])] = bundle_report

    empty_off = directory / "empty_off_reference" / "mos2_gaa_empty_off_reference"
    empty_on = directory / "empty_on_reference" / "mos2_gaa_empty_on_reference"
    programmed_off = (
        directory
        / "programmed_off_reference"
        / "mos2_gaa_programmed_off_reference"
    )
    programmed_on = (
        directory
        / "programmed_on_reference"
        / "mos2_gaa_programmed_on_reference"
    )

    for base in (empty_off, empty_on):
        density_values = read_model_values_from_bundle(base, "TrappedElectronDensity")
        if any(abs(value) > 1.0 for value in density_values):
            raise ValueError(f"Empty-state VTK has nonzero trapped density: {base}")
        charge_values = read_model_values_from_bundle(base, "TrappedChargeDensity")
        if any(abs(value) > 1.0e-30 for value in charge_values):
            raise ValueError(f"Empty-state VTK has nonzero trapped charge: {base}")
    for base in (programmed_off, programmed_on):
        density_values = read_model_values_from_bundle(base, "TrappedElectronDensity")
        expected_density = 2.0e18
        if not density_values or any(
            not math.isclose(
                value,
                expected_density,
                rel_tol=1.0e-12,
                abs_tol=1.0,
            )
            for value in density_values
        ):
            raise ValueError(
                f"Programmed-state VTK trapped density does not equal "
                f"2.0e18 cm^-3: {base}"
            )
        charge_values = read_model_values_from_bundle(base, "TrappedChargeDensity")
        expected_charge_density = -1.602176634e-19 * expected_density
        if any(
            not math.isclose(
                value,
                expected_charge_density,
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            )
            for value in charge_values
        ):
            raise ValueError(
                f"Programmed-state VTK trapped charge does not match -q*Ntrap: {base}"
            )

    return {
        "valid": True,
        "mesh": str(mesh_path.resolve()),
        "mesh_topology": mesh_report,
        "references": bundle_reports,
    }


def export_handoff_mesh(
    output_directory: str | Path = DEFAULT_MESH_DIRECTORY,
) -> dict[str, Any]:
    """Build one device and export final MSH plus four solved VTK references."""

    import devsim
    import run_memory_window as memory_window
    from handoff_metadata import validate_runtime_metadata, write_metadata_csvs
    from state_sweep_helpers import initialize_characterization_device

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)

    geometry, operating_point = initialize_characterization_device()
    runtime_report = validate_runtime_metadata(
        geometry,
        device=memory_window.device,
        runtime_api=devsim,
    )
    metadata_paths = write_metadata_csvs(
        REPOSITORY_ROOT / "shared_data",
        geometry=geometry,
        runtime_verified=True,
    )
    model_report = prepare_visualization_models(
        device=memory_window.device,
        runtime_api=devsim,
    )

    # The restart/mesh file is captured at the initialized empty, zero-bias
    # solution.  It contains the exact final mesh and model definitions.
    zero_reference = {
        "VGS_V": 0.0,
        "VDS_V": 0.0,
        "source_V": 0.0,
        "ntrap_cm3": 0.0,
    }
    assert_reference_bias_and_trap(
        zero_reference,
        device=memory_window.device,
        runtime_api=devsim,
    )
    mesh_path = directory / FINAL_MESH_FILENAME
    devsim.write_devices(
        file=str(mesh_path),
        device=memory_window.device,
        type="devsim",
    )
    _remove_generated_trailing_blank_lines(mesh_path)
    _validate_mesh_file(mesh_path)

    exported_references: dict[str, dict[str, Any]] = {}
    for reference in reference_specifications():
        _apply_reference(reference, operating_point)
        actual = assert_reference_bias_and_trap(
            reference,
            device=memory_window.device,
            runtime_api=devsim,
        )
        reference_directory = directory / str(reference["slug"])
        reference_directory.mkdir(parents=True, exist_ok=True)
        base_path = reference_directory / str(reference["base_name"])
        devsim.write_devices(
            file=str(base_path),
            device=memory_window.device,
            type="vtk",
            include_test=_include_export_model,
        )
        normalize_vtk_bundle_references(base_path)
        bundle = validate_vtk_bundle(base_path)
        exported_references[str(reference["slug"])] = {
            "specification": dict(reference),
            "actual_parameters": actual,
            "bundle": bundle,
        }

    validation = validate_reference_exports(directory)
    return {
        "geometry": dict(geometry),
        "runtime_metadata": runtime_report,
        "metadata_paths": {
            name: str(path.resolve()) for name, path in metadata_paths.items()
        },
        "visualization_models": model_report,
        "mesh": str(mesh_path.resolve()),
        "references": exported_references,
        "validation": validation,
    }


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export or validate the final 3/5/16 nm DEVSIM mesh and "
            "empty/programmed off/on VTK references."
        )
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_MESH_DIRECTORY,
        help="Mesh/VTK output directory (default: shared_data/mesh).",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing MSH/VTM/VTU files without running DEVSIM.",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    if arguments.validate_only:
        report = validate_reference_exports(arguments.output_directory)
    else:
        report = export_handoff_mesh(arguments.output_directory)
    print(f"Final mesh: {report['mesh']}")
    print("Mesh/VTK validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MESH_DIRECTORY",
    "EXPORT_MODEL_NAMES",
    "FINAL_MESH_FILENAME",
    "REQUIRED_VTK_MODELS",
    "UNSUPPORTED_VTK_MODELS",
    "assert_reference_bias_and_trap",
    "export_handoff_mesh",
    "inspect_vtu",
    "normalize_vtk_bundle_references",
    "prepare_visualization_models",
    "read_model_values_from_bundle",
    "read_point_coordinates_from_bundle",
    "read_visit_dataset_paths",
    "read_vtm_dataset_paths",
    "reference_specifications",
    "validate_reference_exports",
    "validate_vtk_bundle",
]
