"""Characterization-only Source/Drain contact-topology adapter.

The parameterized runtime structure remains owned by
``05_program_erase/device_structure.py``.  This module deliberately does not
modify that shared implementation.  Instead, it wraps one call to its public
``create_structure`` function so the existing ``Air`` region is installed as
the mesh background and the active device regions are then overlaid on it.

The axial Air guard cells make the MoS2 source and drain planes true region
boundaries.  The unchanged ``add_2d_contact`` calls can consequently select
the full annular MoS2 end caps rather than two isolated corner nodes.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import math
from typing import Any, Callable, Iterator, Mapping


AIR_REGION_NAME = "Air"
MOS2_REGION_NAME = "MoS2"


def _finite_float(value: Any, description: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def axial_guard_thickness_cm(geometry: Mapping[str, Any]) -> float:
    """Use one MoS2-thickness-wide Air guard cell at each channel end."""

    inner_radius = _finite_float(geometry["r_core"], "MoS2 inner radius")
    outer_radius = _finite_float(geometry["r_mos2"], "MoS2 outer radius")
    guard = outer_radius - inner_radius
    if guard <= 0.0:
        raise ValueError("MoS2 radial thickness must be positive")
    return guard


def expanded_air_bounds(geometry: Mapping[str, Any]) -> dict[str, float]:
    """Return the radial-background and axial-guard bounds in centimetres."""

    guard = axial_guard_thickness_cm(geometry)
    r_axis = _finite_float(geometry["r_axis"], "axis radius")
    r_air_outer = _finite_float(geometry["r_air_outer"], "Air outer radius")
    z_source = _finite_float(geometry["z_source"], "source coordinate")
    z_drain = _finite_float(geometry["z_drain"], "drain coordinate")
    if r_air_outer <= r_axis:
        raise ValueError("Air outer radius must exceed the axis radius")
    if z_drain <= z_source:
        raise ValueError("drain coordinate must exceed source coordinate")
    return {
        "xl": r_axis,
        "xh": r_air_outer,
        "yl": z_source - guard,
        "yh": z_drain + guard,
        "guard_cm": guard,
    }


@contextmanager
def _temporary_contact_topology(
    structure_module: Any,
    geometry: Mapping[str, Any],
) -> Iterator[None]:
    """Temporarily adapt mesh-line and region registration for one build.

    The functions present on ``structure_module`` at entry are treated as the
    next layer in the call chain.  This is important for mesh-convergence
    runners that have already wrapped ``add_2d_mesh_line`` to scale ``ps``:
    the two guard lines pass through that wrapper as well.  Both attributes are
    restored even when mesh construction raises.
    """

    forwarded_add_mesh_line = structure_module.add_2d_mesh_line
    forwarded_add_region = structure_module.add_2d_region
    bounds = expanded_air_bounds(geometry)

    guard_lines_added = False
    regions_flushed = False
    buffered_regions: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def add_mesh_line_with_axial_guards(*args: Any, **kwargs: Any) -> Any:
        nonlocal guard_lines_added
        if not guard_lines_added:
            if args or "mesh" not in kwargs:
                raise RuntimeError(
                    "contact-topology adapter requires keyword mesh-line calls"
                )
            mesh_name = kwargs["mesh"]
            guard_spacing = bounds["guard_cm"]
            forwarded_add_mesh_line(
                mesh=mesh_name,
                dir="y",
                pos=bounds["yl"],
                ps=guard_spacing,
            )
            forwarded_add_mesh_line(
                mesh=mesh_name,
                dir="y",
                pos=bounds["yh"],
                ps=guard_spacing,
            )
            guard_lines_added = True
        return forwarded_add_mesh_line(*args, **kwargs)

    def add_region_with_air_background(*args: Any, **kwargs: Any) -> Any:
        nonlocal regions_flushed
        region_name = kwargs.get("region")

        if regions_flushed:
            if region_name == AIR_REGION_NAME:
                raise RuntimeError("Air region may only be registered once")
            return forwarded_add_region(*args, **kwargs)

        if region_name != AIR_REGION_NAME:
            buffered_regions.append((args, dict(kwargs)))
            return None

        if args:
            raise RuntimeError(
                "contact-topology adapter requires a keyword Air-region call"
            )

        active_region_names = {
            request_kwargs.get("region")
            for _, request_kwargs in buffered_regions
        }
        if MOS2_REGION_NAME not in active_region_names:
            raise RuntimeError("MoS2 region was not registered before Air")

        air_kwargs = dict(kwargs)
        air_kwargs.update(
            {
                "xl": bounds["xl"],
                "xh": bounds["xh"],
                "yl": bounds["yl"],
                "yh": bounds["yh"],
            }
        )
        air_result = forwarded_add_region(**air_kwargs)

        # DEVSIM's internal 2-D mesher gives the last registered region the
        # highest precedence.  Registering the active regions after Air keeps
        # their original geometry while retaining Air beyond each end cap.
        for request_args, request_kwargs in buffered_regions:
            forwarded_add_region(*request_args, **request_kwargs)

        buffered_regions.clear()
        regions_flushed = True
        return air_result

    structure_module.add_2d_mesh_line = add_mesh_line_with_axial_guards
    structure_module.add_2d_region = add_region_with_air_background
    try:
        yield
        if not guard_lines_added:
            raise RuntimeError("runtime structure did not register mesh lines")
        if not regions_flushed:
            raise RuntimeError("runtime structure did not register the Air region")
    finally:
        structure_module.add_2d_mesh_line = forwarded_add_mesh_line
        structure_module.add_2d_region = forwarded_add_region


def bind_characterization_create_structure(
    structure_module: Any,
) -> Callable[..., Mapping[str, Any]]:
    """Bind the topology adapter to a parameterized structure module."""

    original_create_structure = structure_module.create_structure
    calculate_geometry = structure_module.calculate_geometry

    @wraps(original_create_structure)
    def create_structure(*args: Any, **kwargs: Any) -> Mapping[str, Any]:
        expected_geometry = calculate_geometry(*args, **kwargs)
        with _temporary_contact_topology(structure_module, expected_geometry):
            geometry = original_create_structure(*args, **kwargs)
        if geometry is None:
            raise RuntimeError("runtime create_structure returned no geometry")
        return geometry

    return create_structure


__all__ = (
    "AIR_REGION_NAME",
    "MOS2_REGION_NAME",
    "axial_guard_thickness_cm",
    "expanded_air_bounds",
    "bind_characterization_create_structure",
)
