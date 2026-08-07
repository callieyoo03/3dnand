"""Fixed-position local mesh ladder for contact-flux convergence studies.

The production geometry remains owned by ``05_program_erase/device_structure``.
This module is a characterization-only adapter: it intercepts that module's
runtime mesh-line calls, keeps every geometry boundary fixed, and adds a
common set of radial and axial cross-lines.  The built-in DEVSIM line mesher
does not support a truly local unstructured patch, so the added lines are a
fixed, geometrically similar contact/interface refinement surrogate.

Only line spacing changes between family members.  Material parameters,
region coordinates, contact planes, and solver settings are outside this
module's scope.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import math
from typing import Any


NM_TO_CM = 1.0e-7
BASE_AXIAL_SPACING_NM = 0.25
BASE_RADIAL_SPACING_NM = 0.50
POSITION_ABS_TOLERANCE_CM = 1.0e-18

LEVELS = (
    "local_base",
    "local_fine",
    "local_extra_fine",
    "local_ultra_fine",
)
LEVEL_FACTORS = {
    "local_base": 1.0,
    "local_fine": 0.5,
    "local_extra_fine": 0.25,
    "local_ultra_fine": 0.125,
}

AXIAL_DISTANCE_NM = (0.0, 0.25, 0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0)
AXIAL_INTERVAL_SPACING_MULTIPLIERS = (
    1.0,
    1.0,
    1.0,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
    16.0,
)
AXIAL_INTERVAL_MINIMUM_SPACING_NM = (
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.25,
    0.50,
    1.00,
    2.00,
)
# Backward-compatible export.  The values now describe the interval starting
# at each contact-distance anchor, rather than a symmetric point spacing.
AXIAL_SPACING_MULTIPLIERS = AXIAL_INTERVAL_SPACING_MULTIPLIERS
RADIAL_ANCHOR_COUNT = 5
AIR_GUARD_SPACING_NM = 2.0

REQUIRED_GEOMETRY_KEYS = (
    "r_axis",
    "r_core",
    "r_mos2",
    "r_tox",
    "r_trap",
    "r_block",
    "r_gate_outer",
    "r_air_outer",
    "z_source",
    "z_drain",
)
PASSTHROUGH_RADIAL_KEYS = (
    "r_axis",
    "r_tox",
    "r_trap",
    "r_block",
    "r_gate_outer",
    "r_air_outer",
)


def _finite(value: Any, description: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{description} must be finite")
    return result


def _positive(value: Any, description: str) -> float:
    result = _finite(value, description)
    if result <= 0.0:
        raise ValueError(f"{description} must be positive")
    return result


@dataclass(frozen=True)
class LocalMeshLevel:
    """One member of the dyadic local-spacing ladder."""

    name: str
    factor: float
    axial_spacing_cm: float
    radial_spacing_cm: float

    @property
    def axial_spacing_nm(self) -> float:
        return self.axial_spacing_cm / NM_TO_CM

    @property
    def radial_spacing_nm(self) -> float:
        return self.radial_spacing_cm / NM_TO_CM


def get_level_spec(level: str | LocalMeshLevel) -> LocalMeshLevel:
    """Return a validated immutable specification for ``level``."""

    if isinstance(level, LocalMeshLevel):
        if level.name not in LEVEL_FACTORS:
            raise ValueError(f"unknown local mesh level {level.name!r}")
        expected = LEVEL_FACTORS[level.name]
        if not math.isclose(level.factor, expected, rel_tol=0.0, abs_tol=0.0):
            raise ValueError(f"noncanonical factor for {level.name}")
        return level
    name = str(level)
    try:
        factor = LEVEL_FACTORS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown local mesh level {name!r}; expected one of {LEVELS}"
        ) from error
    return LocalMeshLevel(
        name=name,
        factor=factor,
        axial_spacing_cm=BASE_AXIAL_SPACING_NM * NM_TO_CM * factor,
        radial_spacing_cm=BASE_RADIAL_SPACING_NM * NM_TO_CM * factor,
    )


@dataclass(frozen=True)
class MeshLinePolicy:
    """Expected treatment of one unique runtime mesh-line position."""

    direction: str
    position_cm: float
    target: str
    negative_spacing_cm: float | None
    positive_spacing_cm: float | None
    injected: bool
    position_rule: str

    @property
    def controlled_spacing_cm(self) -> float | None:
        """Compatibility view of the positive-side requested spacing."""

        return self.positive_spacing_cm


@dataclass(frozen=True)
class LocalMeshPolicy:
    """Complete fixed-position policy for one local mesh family member."""

    level: LocalMeshLevel
    geometry: Mapping[str, float]
    lines: tuple[MeshLinePolicy, ...]
    position_hash: str

    def match(self, direction: str, position_cm: float) -> MeshLinePolicy | None:
        for line in self.lines:
            if line.direction == direction and math.isclose(
                line.position_cm,
                position_cm,
                rel_tol=0.0,
                abs_tol=POSITION_ABS_TOLERANCE_CM,
            ):
                return line
        return None


def _validated_geometry(geometry: Mapping[str, Any]) -> dict[str, float]:
    missing = [key for key in REQUIRED_GEOMETRY_KEYS if key not in geometry]
    if missing:
        raise ValueError("missing geometry keys: " + ", ".join(missing))
    values = {key: _finite(geometry[key], key) for key in REQUIRED_GEOMETRY_KEYS}
    radial = [values[key] for key in REQUIRED_GEOMETRY_KEYS if key.startswith("r_")]
    if not all(right > left for left, right in zip(radial, radial[1:])):
        raise ValueError("runtime radial boundaries must be strictly increasing")
    if values["z_drain"] <= values["z_source"]:
        raise ValueError("z_drain must exceed z_source")
    required_half_length = 2.0 * max(AXIAL_DISTANCE_NM) * NM_TO_CM
    if values["z_drain"] - values["z_source"] <= required_half_length:
        raise ValueError("channel is too short for disjoint symmetric axial anchors")
    return values


def _radial_anchor_positions(geometry: Mapping[str, float]) -> tuple[float, ...]:
    inner = geometry["r_core"]
    outer = geometry["r_mos2"]
    step = (outer - inner) / float(RADIAL_ANCHOR_COUNT - 1)
    return tuple(inner + index * step for index in range(RADIAL_ANCHOR_COUNT))


def _position_digest(lines: Sequence[MeshLinePolicy]) -> str:
    payload = "\n".join(
        f"{line.direction}|{line.position_cm:.18e}"
        for line in sorted(lines, key=lambda item: (item.direction, item.position_cm))
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_local_mesh_policy(
    geometry: Mapping[str, Any], level: str | LocalMeshLevel
) -> LocalMeshPolicy:
    """Build the fixed anchors and level-specific spacings for one mesh."""

    values = _validated_geometry(geometry)
    spec = get_level_spec(level)
    lines: list[MeshLinePolicy] = []

    for key in PASSTHROUGH_RADIAL_KEYS:
        lines.append(
            MeshLinePolicy(
                direction="x",
                position_cm=values[key],
                target=f"radial_boundary_{key}",
                negative_spacing_cm=None,
                positive_spacing_cm=None,
                injected=False,
                position_rule="runtime geometry boundary; spacing passthrough",
            )
        )

    for index, position in enumerate(_radial_anchor_positions(values)):
        lines.append(
            MeshLinePolicy(
                direction="x",
                position_cm=position,
                target=f"mos2_radial_anchor_{index}",
                negative_spacing_cm=(
                    None if index == 0 else spec.radial_spacing_cm
                ),
                positive_spacing_cm=(
                    None
                    if index == RADIAL_ANCHOR_COUNT - 1
                    else spec.radial_spacing_cm
                ),
                injected=index not in (0, RADIAL_ANCHOR_COUNT - 1),
                position_rule=(
                    f"r_core + {index}/{RADIAL_ANCHOR_COUNT - 1} "
                    "* (r_mos2-r_core)"
                ),
            )
        )

    shell_width = values["r_mos2"] - values["r_core"]
    for side, position in (
        ("source", values["z_source"] - shell_width),
        ("drain", values["z_drain"] + shell_width),
    ):
        lines.append(
            MeshLinePolicy(
                direction="y",
                position_cm=position,
                target=f"{side}_air_guard",
                negative_spacing_cm=(
                    AIR_GUARD_SPACING_NM * NM_TO_CM
                ),
                positive_spacing_cm=(
                    AIR_GUARD_SPACING_NM * NM_TO_CM
                ),
                injected=False,
                position_rule="fixed topology guard position",
            )
        )

    for side, sign, endpoint in (
        ("source", 1.0, values["z_source"]),
        ("drain", -1.0, values["z_drain"]),
    ):
        interval_spacings = tuple(
            max(
                multiplier * spec.axial_spacing_cm,
                minimum_nm * NM_TO_CM,
            )
            for multiplier, minimum_nm in zip(
                AXIAL_INTERVAL_SPACING_MULTIPLIERS,
                AXIAL_INTERVAL_MINIMUM_SPACING_NM,
            )
        )
        guard_spacing = AIR_GUARD_SPACING_NM * NM_TO_CM
        for index, distance_nm in enumerate(AXIAL_DISTANCE_NM):
            if side == "source":
                negative_spacing = (
                    guard_spacing if index == 0 else interval_spacings[index - 1]
                )
                positive_spacing = interval_spacings[index]
            else:
                negative_spacing = interval_spacings[index]
                positive_spacing = (
                    guard_spacing if index == 0 else interval_spacings[index - 1]
                )
            lines.append(
                MeshLinePolicy(
                    direction="y",
                    position_cm=endpoint + sign * distance_nm * NM_TO_CM,
                    target=f"{side}_axial_anchor_{index}",
                    negative_spacing_cm=negative_spacing,
                    positive_spacing_cm=positive_spacing,
                    injected=index != 0,
                    position_rule=f"{side} contact inward distance {distance_nm:g} nm",
                )
            )

    positions = [(line.direction, line.position_cm) for line in lines]
    unique_positions = {
        (direction, round(position, 18)) for direction, position in positions
    }
    if len(unique_positions) != len(positions):
        raise ValueError("local mesh policy contains duplicate line positions")
    line_tuple = tuple(lines)
    return LocalMeshPolicy(
        level=spec,
        geometry=values,
        lines=line_tuple,
        position_hash=_position_digest(line_tuple),
    )


def local_mesh_position_hash(geometry: Mapping[str, Any]) -> str:
    """Hash only mesh directions/positions; spacing is deliberately excluded."""

    return build_local_mesh_policy(geometry, LEVELS[0]).position_hash


def _row_number(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        value = row.get(name)
        if value is not None and value != "":
            return _finite(value, name)
    return None


def validate_local_mesh_line_log(
    rows: Sequence[Mapping[str, Any]],
    geometry: Mapping[str, Any],
    level: str | LocalMeshLevel,
) -> dict[str, Any]:
    """Validate realized positions and spacings without relying on call order.

    Both this module's own log schema and the legacy candidate runner's
    ``base_spacing_cm``/``scaled_spacing_cm`` schema are accepted.  This lets
    callers install the validator directly in the existing QC runner.
    """

    policy = build_local_mesh_policy(geometry, level)
    errors: list[str] = []
    matched: dict[str, Mapping[str, Any]] = {}
    unmatched: list[int] = []

    for index, row in enumerate(rows):
        direction = str(row.get("direction", ""))
        try:
            position = _finite(row.get("position_cm"), f"line {index} position")
        except (TypeError, ValueError):
            errors.append(f"line {index} invalid position")
            continue
        expected = policy.match(direction, position)
        if expected is None:
            unmatched.append(index)
            continue
        if expected.target in matched:
            errors.append(f"duplicate position for {expected.target}")
            continue
        matched[expected.target] = row

        output_positive = _row_number(
            row,
            (
                "output_positive_spacing_cm",
                "output_spacing_cm",
                "scaled_spacing_cm",
                "spacing_cm",
                "ps",
            ),
        )
        output_negative = _row_number(
            row,
            ("output_negative_spacing_cm", "negative_spacing_cm", "ns"),
        )
        if output_positive is None or output_positive <= 0.0:
            errors.append(f"{expected.target} missing positive output spacing")
            continue
        input_spacing = _row_number(
            row, ("input_spacing_cm", "base_spacing_cm")
        )
        expected_positive = expected.positive_spacing_cm
        if expected_positive is None:
            expected_positive = input_spacing
        if expected_positive is None:
            errors.append(f"{expected.target} missing passthrough input spacing")
        elif not math.isclose(
            output_positive,
            expected_positive,
            rel_tol=1.0e-14,
            abs_tol=1.0e-30,
        ):
            errors.append(f"{expected.target} positive-side spacing")

        expected_negative = expected.negative_spacing_cm
        if expected_negative is None:
            expected_negative = input_spacing
        if output_negative is not None and expected_negative is not None:
            if not math.isclose(
                output_negative,
                expected_negative,
                rel_tol=1.0e-14,
                abs_tol=1.0e-30,
            ):
                errors.append(f"{expected.target} negative-side spacing")

    if unmatched:
        errors.append("unexpected line indices=" + ",".join(map(str, unmatched)))
    missing = [line.target for line in policy.lines if line.target not in matched]
    if missing:
        errors.append("missing targets=" + ",".join(missing))
    if len(rows) != len(policy.lines):
        errors.append(f"line_count={len(rows)} expected={len(policy.lines)}")

    return {
        "passed": not errors,
        "errors": errors,
        "level": policy.level.name,
        "factor": policy.level.factor,
        "line_count": len(rows),
        "expected_line_count": len(policy.lines),
        "position_hash": policy.position_hash,
        "lines": [dict(row) for row in rows],
    }


@contextmanager
def local_runtime_mesh_lines(
    structure_module: Any,
    geometry: Mapping[str, Any],
    level: str | LocalMeshLevel,
    line_log: list[dict[str, Any]],
) -> Iterator[None]:
    """Temporarily install one fixed-position local mesh policy.

    The callable present at entry is captured and used for every original and
    injected line.  Calling the captured downstream layer avoids wrapper
    recursion and composes with the existing global mesh logger.
    """

    policy = build_local_mesh_policy(geometry, level)
    downstream = structure_module.add_2d_mesh_line
    radial_injected = False
    axial_injected = False
    seen_original_targets: set[str] = set()

    radial_interior = [
        line
        for line in policy.lines
        if line.target.startswith("mos2_radial_anchor_") and line.injected
    ]
    axial_interior = [
        line
        for line in policy.lines
        if "_axial_anchor_" in line.target and line.injected
    ]

    def forward(
        request: Mapping[str, Any], expected: MeshLinePolicy, *, injected: bool,
        input_spacing: float | None,
    ) -> Any:
        forwarded = dict(request)
        forwarded.update({"dir": expected.direction, "pos": expected.position_cm})
        if input_spacing is None and (
            expected.negative_spacing_cm is None
            or expected.positive_spacing_cm is None
        ):
            raise RuntimeError("one-sided passthrough mesh line has no input spacing")
        output_positive = (
            input_spacing
            if expected.positive_spacing_cm is None
            else expected.positive_spacing_cm
        )
        output_negative = (
            input_spacing
            if expected.negative_spacing_cm is None
            else expected.negative_spacing_cm
        )
        if output_positive is None or output_negative is None:
            raise RuntimeError("mesh line did not resolve both side spacings")
        forwarded["ps"] = output_positive
        # Explicit ns/ps controls each interval independently.  This avoids
        # the tiny grading remainder cells produced by a symmetric point ps
        # at the ultra-fine transition while keeping the tunnel/core sides at
        # their canonical input spacing.
        forwarded["ns"] = output_negative
        result = downstream(**forwarded)
        output_positive = _positive(
            forwarded["ps"], "forwarded positive mesh spacing"
        )
        output_negative = _positive(
            forwarded["ns"], "forwarded negative mesh spacing"
        )
        line_log.append(
            {
                "level": policy.level.name,
                "level_factor": policy.level.factor,
                "direction": expected.direction,
                "position_cm": expected.position_cm,
                "input_spacing_cm": input_spacing,
                "input_negative_spacing_cm": (
                    None if input_spacing is None else input_spacing
                ),
                "output_spacing_cm": output_positive,
                "output_positive_spacing_cm": output_positive,
                "output_negative_spacing_cm": output_negative,
                "base_spacing_cm": input_spacing,
                "scaled_spacing_cm": output_positive,
                "target": expected.target,
                "injected": injected,
                "position_hash": policy.position_hash,
            }
        )
        return result

    def add_local_line(*args: Any, **kwargs: Any) -> Any:
        nonlocal radial_injected, axial_injected
        if args:
            raise RuntimeError("local mesh adapter requires keyword mesh-line calls")
        direction = str(kwargs.get("dir", ""))
        if direction not in {"x", "y"}:
            raise RuntimeError(f"unexpected mesh-line direction {direction!r}")
        position = _finite(kwargs.get("pos"), "mesh-line position")
        input_spacing = _positive(kwargs.get("ps"), "mesh-line spacing")
        expected = policy.match(direction, position)
        if expected is None or expected.injected:
            raise RuntimeError(
                f"unexpected original mesh line at {direction}={position:.18e} cm"
            )
        if expected.target in seen_original_targets:
            raise RuntimeError(f"duplicate original mesh line {expected.target}")
        seen_original_targets.add(expected.target)

        result = forward(
            kwargs, expected, injected=False, input_spacing=input_spacing
        )
        mesh_name = kwargs.get("mesh")
        if mesh_name is None:
            raise RuntimeError("mesh-line call omitted mesh name")

        if (
            expected.target in {"mos2_radial_anchor_0", "mos2_radial_anchor_4"}
            and not radial_injected
        ):
            for added in radial_interior:
                forward(
                    {"mesh": mesh_name}, added, injected=True, input_spacing=None
                )
            radial_injected = True
        if expected.target in {"source_axial_anchor_0", "drain_axial_anchor_0"} and not axial_injected:
            for added in axial_interior:
                forward(
                    {"mesh": mesh_name}, added, injected=True, input_spacing=None
                )
            axial_injected = True
        return result

    structure_module.add_2d_mesh_line = add_local_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = downstream


def local_mesh_family_definition_rows(
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return long-form, CSV-ready definitions for all four mesh levels."""

    rows: list[dict[str, Any]] = []
    for level_name in LEVELS:
        policy = build_local_mesh_policy(geometry, level_name)
        for line in sorted(
            policy.lines, key=lambda item: (item.direction, item.position_cm)
        ):
            negative_spacing = line.negative_spacing_cm
            positive_spacing = line.positive_spacing_cm
            rows.append(
                {
                    "mesh_level": level_name,
                    "level_factor": policy.level.factor,
                    "base_axial_spacing_nm": BASE_AXIAL_SPACING_NM,
                    "base_radial_spacing_nm": BASE_RADIAL_SPACING_NM,
                    "direction": line.direction,
                    "target": line.target,
                    "position_cm": line.position_cm,
                    "position_nm": line.position_cm / NM_TO_CM,
                    "controlled_spacing_cm": (
                        "" if positive_spacing is None else positive_spacing
                    ),
                    "controlled_spacing_nm": (
                        ""
                        if positive_spacing is None
                        else positive_spacing / NM_TO_CM
                    ),
                    "negative_spacing_cm": (
                        "" if negative_spacing is None else negative_spacing
                    ),
                    "negative_spacing_nm": (
                        ""
                        if negative_spacing is None
                        else negative_spacing / NM_TO_CM
                    ),
                    "positive_spacing_cm": (
                        "" if positive_spacing is None else positive_spacing
                    ),
                    "positive_spacing_nm": (
                        ""
                        if positive_spacing is None
                        else positive_spacing / NM_TO_CM
                    ),
                    "spacing_policy": (
                        "passthrough"
                        if negative_spacing is None and positive_spacing is None
                        else "dyadic_interval_controlled"
                    ),
                    "injected": line.injected,
                    "position_rule": line.position_rule,
                    "position_hash_sha256": policy.position_hash,
                }
            )
    return rows


__all__ = (
    "AXIAL_DISTANCE_NM",
    "AXIAL_INTERVAL_SPACING_MULTIPLIERS",
    "AXIAL_INTERVAL_MINIMUM_SPACING_NM",
    "AXIAL_SPACING_MULTIPLIERS",
    "BASE_AXIAL_SPACING_NM",
    "BASE_RADIAL_SPACING_NM",
    "AIR_GUARD_SPACING_NM",
    "LEVELS",
    "LEVEL_FACTORS",
    "LocalMeshLevel",
    "LocalMeshPolicy",
    "MeshLinePolicy",
    "build_local_mesh_policy",
    "get_level_spec",
    "local_mesh_family_definition_rows",
    "local_mesh_position_hash",
    "local_runtime_mesh_lines",
    "validate_local_mesh_line_log",
)
