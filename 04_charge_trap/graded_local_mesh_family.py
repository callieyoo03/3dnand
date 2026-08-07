"""Quality-preserving fixed-position local mesh ladder.

This characterization-only adapter leaves the canonical runtime geometry,
regions, contacts, materials, and solver unchanged.  It adds the same radial
and symmetric axial mesh-line positions at every level and changes only the
requested line spacings.  Explicit ``ns``/``ps`` values grade each active
axial interval by at most a factor of two and keep the worst MoS2 cell
anisotropy constant across the family.

The DEVSIM line mesher creates cross-lines, not a truly local unstructured
patch.  Consequently, halving the MoS2 radial spacing also requires a
corresponding far-field axial reduction to avoid high-aspect MoS2 triangles.
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
AIR_GUARD_SPACING_NM = 2.0
MAX_ACTIVE_ADJACENT_SPACING_RATIO = 2.0

LEVELS = (
    "graded_base",
    "graded_fine",
    "graded_extra_fine",
    "graded_ultra_fine",
)
LEVEL_FACTORS = {
    "graded_base": 1.0,
    "graded_fine": 0.5,
    "graded_extra_fine": 0.25,
    "graded_ultra_fine": 0.125,
}

# Fixed inward distances from either physical contact plane.  The 7-nm anchor
# gives the final 1->2 transition its own exactly controlled interval before
# the central channel.
AXIAL_DISTANCE_NM = (
    0.0,
    0.25,
    0.5,
    1.0,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    7.0,
)
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
    16.0,
)
# Caps apply to the finite 4--5 and 5--7 nm transition intervals.  The final
# entry is the uncapped central interval beginning at the 7-nm anchor.
AXIAL_INTERVAL_MAXIMUM_SPACING_NM = (
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    1.0,
    2.0,
    None,
)
# Compatibility export retained for callers of the previous family API.  The
# graded family deliberately has no absolute anti-refinement floor.
AXIAL_INTERVAL_MINIMUM_SPACING_NM = (0.0,) * len(AXIAL_DISTANCE_NM)
AXIAL_SPACING_MULTIPLIERS = AXIAL_INTERVAL_SPACING_MULTIPLIERS

RADIAL_ANCHOR_COUNT = 5
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


def _spacing_ratio(first: float, second: float) -> float:
    smaller = min(_positive(first, "spacing"), _positive(second, "spacing"))
    return max(first, second) / smaller


@dataclass(frozen=True)
class LocalMeshLevel:
    """One member of the geometrically similar graded ladder."""

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
    """Return the canonical immutable specification for ``level``."""

    if isinstance(level, LocalMeshLevel):
        if level.name not in LEVEL_FACTORS:
            raise ValueError(f"unknown graded mesh level {level.name!r}")
        if not math.isclose(
            level.factor,
            LEVEL_FACTORS[level.name],
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(f"noncanonical factor for {level.name}")
        return level
    name = str(level)
    try:
        factor = LEVEL_FACTORS[name]
    except KeyError as error:
        raise ValueError(
            f"unknown graded mesh level {name!r}; expected one of {LEVELS}"
        ) from error
    return LocalMeshLevel(
        name=name,
        factor=factor,
        axial_spacing_cm=BASE_AXIAL_SPACING_NM * factor * NM_TO_CM,
        radial_spacing_cm=BASE_RADIAL_SPACING_NM * factor * NM_TO_CM,
    )


@dataclass(frozen=True)
class MeshLinePolicy:
    """Expected two-sided request for one unique runtime mesh line."""

    direction: str
    position_cm: float
    target: str
    negative_spacing_cm: float | None
    positive_spacing_cm: float | None
    injected: bool
    position_rule: str
    adjacent_requested_spacing_ratio: float | None = None
    adjacent_ratio_scope: str = "not_applicable"

    @property
    def controlled_spacing_cm(self) -> float | None:
        """Compatibility view of the positive-side requested spacing."""

        return self.positive_spacing_cm


@dataclass(frozen=True)
class LocalMeshPolicy:
    """Complete position and two-sided-spacing policy for one level."""

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
    missing = [name for name in REQUIRED_GEOMETRY_KEYS if name not in geometry]
    if missing:
        raise ValueError("missing geometry keys: " + ", ".join(missing))
    values = {
        name: _finite(geometry[name], name) for name in REQUIRED_GEOMETRY_KEYS
    }
    radial = [
        values[name] for name in REQUIRED_GEOMETRY_KEYS if name.startswith("r_")
    ]
    if not all(right > left for left, right in zip(radial, radial[1:])):
        raise ValueError("runtime radial boundaries must be strictly increasing")
    if values["z_drain"] <= values["z_source"]:
        raise ValueError("z_drain must exceed z_source")
    minimum_channel = 2.0 * max(AXIAL_DISTANCE_NM) * NM_TO_CM
    if values["z_drain"] - values["z_source"] <= minimum_channel:
        raise ValueError("channel is too short for disjoint graded axial anchors")
    return values


def _radial_anchor_positions(geometry: Mapping[str, float]) -> tuple[float, ...]:
    inner = geometry["r_core"]
    outer = geometry["r_mos2"]
    step = (outer - inner) / float(RADIAL_ANCHOR_COUNT - 1)
    return tuple(inner + index * step for index in range(RADIAL_ANCHOR_COUNT))


def axial_interval_spacings_cm(
    level: str | LocalMeshLevel,
) -> tuple[float, ...]:
    """Return the ten source-to-centre interval targets for one level."""

    spec = get_level_spec(level)
    targets: list[float] = []
    for multiplier, cap_nm in zip(
        AXIAL_INTERVAL_SPACING_MULTIPLIERS,
        AXIAL_INTERVAL_MAXIMUM_SPACING_NM,
    ):
        spacing_nm = multiplier * spec.axial_spacing_nm
        if cap_nm is not None:
            spacing_nm = min(spacing_nm, cap_nm)
        targets.append(spacing_nm * NM_TO_CM)
    ratios = [
        _spacing_ratio(left, right)
        for left, right in zip(targets, targets[1:])
    ]
    if any(
        ratio > MAX_ACTIVE_ADJACENT_SPACING_RATIO + 1.0e-14
        for ratio in ratios
    ):
        raise ValueError("graded axial interval ratio exceeds two")
    return tuple(targets)


def _position_digest(lines: Sequence[MeshLinePolicy]) -> str:
    payload = "\n".join(
        f"{line.direction}|{line.position_cm:.18e}"
        for line in sorted(lines, key=lambda item: (item.direction, item.position_cm))
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def build_local_mesh_policy(
    geometry: Mapping[str, Any], level: str | LocalMeshLevel
) -> LocalMeshPolicy:
    """Build one fixed-position, ratio-bounded graded mesh policy."""

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
        interior = index not in (0, RADIAL_ANCHOR_COUNT - 1)
        lines.append(
            MeshLinePolicy(
                direction="x",
                position_cm=position,
                target=f"mos2_radial_anchor_{index}",
                negative_spacing_cm=(
                    spec.radial_spacing_cm if index != 0 else None
                ),
                positive_spacing_cm=(
                    spec.radial_spacing_cm
                    if index != RADIAL_ANCHOR_COUNT - 1
                    else None
                ),
                injected=interior,
                position_rule=(
                    f"r_core + {index}/{RADIAL_ANCHOR_COUNT - 1} "
                    "* (r_mos2-r_core)"
                ),
                adjacent_requested_spacing_ratio=(1.0 if interior else None),
                adjacent_ratio_scope=(
                    "mos2_radial_intervals" if interior else "one_sided_mos2"
                ),
            )
        )

    shell_width = values["r_mos2"] - values["r_core"]
    guard_spacing = AIR_GUARD_SPACING_NM * NM_TO_CM
    for side, position in (
        ("source", values["z_source"] - shell_width),
        ("drain", values["z_drain"] + shell_width),
    ):
        lines.append(
            MeshLinePolicy(
                direction="y",
                position_cm=position,
                target=f"{side}_air_guard",
                negative_spacing_cm=guard_spacing,
                positive_spacing_cm=guard_spacing,
                injected=False,
                position_rule="fixed topology guard position",
                adjacent_requested_spacing_ratio=1.0,
                adjacent_ratio_scope="air_guard_only",
            )
        )

    interval_spacings = axial_interval_spacings_cm(spec)
    for side, inward_sign, endpoint in (
        ("source", 1.0, values["z_source"]),
        ("drain", -1.0, values["z_drain"]),
    ):
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
            ratio = (
                None
                if index == 0
                else _spacing_ratio(
                    interval_spacings[index - 1], interval_spacings[index]
                )
            )
            lines.append(
                MeshLinePolicy(
                    direction="y",
                    position_cm=(
                        endpoint + inward_sign * distance_nm * NM_TO_CM
                    ),
                    target=f"{side}_axial_anchor_{index}",
                    negative_spacing_cm=negative_spacing,
                    positive_spacing_cm=positive_spacing,
                    injected=index != 0,
                    position_rule=(
                        f"{side} contact inward distance {distance_nm:g} nm"
                    ),
                    adjacent_requested_spacing_ratio=ratio,
                    adjacent_ratio_scope=(
                        "one_sided_active_contact_anchor"
                        if index == 0
                        else "adjacent_active_axial_intervals"
                    ),
                )
            )

    rounded_positions = {
        (line.direction, round(line.position_cm, 18)) for line in lines
    }
    if len(rounded_positions) != len(lines):
        raise ValueError("graded mesh policy contains duplicate line positions")
    active_ratios = [
        line.adjacent_requested_spacing_ratio
        for line in lines
        if line.adjacent_ratio_scope == "adjacent_active_axial_intervals"
    ]
    if any(
        ratio is None
        or ratio > MAX_ACTIVE_ADJACENT_SPACING_RATIO + 1.0e-14
        for ratio in active_ratios
    ):
        raise ValueError("graded mesh policy violates adjacent ratio limit")

    line_tuple = tuple(lines)
    return LocalMeshPolicy(
        level=spec,
        geometry=values,
        lines=line_tuple,
        position_hash=_position_digest(line_tuple),
    )


def local_mesh_position_hash(geometry: Mapping[str, Any]) -> str:
    """Hash only fixed directions and positions, excluding all spacings."""

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
    """Validate complete realized requests without relying on call order."""

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

        input_spacing = _row_number(
            row, ("input_spacing_cm", "base_spacing_cm")
        )
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
            row, ("output_negative_spacing_cm", "negative_spacing_cm", "ns")
        )
        if output_positive is None or output_positive <= 0.0:
            errors.append(f"{expected.target} missing positive output spacing")
            continue
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
        if output_negative is None and (
            "target" in row or "output_positive_spacing_cm" in row
        ):
            # The native graded log is required to retain both requests.  A
            # legacy outer logger may expose only base/scaled ``ps`` fields;
            # keep accepting that schema for compatibility with existing QC
            # runner validators.
            errors.append(f"{expected.target} missing negative output spacing")
        elif output_negative is None:
            pass
        elif expected_negative is None:
            errors.append(f"{expected.target} missing negative input spacing")
        elif not math.isclose(
            output_negative,
            expected_negative,
            rel_tol=1.0e-14,
            abs_tol=1.0e-30,
        ):
            errors.append(f"{expected.target} negative-side spacing")

        logged_ratio = _row_number(
            row, ("adjacent_requested_spacing_ratio",)
        )
        if expected.adjacent_requested_spacing_ratio is not None:
            if logged_ratio is not None and not math.isclose(
                logged_ratio,
                expected.adjacent_requested_spacing_ratio,
                rel_tol=1.0e-14,
                abs_tol=1.0e-30,
            ):
                errors.append(f"{expected.target} adjacent spacing ratio")

    if unmatched:
        errors.append("unexpected line indices=" + ",".join(map(str, unmatched)))
    missing = [line.target for line in policy.lines if line.target not in matched]
    if missing:
        errors.append("missing targets=" + ",".join(missing))
    if len(rows) != len(policy.lines):
        errors.append(f"line_count={len(rows)} expected={len(policy.lines)}")

    active_ratios = [
        line.adjacent_requested_spacing_ratio
        for line in policy.lines
        if line.adjacent_ratio_scope == "adjacent_active_axial_intervals"
        and line.adjacent_requested_spacing_ratio is not None
    ]
    return {
        "passed": not errors,
        "errors": errors,
        "level": policy.level.name,
        "factor": policy.level.factor,
        "line_count": len(rows),
        "expected_line_count": len(policy.lines),
        "position_hash": policy.position_hash,
        "maximum_adjacent_requested_spacing_ratio": max(active_ratios),
        "adjacent_requested_spacing_ratio_passed": (
            max(active_ratios) <= MAX_ACTIVE_ADJACENT_SPACING_RATIO
        ),
        "lines": [dict(row) for row in rows],
    }


@contextmanager
def local_runtime_mesh_lines(
    structure_module: Any,
    geometry: Mapping[str, Any],
    level: str | LocalMeshLevel,
    line_log: list[dict[str, Any]],
) -> Iterator[None]:
    """Temporarily install a graded two-sided runtime mesh-line adapter."""

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
        request: Mapping[str, Any],
        expected: MeshLinePolicy,
        *,
        injected: bool,
        input_spacing: float | None,
    ) -> Any:
        forwarded = dict(request)
        forwarded.update({"dir": expected.direction, "pos": expected.position_cm})
        if input_spacing is None and (
            expected.negative_spacing_cm is None
            or expected.positive_spacing_cm is None
        ):
            raise RuntimeError("one-sided passthrough line has no input spacing")
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
            raise RuntimeError("graded line did not resolve both side spacings")
        forwarded["ps"] = output_positive
        forwarded["ns"] = output_negative
        result = downstream(**forwarded)
        line_log.append(
            {
                "level": policy.level.name,
                "level_factor": policy.level.factor,
                "direction": expected.direction,
                "position_cm": expected.position_cm,
                "input_spacing_cm": input_spacing,
                "input_negative_spacing_cm": input_spacing,
                "output_spacing_cm": output_positive,
                "output_positive_spacing_cm": output_positive,
                "output_negative_spacing_cm": output_negative,
                "base_spacing_cm": input_spacing,
                "scaled_spacing_cm": output_positive,
                "target": expected.target,
                "injected": injected,
                "adjacent_requested_spacing_ratio": (
                    expected.adjacent_requested_spacing_ratio
                ),
                "adjacent_ratio_scope": expected.adjacent_ratio_scope,
                "position_hash": policy.position_hash,
            }
        )
        return result

    def add_graded_line(*args: Any, **kwargs: Any) -> Any:
        nonlocal radial_injected, axial_injected
        if args:
            raise RuntimeError("graded mesh adapter requires keyword calls")
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
                    {"mesh": mesh_name},
                    added,
                    injected=True,
                    input_spacing=None,
                )
            radial_injected = True
        if (
            expected.target in {"source_axial_anchor_0", "drain_axial_anchor_0"}
            and not axial_injected
        ):
            for added in axial_interior:
                forward(
                    {"mesh": mesh_name},
                    added,
                    injected=True,
                    input_spacing=None,
                )
            axial_injected = True
        return result

    structure_module.add_2d_mesh_line = add_graded_line
    try:
        yield
    finally:
        structure_module.add_2d_mesh_line = downstream


def local_mesh_family_definition_rows(
    geometry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return long-form CSV-ready definitions including both side requests."""

    rows: list[dict[str, Any]] = []
    for level_name in LEVELS:
        policy = build_local_mesh_policy(geometry, level_name)
        for line in sorted(
            policy.lines, key=lambda item: (item.direction, item.position_cm)
        ):
            negative = line.negative_spacing_cm
            positive = line.positive_spacing_cm
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
                        "" if positive is None else positive
                    ),
                    "controlled_spacing_nm": (
                        "" if positive is None else positive / NM_TO_CM
                    ),
                    "negative_spacing_cm": "" if negative is None else negative,
                    "negative_spacing_nm": (
                        "" if negative is None else negative / NM_TO_CM
                    ),
                    "positive_spacing_cm": "" if positive is None else positive,
                    "positive_spacing_nm": (
                        "" if positive is None else positive / NM_TO_CM
                    ),
                    "adjacent_requested_spacing_ratio": (
                        ""
                        if line.adjacent_requested_spacing_ratio is None
                        else line.adjacent_requested_spacing_ratio
                    ),
                    "adjacent_ratio_scope": line.adjacent_ratio_scope,
                    "spacing_policy": (
                        "passthrough"
                        if negative is None and positive is None
                        else "graded_interval_controlled"
                    ),
                    "injected": line.injected,
                    "position_rule": line.position_rule,
                    "position_hash_sha256": policy.position_hash,
                }
            )
    return rows


__all__ = (
    "AIR_GUARD_SPACING_NM",
    "AXIAL_DISTANCE_NM",
    "AXIAL_INTERVAL_MAXIMUM_SPACING_NM",
    "AXIAL_INTERVAL_MINIMUM_SPACING_NM",
    "AXIAL_INTERVAL_SPACING_MULTIPLIERS",
    "AXIAL_SPACING_MULTIPLIERS",
    "BASE_AXIAL_SPACING_NM",
    "BASE_RADIAL_SPACING_NM",
    "LEVELS",
    "LEVEL_FACTORS",
    "MAX_ACTIVE_ADJACENT_SPACING_RATIO",
    "LocalMeshLevel",
    "LocalMeshPolicy",
    "MeshLinePolicy",
    "axial_interval_spacings_cm",
    "build_local_mesh_policy",
    "get_level_spec",
    "local_mesh_family_definition_rows",
    "local_mesh_position_hash",
    "local_runtime_mesh_lines",
    "validate_local_mesh_line_log",
)
