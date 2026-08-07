"""Parameters for the independent ERASE electron-FN negative control.

This module intentionally contains no DEVSIM dependency.  Geometry and
dielectric values come from the repository's compact-handoff source of truth;
the Fowler--Nordheim constants are imported from the existing *electron*
program model and are labelled as an uncalibrated diagnostic negative control.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import MappingProxyType


MODULE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT_DIRECTORY = MODULE_DIRECTORY.parent

for search_path in (str(MODULE_DIRECTORY), str(PROJECT_ROOT_DIRECTORY)):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)

import compact_handoff_parameters as compact_parameters
import trap_parameters as static_trap_parameters
import tunneling_parameters as electron_tunneling_parameters


def _require_repository_module(module: object, expected_path: Path) -> None:
    actual_path = Path(getattr(module, "__file__", "")).resolve()
    if actual_path != expected_path.resolve():
        raise ImportError(
            f"Expected {expected_path.resolve()}, but imported {actual_path}."
        )


_require_repository_module(
    compact_parameters,
    PROJECT_ROOT_DIRECTORY / "compact_handoff_parameters.py",
)
_require_repository_module(
    static_trap_parameters,
    MODULE_DIRECTORY / "trap_parameters.py",
)
_require_repository_module(
    electron_tunneling_parameters,
    MODULE_DIRECTORY / "tunneling_parameters.py",
)


# Mandatory provenance labels for every generated negative-baseline result.
MODEL_CLASS = "electron_FN_diagnostic"
PHYSICAL_ROLE = "negative_control"
MODEL_STATUS = "baseline_negative_result"
CALIBRATION_STATUS = "uncalibrated"
PREDICTIVE_ERASE_MODEL = False

MODEL_METADATA = MappingProxyType(
    {
        "model_class": MODEL_CLASS,
        "physical_role": PHYSICAL_ROLE,
        "model_status": MODEL_STATUS,
        "calibration_status": CALIBRATION_STATUS,
        "predictive_erase_model": PREDICTIVE_ERASE_MODEL,
    }
)


# Biases used by the report-based independent reimplementation.
SOURCE_VOLTAGE_V = 0.0
DRAIN_VOLTAGE_V = 0.0
ERASE_GATE_TARGETS_V = (0.0, -4.0, -6.0, -8.0, -10.0, -12.0)


# Charge state.  A positive density denotes the number of trapped electrons;
# its physical charge density is therefore negative.
INITIAL_TRAP_SHEET_DENSITY_CM2 = 1.0e12
INITIAL_TRAP_VOLUME_DENSITY_CM3 = 2.0e18
TARGET_RESIDUAL_SHEET_DENSITY_CM2 = 1.0e10
ELEMENTARY_CHARGE_C = 1.602176634e-19


# The active window is a diagnostic/integration window, not a geometry change.
ACTIVE_AXIAL_START_NM = 10.0
ACTIVE_AXIAL_STOP_NM = 90.0


# Existing solver values and continuation scales are kept separate from the
# physical parameter set.  Continuation changes only the path to a fixed state.
SOLVER_ABSOLUTE_ERROR = float(static_trap_parameters.SOLVER_ABSOLUTE_ERROR)
SOLVER_RELATIVE_ERROR = float(static_trap_parameters.SOLVER_RELATIVE_ERROR)
SOLVER_MAXIMUM_ITERATIONS = int(static_trap_parameters.SOLVER_MAXIMUM_ITERATIONS)
TRAP_INITIAL_STEP_CM3 = 2.0e17
TRAP_MINIMUM_STEP_CM3 = 1.0e15
GATE_INITIAL_STEP_V = 0.10
GATE_MINIMUM_STEP_V = 1.0e-3
STEP_GROWTH_FACTOR = 1.5


# Program electron parameters are retained only for a comparison proxy.  The
# legacy tunneling module's 4 nm thickness is deliberately not imported here;
# runtime thickness comes from the canonical 3 nm geometry instead.
ELECTRON_PROXY_BARRIER_HEIGHT_EV = float(
    electron_tunneling_parameters.BARRIER_HEIGHT_EV
)
ELECTRON_PROXY_EFFECTIVE_MASS_RATIO = float(
    electron_tunneling_parameters.TUNNEL_EFFECTIVE_MASS_RATIO
)
ELECTRON_PROXY_PARAMETER_PROVENANCE = (
    "05_program_erase/tunneling_parameters.py program-electron parameters; "
    "negative-control diagnostic only, not physical ERASE parameters"
)


def _finite_nonnegative(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return numeric


def charge_trap_thickness_cm() -> float:
    """Return the canonical HfO2 charge-trap thickness in centimetres."""

    return (
        float(compact_parameters.CHARGE_TRAP_THICKNESS_NM)
        * float(compact_parameters.NM_TO_CM)
    )


def sheet_to_volume_density(sheet_density_cm2: float) -> float:
    """Convert an areal trapped-electron density to a uniform volume density."""

    sheet = _finite_nonnegative("sheet_density_cm2", sheet_density_cm2)
    return sheet / charge_trap_thickness_cm()


def volume_to_sheet_density(volume_density_cm3: float) -> float:
    """Convert a uniform volume density through the canonical trap thickness."""

    volume = _finite_nonnegative("volume_density_cm3", volume_density_cm3)
    return volume * charge_trap_thickness_cm()


def trapped_electron_charge_density_C_cm3(
    volume_density_cm3: float,
) -> float:
    """Return physical charge density for positive trapped-electron occupancy."""

    volume = _finite_nonnegative("volume_density_cm3", volume_density_cm3)
    return -ELEMENTARY_CHARGE_C * volume


def runtime_geometry_expected_nm() -> dict[str, float]:
    """Return the exact canonical geometry expected by the ERASE runner."""

    return {
        **compact_parameters.geometry_dict(),
        **compact_parameters.expected_outer_radii_nm(),
    }


def runtime_material_expected() -> dict[str, float]:
    """Return canonical relative permittivities by solved physical region."""

    return compact_parameters.material_dict()


def validate_parameters() -> None:
    """Fail fast on inconsistent report inputs or provenance constants."""

    compact_parameters.validate_canonical_parameters()

    converted = sheet_to_volume_density(INITIAL_TRAP_SHEET_DENSITY_CM2)
    if not math.isclose(
        converted,
        INITIAL_TRAP_VOLUME_DENSITY_CM3,
        rel_tol=1.0e-12,
        abs_tol=0.0,
    ):
        raise ValueError(
            "Initial Nsheet and Nvolume are inconsistent with the canonical "
            "charge-trap thickness."
        )

    if ACTIVE_AXIAL_START_NM < 0.0 or not (
        ACTIVE_AXIAL_START_NM
        < ACTIVE_AXIAL_STOP_NM
        <= compact_parameters.CHANNEL_LENGTH_NM
    ):
        raise ValueError("The active axial diagnostic window is invalid.")

    if ERASE_GATE_TARGETS_V[0] != 0.0 or any(
        current >= previous
        for previous, current in zip(
            ERASE_GATE_TARGETS_V,
            ERASE_GATE_TARGETS_V[1:],
        )
    ):
        raise ValueError(
            "ERASE gate targets must start at zero and become strictly negative."
        )

    for name, value in (
        ("TRAP_INITIAL_STEP_CM3", TRAP_INITIAL_STEP_CM3),
        ("TRAP_MINIMUM_STEP_CM3", TRAP_MINIMUM_STEP_CM3),
        ("GATE_INITIAL_STEP_V", GATE_INITIAL_STEP_V),
        ("GATE_MINIMUM_STEP_V", GATE_MINIMUM_STEP_V),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")

    if TRAP_MINIMUM_STEP_CM3 > TRAP_INITIAL_STEP_CM3:
        raise ValueError("Trap minimum step exceeds the initial step.")
    if GATE_MINIMUM_STEP_V > GATE_INITIAL_STEP_V:
        raise ValueError("Gate minimum step exceeds the initial step.")

    if (
        ELECTRON_PROXY_BARRIER_HEIGHT_EV <= 0.0
        or ELECTRON_PROXY_EFFECTIVE_MASS_RATIO <= 0.0
    ):
        raise ValueError("Electron-proxy FN parameters must be positive.")


validate_parameters()
