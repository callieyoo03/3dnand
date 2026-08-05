"""Generate and validate the complete static compact-model handoff package.

Every DEVSIM workload runs in a fresh Python process because this repository
uses fixed global mesh/device names.  No ERASE or retention runner is imported
or executed by this orchestrator.
"""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from handoff_metadata import (
    MANIFEST_SELF_HASH_POLICY,
    validate_manifest,
    write_csv_rows,
    write_manifest,
)


MODULE_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_DIRECTORY.parent
SHARED_DATA_DIRECTORY = REPOSITORY_ROOT / "shared_data"
MANIFEST_PATH = SHARED_DATA_DIRECTORY / "handoff_manifest.csv"
REPRODUCIBILITY_REPORT_PATH = (
    SHARED_DATA_DIRECTORY / "reproducibility_report.csv"
)

DC_REPRODUCIBILITY_FILES = (
    "idvg_by_state.csv",
    "idvd_by_state.csv",
    "metrics_by_state.csv",
    "memory_state_map.csv",
    "baseline_comparison_legacy_vs_final.csv",
)

CHARGE_CAPACITANCE_REPRODUCIBILITY_FILES = (
    "terminal_charge_by_state.csv",
    "capacitance_matrix_by_state.csv",
    "capacitance_summary_by_state.csv",
)

REPRODUCIBILITY_FILES = (
    *DC_REPRODUCIBILITY_FILES,
    *CHARGE_CAPACITANCE_REPRODUCIBILITY_FILES,
)

REPRODUCIBILITY_FIELDNAMES = (
    "file",
    "row_count",
    "numeric_cell_count",
    "exact_text_cell_count",
    "max_absolute_difference",
    "max_relative_difference",
    "max_normalized_error",
    "relative_tolerance",
    "absolute_tolerance",
    "passed",
    "notes",
)

REPRODUCIBILITY_RELATIVE_TOLERANCE = 1.0e-4
DC_REPRODUCIBILITY_ABSOLUTE_TOLERANCE = 1.0e-18
CHARGE_CAPACITANCE_ABSOLUTE_TOLERANCE = 1.0e-28

# These two derived diagnostics amplify roundoff by division or cancellation.
# Primary Q/C values retain the tight 1e-28 C/F floor above.  The overrides
# remain much tighter than their independent acceptance limits and are recorded
# in the report notes rather than being applied to any terminal charge or
# capacitance value.
REPRODUCIBILITY_FIELD_ABSOLUTE_TOLERANCES = {
    ("capacitance_summary_by_state.csv", "Cgg_relative_sensitivity"): 1.0e-8,
    ("capacitance_summary_by_state.csv", "Cgd_relative_sensitivity"): 1.0e-8,
    ("capacitance_summary_by_state.csv", "Cgs_relative_sensitivity"): 1.0e-8,
    ("capacitance_summary_by_state.csv", "gate_capacitance_row_sum_F"): 1.0e-23,
}


def run_python_script(script_name: str, *arguments: str) -> None:
    """Run one repository script with the current DEVSIM-capable Python."""

    command = [
        sys.executable,
        str(MODULE_DIRECTORY / script_name),
        *arguments,
    ]
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPOSITORY_ROOT, check=True)


def read_reproducibility_bundle(
) -> dict[str, tuple[tuple[str, ...], list[dict[str, str]]]]:
    """Read the public DC and Q/C CSVs before a fresh repeat process."""

    bundle: dict[str, tuple[tuple[str, ...], list[dict[str, str]]]] = {}
    for name in REPRODUCIBILITY_FILES:
        path = SHARED_DATA_DIRECTORY / name
        with path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            header = tuple(reader.fieldnames or ())
            bundle[name] = (header, list(reader))
    return bundle


def _numeric_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compare_dc_bundles(
    before: dict[str, tuple[tuple[str, ...], list[dict[str, str]]]],
    after: dict[str, tuple[tuple[str, ...], list[dict[str, str]]]],
    *,
    relative_tolerance: float = REPRODUCIBILITY_RELATIVE_TOLERANCE,
    dc_absolute_tolerance: float = DC_REPRODUCIBILITY_ABSOLUTE_TOLERANCE,
    charge_capacitance_absolute_tolerance: float = (
        CHARGE_CAPACITANCE_ABSOLUTE_TOLERANCE
    ),
) -> list[dict[str, object]]:
    """Compare every DC and Q/C CSV cell with quantity-scaled tolerances."""

    reports: list[dict[str, object]] = []
    for name in REPRODUCIBILITY_FILES:
        absolute_tolerance = (
            charge_capacitance_absolute_tolerance
            if name in CHARGE_CAPACITANCE_REPRODUCIBILITY_FILES
            else dc_absolute_tolerance
        )
        before_header, before_rows = before[name]
        after_header, after_rows = after[name]
        passed = before_header == after_header and len(before_rows) == len(after_rows)
        numeric_cells = 0
        exact_text_cells = 0
        max_absolute = 0.0
        max_relative = 0.0
        max_normalized = 0.0
        notes: list[str] = []
        override_fields = sorted(
            field
            for (override_file, field), _ in (
                REPRODUCIBILITY_FIELD_ABSOLUTE_TOLERANCES.items()
            )
            if override_file == name
        )
        if override_fields:
            notes.append(
                "derived-diagnostic absolute overrides: "
                + ",".join(override_fields)
            )

        if before_header != after_header:
            notes.append("header mismatch")
        if len(before_rows) != len(after_rows):
            notes.append(
                f"row-count mismatch {len(before_rows)} != {len(after_rows)}"
            )

        for row_index, (left_row, right_row) in enumerate(
            zip(before_rows, after_rows)
        ):
            for field in before_header:
                left_text = left_row[field]
                right_text = right_row[field]
                left = _numeric_or_none(left_text)
                right = _numeric_or_none(right_text)

                if left is None or right is None:
                    exact_text_cells += 1
                    if left_text != right_text:
                        passed = False
                        notes.append(
                            f"text mismatch row={row_index} field={field}"
                        )
                    continue

                numeric_cells += 1
                if math.isnan(left) or math.isnan(right):
                    if not (math.isnan(left) and math.isnan(right)):
                        passed = False
                        notes.append(
                            f"NaN mismatch row={row_index} field={field}"
                        )
                    continue

                absolute_difference = abs(right - left)
                scale = max(abs(left), abs(right))
                relative_difference = (
                    absolute_difference / scale if scale > 0.0 else 0.0
                )
                cell_absolute_tolerance = (
                    REPRODUCIBILITY_FIELD_ABSOLUTE_TOLERANCES.get(
                        (name, field),
                        absolute_tolerance,
                    )
                )
                allowed = (
                    cell_absolute_tolerance + relative_tolerance * scale
                )
                normalized = absolute_difference / allowed if allowed > 0.0 else 0.0
                max_absolute = max(max_absolute, absolute_difference)
                max_relative = max(max_relative, relative_difference)
                max_normalized = max(max_normalized, normalized)
                if absolute_difference > allowed:
                    passed = False
                    notes.append(
                        f"numeric mismatch row={row_index} field={field}"
                    )

        reports.append(
            {
                "file": name,
                "row_count": len(after_rows),
                "numeric_cell_count": numeric_cells,
                "exact_text_cell_count": exact_text_cells,
                "max_absolute_difference": max_absolute,
                "max_relative_difference": max_relative,
                "max_normalized_error": max_normalized,
                "relative_tolerance": relative_tolerance,
                "absolute_tolerance": absolute_tolerance,
                "passed": passed,
                "notes": "; ".join(notes[:10]),
            }
        )

    if not all(bool(report["passed"]) for report in reports):
        raise RuntimeError(
            f"DC/QC reproducibility comparison failed: {reports}"
        )
    return reports


def write_reproducibility_report(
    reports: list[dict[str, object]],
) -> Path:
    return write_csv_rows(
        REPRODUCIBILITY_REPORT_PATH,
        REPRODUCIBILITY_FIELDNAMES,
        reports,
    )


def collect_manifest_artifacts() -> tuple[Path, ...]:
    """Collect every delivered shared-data file except the manifest itself."""

    artifacts = tuple(
        sorted(
            (
                path
                for path in SHARED_DATA_DIRECTORY.rglob("*")
                if path.is_file() and path.resolve() != MANIFEST_PATH.resolve()
            ),
            key=lambda path: path.relative_to(SHARED_DATA_DIRECTORY).as_posix(),
        )
    )
    if not artifacts:
        raise RuntimeError("No handoff artifacts were found.")
    return artifacts


def generate_manifest() -> dict[str, object]:
    """Write and independently validate the deterministic SHA-256 manifest."""

    artifacts = collect_manifest_artifacts()
    write_manifest(
        MANIFEST_PATH,
        SHARED_DATA_DIRECTORY,
        artifacts,
        generated_by="04_charge_trap/run_final_handoff.py",
    )
    return validate_manifest(
        MANIFEST_PATH,
        root_directory=SHARED_DATA_DIRECTORY,
        expected_files=artifacts,
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate final 3/5/16-nm IV, Q/C, metadata, mesh/VTK, and "
            "SHA-256 handoff artifacts without ERASE or retention simulation."
        )
    )
    parser.add_argument(
        "--reproducibility-check",
        action="store_true",
        help=(
            "Run the full DC and Q/C characterization twice and require "
            "cell-by-cell numeric equivalence within the documented tolerances."
        ),
    )
    parser.add_argument(
        "--reproducibility-only",
        action="store_true",
        help=(
            "Use the existing public DC and Q/C files as the first sample, "
            "run one fresh repeat, and write reproducibility_report.csv."
        ),
    )
    parser.add_argument(
        "--skip-dc",
        action="store_true",
        help="Keep existing final IV/metric files.",
    )
    parser.add_argument(
        "--skip-charge",
        action="store_true",
        help="Keep existing terminal-charge/capacitance files.",
    )
    parser.add_argument(
        "--skip-mesh",
        action="store_true",
        help="Keep existing mesh/VTK and metadata files.",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Only rebuild and validate the manifest from existing artifacts.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_argument_parser().parse_args(argv)

    if arguments.reproducibility_check and arguments.reproducibility_only:
        raise SystemExit(
            "--reproducibility-check and --reproducibility-only are mutually exclusive"
        )
    if (
        arguments.reproducibility_check or arguments.reproducibility_only
    ) and (arguments.skip_dc or arguments.skip_charge):
        raise SystemExit(
            "reproducibility modes require both DC and Q/C generation; "
            "do not use --skip-dc or --skip-charge"
        )

    if not arguments.manifest_only:
        if arguments.reproducibility_only:
            first_bundle = read_reproducibility_bundle()
            run_python_script("run_state_characterization.py", "--all")
            run_python_script("run_terminal_charge_by_state.py")
            run_python_script("run_capacitance_by_state.py")
            reports = compare_dc_bundles(
                first_bundle,
                read_reproducibility_bundle(),
            )
            write_reproducibility_report(reports)
            print("DC/QC numeric reproducibility comparison passed.")
        else:
            if not arguments.skip_dc:
                run_python_script("run_state_characterization.py", "--all")

            if not arguments.skip_charge:
                run_python_script("run_terminal_charge_by_state.py")
                run_python_script("run_capacitance_by_state.py")

            if arguments.reproducibility_check:
                first_bundle = read_reproducibility_bundle()
                run_python_script("run_state_characterization.py", "--all")
                run_python_script("run_terminal_charge_by_state.py")
                run_python_script("run_capacitance_by_state.py")
                reports = compare_dc_bundles(
                    first_bundle,
                    read_reproducibility_bundle(),
                )
                write_reproducibility_report(reports)
                print("DC/QC numeric reproducibility comparison passed.")

        if not arguments.skip_mesh:
            run_python_script("export_handoff_mesh.py")

    report = generate_manifest()
    print(
        f"Validated {report['artifact_count']} manifest entries. "
        f"{MANIFEST_SELF_HASH_POLICY}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
