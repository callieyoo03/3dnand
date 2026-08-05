"""Unit tests for final state-characterization publication safeguards."""

from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import state_characterization_config as config
from run_state_characterization import (
    FINAL_CONDITION,
    LEGACY_CONDITION,
    build_baseline_comparison_rows,
    expected_idvd_coordinates,
    expected_idvg_coordinates,
    preserve_legacy_shared_data,
    recover_legacy_metrics_from_comparison,
    validate_bias_grid,
)


class BiasGridValidationTests(unittest.TestCase):
    def test_full_grid_sizes(self) -> None:
        idvg = expected_idvg_coordinates(
            config.MEMORY_STATES,
            config.IDVG_VDS_VALUES_V,
            config.IDVG_VGS_STEP_V,
        )
        idvd = expected_idvd_coordinates(
            config.MEMORY_STATES,
            config.IDVD_VGS_VALUES_V,
            config.IDVD_VDS_STEP_V,
        )
        self.assertEqual(len(idvg), 615)
        self.assertEqual(len(idvd), 385)

    def test_duplicate_and_missing_points_are_rejected(self) -> None:
        rows = [
            {
                "state_index": 0,
                "VGS_V": 0.0,
                "VDS_V": 0.05,
                "error_message": "",
            },
            {
                "state_index": 0,
                "VGS_V": 0.0,
                "VDS_V": 0.05,
                "error_message": "",
            },
        ]
        report = validate_bias_grid(
            rows,
            expected_coordinates={(0, 0.0, 0.05), (0, 0.1, 0.05)},
        )
        self.assertFalse(report["valid"])
        self.assertEqual(report["duplicate_count"], 1)
        self.assertEqual(report["missing_count"], 1)


class LegacyComparisonTests(unittest.TestCase):
    @staticmethod
    def _metric_row(vth: float, *, final: bool) -> dict[str, object]:
        return {
            "state_index": 0,
            "state": "State_0_Empty",
            "VDS_V": 0.05,
            "Vth_V": vth,
            "SS_mV_dec": 61.0 if final else 62.0,
            "Ion_A": 2.0e-8 if final else 1.0e-8,
            "Ioff_A": 2.0e-20 if final else 1.0e-20,
            "on_off_ratio": 1.0e12,
            "gm_max_S": 3.0e-8 if final else 2.0e-8,
        }

    def test_comparison_is_keyed_and_signed_final_minus_legacy(self) -> None:
        rows = build_baseline_comparison_rows(
            [self._metric_row(0.5, final=False)],
            [self._metric_row(0.7, final=True)],
        )
        self.assertEqual(len(rows), 1)
        self.assertTrue(
            math.isclose(float(rows[0]["Vth_difference_V"]), 0.2)
        )
        self.assertEqual(rows[0]["legacy_condition"], LEGACY_CONDITION)
        self.assertEqual(rows[0]["final_condition"], FINAL_CONDITION)
        self.assertEqual(
            tuple(rows[0]),
            config.BASELINE_COMPARISON_FIELDNAMES,
        )


class LegacyComparisonRecoveryTests(unittest.TestCase):
    @staticmethod
    def _comparison_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for state in config.MEMORY_STATES:
            for vds_V in config.IDVG_VDS_VALUES_V:
                legacy_vth = (
                    0.5
                    + 0.05 * int(state["state_index"])
                    - 0.1 * float(vds_V)
                )
                final_vth = legacy_vth + 0.01
                rows.append(
                    {
                        "state_index": state["state_index"],
                        "state": state["state"],
                        "VDS_V": vds_V,
                        "legacy_Vth_V": legacy_vth,
                        "final_Vth_V": final_vth,
                        "Vth_difference_V": final_vth - legacy_vth,
                        "legacy_SS_mV_dec": 62.0,
                        "final_SS_mV_dec": 63.0,
                        "legacy_Ion_A": 1.0e-8,
                        "final_Ion_A": 0.9e-8,
                        "legacy_Ioff_A": 1.0e-20,
                        "final_Ioff_A": 1.1e-20,
                        "legacy_on_off_ratio": 1.0e12,
                        "final_on_off_ratio": 0.9e12,
                        "legacy_gm_max_S": 2.0e-8,
                        "final_gm_max_S": 1.9e-8,
                        "legacy_condition": LEGACY_CONDITION,
                        "final_condition": FINAL_CONDITION,
                        "ioff_on_off_warning": (
                            "Ioff and ON/OFF are numerical-leakage-floor "
                            "sensitive; prioritize Vth, delta-Vth, and "
                            "complete IV curves for fitting."
                        ),
                    }
                )
        return rows

    @staticmethod
    def _write_comparison(
        path: Path,
        rows: list[dict[str, object]],
    ) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=config.BASELINE_COMPARISON_FIELDNAMES,
            )
            writer.writeheader()
            writer.writerows(rows)

    def test_final_checkout_falls_back_to_tracked_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            comparison_path = root / "baseline_comparison.csv"
            self._write_comparison(comparison_path, self._comparison_rows())

            source_paths = {
                "IDVG_BY_STATE_CSV_PATH": root / "idvg_by_state.csv",
                "IDVD_BY_STATE_CSV_PATH": root / "idvd_by_state.csv",
                "METRICS_BY_STATE_CSV_PATH": root / "metrics_by_state.csv",
                "MEMORY_STATE_MAP_CSV_PATH": root / "memory_state_map.csv",
            }
            for attribute_name, source_path in source_paths.items():
                source_path.write_text(
                    f"final data for {attribute_name}\n",
                    encoding="utf-8",
                )

            snapshot_directory = root / "ignored_legacy_snapshot"
            expected_hashes = {
                source_path.name: "0" * 64
                for source_path in source_paths.values()
            }
            with patch.multiple(
                config,
                **source_paths,
                BASELINE_COMPARISON_CSV_PATH=comparison_path,
                LEGACY_SNAPSHOT_DIRECTORY=snapshot_directory,
                LEGACY_SHARED_DATA_SHA256=expected_hashes,
            ):
                recovered_rows = preserve_legacy_shared_data()

            self.assertEqual(len(recovered_rows), 15)
            self.assertEqual(recovered_rows[0]["state_index"], 0)
            self.assertEqual(recovered_rows[0]["VDS_V"], 0.01)
            self.assertFalse(snapshot_directory.exists())

    def test_malformed_comparison_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            comparison_path = Path(temporary_directory) / "malformed.csv"
            rows = self._comparison_rows()
            rows[0]["Vth_difference_V"] = 999.0
            self._write_comparison(comparison_path, rows)

            with self.assertRaisesRegex(RuntimeError, "Vth difference mismatch"):
                recover_legacy_metrics_from_comparison(comparison_path)


if __name__ == "__main__":
    unittest.main()
