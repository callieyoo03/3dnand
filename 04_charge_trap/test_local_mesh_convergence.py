"""DEVSIM-free tests for local-mesh convergence policy."""

from __future__ import annotations

import copy
import math
import unittest

import local_mesh_convergence as convergence


def _point(
    level: str,
    state_index: int,
    bias_name: str,
    *,
    scale: float = 1.0,
    role: str = "primary",
) -> dict:
    targets = {
        (0, "on"): (3.0, 0.05, 0.0),
        (4, "off"): (-1.0, 0.05, 0.0),
    }
    vgs, vds, vs = targets[(state_index, bias_name)]
    return {
        "mesh_level": level,
        "run_role": role,
        "state_index": state_index,
        "state": "State_0_Empty" if state_index == 0 else "State_4_Programmed",
        "bias_name": bias_name,
        "target_VGS_V": vgs,
        "target_VDS_V": vds,
        "target_VS_V": vs,
        "actual_VGS_V": vgs,
        "actual_VDS_V": vds,
        "actual_VS_V": vs,
        "final_reference_validation_passed": True,
        "topology_passed": True,
        "mesh_line_validation_passed": True,
        "runtime_invariants_passed": True,
        "continuity_passed": True,
        "global_gauss": {"passed": True},
        "derivative_gauss_rows": [
            {"perturbed_terminal": terminal, "passed": True}
            for terminal in ("gate", "drain", "source")
        ],
        "gauge_row_validation": {"passed": True},
        "sensitivities": {
            key: {"value": 0.005, "passed": True}
            for key in ("gate:gate", "gate:drain", "gate:source")
        },
        "ID_A": (1.0e-6 if state_index == 0 else 1.0e-18) * scale,
        "direct": {
            "Qg_contact_C": 3.0e-17 * scale,
            "Qd_contact_C": -1.0e-17 * scale,
            "Qs_contact_C": -2.0e-17 * scale,
        },
        "nominal_matrix_F": {
            "gate:gate": 3.0e-17 * scale,
            "gate:drain": -1.0e-17 * scale,
            "gate:source": -2.0e-17 * scale,
        },
        # Deliberately false: neither is an acceptance criterion here.
        "contact_flux_column_sums_passed": False,
        "ward": {"converged": False},
    }


def _passing_points() -> list[dict]:
    rows = []
    scales = {
        "local_base": 1.03,
        "local_fine": 1.015,
        "local_extra_fine": 1.005,
        "local_ultra_fine": 1.0,
    }
    for level in convergence.MESH_LEVELS:
        for state_index, bias_name in convergence.REFERENCE_STATES:
            rows.append(_point(level, state_index, bias_name, scale=scales[level]))
    for state_index, bias_name in convergence.REFERENCE_STATES:
        rows.append(
            _point(
                "local_ultra_fine", state_index, bias_name,
                scale=1.000001, role="repeat",
            )
        )
    return rows


class CombinedToleranceTests(unittest.TestCase):
    def test_near_zero_uses_explicit_floor_and_absolute_tolerance(self) -> None:
        result = convergence.compare_values(
            0.0,
            0.5e-28,
            floor=convergence.CHARGE_FLOOR_C,
            atol=convergence.CHARGE_ATOL_C,
            rtol=convergence.MESH_CHARGE_RTOL,
        )

        self.assertTrue(result["passed"])
        self.assertFalse(result["near_zero_floor_used"])
        self.assertAlmostEqual(result["allowed_difference"], 1.01e-28)

    def test_near_zero_failure_is_not_hidden_by_relative_scaling(self) -> None:
        result = convergence.compare_values(
            0.0,
            2.0e-28,
            floor=convergence.CHARGE_FLOOR_C,
            atol=convergence.CHARGE_ATOL_C,
            rtol=convergence.MESH_CHARGE_RTOL,
        )

        self.assertFalse(result["passed"])

    def test_floor_is_used_when_both_values_are_below_it(self) -> None:
        result = convergence.compare_values(
            0.0,
            0.0,
            floor=convergence.ID_FLOOR_A,
            atol=convergence.ID_ATOL_A,
            rtol=convergence.MESH_ID_RTOL,
        )

        self.assertTrue(result["near_zero_floor_used"])
        self.assertTrue(result["passed"])

    def test_nonfinite_always_fails(self) -> None:
        result = convergence.compare_values(
            math.nan, 0.0, floor=1.0e-30, atol=1.0e-28, rtol=0.02
        )

        self.assertFalse(result["finite"])
        self.assertFalse(result["passed"])


class ConvergenceRowTests(unittest.TestCase):
    def test_builder_emits_three_pairs_and_ultra_repeat(self) -> None:
        rows = convergence.build_convergence_rows(_passing_points())

        expected = 4 * len(convergence.REFERENCE_STATES) * len(
            convergence.QUANTITY_SPECS
        )
        self.assertEqual(len(rows), expected)
        self.assertEqual(
            {row["comparison_type"] for row in rows}, {"mesh", "repeat"}
        )
        self.assertEqual(
            {row["pair"] for row in rows},
            {
                "local_base->local_fine",
                "local_fine->local_extra_fine",
                "local_extra_fine->local_ultra_fine",
                "local_ultra_fine->local_ultra_fine_repeat",
            },
        )

    def test_non_monotonic_diagnostic_does_not_change_comparison(self) -> None:
        points = _passing_points()
        for point in points:
            if (
                point["run_role"] == "primary"
                and point["state_index"] == 0
                and point["mesh_level"] == "local_fine"
            ):
                point["ID_A"] = 0.99e-6
        rows = convergence.build_convergence_rows(points)
        final_id = next(
            row for row in rows
            if row["pair"] == "local_extra_fine->local_ultra_fine"
            and row["state_index"] == 0
            and row["quantity"] == "ID_A"
        )

        self.assertEqual(final_id["value_monotonicity"], "non_monotonic")
        self.assertTrue(final_id["passed"])


class FinalGateTruthTableTests(unittest.TestCase):
    def _evaluate(self, points: list[dict], quality: dict | None = None) -> dict:
        rows = convergence.build_convergence_rows(points)
        return convergence.evaluate_final_gate(
            points,
            rows,
            quality or {
                "local_extra_fine": True,
                "local_ultra_fine": True,
            },
        )

    def test_all_requested_criteria_pass_but_status_remains_provisional(self) -> None:
        result = self._evaluate(_passing_points())

        self.assertTrue(result["passed"])
        self.assertTrue(result["full_sweep_allowed"])
        self.assertEqual(result["status"], convergence.STATUS_CONVERGED)
        self.assertNotIn("supported", result["status"])

    def test_contact_column_sum_and_ward_dutton_are_not_gates(self) -> None:
        result = self._evaluate(_passing_points())

        self.assertTrue(result["passed"])
        self.assertNotIn("ward", result["criteria"])
        self.assertFalse(any("column" in key for key in result["criteria"]))

    def test_each_top_level_gate_can_block_promotion(self) -> None:
        base_points = _passing_points()
        mutations = {
            "both_ultra_reference_points_valid": lambda points, quality: points[
                6
            ].update(continuity_passed=False),
            "state0_on_id_extra_to_ultra": lambda points, quality: points[
                6
            ].update(ID_A=2.0e-6),
            "both_states_Q_extra_to_ultra": lambda points, quality: points[
                7
            ]["direct"].update(Qg_contact_C=6.0e-17),
            "both_states_supported_C_extra_to_ultra": lambda points, quality: points[
                7
            ]["nominal_matrix_F"].update({"gate:gate": 6.0e-17}),
            "ultra_repeat_reproducibility": lambda points, quality: points[
                8
            ]["direct"].update(Qs_contact_C=-4.0e-17),
            "extra_and_ultra_mesh_quality": lambda points, quality: quality.update(
                local_ultra_fine=False
            ),
        }
        for expected_failed, mutate in mutations.items():
            with self.subTest(criterion=expected_failed):
                points = copy.deepcopy(base_points)
                quality = {
                    "local_extra_fine": True,
                    "local_ultra_fine": True,
                }
                mutate(points, quality)
                result = self._evaluate(points, quality)
                self.assertFalse(result["passed"])
                self.assertFalse(result["criteria"][expected_failed])
                self.assertEqual(result["status"], convergence.STATUS_NOT_CONVERGED)

    def test_supported_delta_sensitivity_limit_is_inclusive(self) -> None:
        point = _point("local_ultra_fine", 0, "on")
        for entry in point["sensitivities"].values():
            entry["value"] = convergence.DELTA_SENSITIVITY_LIMIT

        result = convergence.reference_point_validation(point)

        self.assertTrue(
            result["criteria"]["supported_capacitance_delta_sensitivity"]
        )
        explicit = {
            "supported_delta_sensitivity_max": convergence.DELTA_SENSITIVITY_LIMIT,
            "supported_delta_sensitivity_passed": True,
        }
        self.assertEqual(
            convergence.supported_delta_sensitivity(explicit)["maximum"],
            convergence.DELTA_SENSITIVITY_LIMIT,
        )

    def test_bias_target_uses_absolute_tolerance(self) -> None:
        point = _point("local_ultra_fine", 4, "off")
        point["actual_VGS_V"] += 2.0 * convergence.BIAS_ATOL_V

        self.assertFalse(convergence.bias_target_reached(point))


if __name__ == "__main__":
    unittest.main(verbosity=2)
