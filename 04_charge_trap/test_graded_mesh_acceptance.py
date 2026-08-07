"""Pure tests for candidate-only split transport/electrostatic acceptance."""

from __future__ import annotations

import copy
import math
import unittest

import graded_mesh_acceptance as acceptance


def _point(
    level: str,
    state: tuple[int, str],
    *,
    role: str = "primary",
    id_A: float | None = None,
) -> dict:
    state_index, bias_name = state
    if id_A is None:
        id_A = 1.0e-6 if state == acceptance.STATE0_ON else 2.0e-24
    if state == acceptance.STATE0_ON:
        source = -id_A
    else:
        # Deliberately fails the old strict continuity gate while remaining
        # below the independently derived quantification floor.
        source = 3.0e-24
    return {
        "mesh_level": level,
        "run_role": role,
        "state_index": state_index,
        "bias_name": bias_name,
        "ID_A": id_A,
        "drain_current_A": id_A,
        "source_current_A": source,
        "Qg_C": 1.0e-17,
        "raw_Qd_C": -4.0e-18,
        "raw_Qs_C": -5.0e-18,
        "Cgg_F": 2.0e-17,
        "Cgd_F": -1.2e-17,
        "Cgs_F": -8.0e-18,
        "target_reached": True,
        "final_reference_validation_passed": True,
        "topology_passed": True,
        "mesh_line_validation_passed": True,
        "runtime_invariants_passed": True,
        "global_gauss_passed": True,
        "derivative_gauss_passed": True,
        "gauge_row_sum_passed": True,
        "supported_delta_sensitivity_max": 0.005,
    }


def _points() -> list[dict]:
    result = []
    level_scale = {
        "graded_base": 1.03,
        "graded_fine": 1.02,
        acceptance.EXTRA_FINE: 1.01,
        acceptance.ULTRA_FINE: 1.0,
    }
    for level in acceptance.GRADED_LEVELS:
        for state in acceptance.REFERENCE_STATES:
            scale = level_scale[level]
            id_A = (1.0e-6 * scale) if state == acceptance.STATE0_ON else 2.0e-24
            result.append(_point(level, state, id_A=id_A))
    for state in acceptance.REFERENCE_STATES:
        id_A = 1.00000001e-6 if state == acceptance.STATE0_ON else -2.0e-24
        result.append(_point(acceptance.ULTRA_FINE, state, role="repeat", id_A=id_A))
    return result


def _comparison_row(
    comparison_type: str,
    state: tuple[int, str],
    quantity: str,
    reference: float,
    candidate: float,
) -> dict:
    if comparison_type == "mesh":
        reference_mesh = acceptance.EXTRA_FINE
        candidate_mesh = acceptance.ULTRA_FINE
    else:
        reference_mesh = acceptance.ULTRA_FINE
        candidate_mesh = f"{acceptance.ULTRA_FINE}_repeat"
    return {
        "comparison_type": comparison_type,
        "reference_mesh": reference_mesh,
        "candidate_mesh": candidate_mesh,
        "state_index": state[0],
        "bias_name": state[1],
        "quantity": quantity,
        "reference_value": reference,
        "candidate_value": candidate,
        # These are deliberately ignored and recomputed by the new policy.
        "passed": False,
        "acceptance_gate_applies": False,
    }


def _comparisons() -> list[dict]:
    rows: list[dict] = []
    quantities = {
        "Qg_C": 1.0e-17,
        "raw_Qd_C": -4.0e-18,
        "raw_Qs_C": -5.0e-18,
        "Cgg_F": 2.0e-17,
        "Cgd_F": -1.2e-17,
        "Cgs_F": -8.0e-18,
    }
    rows.append(_comparison_row(
        "mesh", acceptance.STATE0_ON, "ID_A", 1.01e-6, 1.0e-6
    ))
    rows.append(_comparison_row(
        "repeat", acceptance.STATE0_ON, "ID_A", 1.0e-6, 1.00000001e-6
    ))
    # State-4 ID sign flip is present as a diagnostic but is not a DC or Q/C
    # acceptance input below the quantification floor.
    rows.append(_comparison_row(
        "repeat", acceptance.STATE4_OFF, "ID_A", 2.0e-24, -2.0e-24
    ))
    for state in acceptance.REFERENCE_STATES:
        for name, value in quantities.items():
            rows.append(_comparison_row("mesh", state, name, 1.01 * value, value))
            rows.append(_comparison_row("repeat", state, name, value, value))
    return rows


def _quality(passed: bool = True) -> dict[str, bool]:
    return {level: passed for level in acceptance.GRADED_LEVELS}


class CurrentClassificationTests(unittest.TestCase):
    def test_floor_is_derived_not_fitted(self) -> None:
        self.assertEqual(
            acceptance.DC_CURRENT_QUANTIFICATION_FLOOR_A,
            acceptance.ID_ATOL_A / acceptance.ION_MESH_RTOL,
        )
        self.assertTrue(math.isclose(
            acceptance.DC_CURRENT_QUANTIFICATION_FLOOR_A,
            5.0e-23,
            rel_tol=1.0e-15,
            abs_tol=0.0,
        ))

    def test_below_floor_preserves_raw_and_reports_nan(self) -> None:
        point = _point(
            acceptance.ULTRA_FINE, acceptance.STATE4_OFF, id_A=-2.0e-24
        )
        original = copy.deepcopy(point)

        result = acceptance.classify_current_point(point)

        self.assertEqual(point, original)
        self.assertEqual(result["raw_ID_A"], -2.0e-24)
        self.assertEqual(result["raw_source_current_A"], 3.0e-24)
        self.assertTrue(math.isnan(result["reported_ID_A"]))
        self.assertTrue(math.isnan(result["reported_Ioff_A"]))
        self.assertEqual(
            result["dc_current_status"], acceptance.CURRENT_STATUS_BELOW_FLOOR
        )
        self.assertFalse(result["continuity_gate_applicable"])
        self.assertEqual(
            result["dc_continuity_acceptance_status"],
            "not_applicable_below_numerical_floor",
        )

    def test_floor_boundary_is_quantified(self) -> None:
        floor = acceptance.DC_CURRENT_QUANTIFICATION_FLOOR_A
        point = _point(
            acceptance.ULTRA_FINE, acceptance.STATE0_ON, id_A=floor
        )

        result = acceptance.classify_current_point(point)

        self.assertEqual(result["dc_current_status"], acceptance.CURRENT_STATUS_QUANTIFIED)
        self.assertEqual(result["reported_ID_A"], floor)

    def test_nonfinite_is_fail_closed(self) -> None:
        point = _point(acceptance.ULTRA_FINE, acceptance.STATE0_ON)
        point["source_current_A"] = math.nan

        result = acceptance.classify_current_point(point)

        self.assertEqual(result["dc_current_status"], acceptance.CURRENT_STATUS_NONFINITE)
        self.assertTrue(math.isnan(result["reported_ID_A"]))

    def test_stable_scalar_and_annotation_apis(self) -> None:
        scalar = acceptance.classify_current(
            1.0e-6, source_current_A=-1.0e-6,
            state_index=0, bias_name="on",
        )
        local = _point("local_ultra_fine", acceptance.STATE0_ON)
        annotated = acceptance.annotate_reference_point(local)

        self.assertEqual(scalar["dc_current_status"], acceptance.CURRENT_STATUS_QUANTIFIED)
        self.assertEqual(annotated["mesh_level"], acceptance.ULTRA_FINE)
        self.assertTrue(annotated["dc_transport_point_passed"])
        self.assertTrue(annotated["electrostatic_qc_point_passed"])


class SplitAcceptanceTests(unittest.TestCase):
    def _evaluate(
        self,
        *,
        points: list[dict] | None = None,
        comparisons: list[dict] | None = None,
        quality: dict | None = None,
        profile: bool = True,
    ) -> dict:
        return acceptance.evaluate_split_acceptance(
            points or _points(),
            comparisons or _comparisons(),
            quality or _quality(),
            profile_coverage_passed=profile,
        )

    def test_all_domains_pass_with_unquantified_state4_off(self) -> None:
        result = self._evaluate()

        self.assertTrue(result["dc_transport"]["passed"])
        self.assertIn("ioff_below_numerical_floor", result["dc_transport"]["status"])
        self.assertTrue(result["electrostatic_qc"]["passed"])
        self.assertTrue(result["combined_reference"]["full_sweep_allowed"])
        self.assertEqual(
            result["combined_reference"]["physical_contact_flux_status"],
            acceptance.PHYSICAL_STATUS_CONVERGED,
        )

    def test_state4_sign_flip_and_failed_relative_continuity_do_not_gate_qc(self) -> None:
        result = self._evaluate()
        off = [
            point for point in result["classified_points"]
            if int(point["state_index"]) == 4
        ]

        self.assertTrue(any(point["raw_ID_A"] < 0.0 for point in off))
        self.assertTrue(any(not point["raw_continuity_passed"] for point in off))
        self.assertTrue(result["electrostatic_qc"]["passed"])
        off_id_gate_rows = [
            row for row in result["criterion_rows"]
            if row["criterion"] == "state4_off_current_quantification"
        ]
        self.assertEqual(len(off_id_gate_rows), 3)
        self.assertTrue(all(not row["gate_applicable"] for row in off_id_gate_rows))

    def test_on_continuity_failure_blocks_dc_only(self) -> None:
        points = _points()
        point = next(
            row for row in points
            if row["mesh_level"] == acceptance.ULTRA_FINE
            and row["run_role"] == "primary" and int(row["state_index"]) == 0
        )
        point["source_current_A"] = point["drain_current_A"]

        result = self._evaluate(points=points)

        self.assertFalse(result["dc_transport"]["passed"])
        self.assertTrue(result["electrostatic_qc"]["passed"])
        self.assertFalse(result["combined_reference"]["full_sweep_allowed"])

    def test_on_repeat_failure_blocks_dc_only(self) -> None:
        rows = _comparisons()
        repeat = next(
            row for row in rows
            if row["comparison_type"] == "repeat"
            and int(row["state_index"]) == 0 and row["quantity"] == "ID_A"
        )
        repeat["candidate_value"] = 1.1e-6

        result = self._evaluate(comparisons=rows)

        self.assertFalse(result["dc_transport"]["passed"])
        self.assertTrue(result["electrostatic_qc"]["passed"])

    def test_q_failure_blocks_electrostatic_only(self) -> None:
        rows = _comparisons()
        qrow = next(
            row for row in rows
            if row["comparison_type"] == "mesh"
            and int(row["state_index"]) == 4 and row["quantity"] == "Qg_C"
        )
        qrow["candidate_value"] *= 0.5

        result = self._evaluate(comparisons=rows)

        self.assertTrue(result["dc_transport"]["passed"])
        self.assertFalse(result["electrostatic_qc"]["passed"])
        self.assertFalse(result["combined_reference"]["passed"])

    def test_delta_failure_blocks_electrostatic_only(self) -> None:
        points = _points()
        next(
            point for point in points
            if point["mesh_level"] == acceptance.EXTRA_FINE
            and point["state_index"] == acceptance.STATE0_ON[0]
        )["supported_delta_sensitivity_max"] = 0.0100001

        result = self._evaluate(points=points)

        self.assertTrue(result["dc_transport"]["passed"])
        self.assertFalse(result["electrostatic_qc"]["passed"])

    def test_quality_failure_blocks_both_and_full_sweep(self) -> None:
        result = self._evaluate(quality=_quality(False))

        self.assertFalse(result["dc_transport"]["passed"])
        self.assertFalse(result["electrostatic_qc"]["passed"])
        self.assertFalse(result["combined_reference"]["full_sweep_allowed"])
        self.assertEqual(
            result["combined_reference"]["physical_contact_flux_status"],
            acceptance.PHYSICAL_STATUS_NOT_CONVERGED,
        )

    def test_profile_coverage_is_not_a_combined_authorization_gate(self) -> None:
        result = self._evaluate(profile=False)

        self.assertTrue(result["dc_transport"]["passed"])
        self.assertTrue(result["electrostatic_qc"]["passed"])
        self.assertTrue(result["combined_reference"]["passed"])
        self.assertTrue(result["combined_reference"]["full_sweep_allowed"])

    def test_missing_q_comparison_fails_closed(self) -> None:
        rows = [
            row for row in _comparisons()
            if not (
                row["comparison_type"] == "repeat"
                and int(row["state_index"]) == 4
                and row["quantity"] == "raw_Qs_C"
            )
        ]

        result = self._evaluate(comparisons=rows)

        self.assertFalse(result["electrostatic_qc"]["passed"])

    def test_criterion_rows_are_flat_and_have_domain_summaries(self) -> None:
        result = self._evaluate()
        rows = result["criterion_rows"]
        required = {
            "row_type", "criterion_order", "acceptance_domain", "criterion",
            "scope", "gate_applicable", "passed", "status", "observed_value",
            "limit_value", "units", "reason",
        }

        self.assertTrue(rows)
        self.assertTrue(all(set(row) == required for row in rows))
        summaries = {
            row["criterion"] for row in rows
            if row["acceptance_domain"] == "combined_reference"
        }
        self.assertEqual(
            summaries,
            {
                "dc_transport_gate", "electrostatic_qc_gate",
                "profile_coverage_not_in_scope", "conditional_full_sweep_gate",
                "physical_contact_flux_status",
            },
        )

    def test_quality_rows_extract_fail_closed(self) -> None:
        rows = [
            {"row_type": "family_quality_summary", "mesh_level": acceptance.EXTRA_FINE,
             "quality_passed": "True"},
            {"row_type": "graded_quality_summary", "mesh_level": acceptance.ULTRA_FINE,
             "quality_passed": "False"},
        ]

        result = acceptance.quality_mapping_from_rows(rows)

        self.assertEqual(result, {acceptance.EXTRA_FINE: True, acceptance.ULTRA_FINE: False})

    def test_runner_facing_build_evaluate_and_criterion_apis(self) -> None:
        rows = acceptance.build_graded_convergence_rows(_points())
        gate = acceptance.evaluate_reference_gates(
            _points(), rows, _quality(), profile_passed=True
        )
        copied = acceptance.criterion_rows(gate)

        self.assertEqual(len(rows), 56)
        self.assertEqual(
            {row["reference_mesh"] for row in rows},
            set(acceptance.GRADED_LEVELS),
        )
        off_id = [
            row for row in rows
            if int(row["state_index"]) == 4 and row["quantity"] == "ID_A"
        ]
        self.assertEqual(len(off_id), 4)
        self.assertTrue(all(not row["gate_applicable"] for row in off_id))
        self.assertTrue(all(
            row["comparison_status"] == "not_applicable_below_numerical_floor"
            for row in off_id
        ))
        self.assertTrue(gate["combined_reference"]["passed"])
        self.assertEqual(copied, gate["criterion_rows"])
        self.assertIsNot(copied[0], gate["criterion_rows"][0])

    def test_local_quality_aliases_are_accepted(self) -> None:
        gate = acceptance.evaluate_reference_gates(
            _points(), _comparisons(),
            {
                "local_base": True,
                "local_fine": True,
                "local_extra_fine": True,
                "local_ultra_fine": True,
            },
        )

        self.assertTrue(gate["combined_reference"]["passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
