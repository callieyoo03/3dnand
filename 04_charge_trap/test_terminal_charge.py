"""DEVSIM-independent unit tests for terminal charge and capacitance helpers."""

from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch

import state_characterization_config as state_config
from terminal_charge import (
    CAPACITANCE_MATRIX_FIELDNAMES,
    CAPACITANCE_SUMMARY_FIELDNAMES,
    TERMINAL_CHARGE_FIELDNAMES,
    adaptive_micro_ramp,
    build_capacitance_matrix_rows,
    build_capacitance_summary_row,
    build_terminal_charge_row,
    calculate_charge_conservation,
    calculate_relative_sensitivity,
    calculate_semiconductor_charge_components,
    central_difference,
    charge_conservation_within_tolerance,
    extract_terminal_charge_components,
    integrate_cylindrical_node_quantity,
    measure_gate_capacitance,
    validate_capacitance_summary_diagnostics,
    validate_devsim_charge_runtime,
)


STATE = {
    "state_index": 4,
    "state": "State_4_Programmed",
    "ntrap_cm3": 2.0e18,
    "nsheet_cm2": 1.0e12,
}


class ReferenceCoverageTests(unittest.TestCase):
    def test_five_memory_states_are_available_for_off_on_references(self) -> None:
        self.assertEqual(len(state_config.MEMORY_STATES), 5)
        self.assertEqual(
            tuple(state["state_index"] for state in state_config.MEMORY_STATES),
            (0, 1, 2, 3, 4),
        )


class CylindricalIntegrationTests(unittest.TestCase):
    def test_integrates_nodal_density_with_supplied_cylindrical_volumes(self) -> None:
        result = integrate_cylindrical_node_quantity(
            [2.0, -3.0, 4.0],
            [0.5, 2.0, 0.25],
        )
        self.assertEqual(result, -4.0)

    def test_rejects_shape_mismatch_and_negative_volume(self) -> None:
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            integrate_cylindrical_node_quantity([1.0], [1.0, 2.0])
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            integrate_cylindrical_node_quantity([1.0], [-1.0])

    def test_separates_mobile_and_fixed_semiconductor_charge(self) -> None:
        mobile, fixed = calculate_semiconductor_charge_components(
            electrons_cm3=[3.0, 4.0],
            holes_cm3=[1.0, 2.0],
            net_doping_cm3=[5.0, 6.0],
            node_volumes_cm3=[2.0, 3.0],
            elementary_charge_C=2.0,
        )
        self.assertEqual(mobile, -20.0)
        self.assertEqual(fixed, 56.0)


class ChargeConservationTests(unittest.TestCase):
    def test_signed_gauss_residual_and_tolerance(self) -> None:
        signed, absolute = calculate_charge_conservation(
            gate_charge_C=5.0,
            drain_boundary_flux_C=-1.0,
            source_boundary_flux_C=-2.0,
            trap_charge_C=-3.0,
            mobile_charge_C=0.5,
            fixed_charge_C=0.5,
        )
        self.assertEqual(signed, 0.0)
        self.assertEqual(absolute, 0.0)

        components = {
            "Qg_C": 5.0,
            "Qd_poisson_boundary_flux_C": -1.0,
            "Qs_poisson_boundary_flux_C": -2.0,
            "Qtrap_C": -3.0,
            "Qmobile_C": 0.5,
            "Qfixed_C": 0.5,
            "charge_sum_C": 1.0e-12,
        }
        self.assertTrue(
            charge_conservation_within_tolerance(
                components,
                absolute_tolerance_C=1.0e-11,
                relative_tolerance=0.0,
            )
        )
        self.assertFalse(
            charge_conservation_within_tolerance(
                components,
                absolute_tolerance_C=1.0e-13,
                relative_tolerance=0.0,
            )
        )


class FiniteDifferenceTests(unittest.TestCase):
    def test_central_difference_is_exact_for_quadratic_charge(self) -> None:
        base = 0.7
        delta = 0.001
        charge = lambda voltage: 2.0 * voltage**2 - 3.0 * voltage + 4.0
        result = central_difference(
            charge(base + delta),
            charge(base - delta),
            delta,
        )
        self.assertAlmostEqual(result, 4.0 * base - 3.0, places=12)

    def test_relative_sensitivity_uses_nominal_value(self) -> None:
        result = calculate_relative_sensitivity(
            {0.0005: 10.1, 0.001: 10.0, 0.002: 9.8},
            nominal_delta_voltage_V=0.001,
        )
        self.assertAlmostEqual(result, 0.02)

    def test_measurement_holds_trap_and_restores_baseline(self) -> None:
        biases = {"gate": 0.7, "drain": 0.05, "source": 0.0}

        devsim_stub = types.ModuleType("devsim")

        def get_parameter(**kwargs):
            if kwargs.get("region") == "ChargeTrap":
                return 2.0e18
            return biases[kwargs["name"].removesuffix("_bias")]

        devsim_stub.get_parameter = get_parameter
        trap_models_stub = types.ModuleType("trap_models")
        trap_models_stub.TRAPPED_ELECTRON_PARAMETER = (
            "TrappedElectronDensityParameter"
        )

        def ramp(terminal, target, delta, **kwargs):
            del delta, kwargs
            biases[terminal] = target
            return target

        def gate_charge():
            return (
                3.0 * biases["gate"]
                - 2.0 * biases["drain"]
                - biases["source"]
            )

        with patch.dict(
            sys.modules,
            {"devsim": devsim_stub, "trap_models": trap_models_stub},
        ):
            for terminal, expected in (
                ("gate", 3.0),
                ("drain", -2.0),
                ("source", -1.0),
            ):
                with self.subTest(terminal=terminal):
                    result = measure_gate_capacitance(
                        perturbed_terminal=terminal,
                        base_voltage_V=biases[terminal],
                        delta_voltage_V=0.001,
                        expected_trap_density_cm3=2.0e18,
                        ramp_function=ramp,
                        gate_charge_function=gate_charge,
                    )
                    self.assertTrue(result["converged"])
                    self.assertAlmostEqual(result["capacitance_F"], expected)
                    self.assertEqual(
                        biases[terminal],
                        {"gate": 0.7, "drain": 0.05, "source": 0.0}[terminal],
                    )


class RuntimeApiTests(unittest.TestCase):
    @staticmethod
    def _runtime_stub() -> types.ModuleType:
        devsim_stub = types.ModuleType("devsim")
        devsim_stub.get_parameter = lambda **kwargs: {
            "node_volume_model": "CylindricalNodeVolume",
            "edge_couple_model": "CylindricalEdgeCouple",
            "raxis_variable": "x",
            "raxis_zero": 0.0,
        }[kwargs["name"]]
        devsim_stub.get_contact_equation_list = lambda **kwargs: (
            "PotentialEquation",
        )
        devsim_stub.get_contact_equation_command = lambda **kwargs: {
            "edge_charge_model": "PotentialEdgeFlux"
        }
        devsim_stub.get_contact_charge = lambda **kwargs: 0.0
        return devsim_stub

    def test_validates_required_axisymmetric_runtime_and_contact_api(self) -> None:
        with patch.dict(sys.modules, {"devsim": self._runtime_stub()}):
            result = validate_devsim_charge_runtime()
        self.assertEqual(result["node_volume_model"], "CylindricalNodeVolume")
        self.assertEqual(result["edge_couple_model"], "CylindricalEdgeCouple")
        self.assertEqual(result["raxis_variable"], "x")

    def test_extracts_contact_and_volume_charge_without_planar_area(self) -> None:
        devsim_stub = self._runtime_stub()
        contact_charges = {"gate": 22.5, "drain": -1.0, "source": -2.0}
        devsim_stub.get_contact_charge = lambda **kwargs: contact_charges[
            kwargs["contact"]
        ]
        node_values = {
            ("MoS2", "CylindricalNodeVolume"): [2.0, 3.0],
            ("MoS2", "Electrons"): [4.0, 5.0],
            ("MoS2", "EquilibriumHoles"): [1.0, 1.0],
            ("MoS2", "NetDoping"): [0.5, 0.5],
            ("ChargeTrap", "TrappedChargeDensity"): [-2.0, -2.0],
            ("ChargeTrap", "CylindricalNodeVolume"): [1.0, 1.0],
        }
        devsim_stub.get_node_model_values = lambda **kwargs: node_values[
            (kwargs["region"], kwargs["name"])
        ]

        with patch.dict(sys.modules, {"devsim": devsim_stub}):
            components = extract_terminal_charge_components(
                elementary_charge_C=1.0
            )

        self.assertEqual(components["Qg_C"], 22.5)
        self.assertTrue(math.isnan(components["Qd_C"]))
        self.assertTrue(math.isnan(components["Qs_C"]))
        self.assertEqual(components["Qd_poisson_boundary_flux_C"], -1.0)
        self.assertEqual(components["Qs_poisson_boundary_flux_C"], -2.0)
        self.assertEqual(components["Qtrap_C"], -4.0)
        self.assertEqual(components["Qmobile_C"], -18.0)
        self.assertEqual(components["Qfixed_C"], 2.5)
        self.assertEqual(components["charge_sum_C"], 0.0)

    def test_micro_ramp_uses_sub_delta_steps_for_source_too(self) -> None:
        biases = {"source": 0.0}
        calls = []
        devsim_stub = types.ModuleType("devsim")
        devsim_stub.get_parameter = lambda **kwargs: biases[
            kwargs["name"].removesuffix("_bias")
        ]
        memory_stub = types.ModuleType("run_memory_window")

        def adaptive_voltage_ramp(**kwargs):
            calls.append(kwargs)
            biases[kwargs["terminal"]] = kwargs["target_voltage"]
            return kwargs["target_voltage"]

        memory_stub.adaptive_voltage_ramp = adaptive_voltage_ramp
        with patch.dict(
            sys.modules,
            {"devsim": devsim_stub, "run_memory_window": memory_stub},
        ):
            reached = adaptive_micro_ramp("source", 0.001, 0.001)

        self.assertEqual(reached, 0.001)
        self.assertEqual(calls[0]["initial_step"], 0.0005)
        self.assertEqual(calls[0]["minimum_step"], 0.001 / 64.0)


class CsvSchemaTests(unittest.TestCase):
    @staticmethod
    def _components() -> dict[str, float]:
        return {
            "Qg_C": 5.0,
            "Qd_C": math.nan,
            "Qs_C": math.nan,
            "Qd_poisson_boundary_flux_C": -1.0,
            "Qs_poisson_boundary_flux_C": -2.0,
            "Qtrap_C": -3.0,
            "Qmobile_C": 0.5,
            "Qfixed_C": 0.5,
            "charge_sum_C": 0.0,
            "charge_conservation_error_C": 0.0,
        }

    def test_terminal_charge_row_matches_schema_and_marks_qd_qs_nan(self) -> None:
        row = build_terminal_charge_row(
            STATE,
            VGS_V=-1.0,
            VDS_V=0.05,
            components=self._components(),
        )
        self.assertEqual(tuple(row), TERMINAL_CHARGE_FIELDNAMES)
        self.assertTrue(math.isnan(row["Qd_C"]))
        self.assertTrue(math.isnan(row["Qs_C"]))
        self.assertTrue(row["converged"])

    def test_matrix_rows_make_only_gate_measurements_numeric(self) -> None:
        deltas = (0.0005, 0.001, 0.002)
        gate_results = {
            (terminal, delta): {
                "capacitance_F": {
                    "gate": 4.0,
                    "drain": -3.0,
                    "source": -1.0,
                }[terminal],
                "converged": True,
                "error_message": "",
            }
            for delta in deltas
            for terminal in ("gate", "drain", "source")
        }
        rows = build_capacitance_matrix_rows(
            STATE,
            VGS_V=3.0,
            VDS_V=0.05,
            gate_results=gate_results,
        )
        self.assertEqual(len(rows), 27)
        self.assertTrue(
            all(tuple(row) == CAPACITANCE_MATRIX_FIELDNAMES for row in rows)
        )
        supported = [row for row in rows if row["supported"]]
        unsupported = [row for row in rows if not row["supported"]]
        self.assertEqual(len(supported), 9)
        self.assertTrue(all(row["measured_terminal"] == "gate" for row in supported))
        self.assertTrue(all(math.isfinite(row["capacitance_F"]) for row in supported))
        self.assertEqual(len(unsupported), 18)
        self.assertTrue(all(math.isnan(row["capacitance_F"]) for row in unsupported))

    def test_summary_selects_one_millivolt_and_reports_gate_row_sum(self) -> None:
        deltas = (0.0005, 0.001, 0.002)
        nominal = {"gate": 4.0, "drain": -3.0, "source": -1.0}
        gate_results = {
            (terminal, delta): {
                "capacitance_F": nominal[terminal],
                "converged": True,
                "error_message": "",
            }
            for delta in deltas
            for terminal in nominal
        }
        row = build_capacitance_summary_row(
            STATE,
            VGS_V=3.0,
            VDS_V=0.05,
            gate_results=gate_results,
            nominal_delta_voltage_V=0.001,
            sensitivity_delta_voltages_V=deltas,
        )
        self.assertEqual(tuple(row), CAPACITANCE_SUMMARY_FIELDNAMES)
        self.assertEqual(row["Cgg_F"], 4.0)
        self.assertEqual(row["Cgd_F"], -3.0)
        self.assertEqual(row["Cgs_F"], -1.0)
        self.assertEqual(row["gate_capacitance_row_sum_F"], 0.0)
        self.assertEqual(row["Cgg_relative_sensitivity"], 0.0)
        self.assertTrue(row["converged"])

        report = validate_capacitance_summary_diagnostics([row])
        self.assertEqual(report["maximum_relative_sensitivity"], 0.0)
        self.assertEqual(report["maximum_gate_row_sum_F"], 0.0)

        unstable = dict(row)
        unstable["Cgg_relative_sensitivity"] = 0.02
        with self.assertRaisesRegex(RuntimeError, "sensitivity exceeds"):
            validate_capacitance_summary_diagnostics([unstable])

        nonconservative = dict(row)
        nonconservative["gate_capacitance_row_sum_F"] = 2.0e-22
        with self.assertRaisesRegex(RuntimeError, "row-sum residual"):
            validate_capacitance_summary_diagnostics([nonconservative])


if __name__ == "__main__":
    unittest.main(verbosity=2)
