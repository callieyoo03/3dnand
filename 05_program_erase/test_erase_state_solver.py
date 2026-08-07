"""DEVSIM-free unit tests for ERASE parameters, staging, and rollback."""

from __future__ import annotations

import math
import unittest

import erase_parameters as ep
from erase_state_solver import (
    AdaptiveRampError,
    EraseStateSolver,
    negative_ramp_trial_values,
)


class FakeRuntime:
    device_name = "MoS2_GAA"
    regions = ("CoreOxide", "MoS2", "TunnelOxide", "ChargeTrap", "BlockingOxide")
    semiconductor_region = "MoS2"

    def __init__(self) -> None:
        self.biases = {"gate": 0.0, "drain": 0.0, "source": 0.0}
        self.trap_density = 0.0
        self.node_values = {
            (region, "Potential"): [float(index), float(index) + 0.25]
            for index, region in enumerate(self.regions)
        }
        self.node_values[("MoS2", "Electrons")] = [1.0e17, 1.1e17]
        self.solve_calls = 0
        self.stage_calls: list[str] = []
        self.fail_predicate = lambda runtime: False

    def build_poisson_only(self):
        self.stage_calls.append("build_poisson_only")
        return ep.runtime_geometry_expected_nm()

    def initialize_electrons_from_equilibrium(self) -> None:
        self.stage_calls.append("initialize_electrons")
        self.node_values[("MoS2", "Electrons")] = [2.0e17, 2.0e17]

    def activate_electron_dd(self) -> None:
        self.stage_calls.append("activate_electron_dd")

    def solve_dc(self) -> None:
        self.solve_calls += 1
        # Deliberately contaminate every solution vector before a requested
        # failure so the test proves rollback restores vectors, not just bias.
        if self.fail_predicate(self):
            for key, values in self.node_values.items():
                self.node_values[key] = [value + 99.0 for value in values]
            raise RuntimeError("synthetic Newton failure")

    def get_node_values(self, region: str, name: str):
        return tuple(self.node_values[(region, name)])

    def set_node_values(self, region: str, name: str, values) -> None:
        self.node_values[(region, name)] = list(values)

    def get_terminal_bias(self, terminal: str) -> float:
        return self.biases[terminal]

    def set_terminal_bias(self, terminal: str, voltage_V: float) -> None:
        self.biases[terminal] = float(voltage_V)

    def get_trap_density(self) -> float:
        return self.trap_density

    def set_trap_density(self, density_cm3: float) -> None:
        self.trap_density = float(density_cm3)


class EraseParameterTests(unittest.TestCase):
    def test_sheet_volume_round_trip_and_nominal_pair(self) -> None:
        volume = ep.sheet_to_volume_density(1.0e12)
        self.assertEqual(volume, 2.0e18)
        self.assertEqual(ep.volume_to_sheet_density(volume), 1.0e12)

    def test_trapped_charge_sign(self) -> None:
        self.assertLess(ep.trapped_electron_charge_density_C_cm3(2.0e18), 0.0)
        self.assertEqual(ep.trapped_electron_charge_density_C_cm3(0.0), -0.0)

    def test_negative_gate_direction(self) -> None:
        trials = negative_ramp_trial_values(0.0, -1.0, 0.4)
        self.assertEqual(trials, (-0.4, -0.8, -1.0))
        self.assertTrue(all(b < a for a, b in zip((0.0,) + trials, trials)))
        with self.assertRaisesRegex(ValueError, "negative gate ramp"):
            negative_ramp_trial_values(-1.0, 0.0, 0.1)

    def test_required_model_labels(self) -> None:
        self.assertEqual(ep.MODEL_METADATA["model_class"], "electron_FN_diagnostic")
        self.assertEqual(ep.MODEL_METADATA["physical_role"], "negative_control")
        self.assertEqual(ep.MODEL_METADATA["model_status"], "baseline_negative_result")
        self.assertEqual(ep.MODEL_METADATA["calibration_status"], "uncalibrated")
        self.assertFalse(ep.MODEL_METADATA["predictive_erase_model"])


class EraseStateSolverTests(unittest.TestCase):
    def test_staged_initialization_order(self) -> None:
        runtime = FakeRuntime()
        solver = EraseStateSolver(runtime)
        geometry = solver.initialize_staged()

        self.assertEqual(
            runtime.stage_calls,
            [
                "build_poisson_only",
                "initialize_electrons",
                "activate_electron_dd",
            ],
        )
        self.assertEqual(runtime.solve_calls, 2)
        self.assertEqual(geometry["tunnel_oxide_thickness_nm"], 3.0)
        self.assertEqual(
            solver.last_successful_step.label,
            "empty_trap_zero_bias_electron_dd",
        )

    def test_snapshot_restore_covers_all_mutable_state(self) -> None:
        runtime = FakeRuntime()
        solver = EraseStateSolver(runtime)
        solver.last_successful_step = None
        snapshot = solver.checkpoint()

        for key in runtime.node_values:
            runtime.node_values[key] = [-7.0, -8.0]
        runtime.biases.update(gate=-5.0, drain=1.0, source=2.0)
        runtime.trap_density = 9.0e18
        solver.restore(snapshot)

        for region_index, region in enumerate(runtime.regions):
            self.assertEqual(
                runtime.node_values[(region, "Potential")],
                [float(region_index), float(region_index) + 0.25],
            )
        self.assertEqual(runtime.node_values[("MoS2", "Electrons")], [1.0e17, 1.1e17])
        self.assertEqual(runtime.biases, {"gate": 0.0, "drain": 0.0, "source": 0.0})
        self.assertEqual(runtime.trap_density, 0.0)

    def test_failed_trial_rolls_back_vectors_and_retries_half_step(self) -> None:
        runtime = FakeRuntime()
        solver = EraseStateSolver(runtime)
        pristine = solver.checkpoint()
        failures = {"remaining": 1}

        def fail_first_trial(current: FakeRuntime) -> bool:
            if failures["remaining"] and math.isclose(current.biases["gate"], -0.4):
                failures["remaining"] -= 1
                return True
            return False

        runtime.fail_predicate = fail_first_trial
        result = solver.ramp_gate(-0.4, initial_step_V=0.4, minimum_step_V=0.05)

        self.assertEqual(result.final_value, -0.4)
        self.assertFalse(result.attempts[0].success)
        self.assertTrue(result.attempts[0].rolled_back)
        self.assertIn("synthetic Newton failure", result.attempts[0].exception_traceback)
        self.assertEqual(result.attempts[1].step_size, 0.2)
        # Node solutions remain exactly the pristine vectors after rollback;
        # the fake successful solves do not alter them.
        self.assertEqual(solver.checkpoint().potential_by_region, pristine.potential_by_region)
        self.assertEqual(solver.checkpoint().mos2_electrons_cm3, pristine.mos2_electrons_cm3)

    def test_terminal_failure_restores_last_state_and_preserves_original_cause(self) -> None:
        runtime = FakeRuntime()
        solver = EraseStateSolver(runtime)
        runtime.fail_predicate = lambda current: current.biases["gate"] < 0.0

        with self.assertRaises(AdaptiveRampError) as caught:
            solver.ramp_gate(-0.2, initial_step_V=0.2, minimum_step_V=0.075)

        error = caught.exception
        self.assertEqual(error.last_converged_state.gate_voltage_V, 0.0)
        self.assertEqual(runtime.biases["gate"], 0.0)
        self.assertEqual(runtime.node_values[("MoS2", "Electrons")], [1.0e17, 1.1e17])
        self.assertIsInstance(error.__cause__, RuntimeError)
        self.assertTrue(all(attempt.rolled_back for attempt in error.attempts))

    def test_trap_continuation_reaches_exact_target(self) -> None:
        runtime = FakeRuntime()
        solver = EraseStateSolver(runtime)
        result = solver.ramp_trap_density(
            5.0e17,
            initial_step_cm3=2.0e17,
            minimum_step_cm3=1.0e15,
        )
        self.assertEqual(result.final_value, 5.0e17)
        self.assertEqual(runtime.trap_density, 5.0e17)
        self.assertEqual([a.trial_value for a in result.attempts], [2.0e17, 4.0e17, 5.0e17])


if __name__ == "__main__":
    unittest.main()
