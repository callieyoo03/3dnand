"""Synthetic-only tests for the localized electron-trap emission framework."""

import math
import unittest
from dataclasses import replace

import electron_trap_emission as emission


class _FakeDevsim:
    def __init__(self):
        self.values = {
            "TunnelOxide": {
                "x": (1.2e-6, 1.3e-6, 1.2e-6, 1.3e-6),
                "y": (5.0e-6, 5.0e-6, 6.0e-6, 6.0e-6),
                "Potential": (0.0, -0.1, 0.0, -0.2),
            },
            "ChargeTrap": {
                "x": (1.3e-6, 1.5e-6, 1.3e-6, 1.5e-6),
                "y": (5.0e-6, 5.0e-6, 6.0e-6, 6.0e-6),
                "Potential": (-0.1, -0.2, -0.2, -0.4),
            },
            "BlockingOxide": {
                "x": (1.5e-6, 1.7e-6, 1.5e-6, 1.7e-6),
                "y": (5.0e-6, 5.0e-6, 6.0e-6, 6.0e-6),
                "Potential": (-0.2, -0.4, -0.4, -0.8),
            },
        }

    def get_node_model_values(self, *, device, region, name):
        self.last_device = device
        return self.values[region][name]


class ElectronTrapEmissionTests(unittest.TestCase):
    def setUp(self):
        self.profile = emission.build_radial_potential_profile(
            radii_cm=(1.2e-6, 1.3e-6, 1.5e-6, 1.7e-6),
            potentials_V=(0.0, 0.0, 0.0, 0.0),
            interval_materials=("Al2O3", "HfO2", "Al2O3"),
            axial_position_cm=5.0e-6,
            source="synthetic unit-test profile",
        )
        self.parameters = emission.ElectronTrapEmissionParameters(
            attempt_frequency_Hz=1.0e13,
            trap_energy_reference=emission.SUPPORTED_TRAP_ENERGY_REFERENCE,
            band_edge_reference="synthetic common vacuum-like reference",
            conduction_band_edge_eV_by_material={"Al2O3": 3.0, "HfO2": 2.0},
            electron_effective_mass_ratio_by_material={"Al2O3": 0.30, "HfO2": 0.20},
            energy_bins=(emission.TrapEnergyBin(1.0),),
            radial_position_bins=(emission.RadialTrapPositionBin(1.4e-6, material="HfO2"),),
            spatial_distribution=emission.SpatialDistribution.SINGLE_POSITION,
            stable_trap_fraction=0.0,
            attempt_partition=emission.AttemptFrequencyPartition(
                0.5, 0.5, "synthetic symmetric attempts"
            ),
            provenance=emission.ParameterProvenance(
                emission.ProvenanceKind.SYNTHETIC_TEST,
                "test_electron_trap_emission.py synthetic values",
            ),
        )
        self.config = emission.TrapEmissionConfiguration(
            emission.TrapEmissionMode.SENSITIVITY_ONLY, self.parameters
        )

    def test_default_mode_is_disabled_and_explicitly_unsupported(self):
        config = emission.TrapEmissionConfiguration()
        self.assertEqual(emission.TRAP_EMISSION_MODEL_MODE, "disabled")
        result = emission.evaluate_localized_trap_emission(config)
        self.assertFalse(result.supported)
        self.assertEqual(result.model_status, "disabled_unsupported")
        self.assertIsNone(result.gamma_channel_s1)
        self.assertIsNone(result.gamma_gate_s1)
        self.assertIsNone(result.gamma_total_s1)
        self.assertFalse(result.predictive_erase_model)
        self.assertFalse(result.hole_assisted_model_implemented)

    def test_all_required_modes_exist(self):
        self.assertEqual(
            {mode.value for mode in emission.TrapEmissionMode},
            {"disabled", "sensitivity_only", "user_supplied", "literature_parameter_set"},
        )

    def test_missing_parameters_fail_fast_without_defaults(self):
        config = emission.TrapEmissionConfiguration(
            emission.TrapEmissionMode.SENSITIVITY_ONLY,
            emission.ElectronTrapEmissionParameters(),
        )
        with self.assertRaises(emission.MissingEmissionParameterError) as context:
            emission.evaluate_localized_trap_emission(config, self.profile)
        missing = context.exception.missing_parameters
        self.assertIn("attempt_frequency_Hz", missing)
        self.assertIn("HfO2 electron trap energy/depth", missing)
        self.assertIn("Al2O3 electron tunnelling effective mass", missing)
        self.assertNotIn("stable/residual trap fraction", missing)

    def test_stable_fraction_is_audited_but_not_required_by_zero_floor_rate(self):
        parameters = replace(self.parameters, stable_trap_fraction=None)
        result = emission.evaluate_localized_trap_emission(
            emission.TrapEmissionConfiguration(
                emission.TrapEmissionMode.SENSITIVITY_ONLY, parameters
            ),
            self.profile,
        )
        self.assertTrue(result.supported)
        self.assertLess(
            emission.trap_density_derivative_cm3_s(2.0e18, result.gamma_total_s1),
            0.0,
        )

    def test_user_and_literature_modes_enforce_provenance_kind(self):
        with self.assertRaisesRegex(ValueError, "user_supplied provenance"):
            emission.evaluate_localized_trap_emission(
                emission.TrapEmissionConfiguration(
                    emission.TrapEmissionMode.USER_SUPPLIED, self.parameters
                ),
                self.profile,
            )
        with self.assertRaisesRegex(ValueError, "literature provenance"):
            emission.evaluate_localized_trap_emission(
                emission.TrapEmissionConfiguration(
                    emission.TrapEmissionMode.LITERATURE_PARAMETER_SET,
                    self.parameters,
                ),
                self.profile,
            )

    def test_band_edge_uses_potential_not_first_cell_field(self):
        profile = emission.build_radial_potential_profile(
            (1.0e-6, 1.1e-6),
            (0.2, -0.3),
            ("Oxide",),
            5.0e-6,
            source="synthetic",
        )
        band = emission.build_band_edge_profile(profile, {"Oxide": 3.5})[0]
        self.assertAlmostEqual(band.inner_conduction_band_eV, 3.3)
        self.assertAlmostEqual(band.outer_conduction_band_eV, 3.8)

    def test_rectangular_wkb_matches_analytic_action(self):
        band = (
            emission.BandEdgeSegment(1.0e-6, 1.1e-6, 2.0, 2.0, "Oxide"),
        )
        result = emission.calculate_wkb_path(
            band,
            trap_radius_cm=1.0e-6,
            endpoint_radius_cm=1.1e-6,
            trap_energy_eV=1.0,
            electron_effective_mass_ratio_by_material={"Oxide": 0.25},
            destination="gate",
        )
        length_m = 1.0e-9
        expected = (
            math.sqrt(
                2.0
                * 0.25
                * emission.FREE_ELECTRON_MASS_KG
                * emission.ELEMENTARY_CHARGE_C
            )
            / emission.REDUCED_PLANCK_CONSTANT_J_S
            * length_m
        )
        self.assertAlmostEqual(result.action_integral_dimensionless, expected, places=12)
        self.assertAlmostEqual(result.log_transmission, -2.0 * expected, places=12)

    def test_gamma_branching_and_rate_equation_units(self):
        result = emission.evaluate_localized_trap_emission(self.config, self.profile)
        self.assertTrue(result.supported)
        self.assertGreater(result.gamma_total_s1, 0.0)
        self.assertTrue(
            math.isclose(
                result.gamma_channel_s1 + result.gamma_gate_s1,
                result.gamma_total_s1,
                rel_tol=2.0e-15,
                abs_tol=0.0,
            )
        )
        self.assertTrue(
            math.isclose(
                result.channel_branching_ratio + result.gate_branching_ratio,
                1.0,
                rel_tol=2.0e-15,
                abs_tol=2.0e-15,
            )
        )
        density = 2.0e18
        self.assertEqual(
            emission.trap_density_derivative_cm3_s(density, result.gamma_total_s1),
            -result.gamma_total_s1 * density,
        )

    def test_log_rates_survive_linear_underflow(self):
        deep_parameters = emission.ElectronTrapEmissionParameters(
            attempt_frequency_Hz=1.0e12,
            trap_energy_reference=emission.SUPPORTED_TRAP_ENERGY_REFERENCE,
            band_edge_reference="synthetic",
            conduction_band_edge_eV_by_material={"Al2O3": 100.0, "HfO2": 100.0},
            electron_effective_mass_ratio_by_material={"Al2O3": 1.0, "HfO2": 1.0},
            energy_bins=(emission.TrapEnergyBin(1.0e6),),
            radial_position_bins=(emission.RadialTrapPositionBin(1.4e-6, material="HfO2"),),
            spatial_distribution=emission.SpatialDistribution.SINGLE_POSITION,
            stable_trap_fraction=0.0,
            attempt_partition=emission.AttemptFrequencyPartition(0.5, 0.5, "synthetic"),
            provenance=emission.ParameterProvenance(
                emission.ProvenanceKind.SYNTHETIC_TEST, "synthetic underflow test"
            ),
        )
        result = emission.evaluate_localized_trap_emission(
            emission.TrapEmissionConfiguration(
                emission.TrapEmissionMode.SENSITIVITY_ONLY, deep_parameters
            ),
            self.profile,
        )
        self.assertEqual(result.gamma_total_s1, 0.0)
        self.assertTrue(math.isfinite(result.log_gamma_total_s1))

    def test_multiple_energy_and_position_bins_remain_explicit(self):
        parameters = emission.ElectronTrapEmissionParameters(
            attempt_frequency_Hz=1.0e13,
            trap_energy_reference=emission.SUPPORTED_TRAP_ENERGY_REFERENCE,
            band_edge_reference="synthetic",
            conduction_band_edge_eV_by_material={"Al2O3": 3.0, "HfO2": 2.0},
            electron_effective_mass_ratio_by_material={"Al2O3": 0.3, "HfO2": 0.2},
            energy_bins=(emission.TrapEnergyBin(0.8, 0.25), emission.TrapEnergyBin(1.2, 0.75)),
            radial_position_bins=emission.uniform_radial_position_bins(1.3e-6, 1.5e-6, 2, "HfO2"),
            spatial_distribution=emission.SpatialDistribution.UNIFORM,
            stable_trap_fraction=0.1,
            attempt_partition=emission.AttemptFrequencyPartition(0.4, 0.6, "synthetic"),
            provenance=emission.ParameterProvenance(
                emission.ProvenanceKind.SYNTHETIC_TEST, "synthetic grid"
            ),
        )
        grid = emission.evaluate_trap_emission_grid(
            emission.TrapEmissionConfiguration(
                emission.TrapEmissionMode.SENSITIVITY_ONLY, parameters
            ),
            self.profile,
        )
        self.assertEqual(len(grid.results), 4)
        self.assertAlmostEqual(sum(grid.population_weights), 1.0)
        self.assertIsNotNone(grid.initial_population_average_gamma_total_s1)

    def test_devsim_adapter_is_lazy_and_uses_nearest_axial_line(self):
        fake = _FakeDevsim()
        profile = emission.extract_devsim_radial_potential_profile(
            device="synthetic_device",
            axial_position_cm=5.0e-6,
            runtime_api=fake,
            maximum_axial_distance_cm=1.0e-12,
        )
        self.assertEqual(profile.materials, ("TunnelOxide", "ChargeTrap", "BlockingOxide"))
        self.assertEqual(len(profile.segments), 3)
        self.assertAlmostEqual(profile.segments[0].outer_potential_V, -0.1)
        self.assertEqual(fake.last_device, "synthetic_device")

    def test_interface_localized_helper_requires_explicit_radius(self):
        position = emission.interface_localized_position_bin(
            1.3e-6, "tunnel/trap", "ChargeTrap"
        )
        self.assertEqual(position.radius_cm, 1.3e-6)
        self.assertIn("tunnel/trap", position.label)


if __name__ == "__main__":
    unittest.main(verbosity=2)
