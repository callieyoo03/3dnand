"""Schema and metadata tests for the ERASE candidate CSV outputs."""

import unittest

import run_erase_voltage_sweep as runner


class EraseOutputSchemaTests(unittest.TestCase):
    def test_required_output_names(self):
        self.assertEqual(
            set(runner.OUTPUT_FILENAMES),
            {
                "erase_voltage_sweep.csv",
                "erase_layer_voltage_drop.csv",
                "erase_field_consistency.csv",
                "erase_edge_distribution.csv",
                "erase_fn_log_domain.csv",
                "erase_baseline_README.md",
            },
        )

    def test_all_csv_schemas_have_exact_model_labels(self):
        schemas = (
            runner.VOLTAGE_SWEEP_FIELDS,
            runner.LAYER_VOLTAGE_FIELDS,
            runner.FIELD_CONSISTENCY_FIELDS,
            runner.EDGE_DISTRIBUTION_FIELDS,
            runner.FN_LOG_DOMAIN_FIELDS,
        )
        for schema in schemas:
            self.assertEqual(schema[: len(runner.MODEL_FIELDS)], runner.MODEL_FIELDS)
            self.assertEqual(len(schema), len(set(schema)))
        self.assertEqual(
            runner._model_metadata(),
            {
                "model_class": "electron_FN_diagnostic",
                "physical_role": "negative_control",
                "model_status": "baseline_negative_result",
                "calibration_status": "uncalibrated",
                "predictive_erase_model": False,
            },
        )

    def test_schema_validation_rejects_missing_or_extra_columns(self):
        runner.validate_csv_rows(("a", "b"), [{"a": 1, "b": 2}])
        with self.assertRaisesRegex(ValueError, "missing"):
            runner.validate_csv_rows(("a", "b"), [{"a": 1}])
        with self.assertRaisesRegex(ValueError, "extra"):
            runner.validate_csv_rows(("a",), [{"a": 1, "b": 2}])

    def test_predictive_flag_serializes_lowercase(self):
        self.assertEqual(runner._csv_value(False), "false")
        self.assertEqual(runner._csv_value(True), "true")


if __name__ == "__main__":
    unittest.main(verbosity=2)
