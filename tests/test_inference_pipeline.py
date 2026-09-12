"""Tests for feature semantics, model contracts, and cycle assessments."""

import copy
import io
import json
import tempfile
import unittest

from contextlib import redirect_stdout
from pathlib import Path

import joblib
import numpy as np

from maintenance_copilot.cli import main
from maintenance_copilot.features import (
    FEATURE_COLUMNS,
    FEATURE_SPEC,
    SENSORS,
    extract_temperature_features,
)
from maintenance_copilot.inference import (
    assess_cycle,
    load_model_bundle,
)


class FixedScoreModel:
    """Deterministic scoring double for threshold boundary tests."""

    feature_names_in_ = np.array(FEATURE_COLUMNS)

    def score_samples(self, features):
        return np.full(len(features), -0.5)


def make_bundle():
    return {
        "artifact_version": 1,
        "component_scope": "cooler",
        "nominal_cooler_condition_pct": 100,
        "stable_cycles_only": True,
        "feature_columns": FEATURE_COLUMNS.copy(),
        "feature_spec": copy.deepcopy(FEATURE_SPEC),
        "model": FixedScoreModel(),
        "anomaly_threshold": 0.5,
        "ts1_threshold": 35.0,
        "reference_feature_medians": {
            f"{sensor}__mean": 30.0
            for sensor in SENSORS
        },
    }


def make_signals(value=30.0):
    return {
        sensor: np.full(60, value)
        for sensor in SENSORS
    }


def assess(bundle, signals, stable=True):
    return assess_cycle(
        bundle,
        signals,
        machine_id="test-rig",
        cycle_id=1,
        stable_conditions=stable,
    )


class InferencePipelineTests(unittest.TestCase):
    def test_linear_temperature_features(self):
        signal = 20 + 0.1 * np.arange(60)

        result = extract_temperature_features(
            {
                sensor: signal[None, :]
                for sensor in SENSORS
            }
        )

        self.assertEqual(result.shape, (1, 24))
        self.assertAlmostEqual(
            result.loc[0, "TS1__mean"], 22.95
        )
        self.assertAlmostEqual(
            result.loc[0, "TS1__range"], 5.9
        )
        self.assertAlmostEqual(
            result.loc[0, "TS1__slope"], 0.1
        )

    def test_features_do_not_depend_on_other_cycles(self):
        single = {
            sensor: np.arange(60, dtype=float)[None, :]
            for sensor in SENSORS
        }

        batch = {
            sensor: np.vstack([values, values + 1000])
            for sensor, values in single.items()
        }

        np.testing.assert_allclose(
            extract_temperature_features(single).iloc[0],
            extract_temperature_features(batch).iloc[0],
        )

    def test_invalid_sensor_inputs_are_rejected(self):
        valid = {
            sensor: np.ones((1, 60))
            for sensor in SENSORS
        }

        missing = {
            sensor: values
            for sensor, values in valid.items()
            if sensor != "TS4"
        }

        cases = [
            missing,
            {**valid, "TS1": np.ones((1, 59))},
            {**valid, "TS1": np.full((1, 60), np.nan)},
            {**valid, "TS1": np.ones((2, 60))},
        ]

        for signals in cases:
            with self.subTest(signals=list(signals)):
                with self.assertRaises(ValueError):
                    extract_temperature_features(signals)

    def test_score_equal_to_threshold_does_not_alert(self):
        event = assess(
            make_bundle(),
            make_signals(),
        )

        self.assertFalse(
            event["detectors"]["isolation_forest"]["is_anomaly"]
        )

    def test_temperature_evidence_uses_celsius_differences(self):
        event = assess(
            make_bundle(),
            make_signals(40.0),
        )

        self.assertTrue(
            event["detectors"]["ts1_mean_threshold"]["is_anomaly"]
        )

        self.assertEqual(
            event["temperature_evidence"][0]["deviation_c"],
            10.0,
        )

        self.assertEqual(
            event["root_cause_status"],
            "not_assessed",
        )

        json.dumps(event, allow_nan=False)

    def test_unconfirmed_stability_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "stable"):
            assess(
                make_bundle(),
                make_signals(),
                stable=False,
            )

    def test_incompatible_feature_spec_is_rejected(self):
        bundle = make_bundle()

        bundle["feature_spec"]["sampling_rate_hz"] = 10.0

        with self.assertRaisesRegex(ValueError, "specification"):
            assess(
                bundle,
                make_signals(),
            )

    def test_saved_model_reproduces_assessment(self):
        bundle = make_bundle()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.joblib"

            joblib.dump(bundle, path)
            loaded = load_model_bundle(path)

            self.assertEqual(
                assess(bundle, make_signals()),
                assess(loaded, make_signals()),
            )

    def test_cli_does_not_use_cooler_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            for sensor in SENSORS:
                np.savetxt(
                    root / f"{sensor}.txt",
                    np.full((1, 60), 40.0),
                )

            model_path = root / "model.joblib"
            output_path = root / "assessment.json"

            joblib.dump(make_bundle(), model_path)

            events = []

            for cooler_condition in (3, 100):
                (root / "profile.txt").write_text(
                    f"{cooler_condition} 100 0 130 0\n",
                    encoding="utf-8",
                )

                captured = io.StringIO()

                with redirect_stdout(captured):
                    main(
                        [
                            "--cycle-id", "1",
                            "--data-dir", str(root),
                            "--model", str(model_path),
                            "--output", str(output_path),
                        ]
                    )

                event = json.loads(captured.getvalue())

                self.assertEqual(
                    event,
                    json.loads(output_path.read_text()),
                )

                events.append(event)

            self.assertEqual(events[0], events[1])


if __name__ == "__main__":
    unittest.main()