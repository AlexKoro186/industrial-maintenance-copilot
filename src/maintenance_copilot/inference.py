"""Assess completed cycles using the model bundle from Notebook 02."""

from pathlib import Path

import joblib
import numpy as np

from maintenance_copilot.features import (
    FEATURE_COLUMNS,
    FEATURE_SPEC,
    SENSORS,
    extract_temperature_features,
)


def validate_model_bundle(bundle: dict) -> None:
    """Check the model scope, feature contract, and numeric metadata."""
    if (
        not isinstance(bundle, dict)
        or bundle.get("artifact_version") != 1
    ):
        raise ValueError(
            "Expected a version 1 model bundle from Notebook 02."
        )

    if (
        bundle.get("component_scope") != "cooler"
        or bundle.get("nominal_cooler_condition_pct") != 100
        or bundle.get("stable_cycles_only") is not True
    ):
        raise ValueError(
            "The model must use the stable, nominal-cooler reference."
        )

    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            "The saved feature columns do not match this pipeline."
        )

    if bundle.get("feature_spec") != FEATURE_SPEC:
        raise ValueError(
            "The saved feature specification does not match this pipeline."
        )

    model = bundle.get("model")

    if not callable(getattr(model, "score_samples", None)):
        raise ValueError(
            "The bundle does not contain a scoring model."
        )

    if list(getattr(model, "feature_names_in_", [])) != FEATURE_COLUMNS:
        raise ValueError(
            "The fitted model has an incompatible feature schema."
        )

    try:
        numbers = [
            bundle["anomaly_threshold"],
            bundle["ts1_threshold"],
        ]

        reference = bundle["reference_feature_medians"]

        numbers.extend(
            reference[f"{sensor}__mean"]
            for sensor in SENSORS
        )

        finite = np.isfinite(
            np.asarray(numbers, dtype=float)
        ).all()

    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Missing or invalid thresholds or temperature references."
        ) from exc

    if not finite:
        raise ValueError(
            "Thresholds and temperature references must be finite."
        )


def load_model_bundle(path: str | Path) -> dict:
    """Load a locally generated model bundle from Notebook 02."""
    bundle = joblib.load(path)
    validate_model_bundle(bundle)
    return bundle


def assess_cycle(
    bundle: dict,
    signals: dict,
    *,
    machine_id: str,
    cycle_id: int,
    stable_conditions: bool,
) -> dict:
    """Return a JSON-compatible assessment; each sensor contains 60 samples."""
    validate_model_bundle(bundle)

    if stable_conditions is not True:
        raise ValueError(
            "Confirmed stable operating conditions are required."
        )

    if type(cycle_id) is not int or cycle_id < 1:
        raise ValueError(
            "cycle_id must be a positive integer."
        )

    if not isinstance(machine_id, str) or not machine_id.strip():
        raise ValueError(
            "machine_id must be a nonempty string."
        )

    batch = {}

    for sensor, samples in signals.items():
        values = np.asarray(samples, dtype=float)

        if values.shape != (60,):
            raise ValueError(
                f"{sensor} must contain exactly 60 samples in one dimension."
            )

        batch[sensor] = values[None, :]

    features = extract_temperature_features(batch)

    score = float(
        -bundle["model"].score_samples(features)[0]
    )

    if not np.isfinite(score):
        raise ValueError(
            "The model returned a nonfinite anomaly score."
        )

    threshold = float(bundle["anomaly_threshold"])
    ts1_mean = float(features.iloc[0]["TS1__mean"])
    ts1_threshold = float(bundle["ts1_threshold"])

    evidence = []

    for sensor in SENSORS:
        observed = float(
            features.iloc[0][f"{sensor}__mean"]
        )

        reference = float(
            bundle["reference_feature_medians"][f"{sensor}__mean"]
        )

        evidence.append(
            {
                "sensor": sensor,
                "mean_c": observed,
                "reference_median_cycle_mean_c": reference,
                "deviation_c": observed - reference,
            }
        )

    return {
        "schema_version": 1,
        "event_type": "cycle_assessment",
        "machine_id": machine_id.strip(),
        "cycle_id": cycle_id,
        "component_scope": "cooler",
        "operating_conditions": "stable",
        "detectors": {
            "isolation_forest": {
                "anomaly_score": score,
                "threshold": threshold,
                "is_anomaly": score > threshold,
            },
            "ts1_mean_threshold": {
                "mean_c": ts1_mean,
                "threshold_c": ts1_threshold,
                "is_anomaly": ts1_mean > ts1_threshold,
            },
        },
        "temperature_evidence": evidence,
        "root_cause_status": "not_assessed",
    }