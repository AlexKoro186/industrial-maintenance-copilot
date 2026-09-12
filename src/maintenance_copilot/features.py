"""Temperature features for complete 60-second hydraulic cycles."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

SENSORS = ("TS1", "TS2", "TS3", "TS4")
STATISTICS = ("mean", "std", "min", "max", "range", "slope")

FEATURE_COLUMNS = [
    f"{sensor}__{stat}"
    for sensor in SENSORS
    for stat in STATISTICS
]

FEATURE_SPEC = {
    "sensors": list(SENSORS),
    "samples_per_cycle": 60,
    "sampling_rate_hz": 1.0,
    "statistics": list(STATISTICS),
    "std_ddof": 0,
    "slope_method": (
        "Least-squares linear trend in degrees Celsius per second"
    ),
}


def extract_temperature_features(
    signals: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Accept four arrays shaped (number_of_cycles, 60), in matching row order."""
    if set(signals) != set(SENSORS):
        raise ValueError("Expected exactly TS1, TS2, TS3, and TS4.")

    data = {}
    cycle_count = None

    time_s = np.arange(60, dtype=float)
    centered_time = time_s - time_s.mean()

    for sensor in SENSORS:
        values = np.asarray(signals[sensor], dtype=float)

        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != 60
        ):
            raise ValueError(
                f"{sensor} must have shape (number_of_cycles, 60)."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{sensor} contains missing or infinite measurements."
            )

        if cycle_count is not None and values.shape[0] != cycle_count:
            raise ValueError(
                "All sensors must contain the same number of cycles."
            )

        cycle_count = values.shape[0]

        data[f"{sensor}__mean"] = values.mean(axis=1)
        data[f"{sensor}__std"] = values.std(axis=1, ddof=0)
        data[f"{sensor}__min"] = values.min(axis=1)
        data[f"{sensor}__max"] = values.max(axis=1)
        data[f"{sensor}__range"] = np.ptp(values, axis=1)

        data[f"{sensor}__slope"] = (
            values @ centered_time
            / (centered_time @ centered_time)
        )

    features = pd.DataFrame(data, columns=FEATURE_COLUMNS)

    if not np.isfinite(features.to_numpy()).all():
        raise ValueError(
            "Feature extraction produced nonfinite values."
        )

    return features