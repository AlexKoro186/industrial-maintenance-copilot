"""Replay one cycle from the UCI hydraulic dataset and emit a JSON assessment."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from maintenance_copilot.features import SENSORS
from maintenance_copilot.inference import (
    assess_cycle,
    load_model_bundle,
)


def read_benchmark_cycle(
    data_dir: Path,
    cycle_id: int,
) -> tuple[dict, bool]:
    """Read one-based file rows and the benchmark stability flag."""
    if cycle_id < 1:
        raise ValueError("cycle_id must be at least 1.")

    profile = pd.read_csv(
        data_dir / "profile.txt",
        sep=r"\s+",
        header=None,
    )

    if profile.shape[1] != 5 or cycle_id > len(profile):
        raise ValueError(
            "Invalid profile format or cycle_id outside the dataset."
        )

    # Only the stability flag is used as eligibility metadata.
    stability_flag = profile.iloc[cycle_id - 1, 4]

    if stability_flag not in (0, 1):
        raise ValueError(
            "Invalid or missing stability flag."
        )

    signals = {}

    for sensor in SENSORS:
        row = pd.read_csv(
            data_dir / f"{sensor}.txt",
            sep=r"\s+",
            header=None,
            skiprows=cycle_id - 1,
            nrows=1,
            dtype=np.float64,
        )

        signals[sensor] = row.to_numpy().reshape(-1)

    return signals, bool(stability_flag == 0)


def main(argv=None) -> int:
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--cycle-id",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--machine-id",
        default="hydraulic-test-rig",
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data/raw/hydraulic",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=project_root / "models/cooler_isolation_forest.joblib",
    )

    parser.add_argument(
        "--output",
        type=Path,
    )

    args = parser.parse_args(argv)

    try:
        signals, stable = read_benchmark_cycle(
            args.data_dir,
            args.cycle_id,
        )

        bundle = load_model_bundle(args.model)

        event = assess_cycle(
            bundle,
            signals,
            machine_id=args.machine_id,
            cycle_id=args.cycle_id,
            stable_conditions=stable,
        )

        text = json.dumps(
            event,
            indent=2,
            allow_nan=False,
        ) + "\n"

        if args.output is not None:
            args.output.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            args.output.write_text(
                text,
                encoding="utf-8",
            )

    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
