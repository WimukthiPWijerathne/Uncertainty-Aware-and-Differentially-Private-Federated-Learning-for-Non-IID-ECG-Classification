"""
Run the remaining DP-FedAvg noise-multiplier experiments.

The completed noise=1.0 experiment is intentionally omitted by default.
Each child process uses the same DP-FedAvg runner and changes only the
noise multiplier through environment variables.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DP_RUNNER = PROJECT_ROOT / "scripts" / "12_run_dp_fedavg.py"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"

DEFAULT_NOISE_VALUES = [0.75, 1.25, 1.50]
MAX_ROUNDS = 50
LOCAL_EPOCHS = 1
MAX_GRAD_NORM = 1.0
MAX_EPSILON = 8.0


def result_path(noise_multiplier: float) -> Path:
    run_name = (
        f"dp_fedavg_rounds_{MAX_ROUNDS}"
        f"_local_epochs_{LOCAL_EPOCHS}"
        f"_noise_{noise_multiplier:g}"
        f"_clip_{MAX_GRAD_NORM:g}"
        f"_epscap_{MAX_EPSILON:g}"
    )
    return RESULTS_TABLE_DIR / f"{run_name}_test_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a DP-FedAvg noise-multiplier sweep."
    )
    parser.add_argument(
        "--noise-values",
        nargs="+",
        type=float,
        default=DEFAULT_NOISE_VALUES,
        help="Noise multipliers to run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun experiments even if their test CSV exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not DP_RUNNER.is_file():
        raise FileNotFoundError(
            f"DP runner not found: {DP_RUNNER}"
        )

    for index, noise in enumerate(args.noise_values, start=1):
        if noise <= 0:
            raise ValueError(
                f"Noise multiplier must be positive: {noise}"
            )

        output_csv = result_path(noise)

        print("\n" + "=" * 72)
        print(
            f"Noise experiment {index}/{len(args.noise_values)}: "
            f"noise_multiplier={noise:g}"
        )
        print("=" * 72)

        if output_csv.is_file() and not args.force:
            print(
                "Skipping completed experiment because this file exists:"
            )
            print(output_csv)
            continue

        environment = os.environ.copy()
        environment.update(
            {
                "DP_NOISE_MULTIPLIER": str(noise),
                "DP_MAX_GRAD_NORM": str(MAX_GRAD_NORM),
                "DP_MAX_EPSILON": str(MAX_EPSILON),
                "DP_MAX_ROUNDS": str(MAX_ROUNDS),
                "DP_LOCAL_EPOCHS": str(LOCAL_EPOCHS),
                "DP_DELTA": "1e-5",
                "DP_SECURE_MODE": "false",
            }
        )

        subprocess.run(
            [sys.executable, str(DP_RUNNER)],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
        )

    print("\nDP noise sweep finished.")


if __name__ == "__main__":
    main()