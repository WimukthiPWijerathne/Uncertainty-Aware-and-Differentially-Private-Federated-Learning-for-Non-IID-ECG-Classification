"""
Build the final project comparison table and essential figures.

Run this only after the full DP-UA-FedAvg experiment has completed.

The script:
  * locates the latest DP-UA summary JSON, or uses the path supplied
    with --dp-ua-summary,
  * combines it with the already validated baseline results,
  * saves one final comparison CSV,
  * saves a source manifest JSON,
  * creates the essential final figures used by the notebook/report.

Usage
-----
uv run python scripts\\17_finalize_project_results.py

Or specify an exact DP-UA summary:
uv run python scripts\\17_finalize_project_results.py ^
    --dp-ua-summary results\\tables\\<run>_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = (
    RESULTS_DIR
    / "figures"
    / "final_project"
)

FINAL_COMPARISON_PATH = (
    TABLE_DIR
    / "final_model_comparison.csv"
)

SOURCE_MANIFEST_PATH = (
    TABLE_DIR
    / "final_results_source_manifest.json"
)


# These values come from the completed, validated experiment outputs.
# DP-UA-FedAvg is loaded dynamically after its full run.
BASELINE_RESULTS = [
    {
        "method": "Centralized CNN",
        "training_type": "centralized",
        "best_round_or_epoch": 10,
        "test_loss": 0.308395,
        "test_macro_f1": 0.6686,
        "test_weighted_f1": 0.7216,
        "test_macro_auroc": 0.8964,
        "reported_epsilon": None,
        "delta": None,
        "privacy_enabled": False,
        "uncertainty_weighting": False,
        "notes": (
            "Centralized upper-performance baseline."
        ),
    },
    {
        "method": "Standard FedAvg",
        "training_type": "federated",
        "best_round_or_epoch": 15,
        "test_loss": 0.333644,
        "test_macro_f1": 0.6333,
        "test_weighted_f1": 0.6900,
        "test_macro_auroc": 0.8771,
        "reported_epsilon": None,
        "delta": None,
        "privacy_enabled": False,
        "uncertainty_weighting": False,
        "notes": (
            "Sample-count-weighted federated baseline."
        ),
    },
    {
        "method": "Compatible FedAvg",
        "training_type": "federated-control",
        "best_round_or_epoch": 15,
        "test_loss": 0.333640,
        "test_macro_f1": 0.6333,
        "test_weighted_f1": 0.6899,
        "test_macro_auroc": 0.8771,
        "reported_epsilon": None,
        "delta": None,
        "privacy_enabled": False,
        "uncertainty_weighting": False,
        "notes": (
            "Non-private control using the Opacus-compatible model."
        ),
    },
    {
        "method": "UA-FedAvg",
        "training_type": "federated",
        "best_round_or_epoch": 9,
        "test_loss": 0.352530,
        "test_macro_f1": 0.6134,
        "test_weighted_f1": 0.6677,
        "test_macro_auroc": 0.8658,
        "reported_epsilon": None,
        "delta": None,
        "privacy_enabled": False,
        "uncertainty_weighting": True,
        "notes": (
            "Inverse MC-dropout uncertainty-aware aggregation."
        ),
    },
    {
        "method": "DP-FedAvg",
        "training_type": "private-federated",
        "best_round_or_epoch": 9,
        "test_loss": 0.487335,
        "test_macro_f1": 0.2750,
        "test_weighted_f1": 0.3803,
        "test_macro_auroc": 0.7711,
        "reported_epsilon": 5.4551184213239,
        "delta": 1e-5,
        "privacy_enabled": True,
        "uncertainty_weighting": False,
        "notes": (
            "DP-SGD with noise multiplier 1.0 and clipping norm 1.0. "
            "Reported epsilon is total-training maximum client epsilon."
        ),
    },
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create final comparison data and essential report figures."
        )
    )

    parser.add_argument(
        "--dp-ua-summary",
        type=Path,
        default=None,
        help=(
            "Exact DP-UA summary JSON. When omitted, the latest matching "
            "summary in results/tables is selected."
        ),
    )

    return parser.parse_args()


def locate_dp_ua_summary(
    requested_path: Path | None,
) -> Path:
    if requested_path is not None:
        path = requested_path

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.is_file():
            raise FileNotFoundError(
                f"DP-UA summary not found: {path}"
            )

        return path

    candidates = sorted(
        TABLE_DIR.glob(
            "dp_ua_fedavg_*_summary.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No DP-UA summary JSON was found. Complete the full "
            "DP-UA-FedAvg run first, or pass --dp-ua-summary."
        )

    return candidates[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            sort_keys=True,
        )


def build_dp_ua_row(
    summary: dict[str, Any],
) -> dict[str, Any]:
    required_fields = {
        "best_round",
        "test_loss",
        "test_macro_f1",
        "test_weighted_f1",
        "test_macro_auroc",
        "reported_epsilon",
        "delta",
        "run_name",
    }

    missing = required_fields.difference(
        summary
    )

    if missing:
        raise ValueError(
            "DP-UA summary is missing fields: "
            f"{sorted(missing)}"
        )

    return {
        "method": "DP-UA-FedAvg",
        "training_type": (
            "private-uncertainty-federated"
        ),
        "best_round_or_epoch": int(
            summary["best_round"]
        ),
        "test_loss": float(
            summary["test_loss"]
        ),
        "test_macro_f1": float(
            summary["test_macro_f1"]
        ),
        "test_weighted_f1": float(
            summary["test_weighted_f1"]
        ),
        "test_macro_auroc": float(
            summary["test_macro_auroc"]
        ),
        "reported_epsilon": float(
            summary["reported_epsilon"]
        ),
        "delta": float(
            summary["delta"]
        ),
        "privacy_enabled": True,
        "uncertainty_weighting": True,
        "notes": (
            "DP-SGD local training combined with inverse "
            "MC-dropout uncertainty-aware aggregation. "
            f"Source run: {summary['run_name']}."
        ),
    }


def save_bar_chart(
    comparison: pd.DataFrame,
    metric: str,
    ylabel: str,
    filename: str,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        comparison["method"],
        comparison[metric],
    )

    axis.set_title(
        ylabel + " by Method"
    )
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Method")
    axis.tick_params(
        axis="x",
        rotation=30,
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / filename,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def find_dp_history() -> Path:
    path = (
        LOG_DIR
        / (
            "dp_fedavg_rounds_50_local_epochs_1_"
            "noise_1_clip_1_epscap_8"
        )
        / "dp_fedavg_round_history.csv"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"DP-FedAvg round history not found: {path}"
        )

    return path


def find_dp_ua_histories(
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    output_paths = summary.get(
        "output_paths",
        {},
    )

    round_history = Path(
        output_paths.get(
            "round_history",
            "",
        )
    )

    client_history = Path(
        output_paths.get(
            "client_history",
            "",
        )
    )

    if not round_history.is_absolute():
        round_history = (
            PROJECT_ROOT
            / round_history
        )

    if not client_history.is_absolute():
        client_history = (
            PROJECT_ROOT
            / client_history
        )

    if not round_history.is_file():
        raise FileNotFoundError(
            "DP-UA round-history CSV not found: "
            f"{round_history}"
        )

    if not client_history.is_file():
        raise FileNotFoundError(
            "DP-UA client-history CSV not found: "
            f"{client_history}"
        )

    return (
        round_history,
        client_history,
    )


def save_epsilon_plot(
    dp_history_path: Path,
    dp_ua_history_path: Path,
) -> None:
    dp_history = pd.read_csv(
        dp_history_path
    )
    dp_ua_history = pd.read_csv(
        dp_ua_history_path
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.plot(
        dp_history["round"],
        dp_history[
            "maximum_client_epsilon"
        ],
        marker="o",
        label="DP-FedAvg",
    )

    axis.plot(
        dp_ua_history["round"],
        dp_ua_history[
            "maximum_client_epsilon"
        ],
        marker="o",
        label="DP-UA-FedAvg",
    )

    axis.set_title(
        "Maximum Client Epsilon by Federated Round"
    )
    axis.set_xlabel(
        "Federated Round"
    )
    axis.set_ylabel(
        "Maximum Client Epsilon"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / "epsilon_by_round.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_dp_ua_weight_plot(
    client_history_path: Path,
) -> None:
    history = pd.read_csv(
        client_history_path
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    for hospital_id, group in history.groupby(
        "hospital_id"
    ):
        group = group.sort_values(
            "round"
        )

        axis.plot(
            group["round"],
            group[
                "final_aggregation_weight"
            ],
            marker="o",
            label=(
                f"Hospital {hospital_id}"
            ),
        )

    axis.set_title(
        "DP-UA Aggregation Weight by Hospital"
    )
    axis.set_xlabel(
        "Federated Round"
    )
    axis.set_ylabel(
        "Final Aggregation Weight"
    )
    axis.grid(alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / "dp_ua_aggregation_weights.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_privacy_utility_plot(
    comparison: pd.DataFrame,
) -> None:
    private_methods = comparison[
        comparison["privacy_enabled"]
    ].dropna(
        subset=[
            "reported_epsilon",
            "test_macro_f1",
        ]
    )

    figure, axis = plt.subplots(
        figsize=(7, 5)
    )

    axis.scatter(
        private_methods[
            "reported_epsilon"
        ],
        private_methods[
            "test_macro_f1"
        ],
        s=80,
    )

    for _, row in private_methods.iterrows():
        axis.annotate(
            row["method"],
            (
                row[
                    "reported_epsilon"
                ],
                row[
                    "test_macro_f1"
                ],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    axis.set_title(
        "Privacy–Utility Comparison"
    )
    axis.set_xlabel(
        "Total-Training Maximum Client Epsilon"
    )
    axis.set_ylabel(
        "Test Macro-F1"
    )
    axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR
        / "privacy_utility_comparison.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dp_ua_summary_path = (
        locate_dp_ua_summary(
            args.dp_ua_summary
        )
    )

    dp_ua_summary = load_json(
        dp_ua_summary_path
    )

    dp_ua_row = build_dp_ua_row(
        dp_ua_summary
    )

    comparison = pd.DataFrame(
        [
            *BASELINE_RESULTS,
            dp_ua_row,
        ]
    )

    comparison.to_csv(
        FINAL_COMPARISON_PATH,
        index=False,
    )

    save_bar_chart(
        comparison=comparison,
        metric="test_macro_f1",
        ylabel="Test Macro-F1",
        filename=(
            "macro_f1_comparison.png"
        ),
    )

    save_bar_chart(
        comparison=comparison,
        metric="test_macro_auroc",
        ylabel="Test Macro-AUROC",
        filename=(
            "macro_auroc_comparison.png"
        ),
    )

    save_privacy_utility_plot(
        comparison
    )

    dp_history_path = (
        find_dp_history()
    )

    (
        dp_ua_round_history,
        dp_ua_client_history,
    ) = find_dp_ua_histories(
        dp_ua_summary
    )

    save_epsilon_plot(
        dp_history_path,
        dp_ua_round_history,
    )

    save_dp_ua_weight_plot(
        dp_ua_client_history
    )

    manifest = {
        "final_comparison_csv": str(
            FINAL_COMPARISON_PATH
        ),
        "dp_ua_summary_json": str(
            dp_ua_summary_path
        ),
        "dp_fedavg_round_history": str(
            dp_history_path
        ),
        "dp_ua_round_history": str(
            dp_ua_round_history
        ),
        "dp_ua_client_history": str(
            dp_ua_client_history
        ),
        "figure_directory": str(
            FIGURE_DIR
        ),
        "baseline_values_note": (
            "Baseline rows use the completed and validated experiment "
            "results established before the DP-UA run."
        ),
    }

    save_json(
        SOURCE_MANIFEST_PATH,
        manifest,
    )

    print("\nFinal comparison")
    print(
        comparison[
            [
                "method",
                "test_macro_f1",
                "test_weighted_f1",
                "test_macro_auroc",
                "reported_epsilon",
                "delta",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nFinal comparison CSV: "
        f"{FINAL_COMPARISON_PATH}"
    )
    print(
        f"Source manifest: "
        f"{SOURCE_MANIFEST_PATH}"
    )
    print(
        f"Final figures: "
        f"{FIGURE_DIR}"
    )


if __name__ == "__main__":
    main()