"""
Load a calibrated DP-FedAvg or DP-UA-FedAvg checkpoint, display the
DP-compatible ECG1DCNN architecture, generate predictions for held-out
PTB-XL test records, and save outputs for the notebook and presentation.

The script supports two threshold modes:

1. Class-specific thresholds stored in the calibrated checkpoint.
2. A single manual threshold supplied with --threshold.

Default calibrated checkpoint
-----------------------------
results/threshold_calibration/dp_fedavg/
    checkpoint_with_calibrated_thresholds.pt

Outputs
-------
results/predictions/dp_fedavg_calibrated/
    model_summary.txt
    sample_predictions.csv
    example_ecg.png
    example_probabilities.png

Usage
-----
Run with the calibrated DP-FedAvg checkpoint:

    uv run python scripts/18_show_model_predictions.py

Choose a different number of examples:

    uv run python scripts/18_show_model_predictions.py --num-samples 20

Use an exact calibrated checkpoint:

    uv run python scripts/18_show_model_predictions.py ^
        --checkpoint results/threshold_calibration/dp_fedavg/checkpoint_with_calibrated_thresholds.pt ^
        --method-name DP-FedAvg

Force one common threshold for comparison:

    uv run python scripts/18_show_model_predictions.py ^
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.data.ecg_dataset import (
    DEFAULT_CLASS_NAMES,
    PTBXLDataset,
)
from src.data.paths import (
    PROCESSED_METADATA_PATH,
    PTBXL_ROOT,
)
from src.federated.dp_client import (
    create_dp_compatible_ecg_model,
)


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "results"
    / "threshold_calibration"
    / "dp_fedavg"
    / "checkpoint_with_calibrated_thresholds.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "dp_fedavg_calibrated"
)

LEAD_NAMES = [
    "I",
    "II",
    "III",
    "aVR",
    "aVL",
    "aVF",
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate model and prediction outputs using either "
            "class-specific calibrated thresholds or one fixed threshold."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=(
            "Checkpoint containing model_state_dict. A calibrated "
            "checkpoint may also contain class_thresholds."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prediction outputs are saved.",
    )

    parser.add_argument(
        "--method-name",
        type=str,
        default="DP-FedAvg",
        help=(
            "Display name used in model summaries and figure titles."
        ),
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of randomly selected test ECGs to save in the CSV.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used to select reproducible test examples.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Optional single threshold applied to all five classes. "
            "When omitted, class_thresholds from the checkpoint are used. "
            "If the checkpoint has no class_thresholds, its scalar "
            "threshold or 0.5 is used."
        ),
    )

    parser.add_argument(
        "--example-position",
        type=int,
        default=0,
        help=(
            "Position inside the randomly selected examples used for "
            "the ECG and probability figures."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def validate_thresholds(thresholds: np.ndarray) -> None:
    expected_shape = (len(DEFAULT_CLASS_NAMES),)

    if thresholds.shape != expected_shape:
        raise ValueError(
            "Threshold array has the wrong shape. "
            f"Expected {expected_shape}, received {thresholds.shape}."
        )

    if not np.all(np.isfinite(thresholds)):
        raise ValueError(
            "All thresholds must be finite numeric values."
        )

    if np.any(thresholds <= 0) or np.any(thresholds >= 1):
        raise ValueError(
            "Every class threshold must be between 0 and 1."
        )


def load_thresholds_from_checkpoint(
    checkpoint: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """
    Load thresholds in exactly the same order as DEFAULT_CLASS_NAMES.

    Returns
    -------
    thresholds:
        Array with one threshold per diagnostic superclass.
    strategy:
        Human-readable threshold strategy.
    """
    if "class_thresholds" in checkpoint:
        stored_thresholds = checkpoint["class_thresholds"]

        if not isinstance(stored_thresholds, dict):
            raise TypeError(
                "'class_thresholds' must be stored as a dictionary."
            )

        missing_classes = [
            class_name
            for class_name in DEFAULT_CLASS_NAMES
            if class_name not in stored_thresholds
        ]

        if missing_classes:
            raise KeyError(
                "The calibrated checkpoint is missing thresholds for: "
                f"{missing_classes}"
            )

        thresholds = np.asarray(
            [
                float(stored_thresholds[class_name])
                for class_name in DEFAULT_CLASS_NAMES
            ],
            dtype=np.float32,
        )

        validate_thresholds(thresholds)
        return thresholds, "class_specific_validation_calibrated"

    scalar_threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    thresholds = np.full(
        len(DEFAULT_CLASS_NAMES),
        scalar_threshold,
        dtype=np.float32,
    )

    validate_thresholds(thresholds)
    return thresholds, f"fixed_{scalar_threshold:g}"


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    np.ndarray,
    str,
]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain 'model_state_dict'."
        )

    model = create_dp_compatible_ecg_model(
        device=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    thresholds, threshold_strategy = (
        load_thresholds_from_checkpoint(
            checkpoint
        )
    )

    return (
        model,
        checkpoint,
        thresholds,
        threshold_strategy,
    )


def format_threshold_lines(
    thresholds: np.ndarray,
) -> list[str]:
    return [
        f"{class_name}: {float(class_threshold):.4f}"
        for class_name, class_threshold in zip(
            DEFAULT_CLASS_NAMES,
            thresholds,
            strict=True,
        )
    ]


def save_model_summary(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
    method_name: str,
    thresholds: np.ndarray,
    threshold_strategy: str,
) -> None:
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    lines = [
        f"{method_name} Model Summary",
        "=" * 72,
        f"Checkpoint: {checkpoint_path}",
        f"Saved round: {checkpoint.get('round', 'unknown')}",
        (
            "Validation loss at checkpoint: "
            f"{checkpoint.get('validation_loss', 'unknown')}"
        ),
        (
            "Validation macro-F1 at checkpoint: "
            f"{checkpoint.get('validation_macro_f1', 'unknown')}"
        ),
        (
            "Validation macro-AUROC at checkpoint: "
            f"{checkpoint.get('validation_macro_auroc', 'unknown')}"
        ),
        f"Threshold strategy: {threshold_strategy}",
        "Class thresholds:",
        *[
            f"  {line}"
            for line in format_threshold_lines(
                thresholds
            )
        ],
        f"Total parameters: {total_parameters:,}",
        f"Trainable parameters: {trainable_parameters:,}",
        "",
        "Architecture",
        "-" * 72,
        str(model),
    ]

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def load_test_data() -> tuple[
    pd.DataFrame,
    PTBXLDataset,
]:
    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    test_metadata = metadata[
        metadata["split"] == "test"
    ].copy()

    if test_metadata.empty:
        raise RuntimeError(
            "The test split is empty."
        )

    dataset = PTBXLDataset(
        metadata=test_metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=False,
    )

    if len(dataset) != len(test_metadata):
        raise RuntimeError(
            "The test metadata and dataset lengths do not match."
        )

    return test_metadata, dataset


def unpack_dataset_item(
    item: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(item, dict):
        signal = item.get(
            "signal",
            item.get("x"),
        )
        label = item.get(
            "label",
            item.get("y"),
        )
    elif (
        isinstance(item, (tuple, list))
        and len(item) >= 2
    ):
        signal = item[0]
        label = item[1]
    else:
        raise TypeError(
            "PTBXLDataset must return either a dictionary or a "
            "tuple/list containing signal and label."
        )

    if signal is None or label is None:
        raise ValueError(
            "Could not extract signal and label from dataset item."
        )

    signal_tensor = torch.as_tensor(
        signal,
        dtype=torch.float32,
    )

    label_tensor = torch.as_tensor(
        label,
        dtype=torch.float32,
    )

    if signal_tensor.ndim != 2:
        raise ValueError(
            "Expected one ECG signal with shape (12, samples), "
            f"but received {tuple(signal_tensor.shape)}."
        )

    if signal_tensor.shape[0] != 12:
        raise ValueError(
            "Expected 12 ECG leads, but received "
            f"{signal_tensor.shape[0]}."
        )

    if label_tensor.numel() != len(
        DEFAULT_CLASS_NAMES
    ):
        raise ValueError(
            "Expected one target per diagnostic class, but received "
            f"{label_tensor.numel()} values."
        )

    return signal_tensor, label_tensor


def labels_from_binary(
    values: np.ndarray,
) -> list[str]:
    return [
        class_name
        for class_name, value in zip(
            DEFAULT_CLASS_NAMES,
            values,
            strict=True,
        )
        if int(value) == 1
    ]


def run_predictions(
    model: torch.nn.Module,
    dataset: PTBXLDataset,
    test_metadata: pd.DataFrame,
    selected_indices: list[int],
    thresholds: np.ndarray,
    threshold_strategy: str,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    validate_thresholds(thresholds)

    rows: list[dict[str, Any]] = []
    first_example: dict[str, Any] | None = None

    ecg_ids = list(
        test_metadata.index
    )

    model.eval()

    with torch.no_grad():
        for position, dataset_index in enumerate(
            selected_indices
        ):
            signal, target = unpack_dataset_item(
                dataset[dataset_index]
            )

            logits = model(
                signal.unsqueeze(0).to(
                    device,
                    non_blocking=True,
                )
            )

            probabilities = (
                torch.sigmoid(logits)
                .squeeze(0)
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            target_values = (
                target.cpu()
                .numpy()
                .astype(np.int64)
            )

            predicted_values = (
                probabilities >= thresholds
            ).astype(np.int64)

            true_labels = labels_from_binary(
                target_values
            )

            predicted_labels = labels_from_binary(
                predicted_values
            )

            row: dict[str, Any] = {
                "selection_position": position,
                "dataset_index": dataset_index,
                "ecg_id": ecg_ids[dataset_index],
                "true_labels": (
                    ", ".join(true_labels)
                    if true_labels
                    else "None"
                ),
                "predicted_labels": (
                    ", ".join(predicted_labels)
                    if predicted_labels
                    else "None"
                ),
                "exact_label_match": bool(
                    np.array_equal(
                        target_values,
                        predicted_values,
                    )
                ),
                "threshold_strategy": (
                    threshold_strategy
                ),
            }

            for (
                class_name,
                probability,
                class_threshold,
                true_value,
                predicted_value,
            ) in zip(
                DEFAULT_CLASS_NAMES,
                probabilities,
                thresholds,
                target_values,
                predicted_values,
                strict=True,
            ):
                row[
                    f"{class_name}_probability"
                ] = float(probability)

                row[
                    f"{class_name}_threshold"
                ] = float(class_threshold)

                row[
                    f"{class_name}_true"
                ] = int(true_value)

                row[
                    f"{class_name}_predicted"
                ] = int(predicted_value)

            rows.append(row)

            if first_example is None:
                first_example = {
                    "signal": signal.cpu().numpy(),
                    "ecg_id": ecg_ids[dataset_index],
                    "probabilities": probabilities,
                    "thresholds": thresholds.copy(),
                    "target_values": target_values,
                    "predicted_values": predicted_values,
                    "true_labels": true_labels,
                    "predicted_labels": predicted_labels,
                }

    if first_example is None:
        raise RuntimeError(
            "No prediction examples were generated."
        )

    return pd.DataFrame(rows), first_example


def save_ecg_figure(
    example: dict[str, Any],
    output_path: Path,
    method_name: str,
) -> None:
    signal = np.asarray(
        example["signal"],
        dtype=float,
    )

    sample_count = signal.shape[1]
    time_seconds = np.arange(
        sample_count
    ) / 100.0

    robust_scale = float(
        np.percentile(
            np.abs(signal),
            99,
        )
    )

    if robust_scale <= 0:
        robust_scale = 1.0

    vertical_spacing = robust_scale * 3.0
    offsets = (
        np.arange(
            len(LEAD_NAMES)
        )[::-1]
        * vertical_spacing
    )

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    for lead_index, lead_name in enumerate(
        LEAD_NAMES
    ):
        axis.plot(
            time_seconds,
            signal[lead_index]
            + offsets[lead_index],
            linewidth=0.8,
        )

    axis.set_yticks(offsets)
    axis.set_yticklabels(LEAD_NAMES)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("ECG lead")
    axis.set_title(
        f"{method_name}: Example 12-Lead ECG — "
        f"ECG ID {example['ecg_id']}"
    )
    axis.grid(alpha=0.2)

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_probability_figure(
    example: dict[str, Any],
    output_path: Path,
    method_name: str,
) -> None:
    probabilities = np.asarray(
        example["probabilities"],
        dtype=float,
    )

    thresholds = np.asarray(
        example["thresholds"],
        dtype=float,
    )

    positions = np.arange(
        len(DEFAULT_CLASS_NAMES)
    )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    axis.bar(
        positions,
        probabilities,
        width=0.65,
        label="Predicted probability",
    )

    for index, class_threshold in enumerate(
        thresholds
    ):
        axis.hlines(
            y=class_threshold,
            xmin=index - 0.28,
            xmax=index + 0.28,
            linewidth=3,
            label=(
                "Class-specific threshold"
                if index == 0
                else None
            ),
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        DEFAULT_CLASS_NAMES
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel(
        "Diagnostic superclass"
    )
    axis.set_ylabel(
        "Predicted probability"
    )
    axis.set_title(
        f"{method_name} Prediction Probabilities\n"
        f"True: "
        f"{', '.join(example['true_labels']) or 'None'} | "
        f"Predicted: "
        f"{', '.join(example['predicted_labels']) or 'None'}"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )
    axis.legend()

    for (
        index,
        probability,
        class_threshold,
    ) in zip(
        positions,
        probabilities,
        thresholds,
        strict=True,
    ):
        axis.text(
            index,
            min(probability + 0.035, 0.96),
            f"p={probability:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

        axis.text(
            index,
            max(class_threshold - 0.045, 0.02),
            f"t={class_threshold:.2f}",
            ha="center",
            va="top",
            fontsize=8,
        )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_arguments()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    output_dir = resolve_path(
        args.output_dir
    )

    if args.num_samples < 1:
        raise ValueError(
            "num_samples must be at least 1."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Method: {args.method_name}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output directory: {output_dir}")

    (
        model,
        checkpoint,
        thresholds,
        threshold_strategy,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    if args.threshold is not None:
        scalar_threshold = float(
            args.threshold
        )

        thresholds = np.full(
            len(DEFAULT_CLASS_NAMES),
            scalar_threshold,
            dtype=np.float32,
        )

        validate_thresholds(
            thresholds
        )

        threshold_strategy = (
            f"manual_fixed_{scalar_threshold:g}"
        )

    print(
        f"Threshold strategy: "
        f"{threshold_strategy}"
    )

    print("\nPrediction thresholds")
    for line in format_threshold_lines(
        thresholds
    ):
        print(line)

    save_model_summary(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_path=(
            output_dir
            / "model_summary.txt"
        ),
        method_name=args.method_name,
        thresholds=thresholds,
        threshold_strategy=(
            threshold_strategy
        ),
    )

    (
        test_metadata,
        test_dataset,
    ) = load_test_data()

    sample_count = min(
        args.num_samples,
        len(test_dataset),
    )

    selected_indices = sorted(
        random.sample(
            range(len(test_dataset)),
            sample_count,
        )
    )

    if not (
        0
        <= args.example_position
        < sample_count
    ):
        raise ValueError(
            "example_position must refer to one of the selected samples."
        )

    prediction_table, _ = run_predictions(
        model=model,
        dataset=test_dataset,
        test_metadata=test_metadata,
        selected_indices=selected_indices,
        thresholds=thresholds,
        threshold_strategy=(
            threshold_strategy
        ),
        device=device,
    )

    prediction_table.to_csv(
        output_dir
        / "sample_predictions.csv",
        index=False,
    )

    example_dataset_index = selected_indices[
        args.example_position
    ]

    (
        example_prediction_table,
        selected_example,
    ) = run_predictions(
        model=model,
        dataset=test_dataset,
        test_metadata=test_metadata,
        selected_indices=[
            example_dataset_index
        ],
        thresholds=thresholds,
        threshold_strategy=(
            threshold_strategy
        ),
        device=device,
    )

    save_ecg_figure(
        selected_example,
        output_dir
        / "example_ecg.png",
        args.method_name,
    )

    save_probability_figure(
        selected_example,
        output_dir
        / "example_probabilities.png",
        args.method_name,
    )

    print("\nModel")
    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"\nTotal model parameters: "
        f"{total_parameters:,}"
    )

    print(
        f"Saved checkpoint round: "
        f"{checkpoint.get('round', 'unknown')}"
    )

    print("\nSelected example")
    print(
        example_prediction_table[
            [
                "ecg_id",
                "true_labels",
                "predicted_labels",
                "exact_label_match",
                "threshold_strategy",
            ]
        ].to_string(
            index=False
        )
    )

    probability_columns = [
        "ecg_id",
    ]

    for class_name in DEFAULT_CLASS_NAMES:
        probability_columns.extend(
            [
                f"{class_name}_probability",
                f"{class_name}_threshold",
                f"{class_name}_predicted",
            ]
        )

    print(
        "\nSelected example probabilities, "
        "thresholds and decisions"
    )

    print(
        example_prediction_table[
            probability_columns
        ].to_string(
            index=False
        )
    )

    print("\nSaved files")
    for filename in (
        "model_summary.txt",
        "sample_predictions.csv",
        "example_ecg.png",
        "example_probabilities.png",
    ):
        print(
            output_dir
            / filename
        )


if __name__ == "__main__":
    main()