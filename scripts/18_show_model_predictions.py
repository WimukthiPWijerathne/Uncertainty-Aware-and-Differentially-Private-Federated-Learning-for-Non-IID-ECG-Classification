"""
Load the best DP-UA-FedAvg checkpoint, show the model architecture,
run predictions on held-out PTB-XL test records, and save clear outputs
for the final notebook and presentation.

Outputs
-------
results/predictions/dp_ua_fedavg/
    model_summary.txt
    sample_predictions.csv
    example_ecg.png
    example_probabilities.png

Usage
-----
uv run python scripts/18_show_model_predictions.py

Choose a different number of examples:
uv run python scripts/18_show_model_predictions.py --num-samples 12

Choose an exact checkpoint:
uv run python scripts/18_show_model_predictions.py ^
    --checkpoint results/checkpoints/<run>/dp_ua_fedavg_best.pt
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
    / "checkpoints"
    / (
        "dp_ua_fedavg_rounds_50_local_epochs_1_"
        "noise_1_clip_1_mc_10_epscap_8"
    )
    / "dp_ua_fedavg_best.pt"
)

DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "predictions"
    / "dp_ua_fedavg"
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
            "Generate clear model and prediction outputs from the "
            "best DP-UA-FedAvg checkpoint."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Prediction threshold. When omitted, the value saved in "
            "the checkpoint is used, otherwise 0.5 is used."
        ),
    )

    parser.add_argument(
        "--example-position",
        type=int,
        default=0,
        help=(
            "Position within the randomly selected sample list to use "
            "for the ECG and probability figures."
        ),
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], float]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
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

    threshold = float(
        checkpoint.get(
            "threshold",
            0.5,
        )
    )

    return model, checkpoint, threshold


def save_model_summary(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    checkpoint_path: Path,
    output_path: Path,
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
        "DP-UA-FedAvg Model Summary",
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
        f"Threshold: {checkpoint.get('threshold', 0.5)}",
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


def load_test_data() -> tuple[pd.DataFrame, PTBXLDataset]:
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
    elif isinstance(item, (tuple, list)) and len(item) >= 2:
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
    threshold: float,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    first_example: dict[str, Any] | None = None

    ecg_ids = list(
        test_metadata.index
    )

    with torch.no_grad():
        for position, dataset_index in enumerate(
            selected_indices
        ):
            signal, target = unpack_dataset_item(
                dataset[dataset_index]
            )

            logits = model(
                signal.unsqueeze(0).to(device)
            )

            probabilities = torch.sigmoid(
                logits
            ).squeeze(0).cpu().numpy()

            target_values = (
                target.cpu().numpy().astype(int)
            )

            predicted_values = (
                probabilities >= threshold
            ).astype(int)

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
                "threshold": threshold,
            }

            for class_name, probability, true_value, predicted_value in zip(
                DEFAULT_CLASS_NAMES,
                probabilities,
                target_values,
                predicted_values,
                strict=True,
            ):
                row[
                    f"{class_name}_probability"
                ] = float(probability)

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
    offsets = np.arange(
        len(LEAD_NAMES)
    )[::-1] * vertical_spacing

    figure, axis = plt.subplots(
        figsize=(12, 8)
    )

    for lead_index, lead_name in enumerate(
        LEAD_NAMES
    ):
        axis.plot(
            time_seconds,
            signal[lead_index] + offsets[lead_index],
            linewidth=0.8,
        )

    axis.set_yticks(offsets)
    axis.set_yticklabels(LEAD_NAMES)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("ECG lead")
    axis.set_title(
        f"Example 12-Lead ECG — ECG ID {example['ecg_id']}"
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
    threshold: float,
    output_path: Path,
) -> None:
    probabilities = np.asarray(
        example["probabilities"],
        dtype=float,
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        DEFAULT_CLASS_NAMES,
        probabilities,
    )

    axis.axhline(
        threshold,
        linestyle="--",
        label=(
            f"Decision threshold = "
            f"{threshold:.2f}"
        ),
    )

    axis.set_ylim(0, 1)
    axis.set_xlabel("Diagnostic superclass")
    axis.set_ylabel("Predicted probability")
    axis.set_title(
        "DP-UA-FedAvg Prediction Probabilities\n"
        f"True: {', '.join(example['true_labels']) or 'None'} | "
        f"Predicted: {', '.join(example['predicted_labels']) or 'None'}"
    )
    axis.grid(
        axis="y",
        alpha=0.3,
    )
    axis.legend()

    for index, probability in enumerate(
        probabilities
    ):
        axis.text(
            index,
            min(probability + 0.03, 0.97),
            f"{probability:.3f}",
            ha="center",
            va="bottom",
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(
        f"Checkpoint: {checkpoint_path}"
    )
    print(
        f"Output directory: {output_dir}"
    )

    (
        model,
        checkpoint,
        checkpoint_threshold,
    ) = load_checkpoint(
        checkpoint_path,
        device,
    )

    threshold = (
        checkpoint_threshold
        if args.threshold is None
        else float(args.threshold)
    )

    if not 0 < threshold < 1:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    save_model_summary(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        output_path=(
            output_dir
            / "model_summary.txt"
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

    if not 0 <= args.example_position < sample_count:
        raise ValueError(
            "example_position must refer to one of the selected samples."
        )

    prediction_table, first_example = run_predictions(
        model=model,
        dataset=test_dataset,
        test_metadata=test_metadata,
        selected_indices=selected_indices,
        threshold=threshold,
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

    example_prediction_table, selected_example = run_predictions(
        model=model,
        dataset=test_dataset,
        test_metadata=test_metadata,
        selected_indices=[
            example_dataset_index
        ],
        threshold=threshold,
        device=device,
    )

    save_ecg_figure(
        selected_example,
        output_dir
        / "example_ecg.png",
    )

    save_probability_figure(
        selected_example,
        threshold,
        output_dir
        / "example_probabilities.png",
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

    print(
        f"Prediction threshold: "
        f"{threshold:.2f}"
    )

    print(
        "\nSelected example"
    )

    print(
        example_prediction_table[
            [
                "ecg_id",
                "true_labels",
                "predicted_labels",
                "exact_label_match",
            ]
        ].to_string(
            index=False
        )
    )

    probability_columns = [
        "ecg_id",
        *[
            f"{class_name}_probability"
            for class_name
            in DEFAULT_CLASS_NAMES
        ],
    ]

    print(
        "\nSelected example probabilities"
    )

    print(
        example_prediction_table[
            probability_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSaved files"
    )
    print(
        output_dir
        / "model_summary.txt"
    )
    print(
        output_dir
        / "sample_predictions.csv"
    )
    print(
        output_dir
        / "example_ecg.png"
    )
    print(
        output_dir
        / "example_probabilities.png"
    )


if __name__ == "__main__":
    main()