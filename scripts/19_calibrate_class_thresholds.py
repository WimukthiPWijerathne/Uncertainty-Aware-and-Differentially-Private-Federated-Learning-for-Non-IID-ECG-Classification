
"""Calibrate one threshold per PTB-XL class on validation data only.

This does not retrain the model or change its weights.

Example:
uv run python scripts/19_calibrate_class_thresholds.py `
  --checkpoint results/checkpoints/dp_fedavg_rounds_50_local_epochs_1_noise_1_clip_1_epscap_8/dp_fedavg_best.pt `
  --method-name DP-FedAvg
"""

from __future__ import annotations

import argparse
import json
import math
import re
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
from sklearn.metrics import f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES, PTBXLDataset
from src.data.paths import PROCESSED_METADATA_PATH, PTBXL_ROOT
from src.federated.dp_client import create_dp_compatible_ecg_model
from src.models.cnn1d import ECG1DCNN


FIXED_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--method-name", type=str, required=True)
    parser.add_argument(
        "--architecture",
        choices=("auto", "standard", "dp-compatible"),
        default="auto",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(PROCESSED_METADATA_PATH, index_col="ecg_id")
    validation = metadata[metadata["split"] == "validation"].copy()
    test = metadata[metadata["split"] == "test"].copy()
    if validation.empty or test.empty:
        raise RuntimeError("Validation or test split is empty.")
    return validation, test


def create_loader(
    metadata: pd.DataFrame,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = PTBXLDataset(
        metadata=metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=False,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(batch, dict):
        x = batch.get("signal", batch.get("x"))
        y = batch.get("label", batch.get("y"))
    elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x, y = batch[0], batch[1]
    else:
        raise TypeError("Unsupported dataset batch format.")
    if x is None or y is None:
        raise ValueError("Could not extract signals and labels.")
    return torch.as_tensor(x).float(), torch.as_tensor(y).float()


def detect_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    has_batch_norm = any(
        key.endswith("running_mean") or key.endswith("running_var")
        for key in state_dict
    )
    return "standard" if has_batch_norm else "dp-compatible"


def create_model(kind: str, device: torch.device) -> torch.nn.Module:
    if kind == "standard":
        return ECG1DCNN(
            in_channels=12,
            num_classes=len(DEFAULT_CLASS_NAMES),
            dropout_p=0.3,
        ).to(device)
    if kind == "dp-compatible":
        return create_dp_compatible_ecg_model(device=device)
    raise ValueError(f"Unsupported architecture: {kind}")


def load_model(
    checkpoint_path: Path,
    requested_architecture: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], str]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    architecture = requested_architecture
    if architecture == "auto":
        architecture = detect_architecture(state_dict)
    model = create_model(architecture, device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model, checkpoint, architecture


def collect(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_probabilities = []
    all_targets = []
    with torch.no_grad():
        for batch in loader:
            signals, labels = unpack_batch(batch)
            logits = model(signals.to(device, non_blocking=True))
            all_probabilities.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(labels.cpu().numpy().astype(np.int64))
    return (
        np.concatenate(all_probabilities, axis=0),
        np.concatenate(all_targets, axis=0),
    )


def per_class_auroc(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    scores = []
    for index in range(targets.shape[1]):
        if np.unique(targets[:, index]).size < 2:
            scores.append(float("nan"))
        else:
            scores.append(
                roc_auc_score(targets[:, index], probabilities[:, index])
            )
    return np.asarray(scores, dtype=float)


def metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    predictions = (
        probabilities >= thresholds.reshape(1, -1)
    ).astype(np.int64)
    per_f1 = f1_score(
        targets,
        predictions,
        average=None,
        zero_division=0,
    )
    per_auc = per_class_auroc(targets, probabilities)
    return {
        "predictions": predictions,
        "per_class_f1": per_f1,
        "per_class_auroc": per_auc,
        "macro_f1": float(np.mean(per_f1)),
        "weighted_f1": float(
            f1_score(
                targets,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "macro_auroc": float(np.nanmean(per_auc)),
    }


def calibrate(
    targets: np.ndarray,
    probabilities: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    selected = []
    search_rows = []

    for class_index, class_name in enumerate(DEFAULT_CLASS_NAMES):
        scores = []
        for threshold in grid:
            prediction = (
                probabilities[:, class_index] >= threshold
            ).astype(np.int64)
            score = float(
                f1_score(
                    targets[:, class_index],
                    prediction,
                    zero_division=0,
                )
            )
            scores.append((float(threshold), score))
            search_rows.append(
                {
                    "class_name": class_name,
                    "threshold": float(threshold),
                    "validation_f1": score,
                }
            )

        best_score = max(score for _, score in scores)
        candidates = [
            threshold
            for threshold, score in scores
            if math.isclose(score, best_score, abs_tol=1e-12)
        ]
        selected.append(
            min(candidates, key=lambda value: abs(value - 0.5))
        )

    return np.asarray(selected, dtype=float), pd.DataFrame(search_rows)


def labels_as_text(binary_values: np.ndarray) -> str:
    names = [
        class_name
        for class_name, value in zip(
            DEFAULT_CLASS_NAMES,
            binary_values,
            strict=True,
        )
        if int(value) == 1
    ]
    return ", ".join(names) if names else "None"


def save_figures(
    thresholds: np.ndarray,
    test_fixed: dict[str, Any],
    test_calibrated: dict[str, Any],
    output_dir: Path,
    method_name: str,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(DEFAULT_CLASS_NAMES, thresholds)
    axis.axhline(0.5, linestyle="--", label="Original threshold = 0.50")
    axis.set_ylim(0, 1)
    axis.set_xlabel("Diagnostic superclass")
    axis.set_ylabel("Threshold")
    axis.set_title(f"{method_name}: Calibrated Thresholds")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "calibrated_thresholds.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)

    positions = np.arange(len(DEFAULT_CLASS_NAMES))
    width = 0.36
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        positions - width / 2,
        test_fixed["per_class_f1"],
        width,
        label="Fixed 0.50",
    )
    axis.bar(
        positions + width / 2,
        test_calibrated["per_class_f1"],
        width,
        label="Class-specific",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(DEFAULT_CLASS_NAMES)
    axis.set_ylim(0, 1)
    axis.set_xlabel("Diagnostic superclass")
    axis.set_ylabel("Test F1")
    axis.set_title(f"{method_name}: Test F1 Before and After Calibration")
    axis.grid(axis="y", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        output_dir / "test_per_class_f1_before_after.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()

    if not (
        0 < args.threshold_min
        < args.threshold_max
        < 1
    ):
        raise ValueError("Threshold limits must satisfy 0 < min < max < 1.")
    if args.threshold_step <= 0:
        raise ValueError("threshold-step must be positive.")

    checkpoint_path = resolve(args.checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    output_dir = (
        resolve(args.output_dir)
        if args.output_dir is not None
        else PROJECT_ROOT
        / "results"
        / "threshold_calibration"
        / slugify(args.method_name)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, architecture = load_model(
        checkpoint_path,
        args.architecture,
        device,
    )

    validation_metadata, test_metadata = load_splits()
    validation_loader = create_loader(
        validation_metadata,
        args.batch_size,
        args.num_workers,
        device,
    )
    test_loader = create_loader(
        test_metadata,
        args.batch_size,
        args.num_workers,
        device,
    )

    print(f"Method: {args.method_name}")
    print(f"Device: {device}")
    print(f"Architecture: {architecture}")
    print(f"Validation records: {len(validation_metadata):,}")
    print(f"Test records: {len(test_metadata):,}")

    validation_probabilities, validation_targets = collect(
        model,
        validation_loader,
        device,
    )

    grid = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2,
        args.threshold_step,
    )
    grid = np.round(grid[grid <= args.threshold_max + 1e-12], 10)

    thresholds, search_table = calibrate(
        validation_targets,
        validation_probabilities,
        grid,
    )

    fixed_thresholds = np.full(len(DEFAULT_CLASS_NAMES), 0.5)
    validation_fixed = metrics(
        validation_targets,
        validation_probabilities,
        fixed_thresholds,
    )
    validation_calibrated = metrics(
        validation_targets,
        validation_probabilities,
        thresholds,
    )

    test_probabilities, test_targets = collect(
        model,
        test_loader,
        device,
    )
    test_fixed = metrics(
        test_targets,
        test_probabilities,
        fixed_thresholds,
    )
    test_calibrated = metrics(
        test_targets,
        test_probabilities,
        thresholds,
    )

    threshold_rows = []
    test_class_rows = []

    for index, class_name in enumerate(DEFAULT_CLASS_NAMES):
        threshold_rows.append(
            {
                "class_name": class_name,
                "fixed_threshold": 0.5,
                "calibrated_threshold": thresholds[index],
                "validation_f1_fixed": (
                    validation_fixed["per_class_f1"][index]
                ),
                "validation_f1_calibrated": (
                    validation_calibrated["per_class_f1"][index]
                ),
            }
        )
        test_class_rows.append(
            {
                "class_name": class_name,
                "calibrated_threshold": thresholds[index],
                "test_f1_fixed": test_fixed["per_class_f1"][index],
                "test_f1_calibrated": (
                    test_calibrated["per_class_f1"][index]
                ),
                "test_f1_change": (
                    test_calibrated["per_class_f1"][index]
                    - test_fixed["per_class_f1"][index]
                ),
                "test_auroc": test_calibrated["per_class_auroc"][index],
            }
        )

    thresholds_table = pd.DataFrame(threshold_rows)
    test_class_table = pd.DataFrame(test_class_rows)

    metric_rows = []
    for split, threshold_type, result in (
        ("validation", "fixed_0.5", validation_fixed),
        ("validation", "class_specific", validation_calibrated),
        ("test", "fixed_0.5", test_fixed),
        ("test", "class_specific", test_calibrated),
    ):
        metric_rows.append(
            {
                "method": args.method_name,
                "split": split,
                "threshold_type": threshold_type,
                "macro_f1": result["macro_f1"],
                "weighted_f1": result["weighted_f1"],
                "macro_auroc": result["macro_auroc"],
            }
        )
    metrics_table = pd.DataFrame(metric_rows)

    prediction_rows = []
    for row_index, ecg_id in enumerate(test_metadata.index):
        row = {
            "ecg_id": ecg_id,
            "true_labels": labels_as_text(test_targets[row_index]),
            "predicted_labels": labels_as_text(
                test_calibrated["predictions"][row_index]
            ),
        }
        for class_index, class_name in enumerate(DEFAULT_CLASS_NAMES):
            row[f"{class_name}_probability"] = float(
                test_probabilities[row_index, class_index]
            )
            row[f"{class_name}_threshold"] = float(
                thresholds[class_index]
            )
            row[f"{class_name}_true"] = int(
                test_targets[row_index, class_index]
            )
            row[f"{class_name}_predicted"] = int(
                test_calibrated["predictions"][
                    row_index,
                    class_index,
                ]
            )
        prediction_rows.append(row)

    thresholds_table.to_csv(
        output_dir / "class_thresholds.csv",
        index=False,
    )
    search_table.to_csv(
        output_dir / "validation_threshold_search.csv",
        index=False,
    )
    metrics_table.to_csv(
        output_dir / "metrics_comparison.csv",
        index=False,
    )
    test_class_table.to_csv(
        output_dir / "test_per_class_comparison.csv",
        index=False,
    )
    pd.DataFrame(prediction_rows).to_csv(
        output_dir / "test_predictions_calibrated.csv",
        index=False,
    )

    save_figures(
        thresholds,
        test_fixed,
        test_calibrated,
        output_dir,
        args.method_name,
    )

    calibrated_checkpoint = dict(checkpoint)
    calibrated_checkpoint["class_thresholds"] = {
        class_name: float(threshold)
        for class_name, threshold in zip(
            DEFAULT_CLASS_NAMES,
            thresholds,
            strict=True,
        )
    }
    calibrated_checkpoint["threshold_calibration_split"] = "validation"
    calibrated_checkpoint["threshold_calibration_metric"] = "per-class F1"
    torch.save(
        calibrated_checkpoint,
        output_dir / "checkpoint_with_calibrated_thresholds.pt",
    )

    summary = {
        "method": args.method_name,
        "checkpoint": str(checkpoint_path),
        "architecture": architecture,
        "class_thresholds": {
            class_name: float(threshold)
            for class_name, threshold in zip(
                DEFAULT_CLASS_NAMES,
                thresholds,
                strict=True,
            )
        },
        "test_fixed_macro_f1": test_fixed["macro_f1"],
        "test_calibrated_macro_f1": test_calibrated["macro_f1"],
        "test_macro_f1_change": (
            test_calibrated["macro_f1"] - test_fixed["macro_f1"]
        ),
        "test_fixed_weighted_f1": test_fixed["weighted_f1"],
        "test_calibrated_weighted_f1": test_calibrated["weighted_f1"],
        "test_macro_auroc": test_calibrated["macro_auroc"],
    }
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nSelected validation thresholds")
    print(thresholds_table.to_string(index=False))
    print("\nOverall comparison")
    print(metrics_table.to_string(index=False))
    print("\nPer-class test comparison")
    print(test_class_table.to_string(index=False))
    print(
        "\nThe model was not retrained. Thresholds were selected only "
        "on validation data, then frozen before test evaluation."
    )
    print(f"\nSaved outputs: {output_dir}")


if __name__ == "__main__":
    main()