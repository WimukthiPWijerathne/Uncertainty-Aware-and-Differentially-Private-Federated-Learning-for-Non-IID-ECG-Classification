"""
Evaluate the best centralized model on the held-out PTB-XL test split.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    DEFAULT_CLASS_NAMES,
    PTBXLDataset,
)
from src.data.paths import (
    PROCESSED_METADATA_PATH,
    PTBXL_ROOT,
)
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model


BATCH_SIZE = 64
NUM_WORKERS = 0
CLASSIFICATION_THRESHOLD = 0.5
NORMALIZE_PER_RECORD = False

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "centralized_best.pt"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "centralized_test_metrics.csv"
)


def main() -> None:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    test_metadata = metadata[
        metadata["split"] == "test"
    ].copy()

    if test_metadata.empty:
        raise RuntimeError(
            "No test records were found."
        )

    print(f"Test records: {len(test_metadata):,}")

    test_dataset = PTBXLDataset(
        metadata=test_metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=NORMALIZE_PER_RECORD,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = ECG1DCNN(
        in_channels=12,
        num_classes=len(DEFAULT_CLASS_NAMES),
        dropout_p=0.3,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    criterion = nn.BCEWithLogitsLoss()

    metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    print("\nCentralized test results")
    print(f"Test loss: {metrics['loss']:.6f}")
    print(
        f"Macro-F1: {metrics['macro_f1']:.4f}"
    )
    print(
        f"Weighted-F1: {metrics['weighted_f1']:.4f}"
    )
    print(
        f"Macro-AUROC: {metrics['macro_auroc']:.4f}"
    )

    print("\nPer-class results")

    for class_name, f1_value, auroc_value in zip(
        DEFAULT_CLASS_NAMES,
        metrics["per_class_f1"],
        metrics["per_class_auroc"],
    ):
        print(
            f"{class_name}: "
            f"F1={f1_value:.4f}, "
            f"AUROC={auroc_value:.4f}"
        )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for class_name, f1_value, auroc_value in zip(
        DEFAULT_CLASS_NAMES,
        metrics["per_class_f1"],
        metrics["per_class_auroc"],
    ):
        rows.append(
            {
                "metric_scope": class_name,
                "loss": "",
                "f1": f1_value,
                "weighted_f1": "",
                "auroc": auroc_value,
                "threshold": CLASSIFICATION_THRESHOLD,
            }
        )

    rows.insert(
        0,
        {
            "metric_scope": "overall",
            "loss": metrics["loss"],
            "f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "auroc": metrics["macro_auroc"],
            "threshold": CLASSIFICATION_THRESHOLD,
        },
    )

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "metric_scope",
                "loss",
                "f1",
                "weighted_f1",
                "auroc",
                "threshold",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        f"\nSaved test metrics to: {RESULTS_PATH}"
    )


if __name__ == "__main__":
    main()