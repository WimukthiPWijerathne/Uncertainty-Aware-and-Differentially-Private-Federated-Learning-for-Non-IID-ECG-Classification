"""
Train one independent local CNN for each simulated hospital.

Each hospital:
1. Uses only its own local training partition.
2. Starts from a fresh model initialization.
3. Uses the common PTB-XL validation split for checkpoint selection.
4. Is evaluated on the same held-out PTB-XL test split.

Outputs:
    results/checkpoints/local/hospital_X_best.pt
    results/tables/local_baseline_results.csv
"""

from __future__ import annotations

import copy
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    DEFAULT_CLASS_NAMES,
    PTBXLDataset,
)
from src.data.paths import (
    PARTITIONS_DIR,
    PROCESSED_METADATA_PATH,
    PTBXL_ROOT,
)
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

NUM_HOSPITALS = 4

SEED = 42
EPOCHS = 3

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 0

CLASSIFICATION_THRESHOLD = 0.5
NORMALIZE_PER_RECORD = False

EARLY_STOPPING_PATIENCE = 3
MINIMUM_IMPROVEMENT = 1e-4

CHECKPOINT_DIR = (
    PROJECT_ROOT
    / "results"
    / "checkpoints"
    / "local"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "local_baseline_results.csv"
)


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loader(
    metadata: pd.DataFrame,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    """Create a PyTorch DataLoader from metadata."""
    dataset = PTBXLDataset(
        metadata=metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=NORMALIZE_PER_RECORD,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def load_global_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared validation and test metadata."""
    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    required_columns = {
        "split",
        "filename_lr",
        *DEFAULT_CLASS_NAMES,
    }

    missing_columns = required_columns.difference(
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            "Processed metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    validation_metadata = metadata[
        metadata["split"] == "validation"
    ].copy()

    test_metadata = metadata[
        metadata["split"] == "test"
    ].copy()

    if validation_metadata.empty:
        raise RuntimeError(
            "The validation split is empty."
        )

    if test_metadata.empty:
        raise RuntimeError(
            "The test split is empty."
        )

    return validation_metadata, test_metadata


def load_hospital_partition(
    hospital_id: int,
) -> pd.DataFrame:
    """Load one hospital's local training partition."""
    partition_path = (
        PARTITIONS_DIR
        / f"hospital_{hospital_id}.csv"
    )

    if not partition_path.is_file():
        raise FileNotFoundError(
            f"Hospital partition was not found: "
            f"{partition_path}"
        )

    hospital_metadata = pd.read_csv(
        partition_path
    )

    required_columns = {
        "ecg_id",
        "patient_id",
        "hospital_id",
        "filename_lr",
        "split",
        *DEFAULT_CLASS_NAMES,
    }

    missing_columns = required_columns.difference(
        hospital_metadata.columns
    )

    if missing_columns:
        raise ValueError(
            f"Hospital {hospital_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if not (
        hospital_metadata["hospital_id"] == hospital_id
    ).all():
        raise ValueError(
            f"Hospital ID mismatch inside {partition_path}"
        )

    if not (
        hospital_metadata["split"] == "train"
    ).all():
        raise ValueError(
            f"Non-training records were found in "
            f"hospital {hospital_id}"
        )

    return hospital_metadata


def save_checkpoint(
    checkpoint_path: Path,
    hospital_id: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    validation_metrics: dict,
) -> None:
    """Save the best local model checkpoint."""
    torch.save(
        {
            "hospital_id": hospital_id,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_metrics["loss"],
            "validation_macro_f1": validation_metrics[
                "macro_f1"
            ],
            "validation_weighted_f1": validation_metrics[
                "weighted_f1"
            ],
            "validation_macro_auroc": validation_metrics[
                "macro_auroc"
            ],
            "class_names": DEFAULT_CLASS_NAMES,
            "threshold": CLASSIFICATION_THRESHOLD,
        },
        checkpoint_path,
    )


def train_hospital_model(
    hospital_id: int,
    hospital_metadata: pd.DataFrame,
    validation_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Train and evaluate one hospital-specific local model.
    """
    print("\n" + "=" * 65)
    print(f"Training local model for Hospital {hospital_id}")
    print("=" * 65)

    print(
        f"Training records: "
        f"{len(hospital_metadata):,}"
    )

    print(
        f"Training patients: "
        f"{hospital_metadata['patient_id'].nunique():,}"
    )

    print("Class counts:")

    for class_name in DEFAULT_CLASS_NAMES:
        class_count = int(
            hospital_metadata[class_name].sum()
        )

        print(
            f"  {class_name}: {class_count:,}"
        )

    train_loader = create_data_loader(
        metadata=hospital_metadata,
        shuffle=True,
        device=device,
    )

    # Use a reproducible but different seed for each hospital.
    set_random_seed(
        SEED + hospital_id
    )

    model = ECG1DCNN(
        in_channels=12,
        num_classes=len(DEFAULT_CLASS_NAMES),
        dropout_p=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    checkpoint_path = (
        CHECKPOINT_DIR
        / f"hospital_{hospital_id}_best.pt"
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    epochs_without_improvement = 0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_metrics = evaluate_model(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Training loss: "
            f"{train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{validation_metrics['loss']:.6f}"
        )

        print(
            f"Validation macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Validation weighted-F1: "
            f"{validation_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Validation macro-AUROC: "
            f"{validation_metrics['macro_auroc']:.4f}"
        )

        if (
            validation_metrics["loss"]
            < best_validation_loss
            - MINIMUM_IMPROVEMENT
        ):
            best_validation_loss = (
                validation_metrics["loss"]
            )

            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                hospital_id=hospital_id,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                validation_metrics=validation_metrics,
            )

            print(
                f"Saved new best checkpoint: "
                f"{checkpoint_path}"
            )

        else:
            epochs_without_improvement += 1

            print(
                "No validation-loss improvement. "
                f"Patience: "
                f"{epochs_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

            if (
                epochs_without_improvement
                >= EARLY_STOPPING_PATIENCE
            ):
                print(
                    "Early stopping triggered."
                )
                break

    if best_model_state is None:
        raise RuntimeError(
            f"No checkpoint was saved for hospital "
            f"{hospital_id}."
        )

    model.load_state_dict(
        best_model_state
    )

    test_metrics = evaluate_model(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    print("\nCommon test-set results")

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Test loss: "
        f"{test_metrics['loss']:.6f}"
    )

    print(
        f"Test macro-F1: "
        f"{test_metrics['macro_f1']:.4f}"
    )

    print(
        f"Test weighted-F1: "
        f"{test_metrics['weighted_f1']:.4f}"
    )

    print(
        f"Test macro-AUROC: "
        f"{test_metrics['macro_auroc']:.4f}"
    )

    print("Per-class test results:")

    for class_name, f1_score, auroc_score in zip(
        DEFAULT_CLASS_NAMES,
        test_metrics["per_class_f1"],
        test_metrics["per_class_auroc"],
    ):
        print(
            f"  {class_name}: "
            f"F1={f1_score:.4f}, "
            f"AUROC={auroc_score:.4f}"
        )

    result_row = {
        "hospital_id": hospital_id,
        "training_records": len(
            hospital_metadata
        ),
        "training_patients": hospital_metadata[
            "patient_id"
        ].nunique(),
        "best_epoch": best_epoch,
        "best_validation_loss": (
            best_validation_loss
        ),
        "test_loss": test_metrics["loss"],
        "test_macro_f1": test_metrics[
            "macro_f1"
        ],
        "test_weighted_f1": test_metrics[
            "weighted_f1"
        ],
        "test_macro_auroc": test_metrics[
            "macro_auroc"
        ],
    }

    for class_name, f1_score, auroc_score in zip(
        DEFAULT_CLASS_NAMES,
        test_metrics["per_class_f1"],
        test_metrics["per_class_auroc"],
    ):
        result_row[
            f"{class_name}_test_f1"
        ] = f1_score

        result_row[
            f"{class_name}_test_auroc"
        ] = auroc_score

    return result_row


def main() -> None:
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_random_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Number of hospitals: {NUM_HOSPITALS}")
    print(f"Epochs per hospital: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(
        f"Shared validation threshold: "
        f"{CLASSIFICATION_THRESHOLD}"
    )

    validation_metadata, test_metadata = (
        load_global_splits()
    )

    print(
        f"Shared validation records: "
        f"{len(validation_metadata):,}"
    )

    print(
        f"Shared test records: "
        f"{len(test_metadata):,}"
    )

    validation_loader = create_data_loader(
        metadata=validation_metadata,
        shuffle=False,
        device=device,
    )

    test_loader = create_data_loader(
        metadata=test_metadata,
        shuffle=False,
        device=device,
    )

    result_rows: list[dict] = []

    for hospital_id in range(NUM_HOSPITALS):
        hospital_metadata = (
            load_hospital_partition(
                hospital_id
            )
        )

        result_row = train_hospital_model(
            hospital_id=hospital_id,
            hospital_metadata=hospital_metadata,
            validation_loader=validation_loader,
            test_loader=test_loader,
            device=device,
        )

        result_rows.append(
            result_row
        )

        # Save progress after every hospital.
        pd.DataFrame(
            result_rows
        ).to_csv(
            RESULTS_PATH,
            index=False,
        )

        print(
            f"\nProgress saved to: "
            f"{RESULTS_PATH}"
        )

    results = pd.DataFrame(
        result_rows
    )

    print("\n" + "=" * 65)
    print("Local-only baseline summary")
    print("=" * 65)

    print(
        results[
            [
                "hospital_id",
                "training_records",
                "best_epoch",
                "test_macro_f1",
                "test_weighted_f1",
                "test_macro_auroc",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nAll local results saved to: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Checkpoints saved under: "
        f"{CHECKPOINT_DIR}"
    )


if __name__ == "__main__":
    main()