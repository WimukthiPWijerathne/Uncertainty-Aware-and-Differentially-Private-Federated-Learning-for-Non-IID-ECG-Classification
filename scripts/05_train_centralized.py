"""
Train a centralized 1D-CNN baseline on PTB-XL.

First run:
    2,000 training records
    500 validation records
    2 epochs

After confirming that the pipeline works, change DEBUG_MODE
to False for full centralized training.
"""

from __future__ import annotations

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import copy
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
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
    PROCESSED_METADATA_PATH,
    PTBXL_ROOT,
)
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch


# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------

SEED = 42

DEBUG_MODE = False

DEBUG_TRAIN_SAMPLES = 2_000
DEBUG_VALIDATION_SAMPLES = 500

DEBUG_EPOCHS = 2
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

NORMALIZE_PER_RECORD = False
CLASSIFICATION_THRESHOLD = 0.5



RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
LOG_DIR = RESULTS_DIR / "logs"
FIGURE_DIR = RESULTS_DIR / "figures"

BEST_MODEL_PATH = (
    CHECKPOINT_DIR / "centralized_best.pt"
)

LAST_MODEL_PATH = (
    CHECKPOINT_DIR / "centralized_last.pt"
)

HISTORY_PATH = (
    LOG_DIR / "centralized_history.csv"
)

LOSS_FIGURE_PATH = (
    FIGURE_DIR / "centralized_loss.png"
)

F1_FIGURE_PATH = (
    FIGURE_DIR / "centralized_macro_f1.png"
)


def set_random_seed(seed: int) -> None:
    """Configure random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories() -> None:
    """Create experiment output directories."""
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def select_debug_subset(
    metadata: pd.DataFrame,
    sample_count: int,
    seed: int,
) -> pd.DataFrame:
    """
    Randomly select a reproducible subset.

    If the available dataset is smaller than sample_count,
    all available records are returned.
    """
    actual_count = min(
        sample_count,
        len(metadata),
    )

    return metadata.sample(
        n=actual_count,
        random_state=seed,
    ).reset_index(drop=True)


def create_data_loaders(
    train_metadata: pd.DataFrame,
    validation_metadata: pd.DataFrame,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders."""
    train_dataset = PTBXLDataset(
        metadata=train_metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=NORMALIZE_PER_RECORD,
    )

    validation_dataset = PTBXLDataset(
        metadata=validation_metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=NORMALIZE_PER_RECORD,
    )

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, validation_loader


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: dict,
) -> None:
    """Save a resumable model checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_metrics["loss"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_macro_auroc": validation_metrics[
            "macro_auroc"
        ],
        "class_names": DEFAULT_CLASS_NAMES,
        "threshold": CLASSIFICATION_THRESHOLD,
    }

    torch.save(
        checkpoint,
        path,
    )


def save_history(
    history: list[dict],
    output_path: Path,
) -> None:
    """Save training history as CSV."""
    if not history:
        return

    field_names = list(
        history[0].keys()
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=field_names,
        )

        writer.writeheader()
        writer.writerows(history)


def plot_training_history(
    history: list[dict],
) -> None:
    """Create training-loss and validation-F1 figures."""
    epochs = [
        row["epoch"]
        for row in history
    ]

    train_losses = [
        row["train_loss"]
        for row in history
    ]

    validation_losses = [
        row["validation_loss"]
        for row in history
    ]

    macro_f1_scores = [
        row["validation_macro_f1"]
        for row in history
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        train_losses,
        marker="o",
        label="Training loss",
    )
    plt.plot(
        epochs,
        validation_losses,
        marker="o",
        label="Validation loss",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Centralized Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        LOSS_FIGURE_PATH,
        dpi=150,
    )
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        epochs,
        macro_f1_scores,
        marker="o",
        label="Validation macro-F1",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Macro-F1")
    plt.title("Centralized Validation Macro-F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        F1_FIGURE_PATH,
        dpi=150,
    )
    plt.close()


def main() -> None:
    set_random_seed(SEED)
    create_output_directories()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")
    print(f"Debug mode: {DEBUG_MODE}")
    print(f"Dataset root: {PTBXL_ROOT}")
    print(f"Metadata path: {PROCESSED_METADATA_PATH}")

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

    train_metadata = metadata[
        metadata["split"] == "train"
    ].copy()

    validation_metadata = metadata[
        metadata["split"] == "validation"
    ].copy()

    if train_metadata.empty:
        raise RuntimeError(
            "No training records were found."
        )

    if validation_metadata.empty:
        raise RuntimeError(
            "No validation records were found."
        )

    if DEBUG_MODE:
        train_metadata = select_debug_subset(
            metadata=train_metadata,
            sample_count=DEBUG_TRAIN_SAMPLES,
            seed=SEED,
        )

        validation_metadata = select_debug_subset(
            metadata=validation_metadata,
            sample_count=DEBUG_VALIDATION_SAMPLES,
            seed=SEED,
        )

        number_of_epochs = DEBUG_EPOCHS
    else:
        number_of_epochs = FULL_EPOCHS

    print(f"Training records: {len(train_metadata):,}")
    print(
        f"Validation records: "
        f"{len(validation_metadata):,}"
    )
    print(f"Epochs: {number_of_epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    train_loader, validation_loader = create_data_loaders(
        train_metadata=train_metadata,
        validation_metadata=validation_metadata,
        device=device,
    )

    # Change this constructor only if your current ECG1DCNN
    # uses different parameter names.
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

    history: list[dict] = []

    best_validation_loss = float("inf")
    best_model_state = None
    best_epoch = 0

    for epoch in range(1, number_of_epochs + 1):
        print(
            f"\nEpoch {epoch}/{number_of_epochs}"
        )

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

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
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
        }

        history.append(history_row)

        print(f"Training loss: {train_loss:.6f}")
        print(
            "Validation loss: "
            f"{validation_metrics['loss']:.6f}"
        )
        print(
            "Validation macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )
        print(
            "Validation weighted-F1: "
            f"{validation_metrics['weighted_f1']:.4f}"
        )
        print(
            "Validation macro-AUROC: "
            f"{validation_metrics['macro_auroc']:.4f}"
        )

        print("Per-class F1:")

        for class_name, score in zip(
            DEFAULT_CLASS_NAMES,
            validation_metrics["per_class_f1"],
        ):
            print(
                f"  {class_name}: {score:.4f}"
            )

        if (
            validation_metrics["loss"]
            < best_validation_loss
        ):
            best_validation_loss = validation_metrics["loss"]
            best_epoch = epoch

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            save_checkpoint(
                path=BEST_MODEL_PATH,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=validation_metrics,
            )

            print(
                f"Saved new best model: "
                f"{BEST_MODEL_PATH}"
            )

        save_history(
            history=history,
            output_path=HISTORY_PATH,
        )

    final_metrics = evaluate_model(
        model=model,
        data_loader=validation_loader,
        criterion=criterion,
        device=device,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    save_checkpoint(
        path=LAST_MODEL_PATH,
        model=model,
        optimizer=optimizer,
        epoch=number_of_epochs,
        validation_metrics=final_metrics,
    )

    if best_model_state is not None:
        model.load_state_dict(
            best_model_state
        )

    plot_training_history(history)

    print("\nTraining completed.")
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(f"Best model: {BEST_MODEL_PATH}")
    print(f"Last model: {LAST_MODEL_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"Loss figure: {LOSS_FIGURE_PATH}")
    print(f"F1 figure: {F1_FIGURE_PATH}")


if __name__ == "__main__":
    main()