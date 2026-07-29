"""
Run a manual FedAvg experiment across four simulated hospitals.

Workflow:
1. Initialize one global ECG1DCNN.
2. Send global parameters to every hospital.
3. Train each hospital locally.
4. Aggregate updates using sample-weighted FedAvg.
5. Evaluate the global model on the shared validation set.
6. Save round history and the best global checkpoint.
7. Evaluate the best checkpoint on the shared test set.

This is a manual FedAvg implementation and does not require
Flower simulation.
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
from src.federated.client import HospitalClient
from src.federated.fedavg import aggregate_fedavg
from src.federated.parameter_utils import (
    get_model_parameters,
    set_model_parameters,
)
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model


# =========================================================
# Experiment configuration
# =========================================================

SEED = 42

NUM_HOSPITALS = 4
MAX_ROUNDS = 50

EARLY_STOPPING_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4
LOCAL_EPOCHS = 1

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

CLASSIFICATION_THRESHOLD = 0.5
NORMALIZE_PER_RECORD = False

RUN_NAME = (
    f"fedavg_earlystop_max_{MAX_ROUNDS}"
    f"_local_epochs_{LOCAL_EPOCHS}"
)

RESULTS_DIR = PROJECT_ROOT / "results"

CHECKPOINT_DIR = (
    RESULTS_DIR
    / "checkpoints"
    / RUN_NAME
)

LOG_DIR = (
    RESULTS_DIR
    / "logs"
    / RUN_NAME
)

TABLE_DIR = (
    RESULTS_DIR
    / "tables"
)

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "fedavg_best.pt"
)

LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "fedavg_last.pt"
)

ROUND_HISTORY_PATH = (
    LOG_DIR
    / "fedavg_round_history.csv"
)

CLIENT_HISTORY_PATH = (
    LOG_DIR
    / "fedavg_client_history.csv"
)

TEST_RESULTS_PATH = (
    TABLE_DIR
    / f"{RUN_NAME}_test_results.csv"
)

CONFIG_PATH = (
    TABLE_DIR
    / f"{RUN_NAME}_config.csv"
)


# =========================================================
# General utilities
# =========================================================

def set_random_seed(seed: int) -> None:
    """Configure reproducible random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories() -> None:
    """Create all required output directories."""
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_evaluation_loader(
    metadata: pd.DataFrame,
    device: torch.device,
) -> DataLoader:
    """Create a shared validation or test DataLoader."""
    dataset = PTBXLDataset(
        metadata=metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=NORMALIZE_PER_RECORD,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


# =========================================================
# Metadata loading
# =========================================================

def load_global_metadata(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the shared validation and test splits."""
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
            "The shared validation split is empty."
        )

    if test_metadata.empty:
        raise RuntimeError(
            "The shared test split is empty."
        )

    return (
        validation_metadata,
        test_metadata,
    )


def load_hospital_metadata(
    hospital_id: int,
) -> pd.DataFrame:
    """Load one hospital's local training partition."""
    partition_path = (
        PARTITIONS_DIR
        / f"hospital_{hospital_id}.csv"
    )

    if not partition_path.is_file():
        raise FileNotFoundError(
            f"Hospital partition not found: "
            f"{partition_path}"
        )

    metadata = pd.read_csv(
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
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            f"Hospital {hospital_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if not (
        metadata["hospital_id"] == hospital_id
    ).all():
        raise ValueError(
            f"Hospital ID mismatch in {partition_path}"
        )

    if not (
        metadata["split"] == "train"
    ).all():
        raise ValueError(
            f"Non-training records found in "
            f"hospital {hospital_id}"
        )

    return metadata


# =========================================================
# Client creation
# =========================================================

def create_clients(
    validation_metadata: pd.DataFrame,
    device: torch.device,
) -> list[HospitalClient]:
    """Create one federated client for each hospital."""
    clients: list[HospitalClient] = []

    for hospital_id in range(NUM_HOSPITALS):
        hospital_metadata = load_hospital_metadata(
            hospital_id
        )

        print(
            f"Hospital {hospital_id}: "
            f"{len(hospital_metadata):,} records, "
            f"{hospital_metadata['patient_id'].nunique():,} patients"
        )

        client = HospitalClient(
            hospital_id=hospital_id,
            train_metadata=hospital_metadata,
            validation_metadata=validation_metadata,
            device=device,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            local_epochs=LOCAL_EPOCHS,
            num_workers=NUM_WORKERS,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        clients.append(client)

    return clients


# =========================================================
# Saving
# =========================================================

def save_global_checkpoint(
    path: Path,
    model: nn.Module,
    round_number: int,
    validation_metrics: dict,
) -> None:
    """Save a global FedAvg model checkpoint."""
    torch.save(
        {
            "round": round_number,
            "model_state_dict": model.state_dict(),
            "validation_loss": validation_metrics[
                "loss"
            ],
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
            "num_hospitals": NUM_HOSPITALS,
            "local_epochs": LOCAL_EPOCHS,
        },
        path,
    )


def save_configuration(
    device: torch.device,
) -> None:
    """Save experiment settings to CSV."""
    configuration = pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "seed": SEED,
                "num_hospitals": NUM_HOSPITALS,
                "max_rounds": MAX_ROUNDS,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "minimum_improvement": MINIMUM_IMPROVEMENT,
                "local_epochs": LOCAL_EPOCHS,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "num_workers": NUM_WORKERS,
                "classification_threshold": (
                    CLASSIFICATION_THRESHOLD
                ),
                "normalize_per_record": (
                    NORMALIZE_PER_RECORD
                ),
                "device": str(device),
            }
        ]
    )

    configuration.to_csv(
        CONFIG_PATH,
        index=False,
    )


# =========================================================
# Main FedAvg experiment
# =========================================================

def main() -> None:
    set_random_seed(SEED)
    create_output_directories()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(f"Number of hospitals: {NUM_HOSPITALS}")
    print(
        f"Maximum federated rounds: {MAX_ROUNDS}"
    )
    print(
        f"Early-stopping patience: "
        f"{EARLY_STOPPING_PATIENCE}"
    )
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")

    save_configuration(device)

    validation_metadata, test_metadata = (
        load_global_metadata()
    )

    print(
        f"Shared validation records: "
        f"{len(validation_metadata):,}"
    )

    print(
        f"Shared test records: "
        f"{len(test_metadata):,}"
    )

    validation_loader = create_evaluation_loader(
        metadata=validation_metadata,
        device=device,
    )

    test_loader = create_evaluation_loader(
        metadata=test_metadata,
        device=device,
    )

    clients = create_clients(
        validation_metadata=validation_metadata,
        device=device,
    )

    global_model = ECG1DCNN(
        in_channels=12,
        num_classes=len(DEFAULT_CLASS_NAMES),
        dropout_p=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    global_parameters = get_model_parameters(
        global_model
    )

    round_history: list[dict] = []
    client_history: list[dict] = []

    best_validation_loss = float("inf")
    best_round = 0
    best_global_state = None

    rounds_without_improvement = 0
    stopped_early = False
    rounds_completed = 0

    for round_number in range(
        1,
        MAX_ROUNDS + 1,
    ):
        rounds_completed = round_number

        print("\n" + "=" * 70)
        print(
            f"Federated round "
            f"{round_number}/{MAX_ROUNDS}"
        )
        print("=" * 70)

        client_aggregation_results = []

        for client in clients:
            print(
                f"\nTraining Hospital "
                f"{client.hospital_id}"
            )

            (
                updated_parameters,
                sample_count,
                client_metrics,
            ) = client.fit(
                global_parameters
            )

            client_aggregation_results.append(
                (
                    updated_parameters,
                    sample_count,
                )
            )

            client_row = {
                "round": round_number,
                "hospital_id": (
                    client.hospital_id
                ),
                "sample_count": sample_count,
                "train_loss": client_metrics[
                    "train_loss"
                ],
                "validation_loss": client_metrics[
                    "validation_loss"
                ],
                "validation_macro_f1": (
                    client_metrics[
                        "validation_macro_f1"
                    ]
                ),
                "validation_macro_auroc": (
                    client_metrics[
                        "validation_macro_auroc"
                    ]
                ),
            }

            client_history.append(
                client_row
            )

            print(
                f"Hospital {client.hospital_id} | "
                f"samples={sample_count:,} | "
                f"train loss="
                f"{client_metrics['train_loss']:.4f} | "
                f"validation loss="
                f"{client_metrics['validation_loss']:.4f} | "
                f"macro-F1="
                f"{client_metrics['validation_macro_f1']:.4f}"
            )

        global_parameters = aggregate_fedavg(
            client_aggregation_results
        )

        set_model_parameters(
            global_model,
            global_parameters,
        )

        global_validation_metrics = evaluate_model(
            model=global_model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        print("\nGlobal validation results")

        print(
            f"Loss: "
            f"{global_validation_metrics['loss']:.6f}"
        )

        print(
            f"Macro-F1: "
            f"{global_validation_metrics['macro_f1']:.4f}"
        )

        print(
            f"Weighted-F1: "
            f"{global_validation_metrics['weighted_f1']:.4f}"
        )

        print(
            f"Macro-AUROC: "
            f"{global_validation_metrics['macro_auroc']:.4f}"
        )

        round_row = {
            "round": round_number,
            "global_validation_loss": (
                global_validation_metrics["loss"]
            ),
            "global_validation_macro_f1": (
                global_validation_metrics[
                    "macro_f1"
                ]
            ),
            "global_validation_weighted_f1": (
                global_validation_metrics[
                    "weighted_f1"
                ]
            ),
            "global_validation_macro_auroc": (
                global_validation_metrics[
                    "macro_auroc"
                ]
            ),
        }

        for client_row in client_history:
            if client_row["round"] == round_number:
                hospital_id = client_row[
                    "hospital_id"
                ]

                round_row[
                    f"hospital_{hospital_id}_train_loss"
                ] = client_row["train_loss"]

                round_row[
                    f"hospital_{hospital_id}_validation_loss"
                ] = client_row[
                    "validation_loss"
                ]

        round_history.append(
            round_row
        )

        pd.DataFrame(
            client_history
        ).to_csv(
            CLIENT_HISTORY_PATH,
            index=False,
        )

        pd.DataFrame(
            client_history
        ).to_csv(
            CLIENT_HISTORY_PATH,
            index=False,
        )
        improved = (
            global_validation_metrics["loss"]
            < best_validation_loss
            - MINIMUM_IMPROVEMENT
        )
        round_history[-1]["improved"] = improved

        if improved:
            best_validation_loss = (
                global_validation_metrics["loss"]
            )

            best_round = round_number

            best_global_state = copy.deepcopy(
                global_model.state_dict()
            )

            rounds_without_improvement = 0

            save_global_checkpoint(
                path=BEST_CHECKPOINT_PATH,
                model=global_model,
                round_number=round_number,
                validation_metrics=(
                    global_validation_metrics
                ),
            )

            print(
                f"Saved new best global model: "
                f"{BEST_CHECKPOINT_PATH}"
            )

        else:
            rounds_without_improvement += 1

            print(
                "No sufficient validation-loss improvement. "
                f"Patience: {rounds_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        round_history[-1][
            "rounds_without_improvement"
        ] = rounds_without_improvement

        pd.DataFrame(
            round_history
        ).to_csv(
            ROUND_HISTORY_PATH,
            index=False,
        )

        if (
            rounds_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True

            print(
                "\nFederated early stopping triggered."
            )
            print(
                f"Best round was {best_round} with "
                f"validation loss "
                f"{best_validation_loss:.6f}."
            )

            break

    last_validation_metrics = evaluate_model(
        model=global_model,
        data_loader=validation_loader,
        criterion=criterion,
        device=device,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    save_global_checkpoint(
        path=LAST_CHECKPOINT_PATH,
        model=global_model,
        round_number=rounds_completed,
        validation_metrics=last_validation_metrics,
    )

    if best_global_state is None:
        raise RuntimeError(
            "No best global model was saved."
        )

    global_model.load_state_dict(
        best_global_state
    )

    test_metrics = evaluate_model(
        model=global_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=CLASSIFICATION_THRESHOLD,
    )

    print("\n" + "=" * 70)
    print("Best FedAvg global test results")
    print("=" * 70)

    print(
    f"Rounds completed: {rounds_completed}"
    )
    print(
        f"Stopped early: {stopped_early}"
    )
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
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

    print("\nPer-class test results")

    result_rows = [
        {
            "metric_scope": "overall",
            "best_round": best_round,
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
            "test_class_f1": "",
            "test_class_auroc": "",
        }
    ]

    for (
        class_name,
        class_f1,
        class_auroc,
    ) in zip(
        DEFAULT_CLASS_NAMES,
        test_metrics["per_class_f1"],
        test_metrics["per_class_auroc"],
    ):
        print(
            f"{class_name}: "
            f"F1={class_f1:.4f}, "
            f"AUROC={class_auroc:.4f}"
        )

        result_rows.append(
            {
                "metric_scope": class_name,
                "best_round": best_round,
                "rounds_completed": "",
                "stopped_early": "",
                "best_validation_loss": "",
                "test_loss": "",
                "test_macro_f1": "",
                "test_weighted_f1": "",
                "test_macro_auroc": "",
                "test_class_f1": class_f1,
                "test_class_auroc": class_auroc,
            }
        )

    pd.DataFrame(
        result_rows
    ).to_csv(
        TEST_RESULTS_PATH,
        index=False,
    )

    print("\nFedAvg experiment completed.")
    print(
        f"Best checkpoint: "
        f"{BEST_CHECKPOINT_PATH}"
    )
    print(
        f"Last checkpoint: "
        f"{LAST_CHECKPOINT_PATH}"
    )
    print(
        f"Round history: "
        f"{ROUND_HISTORY_PATH}"
    )
    print(
        f"Client history: "
        f"{CLIENT_HISTORY_PATH}"
    )
    print(
        f"Test results: "
        f"{TEST_RESULTS_PATH}"
    )
    print(
        f"Configuration: "
        f"{CONFIG_PATH}"
    )


if __name__ == "__main__":
    main()