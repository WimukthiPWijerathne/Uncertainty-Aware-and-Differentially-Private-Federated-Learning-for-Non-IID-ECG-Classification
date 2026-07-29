"""
Run a non-private FedAvg control using the same Opacus-compatible
model architecture as DP-FedAvg.

Purpose:
Separate performance changes caused by architecture conversion from
changes caused by DP clipping and Gaussian noise.
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
from src.federated.compatible_client import (
    CompatibleHospitalClient,
)
from src.federated.dp_client import (
    create_dp_compatible_ecg_model,
)
from src.federated.fedavg import aggregate_fedavg
from src.federated.parameter_utils import (
    get_model_parameters,
    set_model_parameters,
)
from src.training.evaluate import evaluate_model


SEED = 42

NUM_HOSPITALS = 4
MAX_ROUNDS = 50
LOCAL_EPOCHS = 1

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_WORKERS = 2

EARLY_STOPPING_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4

CLASSIFICATION_THRESHOLD = 0.5
NORMALIZE_PER_RECORD = False

RUN_NAME = (
    f"compatible_fedavg_rounds_{MAX_ROUNDS}"
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
TABLE_DIR = RESULTS_DIR / "tables"

BEST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "compatible_fedavg_best.pt"
)
LAST_CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "compatible_fedavg_last.pt"
)
ROUND_HISTORY_PATH = (
    LOG_DIR
    / "compatible_fedavg_round_history.csv"
)
CLIENT_HISTORY_PATH = (
    LOG_DIR
    / "compatible_fedavg_client_history.csv"
)
TEST_RESULTS_PATH = (
    TABLE_DIR
    / f"{RUN_NAME}_test_results.csv"
)
CONFIG_PATH = (
    TABLE_DIR
    / f"{RUN_NAME}_config.csv"
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories() -> None:
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


def load_global_metadata(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
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

    return (
        validation_metadata,
        test_metadata,
    )


def load_hospital_metadata(
    hospital_id: int,
) -> pd.DataFrame:
    partition_path = (
        PARTITIONS_DIR
        / f"hospital_{hospital_id}.csv"
    )

    if not partition_path.is_file():
        raise FileNotFoundError(
            f"Partition not found: {partition_path}"
        )

    metadata = pd.read_csv(
        partition_path
    )

    if not (
        metadata["hospital_id"] == hospital_id
    ).all():
        raise ValueError(
            f"Hospital ID mismatch in "
            f"{partition_path}"
        )

    if not (
        metadata["split"] == "train"
    ).all():
        raise ValueError(
            f"Non-training records found in "
            f"{partition_path}"
        )

    return metadata


def create_clients(
    validation_metadata: pd.DataFrame,
    device: torch.device,
) -> list[CompatibleHospitalClient]:
    clients: list[
        CompatibleHospitalClient
    ] = []

    for hospital_id in range(
        NUM_HOSPITALS
    ):
        hospital_metadata = (
            load_hospital_metadata(
                hospital_id
            )
        )

        print(
            f"Hospital {hospital_id}: "
            f"{len(hospital_metadata):,} records, "
            f"{hospital_metadata['patient_id'].nunique():,} patients"
        )

        clients.append(
            CompatibleHospitalClient(
                hospital_id=hospital_id,
                train_metadata=(
                    hospital_metadata
                ),
                validation_metadata=(
                    validation_metadata
                ),
                device=device,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                local_epochs=LOCAL_EPOCHS,
                num_workers=NUM_WORKERS,
                threshold=(
                    CLASSIFICATION_THRESHOLD
                ),
                normalize_per_record=(
                    NORMALIZE_PER_RECORD
                ),
            )
        )

    return clients


def save_checkpoint(
    path: Path,
    model: nn.Module,
    round_number: int,
    validation_metrics: dict,
) -> None:
    torch.save(
        {
            "round": round_number,
            "model_state_dict": (
                model.state_dict()
            ),
            "validation_loss": (
                validation_metrics["loss"]
            ),
            "validation_macro_f1": (
                validation_metrics[
                    "macro_f1"
                ]
            ),
            "validation_weighted_f1": (
                validation_metrics[
                    "weighted_f1"
                ]
            ),
            "validation_macro_auroc": (
                validation_metrics[
                    "macro_auroc"
                ]
            ),
            "class_names": (
                DEFAULT_CLASS_NAMES
            ),
            "threshold": (
                CLASSIFICATION_THRESHOLD
            ),
            "model_architecture": (
                "opacus_compatible_non_private"
            ),
        },
        path,
    )


def save_configuration(
    device: torch.device,
) -> None:
    pd.DataFrame(
        [
            {
                "run_name": RUN_NAME,
                "seed": SEED,
                "num_hospitals": (
                    NUM_HOSPITALS
                ),
                "max_rounds": MAX_ROUNDS,
                "local_epochs": (
                    LOCAL_EPOCHS
                ),
                "batch_size": BATCH_SIZE,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "num_workers": NUM_WORKERS,
                "early_stopping_patience": (
                    EARLY_STOPPING_PATIENCE
                ),
                "minimum_improvement": (
                    MINIMUM_IMPROVEMENT
                ),
                "classification_threshold": (
                    CLASSIFICATION_THRESHOLD
                ),
                "normalize_per_record": (
                    NORMALIZE_PER_RECORD
                ),
                "privacy_enabled": False,
                "model_architecture": (
                    "opacus_compatible"
                ),
                "device": str(device),
            }
        ]
    ).to_csv(
        CONFIG_PATH,
        index=False,
    )


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

    print(
        "Experiment: non-private "
        "Opacus-compatible FedAvg control"
    )
    print(
        f"Number of hospitals: "
        f"{NUM_HOSPITALS}"
    )
    print(
        f"Maximum federated rounds: "
        f"{MAX_ROUNDS}"
    )
    print(
        f"Local epochs per round: "
        f"{LOCAL_EPOCHS}"
    )

    save_configuration(
        device
    )

    (
        validation_metadata,
        test_metadata,
    ) = load_global_metadata()

    validation_loader = (
        create_evaluation_loader(
            validation_metadata,
            device,
        )
    )

    test_loader = (
        create_evaluation_loader(
            test_metadata,
            device,
        )
    )

    clients = create_clients(
        validation_metadata,
        device,
    )

    global_model = (
        create_dp_compatible_ecg_model(
            device
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    global_parameters = (
        get_model_parameters(
            global_model
        )
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
            f"Compatible FedAvg round "
            f"{round_number}/{MAX_ROUNDS}"
        )
        print("=" * 70)

        aggregation_results = []

        for client in clients:
            print(
                f"\nTraining Hospital "
                f"{client.hospital_id}"
            )

            (
                parameters,
                sample_count,
                metrics,
            ) = client.fit(
                global_parameters
            )

            aggregation_results.append(
                (
                    parameters,
                    sample_count,
                )
            )

            client_row = {
                "round": round_number,
                "hospital_id": (
                    client.hospital_id
                ),
                "sample_count": (
                    sample_count
                ),
                "train_loss": (
                    metrics["train_loss"]
                ),
                "validation_loss": (
                    metrics[
                        "validation_loss"
                    ]
                ),
                "validation_macro_f1": (
                    metrics[
                        "validation_macro_f1"
                    ]
                ),
                "validation_weighted_f1": (
                    metrics[
                        "validation_weighted_f1"
                    ]
                ),
                "validation_macro_auroc": (
                    metrics[
                        "validation_macro_auroc"
                    ]
                ),
            }

            client_history.append(
                client_row
            )

            print(
                f"Hospital "
                f"{client.hospital_id} | "
                f"train loss="
                f"{metrics['train_loss']:.4f} | "
                f"validation loss="
                f"{metrics['validation_loss']:.4f} | "
                f"macro-F1="
                f"{metrics['validation_macro_f1']:.4f}"
            )

        global_parameters = (
            aggregate_fedavg(
                aggregation_results
            )
        )

        set_model_parameters(
            global_model,
            global_parameters,
        )

        validation_metrics = evaluate_model(
            model=global_model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=(
                CLASSIFICATION_THRESHOLD
            ),
        )

        print(
            "\nGlobal validation results"
        )
        print(
            f"Loss: "
            f"{validation_metrics['loss']:.6f}"
        )
        print(
            f"Macro-F1: "
            f"{validation_metrics['macro_f1']:.4f}"
        )
        print(
            f"Weighted-F1: "
            f"{validation_metrics['weighted_f1']:.4f}"
        )
        print(
            f"Macro-AUROC: "
            f"{validation_metrics['macro_auroc']:.4f}"
        )

        improved = (
            validation_metrics["loss"]
            < best_validation_loss
            - MINIMUM_IMPROVEMENT
        )

        if improved:
            best_validation_loss = (
                validation_metrics["loss"]
            )
            best_round = round_number
            best_global_state = (
                copy.deepcopy(
                    global_model.state_dict()
                )
            )
            rounds_without_improvement = 0

            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                global_model,
                round_number,
                validation_metrics,
            )

            print(
                f"Saved new best model: "
                f"{BEST_CHECKPOINT_PATH}"
            )

        else:
            rounds_without_improvement += 1

            print(
                "No sufficient validation-loss "
                "improvement. "
                f"Patience: "
                f"{rounds_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        round_history.append(
            {
                "round": round_number,
                "global_validation_loss": (
                    validation_metrics[
                        "loss"
                    ]
                ),
                "global_validation_macro_f1": (
                    validation_metrics[
                        "macro_f1"
                    ]
                ),
                "global_validation_weighted_f1": (
                    validation_metrics[
                        "weighted_f1"
                    ]
                ),
                "global_validation_macro_auroc": (
                    validation_metrics[
                        "macro_auroc"
                    ]
                ),
                "improved": improved,
                "rounds_without_improvement": (
                    rounds_without_improvement
                ),
            }
        )

        pd.DataFrame(
            client_history
        ).to_csv(
            CLIENT_HISTORY_PATH,
            index=False,
        )

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
                "\nCompatible FedAvg "
                "early stopping triggered."
            )
            break

    last_validation_metrics = (
        evaluate_model(
            model=global_model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=(
                CLASSIFICATION_THRESHOLD
            ),
        )
    )

    save_checkpoint(
        LAST_CHECKPOINT_PATH,
        global_model,
        rounds_completed,
        last_validation_metrics,
    )

    if best_global_state is None:
        raise RuntimeError(
            "No best global state was saved."
        )

    global_model.load_state_dict(
        best_global_state
    )

    test_metrics = evaluate_model(
        model=global_model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        threshold=(
            CLASSIFICATION_THRESHOLD
        ),
    )

    print("\n" + "=" * 70)
    print(
        "Best compatible FedAvg "
        "global test results"
    )
    print("=" * 70)
    print(
        f"Best round: {best_round}"
    )
    print(
        f"Rounds completed: "
        f"{rounds_completed}"
    )
    print(
        f"Stopped early: "
        f"{stopped_early}"
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

    result_rows = [
        {
            "metric_scope": "overall",
            "best_round": best_round,
            "rounds_completed": (
                rounds_completed
            ),
            "stopped_early": (
                stopped_early
            ),
            "best_validation_loss": (
                best_validation_loss
            ),
            "test_loss": (
                test_metrics["loss"]
            ),
            "test_macro_f1": (
                test_metrics["macro_f1"]
            ),
            "test_weighted_f1": (
                test_metrics[
                    "weighted_f1"
                ]
            ),
            "test_macro_auroc": (
                test_metrics[
                    "macro_auroc"
                ]
            ),
            "test_class_f1": "",
            "test_class_auroc": "",
        }
    ]

    print(
        "\nPer-class test results"
    )

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
                "metric_scope": (
                    class_name
                ),
                "best_round": (
                    best_round
                ),
                "test_class_f1": (
                    class_f1
                ),
                "test_class_auroc": (
                    class_auroc
                ),
            }
        )

    pd.DataFrame(
        result_rows
    ).to_csv(
        TEST_RESULTS_PATH,
        index=False,
    )

    print(
        "\nCompatible FedAvg "
        "experiment completed."
    )
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