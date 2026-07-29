"""
Run a differentially private FedAvg experiment.
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

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES, PTBXLDataset
from src.data.paths import PARTITIONS_DIR, PROCESSED_METADATA_PATH, PTBXL_ROOT
from src.federated.dp_client import (
    DPHospitalClient,
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

NOISE_MULTIPLIER = 1.0
MAX_GRAD_NORM = 1.0
DELTA = 1e-5
MAX_EPSILON = 8.0
PRIVACY_ACCOUNTANT = "prv"
SECURE_MODE = False

EARLY_STOPPING_PATIENCE = 5
MINIMUM_IMPROVEMENT = 1e-4
CLASSIFICATION_THRESHOLD = 0.5
NORMALIZE_PER_RECORD = False

RUN_NAME = (
    f"dp_fedavg_rounds_{MAX_ROUNDS}"
    f"_local_epochs_{LOCAL_EPOCHS}"
    f"_noise_{NOISE_MULTIPLIER:g}"
    f"_clip_{MAX_GRAD_NORM:g}"
    f"_epscap_{MAX_EPSILON:g}"
)

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints" / RUN_NAME
LOG_DIR = RESULTS_DIR / "logs" / RUN_NAME
TABLE_DIR = RESULTS_DIR / "tables"

BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "dp_fedavg_best.pt"
LAST_CHECKPOINT_PATH = CHECKPOINT_DIR / "dp_fedavg_last.pt"
ROUND_HISTORY_PATH = LOG_DIR / "dp_fedavg_round_history.csv"
CLIENT_HISTORY_PATH = LOG_DIR / "dp_fedavg_client_history.csv"
TEST_RESULTS_PATH = TABLE_DIR / f"{RUN_NAME}_test_results.csv"
CONFIG_PATH = TABLE_DIR / f"{RUN_NAME}_config.csv"


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_output_directories() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


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


def load_global_metadata() -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(PROCESSED_METADATA_PATH, index_col="ecg_id")
    validation = metadata[metadata["split"] == "validation"].copy()
    test = metadata[metadata["split"] == "test"].copy()
    if validation.empty or test.empty:
        raise RuntimeError("Validation or test split is empty.")
    return validation, test


def load_hospital_metadata(hospital_id: int) -> pd.DataFrame:
    path = PARTITIONS_DIR / f"hospital_{hospital_id}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    metadata = pd.read_csv(path)
    if not (metadata["hospital_id"] == hospital_id).all():
        raise ValueError(f"Hospital ID mismatch in {path}")
    if not (metadata["split"] == "train").all():
        raise ValueError(f"Non-training records found in {path}")
    return metadata


def create_clients(
    validation_metadata: pd.DataFrame,
    device: torch.device,
) -> list[DPHospitalClient]:
    clients: list[DPHospitalClient] = []

    for hospital_id in range(NUM_HOSPITALS):
        metadata = load_hospital_metadata(hospital_id)
        print(
            f"Hospital {hospital_id}: "
            f"{len(metadata):,} records, "
            f"{metadata['patient_id'].nunique():,} patients"
        )

        clients.append(
            DPHospitalClient(
                hospital_id=hospital_id,
                train_metadata=metadata,
                validation_metadata=validation_metadata,
                device=device,
                batch_size=BATCH_SIZE,
                learning_rate=LEARNING_RATE,
                local_epochs=LOCAL_EPOCHS,
                num_workers=NUM_WORKERS,
                threshold=CLASSIFICATION_THRESHOLD,
                normalize_per_record=NORMALIZE_PER_RECORD,
                noise_multiplier=NOISE_MULTIPLIER,
                max_grad_norm=MAX_GRAD_NORM,
                delta=DELTA,
                accountant=PRIVACY_ACCOUNTANT,
                secure_mode=SECURE_MODE,
            )
        )

    return clients


def save_checkpoint(
    path: Path,
    model: nn.Module,
    round_number: int,
    validation_metrics: dict,
    privacy_rows: list[dict],
) -> None:
    torch.save(
        {
            "round": round_number,
            "model_state_dict": model.state_dict(),
            "validation_loss": validation_metrics["loss"],
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_weighted_f1": validation_metrics["weighted_f1"],
            "validation_macro_auroc": validation_metrics["macro_auroc"],
            "epsilon_by_hospital": {
                int(row["hospital_id"]): float(row["epsilon"])
                for row in privacy_rows
            },
            "delta": DELTA,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "class_names": DEFAULT_CLASS_NAMES,
            "threshold": CLASSIFICATION_THRESHOLD,
        },
        path,
    )


def save_configuration(device: torch.device) -> None:
    pd.DataFrame(
        [{
            "run_name": RUN_NAME,
            "seed": SEED,
            "num_hospitals": NUM_HOSPITALS,
            "max_rounds": MAX_ROUNDS,
            "local_epochs": LOCAL_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "num_workers": NUM_WORKERS,
            "noise_multiplier": NOISE_MULTIPLIER,
            "max_grad_norm": MAX_GRAD_NORM,
            "delta": DELTA,
            "max_epsilon": MAX_EPSILON,
            "privacy_accountant": PRIVACY_ACCOUNTANT,
            "secure_mode": SECURE_MODE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "minimum_improvement": MINIMUM_IMPROVEMENT,
            "device": str(device),
        }]
    ).to_csv(CONFIG_PATH, index=False)


def main() -> None:
    set_random_seed(SEED)
    create_output_directories()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Number of hospitals: {NUM_HOSPITALS}")
    print(f"Maximum federated rounds: {MAX_ROUNDS}")
    print(f"Local epochs per round: {LOCAL_EPOCHS}")
    print(f"Noise multiplier: {NOISE_MULTIPLIER}")
    print(f"Maximum gradient norm: {MAX_GRAD_NORM}")
    print(f"Delta: {DELTA:.1e}")
    print(f"Maximum allowed epsilon: {MAX_EPSILON:.4f}")

    save_configuration(device)

    validation_metadata, test_metadata = load_global_metadata()
    validation_loader = create_evaluation_loader(validation_metadata, device)
    test_loader = create_evaluation_loader(test_metadata, device)
    clients = create_clients(validation_metadata, device)

    global_model = create_dp_compatible_ecg_model(device)
    criterion = nn.BCEWithLogitsLoss()
    global_parameters = get_model_parameters(global_model)

    round_history: list[dict] = []
    client_history: list[dict] = []

    best_validation_loss = float("inf")
    best_round = 0
    best_global_state = None
    best_privacy_rows: list[dict] = []

    rounds_without_improvement = 0
    stopped_early = False
    privacy_budget_reached = False
    rounds_completed = 0

    for round_number in range(1, MAX_ROUNDS + 1):
        rounds_completed = round_number
        print("\n" + "=" * 70)
        print(f"DP-FedAvg round {round_number}/{MAX_ROUNDS}")
        print("=" * 70)

        aggregation_results = []
        current_round_rows: list[dict] = []

        for client in clients:
            print(f"\nTraining Hospital {client.hospital_id}")
            parameters, sample_count, metrics = client.fit(global_parameters)

            aggregation_results.append((parameters, sample_count))

            row = {
                "round": round_number,
                "hospital_id": client.hospital_id,
                "sample_count": sample_count,
                "train_loss": metrics["train_loss"],
                "validation_loss": metrics["validation_loss"],
                "validation_macro_f1": metrics["validation_macro_f1"],
                "validation_weighted_f1": metrics["validation_weighted_f1"],
                "validation_macro_auroc": metrics["validation_macro_auroc"],
                "epsilon": metrics["epsilon"],
                "delta": metrics["delta"],
                "noise_multiplier": metrics["noise_multiplier"],
                "max_grad_norm": metrics["max_grad_norm"],
            }
            client_history.append(row)
            current_round_rows.append(row)

            print(
                f"Hospital {client.hospital_id} | "
                f"train loss={metrics['train_loss']:.4f} | "
                f"validation loss={metrics['validation_loss']:.4f} | "
                f"macro-F1={metrics['validation_macro_f1']:.4f} | "
                f"epsilon={metrics['epsilon']:.4f}"
            )

        global_parameters = aggregate_fedavg(aggregation_results)
        set_model_parameters(global_model, global_parameters)

        validation_metrics = evaluate_model(
            model=global_model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
            threshold=CLASSIFICATION_THRESHOLD,
        )

        epsilons = [row["epsilon"] for row in current_round_rows]

        print("\nGlobal validation results")
        print(f"Loss: {validation_metrics['loss']:.6f}")
        print(f"Macro-F1: {validation_metrics['macro_f1']:.4f}")
        print(f"Weighted-F1: {validation_metrics['weighted_f1']:.4f}")
        print(f"Macro-AUROC: {validation_metrics['macro_auroc']:.4f}")
        print(
            f"Epsilon range: {min(epsilons):.4f} - "
            f"{max(epsilons):.4f} at delta={DELTA:.1e}"
        )

        improved = (
            validation_metrics["loss"]
            < best_validation_loss - MINIMUM_IMPROVEMENT
        )

        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_round = round_number
            best_global_state = copy.deepcopy(global_model.state_dict())
            best_privacy_rows = copy.deepcopy(current_round_rows)
            rounds_without_improvement = 0
            save_checkpoint(
                BEST_CHECKPOINT_PATH,
                global_model,
                round_number,
                validation_metrics,
                current_round_rows,
            )
            print(f"Saved new best global model: {BEST_CHECKPOINT_PATH}")
        else:
            rounds_without_improvement += 1
            print(
                "No sufficient validation-loss improvement. "
                f"Patience: {rounds_without_improvement}/"
                f"{EARLY_STOPPING_PATIENCE}"
            )

        round_history.append(
            {
                "round": round_number,
                "global_validation_loss": validation_metrics["loss"],
                "global_validation_macro_f1": validation_metrics["macro_f1"],
                "global_validation_weighted_f1": validation_metrics["weighted_f1"],
                "global_validation_macro_auroc": validation_metrics["macro_auroc"],
                "minimum_client_epsilon": min(epsilons),
                "maximum_client_epsilon": max(epsilons),
                "mean_client_epsilon": float(np.mean(epsilons)),
                "delta": DELTA,
                "improved": improved,
                "rounds_without_improvement": rounds_without_improvement,
            }
        )

        pd.DataFrame(client_history).to_csv(
            CLIENT_HISTORY_PATH,
            index=False,
        )
        pd.DataFrame(round_history).to_csv(
            ROUND_HISTORY_PATH,
            index=False,
        )

        if max(epsilons) >= MAX_EPSILON:
            privacy_budget_reached = True
            print(
                "\nMaximum privacy budget reached."
            )
            print(
                f"Maximum client epsilon "
                f"{max(epsilons):.4f} >= "
                f"configured cap {MAX_EPSILON:.4f}."
            )
            break

        if (
            rounds_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            print(
                "\nDP-FedAvg early stopping triggered."
            )
            break

    final_rows = [
        row for row in client_history
        if row["round"] == rounds_completed
    ]

    last_validation_metrics = evaluate_model(
        global_model,
        validation_loader,
        criterion,
        device,
        CLASSIFICATION_THRESHOLD,
    )
    save_checkpoint(
        LAST_CHECKPOINT_PATH,
        global_model,
        rounds_completed,
        last_validation_metrics,
        final_rows,
    )

    if best_global_state is None:
        raise RuntimeError("No best global model was saved.")

    global_model.load_state_dict(best_global_state)

    test_metrics = evaluate_model(
        global_model,
        test_loader,
        criterion,
        device,
        CLASSIFICATION_THRESHOLD,
    )

    best_epsilons = [
        row["epsilon"]
        for row in best_privacy_rows
    ]
    final_epsilons = [
        row["epsilon"]
        for row in final_rows
    ]

    if privacy_budget_reached:
        stopping_reason = "privacy_budget"
    elif stopped_early:
        stopping_reason = "validation_early_stopping"
    else:
        stopping_reason = "maximum_rounds"

    print("\n" + "=" * 70)
    print("Best DP-FedAvg global test results")
    print("=" * 70)
    print(f"Best round: {best_round}")
    print(f"Rounds completed: {rounds_completed}")
    print(f"Stopped early: {stopped_early}")
    print(
        f"Privacy budget reached: "
        f"{privacy_budget_reached}"
    )
    print(f"Best validation loss: {best_validation_loss:.6f}")
    print(f"Stopping reason: {stopping_reason}")
    print(
        f"Best-checkpoint epsilon range: "
        f"{min(best_epsilons):.4f} - "
        f"{max(best_epsilons):.4f}"
    )
    print(
        f"Total-training epsilon range: "
        f"{min(final_epsilons):.4f} - "
        f"{max(final_epsilons):.4f}"
    )
    print(
        "Primary privacy guarantee to report: "
        f"epsilon={max(final_epsilons):.4f}, "
        f"delta={DELTA:.1e}"
    )
    print(f"Test loss: {test_metrics['loss']:.6f}")
    print(f"Test macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test weighted-F1: {test_metrics['weighted_f1']:.4f}")
    print(f"Test macro-AUROC: {test_metrics['macro_auroc']:.4f}")

    rows = [{
        "metric_scope": "overall",
        "best_round": best_round,
        "rounds_completed": rounds_completed,
        "stopped_early": stopped_early,
        "privacy_budget_reached": privacy_budget_reached,
        "stopping_reason": stopping_reason,
        "configured_max_epsilon": MAX_EPSILON,
        "best_validation_loss": best_validation_loss,
        "best_checkpoint_minimum_epsilon": min(
            best_epsilons
        ),
        "best_checkpoint_maximum_epsilon": max(
            best_epsilons
        ),
        "best_checkpoint_mean_epsilon": float(
            np.mean(best_epsilons)
        ),
        "total_training_minimum_epsilon": min(
            final_epsilons
        ),
        "total_training_maximum_epsilon": max(
            final_epsilons
        ),
        "total_training_mean_epsilon": float(
            np.mean(final_epsilons)
        ),
        "reported_epsilon": max(final_epsilons),
        "delta": DELTA,
        "noise_multiplier": NOISE_MULTIPLIER,
        "max_grad_norm": MAX_GRAD_NORM,
        "test_loss": test_metrics["loss"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_weighted_f1": test_metrics["weighted_f1"],
        "test_macro_auroc": test_metrics["macro_auroc"],
        "test_class_f1": "",
        "test_class_auroc": "",
    }]

    print("\nPer-class test results")
    for class_name, class_f1, class_auroc in zip(
        DEFAULT_CLASS_NAMES,
        test_metrics["per_class_f1"],
        test_metrics["per_class_auroc"],
    ):
        print(f"{class_name}: F1={class_f1:.4f}, AUROC={class_auroc:.4f}")
        rows.append({
            "metric_scope": class_name,
            "best_round": best_round,
            "test_class_f1": class_f1,
            "test_class_auroc": class_auroc,
        })

    pd.DataFrame(rows).to_csv(TEST_RESULTS_PATH, index=False)

    print("\nDP-FedAvg experiment completed.")
    print(f"Best checkpoint: {BEST_CHECKPOINT_PATH}")
    print(f"Last checkpoint: {LAST_CHECKPOINT_PATH}")
    print(f"Round history: {ROUND_HISTORY_PATH}")
    print(f"Client history: {CLIENT_HISTORY_PATH}")
    print(f"Test results: {TEST_RESULTS_PATH}")
    print(f"Configuration: {CONFIG_PATH}")


if __name__ == "__main__":
    main()