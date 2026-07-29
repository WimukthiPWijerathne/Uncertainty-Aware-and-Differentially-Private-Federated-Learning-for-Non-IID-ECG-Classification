"""
Run differentially private uncertainty-aware FedAvg (DP-UA-FedAvg).

The experiment combines:
  1. Private local DP-SGD training with one independent Opacus
     privacy accountant per hospital.
  2. MC-dropout predictive-uncertainty estimation per hospital.
  3. Sample-size and inverse-uncertainty server aggregation.

The runner saves:
  * best and last checkpoints,
  * per-client/per-round CSV logs,
  * global round-history CSV,
  * configuration CSV and JSON,
  * final test-results CSV,
  * final machine-readable summary JSON,
  * run-status JSON updated after every completed round.

Examples
--------
Three-round smoke test:
    uv run python scripts\\16_run_dp_ua_fedavg.py

Full experiment:
    uv run python scripts\\16_run_dp_ua_fedavg.py ^
        --max-rounds 50 ^
        --mc-passes 10
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
from src.federated.dp_client import (
    create_dp_compatible_ecg_model,
)
from src.federated.dp_ua_client import (
    DPUAHospitalClient,
)
from src.federated.parameter_utils import (
    get_model_parameters,
    set_model_parameters,
)
from src.federated.ua_fedavg import (
    aggregate_uncertainty_fedavg,
)
from src.training.evaluate import evaluate_model


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete configuration for one DP-UA-FedAvg run."""

    seed: int = 42

    num_hospitals: int = 4
    max_rounds: int = 3
    local_epochs: int = 1

    batch_size: int = 64
    learning_rate: float = 1e-3
    num_workers: int = 2

    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    max_epsilon: float = 8.0
    privacy_accountant: str = "prv"
    secure_mode: bool = False

    mc_passes: int = 5
    uncertainty_epsilon: float = 1e-8

    early_stopping_patience: int = 5
    minimum_improvement: float = 1e-4

    classification_threshold: float = 0.5
    normalize_per_record: bool = False

    overwrite: bool = False


@dataclass(frozen=True)
class OutputPaths:
    """All output paths belonging to one experiment."""

    run_name: str
    checkpoint_dir: Path
    log_dir: Path
    table_dir: Path

    best_checkpoint: Path
    last_checkpoint: Path

    round_history: Path
    client_history: Path
    status_json: Path

    test_results: Path
    config_csv: Path
    config_json: Path
    final_summary_json: Path


def parse_arguments() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run DP-UA-FedAvg with clear, run-specific result storage."
        )
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-hospitals", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--local-epochs", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--noise-multiplier", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--max-epsilon", type=float, default=8.0)
    parser.add_argument(
        "--privacy-accountant",
        type=str,
        default="prv",
        choices=("prv", "rdp", "gdp"),
    )
    parser.add_argument(
        "--secure-mode",
        action="store_true",
        help=(
            "Enable Opacus secure RNG. Leave disabled for the course "
            "experiment unless a production-grade run is required."
        ),
    )

    parser.add_argument("--mc-passes", type=int, default=5)
    parser.add_argument(
        "--uncertainty-epsilon",
        type=float,
        default=1e-8,
    )

    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--minimum-improvement",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--normalize-per-record",
        action="store_true",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete only this run's existing outputs before starting. "
            "Without this flag, existing outputs are protected."
        ),
    )

    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        num_hospitals=args.num_hospitals,
        max_rounds=args.max_rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        delta=args.delta,
        max_epsilon=args.max_epsilon,
        privacy_accountant=args.privacy_accountant,
        secure_mode=args.secure_mode,
        mc_passes=args.mc_passes,
        uncertainty_epsilon=args.uncertainty_epsilon,
        early_stopping_patience=(
            args.early_stopping_patience
        ),
        minimum_improvement=args.minimum_improvement,
        classification_threshold=(
            args.classification_threshold
        ),
        normalize_per_record=(
            args.normalize_per_record
        ),
        overwrite=args.overwrite,
    )

    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    if config.num_hospitals < 1:
        raise ValueError(
            "num_hospitals must be at least 1."
        )

    if config.max_rounds < 1:
        raise ValueError(
            "max_rounds must be at least 1."
        )

    if config.local_epochs < 1:
        raise ValueError(
            "local_epochs must be at least 1."
        )

    if config.batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    if config.learning_rate <= 0:
        raise ValueError(
            "learning_rate must be positive."
        )

    if config.num_workers < 0:
        raise ValueError(
            "num_workers cannot be negative."
        )

    if config.noise_multiplier <= 0:
        raise ValueError(
            "noise_multiplier must be positive."
        )

    if config.max_grad_norm <= 0:
        raise ValueError(
            "max_grad_norm must be positive."
        )

    if not 0 < config.delta < 1:
        raise ValueError(
            "delta must be between 0 and 1."
        )

    if config.max_epsilon <= 0:
        raise ValueError(
            "max_epsilon must be positive."
        )

    if config.mc_passes < 2:
        raise ValueError(
            "mc_passes must be at least 2."
        )

    if config.uncertainty_epsilon <= 0:
        raise ValueError(
            "uncertainty_epsilon must be positive."
        )

    if config.early_stopping_patience < 1:
        raise ValueError(
            "early_stopping_patience must be at least 1."
        )

    if config.minimum_improvement < 0:
        raise ValueError(
            "minimum_improvement cannot be negative."
        )

    if not 0 < config.classification_threshold < 1:
        raise ValueError(
            "classification_threshold must be between 0 and 1."
        )


def number_token(value: float) -> str:
    """Create a concise filesystem-safe number token."""
    return f"{value:g}".replace("-", "m")


def create_output_paths(
    config: ExperimentConfig,
) -> OutputPaths:
    run_name = (
        f"dp_ua_fedavg"
        f"_rounds_{config.max_rounds}"
        f"_local_epochs_{config.local_epochs}"
        f"_noise_{number_token(config.noise_multiplier)}"
        f"_clip_{number_token(config.max_grad_norm)}"
        f"_mc_{config.mc_passes}"
        f"_epscap_{number_token(config.max_epsilon)}"
    )

    results_dir = PROJECT_ROOT / "results"

    checkpoint_dir = (
        results_dir
        / "checkpoints"
        / run_name
    )

    log_dir = (
        results_dir
        / "logs"
        / run_name
    )

    table_dir = results_dir / "tables"

    return OutputPaths(
        run_name=run_name,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        table_dir=table_dir,
        best_checkpoint=(
            checkpoint_dir
            / "dp_ua_fedavg_best.pt"
        ),
        last_checkpoint=(
            checkpoint_dir
            / "dp_ua_fedavg_last.pt"
        ),
        round_history=(
            log_dir
            / "dp_ua_fedavg_round_history.csv"
        ),
        client_history=(
            log_dir
            / "dp_ua_fedavg_client_history.csv"
        ),
        status_json=(
            log_dir
            / "dp_ua_fedavg_run_status.json"
        ),
        test_results=(
            table_dir
            / f"{run_name}_test_results.csv"
        ),
        config_csv=(
            table_dir
            / f"{run_name}_config.csv"
        ),
        config_json=(
            table_dir
            / f"{run_name}_config.json"
        ),
        final_summary_json=(
            table_dir
            / f"{run_name}_summary.json"
        ),
    )


def prepare_output_locations(
    config: ExperimentConfig,
    paths: OutputPaths,
) -> None:
    protected_files = (
        paths.best_checkpoint,
        paths.last_checkpoint,
        paths.round_history,
        paths.client_history,
        paths.status_json,
        paths.test_results,
        paths.config_csv,
        paths.config_json,
        paths.final_summary_json,
    )

    existing_files = [
        path
        for path in protected_files
        if path.exists()
    ]

    if existing_files and not config.overwrite:
        formatted = "\n".join(
            f"  - {path}"
            for path in existing_files
        )

        raise FileExistsError(
            "This exact run already has saved outputs:\n"
            f"{formatted}\n\n"
            "Use a different configuration, or deliberately repeat "
            "it with --overwrite."
        )

    if config.overwrite:
        if paths.checkpoint_dir.exists():
            shutil.rmtree(paths.checkpoint_dir)

        if paths.log_dir.exists():
            shutil.rmtree(paths.log_dir)

        for table_file in (
            paths.test_results,
            paths.config_csv,
            paths.config_json,
            paths.final_summary_json,
        ):
            table_file.unlink(
                missing_ok=True
            )

    paths.checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.log_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.table_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_safe(value: Any) -> Any:
    """Convert NumPy and Path values to JSON-compatible values."""
    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]

    return value


def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            json_safe(payload),
            file,
            indent=2,
            sort_keys=True,
        )

    temporary_path.replace(path)


def save_configuration(
    config: ExperimentConfig,
    paths: OutputPaths,
    device: torch.device,
) -> None:
    payload = {
        **asdict(config),
        "run_name": paths.run_name,
        "device": str(device),
        "project_root": str(PROJECT_ROOT),
        "processed_metadata_path": str(
            PROCESSED_METADATA_PATH
        ),
        "partitions_dir": str(
            PARTITIONS_DIR
        ),
        "ptbxl_root": str(
            PTBXL_ROOT
        ),
    }

    pd.DataFrame(
        [payload]
    ).to_csv(
        paths.config_csv,
        index=False,
    )

    save_json(
        paths.config_json,
        payload,
    )


def load_global_metadata(
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    required_columns = {
        "split",
        "filename_lr",
        *DEFAULT_CLASS_NAMES,
    }

    missing_columns = (
        required_columns
        .difference(metadata.columns)
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

    missing_columns = (
        required_columns
        .difference(metadata.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Hospital {hospital_id} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if not (
        metadata["hospital_id"]
        == hospital_id
    ).all():
        raise ValueError(
            f"Hospital-ID mismatch in "
            f"{partition_path}"
        )

    if not (
        metadata["split"] == "train"
    ).all():
        raise ValueError(
            f"Non-training records found in "
            f"hospital {hospital_id}"
        )

    return metadata


def create_evaluation_loader(
    metadata: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
) -> DataLoader:
    dataset = PTBXLDataset(
        metadata=metadata,
        dataset_root=PTBXL_ROOT,
        class_names=DEFAULT_CLASS_NAMES,
        normalize_per_record=(
            config.normalize_per_record
        ),
    )

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )


def create_clients(
    validation_metadata: pd.DataFrame,
    config: ExperimentConfig,
    device: torch.device,
) -> list[DPUAHospitalClient]:
    clients: list[
        DPUAHospitalClient
    ] = []

    for hospital_id in range(
        config.num_hospitals
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

        client = DPUAHospitalClient(
            hospital_id=hospital_id,
            train_metadata=(
                hospital_metadata
            ),
            validation_metadata=(
                validation_metadata
            ),
            device=device,
            batch_size=config.batch_size,
            learning_rate=(
                config.learning_rate
            ),
            local_epochs=(
                config.local_epochs
            ),
            num_workers=(
                config.num_workers
            ),
            threshold=(
                config.classification_threshold
            ),
            normalize_per_record=(
                config.normalize_per_record
            ),
            noise_multiplier=(
                config.noise_multiplier
            ),
            max_grad_norm=(
                config.max_grad_norm
            ),
            delta=config.delta,
            accountant=(
                config.privacy_accountant
            ),
            secure_mode=(
                config.secure_mode
            ),
            mc_passes=config.mc_passes,
        )

        clients.append(client)

    return clients


def save_checkpoint(
    path: Path,
    model: nn.Module,
    round_number: int,
    validation_metrics: dict[str, Any],
    client_rows: list[dict[str, Any]],
    config: ExperimentConfig,
    paths: OutputPaths,
) -> None:
    epsilon_by_hospital = {
        int(row["hospital_id"]): float(
            row["epsilon"]
        )
        for row in client_rows
    }

    uncertainty_by_hospital = {
        int(row["hospital_id"]): float(
            row["uncertainty"]
        )
        for row in client_rows
    }

    aggregation_weight_by_hospital = {
        int(row["hospital_id"]): float(
            row["final_aggregation_weight"]
        )
        for row in client_rows
    }

    torch.save(
        {
            "run_name": paths.run_name,
            "round": round_number,
            "model_state_dict": (
                model.state_dict()
            ),
            "validation_loss": float(
                validation_metrics["loss"]
            ),
            "validation_macro_f1": float(
                validation_metrics[
                    "macro_f1"
                ]
            ),
            "validation_weighted_f1": float(
                validation_metrics[
                    "weighted_f1"
                ]
            ),
            "validation_macro_auroc": float(
                validation_metrics[
                    "macro_auroc"
                ]
            ),
            "epsilon_by_hospital": (
                epsilon_by_hospital
            ),
            "uncertainty_by_hospital": (
                uncertainty_by_hospital
            ),
            "aggregation_weight_by_hospital": (
                aggregation_weight_by_hospital
            ),
            "delta": config.delta,
            "noise_multiplier": (
                config.noise_multiplier
            ),
            "max_grad_norm": (
                config.max_grad_norm
            ),
            "mc_passes": config.mc_passes,
            "uncertainty_epsilon": (
                config.uncertainty_epsilon
            ),
            "class_names": list(
                DEFAULT_CLASS_NAMES
            ),
            "threshold": (
                config.classification_threshold
            ),
            "num_hospitals": (
                config.num_hospitals
            ),
            "local_epochs": (
                config.local_epochs
            ),
        },
        path,
    )


def save_progress(
    client_history: list[dict[str, Any]],
    round_history: list[dict[str, Any]],
    status: dict[str, Any],
    paths: OutputPaths,
) -> None:
    pd.DataFrame(
        client_history
    ).to_csv(
        paths.client_history,
        index=False,
    )

    pd.DataFrame(
        round_history
    ).to_csv(
        paths.round_history,
        index=False,
    )

    save_json(
        paths.status_json,
        status,
    )


def validate_aggregation_weights(
    weights: list[float],
) -> None:
    if not weights:
        raise RuntimeError(
            "No aggregation weights were returned."
        )

    for weight in weights:
        if not math.isfinite(weight):
            raise RuntimeError(
                "A non-finite aggregation weight "
                "was returned."
            )

        if weight < 0:
            raise RuntimeError(
                "A negative aggregation weight "
                "was returned."
            )

    weight_sum = float(sum(weights))

    if not math.isclose(
        weight_sum,
        1.0,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise RuntimeError(
            "Aggregation weights do not sum to 1. "
            f"Observed sum: {weight_sum}"
        )


def print_configuration(
    config: ExperimentConfig,
    paths: OutputPaths,
    device: torch.device,
) -> None:
    print(f"Run name: {paths.run_name}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        f"Number of hospitals: "
        f"{config.num_hospitals}"
    )
    print(
        f"Maximum federated rounds: "
        f"{config.max_rounds}"
    )
    print(
        f"Local epochs per round: "
        f"{config.local_epochs}"
    )
    print(
        f"Batch size: "
        f"{config.batch_size}"
    )
    print(
        f"Learning rate: "
        f"{config.learning_rate}"
    )
    print(
        f"Noise multiplier: "
        f"{config.noise_multiplier}"
    )
    print(
        f"Maximum gradient norm: "
        f"{config.max_grad_norm}"
    )
    print(
        f"Delta: "
        f"{config.delta:.1e}"
    )
    print(
        f"Maximum allowed epsilon: "
        f"{config.max_epsilon:.4f}"
    )
    print(
        f"MC-dropout passes: "
        f"{config.mc_passes}"
    )
    print(
        f"Uncertainty epsilon: "
        f"{config.uncertainty_epsilon:.1e}"
    )
    print(
        f"Early-stopping patience: "
        f"{config.early_stopping_patience}"
    )


def main() -> None:
    config = parse_arguments()
    paths = create_output_paths(config)

    prepare_output_locations(
        config,
        paths,
    )

    set_random_seed(
        config.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print_configuration(
        config,
        paths,
        device,
    )

    save_configuration(
        config,
        paths,
        device,
    )

    (
        validation_metadata,
        test_metadata,
    ) = load_global_metadata()

    print(
        f"Shared validation records: "
        f"{len(validation_metadata):,}"
    )
    print(
        f"Shared test records: "
        f"{len(test_metadata):,}"
    )

    validation_loader = (
        create_evaluation_loader(
            validation_metadata,
            config,
            device,
        )
    )

    test_loader = (
        create_evaluation_loader(
            test_metadata,
            config,
            device,
        )
    )

    clients = create_clients(
        validation_metadata,
        config,
        device,
    )

    global_model = (
        create_dp_compatible_ecg_model(
            device=device,
        )
    )

    criterion = nn.BCEWithLogitsLoss()

    global_parameters = (
        get_model_parameters(
            global_model
        )
    )

    client_history: list[
        dict[str, Any]
    ] = []

    round_history: list[
        dict[str, Any]
    ] = []

    best_validation_loss = float("inf")
    best_round = 0
    best_global_state = None
    best_client_rows: list[
        dict[str, Any]
    ] = []

    rounds_without_improvement = 0
    rounds_completed = 0
    stopped_early = False
    privacy_budget_reached = False
    stopping_reason = "running"

    initial_status = {
        "run_name": paths.run_name,
        "state": "running",
        "rounds_completed": 0,
        "best_round": 0,
        "best_validation_loss": None,
        "stopping_reason": None,
    }

    save_json(
        paths.status_json,
        initial_status,
    )

    for round_number in range(
        1,
        config.max_rounds + 1,
    ):
        rounds_completed = round_number

        print("\n" + "=" * 78)
        print(
            f"DP-UA-FedAvg round "
            f"{round_number}/"
            f"{config.max_rounds}"
        )
        print("=" * 78)

        aggregation_inputs = []
        current_round_rows: list[
            dict[str, Any]
        ] = []

        for client in clients:
            print(
                f"\nTraining Hospital "
                f"{client.hospital_id}"
            )

            (
                updated_parameters,
                sample_count,
                metrics,
            ) = client.fit(
                global_parameters
            )

            uncertainty = float(
                metrics["uncertainty"]
            )

            aggregation_inputs.append(
                (
                    updated_parameters,
                    sample_count,
                    uncertainty,
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
                "train_loss": float(
                    metrics["train_loss"]
                ),
                "validation_loss": float(
                    metrics[
                        "validation_loss"
                    ]
                ),
                "validation_macro_f1": float(
                    metrics[
                        "validation_macro_f1"
                    ]
                ),
                "validation_weighted_f1": float(
                    metrics[
                        "validation_weighted_f1"
                    ]
                ),
                "validation_macro_auroc": float(
                    metrics[
                        "validation_macro_auroc"
                    ]
                ),
                "uncertainty": uncertainty,
                "mc_passes": int(
                    metrics["mc_passes"]
                ),
                "epsilon": float(
                    metrics["epsilon"]
                ),
                "delta": float(
                    metrics["delta"]
                ),
                "noise_multiplier": float(
                    metrics[
                        "noise_multiplier"
                    ]
                ),
                "max_grad_norm": float(
                    metrics[
                        "max_grad_norm"
                    ]
                ),
            }

            current_round_rows.append(
                client_row
            )

            print(
                f"Hospital {client.hospital_id} | "
                f"samples={sample_count:,} | "
                f"train loss="
                f"{metrics['train_loss']:.4f} | "
                f"validation loss="
                f"{metrics['validation_loss']:.4f} | "
                f"macro-F1="
                f"{metrics['validation_macro_f1']:.4f} | "
                f"epsilon="
                f"{metrics['epsilon']:.4f} | "
                f"uncertainty="
                f"{uncertainty:.8f}"
            )

        (
            global_parameters,
            aggregation_weights,
            raw_uncertainty_weights,
        ) = aggregate_uncertainty_fedavg(
            client_results=(
                aggregation_inputs
            ),
            epsilon=(
                config.uncertainty_epsilon
            ),
        )

        aggregation_weights = [
            float(weight)
            for weight in aggregation_weights
        ]

        raw_uncertainty_weights = [
            float(weight)
            for weight
            in raw_uncertainty_weights
        ]

        validate_aggregation_weights(
            aggregation_weights
        )

        total_samples = sum(
            sample_count
            for (
                _,
                sample_count,
                _,
            ) in aggregation_inputs
        )

        sample_only_weights = [
            sample_count / total_samples
            for (
                _,
                sample_count,
                _,
            ) in aggregation_inputs
        ]

        for (
            row,
            sample_only_weight,
            raw_weight,
            final_weight,
        ) in zip(
            current_round_rows,
            sample_only_weights,
            raw_uncertainty_weights,
            aggregation_weights,
            strict=True,
        ):
            row["sample_only_weight"] = float(
                sample_only_weight
            )
            row[
                "uncertainty_raw_weight"
            ] = float(raw_weight)
            row[
                "final_aggregation_weight"
            ] = float(final_weight)

            print(
                f"Hospital "
                f"{row['hospital_id']} weights | "
                f"FedAvg="
                f"{sample_only_weight:.4f} | "
                f"raw-UA="
                f"{raw_weight:.4f} | "
                f"final-UA="
                f"{final_weight:.4f}"
            )

        client_history.extend(
            current_round_rows
        )

        weight_sum = sum(
            aggregation_weights
        )

        print(
            f"DP-UA aggregation-weight sum: "
            f"{weight_sum:.8f}"
        )

        maximum_weight = max(
            aggregation_weights
        )

        if maximum_weight > 0.80:
            print(
                "Warning: one client received more "
                "than 80% of the aggregation weight."
            )

        set_model_parameters(
            global_model,
            global_parameters,
        )

        validation_metrics = (
            evaluate_model(
                model=global_model,
                data_loader=(
                    validation_loader
                ),
                criterion=criterion,
                device=device,
                threshold=(
                    config
                    .classification_threshold
                ),
            )
        )

        epsilon_values = [
            float(row["epsilon"])
            for row in current_round_rows
        ]

        uncertainty_values = [
            float(row["uncertainty"])
            for row in current_round_rows
        ]

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
        print(
            f"Epsilon range: "
            f"{min(epsilon_values):.4f} - "
            f"{max(epsilon_values):.4f} "
            f"at delta={config.delta:.1e}"
        )
        print(
            f"Uncertainty range: "
            f"{min(uncertainty_values):.8f} - "
            f"{max(uncertainty_values):.8f}"
        )

        improved = (
            validation_metrics["loss"]
            < best_validation_loss
            - config.minimum_improvement
        )

        if improved:
            best_validation_loss = float(
                validation_metrics["loss"]
            )
            best_round = round_number
            best_global_state = (
                copy.deepcopy(
                    global_model.state_dict()
                )
            )
            best_client_rows = (
                copy.deepcopy(
                    current_round_rows
                )
            )
            rounds_without_improvement = 0

            save_checkpoint(
                path=(
                    paths.best_checkpoint
                ),
                model=global_model,
                round_number=(
                    round_number
                ),
                validation_metrics=(
                    validation_metrics
                ),
                client_rows=(
                    current_round_rows
                ),
                config=config,
                paths=paths,
            )

            print(
                f"Saved new best global model: "
                f"{paths.best_checkpoint}"
            )
        else:
            rounds_without_improvement += 1

            print(
                "No sufficient validation-loss "
                "improvement. "
                f"Patience: "
                f"{rounds_without_improvement}/"
                f"{config.early_stopping_patience}"
            )

        round_row = {
            "round": round_number,
            "global_validation_loss": float(
                validation_metrics["loss"]
            ),
            "global_validation_macro_f1": float(
                validation_metrics[
                    "macro_f1"
                ]
            ),
            "global_validation_weighted_f1": float(
                validation_metrics[
                    "weighted_f1"
                ]
            ),
            "global_validation_macro_auroc": float(
                validation_metrics[
                    "macro_auroc"
                ]
            ),
            "minimum_client_epsilon": min(
                epsilon_values
            ),
            "maximum_client_epsilon": max(
                epsilon_values
            ),
            "mean_client_epsilon": float(
                np.mean(epsilon_values)
            ),
            "minimum_client_uncertainty": min(
                uncertainty_values
            ),
            "maximum_client_uncertainty": max(
                uncertainty_values
            ),
            "mean_client_uncertainty": float(
                np.mean(uncertainty_values)
            ),
            "minimum_aggregation_weight": min(
                aggregation_weights
            ),
            "maximum_aggregation_weight": max(
                aggregation_weights
            ),
            "aggregation_weight_sum": float(
                weight_sum
            ),
            "improved": improved,
            "rounds_without_improvement": (
                rounds_without_improvement
            ),
        }

        round_history.append(
            round_row
        )

        current_status = {
            "run_name": paths.run_name,
            "state": "running",
            "rounds_completed": (
                rounds_completed
            ),
            "best_round": best_round,
            "best_validation_loss": (
                None
                if best_round == 0
                else best_validation_loss
            ),
            "current_maximum_epsilon": max(
                epsilon_values
            ),
            "privacy_cap": (
                config.max_epsilon
            ),
            "stopping_reason": None,
        }

        save_progress(
            client_history,
            round_history,
            current_status,
            paths,
        )

        if (
            max(epsilon_values)
            >= config.max_epsilon
        ):
            privacy_budget_reached = True
            stopping_reason = (
                "privacy_budget"
            )

            print(
                "\nMaximum privacy budget reached."
            )
            print(
                f"Maximum client epsilon "
                f"{max(epsilon_values):.4f} "
                f">= configured cap "
                f"{config.max_epsilon:.4f}."
            )
            break

        if (
            rounds_without_improvement
            >= config.early_stopping_patience
        ):
            stopped_early = True
            stopping_reason = (
                "validation_early_stopping"
            )

            print(
                "\nDP-UA-FedAvg early "
                "stopping triggered."
            )
            print(
                f"Best round was "
                f"{best_round} with "
                f"validation loss "
                f"{best_validation_loss:.6f}."
            )
            break

    if stopping_reason == "running":
        stopping_reason = (
            "maximum_rounds"
        )

    final_round_rows = [
        row
        for row in client_history
        if int(row["round"])
        == rounds_completed
    ]

    if len(final_round_rows) != len(clients):
        raise RuntimeError(
            "The final round does not contain "
            "one result row per hospital."
        )

    last_validation_metrics = (
        evaluate_model(
            model=global_model,
            data_loader=(
                validation_loader
            ),
            criterion=criterion,
            device=device,
            threshold=(
                config.classification_threshold
            ),
        )
    )

    save_checkpoint(
        path=paths.last_checkpoint,
        model=global_model,
        round_number=rounds_completed,
        validation_metrics=(
            last_validation_metrics
        ),
        client_rows=final_round_rows,
        config=config,
        paths=paths,
    )

    if best_global_state is None:
        raise RuntimeError(
            "No best global model was saved."
        )

    if not best_client_rows:
        raise RuntimeError(
            "No best-round client metrics "
            "were saved."
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
            config.classification_threshold
        ),
    )

    best_epsilon_values = [
        float(row["epsilon"])
        for row in best_client_rows
    ]

    final_epsilon_values = [
        float(row["epsilon"])
        for row in final_round_rows
    ]

    best_uncertainty_values = [
        float(row["uncertainty"])
        for row in best_client_rows
    ]

    reported_epsilon = max(
        final_epsilon_values
    )

    print("\n" + "=" * 78)
    print(
        "Best DP-UA-FedAvg global "
        "test results"
    )
    print("=" * 78)
    print(f"Best round: {best_round}")
    print(
        f"Rounds completed: "
        f"{rounds_completed}"
    )
    print(
        f"Stopping reason: "
        f"{stopping_reason}"
    )
    print(
        f"Stopped early: "
        f"{stopped_early}"
    )
    print(
        f"Privacy budget reached: "
        f"{privacy_budget_reached}"
    )
    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )
    print(
        f"Best-checkpoint epsilon range: "
        f"{min(best_epsilon_values):.4f} - "
        f"{max(best_epsilon_values):.4f}"
    )
    print(
        f"Total-training epsilon range: "
        f"{min(final_epsilon_values):.4f} - "
        f"{max(final_epsilon_values):.4f}"
    )
    print(
        "Primary privacy guarantee: "
        f"epsilon={reported_epsilon:.4f}, "
        f"delta={config.delta:.1e}"
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

    overall_result = {
        "metric_scope": "overall",
        "method": "DP-UA-FedAvg",
        "run_name": paths.run_name,
        "best_round": best_round,
        "rounds_completed": (
            rounds_completed
        ),
        "stopping_reason": (
            stopping_reason
        ),
        "stopped_early": (
            stopped_early
        ),
        "privacy_budget_reached": (
            privacy_budget_reached
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
        "best_checkpoint_minimum_epsilon": min(
            best_epsilon_values
        ),
        "best_checkpoint_maximum_epsilon": max(
            best_epsilon_values
        ),
        "best_checkpoint_mean_epsilon": float(
            np.mean(best_epsilon_values)
        ),
        "total_training_minimum_epsilon": min(
            final_epsilon_values
        ),
        "total_training_maximum_epsilon": max(
            final_epsilon_values
        ),
        "total_training_mean_epsilon": float(
            np.mean(final_epsilon_values)
        ),
        "reported_epsilon": (
            reported_epsilon
        ),
        "delta": config.delta,
        "noise_multiplier": (
            config.noise_multiplier
        ),
        "max_grad_norm": (
            config.max_grad_norm
        ),
        "mc_passes": config.mc_passes,
        "best_round_mean_uncertainty": float(
            np.mean(
                best_uncertainty_values
            )
        ),
        "test_loss": float(
            test_metrics["loss"]
        ),
        "test_macro_f1": float(
            test_metrics["macro_f1"]
        ),
        "test_weighted_f1": float(
            test_metrics["weighted_f1"]
        ),
        "test_macro_auroc": float(
            test_metrics["macro_auroc"]
        ),
        "test_class_f1": "",
        "test_class_auroc": "",
    }

    result_rows = [
        overall_result
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
        strict=True,
    ):
        class_f1 = float(class_f1)
        class_auroc = float(
            class_auroc
        )

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
                "method": (
                    "DP-UA-FedAvg"
                ),
                "run_name": (
                    paths.run_name
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
        paths.test_results,
        index=False,
    )

    final_summary = {
        **overall_result,
        "per_class": {
            class_name: {
                "f1": float(class_f1),
                "auroc": float(
                    class_auroc
                ),
            }
            for (
                class_name,
                class_f1,
                class_auroc,
            ) in zip(
                DEFAULT_CLASS_NAMES,
                test_metrics[
                    "per_class_f1"
                ],
                test_metrics[
                    "per_class_auroc"
                ],
                strict=True,
            )
        },
        "output_paths": {
            "best_checkpoint": (
                paths.best_checkpoint
            ),
            "last_checkpoint": (
                paths.last_checkpoint
            ),
            "round_history": (
                paths.round_history
            ),
            "client_history": (
                paths.client_history
            ),
            "test_results": (
                paths.test_results
            ),
            "config_csv": (
                paths.config_csv
            ),
            "config_json": (
                paths.config_json
            ),
        },
    }

    save_json(
        paths.final_summary_json,
        final_summary,
    )

    completed_status = {
        "run_name": paths.run_name,
        "state": "completed",
        "rounds_completed": (
            rounds_completed
        ),
        "best_round": best_round,
        "best_validation_loss": (
            best_validation_loss
        ),
        "stopping_reason": (
            stopping_reason
        ),
        "reported_epsilon": (
            reported_epsilon
        ),
        "delta": config.delta,
        "test_macro_f1": float(
            test_metrics["macro_f1"]
        ),
        "test_weighted_f1": float(
            test_metrics["weighted_f1"]
        ),
        "test_macro_auroc": float(
            test_metrics["macro_auroc"]
        ),
    }

    save_progress(
        client_history,
        round_history,
        completed_status,
        paths,
    )

    print(
        "\nDP-UA-FedAvg experiment "
        "completed."
    )
    print(
        f"Best checkpoint: "
        f"{paths.best_checkpoint}"
    )
    print(
        f"Last checkpoint: "
        f"{paths.last_checkpoint}"
    )
    print(
        f"Round history: "
        f"{paths.round_history}"
    )
    print(
        f"Client history: "
        f"{paths.client_history}"
    )
    print(
        f"Run status: "
        f"{paths.status_json}"
    )
    print(
        f"Test results: "
        f"{paths.test_results}"
    )
    print(
        f"Configuration CSV: "
        f"{paths.config_csv}"
    )
    print(
        f"Configuration JSON: "
        f"{paths.config_json}"
    )
    print(
        f"Final summary JSON: "
        f"{paths.final_summary_json}"
    )


if __name__ == "__main__":
    main()