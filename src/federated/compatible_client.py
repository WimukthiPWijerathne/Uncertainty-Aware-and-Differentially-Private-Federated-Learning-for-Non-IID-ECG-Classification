"""
Non-private federated client using the same Opacus-compatible
architecture as DP-FedAvg.

This control isolates the effect of architecture conversion
(e.g., BatchNorm replacement) from DP clipping and noise.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.ecg_dataset import (
    DEFAULT_CLASS_NAMES,
    PTBXLDataset,
)
from src.data.paths import PTBXL_ROOT
from src.federated.dp_client import (
    create_dp_compatible_ecg_model,
)
from src.federated.parameter_utils import (
    get_model_parameters,
    set_model_parameters,
)
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch


class CompatibleHospitalClient:
    """
    Standard non-private local client using the exact model
    architecture used by DP-FedAvg.
    """

    def __init__(
        self,
        hospital_id: int,
        train_metadata: pd.DataFrame,
        validation_metadata: pd.DataFrame,
        device: torch.device,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        local_epochs: int = 1,
        num_workers: int = 2,
        threshold: float = 0.5,
        normalize_per_record: bool = False,
    ) -> None:
        self.hospital_id = hospital_id
        self.device = device
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.threshold = threshold

        train_dataset = PTBXLDataset(
            metadata=train_metadata,
            dataset_root=PTBXL_ROOT,
            class_names=DEFAULT_CLASS_NAMES,
            normalize_per_record=normalize_per_record,
        )

        validation_dataset = PTBXLDataset(
            metadata=validation_metadata,
            dataset_root=PTBXL_ROOT,
            class_names=DEFAULT_CLASS_NAMES,
            normalize_per_record=normalize_per_record,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        self.validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
            drop_last=False,
        )

        self.num_train_samples = len(train_dataset)

        self.model = create_dp_compatible_ecg_model(
            device=device,
        )

        self.criterion = nn.BCEWithLogitsLoss()

    def fit(
        self,
        global_parameters: list,
    ) -> tuple[list, int, dict[str, Any]]:
        """Train one non-private local update."""
        set_model_parameters(
            self.model,
            global_parameters,
        )

        # A fresh optimizer each round matches the original
        # standard FedAvg client protocol.
        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        losses: list[float] = []

        for local_epoch in range(
            1,
            self.local_epochs + 1,
        ):
            train_loss = train_one_epoch(
                model=self.model,
                data_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=optimizer,
                device=self.device,
            )

            losses.append(
                float(train_loss)
            )

            print(
                f"  Local epoch "
                f"{local_epoch}/{self.local_epochs} "
                f"loss={train_loss:.6f}"
            )

        validation_metrics = evaluate_model(
            model=self.model,
            data_loader=self.validation_loader,
            criterion=self.criterion,
            device=self.device,
            threshold=self.threshold,
        )

        metrics: dict[str, Any] = {
            "train_loss": sum(losses) / len(losses),
            "validation_loss": float(
                validation_metrics["loss"]
            ),
            "validation_macro_f1": float(
                validation_metrics["macro_f1"]
            ),
            "validation_weighted_f1": float(
                validation_metrics["weighted_f1"]
            ),
            "validation_macro_auroc": float(
                validation_metrics["macro_auroc"]
            ),
        }

        return (
            get_model_parameters(self.model),
            self.num_train_samples,
            metrics,
        )
