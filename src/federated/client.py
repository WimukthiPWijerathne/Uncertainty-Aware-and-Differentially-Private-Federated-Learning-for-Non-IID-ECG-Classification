"""
Federated-learning client for one simulated hospital.
"""

from __future__ import annotations

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
from src.federated.parameter_utils import (
    get_model_parameters,
    set_model_parameters,
)
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch


class HospitalClient:
    """One simulated hospital participating in FedAvg."""

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

        self.model = ECG1DCNN(
            in_channels=12,
            num_classes=len(DEFAULT_CLASS_NAMES),
            dropout_p=0.3,
        ).to(device)

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
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self) -> list:
        """Return this client's current model parameters."""
        return get_model_parameters(self.model)

    def set_parameters(
        self,
        parameters: list,
    ) -> None:
        """Load global parameters into the client model."""
        set_model_parameters(
            self.model,
            parameters,
        )

    def fit(
        self,
        global_parameters: list,
    ) -> tuple[list, int, dict]:
        """
        Train locally from the current global parameters.

        Returns:
            updated_parameters
            number_of_training_samples
            client_metrics
        """
        self.set_parameters(global_parameters)

        optimizer = Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        training_losses: list[float] = []

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

            training_losses.append(train_loss)

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

        mean_train_loss = (
            sum(training_losses)
            / len(training_losses)
        )

        metrics = {
            "hospital_id": self.hospital_id,
            "train_loss": float(mean_train_loss),
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
            self.get_parameters(),
            self.num_train_samples,
            metrics,
        )