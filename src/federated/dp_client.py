"""
Differentially private federated client using Opacus.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import torch
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES, PTBXLDataset
from src.data.paths import PTBXL_ROOT
from src.federated.parameter_utils import get_model_parameters, set_model_parameters
from src.models.cnn1d import ECG1DCNN
from src.training.evaluate import evaluate_model
from src.training.train import train_one_epoch


def disable_inplace_activations(
    model: nn.Module,
) -> None:
    """
    Disable in-place activation operations.

    Opacus uses backward hooks for per-sample gradients.
    In-place activations can modify hook-generated tensor
    views and cause autograd errors.
    """
    inplace_activation_types = (
        nn.ReLU,
        nn.ReLU6,
        nn.LeakyReLU,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.RReLU,
        nn.SiLU,
        nn.Hardswish,
    )

    for module in model.modules():
        if isinstance(
            module,
            inplace_activation_types,
        ):
            module.inplace = False


def create_dp_compatible_ecg_model(
    device: torch.device,
) -> nn.Module:
    """
    Create an Opacus-compatible ECG model.
    """
    model = ECG1DCNN(
        in_channels=12,
        num_classes=len(DEFAULT_CLASS_NAMES),
        dropout_p=0.3,
    )

    # Prevent backward-hook conflicts with in-place ReLU.
    disable_inplace_activations(model)

    # Replace unsupported trainable modules, such as
    # BatchNorm, with privacy-compatible alternatives.
    model = ModuleValidator.fix(model)

    # Apply again in case the model fixer introduced or
    # retained any in-place activation modules.
    disable_inplace_activations(model)

    ModuleValidator.validate(
        model,
        strict=True,
    )

    return model.to(device)


class DPHospitalClient:
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
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
        delta: float = 1e-5,
        accountant: str = "prv",
        secure_mode: bool = False,
    ) -> None:
        self.hospital_id = hospital_id
        self.device = device
        self.local_epochs = local_epochs
        self.threshold = threshold
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.delta = delta

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

        ordinary_train_loader = DataLoader(
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
        model = create_dp_compatible_ecg_model(device)
        optimizer = Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCEWithLogitsLoss()

        self.privacy_engine = PrivacyEngine(
            accountant=accountant,
            secure_mode=secure_mode,
        )

        self.model, self.optimizer, self.train_loader = (
            self.privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=ordinary_train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=True,
                clipping="flat",
                grad_sample_mode="hooks",
            )
        )

    def get_parameters(self) -> list:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: list) -> None:
        set_model_parameters(self.model, parameters)

    def fit(
        self,
        global_parameters: list,
    ) -> tuple[list, int, dict[str, Any]]:
        self.set_parameters(global_parameters)

        # Reset Adam state to match the non-DP FedAvg client behavior.
        self.optimizer.state.clear()

        losses: list[float] = []

        for local_epoch in range(1, self.local_epochs + 1):
            loss = train_one_epoch(
                model=self.model,
                data_loader=self.train_loader,
                criterion=self.criterion,
                optimizer=self.optimizer,
                device=self.device,
            )
            losses.append(float(loss))
            print(
                f"  Private local epoch "
                f"{local_epoch}/{self.local_epochs} "
                f"loss={loss:.6f}"
            )

        validation_metrics = evaluate_model(
            model=self.model,
            data_loader=self.validation_loader,
            criterion=self.criterion,
            device=self.device,
            threshold=self.threshold,
        )

        epsilon = float(
            self.privacy_engine.get_epsilon(delta=self.delta)
        )

        metrics: dict[str, Any] = {
            "hospital_id": self.hospital_id,
            "train_loss": sum(losses) / len(losses),
            "validation_loss": float(validation_metrics["loss"]),
            "validation_macro_f1": float(validation_metrics["macro_f1"]),
            "validation_weighted_f1": float(
                validation_metrics["weighted_f1"]
            ),
            "validation_macro_auroc": float(
                validation_metrics["macro_auroc"]
            ),
            "epsilon": epsilon,
            "delta": self.delta,
            "noise_multiplier": self.noise_multiplier,
            "max_grad_norm": self.max_grad_norm,
        }

        print(
            f"  Privacy: epsilon={epsilon:.4f}, "
            f"delta={self.delta:.1e}"
        )

        return self.get_parameters(), self.num_train_samples, metrics