"""
Uncertainty-aware federated client.
"""

from __future__ import annotations

import pandas as pd
import torch

from src.federated.client import HospitalClient
from src.federated.uncertainty import (
    estimate_predictive_uncertainty,
)


class UncertaintyAwareHospitalClient(
    HospitalClient
):
    """
    Hospital client that estimates predictive uncertainty
    after local training.
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
        mc_passes: int = 10,
    ) -> None:
        super().__init__(
            hospital_id=hospital_id,
            train_metadata=train_metadata,
            validation_metadata=validation_metadata,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            local_epochs=local_epochs,
            num_workers=num_workers,
            threshold=threshold,
            normalize_per_record=(
                normalize_per_record
            ),
        )

        if mc_passes < 2:
            raise ValueError(
                "mc_passes must be at least 2."
            )

        self.mc_passes = mc_passes

    def fit(
        self,
        global_parameters: list,
    ) -> tuple[list, int, dict]:
        """
        Train locally and calculate MC-dropout uncertainty.
        """
        (
            updated_parameters,
            sample_count,
            metrics,
        ) = super().fit(
            global_parameters
        )

        uncertainty = (
            estimate_predictive_uncertainty(
                model=self.model,
                data_loader=self.validation_loader,
                device=self.device,
                mc_passes=self.mc_passes,
            )
        )

        metrics["uncertainty"] = float(
            uncertainty
        )

        print(
            f"  Predictive uncertainty="
            f"{uncertainty:.8f}"
        )

        return (
            updated_parameters,
            sample_count,
            metrics,
        )