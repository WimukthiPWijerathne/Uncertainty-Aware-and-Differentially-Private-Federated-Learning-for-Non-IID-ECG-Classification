"""
Differentially private, uncertainty-aware federated client.

This client reuses the validated DP-SGD implementation from
``src.federated.dp_client.DPHospitalClient`` and adds MC-dropout
predictive-uncertainty estimation after each private local update.
"""

from __future__ import annotations

import math
from typing import Any

from src.federated.dp_client import DPHospitalClient
from src.federated.uncertainty import estimate_predictive_uncertainty


class DPUAHospitalClient(DPHospitalClient):
    """
    Hospital client using both DP-SGD and MC-dropout uncertainty.

    The parent class:
      * receives global model parameters,
      * performs private local training,
      * evaluates the local model,
      * updates the hospital-specific privacy accountant,
      * returns the private model parameters and metrics.

    This subclass then estimates predictive uncertainty using the
    existing validation loader and appends it to the returned metrics.
    """

    def __init__(
        self,
        *args: Any,
        mc_passes: int = 10,
        **kwargs: Any,
    ) -> None:
        if mc_passes < 2:
            raise ValueError(
                "mc_passes must be at least 2 so that predictive "
                "variance can be estimated."
            )

        super().__init__(*args, **kwargs)
        self.mc_passes = int(mc_passes)

    def fit(
        self,
        global_parameters: list,
    ) -> tuple[list, int, dict[str, Any]]:
        """
        Perform one private local update and estimate uncertainty.

        Returns
        -------
        parameters:
            Updated local model parameters.
        sample_count:
            Number of local training records.
        metrics:
            Local training, validation, privacy and uncertainty metrics.
        """
        (
            parameters,
            sample_count,
            metrics,
        ) = super().fit(global_parameters)

        uncertainty = estimate_predictive_uncertainty(
            model=self.model,
            data_loader=self.validation_loader,
            device=self.device,
            mc_passes=self.mc_passes,
        )

        uncertainty = float(uncertainty)

        if not math.isfinite(uncertainty):
            raise RuntimeError(
                f"Hospital {self.hospital_id} produced a non-finite "
                f"uncertainty value: {uncertainty}"
            )

        if uncertainty < 0:
            raise RuntimeError(
                f"Hospital {self.hospital_id} produced a negative "
                f"uncertainty value: {uncertainty}"
            )

        metrics["uncertainty"] = uncertainty
        metrics["mc_passes"] = self.mc_passes

        print(
            f"  Predictive uncertainty="
            f"{uncertainty:.8f} "
            f"(MC passes={self.mc_passes})"
        )

        return (
            parameters,
            sample_count,
            metrics,
        )