"""
Monte Carlo dropout uncertainty estimation.

The model is evaluated multiple times with dropout active.
Batch-normalization layers remain in evaluation mode.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


def _enable_dropout_only(
    module: nn.Module,
) -> None:
    """
    Enable dropout layers while keeping the rest of the
    model, including batch normalization, in evaluation mode.
    """
    if isinstance(
        module,
        (
            nn.Dropout,
            nn.Dropout1d,
            nn.Dropout2d,
            nn.Dropout3d,
        ),
    ):
        module.train()


def estimate_predictive_uncertainty(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mc_passes: int = 10,
) -> float:
    """
    Estimate client predictive uncertainty using MC dropout.

    For every batch:
    1. Run multiple stochastic forward passes.
    2. Convert logits to sigmoid probabilities.
    3. Calculate variance across MC passes.
    4. Average variance across samples and classes.

    Returns:
        One scalar mean predictive-variance value.
    """
    if mc_passes < 2:
        raise ValueError(
            "mc_passes must be at least 2."
        )

    previous_training_state = model.training

    # Put the complete model into evaluation mode first.
    # This prevents batch-normalization statistics from changing.
    model.eval()

    # Reactivate only dropout layers.
    model.apply(_enable_dropout_only)

    total_variance = 0.0
    total_probability_values = 0

    try:
        with torch.inference_mode():
            for signals, _ in data_loader:
                signals = signals.to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )

                stochastic_predictions = []

                for _ in range(mc_passes):
                    logits = model(signals)

                    probabilities = torch.sigmoid(
                        logits
                    )

                    stochastic_predictions.append(
                        probabilities
                    )

                stacked_predictions = torch.stack(
                    stochastic_predictions,
                    dim=0,
                )

                predictive_variance = torch.var(
                    stacked_predictions,
                    dim=0,
                    unbiased=False,
                )

                total_variance += float(
                    predictive_variance.sum().item()
                )

                total_probability_values += (
                    predictive_variance.numel()
                )

    finally:
        # Restore the model's previous overall mode.
        if previous_training_state:
            model.train()
        else:
            model.eval()

    if total_probability_values == 0:
        raise RuntimeError(
            "Uncertainty DataLoader produced no samples."
        )

    mean_uncertainty = (
        total_variance
        / total_probability_values
    )

    return float(mean_uncertainty)