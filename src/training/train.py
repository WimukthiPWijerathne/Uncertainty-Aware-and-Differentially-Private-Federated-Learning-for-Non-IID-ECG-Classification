"""
Training utilities for centralized ECG classification.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn
from torch.optim import Optimizer


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> float:
    """
    Train the model for one complete epoch.

    Parameters
    ----------
    model:
        PyTorch neural network.

    data_loader:
        Training DataLoader.

    criterion:
        Loss function, normally BCEWithLogitsLoss.

    optimizer:
        PyTorch optimizer.

    device:
        CPU or CUDA device.

    Returns
    -------
    float
        Mean training loss across all samples.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

    for signals, labels in data_loader:
        signals = signals.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        labels = labels.to(
            device=device,
            dtype=torch.float32,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(signals)

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite training loss encountered: {loss.item()}"
            )

        loss.backward()

        optimizer.step()

        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("Training DataLoader contains no samples.")

    return total_loss / total_samples
