"""
Evaluation utilities for multi-label ECG classification.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
)
from torch import nn


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate a multi-label ECG classification model.

    Returns validation loss, macro-F1, weighted-F1,
    macro-AUROC and per-class metrics.
    """
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for signals, labels in data_loader:
        signals = signals.to(
            device=device,
            dtype=torch.float32,
        )

        labels = labels.to(
            device=device,
            dtype=torch.float32,
        )

        logits = model(signals)

        loss = criterion(
            logits,
            labels,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite validation loss encountered: {loss.item()}"
            )

        probabilities = torch.sigmoid(logits)

        batch_size = signals.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.append(
            labels.cpu().numpy()
        )

        all_probabilities.append(
            probabilities.cpu().numpy()
        )

    if total_samples == 0:
        raise RuntimeError("Validation DataLoader contains no samples.")

    labels_array = np.concatenate(
        all_labels,
        axis=0,
    )

    probability_array = np.concatenate(
        all_probabilities,
        axis=0,
    )

    prediction_array = (
        probability_array >= threshold
    ).astype(np.int32)

    macro_f1 = f1_score(
        labels_array,
        prediction_array,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels_array,
        prediction_array,
        average="weighted",
        zero_division=0,
    )

    per_class_f1 = f1_score(
        labels_array,
        prediction_array,
        average=None,
        zero_division=0,
    )

    per_class_auroc: list[float] = []

    for class_index in range(labels_array.shape[1]):
        class_labels = labels_array[:, class_index]
        class_probabilities = probability_array[:, class_index]

        # AUROC is undefined when a class contains only one
        # target value in the evaluated subset.
        if np.unique(class_labels).size < 2:
            per_class_auroc.append(float("nan"))
            continue

        class_auroc = roc_auc_score(
            class_labels,
            class_probabilities,
        )

        per_class_auroc.append(
            float(class_auroc)
        )

    valid_aurocs = [
        score
        for score in per_class_auroc
        if not np.isnan(score)
    ]

    macro_auroc = (
        float(np.mean(valid_aurocs))
        if valid_aurocs
        else float("nan")
    )

    return {
        "loss": total_loss / total_samples,
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_auroc": macro_auroc,
        "per_class_f1": per_class_f1.tolist(),
        "per_class_auroc": per_class_auroc,
        "labels": labels_array,
        "probabilities": probability_array,
    }
