"""
Compact 1D-CNN for PTB-XL multi-label ECG classification.

Input:
    (batch_size, 12, 1000)

Output:
    (batch_size, num_classes) raw logits
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


DROPOUT_TYPES = (
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.Dropout3d,
    nn.AlphaDropout,
)


class ECG1DCNN(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        dropout_p: float = 0.3,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels,
                32,
                kernel_size=7,
                padding=3,
                bias=False,
            ),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(
                32,
                64,
                kernel_size=5,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )

        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, channels, samples), "
                f"but received {tuple(x.shape)}"
            )

        x = self.features(x)
        x = x.squeeze(-1)
        x = self.dropout(x)

        return self.classifier(x)


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Keep the model in evaluation mode while enabling dropout layers.

    Normalization layers remain in evaluation mode.
    """
    model.eval()

    for module in model.modules():
        if isinstance(module, DROPOUT_TYPES):
            module.train()


def binary_entropy(
    probabilities: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate Bernoulli entropy for multi-label probabilities.
    """
    probabilities = probabilities.clamp(
        min=epsilon,
        max=1.0 - epsilon,
    )

    return -(
        probabilities * torch.log(probabilities)
        + (1.0 - probabilities)
        * torch.log(1.0 - probabilities)
    )


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module,
    x: torch.Tensor,
    n_passes: int = 20,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Perform Monte Carlo dropout inference for multi-label classification.
    """
    if n_passes < 2:
        raise ValueError("n_passes must be at least 2")

    original_training_state = model.training

    try:
        enable_mc_dropout(model)

        all_probabilities = torch.stack(
            [
                torch.sigmoid(model(x))
                for _ in range(n_passes)
            ],
            dim=0,
        )

        mean_probabilities = all_probabilities.mean(dim=0)

        probability_variance = all_probabilities.var(
            dim=0,
            unbiased=False,
        ).mean(dim=-1)

        predictive_entropy = binary_entropy(
            mean_probabilities
        ).mean(dim=-1)

        expected_entropy = binary_entropy(
            all_probabilities
        ).mean(dim=0).mean(dim=-1)

        mutual_information = (
            predictive_entropy - expected_entropy
        ).clamp_min(0.0)

        # Normalize entropy values into approximately [0, 1].
        predictive_entropy = predictive_entropy / math.log(2.0)
        expected_entropy = expected_entropy / math.log(2.0)
        mutual_information = mutual_information / math.log(2.0)

        uncertainty = {
            "variance": probability_variance,
            "predictive_entropy": predictive_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
        }

        return mean_probabilities, uncertainty

    finally:
        model.train(original_training_state)


def hospital_uncertainty_score(
    model: nn.Module,
    val_loader,
    n_passes: int = 20,
    device: str | torch.device = "cpu",
    metric: str = "mutual_information",
) -> float:
    """
    Calculate sample-weighted hospital uncertainty.
    """
    device = torch.device(device)
    model.to(device)

    total_uncertainty = 0.0
    total_samples = 0

    for x_batch, _ in val_loader:
        x_batch = x_batch.to(
            device,
            non_blocking=True,
        )

        _, uncertainty = mc_dropout_predict(
            model,
            x_batch,
            n_passes=n_passes,
        )

        batch_scores = uncertainty[metric]

        total_uncertainty += batch_scores.sum().item()
        total_samples += batch_scores.numel()

    if total_samples == 0:
        raise ValueError("Validation loader is empty")

    return total_uncertainty / total_samples


if __name__ == "__main__":
    model = ECG1DCNN()

    dummy_input = torch.randn(8, 12, 1000)

    logits = model(dummy_input)

    print("Logits shape:", logits.shape)

    mean_probabilities, uncertainty = mc_dropout_predict(
        model,
        dummy_input,
        n_passes=10,
    )

    print(
        "Mean probability shape:",
        mean_probabilities.shape,
    )

    for name, values in uncertainty.items():
        print(name, values.shape, values.mean().item())