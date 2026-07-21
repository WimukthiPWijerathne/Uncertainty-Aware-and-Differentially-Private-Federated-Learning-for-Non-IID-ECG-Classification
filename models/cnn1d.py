"""
Shared 1D-CNN architecture for PTB-XL ECG classification.
Input shape: (batch, 12, 1000) — 12 leads, 10 seconds at 100Hz.
Output: (batch, num_classes) raw logits — default 5 PTB-XL superclasses.
"""

import torch
import torch.nn as nn


class ECG1DCNN(nn.Module):
    def __init__(self, in_channels: int = 12, num_classes: int = 5, dropout_p: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


def enable_mc_dropout(model: nn.Module) -> None:
    """Eval mode overall, but re-enable Dropout layers for MC sampling."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_passes: int = 20):
    enable_mc_dropout(model)
    all_probs = torch.stack(
        [torch.softmax(model(x), dim=-1) for _ in range(n_passes)]
    )
    mean_probs = all_probs.mean(dim=0)
    uncertainty = all_probs.var(dim=0).mean(dim=-1)
    return mean_probs, uncertainty


def hospital_uncertainty_score(model: nn.Module, val_loader, n_passes: int = 20, device: str = "cpu") -> float:
    model.to(device)
    scores = []
    for x_batch, _ in val_loader:
        x_batch = x_batch.to(device)
        _, uncertainty = mc_dropout_predict(model, x_batch, n_passes=n_passes)
        scores.append(uncertainty.mean().item())
    return sum(scores) / len(scores)


if __name__ == "__main__":
    model = ECG1DCNN()
    dummy_input = torch.randn(8, 12, 1000)
    out = model(dummy_input)
    print("Forward pass output shape:", out.shape)

    mean_probs, uncertainty = mc_dropout_predict(model, dummy_input, n_passes=10)
    print("MC dropout mean_probs shape:", mean_probs.shape)
    print("MC dropout uncertainty shape:", uncertainty.shape)