from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.ecg_dataset import PTBXLDataset
from src.models.ecg_cnn import ECG1DCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "ptb-xl"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ptbxl_superclasses.csv"
)


def main() -> None:
    metadata = pd.read_csv(
        METADATA_PATH,
        index_col="ecg_id",
    )

    train_metadata = metadata[
        metadata["split"] == "train"
    ].copy()

    dataset = PTBXLDataset(
        metadata=train_metadata,
        dataset_root=DATASET_ROOT,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    signals, labels = next(iter(loader))

    model = ECG1DCNN(
        in_channels=12,
        num_classes=5,
        dropout_p=0.3,
    )

    logits = model(signals)

    criterion = torch.nn.BCEWithLogitsLoss()

    loss = criterion(
        logits,
        labels.float(),
    )

    probabilities = torch.sigmoid(logits)

    print("Input shape:", signals.shape)
    print("Label shape:", labels.shape)
    print("Logit shape:", logits.shape)
    print("Probability shape:", probabilities.shape)
    print("Initial loss:", loss.item())

    loss.backward()

    print("Forward and backward passes succeeded.")


if __name__ == "__main__":
    main()net