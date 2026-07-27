from pathlib import Path
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.ecg_dataset import PTBXLDataset
from src.data.paths import PROCESSED_DATA_ROOT, get_dataset_root
from src.models.cnn1d import ECG1DCNN


DATASET_ROOT = get_dataset_root()
METADATA_PATH = PROCESSED_DATA_ROOT / "ptbxl_superclasses.csv"


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
    main()
