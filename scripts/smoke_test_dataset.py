"""
Verify that PTB-XL records can be loaded through the PyTorch Dataset.
"""

from pathlib import Path
import sys

import pandas as pd
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.ecg_dataset import PTBXLDataset
from data.paths import PROCESSED_DATA_ROOT, get_dataset_root


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
        normalize_per_record=False,
    )

    print("Training records:", len(dataset))

    signal, labels = dataset[0]

    print("\nSingle sample:")
    print("Signal shape:", signal.shape)
    print("Signal dtype:", signal.dtype)
    print("Labels shape:", labels.shape)
    print("Labels:", labels)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
    )

    signal_batch, label_batch = next(
        iter(data_loader)
    )

    print("\nBatch:")
    print("Signal batch shape:", signal_batch.shape)
    print("Label batch shape:", label_batch.shape)

    assert signal.shape == (12, 1000)
    assert labels.shape == (5,)
    assert signal_batch.shape == (8, 12, 1000)
    assert label_batch.shape == (8, 5)

    print("\nDataset smoke test passed.")


if __name__ == "__main__":
    main()
