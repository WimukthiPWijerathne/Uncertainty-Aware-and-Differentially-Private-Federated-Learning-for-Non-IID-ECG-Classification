"""
PyTorch dataset for PTB-XL 100 Hz waveform records.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import Dataset


DEFAULT_CLASS_NAMES = [
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
]


class PTBXLDataset(Dataset):
    """
    Lazily loads PTB-XL waveform records.

    WFDB returns each ECG as:
        (time_samples, leads) = (1000, 12)

    Conv1d requires:
        (channels, sequence_length) = (12, 1000)
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        dataset_root: str | Path,
        class_names: Sequence[str] = DEFAULT_CLASS_NAMES,
        normalize_per_record: bool = False,
    ) -> None:
        self.metadata = metadata.reset_index(drop=False).copy()
        self.dataset_root = Path(dataset_root)
        self.class_names = list(class_names)
        self.normalize_per_record = normalize_per_record

        missing_classes = set(self.class_names).difference(
            self.metadata.columns
        )

        if missing_classes:
            raise ValueError(
                f"Missing label columns: {sorted(missing_classes)}"
            )

        if "filename_lr" not in self.metadata.columns:
            raise ValueError("Metadata is missing the 'filename_lr' column")

        if not self.dataset_root.is_dir():
            raise FileNotFoundError(
                f"PTB-XL dataset root was not found: {self.dataset_root}"
            )

    def __len__(self) -> int:
        return len(self.metadata)

    def _normalize_signal(
        self,
        signal: np.ndarray,
    ) -> np.ndarray:
        """
        Perform optional per-lead, per-record standardization.

        This is useful for early experiments but can reduce absolute
        amplitude information. Keep it configurable.
        """
        mean = signal.mean(
            axis=1,
            keepdims=True,
        )

        standard_deviation = signal.std(
            axis=1,
            keepdims=True,
        )

        standard_deviation = np.maximum(
            standard_deviation,
            1e-6,
        )

        return (
            signal - mean
        ) / standard_deviation

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]

        record_path = (
            self.dataset_root / str(row["filename_lr"])
        )

        if not record_path.with_suffix(".hea").is_file():
            raise FileNotFoundError(
                f"PTB-XL waveform header was not found: "
                f"{record_path.with_suffix('.hea')}"
            )

        signal, signal_information = wfdb.rdsamp(
            str(record_path)
        )

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        signal = np.nan_to_num(
            signal,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        expected_leads = 12
        expected_samples = 1000

        if signal.shape != (
            expected_samples,
            expected_leads,
        ):
            raise ValueError(
                f"Unexpected ECG shape for {record_path}: "
                f"{signal.shape}. Expected "
                f"({expected_samples}, {expected_leads})."
            )

        # Convert from (1000, 12) to (12, 1000).
        signal = signal.T

        if self.normalize_per_record:
            signal = self._normalize_signal(signal)

        labels = row[self.class_names].to_numpy(
            dtype=np.float32
        )

        signal_tensor = torch.from_numpy(
            signal.copy()
        )

        label_tensor = torch.from_numpy(
            labels.copy()
        )

        return signal_tensor, label_tensor
