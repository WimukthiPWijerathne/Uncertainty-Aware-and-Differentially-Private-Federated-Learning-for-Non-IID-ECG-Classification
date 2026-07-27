"""Verify the local PTB-XL metadata and waveform files."""

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import get_dataset_root


def main() -> None:
    dataset_root = get_dataset_root()
    metadata = pd.read_csv(dataset_root / "ptbxl_database.csv")
    missing: list[Path] = []

    for column in ("filename_lr", "filename_hr"):
        for record in metadata[column].astype(str):
            record_path = dataset_root / record
            missing.extend(
                path
                for path in (
                    record_path.with_suffix(".hea"),
                    record_path.with_suffix(".dat"),
                )
                if not path.is_file()
            )

    if missing:
        examples = "\n".join(f"  - {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} waveform files. Examples:\n{examples}"
        )

    print(f"Dataset root: {dataset_root}")
    print(f"Metadata records: {len(metadata):,}")
    print("All 100 Hz and 500 Hz waveform pairs are present.")


if __name__ == "__main__":
    main()
