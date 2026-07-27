"""Project data paths and PTB-XL dataset discovery."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "ptb-xl"
PROCESSED_DATA_ROOT = PROJECT_ROOT / "data" / "processed"


def get_dataset_root() -> Path:
    """Return the configured PTB-XL root and verify its required files."""
    dataset_root = Path(
        os.environ.get("PTBXL_DATASET_ROOT", DEFAULT_DATASET_ROOT)
    ).expanduser().resolve()

    required_paths = (
        dataset_root / "ptbxl_database.csv",
        dataset_root / "scp_statements.csv",
        dataset_root / "records100",
    )
    missing = [path for path in required_paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "PTB-XL dataset is incomplete or its path is incorrect.\n"
            f"Resolved dataset root: {dataset_root}\n"
            f"Missing:\n{formatted}\n"
            "Set PTBXL_DATASET_ROOT to override the default location."
        )

    return dataset_root
