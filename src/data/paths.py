"""Project data paths and PTB-XL dataset discovery."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw"
PTBXL_ROOT = RAW_DATA_DIR / "ptb-xl"

PROCESSED_DATA_DIR = DATA_DIR / "processed"
PARTITIONS_DIR = DATA_DIR / "partitions"

PTBXL_METADATA_PATH = PTBXL_ROOT / "ptbxl_database.csv"
SCP_STATEMENTS_PATH = PTBXL_ROOT / "scp_statements.csv"

PROCESSED_METADATA_PATH = (
    PROCESSED_DATA_DIR / "ptbxl_superclasses.csv"
)
CLASS_NAMES_PATH = PROCESSED_DATA_DIR / "class_names.txt"

RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = RESULTS_DIR / "checkpoints"
LOGS_DIR = RESULTS_DIR / "logs"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Backward-compatible names used by the existing preparation and smoke tests.
DEFAULT_DATASET_ROOT = PTBXL_ROOT
PROCESSED_DATA_ROOT = PROCESSED_DATA_DIR


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
