"""
Prepare PTB-XL metadata for five-superclass multi-label classification.

Outputs:
    data/processed/ptbxl_superclasses.csv
    data/processed/class_names.txt
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.paths import PROCESSED_DATA_ROOT, get_dataset_root


DATASET_ROOT = get_dataset_root()
OUTPUT_DIRECTORY = PROCESSED_DATA_ROOT

METADATA_PATH = DATASET_ROOT / "ptbxl_database.csv"
STATEMENTS_PATH = DATASET_ROOT / "scp_statements.csv"

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


def extract_superclasses(
    code_dictionary: dict[str, float],
    diagnostic_mapping: dict[str, str],
) -> list[str]:
    """
    Convert detailed SCP-ECG diagnostic codes into the five selected
    PTB-XL diagnostic superclasses.
    """
    labels = {
        diagnostic_mapping[code]
        for code in code_dictionary
        if code in diagnostic_mapping
        and diagnostic_mapping[code] in CLASS_NAMES
    }

    return sorted(labels)


def assign_split(strat_fold: int) -> str:
    """
    Use the official PTB-XL patient-safe split.

    Folds 1-8: training
    Fold 9: validation
    Fold 10: testing
    """
    if strat_fold <= 8:
        return "train"

    if strat_fold == 9:
        return "validation"

    if strat_fold == 10:
        return "test"

    raise ValueError(f"Unexpected strat_fold value: {strat_fold}")


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file was not found: {METADATA_PATH}"
        )

    if not STATEMENTS_PATH.exists():
        raise FileNotFoundError(
            f"SCP statements file was not found: {STATEMENTS_PATH}"
        )

    metadata = pd.read_csv(
        METADATA_PATH,
        index_col="ecg_id",
    )

    statements = pd.read_csv(
        STATEMENTS_PATH,
        index_col=0,
    )

    print("Original metadata shape:", metadata.shape)
    print("SCP statements shape:", statements.shape)

    required_columns = {
        "patient_id",
        "scp_codes",
        "strat_fold",
        "filename_lr",
    }

    missing_columns = required_columns.difference(metadata.columns)

    if missing_columns:
        raise ValueError(
            f"Missing metadata columns: {sorted(missing_columns)}"
        )

    # Convert the string representation of the SCP dictionary
    # into a real Python dictionary.
    metadata["scp_codes"] = metadata["scp_codes"].apply(
        ast.literal_eval
    )

    # Keep only diagnostic statements.
    diagnostic_statements = statements[
        statements["diagnostic"] == 1
    ].copy()

    diagnostic_mapping = (
        diagnostic_statements["diagnostic_class"]
        .dropna()
        .to_dict()
    )

    metadata["diagnostic_superclasses"] = metadata[
        "scp_codes"
    ].apply(
        lambda codes: extract_superclasses(
            codes,
            diagnostic_mapping,
        )
    )

    # Remove records that do not belong to one of the five classes.
    metadata = metadata[
        metadata["diagnostic_superclasses"].map(len) > 0
    ].copy()

    for class_name in CLASS_NAMES:
        metadata[class_name] = metadata[
            "diagnostic_superclasses"
        ].map(lambda labels: np.float32(class_name in labels))

    metadata["split"] = metadata["strat_fold"].apply(
        assign_split
    )

    selected_columns = [
        "patient_id",
        "age",
        "sex",
        "height",
        "weight",
        "strat_fold",
        "split",
        "filename_lr",
        "diagnostic_superclasses",
        *CLASS_NAMES,
    ]

    # Keep only columns that are actually present.
    selected_columns = [
        column
        for column in selected_columns
        if column in metadata.columns
    ]

    output_metadata = metadata[selected_columns].copy()

    output_path = (
        OUTPUT_DIRECTORY / "ptbxl_superclasses.csv"
    )

    output_metadata.to_csv(
        output_path,
        index=True,
    )

    class_names_path = (
        OUTPUT_DIRECTORY / "class_names.txt"
    )

    class_names_path.write_text(
        "\n".join(CLASS_NAMES),
        encoding="utf-8",
    )

    print("\nProcessed metadata shape:", output_metadata.shape)

    print("\nSplit counts:")
    print(output_metadata["split"].value_counts())

    print("\nClass counts:")
    print(
        output_metadata[CLASS_NAMES]
        .sum()
        .astype(int)
    )

    print("\nRecords with multiple labels:")

    multi_label_count = (
        output_metadata[CLASS_NAMES].sum(axis=1) > 1
    ).sum()

    print(
        f"{multi_label_count:,} / "
        f"{len(output_metadata):,}"
    )

    print("\nSaved:")
    print(output_path)
    print(class_names_path)


if __name__ == "__main__":
    main()
