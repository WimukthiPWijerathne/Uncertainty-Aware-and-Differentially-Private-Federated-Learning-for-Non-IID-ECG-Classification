"""
Verify patient-level hospital partitions.

Checks:
1. Four hospital files exist.
2. Only training records are present.
3. No ECG occurs in multiple hospitals.
4. No patient occurs in multiple hospitals.
5. Every original training ECG is assigned exactly once.
6. Validation and test records are excluded.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES
from src.data.paths import (
    PARTITIONS_DIR,
    PROCESSED_METADATA_PATH,
)


NUM_HOSPITALS = 4


def main() -> None:
    original_metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    original_train = original_metadata[
        original_metadata["split"] == "train"
    ].copy()

    hospital_frames: list[pd.DataFrame] = []

    for hospital_id in range(NUM_HOSPITALS):
        path = (
            PARTITIONS_DIR
            / f"hospital_{hospital_id}.csv"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Missing hospital partition: {path}"
            )

        hospital_data = pd.read_csv(path)

        required_columns = {
            "ecg_id",
            "patient_id",
            "hospital_id",
            "filename_lr",
            "split",
            *DEFAULT_CLASS_NAMES,
        }

        missing_columns = required_columns.difference(
            hospital_data.columns
        )

        if missing_columns:
            raise ValueError(
                f"Hospital {hospital_id} is missing columns: "
                f"{sorted(missing_columns)}"
            )

        if not (
            hospital_data["hospital_id"] == hospital_id
        ).all():
            raise ValueError(
                f"Hospital ID mismatch in {path}"
            )

        if not (
            hospital_data["split"] == "train"
        ).all():
            raise ValueError(
                f"Non-training records found in hospital {hospital_id}"
            )

        hospital_frames.append(hospital_data)

        print(
            f"Hospital {hospital_id}: "
            f"{len(hospital_data):,} records, "
            f"{hospital_data['patient_id'].nunique():,} patients"
        )

    combined = pd.concat(
        hospital_frames,
        ignore_index=True,
    )

    duplicated_ecgs = combined[
        "ecg_id"
    ].duplicated().sum()

    if duplicated_ecgs:
        raise ValueError(
            f"{duplicated_ecgs} duplicated ECG IDs were found."
        )

    patient_hospital_counts = combined.groupby(
        "patient_id"
    )["hospital_id"].nunique()

    overlapping_patients = patient_hospital_counts[
        patient_hospital_counts > 1
    ]

    if not overlapping_patients.empty:
        raise ValueError(
            f"{len(overlapping_patients)} patients appear "
            "in multiple hospitals."
        )

    original_ecg_ids = set(
        original_train.index.tolist()
    )

    partitioned_ecg_ids = set(
        combined["ecg_id"].tolist()
    )

    missing_ecgs = (
        original_ecg_ids - partitioned_ecg_ids
    )

    unexpected_ecgs = (
        partitioned_ecg_ids - original_ecg_ids
    )

    if missing_ecgs:
        raise ValueError(
            f"{len(missing_ecgs)} training ECGs were not assigned."
        )

    if unexpected_ecgs:
        raise ValueError(
            f"{len(unexpected_ecgs)} unexpected ECGs were assigned."
        )

    print("\nVerification results")
    print(
        f"Original training records: "
        f"{len(original_train):,}"
    )
    print(
        f"Partitioned records: "
        f"{len(combined):,}"
    )
    print(
        f"Partitioned patients: "
        f"{combined['patient_id'].nunique():,}"
    )
    print("Duplicated ECG IDs: 0")
    print("Patients shared between hospitals: 0")
    print("Validation records included: 0")
    print("Test records included: 0")
    print(
        "\nHospital partition verification passed."
    )


if __name__ == "__main__":
    main()