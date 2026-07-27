"""
Create four patient-level, non-IID PTB-XL hospital partitions.

Important rules:
1. Only the original training split is partitioned.
2. All ECGs belonging to one patient remain in one hospital.
3. Validation and test records are never included.
4. A Dirichlet distribution creates different class proportions
   across the simulated hospitals.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES
from src.data.paths import (
    PARTITIONS_DIR,
    PROCESSED_METADATA_PATH,
)


NUM_HOSPITALS = 4
DIRICHLET_ALPHA = 0.5
SEED = 42

MAX_ATTEMPTS = 100
MIN_PATIENTS_PER_HOSPITAL = 100

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

SUMMARY_PATH = (
    PARTITIONS_DIR / "hospital_partition_summary.csv"
)

CLASS_COUNT_FIGURE_PATH = (
    FIGURES_DIR / "hospital_class_counts.png"
)

CLASS_PREVALENCE_FIGURE_PATH = (
    FIGURES_DIR / "hospital_class_prevalence.png"
)


def assign_primary_class(
    patient_class_counts: pd.DataFrame,
    global_class_counts: pd.Series,
) -> pd.Series:
    """
    Assign one primary class to each patient.

    PTB-XL is multi-label, so one patient may have several positive
    diagnostic classes. The primary class is selected using:

    1. The class appearing most often in that patient's ECGs.
    2. For ties, the globally rarer class is selected.

    The primary class is used only for partitioning. All original
    multi-label targets are preserved in the output CSV files.
    """
    primary_classes: dict[object, str] = {}

    for patient_id, row in patient_class_counts.iterrows():
        highest_count = row.max()

        candidate_classes = [
            class_name
            for class_name in DEFAULT_CLASS_NAMES
            if row[class_name] == highest_count
            and row[class_name] > 0
        ]

        if not candidate_classes:
            raise ValueError(
                f"Patient {patient_id} has no positive superclass labels."
            )

        primary_class = min(
            candidate_classes,
            key=lambda name: global_class_counts[name],
        )

        primary_classes[patient_id] = primary_class

    return pd.Series(
        primary_classes,
        name="primary_class",
    )


def generate_patient_assignment(
    patient_table: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[int, list]:
    """
    Create one patient assignment using class-wise Dirichlet sampling.
    """
    assignments: dict[int, list] = {
        hospital_id: []
        for hospital_id in range(NUM_HOSPITALS)
    }

    for class_name in DEFAULT_CLASS_NAMES:
        class_patients = patient_table.index[
            patient_table["primary_class"] == class_name
        ].to_numpy(copy=True)

        rng.shuffle(class_patients)

        proportions = rng.dirichlet(
            np.full(
                NUM_HOSPITALS,
                DIRICHLET_ALPHA,
            )
        )

        patient_counts = rng.multinomial(
            len(class_patients),
            proportions,
        )

        start_index = 0

        for hospital_id, patient_count in enumerate(
            patient_counts
        ):
            end_index = start_index + patient_count

            assigned_patients = class_patients[
                start_index:end_index
            ].tolist()

            assignments[hospital_id].extend(
                assigned_patients
            )

            start_index = end_index

    return assignments


def assignment_is_valid(
    assignments: dict[int, list],
    total_patients: int,
) -> bool:
    """Check basic validity of a generated assignment."""
    all_assigned_patients = [
        patient_id
        for patient_ids in assignments.values()
        for patient_id in patient_ids
    ]

    if len(all_assigned_patients) != total_patients:
        return False

    if len(set(all_assigned_patients)) != total_patients:
        return False

    for patient_ids in assignments.values():
        if len(patient_ids) < MIN_PATIENTS_PER_HOSPITAL:
            return False

    return True


def create_summary(
    hospital_frames: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Create one summary row for each hospital."""
    rows: list[dict] = []

    for hospital_id, hospital_data in hospital_frames.items():
        row = {
            "hospital_id": hospital_id,
            "records": len(hospital_data),
            "patients": hospital_data[
                "patient_id"
            ].nunique(),
        }

        for class_name in DEFAULT_CLASS_NAMES:
            positive_count = int(
                hospital_data[class_name].sum()
            )

            prevalence = (
                positive_count / len(hospital_data)
                if len(hospital_data) > 0
                else 0.0
            )

            row[f"{class_name}_count"] = positive_count
            row[f"{class_name}_prevalence"] = prevalence

        rows.append(row)

    return pd.DataFrame(rows)


def plot_class_counts(
    summary: pd.DataFrame,
) -> None:
    """Plot positive-label counts for each hospital."""
    plot_data = summary.set_index(
        "hospital_id"
    )[
        [
            f"{class_name}_count"
            for class_name in DEFAULT_CLASS_NAMES
        ]
    ].copy()

    plot_data.columns = DEFAULT_CLASS_NAMES

    axis = plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    axis.set_title(
        "Positive ECG Class Counts by Simulated Hospital"
    )
    axis.set_xlabel("Hospital")
    axis.set_ylabel("Positive ECG records")
    axis.legend(title="Superclass")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        CLASS_COUNT_FIGURE_PATH,
        dpi=150,
    )
    plt.close()


def plot_class_prevalence(
    summary: pd.DataFrame,
) -> None:
    """Plot label prevalence for each hospital."""
    plot_data = summary.set_index(
        "hospital_id"
    )[
        [
            f"{class_name}_prevalence"
            for class_name in DEFAULT_CLASS_NAMES
        ]
    ].copy()

    plot_data.columns = DEFAULT_CLASS_NAMES

    axis = plot_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    axis.set_title(
        "Diagnostic Class Prevalence by Simulated Hospital"
    )
    axis.set_xlabel("Hospital")
    axis.set_ylabel("Positive-label prevalence")
    axis.legend(title="Superclass")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        CLASS_PREVALENCE_FIGURE_PATH,
        dpi=150,
    )
    plt.close()


def main() -> None:
    PARTITIONS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = pd.read_csv(
        PROCESSED_METADATA_PATH,
        index_col="ecg_id",
    )

    required_columns = {
        "patient_id",
        "split",
        "filename_lr",
        *DEFAULT_CLASS_NAMES,
    }

    missing_columns = required_columns.difference(
        metadata.columns
    )

    if missing_columns:
        raise ValueError(
            "Processed metadata is missing columns: "
            f"{sorted(missing_columns)}"
        )

    train_metadata = metadata[
        metadata["split"] == "train"
    ].copy()

    if train_metadata.empty:
        raise RuntimeError(
            "The training split is empty."
        )

    print(
        f"Training records available: "
        f"{len(train_metadata):,}"
    )

    print(
        "Unique training patients: "
        f"{train_metadata['patient_id'].nunique():,}"
    )

    # Count how frequently each class occurs for each patient.
    patient_class_counts = train_metadata.groupby(
        "patient_id"
    )[DEFAULT_CLASS_NAMES].sum()

    global_class_counts = train_metadata[
        DEFAULT_CLASS_NAMES
    ].sum()

    primary_classes = assign_primary_class(
        patient_class_counts=patient_class_counts,
        global_class_counts=global_class_counts,
    )

    patient_table = patient_class_counts.copy()
    patient_table["primary_class"] = primary_classes

    print("\nPrimary-class patient counts:")
    print(
        patient_table["primary_class"]
        .value_counts()
        .reindex(DEFAULT_CLASS_NAMES, fill_value=0)
    )

    rng = np.random.default_rng(SEED)

    assignments = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        candidate_assignments = generate_patient_assignment(
            patient_table=patient_table,
            rng=rng,
        )

        if assignment_is_valid(
            candidate_assignments,
            total_patients=len(patient_table),
        ):
            assignments = candidate_assignments

            print(
                f"\nValid assignment found on attempt {attempt}."
            )
            break

    if assignments is None:
        raise RuntimeError(
            "Could not create a valid hospital assignment. "
            "Try increasing DIRICHLET_ALPHA or reducing "
            "MIN_PATIENTS_PER_HOSPITAL."
        )

    hospital_frames: dict[int, pd.DataFrame] = {}

    for hospital_id, patient_ids in assignments.items():
        hospital_data = train_metadata[
            train_metadata["patient_id"].isin(
                patient_ids
            )
        ].copy()

        hospital_data["hospital_id"] = hospital_id

        hospital_data = hospital_data.reset_index()

        output_columns = [
            "ecg_id",
            "patient_id",
            "hospital_id",
            "filename_lr",
            "split",
            *DEFAULT_CLASS_NAMES,
        ]

        hospital_data = hospital_data[
            output_columns
        ]

        output_path = (
            PARTITIONS_DIR
            / f"hospital_{hospital_id}.csv"
        )

        hospital_data.to_csv(
            output_path,
            index=False,
        )

        hospital_frames[hospital_id] = hospital_data

        print(
            f"Hospital {hospital_id}: "
            f"{len(hospital_data):,} records, "
            f"{hospital_data['patient_id'].nunique():,} patients"
        )

        print(f"  Saved: {output_path}")

    summary = create_summary(
        hospital_frames
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    plot_class_counts(summary)
    plot_class_prevalence(summary)

    print("\nPartition summary:")
    print(
        summary[
            [
                "hospital_id",
                "records",
                "patients",
                *[
                    f"{class_name}_count"
                    for class_name in DEFAULT_CLASS_NAMES
                ],
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSummary saved to: {SUMMARY_PATH}"
    )
    print(
        f"Class-count figure saved to: "
        f"{CLASS_COUNT_FIGURE_PATH}"
    )
    print(
        f"Prevalence figure saved to: "
        f"{CLASS_PREVALENCE_FIGURE_PATH}"
    )


if __name__ == "__main__":
    main()