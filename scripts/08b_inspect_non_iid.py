from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data.ecg_dataset import DEFAULT_CLASS_NAMES
from src.data.paths import PARTITIONS_DIR


SUMMARY_PATH = (
    PARTITIONS_DIR / "hospital_partition_summary.csv"
)

MIN_RECORDS = 500
MIN_PATIENTS = 300
MIN_POSITIVES_PER_CLASS = 20
MIN_PREVALENCE_RANGE = 0.05


def main() -> None:
    if not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Partition summary was not found: {SUMMARY_PATH}"
        )

    summary = pd.read_csv(SUMMARY_PATH)

    print("Hospital sizes")
    print(
        summary[
            ["hospital_id", "records", "patients"]
        ].to_string(index=False)
    )

    print("\nClass counts")
    print(
        summary[
            [
                "hospital_id",
                *[
                    f"{name}_count"
                    for name in DEFAULT_CLASS_NAMES
                ],
            ]
        ].to_string(index=False)
    )

    print("\nClass prevalence")
    prevalence_columns = [
        f"{name}_prevalence"
        for name in DEFAULT_CLASS_NAMES
    ]

    print(
        summary[
            ["hospital_id", *prevalence_columns]
        ].round(4).to_string(index=False)
    )

    problems: list[str] = []

    for _, row in summary.iterrows():
        hospital_id = int(row["hospital_id"])

        if row["records"] < MIN_RECORDS:
            problems.append(
                f"Hospital {hospital_id} has only "
                f"{int(row['records'])} records."
            )

        if row["patients"] < MIN_PATIENTS:
            problems.append(
                f"Hospital {hospital_id} has only "
                f"{int(row['patients'])} patients."
            )

        for class_name in DEFAULT_CLASS_NAMES:
            positive_count = int(
                row[f"{class_name}_count"]
            )

            if positive_count < MIN_POSITIVES_PER_CLASS:
                problems.append(
                    f"Hospital {hospital_id} has only "
                    f"{positive_count} positive {class_name} records."
                )

    print("\nPrevalence ranges across hospitals")

    non_iid_classes = 0

    for class_name in DEFAULT_CLASS_NAMES:
        column = f"{class_name}_prevalence"

        minimum = summary[column].min()
        maximum = summary[column].max()
        difference = maximum - minimum

        print(
            f"{class_name}: "
            f"min={minimum:.4f}, "
            f"max={maximum:.4f}, "
            f"range={difference:.4f}"
        )

        if difference >= MIN_PREVALENCE_RANGE:
            non_iid_classes += 1

    if non_iid_classes < 2:
        problems.append(
            "The hospital distributions may be too similar. "
            "Fewer than two classes have a prevalence range "
            f"of at least {MIN_PREVALENCE_RANGE:.2f}."
        )

    if problems:
        print("\nWarnings")

        for problem in problems:
            print(f"  - {problem}")

        print(
            "\nConsider regenerating the partitions with a "
            "different Dirichlet alpha or random seed."
        )
    else:
        print(
            "\nThe hospital partitions have reasonable sizes, "
            "class coverage, and visible non-IID variation."
        )


if __name__ == "__main__":
    main()