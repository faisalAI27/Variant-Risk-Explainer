#!/usr/bin/env python3
"""Check the larger ClinVar datasets before DNABERT-2 training.

This script checks generated larger datasets when present:
- training/csv_files_20k_alt/ and training/csv_files_20k/
- training/csv_files_10k_alt/ and training/csv_files_10k/
- training/csv_files_large_alt/ and training/csv_files_large/
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

import pandas as pd


DATASETS = [
    (
        "20k alternate-sequence dataset",
        Path("training/csv_files_20k_alt"),
        {
            "train": "train_with_alt_sequences.csv",
            "val": "val_with_alt_sequences.csv",
            "test": "test_with_alt_sequences.csv",
        },
    ),
    (
        "20k reference-sequence dataset",
        Path("training/csv_files_20k"),
        {
            "train": "train_with_sequences.csv",
            "val": "val_with_sequences.csv",
            "test": "test_with_sequences.csv",
        },
    ),
    (
        "10k alternate-sequence dataset",
        Path("training/csv_files_10k_alt"),
        {
            "train": "train_with_alt_sequences.csv",
            "val": "val_with_alt_sequences.csv",
            "test": "test_with_alt_sequences.csv",
        },
    ),
    (
        "10k reference-sequence dataset",
        Path("training/csv_files_10k"),
        {
            "train": "train_with_sequences.csv",
            "val": "val_with_sequences.csv",
            "test": "test_with_sequences.csv",
        },
    ),
    (
        "large alternate-sequence dataset",
        Path("training/csv_files_large_alt"),
        {
            "train": "train_with_alt_sequences.csv",
            "val": "val_with_alt_sequences.csv",
            "test": "test_with_alt_sequences.csv",
        },
    ),
    (
        "large reference-sequence dataset",
        Path("training/csv_files_large"),
        {
            "train": "train_with_sequences.csv",
            "val": "val_with_sequences.csv",
            "test": "test_with_sequences.csv",
        },
    ),
]

REQUIRED_COLUMNS = {"sequence", "label"}
EXAMPLE_COLUMNS = ["variant_id", "REF", "ALT", "label"]
SUSPICIOUS_CLNSIG_TERMS = (
    "conflicting",
    "uncertain",
    "not provided",
    "not_provided",
    "risk_factor",
    "risk factor",
    "association",
    "drug_response",
    "drug response",
    "protective",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_clnsig(value: object) -> str:
    decoded = unquote(str(value))
    return (
        decoded.replace("_", " ")
        .replace("-", " ")
        .replace("/", " ")
        .replace("|", " ")
        .replace(",", " ")
        .strip()
        .lower()
    )


def suspicious_clnsig_count(df: pd.DataFrame) -> int:
    if "CLNSIG" not in df.columns:
        return 0
    normalized = df["CLNSIG"].fillna("").apply(normalize_clnsig)
    return int(normalized.apply(lambda value: any(term in value for term in SUSPICIOUS_CLNSIG_TERMS)).sum())


def sequence_lengths(df: pd.DataFrame) -> pd.Series:
    return df["sequence"].fillna("").astype(str).str.strip().str.len()


def missing_sequence_count(df: pd.DataFrame) -> int:
    cleaned = df["sequence"].fillna("").astype(str).str.strip()
    return int((cleaned == "").sum())


def print_sequence_length_summary(lengths: pd.Series) -> None:
    non_empty = lengths[lengths > 0]
    if non_empty.empty:
        print("Sequence length min/mean/max: no non-empty sequences")
        return

    print(
        "Sequence length min/mean/max: "
        f"{int(non_empty.min())} / "
        f"{float(non_empty.mean()):.2f} / "
        f"{int(non_empty.max())}"
    )


def print_examples(df: pd.DataFrame, lengths: pd.Series) -> None:
    print("First 3 examples:")
    if df.empty:
        print("  no rows")
        return

    columns = [column for column in EXAMPLE_COLUMNS if column in df.columns]
    examples = df[columns].head(3).copy()
    examples["sequence_length"] = lengths.head(3).to_list()
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.max_colwidth", 80):
        print(examples.to_string(index=False))


def check_split(split_name: str, csv_path: Path) -> tuple[pd.DataFrame | None, bool]:
    print("-" * 80)
    print(f"{split_name.upper()} SPLIT")
    print("-" * 80)
    print(f"File path: {csv_path}")

    if not csv_path.exists():
        print("ERROR: file is missing.")
        print()
        return None, False

    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        print(f"ERROR: missing required columns: {missing_columns}")
        print()
        return df, False

    labels = pd.to_numeric(df["label"], errors="coerce")
    print("Label distribution:")
    print(labels.value_counts(dropna=False).sort_index().to_string())

    missing_sequences = missing_sequence_count(df)
    lengths = sequence_lengths(df)
    print(f"Missing sequence count: {missing_sequences:,}")
    print_sequence_length_summary(lengths)

    if "CLNSIG" in df.columns:
        suspicious_count = suspicious_clnsig_count(df)
        print(f"CLNSIG suspicious rows: {suspicious_count:,}")
    else:
        print("CLNSIG suspicious rows: CLNSIG column not present")

    print_examples(df, lengths)
    print()

    valid_labels = labels.isin([0, 1]).all()
    has_sequences = missing_sequences == 0 and int((lengths > 0).sum()) == len(df)
    usable = len(df) > 0 and valid_labels and has_sequences
    return df, usable


def check_dataset(dataset_name: str, dataset_dir: Path, split_files: dict[str, str]) -> tuple[int, bool]:
    print("=" * 80)
    print(dataset_name.upper())
    print("=" * 80)
    print(f"Dataset directory: {dataset_dir}")

    if not dataset_dir.exists():
        print("Dataset directory is missing, skipping.")
        print()
        return 0, True

    total_rows = 0
    dataset_usable = True
    frames = []

    for split_name, filename in split_files.items():
        df, split_usable = check_split(split_name, dataset_dir / filename)
        dataset_usable = dataset_usable and split_usable
        if df is not None:
            total_rows += len(df)
            if "label" in df.columns:
                temp = df[["label"]].copy()
                temp["split"] = split_name
                frames.append(temp)

    print(f"Total rows across train/val/test: {total_rows:,}")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined["label"] = pd.to_numeric(combined["label"], errors="coerce")
        print("Combined label distribution:")
        print(combined["label"].value_counts(dropna=False).sort_index().to_string())

    if dataset_usable:
        print("Usability check: OK, this dataset has rows, labels, and non-empty sequences.")
    else:
        print("Usability check: FAILED, fix the issues above before training.")
    print()

    return total_rows, dataset_usable


def main() -> None:
    root = project_root()
    print("Large ClinVar dataset check")
    print(f"Project root: {root}")
    print()

    overall_usable = True
    datasets_found = 0
    for dataset_name, relative_dir, split_files in DATASETS:
        dataset_dir = root / relative_dir
        total_rows, usable = check_dataset(dataset_name, dataset_dir, split_files)
        if dataset_dir.exists():
            datasets_found += 1
            overall_usable = overall_usable and usable and total_rows > 0

    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)
    if datasets_found == 0:
        print("No larger dataset folders were found yet. Run prepare_larger_clinvar_dataset.py first.")
    elif overall_usable:
        print("The larger dataset files found look usable for training.")
    else:
        print("At least one larger dataset found is not fully usable. Review the errors above.")


if __name__ == "__main__":
    main()
