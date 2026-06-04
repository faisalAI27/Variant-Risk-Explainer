#!/usr/bin/env python3
"""Check sequence CSV files before local model training.

Expected files:
- train_with_sequences.csv
- val_with_sequences.csv
- test_with_sequences.csv

The script prefers data/processed/ and falls back to training/csv_files/ so it
works with both the planned project layout and the current sample CSV location.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.parse import unquote

import pandas as pd


SPLIT_FILES = {
    "train": "train_with_sequences.csv",
    "val": "val_with_sequences.csv",
    "test": "test_with_sequences.csv",
}

REQUIRED_COLUMNS = {"sequence", "label"}
VALID_LABELS = {0, 1}
LABEL_MEANINGS = {
    0: "Benign/Likely benign",
    1: "Pathogenic/Likely pathogenic",
}

DROP_CLNSIG_TERMS = (
    "conflicting",
    "uncertain",
    "risk",
    "association",
    "drug",
    "protective",
    "not provided",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing train/val/test CSV files. Defaults to data/processed, then training/csv_files.",
    )
    parser.add_argument(
        "--min-reasonable-length",
        type=int,
        default=200,
        help="Warn when non-empty sequences are shorter than this length.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def choose_data_dir(root: Path, requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        return requested_dir.expanduser().resolve()

    preferred = root / "data" / "processed"
    fallback = root / "training" / "csv_files"

    if all((preferred / filename).exists() for filename in SPLIT_FILES.values()):
        return preferred
    if all((fallback / filename).exists() for filename in SPLIT_FILES.values()):
        return fallback
    return preferred


def find_missing_files(data_dir: Path) -> list[Path]:
    return [data_dir / filename for filename in SPLIT_FILES.values() if not (data_dir / filename).exists()]


def sequence_lengths(sequence_series: pd.Series) -> pd.Series:
    cleaned = sequence_series.fillna("").astype(str).str.strip()
    return cleaned.str.len()


def missing_sequence_count(sequence_series: pd.Series) -> int:
    cleaned = sequence_series.fillna("").astype(str).str.strip()
    return int((cleaned == "").sum())


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


def expected_label_from_clnsig(value: object) -> int | None:
    normalized = normalize_clnsig(value)
    if not normalized or normalized == "." or any(term in normalized for term in DROP_CLNSIG_TERMS):
        return None

    has_pathogenic = "pathogenic" in normalized
    has_benign = "benign" in normalized
    if has_pathogenic and has_benign:
        return None
    if has_pathogenic:
        return 1
    if has_benign:
        return 0
    return None


def print_label_meaning() -> None:
    print("Label encoding:")
    for label, meaning in LABEL_MEANINGS.items():
        print(f"  {label} = {meaning}")
    print()


def audit_split(split_name: str, csv_path: Path, min_reasonable_length: int) -> pd.DataFrame:
    print("=" * 80)
    print(f"{split_name.upper()} SPLIT")
    print("=" * 80)
    print(f"Path: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        print(f"ERROR: missing required columns: {missing_columns}")
        print()
        return df

    labels = pd.to_numeric(df["label"], errors="coerce")
    label_distribution = labels.value_counts(dropna=False).sort_index()
    print("Label distribution:")
    print(label_distribution.to_string())

    invalid_labels = sorted(set(labels.dropna().astype(int).unique()) - VALID_LABELS)
    if invalid_labels:
        print(f"WARNING: unexpected label values found: {invalid_labels}")
    else:
        print("Label check: OK, labels are encoded as 0/1.")

    if "label_name" in df.columns:
        label_name = df["label_name"].fillna("").astype(str).str.lower()
        label_name_mismatch = int(
            (((labels == 0) & ~label_name.str.contains("benign")) | ((labels == 1) & ~label_name.str.contains("pathogenic"))).sum()
        )
        if label_name_mismatch:
            print(f"WARNING: {label_name_mismatch:,} rows have label_name values that do not match label.")
        else:
            print("Label name check: OK.")

    if "CLNSIG" in df.columns:
        expected_labels = df["CLNSIG"].apply(expected_label_from_clnsig)
        unsupported_clnsig = int(expected_labels.isna().sum())
        comparable = expected_labels.notna() & labels.notna()
        clnsig_mismatch = int((expected_labels[comparable].astype(int) != labels[comparable].astype(int)).sum())
        print(f"CLNSIG rows that should be excluded before training: {unsupported_clnsig:,}")
        if clnsig_mismatch:
            print(f"WARNING: {clnsig_mismatch:,} rows have CLNSIG values that disagree with label.")
        if unsupported_clnsig:
            examples = df.loc[expected_labels.isna(), "CLNSIG"].dropna().astype(str).unique()[:5]
            print(f"Excluded CLNSIG examples: {list(examples)}")

    missing_sequences = missing_sequence_count(df["sequence"])
    lengths = sequence_lengths(df["sequence"])
    non_empty_lengths = lengths[lengths > 0]

    print(f"Missing or empty sequence count: {missing_sequences:,}")
    if non_empty_lengths.empty:
        print("Sequence length min/mean/max: no non-empty sequences")
    else:
        print(
            "Sequence length min/mean/max: "
            f"{int(non_empty_lengths.min())} / "
            f"{float(non_empty_lengths.mean()):.2f} / "
            f"{int(non_empty_lengths.max())}"
        )
        short_count = int((non_empty_lengths < min_reasonable_length).sum())
        if short_count:
            print(f"WARNING: {short_count:,} sequences are shorter than {min_reasonable_length} bp.")

    print("First 3 example rows:")
    example_columns = [column for column in ["variant_id", "CHROM", "POS", "REF", "ALT", "label", "label_name"] if column in df.columns]
    examples = df[example_columns].head(3).copy()
    examples["sequence_length"] = lengths.head(3).to_list()
    examples["sequence_preview"] = df["sequence"].fillna("").astype(str).str.slice(0, 80).head(3).to_list()
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.max_colwidth", 80):
        print(examples.to_string(index=False))
    print()
    return df


def print_overall_balance(frames: dict[str, pd.DataFrame]) -> None:
    print("=" * 80)
    print("OVERALL CLASS BALANCE")
    print("=" * 80)

    usable_frames = []
    for split_name, df in frames.items():
        if "label" not in df.columns:
            continue
        temp = df[["label"]].copy()
        temp["split"] = split_name
        usable_frames.append(temp)

    if not usable_frames:
        print("No label columns found, so balance cannot be checked.")
        return

    combined = pd.concat(usable_frames, ignore_index=True)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce")
    counts = combined["label"].value_counts().sort_index()
    print(counts.to_string())

    valid_counts = counts[counts.index.isin(list(VALID_LABELS))]
    if len(valid_counts) == 2 and valid_counts.sum() > 0:
        minority_fraction = float(valid_counts.min() / valid_counts.sum())
        print(f"Minority class fraction: {minority_fraction:.2%}")
        if minority_fraction < 0.10:
            print("Balance note: very imbalanced. Consider class weights, sampling, or more data.")
        elif minority_fraction < 0.25:
            print("Balance note: moderately imbalanced, but usable for a demo with stratified metrics.")
        else:
            print("Balance note: reasonably balanced for a research demo.")
    else:
        print("Balance note: expected both labels 0 and 1.")


def main() -> None:
    args = parse_args()
    root = project_root()
    data_dir = choose_data_dir(root, args.data_dir)

    print("Variant Risk Explainer dataset check")
    print(f"Project root: {root}")
    print(f"Data directory: {data_dir}")
    print()
    print_label_meaning()

    missing_files = find_missing_files(data_dir)
    if missing_files:
        print("ERROR: missing expected CSV files:")
        for path in missing_files:
            print(f"  {path}")
        raise SystemExit(1)

    frames = {}
    for split_name, filename in SPLIT_FILES.items():
        frames[split_name] = audit_split(split_name, data_dir / filename, args.min_reasonable_length)

    print_overall_balance(frames)


if __name__ == "__main__":
    main()
