#!/usr/bin/env python3
"""Audit whether sequence strings contain REF or ALT alleles near the center.

The prepared sequence files were fetched as roughly +/-512 bp around each
variant, so the variant should begin around index 512 in the sequence string.
This script checks that assumption for SNVs and small indels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ORIGINAL_SPLIT_FILES = {
    "train": "train_with_sequences.csv",
    "val": "val_with_sequences.csv",
    "test": "test_with_sequences.csv",
}

ALT_SPLIT_FILES = {
    "train": "train_with_alt_sequences.csv",
    "val": "val_with_alt_sequences.csv",
    "test": "test_with_alt_sequences.csv",
}

REQUIRED_COLUMNS = {"sequence", "REF", "ALT"}
OPTIONAL_EXAMPLE_COLUMNS = [
    "variant_id",
    "CHROM",
    "POS",
    "CLNHGVS",
    "gene_symbol",
    "GENEINFO",
]

CENTER_INDEX = 512
SNIPPET_FLANK = 20
MAX_INDEL_SIZE = 50
VALID_BASES = set("ACGTN")


@dataclass
class AuditCounts:
    total_rows_seen: int = 0
    total_rows_checked: int = 0
    snv_rows_checked: int = 0
    indel_rows_checked: int = 0
    reference_matches: int = 0
    alternate_matches: int = 0
    mismatches: int = 0
    skipped_missing_values: int = 0
    skipped_short_sequence: int = 0
    skipped_multiple_alt: int = 0
    skipped_symbolic_alt: int = 0
    skipped_non_acgtn_allele: int = 0
    skipped_not_snv_or_small_indel: int = 0

    def add(self, other: "AuditCounts") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


@dataclass
class DatasetChoice:
    data_dir: Path
    split_files: dict[str, str]
    is_alternate_dataset: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing train/val/test CSV files. Defaults to "
            "training/csv_files_20k_alt, training/csv_files_10k_alt, "
            "training/csv_files_large_alt, training/csv_files_alt, "
            "training/csv_files_20k, training/csv_files_10k, "
            "training/csv_files_large, training/csv_files, then data/processed."
        ),
    )
    parser.add_argument(
        "--center-index",
        type=int,
        default=CENTER_INDEX,
        help="0-based index where the variant is expected to start. Default: 512.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def has_all_files(directory: Path, split_files: dict[str, str]) -> bool:
    return all((directory / filename).exists() for filename in split_files.values())


def choose_dataset(root: Path, requested_dir: Path | None) -> DatasetChoice:
    if requested_dir is not None:
        data_dir = requested_dir.expanduser().resolve()
        if has_all_files(data_dir, ALT_SPLIT_FILES):
            return DatasetChoice(data_dir, ALT_SPLIT_FILES, True)
        if has_all_files(data_dir, ORIGINAL_SPLIT_FILES):
            return DatasetChoice(data_dir, ORIGINAL_SPLIT_FILES, False)
        raise FileNotFoundError(
            "The requested directory does not contain a complete alternate or original dataset:\n"
            f"{data_dir}"
        )

    candidates: list[tuple[Path, dict[str, str], bool]] = [
        (root / "training" / "csv_files_20k_alt", ALT_SPLIT_FILES, True),
        (root / "training" / "csv_files_10k_alt", ALT_SPLIT_FILES, True),
        (root / "training" / "csv_files_large_alt", ALT_SPLIT_FILES, True),
        (root / "training" / "csv_files_alt", ALT_SPLIT_FILES, True),
        (root / "training" / "csv_files_20k", ORIGINAL_SPLIT_FILES, False),
        (root / "training" / "csv_files_10k", ORIGINAL_SPLIT_FILES, False),
        (root / "training" / "csv_files_large", ORIGINAL_SPLIT_FILES, False),
        (root / "training" / "csv_files", ORIGINAL_SPLIT_FILES, False),
        (root / "data" / "processed", ORIGINAL_SPLIT_FILES, False),
    ]

    for directory, split_files, is_alternate_dataset in candidates:
        if has_all_files(directory, split_files):
            return DatasetChoice(directory, split_files, is_alternate_dataset)

    searched = "\n".join(str(path) for path, _split_files, _is_alt in candidates)
    raise FileNotFoundError(
        "Could not find a complete alternate or original sequence dataset.\n"
        f"Searched:\n{searched}"
    )


def normalize_sequence(value: object) -> str:
    return str(value).strip().upper()


def normalize_allele(value: object) -> str:
    return str(value).strip().upper()


def is_symbolic_alt(alt: str) -> bool:
    return alt.startswith("<") or alt.endswith(">") or "[" in alt or "]" in alt


def is_acgtn(value: str) -> bool:
    return bool(value) and set(value).issubset(VALID_BASES)


def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1


def is_small_indel(ref: str, alt: str) -> bool:
    return ref != alt and abs(len(ref) - len(alt)) <= MAX_INDEL_SIZE


def center_snippet(sequence: str, center_index: int) -> str:
    start = max(0, center_index - SNIPPET_FLANK)
    end = min(len(sequence), center_index + SNIPPET_FLANK + 1)
    snippet = sequence[start:end]
    marker_position = center_index - start
    if 0 <= marker_position < len(snippet):
        return snippet[:marker_position] + "[" + snippet[marker_position] + "]" + snippet[marker_position + 1 :]
    return snippet


def classify_row(row: pd.Series, center_index: int, is_alternate_dataset: bool) -> tuple[str, str, str]:
    sequence = normalize_sequence(row["sequence"])
    ref = normalize_allele(row["REF"])
    alt = normalize_allele(row["ALT"])

    if not sequence or not ref or not alt or sequence == "NAN" or ref == "NAN" or alt == "NAN":
        return "skipped_missing_values", "", ""

    if "," in alt:
        return "skipped_multiple_alt", "", ""

    if is_symbolic_alt(alt):
        return "skipped_symbolic_alt", "", ""

    if not is_acgtn(ref) or not is_acgtn(alt):
        return "skipped_non_acgtn_allele", "", ""

    if len(sequence) <= center_index:
        return "skipped_short_sequence", "", ""

    if is_snv(ref, alt):
        observed = sequence[center_index]
        if is_alternate_dataset:
            if observed == alt:
                return "alternate_match_snv", observed, center_snippet(sequence, center_index)
            if observed == ref:
                return "reference_match_snv", observed, center_snippet(sequence, center_index)
        else:
            if observed == ref:
                return "reference_match_snv", observed, center_snippet(sequence, center_index)
            if observed == alt:
                return "alternate_match_snv", observed, center_snippet(sequence, center_index)
        return "mismatch_snv", observed, center_snippet(sequence, center_index)

    if is_small_indel(ref, alt):
        expected_allele = alt if is_alternate_dataset else ref
        if len(sequence) < center_index + len(expected_allele):
            return "skipped_short_sequence", "", ""

        ref_window = sequence[center_index : center_index + len(ref)] if len(sequence) >= center_index + len(ref) else ""
        alt_window = sequence[center_index : center_index + len(alt)] if len(sequence) >= center_index + len(alt) else ""

        if is_alternate_dataset:
            if alt_window == alt:
                return "alternate_match_indel", alt_window, center_snippet(sequence, center_index)
            if ref_window == ref:
                return "reference_match_indel", ref_window, center_snippet(sequence, center_index)
            return "mismatch_indel", alt_window, center_snippet(sequence, center_index)

        if ref_window == ref:
            return "reference_match_indel", ref_window, center_snippet(sequence, center_index)
        if alt_window == alt:
            return "alternate_match_indel", alt_window, center_snippet(sequence, center_index)
        return "mismatch_indel", ref_window, center_snippet(sequence, center_index)

    return "skipped_not_snv_or_small_indel", "", ""


def update_counts(counts: AuditCounts, status: str) -> None:
    if status.startswith("skipped_"):
        current_value = getattr(counts, status)
        setattr(counts, status, current_value + 1)
        return

    counts.total_rows_checked += 1

    if status.endswith("_snv"):
        counts.snv_rows_checked += 1
    elif status.endswith("_indel"):
        counts.indel_rows_checked += 1

    if status.startswith("reference_match"):
        counts.reference_matches += 1
    elif status.startswith("alternate_match"):
        counts.alternate_matches += 1
    elif status.startswith("mismatch"):
        counts.mismatches += 1


def build_example(row: pd.Series, split_name: str, status: str, observed: str, snippet: str) -> dict[str, object]:
    example = {
        "split": split_name,
        "status": status,
        "REF": normalize_allele(row["REF"]),
        "ALT": normalize_allele(row["ALT"]),
        "observed_at_center": observed,
        "center_sequence_snippet": snippet,
    }

    for column in OPTIONAL_EXAMPLE_COLUMNS:
        if column in row.index:
            example[column] = row[column]

    return example


def audit_split(
    split_name: str,
    csv_path: Path,
    center_index: int,
    is_alternate_dataset: bool,
) -> tuple[AuditCounts, list[dict[str, object]]]:
    print("=" * 80)
    print(f"{split_name.upper()} SPLIT")
    print("=" * 80)
    print(f"Path: {csv_path}")

    df = pd.read_csv(csv_path)
    counts = AuditCounts(total_rows_seen=len(df))
    examples: list[dict[str, object]] = []

    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        print(f"ERROR: missing required columns: {missing_columns}")
        print()
        return counts, examples

    for _, row in df.iterrows():
        status, observed, snippet = classify_row(row, center_index, is_alternate_dataset)
        update_counts(counts, status)

        if len(examples) < 10 and not status.startswith("skipped_"):
            examples.append(build_example(row, split_name, status, observed, snippet))

    print_counts(counts)
    print()
    return counts, examples


def print_counts(counts: AuditCounts) -> None:
    print(f"Rows seen: {counts.total_rows_seen:,}")
    print(f"Total rows checked: {counts.total_rows_checked:,}")
    print(f"SNV rows checked: {counts.snv_rows_checked:,}")
    print(f"Indel rows checked: {counts.indel_rows_checked:,}")
    print(f"Reference allele matches: {counts.reference_matches:,}")
    print(f"Alternate allele matches: {counts.alternate_matches:,}")
    print(f"Mismatches: {counts.mismatches:,}")
    print("Skipped rows:")
    print(f"  missing sequence/REF/ALT: {counts.skipped_missing_values:,}")
    print(f"  sequence too short for center check: {counts.skipped_short_sequence:,}")
    print(f"  multiple ALT alleles: {counts.skipped_multiple_alt:,}")
    print(f"  symbolic ALT allele: {counts.skipped_symbolic_alt:,}")
    print(f"  non-ACGTN allele: {counts.skipped_non_acgtn_allele:,}")
    print(f"  not SNV or small indel: {counts.skipped_not_snv_or_small_indel:,}")


def print_examples(examples: list[dict[str, object]]) -> None:
    print("=" * 80)
    print("FIRST 10 CHECKED EXAMPLES")
    print("=" * 80)

    if not examples:
        print("No checked examples available.")
        print()
        return

    examples_df = pd.DataFrame(examples)
    display_columns = [
        column
        for column in [
            "split",
            "variant_id",
            "CHROM",
            "POS",
            "gene_symbol",
            "REF",
            "ALT",
            "status",
            "observed_at_center",
            "center_sequence_snippet",
            "CLNHGVS",
            "GENEINFO",
        ]
        if column in examples_df.columns
    ]

    with pd.option_context("display.max_columns", None, "display.width", 220, "display.max_colwidth", 90):
        print(examples_df[display_columns].to_string(index=False))
    print()


def print_conclusion(counts: AuditCounts) -> None:
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print_counts(counts)

    checked = counts.total_rows_checked
    if checked == 0:
        conclusion = "Could not determine clearly."
    else:
        ref_fraction = counts.reference_matches / checked
        alt_fraction = counts.alternate_matches / checked
        print(f"Reference match fraction: {ref_fraction:.2%}")
        print(f"Alternate match fraction: {alt_fraction:.2%}")

        if ref_fraction >= 0.80 and counts.reference_matches > counts.alternate_matches:
            conclusion = "The sequence appears to be reference sequence."
        elif alt_fraction >= 0.80 and counts.alternate_matches > counts.reference_matches:
            conclusion = "The sequence appears to contain alternate alleles."
        else:
            conclusion = "Could not determine clearly."

    print()
    print(f"Conclusion: {conclusion}")


def main() -> None:
    args = parse_args()
    root = project_root()
    dataset = choose_dataset(root, args.data_dir)

    print("Variant sequence encoding audit")
    print(f"Selected data directory: {dataset.data_dir}")
    print(f"Auditing alternate dataset: {dataset.is_alternate_dataset}")
    print(f"Expected variant start index: {args.center_index}")
    print()

    total_counts = AuditCounts()
    all_examples: list[dict[str, object]] = []

    for split_name, filename in dataset.split_files.items():
        csv_path = dataset.data_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing required CSV file: {csv_path}")

        split_counts, split_examples = audit_split(
            split_name,
            csv_path,
            args.center_index,
            dataset.is_alternate_dataset,
        )
        total_counts.add(split_counts)
        remaining_example_slots = 10 - len(all_examples)
        if remaining_example_slots > 0:
            all_examples.extend(split_examples[:remaining_example_slots])

    print_examples(all_examples)
    print_conclusion(total_counts)


if __name__ == "__main__":
    main()
