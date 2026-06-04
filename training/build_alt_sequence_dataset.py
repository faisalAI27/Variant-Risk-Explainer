#!/usr/bin/env python3
"""Build alternate-allele sequence CSVs from reference sequence CSVs.

The current prepared CSVs contain reference genome sequence around each
variant. This script verifies that REF is present at index 512, replaces that
REF allele with ALT, and writes new CSV files with mutated/alternate sequences.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


FLANK_SIZE = 512
VALID_BASES = set("ACGTN")

SPLIT_FILES = {
    "train": ("train_with_sequences.csv", "train_with_alt_sequences.csv"),
    "val": ("val_with_sequences.csv", "val_with_alt_sequences.csv"),
    "test": ("test_with_sequences.csv", "test_with_alt_sequences.csv"),
}

REQUIRED_COLUMNS = {"sequence", "REF", "ALT", "label"}
USEFUL_METADATA_COLUMNS = [
    "variant_id",
    "CHROM",
    "POS",
    "ID",
    "variant_type",
    "gene_symbol",
    "GENEINFO",
    "CLNSIG",
    "CLNHGVS",
    "CLNVC",
    "label_name",
]


@dataclass
class BuildStats:
    input_rows: int = 0
    successful_rows: int = 0
    failed_rows: int = 0
    snv_count: int = 0
    indel_count: int = 0
    skipped_missing_values: int = 0
    skipped_multiple_alt: int = 0
    skipped_symbolic_alt: int = 0
    skipped_non_acgtn: int = 0
    skipped_short_sequence: int = 0
    failed_ref_mismatch: int = 0

    def add(self, other: "BuildStats") -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing the input CSV files. Defaults to data/processed, then training/csv_files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for alternate-sequence CSV files. Default: training/csv_files_alt.",
    )
    return parser.parse_args()


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(root: Path, path: Path) -> Path:
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def choose_input_dir(root: Path, requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        return resolve_path(root, requested_dir)

    candidates = [
        root / "data" / "processed",
        root / "training" / "csv_files",
    ]

    for directory in candidates:
        if all((directory / input_name).exists() for input_name, _output_name in SPLIT_FILES.values()):
            return directory

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find all input sequence CSV files.\n"
        f"Searched:\n{searched}"
    )


def choose_output_dir(root: Path, requested_dir: Path | None) -> Path:
    if requested_dir is not None:
        return resolve_path(root, requested_dir)
    return root / "training" / "csv_files_alt"


def normalize_text(value: object) -> str:
    return str(value).strip().upper()


def is_missing_text(value: str) -> bool:
    return value == "" or value == "NAN" or value == "NONE"


def is_symbolic_alt(alt: str) -> bool:
    return alt.startswith("<") or alt.endswith(">") or "[" in alt or "]" in alt


def contains_only_valid_bases(value: str) -> bool:
    return bool(value) and set(value).issubset(VALID_BASES)


def is_snv(ref: str, alt: str) -> bool:
    return len(ref) == 1 and len(alt) == 1


def is_small_indel_or_complex(ref: str, alt: str) -> bool:
    return ref != alt and abs(len(ref) - len(alt)) <= 50


def build_alt_sequence(row: pd.Series) -> tuple[dict[str, object] | None, str]:
    ref_sequence = normalize_text(row["sequence"])
    ref = normalize_text(row["REF"])
    alt = normalize_text(row["ALT"])

    if is_missing_text(ref_sequence) or is_missing_text(ref) or is_missing_text(alt):
        return None, "skipped_missing_values"

    if "," in alt:
        return None, "skipped_multiple_alt"

    if is_symbolic_alt(alt):
        return None, "skipped_symbolic_alt"

    if not (
        contains_only_valid_bases(ref_sequence)
        and contains_only_valid_bases(ref)
        and contains_only_valid_bases(alt)
    ):
        return None, "skipped_non_acgtn"

    if len(ref_sequence) < FLANK_SIZE + len(ref):
        return None, "skipped_short_sequence"

    upstream = ref_sequence[:FLANK_SIZE]
    observed_ref = ref_sequence[FLANK_SIZE : FLANK_SIZE + len(ref)]
    downstream = ref_sequence[FLANK_SIZE + len(ref) :]

    if observed_ref != ref:
        return None, "failed_ref_mismatch"

    alt_sequence = upstream + alt + downstream
    output_row = row.to_dict()
    output_row["REF"] = ref
    output_row["ALT"] = alt
    output_row["ref_sequence"] = ref_sequence
    output_row["alt_sequence"] = alt_sequence
    output_row["sequence"] = alt_sequence
    output_row["ref_center"] = observed_ref
    output_row["alt_center"] = alt_sequence[FLANK_SIZE : FLANK_SIZE + len(alt)]
    return output_row, "success"


def ordered_columns(df: pd.DataFrame) -> list[str]:
    important_columns = [
        column
        for column in USEFUL_METADATA_COLUMNS + ["REF", "ALT", "label", "ref_sequence", "alt_sequence", "sequence"]
        if column in df.columns
    ]
    remaining_columns = [column for column in df.columns if column not in important_columns]
    return important_columns + remaining_columns


def update_failed_stat(stats: BuildStats, reason: str) -> None:
    stats.failed_rows += 1
    current_value = getattr(stats, reason)
    setattr(stats, reason, current_value + 1)


def process_split(split_name: str, input_path: Path, output_path: Path) -> tuple[BuildStats, pd.DataFrame]:
    print("=" * 80)
    print(f"{split_name.upper()} SPLIT")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")

    df = pd.read_csv(input_path)
    missing_columns = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing_columns:
        raise ValueError(f"{input_path} is missing required columns: {missing_columns}")

    stats = BuildStats(input_rows=len(df))
    output_rows: list[dict[str, object]] = []

    for _, row in df.iterrows():
        output_row, status = build_alt_sequence(row)
        if output_row is None:
            update_failed_stat(stats, status)
            continue

        output_rows.append(output_row)
        stats.successful_rows += 1

        ref = output_row["REF"]
        alt = output_row["ALT"]
        if is_snv(ref, alt):
            stats.snv_count += 1
        elif is_small_indel_or_complex(ref, alt):
            stats.indel_count += 1

    output_df = pd.DataFrame(output_rows)
    if not output_df.empty:
        output_df = output_df[ordered_columns(output_df)]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print_split_summary(stats, output_df)
    print_examples(output_df)
    return stats, output_df


def print_split_summary(stats: BuildStats, output_df: pd.DataFrame) -> None:
    print(f"Input rows: {stats.input_rows:,}")
    print(f"Successful rows: {stats.successful_rows:,}")
    print(f"Failed rows: {stats.failed_rows:,}")
    print(f"SNV count: {stats.snv_count:,}")
    print(f"Indel count: {stats.indel_count:,}")

    if not output_df.empty and "label" in output_df.columns:
        print("Label distribution:")
        print(output_df["label"].value_counts().sort_index().to_string())
    else:
        print("Label distribution: no successful rows")

    if stats.failed_rows:
        print("Failure reasons:")
        print(f"  missing sequence/REF/ALT: {stats.skipped_missing_values:,}")
        print(f"  multiple ALT alleles: {stats.skipped_multiple_alt:,}")
        print(f"  symbolic ALT allele: {stats.skipped_symbolic_alt:,}")
        print(f"  non-ACGTN sequence/REF/ALT: {stats.skipped_non_acgtn:,}")
        print(f"  sequence too short: {stats.skipped_short_sequence:,}")
        print(f"  REF mismatch at center: {stats.failed_ref_mismatch:,}")
    print()


def print_examples(output_df: pd.DataFrame) -> None:
    print("First 5 examples:")
    if output_df.empty:
        print("No successful rows.")
        print()
        return

    columns = [column for column in ["variant_id", "REF", "ALT", "label", "ref_center", "alt_center"] if column in output_df.columns]
    examples = output_df[columns].head(5)
    with pd.option_context("display.max_columns", None, "display.width", 160, "display.max_colwidth", 80):
        print(examples.to_string(index=False))
    print()


def print_overall_summary(total_stats: BuildStats, all_outputs: list[pd.DataFrame]) -> None:
    print("=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)

    combined = pd.concat(all_outputs, ignore_index=True) if all_outputs else pd.DataFrame()
    print_split_summary(total_stats, combined)


def main() -> None:
    args = parse_args()
    root = project_root()
    input_dir = choose_input_dir(root, args.data_dir)
    output_dir = choose_output_dir(root, args.output_dir)

    print("Build alternate-allele sequence dataset")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Assumed flank size / variant center index: {FLANK_SIZE}")
    print()

    total_stats = BuildStats()
    all_outputs: list[pd.DataFrame] = []

    for split_name, (input_name, output_name) in SPLIT_FILES.items():
        input_path = input_dir / input_name
        output_path = output_dir / output_name
        if not input_path.exists():
            raise FileNotFoundError(f"Missing input CSV file: {input_path}")

        split_stats, output_df = process_split(split_name, input_path, output_path)
        total_stats.add(split_stats)
        all_outputs.append(output_df)

    print_overall_summary(total_stats, all_outputs)
    print(f"Alternate-sequence CSV files saved to: {output_dir}")


if __name__ == "__main__":
    main()
