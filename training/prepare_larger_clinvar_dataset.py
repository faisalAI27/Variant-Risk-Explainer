#!/usr/bin/env python3
"""Prepare a larger balanced ClinVar sequence dataset for DNABERT-2.

The script prefers existing parsed/filtered CSV data when enough rows are
available. If not, it downloads/parses the ClinVar GRCh38 VCF, samples a more
balanced binary dataset, fetches GRCh38 reference sequence windows with resume
support, and writes both reference-sequence and alternate-sequence CSV files.
"""

from __future__ import annotations

import argparse
import gzip
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.utils.clinvar_parser import (
    add_variant_id,
    classify_variant_type,
    extract_gene_symbol,
    has_multiple_alt,
    is_sequence_allele,
    is_symbolic_alt,
    parse_info_field,
)
from training.utils.label_utils import assign_binary_label, label_name
from training.utils.sequence_fetcher import build_sequence_fetcher, clean_sequence


CLINVAR_GRCH38_VCF_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
FLANK_SIZE_DEFAULT = 512
MIN_SEQUENCE_LENGTH = 200
PROGRESS_EVERY = 100

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

DROP_OUTPUT_COLUMNS = {
    "_source_split",
    "_source_kind",
    "ref_sequence",
    "alt_sequence",
    "ref_center",
    "alt_center",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target_total", type=int, default=5000)
    parser.add_argument("--max_pathogenic", type=int, default=None)
    parser.add_argument("--max_benign", type=int, default=None)
    parser.add_argument("--flank_size", type=int, default=FLANK_SIZE_DEFAULT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Reference-sequence output directory. If omitted, target_total 5000 writes "
            "training/csv_files_large, 10000 writes training/csv_files_10k, and 20000 "
            "writes training/csv_files_20k."
        ),
    )
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--clinvar_vcf", type=Path, default=None, help="Optional local ClinVar GRCh38 VCF.GZ path.")
    parser.add_argument("--fetch_mode", choices=["ucsc", "fasta"], default="ucsc")
    parser.add_argument("--fasta_path", type=Path, default=None, help="Optional local GRCh38 FASTA for --fetch_mode fasta.")
    parser.add_argument("--sequence_cache", type=Path, default=None, help="Optional JSON sequence cache path.")
    parser.add_argument("--sleep_seconds", type=float, default=0.15)
    parser.add_argument("--max_retries", type=int, default=3)
    return parser.parse_args()


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    return path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()


def alt_output_dir_for(output_dir: Path) -> Path:
    return output_dir.with_name(f"{output_dir.name}_alt")


def default_output_dir_for_target(target_total: int) -> Path:
    if target_total == 5000:
        return Path("training/csv_files_large")
    if target_total == 10000:
        return Path("training/csv_files_10k")
    if target_total == 20000:
        return Path("training/csv_files_20k")
    return Path(f"training/csv_files_{target_total}")


def has_complete_split(directory: Path, split_files: dict[str, str]) -> bool:
    return all((directory / filename).exists() for filename in split_files.values())


def read_split_directory(directory: Path, split_files: dict[str, str], source_kind: str) -> pd.DataFrame:
    frames = []
    for split_name, filename in split_files.items():
        path = directory / filename
        df = pd.read_csv(path)
        df["_source_split"] = split_name
        df["_source_kind"] = source_kind

        # Alternate CSVs keep the true reference sequence in ref_sequence.
        if source_kind == "alternate_split" and "ref_sequence" in df.columns:
            df["sequence"] = df["ref_sequence"]

        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def existing_source_candidates(root: Path) -> list[tuple[str, pd.DataFrame]]:
    candidates: list[tuple[str, pd.DataFrame]] = []

    for path in [
        root / "data" / "processed" / "clinvar_binary_variants.csv",
        root / "training" / "csv_files" / "clinvar_binary_variants.csv",
        root / "training" / "csv_files_20k" / "clinvar_binary_variants.csv",
        root / "training" / "csv_files_10k" / "clinvar_binary_variants.csv",
        root / "training" / "csv_files_large" / "clinvar_binary_variants.csv",
    ]:
        if path.exists():
            candidates.append((str(path), pd.read_csv(path)))

    for directory in [
        root / "training" / "csv_files_20k",
        root / "training" / "csv_files_10k",
        root / "training" / "csv_files_large",
        root / "training" / "csv_files",
        root / "data" / "processed",
    ]:
        if has_complete_split(directory, ORIGINAL_SPLIT_FILES):
            candidates.append((str(directory), read_split_directory(directory, ORIGINAL_SPLIT_FILES, "reference_split")))

    for directory in [
        root / "training" / "csv_files_20k_alt",
        root / "training" / "csv_files_10k_alt",
        root / "training" / "csv_files_large_alt",
        root / "training" / "csv_files_alt",
    ]:
        if has_complete_split(directory, ALT_SPLIT_FILES):
            candidates.append((str(directory), read_split_directory(directory, ALT_SPLIT_FILES, "alternate_split")))

    return candidates


def normalize_allele(value: object) -> str:
    return str(value).strip().upper()


def normalize_clnsig_display(value: object) -> str:
    return unquote(str(value)).strip()


def filter_clear_variants(df: pd.DataFrame) -> pd.DataFrame:
    required = {"CHROM", "POS", "REF", "ALT"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Variant table is missing required columns: {missing}")

    filtered = df.copy()
    filtered["REF"] = filtered["REF"].apply(normalize_allele)
    filtered["ALT"] = filtered["ALT"].apply(normalize_allele)
    filtered["POS"] = pd.to_numeric(filtered["POS"], errors="coerce")
    filtered = filtered.loc[filtered["POS"].notna()].copy()
    filtered["POS"] = filtered["POS"].astype(int)

    if "CLNSIG" in filtered.columns:
        filtered["CLNSIG"] = filtered["CLNSIG"].apply(normalize_clnsig_display)
        filtered["label"] = filtered["CLNSIG"].apply(assign_binary_label)
    elif "label" in filtered.columns:
        filtered["label"] = pd.to_numeric(filtered["label"], errors="coerce")
        filtered = filtered.loc[filtered["label"].isin([0, 1])].copy()
    else:
        raise ValueError("Variant table needs either CLNSIG or label.")

    filtered["variant_type"] = filtered.apply(lambda row: classify_variant_type(row["REF"], row["ALT"]), axis=1)

    keep_mask = (
        filtered["label"].notna()
        & ~filtered["ALT"].apply(has_multiple_alt)
        & ~filtered["ALT"].apply(is_symbolic_alt)
        & filtered["REF"].apply(is_sequence_allele)
        & filtered["ALT"].apply(is_sequence_allele)
        & filtered["variant_type"].notna()
    )
    filtered = filtered.loc[keep_mask].copy()
    filtered["label"] = filtered["label"].astype(int)

    if "GENEINFO" in filtered.columns and "gene_symbol" not in filtered.columns:
        filtered["gene_symbol"] = filtered["GENEINFO"].apply(extract_gene_symbol)
    if "label_name" not in filtered.columns:
        filtered["label_name"] = filtered["label"].apply(label_name)
    if "variant_id" not in filtered.columns:
        filtered["variant_id"] = filtered.apply(add_variant_id, axis=1)

    filtered = filtered.drop_duplicates(subset=["variant_id"]).reset_index(drop=True)
    return filtered


def download_clinvar_vcf(vcf_path: Path) -> Path:
    vcf_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = vcf_path.with_suffix(vcf_path.suffix + ".tmp")

    print(f"Downloading ClinVar GRCh38 VCF to: {vcf_path}")
    with requests.get(CLINVAR_GRCH38_VCF_URL, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with tmp_path.open("wb") as handle, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading VCF") as bar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                bar.update(len(chunk))

    shutil.move(str(tmp_path), str(vcf_path))
    return vcf_path


def find_or_download_vcf(root: Path, requested_vcf: Path | None) -> Path:
    if requested_vcf is not None:
        vcf_path = resolve_path(root, requested_vcf)
        if vcf_path is None or not vcf_path.exists():
            raise FileNotFoundError(f"ClinVar VCF not found: {vcf_path}")
        return vcf_path

    candidates = [
        root / "data" / "raw" / "clinvar.vcf.gz",
        root / "training" / "data" / "raw" / "clinvar.vcf.gz",
        root / "clinvar.vcf.gz",
    ]
    for path in candidates:
        if path.exists():
            return path

    return download_clinvar_vcf(root / "data" / "raw" / "clinvar.vcf.gz")


def parse_and_filter_vcf(vcf_path: Path) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, object]] = []
    total_records = 0

    print(f"Parsing ClinVar VCF: {vcf_path}")
    with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Parsing VCF records"):
            if line.startswith("#"):
                continue

            total_records += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos_raw, clinvar_id, ref, alt, _qual, _filter, info_raw = fields[:8]
            info = parse_info_field(info_raw)
            clnsig = info.get("CLNSIG")
            label = assign_binary_label(clnsig)
            ref = ref.upper()
            alt = alt.upper()
            variant_type = classify_variant_type(ref, alt)

            if (
                label is None
                or has_multiple_alt(alt)
                or is_symbolic_alt(alt)
                or not is_sequence_allele(ref)
                or not is_sequence_allele(alt)
                or variant_type is None
            ):
                continue

            gene_info = info.get("GENEINFO")
            record = {
                "CHROM": chrom,
                "POS": int(pos_raw),
                "ID": clinvar_id,
                "REF": ref,
                "ALT": alt,
                "variant_type": variant_type,
                "gene_symbol": extract_gene_symbol(gene_info),
                "GENEINFO": gene_info,
                "CLNSIG": clnsig,
                "CLNHGVS": info.get("CLNHGVS"),
                "CLNVC": info.get("CLNVC"),
                "label": int(label),
                "label_name": label_name(label),
            }
            record["variant_id"] = add_variant_id(pd.Series(record))
            records.append(record)

    filtered = pd.DataFrame.from_records(records)
    if not filtered.empty:
        filtered = filtered.drop_duplicates(subset=["variant_id"]).reset_index(drop=True)
    return filtered, total_records


def choose_variant_source(root: Path, target_total: int, requested_vcf: Path | None) -> tuple[pd.DataFrame, str, int, int]:
    best_source_name = ""
    best_source = pd.DataFrame()
    best_before = 0

    for source_name, source_df in existing_source_candidates(root):
        before = len(source_df)
        filtered = filter_clear_variants(source_df)
        print(f"Existing source candidate: {source_name}")
        print(f"  rows before filtering: {before:,}")
        print(f"  rows after filtering: {len(filtered):,}")

        if len(filtered) > len(best_source):
            best_source_name = source_name
            best_source = filtered
            best_before = before

        if len(filtered) >= target_total:
            return filtered, source_name, before, len(filtered)

    if not best_source.empty and len(best_source) >= target_total:
        return best_source, best_source_name, best_before, len(best_source)

    if not best_source.empty:
        print(
            "Existing data is available but smaller than target_total. "
            "Parsing ClinVar VCF to build a larger candidate set."
        )

    vcf_path = find_or_download_vcf(root, requested_vcf)
    parsed_df, total_records = parse_and_filter_vcf(vcf_path)
    return parsed_df, str(vcf_path), total_records, len(parsed_df)


def validate_positive_int(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive when provided.")


def sample_or_all(df: pd.DataFrame, n: int, random_state: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=random_state)


def balanced_sample(
    df: pd.DataFrame,
    target_total: int,
    max_pathogenic: int | None,
    max_benign: int | None,
    random_state: int,
) -> pd.DataFrame:
    if target_total <= 0:
        raise ValueError("--target_total must be positive.")
    validate_positive_int("--max_pathogenic", max_pathogenic)
    validate_positive_int("--max_benign", max_benign)

    pathogenic = df.loc[df["label"] == 1].copy()
    benign = df.loc[df["label"] == 0].copy()
    target_per_class = max(1, target_total // 2)

    pathogenic_limit = min(len(pathogenic), target_per_class)
    if max_pathogenic is not None:
        pathogenic_limit = min(pathogenic_limit, max_pathogenic)

    pathogenic_sample = sample_or_all(pathogenic, pathogenic_limit, random_state)

    benign_limit = min(len(benign), len(pathogenic_sample))
    if max_benign is not None:
        benign_limit = min(benign_limit, max_benign)

    benign_sample = sample_or_all(benign, benign_limit, random_state)

    sampled = pd.concat([pathogenic_sample, benign_sample], ignore_index=True)
    sampled = sampled.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return sampled


def sequence_matches_ref(sequence: str | None, ref: str, flank_size: int) -> bool:
    sequence = clean_sequence(sequence)
    if sequence is None or len(sequence) < flank_size + len(ref):
        return False
    return sequence[flank_size : flank_size + len(ref)] == ref


def load_progress_sequences(progress_path: Path) -> dict[str, str]:
    if not progress_path.exists():
        return {}

    progress_df = pd.read_csv(progress_path)
    if "variant_id" not in progress_df.columns or "sequence" not in progress_df.columns:
        return {}

    sequences: dict[str, str] = {}
    for row in progress_df.itertuples(index=False):
        variant_id = str(getattr(row, "variant_id"))
        sequence = clean_sequence(getattr(row, "sequence"))
        if sequence is not None:
            sequences[variant_id] = sequence
    return sequences


def save_progress(records: list[dict[str, object]], progress_path: Path) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(records).to_csv(progress_path, index=False)


def save_dataframe(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_fetcher_cache(fetcher) -> None:
    if hasattr(fetcher, "save_cache"):
        fetcher.save_cache()


def fetch_reference_sequences(
    df: pd.DataFrame,
    output_dir: Path,
    flank_size: int,
    fetch_mode: str,
    fasta_path: Path | None,
    sequence_cache: Path | None,
    sleep_seconds: float,
    max_retries: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "sequence_progress.csv"
    cache_path = sequence_cache or (output_dir / f"sequence_cache_flank{flank_size}.json")
    progress_sequences = load_progress_sequences(progress_path)

    fetcher = build_sequence_fetcher(
        mode=fetch_mode,
        fasta_path=fasta_path,
        cache_path=cache_path,
        sleep_seconds=sleep_seconds,
        max_retries=max_retries,
    )

    records: list[dict[str, object]] = []
    failed_fetches = 0
    ref_mismatches = 0
    too_short = 0
    reused_source_sequences = 0
    reused_progress_sequences = 0

    for processed, row_dict in enumerate(tqdm(df.to_dict("records"), total=len(df), desc="Fetching reference sequences"), start=1):
        variant_id = str(row_dict["variant_id"])
        ref = str(row_dict["REF"]).upper()

        sequence = clean_sequence(row_dict.get("sequence"))
        if sequence_matches_ref(sequence, ref, flank_size):
            reused_source_sequences += 1
        else:
            sequence = clean_sequence(progress_sequences.get(variant_id))
            if sequence_matches_ref(sequence, ref, flank_size):
                reused_progress_sequences += 1
            else:
                sequence = fetcher.fetch(row_dict["CHROM"], int(row_dict["POS"]), ref, flank_size)
                sequence = clean_sequence(sequence)

        if sequence is None:
            failed_fetches += 1
        elif len(sequence) < MIN_SEQUENCE_LENGTH:
            too_short += 1
        elif not sequence_matches_ref(sequence, ref, flank_size):
            ref_mismatches += 1
        else:
            row_dict["sequence"] = sequence
            records.append(row_dict)

        if processed % PROGRESS_EVERY == 0:
            save_progress(records, progress_path)
            save_fetcher_cache(fetcher)
            print(f"Saved sequence progress after {processed:,} variants: {progress_path}")

    save_progress(records, progress_path)
    save_fetcher_cache(fetcher)

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(records)),
        "failed_fetches": int(failed_fetches),
        "too_short": int(too_short),
        "ref_mismatches": int(ref_mismatches),
        "reused_source_sequences": int(reused_source_sequences),
        "reused_progress_sequences": int(reused_progress_sequences),
    }
    return pd.DataFrame.from_records(records), stats


def stratified_train_val_test_split(df: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df["label"] if df["label"].nunique() == 2 and df["label"].value_counts().min() >= 3 else None
    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=random_state,
        stratify=stratify,
    )

    temp_stratify = temp_df["label"] if temp_df["label"].nunique() == 2 and temp_df["label"].value_counts().min() >= 2 else None
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=random_state,
        stratify=temp_stratify,
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def clean_reference_output(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = [column for column in DROP_OUTPUT_COLUMNS if column in df.columns]
    return df.drop(columns=columns_to_drop, errors="ignore")


def build_alt_sequences(df: pd.DataFrame, flank_size: int) -> tuple[pd.DataFrame, int]:
    records: list[dict[str, object]] = []
    failed = 0

    for row_dict in df.to_dict("records"):
        ref_sequence = clean_sequence(row_dict["sequence"])
        ref = str(row_dict["REF"]).upper()
        alt = str(row_dict["ALT"]).upper()

        if ref_sequence is None or not sequence_matches_ref(ref_sequence, ref, flank_size):
            failed += 1
            continue

        upstream = ref_sequence[:flank_size]
        downstream = ref_sequence[flank_size + len(ref) :]
        alt_sequence = upstream + alt + downstream

        output_row = row_dict.copy()
        output_row["ref_sequence"] = ref_sequence
        output_row["alt_sequence"] = alt_sequence
        output_row["sequence"] = alt_sequence
        output_row["ref_center"] = ref_sequence[flank_size : flank_size + len(ref)]
        output_row["alt_center"] = alt_sequence[flank_size : flank_size + len(alt)]
        records.append(output_row)

    return pd.DataFrame.from_records(records), failed


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
    alt_output_dir: Path,
    flank_size: int,
) -> tuple[dict[str, Path], dict[str, Path], int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    alt_output_dir.mkdir(parents=True, exist_ok=True)

    reference_paths: dict[str, Path] = {}
    alt_paths: dict[str, Path] = {}
    total_alt_failures = 0

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        reference_path = output_dir / ORIGINAL_SPLIT_FILES[split_name]
        reference_df = clean_reference_output(split_df)
        reference_df.to_csv(reference_path, index=False)
        reference_paths[split_name] = reference_path

        alt_df, alt_failures = build_alt_sequences(reference_df, flank_size)
        total_alt_failures += alt_failures
        alt_path = alt_output_dir / ALT_SPLIT_FILES[split_name]
        alt_df.to_csv(alt_path, index=False)
        alt_paths[split_name] = alt_path

    return reference_paths, alt_paths, total_alt_failures


def print_distribution(title: str, df: pd.DataFrame) -> None:
    print(title)
    if df.empty or "label" not in df.columns:
        print("  no rows")
    else:
        print(df["label"].value_counts().sort_index().to_string())
    print()


def warn_if_imbalanced(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    if len(counts) < 2:
        print("WARNING: selected sample contains only one class. Training will not be useful.")
        return
    if counts.iloc[0] != counts.iloc[1]:
        print("WARNING: exact class balance was not possible with the available/capped variants.")
        print(counts.to_string())
        print()


def main() -> None:
    args = parse_args()
    root = project_root()
    output_dir_arg = args.output_dir or default_output_dir_for_target(args.target_total)
    output_dir = resolve_path(root, output_dir_arg)
    if output_dir is None:
        raise ValueError("--output_dir is required")
    alt_output_dir = alt_output_dir_for(output_dir)
    requested_vcf = resolve_path(root, args.clinvar_vcf)
    fasta_path = resolve_path(root, args.fasta_path)
    sequence_cache = resolve_path(root, args.sequence_cache)

    print("Prepare larger ClinVar dataset")
    print(f"Target total rows: {args.target_total:,}")
    print(f"Output reference directory: {output_dir}")
    print(f"Output alternate directory: {alt_output_dir}")
    print(f"Flank size: {args.flank_size}")
    print()

    variants_df, source_name, before_filter_count, after_filter_count = choose_variant_source(
        root=root,
        target_total=args.target_total,
        requested_vcf=requested_vcf,
    )

    print("=" * 80)
    print("FILTERING SUMMARY")
    print("=" * 80)
    print(f"Selected source: {source_name}")
    print(f"Total variants before filtering: {before_filter_count:,}")
    print(f"Total variants after filtering: {after_filter_count:,}")
    print_distribution("Class distribution after filtering:", variants_df)
    filtered_variants_path = save_dataframe(variants_df, output_dir / "clinvar_binary_variants.csv")
    print(f"Saved filtered variant table: {filtered_variants_path}")
    print()

    if variants_df.empty:
        raise RuntimeError("No variants remained after filtering. Check the input VCF/CSV and CLNSIG labels.")

    sampled_df = balanced_sample(
        variants_df,
        target_total=args.target_total,
        max_pathogenic=args.max_pathogenic,
        max_benign=args.max_benign,
        random_state=args.random_state,
    )

    print("=" * 80)
    print("SAMPLING SUMMARY")
    print("=" * 80)
    print(f"Selected sample size: {len(sampled_df):,}")
    print_distribution("Class distribution after balanced sampling:", sampled_df)
    warn_if_imbalanced(sampled_df)
    selected_variants_path = save_dataframe(sampled_df, output_dir / "selected_variants.csv")
    print(f"Saved selected variant table: {selected_variants_path}")
    print()

    if sampled_df.empty or sampled_df["label"].nunique() < 2:
        raise RuntimeError("Balanced sampling did not produce both classes. Adjust target_total/max caps.")

    sequenced_df, fetch_stats = fetch_reference_sequences(
        sampled_df,
        output_dir=output_dir,
        flank_size=args.flank_size,
        fetch_mode=args.fetch_mode,
        fasta_path=fasta_path,
        sequence_cache=sequence_cache,
        sleep_seconds=args.sleep_seconds,
        max_retries=args.max_retries,
    )

    print("=" * 80)
    print("SEQUENCE FETCH SUMMARY")
    print("=" * 80)
    print(f"Rows requested for sequence fetching: {fetch_stats['input_rows']:,}")
    print(f"Rows with usable reference sequences: {fetch_stats['output_rows']:,}")
    print(f"Failed sequence fetches: {fetch_stats['failed_fetches']:,}")
    print(f"Too-short sequences: {fetch_stats['too_short']:,}")
    print(f"Reference mismatches at variant center: {fetch_stats['ref_mismatches']:,}")
    print(f"Reused source sequences: {fetch_stats['reused_source_sequences']:,}")
    print(f"Reused progress sequences: {fetch_stats['reused_progress_sequences']:,}")
    print_distribution("Class distribution after sequence filtering:", sequenced_df)

    if sequenced_df.empty:
        raise RuntimeError("No usable sequences were created. Check VCF/input data and sequence fetching settings.")

    train_df, val_df, test_df = stratified_train_val_test_split(sequenced_df, args.random_state)

    print("=" * 80)
    print("SPLIT SUMMARY")
    print("=" * 80)
    print(f"Train rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print_distribution("Train label distribution:", train_df)
    print_distribution("Validation label distribution:", val_df)
    print_distribution("Test label distribution:", test_df)

    reference_paths, alt_paths, alt_failures = save_splits(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_dir=output_dir,
        alt_output_dir=alt_output_dir,
        flank_size=args.flank_size,
    )

    print("=" * 80)
    print("OUTPUT SUMMARY")
    print("=" * 80)
    print(f"Alternate sequence build failures: {alt_failures:,}")
    print("Reference-sequence output paths:")
    for split_name, path in reference_paths.items():
        print(f"  {split_name}: {path}")
    print("Alternate-sequence output paths:")
    for split_name, path in alt_paths.items():
        print(f"  {split_name}: {path}")
    print()
    print("Larger ClinVar dataset preparation completed.")


if __name__ == "__main__":
    main()
