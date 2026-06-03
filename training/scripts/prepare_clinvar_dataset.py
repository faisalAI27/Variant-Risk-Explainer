#!/usr/bin/env python
"""Prepare ClinVar GRCh38 SNV sequence windows for DNABERT-2 fine-tuning.

This script is intended for Google Colab. It expects a ClinVar GRCh38 VCF and
a GRCh38 FASTA indexed by pysam. The output is JSONL, one example per variant.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable

import pysam


LABEL_TO_ID = {
    "likely_benign": 0,
    "likely_pathogenic": 1,
}

SKIP_CLNSIG_TERMS = (
    "conflicting",
    "uncertain",
    "not provided",
    "not_provided",
    "other",
    "risk factor",
    "association",
    "drug response",
    "protective",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinvar-vcf", required=True, help="Path to ClinVar GRCh38 VCF or VCF.GZ.")
    parser.add_argument("--reference-fasta", required=True, help="Path to indexed GRCh38 FASTA.")
    parser.add_argument("--output-jsonl", required=True, help="Destination JSONL file.")
    parser.add_argument("--window-size", type=int, default=251, help="Odd sequence window size centered on the variant.")
    parser.add_argument("--max-records", type=int, default=0, help="Optional cap for quick Colab smoke tests. 0 means no cap.")
    return parser.parse_args()


def open_text(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "rt", encoding="utf-8")


def parse_info(info: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for item in info.split(";"):
        if not item:
            continue
        if "=" not in item:
            parsed[item] = "true"
            continue
        key, value = item.split("=", 1)
        parsed[key] = value
    return parsed


def normalize_clnsig(raw: str) -> str:
    return (
        raw.replace("%2C", ",")
        .replace("%2c", ",")
        .replace("_", " ")
        .replace("/", " ")
        .replace("|", " ")
        .lower()
    )


def map_clnsig(raw: str | None) -> str | None:
    if not raw:
        return None

    normalized = normalize_clnsig(raw)
    if any(term in normalized for term in SKIP_CLNSIG_TERMS):
        return None

    has_pathogenic = "pathogenic" in normalized
    has_benign = "benign" in normalized

    if has_pathogenic and has_benign:
        return None
    if has_pathogenic:
        return "likely_pathogenic"
    if has_benign:
        return "likely_benign"
    return None


def extract_gene(info: dict[str, str]) -> str | None:
    gene_info = info.get("GENEINFO")
    if not gene_info:
        return None
    first_gene = gene_info.split("|", 1)[0]
    return first_gene.split(":", 1)[0] or None


def contig_candidates(chrom: str) -> Iterable[str]:
    yield chrom
    if chrom.startswith("chr"):
        yield chrom.removeprefix("chr")
    else:
        yield f"chr{chrom}"
    if chrom == "MT":
        yield "chrM"
    if chrom == "chrM":
        yield "MT"


def fetch_window(reference: pysam.FastaFile, chrom: str, pos: int, window_size: int) -> tuple[str, str] | None:
    half = window_size // 2
    start = pos - 1 - half
    end = pos - 1 + half + 1

    if start < 0:
        return None

    for contig in contig_candidates(chrom):
        if contig not in reference.references:
            continue
        if end > reference.get_reference_length(contig):
            return None
        return contig, reference.fetch(contig, start, end).upper()
    return None


def make_alt_sequence(reference_sequence: str, alt: str) -> str:
    center = len(reference_sequence) // 2
    return reference_sequence[:center] + alt.upper() + reference_sequence[center + 1 :]


def prepare_examples(args: argparse.Namespace) -> tuple[int, int]:
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.window_size % 2 == 0:
        raise ValueError("--window-size must be odd so the variant has a center base.")

    written = 0
    scanned = 0

    reference = pysam.FastaFile(args.reference_fasta)
    with open_text(args.clinvar_vcf) as vcf, output_path.open("w", encoding="utf-8") as output:
        for line in vcf:
            if line.startswith("#"):
                continue
            scanned += 1
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos_raw, variant_id, ref, alts, _qual, _filter, info_raw = fields[:8]
            pos = int(pos_raw)
            info = parse_info(info_raw)
            label = map_clnsig(info.get("CLNSIG"))
            if label is None:
                continue

            ref = ref.upper()
            for alt in alts.split(","):
                alt = alt.upper()
                if len(ref) != 1 or len(alt) != 1:
                    continue
                if ref not in {"A", "C", "G", "T"} or alt not in {"A", "C", "G", "T"}:
                    continue

                fetched = fetch_window(reference, chrom, pos, args.window_size)
                if fetched is None:
                    continue
                resolved_contig, reference_sequence = fetched
                if len(reference_sequence) != args.window_size:
                    continue

                center_base = reference_sequence[len(reference_sequence) // 2]
                if center_base != ref:
                    continue

                example = {
                    "id": variant_id,
                    "chromosome": resolved_contig,
                    "position": pos,
                    "reference": ref,
                    "alternate": alt,
                    "gene": extract_gene(info),
                    "clnsig": info.get("CLNSIG"),
                    "sequence": make_alt_sequence(reference_sequence, alt),
                    "reference_sequence": reference_sequence,
                    "label": label,
                    "label_id": LABEL_TO_ID[label],
                    "grch_build": "GRCh38",
                }
                output.write(json.dumps(example) + "\n")
                written += 1

                if args.max_records and written >= args.max_records:
                    return scanned, written

    return scanned, written


def main() -> None:
    args = parse_args()
    scanned, written = prepare_examples(args)
    print(f"Scanned {scanned:,} ClinVar records.")
    print(f"Wrote {written:,} GRCh38 SNV examples to {args.output_jsonl}.")


if __name__ == "__main__":
    main()
