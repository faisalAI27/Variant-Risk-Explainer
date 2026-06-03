"""Manual ClinVar VCF parsing helpers for Colab preprocessing.

The parser intentionally uses gzip and simple VCF column handling so beginners
can inspect the preprocessing logic without needing a genomics-specific parser.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from urllib.parse import unquote

import pandas as pd

from training.utils.label_utils import assign_binary_label, label_name


INFO_FIELDS = ("CLNSIG", "GENEINFO", "CLNHGVS", "CLNVC")
DNA_BASES = set("ACGTN")


def parse_info_field(info_value: str) -> dict[str, str]:
    """Parse the VCF INFO column into a dictionary."""
    parsed: dict[str, str] = {}
    for item in info_value.split(";"):
        if not item:
            continue
        if "=" not in item:
            parsed[item] = "true"
            continue
        key, value = item.split("=", 1)
        parsed[key] = unquote(value)
    return parsed


def parse_clinvar_vcf(vcf_path: str | Path, max_records: int | None = None) -> pd.DataFrame:
    """Read a ClinVar GRCh38 VCF.GZ file into a DataFrame with selected fields."""
    records: list[dict[str, object]] = []
    vcf_path = Path(vcf_path)

    with gzip.open(vcf_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("##"):
                continue
            if line.startswith("#"):
                continue

            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8:
                continue

            chrom, pos_raw, clinvar_id, ref, alt, _qual, _filter, info_raw = fields[:8]
            info = parse_info_field(info_raw)

            record = {
                "CHROM": chrom,
                "POS": int(pos_raw),
                "ID": clinvar_id,
                "REF": ref.upper(),
                "ALT": alt.upper(),
                "INFO": info_raw,
                "CLNSIG": info.get("CLNSIG"),
                "GENEINFO": info.get("GENEINFO"),
                "CLNHGVS": info.get("CLNHGVS"),
                "CLNVC": info.get("CLNVC"),
            }
            records.append(record)

            if max_records is not None and len(records) >= max_records:
                break

    return pd.DataFrame.from_records(records)


def has_multiple_alt(alt: str | None) -> bool:
    """Return True when the VCF ALT field contains multiple alternate alleles."""
    return bool(alt and "," in str(alt))


def is_symbolic_alt(alt: str | None) -> bool:
    """Return True for symbolic or breakend-style alternate alleles."""
    if alt is None:
        return True

    value = str(alt).strip().upper()
    if not value or value == ".":
        return True
    return value.startswith("<") or value.endswith(">") or "[" in value or "]" in value


def is_sequence_allele(value: str | None) -> bool:
    """Return True when an allele contains only simple DNA bases."""
    if value is None:
        return False
    allele = str(value).strip().upper()
    return bool(allele) and all(base in DNA_BASES for base in allele)


def classify_variant_type(ref: str | None, alt: str | None) -> str | None:
    """Classify supported MVP variants as SNV or INDEL."""
    if not is_sequence_allele(ref) or not is_sequence_allele(alt):
        return None

    ref_value = str(ref).upper()
    alt_value = str(alt).upper()

    if len(ref_value) == 1 and len(alt_value) == 1:
        return "SNV"

    length_delta = abs(len(ref_value) - len(alt_value))
    if len(ref_value) != len(alt_value) and length_delta <= 50:
        return "INDEL"

    return None


def extract_gene_symbol(gene_info: str | None) -> str | None:
    """Extract the first gene symbol from ClinVar GENEINFO."""
    if gene_info is None:
        return None

    value = str(gene_info).strip()
    if not value or value == ".":
        return None

    first_gene = value.split("|", 1)[0]
    symbol = first_gene.split(":", 1)[0].strip()
    return symbol or None


def add_variant_id(row: pd.Series) -> str:
    """Build a stable GRCh38 variant identifier."""
    return f"GRCh38-{row['CHROM']}-{row['POS']}-{row['REF']}-{row['ALT']}"


def prepare_binary_variants(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Apply MVP filters and binary labels to parsed ClinVar rows."""
    df = raw_df.copy()
    df["has_multiple_alt"] = df["ALT"].apply(has_multiple_alt)
    df["is_symbolic_alt"] = df["ALT"].apply(is_symbolic_alt)
    df["variant_type"] = df.apply(lambda row: classify_variant_type(row["REF"], row["ALT"]), axis=1)
    df["label"] = df["CLNSIG"].apply(assign_binary_label)
    df["label_name"] = df["label"].apply(label_name)
    df["gene_symbol"] = df["GENEINFO"].apply(extract_gene_symbol)

    keep_mask = (
        ~df["has_multiple_alt"]
        & ~df["is_symbolic_alt"]
        & df["variant_type"].notna()
        & df["label"].notna()
    )

    filtered = df.loc[keep_mask].copy()
    filtered["label"] = filtered["label"].astype(int)
    filtered["variant_id"] = filtered.apply(add_variant_id, axis=1)

    output_columns = [
        "variant_id",
        "CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "variant_type",
        "gene_symbol",
        "GENEINFO",
        "CLNSIG",
        "CLNHGVS",
        "CLNVC",
        "label",
        "label_name",
    ]
    return filtered[output_columns].reset_index(drop=True)
