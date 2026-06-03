"""Sequence fetching helpers for GRCh38 ClinVar preprocessing.

Supports two Colab-friendly modes:

- UCSC API mode using https://api.genome.ucsc.edu/getData/sequence
- Local FASTA mode using pyfaidx when a GRCh38 FASTA is available
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
import requests
from tqdm.auto import tqdm


UCSC_SEQUENCE_ENDPOINT = "https://api.genome.ucsc.edu/getData/sequence"
VALID_SEQUENCE_BASES = set("ACGTN")


def normalize_ucsc_chromosome(chromosome: str) -> str:
    """Convert ClinVar chromosome values such as 1, chr1, MT, or M to UCSC names."""
    value = str(chromosome).strip()
    if not value:
        raise ValueError("chromosome cannot be empty")

    if value.lower().startswith("chr"):
        suffix = value[3:]
    else:
        suffix = value

    suffix_upper = suffix.upper()
    if suffix_upper in {"M", "MT", "MITO"}:
        return "chrM"
    return f"chr{suffix_upper if suffix_upper in {'X', 'Y'} else suffix}"


def candidate_fasta_chromosomes(chromosome: str) -> list[str]:
    """Return possible chromosome names for common GRCh38 FASTA conventions."""
    ucsc = normalize_ucsc_chromosome(chromosome)
    suffix = ucsc[3:]

    candidates = [ucsc]
    if suffix == "M":
        candidates.extend(["MT", "M"])
    else:
        candidates.append(suffix)

    seen: set[str] = set()
    return [item for item in candidates if not (item in seen or seen.add(item))]


def make_sequence_interval(chromosome: str, position: int, ref: str, flank_size: int) -> tuple[str, int, int]:
    """Create the 0-based half-open interval requested from UCSC or FASTA."""
    chrom = normalize_ucsc_chromosome(chromosome)
    start = int(position) - 1 - int(flank_size)
    end = int(position) - 1 + len(str(ref)) + int(flank_size)
    return chrom, max(0, start), max(0, end)


def make_cache_key(chromosome: str, position: int, ref: str, flank_size: int) -> str:
    """Build a stable cache key for sequence fetches."""
    chrom, start, end = make_sequence_interval(chromosome, position, ref, flank_size)
    return f"hg38:{chrom}:{start}-{end}"


def clean_sequence(sequence: str | None) -> str | None:
    """Uppercase sequence and keep only A/C/G/T/N characters."""
    if sequence is None:
        return None
    cleaned = "".join(base for base in str(sequence).upper() if base in VALID_SEQUENCE_BASES)
    return cleaned or None


def load_sequence_cache(cache_path: str | Path | None) -> dict[str, str]:
    """Load a JSON sequence cache if it exists."""
    if cache_path is None:
        return {}

    path = Path(cache_path)
    if not path.exists():
        return {}

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

    return {str(key): str(value) for key, value in data.items()}


def save_sequence_cache(cache: dict[str, str], cache_path: str | Path | None) -> None:
    """Persist the sequence cache as JSON."""
    if cache_path is None:
        return

    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8")


class SequenceFetcher(Protocol):
    """Common interface for sequence fetchers."""

    def fetch(self, chromosome: str, position: int, ref: str, flank_size: int) -> str | None:
        ...


@dataclass
class UcscSequenceFetcher:
    """Fetch GRCh38 sequence windows from the UCSC Genome Browser API."""

    genome: str = "hg38"
    endpoint: str = UCSC_SEQUENCE_ENDPOINT
    sleep_seconds: float = 0.15
    max_retries: int = 3
    timeout_seconds: int = 30
    cache_path: str | Path | None = None
    session: requests.Session = field(default_factory=requests.Session)

    def __post_init__(self) -> None:
        self.cache: dict[str, str] = load_sequence_cache(self.cache_path)
        self.cache_hits = 0
        self.cache_misses = 0
        self.failed_fetches = 0
        self._last_request_at = 0.0

    def fetch(self, chromosome: str, position: int, ref: str, flank_size: int) -> str | None:
        chrom, start, end = make_sequence_interval(chromosome, position, ref, flank_size)
        cache_key = f"{self.genome}:{chrom}:{start}-{end}"

        if cache_key in self.cache:
            self.cache_hits += 1
            return self.cache[cache_key]

        self.cache_misses += 1
        params = {
            "genome": self.genome,
            "chrom": chrom,
            "start": start,
            "end": end,
        }

        for attempt in range(1, self.max_retries + 1):
            self._sleep_between_requests()
            try:
                response = self.session.get(self.endpoint, params=params, timeout=self.timeout_seconds)
                response.raise_for_status()
                payload: dict[str, Any] = response.json()
            except Exception:
                if attempt == self.max_retries:
                    self.failed_fetches += 1
                    return None
                time.sleep(min(8.0, self.sleep_seconds * (2**attempt)))
                continue

            if "error" in payload:
                if attempt == self.max_retries:
                    self.failed_fetches += 1
                    return None
                time.sleep(min(8.0, self.sleep_seconds * (2**attempt)))
                continue

            sequence = clean_sequence(payload.get("dna") or payload.get("sequence"))
            if sequence is None:
                self.failed_fetches += 1
                return None

            self.cache[cache_key] = sequence
            return sequence

        self.failed_fetches += 1
        return None

    def save_cache(self) -> None:
        """Persist any fetched UCSC sequences."""
        save_sequence_cache(self.cache, self.cache_path)

    def _sleep_between_requests(self) -> None:
        if self.sleep_seconds <= 0:
            return

        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.sleep_seconds:
            time.sleep(self.sleep_seconds - elapsed)
        self._last_request_at = time.monotonic()


@dataclass
class LocalFastaSequenceFetcher:
    """Fetch GRCh38 sequence windows from a local FASTA file with pyfaidx."""

    fasta_path: str | Path

    def __post_init__(self) -> None:
        from pyfaidx import Fasta

        self.fasta = Fasta(str(self.fasta_path), as_raw=True, sequence_always_upper=True)
        self.failed_fetches = 0

    def fetch(self, chromosome: str, position: int, ref: str, flank_size: int) -> str | None:
        _ucsc_chrom, start, end = make_sequence_interval(chromosome, position, ref, flank_size)

        for contig in candidate_fasta_chromosomes(chromosome):
            if contig not in self.fasta.keys():
                continue
            try:
                sequence = clean_sequence(self.fasta[contig][start:end])
            except Exception:
                continue
            if sequence is not None:
                return sequence

        self.failed_fetches += 1
        return None


def build_sequence_fetcher(
    mode: str,
    fasta_path: str | Path | None = None,
    cache_path: str | Path | None = None,
    sleep_seconds: float = 0.15,
    max_retries: int = 3,
) -> SequenceFetcher:
    """Create a UCSC API or local FASTA sequence fetcher."""
    normalized_mode = mode.strip().lower()
    if normalized_mode == "ucsc":
        return UcscSequenceFetcher(
            cache_path=cache_path,
            sleep_seconds=sleep_seconds,
            max_retries=max_retries,
        )
    if normalized_mode == "fasta":
        if fasta_path is None:
            raise ValueError("fasta_path is required when mode='fasta'")
        return LocalFastaSequenceFetcher(fasta_path=fasta_path)
    raise ValueError("mode must be either 'ucsc' or 'fasta'")


def add_sequences_to_dataframe(
    df: pd.DataFrame,
    fetcher: SequenceFetcher,
    flank_size: int = 512,
    min_seq_len: int = 200,
    progress_desc: str = "Fetching sequences",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Fetch sequence windows, drop failures, and return a filtered DataFrame."""
    records: list[dict[str, Any]] = []
    failed_fetches = 0
    too_short = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc=progress_desc):
        row_dict = row._asdict()
        sequence = fetcher.fetch(row_dict["CHROM"], int(row_dict["POS"]), row_dict["REF"], flank_size)
        sequence = clean_sequence(sequence)

        if sequence is None:
            failed_fetches += 1
            continue
        if len(sequence) < min_seq_len:
            too_short += 1
            continue

        row_dict["sequence"] = sequence
        records.append(row_dict)

    if records:
        output_df = pd.DataFrame.from_records(records)
    else:
        output_df = df.iloc[0:0].copy()
        output_df["sequence"] = pd.Series(dtype="object")

    stats = {
        "input_rows": int(len(df)),
        "output_rows": int(len(output_df)),
        "failed_fetches": int(failed_fetches),
        "too_short": int(too_short),
        "dropped_rows": int(len(df) - len(output_df)),
    }
    return output_df, stats
