"""ClinVar clinical-significance label helpers for binary classification."""

from __future__ import annotations

from urllib.parse import unquote


DROP_LABEL_TERMS = (
    "conflicting interpretations",
    "uncertain significance",
    "risk factor",
    "association",
    "drug response",
    "protective",
    "not provided",
)


def normalize_clnsig(value: str | None) -> str:
    """Normalize a raw ClinVar CLNSIG value for matching."""
    if value is None:
        return ""

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


def should_drop_clnsig(value: str | None) -> bool:
    """Return True when a CLNSIG value should be excluded from the MVP dataset."""
    normalized = normalize_clnsig(value)
    if not normalized or normalized == ".":
        return True
    return any(term in normalized for term in DROP_LABEL_TERMS)


def assign_binary_label(value: str | None) -> int | None:
    """Map ClinVar CLNSIG to 1 for pathogenic and 0 for benign.

    Rows with uncertain, conflicting, unsupported, or mixed pathogenic/benign
    labels return None so they can be dropped before splitting.
    """
    if should_drop_clnsig(value):
        return None

    normalized = normalize_clnsig(value)
    has_pathogenic = "pathogenic" in normalized
    has_benign = "benign" in normalized

    if has_pathogenic and has_benign:
        return None
    if has_pathogenic:
        return 1
    if has_benign:
        return 0
    return None


def label_name(label: int | None) -> str | None:
    """Return a human-readable binary label name."""
    if label == 1:
        return "pathogenic"
    if label == 0:
        return "benign_or_likely_benign"
    return None
