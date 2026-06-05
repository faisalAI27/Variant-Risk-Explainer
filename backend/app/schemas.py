from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


VALID_CHROMOSOMES = {str(value) for value in range(1, 23)} | {"X", "Y", "MT"}
VALID_BASES = set("ACGTN")
DISCLAIMER = (
    "Research/demo use only. This model is not a clinical diagnostic system "
    "and must not be used for medical decisions."
)


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "variant_name": "GRCh38-7-140753336-A-T",
                "gene": "BRAF",
                "sequence": "ACGTACGTACGTACGTACGT",
                "notes": "Example sequence for research demo testing.",
            }
        }
    )

    sequence: str = Field(..., description="DNA sequence using A/C/G/T/N.")
    variant_name: str | None = Field(default=None, max_length=128)
    gene: str | None = Field(default=None, max_length=64)
    notes: str | None = Field(default=None, max_length=1000)


class AnalyzeResponse(BaseModel):
    variant_name: str | None
    gene: str | None
    prediction_class: Literal[0, 1]
    prediction_label: str
    risk_level: Literal["Lower", "Elevated"]
    benign_probability: float = Field(..., ge=0.0, le=1.0)
    pathogenic_probability: float = Field(..., ge=0.0, le=1.0)
    threshold: float = Field(..., ge=0.0, le=1.0)
    model_name: str
    sequence_length_used: int
    disclaimer: str = DISCLAIMER


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    device: str
    model_dir: str
    threshold: float
    model_name: str
    load_error: str | None = None


# Legacy schemas kept so older modules/tests importing them do not break.
def normalize_chromosome(value: str) -> str:
    chrom = str(value).strip().upper()
    if chrom.startswith("CHR"):
        chrom = chrom[3:]
    if chrom == "M":
        chrom = "MT"
    return chrom


class VariantRequest(BaseModel):
    chromosome: str = Field(..., description="GRCh38 chromosome: 1-22, X, Y, MT, with optional chr prefix.")
    position: int = Field(..., description="1-based GRCh38 coordinate.")
    reference: str = Field(..., description="Reference allele.")
    alternate: str = Field(..., description="Alternate allele.")
    gene: str | None = Field(default=None, max_length=32, description="Optional gene symbol.")
    sequence_context: str | None = Field(default=None, description="Optional GRCh38 sequence context.")

    @field_validator("chromosome", mode="before")
    @classmethod
    def validate_chromosome(cls, value: str) -> str:
        chrom = normalize_chromosome(value)
        if chrom not in VALID_CHROMOSOMES:
            raise ValueError("chromosome must be GRCh38 chromosome 1-22, X, Y, or MT")
        return chrom

    @field_validator("position")
    @classmethod
    def validate_position(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("position must be a positive GRCh38 coordinate")
        return value

    @field_validator("reference", "alternate", mode="before")
    @classmethod
    def validate_allele(cls, value: str) -> str:
        allele = str(value).strip().upper()
        if not allele:
            raise ValueError("allele cannot be empty")
        if len(allele) > 50:
            raise ValueError("allele must be 50 bases or fewer for this demo API")
        if any(base not in VALID_BASES for base in allele):
            raise ValueError("allele must contain only A, C, G, T, or N")
        return allele

    @field_validator("gene", mode="before")
    @classmethod
    def normalize_gene(cls, value: str | None) -> str | None:
        if value is None:
            return None
        gene = str(value).strip().upper()
        return gene or None

    @field_validator("sequence_context", mode="before")
    @classmethod
    def validate_sequence_context(cls, value: str | None) -> str | None:
        if value is None:
            return None
        sequence = str(value).strip().upper().replace(" ", "").replace("\n", "")
        if not sequence:
            return None
        if any(base not in VALID_BASES for base in sequence):
            raise ValueError("sequence_context must contain only A, C, G, T, or N")
        return sequence

    @model_validator(mode="after")
    def validate_variant(self) -> "VariantRequest":
        if self.reference == self.alternate:
            raise ValueError("reference and alternate must differ")
        return self


class VariantAnalysisResponse(BaseModel):
    request_id: str
    submitted_at: str
    input: VariantRequest
    grch_build: Literal["GRCh38"] = "GRCh38"
    risk_label: Literal["likely_benign", "uncertain", "likely_pathogenic"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_mode: Literal["mock", "trained"]
    explanation: str
    limitations: list[str]
    disclaimer: str
