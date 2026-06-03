from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.core.config import Settings
from app.schemas import VariantRequest


RISK_LABELS = ("likely_benign", "uncertain", "likely_pathogenic")
LIMITATIONS = [
    "Research demo only.",
    "Not validated for diagnosis or treatment decisions.",
    "ClinVar labels may be incomplete, conflicting, or biased.",
]
DISCLAIMER = "For research and education only. Not for medical diagnosis."


@dataclass(frozen=True)
class ModelPrediction:
    risk_label: str
    confidence: float
    model_mode: str
    explanation: str


class VariantModel(Protocol):
    mode: str

    def analyze(self, request: VariantRequest) -> ModelPrediction:
        ...


class MockVariantModel:
    mode = "mock"

    def __init__(self, fallback_reason: str | None = None) -> None:
        self.fallback_reason = fallback_reason

    def analyze(self, request: VariantRequest) -> ModelPrediction:
        score = self._score(request)
        if score >= 0.66:
            label = "likely_pathogenic"
        elif score <= 0.34:
            label = "likely_benign"
        else:
            label = "uncertain"

        confidence = self._confidence(score, label)
        reason = "Mock mode produced a deterministic research-only score from variant features."
        if self.fallback_reason:
            reason += f" Fallback reason: {self.fallback_reason}."
        reason += " No clinical meaning should be inferred."
        return ModelPrediction(
            risk_label=label,
            confidence=confidence,
            model_mode=self.mode,
            explanation=reason,
        )

    def _score(self, request: VariantRequest) -> float:
        variant_key = f"{request.chromosome}:{request.position}:{request.reference}>{request.alternate}:{request.gene or ''}"
        digest = hashlib.sha256(variant_key.encode("utf-8")).hexdigest()
        jitter = int(digest[:8], 16) / 0xFFFFFFFF

        score = 0.5 + ((jitter - 0.5) * 0.24)

        if len(request.reference) != len(request.alternate):
            score += 0.08
        if request.sequence_context:
            gc_count = request.sequence_context.count("G") + request.sequence_context.count("C")
            gc_fraction = gc_count / len(request.sequence_context)
            score += (gc_fraction - 0.5) * 0.12
            if "N" in request.sequence_context:
                score -= 0.04
        if request.gene:
            score += 0.02

        return min(0.95, max(0.05, score))

    def _confidence(self, score: float, label: str) -> float:
        if label == "uncertain":
            return round(0.5 + min(0.12, abs(score - 0.5)), 2)
        distance_from_boundary = min(abs(score - 0.34), abs(score - 0.66))
        return round(min(0.92, 0.62 + distance_from_boundary), 2)


class HuggingFaceVariantModel:
    mode = "trained"

    def __init__(self, model_dir: str) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install backend/requirements-model.txt to use trained mode.") from exc

        self.torch = torch
        model_path = Path(model_dir).expanduser()
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path), trust_remote_code=True)
        self.model.eval()

    def analyze(self, request: VariantRequest) -> ModelPrediction:
        if not request.sequence_context:
            raise ValueError("trained model mode requires sequence_context from GRCh38")

        sequence = self._sequence_with_alt(request)
        encoded = self.tokenizer(sequence, return_tensors="pt", truncation=True, max_length=256)

        with self.torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probabilities = self.torch.softmax(logits, dim=-1).cpu().numpy()

        pathogenic_probability = float(probabilities[1]) if len(probabilities) > 1 else float(probabilities[0])
        if pathogenic_probability >= 0.66:
            label = "likely_pathogenic"
            confidence = pathogenic_probability
        elif pathogenic_probability <= 0.34:
            label = "likely_benign"
            confidence = 1.0 - pathogenic_probability
        else:
            label = "uncertain"
            confidence = 1.0 - abs(pathogenic_probability - 0.5)

        return ModelPrediction(
            risk_label=label,
            confidence=round(float(confidence), 4),
            model_mode=self.mode,
            explanation=(
                "Trained DNABERT-2 research model scored the submitted GRCh38 sequence context. "
                "This output is experimental and not clinical evidence."
            ),
        )

    def _sequence_with_alt(self, request: VariantRequest) -> str:
        sequence = request.sequence_context or ""
        if len(sequence) < 3 or len(sequence) % 2 == 0:
            return sequence
        center = math.floor(len(sequence) / 2)
        if len(request.reference) == 1 and len(request.alternate) == 1:
            return sequence[:center] + request.alternate + sequence[center + 1 :]
        return sequence


def build_model(settings: Settings) -> VariantModel:
    if settings.model_mode == "mock":
        return MockVariantModel()

    if settings.model_dir and Path(settings.model_dir).expanduser().exists():
        try:
            return HuggingFaceVariantModel(settings.model_dir)
        except Exception as exc:
            if settings.model_mode == "trained":
                raise
            return MockVariantModel(fallback_reason=str(exc))

    if settings.model_mode == "trained":
        raise RuntimeError("MODEL_MODE=trained requires MODEL_DIR to point to an exported model directory.")

    return MockVariantModel(fallback_reason="MODEL_DIR is not configured")
