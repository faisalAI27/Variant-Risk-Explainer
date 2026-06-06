from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.core.config import Settings


VALID_DNA_PATTERN = re.compile(r"^[ACGTN]+$")
DISCLAIMER = (
    "Research/demo use only. This model is not a clinical diagnostic system "
    "and must not be used for medical decisions."
)


@dataclass(frozen=True)
class PredictionResult:
    prediction_class: int
    prediction_label: str
    risk_level: str
    benign_probability: float
    pathogenic_probability: float
    threshold: float
    model_name: str
    sequence_length_used: int
    disclaimer: str


class ModelService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_dir = settings.resolved_model_source()
        self.threshold = settings.model_threshold
        self.max_length = settings.model_max_length
        self.model_name = settings.model_name
        self.device = self._select_device(settings.device)
        self.model_loaded = False
        self.load_error: str | None = None
        self.tokenizer = None
        self.model = None

        self._load_model()

    def _is_local_model_source(self) -> bool:
        configured = self.settings.model_dir.strip()
        return Path(self.model_dir).is_absolute() or configured.startswith(("./", "../", "~"))

    def _select_device(self, requested_device: str) -> str:
        if requested_device != "auto":
            if requested_device == "cuda" and not torch.cuda.is_available():
                return "cpu"
            if requested_device == "mps":
                mps_backend = getattr(torch.backends, "mps", None)
                if mps_backend is None or not mps_backend.is_available():
                    return "cpu"
            return requested_device

        if torch.cuda.is_available():
            return "cuda"

        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return "mps"

        return "cpu"

    def _load_model(self) -> None:
        if self._is_local_model_source() and not Path(self.model_dir).exists():
            self.load_error = f"Model directory not found: {self.model_dir}"
            return

        try:
            token = os.getenv("HF_TOKEN", "").strip() or None
            auth_kwargs = {"token": token} if token else {}
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                **auth_kwargs,
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                trust_remote_code=True,
                low_cpu_mem_usage=False,
                **auth_kwargs,
            )
            self.model.to(torch.device(self.device))
            self.model.eval()
            self.model_loaded = True
            self.load_error = None
        except Exception as exc:  # pragma: no cover - exact HF failures vary by platform.
            self.model_loaded = False
            self.load_error = f"{type(exc).__name__}: {exc}"

    def clean_sequence(self, sequence: str) -> str:
        cleaned = str(sequence or "").upper()
        cleaned = re.sub(r"\s+", "", cleaned)

        if not cleaned:
            raise ValueError("sequence cannot be empty")
        if not VALID_DNA_PATTERN.fullmatch(cleaned):
            invalid = sorted(set(cleaned) - set("ACGTN"))
            raise ValueError(f"sequence contains invalid DNA characters: {''.join(invalid)}")

        return cleaned

    def crop_sequence(self, sequence: str) -> str:
        if len(sequence) <= self.max_length:
            return sequence

        start = max(0, (len(sequence) - self.max_length) // 2)
        end = start + self.max_length
        return sequence[start:end]

    def predict(self, sequence: str) -> PredictionResult:
        if not self.model_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not available. Please configure MODEL_DIR correctly.")

        cleaned_sequence = self.clean_sequence(sequence)
        model_sequence = self.crop_sequence(cleaned_sequence)

        encoded = self.tokenizer(
            model_sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(torch.device(self.device)) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probabilities = torch.softmax(logits.float(), dim=-1)[0].detach().cpu().numpy()

        if probabilities.shape[0] < 2:
            raise RuntimeError("model output does not contain two class probabilities")

        benign_probability = float(np.round(probabilities[0], 6))
        pathogenic_probability = float(np.round(probabilities[1], 6))

        if pathogenic_probability >= self.threshold:
            prediction_class = 1
            prediction_label = "Pathogenic / Likely pathogenic"
            risk_level = "Elevated"
        else:
            prediction_class = 0
            prediction_label = "Benign / Likely benign"
            risk_level = "Lower"

        return PredictionResult(
            prediction_class=prediction_class,
            prediction_label=prediction_label,
            risk_level=risk_level,
            benign_probability=benign_probability,
            pathogenic_probability=pathogenic_probability,
            threshold=float(self.threshold),
            model_name=self.model_name,
            sequence_length_used=len(model_sequence),
            disclaimer=DISCLAIMER,
        )
