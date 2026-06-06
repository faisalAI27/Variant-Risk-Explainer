from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_model_dir() -> str:
    return str(_repo_root() / "training" / "training_model_files")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ai_explanation_enabled() -> bool:
    if os.getenv("USE_AI_EXPLANATION") is not None:
        return _env_bool("USE_AI_EXPLANATION", False)
    return _env_bool("USE_OPENAI_EXPLANATION", False)


@dataclass(frozen=True)
class Settings:
    app_name: str = "variant-risk-explainer"
    model_mode: str = os.getenv("MODEL_MODE", "auto").strip().lower()
    model_dir: str = os.getenv("MODEL_DIR", _default_model_dir()).strip() or _default_model_dir()
    model_threshold: float = float(os.getenv("MODEL_THRESHOLD", "0.16"))
    model_max_length: int = int(os.getenv("MODEL_MAX_LENGTH", "512"))
    model_name: str = os.getenv("MODEL_NAME", "DNABERT-2 ClinVar 20k").strip() or "DNABERT-2 ClinVar 20k"
    device: str = os.getenv("DEVICE", "auto").strip().lower() or "auto"
    max_sequence_context_length: int = int(os.getenv("MAX_SEQUENCE_CONTEXT_LENGTH", "2000"))
    use_openai_explanation: bool = _ai_explanation_enabled()
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
    openai_explanation_model: str = (
        os.getenv("OPENAI_EXPLANATION_MODEL", "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    )
    openai_explanation_timeout: float = float(os.getenv("OPENAI_EXPLANATION_TIMEOUT", "12"))
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    )

    def resolved_model_source(self) -> str:
        raw_source = self.model_dir.strip()
        model_path = Path(raw_source).expanduser()
        if model_path.is_absolute():
            return str(model_path)

        candidates = [
            (_repo_root() / model_path).resolve(),
            (Path(__file__).resolve().parents[2] / model_path).resolve(),
            (Path.cwd() / model_path).resolve(),
        ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        if raw_source.startswith(("./", "../", "~")):
            return str(candidates[0])

        # Non-local values such as username/model-repo are Hugging Face repo IDs.
        return raw_source

    def validate(self) -> "Settings":
        if self.model_mode not in {"auto", "mock", "trained"}:
            raise ValueError("MODEL_MODE must be one of: auto, mock, trained")
        if not 0.0 <= self.model_threshold <= 1.0:
            raise ValueError("MODEL_THRESHOLD must be between 0 and 1")
        if self.model_max_length <= 0:
            raise ValueError("MODEL_MAX_LENGTH must be positive")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("DEVICE must be one of: auto, cuda, mps, cpu")
        if self.openai_explanation_timeout <= 0:
            raise ValueError("OPENAI_EXPLANATION_TIMEOUT must be positive")
        return self


def get_settings() -> Settings:
    return Settings().validate()
