from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    app_name: str = "variant-risk-explainer"
    model_mode: str = os.getenv("MODEL_MODE", "auto").strip().lower()
    model_dir: str | None = os.getenv("MODEL_DIR") or None
    allowed_origins: tuple[str, ...] = tuple(
        origin.strip()
        for origin in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    )
    max_sequence_context_length: int = int(os.getenv("MAX_SEQUENCE_CONTEXT_LENGTH", "2000"))

    def validate(self) -> "Settings":
        if self.model_mode not in {"auto", "mock", "trained"}:
            raise ValueError("MODEL_MODE must be one of: auto, mock, trained")
        if self.model_dir and not Path(self.model_dir).expanduser().exists():
            if self.model_mode == "trained":
                raise ValueError(f"MODEL_DIR does not exist: {self.model_dir}")
        return self


def get_settings() -> Settings:
    return Settings().validate()
