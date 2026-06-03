from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.model import DISCLAIMER, LIMITATIONS, build_model
from app.schemas import HealthResponse, VariantAnalysisResponse, VariantRequest


settings = get_settings()
model_runner = build_model(settings)

app = FastAPI(
    title="Variant Risk Explainer API",
    version="0.1.0",
    description="Research-only FastAPI backend for GRCh38 variant risk exploration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, model_mode=model_runner.mode)


@app.post("/analyze", response_model=VariantAnalysisResponse)
def analyze_variant(request: VariantRequest) -> VariantAnalysisResponse:
    if request.sequence_context and len(request.sequence_context) > settings.max_sequence_context_length:
        raise HTTPException(
            status_code=422,
            detail=f"sequence_context must be {settings.max_sequence_context_length} bases or fewer",
        )

    try:
        prediction = model_runner.analyze(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return VariantAnalysisResponse(
        request_id=str(uuid4()),
        submitted_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        input=request,
        risk_label=prediction.risk_label,
        confidence=prediction.confidence,
        model_mode=prediction.model_mode,
        explanation=prediction.explanation,
        limitations=LIMITATIONS,
        disclaimer=DISCLAIMER,
    )
