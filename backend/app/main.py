from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse
from app.services.explanation_service import generate_explanation
from app.services.model_service import ModelService


settings = get_settings()
model_service = ModelService(settings)

app = FastAPI(
    title="Variant Risk Explainer API",
    version="0.2.0",
    description="Research-only FastAPI backend for DNABERT-2 ClinVar variant risk exploration.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Variant Risk Explainer API",
        "model": settings.model_name,
        "disclaimer": "Research/demo use only. Not for clinical diagnosis.",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if model_service.model_loaded else "degraded",
        model_loaded=model_service.model_loaded,
        device=model_service.device,
        model_dir=str(model_service.model_dir),
        threshold=model_service.threshold,
        model_name=model_service.model_name,
        load_error=model_service.load_error,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    try:
        prediction = model_service.predict(request.sequence)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary for inference failures.
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}") from exc

    explanation = generate_explanation(
        prediction_class=prediction.prediction_class,
        prediction_label=prediction.prediction_label,
        risk_level=prediction.risk_level,
        benign_probability=prediction.benign_probability,
        pathogenic_probability=prediction.pathogenic_probability,
        threshold=prediction.threshold,
        variant_name=request.variant_name,
        gene=request.gene,
        sequence_length_used=prediction.sequence_length_used,
        use_openai=settings.use_openai_explanation,
        openai_api_key=settings.openai_api_key,
        openai_model=settings.openai_explanation_model,
        openai_timeout=settings.openai_explanation_timeout,
    )

    return AnalyzeResponse(
        variant_name=request.variant_name,
        gene=request.gene,
        prediction_class=prediction.prediction_class,
        prediction_label=prediction.prediction_label,
        risk_level=prediction.risk_level,
        benign_probability=prediction.benign_probability,
        pathogenic_probability=prediction.pathogenic_probability,
        threshold=prediction.threshold,
        model_name=prediction.model_name,
        sequence_length_used=prediction.sequence_length_used,
        explanation=explanation["explanation"],
        confidence_level=explanation["confidence_level"],
        recommendation=explanation["recommendation"],
        limitations=explanation["limitations"],
        disclaimer=prediction.disclaimer,
    )
