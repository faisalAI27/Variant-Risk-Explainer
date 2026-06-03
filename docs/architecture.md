# Architecture

Variant Risk Explainer is split into four top-level folders:

```text
training/  -> Colab training pipeline
backend/   -> FastAPI inference API
frontend/  -> Next.js research demo UI
docs/      -> project documentation
```

## Component Flow

```text
User
  |
  v
Next.js frontend
  |
  | POST /analyze
  v
FastAPI backend
  |
  | mock mode or trained DNABERT-2 model directory
  v
Variant risk response
```

## Training Flow

1. Download ClinVar GRCh38 VCF data in Google Colab.
2. Load a GRCh38 reference FASTA.
3. Extract sequence windows around single nucleotide variants.
4. Map ClinVar clinical significance labels into research classes.
5. Fine-tune DNABERT-2 with Hugging Face Transformers.
6. Save an exported model directory for backend inference.

Training is intentionally not wired into the local runtime. The backend can run in mock mode until a trained model is available.

## Backend Flow

The backend exposes `POST /analyze`. It validates the submitted GRCh38 variant, chooses a model implementation, returns a risk label, confidence, explanation, and research-only disclaimers.

Model mode is controlled by environment variables:

- `MODEL_MODE=mock`: always use deterministic mock inference.
- `MODEL_MODE=trained`: require a trained model at `MODEL_DIR`.
- `MODEL_MODE=auto`: use the trained model when available, otherwise fall back to mock mode.

## Frontend Flow

The frontend provides:

- Variant input form.
- Loading and error states.
- Result card.
- Local in-browser history panel.

The frontend uses `NEXT_PUBLIC_API_BASE_URL` to find the backend.

## Safety Boundary

This is a research and educational demo. It does not diagnose disease, recommend care, or replace genetic counseling or clinical interpretation. All user-facing layers should preserve that boundary.
