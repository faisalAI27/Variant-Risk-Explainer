# Architecture

Variant Risk Explainer is split into four top-level folders:

```text
training/  -> ClinVar GRCh38 preparation, DNABERT-2 training, evaluation scripts
backend/   -> FastAPI inference API
frontend/  -> Next.js research demo UI
docs/      -> project documentation
```

## Runtime Flow

```text
User
  |
  v
Next.js Frontend
  |
  | POST /analyze
  v
FastAPI Backend
  |
  v
DNABERT-2 Prediction Service
  |
  v
Explanation Layer
  |
  v
Research/Demo Result
```

## Backend Flow

1. Load settings from `backend/.env`.
2. Load the DNABERT-2 tokenizer and sequence-classification model from `MODEL_DIR`.
3. Select device automatically: CUDA, then MPS, then CPU.
4. Clean the submitted DNA sequence.
5. Center crop sequences longer than `MODEL_MAX_LENGTH`.
6. Run DNABERT-2 in inference mode.
7. Apply the tuned pathogenic threshold, currently `0.16`.
8. Generate a cautious explanation.
9. Return prediction probabilities, label, explanation, limitations, and disclaimer.

## Explanation Layer

The explanation layer is designed to be safe for a research demo:

- `rule-based`: local deterministic explanation.
- `openai`: optional OpenAI-generated explanation paragraph.
- `rule-based-fallback`: local explanation used because AI explanation failed or was missing configuration.

The LLM, when enabled, rewrites only the explanation paragraph. It does not control the prediction, probabilities, threshold, confidence level, recommendation, limitations, or disclaimer.

## Frontend Flow

The frontend provides:

- backend health indicator
- DNA sequence input form
- loading and error states
- result card
- explanation source display
- local in-browser history panel
- research/demo disclaimer

## Training Flow

1. Prepare ClinVar GRCh38 records.
2. Extract reference sequence windows.
3. Build alternate-allele sequence windows.
4. Filter uncertain/conflicting labels.
5. Fine-tune DNABERT-2 on the alternate-sequence dataset.
6. Evaluate on held-out validation and test splits.
7. Export a self-contained Hugging Face model folder for backend inference.

## Safety Boundary

This is a research and educational demo. It does not diagnose disease, recommend care, or replace genetic counseling or clinical variant interpretation. All user-facing layers should preserve that boundary.
