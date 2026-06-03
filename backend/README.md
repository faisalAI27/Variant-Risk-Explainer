# Backend

This folder contains the FastAPI backend for Variant Risk Explainer.

The API exposes `POST /analyze` and returns a research-only variant risk explanation. It runs in mock mode by default when no trained model is available.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for the interactive API docs.

## Model Modes

- `MODEL_MODE=mock`: deterministic mock responses for development.
- `MODEL_MODE=auto`: use `MODEL_DIR` when available, otherwise mock mode.
- `MODEL_MODE=trained`: require a trained Hugging Face model directory at `MODEL_DIR`.

For trained mode, install optional model dependencies:

```bash
pip install -r requirements-model.txt
```

## Tests

```bash
pytest
```

## Safety Notice

The backend response includes a research-only disclaimer. Do not remove it from user-facing responses.
