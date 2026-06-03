# Variant Risk Explainer

Variant Risk Explainer is a full-stack AI genomics research demo for exploring how a sequence model might explain variant risk signals from ClinVar-style data.

This repository is for research and education only. It is not a medical device, not a diagnostic tool, and must not be used to make clinical decisions.

## Repository Structure

```text
training/
backend/
frontend/
docs/
```

- `training/`: Google Colab-oriented ClinVar + DNABERT-2 fine-tuning pipeline.
- `backend/`: FastAPI service exposing `POST /analyze`.
- `frontend/`: Next.js interface with variant input, loading state, result card, and history.
- `docs/`: Architecture, API contract, and model card.

## Prerequisites

- Python 3.11+
- Node.js 20+
- Google Colab for model training
- GRCh38 reference assets for training data preparation

## Backend Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

The backend defaults to mock model mode when no trained model directory is configured.

Run tests:

```bash
cd backend
pytest
```

## Frontend Quick Start

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Training Quick Start

Training is intended for Google Colab only. Do not train the model locally.

1. Open `training/colab_dnabert2_clinvar_finetune.ipynb` in Google Colab.
2. Follow the notebook cells to install dependencies, download ClinVar GRCh38 data, prepare examples, and fine-tune DNABERT-2.
3. Export the trained Hugging Face model directory.
4. Point the backend `MODEL_DIR` environment variable to that exported model directory.

## Environment Files

Each app folder includes a `.env.example`. Do not commit `.env`, `.env.local`, API keys, model checkpoints, or private data.

## Safety Notice

Predictions and explanations are experimental model outputs. They can be wrong, incomplete, biased by the training data, or invalid for variants outside the training distribution. Use this project only for software, ML, and genomics workflow demonstrations.
