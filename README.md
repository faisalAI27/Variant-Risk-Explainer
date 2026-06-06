# Variant Risk Explainer

Variant Risk Explainer is a full-stack AI-powered genomic variant analysis system. It uses a fine-tuned DNABERT-2 model to estimate whether a submitted DNA sequence looks more similar to benign/likely benign or pathogenic/likely pathogenic ClinVar examples.

This project is for AI/ML research and education only. It is not a medical device, not a diagnostic system, and must not be used for clinical decisions.

## Project Overview

- `training/`: ClinVar GRCh38 data preparation, DNABERT-2 training notebooks, local evaluation scripts.
- `backend/`: FastAPI inference API with DNABERT-2 prediction and explanation services.
- `frontend/`: Next.js analysis interface with input form, service status, result card, explanation, and history.
- `docs/`: Architecture notes, API contract, model card, demo examples, limitations, and testing checklist.

## Architecture

```text
User
  ↓
Next.js Frontend
  ↓
FastAPI Backend
  ↓
DNABERT-2 Prediction Service
  ↓
Explanation Layer
  ↓
AI-Assisted Result
```

The frontend sends a DNA sequence to `POST /analyze`. The backend cleans and crops the sequence, runs the DNABERT-2 classifier, applies the tuned threshold, then returns probabilities, a research-only label, and a cautious explanation.

## Model Training Summary

- Base model: DNABERT-2
- Dataset: 20k ClinVar alternate-sequence dataset
- Genome build: GRCh38
- Task: binary research classification
- Label `0`: Benign / Likely benign
- Label `1`: Pathogenic / Likely pathogenic
- Decision threshold: `0.16`

## Final Metrics

| Metric | Value |
| --- | ---: |
| Accuracy | 0.5537 |
| Precision | 0.5384 |
| Recall | 0.7533 |
| F1 | 0.6280 |
| MCC | 0.1171 |
| AUC ROC | 0.5928 |

These metrics are limited and support educational and research-oriented analysis only, not clinical interpretation.

## Run Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload
```

Open `http://localhost:8000/docs`.

## Run Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Environment Variables

Backend values live in `backend/.env`:

```bash
MODEL_DIR=../training/training_model_files
MODEL_THRESHOLD=0.16
MODEL_MAX_LENGTH=512
MODEL_NAME=DNABERT-2 ClinVar 20k
DEVICE=auto
OPENAI_API_KEY=your_openai_api_key_here
USE_AI_EXPLANATION=true
```

Frontend values live in `frontend/.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

Never commit `.env`, `.env.local`, API keys, datasets, or model weights.

## Data and Model Artifact Policy

Large files are intentionally ignored by Git:

- trained model folders such as `training/training_model_files/`
- generated datasets such as `training/csv_files_20k_alt/`
- model weight files such as `.safetensors`, `.bin`, `.pt`, and `.ckpt`
- local environment files such as `.env` and `.env.local`

Use local storage, Google Drive, or another private artifact store for trained models and datasets.

## Responsible Use

Predictions and explanations are experimental model outputs. They can be wrong, incomplete, biased by ClinVar labels, or invalid outside the training distribution. This project is intended for educational and research-oriented AI/ML analysis and must not be used for diagnosis, treatment, or medical decision-making.
