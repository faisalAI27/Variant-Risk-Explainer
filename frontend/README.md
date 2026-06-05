# Frontend

Next.js frontend for the Variant Risk Explainer research demo.

The app connects to the FastAPI backend, submits a DNA sequence to
`POST /analyze`, shows the DNABERT-2 prediction result, and displays backend
health/model status.

This is for research/demo use only. It is not a clinical diagnostic system.

## Run Backend

From the repository root:

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload
```

If port `8000` is busy, use:

```bash
uvicorn app.main:app --reload --port 8001
```

When using a non-default backend port, update `frontend/.env.local`.

## Run Frontend

From the repository root:

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open:

```text
http://localhost:3000
```

## Environment

Set the backend URL in `.env.local`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

For a backend on port `8001`:

```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8001
```

## Analyze Flow

1. Start FastAPI.
2. Start Next.js.
3. Confirm the page shows `Backend connected`.
4. Enter a DNA sequence containing only `A`, `C`, `G`, `T`, or `N`.
5. Click `Analyze Variant`.

The result card shows benign/pathogenic probabilities, threshold, model name,
and sequence length used after center cropping.

## Safety Notice

The UI displays research-only disclaimers from the backend. Do not present model
output as medical advice or diagnosis.
