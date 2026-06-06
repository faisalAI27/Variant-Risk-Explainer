# Frontend

Next.js frontend for the Variant Risk Explainer AI-powered genomic variant analysis system.

The app connects to the FastAPI backend, submits a DNA sequence to
`POST /analyze`, shows the DNABERT-2 prediction result, and displays backend
health/model status. The result card also displays the backend's rule-based
or optional OpenAI explanation, confidence level, recommendation, and
limitations.

The interface presents model-assisted analysis without making clinical claims.
Responsible-use guidance is placed in the application footer and technical
limitations remain available in the project documentation.

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
5. Click `Analyze Sequence`.

The main result card shows the overall assessment, risk category, benign and
pathogenic likelihoods, confidence level, explanation, and recommendation.
Model name, threshold, sequence length, prediction class, performance metrics,
and explanation source are available in the collapsed `Technical details`
section.

The explanation is generated from the backend model output and threshold. It is
not medical advice and is not a clinical interpretation. If the backend has
`USE_AI_EXPLANATION=true`, the backend may use OpenAI to improve the
explanation paragraph. Do not put the OpenAI API key in the frontend; keep it in
`backend/.env` only.

## Responsible Use

The UI includes a professional notice explaining that AI-assisted interpretation
does not replace clinical genetic testing or professional medical evaluation.
