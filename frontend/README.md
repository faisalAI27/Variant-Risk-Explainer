# Frontend

This folder contains the Next.js frontend for Variant Risk Explainer.

The app provides a GRCh38 variant form, loading state, result card, and local history panel.

## Setup

```bash
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Environment

Set the backend URL in `.env.local`:

```bash
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

## Safety Notice

The UI displays research-only disclaimers from the backend. This frontend must not present model output as medical advice or diagnosis.
