# Testing Checklist

Use this checklist before pushing or presenting the demo.

## Backend

- [ ] `GET /health` returns `status: ok`.
- [ ] `GET /health` shows the expected model directory and threshold `0.16`.
- [ ] `POST /analyze` works for a valid synthetic DNA sequence.
- [ ] `POST /analyze` returns prediction probabilities and `prediction_class`.
- [ ] Invalid DNA sequence returns a clear HTTP `400` error.
- [ ] Response includes `explanation`.
- [ ] Response includes `explanation_source`.
- [ ] Response includes `confidence_level`, `recommendation`, `limitations`, and `disclaimer`.
- [ ] OpenAI explanation works when `USE_AI_EXPLANATION=true` and `OPENAI_API_KEY` is set.
- [ ] OpenAI fallback works if the key is missing or the OpenAI request fails.

## Frontend

- [ ] Frontend loads at `http://localhost:3000`.
- [ ] Backend status indicator shows connected/model loaded.
- [ ] Form accepts variant name, gene, sequence, and notes.
- [ ] Result card displays prediction label and risk level.
- [ ] Result card displays pathogenic and benign probabilities.
- [ ] Result card displays explanation and explanation source.
- [ ] Result card displays limitations and research/demo disclaimer.
- [ ] History panel records recent demo analyses.

## Security And Artifacts

- [ ] No `.env` files are tracked by Git.
- [ ] No `.env.local` files are tracked by Git.
- [ ] No OpenAI API key appears in tracked source, README, or docs files.
- [ ] Trained model folders are ignored by Git.
- [ ] Dataset folders are ignored by Git.
- [ ] Model weight files such as `.safetensors`, `.bin`, `.pt`, and `.ckpt` are ignored by Git.
