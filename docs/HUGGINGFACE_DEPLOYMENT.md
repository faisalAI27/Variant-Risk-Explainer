# Hugging Face Spaces Deployment

Variant Risk Explainer is prepared for a Hugging Face Space using the Docker SDK.
The Space exposes one public port, `7860`.

## Deployment Architecture

```text
User browser
  |
  v
Hugging Face Space
  |
  v
FastAPI on port 7860
  |-- serves the exported Next.js frontend
  |-- exposes /api/health and /api/analyze
  v
DNABERT-2 prediction service
  |
  v
OpenAI or rule-based explanation layer
```

The Docker build exports the Next.js application to `frontend/out/`, copies it
to `backend/static/`, and serves it from FastAPI. The static directories are
generated during the Docker build and are not committed.

## Create The Space

1. Create a new Hugging Face Space.
2. Select **Docker** as the SDK.
3. Keep the public app port as `7860`.
4. Push this repository to the Space.
5. Configure the model and environment settings described below.

## Model Choice 1: Store The Model In The Space

Place the complete exported model folder at:

```text
models/final_model/
```

Set:

```bash
MODEL_DIR=./models/final_model
```

Required model files include:

- `config.json`
- `configuration_bert.py`
- `bert_layers.py`
- `bert_padding.py`
- `flash_attn_triton.py` if the exported model code requires it
- `model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- special-token files or other tokenizer configuration files
- any other Python file imported by the custom DNABERT-2 model code

Model weight extensions are ignored by the project `.gitignore`. If you
intentionally store weights in the Space repository, use Git LFS and explicitly
add the files:

```bash
git lfs install
git lfs track "*.safetensors"
git add .gitattributes
git add -f models/final_model/
```

This option makes the Space repository large. The separate model repository
approach below is cleaner.

## Model Choice 2: Separate Hugging Face Model Repository

Create a Hugging Face model repository and upload the contents of:

```text
training/training_model_files/
```

For example:

```bash
hf upload your-username/variant-risk-dnabert2-20k \
  training/training_model_files \
  . \
  --repo-type model
```

You can also upload the files through the Hugging Face model repository web
interface.

Set the Space variable:

```bash
MODEL_DIR=your-username/variant-risk-dnabert2-20k
```

Transformers loads the tokenizer and model directly from this repository using
`trust_remote_code=True`.

If the model repository is private, create a read token and add it to the Space
as the `HF_TOKEN` secret.

## Space Variables

Add these under **Space Settings -> Variables**:

```text
MODEL_DIR=your-username/variant-risk-dnabert2-20k
MODEL_THRESHOLD=0.16
MODEL_MAX_LENGTH=512
MODEL_NAME=DNABERT-2 ClinVar 20k
DEVICE=auto
USE_AI_EXPLANATION=true
```

For a model stored inside the Space, use:

```text
MODEL_DIR=./models/final_model
```

## Space Secrets

Add secrets under **Space Settings -> Secrets**:

```text
OPENAI_API_KEY=your_real_key
HF_TOKEN=only_needed_if_model_repo_is_private
```

Never put `OPENAI_API_KEY` or a private `HF_TOKEN` in the README, source code,
Dockerfile, or committed environment files.

If `OPENAI_API_KEY` is missing or the OpenAI request fails, the backend returns
the rule-based explanation and reports `rule-based-fallback` as the explanation
source.

## Health Check

After deployment, open:

```text
https://your-space-name.hf.space/api/health
```

The response includes:

- service status
- whether the model loaded
- configured model source
- selected device
- decision threshold
- explanation mode
- whether AI explanation is enabled
- whether an OpenAI key is configured
- model load error, when present

The Space still starts and serves the frontend if the model cannot load.
`POST /api/analyze` then returns:

```text
Model is not available. Please configure MODEL_DIR correctly.
```

## Local Docker Test

Place a model at `models/final_model/`, then run:

```bash
docker build -t variant-risk-explainer .
docker run -p 7860:7860 \
  -e MODEL_DIR=./models/final_model \
  -e MODEL_THRESHOLD=0.16 \
  -e USE_AI_EXPLANATION=false \
  variant-risk-explainer
```

Open:

```text
http://localhost:7860
```

Health endpoint:

```text
http://localhost:7860/api/health
```

## Push To The Space

After creating the Space:

```bash
git remote add space https://huggingface.co/spaces/your-username/variant-risk-explainer
git push space main
```

If your local default branch is not `main`, push it explicitly:

```bash
git push space HEAD:main
```

Do not push local `.env` files, datasets, `training/outputs/`, or
`training/csv_files_20k_alt/`.
