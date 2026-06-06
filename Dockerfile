FROM node:20-bookworm-slim AS frontend-builder

WORKDIR /frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
ENV NEXT_PUBLIC_API_URL=""
RUN npm run build


FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEVICE=auto \
    HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /app

COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r /tmp/requirements.txt

COPY backend/app /app/backend/app
COPY models /app/models
COPY --from=frontend-builder /frontend/out /app/backend/static

RUN useradd --create-home --uid 1000 user \
    && chown -R user:user /app /home/user

USER user

WORKDIR /app/backend

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
