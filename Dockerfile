# syntax=docker/dockerfile:1
# Build natively on each host: arm64 on the Mac (dev), amd64 on Ubuntu (prod).
# Do NOT cross-build with QEMU — torch/onnxruntime are flaky and very slow
# under emulation. On the server: git pull && docker compose build.
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf
WORKDIR /srv/app

# ML deps first: the torch layer is huge and changes rarely, so bumps to the
# core requirements must not invalidate it.
COPY requirements-ml.txt .
RUN pip install -r requirements-ml.txt

# Bake the embedding model into the image so runtime is fully offline.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN useradd --create-home --uid 1000 app
COPY app ./app
COPY README.md .
USER app
EXPOSE 8090

FROM base AS dev
USER root
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
USER app

FROM base AS prod
# The SSE bus, run registry, and job queue are in-process state:
# this MUST stay a single uvicorn worker. Never add --workers.
CMD ["uvicorn", "app.web.server:create_app", "--factory", "--host", "0.0.0.0", "--port", "8090"]
