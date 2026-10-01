# syntax=docker/dockerfile:1.7
#
# Build args:
#   INSTALL_ML=false|true     install the `ml` extra (torch, pyannote.audio, faster-whisper)
#   INSTALL_INDIC=false|true  also install the `indic` extra (Odia ASR: transformers, onnxruntime)
#   TORCH_VARIANT=cpu         PyTorch wheel variant: cpu (small image) or cu124 / cu128 (CUDA;
#                             cu128 is required for RTX 50xx / Blackwell GPUs). The CUDA
#                             runtime ships inside the torch wheels, so the base stays slim.
#                             (run the GPU image with `--gpus all` on a host with the NVIDIA runtime)
#
#   docker build -f docker/Dockerfile -t polymom .                                   # light, no ML
#   docker build -f docker/Dockerfile --build-arg INSTALL_ML=true -t polymom:ml .    # CPU torch
#   docker build -f docker/Dockerfile --build-arg INSTALL_ML=true \
#       --build-arg INSTALL_INDIC=true -t polymom:ml-indic .                         # + Odia ASR
#   docker build -f docker/Dockerfile --build-arg INSTALL_ML=true \
#       --build-arg TORCH_VARIANT=cu128 -t polymom:cuda .                            # CUDA torch
#
# Base images are pinned by digest; Dependabot (docker ecosystem) proposes updates.

# ---------- builder ----------
FROM python:3.11.16-slim-trixie@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e AS builder
ARG INSTALL_ML=false
ARG INSTALL_INDIC=false
ARG TORCH_VARIANT=cpu
COPY --from=ghcr.io/astral-sh/uv:0.9.30@sha256:538e0b39736e7feae937a65983e49d2ab75e1559d35041f9878b7b7e51de91e4 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv UV_HTTP_TIMEOUT=300
WORKDIR /build
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    EXTRAS="--extra postgres --extra s3"; \
    if [ "$INSTALL_ML" = "true" ]; then EXTRAS="$EXTRAS --extra ml"; fi; \
    if [ "$INSTALL_INDIC" = "true" ]; then EXTRAS="$EXTRAS --extra indic"; fi; \
    uv sync --frozen --no-dev --no-install-project $EXTRAS
# Swap the locked CPU torch wheels for a CUDA build of the same version.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$INSTALL_ML" = "true" ] && [ "$TORCH_VARIANT" != "cpu" ]; then \
        PINS=$(uv export --frozen --no-dev --extra ml --no-hashes \
               | grep -E '^torch(audio)?==' | sed 's/+[^ ;]*//; s/ *;.*//'); \
        uv pip install --python /opt/venv/bin/python --reinstall $PINS \
            --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}"; \
    fi
COPY app ./app
# For CUDA builds keep the swapped torch wheels and their nvidia-* dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    FLAGS="--extra postgres --extra s3"; \
    if [ "$INSTALL_INDIC" = "true" ]; then FLAGS="$FLAGS --extra indic"; fi; \
    if [ "$INSTALL_ML" = "true" ]; then FLAGS="$FLAGS --extra ml"; \
        if [ "$TORCH_VARIANT" != "cpu" ]; then \
            FLAGS="$FLAGS --inexact --no-install-package torch --no-install-package torchaudio"; \
        fi; \
    fi; \
    uv sync --frozen --no-dev $FLAGS

# ---------- runtime ----------
FROM python:3.11.16-slim-trixie@sha256:e41613d42d4891e4930f79523f93f81bbc7632584ec65e36ab055f41a800b41e AS runtime
ARG VERSION=dev
LABEL org.opencontainers.image.title="polymom" \
      org.opencontainers.image.description="Multilingual minutes-of-meeting pipeline" \
      org.opencontainers.image.source="https://github.com/PrayasPanda/polymom" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${VERSION}"
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app && useradd --system --gid app --home /app app
# HF_HOME holds downloaded models (pyannote, Whisper, IndicConformer); mount a volume there.
ENV PATH="/opt/venv/bin:$PATH" PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    APP_ENV=production STORAGE_DIR=/app/storage \
    HF_HOME=/app/.cache/huggingface TORCH_HOME=/app/.cache/torch \
    MPLCONFIGDIR=/app/.cache/matplotlib
WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app app ./app
# Operational scripts (API keys, retention cleanup) run with `docker compose exec`.
COPY --chown=app:app scripts ./scripts
RUN mkdir -p /app/storage /app/.cache/huggingface /app/.cache/torch \
    && chown -R app:app /app/storage /app/.cache
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD curl -fsS http://localhost:8000/api/v1/health/live || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
