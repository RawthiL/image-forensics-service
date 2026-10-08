# One Dockerfile, two targets. They differ only in which torch extra is synced.
#
#   CPU:  docker build --target cpu -t ifs:cpu .
#   GPU:  docker build --target gpu -t ifs:gpu .
#
# No config.yaml is baked in: the service must be given one (compose mounts it),
# and an absent config fails loudly rather than resolving to some default model.

# ---------- shared base ----------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS base
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_CACHE=/weights/hub
WORKDIR /app
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# ---------- CPU ----------
FROM base AS cpu
# The +cpu wheel is ~200MB against ~2.5GB for the CUDA build.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --extra community-forensics
ENV IFS_RUNTIME__DEVICE=cpu IFS_RUNTIME__DTYPE=float32
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- GPU ----------
FROM base AS gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cu124 --extra community-forensics
ENV IFS_RUNTIME__DEVICE=cuda IFS_RUNTIME__DTYPE=float16
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]
