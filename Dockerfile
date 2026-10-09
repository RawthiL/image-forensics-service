# One Dockerfile, four targets, differing only in which runtime extra is synced.
#
#   Triton client:     docker build --target triton  -t ifs:triton .
#   ONNX (lean, CPU):  docker build --target onnx    -t ifs:onnx .
#   ONNX (lean, CUDA): docker build --target onnx-gpu -t ifs:onnx-gpu .
#   torch CPU:         docker build --target cpu     -t ifs:cpu .
#   torch CUDA:        docker build --target gpu     -t ifs:gpu .
#
# The onnx targets carry no torch at all: ~173 MB of dependencies against
# ~1.1 GB (torch CPU) and ~5.5 GB (torch CUDA), serving the identical model
# with scores that agree to 1e-6. Prefer them unless you need the torch
# runtime. The `triton` runtime needs no extra at all -- inference is remote --
# so the onnx image serves it too.
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
    HF_HUB_CACHE=/weights/hub \
    IFS_CONFIG_FILE=/etc/ifs/config.yaml
WORKDIR /app
# Config lives OUTSIDE /app on purpose. The virtualenv is at /app/.venv, so a
# volume mounted at /app -- which is what a Kubernetes ConfigMap does unless you
# remember `subPath` -- shadows it and the entrypoint dies with
# `exec: "uvicorn": executable file not found in $PATH`.
# Mounting a directory at /etc/ifs has no such hazard, and unlike a subPath
# mount a ConfigMap mounted this way still receives updates.
RUN mkdir -p /etc/ifs
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# ---------- torch, CPU ----------
FROM base AS cpu
# The +cpu wheel is ~200MB against ~2.5GB for the CUDA build.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --extra community-forensics
ENV IFS_RUNTIME__DEVICE=cpu IFS_RUNTIME__DTYPE=float32
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- Triton client (leanest: no local model runtime at all) ----------
FROM base AS triton
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra triton
# device/dtype are meaningless here: inference happens on the Triton server.
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- ONNX, CPU (the lean default) ----------
FROM base AS onnx
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra onnx-cpu
ENV IFS_RUNTIME__DEVICE=cpu IFS_RUNTIME__DTYPE=float32
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- ONNX, CUDA ----------
FROM base AS onnx-gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra onnx-gpu
ENV IFS_RUNTIME__DEVICE=cuda IFS_RUNTIME__DTYPE=float32
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------- torch, CUDA ----------
FROM base AS gpu
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cu124 --extra community-forensics
ENV IFS_RUNTIME__DEVICE=cuda IFS_RUNTIME__DTYPE=float16
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s \
  CMD curl -fsS http://localhost:8000/readyz || exit 1
CMD ["uvicorn", "ifs.main:app", "--host", "0.0.0.0", "--port", "8000"]
