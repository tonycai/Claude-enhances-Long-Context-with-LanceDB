# ============================================================================
# Multi-stage Dockerfile for lancedb-code MCP server
# ============================================================================
# Stage 1 (builder): installs uv, syncs all deps (CPU-only torch), prefetches
#                     the embedding model into the HuggingFace cache.
# Stage 2 (runtime): slim Python image + git, copies .venv and HF cache from
#                     builder. Runs the MCP server over stdio transport.
# ============================================================================

# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Avoid interactive prompts during package installs.
ENV DEBIAN_FRONTEND=noninteractive

# Install build-time system dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy dependency manifests first for better layer caching.
COPY pyproject.toml uv.lock* ./

# Sync all dependencies (including optional language grammars).
# UV_TORCH_BACKEND=cpu avoids pulling ~2.4 GB of CUDA libraries.
ENV UV_TORCH_BACKEND=cpu
RUN uv sync --extra all-languages --no-dev

# Copy application source.
COPY . .

# Pre-download the embedding model so it is baked into the image.
# The model is cached under /root/.cache/huggingface.
RUN uv run python scripts/prefetch_model.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# git is needed for .gitignore-aware file discovery.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (needed for `uv run` in the entrypoint).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy virtual environment and application source from builder.
COPY --from=builder /app /app

# Copy the pre-downloaded HuggingFace model cache from builder.
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Default environment: index data lives in a mountable volume,
# workspace is the bind-mounted source repo.
ENV LANCEDB_PATH=/data/lancedb \
    LANCEDB_REPO_ROOT=/workspace

# The MCP server communicates over stdio — no ports to expose.
# Use `docker run -i` (interactive, no TTY) to preserve the JSON-RPC stream.
ENTRYPOINT ["uv", "run", "python", "server.py"]
