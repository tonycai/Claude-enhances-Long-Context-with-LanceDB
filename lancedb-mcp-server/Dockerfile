# ============================================================================
# Multi-stage Dockerfile for lancedb-code MCP server
# ============================================================================
# Stage 1 (builder): installs uv, syncs all deps (CPU-only torch), prefetches
#                     the embedding model into the HuggingFace cache.
# Stage 2 (runtime): slim Python image + git, copies .venv and HF cache from
#                     builder. Runs the MCP server over stdio transport.
#
# Embedding model (multilingual by default — Chinese, Japanese, English):
#   docker compose build                                   # multilingual-e5-base
#   EMBEDDING_MODEL=intfloat/multilingual-e5-small docker compose build
#   EMBEDDING_MODEL=BAAI/bge-m3 docker compose build       # best, largest
#   EMBEDDING_MODEL=all-MiniLM-L6-v2 docker compose build  # old English model
# ============================================================================

ARG EMBEDDING_MODEL=intfloat/multilingual-e5-base

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
# The model is cached under /root/.cache/huggingface. The build arg is used
# only from here on, so switching models keeps the dependency layers cached.
ARG EMBEDDING_MODEL
ENV LANCEDB_EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN uv run python scripts/prefetch_model.py

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ARG EMBEDDING_MODEL
ENV DEBIAN_FRONTEND=noninteractive

# git is needed for .gitignore-aware file discovery.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (handy for debugging inside the container; the entrypoint
# below runs the venv Python directly).
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

# Multilingual defaults: the model baked in above, n-gram full-text search
# (works for Chinese / Japanese, which have no spaces between words), and no
# network access to Hugging Face at runtime.
ENV LANCEDB_EMBEDDING_MODEL=${EMBEDDING_MODEL} \
    LANCEDB_FTS_TOKENIZER=ngram \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# The MCP server communicates over stdio — no ports to expose.
# Use `docker run -i` (interactive, no TTY) to preserve the JSON-RPC stream.
# Run the venv's Python directly: `uv run` would check the lock file on every
# start, which is slower and can try to reach the network.
ENTRYPOINT ["/app/.venv/bin/python", "server.py"]
