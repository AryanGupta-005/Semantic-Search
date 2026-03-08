# Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): installs dependencies into a venv
#   Stage 2 (runtime): copies only what's needed — smaller final image
#
# Why python:3.10-slim (not 3.10-full or 3.11):
#   slim removes docs, tests, dev headers — not needed in production.
#   3.10 is the stable LTS version with best library compatibility.

# ── Stage 1: Build ─────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies for packages that compile C extensions
# (faiss-cpu, umap-learn need these)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ───────────────────────────────────────────────────
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY app/ ./app/
COPY pipeline/ ./pipeline/

# Copy pre-built artifacts (FAISS index, GMM model, metadata)
# These must exist before building the image.
# Run the pipeline locally first: ./pipeline/run_pipeline.sh
COPY artifacts/ ./artifacts/

# Expose port 8000
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

# Start server
# host 0.0.0.0 makes it accessible outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
