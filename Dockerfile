# ── Production Dockerfile for RegBrain FastAPI Backend ──
FROM python:3.11-slim

# Set environment variables for Python runtime optimization & memory efficiency
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

WORKDIR /app

# Install curl for health check support and clean up apt cache
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first to avoid multi-gigabyte CUDA bloat, then install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and artifacts
COPY api/ /app/api/
COPY retrieval/ /app/retrieval/
COPY generation/ /app/generation/
COPY query_planner/ /app/query_planner/
COPY ingestion/ /app/ingestion/

# Create logs directory for the structured JSON rotating file handler
RUN mkdir -p /app/logs

# Expose default backend port
EXPOSE 8000

# Health check to ensure container is responsive
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Launch FastAPI app with Uvicorn (single worker to conserve RAM within 512MB limits)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
