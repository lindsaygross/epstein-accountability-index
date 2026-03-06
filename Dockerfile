# Attribution: Scaffolded with AI assistance (Claude, Anthropic)
# Multi-stage build for The Impunity Index — Google Cloud Run deployment
#
# Build: docker build -t impunity-index .
# Run locally: docker run -p 8080:8080 -e PORT=8080 impunity-index
# Deploy: gcloud run deploy impunity-index --source . --region us-central1

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FLASK_ENV=production \
    PORT=8080

WORKDIR /app

# Install system dependencies (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (use app/requirements.txt — no training deps)
COPY app/requirements.txt ./app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r app/requirements.txt

# Pre-download sentence-transformer model weights at build time
# so there's no cold-start download delay in production
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application code
COPY app/ ./app/

# Copy trained models (pkl files)
COPY models/ ./models/

# Copy processed data files needed at runtime
COPY data/processed/ ./data/processed/
COPY data/outputs/ ./data/outputs/

# Copy ChromaDB vector store (citations endpoint)
# If chroma_db/ is large, you can remove this line and citations will return empty
COPY chroma_db/ ./chroma_db/

# Expose port (Cloud Run injects PORT env var)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/people | python3 -c "import json,sys; d=json.load(sys.stdin); exit(0 if len(d)>0 else 1)" || exit 1

# Run with gunicorn (production WSGI server)
# --timeout 120: allow slow first requests (ChromaDB cold load)
# --workers 1: single worker to share loaded models in memory
# --threads 4: handle concurrent requests within one worker
CMD exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers 1 \
    --threads 4 \
    --timeout 120 \
    --access-logfile - \
    --error-logfile - \
    "app.main:app"
