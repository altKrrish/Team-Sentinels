# ==============================================================================
# Sentinel Hardened Microservice Dockerfile
# Production-ready container image for rig-site edge and cloud deployment
# ==============================================================================

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and trained models
COPY sentinel/ ./sentinel/
COPY service/ ./service/
COPY data/ ./data/
COPY src/ ./src/
COPY models/ ./models/
COPY test_inference.py .
COPY run_tests.py .

# Expose microservice HTTP port
EXPOSE 8000

# Health check probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Launch ASGI FastAPI server
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
