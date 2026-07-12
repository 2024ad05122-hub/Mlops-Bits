# syntax=docker/dockerfile:1
# ---- Heart Disease classifier serving image ----
FROM python:3.12-slim

# Prevent .pyc files, unbuffered logs (so container logs stream to stdout)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# System deps kept minimal; scikit-learn wheels are prebuilt
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install python deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source, trained model, and config
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/

# Run as non-root
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# 2 workers is plenty for a demo; tune for your deployment
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "--access-logfile", "-", "app.main:app"]
