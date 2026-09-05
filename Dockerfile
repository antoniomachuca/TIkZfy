# Base image: Python 3.10 Slim
FROM python:3.10-slim

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=8080
ENV CHECKPOINT_PATH=/app/checkpoints/curriculum_v4_best.pt
ENV VOCABULARY_PATH=/app/checkpoints/vocabulary_v4.json

# Installation of system dependencies and TeX Live compiler
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-pictures \
    texlive-science \
    ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Working directory configuration
WORKDIR /app

# Injection of structural and mathematical dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Explicit copy of application layers conforming to Clean Architecture
COPY pyproject.toml ./
COPY core/ ./core/
COPY ports/ ./ports/
COPY adapters/ ./adapters/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY checkpoints/ ./checkpoints/

# Expose default Cloud Run port
EXPOSE 8080

# Production ASGI server entrypoint with signal forwarding
CMD ["sh", "-c", "exec uvicorn adapters.api.app:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
