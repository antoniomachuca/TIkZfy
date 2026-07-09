# Base image: Python 3.10 Slim
FROM python:3.10-slim

# Environment configuration
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

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

# Copying core and adapters
COPY . .

# Default entrypoint delegable to external orchestrators
CMD ["python", "-c", "print('Engine container initialized successfully.')"]
