# -----------------------------
# Stage 1: Base image
# -----------------------------
FROM python:3.10-slim AS base

# Prevent Python from writing pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# -----------------------------
# Stage 2: System dependencies
# -----------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Stage 3: Python dependencies
# -----------------------------
# Copy dependency file first (for caching)
COPY requirements.txt .

# If you don’t have a requirements.txt yet, we’ll generate it below.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# -----------------------------
# Stage 4: Copy source code
# -----------------------------
COPY . .

# -----------------------------
# Stage 5: NLTK setup
# -----------------------------
RUN python -m nltk.downloader punkt stopwords wordnet

# -----------------------------
# Stage 6: Port + Entrypoint
# -----------------------------
EXPOSE 8000

# Default command to launch the API
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
