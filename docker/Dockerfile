# =============================
#  Base Image
# =============================
FROM python:3.10-slim

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# =============================
#  System dependencies
# =============================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================
#  Copy project files
# =============================
# Copy everything from the project root (context will be ".")
COPY . /app

# =============================
#  Install Python dependencies
# =============================
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================
#  Download NLTK data
# =============================
RUN python -m nltk.downloader punkt stopwords wordnet punkt_tab

# =============================
#  Expose port and run app
# =============================
EXPOSE 8000

# Default command: run FastAPI
CMD ["uvicorn", "scripts.api:app", "--host", "0.0.0.0", "--port", "8000"]
