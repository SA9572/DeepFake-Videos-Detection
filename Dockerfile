# ─────────────────────────────────────────────────────────────────────────────
# TriGuard-DF Deployment Dockerfile
# Requires: Python 3.11, ~2GB RAM, ~500MB disk
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# ─────────────────────────────────────────────────────────────────────────────
# Set working directory
# ─────────────────────────────────────────────────────────────────────────────

WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# Install system dependencies
# ─────────────────────────────────────────────────────────────────────────────

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    libhdf5-dev \
    pkg-config \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Upgrade pip
# ─────────────────────────────────────────────────────────────────────────────

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# ─────────────────────────────────────────────────────────────────────────────
# Copy project files
# ─────────────────────────────────────────────────────────────────────────────

COPY requirements.txt .
COPY app.py .
COPY main.py .
COPY configs/ configs/
COPY models/ models/
COPY src/ src/
COPY static/ static/
COPY templates/ templates/

# ─────────────────────────────────────────────────────────────────────────────
# Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────

RUN pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Create output directories
# ─────────────────────────────────────────────────────────────────────────────

RUN mkdir -p output/logs output/uploads

# ─────────────────────────────────────────────────────────────────────────────
# Expose port
# ─────────────────────────────────────────────────────────────────────────────

EXPOSE 5000

# ─────────────────────────────────────────────────────────────────────────────
# Health check
# ─────────────────────────────────────────────────────────────────────────────

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health').read()"

# ─────────────────────────────────────────────────────────────────────────────
# Run with gunicorn
# ─────────────────────────────────────────────────────────────────────────────

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "--access-logfile", "-", "--error-logfile", "-", "app:app"]
