FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for models and data
RUN mkdir -p /app/models /app/data /app/checkpoints /app/tokenizer

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV LINGOLITE_MODEL_SIZE=small
ENV LINGOLITE_DEVICE=cpu
ENV LINGOLITE_ALLOWED_ORIGINS="http://localhost,http://127.0.0.1"

# Health check - verify API server is healthy (if running)
# Falls back to checking torch import if API server is not running
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || \
        python -c "import torch; import sys; sys.exit(0)" || exit 1

# Expose port for API server (when implemented)
EXPOSE 8000

# Default command - run lightweight validation (no PyTorch needed)
CMD ["python", "scripts/install.py"]
