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
ENV LINGOLITE_BIND_HOST=0.0.0.0
ENV LINGOLITE_ALLOWED_ORIGINS="http://localhost,http://127.0.0.1"

# Route traffic only after the model, tokenizer and batching worker are ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/readiness', timeout=5).read()" || exit 1

# Expose the translation API.
EXPOSE 8000

# Requires a matching trained checkpoint and tokenizer mounted into /app.
CMD ["python", "-m", "scripts.api_server"]
