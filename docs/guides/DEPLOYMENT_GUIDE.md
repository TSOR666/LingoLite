# LingoLite Deployment Guide

This guide provides step-by-step instructions for deploying LingoLite in various environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Local Development](#local-development)
3. [Docker Deployment](#docker-deployment)
4. [Production Deployment](#production-deployment)
5. [API Server](#api-server)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites

- Python 3.8+
- 4GB+ RAM (16GB+ recommended)
- Optional: NVIDIA GPU with CUDA 11.8+

### Installation

```bash
# Clone repository
git clone https://github.com/TSOR666/LingoLite.git
cd LingoLite

# Install editable package with API extras
pip install -e .[api]

# Verify installation
python scripts/install.py
```

### Train a Tokenizer

```python
from lingolite.translation_tokenizer import TranslationTokenizer

tokenizer = TranslationTokenizer(
    languages=['en', 'es', 'fr', 'de', 'it', 'da'],
    vocab_size=24000
)

# Train on your corpus files
tokenizer.train([
    'data/corpus_en.txt',
    'data/corpus_es.txt',
    # ... more files
])

# Save tokenizer
tokenizer.save('./tokenizer')
```

### Train a Model

```python
from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationTrainer, TranslationDataset

# Create model
model = create_model(vocab_size=24000, model_size='small')

# Prepare data
dataset = TranslationDataset(data, tokenizer, max_length=128)
train_loader = DataLoader(dataset, batch_size=32)

# Train
trainer = TranslationTrainer(model, train_loader)
trainer.train(num_epochs=10)

# Save model
torch.save(model.state_dict(), './models/translation_model.pt')
```

---

## Local Development

### Run Examples

> Tip: Generate sample data with `python scripts/make_tiny_dataset.py` if you want a quick playground dataset.

### Run Tests

```bash
# Installation check
python scripts/install.py

# Structure validation (no torch required)
python scripts/validate_improvements.py

# Unit tests
pytest -v tests
```

### Interactive Usage

```python
import torch
from mobile_translation_model import create_model
from lingolite.translation_tokenizer import TranslationTokenizer

# Load tokenizer and model
tokenizer = TranslationTokenizer.from_pretrained('./tokenizer')
model = create_model(vocab_size=tokenizer.get_vocab_size(), model_size='small')

# Load trained weights
checkpoint = torch.load('./models/translation_model.pt')
model.load_state_dict(checkpoint)
model.eval()

# Translate
text = "Hello, how are you?"
input_ids = tokenizer.encode(text, src_lang='en', tgt_lang='es')
input_tensor = torch.tensor([input_ids])

output = model.generate(
    src_input_ids=input_tensor,
    max_length=128,
    sos_token_id=tokenizer.sos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

translation = tokenizer.decode(output[0].tolist())
print(f"Translation: {translation}")
```

---

## Docker Deployment

### Build Docker Image

```bash
# Build image
docker build -t lingolite:latest .

# Check image
docker images | grep lingolite
```

### Run Container (CPU)

```bash
docker run -d \
  --name lingolite \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/tokenizer:/app/tokenizer \
  -e LINGOLITE_MODEL_SIZE=small \
  -e LINGOLITE_DEVICE=cpu \
  -e LINGOLITE_ALLOWED_ORIGINS=http://localhost,http://127.0.0.1 \
  lingolite:latest
```

### Run Container (GPU)

```bash
docker run -d \
  --name lingolite \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/tokenizer:/app/tokenizer \
  -e LINGOLITE_MODEL_SIZE=small \
  -e LINGOLITE_DEVICE=cuda \
  -e LINGOLITE_ALLOWED_ORIGINS=https://example.com \
  lingolite:latest
```

### Using Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f lingolite

# Stop services
docker-compose down
```

### Docker Compose with GPU

Edit `docker-compose.yml` and uncomment the GPU section:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## Production Deployment

### Prerequisites

Before deploying to production:

1. **Train your model** on quality parallel corpora
2. **Evaluate BLEU score** to ensure quality
3. **Save model checkpoint** to `./models/translation_model.pt`
4. **Train and save tokenizer** to `./tokenizer`
5. **Configure monitoring** (see Monitoring section)
6. **Setup CI/CD** (GitHub Actions already configured)

### API Server Deployment

#### Option 1: Direct Python

```bash
pip install -e .[api]
export LINGOLITE_USE_STUB_TOKENIZER=1  # optional, dev mode
export LINGOLITE_ALLOW_RANDOM_MODEL=1 # optional, dev mode
lingolite-api
```

#### Option 2: Uvicorn with Workers

```bash
# Production settings with multiple workers
uvicorn scripts.api_server:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log
```

#### Option 3: Behind Nginx

**Nginx configuration** (`/etc/nginx/sites-available/lingolite`):

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for translation
        proxy_read_timeout 300;
        proxy_connect_timeout 300;
        proxy_send_timeout 300;
    }

    # Health checks
    location /health {
        access_log off;
        proxy_pass http://127.0.0.1:8000/health;
    }
}
```

Enable and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/lingolite /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Kubernetes Deployment

#### Create Kubernetes manifests

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lingolite
  labels:
    app: lingolite
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lingolite
  template:
    metadata:
      labels:
        app: lingolite
    spec:
      containers:
      - name: lingolite
        image: ghcr.io/your-org/lingolite:latest
        ports:
        - containerPort: 8000
        env:
        - name: LINGOLITE_MODEL_SIZE
          value: "small"
        - name: LINGOLITE_DEVICE
          value: "cuda"
        - name: LINGOLITE_ALLOWED_ORIGINS
          value: "https://api.example.com"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health/liveness
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/readiness
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        volumeMounts:
        - name: models
          mountPath: /app/models
        - name: tokenizer
          mountPath: /app/tokenizer
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: lingolite-models-pvc
      - name: tokenizer
        persistentVolumeClaim:
          claimName: lingolite-tokenizer-pvc
```

**service.yaml:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: lingolite
spec:
  selector:
    app: lingolite
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Check status
kubectl get pods -l app=lingolite
kubectl get svc lingolite
```

---

## API Server

### API Endpoints

#### POST /translate

Translate text from source to target language.

**Request:**

```json
{
  "text": "Hello, how are you?",
  "src_lang": "en",
  "tgt_lang": "es",
  "max_length": 128,
  "method": "beam",
  "num_beams": 4,
  "temperature": 1.0
}
```

**Response:**

```json
{
  "translation": "Hola, ¿cómo estás?",
  "src_lang": "en",
  "tgt_lang": "es",
  "method": "beam",
  "inference_time_ms": 156.32,
  "input_length": 12,
  "output_length": 9
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "src_lang": "en",
    "tgt_lang": "es",
    "method": "greedy"
  }'
```

#### GET /health

Check server health status.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "tokenizer_loaded": true,
  "device": "cuda",
  "model_size": "small"
}
```

#### GET /languages

Get supported languages.

**Response:**

```json
{
  "languages": ["en", "es", "fr", "de", "it", "da"],
  "count": 6
}
```

#### GET /docs

Interactive API documentation (Swagger UI).

#### GET /redoc

Alternative API documentation (ReDoc).

### API Client Examples

#### Python

```python
import requests

response = requests.post(
    'http://localhost:8000/translate',
    json={
        'text': 'Hello, world!',
        'src_lang': 'en',
        'tgt_lang': 'es',
        'method': 'beam',
        'num_beams': 4
    }
)

result = response.json()
print(f"Translation: {result['translation']}")
print(f"Time: {result['inference_time_ms']}ms")
```

#### JavaScript

```javascript
fetch('http://localhost:8000/translate', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Hello, world!',
    src_lang: 'en',
    tgt_lang: 'es',
    method: 'greedy'
  })
})
.then(res => res.json())
.then(data => {
  console.log('Translation:', data.translation);
  console.log('Time:', data.inference_time_ms, 'ms');
});
```

---

## Monitoring

### Logging

LingoLite uses structured logging. Configure log level:

```python
from lingolite.utils import setup_logger
import logging

logger = setup_logger(level=logging.INFO)  # or DEBUG, WARNING, ERROR
```

### Metrics (Recommended)

For production, integrate Prometheus metrics:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Add to scripts/api_server.py
translation_requests = Counter(
    'translation_requests_total',
    'Total translation requests',
    ['src_lang', 'tgt_lang', 'method']
)

translation_latency = Histogram(
    'translation_latency_seconds',
    'Translation latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

# Start metrics server
start_http_server(9090)
```

### Health Checks

Kubernetes health check endpoints:

- **Liveness:** `/health/liveness` - Is the service running?
- **Readiness:** `/health/readiness` - Is the service ready to accept traffic?

```bash
# Test health
curl http://localhost:8000/health

# Test liveness
curl http://localhost:8000/health/liveness

# Test readiness
curl http://localhost:8000/health/readiness
```

---

## Troubleshooting

### Model Not Loading

**Problem:** API returns "Model not loaded" error.

**Solution:**

1. Check model file exists: `ls -lh ./models/translation_model.pt`
2. Check tokenizer exists: `ls -lh ./tokenizer/`
3. Review startup logs: `docker logs lingolite`

### Out of Memory

**Problem:** Process killed with OOM error.

**Solutions:**

- **Reduce batch size** during training
- **Use smaller model** (tiny or small)
- **Enable gradient accumulation**
- **Reduce max_length** for inference
- **Use CPU instead of GPU** for very large batches

### Slow Inference

**Problem:** Translations taking too long.

**Solutions:**

- **Use greedy decoding** instead of beam search
- **Enable KV caching** with `model.generate_fast()`
- **Reduce max_length**
- **Use quantized model** (INT8)
- **Enable GPU** if available
- **Batch multiple requests** together

### CUDA Out of Memory

**Problem:** GPU runs out of memory.

**Solutions:**

```bash
# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Use smaller batch size
# Reduce model size (use 'tiny' or 'small')
# Reduce max_length
```

### Docker Build Fails

**Problem:** Docker build fails.

**Solutions:**

```bash
# Check Docker daemon
sudo systemctl status docker

# Clean Docker cache
docker system prune -a

# Rebuild with no cache
docker build --no-cache -t lingolite:latest .
```

### API Server Not Starting

**Problem:** FastAPI server fails to start.

**Check logs:**

```bash
# Direct
lingolite-api

# Docker
docker logs lingolite

# Kubernetes
kubectl logs -f deployment/lingolite
```

**Common issues:**

- Port 8000 already in use: `lsof -i :8000`
- Missing dependencies: `pip install -e .[api]`
- Model/tokenizer not found: Check file paths

---

## Performance Optimization

### Model Quantization

Reduce model size by 75%:

```python
from lingolite.quantization_utils import quantize_model

# Load model
model = create_model(vocab_size=24000, model_size='small')
model.load_state_dict(torch.load('./models/translation_model.pt'))

# Quantize
quantized_model = quantize_model(model)

# Save
torch.save(quantized_model.state_dict(), './models/translation_model_int8.pt')
```

### ONNX Export

Export for mobile deployment:

```python
from export_onnx import export_to_onnx

export_to_onnx(
    model=model,
    output_path='./models/translation_model.onnx',
    opset_version=14
)
```

### Batch Processing

Process multiple translations efficiently:

```python
# Batch encode
texts = ["Hello!", "Goodbye!", "Thank you!"]
batch = tokenizer.batch_encode(
    texts,
    src_lang='en',
    tgt_lang='es',
    padding=True,
    return_tensors=True
)

# Batch generate
outputs = model.generate(
    src_input_ids=batch['input_ids'],
    src_attention_mask=batch['attention_mask'],
    max_length=128
)

# Batch decode
translations = tokenizer.batch_decode(outputs.tolist())
```

---

## Security Best Practices

1. **Input Validation:** Already implemented via `InputValidator`
2. **Rate Limiting:** Add to production API
3. **Authentication:** Add API keys or OAuth
4. **HTTPS:** Use TLS certificates in production
5. **Firewall:** Restrict access to necessary ports
6. **Updates:** Keep dependencies up to date

---

## Support

For issues and questions:

- **GitHub Issues:** https://github.com/TSOR666/LingoLite/issues
- **Documentation:** See README.md, SECURITY.md, COMMUNITY_DEPLOYMENT_REVIEW.md
- **Examples:** See scripts/examples.py

---

## License

MIT License - See LICENSE file for details.
