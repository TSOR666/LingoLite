# LingoLite

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**LingoLite** is a lightweight, mobile-optimized neural machine translation (NMT) framework designed for efficient multilingual translation on resource-constrained devices. Built with PyTorch, it features a modern transformer architecture with state-of-the-art optimizations for mobile deployment.

---

## ⚠️ Development Status

**LingoLite is currently in active development and is NOT production-ready.**

- ❌ **No pre-trained models included** - You must train models from scratch
- ❌ **Training pipeline experimental** - Tested only with dummy data, not validated on real datasets
- ❌ **No example training data** - Users must source and prepare their own datasets
- ⚠️ **API server requires trained artifacts** - Will not start without model checkpoint and tokenizer
- 🤝 **Community-maintained** - Provided as-is for contributors to train, evaluate, and harden
- ℹ️ **Suitable for research and experimentation** - Good foundation for building custom translation systems

**For detailed production readiness assessment, see [PRODUCTION_READINESS.md](docs/reports/PRODUCTION_READINESS.md)**

---

## Table of Contents

- [Features](#features)
- [Recent Updates](#recent-updates)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [REST API Server](#rest-api-server)
- [Docker Deployment](#docker-deployment)
- [Getting Started](#getting-started)
- [Usage Examples](#usage-examples)
- [Model Quantization](#model-quantization)
- [ONNX Export for Mobile Deployment](#onnx-export-for-mobile-deployment)
- [Model Evaluation](#model-evaluation)
- [Training](#training)
- [Testing](#testing)
- [Model Configuration](#model-configuration)
- [Generation Parameters](#generation-parameters)
- [Security](#security)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Documentation](#documentation)
- [Support](#support)

## Features

- **Mobile-Optimized Architecture**: Designed specifically for efficient inference on mobile devices
  - Grouped Query Attention (GQA) reduces memory footprint by 4-8x
  - Rotary Position Embeddings (RoPE) eliminates learned position parameters
  - SwiGLU Feed-Forward Networks for efficient computation
  - Weight tying between encoder/decoder embeddings

- **Multilingual Translation**: Supports 6 languages out of the box
  - English (en), Spanish (es), French (fr), German (de), Italian (it), Danish (da)
  - Easy to extend to additional languages

- **Advanced Generation Methods**:
  - Greedy decoding for fastest inference
  - Beam search for higher quality translations
  - KV caching for efficient autoregressive generation
  - Temperature-based sampling for diverse outputs

- **Development Infrastructure**:
  - FastAPI REST API server with async support (requires trained model)
  - Docker and Docker Compose deployment configurations
  - Comprehensive input validation and error handling
  - Security-hardened file operations
  - Professional logging infrastructure
  - Automated test suite with pytest (unit tests only, no integration tests)
  - Model quantization (INT8) utilities and ONNX export scripts
  - BLEU evaluation scripts (untested on real data)

- **Flexible Model Sizes**:
  - **Tiny**: ~7M parameters (~30MB FP32, ~7.5MB INT8)
  - **Small**: ~60M parameters (~240MB FP32, ~60MB INT8)
  - **Medium**: ~140M parameters (~560MB FP32, ~140MB INT8)

## Recent Updates

**October 26, 2025** - Production readiness fixes:
- ✅ **Fixed Training Pipeline**: Resolved OneCycleLR crash; training loop now respects max_steps
- ✅ **Proper Training Entry Point**: Command-line interface with validation and error handling
- ✅ **Fixed Dependencies**: Added missing numpy to requirements.txt
- ✅ **Automated Testing**: Converted manual tests to pytest with proper assertions
- ✅ **Fail-Closed Deployment**: API server now requires trained model and tokenizer to start
- ✅ **Honest Documentation**: Added PRODUCTION_READINESS.md with accurate assessment
- ⚠️ **Status Disclaimer**: Clear warning that project is not production-ready

**Previous Updates** (framework components):
- ✅ **REST API Server**: FastAPI-based HTTP endpoints (requires trained model)
- ✅ **Docker Support**: Containerization configurations
- ✅ **Model Quantization**: Utility scripts for INT8 quantization
- ✅ **ONNX Export**: Mobile deployment export scripts
- ✅ **BLEU Evaluation**: Translation quality assessment scripts
- ✅ **Danish Language Support**: 6 language support (en, es, fr, de, it, da)

See [PRODUCTION_READINESS.md](docs/reports/PRODUCTION_READINESS.md) for current status.

## Architecture

LingoLite uses a **Transformer encoder-decoder** architecture with modern optimizations:

```
┌─────────────────────────────────────────────┐
│           Source Text (e.g., English)       │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────▼──────────┐
         │  TranslationTokenizer│
         │   (SentencePiece)   │
         └─────────┬──────────┘
                   │ Token IDs
         ┌─────────▼──────────┐
         │   Token Embeddings │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Transformer       │
         │  Encoder           │
         │  (Bidirectional)   │
         │  • RoPE Position   │
         │  • GQA Attention   │
         │  • SwiGLU FFN      │
         └─────────┬──────────┘
                   │ Context
                   │
         ┌─────────▼──────────┐
         │  Transformer       │
         │  Decoder           │
         │  (Causal)          │
         │  • Self-Attention  │
         │  • Cross-Attention │
         │  • SwiGLU FFN      │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  Output Projection │
         └─────────┬──────────┘
                   │
         ┌─────────▼──────────┐
         │  TranslationTokenizer│
         │      (Decode)       │
         └─────────┬──────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        Target Text (e.g., Spanish)          │
└─────────────────────────────────────────────┘
```

### Key Components

- **RMSNorm**: Efficient normalization layer (lighter than LayerNorm)
- **Rotary Position Embeddings (RoPE)**: Relative position encoding without learned parameters
- **Grouped Query Attention (GQA)**: Reduces KV cache size while maintaining quality
- **SwiGLU**: Gated Linear Unit with Swish activation for efficient feed-forward networks

## Installation

### Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- 4GB+ RAM (for tiny model), 16GB+ recommended (for larger models)

### Install Dependencies

Install in editable mode so local changes are picked up automatically:

```bash
# Minimal runtime (core + REST API)
pip install -e .[api]

# Full developer setup (tests, linting, REST API)
pip install -e .[api,dev]
```

Key dependencies (see `pyproject.toml` for details):
- `torch>=2.0.0` – Deep learning framework
- `sentencepiece>=0.1.99` – Tokenization
- `sacrebleu>=2.3.1` – Translation evaluation
- `tqdm>=4.65.0` – Progress bars

### Verify Installation

```bash
python scripts/install.py
```

This will verify that all required files are present and properly structured.

## Quick Start

### 1. Train a Tokenizer

```python
from lingolite.translation_tokenizer import TranslationTokenizer

# Prepare training data file paths (parallel corpora recommended)
corpus_files = [
    "data/corpus_en.txt",
    "data/corpus_es.txt",
    "data/corpus_fr.txt",
    "data/corpus_de.txt",
    "data/corpus_it.txt",
    "data/corpus_da.txt",
]

# Train tokenizer and save artifacts
tokenizer = TranslationTokenizer(vocab_size=24000)
tokenizer.train(corpus_files)
tokenizer.save("tokenizer_model")
```

### 2. Create a Translation Model

```python
from lingolite.mobile_translation_model import create_model

# Create a tiny model for exploratory work
model = create_model(vocab_size=24000, model_size="tiny")
params = model.count_parameters()
print(f"Model has {params['total']:,} trainable parameters")
```

### 3. Translate Text

```python
import torch

# Prepare input
text = "Hello, world!"
input_ids = tokenizer.encode(
    text,
    src_lang="en",
    tgt_lang="es",
    add_special_tokens=True,
)
input_tensor = torch.tensor([input_ids])

# Generate translation (greedy)
output_ids = model.generate(
    src_input_ids=input_tensor,
    max_length=128,
    sos_token_id=tokenizer.sos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

# Decode output
translation = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(f"Translation: {translation}")
```

### 4. Use Beam Search for Better Quality

```python
# Generate with beam search (slower but higher quality)
output_ids = model.generate_beam(
    src_input_ids=input_tensor,
    max_length=128,
    num_beams=4,
    length_penalty=1.0,
    sos_token_id=tokenizer.sos_token_id,
    eos_token_id=tokenizer.eos_token_id
)

translation = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
print(f"Beam search translation: {translation}")
```

## REST API Server

LingoLite includes a production-ready FastAPI server for serving translations via HTTP endpoints.

### Starting the Server

```bash
pip install -e .[api]                 # install server dependencies
export LINGOLITE_USE_STUB_TOKENIZER=1 # optional: use stub tokenizer (no artifacts)
export LINGOLITE_ALLOW_RANDOM_MODEL=1 # optional: create random tiny model
lingolite-api
```

Windows PowerShell:

```powershell
pip install -e .[api]
$env:LINGOLITE_USE_STUB_TOKENIZER = "1"
$env:LINGOLITE_ALLOW_RANDOM_MODEL = "1"
lingolite-api
```

### API Endpoints

**Health Check**
```bash
curl http://localhost:8000/health
```

**Translate Text**
```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, world!",
    "src_lang": "en",
    "tgt_lang": "es",
    "max_length": 128,
    "method": "beam",
    "num_beams": 4
  }'
```

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

> **Tip**: Set `LINGOLITE_ECHO_MODE=1` to echo inputs without running the model (useful for smoke tests).

## Docker Deployment

LingoLite supports containerized deployment with Docker and Docker Compose.

### Quick Start with Docker

```bash
# Build the Docker image
docker build -t lingolite:latest .

# Run the container
docker run -p 8000:8000 lingolite:latest
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

The Docker setup includes:
- Multi-stage build for optimized image size
- Health checks and automatic restarts
- Volume mounts for model persistence
- Configurable resource limits
- Production-ready security settings

See [DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md) for detailed deployment instructions.

## Usage Examples

See `scripts/examples.py` for comprehensive examples including:

1. **Tokenizer Training** - Train a multilingual SentencePiece tokenizer
2. **Model Creation** - Create models of different sizes
3. **Basic Inference** - Simple translation with greedy decoding
4. **Advanced Generation** - Beam search and temperature sampling
5. **Model Quantization** - Reduce model size with INT8 quantization
6. **Complete Workflow** - End-to-end training and inference pipeline

Run examples:
```bash
python scripts/examples.py
```

## Model Quantization

LingoLite includes comprehensive quantization utilities to reduce model size and improve inference speed.

### Dynamic Quantization (Post-Training)

```python
from lingolite.quantization_utils import quantize_model_dynamic

# Quantize model to INT8
quantized_model = quantize_model_dynamic(
    model,
    dtype=torch.qint8,
    output_path="model_quantized.pt"
)

# Model size reduced by ~75% (FP32 → INT8)
print(f"Size reduction: {model.num_parameters() * 4 / (1024**2):.1f}MB → "
      f"{model.num_parameters() / (1024**2):.1f}MB")
```

### Static Quantization (Calibration-Based)

```python
from lingolite.quantization_utils import quantize_model_static

# Prepare calibration dataset
calibration_data = [...]  # Your representative samples

# Static quantization for maximum efficiency
quantized_model = quantize_model_static(
    model,
    calibration_data,
    output_path="model_static_quantized.pt"
)
```

### Quantization-Aware Training (QAT)

```python
from lingolite.quantization_utils import prepare_qat_model, convert_qat_model

# Prepare model for QAT
qat_model = prepare_qat_model(model)

# Train with quantization simulation
trainer.train(qat_model)

# Convert to quantized model
quantized_model = convert_qat_model(qat_model)
```

Quantization features:
- **Dynamic Quantization**: Fast post-training quantization
- **Static Quantization**: Calibration-based for optimal accuracy
- **Quantization-Aware Training**: Train with quantization in the loop
- **Compression Analysis**: Detailed size and performance metrics

## ONNX Export for Mobile Deployment

Export models to ONNX format for deployment on mobile devices (TensorFlow Lite, CoreML, etc.).

### Export to ONNX

```python
from export_onnx import export_to_onnx

# Export encoder and decoder separately for mobile optimization
export_to_onnx(
    model,
    encoder_path="encoder.onnx",
    decoder_path="decoder.onnx",
    vocab_size=24000,
    max_seq_length=128
)
```

### Command-Line Export

```bash
python scripts/export_onnx.py \
    --model-path translation_model.pt \
    --tokenizer-path tokenizer_model \
    --output-dir ./onnx_models \
    --max-seq-length 128 \
    --opset-version 14
```

### Verify ONNX Model

```python
import onnxruntime as ort

# Load and verify ONNX model
session = ort.InferenceSession("encoder.onnx")
print(f"Inputs: {[i.name for i in session.get_inputs()]}")
print(f"Outputs: {[o.name for o in session.get_outputs()]}")
```

ONNX export features:
- **Separate encoder/decoder**: Optimized for mobile architectures
- **Dynamic shapes**: Support variable sequence lengths
- **Quantization-ready**: Export quantized models
- **Validation**: Automatic output verification
- **Mobile-optimized**: TensorFlow Lite and CoreML compatible

## Model Evaluation

Evaluate translation quality using industry-standard BLEU scores.

### BLEU Evaluation

```python
from pathlib import Path

from evaluate_model import evaluate_model
from evaluate_bleu import compute_bleu

# Evaluate a trained checkpoint against a dataset of source/target pairs
results = evaluate_model(
    model_path=Path("checkpoints/model.pt"),
    tokenizer_path=Path("tokenizer"),
    source_file=Path("data/test.src"),
    target_file=Path("data/test.tgt"),
)

print(f"BLEU Score: {results['bleu']:.2f}")
print(f"chrF Score: {results['chrf']:.2f}")
```

### Command-Line Evaluation

```bash
python scripts/evaluate_model.py \
    --model checkpoints/model.pt \
    --tokenizer tokenizer \
    --source data/test.src \
    --target data/test.tgt \
    --output reports/eval.json
```

### Evaluation Metrics

The evaluation suite provides:
- **BLEU scores**: Standard MT quality metric (sacrebleu)
- **Per-language pair analysis**: Individual scores for each translation direction
- **Inference speed**: Tokens per second, latency analysis
- **Memory profiling**: Peak memory usage during inference
- **Error analysis**: Common failure patterns and edge cases

See [EVALUATION_REPORT.md](docs/reports/EVALUATION_REPORT.md) for comprehensive benchmark results.

## Training

### Prepare Training Data

Use high-quality, balanced corpora for each supported language. Public datasets that work
well for compact translation models include:

- [Europarl v10](https://www.statmt.org/europarl/) – Parliamentary proceedings with
  consistent domain coverage across many European languages.
- [Tatoeba Challenge](https://tatoeba.org/eng/downloads) – Sentence-aligned community
  translations that provide colloquial phrasing and short-form utterances.
- [OPUS OpenSubtitles](https://opus.nlpl.eu/OpenSubtitles-v2018.php) – Informal movie and
  TV dialog suitable for conversational styles (ensure proper cleaning).
- [Global Voices](https://globalvoices.org/lingua/) – News articles translated by native
  speakers; useful for narrative and journalistic tone.
- [CCMatrix](https://opus.nlpl.eu/CCMatrix.php) – Large-scale web-mined parallel corpus
  that is helpful for pretraining before domain-specific fine-tuning.
- [JW300](https://opus.nlpl.eu/JW300.php) – Religious text translations that can improve
  coverage for low-resource language pairs when filtered appropriately.

Combine multiple corpora to diversify styles and reduce domain bias. When expanding to
new languages, prefer resources that include explicit language codes or metadata for
clean filtering.

```python
from lingolite.training import TranslationDataset
from torch.utils.data import DataLoader

# Your parallel corpus
data = [
    {"src": "Hello", "tgt": "Hola", "src_lang": "en", "tgt_lang": "es"},
    {"src": "Goodbye", "tgt": "Adiós", "src_lang": "en", "tgt_lang": "es"},
    # ... more examples
]

# Create dataset
dataset = TranslationDataset(
    data=data,
    tokenizer=tokenizer,
    max_length=128
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id)
)
```

### Train the Model

```python
from lingolite.training import TranslationTrainer

# Initialize trainer
trainer = TranslationTrainer(
    model=model,
    train_dataloader=dataloader,
    learning_rate=1e-4,
    num_epochs=10,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Train
trainer.train()

# Save model
torch.save(model.state_dict(), "translation_model.pt")
```

### Exhaustive Training Strategy

1. **Preprocess & Normalize** – Lowercase consistently, normalize punctuation with
   [Moses scripts](https://github.com/moses-smt/mosesdecoder/tree/master/scripts), remove
   duplicates, and filter out noisy or misaligned sentence pairs.
2. **Split Strategically** – Build stratified train/validation/test splits for each
   language pair to monitor overfitting and domain drift. Ensure held-out sets cover
   varied sequence lengths and styles.
3. **Tokenizer Iteration** – Train the tokenizer on the full multilingual mix, inspect
   coverage statistics, and retrain with adjusted `character_coverage` if rare glyphs are
   dropped.
4. **Curriculum Training** – Start with the highest-resource pairs (e.g., en↔es, en↔fr)
   for stable convergence, then gradually interleave medium- and low-resource pairs using
   temperature-based sampling to avoid forgetting.
5. **Regular Evaluation** – Track BLEU/chrF scores per language pair with
   [SacreBLEU](https://github.com/mjpost/sacrebleu). Complement metrics with human review
   of edge cases (idioms, named entities).
6. **Fine-Tune & Distill** – After base training, fine-tune on target-domain data (e.g.,
   customer support) and optionally distill from a larger teacher model to maintain
   quality under mobile constraints.
7. **Quantization-Aware Training** – Enable INT8-aware fine-tuning before deployment to
   minimize accuracy loss when compressing the model.

### Training Features

- **Mixed Precision Training**: Automatic with `torch.cuda.amp` (GPU only)
- **Gradient Accumulation**: For effective larger batch sizes
- **Learning Rate Scheduling**: OneCycleLR for optimal convergence
- **Progress Tracking**: Real-time loss and metrics with tqdm
- **Checkpointing**: Save model at regular intervals

## Testing

Run the automated test suite:

```bash
# Run targeted tests (recommended)
pytest -v tests

# Skip slow markers if desired
pytest -v tests -m "not slow"

# With coverage reporting
pytest -v tests --cov=lingolite
```

**Test Coverage:**
- ✅ Input validation for all parameters
- ✅ Tensor dimension checking
- ✅ Token ID range validation
- ✅ KV cache functionality
- ✅ Beam search generation
- ✅ Helper functions (format_size, format_time, device selection)
- ✅ Model generation methods
- ❌ Training pipeline (not tested)
- ❌ API endpoints (not tested)
- ❌ Integration tests (not implemented)

Validate code structure:
```bash
python scripts/validate_improvements.py
```

## Model Configuration

### Tiny Model (Mobile Devices)
```python
model = MobileTranslationModel(
    vocab_size=24000,
    d_model=256,
    num_encoder_layers=4,
    num_decoder_layers=4,
    num_heads=4,
    d_ff=1024,
    dropout=0.1
)
# ~7M parameters, ~30MB FP32, ~7.5MB INT8
```

### Small Model (Tablets/Desktop)
```python
model = MobileTranslationModel(
    vocab_size=24000,
    d_model=512,
    num_encoder_layers=6,
    num_decoder_layers=6,
    num_heads=8,
    d_ff=2048,
    dropout=0.1
)
# ~60M parameters, ~240MB FP32, ~60MB INT8
```

### Medium Model (Desktop/Server)
```python
model = MobileTranslationModel(
    vocab_size=24000,
    d_model=768,
    num_encoder_layers=8,
    num_decoder_layers=8,
    num_heads=12,
    d_ff=3072,
    dropout=0.1
)
# ~140M parameters, ~560MB FP32, ~140MB INT8
```

## Generation Parameters

### Greedy Decoding (Fastest)
```python
output = model.generate(
    src_input_ids=input_ids,
    max_length=128,
    temperature=1.0,  # Lower = more deterministic
    sos_token_id=1,
    eos_token_id=2
)
```

### Beam Search (Higher Quality)
```python
output = model.generate_beam(
    src_input_ids=input_ids,
    max_length=128,
    num_beams=4,           # More beams = better quality but slower
    length_penalty=1.0,     # >1.0 favors longer, <1.0 favors shorter
    early_stopping=True,    # Stop when all beams finish
    sos_token_id=1,
    eos_token_id=2
)
```

## Security

LingoLite implements comprehensive security measures:

- **Input Validation**: All inputs validated for type, shape, and range
- **Path Validation**: File operations protected against directory traversal
- **Resource Limits**: Max length constraints prevent memory exhaustion
- **Token ID Validation**: Prevents out-of-bounds access
- **No Code Execution**: Pure data processing, no eval() or exec()

See `AUDIT_REPORT.md` for detailed security audit results.

## Performance

### Memory Usage (Inference)

| Model | FP32 | INT8 | KV Cache (128 tokens) |
|-------|------|------|-----------------------|
| Tiny  | 30MB | 7.5MB | ~2MB |
| Small | 240MB | 60MB | ~8MB |
| Medium | 560MB | 140MB | ~18MB |

### Generation Speed (Approximate, CPU)

- **Greedy**: 5-10 tokens/second (tiny), 1-3 tokens/second (medium)
- **Beam Search (4 beams)**: 2-5 tokens/second (tiny), 0.5-1 tokens/second (medium)
- **With KV Cache**: 2-3x speedup for greedy decoding

*Note: Actual speed depends on hardware, sequence length, and batch size*

## Project Structure

```
LingoLite/
|-- docs/
|   |-- guides/
|   |   `-- DEPLOYMENT_GUIDE.md
|   |-- policies/
|   |   |-- CODE_OF_CONDUCT.md
|   |   |-- CONTRIBUTING.md
|   |   `-- SECURITY.md
|   |-- reference/
|   |   |-- CHANGELOG.md
|   |   |-- RELEASE_CHECKLIST.md
|   |   `-- RELEASE_NOTES_v0.1.0.md
|   `-- reports/
|       |-- IMPROVEMENTS.md
|       |-- OPEN_SOURCE_READINESS_REPORT.md
|       `-- PRODUCTION_READINESS.md
|-- examples/
|   `-- data/
|       `-- tiny_dataset.json
|-- lingolite/
|   |-- __init__.py
|   |-- encoder_decoder.py
|   |-- generation_utils.py
|   |-- mobile_translation_model.py
|   |-- model_components.py
|   |-- quantization_utils.py
|   |-- tokenizer_stub.py
|   |-- translation_tokenizer.py
|   `-- training.py
|-- scripts/
|   |-- api_server.py
|   |-- install.py
|   |-- make_tiny_dataset.py
|   `-- validate_improvements.py
|-- tests/
|   |-- test_api_bypass_startup.py
|   `-- ... (beam search, cache, and generation tests)
|-- pyproject.toml
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Contributing

Contributions are welcome! Areas for improvement:

1. **Complete KV Cache Integration**: Full integration with decoder layers for maximum speedup
2. **Additional Languages**: Extend tokenizer for more language pairs (currently supports 6)
3. **Mobile Framework Integration**: Convert ONNX models to TensorFlow Lite/CoreML
4. **Model Distillation**: Implement knowledge distillation from larger teacher models
5. **More Tests**: Edge cases, stress tests, integration tests for new features
6. **Benchmarks**: BLEU scores on standard datasets (WMT, OPUS, Flores)
7. **Monitoring**: Add Prometheus metrics and Grafana dashboards
8. **Multi-GPU Training**: Distributed training support for large-scale datasets

### Development Setup

```bash
# Install development dependencies
pip install pytest black flake8

# Run tests
pytest -v

# Format code
black *.py

# Lint code
flake8 *.py --max-line-length=100
```

## Citation

If you use LingoLite in your research or project, please cite:

```bibtex
@software{lingolite2025,
  title = {LingoLite: Mobile-Optimized Neural Machine Translation},
  author = {LingoLite Contributors},
  year = {2025},
  url = {https://github.com/TSOR666/LingoLite}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Transformer Architecture**: Vaswani et al., "Attention is All You Need" (2017)
- **Rotary Position Embeddings**: Su et al., "RoFormer" (2021)
- **Grouped Query Attention**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models" (2023)
- **SwiGLU**: Shazeer, "GLU Variants Improve Transformer" (2020)
- **PyTorch**: Paszke et al., "PyTorch: An Imperative Style, High-Performance Deep Learning Library" (2019)
- **SentencePiece**: Kudo & Richardson, "SentencePiece: A simple and language independent approach to subword tokenization" (2018)

## Documentation

Comprehensive documentation is available:

- **[README.md](README.md)** - Quick start guide and API reference (this file)
- **[PRODUCTION_READINESS.md](docs/reports/PRODUCTION_READINESS.md)** - **START HERE: Honest assessment of current state**
- **[SECURITY.md](docs/policies/SECURITY.md)** - Security policy and vulnerability reporting
- **[CHANGELOG.md](docs/reference/CHANGELOG.md)** - Version history and release notes
- **[CONTRIBUTING.md](docs/policies/CONTRIBUTING.md)** - Contribution guidelines and development setup
- **[CODE_OF_CONDUCT.md](docs/policies/CODE_OF_CONDUCT.md)** - Community guidelines
- **[DEPLOYMENT_GUIDE.md](docs/guides/DEPLOYMENT_GUIDE.md)** - Deployment instructions (requires trained model)
- **[EVALUATION_REPORT.md](docs/reports/EVALUATION_REPORT.md)** - Evaluation scripts and utilities
- **[AUDIT_REPORT.md](docs/reports/AUDIT_REPORT.md)** - Security audit documentation
- **[IMPROVEMENTS.md](docs/reports/IMPROVEMENTS.md)** - Recent improvements and changes
- **[scripts/examples.py](scripts/examples.py)** - Code examples and usage patterns

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check the comprehensive documentation listed above
- Review API documentation at `/docs` when running the API server
- Review `scripts/examples.py` for usage patterns

---

**Built with modern ML best practices for efficient mobile translation**
## Getting Started

The fastest way to explore LingoLite locally without training artifacts:

- Install (API extras)
  - `pip install -e .[api]`

- Run the API with a stub tokenizer and a random tiny model (dev mode):
  - Linux/macOS:
    - `export LINGOLITE_USE_STUB_TOKENIZER=1`
    - `export LINGOLITE_ALLOW_RANDOM_MODEL=1`
    - `lingolite-api`
  - Windows PowerShell:
    - `$env:LINGOLITE_USE_STUB_TOKENIZER=1`
    - `$env:LINGOLITE_ALLOW_RANDOM_MODEL=1`
    - `lingolite-api`

- Optional echo mode (bypass model execution):
  - Set `LINGOLITE_ECHO_MODE=1` to return the input text directly.

- Quick tiny dataset for experimentation:
  - `python scripts/make_tiny_dataset.py`
  - Writes `examples/data/tiny_dataset.json`

To train for real usage, follow the tokenizer training steps and the training CLI described above; then place artifacts under `./tokenizer/` and a model checkpoint under `./models/translation_model.pt`.
