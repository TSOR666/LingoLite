# Changelog

All notable changes to LingoLite will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No entries yet._

## [0.1.0] - 2025-10-27

### Added
- Packaging via `pyproject.toml` with editable installs and extras
- Console scripts: `lingolite-train`, `lingolite-api`
- FastAPI dev modes (stub tokenizer, random model, echo mode)
- Stub tokenizer (`lingolite/tokenizer_stub.py`) for artifact-free exploration
- Tiny dataset generator (`scripts/make_tiny_dataset.py`) for quick demos
- API readiness pytest and documentation updates

### Fixed
- Removed fragile interactive demos that broke under different encodings
- Corrected CI paths for validator/test scripts
- Docker image now runs validation by default

### Changed
- Updated README and deployment guide with editable installs and dev-mode flags
- Rewrote CONTRIBUTING instructions for the packaging workflow
- Added release checklist and refreshed release notes

## [0.0.1] - 2025-10-25

### Added
- Mobile-optimized transformer architecture with GQA, RoPE, and SwiGLU
- Encoder-decoder model for sequence-to-sequence translation
- SentencePiece tokenizer with multilingual support (6 languages)
- Greedy and beam search generation methods
- KV caching for efficient autoregressive generation
- FastAPI REST API server for translation services
- Docker and Docker Compose deployment configurations
- Model quantization utilities (INT8, QAT support)
- ONNX export for mobile deployment
- BLEU evaluation scripts
- Comprehensive documentation and examples
- Unit test suite with pytest
- GitHub Actions CI/CD workflows
- Code of Conduct and Contributing guidelines
- MIT License

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Danish (da)

### Model Sizes
- Tiny: ~7M parameters (~30MB FP32, ~7.5MB INT8)
- Small: ~60M parameters (~240MB FP32, ~60MB INT8)
- Medium: ~140M parameters (~560MB FP32, ~140MB INT8)

---

## Release Notes

### Version Numbering

LingoLite uses semantic versioning:
- **MAJOR** version for incompatible API changes
- **MINOR** version for new functionality in a backward compatible manner
- **PATCH** version for backward compatible bug fixes

### Pre-1.0 Status

LingoLite is currently in pre-1.0 development status, meaning:
- The API may change between releases
- No production guarantees or stability promises
- Features are experimental and subject to change
- No pre-trained models included (train from scratch required)

Version 1.0.0 will be released when:
- A trained baseline model is provided
- Training pipeline is validated on real datasets
- API is stabilized and well-documented
- Comprehensive integration tests are in place
- Production deployment guide is validated

---

## Contributing

See [CONTRIBUTING.md](../policies/CONTRIBUTING.md) for guidelines on proposing changes and submitting pull requests.

## Security

See [SECURITY.md](../policies/SECURITY.md) for information on reporting vulnerabilities.

---

**Last Updated**: October 27, 2025
