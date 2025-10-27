# LingoLite v0.1.0 Release Notes

Date: 2025-10-27

Highlights:
- Packaging: Added `pyproject.toml` and console scripts `lingolite-train`, `lingolite-api`.
- Dev Modes: Environment flags for API to ease community testing.
  - `LINGOLITE_DISABLE_STARTUP=1` (skip artifacts, readiness 503)
  - `LINGOLITE_USE_STUB_TOKENIZER=1` (stub tokenizer)
  - `LINGOLITE_ALLOW_RANDOM_MODEL=1` (random tiny model if no checkpoint)
  - `LINGOLITE_ECHO_MODE=1` (echo input as translation)
- Installer: Robust ASCII-only `scripts/install.py`.
- Tests: API TestClient-based readiness test; cleaned fragile prints.
- Docker: Default command runs validation.
- Docs: Added Getting Started and Release Checklist.

Breaking Changes:
- None.

Upgrade Notes:
- For API usage without artifacts, opt-in to stub/random modes via env vars.
