# Contributing to LingoLite

Thank you for your interest in contributing! This document explains how to get set up, the coding standards we follow, and what we look for in pull requests.

## Project Status

LingoLite is an active research/development project and **not production ready**. The codebase is intentionally released for community training, experimentation, and hardening. Please review [PRODUCTION_READINESS.md](docs/reports/PRODUCTION_READINESS.md) to understand the current limitations before contributing.

## Code of Conduct

We follow the [Contributor Covenant](docs/policies/CODE_OF_CONDUCT.md). Please report unacceptable behaviour via GitHub issues or privately to the maintainers.

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/LingoLite.git
   cd LingoLite
   ```
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   ```
3. Install dependencies from the editable package:
   ```bash
   # Core runtime + REST API extras
   pip install -e .[api]

   # Full developer tooling (tests, linting, REST API)
   pip install -e .[api,dev]
   ```
4. Generate optional sample data (helps with documentation/testing):
   ```bash
   python scripts/make_tiny_dataset.py
   ```
5. Verify the installation:
   ```bash
   python scripts/install.py
   python scripts/validate_improvements.py
   pytest -v tests
   ```

## Coding Standards

- Follow [PEP 8](https://peps.python.org/pep-0008/) and keep lines <= 100 characters.
- Use type hints on public functions.
- Prefer informative logging over `print`.
- Keep documentation ASCII unless there is a strong reason not to.
- Format and lint before opening a PR:
  ```bash
  black .
  flake8
  ```

## Testing

- Add unit tests for new behaviour.
- Run the targeted suite before committing:
  ```bash
  pytest -v tests
  ```
- If you add slow/performance tests, mark them with `@pytest.mark.slow`.
- API tests assume the `api` extra is installed. Use the environment variables described in the README (`LINGOLITE_USE_STUB_TOKENIZER`, etc.) when writing integration tests.

## Submitting Changes

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/my-improvement
   ```
2. **Make your changes** and keep commits focused.
3. **Run the verification commands** listed above.
4. **Commit with a descriptive message**:
   ```bash
   git commit -am "feat: add beam-search stub for dev API"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/my-improvement
   ```
6. **Open a Pull Request** and include:
   - Summary of what changed and why
   - Testing performed
   - Any follow-up work or known limitations

### Pull Request Checklist

- [ ] Code follows project style (PEP 8, type hints).
- [ ] `black .` and `flake8` pass.
- [ ] `pytest -v tests` passes (or failed tests are justified).
- [ ] Documentation updated (`README`, guides, changelog, etc.).
- [ ] No secrets, API keys, or credentials.
- [ ] Added or updated tests.

## Priority Areas

We especially welcome contributions in the following areas:

1. **Training pipeline validation** on real datasets.
2. **Improved testing** (integration, stress, failure modes).
3. **Documentation** (tutorials, troubleshooting, performance tips).
4. **Model quality** evaluations and benchmarks.
5. **Infrastructure** (monitoring, deployment tooling, dataset tooling).

## Questions?

If you need help:
- Open a GitHub issue with the `question` label.
- Join ongoing discussions in existing issues or PRs.
- Review the documentation in `docs/` for additional context.

Thanks for helping make LingoLite better!
