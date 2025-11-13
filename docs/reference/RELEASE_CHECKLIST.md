# LingoLite Release Checklist

Use this checklist before publishing a community release or tagging a version.

## 1. Versioning
- [ ] Bump the version in `pyproject.toml`
- [ ] Bump the version in `lingolite/__init__.py`
- [ ] Update `docs/reference/RELEASE_NOTES_<version>.md`

## 2. Quality Gates
- [ ] `python scripts/install.py`
- [ ] `python scripts/validate_improvements.py`
- [ ] `pytest -v` (optionally skip slow tests with `-m "not slow"`)
- [ ] `docker build -t lingolite:<version> .`

## 3. Artifacts
- [ ] Tokenizer artifacts under `./tokenizer/`
- [ ] Model checkpoint at `./models/translation_model.pt`
- [ ] Sample data or instructions refreshed (if applicable)

## 4. CI/CD
- [ ] GitHub Actions workflow green on `main`
- [ ] Docker publishing workflow configured for the new tag

## 5. Documentation
- [ ] README “Getting Started” section verified
- [ ] Guides updated for any behavior or CLI changes
- [ ] Deployment docs mention new environment variables or requirements

## 6. Release
- [ ] Create Git tag `v<version>`
- [ ] Publish GitHub Release referencing the release notes
- [ ] (Optional) Publish to PyPI:
  ```bash
  pipx run build
  twine upload dist/*
  ```
