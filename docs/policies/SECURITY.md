# LingoLite Security Policy

_Last updated: November 13, 2025_

## Supported Versions

| Version | Supported |
| --- | --- |
| `main` (latest development build) | Yes |
| `< 1.0.0` tagged releases | No (development preview only) |

Security fixes are provided on the active `main` branch until a stable release line is established.

## Reporting a Vulnerability

We take security issues seriously. **Please do not open public GitHub issues for vulnerabilities.**

Report privately using one of the following channels:

1. **GitHub Security Advisories (preferred)**  
   - Navigate to the repository's Security -> Advisories page  
   - Click "Report a vulnerability" and include reproduction details
2. **Email**  
   - Send a message to the maintainer listed in `pyproject.toml` via GitHub  
   - Use the subject line `SECURITY REPORT: <short summary>`  

### What to Include
- Description of the issue and potential impact
- Step-by-step reproduction instructions
- Affected versions or commit SHAs
- Suggested remediation ideas (optional)
- Whether you prefer public credit in advisories

### Coordinated Disclosure Process
1. We acknowledge new reports within **48 hours**
2. We aim to validate the issue within **7 days**
3. Critical fixes target a **30-day** remediation window
4. Once a fix is available, we will coordinate disclosure timing with the reporter

## Security Best Practices

### For Users & Deployers
- **Keep dependencies current**: `pip install -e .[api,dev] --upgrade`
- **Use trusted checkpoints** only; untrusted PyTorch checkpoints can execute arbitrary code
- **Run the API server behind HTTPS** with proper authentication and rate limiting
- Restrict CORS origins (see `LINGOLITE_ALLOWED_ORIGINS` in `scripts/api_server.py`)
- Limit request sizes (`max_length`, `batch_size`) when exposing the API publicly
- Store secrets in environment variables; never commit `.env` files

### For Contributors
- Do not commit credentials, access tokens, or API keys
- Validate all external inputs (use `lingolite.utils.InputValidator`)
- Resolve filesystem paths with `Path.resolve()` before reading/writing
- Prefer well-maintained dependencies and run `pip-audit` before proposing new ones
- Request focused security review for changes touching serialization, checkpoint loading, or network boundaries

## Known Security Considerations

| Component | Risk | Status |
| --- | --- | --- |
| PyTorch checkpoint loading | Pickle execution when loading arbitrary checkpoints | Mitigated via documentation; production deployers should implement signing |
| FastAPI server | Ships without authentication and with permissive CORS for local dev | Documented limitation; production deployers must configure auth, HTTPS, and rate limiting |
| Tokenizer training | Malicious corpora can poison vocabularies | Covered in docs; sanitize corpora before training |

## Security Audit History

| Date | Scope | Result |
| --- | --- | --- |
| 2025-10-27 | Codebase scan for secrets & validation | Yes Pass |
| 2025-10-27 | Dependency review | Yes Pass |

See [`docs/reports/OPEN_SOURCE_READINESS_REPORT.md`](../reports/OPEN_SOURCE_READINESS_REPORT.md) for details.

## Recommended Production Configuration

```python
CORS_ORIGINS = ["https://yourdomain.com"]  # Never "*"
MAX_REQUEST_SIZE = 5000                    # Characters
RATE_LIMIT = "100/hour"
REQUIRE_AUTH = True                        # Token/JWT/OAuth
MAX_LENGTH = 512                           # Decoder cap
BATCH_SIZE = 32                            # Prevent large fan-out
TIMEOUT = 30                               # Seconds
```

## Additional Resources
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [PyTorch Serialization Guidance](https://pytorch.org/docs/stable/notes/serialization.html)
- [FastAPI Security Best Practices](https://fastapi.tiangolo.com/advanced/security/)

## Contact

Questions that are not security vulnerabilities can be discussed via GitHub Discussions or Issues. For vulnerabilities, please use the private channels described above.
