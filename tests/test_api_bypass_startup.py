import os
import pytest

try:
    from fastapi.testclient import TestClient  # starlette + httpx are required
    HAS_TESTCLIENT = True
except Exception:
    HAS_TESTCLIENT = False


@pytest.mark.skipif(not HAS_TESTCLIENT, reason="FastAPI TestClient not available")
def test_api_liveness_and_readiness_without_startup():
    os.environ["LINGOLITE_DISABLE_STARTUP"] = "1"
    # Import late to ensure env var is set before app is created
    from scripts.api_server import app

    with TestClient(app, raise_server_exceptions=False) as client:
        # Liveness should be OK regardless of startup
        r = client.get("/health/liveness")
        assert r.status_code == 200

        # Readiness should fail (503) because model/tokenizer are not loaded
        r = client.get("/health/readiness")
        assert r.status_code == 503

        # Languages endpoint should also fail since tokenizer is not loaded
        r = client.get("/languages")
        assert r.status_code in (503, 500)
