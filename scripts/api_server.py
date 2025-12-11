"""FastAPI Server for LingoLite Translation Model
Provides REST API endpoints for translation services.

This module is ASCII-only to avoid encoding issues in various consoles.
Supports dev modes via environment variables:
- LINGOLITE_DISABLE_STARTUP=1: skip loading artifacts (liveness OK, readiness 503)
- LINGOLITE_USE_STUB_TOKENIZER=1: use a lightweight tokenizer stub
- LINGOLITE_ALLOW_RANDOM_MODEL=1: create a random tiny model if no checkpoint
- LINGOLITE_ECHO_MODE=1: return the input text as the translation
- LINGOLITE_MODEL_SIZE / MODEL_SIZE: override model preset (tiny/small/medium/large)
- LINGOLITE_DEVICE / DEVICE: choose cuda/cpu/auto device preference
- LINGOLITE_ALLOWED_ORIGINS: comma-separated list of allowed CORS origins ("*" to disable protection)
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import torch
import time
import logging
from typing import Optional, List
from pathlib import Path
import os

from lingolite.mobile_translation_model import create_model, MobileTranslationModel
from lingolite.translation_tokenizer import TranslationTokenizer
from lingolite.utils import setup_logger, get_device


logger = setup_logger(name="lingolite_api", level=logging.INFO)

DEFAULT_ALLOWED_ORIGINS = ["http://localhost", "http://127.0.0.1"]
MODEL_SIZE_CHOICES = {"tiny", "small", "medium", "large"}
DEVICE_PREFERENCES = {"auto", "cpu", "cuda"}


def _parse_allowed_origins() -> List[str]:
    env_value = os.getenv("LINGOLITE_ALLOWED_ORIGINS")
    if not env_value:
        return DEFAULT_ALLOWED_ORIGINS

    parsed = [origin.strip() for origin in env_value.split(",") if origin.strip()]
    if not parsed:
        logger.warning("LINGOLITE_ALLOWED_ORIGINS was provided but empty; reverting to defaults.")
        return DEFAULT_ALLOWED_ORIGINS

    if parsed == ["*"]:
        logger.warning("CORS is unrestricted (LINGOLITE_ALLOWED_ORIGINS='*'). Only use this in trusted networks.")
    return parsed


def _resolve_model_size() -> str:
    env_value = (os.getenv("LINGOLITE_MODEL_SIZE") or os.getenv("MODEL_SIZE") or "small").lower()
    if env_value not in MODEL_SIZE_CHOICES:
        logger.warning("Unsupported model size '%s'. Falling back to 'small'.", env_value)
        return "small"
    return env_value


def _resolve_device_preference() -> str:
    env_value = (os.getenv("LINGOLITE_DEVICE") or os.getenv("DEVICE") or "auto").lower()
    if env_value not in DEVICE_PREFERENCES:
        logger.warning("Unsupported device preference '%s'. Falling back to 'auto'.", env_value)
        return "auto"
    return env_value


def _select_device(preference: str) -> torch.device:
    prefer_cuda = preference != "cpu"
    resolved = get_device(prefer_cuda=prefer_cuda)
    if preference == "cuda" and resolved.type != "cuda":
        logger.warning("CUDA requested but unavailable. Using %s instead.", resolved)
    return resolved

app = FastAPI(
    title="LingoLite Translation API",
    description="Mobile-optimized neural machine translation service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration - defaults to localhost-only for safety
allowed_origins = _parse_allowed_origins()

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[MobileTranslationModel] = None
tokenizer: Optional[TranslationTokenizer] = None
device: torch.device = torch.device("cpu")
device_preference: str = "auto"
configured_model_size: str = "small"
current_model_size: Optional[str] = None


class TranslationRequest(BaseModel):
    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)
    src_lang: str = Field("en", description="Source language code")
    tgt_lang: str = Field("es", description="Target language code")
    max_length: int = Field(128, description="Maximum output length", ge=10, le=512)
    method: str = Field("greedy", description="Generation method: greedy, beam, fast")
    num_beams: int = Field(4, description="Number of beams", ge=1, le=10)
    temperature: float = Field(1.0, description="Sampling temperature", gt=0.0, le=2.0)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ["greedy", "beam", "fast"]:
            raise ValueError("Method must be one of: greedy, beam, fast")
        return v


class TranslationResponse(BaseModel):
    translation: str
    src_lang: str
    tgt_lang: str
    method: str
    inference_time_ms: float
    input_length: int
    output_length: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    tokenizer_loaded: bool
    device: str
    model_size: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


@app.on_event("startup")
async def startup_event() -> None:
    global model, tokenizer, device, device_preference, configured_model_size, current_model_size
    logger.info("Starting LingoLite API server...")

    # Lightweight mode for tests/dev
    if os.getenv("LINGOLITE_DISABLE_STARTUP") == "1":
        logger.info("Startup disabled by LINGOLITE_DISABLE_STARTUP=1 (tests/dev mode)")
        model = None
        tokenizer = None
        device = torch.device("cpu")
        configured_model_size = _resolve_model_size()
        current_model_size = None
        return

    try:
        # Device
        device_preference = _resolve_device_preference()
        device = _select_device(device_preference)
        logger.info(f"Using device: {device}")

        # Tokenizer
        use_stub = os.getenv("LINGOLITE_USE_STUB_TOKENIZER") == "1"
        tokenizer_path = Path("./tokenizer")
        if use_stub:
            from lingolite.tokenizer_stub import StubTranslationTokenizer

            logger.warning("Using StubTranslationTokenizer (dev mode)")
            tokenizer = StubTranslationTokenizer(languages=["en", "es"])  # type: ignore
        else:
            if not tokenizer_path.exists():
                raise RuntimeError("Tokenizer not found at ./tokenizer")
            logger.info("Loading tokenizer...")
            tokenizer = TranslationTokenizer.from_pretrained(str(tokenizer_path))
        logger.info(f"Tokenizer ready: vocab_size={tokenizer.get_vocab_size()}")

        # Model
        configured_model_size = _resolve_model_size()
        model_checkpoint = Path("./models/translation_model.pt")
        if model_checkpoint.exists():
            logger.info(f"Loading model from {model_checkpoint}...")
            # SECURITY: Use weights_only=True to prevent arbitrary code execution
            checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=True)
            vocab_size = tokenizer.get_vocab_size()
            model = create_model(vocab_size=vocab_size, model_size=configured_model_size)
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            logger.info("Model loaded from checkpoint")
            current_model_size = configured_model_size
        else:
            if os.getenv("LINGOLITE_ALLOW_RANDOM_MODEL") == "1":
                logger.warning("Using randomly initialized model (dev mode)")
                vocab_size = tokenizer.get_vocab_size()
                model = create_model(vocab_size=vocab_size, model_size=configured_model_size).to(device)
                model.eval()
                current_model_size = configured_model_size
            else:
                raise RuntimeError("Model checkpoint not found at ./models/translation_model.pt")

        params = model.count_parameters()
        logger.info(
            "Model ready: size=%s total_params=%.1fM",
            current_model_size,
            params["total"] / 1e6,
        )
        logger.info("LingoLite API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down LingoLite API server...")


@app.get("/", response_model=dict)
async def root() -> dict:
    return {
        "name": "LingoLite Translation API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {"translate": "/translate", "health": "/health", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        tokenizer_loaded=tokenizer is not None,
        device=str(device),
        model_size=current_model_size,
    )


@app.get("/health/liveness")
async def liveness() -> dict:
    return {"status": "alive"}


@app.get("/health/readiness")
async def readiness() -> dict:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")

    # Echo mode for dev/testing
    if os.getenv("LINGOLITE_ECHO_MODE") == "1":
        return TranslationResponse(
            translation=request.text,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
            method=request.method,
            inference_time_ms=0.0,
            input_length=len(request.text.split()),
            output_length=len(request.text.split()),
        )

    start_time = time.time()

    if request.src_lang not in tokenizer.languages or request.tgt_lang not in tokenizer.languages:
        raise HTTPException(status_code=400, detail="Unsupported language code")

    input_ids = tokenizer.encode(
        text=request.text,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        add_special_tokens=True,
        max_length=512,
    )
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        if request.method == "beam" and hasattr(model, "generate_beam"):
            output_ids = model.generate_beam(
                src_input_ids=input_tensor,
                max_length=request.max_length,
                num_beams=request.num_beams,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elif request.method == "fast" and hasattr(model, "generate_fast"):
            output_ids = model.generate_fast(
                src_input_ids=input_tensor,
                max_length=request.max_length,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=request.temperature,
            )
        else:
            output_ids = model.generate(
                src_input_ids=input_tensor,
                max_length=request.max_length,
                sos_token_id=tokenizer.sos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                temperature=request.temperature,
            )

    translation = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
    inference_time = (time.time() - start_time) * 1000.0

    return TranslationResponse(
        translation=translation,
        src_lang=request.src_lang,
        tgt_lang=request.tgt_lang,
        method=request.method,
        inference_time_ms=round(inference_time, 2),
        input_length=len(input_ids),
        output_length=len(output_ids[0]),
    )


@app.get("/languages", response_model=dict)
async def get_supported_languages() -> dict:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    return {"languages": tokenizer.languages, "count": len(tokenizer.languages)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=exc.detail).dict())


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error").dict())


def main() -> None:
    import uvicorn

    # SECURITY: Default to localhost-only binding
    # Set LINGOLITE_BIND_HOST=0.0.0.0 to expose externally (with proper firewall/VPN)
    host = os.getenv("LINGOLITE_BIND_HOST", "127.0.0.1")
    port = int(os.getenv("LINGOLITE_PORT", "8000"))

    if host == "0.0.0.0":
        logger.warning(
            "SECURITY WARNING: Binding to 0.0.0.0 exposes API to all network interfaces. "
            "Ensure proper firewall rules, authentication, and rate limiting are configured."
        )

    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("scripts.api_server:app", host=host, port=port, log_level="info", reload=False)


if __name__ == "__main__":
    main()

