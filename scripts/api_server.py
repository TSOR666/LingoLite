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
from pydantic import BaseModel, Field, field_validator, model_validator
import asyncio
from contextlib import asynccontextmanager
import torch
import time
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import os

from lingolite.mobile_translation_model import (
    create_model,
    load_model_from_checkpoint,
    MobileTranslationModel,
)
from lingolite.translation_tokenizer import TranslationTokenizer
from lingolite.utils import setup_logger, get_device
from lingolite.inference import GenerationOptions, TranslationEngine, TranslationInput
from lingolite.serving import TranslationService, ServiceBusyError, ServiceUnavailableError


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

@asynccontextmanager
async def lifespan(application: FastAPI):
    try:
        await startup_event()
        yield
    finally:
        await shutdown_event()


app = FastAPI(
    title="LingoLite Translation API",
    description="Mobile-optimized neural machine translation service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
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
translation_service: Optional[TranslationService] = None
device: torch.device = torch.device("cpu")
device_preference: str = "auto"
configured_model_size: str = "small"
current_model_size: Optional[str] = None


class TranslationParameters(BaseModel):
    src_lang: str = Field("en", description="Source language code")
    tgt_lang: str = Field("es", description="Target language code")
    max_length: int = Field(128, description="Maximum output length", ge=10, le=512)
    method: str = Field("greedy", description="Generation method: greedy, beam, fast")
    num_beams: int = Field(4, description="Number of beams", ge=1, le=10)
    temperature: float = Field(1.0, description="Sampling temperature", gt=0.0, le=2.0)
    length_penalty: float = Field(1.0, description="Beam length normalization", ge=0.0, le=5.0)
    min_new_tokens: int = Field(0, description="Minimum tokens before EOS is allowed", ge=0, le=511)
    no_repeat_ngram_size: int = Field(0, description="Block repeated n-grams; 0 disables", ge=0, le=32)

    @field_validator("method")
    @classmethod
    def validate_method(cls, v: str) -> str:
        if v not in ["greedy", "beam", "fast"]:
            raise ValueError("Method must be one of: greedy, beam, fast")
        return v

    @model_validator(mode="after")
    def validate_lengths(self):
        if self.min_new_tokens >= self.max_length:
            raise ValueError("min_new_tokens must be smaller than max_length")
        return self

    def generation_options(self) -> GenerationOptions:
        return GenerationOptions(**{
            name: getattr(self, name) for name in GenerationOptions.__dataclass_fields__
        })


class TranslationRequest(TranslationParameters):
    text: str = Field(..., description="Text to translate", min_length=1, max_length=5000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text must contain non-whitespace characters")
        return value


class BatchTranslationRequest(TranslationParameters):
    texts: List[str] = Field(..., description="Texts in result order", min_length=1, max_length=64)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: List[str]) -> List[str]:
        if any(not value.strip() or len(value) > 5000 for value in values):
            raise ValueError("Each text must contain 1-5000 characters and not be blank")
        return values


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
    outstanding_requests: int = 0
    completed_batches: int = 0


class BatchTranslationResponse(BaseModel):
    translations: List[TranslationResponse]


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


async def startup_event() -> None:
    global model, tokenizer, device, device_preference, configured_model_size, current_model_size
    global translation_service
    logger.info("Starting LingoLite API server...")
    if translation_service is not None:
        await translation_service.close()
        translation_service = None

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
        tokenizer_path = Path(os.getenv("LINGOLITE_TOKENIZER_PATH", "./tokenizer"))
        if use_stub:
            from lingolite.tokenizer_stub import StubTranslationTokenizer

            logger.warning("Using StubTranslationTokenizer (dev mode)")
            tokenizer = StubTranslationTokenizer(languages=["en", "es"])  # type: ignore
        else:
            if not tokenizer_path.exists():
                raise RuntimeError(f"Tokenizer not found at {tokenizer_path}")
            logger.info("Loading tokenizer...")
            tokenizer = TranslationTokenizer.from_pretrained(str(tokenizer_path))
        if tokenizer is None:
            raise RuntimeError("Tokenizer failed to load")
        logger.info(f"Tokenizer ready: vocab_size={tokenizer.get_vocab_size()}")

        # Model
        configured_model_size = _resolve_model_size()
        model_checkpoint = Path(os.getenv("LINGOLITE_MODEL_PATH", "./models/translation_model.pt"))
        if model_checkpoint.exists():
            logger.info(f"Loading model from {model_checkpoint}...")
            # SECURITY: Use weights_only=True to prevent arbitrary code execution
            checkpoint = torch.load(model_checkpoint, map_location=device, weights_only=True)
            vocab_size = tokenizer.get_vocab_size()
            model = load_model_from_checkpoint(
                checkpoint,
                fallback_vocab_size=vocab_size,
                fallback_model_size=configured_model_size,
            )
            model = model.to(device)
            model.eval()
            logger.info("Model loaded from checkpoint")
            current_model_size = (
                "custom"
                if isinstance(checkpoint, dict) and "config" in checkpoint
                else configured_model_size
            )
        else:
            if os.getenv("LINGOLITE_ALLOW_RANDOM_MODEL") == "1":
                logger.warning("Using randomly initialized model (dev mode)")
                vocab_size = tokenizer.get_vocab_size()
                model = create_model(vocab_size=vocab_size, model_size=configured_model_size).to(device)
                model.eval()
                current_model_size = configured_model_size
            else:
                raise RuntimeError(f"Model checkpoint not found at {model_checkpoint}")

        params = model.count_parameters()
        logger.info(
            "Model ready: size=%s total_params=%.1fM",
            current_model_size,
            params["total"] / 1e6,
        )
        translation_service = TranslationService(
            TranslationEngine(model, tokenizer, device),
            max_batch_size=int(os.getenv("LINGOLITE_MAX_BATCH_SIZE", "8")),
            max_pending=int(os.getenv("LINGOLITE_MAX_PENDING", "128")),
            max_batch_tokens=int(os.getenv("LINGOLITE_MAX_BATCH_TOKENS", "8192")),
            batch_wait_ms=float(os.getenv("LINGOLITE_BATCH_WAIT_MS", "5")),
            request_timeout=float(os.getenv("LINGOLITE_REQUEST_TIMEOUT", "30")),
        )
        await translation_service.start()
        logger.info("LingoLite API server started successfully")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise


async def shutdown_event() -> None:
    global translation_service, model, tokenizer, current_model_size
    logger.info("Shutting down LingoLite API server...")
    if translation_service is not None:
        await translation_service.close()
        translation_service = None
    model = None
    tokenizer = None
    current_model_size = None


@app.get("/", response_model=dict)
async def root() -> Dict[str, Any]:
    return {
        "name": "LingoLite Translation API",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {"translate": "/translate", "batch": "/translate/batch", "health": "/health", "docs": "/docs"},
    }


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        tokenizer_loaded=tokenizer is not None,
        device=str(device),
        model_size=current_model_size,
        outstanding_requests=translation_service.outstanding if translation_service else 0,
        completed_batches=translation_service.completed_batches if translation_service else 0,
    )


@app.get("/health/liveness")
async def liveness() -> Dict[str, str]:
    return {"status": "alive"}


@app.get("/health/readiness")
async def readiness() -> Dict[str, str]:
    if model is None or tokenizer is None or translation_service is None or not translation_service.ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    return {"status": "ready"}


@app.post("/translate", response_model=TranslationResponse)
async def translate(request: TranslationRequest) -> TranslationResponse:
    return (await _translate_texts([request.text], request))[0]


@app.post("/translate/batch", response_model=BatchTranslationResponse)
async def translate_batch(request: BatchTranslationRequest) -> BatchTranslationResponse:
    return BatchTranslationResponse(translations=await _translate_texts(request.texts, request))


async def _translate_texts(texts: List[str], request: TranslationParameters) -> List[TranslationResponse]:
    if tokenizer is None or translation_service is None or not translation_service.ready:
        raise HTTPException(status_code=503, detail="Model or tokenizer not loaded")
    if request.src_lang not in tokenizer.languages or request.tgt_lang not in tokenizer.languages:
        raise HTTPException(status_code=400, detail="Unsupported language code")
    if os.getenv("LINGOLITE_ECHO_MODE") == "1":
        return [TranslationResponse(
            translation=text, src_lang=request.src_lang, tgt_lang=request.tgt_lang,
            method=request.method, inference_time_ms=0.0,
            input_length=len(text.split()), output_length=len(text.split()),
        ) for text in texts]
    started = time.perf_counter()
    try:
        results = await translation_service.translate_many(
            [TranslationInput(text, request.src_lang, request.tgt_lang) for text in texts],
            request.generation_options(),
        )
    except ServiceBusyError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "1"}) from exc
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Translation deadline exceeded") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    return [TranslationResponse(
        translation=result.translation, src_lang=request.src_lang, tgt_lang=request.tgt_lang,
        method=request.method, inference_time_ms=elapsed,
        input_length=result.input_length, output_length=result.output_length,
    ) for result in results]


@app.get("/languages", response_model=dict)
async def get_supported_languages() -> Dict[str, Any]:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="Tokenizer not loaded")
    return {"languages": tokenizer.languages, "count": len(tokenizer.languages)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(error=exc.detail).model_dump(), headers=exc.headers)


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error").model_dump())


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
