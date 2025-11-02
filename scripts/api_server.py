"""FastAPI Server for LingoLite Translation Model
Provides REST API endpoints for translation services.

This module is ASCII-only to avoid encoding issues in various consoles.
Supports dev modes via environment variables:
- LINGOLITE_DISABLE_STARTUP=1: skip loading artifacts (liveness OK, readiness 503)
- LINGOLITE_USE_STUB_TOKENIZER=1: use a lightweight tokenizer stub
- LINGOLITE_ALLOW_RANDOM_MODEL=1: create a random tiny model if no checkpoint
- LINGOLITE_ECHO_MODE=1: return the input text as the translation
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

app = FastAPI(
    title="LingoLite Translation API",
    description="Mobile-optimized neural machine translation service",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS configuration - can be restricted via environment variable
# For production, set LINGOLITE_ALLOWED_ORIGINS="https://yourdomain.com,https://anotherdomain.com"
allowed_origins_env = os.getenv("LINGOLITE_ALLOWED_ORIGINS", "*")
allowed_origins = allowed_origins_env.split(",") if allowed_origins_env != "*" else ["*"]

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
    global model, tokenizer, device
    logger.info("Starting LingoLite API server...")

    # Lightweight mode for tests/dev
    if os.getenv("LINGOLITE_DISABLE_STARTUP") == "1":
        logger.info("Startup disabled by LINGOLITE_DISABLE_STARTUP=1 (tests/dev mode)")
        model = None
        tokenizer = None
        device = torch.device("cpu")
        return

    try:
        # Device
        device = get_device(prefer_cuda=True)
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
        model_checkpoint = Path("./models/translation_model.pt")
        if model_checkpoint.exists():
            logger.info(f"Loading model from {model_checkpoint}...")
            checkpoint = torch.load(model_checkpoint, map_location=device)
            vocab_size = tokenizer.get_vocab_size()
            model = create_model(vocab_size=vocab_size, model_size="small")
            state = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state)
            model = model.to(device)
            model.eval()
            logger.info("Model loaded from checkpoint")
        else:
            if os.getenv("LINGOLITE_ALLOW_RANDOM_MODEL") == "1":
                logger.warning("Using randomly initialized model (dev mode)")
                vocab_size = tokenizer.get_vocab_size()
                model = create_model(vocab_size=vocab_size, model_size="tiny").to(device)
                model.eval()
            else:
                raise RuntimeError("Model checkpoint not found at ./models/translation_model.pt")

        params = model.count_parameters()
        logger.info(f"Model ready: {params['total']/1e6:.1f}M parameters")
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
        model_size="tiny" if model is not None else None,
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

    uvicorn.run("scripts.api_server:app", host="0.0.0.0", port=8000, log_level="info", reload=False)


if __name__ == "__main__":
    main()

