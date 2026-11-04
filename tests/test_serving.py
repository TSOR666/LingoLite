"""Behavioral coverage for padding, bounded scheduling and HTTP integration."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest
import torch

from lingolite.inference import GenerationOptions, TranslationEngine, TranslationInput, TranslationResult
from lingolite.serving import TranslationService, ServiceBusyError, ServiceUnavailableError


class ToyTokenizer:
    languages = ["en", "es"]
    pad_token_id, sos_token_id, eos_token_id = 0, 1, 2

    def get_vocab_size(self):
        return 64

    def encode(self, text, **kwargs):
        assert kwargs["max_length"] is None
        return [1] + [int(word) for word in text.split()] + [2]

    def decode(self, ids, **kwargs):
        return " ".join(str(token) for token in ids if token > 2)


class ToyModel:
    vocab_size, pad_token_id = 64, 0
    config = {"max_seq_len": 32}

    def __init__(self):
        self.calls = []

    def get_config(self):
        return self.config

    def generate(self, **kwargs):
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        ids = kwargs["src_input_ids"]
        return torch.tensor([[1, int(row[1]), 2, 0, 0] for row in ids])

    generate_fast = generate
    generate_beam = generate


@pytest.mark.parametrize("method", ["greedy", "beam", "fast"])
def test_engine_pads_masks_and_preserves_result_order(method):
    model = ToyModel()
    engine = TranslationEngine(model, ToyTokenizer(), torch.device("cpu"))
    options = GenerationOptions(max_length=16, method=method, min_new_tokens=2,
                                no_repeat_ngram_size=3, length_penalty=0.8)
    results = engine.translate_batch([TranslationInput("10"), TranslationInput("11 12 13")], options)
    assert results == [TranslationResult("10", 3, 3), TranslationResult("11", 5, 3)]
    call = model.calls[0]
    assert call["src_input_ids"].tolist() == [[1, 10, 2, 0, 0], [1, 11, 12, 13, 2]]
    assert call["src_attention_mask"].tolist() == [[True, True, True, False, False], [True] * 5]
    assert call["min_new_tokens"] == 2 and call["no_repeat_ngram_size"] == 3
    if method == "beam":
        assert call["length_penalty"] == 0.8


def test_engine_isolates_invalid_inputs_without_truncation():
    model = ToyModel()
    engine = TranslationEngine(model, ToyTokenizer(), torch.device("cpu"), max_source_length=4)
    results = engine.translate_batch([
        TranslationInput("10 11 12"), TranslationInput("10"), TranslationInput("10", "xx"),
    ], GenerationOptions(max_length=16))
    assert isinstance(results[0], ValueError) and "5 tokens" in str(results[0])
    assert results[1] == TranslationResult("10", 3, 3)
    assert isinstance(results[2], ValueError)
    assert len(model.calls[0]["src_input_ids"]) == 1


@pytest.mark.parametrize("attribute,value", [("vocab_size", 60), ("pad_token_id", 3)])
def test_engine_rejects_incompatible_artifacts(attribute, value):
    model = ToyModel()
    setattr(model, attribute, value)
    with pytest.raises(ValueError):
        TranslationEngine(model, ToyTokenizer(), torch.device("cpu"))


class RecordingEngine:
    def __init__(self, block=False):
        self.calls = []
        self.started = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()

    def token_cost(self, options):
        return 10 * (options.num_beams if options.method == "beam" else 1)

    def translate_batch(self, items, options):
        self.calls.append((items, options))
        self.started.set()
        assert self.release.wait(timeout=5), "Test worker was not released"
        if items[0].text == "fail":
            raise RuntimeError("Synthetic generation failure")
        return [ValueError("bad input") if item.text == "bad" else TranslationResult(item.text, 1, 1)
                for item in items]


async def wait_started(engine):
    for _ in range(1000):
        if engine.started.is_set():
            return
        await asyncio.sleep(0.001)
    pytest.fail("Worker did not start")


def test_dynamic_batches_group_options_and_honor_token_budget():
    async def scenario():
        engine = RecordingEngine()
        service = TranslationService(engine, batch_wait_ms=10, max_batch_size=8, max_batch_tokens=20)
        await service.start()
        try:
            options = [GenerationOptions(), GenerationOptions(method="fast"), GenerationOptions(), GenerationOptions()]
            results = await asyncio.gather(*[
                service.translate(TranslationInput(str(i)), option) for i, option in enumerate(options)
            ])
            assert [r.translation for r in results] == ["0", "1", "2", "3"]
            assert [[item.text for item in call[0]] for call in engine.calls] == [["0", "2"], ["1"], ["3"]]
            assert service.outstanding == 0
            with pytest.raises(ValueError, match="token budget"):
                await service.translate(TranslationInput("beam"), GenerationOptions(method="beam"))
        finally:
            await service.close()
    asyncio.run(scenario())


def test_overload_counts_active_work_and_batch_admission_is_atomic():
    async def scenario():
        engine = RecordingEngine(block=True)
        service = TranslationService(engine, max_pending=2, batch_wait_ms=0)
        await service.start()
        first = asyncio.create_task(service.translate(TranslationInput("first"), GenerationOptions()))
        try:
            await wait_started(engine)
            with pytest.raises(ServiceBusyError):
                await service.translate_many([TranslationInput("a"), TranslationInput("b")], GenerationOptions())
            assert service.outstanding == 1
            # Event loop remains responsive while the model worker is blocked.
            await asyncio.sleep(0)
            engine.release.set()
            assert (await first).translation == "first"
        finally:
            engine.release.set()
            await service.close()
    asyncio.run(scenario())


def test_timeout_retains_active_capacity_and_cancels_queued_work():
    async def scenario():
        engine = RecordingEngine(block=True)
        service = TranslationService(engine, max_pending=1, batch_wait_ms=0, request_timeout=0.05)
        await service.start()
        first = asyncio.create_task(service.translate(TranslationInput("first"), GenerationOptions()))
        try:
            await wait_started(engine)
            with pytest.raises(asyncio.TimeoutError):
                await first
            assert service.outstanding == 1
            with pytest.raises(ServiceBusyError):
                await service.translate(TranslationInput("second"), GenerationOptions())
        finally:
            engine.release.set()
            await service.close()
        assert service.outstanding == 0
    asyncio.run(scenario())


def test_cancelled_queued_request_is_not_generated():
    async def scenario():
        engine = RecordingEngine(block=True)
        service = TranslationService(engine, max_batch_size=1, batch_wait_ms=0)
        await service.start()
        first = asyncio.create_task(service.translate(TranslationInput("first"), GenerationOptions()))
        try:
            await wait_started(engine)
            cancelled = asyncio.create_task(service.translate(TranslationInput("cancelled"), GenerationOptions()))
            await asyncio.sleep(0)
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
            engine.release.set()
            await first
            assert (await service.translate(TranslationInput("last"), GenerationOptions())).translation == "last"
            assert [[item.text for item in call[0]] for call in engine.calls] == [["first"], ["last"]]
        finally:
            engine.release.set()
            await service.close()
    asyncio.run(scenario())


def test_input_and_generation_failures_do_not_stop_worker():
    async def scenario():
        engine = RecordingEngine()
        service = TranslationService(engine, batch_wait_ms=5)
        await service.start()
        try:
            results = await asyncio.gather(
                service.translate(TranslationInput("bad"), GenerationOptions()),
                service.translate(TranslationInput("good"), GenerationOptions()), return_exceptions=True,
            )
            assert isinstance(results[0], ValueError) and results[1].translation == "good"
            with pytest.raises(RuntimeError, match="Synthetic"):
                await service.translate(TranslationInput("fail"), GenerationOptions())
            assert (await service.translate(TranslationInput("recovered"), GenerationOptions())).translation == "recovered"
        finally:
            await service.close()
    asyncio.run(scenario())


def test_shutdown_rejects_queued_work_and_finishes_active_batch():
    async def scenario():
        engine = RecordingEngine(block=True)
        service = TranslationService(engine, max_batch_size=1, batch_wait_ms=0)
        await service.start()
        first = asyncio.create_task(service.translate(TranslationInput("first"), GenerationOptions()))
        await wait_started(engine)
        queued = asyncio.create_task(service.translate(TranslationInput("queued"), GenerationOptions()))
        await asyncio.sleep(0)
        closing = asyncio.create_task(service.close())
        try:
            with pytest.raises(ServiceUnavailableError):
                await queued
            assert not closing.done()
        finally:
            engine.release.set()
            await first
            await closing
        assert service.outstanding == 0 and not service.ready
    asyncio.run(scenario())


@pytest.mark.parametrize("method", ["greedy", "fast", "beam"])
def test_real_model_batch_matches_single_translation(method):
    from lingolite.mobile_translation_model import MobileTranslationModel
    torch.manual_seed(42)
    model = MobileTranslationModel(vocab_size=64, d_model=16, n_encoder_layers=1,
                                   n_decoder_layers=1, n_heads=2, n_kv_heads=1,
                                   d_ff=32, dropout=0, max_seq_len=32)
    engine = TranslationEngine(model, ToyTokenizer(), torch.device("cpu"))
    assert engine.max_source_length == 32
    with pytest.raises(ValueError, match="model limit"):
        engine.translate_batch([TranslationInput("10")], GenerationOptions(max_length=33))
    oversized = engine.translate_batch([TranslationInput(" ".join(["10"] * 31))], GenerationOptions(max_length=10))
    assert isinstance(oversized[0], ValueError)
    options = GenerationOptions(max_length=10, method=method, num_beams=2)
    inputs = [TranslationInput("10"), TranslationInput("11 12 13")]
    batch = engine.translate_batch(inputs, options)
    separate = [engine.translate_batch([item], options)[0] for item in inputs]
    assert batch == separate


def test_http_batch_validation_and_overload_headers(monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from scripts import api_server
    monkeypatch.setenv("LINGOLITE_DISABLE_STARTUP", "1")
    monkeypatch.delenv("LINGOLITE_ECHO_MODE", raising=False)
    engine = RecordingEngine(block=True)
    service = TranslationService(engine, max_pending=2, batch_wait_ms=0)
    with TestClient(api_server.app) as client:
        monkeypatch.setattr(api_server, "model", ToyModel())
        monkeypatch.setattr(api_server, "tokenizer", ToyTokenizer())
        monkeypatch.setattr(api_server, "translation_service", service)
        client.portal.call(service.start)
        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                active = pool.submit(client.post, "/translate/batch", json={"texts": ["a", "b"]})
                try:
                    assert engine.started.wait(5)
                    assert client.get("/health/liveness").status_code == 200
                    assert client.get("/health/readiness").status_code == 200
                    overloaded = client.post("/translate", json={"text": "c"})
                    assert overloaded.status_code == 429
                    assert overloaded.headers["retry-after"] == "1"
                finally:
                    engine.release.set()
                response = active.result(5)
                assert response.status_code == 200
                assert [r["translation"] for r in response.json()["translations"]] == ["a", "b"]
            assert client.post("/translate", json={"text": " "}).status_code == 422
            assert client.post("/translate", json={"text": "a", "src_lang": "xx"}).status_code == 400
            assert client.post("/translate/batch", json={"texts": []}).status_code == 422
        finally:
            engine.release.set()


def test_api_loads_custom_checkpoint_and_closes_worker(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from lingolite.mobile_translation_model import MobileTranslationModel
    from scripts import api_server

    model = MobileTranslationModel(vocab_size=2048, d_model=16, n_encoder_layers=1,
                                   n_decoder_layers=1, n_heads=2, n_kv_heads=1,
                                   d_ff=32, dropout=0, max_seq_len=32)
    checkpoint = tmp_path / "model.pt"
    torch.save({"config": model.get_config(), "model_state_dict": model.state_dict()}, checkpoint)
    monkeypatch.setenv("LINGOLITE_DISABLE_STARTUP", "0")
    monkeypatch.setenv("LINGOLITE_USE_STUB_TOKENIZER", "1")
    monkeypatch.setenv("LINGOLITE_MODEL_PATH", str(checkpoint))
    monkeypatch.setenv("LINGOLITE_DEVICE", "cpu")
    monkeypatch.delenv("LINGOLITE_ECHO_MODE", raising=False)
    with TestClient(api_server.app) as client:
        service = api_server.translation_service
        assert service.ready
        assert service.engine.max_source_length == 32
        assert client.get("/health/readiness").status_code == 200
        response = client.post("/translate/batch", json={"texts": ["Hello", "Good morning"], "max_length": 10})
        assert response.status_code == 200
        assert len(response.json()["translations"]) == 2
        overlong = client.post("/translate", json={"text": "word " * 40, "max_length": 10})
        assert overlong.status_code == 400
        assert "supported limit is 32" in overlong.json()["error"]
        assert client.get("/health").json()["outstanding_requests"] == 0
    assert not service.ready
    assert api_server.translation_service is None
    assert api_server.model is None
