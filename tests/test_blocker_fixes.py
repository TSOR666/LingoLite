from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, List

import torch

from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationTrainer
from scripts.evaluate_model import translate_batch


def _mock_batch(vocab_size: int = 100) -> Dict[str, torch.Tensor]:
    return {
        "src_input_ids": torch.randint(0, vocab_size, (2, 10)),
        "tgt_input_ids": torch.randint(0, vocab_size, (2, 8)),
        "src_attention_mask": torch.ones(2, 10),
        "tgt_attention_mask": torch.ones(2, 8),
    }


def test_translation_trainer_clamps_warmup_steps() -> None:
    model = create_model(vocab_size=100, model_size="tiny")
    trainer = TranslationTrainer(
        model=model,
        train_loader=[_mock_batch()],
        warmup_steps=200,
        max_steps=20,
        device="cpu",
        save_dir=str(Path(".tmp_manual") / "test_ckpt"),
    )

    assert trainer.warmup_steps == 19
    assert trainer.scheduler.total_steps == 20


def test_generate_is_deterministic_by_default() -> None:
    torch.manual_seed(0)
    model = create_model(vocab_size=100, model_size="tiny")
    model.eval()

    src_input_ids = torch.randint(0, 100, (1, 12))
    src_attention_mask = torch.ones(1, 12)

    with torch.no_grad():
        out1 = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=12,
            sos_token_id=1,
            eos_token_id=2,
        )
        out2 = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=12,
            sos_token_id=1,
            eos_token_id=2,
        )

    assert torch.equal(out1, out2)


def test_generate_accepts_temperature_above_one() -> None:
    model = create_model(vocab_size=100, model_size="tiny")
    model.eval()

    src_input_ids = torch.randint(0, 100, (1, 8))
    src_attention_mask = torch.ones(1, 8)

    with torch.no_grad():
        out = model.generate(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=8,
            sos_token_id=1,
            eos_token_id=2,
            temperature=1.5,
        )

    assert out.shape[0] == 1


def test_generate_num_beams_uses_beam_path() -> None:
    model = create_model(vocab_size=100, model_size="tiny")
    src_input_ids = torch.randint(0, 100, (1, 6))

    called: Dict[str, int] = {}

    def _fake_generate_beam(self, **kwargs):  # type: ignore[no-untyped-def]
        called["num_beams"] = int(kwargs["num_beams"])
        return torch.full((1, 4), 2, dtype=torch.long)

    model.generate_beam = types.MethodType(_fake_generate_beam, model)

    out = model.generate(
        src_input_ids=src_input_ids,
        max_length=8,
        num_beams=3,
        sos_token_id=1,
        eos_token_id=2,
    )

    assert called["num_beams"] == 3
    assert out.shape == (1, 4)


class _DummyTokenizer:
    def __init__(self) -> None:
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.batch_encode_calls: List[Dict[str, object]] = []

    def batch_encode(self, texts: List[str], **_: object) -> Dict[str, torch.Tensor]:
        self.batch_encode_calls.append(_)
        batch_size = len(texts)
        input_ids = torch.ones(batch_size, 5, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def batch_decode(self, token_ids_batch: List[List[int]], skip_special_tokens: bool = True) -> List[str]:
        _ = skip_special_tokens
        return [f"decoded-{len(row)}" for row in token_ids_batch]


class _DummyModel:
    def eval(self) -> "_DummyModel":
        return self

    def generate_with_cache(self, src_input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        batch = src_input_ids.shape[0]
        return torch.full((batch, 3), 2, dtype=torch.long)

    def generate(self, src_input_ids: torch.Tensor, **_: object) -> torch.Tensor:
        batch = src_input_ids.shape[0]
        return torch.full((batch, 4), 2, dtype=torch.long)


def test_translate_batch_uses_batch_encode_decode() -> None:
    model = _DummyModel()
    tokenizer = _DummyTokenizer()
    source_sentences = ["a", "b", "c"]

    out_cached = translate_batch(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        source_sentences=source_sentences,
        device=torch.device("cpu"),
        batch_size=2,
        max_length=8,
        use_cache=True,
    )
    out_no_cache = translate_batch(
        model=model,  # type: ignore[arg-type]
        tokenizer=tokenizer,  # type: ignore[arg-type]
        source_sentences=source_sentences,
        device=torch.device("cpu"),
        batch_size=2,
        max_length=8,
        use_cache=False,
    )

    assert len(out_cached) == len(source_sentences)
    assert len(out_no_cache) == len(source_sentences)
    assert len(tokenizer.batch_encode_calls) > 0
    assert tokenizer.batch_encode_calls[0]["src_lang"] == "en"
    assert tokenizer.batch_encode_calls[0]["tgt_lang"] == "es"


def test_checkpoint_save_load_with_absolute_path() -> None:
    model = create_model(vocab_size=100, model_size="tiny")
    trainer = TranslationTrainer(
        model=model,
        train_loader=[_mock_batch()],
        warmup_steps=2,
        max_steps=10,
        device="cpu",
        save_dir=str(Path(".tmp_manual") / "ckpt_dir"),
    )

    _ = trainer.train_step(_mock_batch())

    checkpoint_path = (Path(".tmp_manual") / "absolute_checkpoint.pt").resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_checkpoint(str(checkpoint_path))
    trainer.load_checkpoint(str(checkpoint_path))

    assert checkpoint_path.exists()
