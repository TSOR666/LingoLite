"""CPU smoke test for the full train-to-translate path.

The test overfits a tiny model on synthetic translation pairs and then
checks that greedy and cached decoding reconstruct the target exactly,
while beam decoding still produces a valid overlapping translation on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationDataset, TranslationTrainer, collate_fn


PAD, SOS, EOS, UNK = 0, 1, 2, 3
SRC_TOK, TGT_TOK = 4, 5
EN_TOK, ES_TOK = 6, 7
VOCAB_SIZE = 64
TEXT_TO_BODY_TOKENS = {
    "abc": [8, 9],
    "hello": [10, 11],
    "one": [12, 13],
    "xyz": [20, 21],
    "hola": [22, 23],
    "uno": [24, 25],
}


@dataclass
class _DummyTokenizer:
    """Minimal tokenizer compatible with TranslationDataset."""

    languages: List[str]
    token_to_id: Dict[str, int]
    eos_token_id: int = EOS
    pad_token_id: int = PAD

    def encode(
        self,
        text: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        add_special_tokens: bool = True,
        max_length: int = 128,
    ) -> List[int]:
        # Keep the synthetic task tiny and separable: each sentence maps to a
        # fixed token sequence so the smoke test can check exact translation.
        body = list(TEXT_TO_BODY_TOKENS.get(text, []))
        if not body:
            body = [8 + (ord(c) % (VOCAB_SIZE - 8)) for c in text][: max(1, max_length - 4)]
        if add_special_tokens and src_lang and tgt_lang:
            return [
                self.token_to_id["<src>"],
                self.token_to_id[f"<{src_lang}>"],
                *body,
                self.eos_token_id,
                self.token_to_id["<tgt>"],
                self.token_to_id[f"<{tgt_lang}>"],
            ][:max_length]
        if add_special_tokens:
            return [self.token_to_id["<s>"], *body, self.eos_token_id][:max_length]
        return body[:max_length]


def _make_tokenizer() -> _DummyTokenizer:
    return _DummyTokenizer(
        languages=["en", "es"],
        token_to_id={
            "<pad>": PAD,
            "<s>": SOS,
            "</s>": EOS,
            "<unk>": UNK,
            "<src>": SRC_TOK,
            "<tgt>": TGT_TOK,
            "<en>": EN_TOK,
            "<es>": ES_TOK,
        },
    )


PAIRS = [
    {"src_text": "abc", "tgt_text": "xyz", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "hello", "tgt_text": "hola", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "one", "tgt_text": "uno", "src_lang": "en", "tgt_lang": "es"},
]


def _decode_without_special(tokens: torch.Tensor) -> List[int]:
    """Strip all control / language / framing tokens from a 1-D tensor."""
    special_ids = {PAD, SOS, EOS, UNK, SRC_TOK, TGT_TOK, EN_TOK, ES_TOK}
    return [t for t in tokens.tolist() if t not in special_ids]


def _train_until_overfit(
    model: torch.nn.Module,
    loader: DataLoader,
    save_dir: str,
    max_epochs: int = 80,
    target_loss: float = 0.05,
) -> tuple[TranslationTrainer, float]:
    """Train through the real trainer so the smoke test covers that path."""
    trainer = TranslationTrainer(
        model=model,
        train_loader=loader,
        learning_rate=3e-3,
        warmup_steps=8,
        max_steps=96,
        gradient_clip=1.0,
        label_smoothing=0.0,
        device="cpu",
        save_dir=save_dir,
    )
    last_loss = float("inf")
    for _epoch in range(max_epochs):
        for batch in loader:
            if trainer.global_step >= trainer.max_steps:
                break
            loss, _metrics = trainer.train_step(batch)
            last_loss = loss
        if last_loss < target_loss or trainer.global_step >= trainer.max_steps:
            break
    return trainer, last_loss


@pytest.fixture(scope="module")
def trained_model_and_loader():
    """Train a tiny model once and share it across all tests in this module."""
    torch.manual_seed(42)

    tokenizer = _make_tokenizer()
    dataset = TranslationDataset(PAIRS, tokenizer, max_length=24)
    loader = DataLoader(
        dataset,
        batch_size=len(PAIRS),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=PAD),
    )

    model = create_model(
        vocab_size=VOCAB_SIZE,
        model_size="tiny",
        d_model=64,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        pad_token_id=PAD,
    )

    trainer, final_loss = _train_until_overfit(
        model,
        loader,
        save_dir=str(Path(".tmp_manual") / "smoke_translate"),
    )

    return model, loader, tokenizer, trainer, final_loss


def _assert_translation_matches(
    *,
    translated: torch.Tensor,
    expected: torch.Tensor,
    pair: Dict[str, str],
    mode: str,
) -> None:
    expected_body = _decode_without_special(expected)
    translated_body = _decode_without_special(translated)
    assert translated_body == expected_body, (
        f"{mode} translation mismatch for {pair['src_text']} -> {pair['tgt_text']}: "
        f"expected {expected_body}, got {translated_body}"
    )


class TestSmokeTranslate:
    """End-to-end CPU translation smoke tests."""

    def test_training_converged(self, trained_model_and_loader):
        _, _, _, trainer, final_loss = trained_model_and_loader
        assert trainer.global_step > 0
        assert final_loss < 0.15, (
            f"Model did not converge: final loss {final_loss:.4f} (expected < 0.15)"
        )

    def test_greedy_reproduces_targets(self, trained_model_and_loader):
        model, loader, _, _, _ = trained_model_and_loader
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            generated = model.generate(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                max_length=24,
                sos_token_id=SOS,
                eos_token_id=EOS,
            )

        for i, pair in enumerate(PAIRS):
            _assert_translation_matches(
                translated=generated[i],
                expected=batch["tgt_input_ids"][i],
                pair=pair,
                mode="greedy",
            )

    def test_cached_reproduces_targets(self, trained_model_and_loader):
        model, loader, _, _, _ = trained_model_and_loader
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            cached = model.generate_fast(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                max_length=24,
                sos_token_id=SOS,
                eos_token_id=EOS,
                temperature=1.0,
            )

        for i, pair in enumerate(PAIRS):
            _assert_translation_matches(
                translated=cached[i],
                expected=batch["tgt_input_ids"][i],
                pair=pair,
                mode="cached",
            )

    def test_beam_search_smoke(self, trained_model_and_loader):
        model, loader, _, _, _ = trained_model_and_loader
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            beam = model.generate(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                max_length=24,
                sos_token_id=SOS,
                eos_token_id=EOS,
                num_beams=3,
            )

        assert beam.ndim == 2
        assert beam.shape[0] == len(PAIRS)
        assert beam.dtype == torch.long
        assert beam.shape[1] <= 24

        for i, pair in enumerate(PAIRS):
            expected_body = set(_decode_without_special(batch["tgt_input_ids"][i]))
            translated_body = set(_decode_without_special(beam[i]))
            assert translated_body, (
                f"beam search produced an empty translation for "
                f"{pair['src_text']} -> {pair['tgt_text']}"
            )
            assert expected_body & translated_body, (
                f"beam search output shares no tokens with the target for "
                f"{pair['src_text']} -> {pair['tgt_text']}"
            )

    def test_output_shape_and_dtype(self, trained_model_and_loader):
        model, loader, _, _, _ = trained_model_and_loader
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            generated = model.generate(
                src_input_ids=batch["src_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                max_length=24,
                sos_token_id=SOS,
                eos_token_id=EOS,
            )

        assert generated.dtype == torch.long
        assert generated.shape[0] == len(PAIRS)
        assert generated.shape[1] <= 24

    def test_no_nan_in_forward(self, trained_model_and_loader):
        model, loader, _, _, _ = trained_model_and_loader
        model.eval()
        batch = next(iter(loader))

        with torch.no_grad():
            logits, _, enc_out = model.forward(
                src_input_ids=batch["src_input_ids"],
                tgt_input_ids=batch["tgt_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                tgt_attention_mask=batch["tgt_attention_mask"],
            )

        assert torch.isfinite(logits).all(), "Logits contain NaN/Inf"
        assert torch.isfinite(enc_out).all(), "Encoder output contains NaN/Inf"
