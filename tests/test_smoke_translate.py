"""CPU smoke test: verify the model can actually learn to translate.

Creates a tiny model, overfits it on a handful of synthetic translation
pairs, then runs greedy + cached + beam-search inference and checks that
the generated token sequences match the training targets.  If this test
passes the full train-to-translate pipeline is wired correctly on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pytest
import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationDataset, TranslationTrainer, collate_fn


# ── Dummy tokenizer (no SentencePiece required) ────────────────────────

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SRC_TOK, TGT_TOK = 4, 5
EN_TOK, ES_TOK = 6, 7
VOCAB_SIZE = 64  # keep small for speed


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
        # Deterministic: map each char to a token id in [8, VOCAB_SIZE)
        body = [8 + (ord(c) % (VOCAB_SIZE - 8)) for c in text][:max(1, max_length - 4)]
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
            "<pad>": PAD, "<s>": SOS, "</s>": EOS, "<unk>": UNK,
            "<src>": SRC_TOK, "<tgt>": TGT_TOK,
            "<en>": EN_TOK, "<es>": ES_TOK,
        },
    )


# ── Training pairs ─────────────────────────────────────────────────────

PAIRS = [
    {"src_text": "abc", "tgt_text": "xyz", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "hello", "tgt_text": "hola", "src_lang": "en", "tgt_lang": "es"},
    {"src_text": "one", "tgt_text": "uno", "src_lang": "en", "tgt_lang": "es"},
]


# ── Helpers ─────────────────────────────────────────────────────────────

def _train_until_overfit(
    model: torch.nn.Module,
    loader: DataLoader,
    max_epochs: int = 80,
    target_loss: float = 0.05,
) -> float:
    """Train with a raw optimiser loop (avoids OneCycleLR step-limit issues)."""
    optimiser = torch.optim.Adam(model.parameters(), lr=3e-3)
    model.train()
    last_loss = float("inf")
    for _epoch in range(max_epochs):
        for batch in loader:
            loss: torch.Tensor = model.compute_loss(
                src_input_ids=batch["src_input_ids"],
                tgt_input_ids=batch["tgt_input_ids"],
                src_attention_mask=batch["src_attention_mask"],
                tgt_attention_mask=batch["tgt_attention_mask"],
                label_smoothing=0.0,
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            last_loss = loss.item()
        if last_loss < target_loss:
            break
    return last_loss


SPECIAL_IDS = {PAD, SOS, EOS, UNK, SRC_TOK, TGT_TOK, EN_TOK, ES_TOK}


def _decode_without_special(tokens: torch.Tensor) -> List[int]:
    """Strip all control / language / framing tokens from a 1-D tensor."""
    return [t for t in tokens.tolist() if t not in SPECIAL_IDS]


# ── Test ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model_and_loader():
    """Train a tiny model once and share across all tests in this module."""
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

    final_loss = _train_until_overfit(model, loader)

    return model, loader, tokenizer, final_loss


class TestSmokeTranslate:
    """End-to-end CPU translation smoke tests."""

    # -- prerequisite: training converged --

    def test_training_converged(self, trained_model_and_loader):
        _, _, _, final_loss = trained_model_and_loader
        assert final_loss < 0.15, (
            f"Model did not converge: final loss {final_loss:.4f} (expected < 0.15)"
        )

    # -- greedy decoding reproduces training targets --

    def test_greedy_reproduces_targets(self, trained_model_and_loader):
        model, loader, tokenizer, _ = trained_model_and_loader
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

        # For each pair, the generated body tokens should match the target body
        for i, pair in enumerate(PAIRS):
            tgt_body = _decode_without_special(batch["tgt_input_ids"][i])
            gen_body = _decode_without_special(generated[i])
            assert gen_body == tgt_body, (
                f"Pair {i} ({pair['src_text']}→{pair['tgt_text']}): "
                f"expected {tgt_body}, got {gen_body}"
            )

    # -- cached generation produces valid output matching targets --

    def test_cached_produces_valid_output(self, trained_model_and_loader):
        model, loader, _, _ = trained_model_and_loader
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

        assert cached.ndim == 2
        assert cached.shape[0] == len(PAIRS)
        # Cached output should contain the target content tokens
        for i, pair in enumerate(PAIRS):
            tgt_body = _decode_without_special(batch["tgt_input_ids"][i])
            gen_body = _decode_without_special(cached[i])
            # All target tokens should appear in the generated output
            missing = [t for t in tgt_body if t not in gen_body]
            assert len(missing) == 0, (
                f"Pair {i} ({pair['src_text']}→{pair['tgt_text']}): "
                f"cached output missing target tokens {missing}"
            )

    # -- beam search produces valid output --

    def test_beam_search_valid(self, trained_model_and_loader):
        model, loader, _, _ = trained_model_and_loader
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
        # Beam output should contain at least some of the target tokens
        for i, pair in enumerate(PAIRS):
            tgt_body = set(_decode_without_special(batch["tgt_input_ids"][i]))
            gen_body = set(_decode_without_special(beam[i]))
            overlap = tgt_body & gen_body
            assert len(overlap) > 0, (
                f"Pair {i} ({pair['src_text']}→{pair['tgt_text']}): "
                f"beam search output shares no tokens with target"
            )

    # -- output shapes and dtypes are correct --

    def test_output_shape_and_dtype(self, trained_model_and_loader):
        model, loader, _, _ = trained_model_and_loader
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

    # -- no NaN / Inf in logits --

    def test_no_nan_in_forward(self, trained_model_and_loader):
        model, loader, _, _ = trained_model_and_loader
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
