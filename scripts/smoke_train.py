"""Synthetic smoke training run for LingoLite.

This script validates that the training stack is wired correctly without
requiring SentencePiece assets or external datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from lingolite.mobile_translation_model import create_model
from lingolite.training import TranslationDataset, TranslationTrainer, collate_fn


@dataclass
class DummyTokenizer:
    """Minimal tokenizer stub compatible with TranslationDataset."""

    languages: List[str]
    token_to_id: Dict[str, int]
    eos_token_id: int
    pad_token_id: int

    def encode(
        self,
        text: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        add_special_tokens: bool = True,
        max_length: int = 128,
    ) -> List[int]:
        token_body = [10 + (ord(ch) % 20) for ch in text][: max(1, max_length - 4)]
        if add_special_tokens and src_lang is not None and tgt_lang is not None:
            return [
                self.token_to_id["<src>"],
                self.token_to_id[f"<{src_lang}>"],
                *token_body,
                self.eos_token_id,
                self.token_to_id["<tgt>"],
                self.token_to_id[f"<{tgt_lang}>"],
            ][:max_length]
        if add_special_tokens:
            return [self.token_to_id["<s>"], *token_body, self.eos_token_id][:max_length]
        return token_body[:max_length]


def build_dummy_tokenizer() -> DummyTokenizer:
    langs = ["en", "es"]
    token_to_id = {
        "<pad>": 0,
        "<s>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<src>": 4,
        "<tgt>": 5,
        "<en>": 6,
        "<es>": 7,
    }
    return DummyTokenizer(
        languages=langs,
        token_to_id=token_to_id,
        eos_token_id=2,
        pad_token_id=0,
    )


def main() -> None:
    torch.manual_seed(0)

    tokenizer = build_dummy_tokenizer()
    synthetic_data = [
        {"src_text": "hello world", "tgt_text": "hola mundo", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "good morning", "tgt_text": "buenos dias", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "thank you", "tgt_text": "gracias", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "how are you", "tgt_text": "como estas", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "test sentence", "tgt_text": "frase de prueba", "src_lang": "en", "tgt_lang": "es"},
        {"src_text": "small batch", "tgt_text": "lote pequeno", "src_lang": "en", "tgt_lang": "es"},
    ]

    dataset = TranslationDataset(synthetic_data, tokenizer, max_length=24)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
    )

    model = create_model(vocab_size=128, model_size="tiny", pad_token_id=tokenizer.pad_token_id)
    trainer = TranslationTrainer(
        model=model,
        train_loader=loader,
        learning_rate=3e-4,
        warmup_steps=20,
        max_steps=4,
        gradient_clip=1.0,
        label_smoothing=0.0,
        device="cpu",
        save_dir=str(Path(".tmp_manual") / "smoke_checkpoints"),
    )

    losses: List[float] = []
    for batch in loader:
        if trainer.global_step >= trainer.max_steps:
            break
        loss, metrics = trainer.train_step(batch)
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError("Non-finite loss detected during smoke training.")
        losses.append(loss)
        print(f"step={trainer.global_step} loss={loss:.4f} lr={metrics['lr']:.2e}")

    if len(losses) == 0:
        raise RuntimeError("Smoke training executed zero steps.")

    print(f"SMOKE TRAIN PASSED: steps={trainer.global_step}, final_loss={losses[-1]:.4f}")


if __name__ == "__main__":
    main()
