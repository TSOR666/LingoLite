"""Regression coverage for the fit-for-purpose optimization pass."""

from __future__ import annotations

import inspect

import pytest
import torch

from lingolite.generation_utils import KVCache, _generate_with_beam_search_impl
from lingolite.mobile_translation_model import create_model
from lingolite.model_components import GroupedQueryAttention
from lingolite.training import TranslationDataset, TranslationTrainer


def _small_model(vocab_size: int = 257):
    torch.manual_seed(0)
    return create_model(
        vocab_size=vocab_size,
        model_size="tiny",
        d_model=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
    )


def test_chunked_loss_matches_full_loss_and_bounds_projection() -> None:
    model = _small_model()
    model.eval()
    src = torch.randint(1, model.vocab_size, (3, 7))
    tgt = torch.randint(1, model.vocab_size, (3, 12))
    tgt[0, -2:] = model.pad_token_id

    projected_token_counts: list[int] = []

    def record_projection(_module, inputs) -> None:
        projected_token_counts.append(int(inputs[0].shape[0]))

    hook = model.decoder.lm_head.register_forward_pre_hook(record_projection)
    try:
        chunked = model.compute_loss(
            src,
            tgt,
            label_smoothing=0.1,
            logits_chunk_size=7,
        )
    finally:
        hook.remove()

    full = model.compute_loss(
        src,
        tgt,
        label_smoothing=0.1,
        logits_chunk_size=0,
    )
    torch.testing.assert_close(chunked, full, rtol=1e-5, atol=1e-6)
    assert len(projected_token_counts) > 1
    assert max(projected_token_counts) <= 7


def test_efficient_ffn_uses_compute_matched_swiglu_width() -> None:
    common = {
        "vocab_size": 128,
        "model_size": "tiny",
        "d_model": 96,
        "n_encoder_layers": 1,
        "n_decoder_layers": 1,
        "n_heads": 4,
        "n_kv_heads": 2,
        "max_seq_len": 32,
        "dropout": 0.0,
    }
    legacy = create_model(**common, d_ff=384)
    efficient = create_model(**common, efficient_ffn=True)

    assert efficient.get_config()["d_ff"] == 256
    assert efficient.count_parameters()["total"] < legacy.count_parameters()["total"]


class _CountingTokenizer:
    languages = ["en", "es"]
    token_to_id = {"<s>": 1}
    sos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def encode(
        self,
        text: str,
        src_lang: str | None = None,
        tgt_lang: str | None = None,
        add_special_tokens: bool = True,
        max_length: int = 128,
    ) -> list[int]:
        self.calls += 1
        ids = [4, 5, 6]
        if add_special_tokens:
            ids = [1, *ids, 2]
        return ids[:max_length]


def test_pretokenized_dataset_encodes_each_example_once() -> None:
    data = [
        {
            "src_text": "hello",
            "tgt_text": "hola",
            "src_lang": "en",
            "tgt_lang": "es",
        },
        {
            "src_text": "thanks",
            "tgt_text": "gracias",
            "src_lang": "en",
            "tgt_lang": "es",
        },
    ]
    tokenizer = _CountingTokenizer()
    dataset = TranslationDataset(data, tokenizer, pretokenize=True)
    calls_after_construction = tokenizer.calls

    assert calls_after_construction == 2 * len(data)
    assert dataset[0] == dataset[0]
    assert dataset[1] == dataset[1]
    assert tokenizer.calls == calls_after_construction


def test_train_step_can_defer_host_metric_sync(tmp_path) -> None:
    model = _small_model(vocab_size=64)
    batch = {
        "src_input_ids": torch.randint(1, 64, (2, 7)),
        "tgt_input_ids": torch.randint(1, 64, (2, 6)),
        "src_attention_mask": torch.ones(2, 7),
        "tgt_attention_mask": torch.ones(2, 6),
    }
    trainer = TranslationTrainer(
        model=model,
        train_loader=[batch],
        max_steps=4,
        warmup_steps=1,
        device="cpu",
        save_dir=str(tmp_path),
        loss_chunk_size=4,
    )

    loss, metrics = trainer.train_step(batch, sync_metrics=False)
    assert loss == 0.0
    assert metrics["loss"] == 0.0
    assert trainer._deferred_loss_count == 1
    assert trainer._consume_deferred_loss() > 0.0
    assert trainer._deferred_loss_count == 0


def test_negative_loss_chunk_size_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="loss_chunk_size"):
        TranslationTrainer(
            model=_small_model(vocab_size=64),
            train_loader=[],
            max_steps=4,
            warmup_steps=1,
            device="cpu",
            save_dir=str(tmp_path),
            loss_chunk_size=-1,
        )


def test_beam_search_candidate_bookkeeping_has_no_cpu_roundtrip() -> None:
    source = inspect.getsource(_generate_with_beam_search_impl)
    assert ".cpu()" not in source
    assert ".tolist()" not in source


def test_cross_attention_shares_source_kv_across_beams() -> None:
    torch.manual_seed(0)
    batch_size, num_beams = 2, 3
    attention = GroupedQueryAttention(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        is_cross_attn=True,
    ).eval()
    query = torch.randn(batch_size * num_beams, 2, 32)
    source = torch.randn(batch_size, 5, 32)
    source_mask = torch.ones(batch_size, 5, dtype=torch.bool)
    cache = KVCache()

    shared, updated_cache = attention(
        query,
        key=source,
        value=source,
        attention_mask=source_mask,
        kv_cache=cache,
        use_cache=True,
    )
    duplicated, _ = attention(
        query,
        key=source.repeat_interleave(num_beams, dim=0),
        value=source.repeat_interleave(num_beams, dim=0),
        attention_mask=source_mask.repeat_interleave(num_beams, dim=0),
        use_cache=False,
    )

    torch.testing.assert_close(shared, duplicated, rtol=1e-5, atol=1e-6)
    assert updated_cache is cache
    assert cache.key is not None
    assert cache.key.shape[0] == batch_size
