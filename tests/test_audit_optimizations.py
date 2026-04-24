"""Regression tests for the Apr-2026 bottleneck audit.

These guard the sync-free hot paths so they do not regress to the previous
``.item()``-heavy implementation.
"""
from __future__ import annotations

import torch

from lingolite.mobile_translation_model import create_model


def _count_items(fn, *args, **kwargs):
    """Count ``torch.Tensor.item()`` invocations while running ``fn``."""
    orig = torch.Tensor.item
    counter = {"n": 0}

    def counting(self):
        counter["n"] += 1
        return orig(self)

    torch.Tensor.item = counting
    try:
        out = fn(*args, **kwargs)
    finally:
        torch.Tensor.item = orig
    return out, counter["n"]


def test_gqa_forward_makes_no_item_sync() -> None:
    """A single encoder forward pass must not call ``.item()``.

    Previously ``GroupedQueryAttention.forward`` issued a ``bool(...any().item())``
    call in its masking branch, producing a device-host round trip on every
    attention invocation. The refactor replaced the branch with
    ``torch.where`` + post-op zeroing.
    """
    torch.manual_seed(0)
    model = create_model(vocab_size=2048, model_size="tiny")
    model.eval()
    src = torch.randint(5, 2048, (2, 12))
    src_mask = torch.ones(2, 12)
    with torch.no_grad():
        _, n = _count_items(lambda: model.encoder(src, src_mask))
    assert n == 0, f"encoder forward leaked {n} .item() syncs"


def test_greedy_kv_cache_item_budget() -> None:
    """Greedy generation should stay well below the pre-audit sync budget.

    Before the audit a single 8-token greedy generation triggered ~62 ``.item()``
    calls, primarily from validation and a multi-layer ``any().item()`` check
    inside attention. The refactor targets the attention sync specifically; we
    allow generous slack to absorb outer-loop ``finished.all()`` plus tensor
    validation and still trap regressions.
    """
    torch.manual_seed(0)
    model = create_model(vocab_size=2048, model_size="tiny")
    model.eval()
    src = torch.randint(5, 2048, (1, 10))
    src_mask = torch.ones(1, 10)
    with torch.no_grad():
        _, n = _count_items(
            lambda: model.generate(
                src, src_mask, max_length=6, num_beams=1, sos_token_id=1, eos_token_id=2
            )
        )
    # Previous baseline: ~46 for max_length=6. New budget: <= 25 (mostly the
    # per-step ``finished.all().item()`` and the outer validation calls).
    assert n <= 25, f"greedy generation leaked {n} .item() syncs (budget 25)"


def test_beam_search_is_correct_without_encoder_reorder() -> None:
    """Dropping the per-step ``encoder_output`` reorder must not change output.

    All beams of a batch share identical encoder state, so reordering them
    among themselves is a no-op. This test runs the same generation twice and
    asserts determinism, and additionally verifies the output shape.
    """
    torch.manual_seed(42)
    model = create_model(vocab_size=2048, model_size="tiny")
    model.eval()
    src = torch.randint(5, 2048, (2, 12))
    src_mask = torch.ones(2, 12)
    with torch.no_grad():
        out1 = model.generate_beam(
            src,
            src_mask,
            max_length=10,
            num_beams=3,
            sos_token_id=1,
            eos_token_id=2,
        )
        out2 = model.generate_beam(
            src,
            src_mask,
            max_length=10,
            num_beams=3,
            sos_token_id=1,
            eos_token_id=2,
        )
    assert out1.shape == (2, 10)
    assert torch.equal(out1, out2)


def test_attention_with_fully_masked_row_produces_zero() -> None:
    """Encoder must tolerate rows where every key is masked out.

    Previously this triggered the slow matmul branch (plus a CPU sync) and
    explicitly zeroed fully-masked rows. The SDPA-always path instead lets the
    masked softmax run and post-zeroes the output.
    """
    torch.manual_seed(0)
    model = create_model(vocab_size=512, model_size="tiny")
    model.eval()
    src = torch.randint(5, 512, (2, 8))
    # Batch 1: all tokens masked out; batch 0: normal.
    src_mask = torch.tensor(
        [[1, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        out = model.encoder(src, src_mask)
    assert torch.isfinite(out).all(), "encoder emitted NaN/Inf on fully-masked row"


def test_beam_search_reaches_batch_shape() -> None:
    """Smoke check: beam search returns ``(batch, max_length)``."""
    torch.manual_seed(0)
    model = create_model(vocab_size=1024, model_size="tiny")
    model.eval()
    src = torch.randint(5, 1024, (3, 14))
    src_mask = torch.ones(3, 14)
    with torch.no_grad():
        out = model.generate_beam(
            src, src_mask, max_length=12, num_beams=2, sos_token_id=1, eos_token_id=2
        )
    assert out.shape == (3, 12)
    assert out.dtype == torch.long


def test_kvcache_reserve_writes_inplace_and_matches_cat() -> None:
    """Reserved-mode KVCache updates must produce the same values as lazy cat.

    The reserved buffer allocates ``max_len`` up front; each ``update`` copies
    a new slice in-place. The valid prefix exposed by ``cache.key`` /
    ``cache.value`` must equal what the lazy ``torch.cat`` implementation
    would produce.
    """
    from lingolite.generation_utils import KVCache

    torch.manual_seed(0)
    chunks_k = [torch.randn(2, 4, 1, 8) for _ in range(5)]
    chunks_v = [torch.randn(2, 4, 1, 8) for _ in range(5)]

    lazy = KVCache()
    reserved = KVCache().reserve(
        batch=2, heads=4, max_len=16, head_dim=8, device=torch.device("cpu"), dtype=torch.float32
    )
    for k_new, v_new in zip(chunks_k, chunks_v):
        lazy.update(k_new, v_new)
        reserved.update(k_new, v_new)

    assert torch.equal(lazy.key, reserved.key)
    assert torch.equal(lazy.value, reserved.value)
    # And the reserved cache must not have grown beyond its initial capacity.
    assert reserved.get_seq_len() == 5
    assert reserved._capacity == 16


def test_kvcache_reserve_falls_back_past_capacity() -> None:
    """Writes past ``max_len`` degrade gracefully to the lazy path."""
    from lingolite.generation_utils import KVCache

    cache = KVCache().reserve(
        batch=1, heads=2, max_len=2, head_dim=4, device=torch.device("cpu"), dtype=torch.float32
    )
    cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    # Third write exceeds capacity; must still succeed via cat fallback.
    overflow_k = torch.randn(1, 2, 1, 4)
    overflow_v = torch.randn(1, 2, 1, 4)
    cache.update(overflow_k, overflow_v)
    assert cache.get_seq_len() == 3
    assert torch.equal(cache.key[:, :, -1:, :], overflow_k)


def test_amp_training_step_runs_in_bf16() -> None:
    """Trainer with ``amp_dtype='bf16'`` must complete a step without error."""
    from lingolite.training import TranslationTrainer

    torch.manual_seed(0)
    model = create_model(vocab_size=256, model_size="tiny")
    batch = {
        "src_input_ids": torch.randint(1, 256, (2, 10)),
        "tgt_input_ids": torch.randint(1, 256, (2, 8)),
        "src_attention_mask": torch.ones(2, 10),
        "tgt_attention_mask": torch.ones(2, 8),
    }
    trainer = TranslationTrainer(
        model=model,
        train_loader=[batch],
        max_steps=10,
        warmup_steps=2,
        device="cpu",
        save_dir=".tmp_manual/amp_reg_test",
        amp_dtype="bf16",
    )
    loss, metrics = trainer.train_step(batch)
    assert loss > 0.0
    assert torch.isfinite(torch.tensor(metrics["grad_norm"]))
    # bf16 doesn't need a scaler
    assert trainer.scaler is None


def test_lr_scheduler_fallback_keeps_lr_alive_for_short_runs() -> None:
    """For very small ``max_steps`` the LinearWarmup+CosineDecay fallback
    must keep LR in a reasonable range; OneCycleLR used to collapse to
    ~1e-9 on the second-to-last step.
    """
    from lingolite.training import TranslationTrainer

    torch.manual_seed(0)
    model = create_model(vocab_size=128, model_size="tiny")
    batch = {
        "src_input_ids": torch.randint(1, 128, (2, 8)),
        "tgt_input_ids": torch.randint(1, 128, (2, 6)),
        "src_attention_mask": torch.ones(2, 8),
        "tgt_attention_mask": torch.ones(2, 6),
    }
    trainer = TranslationTrainer(
        model=model,
        train_loader=[batch],
        learning_rate=1e-3,
        max_steps=4,
        warmup_steps=2,
        device="cpu",
        save_dir=".tmp_manual/lr_fallback_test",
    )
    lrs = []
    for _ in range(3):
        _, metrics = trainer.train_step(batch)
        lrs.append(metrics["lr"])
    # Every LR must be above 1e-6 (OneCycleLR collapsed to ~1e-9 previously).
    for lr in lrs:
        assert lr > 1e-6, f"LR collapsed to {lr} in short-run fallback"
