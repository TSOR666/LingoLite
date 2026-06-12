"""Regression tests for the deep-learning bottleneck audit fixes.

Covers:
- checkpoint save/load roundtrip under torch.load(weights_only=True) with
  OneCycleLR (whose state_dict contains a bound method on torch < 2.4),
- picklable collate_fn for DataLoader workers on spawn platforms,
- one-time SDPA enable_gqa capability probe + GQA numerical parity,
- KV cache reorder: in-place prefix gather in reserved mode, cross-attention
  cache skipping in beam search,
- generate() restoring the model's training mode.
"""

from __future__ import annotations

import functools
import math
import pickle

import pytest
import torch
import torch.nn.functional as F

from lingolite import generation_utils
from lingolite.generation_utils import KVCache, LayerKVCache
from lingolite.mobile_translation_model import create_model
from lingolite.model_components import _SDPA_SUPPORTS_GQA, GroupedQueryAttention
from lingolite.training import TranslationTrainer, collate_fn


def _tiny_model(vocab_size: int = 64):
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


# ---------------------------------------------------------------------------
# Checkpoint roundtrip (BLOCKER fix)
# ---------------------------------------------------------------------------

class TestCheckpointWeightsOnlySafe:
    def _make_trainer(self, tmp_path, max_steps: int) -> TranslationTrainer:
        return TranslationTrainer(
            model=_tiny_model(),
            train_loader=None,  # not needed for save/load
            val_loader=None,
            max_steps=max_steps,
            warmup_steps=5,
            device="cpu",
            save_dir=str(tmp_path),
        )

    def test_onecycle_checkpoint_loads_with_weights_only(self, tmp_path) -> None:
        # max_steps large enough to select the OneCycleLR branch.
        trainer = self._make_trainer(tmp_path, max_steps=100)
        from torch.optim.lr_scheduler import OneCycleLR

        assert isinstance(trainer.scheduler, OneCycleLR)
        trainer.save_checkpoint("ckpt.pt")

        # The save must be loadable with weights_only=True on every supported
        # torch version (OneCycleLR.state_dict() contains a bound method on
        # torch < 2.4 which weights_only refuses).
        checkpoint = torch.load(tmp_path / "ckpt.pt", map_location="cpu", weights_only=True)
        assert "model_state_dict" in checkpoint
        assert not any(callable(v) for v in checkpoint["scheduler_state_dict"].values())

    def test_scheduler_state_restores_and_steps(self, tmp_path) -> None:
        trainer = self._make_trainer(tmp_path, max_steps=100)
        # Advance the schedule a few steps so there is real state to restore.
        for _ in range(3):
            trainer.optimizer.step()
            trainer.scheduler.step()
        lr_before = trainer.scheduler.get_last_lr()[0]
        trainer.global_step = 3
        trainer.save_checkpoint("ckpt.pt")

        fresh = self._make_trainer(tmp_path, max_steps=100)
        fresh.load_checkpoint("ckpt.pt")
        assert fresh.global_step == 3
        assert fresh.scheduler.get_last_lr()[0] == pytest.approx(lr_before)
        # The restored scheduler must still be steppable (anneal_func intact).
        fresh.optimizer.step()
        fresh.scheduler.step()


# ---------------------------------------------------------------------------
# DataLoader collate picklability (Windows/macOS spawn workers)
# ---------------------------------------------------------------------------

def test_collate_partial_is_picklable_and_works() -> None:
    collate = functools.partial(collate_fn, pad_token_id=0)
    restored = pickle.loads(pickle.dumps(collate))
    batch = [
        {
            "src_input_ids": [4, 5, 6],
            "tgt_input_ids": [1, 7, 2],
            "src_attention_mask": [1, 1, 1],
            "tgt_attention_mask": [1, 1, 1],
        },
        {
            "src_input_ids": [4, 5],
            "tgt_input_ids": [1, 2],
            "src_attention_mask": [1, 1],
            "tgt_attention_mask": [1, 1],
        },
    ]
    out = restored(batch)
    assert out["src_input_ids"].shape == (2, 3)
    assert out["src_attention_mask"][1].tolist() == [1.0, 1.0, 0.0]


# ---------------------------------------------------------------------------
# SDPA enable_gqa probe (HIGH speed fix)
# ---------------------------------------------------------------------------

class TestSdpaGqaProbe:
    def test_probe_matches_runtime_capability(self) -> None:
        q = torch.zeros(1, 2, 1, 2)
        kv = torch.zeros(1, 1, 1, 2)
        try:
            F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
            supported = True
        except TypeError:
            supported = False
        assert _SDPA_SUPPORTS_GQA == supported

    def test_gqa_forward_matches_manual_reference(self) -> None:
        torch.manual_seed(0)
        attn = GroupedQueryAttention(d_model=32, n_heads=4, n_kv_heads=2, dropout=0.0)
        attn.eval()
        x = torch.randn(2, 5, 32)
        out, _ = attn(x)

        # Hand-rolled reference: explicit head repeat + softmax attention.
        B, L, _ = x.shape
        Q = attn.q_proj(x).view(B, L, 4, 8).transpose(1, 2)
        K = attn.k_proj(x).view(B, L, 2, 8).transpose(1, 2).repeat_interleave(2, dim=1)
        V = attn.v_proj(x).view(B, L, 2, 8).transpose(1, 2).repeat_interleave(2, dim=1)
        scores = Q @ K.transpose(-2, -1) / math.sqrt(8)
        ref = attn.o_proj((scores.softmax(dim=-1) @ V).transpose(1, 2).reshape(B, L, 32))
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# KV cache reorder (HIGH speed fix)
# ---------------------------------------------------------------------------

class TestKVCacheReorder:
    def test_reserved_reorder_keeps_capacity_semantics(self) -> None:
        cache = KVCache().reserve(
            batch=2, heads=2, max_len=8, head_dim=4, device=torch.device("cpu"), dtype=torch.float32
        )
        k1, v1 = torch.randn(2, 2, 3, 4), torch.randn(2, 2, 3, 4)
        cache.update(k1, v1)

        cache.reorder(torch.tensor([1, 0]))
        # Valid prefix permuted...
        torch.testing.assert_close(cache.key, k1[[1, 0]])
        torch.testing.assert_close(cache.value, v1[[1, 0]])
        # ...and reserved-mode capacity survives the reorder.
        assert cache._capacity == 8
        assert cache._key_buf.shape[2] == 8

        # Subsequent updates still append into the reserved region without cat.
        k2, v2 = torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4)
        cache.update(k2, v2)
        assert cache.get_seq_len() == 4
        assert cache._capacity == 8
        torch.testing.assert_close(cache.key[:, :, 3:, :], k2)

    def test_lazy_reorder_unchanged(self) -> None:
        cache = KVCache()
        k = torch.randn(2, 2, 3, 4)
        v = torch.randn(2, 2, 3, 4)
        cache.update(k, v)
        cache.reorder(torch.tensor([1, 0]))
        torch.testing.assert_close(cache.key, k[[1, 0]])

    def test_layer_reorder_self_only_keeps_cross_cache(self) -> None:
        layer = LayerKVCache()
        self_k = torch.randn(2, 2, 3, 4)
        cross_k = torch.randn(2, 2, 5, 4)
        layer.self_attn_cache.update(self_k, self_k.clone())
        layer.cross_attn_cache.update(cross_k, cross_k.clone())

        layer.reorder(torch.tensor([1, 0]), self_only=True)
        torch.testing.assert_close(layer.self_attn_cache.key, self_k[[1, 0]])
        # Cross-attention cache must be untouched.
        torch.testing.assert_close(layer.cross_attn_cache.key, cross_k)

    def test_beam_search_output_identical_with_full_reorder(self, monkeypatch) -> None:
        """self_only reordering must not change beam search results."""
        model = _tiny_model()
        model.eval()
        torch.manual_seed(1)
        src = torch.randint(4, 64, (2, 7))
        mask = torch.ones(2, 7)

        out_self_only = model.generate_beam(src, mask, max_length=12, num_beams=3)

        original = generation_utils._reorder_past_key_values

        def force_full_reorder(past, idx, self_only=False):
            return original(past, idx, self_only=False)

        monkeypatch.setattr(generation_utils, "_reorder_past_key_values", force_full_reorder)
        out_full = model.generate_beam(src, mask, max_length=12, num_beams=3)
        assert torch.equal(out_self_only, out_full)


# ---------------------------------------------------------------------------
# generate() training-mode restoration (HIGH correctness fix)
# ---------------------------------------------------------------------------

class TestGenerateRestoresMode:
    def test_greedy_restores_train_mode(self) -> None:
        model = _tiny_model()
        model.train()
        src = torch.randint(4, 64, (1, 5))
        model.generate(src, torch.ones(1, 5), max_length=8)
        assert model.training, "generate() must restore the caller's training mode"

    def test_beam_restores_train_mode(self) -> None:
        model = _tiny_model()
        model.train()
        src = torch.randint(4, 64, (1, 5))
        model.generate_beam(src, torch.ones(1, 5), max_length=8, num_beams=2)
        assert model.training

    def test_eval_mode_stays_eval(self) -> None:
        model = _tiny_model()
        model.eval()
        src = torch.randint(4, 64, (1, 5))
        model.generate(src, torch.ones(1, 5), max_length=8)
        assert not model.training
