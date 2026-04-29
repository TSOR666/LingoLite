"""Tests for the cache-aware ONNX decoder wrapper.

These run entirely in PyTorch (no ONNX runtime required), exercising the
wrapper's input/output contract and verifying that its outputs match what
the model's decoder produces when fed the same cache state directly. The
goal is to catch regressions in the cache-tensor packing / unpacking that
sit between the decoder's Python-object API and the flat-tensor API ONNX
needs - the actual ``torch.onnx.export`` call is best-effort and will be
skipped when ``onnx``/``onnxruntime`` aren't installed.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from lingolite.generation_utils import KVCache, LayerKVCache
from lingolite.mobile_translation_model import create_model
from scripts.export_onnx import CachedDecoderWrapper
from tests.tmp_utils import writable_tmp_dir


@pytest.fixture(scope="module")
def small_model_and_dims() -> tuple:
    torch.manual_seed(0)
    vocab_size = 64
    model = create_model(vocab_size=vocab_size, model_size="tiny")
    model.eval()

    decoder_layers = model.decoder._decoder_layers
    n_layers = len(decoder_layers)
    n_kv_heads = int(decoder_layers[0].self_attn.n_kv_heads)
    head_dim = int(decoder_layers[0].self_attn.head_dim)
    d_model = int(model.d_model)

    return model, vocab_size, n_layers, n_kv_heads, head_dim, d_model


class TestCachedDecoderWrapperShapes:
    def test_output_shapes_match_contract(self, small_model_and_dims: tuple) -> None:
        model, vocab_size, n_layers, n_kv_heads, head_dim, d_model = small_model_and_dims
        wrapper = CachedDecoderWrapper(model.decoder, n_layers=n_layers)
        wrapper.eval()

        batch, src_len, past_len = 2, 7, 3
        input_ids = torch.randint(0, vocab_size, (batch, 1), dtype=torch.long)
        encoder_output = torch.randn(batch, src_len, d_model)
        cross_mask = torch.ones(batch, src_len, dtype=torch.bool)
        past_kvs: List[torch.Tensor] = []
        for _ in range(n_layers):
            past_kvs.append(torch.randn(batch, n_kv_heads, past_len, head_dim))
            past_kvs.append(torch.randn(batch, n_kv_heads, past_len, head_dim))

        with torch.inference_mode():
            outputs = wrapper(input_ids, encoder_output, cross_mask, *past_kvs)

        # 1 logits tensor + 2 (key, value) per layer.
        assert len(outputs) == 1 + 2 * n_layers
        logits, *kv_outputs = outputs

        assert logits.shape == (batch, 1, vocab_size)
        for i in range(n_layers):
            new_k = kv_outputs[2 * i]
            new_v = kv_outputs[2 * i + 1]
            # Past length grew by exactly 1 (the new token).
            assert new_k.shape == (batch, n_kv_heads, past_len + 1, head_dim)
            assert new_v.shape == (batch, n_kv_heads, past_len + 1, head_dim)

    def test_wrong_number_of_past_kvs_raises(self, small_model_and_dims: tuple) -> None:
        model, vocab_size, n_layers, n_kv_heads, head_dim, d_model = small_model_and_dims
        wrapper = CachedDecoderWrapper(model.decoder, n_layers=n_layers)

        batch, src_len, past_len = 1, 4, 2
        input_ids = torch.randint(0, vocab_size, (batch, 1), dtype=torch.long)
        encoder_output = torch.randn(batch, src_len, d_model)
        cross_mask = torch.ones(batch, src_len, dtype=torch.bool)

        # Only one layer's worth of KVs - should be rejected.
        bad_past = [
            torch.randn(batch, n_kv_heads, past_len, head_dim),
            torch.randn(batch, n_kv_heads, past_len, head_dim),
        ]
        with pytest.raises(ValueError, match="expected"):
            wrapper(input_ids, encoder_output, cross_mask, *bad_past)


class TestCachedDecoderWrapperEquivalence:
    """The wrapper must produce the same logits as a direct decoder call.

    We construct the equivalent ``LayerKVCache`` list by hand, run the
    decoder directly with ``use_cache=True``, and compare the logits against
    the wrapper's first output. They must be bit-identical because both
    paths execute the same arithmetic - the wrapper just packs/unpacks the
    cache tensors.
    """

    def test_logits_match_direct_decoder(self, small_model_and_dims: tuple) -> None:
        model, vocab_size, n_layers, n_kv_heads, head_dim, d_model = small_model_and_dims
        wrapper = CachedDecoderWrapper(model.decoder, n_layers=n_layers)
        wrapper.eval()

        batch, src_len, past_len = 2, 5, 3
        torch.manual_seed(123)
        input_ids = torch.randint(0, vocab_size, (batch, 1), dtype=torch.long)
        encoder_output = torch.randn(batch, src_len, d_model)
        cross_mask = torch.ones(batch, src_len, dtype=torch.bool)

        # Build matched past KV tensors and a parallel set of LayerKVCache
        # objects. Both paths must see the same starting state.
        past_kvs: List[torch.Tensor] = []
        layer_caches_direct: List[LayerKVCache] = []
        for _ in range(n_layers):
            pk = torch.randn(batch, n_kv_heads, past_len, head_dim)
            pv = torch.randn(batch, n_kv_heads, past_len, head_dim)
            past_kvs.extend([pk, pv])
            cache = LayerKVCache()
            cache.self_attn_cache = KVCache(key=pk.clone(), value=pv.clone())
            layer_caches_direct.append(cache)

        with torch.inference_mode():
            wrapper_outputs = wrapper(input_ids, encoder_output, cross_mask, *past_kvs)
            wrapper_logits = wrapper_outputs[0]

            direct_logits, _ = model.decoder(
                input_ids=input_ids,
                encoder_output=encoder_output,
                self_attention_mask=torch.ones(batch, 1, dtype=torch.bool),
                cross_attention_mask=cross_mask,
                past_key_values=layer_caches_direct,
                use_cache=True,
            )

        assert torch.allclose(wrapper_logits, direct_logits, atol=1e-6), (
            "wrapper logits diverged from a direct decoder call"
        )


class TestOnnxExport:
    """Best-effort check that ``torch.onnx.export`` can trace the wrapper.

    Skipped when ``onnx``/``onnxruntime`` aren't installed - this isn't a
    hard project dep, and the wrapper itself is the load-bearing piece.
    """

    def test_export_smoke(self, small_model_and_dims: tuple) -> None:
        pytest.importorskip("onnx")
        from scripts.export_onnx import export_cached_decoder_to_onnx

        # Project-local tmp dir to dodge the stale Windows %TEMP% perms that
        # break pytest's ``tmp_path`` on this machine.
        with writable_tmp_dir("onnx_export_") as out_dir:
            model, *_ = small_model_and_dims
            out_path = out_dir / "decoder_cached.onnx"
            export_cached_decoder_to_onnx(
                model=model,
                output_path=out_path,
                src_len=8,
                past_len=1,
            )
            assert out_path.exists()
            assert out_path.stat().st_size > 0
