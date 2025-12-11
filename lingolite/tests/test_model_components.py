import sys
from pathlib import Path

import pytest
import torch

# Ensure package imports work when running from the repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lingolite.generation_utils import KVCache  # noqa: E402
from lingolite.model_components import GroupedQueryAttention, RotaryPositionEmbedding  # noqa: E402


def test_rope_even_dim_guard() -> None:
    with pytest.raises(ValueError):
        RotaryPositionEmbedding(dim=3)


def test_kvcache_update_enforces_heads_and_batch() -> None:
    cache = KVCache()
    key1 = torch.randn(2, 2, 3, 4)
    val1 = torch.randn(2, 2, 3, 4)
    cache.update(key1, val1)
    assert cache.num_heads == 2
    assert cache.head_dim == 4

    # Append compatible shapes
    cache.update(torch.randn(2, 2, 1, 4), torch.randn(2, 2, 1, 4))
    assert cache.get_seq_len() == 4

    # Head count, batch size, and head_dim mismatches should be rejected
    with pytest.raises(ValueError):
        cache.update(torch.randn(2, 3, 1, 4), torch.randn(2, 3, 1, 4))
    with pytest.raises(ValueError):
        cache.update(torch.randn(1, 2, 1, 4), torch.randn(1, 2, 1, 4))
    with pytest.raises(ValueError):
        cache.update(torch.randn(2, 2, 1, 5), torch.randn(2, 2, 1, 5))


def test_kvcache_to_moves_dtype() -> None:
    cache = KVCache()
    key = torch.randn(1, 1, 1, 2, dtype=torch.float32)
    val = torch.randn(1, 1, 1, 2, dtype=torch.float32)
    cache.update(key, val)
    cache.to(torch.device("cpu"), dtype=torch.float16)
    assert cache.key is not None and cache.key.dtype == torch.float16
    assert cache.value is not None and cache.value.dtype == torch.float16


def test_gqa_cache_shapes_and_nan_safe() -> None:
    gqa = GroupedQueryAttention(
        d_model=8,
        n_heads=4,
        n_kv_heads=2,
        dropout=0.0,
        is_causal=True,
    )
    query = torch.zeros(1, 1, 8)
    attn_mask = torch.full((1, 1, 1, 1), float("-inf"))

    output, cache = gqa(query, attention_mask=attn_mask, use_cache=True)
    assert cache is not None
    assert cache.num_heads == 2
    assert cache.head_dim == 2
    assert cache.key is not None and cache.key.shape == (1, 2, 1, 2)
    assert torch.isfinite(output).all()

    # Second step with existing cache should append without NaNs
    q2 = torch.zeros(1, 1, 8)
    out2, cache2 = gqa(q2, kv_cache=cache, use_cache=True)
    assert cache2 is not None
    assert cache2.get_seq_len() == 2
    assert torch.isfinite(out2).all()
