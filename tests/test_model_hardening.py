import torch
import pytest

from lingolite.model_components import RotaryPositionEmbedding
from lingolite.generation_utils import _apply_top_p_filter


def _randn_like(shape, dtype):
    """Helper to create deterministic dtype tensors on CPU."""
    return torch.randn(*shape, dtype=torch.float32).to(dtype)


def test_rope_preserves_dtype_and_expands_cache():
    rope = RotaryPositionEmbedding(dim=8, max_seq_len=2)

    q = _randn_like((1, 2, 3, 8), torch.float16)
    k = _randn_like((1, 2, 3, 8), torch.float16)

    # Force cache growth beyond initial max_seq_len
    q_rot, k_rot = rope(q, k, offset=3)

    assert q_rot.dtype == torch.float16
    assert k_rot.dtype == torch.float16
    assert q_rot.device == q.device
    assert rope.cos_cached.shape[0] >= 6  # seq_len (3) + offset (3)


def test_top_p_filter_masks_tail_tokens():
    logits = torch.tensor([[10.0, 1.0, 0.5, -1.0]])
    filtered = _apply_top_p_filter(logits.clone(), top_p=0.8)

    # Only the most likely token should remain unmasked
    assert filtered[0, 0] == pytest.approx(10.0)
    assert torch.isneginf(filtered[0, 1:]).all().item()


def test_top_p_filter_validation():
    logits = torch.tensor([[1.0, 0.0]])
    # Invalid values should raise
    for invalid in (-0.1, 0.0, 1.5):
        with pytest.raises(ValueError):
            _apply_top_p_filter(logits.clone(), top_p=invalid)

    # top_p == 1.0 should return logits untouched
    restored = _apply_top_p_filter(logits.clone(), top_p=1.0)
    assert torch.equal(logits, restored)
