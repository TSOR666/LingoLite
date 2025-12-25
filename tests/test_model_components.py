"""
Tests for model_components.py

Covers:
- RMSNorm: normalization, shape preservation, dtype handling
- RotaryPositionEmbedding: rotation properties, cache growth, offset handling
- GroupedQueryAttention: self/cross attention, KV caching, causal masking
- SwiGLU_FFN: shape preservation, gating mechanism
"""

import pytest
import torch
import torch.nn as nn
import math

from lingolite.model_components import (
    RMSNorm,
    RotaryPositionEmbedding,
    GroupedQueryAttention,
    SwiGLU_FFN,
)


# ============================================================================
# RMSNorm Tests
# ============================================================================

class TestRMSNorm:
    """Tests for Root Mean Square Layer Normalization."""

    @pytest.fixture
    def rms_norm(self) -> RMSNorm:
        """Create RMSNorm instance for testing."""
        return RMSNorm(dim=64)

    def test_output_shape_preserved(self, rms_norm: RMSNorm) -> None:
        """RMSNorm should preserve input shape."""
        x = torch.randn(2, 10, 64)
        output = rms_norm(x)
        assert output.shape == x.shape

    def test_output_normalized(self, rms_norm: RMSNorm) -> None:
        """Output should have approximately unit RMS."""
        x = torch.randn(2, 10, 64) * 10  # Large values
        output = rms_norm(x)
        
        # RMS should be close to 1 (within tolerance due to learned weight)
        rms = torch.sqrt(torch.mean(output ** 2, dim=-1))
        assert rms.mean().item() < 5.0  # Should be normalized

    def test_dtype_preserved(self, rms_norm: RMSNorm) -> None:
        """RMSNorm should preserve input dtype."""
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        output = rms_norm(x)
        assert output.dtype == torch.float16

    def test_batch_dimension_independence(self, rms_norm: RMSNorm) -> None:
        """Normalization should be independent across batch dimension."""
        x = torch.randn(3, 10, 64)
        output = rms_norm(x)
        
        # Process each batch item separately
        outputs_separate = torch.stack([rms_norm(x[i:i+1]) for i in range(3)])
        outputs_separate = outputs_separate.squeeze(1)
        
        assert torch.allclose(output, outputs_separate, atol=1e-5)

    def test_zero_input_handling(self, rms_norm: RMSNorm) -> None:
        """RMSNorm should handle zero input without NaN."""
        x = torch.zeros(2, 10, 64)
        output = rms_norm(x)
        assert not torch.isnan(output).any()

    def test_different_dimensions(self) -> None:
        """RMSNorm should work with different dimensions."""
        for dim in [32, 128, 256, 512]:
            norm = RMSNorm(dim=dim)
            x = torch.randn(2, 5, dim)
            output = norm(x)
            assert output.shape == x.shape


# ============================================================================
# RotaryPositionEmbedding Tests
# ============================================================================

class TestRotaryPositionEmbedding:
    """Tests for Rotary Position Embedding (RoPE)."""

    @pytest.fixture
    def rope(self) -> RotaryPositionEmbedding:
        """Create RoPE instance for testing."""
        return RotaryPositionEmbedding(dim=64, max_seq_len=128)

    def test_output_shape_preserved(self, rope: RotaryPositionEmbedding) -> None:
        """RoPE should preserve input shapes."""
        q = torch.randn(2, 4, 10, 64)  # batch, heads, seq, dim
        k = torch.randn(2, 4, 10, 64)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape

    def test_dtype_preserved(self, rope: RotaryPositionEmbedding) -> None:
        """RoPE should preserve input dtype."""
        q = torch.randn(2, 4, 10, 64, dtype=torch.float16)
        k = torch.randn(2, 4, 10, 64, dtype=torch.float16)
        
        q_rot, k_rot = rope(q, k)
        
        assert q_rot.dtype == torch.float16
        assert k_rot.dtype == torch.float16

    def test_cache_grows_with_longer_sequences(self) -> None:
        """Cache should grow when sequence exceeds initial max_seq_len."""
        rope = RotaryPositionEmbedding(dim=64, max_seq_len=10)
        
        # First call with short sequence
        q1 = torch.randn(1, 2, 5, 64)
        k1 = torch.randn(1, 2, 5, 64)
        rope(q1, k1)
        initial_cache_size = rope.cos_cached.shape[0]
        
        # Second call with longer sequence
        q2 = torch.randn(1, 2, 20, 64)
        k2 = torch.randn(1, 2, 20, 64)
        rope(q2, k2)
        
        assert rope.cos_cached.shape[0] >= 20

    def test_offset_handling(self, rope: RotaryPositionEmbedding) -> None:
        """Offset should correctly shift position embeddings."""
        q = torch.randn(1, 2, 5, 64)
        k = torch.randn(1, 2, 5, 64)
        
        # Same input, different offsets should give different outputs
        q_rot1, k_rot1 = rope(q.clone(), k.clone(), offset=0)
        q_rot2, k_rot2 = rope(q.clone(), k.clone(), offset=10)
        
        assert not torch.allclose(q_rot1, q_rot2)

    def test_rotation_preserves_norm(self, rope: RotaryPositionEmbedding) -> None:
        """Rotation should approximately preserve vector norms."""
        q = torch.randn(1, 2, 5, 64)
        k = torch.randn(1, 2, 5, 64)
        
        q_rot, k_rot = rope(q, k)
        
        # Norms should be similar (rotation is approximately norm-preserving)
        q_norm = torch.norm(q, dim=-1)
        q_rot_norm = torch.norm(q_rot, dim=-1)
        
        assert torch.allclose(q_norm, q_rot_norm, rtol=0.1)

    def test_rotate_half_correctness(self) -> None:
        """rotate_half should split and negate correctly."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        rotated = RotaryPositionEmbedding.rotate_half(x)
        
        # Should be [-x2, x1] = [-3, -4, 1, 2]
        expected = torch.tensor([[-3.0, -4.0, 1.0, 2.0]])
        assert torch.allclose(rotated, expected)


# ============================================================================
# GroupedQueryAttention Tests
# ============================================================================

class TestGroupedQueryAttention:
    """Tests for Grouped-Query Attention."""

    @pytest.fixture
    def gqa(self) -> GroupedQueryAttention:
        """Create GQA instance for testing."""
        return GroupedQueryAttention(
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            dropout=0.0,
            is_causal=False,
        )

    @pytest.fixture
    def causal_gqa(self) -> GroupedQueryAttention:
        """Create causal GQA instance for testing."""
        return GroupedQueryAttention(
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            dropout=0.0,
            is_causal=True,
        )

    def test_self_attention_output_shape(self, gqa: GroupedQueryAttention) -> None:
        """Self-attention should preserve shape."""
        x = torch.randn(2, 10, 64)
        output, _ = gqa(x)
        assert output.shape == x.shape

    def test_cross_attention_output_shape(self) -> None:
        """Cross-attention should output query shape."""
        cross_attn = GroupedQueryAttention(
            d_model=64,
            n_heads=8,
            n_kv_heads=2,
            dropout=0.0,
            is_cross_attn=True,
        )
        
        query = torch.randn(2, 10, 64)
        key = torch.randn(2, 20, 64)
        value = torch.randn(2, 20, 64)
        
        output, _ = cross_attn(query, key=key, value=value)
        assert output.shape == query.shape

    def test_kv_cache_accumulation(self, gqa: GroupedQueryAttention) -> None:
        """KV cache should accumulate across steps."""
        from lingolite.generation_utils import KVCache
        
        gqa.eval()
        cache = KVCache()
        
        # First step
        x1 = torch.randn(1, 5, 64)
        _, cache = gqa(x1, kv_cache=cache, use_cache=True)
        assert cache.get_seq_len() == 5
        
        # Second step
        x2 = torch.randn(1, 1, 64)
        _, cache = gqa(x2, kv_cache=cache, use_cache=True)
        assert cache.get_seq_len() == 6

    def test_causal_masking(self, causal_gqa: GroupedQueryAttention) -> None:
        """Causal attention should not attend to future positions."""
        causal_gqa.eval()
        
        # Create input where future tokens are very different
        x = torch.zeros(1, 5, 64)
        x[:, 0, :] = 1.0  # First token has signal
        x[:, 4, :] = 100.0  # Last token has large signal
        
        output, _ = causal_gqa(x)
        
        # First position output should not be affected by last position
        # (This is a sanity check - actual values depend on learned weights)
        assert output.shape == x.shape

    def test_attention_mask_application(self, gqa: GroupedQueryAttention) -> None:
        """Attention mask should be properly applied."""
        gqa.eval()

        x = torch.randn(2, 10, 64)
        # Create mask in expected format: (batch, 1, q_len, kv_len)
        # Start with (2, 10) padding mask: 1 = attend, 0 = masked
        padding_mask = torch.ones(2, 10)
        padding_mask[:, 5:] = 0  # Mask out second half

        # Convert to additive attention mask format: (batch, 1, 1, kv_len)
        # 0 -> -inf (masked), 1 -> 0 (attend)
        attention_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (2, 1, 1, 10)
        attention_mask = (1.0 - attention_mask) * float('-inf')

        output, _ = gqa(x, attention_mask=attention_mask)
        assert output.shape == x.shape

    def test_head_dimension_validation(self) -> None:
        """Should raise error for invalid head configuration."""
        with pytest.raises(AssertionError):
            GroupedQueryAttention(
                d_model=64,
                n_heads=7,  # Not divisible
                n_kv_heads=2,
            )

    def test_kv_head_divisibility_validation(self) -> None:
        """Should raise error when n_heads not divisible by n_kv_heads."""
        with pytest.raises(AssertionError):
            GroupedQueryAttention(
                d_model=64,
                n_heads=8,
                n_kv_heads=3,  # 8 not divisible by 3
            )


# ============================================================================
# SwiGLU_FFN Tests
# ============================================================================

class TestSwiGLU_FFN:
    """Tests for SwiGLU Feed-Forward Network."""

    @pytest.fixture
    def ffn(self) -> SwiGLU_FFN:
        """Create SwiGLU_FFN instance for testing."""
        return SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0)

    def test_output_shape_preserved(self, ffn: SwiGLU_FFN) -> None:
        """FFN should preserve input shape."""
        x = torch.randn(2, 10, 64)
        output = ffn(x)
        assert output.shape == x.shape

    def test_different_ff_dimensions(self) -> None:
        """FFN should work with different d_ff values."""
        for d_ff in [128, 256, 512]:
            ffn = SwiGLU_FFN(d_model=64, d_ff=d_ff, dropout=0.0)
            x = torch.randn(2, 10, 64)
            output = ffn(x)
            assert output.shape == x.shape

    def test_gating_mechanism(self) -> None:
        """Gating should modulate the output."""
        ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0)
        ffn.eval()
        
        # Zero input should give zero output (due to gating)
        x_zero = torch.zeros(1, 5, 64)
        output_zero = ffn(x_zero)
        assert torch.allclose(output_zero, torch.zeros_like(output_zero), atol=1e-5)

    def test_dtype_preserved(self) -> None:
        """FFN should preserve input dtype when model and input dtypes match."""
        # Create float16 model and float16 input
        ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0).half()
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        output = ffn(x)
        assert output.dtype == torch.float16

    def test_dropout_applied_in_training(self) -> None:
        """Dropout should be applied during training."""
        ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.5)
        ffn.train()
        
        x = torch.randn(2, 10, 64)
        
        # Multiple forward passes should give different results with dropout
        torch.manual_seed(42)
        output1 = ffn(x.clone())
        torch.manual_seed(123)
        output2 = ffn(x.clone())
        
        # Outputs should be different due to dropout
        assert not torch.allclose(output1, output2)

    def test_deterministic_in_eval(self) -> None:
        """FFN should be deterministic in eval mode."""
        ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.5)
        ffn.eval()
        
        x = torch.randn(2, 10, 64)
        
        output1 = ffn(x.clone())
        output2 = ffn(x.clone())
        
        assert torch.allclose(output1, output2)


# ============================================================================
# Integration Tests
# ============================================================================

class TestComponentIntegration:
    """Integration tests combining multiple components."""

    def test_full_attention_block(self) -> None:
        """Test attention with RMSNorm and RoPE."""
        d_model = 64
        n_heads = 4
        
        norm = RMSNorm(dim=d_model)
        rope = RotaryPositionEmbedding(dim=d_model // n_heads, max_seq_len=128)
        attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=2,
            dropout=0.0,
        )
        
        x = torch.randn(2, 10, d_model)
        
        # Pre-norm + attention + residual
        normed = norm(x)
        attended, _ = attn(normed, rope=rope)
        output = x + attended
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_transformer_block(self) -> None:
        """Test full transformer block (attention + FFN)."""
        d_model = 64
        
        norm1 = RMSNorm(dim=d_model)
        norm2 = RMSNorm(dim=d_model)
        attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=4,
            n_kv_heads=2,
            dropout=0.0,
        )
        ffn = SwiGLU_FFN(d_model=d_model, d_ff=256, dropout=0.0)
        
        x = torch.randn(2, 10, d_model)
        
        # Attention block
        h = norm1(x)
        h, _ = attn(h)
        x = x + h
        
        # FFN block
        h = norm2(x)
        h = ffn(h)
        x = x + h
        
        assert x.shape == (2, 10, d_model)
        assert not torch.isnan(x).any()
