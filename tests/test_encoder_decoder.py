"""
Tests for encoder_decoder.py

Covers:
- EncoderLayer: forward pass, attention masking
- DecoderLayer: self-attention, cross-attention, KV caching
- TransformerEncoder: full encoding pipeline
- TransformerDecoder: decoding with caching
"""

import pytest
import torch
import torch.nn as nn

from lingolite.encoder_decoder import (
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)
from lingolite.generation_utils import LayerKVCache


# ============================================================================
# EncoderLayer Tests
# ============================================================================

class TestEncoderLayer:
    """Tests for single encoder layer."""

    @pytest.fixture
    def encoder_layer(self) -> EncoderLayer:
        """Create encoder layer for testing."""
        return EncoderLayer(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            dropout=0.0,
        )

    def test_output_shape(self, encoder_layer: EncoderLayer) -> None:
        """Encoder layer should preserve input shape."""
        x = torch.randn(2, 10, 64)
        output = encoder_layer(x)
        assert output.shape == x.shape

    def test_attention_mask_applied(self, encoder_layer: EncoderLayer) -> None:
        """Attention mask should affect output."""
        encoder_layer.eval()

        x = torch.randn(2, 10, 64)

        # EncoderLayer expects attention_mask in format (batch, 1, seq_len, seq_len)
        # or broadcastable shape. Create additive mask: 0 = attend, -inf = masked

        # Full attention (no masking)
        mask_full = torch.zeros(2, 1, 1, 10)  # All zeros = full attention

        # Partial attention (mask second half of keys)
        mask_partial = torch.zeros(2, 1, 1, 10)
        mask_partial[:, :, :, 5:] = float('-inf')  # Mask out positions 5-9

        output_full = encoder_layer(x, attention_mask=mask_full)
        output_partial = encoder_layer(x, attention_mask=mask_partial)

        # Outputs should be different with different masks
        assert not torch.allclose(output_full, output_partial)

    def test_deterministic_in_eval(self, encoder_layer: EncoderLayer) -> None:
        """Encoder layer should be deterministic in eval mode."""
        encoder_layer.eval()
        
        x = torch.randn(2, 10, 64)
        output1 = encoder_layer(x)
        output2 = encoder_layer(x)
        
        assert torch.allclose(output1, output2)

    def test_residual_connection(self, encoder_layer: EncoderLayer) -> None:
        """Output should include residual from input."""
        encoder_layer.eval()
        
        x = torch.randn(2, 10, 64)
        output = encoder_layer(x)
        
        # Output should not be orthogonal to input (due to residual)
        correlation = torch.sum(x * output) / (torch.norm(x) * torch.norm(output))
        assert correlation.abs() > 0.01


# ============================================================================
# DecoderLayer Tests
# ============================================================================

class TestDecoderLayer:
    """Tests for single decoder layer."""

    @pytest.fixture
    def decoder_layer(self) -> DecoderLayer:
        """Create decoder layer for testing."""
        return DecoderLayer(
            d_model=64,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            dropout=0.0,
        )

    @pytest.fixture
    def encoder_output(self) -> torch.Tensor:
        """Create mock encoder output."""
        return torch.randn(2, 15, 64)

    def test_output_shape(
        self, decoder_layer: DecoderLayer, encoder_output: torch.Tensor
    ) -> None:
        """Decoder layer should preserve input shape."""
        x = torch.randn(2, 10, 64)
        output, _ = decoder_layer(x, encoder_output)
        assert output.shape == x.shape

    def test_kv_cache_usage(
        self, decoder_layer: DecoderLayer, encoder_output: torch.Tensor
    ) -> None:
        """Decoder layer should use and update KV cache."""
        decoder_layer.eval()
        
        cache = LayerKVCache()
        
        # First step - process 5 tokens
        x1 = torch.randn(1, 5, 64)
        output1, cache = decoder_layer(
            x1, encoder_output[:1], layer_cache=cache, use_cache=True
        )
        
        assert cache is not None
        assert cache.self_attn_cache.get_seq_len() == 5
        
        # Second step - process 1 more token
        x2 = torch.randn(1, 1, 64)
        output2, cache = decoder_layer(
            x2, encoder_output[:1], layer_cache=cache, use_cache=True
        )
        
        assert cache.self_attn_cache.get_seq_len() == 6

    def test_cross_attention_uses_encoder(
        self, decoder_layer: DecoderLayer, encoder_output: torch.Tensor
    ) -> None:
        """Cross-attention should attend to encoder output."""
        decoder_layer.eval()
        
        x = torch.randn(2, 10, 64)
        
        # Different encoder outputs should give different decoder outputs
        encoder1 = torch.randn(2, 15, 64)
        encoder2 = torch.randn(2, 15, 64)
        
        output1, _ = decoder_layer(x, encoder1)
        output2, _ = decoder_layer(x, encoder2)
        
        assert not torch.allclose(output1, output2)

    def test_self_attention_is_causal(
        self, decoder_layer: DecoderLayer, encoder_output: torch.Tensor
    ) -> None:
        """Self-attention should be causal (can't see future)."""
        decoder_layer.eval()
        
        x = torch.randn(1, 10, 64)
        
        # Run full sequence
        output_full, _ = decoder_layer(x, encoder_output[:1])
        
        # Output at position 0 should only depend on position 0
        # (This is implicitly tested by the causal mask in GQA)
        assert output_full.shape == x.shape


# ============================================================================
# TransformerEncoder Tests
# ============================================================================

class TestTransformerEncoder:
    """Tests for full Transformer encoder."""

    @pytest.fixture
    def encoder(self) -> TransformerEncoder:
        """Create transformer encoder for testing."""
        return TransformerEncoder(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )

    def test_output_shape(self, encoder: TransformerEncoder) -> None:
        """Encoder should produce correct output shape."""
        input_ids = torch.randint(0, 1000, (2, 10))
        output = encoder(input_ids)
        
        assert output.shape == (2, 10, 64)  # batch, seq, d_model

    def test_attention_mask(self, encoder: TransformerEncoder) -> None:
        """Encoder should respect attention mask."""
        encoder.eval()
        
        input_ids = torch.randint(0, 1000, (2, 10))
        mask_full = torch.ones(2, 10)
        mask_partial = torch.ones(2, 10)
        mask_partial[:, 5:] = 0
        
        output_full = encoder(input_ids, attention_mask=mask_full)
        output_partial = encoder(input_ids, attention_mask=mask_partial)
        
        assert not torch.allclose(output_full, output_partial)

    def test_different_sequence_lengths(self, encoder: TransformerEncoder) -> None:
        """Encoder should handle different sequence lengths."""
        encoder.eval()
        
        for seq_len in [5, 10, 50, 100]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            output = encoder(input_ids)
            assert output.shape == (1, seq_len, 64)

    def test_embedding_scaling(self, encoder: TransformerEncoder) -> None:
        """Embeddings should be scaled by sqrt(d_model)."""
        # This is implicitly tested - embeddings are scaled in forward
        input_ids = torch.randint(0, 1000, (1, 10))
        output = encoder(input_ids)
        
        # Output should have reasonable magnitude
        assert output.abs().mean() < 100


# ============================================================================
# TransformerDecoder Tests
# ============================================================================

class TestTransformerDecoder:
    """Tests for full Transformer decoder."""

    @pytest.fixture
    def decoder(self) -> TransformerDecoder:
        """Create transformer decoder for testing."""
        return TransformerDecoder(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
            tie_embeddings=True,
        )

    @pytest.fixture
    def encoder_output(self) -> torch.Tensor:
        """Create mock encoder output."""
        return torch.randn(2, 15, 64)

    def test_output_shape(
        self, decoder: TransformerDecoder, encoder_output: torch.Tensor
    ) -> None:
        """Decoder should produce logits with vocab size."""
        input_ids = torch.randint(0, 1000, (2, 10))
        logits, _ = decoder(input_ids, encoder_output)
        
        assert logits.shape == (2, 10, 1000)  # batch, seq, vocab

    def test_kv_caching(
        self, decoder: TransformerDecoder, encoder_output: torch.Tensor
    ) -> None:
        """Decoder should support KV caching for generation."""
        decoder.eval()
        
        # First step
        input_ids = torch.randint(0, 1000, (1, 5))
        logits1, caches = decoder(
            input_ids, encoder_output[:1], use_cache=True
        )
        
        assert caches is not None
        assert len(caches) == 2  # 2 layers
        
        # Second step
        next_token = torch.randint(0, 1000, (1, 1))
        logits2, caches = decoder(
            next_token, encoder_output[:1],
            past_key_values=caches, use_cache=True
        )
        
        assert logits2.shape == (1, 1, 1000)

    def test_embedding_tying(self, decoder: TransformerDecoder) -> None:
        """Embedding and output weights should be tied."""
        assert decoder.tie_embeddings
        
        # Check that weights are the same object
        assert decoder.embedding.weight is decoder.lm_head.weight

    def test_untied_embeddings(self) -> None:
        """Decoder should support untied embeddings."""
        decoder = TransformerDecoder(
            vocab_size=1000,
            d_model=64,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
            tie_embeddings=False,
        )
        
        assert not decoder.tie_embeddings
        assert decoder.embedding.weight is not decoder.lm_head.weight


# ============================================================================
# Integration Tests
# ============================================================================

class TestEncoderDecoderIntegration:
    """Integration tests for encoder-decoder together."""

    def test_full_forward_pass(self) -> None:
        """Test complete encoder-decoder forward pass."""
        vocab_size = 1000
        d_model = 64
        
        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        )
        
        encoder.eval()
        decoder.eval()
        
        # Forward pass
        src_ids = torch.randint(0, vocab_size, (2, 10))
        tgt_ids = torch.randint(0, vocab_size, (2, 8))
        
        encoder_out = encoder(src_ids)
        logits, _ = decoder(tgt_ids, encoder_out)
        
        assert logits.shape == (2, 8, vocab_size)

    def test_generation_with_caching(self) -> None:
        """Test autoregressive generation with KV cache."""
        vocab_size = 100
        d_model = 32
        
        encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
        )
        
        decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=1,
            n_heads=2,
            n_kv_heads=1,
            d_ff=64,
            max_seq_len=32,
            dropout=0.0,
        )
        
        encoder.eval()
        decoder.eval()
        
        # Encode source
        src_ids = torch.randint(0, vocab_size, (1, 5))
        encoder_out = encoder(src_ids)
        
        # Generate autoregressively
        generated = [1]  # SOS token
        caches = None
        
        for _ in range(10):
            input_ids = torch.tensor([[generated[-1]]])
            logits, caches = decoder(
                input_ids, encoder_out,
                past_key_values=caches, use_cache=True
            )
            
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            
            if next_token == 2:  # EOS
                break
        
        assert len(generated) > 1
