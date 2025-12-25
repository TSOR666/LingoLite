"""
Step 5 - Mock-Free Verification Test for LingoLite Audit

This script verifies the corrected code runs correctly with real PyTorch operations.
It validates output shapes, numerical stability (no NaN/Inf), and gradient flow.

Prints "VERIFICATION PASSED" only if ALL checks succeed.
"""

import sys
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("tests", 1)[0])

from lingolite.model_components import (
    RMSNorm,
    RotaryPositionEmbedding,
    GroupedQueryAttention,
    SwiGLU_FFN,
)
from lingolite.encoder_decoder import (
    EncoderLayer,
    DecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
)
from lingolite.mobile_translation_model import MobileTranslationModel, create_model
from lingolite.generation_utils import (
    KVCache,
    LayerKVCache,
    generate_with_kv_cache,
    generate_with_beam_search,
)


def check_finite(tensor: torch.Tensor, name: str) -> None:
    """Assert tensor contains no NaN or Inf values."""
    if not torch.isfinite(tensor).all():
        num_nan = torch.isnan(tensor).sum().item()
        num_inf = torch.isinf(tensor).sum().item()
        raise AssertionError(
            f"{name} contains {num_nan} NaN and {num_inf} Inf values"
        )


def check_shape(tensor: torch.Tensor, expected: tuple, name: str) -> None:
    """Assert tensor has expected shape."""
    if tensor.shape != expected:
        raise AssertionError(
            f"{name} shape mismatch: expected {expected}, got {tensor.shape}"
        )


def verify_rmsnorm() -> None:
    """Verify RMSNorm component."""
    print("  Testing RMSNorm...", end=" ")

    # Test shapes: (batch=2, seq_len=10, dim=64)
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 10, 64)

    output = norm(x)
    check_shape(output, (2, 10, 64), "RMSNorm output")
    check_finite(output, "RMSNorm output")

    # Test with large values (stability)
    x_large = torch.randn(2, 10, 64) * 1000
    output_large = norm(x_large)
    check_finite(output_large, "RMSNorm output (large input)")

    # Test with near-zero values (stability)
    x_small = torch.randn(2, 10, 64) * 1e-6
    output_small = norm(x_small)
    check_finite(output_small, "RMSNorm output (small input)")

    # Test dtype preservation
    norm_half = RMSNorm(dim=64).half()
    x_half = torch.randn(2, 10, 64, dtype=torch.float16)
    output_half = norm_half(x_half)
    assert output_half.dtype == torch.float16, "RMSNorm should preserve float16 dtype"
    check_finite(output_half.float(), "RMSNorm output (float16)")

    print("OK")


def verify_rope() -> None:
    """Verify Rotary Position Embedding."""
    print("  Testing RotaryPositionEmbedding...", end=" ")

    # Test shapes: head_dim=32
    rope = RotaryPositionEmbedding(dim=32, max_seq_len=512)

    # Shape: (batch=2, n_heads=8, seq_len=10, head_dim=32)
    q = torch.randn(2, 8, 10, 32)
    k = torch.randn(2, 8, 10, 32)

    q_rot, k_rot = rope(q, k)
    check_shape(q_rot, (2, 8, 10, 32), "RoPE q output")
    check_shape(k_rot, (2, 8, 10, 32), "RoPE k output")
    check_finite(q_rot, "RoPE q output")
    check_finite(k_rot, "RoPE k output")

    # Test with offset (for KV caching)
    q_short = torch.randn(2, 8, 1, 32)  # Single new position
    k_short = torch.randn(2, 8, 1, 32)
    q_rot_offset, k_rot_offset = rope(q_short, k_short, offset=10)
    check_shape(q_rot_offset, (2, 8, 1, 32), "RoPE q output (offset)")
    check_finite(q_rot_offset, "RoPE q output (offset)")

    print("OK")


def verify_gqa() -> None:
    """Verify Grouped Query Attention."""
    print("  Testing GroupedQueryAttention...", end=" ")

    # d_model=64, n_heads=8, n_kv_heads=2 (4:1 ratio)
    # head_dim = d_model / n_heads = 64 / 8 = 8
    d_model = 64
    n_heads = 8
    n_kv_heads = 2
    head_dim = d_model // n_heads  # = 8

    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
        is_causal=True,
    )
    gqa.eval()

    # Shape: (batch=2, seq_len=10, d_model=64)
    x = torch.randn(2, 10, d_model)

    output, cache = gqa(x, use_cache=True)
    check_shape(output, (2, 10, d_model), "GQA output")
    check_finite(output, "GQA output")

    # Verify cache shape
    assert cache is not None, "GQA should return cache when use_cache=True"
    assert cache.key is not None, "Cache key should not be None"
    # Cache shape: (batch, n_kv_heads, seq_len, head_dim)
    check_shape(cache.key, (2, n_kv_heads, 10, head_dim), "GQA cache key")
    check_finite(cache.key, "GQA cache key")

    # Test incremental decoding with cache
    x_new = torch.randn(2, 1, d_model)  # Single new token
    output_new, cache_updated = gqa(x_new, kv_cache=cache, use_cache=True)
    check_shape(output_new, (2, 1, d_model), "GQA output (cached)")
    check_finite(output_new, "GQA output (cached)")
    check_shape(cache_updated.key, (2, n_kv_heads, 11, head_dim), "GQA updated cache key")

    print("OK")


def verify_swiglu() -> None:
    """Verify SwiGLU Feed-Forward Network."""
    print("  Testing SwiGLU_FFN...", end=" ")

    ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0)
    ffn.eval()

    # Shape: (batch=2, seq_len=10, d_model=64)
    x = torch.randn(2, 10, 64)

    output = ffn(x)
    check_shape(output, (2, 10, 64), "SwiGLU output")
    check_finite(output, "SwiGLU output")

    # Test dtype preservation
    ffn_half = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0).half()
    x_half = torch.randn(2, 10, 64, dtype=torch.float16)
    output_half = ffn_half(x_half)
    assert output_half.dtype == torch.float16, "SwiGLU should preserve float16 dtype"
    check_finite(output_half.float(), "SwiGLU output (float16)")

    print("OK")


def verify_encoder_decoder() -> None:
    """Verify Encoder and Decoder layers."""
    print("  Testing Encoder/Decoder layers...", end=" ")

    # Encoder Layer
    enc_layer = EncoderLayer(
        d_model=64, n_heads=8, n_kv_heads=2, d_ff=256, dropout=0.0
    )
    enc_layer.eval()

    x = torch.randn(2, 10, 64)
    output = enc_layer(x)
    check_shape(output, (2, 10, 64), "EncoderLayer output")
    check_finite(output, "EncoderLayer output")

    # Decoder Layer
    dec_layer = DecoderLayer(
        d_model=64, n_heads=8, n_kv_heads=2, d_ff=256, dropout=0.0
    )
    dec_layer.eval()

    encoder_output = torch.randn(2, 15, 64)  # Different length for cross-attention
    x_dec = torch.randn(2, 10, 64)

    output_dec, cache = dec_layer(x_dec, encoder_output, use_cache=True)
    check_shape(output_dec, (2, 10, 64), "DecoderLayer output")
    check_finite(output_dec, "DecoderLayer output")
    assert cache is not None, "DecoderLayer should return cache"

    print("OK")


def verify_full_model() -> None:
    """Verify complete MobileTranslationModel."""
    print("  Testing MobileTranslationModel...", end=" ")

    vocab_size = 1000
    model = create_model(vocab_size=vocab_size, model_size='tiny')
    model.eval()

    # Shapes: (batch=2, src_len=15), (batch=2, tgt_len=10)
    src_ids = torch.randint(0, vocab_size, (2, 15))
    tgt_ids = torch.randint(0, vocab_size, (2, 10))
    src_mask = torch.ones(2, 15)
    tgt_mask = torch.ones(2, 10)

    # Forward pass
    with torch.no_grad():
        logits, cache, encoder_output = model(
            src_input_ids=src_ids,
            tgt_input_ids=tgt_ids,
            src_attention_mask=src_mask,
            tgt_attention_mask=tgt_mask,
            use_cache=False,
        )

    check_shape(logits, (2, 10, vocab_size), "Model logits")
    check_finite(logits, "Model logits")
    check_shape(encoder_output, (2, 15, 256), "Model encoder output")  # tiny uses d_model=256
    check_finite(encoder_output, "Model encoder output")

    print("OK")


def verify_generation() -> None:
    """Verify generation functions."""
    print("  Testing generation...", end=" ")

    vocab_size = 1000
    model = create_model(vocab_size=vocab_size, model_size='tiny')
    model.eval()

    src_ids = torch.randint(0, vocab_size, (2, 10))
    src_mask = torch.ones(2, 10)

    # Standard generation
    with torch.no_grad():
        generated = model.generate(
            src_input_ids=src_ids,
            src_attention_mask=src_mask,
            max_length=20,
            sos_token_id=1,
            eos_token_id=2,
            temperature=1.0,
        )

    assert generated.shape[0] == 2, "Batch size should be preserved"
    assert generated.shape[1] <= 20, "Should not exceed max_length"
    check_finite(generated.float(), "Generated token IDs")

    # KV-cached generation
    with torch.no_grad():
        generated_cached = generate_with_kv_cache(
            model=model,
            src_input_ids=src_ids,
            src_attention_mask=src_mask,
            max_length=20,
            sos_token_id=1,
            eos_token_id=2,
        )

    assert generated_cached.shape[0] == 2, "Batch size should be preserved (cached)"
    check_finite(generated_cached.float(), "Generated token IDs (cached)")

    # Beam search generation
    with torch.no_grad():
        generated_beam = generate_with_beam_search(
            model=model,
            src_input_ids=src_ids,
            src_attention_mask=src_mask,
            max_length=20,
            num_beams=4,
            sos_token_id=1,
            eos_token_id=2,
        )

    check_shape(generated_beam, (2, 20), "Beam search output")
    check_finite(generated_beam.float(), "Generated token IDs (beam search)")

    print("OK")


def verify_gradient_flow() -> None:
    """Verify gradients flow correctly through the model."""
    print("  Testing gradient flow...", end=" ")

    vocab_size = 1000
    model = create_model(vocab_size=vocab_size, model_size='tiny')
    model.train()

    src_ids = torch.randint(0, vocab_size, (2, 10))
    tgt_ids = torch.randint(0, vocab_size, (2, 8))

    # Compute loss
    loss = model.compute_loss(
        src_input_ids=src_ids,
        tgt_input_ids=tgt_ids,
        label_smoothing=0.1,
    )

    check_finite(loss, "Loss")
    assert loss.requires_grad, "Loss should require gradients"

    # Backward pass
    loss.backward()

    # Check gradients are finite
    for name, param in model.named_parameters():
        if param.grad is not None:
            if not torch.isfinite(param.grad).all():
                raise AssertionError(f"Gradient for {name} contains NaN/Inf")

    print("OK")


def verify_numerical_stability() -> None:
    """Verify numerical stability under edge cases."""
    print("  Testing numerical stability...", end=" ")

    # Test RMSNorm with extreme values
    norm = RMSNorm(dim=64)

    # Very large values
    x_large = torch.randn(2, 10, 64) * 1e4
    output_large = norm(x_large)
    check_finite(output_large, "RMSNorm (large values)")

    # Very small values
    x_small = torch.randn(2, 10, 64) * 1e-6
    output_small = norm(x_small)
    check_finite(output_small, "RMSNorm (small values)")

    # Test SwiGLU with potentially large activations
    ffn = SwiGLU_FFN(d_model=64, d_ff=256, dropout=0.0)
    ffn.eval()
    x_large_ffn = torch.randn(2, 10, 64) * 10
    output_ffn = ffn(x_large_ffn)
    check_finite(output_ffn, "SwiGLU (large values)")

    # Test attention with challenging mask patterns
    gqa = GroupedQueryAttention(d_model=64, n_heads=8, n_kv_heads=2, is_causal=True)
    gqa.eval()
    x = torch.randn(2, 10, 64)

    # Mask that leaves only first position visible
    mask = torch.zeros(2, 1, 1, 10)
    mask[:, :, :, 1:] = float('-inf')

    output_masked, _ = gqa(x, attention_mask=mask)
    check_finite(output_masked, "GQA (heavily masked)")

    print("OK")


def main() -> None:
    """Run all verification tests."""
    print("\n" + "=" * 60)
    print("STEP 5: MOCK-FREE VERIFICATION TEST")
    print("=" * 60 + "\n")

    print("Running verification tests...\n")

    try:
        verify_rmsnorm()
        verify_rope()
        verify_gqa()
        verify_swiglu()
        verify_encoder_decoder()
        verify_full_model()
        verify_generation()
        verify_gradient_flow()
        verify_numerical_stability()

        print("\n" + "=" * 60)
        print("VERIFICATION PASSED")
        print("=" * 60 + "\n")

    except AssertionError as e:
        print(f"\n\nVERIFICATION FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nVERIFICATION FAILED (unexpected error): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
