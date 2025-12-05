"""
Minimal verification script for lingolite.model_components.
Runs key modules on random tensors and checks shapes/finite outputs.
"""

import torch

from lingolite.model_components import (
    GroupedQueryAttention,
    RMSNorm,
    RotaryPositionEmbedding,
    SwiGLU_FFN,
)


def _assert_shape(tensor: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    if tensor.shape != expected:
        raise RuntimeError(f"{name} shape mismatch: expected {expected}, got {tuple(tensor.shape)}")


def _assert_finite(tensor: torch.Tensor, name: str) -> None:
    if not torch.isfinite(tensor).all():
        raise RuntimeError(f"{name} contains NaN or Inf")


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch, q_len, kv_len = 2, 3, 3
    d_model, n_heads, n_kv_heads, d_ff = 64, 8, 4, 128
    head_dim = d_model // n_heads

    # RMSNorm
    x = torch.randn(batch, q_len, d_model, device=device)
    rms = RMSNorm(d_model).to(device)
    rms_out = rms(x)
    _assert_shape(rms_out, (batch, q_len, d_model), "RMSNorm output")
    _assert_finite(rms_out, "RMSNorm output")

    # Rotary Position Embedding on per-head inputs
    rope = RotaryPositionEmbedding(dim=head_dim, max_seq_len=16).to(device)
    q_heads = torch.randn(batch, n_heads, q_len, head_dim, device=device)
    k_heads = torch.randn(batch, n_kv_heads, kv_len, head_dim, device=device)
    q_rot, k_rot = rope(q_heads, k_heads, seq_len=q_len)
    _assert_shape(q_rot, (batch, n_heads, q_len, head_dim), "RoPE q_rot")
    _assert_shape(k_rot, (batch, n_kv_heads, kv_len, head_dim), "RoPE k_rot")
    _assert_finite(q_rot, "RoPE q_rot")
    _assert_finite(k_rot, "RoPE k_rot")

    # Grouped Query Attention with cache and RoPE
    gqa = GroupedQueryAttention(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        dropout=0.0,
        is_causal=False,
        is_cross_attn=False,
    ).to(device)
    attn_mask = torch.zeros(batch, 1, q_len, kv_len, device=device)
    attn_out, cache = gqa(query=x, attention_mask=attn_mask, rope=rope, use_cache=True)
    _assert_shape(attn_out, (batch, q_len, d_model), "Attention output")
    _assert_finite(attn_out, "Attention output")
    if cache is None:
        raise RuntimeError("KV cache not returned when use_cache=True")

    # Incremental step with cache
    cache_kv_len = cache.key.shape[2]
    incremental_q = torch.randn(batch, 1, d_model, device=device)
    incremental_mask = torch.zeros(batch, 1, 1, cache_kv_len + 1, device=device)
    attn_inc, cache = gqa(
        query=incremental_q,
        attention_mask=incremental_mask,
        rope=rope,
        kv_cache=cache,
        use_cache=True,
    )
    _assert_shape(attn_inc, (batch, 1, d_model), "Incremental attention output")
    _assert_finite(attn_inc, "Incremental attention output")

    # SwiGLU FFN
    ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, dropout=0.0).to(device)
    ffn_out = ffn(x)
    _assert_shape(ffn_out, (batch, q_len, d_model), "SwiGLU output")
    _assert_finite(ffn_out, "SwiGLU output")

    print("VERIFICATION PASSED")


if __name__ == "__main__":
    main()
