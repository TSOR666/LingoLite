"""
Core Components for Mobile Translation Model
Efficient building blocks optimized for mobile deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .generation_utils import KVCache


def _probe_sdpa_gqa_support() -> bool:
    """Return True when F.scaled_dot_product_attention accepts ``enable_gqa``.

    The kwarg only exists on torch >= 2.5. Probing once at import time lets
    GroupedQueryAttention.forward branch on a bool instead of raising and
    catching a TypeError on every attention call (the failed call costs
    ~0.2 ms per invocation, multiplied by layers x steps during decoding).
    """
    if not hasattr(F, "scaled_dot_product_attention"):
        return False
    try:
        q = torch.zeros(1, 2, 1, 2)
        kv = torch.zeros(1, 1, 1, 2)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except TypeError:
        return False
    except Exception:
        # Any other failure: fall back to the manual repeat path for safety.
        return False


_SDPA_SUPPORTS_GQA = _probe_sdpa_gqa_support()


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler and faster than LayerNorm, no bias/mean subtraction.
    Used in: LLaMA, Mistral, Gemma, etc.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Validate eps is positive (required for numerical stability)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            Normalized tensor (..., dim)
        """
        # Compute normalization in float32 for stability, then cast back.
        x_float = x.float()
        rms = torch.rsqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x_float * rms
        weight = self.weight.to(dtype=x_float.dtype)
        return (weight * x_normed).to(dtype=x.dtype)


class RotaryPositionEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Encodes position information by rotating query and key vectors.
    
    Advantages:
    - No learned parameters (saves space)
    - Better extrapolation to longer sequences
    - Naturally encodes relative positions
    
    Used in: GPT-NeoX, LLaMA, PaLM, GPT-J, etc.
    """

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RotaryPositionEmbedding expects even dim, got {dim}")
        # Validate base is positive (required for frequency computation)
        if base <= 0:
            raise ValueError(f"base must be positive, got {base}")
        # Validate max_seq_len is positive
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('cos_cached', torch.empty(0), persistent=False)
        self.register_buffer('sin_cached', torch.empty(0), persistent=False)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        """Precompute cos and sin for positions on the requested device."""
        if device is None:
            device = self.inv_freq.device
        else:
            device = torch.device(device)

        # Keep inv_freq on the same device as the cached tensors
        if self.inv_freq.device != device:
            self.register_buffer('inv_freq', self.inv_freq.to(device), persistent=False)

        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        inv_freq = self.inv_freq
        freqs = torch.outer(positions, inv_freq)  # (seq_len, dim//2)
        
        # Concatenate for efficient application
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
        
        # Register cos and sin as buffers
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embeddings to queries and keys.
        
        Args:
            q: Query tensor (..., seq_len, dim)
            k: Key tensor (..., seq_len, dim)
            seq_len: Sequence length (if None, use q.shape[-2])
            offset: Position offset for cached KV generation
        
        Returns:
            Rotated q and k tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]

        total_len = seq_len + offset
        device = q.device

        # Extend or relocate cache if needed
        if (
            not hasattr(self, 'cos_cached') or 
            total_len > self.cos_cached.shape[0]
            or self.cos_cached.device != device
        ):
            self._precompute_freqs(total_len, device=device)
            self.max_seq_len = max(self.max_seq_len, total_len)

        # Slice the portion of cos/sin we need for this offset and match dtype/device
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).to(dtype=q.dtype, device=device)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).to(dtype=q.dtype, device=device)
        
        # Apply rotation
        assert q.shape[-1] == k.shape[-1], f"Q/K head_dim mismatch: {q.shape[-1]} vs {k.shape[-1]}"
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    
    Memory-efficient variant of Multi-Head Attention where multiple
    query heads share key/value heads.
    
    Example with 32 query heads and 4 KV groups:
    - MHA: 32 Q heads, 32 K heads, 32 V heads = 96 total
    - GQA: 32 Q heads, 4 K heads, 4 V heads = 40 total
    
    Benefits:
    - KV cache 8x smaller (critical for mobile)
    - Faster inference (fewer KV computations)
    - Minimal quality loss vs full MHA
    
    Used in: LLaMA 2, Mistral, Gemma, etc.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: float = 0.0,
        is_causal: bool = False,
        is_cross_attn: bool = False,
    ) -> None:
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        assert n_heads % n_kv_heads == 0, f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim > 1, f"head_dim must be > 1 for numerical stability, got {self.head_dim}"
        self.n_rep = n_heads // n_kv_heads  # Repetition factor
        self.is_causal = is_causal
        self.is_cross_attn = is_cross_attn
        
        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        # Store dropout probability as a float; a full ``nn.Dropout`` module is
        # unused (SDPA takes the probability directly and the layer already has
        # its own residual dropout).
        self.dropout_p = float(dropout)
        # Kept for backwards compatibility: older checkpoints / code paths may
        # reference ``self.dropout`` as a Module. It is never applied to the
        # attention output in the forward pass.
        self.dropout = nn.Dropout(dropout)

        # Causal mask cache. Building ``torch.arange(kv_len)`` + compare on
        # every forward pass is cheap per call but multiplies across layers and
        # steps; cache the largest triangular mask we've seen and slice into
        # it. Registered as a non-persistent buffer so ``.to(device)`` moves
        # it along with the module and it is kept out of checkpoints.
        self.register_buffer(
            "_causal_mask_cache", torch.empty(0, dtype=torch.bool), persistent=False
        )

    def _get_causal_mask(
        self, q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """Return a cached ``(1, 1, q_len, kv_len)`` bool causal mask.

        The cache stores a single ``(N, N)`` triangular mask sized to the
        largest ``kv_len`` seen and slices into it on subsequent calls.
        """
        cache = self._causal_mask_cache
        need_rebuild = (
            cache.numel() == 0
            or cache.shape[0] < kv_len
            or cache.device != device
        )
        if need_rebuild:
            new_size = max(kv_len, cache.shape[0] if cache.numel() else 0)
            positions = torch.arange(new_size, device=device)
            full = positions.unsqueeze(0) <= positions.unsqueeze(-1)
            self._causal_mask_cache = full
            cache = full

        if q_len == kv_len:
            sub = cache[:q_len, :kv_len]
        else:
            # During cached generation q_len < kv_len: the new query positions
            # are the final q_len rows of the larger triangular matrix.
            start = kv_len - q_len
            sub = cache[start:start + q_len, :kv_len]
        return sub.unsqueeze(0).unsqueeze(0)

    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional['KVCache'] = None,
        past_key_value: Optional['KVCache'] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional['KVCache']]:
        """
        Args:
            query: (batch, q_len, d_model)
            key: (batch, kv_len, d_model) - if None, uses query (self-attention)
            value: (batch, kv_len, d_model) - if None, uses query
            attention_mask: (batch, 1, q_len, kv_len) - attention mask
            rope: RotaryPositionEmbedding instance for position encoding
            kv_cache: Optional KVCache for caching key/value tensors
            use_cache: Whether to return updated cache

        Returns:
            output: (batch, q_len, d_model)
            updated_cache: Updated KVCache if use_cache=True, else None
        """
        batch, q_len, _ = query.shape

        cache_input = past_key_value
        if cache_input is not None:
            if kv_cache is not None and kv_cache is not cache_input:
                raise ValueError("Pass only one of 'kv_cache' or 'past_key_value'.")
            kv_cache = cache_input

        # Always project queries
        Q = self.q_proj(query)  # (B, q_len, d_model) -> (B, q_len, n_heads * head_dim)
        Q = Q.view(batch, q_len, self.n_heads, self.head_dim).transpose(1, 2)  # (B, q_len, n_heads * head_dim) -> (B, n_heads, q_len, head_dim)

        past_key = None
        past_value = None
        if kv_cache is not None and kv_cache.key is not None:
            past_key = kv_cache.key
            past_value = kv_cache.value

        # Determine whether we need to compute new keys/values
        if self.is_cross_attn and past_key is not None and past_value is not None:
            # Cross-attention: reuse the single cached projection.
            K = past_key
            V = past_value
            # Align cached tensors with the query dtype/device.
            if K.device != query.device or K.dtype != Q.dtype:
                K = K.to(device=query.device, dtype=Q.dtype)
                V = V.to(device=query.device, dtype=Q.dtype)
        else:
            # For self-attention, reuse query as key/value
            if key is None:
                key = query
            if value is None:
                value = query

            key_batch, new_kv_len, _ = key.shape
            if value.shape[0] != key_batch:
                raise ValueError("Cross-attention key and value batch sizes must match")

            K_new = self.k_proj(key)
            V_new = self.v_proj(value)

            K_new = K_new.view(
                key_batch, new_kv_len, self.n_kv_heads, self.head_dim
            ).permute(0, 2, 1, 3)
            V_new = V_new.view(
                key_batch, new_kv_len, self.n_kv_heads, self.head_dim
            ).permute(0, 2, 1, 3)

            if rope is not None and not self.is_cross_attn:
                past_len = 0 if past_key is None else past_key.shape[2]
                Q, K_new = rope(Q, K_new, offset=past_len)

            if use_cache and kv_cache is not None:
                # Append the new slice through the cache: in reserved mode this
                # writes into the pre-allocated buffer (no torch.cat); in lazy
                # mode it falls back to the historical concatenation path.
                if K_new.device != query.device or K_new.dtype != Q.dtype:
                    K_new = K_new.to(device=query.device, dtype=Q.dtype)
                    V_new = V_new.to(device=query.device, dtype=Q.dtype)
                kv_cache.update(K_new, V_new)
                K = kv_cache.key
                V = kv_cache.value
            elif past_key is not None and past_value is not None:
                # Legacy path: caller supplied raw past tensors without a cache
                # object. Mirror the previous torch.cat semantics.
                past_key_aligned = past_key.to(device=K_new.device, dtype=K_new.dtype)
                past_value_aligned = past_value.to(device=V_new.device, dtype=V_new.dtype)
                K = torch.cat([past_key_aligned, K_new], dim=2)
                V = torch.cat([past_value_aligned, V_new], dim=2)
            else:
                K = K_new
                V = V_new

        # Final alignment (covers the cross-attn-no-cache path and any stale
        # tensors that slipped through). The checks above mean this is usually
        # a no-op.
        if K.device != query.device or V.device != query.device:
            K = K.to(device=query.device)
            V = V.to(device=query.device)
        if K.dtype != Q.dtype or V.dtype != Q.dtype:
            K = K.to(dtype=Q.dtype)
            V = V.to(dtype=Q.dtype)

        kv_batch = K.shape[0]
        kv_len = K.shape[2]
        beam_group_size = 1
        if kv_batch != batch:
            if not self.is_cross_attn or batch % kv_batch != 0:
                raise ValueError(
                    "Attention query batch must match key/value batch, or be "
                    "an integer beam multiple for cross-attention"
                )
            beam_group_size = batch // kv_batch
            Q = Q.view(
                kv_batch,
                beam_group_size,
                self.n_heads,
                q_len,
                self.head_dim,
            )

        valid_mask: Optional[torch.Tensor] = None
        if attention_mask is not None:
            # Support masks shaped (B, kv_len), (B, q_len, kv_len), or (B, 1, q_len, kv_len).
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[:, None, None, :]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() != 4:
                raise ValueError(
                    f"attention_mask must have 2, 3, or 4 dimensions, got {attention_mask.dim()}"
                )
            attention_mask = attention_mask.to(device=Q.device)
            if attention_mask.dtype == torch.bool:
                valid_mask = attention_mask
            else:
                # Support binary (1=keep, 0=mask) and additive (0=keep, neg=mask)
                # masks without a CPU sync. ``any()``/``all()`` branching forces
                # a device-host round trip on every attention call; instead we
                # derive both interpretations and pick element-wise.
                is_numeric = attention_mask.to(torch.float32)
                binary_valid = attention_mask != 0
                additive_valid = attention_mask >= 0
                has_negative = (is_numeric < 0.0).any(dim=-1, keepdim=True)
                valid_mask = torch.where(has_negative, additive_valid, binary_valid)

            if beam_group_size > 1:
                if valid_mask.shape[0] == kv_batch:
                    valid_mask = valid_mask.unsqueeze(1)
                elif valid_mask.shape[0] == batch:
                    valid_mask = valid_mask.view(
                        kv_batch, beam_group_size, *valid_mask.shape[1:]
                    )
                else:
                    raise ValueError(
                        "Cross-attention mask batch must match the source or query batch"
                    )

        if self.is_causal:
            causal_valid_mask = self._get_causal_mask(q_len, kv_len, Q.device)
            valid_mask = (
                causal_valid_mask
                if valid_mask is None
                else valid_mask & causal_valid_mask
            )

        # Compute ``fully_masked`` unconditionally; zeroing out its rows is cheap
        # and avoids the CPU sync of an ``any().item()`` check that was present
        # on the previous fast/slow path branch.
        fully_masked: Optional[torch.Tensor] = None
        if valid_mask is not None:
            fully_masked = ~valid_mask.any(dim=-1, keepdim=True)
            # Neutralise fully-masked rows inside the mask itself so both SDPA
            # and the manual path see at least one valid key. We then zero the
            # output rows after the attention op.
            valid_mask = valid_mask | fully_masked

        dropout_p = self.dropout_p if self.training else 0.0
        if beam_group_size > 1:
            K_for_attention = K.unsqueeze(1)
            V_for_attention = V.unsqueeze(1)
            kv_head_dim = 2
        else:
            K_for_attention = K
            V_for_attention = V
            kv_head_dim = 1

        if hasattr(F, "scaled_dot_product_attention"):
            if _SDPA_SUPPORTS_GQA and self.n_rep > 1:
                output = F.scaled_dot_product_attention(
                    Q,
                    K_for_attention,
                    V_for_attention,
                    attn_mask=valid_mask,
                    dropout_p=dropout_p,
                    enable_gqa=True,
                )
            else:
                K_for_scores = (
                    K_for_attention.repeat_interleave(self.n_rep, dim=kv_head_dim)
                    if self.n_rep > 1
                    else K_for_attention
                )
                V_for_scores = (
                    V_for_attention.repeat_interleave(self.n_rep, dim=kv_head_dim)
                    if self.n_rep > 1
                    else V_for_attention
                )
                output = F.scaled_dot_product_attention(
                    Q,
                    K_for_scores,
                    V_for_scores,
                    attn_mask=valid_mask,
                    dropout_p=dropout_p,
                )
        else:
            if self.n_rep > 1:
                K_for_scores = K_for_attention.repeat_interleave(
                    self.n_rep, dim=kv_head_dim
                )
                V_for_scores = V_for_attention.repeat_interleave(
                    self.n_rep, dim=kv_head_dim
                )
            else:
                K_for_scores = K_for_attention
                V_for_scores = V_for_attention

            scores = torch.matmul(Q, K_for_scores.transpose(-2, -1)) / math.sqrt(self.head_dim)

            if valid_mask is not None:
                mask_for_scores = valid_mask
                if mask_for_scores.shape != scores.shape:
                    mask_for_scores = mask_for_scores.expand_as(scores)
                scores = scores.masked_fill(~mask_for_scores, float('-inf'))

            attn = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
            if dropout_p > 0.0:
                attn = F.dropout(attn, p=dropout_p, training=True)

            output = torch.matmul(attn, V_for_scores)

        if fully_masked is not None:
            # Zero out query positions that had no valid keys (sync-free).
            output = torch.where(fully_masked, torch.zeros_like(output), output)

        if beam_group_size > 1:
            output = output.permute(0, 1, 3, 2, 4).contiguous().view(
                batch, q_len, -1
            )
        else:
            output = output.transpose(1, 2).contiguous().view(batch, q_len, -1)
        output = self.o_proj(output)

        # Return the cache object that was just updated in-place (same instance
        # as ``kv_cache``/``past_key_value`` when the caller provided one). The
        # caller can treat it as opaque; its ``.key`` / ``.value`` properties
        # expose the valid history slice.
        updated_cache = None
        if use_cache:
            if kv_cache is None:
                # Caller used the legacy "no cache object" path; synthesize one
                # around the freshly materialized K / V tensors.
                from .generation_utils import KVCache as _KVCache
                updated_cache = _KVCache(
                    key=K,
                    value=V,
                    num_heads=self.n_kv_heads,
                    head_dim=self.head_dim,
                )
            else:
                updated_cache = kv_cache

        return output, updated_cache


class SwiGLU_FFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    
    GLU (Gated Linear Unit) variant using Swish (SiLU) activation.
    Better than ReLU for language models.
    
    Architecture:
        FFN(x) = (Swish(xW_gate) * xW_up) W_down
    
    Where * is element-wise multiplication.
    
    Used in: PaLM, LLaMA, Mistral, etc.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        # Three projections for SwiGLU
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            output: (batch, seq_len, d_model)
        """
        def _linear(module: nn.Linear, inp: torch.Tensor) -> torch.Tensor:
            weight = getattr(module, "weight", None)
            bias = getattr(module, "bias", None)
            # Quantized dynamic linear modules expose weight as a callable, so fall back to module(inp)
            if isinstance(weight, torch.Tensor):
                bias_cast = bias.to(dtype=inp.dtype) if isinstance(bias, torch.Tensor) else bias
                return F.linear(inp, weight.to(dtype=inp.dtype), bias_cast)
            return cast(torch.Tensor, module(inp))

        gate: torch.Tensor = F.silu(_linear(self.gate_proj, x))  # Swish activation
        up: torch.Tensor = _linear(self.up_proj, x)
        hidden: torch.Tensor = gate * up  # Gated activation
        
        # Project back down
        output: torch.Tensor = _linear(self.down_proj, hidden)
        output = self.dropout(output)

        return output


# Test the components
if __name__ == "__main__":
    pass
