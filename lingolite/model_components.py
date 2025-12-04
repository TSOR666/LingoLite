"""
Core Components for Mobile Translation Model
Efficient building blocks optimized for mobile deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .generation_utils import KVCache


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    Simpler and faster than LayerNorm, no bias/mean subtraction.
    Used in: LLaMA, Mistral, Gemma, etc.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (..., dim)
        Returns:
            Normalized tensor (..., dim)
        """
        # Compute RMS
        x_float = x.float()  # compute mean in float32 for stability under mixed precision
        rms = torch.sqrt(torch.mean(x_float ** 2, dim=-1, keepdim=True) + self.eps)
        rms = rms.to(dtype=x.dtype)
        
        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


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
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute frequency bands
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs(max_seq_len)
    
    def _precompute_freqs(self, seq_len: int, device: Optional[torch.device] = None) -> None:
        """Precompute cos and sin for positions on the requested device."""
        if device is None:
            device = self.inv_freq.device

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
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot
    
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seq_len: Optional[int] = None,
        offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for forward to support both calling conventions."""
        return self.forward(q, k, seq_len=seq_len, offset=offset)


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
        self.n_rep = n_heads // n_kv_heads  # Repetition factor
        self.is_causal = is_causal
        self.is_cross_attn = is_cross_attn
        
        # Projections
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionEmbedding] = None,
        kv_cache: Optional['KVCache'] = None,
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
            K = past_key
            V = past_value
        else:
            # For self-attention, reuse query as key/value
            if key is None:
                key = query
            if value is None:
                value = query

            kv_len = key.shape[1]

            K = self.k_proj(key)  # (B, kv_len, d_model) -> (B, kv_len, n_kv_heads * head_dim)
            V = self.v_proj(value)  # (B, kv_len, d_model) -> (B, kv_len, n_kv_heads * head_dim)

            K = K.view(batch, kv_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, kv_len, n_kv_heads * head_dim) -> (B, n_kv_heads, kv_len, head_dim)
            V = V.view(batch, kv_len, self.n_kv_heads, self.head_dim).permute(0, 2, 1, 3)  # (B, kv_len, n_kv_heads * head_dim) -> (B, n_kv_heads, kv_len, head_dim)

            if rope is not None and not self.is_cross_attn:
                past_len = 0 if past_key is None else past_key.shape[2]
                Q, K = rope(Q, K, offset=past_len)

            if past_key is not None and past_value is not None:
                K = torch.cat([past_key, K], dim=2)
                V = torch.cat([past_value, V], dim=2)

        # Keep K/V aligned with the query for safe device/dtype usage
        if K.device != query.device or V.device != query.device:
            K = K.to(device=query.device)
            V = V.to(device=query.device)
        if K.dtype != Q.dtype or V.dtype != Q.dtype:
            K = K.to(dtype=Q.dtype)
            V = V.to(dtype=Q.dtype)

        present = (K, V) if use_cache else None

        # Repeat K and V to match the number of query heads
        if self.n_rep > 1:
            K_for_scores = K.repeat_interleave(self.n_rep, dim=1)  # (B, n_kv_heads, kv_len, head_dim) -> (B, n_heads, kv_len, head_dim)
            V_for_scores = V.repeat_interleave(self.n_rep, dim=1)  # (B, n_kv_heads, kv_len, head_dim) -> (B, n_heads, kv_len, head_dim)
        else:
            K_for_scores = K
            V_for_scores = V

        kv_len = K_for_scores.shape[2]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K_for_scores.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, n_heads, q_len, head_dim) @ (B, n_heads, head_dim, kv_len) -> (B, n_heads, q_len, kv_len)

        if self.is_causal:
            kv_positions = torch.arange(kv_len, device=scores.device)
            query_positions = torch.arange(q_len, device=scores.device)
            if past_key is not None and kv_len >= q_len:
                query_positions = query_positions + (kv_len - q_len)
            causal_mask = kv_positions.unsqueeze(0) > query_positions.unsqueeze(-1)
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=scores.device, dtype=scores.dtype)
            scores = scores + attention_mask

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        output = torch.matmul(attn, V_for_scores)  # (B, n_heads, q_len, kv_len) @ (B, n_heads, kv_len, head_dim) -> (B, n_heads, q_len, head_dim)
        output = output.transpose(1, 2).contiguous().view(batch, q_len, -1)  # (B, n_heads, q_len, head_dim) -> (B, q_len, n_heads * head_dim)
        output = self.o_proj(output)

        # Convert present tuple to KVCache if needed
        updated_cache = None
        if use_cache and present is not None:
            from .generation_utils import KVCache
            updated_cache = KVCache(key=present[0], value=present[1])

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
        # SwiGLU activation
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        hidden = gate * up  # Gated activation
        
        # Project back down
        output = self.down_proj(hidden)
        output = self.dropout(output)
        
        return output


# Test the components
if __name__ == "__main__":
    pass
