"""
Encoder-Decoder Architecture for Translation
Mobile-optimized transformer with GQA, RoPE, and SwiGLU
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, List, TYPE_CHECKING
from .model_components import (
    RMSNorm,
    RotaryPositionEmbedding,
    GroupedQueryAttention,
    SwiGLU_FFN
)

if TYPE_CHECKING:
    from .generation_utils import LayerKVCache


class EncoderLayer(nn.Module):
    """
    Single Transformer Encoder Layer.
    
    Architecture:
        x = x + SelfAttention(Norm(x))
        x = x + FFN(Norm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Self-attention (bidirectional for encoder)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            is_causal=False,  # Bidirectional
        )
        
        # Feed-forward network
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # Layer normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionEmbedding] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: (batch, 1, seq_len, seq_len)
            rope: RoPE instance

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(x, attention_mask=attention_mask, rope=rope, use_cache=False)
        x = self.dropout(x)
        x = residual + x

        # Feed-forward with pre-norm
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class DecoderLayer(nn.Module):
    """
    Single Transformer Decoder Layer.
    
    Architecture:
        x = x + MaskedSelfAttention(Norm(x))
        x = x + CrossAttention(Norm(x), encoder_output)
        x = x + FFN(Norm(x))
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Masked self-attention (causal)
        self.self_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            is_causal=True,  # Causal masking
        )
        
        # Cross-attention to encoder
        self.cross_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            is_causal=False,
            is_cross_attn=True,
        )
        
        # Feed-forward network
        self.ffn = SwiGLU_FFN(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
        # Layer normalization
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        rope: Optional[RotaryPositionEmbedding] = None,
        layer_cache: Optional['LayerKVCache'] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional['LayerKVCache']]:
        """
        Args:
            x: (batch, tgt_len, d_model)
            encoder_output: (batch, src_len, d_model)
            self_attention_mask: (batch, 1, tgt_len, tgt_len)
            cross_attention_mask: (batch, 1, tgt_len, src_len)
            rope: RoPE instance
            layer_cache: Optional LayerKVCache for this layer
            use_cache: Whether to return updated cache

        Returns:
            output: (batch, tgt_len, d_model)
            updated_cache: Updated LayerKVCache if use_cache=True, else None
        """
        # Initialize cache if needed
        if use_cache and layer_cache is None:
            from .generation_utils import LayerKVCache
            layer_cache = LayerKVCache()

        # Masked self-attention with pre-norm
        residual = x
        x = self.norm1(x)
        x, self_attn_cache = self.self_attn(
            x,
            attention_mask=self_attention_mask,
            rope=rope,
            kv_cache=layer_cache.self_attn_cache if layer_cache else None,
            use_cache=use_cache,
        )
        x = self.dropout(x)
        x = residual + x

        # Update self-attention cache
        if use_cache and self_attn_cache is not None and layer_cache is not None:
            layer_cache.self_attn_cache = self_attn_cache

        # Cross-attention with pre-norm
        residual = x
        x = self.norm2(x)
        x, cross_attn_cache = self.cross_attn(
            query=x,
            key=encoder_output,
            value=encoder_output,
            attention_mask=cross_attention_mask,
            rope=None,  # Don't apply RoPE to cross-attention
            kv_cache=layer_cache.cross_attn_cache if layer_cache else None,
            use_cache=use_cache,
        )
        x = self.dropout(x)
        x = residual + x

        # Update cross-attention cache
        if use_cache and cross_attn_cache is not None and layer_cache is not None:
            layer_cache.cross_attn_cache = cross_attn_cache

        # Feed-forward with pre-norm
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = residual + x

        return x, layer_cache if use_cache else None


class TransformerEncoder(nn.Module):
    """
    Stack of Encoder Layers with embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.d_model = d_model
        # Cache scaling factor to avoid computing sqrt every forward pass
        self.embedding_scale = math.sqrt(d_model)

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len
        )
        
        # Encoder layers (tracked only via the ModuleList)
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            encoder_output: (batch, seq_len, d_model)
        """
        # Embed tokens (use cached scaling factor)
        x: torch.Tensor = self.embedding(input_ids) * self.embedding_scale
        x = self.dropout(x)
        
        # Create attention mask for padding
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=x.device, dtype=x.dtype)
            # Convert to (batch, 1, 1, seq_len) for broadcasting
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # Convert 0/1 to -inf/0 for masking
            attention_mask = (1.0 - attention_mask) * torch.finfo(x.dtype).min

        # Apply encoder layers
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask, rope=self.rope)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class TransformerDecoder(nn.Module):
    """
    Stack of Decoder Layers with embeddings and output projection.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        d_ff: int,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.tie_embeddings = tie_embeddings
        # Cache scaling factor to avoid computing sqrt every forward pass
        self.embedding_scale = math.sqrt(d_model)

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Rotary position embeddings
        self.rope = RotaryPositionEmbedding(
            dim=d_model // n_heads,
            max_seq_len=max_seq_len
        )
        
        # Decoder layers
        layers: List[DecoderLayer] = [
            DecoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self._decoder_layers: List[DecoderLayer] = layers
        
        # Final normalization
        self.final_norm = RMSNorm(d_model)
        
        # Output projection
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output)
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List['LayerKVCache']] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List['LayerKVCache']]]:
        """
        Args:
            input_ids: (batch, tgt_len)
            encoder_output: (batch, src_len, d_model)
            self_attention_mask: (batch, tgt_len) - for padding
            cross_attention_mask: (batch, src_len) - for encoder padding
            past_key_values: Optional list of LayerKVCache for each layer
            use_cache: Whether to return updated cache

        Returns:
            logits: (batch, tgt_len, vocab_size)
            updated_caches: List of LayerKVCache if use_cache=True, else None
        """
        # Embed tokens (use cached scaling factor)
        x = self.embedding(input_ids) * self.embedding_scale
        x = self.dropout(x)

        # Create self-attention mask (causal mask is handled in layer)
        if self_attention_mask is not None:
            self_attention_mask = self_attention_mask.unsqueeze(1).unsqueeze(2)
            self_attention_mask = (1.0 - self_attention_mask) * torch.finfo(x.dtype).min

        # Create cross-attention mask
        if cross_attention_mask is not None:
            cross_attention_mask = cross_attention_mask.unsqueeze(1).unsqueeze(2)
            cross_attention_mask = (1.0 - cross_attention_mask) * torch.finfo(x.dtype).min
        
        if use_cache:
            if past_key_values is None:
                from .generation_utils import LayerKVCache as _LayerKVCache
                past_key_values = [_LayerKVCache() for _ in self._decoder_layers]
            elif len(past_key_values) != len(self._decoder_layers):
                raise ValueError(
                    f"Expected {len(self._decoder_layers)} past key values, got {len(past_key_values)}"
                )

        # Initialize caches list if needed
        updated_caches: Optional[List['LayerKVCache']] = [] if use_cache else None

        # Apply decoder layers
        for i, layer in enumerate(self._decoder_layers):
            layer_cache = past_key_values[i] if past_key_values else None
            x, new_cache = layer(
                x,
                encoder_output=encoder_output,
                self_attention_mask=self_attention_mask,
                cross_attention_mask=cross_attention_mask,
                rope=self.rope,
                layer_cache=layer_cache,
                use_cache=use_cache,
            )

            if use_cache and updated_caches is not None:
                if new_cache is None:
                    raise ValueError("Decoder layer did not return cache while use_cache=True")
                updated_caches.append(new_cache)

        # Final normalization and projection
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, updated_caches


# Test encoder and decoder
if __name__ == "__main__":
    pass
