"""
Generation Utilities for Mobile Translation Model
Includes KV caching and beam search for efficient and high-quality generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import InputValidator, logger

if TYPE_CHECKING:
    from .mobile_translation_model import MobileTranslationModel

__all__ = [
    "KVCache",
    "LayerKVCache",
    "BeamHypothesis",
    "BeamSearchScorer",
    "generate_with_kv_cache",
    "generate_with_beam_search",
]


# ============================================================================
# KV CACHE DATA STRUCTURE
# ============================================================================

class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.

    Two storage modes are supported transparently:

    * **Lazy mode** (default): the cache starts empty; ``update`` concatenates
      each new step into a fresh ``torch.cat`` tensor. O(n) work per step and
      O(n²) total memory churn over a decode. Matches the historical behaviour.
    * **Reserved mode**: call :meth:`reserve` once at the start of a decode
      with an upper-bound ``max_len``. ``update`` then copies new K/V slices
      into a pre-allocated buffer in-place and only exposes the valid prefix.
      Converts a full decode to O(n) work *and* O(n) peak memory.
    """

    __slots__ = (
        "_key_buf",
        "_value_buf",
        "_length",
        "_capacity",
        "num_heads",
        "head_dim",
    )

    def __init__(
        self,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        num_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ) -> None:
        self._key_buf: Optional[torch.Tensor] = key
        self._value_buf: Optional[torch.Tensor] = value
        self._length: int = 0 if key is None else int(key.shape[2])
        # ``_capacity`` is ``None`` in lazy mode; set in :meth:`reserve`.
        self._capacity: Optional[int] = None
        self.num_heads = num_heads if num_heads is not None else (key.shape[1] if key is not None else None)
        self.head_dim = head_dim if head_dim is not None else (key.shape[3] if key is not None else None)

    # ------------------------------------------------------------------
    # Public tensor views (backward compatible with the dataclass API)
    # ------------------------------------------------------------------
    @property
    def key(self) -> Optional[torch.Tensor]:
        if self._key_buf is None or self._length == 0:
            return None
        if self._capacity is None:
            return self._key_buf
        return self._key_buf[:, :, : self._length, :]

    @key.setter
    def key(self, value: Optional[torch.Tensor]) -> None:
        self._key_buf = value
        self._length = 0 if value is None else int(value.shape[2])
        self._capacity = None

    @property
    def value(self) -> Optional[torch.Tensor]:
        if self._value_buf is None or self._length == 0:
            return None
        if self._capacity is None:
            return self._value_buf
        return self._value_buf[:, :, : self._length, :]

    @value.setter
    def value(self, v: Optional[torch.Tensor]) -> None:
        self._value_buf = v
        # length already set by key setter in normal usage; keep in sync if
        # only ``value`` is reassigned.
        if v is not None and self._length == 0:
            self._length = int(v.shape[2])

    # ------------------------------------------------------------------
    # Pre-allocation
    # ------------------------------------------------------------------
    def reserve(
        self,
        batch: int,
        heads: int,
        max_len: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> 'KVCache':
        """Allocate a fixed (batch, heads, max_len, head_dim) buffer.

        After ``reserve`` the cache is empty (``_length = 0``); call
        :meth:`update` to append new K/V slices in-place.
        """
        if max_len <= 0:
            raise ValueError("max_len must be positive")
        self._key_buf = torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
        self._value_buf = torch.empty(batch, heads, max_len, head_dim, device=device, dtype=dtype)
        self._capacity = max_len
        self._length = 0
        self.num_heads = heads
        self.head_dim = head_dim
        return self

    def _validate_new(self, new_key: torch.Tensor, new_value: torch.Tensor) -> None:
        if new_key.ndim != 4 or new_value.ndim != 4:
            raise ValueError("KVCache expects 4D tensors shaped (batch, heads, seq_len, head_dim)")
        if new_key.shape != new_value.shape:
            raise ValueError("Key and value tensors must share the same shape")
        batch, heads, _, head_dim = new_key.shape

        if self.num_heads is None:
            self.num_heads = heads
        if self.head_dim is None:
            self.head_dim = head_dim

        if heads != self.num_heads or head_dim != self.head_dim:
            raise ValueError(
                f"KVCache head mismatch: expected heads={self.num_heads}, head_dim={self.head_dim}, "
                f"got heads={heads}, head_dim={head_dim}"
            )

        if self._key_buf is not None:
            if self._key_buf.shape[0] != batch:
                raise ValueError("KVCache batch size mismatch during update")
            if self._key_buf.shape[1] != heads or self._key_buf.shape[3] != head_dim:
                raise ValueError("KVCache head dimensions changed during update")

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> 'KVCache':
        """Append new K/V slices.

        In reserved mode the new slice is copied into the pre-allocated buffer
        at the current write head (no ``torch.cat``). In lazy mode this falls
        back to ``torch.cat`` along the sequence dim.
        """
        self._validate_new(new_key, new_value)
        new_len = int(new_key.shape[2])

        if self._capacity is not None and self._key_buf is not None and self._value_buf is not None:
            end = self._length + new_len
            if end <= self._capacity:
                self._key_buf[:, :, self._length:end, :].copy_(new_key)
                self._value_buf[:, :, self._length:end, :].copy_(new_value)
                self._length = end
                return self
            # Over capacity: drop back to lazy ``torch.cat`` semantics so the
            # cache still works, at the cost of an extra allocation.
            existing_key = self._key_buf[:, :, : self._length, :]
            existing_value = self._value_buf[:, :, : self._length, :]
            self._key_buf = torch.cat([existing_key, new_key], dim=2)
            self._value_buf = torch.cat([existing_value, new_value], dim=2)
            self._length = self._key_buf.shape[2]
            self._capacity = None
            return self

        if self._key_buf is None or self._value_buf is None:
            self._key_buf = new_key
            self._value_buf = new_value
            self._length = new_len
        else:
            self._key_buf = torch.cat([self._key_buf, new_key], dim=2)
            self._value_buf = torch.cat([self._value_buf, new_value], dim=2)
            self._length = self._key_buf.shape[2]
        return self

    def as_tuple(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return cache as tuple if populated."""
        k, v = self.key, self.value
        if k is None or v is None:
            return None
        return k, v

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'KVCache':
        """Move cache tensors to device/dtype."""
        if self._key_buf is not None:
            self._key_buf = self._key_buf.to(device=device, dtype=dtype)
        if self._value_buf is not None:
            self._value_buf = self._value_buf.to(device=device, dtype=dtype)
        return self

    def reorder(self, beam_indices: torch.Tensor) -> 'KVCache':
        """Reorder cached batch entries after beam selection."""
        if self._key_buf is None or self._value_buf is None:
            return self
        self._key_buf = self._key_buf.index_select(0, beam_indices.to(device=self._key_buf.device))
        self._value_buf = self._value_buf.index_select(0, beam_indices.to(device=self._value_buf.device))
        return self

    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        return self._length


class LayerKVCache:
    """Cache for a single layer containing self-attention and cross-attention KVs."""

    def __init__(self) -> None:
        self.self_attn_cache: KVCache = KVCache()
        self.cross_attn_cache: KVCache = KVCache()  # Only computed once per generation

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'LayerKVCache':
        self.self_attn_cache.to(device, dtype=dtype)
        self.cross_attn_cache.to(device, dtype=dtype)
        return self

    def reorder(self, beam_indices: torch.Tensor) -> 'LayerKVCache':
        self.self_attn_cache.reorder(beam_indices)
        self.cross_attn_cache.reorder(beam_indices)
        return self


def _reorder_past_key_values(
    past_key_values: Optional[List[LayerKVCache]],
    beam_indices: torch.Tensor,
) -> Optional[List[LayerKVCache]]:
    """Reorder decoder caches to match the new beam ordering."""
    if past_key_values is None:
        return None
    return [layer_cache.reorder(beam_indices) for layer_cache in past_key_values]


def _build_preallocated_caches(
    model: "MobileTranslationModel",
    batch_size: int,
    max_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[LayerKVCache]:
    """Create pre-allocated KV caches for the decoder's self-attention blocks.

    Falls back to empty (lazy-cat) caches if the decoder does not expose the
    expected introspection attributes (e.g. under tests that monkey-patch
    ``model.decoder`` with a non-standard object).
    """
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "_decoder_layers", None) or getattr(decoder, "layers", None)
    if layers is None:
        return []

    try:
        first_self_attn = layers[0].self_attn
        n_kv_heads = int(first_self_attn.n_kv_heads)
        head_dim = int(first_self_attn.head_dim)
    except (AttributeError, IndexError, TypeError):
        # Introspection failed; return lazy caches so behaviour matches the
        # pre-audit code path.
        return [LayerKVCache() for _ in layers]

    caches: List[LayerKVCache] = []
    for _ in layers:
        layer_cache = LayerKVCache()
        layer_cache.self_attn_cache.reserve(
            batch=batch_size,
            heads=n_kv_heads,
            max_len=max_length,
            head_dim=head_dim,
            device=device,
            dtype=dtype,
        )
        caches.append(layer_cache)
    return caches


def _apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Apply nucleus (top-p) filtering to logits.

    Tokens outside the cumulative probability mass are masked to -inf so they
    never get sampled.
    """
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be within (0, 1].")

    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits.float(), dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
    return logits.masked_fill(indices_to_remove, float('-inf'))


# ============================================================================
# BEAM SEARCH
# ============================================================================

class BeamHypothesis:
    """
    Single hypothesis in beam search.
    """
    
    def __init__(
        self,
        tokens: torch.Tensor,
        score: float,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Args:
            tokens: Token sequence (seq_len,)
            score: Log probability score
            attention_mask: Attention mask (seq_len,)
        """
        self.tokens: torch.Tensor = tokens
        self.score: float = float(score)
        self.attention_mask = attention_mask
    
    def __len__(self) -> int:
        return int(self.tokens.shape[0])
    
    def average_score(self, length_penalty: float = 1.0) -> float:
        """
        Get length-normalized score.

        Args:
            length_penalty: Length penalty (>1.0 encourages longer sequences)

        Returns:
            Normalized score: score / (length ** length_penalty)
        """
        length = float(len(self))
        if length == 0.0:
            # Avoid division by zero on malformed hypotheses
            return float("-inf")
        penalized_length: float = math.pow(length, float(length_penalty))
        return float(self.score) / penalized_length


class BeamSearchScorer:
    """
    Manages beam search for multiple batches.
    """
    
    def __init__(
        self,
        batch_size: int,
        num_beams: int,
        device: torch.device,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        eos_token_id: int = 2,
    ) -> None:
        """
        Args:
            batch_size: Number of sequences in batch
            num_beams: Number of beams per sequence
            device: Device to use
            length_penalty: Length penalty (>1.0 encourages longer sequences)
            early_stopping: Whether to stop when num_beams hypotheses are done
            eos_token_id: End-of-sequence token ID
        """
        self.batch_size = batch_size
        self.num_beams = num_beams
        self.device = device
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.eos_token_id = eos_token_id

        # Store finished hypotheses for each batch
        self.finished_hypotheses: List[List[BeamHypothesis]] = [[] for _ in range(batch_size)]

        # Track which sequences are done
        self.done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Track which individual beams have emitted EOS to avoid duplicate storage
        self.beam_is_finished = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device
        )
    
    def process(
        self,
        input_ids: torch.Tensor,
        next_scores: torch.Tensor,
        next_tokens: torch.Tensor,
        next_indices: torch.Tensor,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process scores and select next beams.
        
        Args:
            input_ids: Current sequences (batch * num_beams, seq_len)
            next_scores: Scores for next tokens (batch * num_beams, vocab_size)
            next_tokens: Selected next tokens (batch * num_beams,)
            next_indices: Beam indices for next tokens (batch * num_beams,)
            eos_token_id: EOS token ID (uses self.eos_token_id if None)
        
        Returns:
            Tuple of (updated input_ids, scores, finished flags)
        """
        if eos_token_id is None:
            eos_token_id = self.eos_token_id
        
        batch_size = self.batch_size
        num_beams = self.num_beams
        
        # Check which beams hit EOS
        eos_mask = next_tokens == eos_token_id
        
        # Process each batch separately
        for batch_idx in range(batch_size):
            if self.done[batch_idx]:
                continue

            # Get beams for this batch
            start_idx = batch_idx * num_beams
            end_idx = start_idx + num_beams

            batch_eos_mask = eos_mask[start_idx:end_idx]

            # Save finished hypotheses
            for beam_idx, is_eos in enumerate(batch_eos_mask):
                if is_eos:
                    global_idx = start_idx + beam_idx

                    # Skip if this beam was already marked finished in a previous step
                    if self.beam_is_finished[batch_idx, beam_idx]:
                        continue

                    # Create hypothesis
                    hypothesis = BeamHypothesis(
                        tokens=input_ids[global_idx].clone(),
                        score=next_scores[global_idx].item(),
                    )

                    self.finished_hypotheses[batch_idx].append(hypothesis)
                    self.beam_is_finished[batch_idx, beam_idx] = True

            # Check if this batch is done
            finished_count = len(self.finished_hypotheses[batch_idx])

            # Respect early_stopping while still terminating when all beams are done.
            if self.early_stopping:
                done_now = finished_count >= num_beams
            else:
                done_now = bool(self.beam_is_finished[batch_idx].all().item())

            if done_now:
                self.done[batch_idx] = True

        return input_ids, next_scores, self.done
    
    def finalize(
        self,
        input_ids: torch.Tensor,
        final_scores: torch.Tensor,
        max_length: int,
    ) -> torch.Tensor:
        """
        Finalize beam search and return best hypotheses.
        
        Args:
            input_ids: Final sequences (batch * num_beams, seq_len)
            final_scores: Final scores (batch * num_beams,)
            max_length: Maximum sequence length
        
        Returns:
            Best sequences (batch, max_length)
        """
        batch_size = self.batch_size
        num_beams = self.num_beams
        
        # Get best hypothesis for each batch
        best_sequences = []
        
        for batch_idx in range(batch_size):
            if len(self.finished_hypotheses[batch_idx]) > 0:
                # Sort by score and get best
                hypotheses = self.finished_hypotheses[batch_idx]
                hypotheses.sort(key=lambda h: h.average_score(self.length_penalty), reverse=True)
                best = hypotheses[0]
                best_seq = best.tokens
            else:
                # No finished hypothesis, use best current beam
                start_idx = batch_idx * num_beams
                end_idx = start_idx + num_beams
                
                batch_scores = final_scores[start_idx:end_idx]
                best_beam_idx = batch_scores.argmax()
                best_seq = input_ids[start_idx + best_beam_idx]
            
            # Pad to max_length if needed
            if len(best_seq) < max_length:
                padding = torch.full(
                    (max_length - len(best_seq),),
                    self.eos_token_id,
                    dtype=best_seq.dtype,
                    device=best_seq.device,
                )
                best_seq = torch.cat([best_seq, padding])
            elif len(best_seq) > max_length:
                best_seq = best_seq[:max_length]
            
            best_sequences.append(best_seq)
        
        return torch.stack(best_sequences)


# ============================================================================
# GENERATION WITH KV CACHING
# ============================================================================

def generate_with_kv_cache(
    model: "MobileTranslationModel",
    src_input_ids: torch.Tensor,
    src_attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 128,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float = 1.0,
    pad_token_id: int = 0,
    do_sample: bool = False,
) -> torch.Tensor:
    """
    Generate translation with KV caching for 2-3x speedup.

    Args:
        model: Translation model
        src_input_ids: Source token IDs (batch, src_len)
        src_attention_mask: Source attention mask
        max_length: Maximum generation length
        sos_token_id: Start-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        temperature: Sampling temperature
        top_k: Top-k sampling (0 = disabled)
        top_p: Nucleus sampling (1.0 = disabled)
        pad_token_id: Padding token ID
        do_sample: If True, sample via multinomial; if False (default), take argmax.
            Prior versions always sampled, which diverged from the non-cached
            ``generate()`` default. Keeping ``do_sample=False`` makes greedy
            decoding deterministic and matches the reference path.

    Returns:
        Generated sequences (batch, gen_len)
    """
    InputValidator.validate_positive_int(max_length, "max_length", min_value=1, max_value=2048)
    model.eval()
    device = src_input_ids.device
    batch_size = src_input_ids.shape[0]

    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative or None")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be within (0, 1].")

    InputValidator.validate_positive_float(temperature, "temperature", min_value=1e-8)
    temperature = max(0.01, float(temperature))

    logger.info(
        "Generating with KV cache: batch_size=%d, max_length=%d", batch_size, max_length
    )

    with torch.no_grad():
        encoder_output = model.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
        )

        generated = torch.full(
            (batch_size, 1),
            sos_token_id,
            dtype=torch.long,
            device=device,
        )

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Pre-allocate self-attention KV buffers for the whole decode up-front
        # so each step writes into a fixed buffer instead of allocating a new
        # concatenated tensor. Cross-attention KV is populated on the first
        # decoder call and then reused (shape stays constant).
        past_key_values = _build_preallocated_caches(
            model=model,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
            dtype=encoder_output.dtype,
        )

        # Per-step target-side mask is always a single valid token; pre-allocate
        # it once so the decode loop doesn't repeatedly ``ones_like`` / move a
        # fresh tensor onto the device on every step.
        tgt_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)

        for step in range(max_length - 1):
            decoder_input = generated[:, -1:]

            decoder_outputs = model.decoder(
                input_ids=decoder_input,
                encoder_output=encoder_output,
                self_attention_mask=tgt_mask,
                cross_attention_mask=src_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )

            logits, past_key_values = decoder_outputs

            next_token_logits = logits[:, -1, :] / temperature

            if top_k is not None and top_k > 0:
                top_k_val = min(top_k, next_token_logits.shape[-1])
                threshold = torch.topk(next_token_logits, top_k_val)[0][..., -1, None]
                next_token_logits = next_token_logits.masked_fill(
                    next_token_logits < threshold, float('-inf')
                )

            if top_p < 1.0:
                next_token_logits = _apply_top_p_filter(next_token_logits, top_p)

            if not do_sample:
                # Deterministic argmax path (matches default `generate()` semantics).
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                # Nan-safe softmax without three separate CPU-GPU syncs.
                # nan_to_num replaces inf/nan in-place on the GPU; rows where
                # every logit was masked out collapse to a uniform distribution
                # which still gives a valid categorical sample.
                safe_logits = torch.nan_to_num(
                    next_token_logits,
                    nan=0.0,
                    posinf=0.0,
                    neginf=float("-1e30"),
                )
                probs = F.softmax(safe_logits, dim=-1)
                # Guard against rows that summed to 0 (all -inf post-filtering)
                # by falling back to argmax on the original logits for those rows.
                row_sums = probs.sum(dim=-1, keepdim=True)
                empty_rows = row_sums <= 0
                fallback = F.one_hot(
                    next_token_logits.argmax(dim=-1), probs.shape[-1]
                ).to(probs.dtype)
                probs = torch.where(empty_rows, fallback, probs)
                next_token = torch.multinomial(probs, num_samples=1)

            next_token = torch.where(
                finished.unsqueeze(1),
                torch.full_like(next_token, pad_token_id),
                next_token,
            )

            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == eos_token_id)

            if finished.all():
                logger.info("Early stopping at step %d", step + 1)
                break

    logger.info("Generation complete: final_length=%d", generated.shape[1])
    return generated


# ============================================================================
# BEAM SEARCH GENERATION
# ============================================================================

def generate_with_beam_search(
    model: "MobileTranslationModel",
    src_input_ids: torch.Tensor,
    src_attention_mask: Optional[torch.Tensor] = None,
    max_length: int = 128,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    early_stopping: bool = True,
    sos_token_id: int = 1,
    eos_token_id: int = 2,
    pad_token_id: int = 0,
) -> torch.Tensor:
    """
    Generate translation with beam search for higher quality.
    Expected improvement: +2-4 BLEU points over greedy decoding.

    **Performance Note**: This recomputes the full decoder sequence at each
    step (O(n²) per beam). Use ``generate_with_kv_cache`` for O(n) greedy.
    
    Args:
        model: Translation model
        src_input_ids: Source token IDs (batch, src_len)
        src_attention_mask: Source attention mask
        max_length: Maximum generation length
        num_beams: Number of beams
        length_penalty: Length penalty (>1.0 encourages longer sequences)
        early_stopping: Stop when num_beams hypotheses complete
        sos_token_id: Start token ID
        eos_token_id: End token ID
        pad_token_id: Padding token ID
    
    Returns:
        Best sequences (batch, max_length)
    """
    InputValidator.validate_positive_int(max_length, "max_length", min_value=1, max_value=2048)
    InputValidator.validate_positive_int(num_beams, "num_beams", min_value=1)
    model.eval()
    device = src_input_ids.device
    batch_size = src_input_ids.shape[0]
    
    logger.info(f"Generating with beam search: batch_size={batch_size}, num_beams={num_beams}")
    
    with torch.no_grad():
        # Encode source
        encoder_output = model.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
        )
        
        # Expand for beams
        encoder_output = encoder_output.unsqueeze(1).repeat(1, num_beams, 1, 1)
        # (B, src_len, d_model) -> (B, 1, src_len, d_model) -> (B, num_beams, src_len, d_model)
        encoder_output = encoder_output.view(batch_size * num_beams, -1, encoder_output.shape[-1])
        # (B, num_beams, src_len, d_model) -> (B*num_beams, src_len, d_model)
        
        if src_attention_mask is not None:
            src_attention_mask = src_attention_mask.unsqueeze(1).repeat(1, num_beams, 1)
            src_attention_mask = src_attention_mask.view(batch_size * num_beams, -1)
        
        # Initialize beam search
        beam_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            eos_token_id=eos_token_id,
        )
        
        # Start with SOS token for all beams
        input_ids = torch.full(
            (batch_size * num_beams, 1),
            sos_token_id,
            dtype=torch.long,
            device=device,
        )
        
        # Initialize beam scores (0 for first beam of each batch, -inf for others)
        beam_scores = torch.full((batch_size * num_beams,), float('-inf'), device=device)
        # Activate first beam of each batch (indices 0, num_beams, 2*num_beams, ...)
        beam_scores[::num_beams] = 0.0

        # Pre-allocate self-attention KV cache buffers per-beam. Cross-attention
        # is populated once on the first decoder call and never grows.
        past_key_values: Optional[List[LayerKVCache]] = _build_preallocated_caches(
            model=model,
            batch_size=batch_size * num_beams,
            max_length=max_length,
            device=device,
            dtype=encoder_output.dtype,
        )
        
        # Per-step self-attention mask is a single valid token per beam;
        # pre-allocate once to avoid a fresh ``ones_like`` every step.
        tgt_mask = torch.ones(batch_size * num_beams, 1, dtype=torch.bool, device=device)

        # Generate
        for step in range(max_length - 1):
            decoder_input = input_ids[:, -1:]

            logits, updated_past_key_values = model.decoder(
                input_ids=decoder_input,
                encoder_output=encoder_output,
                self_attention_mask=tgt_mask,
                cross_attention_mask=src_attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            if updated_past_key_values is not None:
                past_key_values = updated_past_key_values

            # Get next token logits
            next_token_logits = logits[:, -1, :]  # (batch * num_beams, vocab_size)
            
            # Convert to log probabilities
            next_token_scores = F.log_softmax(next_token_logits.float(), dim=-1)
            
            # Add beam scores
            next_token_scores = next_token_scores + beam_scores[:, None]
            
            # Reshape for beam selection: (batch, num_beams * vocab_size)
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)
            
            # Select top 2 * num_beams (to have options after removing finished beams)
            next_scores, next_tokens = torch.topk(
                next_token_scores,
                2 * num_beams,
                dim=1,
                largest=True,
                sorted=True,
            )
            
            next_indices = torch.div(next_tokens, vocab_size, rounding_mode='floor')
            next_tokens = next_tokens % vocab_size

            # Pull 2*num_beams candidates to host in a single sync to avoid
            # per-candidate `.item()` calls in the inner selection loop below.
            cand_tokens_cpu = next_tokens.cpu().tolist()
            cand_scores_cpu = next_scores.cpu().tolist()
            cand_beams_cpu = next_indices.cpu().tolist()

            # Snapshot the per-beam-finished flags and the done flags once per
            # step as Python lists so the inner candidate loop does not perform
            # a CPU-GPU sync on every iteration (previously ``.item()`` was
            # called O(batch * 2 * num_beams) times per step).
            beam_finished_cpu = beam_scorer.beam_is_finished.tolist()
            done_cpu = beam_scorer.done.tolist()

            # Pre-allocate per-batch selection buffers sized for num_beams.
            new_beam_idx_global = [0] * (batch_size * num_beams)
            new_beam_next_tokens = [pad_token_id] * (batch_size * num_beams)
            new_beam_scores = [float('-inf')] * (batch_size * num_beams)

            # Collect per-batch finished-state updates to apply to the scorer
            # tensors once at the end of the step (one scatter instead of many).
            finished_updates: List[Tuple[int, int]] = []
            done_updates: List[int] = []

            for batch_idx in range(batch_size):
                first_beam_global = batch_idx * num_beams

                if done_cpu[batch_idx]:
                    # This batch is done, pad all num_beams to maintain shape.
                    for i in range(num_beams):
                        new_beam_idx_global[first_beam_global + i] = first_beam_global + i
                        new_beam_next_tokens[first_beam_global + i] = pad_token_id
                        new_beam_scores[first_beam_global + i] = float('-inf')
                    continue

                selected = 0
                for cand_pos in range(2 * num_beams):
                    if selected >= num_beams:
                        break

                    token_id = int(cand_tokens_cpu[batch_idx][cand_pos])
                    score = float(cand_scores_cpu[batch_idx][cand_pos])
                    beam_offset = int(cand_beams_cpu[batch_idx][cand_pos])
                    source_global = first_beam_global + beam_offset

                    if token_id == eos_token_id:
                        # Finish this hypothesis but do not consume a live beam slot.
                        if not beam_finished_cpu[batch_idx][beam_offset]:
                            prev_seq = input_ids[source_global]
                            eos_tensor = torch.tensor(
                                [eos_token_id], device=prev_seq.device, dtype=prev_seq.dtype
                            )
                            full_seq = torch.cat([prev_seq, eos_tensor])
                            beam_scorer.finished_hypotheses[batch_idx].append(
                                BeamHypothesis(tokens=full_seq.clone(), score=score)
                            )
                            beam_finished_cpu[batch_idx][beam_offset] = True
                            finished_updates.append((batch_idx, beam_offset))
                        continue

                    slot = first_beam_global + selected
                    new_beam_idx_global[slot] = source_global
                    new_beam_next_tokens[slot] = token_id
                    new_beam_scores[slot] = score
                    selected += 1

                # If we could not fill all beam slots (rare: all candidates were EOS),
                # pad remaining slots with the first candidate's source beam so the
                # tensor shape stays consistent. These slots are suppressed to -inf.
                while selected < num_beams:
                    slot = first_beam_global + selected
                    fallback_beam = int(cand_beams_cpu[batch_idx][0])
                    new_beam_idx_global[slot] = first_beam_global + fallback_beam
                    new_beam_next_tokens[slot] = pad_token_id
                    new_beam_scores[slot] = float('-inf')
                    selected += 1

                # Early-stopping bookkeeping: a batch is done when it has gathered
                # num_beams finished hypotheses (mirrors BeamSearchScorer semantics).
                if early_stopping:
                    if len(beam_scorer.finished_hypotheses[batch_idx]) >= num_beams:
                        done_updates.append(batch_idx)
                else:
                    if all(beam_finished_cpu[batch_idx]):
                        done_updates.append(batch_idx)

            # Flush per-step bookkeeping to the scorer tensors in one shot.
            for bi, bo in finished_updates:
                beam_scorer.beam_is_finished[bi, bo] = True
            for bi in done_updates:
                beam_scorer.done[bi] = True

            # Create new input_ids using the selected live beams.
            beam_idx_tensor = torch.tensor(
                new_beam_idx_global, device=device, dtype=torch.long
            )
            next_token_tensor = torch.tensor(
                new_beam_next_tokens, device=device, dtype=torch.long
            ).unsqueeze(1)
            input_ids = torch.cat([input_ids[beam_idx_tensor], next_token_tensor], dim=1)

            beam_scores = torch.tensor(
                new_beam_scores, device=device, dtype=torch.float32
            )
            # ``encoder_output`` and ``src_attention_mask`` are already duplicated
            # per-beam and are identical across beams within a batch (they only
            # depend on the source sentence). Reordering by ``beam_idx_tensor``
            # produces a tensor with the same values but forces a new allocation
            # every step, so we skip it entirely and only reorder the decoder
            # KV caches, which actually track per-beam history.
            past_key_values = _reorder_past_key_values(past_key_values, beam_idx_tensor)

            # Early stopping. ``done_cpu`` already reflects the start-of-step
            # state; we augment it with the updates queued this step to avoid
            # an extra CPU-GPU sync on ``beam_scorer.done``.
            for bi in done_updates:
                done_cpu[bi] = True
            if all(done_cpu):
                logger.info(f"Beam search early stopping at step {step + 1}")
                break
        
        # Finalize and get best sequences
        best_sequences = beam_scorer.finalize(
            input_ids=input_ids,
            final_scores=beam_scores,
            max_length=max_length,
        )
    
    logger.info(f"Beam search complete: final_length={best_sequences.shape[1]}")
    return best_sequences


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    pass
