"""
Generation Utilities for Mobile Translation Model
Includes KV caching and beam search for efficient and high-quality generation
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .utils import logger

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

@dataclass
class KVCache:
    """
    Key-Value cache for efficient autoregressive generation.
    Stores past key and value tensors to avoid recomputation.
    Shapes are tracked to prevent mixing full heads and grouped KV heads.
    """
    key: Optional[torch.Tensor] = None  # (batch, n_kv_heads, seq_len, head_dim)
    value: Optional[torch.Tensor] = None  # (batch, n_kv_heads, seq_len, head_dim)
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    
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

        if self.key is not None:
            if self.key.shape[0] != batch:
                raise ValueError("KVCache batch size mismatch during update")
            if self.key.shape[1] != heads or self.key.shape[3] != head_dim:
                raise ValueError("KVCache head dimensions changed during update")

    def update(
        self,
        new_key: torch.Tensor,
        new_value: torch.Tensor,
    ) -> 'KVCache':
        """
        Update cache with new key-value pairs.

        Args:
            new_key: New key tensor (batch, n_kv_heads, new_len, head_dim)
            new_value: New value tensor (batch, n_kv_heads, new_len, head_dim)

        Returns:
            Updated KVCache instance
        """
        self._validate_new(new_key, new_value)

        if self.key is None or self.value is None:
            self.key = new_key
            self.value = new_value
        else:
            # Concatenate along sequence dimension
            self.key = torch.cat([self.key, new_key], dim=2)
            self.value = torch.cat([self.value, new_value], dim=2)

        return self

    def as_tuple(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Return cache as tuple if populated."""
        if self.key is None or self.value is None:
            return None
        return self.key, self.value

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'KVCache':
        """Move cache tensors to device/dtype."""
        if self.key is not None:
            self.key = self.key.to(device=device, dtype=dtype)
        if self.value is not None:
            self.value = self.value.to(device=device, dtype=dtype)
        return self
    
    def get_seq_len(self) -> int:
        """Get current sequence length in cache."""
        if self.key is None:
            return 0
        return self.key.shape[2]


class LayerKVCache:
    """Cache for a single layer containing self-attention and cross-attention KVs."""

    def __init__(self) -> None:
        self.self_attn_cache: KVCache = KVCache()
        self.cross_attn_cache: KVCache = KVCache()  # Only computed once per generation

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None) -> 'LayerKVCache':
        self.self_attn_cache.to(device, dtype=dtype)
        self.cross_attn_cache.to(device, dtype=dtype)
        return self


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

            # Mark batch done as soon as we have num_beams finished hypotheses,
            # regardless of early_stopping flag. This prevents endless decoding
            # when a sufficient number of hypotheses are complete.
            if finished_count >= num_beams:
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
    
    Returns:
        Generated sequences (batch, gen_len)
    """
    model.eval()
    device = src_input_ids.device
    batch_size = src_input_ids.shape[0]

    if top_k is not None and top_k < 0:
        raise ValueError("top_k must be non-negative or None")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be within (0, 1].")

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

        past_key_values = [LayerKVCache().to(device) for _ in model.decoder.layers]

        for step in range(max_length - 1):
            decoder_input = generated[:, -1:]
            tgt_mask = torch.ones_like(decoder_input, dtype=torch.float, device=device)

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

            # Check for invalid logits before softmax (e.g., all -inf)
            if torch.isinf(next_token_logits).all(dim=-1).any() or torch.isnan(next_token_logits).any():
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits, dim=-1)
                if torch.isnan(probs).any() or (probs.sum(dim=-1) == 0).any():
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                else:
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
        
        # Generate
        for step in range(max_length - 1):
            # Forward pass
            tgt_mask = torch.ones_like(input_ids, dtype=torch.float)

            logits, _ = model.decoder(
                input_ids=input_ids,
                encoder_output=encoder_output,
                self_attention_mask=tgt_mask,
                cross_attention_mask=src_attention_mask,
            )

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
            # Convert flat indices to beam indices: next_tokens ∈ [0, num_beams*vocab_size) -> [0, num_beams)
            assert next_indices.max().item() < num_beams, "Beam index out of bounds"
            next_tokens = next_tokens % vocab_size
            
            # Select best num_beams for each batch
            beam_outputs: List[torch.Tensor] = []
            beam_next_tokens: List[int] = []
            beam_idx_list: List[int] = []
            
            for batch_idx in range(batch_size):
                if beam_scorer.done[batch_idx]:
                    # This batch is done, pad all num_beams to maintain shape
                    first_beam_idx = batch_idx * num_beams
                    for i in range(num_beams):
                        beam_outputs.append(input_ids[first_beam_idx + i])
                        beam_next_tokens.append(pad_token_id)
                        beam_idx_list.append(first_beam_idx + i)
                    continue
                
                # Select num_beams
                batch_next_scores: List[float] = []
                batch_next_tokens: List[torch.Tensor] = []
                batch_next_indices: List[int] = []
                
                for beam_idx in range(num_beams):
                    if len(batch_next_scores) >= num_beams:
                        break
                    
                    idx = beam_idx
                    next_token = next_tokens[batch_idx, idx]
                    next_score = next_scores[batch_idx, idx]
                    beam_idx_in_batch = int(next_indices[batch_idx, idx].item())
                    
                    batch_next_scores.append(float(next_score.item()))
                    batch_next_tokens.append(next_token)
                    batch_next_indices.append(beam_idx_in_batch)
                
                # Update this batch's beams
                for i in range(num_beams):
                    beam_idx_global = batch_idx * num_beams + batch_next_indices[i]
                    prev_input = input_ids[beam_idx_global]

                    beam_outputs.append(prev_input)
                    # Convert 0-D tensor to Python int to avoid indexing/concatenation errors
                    beam_next_tokens.append(int(batch_next_tokens[i].item()))
                    beam_idx_list.append(beam_idx_global)
            
            # Create new input_ids
            input_ids = torch.cat([
                input_ids[beam_idx_list],
                torch.tensor(beam_next_tokens, device=device, dtype=torch.long).unsqueeze(1)
            ], dim=1)
            
            # Flatten scores (avoid Python loop to prevent CPU migration on GPU)
            beam_scores = next_scores[:, :num_beams].flatten()
            
            # Check for EOS and update beam scorer
            input_ids, beam_scores, done = beam_scorer.process(
                input_ids=input_ids,
                next_scores=beam_scores,
                next_tokens=input_ids[:, -1],
                next_indices=torch.tensor(beam_idx_list, device=device),
            )

            # Suppress beams that have emitted EOS to prevent them from continuing
            # This prevents completed beams from crowding out unfinished ones
            eos_mask = input_ids[:, -1] == eos_token_id
            beam_scores = beam_scores.masked_fill(eos_mask, float('-inf'))

            # Early stopping
            if done.all():
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
