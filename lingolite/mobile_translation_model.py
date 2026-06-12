"""
Complete Mobile Translation Model
Encoder-Decoder transformer optimized for mobile deployment
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple, TYPE_CHECKING, TypedDict, Set, SupportsFloat, SupportsInt, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_decoder import TransformerDecoder, TransformerEncoder
from .generation_utils import generate_with_kv_cache, generate_with_beam_search
from .utils import InputValidator, logger

if TYPE_CHECKING:
    from .generation_utils import LayerKVCache


class MobileTranslationModel(nn.Module):
    """
    Complete translation model combining encoder and decoder.
    Optimized for mobile deployment with GQA, RoPE, and SwiGLU.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_encoder_layers: int = 6,
        n_decoder_layers: int = 6,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
        pad_token_id: int = 0,
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (hidden size)
            n_encoder_layers: Number of encoder layers
            n_decoder_layers: Number of decoder layers
            n_heads: Number of attention heads (queries)
            n_kv_heads: Number of key-value heads (for GQA)
            d_ff: Feed-forward dimension
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            tie_embeddings: Whether to tie input/output embeddings
            pad_token_id: ID for padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self._config: Dict[str, Any] = {
            "vocab_size": int(vocab_size),
            "d_model": int(d_model),
            "n_encoder_layers": int(n_encoder_layers),
            "n_decoder_layers": int(n_decoder_layers),
            "n_heads": int(n_heads),
            "n_kv_heads": int(n_kv_heads),
            "d_ff": int(d_ff),
            "max_seq_len": int(max_seq_len),
            "dropout": float(dropout),
            "tie_embeddings": bool(tie_embeddings),
            "pad_token_id": int(pad_token_id),
        }
        
        # Encoder
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_encoder_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        
        # Decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_decoder_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            dropout=dropout,
            tie_embeddings=tie_embeddings,
        )
        
        # Initialize weights BEFORE sharing/tying so the shared embedding is
        # not double-initialized (the previous order initialized the embedding
        # Parameter three times — once as encoder.embedding, again as
        # decoder.embedding, and a third time via decoder.lm_head's Linear init;
        # harmless while all three call ``normal_(0, 0.02)`` but a footgun if
        # ``_init_weights`` ever diverges between Embedding and Linear).
        self.apply(self._init_weights)

        # Share embeddings between encoder and decoder
        self.decoder.embedding = self.encoder.embedding
        # Re-tie LM head to the shared embedding if weight tying is enabled
        if tie_embeddings:
            self.decoder.lm_head.weight = self.decoder.embedding.weight
        
    def gradient_checkpointing_enable(self) -> None:
        """Turn on activation checkpointing for encoder + decoder layers.

        Reduces training-time activation memory at the cost of one extra
        forward per layer during backward. Only takes effect when the module
        is in training mode and ``use_cache`` is False, so inference paths
        (greedy/beam decoding) remain unaffected.
        """
        self.encoder.gradient_checkpointing = True
        self.decoder.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        """Turn off activation checkpointing."""
        self.encoder.gradient_checkpointing = False
        self.decoder.gradient_checkpointing = False

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with appropriate distributions."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List["LayerKVCache"]] = None,
        use_cache: bool = False,
        encoder_output: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[List["LayerKVCache"]], torch.Tensor]:
        """
        Forward pass for training or inference.

        Args:
            src_input_ids: Source token IDs (batch, src_len)
            tgt_input_ids: Target token IDs (batch, tgt_len)
            src_attention_mask: Source attention mask (batch, src_len)
            tgt_attention_mask: Target attention mask (batch, tgt_len)
            past_key_values: Optional list of LayerKVCache for cached generation
            use_cache: Whether to return cache for next generation step
            encoder_output: Pre-computed encoder output (optional, for cached generation)

        Returns:
            logits: Predicted logits (batch, tgt_len, vocab_size)
            updated_caches: List of LayerKVCache if use_cache=True, else None
            encoder_output: Encoder output (returned for caching)
        """
        # Lightweight shape-only validation: full finite/range checks are intentionally
        # omitted here because they force CPU-GPU synchronisation on every forward pass
        # (they are still performed in the public `generate*` APIs that receive raw
        # user inputs). Use `InputValidator` directly when inputs come from untrusted
        # sources.
        InputValidator.validate_tensor(
            src_input_ids, "src_input_ids", expected_dim=2, check_finite=False
        )
        InputValidator.validate_tensor(
            tgt_input_ids, "tgt_input_ids", expected_dim=2, check_finite=False
        )
        if src_attention_mask is not None:
            InputValidator.validate_tensor(
                src_attention_mask, "src_attention_mask", expected_dim=2, check_finite=False
            )
        if tgt_attention_mask is not None:
            InputValidator.validate_tensor(
                tgt_attention_mask, "tgt_attention_mask", expected_dim=2, check_finite=False
            )

        # Encode source (or use cached encoder output)
        if encoder_output is None:
            encoder_output = self.encoder(
                input_ids=src_input_ids,
                attention_mask=src_attention_mask,
            )

        # Decode target
        logits, updated_caches = self.decoder(
            input_ids=tgt_input_ids,
            encoder_output=encoder_output,
            self_attention_mask=tgt_attention_mask,
            cross_attention_mask=src_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        return logits, updated_caches, encoder_output
    
    def generate(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        num_beams: int = 1,
        do_sample: bool = False,
    ) -> torch.Tensor:
        """
        Generate translation (inference).
        
        Args:
            src_input_ids: Source token IDs (batch, src_len)
            src_attention_mask: Source attention mask (batch, src_len)
            max_length: Maximum length of generated sequence
            sos_token_id: Start-of-sequence token ID
            eos_token_id: End-of-sequence token ID
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus (top-p) sampling
            num_beams: Number of beams for beam search (1 = greedy)
            do_sample: Whether to sample stochastically (False = greedy/argmax)
        
        Returns:
            generated_ids: Generated token IDs (batch, gen_len)
        """
        # Input validation
        InputValidator.validate_tensor(src_input_ids, "src_input_ids", expected_dim=2)
        InputValidator.validate_token_ids(src_input_ids, self.vocab_size, "src_input_ids")
        
        if src_attention_mask is not None:
            InputValidator.validate_tensor(src_attention_mask, "src_attention_mask", expected_dim=2)
        
        InputValidator.validate_positive_int(max_length, "max_length", min_value=1, max_value=2048)
        InputValidator.validate_positive_int(sos_token_id, "sos_token_id", min_value=0)
        InputValidator.validate_positive_int(eos_token_id, "eos_token_id", min_value=0)
        InputValidator.validate_positive_float(temperature, "temperature", min_value=1e-8)
        InputValidator.validate_positive_int(top_k, "top_k", min_value=0)
        InputValidator.validate_probability(top_p, "top_p")
        InputValidator.validate_positive_int(num_beams, "num_beams", min_value=1)
        if not isinstance(do_sample, bool):
            raise TypeError(f"do_sample must be a bool, got {type(do_sample).__name__}")

        # num_beams is part of this API: route multi-beam decoding explicitly.
        if num_beams > 1:
            return self.generate_beam(
                src_input_ids=src_input_ids,
                src_attention_mask=src_attention_mask,
                max_length=max_length,
                num_beams=num_beams,
                sos_token_id=sos_token_id,
                eos_token_id=eos_token_id,
            )

        logger.debug(
            "Generating translation: src_len=%d, max_length=%d",
            src_input_ids.shape[1],
            max_length,
        )

        # FIXED: Validate and clamp temperature before generation loop
        temperature = max(0.01, float(temperature))

        # Fast path: decoder KV caching turns O(n^2) autoregressive decoding into O(n).
        # The cached implementation handles argmax (greedy), temperature, top-k, and
        # top-p sampling; behaviour was aligned with the reference implementation by
        # threading ``do_sample`` through ``generate_with_kv_cache``.
        return generate_with_kv_cache(
            model=self,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=self.pad_token_id,
            temperature=temperature,
            top_k=top_k if top_k > 0 else None,
            top_p=top_p,
            do_sample=do_sample,
        )
    
    def compute_loss(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        logits_chunk_size: Optional[int] = 1024,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for training.

        Args:
            src_input_ids: Source token IDs (batch, src_len)
            tgt_input_ids: Target token IDs (batch, tgt_len)
            src_attention_mask: Source attention mask
            tgt_attention_mask: Target attention mask
            label_smoothing: Label smoothing factor
            logits_chunk_size: Maximum number of flattened target tokens to
                project through the vocabulary head at once. The default keeps
                peak training logits bounded for large vocabularies. Set to 0
                or None to materialize all logits in one tensor.

        Returns:
            loss: Cross-entropy loss
        """
        if tgt_input_ids.shape[1] < 2:
            raise ValueError("tgt_input_ids must contain at least SOS and one label token")

        encoder_output = self.encoder(
            input_ids=src_input_ids,
            attention_mask=src_attention_mask,
        )
        hidden_states, _ = self.decoder.forward_hidden(
            input_ids=tgt_input_ids[:, :-1],
            encoder_output=encoder_output,
            self_attention_mask=(
                tgt_attention_mask[:, :-1] if tgt_attention_mask is not None else None
            ),
            cross_attention_mask=src_attention_mask,
            use_cache=False,
        )

        hidden_flat = hidden_states.reshape(-1, self.d_model)
        labels = tgt_input_ids[:, 1:].reshape(-1)
        total_tokens = hidden_flat.shape[0]
        chunk_size = total_tokens if not logits_chunk_size else int(logits_chunk_size)
        if chunk_size <= 0:
            chunk_size = total_tokens

        if chunk_size >= total_tokens:
            return F.cross_entropy(
                self.decoder.lm_head(hidden_flat),
                labels,
                ignore_index=self.pad_token_id,
                label_smoothing=label_smoothing,
            )

        loss_sum = hidden_flat.new_zeros(())
        for start in range(0, total_tokens, chunk_size):
            end = min(start + chunk_size, total_tokens)
            chunk_logits = self.decoder.lm_head(hidden_flat[start:end])
            loss_sum = loss_sum + F.cross_entropy(
                chunk_logits,
                labels[start:end],
                ignore_index=self.pad_token_id,
                label_smoothing=label_smoothing,
                reduction="sum",
            )

        valid_tokens = labels.ne(self.pad_token_id).sum().clamp_min(1)
        return loss_sum / valid_tokens
    
    def count_parameters(self) -> Dict[str, int]:
        """
        Count model parameters.
        
        Returns:
            Dictionary with parameter counts
        """
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        
        # Account for shared embeddings
        shared_embedding_params = sum(p.numel() for p in self.encoder.embedding.parameters())
        
        return {
            'encoder': encoder_params,
            'decoder': decoder_params,
            'shared_embeddings': shared_embedding_params,
            'total': encoder_params + decoder_params - shared_embedding_params,
        }

    def get_config(self) -> Dict[str, Any]:
        """Return the exact constructor config required to rebuild this model."""
        return dict(self._config)
    
    def generate_fast(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Fast generation with KV caching.
        """
        return self.generate_with_cache(
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )
    
    def generate_with_cache(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate with KV caching for efficiency.
        """
        return generate_with_kv_cache(
            model=self,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=getattr(self, "pad_token_id", 0),
            temperature=temperature,
            top_k=top_k,
            top_p=1.0 if top_p is None else float(top_p),
        )

    def generate_beam(
        self,
        src_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 128,
        num_beams: int = 4,
        length_penalty: float = 1.0,
        early_stopping: bool = True,
        sos_token_id: int = 1,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Generate with beam search.

        Quality is model- and dataset-dependent; tune ``length_penalty`` on a
        validation set rather than assuming beam search beats greedy decoding.

        Args:
            src_input_ids: Source token IDs (batch, src_len)
            src_attention_mask: Source attention mask
            max_length: Maximum generation length
            num_beams: Number of beams (higher = better quality, slower)
            length_penalty: Length penalty (>1.0 encourages longer sequences)
            early_stopping: Stop when num_beams hypotheses complete
            sos_token_id: Start-of-sequence token ID
            eos_token_id: End-of-sequence token ID

        Returns:
            Best sequences (batch, max_length)
        """
        return generate_with_beam_search(
            model=self,
            src_input_ids=src_input_ids,
            src_attention_mask=src_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=self.pad_token_id,
        )


def extract_model_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    """Extract model weights from trainer, generic, or raw state-dict checkpoints."""
    candidate: object = checkpoint
    if isinstance(checkpoint, Mapping):
        if "model_state_dict" in checkpoint:
            candidate = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            candidate = checkpoint["state_dict"]

    if not isinstance(candidate, Mapping) or not candidate:
        raise ValueError(
            "Checkpoint must be a non-empty state dict or contain "
            "'model_state_dict'/'state_dict'."
        )

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in candidate.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise ValueError("Checkpoint state dict must map string keys to tensors.")
        state_dict[key] = value

    # torch.compile OptimizedModule checkpoints historically prefixed every
    # parameter with "_orig_mod.". Normalize those files at the loader boundary.
    if state_dict and all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key.removeprefix("_orig_mod."): value
            for key, value in state_dict.items()
        }

    return state_dict


def load_model_from_checkpoint(
    checkpoint: object,
    *,
    fallback_vocab_size: Optional[int] = None,
    fallback_model_size: str = "small",
) -> MobileTranslationModel:
    """Rebuild and strictly load a model from a checkpoint.

    New trainer checkpoints include an exact ``config`` mapping. Legacy
    checkpoints can still be loaded by supplying the preset and vocabulary
    used to create them.
    """
    state_dict = extract_model_state_dict(checkpoint)

    config: Optional[Mapping[str, Any]] = None
    if isinstance(checkpoint, Mapping):
        if checkpoint.get("quantization") is not None:
            # Lazy import avoids a module cycle during package initialization.
            from .quantization_utils import load_quantized_model_from_checkpoint

            return load_quantized_model_from_checkpoint(checkpoint)
        raw_config = checkpoint.get("config")
        if raw_config is not None:
            if not isinstance(raw_config, Mapping):
                raise ValueError("Checkpoint 'config' must be a mapping.")
            config = cast(Mapping[str, Any], raw_config)

    if config is not None:
        model = MobileTranslationModel(**dict(config))
    else:
        if fallback_vocab_size is None:
            embedding = state_dict.get("encoder.embedding.weight")
            if embedding is None or embedding.ndim != 2:
                raise ValueError(
                    "Legacy checkpoint has no model config; provide fallback_vocab_size."
                )
            fallback_vocab_size = int(embedding.shape[0])
        model = create_model(
            vocab_size=int(fallback_vocab_size),
            model_size=fallback_model_size,
        )

    model.load_state_dict(state_dict, strict=True)
    return model


# FIXED: Add missing create_model function
def create_model(
    vocab_size: int,
    model_size: str = 'small',
    **kwargs: SupportsInt | SupportsFloat | bool,
) -> MobileTranslationModel:
    """
    Factory function to create models of different sizes.
    
    Args:
        vocab_size: Size of vocabulary
        model_size: One of 'tiny', 'small', 'medium', 'large'
        **kwargs: Additional arguments to override defaults
    
    Returns:
        MobileTranslationModel instance
    """
    class PresetConfig(TypedDict):
        d_model: int
        n_encoder_layers: int
        n_decoder_layers: int
        n_heads: int
        n_kv_heads: int
        d_ff: int
        max_seq_len: int
        dropout: float

    MODEL_CONFIGS: Dict[str, PresetConfig] = {
        'tiny': PresetConfig(
            d_model=256,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_heads=4,
            n_kv_heads=2,
            d_ff=1024,
            max_seq_len=512,
            dropout=0.1,
        ),
        'small': PresetConfig(
            d_model=512,
            n_encoder_layers=6,
            n_decoder_layers=6,
            n_heads=8,
            n_kv_heads=2,
            d_ff=2048,
            max_seq_len=512,
            dropout=0.1,
        ),
        'medium': PresetConfig(
            d_model=768,
            n_encoder_layers=8,
            n_decoder_layers=8,
            n_heads=12,
            n_kv_heads=4,
            d_ff=3072,
            max_seq_len=512,
            dropout=0.1,
        ),
        'large': PresetConfig(
            d_model=1024,
            n_encoder_layers=12,
            n_decoder_layers=12,
            n_heads=16,
            n_kv_heads=4,
            d_ff=4096,
            max_seq_len=512,
            dropout=0.1,
        ),
    }
    
    if model_size not in MODEL_CONFIGS:
        raise ValueError(
            f"Unknown model size '{model_size}'. "
            f"Choose from: {list(MODEL_CONFIGS.keys())}"
        )
    
    base = MODEL_CONFIGS[model_size]

    allowed_override_keys: Set[str] = {
        'd_model',
        'n_encoder_layers',
        'n_decoder_layers',
        'n_heads',
        'n_kv_heads',
        'd_ff',
        'max_seq_len',
        'dropout',
        'tie_embeddings',
        'pad_token_id',
        'efficient_ffn',
    }
    unexpected = [k for k in kwargs.keys() if k not in allowed_override_keys]
    if unexpected:
        raise ValueError(f"Unexpected overrides: {unexpected}")

    d_model = int(cast(SupportsInt, kwargs.pop('d_model', base['d_model'])))
    n_encoder_layers = int(cast(SupportsInt, kwargs.pop('n_encoder_layers', base['n_encoder_layers'])))
    n_decoder_layers = int(cast(SupportsInt, kwargs.pop('n_decoder_layers', base['n_decoder_layers'])))
    n_heads = int(cast(SupportsInt, kwargs.pop('n_heads', base['n_heads'])))
    n_kv_heads = int(cast(SupportsInt, kwargs.pop('n_kv_heads', base['n_kv_heads'])))
    explicit_d_ff = 'd_ff' in kwargs
    d_ff = int(cast(SupportsInt, kwargs.pop('d_ff', base['d_ff'])))
    efficient_ffn = bool(kwargs.pop('efficient_ffn', False))
    if efficient_ffn and not explicit_d_ff:
        # SwiGLU has three projections, so ~8/3*d_model matches the parameter
        # and FLOP budget of a conventional 4*d_model two-projection FFN.
        multiple = 64
        d_ff = ((8 * d_model + 3 * multiple - 1) // (3 * multiple)) * multiple
    max_seq_len = int(cast(SupportsInt, kwargs.pop('max_seq_len', base['max_seq_len'])))
    dropout = float(cast(SupportsFloat, kwargs.pop('dropout', base['dropout'])))
    tie_embeddings = bool(kwargs.pop('tie_embeddings', True))
    pad_token_id = int(cast(SupportsInt, kwargs.pop('pad_token_id', 0)))

    return MobileTranslationModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        tie_embeddings=tie_embeddings,
        pad_token_id=pad_token_id,
    )


# Test the model
if __name__ == "__main__":
    pass
