"""
ONNX Export Utilities for Mobile Deployment
Export PyTorch models to ONNX format for TFLite/CoreML conversion
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, cast
import json

try:
    import onnx  # type: ignore[import-not-found]
    import onnxruntime as ort  # type: ignore[import-not-found]
    ONNX_AVAILABLE = True
except ImportError:
    print("WARNING: ONNX not installed. Install with: pip install onnx onnxruntime")
    ONNX_AVAILABLE = False

from lingolite.mobile_translation_model import MobileTranslationModel, load_model_from_checkpoint
from lingolite.translation_tokenizer import TranslationTokenizer
from lingolite.utils import logger


class EncoderWrapper(nn.Module):
    """Wrapper for encoder to export to ONNX."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)

        Returns:
            encoder_output: (batch, seq_len, d_model)
        """
        return cast(torch.Tensor, self.encoder(input_ids, attention_mask))


class DecoderWrapper(nn.Module):
    """Wrapper for decoder to export to ONNX."""

    def __init__(self, decoder: nn.Module):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, tgt_len)
            encoder_output: (batch, src_len, d_model)
            self_attention_mask: (batch, tgt_len)
            cross_attention_mask: (batch, src_len)

        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        logits, _ = self.decoder(
            input_ids,
            encoder_output,
            self_attention_mask,
            cross_attention_mask,
            use_cache=False,
        )
        return cast(torch.Tensor, logits)


class CachedDecoderWrapper(nn.Module):
    """Cache-aware decoder wrapper for efficient autoregressive ONNX inference.

    Unlike :class:`DecoderWrapper`, this version makes each layer's
    self-attention KV cache an explicit input/output. Runtimes (ONNX Runtime,
    TFLite, Core ML) can then run one decoder step at a time, feeding the
    previous step's KV tensors back in instead of re-running the full target
    prefix every step (O(n) total instead of O(n^2)).

    Tracing semantics:

    * The traced graph assumes ``past_len >= 1`` because the underlying
      ``LayerKVCache`` exposes a None-typed ``key``/``value`` when the buffer
      is empty, and that conditional doesn't survive ONNX tracing. Use this
      graph from decode step 2 onwards; for step 1, run the regular
      non-cached :class:`DecoderWrapper` to seed the caches with shape
      ``(batch, n_kv_heads, 1, head_dim)``.
    * Cross-attention K/V is recomputed from ``encoder_output`` on every
      call. This is wasteful but keeps the signature minimal; runtimes can
      apply their own constant-folding on the encoder output to mitigate.

    Inputs (positional, then variadic):

        ``input_ids`` ``(B, 1)``                          - one new target token per beam
        ``encoder_output`` ``(B, S, D)``                  - shared across decode steps
        ``cross_attention_mask`` ``(B, S)``               - encoder padding mask
        Then 2*n_layers KV tensors interleaved as
        ``past_key_0, past_value_0, past_key_1, past_value_1, ...``
        each shaped ``(B, H, past_len, Dh)``.

    Outputs:

        ``logits`` ``(B, 1, V)`` followed by 2*n_layers updated KV tensors
        each shaped ``(B, H, past_len + 1, Dh)``.
    """

    def __init__(self, decoder: nn.Module, n_layers: int) -> None:
        super().__init__()
        self.decoder = decoder
        self.n_layers = n_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_output: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        *past_kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        from lingolite.generation_utils import KVCache, LayerKVCache

        if len(past_kv) != 2 * self.n_layers:
            raise ValueError(
                f"expected {2 * self.n_layers} past KV tensors (n_layers * 2), got {len(past_kv)}"
            )

        layer_caches = []
        for i in range(self.n_layers):
            past_k = past_kv[2 * i]
            past_v = past_kv[2 * i + 1]
            layer_cache = LayerKVCache()
            # Lazy-mode KVCache (no ``reserve``): subsequent ``update`` calls
            # use ``torch.cat``, which traces cleanly into ONNX.
            layer_cache.self_attn_cache = KVCache(key=past_k, value=past_v)
            layer_caches.append(layer_cache)

        # New target slot is always a single valid token.
        batch_size = int(input_ids.shape[0])
        self_attention_mask = torch.ones(
            batch_size, 1, dtype=torch.bool, device=input_ids.device
        )

        logits, updated_caches = self.decoder(
            input_ids=input_ids,
            encoder_output=encoder_output,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=layer_caches,
            use_cache=True,
        )
        if updated_caches is None:
            raise RuntimeError("decoder returned no updated caches under use_cache=True")

        outputs: List[torch.Tensor] = [cast(torch.Tensor, logits)]
        for layer_cache in updated_caches:
            new_k = layer_cache.self_attn_cache.key
            new_v = layer_cache.self_attn_cache.value
            if new_k is None or new_v is None:
                raise RuntimeError("layer cache produced no updated key/value tensors")
            outputs.append(new_k)
            outputs.append(new_v)
        return tuple(outputs)


def export_encoder_to_onnx(
    model: MobileTranslationModel,
    output_path: Path,
    max_seq_len: int = 512,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """
    Export encoder to ONNX format.

    Args:
        model: Translation model
        output_path: Path to save ONNX model
        max_seq_len: Maximum sequence length for static shapes
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes (recommended)
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not installed. Install with: pip install onnx onnxruntime")

    logger.info("Exporting encoder to ONNX...")

    # Wrap encoder
    encoder_wrapper = EncoderWrapper(model.encoder)
    encoder_wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    dummy_input_ids = torch.randint(0, model.vocab_size, (batch_size, max_seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.float)

    # Define input/output names
    input_names = ['input_ids', 'attention_mask']
    output_names = ['encoder_output']

    # Define dynamic axes if enabled
    if dynamic_axes:
        dynamic_axes_dict = {
            'input_ids': {0: 'batch', 1: 'seq_len'},
            'attention_mask': {0: 'batch', 1: 'seq_len'},
            'encoder_output': {0: 'batch', 1: 'seq_len'},
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    torch.onnx.export(
        encoder_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
    )

    logger.info(f"Encoder exported to {output_path}")

    # Verify ONNX model
    verify_onnx_model(output_path)


def export_decoder_to_onnx(
    model: MobileTranslationModel,
    output_path: Path,
    max_seq_len: int = 128,
    d_model: int = 512,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """
    Export decoder to ONNX format.

    Args:
        model: Translation model
        output_path: Path to save ONNX model
        max_seq_len: Maximum sequence length for static shapes
        d_model: Model dimension
        opset_version: ONNX opset version
        dynamic_axes: Whether to use dynamic axes
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not installed. Install with: pip install onnx onnxruntime")

    logger.info("Exporting decoder to ONNX...")

    # Wrap decoder
    decoder_wrapper = DecoderWrapper(model.decoder)
    decoder_wrapper.eval()

    # Create dummy inputs
    batch_size = 1
    src_len = 64
    tgt_len = 32

    dummy_input_ids = torch.randint(0, model.vocab_size, (batch_size, tgt_len), dtype=torch.long)
    dummy_encoder_output = torch.randn(batch_size, src_len, d_model)
    dummy_self_attn_mask = torch.ones(batch_size, tgt_len, dtype=torch.float)
    dummy_cross_attn_mask = torch.ones(batch_size, src_len, dtype=torch.float)

    # Define input/output names
    input_names = ['input_ids', 'encoder_output', 'self_attention_mask', 'cross_attention_mask']
    output_names = ['logits']

    # Define dynamic axes if enabled
    if dynamic_axes:
        dynamic_axes_dict = {
            'input_ids': {0: 'batch', 1: 'tgt_len'},
            'encoder_output': {0: 'batch', 1: 'src_len'},
            'self_attention_mask': {0: 'batch', 1: 'tgt_len'},
            'cross_attention_mask': {0: 'batch', 1: 'src_len'},
            'logits': {0: 'batch', 1: 'tgt_len'},
        }
    else:
        dynamic_axes_dict = None

    # Export to ONNX
    torch.onnx.export(
        decoder_wrapper,
        (dummy_input_ids, dummy_encoder_output, dummy_self_attn_mask, dummy_cross_attn_mask),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_dict,
    )

    logger.info(f"Decoder exported to {output_path}")

    # Verify ONNX model
    verify_onnx_model(output_path)


def export_cached_decoder_to_onnx(
    model: MobileTranslationModel,
    output_path: Path,
    src_len: int = 32,
    past_len: int = 1,
    opset_version: int = 14,
    dynamic_axes: bool = True,
) -> None:
    """Export the cache-aware decoder wrapper to ONNX.

    Args:
        model: Translation model.
        output_path: Path to save ONNX model.
        src_len: Encoder source length to use for dummy inputs (becomes a
            dynamic axis when ``dynamic_axes=True``).
        past_len: Past self-attention sequence length to trace with. Must be
            >= 1 - see :class:`CachedDecoderWrapper` docstring.
        opset_version: ONNX opset version.
        dynamic_axes: When True, declare batch / source-len / past-len as
            dynamic so the same exported graph handles any input shape.
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not installed. Install with: pip install onnx onnxruntime")
    if past_len < 1:
        raise ValueError("past_len must be >= 1; trace with past_len=1 for a generic graph")

    logger.info("Exporting cache-aware decoder to ONNX...")

    decoder_layers = model.decoder._decoder_layers  # type: ignore[attr-defined]
    n_layers = len(decoder_layers)
    first_attn = decoder_layers[0].self_attn
    n_kv_heads = int(first_attn.n_kv_heads)
    head_dim = int(first_attn.head_dim)
    d_model = int(model.d_model)

    wrapper = CachedDecoderWrapper(model.decoder, n_layers=n_layers)
    wrapper.eval()

    # Dummy inputs - shapes must be self-consistent so the wrapper can run.
    batch = 1
    dummy_input_ids = torch.randint(0, model.vocab_size, (batch, 1), dtype=torch.long)
    dummy_encoder_output = torch.randn(batch, src_len, d_model)
    dummy_cross_mask = torch.ones(batch, src_len, dtype=torch.bool)
    dummy_past_kvs: List[torch.Tensor] = []
    for _ in range(n_layers):
        dummy_past_kvs.append(torch.randn(batch, n_kv_heads, past_len, head_dim))  # key
        dummy_past_kvs.append(torch.randn(batch, n_kv_heads, past_len, head_dim))  # value

    input_names = ["input_ids", "encoder_output", "cross_attention_mask"]
    for i in range(n_layers):
        input_names.append(f"past_key_{i}")
        input_names.append(f"past_value_{i}")

    output_names = ["logits"]
    for i in range(n_layers):
        output_names.append(f"present_key_{i}")
        output_names.append(f"present_value_{i}")

    if dynamic_axes:
        axes_dict: Dict[str, Dict[int, str]] = {
            "input_ids": {0: "batch"},
            "encoder_output": {0: "batch", 1: "src_len"},
            "cross_attention_mask": {0: "batch", 1: "src_len"},
            "logits": {0: "batch"},
        }
        for i in range(n_layers):
            axes_dict[f"past_key_{i}"] = {0: "batch", 2: "past_len"}
            axes_dict[f"past_value_{i}"] = {0: "batch", 2: "past_len"}
            axes_dict[f"present_key_{i}"] = {0: "batch", 2: "present_len"}
            axes_dict[f"present_value_{i}"] = {0: "batch", 2: "present_len"}
    else:
        axes_dict = None  # type: ignore[assignment]

    args = (dummy_input_ids, dummy_encoder_output, dummy_cross_mask, *dummy_past_kvs)

    torch.onnx.export(
        wrapper,
        args,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=axes_dict,
    )

    logger.info(f"Cache-aware decoder exported to {output_path}")
    verify_onnx_model(output_path)


def verify_onnx_model(onnx_path: Path) -> bool:
    """
    Verify ONNX model is valid.

    Args:
        onnx_path: Path to ONNX model

    Returns:
        True if valid, raises exception otherwise
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX not available, skipping verification")
        return False

    logger.info(f"Verifying ONNX model: {onnx_path}")

    # Load and check model
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    logger.info("ONNX model is valid")
    return True


def optimize_onnx_model(
    input_path: Path,
    output_path: Path,
) -> None:
    """
    Optimize ONNX model for inference.

    Args:
        input_path: Path to input ONNX model
        output_path: Path to save optimized model
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not installed")

    try:
        from onnxruntime.transformers import optimizer  # type: ignore[import-not-found]
        from onnxruntime.transformers.fusion_options import FusionOptions  # type: ignore[import-not-found]

        logger.info(f"Optimizing ONNX model: {input_path}")

        # Create optimization options
        opt_options = FusionOptions('bert')  # Use bert optimizations (works for transformers)
        opt_options.enable_gelu_approximation = False

        # Optimize model
        optimized_model = optimizer.optimize_model(
            str(input_path),
            model_type='bert',
            num_heads=8,  # Adjust based on your model
            hidden_size=512,  # Adjust based on your model
            optimization_options=opt_options,
        )

        optimized_model.save_model_to_file(str(output_path))
        logger.info(f"Optimized model saved to {output_path}")

    except ImportError:
        logger.warning("onnxruntime optimizer not available, skipping optimization")
        # Just copy the file
        import shutil
        shutil.copy(input_path, output_path)


def export_full_model(
    model_path: Path,
    output_dir: Path,
    max_seq_len: int = 512,
    opset_version: int = 14,
    optimize: bool = True,
    verify: bool = True,
    cached_decoder: bool = False,
) -> Dict[str, Path]:
    """
    Export full translation model (encoder + decoder) to ONNX.

    Args:
        model_path: Path to PyTorch model checkpoint
        output_dir: Directory to save ONNX models
        max_seq_len: Maximum sequence length
        opset_version: ONNX opset version
        optimize: Whether to optimize ONNX models
        verify: Whether to verify ONNX models

    Returns:
        Dictionary mapping component names to exported paths
    """
    if not ONNX_AVAILABLE:
        raise ImportError("ONNX not installed. Install with: pip install onnx onnxruntime")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    # SECURITY: Use weights_only=True to prevent arbitrary code execution
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)

    model = load_model_from_checkpoint(
        checkpoint,
        fallback_vocab_size=24000,
        fallback_model_size="small",
    )
    d_model = model.d_model
    model.eval()

    logger.info(f"Model loaded: {model.count_parameters()['total']:,} parameters")

    # Export encoder
    encoder_path = output_dir / "encoder.onnx"
    export_encoder_to_onnx(
        model=model,
        output_path=encoder_path,
        max_seq_len=max_seq_len,
        opset_version=opset_version,
    )

    # Export decoder
    decoder_path = output_dir / "decoder.onnx"
    export_decoder_to_onnx(
        model=model,
        output_path=decoder_path,
        max_seq_len=max_seq_len,
        d_model=d_model,
        opset_version=opset_version,
    )

    exported_paths = {
        'encoder': encoder_path,
        'decoder': decoder_path,
    }

    # Optionally export the cache-aware decoder for efficient
    # autoregressive runtime decoding (O(n) instead of O(n^2)).
    if cached_decoder:
        cached_decoder_path = output_dir / "decoder_cached.onnx"
        export_cached_decoder_to_onnx(
            model=model,
            output_path=cached_decoder_path,
            opset_version=opset_version,
        )
        exported_paths['decoder_cached'] = cached_decoder_path

    # Optimize if requested
    if optimize:
        logger.info("Optimizing ONNX models...")
        encoder_opt_path = output_dir / "encoder_optimized.onnx"
        decoder_opt_path = output_dir / "decoder_optimized.onnx"

        optimize_onnx_model(encoder_path, encoder_opt_path)
        optimize_onnx_model(decoder_path, decoder_opt_path)

        exported_paths['encoder_optimized'] = encoder_opt_path
        exported_paths['decoder_optimized'] = decoder_opt_path

    # Save export config
    export_config = {
        'max_seq_len': max_seq_len,
        'opset_version': opset_version,
        'd_model': d_model,
        'vocab_size': model.vocab_size,
    }

    config_path = output_dir / "export_config.json"
    with open(config_path, 'w') as f:
        json.dump(export_config, f, indent=2)

    logger.info(f"Export config saved to {config_path}")

    print("\n" + "=" * 80)
    print("ONNX EXPORT COMPLETE")
    print("=" * 80)
    print(f"Encoder:  {encoder_path}")
    print(f"Decoder:  {decoder_path}")
    if optimize:
        print(f"Encoder (optimized): {encoder_opt_path}")
        print(f"Decoder (optimized): {decoder_opt_path}")
    print(f"Config:   {config_path}")
    print("\nNext steps for mobile deployment:")
    print("  1. TensorFlow Lite: Use tf2onnx + TFLite converter")
    print("  2. Core ML: Use onnx-coreml converter")
    print("  3. ONNX Runtime Mobile: Use models directly")
    print("=" * 80 + "\n")

    return exported_paths


def test_onnx_inference(
    encoder_path: Path,
    decoder_path: Path,
    test_input_ids: Optional[torch.Tensor] = None,
) -> None:
    """
    Test ONNX models with dummy inference.

    Args:
        encoder_path: Path to encoder ONNX model
        decoder_path: Path to decoder ONNX model
        test_input_ids: Optional test input tensor
    """
    if not ONNX_AVAILABLE:
        logger.warning("ONNX Runtime not available, skipping inference test")
        return

    logger.info("Testing ONNX inference...")

    # Create ONNX Runtime sessions
    encoder_session = ort.InferenceSession(str(encoder_path))
    decoder_session = ort.InferenceSession(str(decoder_path))

    # Create test inputs if not provided
    if test_input_ids is None:
        batch_size = 1
        seq_len = 32
        test_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)

    test_attention_mask = torch.ones_like(test_input_ids, dtype=torch.float)

    # Run encoder
    encoder_outputs = encoder_session.run(
        None,
        {
            'input_ids': test_input_ids.numpy(),
            'attention_mask': test_attention_mask.numpy(),
        }
    )
    encoder_output = encoder_outputs[0]

    logger.info(f"Encoder output shape: {encoder_output.shape}")

    # Run decoder
    decoder_input_ids = torch.randint(0, 1000, (1, 10), dtype=torch.long)
    decoder_self_mask = torch.ones_like(decoder_input_ids, dtype=torch.float)
    decoder_cross_mask = test_attention_mask

    decoder_outputs = decoder_session.run(
        None,
        {
            'input_ids': decoder_input_ids.numpy(),
            'encoder_output': encoder_output,
            'self_attention_mask': decoder_self_mask.numpy(),
            'cross_attention_mask': decoder_cross_mask.numpy(),
        }
    )

    logits = decoder_outputs[0]
    logger.info(f"Decoder output shape: {logits.shape}")
    logger.info("ONNX inference test successful!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export translation model to ONNX")
    parser.add_argument('--model', type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument('--output-dir', type=Path, required=True, help="Directory to save ONNX models")
    parser.add_argument('--max-seq-len', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--opset-version', type=int, default=14, help="ONNX opset version")
    parser.add_argument('--no-optimize', action='store_true', help="Skip optimization")
    parser.add_argument('--test', action='store_true', help="Test ONNX inference after export")
    parser.add_argument(
        '--cached-decoder',
        action='store_true',
        help="Also export a cache-aware decoder (decoder_cached.onnx) for O(n) autoregressive decoding",
    )

    args = parser.parse_args()

    exported_paths = export_full_model(
        model_path=args.model,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        opset_version=args.opset_version,
        optimize=not args.no_optimize,
        cached_decoder=args.cached_decoder,
    )

    if args.test:
        test_onnx_inference(
            encoder_path=exported_paths['encoder'],
            decoder_path=exported_paths['decoder'],
        )


if __name__ == '__main__':
    main()
