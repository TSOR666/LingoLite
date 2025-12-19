"""
ONNX Export Utilities for Mobile Deployment
Export PyTorch models to ONNX format for TFLite/CoreML conversion
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict
import json

try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    print("WARNING: ONNX not installed. Install with: pip install onnx onnxruntime")
    ONNX_AVAILABLE = False

from lingolite.mobile_translation_model import MobileTranslationModel
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
        return self.encoder(input_ids, attention_mask)


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
        return logits


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
        from onnxruntime.transformers import optimizer
        from onnxruntime.transformers.fusion_options import FusionOptions

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

    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MobileTranslationModel(**config)
        d_model = config.get('d_model', 512)
    else:
        # Default config
        model = MobileTranslationModel(
            vocab_size=24000,
            d_model=512,
            n_encoder_layers=6,
            n_decoder_layers=6,
        )
        d_model = 512

    model.load_state_dict(checkpoint['model_state_dict'])
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


def main():
    parser = argparse.ArgumentParser(description="Export translation model to ONNX")
    parser.add_argument('--model', type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument('--output-dir', type=Path, required=True, help="Directory to save ONNX models")
    parser.add_argument('--max-seq-len', type=int, default=512, help="Maximum sequence length")
    parser.add_argument('--opset-version', type=int, default=14, help="ONNX opset version")
    parser.add_argument('--no-optimize', action='store_true', help="Skip optimization")
    parser.add_argument('--test', action='store_true', help="Test ONNX inference after export")

    args = parser.parse_args()

    exported_paths = export_full_model(
        model_path=args.model,
        output_dir=args.output_dir,
        max_seq_len=args.max_seq_len,
        opset_version=args.opset_version,
        optimize=not args.no_optimize,
    )

    if args.test:
        test_onnx_inference(
            encoder_path=exported_paths['encoder'],
            decoder_path=exported_paths['decoder'],
        )


if __name__ == '__main__':
    main()
