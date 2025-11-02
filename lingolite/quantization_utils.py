"""
Quantization Utilities for Mobile Deployment
INT8 quantization and Quantization-Aware Training (QAT) support
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import QuantStub, DeQuantStub
import argparse
from pathlib import Path
from typing import Optional, Dict
import json
import copy

from .mobile_translation_model import MobileTranslationModel
from .utils import logger


class QuantizableModel(nn.Module):
    """
    Wrapper to make MobileTranslationModel quantizable.
    Adds QuantStub and DeQuantStub for quantization.
    """

    def __init__(self, model: MobileTranslationModel):
        super().__init__()
        self.model = model

        # Quantization stubs
        self.quant_src = QuantStub()
        self.quant_tgt = QuantStub()
        self.dequant = DeQuantStub()

    def forward(
        self,
        src_input_ids: torch.Tensor,
        tgt_input_ids: torch.Tensor,
        src_attention_mask: Optional[torch.Tensor] = None,
        tgt_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with quantization stubs."""
        # Note: Quantization is typically applied to activations, not input IDs
        # For embeddings, we quantize after embedding lookup

        logits, _, _ = self.model(
            src_input_ids=src_input_ids,
            tgt_input_ids=tgt_input_ids,
            src_attention_mask=src_attention_mask,
            tgt_attention_mask=tgt_attention_mask,
            use_cache=False,
        )

        return logits

    def generate(self, *args, **kwargs):
        """Pass through to model's generate method."""
        return self.model.generate(*args, **kwargs)


def prepare_model_for_qat(
    model: MobileTranslationModel,
    qconfig: str = 'fbgemm',
) -> QuantizableModel:
    """
    Prepare model for Quantization-Aware Training (QAT).

    Args:
        model: Translation model
        qconfig: Quantization configuration ('fbgemm' for x86, 'qnnpack' for ARM)

    Returns:
        Model prepared for QAT
    """
    logger.info(f"Preparing model for QAT with qconfig: {qconfig}")

    # Wrap model
    quantizable_model = QuantizableModel(model)

    # Set to training mode for QAT
    quantizable_model.train()

    # Configure quantization
    if qconfig == 'fbgemm':
        qconfig_spec = torch.quantization.get_default_qat_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        qconfig_spec = torch.quantization.get_default_qat_qconfig('qnnpack')
    else:
        raise ValueError(f"Unknown qconfig: {qconfig}")

    quantizable_model.qconfig = qconfig_spec

    # Prepare for QAT
    torch.quantization.prepare_qat(quantizable_model, inplace=True)

    logger.info("Model prepared for QAT")
    return quantizable_model


def convert_qat_model(
    qat_model: QuantizableModel,
) -> QuantizableModel:
    """
    Convert QAT model to quantized model for inference.

    Args:
        qat_model: Model prepared with QAT

    Returns:
        Quantized model for inference
    """
    logger.info("Converting QAT model to quantized model...")

    # Set to eval mode
    qat_model.eval()

    # Convert to quantized model
    quantized_model = torch.quantization.convert(qat_model, inplace=False)

    logger.info("QAT model converted to quantized model")
    return quantized_model


def apply_dynamic_quantization(
    model: MobileTranslationModel,
    dtype: torch.dtype = torch.qint8,
) -> MobileTranslationModel:
    """
    Apply dynamic quantization to model.
    Quantizes weights ahead of time, activations dynamically during inference.

    Best for models with:
    - High ratio of memory bandwidth to compute
    - LSTM/RNN/Transformer models (good for translation!)

    Args:
        model: Translation model
        dtype: Quantization dtype (torch.qint8 or torch.float16)

    Returns:
        Dynamically quantized model
    """
    logger.info(f"Applying dynamic quantization with dtype: {dtype}")

    # Copy model to avoid modifying original
    quantized_model = copy.deepcopy(model)
    quantized_model.eval()

    # Apply dynamic quantization to Linear layers
    quantized_model = torch.quantization.quantize_dynamic(
        quantized_model,
        qconfig_spec={nn.Linear},  # Quantize all Linear layers
        dtype=dtype,
    )

    logger.info("Dynamic quantization applied")
    return quantized_model


def apply_static_quantization(
    model: MobileTranslationModel,
    calibration_data_loader,
    qconfig: str = 'fbgemm',
) -> QuantizableModel:
    """
    Apply static quantization to model.
    Requires calibration data to determine quantization parameters.

    Args:
        model: Translation model
        calibration_data_loader: DataLoader with calibration data
        qconfig: Quantization configuration

    Returns:
        Statically quantized model
    """
    logger.info(f"Applying static quantization with qconfig: {qconfig}")

    # Wrap model
    quantizable_model = QuantizableModel(model)
    quantizable_model.eval()

    # Set quantization config
    if qconfig == 'fbgemm':
        backend = 'fbgemm'
        qconfig_spec = torch.quantization.get_default_qconfig('fbgemm')
    elif qconfig == 'qnnpack':
        backend = 'qnnpack'
        qconfig_spec = torch.quantization.get_default_qconfig('qnnpack')
    else:
        raise ValueError(f"Unknown qconfig: {qconfig}")

    torch.backends.quantized.engine = backend
    quantizable_model.qconfig = qconfig_spec

    # Prepare for static quantization
    torch.quantization.prepare(quantizable_model, inplace=True)

    # Calibrate with representative data
    logger.info("Calibrating quantization parameters...")
    with torch.no_grad():
        for i, batch in enumerate(calibration_data_loader):
            if i >= 100:  # Limit calibration samples
                break

            src_ids = batch['src_input_ids']
            tgt_ids = batch['tgt_input_ids']
            src_mask = batch.get('src_attention_mask')
            tgt_mask = batch.get('tgt_attention_mask')

            quantizable_model(src_ids, tgt_ids, src_mask, tgt_mask)

            if (i + 1) % 10 == 0:
                logger.info(f"Calibrated {i + 1} batches")

    # Convert to quantized model
    quantized_model = torch.quantization.convert(quantizable_model, inplace=False)

    logger.info("Static quantization applied")
    return quantized_model


def measure_model_size(model: nn.Module) -> Dict[str, float]:
    """
    Measure model size in MB.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with size metrics
    """
    # Save model to temporary buffer
    import io
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.tell() / (1024 ** 2)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())

    return {
        'size_mb': size_mb,
        'num_params': num_params,
        'params_millions': num_params / 1e6,
    }


def benchmark_model(
    model: nn.Module,
    input_shape: tuple = (1, 32),
    num_iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark model inference speed.

    Args:
        model: PyTorch model
        input_shape: Input tensor shape (batch, seq_len)
        num_iterations: Number of iterations for benchmarking
        warmup: Number of warmup iterations

    Returns:
        Dictionary with timing metrics
    """
    import time

    model.eval()
    device = next(model.parameters()).device

    # Create dummy inputs
    dummy_src = torch.randint(0, 1000, input_shape, device=device)
    dummy_tgt = torch.randint(0, 1000, input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            if hasattr(model, 'model'):  # QuantizableModel
                _ = model(dummy_src, dummy_tgt)
            else:
                _ = model(dummy_src, dummy_tgt, use_cache=False)

    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            if hasattr(model, 'model'):
                _ = model(dummy_src, dummy_tgt)
            else:
                _ = model(dummy_src, dummy_tgt, use_cache=False)
            end = time.perf_counter()
            times.append(end - start)

    import numpy as np
    return {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'median_ms': np.median(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
    }


def compare_quantization_methods(
    model_path: Path,
    output_dir: Path,
    calibration_data_loader = None,
) -> Dict[str, Dict]:
    """
    Compare different quantization methods.

    Args:
        model_path: Path to model checkpoint
        output_dir: Directory to save results
        calibration_data_loader: Optional calibration data for static quantization

    Returns:
        Dictionary with comparison results
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    if 'config' in checkpoint:
        config = checkpoint['config']
        model = MobileTranslationModel(**config)
    else:
        model = MobileTranslationModel(
            vocab_size=24000,
            d_model=512,
            n_encoder_layers=6,
            n_decoder_layers=6,
        )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    results = {}

    # Baseline (FP32)
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating FP32 baseline...")
    logger.info("=" * 80)
    baseline_size = measure_model_size(model)
    baseline_speed = benchmark_model(model)
    results['fp32'] = {
        'size': baseline_size,
        'speed': baseline_speed,
    }
    logger.info(f"FP32 - Size: {baseline_size['size_mb']:.2f} MB, "
                f"Speed: {baseline_speed['mean_ms']:.2f} ms")

    # Dynamic INT8
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Dynamic INT8 quantization...")
    logger.info("=" * 80)
    dynamic_int8_model = apply_dynamic_quantization(model, dtype=torch.qint8)
    dynamic_int8_size = measure_model_size(dynamic_int8_model)
    dynamic_int8_speed = benchmark_model(dynamic_int8_model)
    results['dynamic_int8'] = {
        'size': dynamic_int8_size,
        'speed': dynamic_int8_speed,
        'compression_ratio': baseline_size['size_mb'] / dynamic_int8_size['size_mb'],
        'speedup': baseline_speed['mean_ms'] / dynamic_int8_speed['mean_ms'],
    }
    logger.info(f"Dynamic INT8 - Size: {dynamic_int8_size['size_mb']:.2f} MB "
                f"({results['dynamic_int8']['compression_ratio']:.2f}x compression), "
                f"Speed: {dynamic_int8_speed['mean_ms']:.2f} ms "
                f"({results['dynamic_int8']['speedup']:.2f}x speedup)")

    # Save quantized model
    torch.save(
        dynamic_int8_model.state_dict(),
        output_dir / 'model_dynamic_int8.pt'
    )

    # Dynamic FP16
    logger.info("\n" + "=" * 80)
    logger.info("Evaluating Dynamic FP16 quantization...")
    logger.info("=" * 80)
    try:
        dynamic_fp16_model = apply_dynamic_quantization(model, dtype=torch.float16)
        dynamic_fp16_size = measure_model_size(dynamic_fp16_model)
        dynamic_fp16_speed = benchmark_model(dynamic_fp16_model)
        results['dynamic_fp16'] = {
            'size': dynamic_fp16_size,
            'speed': dynamic_fp16_speed,
            'compression_ratio': baseline_size['size_mb'] / dynamic_fp16_size['size_mb'],
            'speedup': baseline_speed['mean_ms'] / dynamic_fp16_speed['mean_ms'],
        }
        logger.info(f"Dynamic FP16 - Size: {dynamic_fp16_size['size_mb']:.2f} MB "
                    f"({results['dynamic_fp16']['compression_ratio']:.2f}x compression), "
                    f"Speed: {dynamic_fp16_speed['mean_ms']:.2f} ms "
                    f"({results['dynamic_fp16']['speedup']:.2f}x speedup)")

        torch.save(
            dynamic_fp16_model.state_dict(),
            output_dir / 'model_dynamic_fp16.pt'
        )
    except Exception as e:
        logger.warning(f"FP16 quantization failed: {e}")

    # Static quantization (if calibration data provided)
    if calibration_data_loader is not None:
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating Static INT8 quantization...")
        logger.info("=" * 80)
        try:
            static_int8_model = apply_static_quantization(
                model, calibration_data_loader, qconfig='fbgemm'
            )
            static_int8_size = measure_model_size(static_int8_model)
            static_int8_speed = benchmark_model(static_int8_model)
            results['static_int8'] = {
                'size': static_int8_size,
                'speed': static_int8_speed,
                'compression_ratio': baseline_size['size_mb'] / static_int8_size['size_mb'],
                'speedup': baseline_speed['mean_ms'] / static_int8_speed['mean_ms'],
            }
            logger.info(f"Static INT8 - Size: {static_int8_size['size_mb']:.2f} MB "
                        f"({results['static_int8']['compression_ratio']:.2f}x compression), "
                        f"Speed: {static_int8_speed['mean_ms']:.2f} ms "
                        f"({results['static_int8']['speedup']:.2f}x speedup)")

            torch.save(
                static_int8_model.state_dict(),
                output_dir / 'model_static_int8.pt'
            )
        except Exception as e:
            logger.warning(f"Static quantization failed: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Method':<20} {'Size (MB)':<12} {'Compression':<12} {'Speed (ms)':<12} {'Speedup':<12}")
    print("-" * 80)

    for method, data in results.items():
        size = data['size']['size_mb']
        speed = data['speed']['mean_ms']
        compression = data.get('compression_ratio', 1.0)
        speedup = data.get('speedup', 1.0)
        print(f"{method:<20} {size:>10.2f}   {compression:>10.2f}x   {speed:>10.2f}   {speedup:>10.2f}x")

    print("=" * 80 + "\n")

    # Save results
    results_file = output_dir / 'quantization_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Quantization utilities for translation model")
    parser.add_argument('--model', type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument('--output-dir', type=Path, required=True, help="Directory to save quantized models")
    parser.add_argument('--method', type=str, choices=['dynamic_int8', 'dynamic_fp16', 'compare'],
                        default='compare', help="Quantization method")
    parser.add_argument('--qconfig', type=str, choices=['fbgemm', 'qnnpack'],
                        default='fbgemm', help="Quantization backend")

    args = parser.parse_args()

    if args.method == 'compare':
        compare_quantization_methods(
            model_path=args.model,
            output_dir=args.output_dir,
        )
    elif args.method == 'dynamic_int8':
        # Load and quantize
        checkpoint = torch.load(args.model, map_location='cpu')
        if 'config' in checkpoint:
            model = MobileTranslationModel(**checkpoint['config'])
        else:
            model = MobileTranslationModel(vocab_size=24000)
        model.load_state_dict(checkpoint['model_state_dict'])

        quantized = apply_dynamic_quantization(model, dtype=torch.qint8)
        output_path = args.output_dir / 'model_quantized_int8.pt'
        torch.save(quantized.state_dict(), output_path)
        logger.info(f"Quantized model saved to {output_path}")

    elif args.method == 'dynamic_fp16':
        # Load and quantize
        checkpoint = torch.load(args.model, map_location='cpu')
        if 'config' in checkpoint:
            model = MobileTranslationModel(**checkpoint['config'])
        else:
            model = MobileTranslationModel(vocab_size=24000)
        model.load_state_dict(checkpoint['model_state_dict'])

        quantized = apply_dynamic_quantization(model, dtype=torch.float16)
        output_path = args.output_dir / 'model_quantized_fp16.pt'
        torch.save(quantized.state_dict(), output_path)
        logger.info(f"Quantized model saved to {output_path}")


if __name__ == '__main__':
    main()
