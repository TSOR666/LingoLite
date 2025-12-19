"""Utilities to export models for mobile deployment.

Provides helpers to export the translation model to ONNX and convert the
resulting graph to TFLite/CoreML formats when the required dependencies
are available. Conversions rely on optional packages and provide
actionable error messages when they are missing.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Optional

import torch


def export_to_onnx(
    model: torch.nn.Module,
    src_input_ids: torch.Tensor,
    tgt_input_ids: torch.Tensor,
    src_attention_mask: Optional[torch.Tensor],
    tgt_attention_mask: Optional[torch.Tensor],
    output_path: Path,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> Path:
    """Export the full model forward pass to ONNX."""

    model.eval()

    inputs = (
        src_input_ids,
        tgt_input_ids,
        src_attention_mask,
        tgt_attention_mask,
    )

    input_names = [
        "src_input_ids",
        "tgt_input_ids",
        "src_attention_mask",
        "tgt_attention_mask",
    ]
    output_names = ["logits"]

    dynamic_axes_spec = None
    if dynamic_axes:
        dynamic_axes_spec = {
            "src_input_ids": {0: "batch", 1: "src_len"},
            "tgt_input_ids": {0: "batch", 1: "tgt_len"},
            "src_attention_mask": {0: "batch", 1: "src_len"},
            "tgt_attention_mask": {0: "batch", 1: "tgt_len"},
            "logits": {0: "batch", 1: "tgt_len"},
        }

    torch.onnx.export(
        model,
        inputs,
        f=str(output_path),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes_spec,
    )

    return output_path


def convert_onnx_to_tflite(onnx_path: Path, output_path: Path) -> Path:
    """Convert an ONNX model to TFLite via onnx-tf and TensorFlow."""

    try:
        import onnx  # type: ignore
        from onnx_tf.backend import prepare  # type: ignore
        import tensorflow as tf  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Converting to TFLite requires onnx, onnx-tf, and tensorflow"
        ) from exc

    model = onnx.load(str(onnx_path))
    tf_rep = prepare(model)

    with tempfile.TemporaryDirectory() as tmpdir:
        tf_rep.export_graph(tmpdir)
        converter = tf.lite.TFLiteConverter.from_saved_model(tmpdir)
        tflite_model = converter.convert()
        output_path.write_bytes(tflite_model)

    return output_path


def convert_onnx_to_coreml(onnx_path: Path, output_path: Path) -> Path:
    """Convert an ONNX model to CoreML format."""

    try:
        import coremltools as ct  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Converting to CoreML requires coremltools") from exc

    mlmodel = ct.converters.onnx.convert(model=str(onnx_path))
    mlmodel.save(str(output_path))
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export model to mobile formats")
    parser.add_argument("checkpoint", type=Path, help="Path to a PyTorch checkpoint (.pt)")
    parser.add_argument("onnx", type=Path, help="Output ONNX file")
    parser.add_argument(
        "--tflite",
        type=Path,
        default=None,
        help="Optional TFLite output path (requires tensorflow + onnx-tf)",
    )
    parser.add_argument(
        "--coreml",
        type=Path,
        default=None,
        help="Optional CoreML output path (requires coremltools)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--no-dynamic-axes",
        action="store_true",
        help="Disable dynamic axes for fixed-shape exports",
    )
    parser.add_argument(
        "--src-len",
        type=int,
        default=32,
        help="Example source length used for tracing",
    )
    parser.add_argument(
        "--tgt-len",
        type=int,
        default=32,
        help="Example target length used for tracing",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32000,
        help="Model vocabulary size (used to create dummy inputs)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        help="Model size identifier for create_model (tiny/small/medium/large)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from lingolite.mobile_translation_model import create_model

    model = create_model(vocab_size=args.vocab_size, model_size=args.model_size)
    # SECURITY: Use weights_only=True to prevent arbitrary code execution
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    model.load_state_dict(checkpoint, strict=False)

    batch_size = 1
    src_input_ids = torch.ones(batch_size, args.src_len, dtype=torch.long)
    tgt_input_ids = torch.ones(batch_size, args.tgt_len, dtype=torch.long)
    src_attention_mask = torch.ones_like(src_input_ids)
    tgt_attention_mask = torch.ones_like(tgt_input_ids)

    export_to_onnx(
        model=model,
        src_input_ids=src_input_ids,
        tgt_input_ids=tgt_input_ids,
        src_attention_mask=src_attention_mask,
        tgt_attention_mask=tgt_attention_mask,
        output_path=args.onnx,
        opset=args.opset,
        dynamic_axes=not args.no_dynamic_axes,
    )

    print(f"✓ Exported ONNX model to {args.onnx}")

    if args.tflite is not None:
        convert_onnx_to_tflite(args.onnx, args.tflite)
        print(f"✓ Converted to TFLite: {args.tflite}")

    if args.coreml is not None:
        convert_onnx_to_coreml(args.onnx, args.coreml)
        print(f"✓ Converted to CoreML: {args.coreml}")


if __name__ == "__main__":
    main()
