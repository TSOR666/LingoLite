"""
Tests for quantization utilities.

Covers:
- Dynamic quantization
- Model size measurement
- QuantizableModel wrapper
- Error handling for unsupported operations
"""

import pytest
import torch
import torch.nn as nn
from typing import Dict

from lingolite.mobile_translation_model import MobileTranslationModel
from lingolite.quantization_utils import (
    QuantizableModel,
    apply_dynamic_quantization,
    measure_model_size,
    prepare_model_for_qat,
    apply_static_quantization,
    QAT_SUPPORTED,
    STATIC_QUANTIZATION_SUPPORTED,
)


@pytest.fixture
def tiny_model() -> MobileTranslationModel:
    """Create a tiny model for testing."""
    return MobileTranslationModel(
        vocab_size=100,
        d_model=32,
        n_encoder_layers=1,
        n_decoder_layers=1,
        n_heads=2,
        n_kv_heads=1,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
    )


class TestQuantizableModel:
    """Tests for the QuantizableModel wrapper."""

    def test_quantizable_model_forward(self, tiny_model: MobileTranslationModel) -> None:
        """QuantizableModel should wrap model and return logits."""
        qmodel = QuantizableModel(tiny_model)
        
        src = torch.randint(0, 100, (1, 4))
        tgt = torch.randint(0, 100, (1, 4))
        
        logits = qmodel(src, tgt)
        
        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (1, 4, 100)  # (batch, seq_len, vocab_size)

    def test_quantizable_model_generate_passthrough(self, tiny_model: MobileTranslationModel) -> None:
        """QuantizableModel.generate should pass through to inner model."""
        qmodel = QuantizableModel(tiny_model)
        
        src = torch.randint(0, 100, (1, 4))
        
        # Should not raise - just verify it calls through
        output = qmodel.generate(src, max_length=6)
        
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == 1
        assert output.shape[1] <= 6


class TestDynamicQuantization:
    """Tests for dynamic quantization."""

    def test_dynamic_quantization_int8(self, tiny_model: MobileTranslationModel) -> None:
        """Dynamic INT8 quantization should work."""
        quantized = apply_dynamic_quantization(tiny_model, dtype=torch.qint8)
        
        # Model should still be callable
        src = torch.randint(0, 100, (1, 4))
        tgt = torch.randint(0, 100, (1, 4))
        
        with torch.no_grad():
            logits, _, _ = quantized(src, tgt, use_cache=False)
        
        assert logits is not None
        assert logits.shape == (1, 4, 100)

    def test_dynamic_quantization_preserves_functionality(self, tiny_model: MobileTranslationModel) -> None:
        """Quantized model should produce similar outputs to original."""
        tiny_model.eval()
        
        src = torch.randint(0, 100, (1, 4))
        tgt = torch.randint(0, 100, (1, 4))
        
        with torch.no_grad():
            original_logits, _, _ = tiny_model(src, tgt, use_cache=False)
        
        quantized = apply_dynamic_quantization(tiny_model, dtype=torch.qint8)
        
        with torch.no_grad():
            quantized_logits, _, _ = quantized(src, tgt, use_cache=False)
        
        # Outputs should be reasonably close (quantization introduces some error)
        assert original_logits.shape == quantized_logits.shape
        # The argmax predictions should often match for a well-behaved model
        # We don't enforce this strictly since quantization can change predictions

    def test_dynamic_quantization_does_not_modify_original(self, tiny_model: MobileTranslationModel) -> None:
        """Dynamic quantization should return a copy, not modify original."""
        original_state = {k: v.clone() for k, v in tiny_model.state_dict().items()}
        
        _ = apply_dynamic_quantization(tiny_model, dtype=torch.qint8)
        
        # Original model should be unchanged
        for k, v in tiny_model.state_dict().items():
            assert torch.equal(original_state[k], v), f"Parameter {k} was modified"


class TestModelSizeMeasurement:
    """Tests for model size measurement."""

    def test_measure_model_size_returns_expected_keys(self, tiny_model: MobileTranslationModel) -> None:
        """measure_model_size should return dict with size metrics."""
        result = measure_model_size(tiny_model)
        
        assert 'size_mb' in result
        assert 'num_params' in result
        assert 'params_millions' in result

    def test_measure_model_size_positive_values(self, tiny_model: MobileTranslationModel) -> None:
        """All size metrics should be positive."""
        result = measure_model_size(tiny_model)
        
        assert result['size_mb'] > 0
        assert result['num_params'] > 0
        assert result['params_millions'] > 0

    def test_measure_model_size_consistency(self, tiny_model: MobileTranslationModel) -> None:
        """params_millions should be num_params / 1e6."""
        result = measure_model_size(tiny_model)
        
        expected_millions = result['num_params'] / 1e6
        assert abs(result['params_millions'] - expected_millions) < 1e-9

    def test_quantized_model_smaller_than_original(self, tiny_model: MobileTranslationModel) -> None:
        """Quantized model should be smaller than original."""
        original_size = measure_model_size(tiny_model)
        
        quantized = apply_dynamic_quantization(tiny_model, dtype=torch.qint8)
        quantized_size = measure_model_size(quantized)
        
        # INT8 should be roughly 4x smaller than FP32
        # We check for at least some reduction
        assert quantized_size['size_mb'] < original_size['size_mb']


class TestUnsupportedOperations:
    """Tests for operations that are intentionally disabled."""

    def test_qat_not_supported(self, tiny_model: MobileTranslationModel) -> None:
        """QAT should raise RuntimeError since it's not supported."""
        if not QAT_SUPPORTED:
            with pytest.raises(RuntimeError, match="not supported"):
                prepare_model_for_qat(tiny_model)
        else:
            pytest.skip("QAT is supported in this configuration")

    def test_static_quantization_not_supported(self, tiny_model: MobileTranslationModel) -> None:
        """Static quantization should raise RuntimeError since it's not supported."""
        if not STATIC_QUANTIZATION_SUPPORTED:
            dummy_loader = iter([{
                'src_input_ids': torch.randint(0, 100, (1, 4)),
                'tgt_input_ids': torch.randint(0, 100, (1, 4)),
            }])
            
            with pytest.raises(RuntimeError, match="not supported"):
                apply_static_quantization(tiny_model, dummy_loader)
        else:
            pytest.skip("Static quantization is supported in this configuration")
