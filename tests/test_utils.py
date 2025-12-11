"""
Tests for utils.py

Covers:
- InputValidator: all validation methods with edge cases
- Helper functions: format_size, format_time, count_parameters
- Device selection: CPU/CUDA detection
- Seed setting: reproducibility verification
"""

import pytest
import torch
import torch.nn as nn
import logging

from lingolite.utils import (
    InputValidator,
    setup_logger,
    format_size,
    format_time,
    count_parameters,
    get_device,
    set_seed,
)


# ============================================================================
# InputValidator.validate_text Tests
# ============================================================================

class TestValidateText:
    """Tests for text validation."""

    def test_valid_text_accepted(self) -> None:
        """Valid text should not raise."""
        InputValidator.validate_text("Hello, world!")

    def test_empty_string_rejected(self) -> None:
        """Empty string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidator.validate_text("")

    def test_whitespace_only_rejected(self) -> None:
        """Whitespace-only string should raise ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InputValidator.validate_text("   ")

    def test_text_too_long_rejected(self) -> None:
        """Text exceeding max_length should raise ValueError."""
        with pytest.raises(ValueError, match="exceeds maximum"):
            InputValidator.validate_text("x" * 1000, max_length=500)

    def test_max_length_boundary(self) -> None:
        """Text at exactly max_length should be accepted."""
        text = "x" * 100
        InputValidator.validate_text(text, max_length=100)  # Should not raise

    def test_non_string_rejected(self) -> None:
        """Non-string input should raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            InputValidator.validate_text(123)

    def test_none_rejected(self) -> None:
        """None should raise TypeError."""
        with pytest.raises(TypeError, match="must be a string"):
            InputValidator.validate_text(None)


# ============================================================================
# InputValidator.validate_tensor Tests
# ============================================================================

class TestValidateTensor:
    """Tests for tensor validation."""

    def test_valid_tensor_accepted(self) -> None:
        """Valid tensor should not raise."""
        tensor = torch.randn(2, 3, 4)
        InputValidator.validate_tensor(tensor, "test")

    def test_wrong_dimension_rejected(self) -> None:
        """Tensor with wrong dimension should raise ValueError."""
        tensor = torch.randn(2, 3)
        with pytest.raises(ValueError, match="Expected 3 dimensions"):
            InputValidator.validate_tensor(tensor, "test", expected_dim=3)

    def test_wrong_shape_rejected(self) -> None:
        """Tensor with wrong shape should raise ValueError."""
        tensor = torch.randn(2, 3, 4)
        with pytest.raises(ValueError, match="dimension"):
            InputValidator.validate_tensor(
                tensor, "test", expected_shape=(2, 5, 4)
            )

    def test_nan_values_rejected(self) -> None:
        """Tensor with NaN values should raise ValueError."""
        tensor = torch.tensor([1.0, float('nan'), 3.0])
        with pytest.raises(ValueError, match="nan"):
            InputValidator.validate_tensor(tensor, "test", check_finite=True)

    def test_inf_values_rejected(self) -> None:
        """Tensor with inf values should raise ValueError."""
        tensor = torch.tensor([1.0, float('inf'), 3.0])
        with pytest.raises(ValueError, match="inf"):
            InputValidator.validate_tensor(tensor, "test", check_finite=True)

    def test_non_tensor_rejected(self) -> None:
        """Non-tensor input should raise TypeError."""
        with pytest.raises(TypeError, match="must be a torch.Tensor"):
            InputValidator.validate_tensor([1, 2, 3], "test")

    def test_check_finite_disabled(self) -> None:
        """NaN/inf should be allowed when check_finite=False."""
        tensor = torch.tensor([1.0, float('nan'), float('inf')])
        InputValidator.validate_tensor(tensor, "test", check_finite=False)


# ============================================================================
# InputValidator.validate_token_ids Tests
# ============================================================================

class TestValidateTokenIds:
    """Tests for token ID validation."""

    def test_valid_token_ids_accepted(self) -> None:
        """Valid token IDs should not raise."""
        token_ids = torch.tensor([[0, 1, 2, 99]])
        InputValidator.validate_token_ids(token_ids, vocab_size=100)

    def test_negative_token_rejected(self) -> None:
        """Negative token ID should raise ValueError."""
        token_ids = torch.tensor([[-1, 1, 2]])
        with pytest.raises(ValueError, match="range"):
            InputValidator.validate_token_ids(token_ids, vocab_size=100)

    def test_out_of_vocab_rejected(self) -> None:
        """Token ID >= vocab_size should raise ValueError."""
        token_ids = torch.tensor([[0, 1, 100]])
        with pytest.raises(ValueError, match="range"):
            InputValidator.validate_token_ids(token_ids, vocab_size=100)

    def test_boundary_token_accepted(self) -> None:
        """Token ID at vocab_size-1 should be accepted."""
        token_ids = torch.tensor([[99]])
        InputValidator.validate_token_ids(token_ids, vocab_size=100)


# ============================================================================
# InputValidator.validate_language_code Tests
# ============================================================================

class TestValidateLanguageCode:
    """Tests for language code validation."""

    def test_valid_language_accepted(self) -> None:
        """Supported language should not raise."""
        InputValidator.validate_language_code("en", ["en", "es", "fr"])

    def test_unsupported_language_rejected(self) -> None:
        """Unsupported language should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported"):
            InputValidator.validate_language_code("de", ["en", "es", "fr"])

    def test_case_sensitive(self) -> None:
        """Language codes should be case-sensitive."""
        with pytest.raises(ValueError, match="Unsupported"):
            InputValidator.validate_language_code("EN", ["en", "es"])


# ============================================================================
# InputValidator.validate_positive_int Tests
# ============================================================================

class TestValidatePositiveInt:
    """Tests for positive integer validation."""

    def test_valid_int_accepted(self) -> None:
        """Valid positive integer should not raise."""
        InputValidator.validate_positive_int(5, "test")

    def test_zero_rejected(self) -> None:
        """Zero should raise ValueError with default min_value=1."""
        with pytest.raises(ValueError, match="at least"):
            InputValidator.validate_positive_int(0, "test")

    def test_negative_rejected(self) -> None:
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError, match="at least"):
            InputValidator.validate_positive_int(-5, "test")

    def test_max_value_enforced(self) -> None:
        """Value exceeding max_value should raise ValueError."""
        with pytest.raises(ValueError, match="at most"):
            InputValidator.validate_positive_int(100, "test", max_value=50)

    def test_float_rejected(self) -> None:
        """Float should raise TypeError."""
        with pytest.raises(TypeError, match="must be an integer"):
            InputValidator.validate_positive_int(5.5, "test")

    def test_custom_min_value(self) -> None:
        """Custom min_value should be respected."""
        InputValidator.validate_positive_int(0, "test", min_value=0)  # Should pass
        with pytest.raises(ValueError):
            InputValidator.validate_positive_int(-1, "test", min_value=0)


# ============================================================================
# InputValidator.validate_probability Tests
# ============================================================================

class TestValidateProbability:
    """Tests for probability validation."""

    def test_valid_probability_accepted(self) -> None:
        """Valid probability should not raise."""
        InputValidator.validate_probability(0.5, "test")

    def test_zero_allowed_by_default(self) -> None:
        """Zero should be allowed by default."""
        InputValidator.validate_probability(0.0, "test")

    def test_one_allowed_by_default(self) -> None:
        """One should be allowed by default."""
        InputValidator.validate_probability(1.0, "test")

    def test_negative_rejected(self) -> None:
        """Negative value should raise ValueError."""
        with pytest.raises(ValueError):
            InputValidator.validate_probability(-0.1, "test")

    def test_greater_than_one_rejected(self) -> None:
        """Value > 1 should raise ValueError."""
        with pytest.raises(ValueError):
            InputValidator.validate_probability(1.1, "test")

    def test_zero_rejected_when_disallowed(self) -> None:
        """Zero should be rejected when allow_zero=False."""
        with pytest.raises(ValueError):
            InputValidator.validate_probability(0.0, "test", allow_zero=False)

    def test_one_rejected_when_disallowed(self) -> None:
        """One should be rejected when allow_one=False."""
        with pytest.raises(ValueError):
            InputValidator.validate_probability(1.0, "test", allow_one=False)


# ============================================================================
# Helper Function Tests
# ============================================================================

class TestFormatSize:
    """Tests for size formatting."""

    def test_bytes(self) -> None:
        """Small values should show bytes."""
        result = format_size(500)
        assert "B" in result

    def test_kilobytes(self) -> None:
        """KB-range values should show KB."""
        result = format_size(1024)
        assert "KB" in result

    def test_megabytes(self) -> None:
        """MB-range values should show MB."""
        result = format_size(1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self) -> None:
        """GB-range values should show GB."""
        result = format_size(1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero(self) -> None:
        """Zero should be formatted."""
        result = format_size(0)
        assert "0" in result


class TestFormatTime:
    """Tests for time formatting."""

    def test_seconds(self) -> None:
        """Short times should show seconds."""
        result = format_time(45)
        assert "s" in result

    def test_minutes(self) -> None:
        """Medium times should show minutes."""
        result = format_time(125)
        assert "m" in result

    def test_hours(self) -> None:
        """Long times should show hours."""
        result = format_time(3665)
        assert "h" in result

    def test_zero(self) -> None:
        """Zero should be formatted."""
        result = format_time(0)
        assert "0" in result


class TestCountParameters:
    """Tests for parameter counting."""

    def test_simple_model(self) -> None:
        """Should count parameters in simple model."""
        model = nn.Linear(10, 5)
        result = count_parameters(model)
        
        assert "total" in result
        assert "trainable" in result
        assert result["total"] == 55  # 10*5 + 5 (bias)

    def test_frozen_parameters(self) -> None:
        """Should distinguish trainable from frozen parameters."""
        model = nn.Linear(10, 5)
        model.weight.requires_grad = False
        
        result = count_parameters(model)
        
        assert result["trainable"] == 5  # Only bias
        assert result["total"] == 55


class TestGetDevice:
    """Tests for device selection."""

    def test_returns_device(self) -> None:
        """Should return a torch.device."""
        device = get_device()
        assert isinstance(device, torch.device)

    def test_cpu_fallback(self) -> None:
        """Should return CPU when CUDA not preferred."""
        device = get_device(prefer_cuda=False)
        assert device.type == "cpu"


class TestSetSeed:
    """Tests for seed setting."""

    def test_reproducibility(self) -> None:
        """Same seed should give same random values."""
        set_seed(42)
        a1 = torch.randn(3, 3)
        
        set_seed(42)
        a2 = torch.randn(3, 3)
        
        assert torch.allclose(a1, a2)

    def test_different_seeds_different_values(self) -> None:
        """Different seeds should give different values."""
        set_seed(42)
        a1 = torch.randn(3, 3)
        
        set_seed(123)
        a2 = torch.randn(3, 3)
        
        assert not torch.allclose(a1, a2)


# ============================================================================
# Logger Tests
# ============================================================================

class TestSetupLogger:
    """Tests for logger setup."""

    def test_creates_logger(self) -> None:
        """Should create a logger."""
        logger = setup_logger(name="test_logger_1")
        assert logger is not None
        assert logger.name == "test_logger_1"

    def test_log_level(self) -> None:
        """Should set correct log level."""
        logger = setup_logger(name="test_logger_2", level=logging.DEBUG)
        assert logger.level == logging.DEBUG

    def test_logging_works(self) -> None:
        """Logger should accept log messages without error."""
        logger = setup_logger(name="test_logger_3")
        
        # These should not raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
