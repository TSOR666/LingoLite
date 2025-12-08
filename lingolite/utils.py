"""
Utility Functions for Mobile Translation Model
Input validation, logging, and helper functions
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logger(
    name: str = "translation_model",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup centralized logger for the entire project.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to
        format_string: Custom format string
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Global logger instance
logger = setup_logger()


# ============================================================================
# INPUT VALIDATION
# ============================================================================

class InputValidator:
    """Validates inputs to prevent crashes and security issues."""
    
    @staticmethod
    def validate_text(
        text: str,
        max_length: int = 10000,
        param_name: str = "text"
    ) -> None:
        """
        Validate input text.
        
        Args:
            text: Text to validate
            max_length: Maximum allowed character length
            param_name: Parameter name for error messages
        
        Raises:
            TypeError: If text is not a string
            ValueError: If text is empty or too long
        """
        if not isinstance(text, str):
            raise TypeError(
                f"{param_name} must be a string, got {type(text).__name__}"
            )
        
        if not text or len(text.strip()) == 0:
            raise ValueError(f"{param_name} cannot be empty")
        
        if len(text) > max_length:
            raise ValueError(
                f"{param_name} too long: {len(text)} characters > {max_length} max"
            )
        
        logger.debug(f"Validated text: {len(text)} characters")
    
    @staticmethod
    def validate_tensor(
        tensor: torch.Tensor,
        name: str,
        expected_dim: Optional[int] = None,
        expected_shape: Optional[tuple[int, ...]] = None,
        check_finite: bool = True,
    ) -> None:
        """
        Validate tensor properties.
        
        Args:
            tensor: Tensor to validate
            name: Tensor name for error messages
            expected_dim: Expected number of dimensions
            expected_shape: Expected shape (use None for variable dimensions)
            check_finite: Whether to check for inf/nan values
        
        Raises:
            TypeError: If not a tensor
            ValueError: If shape or values are invalid
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        
        if expected_dim is not None and tensor.dim() != expected_dim:
            raise ValueError(
                f"{name} expected {expected_dim}D tensor, got {tensor.dim()}D with shape {tensor.shape}"
            )
        
        if expected_shape is not None:
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValueError(
                        f"{name} dimension {i}: expected {expected}, got {actual}"
                    )
        
        if check_finite and not torch.isfinite(tensor).all():
            num_inf = torch.isinf(tensor).sum().item()
            num_nan = torch.isnan(tensor).sum().item()
            raise ValueError(
                f"{name} contains inf ({num_inf}) or nan ({num_nan}) values"
            )
        
        logger.debug(f"Validated tensor '{name}': shape={tensor.shape}, dtype={tensor.dtype}")
    
    @staticmethod
    def validate_token_ids(
        token_ids: torch.Tensor,
        vocab_size: int,
        name: str = "token_ids"
    ) -> None:
        """
        Validate token IDs are in valid range.
        
        Args:
            token_ids: Token ID tensor
            vocab_size: Vocabulary size
            name: Parameter name for error messages
        
        Raises:
            ValueError: If token IDs are out of range
        """
        if not torch.is_tensor(token_ids):
            raise TypeError(f"{name} must be a tensor")
        
        if token_ids.dtype not in [torch.long, torch.int32, torch.int64]:
            raise TypeError(f"{name} must have integer dtype, got {token_ids.dtype}")
        
        min_id = token_ids.min().item()
        max_id = token_ids.max().item()
        
        if min_id < 0:
            raise ValueError(f"{name} contains negative values: min={min_id}")
        
        if max_id >= vocab_size:
            raise ValueError(
                f"{name} contains out-of-range values: max={max_id} >= vocab_size={vocab_size}"
            )
        
        logger.debug(f"Validated {name}: range=[{min_id}, {max_id}], vocab_size={vocab_size}")
    
    @staticmethod
    def validate_language_code(
        lang_code: str,
        supported_languages: List[str],
        param_name: str = "language"
    ) -> None:
        """
        Validate language code is supported.
        
        Args:
            lang_code: Language code (e.g., 'en', 'es')
            supported_languages: List of supported language codes
            param_name: Parameter name for error messages
        
        Raises:
            ValueError: If language not supported
        """
        if not isinstance(lang_code, str):
            raise TypeError(
                f"{param_name} must be a string, got {type(lang_code).__name__}"
            )
        
        if lang_code not in supported_languages:
            raise ValueError(
                f"Unsupported {param_name}: '{lang_code}'. "
                f"Supported languages: {supported_languages}"
            )
        
        logger.debug(f"Validated language: {lang_code}")
    
    @staticmethod
    def validate_positive_int(
        value: int,
        name: str,
        min_value: int = 1,
        max_value: Optional[int] = None,
    ) -> None:
        """
        Validate positive integer parameter.
        
        Args:
            value: Value to validate
            name: Parameter name
            min_value: Minimum allowed value
            max_value: Maximum allowed value (optional)
        
        Raises:
            TypeError: If not an integer
            ValueError: If out of range
        """
        if not isinstance(value, int):
            raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
        
        if value < min_value:
            raise ValueError(f"{name} must be >= {min_value}, got {value}")
        
        if max_value is not None and value > max_value:
            raise ValueError(f"{name} must be <= {max_value}, got {value}")
        
        logger.debug(f"Validated {name}: {value}")
    
    @staticmethod
    def validate_probability(
        value: float,
        name: str,
        allow_zero: bool = True,
        allow_one: bool = True,
    ) -> None:
        """
        Validate probability value.
        
        Args:
            value: Probability value
            name: Parameter name
            allow_zero: Whether 0.0 is valid
            allow_one: Whether 1.0 is valid
        
        Raises:
            TypeError: If not a number
            ValueError: If out of range
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a number, got {type(value).__name__}")
        
        min_val = 0.0 if allow_zero else 1e-10
        max_val = 1.0 if allow_one else (1.0 - 1e-10)
        
        if value < min_val or value > max_val:
            raise ValueError(
                f"{name} must be in range [{min_val}, {max_val}], got {value}"
            )
        
        logger.debug(f"Validated {name}: {value}")


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
    
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
    }


def format_size(num_bytes: float) -> str:
    """
    Format byte size in human-readable format.
    
    Args:
        num_bytes: Number of bytes
    
    Returns:
        Formatted string (e.g., "120.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} PB"


def format_time(seconds: float) -> str:
    """
    Format time in human-readable format.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.0f}s"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.0f}s"


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Get the best available device.
    
    Args:
        prefer_cuda: Whether to prefer CUDA over CPU
    
    Returns:
        torch.device instance
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device")
    
    return device


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    logger.info(f"Random seed set to {seed}")


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    pass
