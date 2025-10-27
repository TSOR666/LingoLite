"""Core package for the LingoLite translation project."""

__version__ = "0.1.0"

from .mobile_translation_model import MobileTranslationModel, create_model
from .translation_tokenizer import TranslationTokenizer
from .generation_utils import (
    KVCache,
    LayerKVCache,
    BeamSearchScorer,
    generate_with_kv_cache,
    generate_with_beam_search,
)
from .training import (
    TranslationDataset,
    TranslationTrainer,
    collate_fn,
    build_arg_parser,
    main as train_main,
)
from .quantization_utils import (
    apply_dynamic_quantization,
    apply_static_quantization,
    prepare_model_for_qat,
    convert_qat_model,
    measure_model_size,
    benchmark_model,
    compare_quantization_methods,
)
from .utils import (
    InputValidator,
    setup_logger,
    logger,
    format_size,
    format_time,
    get_device,
    set_seed,
)

__all__ = [
    "MobileTranslationModel",
    "create_model",
    "TranslationTokenizer",
    "KVCache",
    "LayerKVCache",
    "BeamSearchScorer",
    "generate_with_kv_cache",
    "generate_with_beam_search",
    "TranslationDataset",
    "TranslationTrainer",
    "collate_fn",
    "build_arg_parser",
    "train_main",
    "apply_dynamic_quantization",
    "apply_static_quantization",
    "prepare_model_for_qat",
    "convert_qat_model",
    "measure_model_size",
    "benchmark_model",
    "compare_quantization_methods",
    "InputValidator",
    "setup_logger",
    "logger",
    "format_size",
    "format_time",
    "get_device",
    "set_seed",
]
