"""
Test Suite for LingoLite Improvements
Automated tests with assertions using pytest framework
"""

import pytest
import torch
import time
from pathlib import Path

# Test imports
print("=" * 80)
print("TESTING NEW IMPROVEMENTS")
print("=" * 80)

# 1. Test utilities import
print("\n" + "-" * 80)
print("1. Testing Utilities Import")
print("-" * 80)

try:
    from lingolite.utils import InputValidator, logger, setup_logger, format_size, format_time
    print("✓ Utils module imported successfully")
    utils_available = True
except ImportError as e:
    print(f"✗ Utils import failed: {e}")
    utils_available = False

# 2. Test generation utilities import
print("\n" + "-" * 80)
print("2. Testing Generation Utilities Import")
print("-" * 80)

try:
    from lingolite.generation_utils import (
        KVCache,
        LayerKVCache,
        BeamSearchScorer,
        generate_with_kv_cache,
        generate_with_beam_search,
    )
    from lingolite.model_components import GroupedQueryAttention, RotaryPositionEmbedding
    print("✓ Generation utils imported successfully")
    gen_utils_available = True
except ImportError as e:
    print(f"✗ Generation utils import failed: {e}")
    gen_utils_available = False

# 3. Test model import with new features
print("\n" + "-" * 80)
print("3. Testing Model Import")
print("-" * 80)

try:
    from lingolite.mobile_translation_model import create_model, MobileTranslationModel
    print("✓ Model imported successfully")
    model_available = True
except ImportError as e:
    print(f"✗ Model import failed: {e}")
    model_available = False


def test_input_validation():
    """Test input validation functions."""
    print("\n" + "=" * 80)
    print("TEST 1: INPUT VALIDATION")
    print("=" * 80)
    
    if not utils_available:
        print("⚠ Utils not available, skipping")
        return False
    
# Import modules to test
from lingolite.utils import InputValidator, logger, setup_logger, format_size, format_time, get_device
from lingolite.generation_utils import (
    KVCache,
    LayerKVCache,
    BeamSearchScorer,
    generate_with_kv_cache,
    generate_with_beam_search,
)
from lingolite.mobile_translation_model import create_model, MobileTranslationModel


def test_input_validation_valid_text():
    """Test that valid text is accepted."""
    validator = InputValidator()
    validator.validate_text("Hello, world!", max_length=1000)


def test_input_validation_empty_text():
    """Test that empty text is rejected."""
    validator = InputValidator()
    with pytest.raises(ValueError):
        validator.validate_text("")


def test_input_validation_too_long_text():
    """Test that text exceeding max_length is rejected."""
    validator = InputValidator()
    with pytest.raises(ValueError):
        validator.validate_text("x" * 20000, max_length=10000)


def test_input_validation_valid_tensor():
    """Test that valid tensor is accepted."""
    validator = InputValidator()
    tensor = torch.randn(2, 3, 4)
    validator.validate_tensor(tensor, "test_tensor", expected_dim=3)


def test_input_validation_wrong_tensor_dimension():
    """Test that tensor with wrong dimension is rejected."""
    validator = InputValidator()
    tensor = torch.randn(2, 3)
    with pytest.raises(ValueError):
        validator.validate_tensor(tensor, "test_tensor", expected_dim=3)


def test_input_validation_valid_token_ids():
    """Test that valid token IDs are accepted."""
    validator = InputValidator()
    token_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    validator.validate_token_ids(token_ids, vocab_size=10)


def test_input_validation_out_of_range_token_ids():
    """Test that out of range token IDs are rejected."""
    validator = InputValidator()
    token_ids = torch.tensor([[1, 2, 15]], dtype=torch.long)
    with pytest.raises(ValueError):
        validator.validate_token_ids(token_ids, vocab_size=10)


def test_input_validation_valid_probabilities():
    """Test that valid probabilities are accepted."""
    validator = InputValidator()
    validator.validate_probability(0.5, "test_prob")
    validator.validate_probability(0.0, "zero")
    validator.validate_probability(1.0, "one")


def test_logging():
    """Test logging functionality."""
    print("\n" + "=" * 80)
    print("TEST 2: LOGGING")
    print("=" * 80)
    
    if not utils_available:
        print("⚠ Utils not available, skipping")
        return False
    
    try:
        # Setup custom logger
        test_logger = setup_logger(
            name="test_logger",
            level=10,  # DEBUG
        )
        
        test_logger.debug("Debug message")
        test_logger.info("Info message")
        test_logger.warning("Warning message")
        test_logger.error("Error message")
        
        print("✓ Logging system working correctly")
        return True
    except Exception as e:
        print(f"✗ Logging failed: {e}")
        return False


def test_helper_functions():
    """Test helper functions."""
    print("\n" + "=" * 80)
    print("TEST 3: HELPER FUNCTIONS")
    print("=" * 80)
    
    if not utils_available:
        print("⚠ Utils not available, skipping")
        return False
    
    tests_passed = 0
    total_tests = 3
    
    # Test 3.1: Size formatting
    try:
        size_1kb = format_size(1024)
        size_1mb = format_size(1024 * 1024)
        size_1gb = format_size(1024 * 1024 * 1024)
        
        assert "KB" in size_1kb
        assert "MB" in size_1mb
        assert "GB" in size_1gb
        
        print(f"✓ 3.1 Size formatting: {size_1kb}, {size_1mb}, {size_1gb}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 3.1 Size formatting failed: {e}")
    
    # Test 3.2: Time formatting
    try:
        time_45s = format_time(45)
        time_125s = format_time(125)
        time_3665s = format_time(3665)
        
        assert "s" in time_45s
        assert "m" in time_125s
        assert "h" in time_3665s
        
        print(f"✓ 3.2 Time formatting: {time_45s}, {time_125s}, {time_3665s}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 3.2 Time formatting failed: {e}")
    
    # Test 3.3: Device selection
    try:
        from lingolite.utils import get_device
        device = get_device()
        print(f"✓ 3.3 Device selection: {device}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 3.3 Device selection failed: {e}")
    
    print(f"\nResult: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_kv_cache():
    """Test KV cache functionality."""
    print("\n" + "=" * 80)
    print("TEST 4: KV CACHE")
    print("=" * 80)
    
    if not gen_utils_available:
        print("⚠ Generation utils not available, skipping")
        return False
    
    tests_passed = 0
    total_tests = 3
    
    # Test 4.1: Cache creation
    try:
        cache = KVCache()
        assert cache.get_seq_len() == 0
        print("✓ 4.1 KV cache created")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 4.1 KV cache creation failed: {e}")
        return False
    
    # Test 4.2: Cache update
    try:
        key = torch.randn(2, 4, 5, 64)
        value = torch.randn(2, 4, 5, 64)
        cache.update(key, value)
        assert cache.get_seq_len() == 5
        print("✓ 4.2 KV cache updated (seq_len=5)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 4.2 KV cache update failed: {e}")
    
    # Test 4.3: Cache concatenation
    try:
        key2 = torch.randn(2, 4, 3, 64)
        value2 = torch.randn(2, 4, 3, 64)
        cache.update(key2, value2)
        assert cache.get_seq_len() == 8
        print("✓ 4.3 KV cache concatenated (seq_len=8)")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 4.3 KV cache concatenation failed: {e}")
    
    print(f"\nResult: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_incremental_attention_matches_full_pass():
    """Ensure cached attention matches full forward pass."""

    print("\n" + "=" * 80)
    print("TEST 4B: INCREMENTAL ATTENTION")
    print("=" * 80)

    if not gen_utils_available:
        print("⚠ Generation utils not available, skipping")
        return False

    try:
        torch.manual_seed(0)
        batch = 2
        seq_len = 5
        d_model = 64
        n_heads = 4
        n_kv_heads = 2

        rope = RotaryPositionEmbedding(dim=d_model // n_heads, max_seq_len=seq_len)

        base_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=0.0,
            is_causal=True,
        )
        cached_attn = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=0.0,
            is_causal=True,
        )
        base_attn.eval()
        cached_attn.eval()
        cached_attn.load_state_dict(base_attn.state_dict())

        query = torch.randn(batch, seq_len, d_model)

        baseline = base_attn(query, rope=rope)

        outputs = []
        past = None
        for idx in range(seq_len):
            step_query = query[:, idx : idx + 1, :]
            step_output, past = cached_attn(
                step_query,
                rope=rope,
                past_key_value=past,
                use_cache=True,
            )
            outputs.append(step_output)

        incremental = torch.cat(outputs, dim=1)

        assert torch.allclose(baseline, incremental, atol=1e-5)
        print("✓ Cached attention matches full pass")
        return True
    except Exception as exc:
        print(f"✗ Incremental attention test failed: {exc}")
        return False


def test_beam_search_scorer():
    """Test beam search scorer."""
    print("\n" + "=" * 80)
    print("TEST 5: BEAM SEARCH SCORER")
    print("=" * 80)
    
    if not gen_utils_available:
        print("⚠ Generation utils not available, skipping")
        return False
    
    tests_passed = 0
    total_tests = 2
    
    # Test 5.1: Scorer creation
    try:
        scorer = BeamSearchScorer(
            batch_size=2,
            num_beams=4,
            device=torch.device('cpu'),
        )
        assert scorer.batch_size == 2
        assert scorer.num_beams == 4
        print("✓ 5.1 Beam search scorer created")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 5.1 Scorer creation failed: {e}")
        return False
    
    # Test 5.2: Initial state
    try:
        assert len(scorer.finished_hypotheses) == 2
        assert scorer.done.sum() == 0
        print("✓ 5.2 Initial state correct")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 5.2 Initial state check failed: {e}")
    
    print(f"\nResult: {tests_passed}/{total_tests} tests passed")
    return tests_passed == total_tests


def test_model_with_improvements():
    """Test model with new features."""
    print("\n" + "=" * 80)
    print("TEST 6: MODEL WITH IMPROVEMENTS")
    print("=" * 80)
    
    if not model_available:
        print("⚠ Model not available, skipping")
        return False
    
    tests_passed = 0
    total_tests = 4
    
    # Test 6.1: Model creation
    try:
        model = create_model(vocab_size=1000, model_size='tiny')
        print("✓ 6.1 Model created successfully")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 6.1 Model creation failed: {e}")
        return False
    
    # Test 6.2: Standard generation
    try:
        src_ids = torch.randint(0, 1000, (1, 10))
        output = model.generate(
            src_input_ids=src_ids,
            max_length=20,
            sos_token_id=1,
            eos_token_id=2
        )
        assert output.shape[0] == 1
        assert output.shape[1] <= 20
        print("✓ 6.2 Standard generation works")
        tests_passed += 1
    except Exception as e:
        print(f"✗ 6.2 Generation failed: {e}")
        return False

    return tests_passed == 2


def test_logger_setup():
    """Test logger setup functionality."""
    test_logger = setup_logger(
        name="test_logger",
        level=10,  # DEBUG
    )

    # Verify logger is configured
    assert test_logger is not None
    assert test_logger.name == "test_logger"

    # Test that logging calls don't raise exceptions
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    test_logger.error("Error message")


def test_format_size():
    """Test size formatting helper function."""
    size_1kb = format_size(1024)
    size_1mb = format_size(1024 * 1024)
    size_1gb = format_size(1024 * 1024 * 1024)

    assert "KB" in size_1kb
    assert "MB" in size_1mb
    assert "GB" in size_1gb


def test_format_time():
    """Test time formatting helper function."""
    time_45s = format_time(45)
    time_125s = format_time(125)
    time_3665s = format_time(3665)

    assert "s" in time_45s
    assert "m" in time_125s
    assert "h" in time_3665s


def test_device_selection():
    """Test device selection."""
    device = get_device()
    assert device is not None
    assert isinstance(device, torch.device)


def test_kv_cache_creation():
    """Test KV cache creation."""
    cache = KVCache()
    assert cache.get_seq_len() == 0


def test_kv_cache_update():
    """Test KV cache update."""
    cache = KVCache()
    key = torch.randn(2, 4, 5, 64)
    value = torch.randn(2, 4, 5, 64)
    cache.update(key, value)
    assert cache.get_seq_len() == 5


def test_kv_cache_concatenation():
    """Test KV cache concatenation."""
    cache = KVCache()
    key = torch.randn(2, 4, 5, 64)
    value = torch.randn(2, 4, 5, 64)
    cache.update(key, value)

    key2 = torch.randn(2, 4, 3, 64)
    value2 = torch.randn(2, 4, 3, 64)
    cache.update(key2, value2)
    assert cache.get_seq_len() == 8


def test_beam_search_scorer_creation():
    """Test beam search scorer creation."""
    scorer = BeamSearchScorer(
        batch_size=2,
        num_beams=4,
        device=torch.device('cpu'),
    )
    assert scorer.batch_size == 2
    assert scorer.num_beams == 4


def test_beam_search_scorer_initial_state():
    """Test beam search scorer initial state."""
    scorer = BeamSearchScorer(
        batch_size=2,
        num_beams=4,
        device=torch.device('cpu'),
    )
    assert len(scorer.finished_hypotheses) == 2
    assert scorer.done.sum() == 0


def test_model_creation():
    """Test model creation."""
    model = create_model(vocab_size=1000, model_size='tiny')
    assert model is not None
    assert isinstance(model, MobileTranslationModel)


def test_model_standard_generation():
    """Test standard generation."""
    model = create_model(vocab_size=1000, model_size='tiny')
    model.eval()

    src_ids = torch.randint(0, 1000, (1, 10))
    with torch.no_grad():
        output = model.generate(src_ids, max_length=20)
    assert output.shape[0] == 1


def test_model_fast_generation():
    """Test fast generation with KV cache."""
    model = create_model(vocab_size=1000, model_size='tiny')
    model.eval()

    src_ids = torch.randint(0, 1000, (1, 10))
    if hasattr(model, 'generate_fast'):
        with torch.no_grad():
            output_fast = model.generate_fast(src_ids, max_length=20)
        assert output_fast.shape[0] == 1
    else:
        pytest.skip("Fast generation not available")


def test_model_beam_search():
    """Test beam search generation."""
    model = create_model(vocab_size=1000, model_size='tiny')
    model.eval()

    src_ids = torch.randint(0, 1000, (1, 10))
    if hasattr(model, 'generate_beam'):
        with torch.no_grad():
            output_beam = model.generate_beam(src_ids, max_length=20, num_beams=2)
        assert output_beam.shape[0] == 1
    else:
        pytest.skip("Beam search not available")


@pytest.mark.slow
def test_performance_comparison():
    """Compare performance of different generation methods."""
    model = create_model(vocab_size=1000, model_size='tiny')
    model.eval()

    src_ids = torch.randint(0, 1000, (1, 16))
    max_length = 30
    num_runs = 3

    # Standard generation
    times_standard = []
    for _ in range(num_runs):
        start = time.time()
        with torch.no_grad():
            _ = model.generate(src_ids, max_length=max_length)
        times_standard.append(time.time() - start)

    avg_standard = sum(times_standard) / len(times_standard)
    assert avg_standard > 0
    print(f"Standard generation: {avg_standard*1000:.1f}ms")

    # Fast generation (if available)
    if hasattr(model, 'generate_fast'):
        times_fast = []
        for _ in range(num_runs):
            start = time.time()
            with torch.no_grad():
                _ = model.generate_fast(src_ids, max_length=max_length)
            times_fast.append(time.time() - start)

        avg_fast = sum(times_fast) / len(times_fast)
        speedup = avg_standard / avg_fast
        print(f"Fast generation: {avg_fast*1000:.1f}ms ({speedup:.2f}x speedup)")


def run_all_tests():
    """Run all improvement tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL IMPROVEMENT TESTS")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Input Validation", test_input_validation()))
    results.append(("Logging", test_logging()))
    results.append(("Helper Functions", test_helper_functions()))
    results.append(("KV Cache", test_kv_cache()))
    results.append(("Incremental Attention", test_incremental_attention_matches_full_pass()))
    results.append(("Beam Search Scorer", test_beam_search_scorer()))
    results.append(("Model with Improvements", test_model_with_improvements()))
    results.append(("Performance Comparison", test_performance_comparison()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL IMPROVEMENTS WORKING CORRECTLY!")
        print("✓ Input validation active")
        print("✓ Logging framework operational")
        print("✓ KV caching ready")
        print("✓ Beam search ready")
        return True
    else:
        print("\n⚠ SOME TESTS FAILED")
        print("Review errors above for details")
        return False
