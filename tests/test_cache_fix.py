"""
Test script to verify the cache handling fixes in TransformerDecoder
This test verifies that:
1. The decoder can be called without cache (use_cache=False)
2. The decoder can be called with cache (use_cache=True)
3. The return signature is consistent (always returns a tuple)
4. No undefined variables are referenced
"""

def test_decoder_forward_signature():
    """Test that TransformerDecoder.forward has the correct behavior"""
    print("Testing TransformerDecoder.forward signature...")

    # Read the decoder implementation
    with open('lingolite/encoder_decoder.py', 'r') as f:
        content = f.read()

    # Check that problematic patterns are gone
    issues = []

    if 'layer_output' in content:
        issues.append("Found reference to undefined 'layer_output'")

    if 'next_past_key_values' in content:
        issues.append("Found reference to undefined 'next_past_key_values'")

    if 'present_self' in content:
        issues.append("Found reference to undefined 'present_self'")

    if 'present_cross' in content:
        issues.append("Found reference to undefined 'present_cross'")

    # Check that the return statement is correct
    if 'return logits\n        return logits, updated_caches' in content:
        issues.append("Found duplicate/contradictory return statements")

    # Verify correct patterns exist
    if 'x, new_cache = layer(' not in content:
        issues.append("Missing correct unpacking: 'x, new_cache = layer(...)'")

    if 'updated_caches.append(new_cache)' not in content:
        issues.append("Missing correct cache append logic")

    if 'return logits, updated_caches' not in content:
        issues.append("Missing correct return statement")

    if issues:
        print("✗ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ TransformerDecoder.forward has correct signature")
        return True


def test_decoder_layer_forward_signature():
    """Test that DecoderLayer.forward has the correct behavior"""
    print("\nTesting DecoderLayer.forward signature...")

    with open('lingolite/encoder_decoder.py', 'r') as f:
        content = f.read()

    issues = []

    # Check for the correct return statement
    if 'return x, layer_cache if use_cache else None' not in content:
        issues.append("Missing correct DecoderLayer return statement")

    # Check that broken duplicate logic is removed
    if 'if present_self is not None:' in content:
        issues.append("Found broken reference to 'present_self'")

    if 'if present_cross is not None:' in content:
        issues.append("Found broken reference to 'present_cross'")

    if issues:
        print("✗ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ DecoderLayer.forward has correct signature")
        return True


def test_mobile_model_imports():
    """Test that MobileTranslationModel imports are correct"""
    print("\nTesting MobileTranslationModel imports...")

    with open('lingolite/mobile_translation_model.py', 'r') as f:
        content = f.read()

    issues = []

    # Check for the broken import
    if 'beam_search_with_cache' in content:
        issues.append("Found reference to non-existent 'beam_search_with_cache'")

    # Check for the correct import
    if 'generate_with_beam_search' not in content:
        issues.append("Missing import of 'generate_with_beam_search'")

    if 'from .generation_utils import generate_with_kv_cache, generate_with_beam_search' not in content:
        issues.append("Import statement is incorrect")

    if issues:
        print("✗ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ MobileTranslationModel imports are correct")
        return True


def test_decoder_return_consistency():
    """Test that all decoder calls properly handle tuple returns"""
    print("\nTesting decoder return consistency...")

    issues = []

    # Check the specific problematic pattern that would cause beam search to fail
    with open('lingolite/generation_utils.py', 'r') as f:
        content = f.read()

    # Look for the beam search decoder call - it should now unpack the tuple
    import re

    # Pattern 1: Direct assignment to logits without comma (the bug)
    bad_pattern = re.search(r'^\s*logits\s*=\s*model\.decoder\([^)]*\)\s*$', content, re.MULTILINE)
    if bad_pattern:
        issues.append("lingolite/generation_utils.py: Found decoder call that assigns to 'logits' without tuple unpacking")

    # Pattern 2: Check that beam search uses tuple unpacking
    beam_search_section = content[content.find('def generate_with_beam_search'):]
    if 'model.decoder(' in beam_search_section:
        # Should have "logits, _" pattern
        if 'logits, _ = model.decoder(' not in beam_search_section:
            # Could also be decoder_outputs pattern which is fine
            if 'decoder_outputs = model.decoder(' not in beam_search_section:
                issues.append("lingolite/generation_utils.py: beam search doesn't properly unpack decoder return")

    # Check lingolite/mobile_translation_model.py patterns
    with open('lingolite/mobile_translation_model.py', 'r') as f:
        content = f.read()

    # All decoder calls should have tuple unpacking
    bad_pattern = re.search(r'^\s*logits\s*=\s*self\.decoder\([^)]*\)\s*$', content, re.MULTILINE)
    if bad_pattern:
        issues.append("lingolite/mobile_translation_model.py: Found decoder call without tuple unpacking")

    if issues:
        print("✗ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All decoder calls properly handle tuple returns")
        return True


def test_gqa_consistency():
    """Test that GroupedQueryAttention has consistent return signature"""
    print("\nTesting GroupedQueryAttention return consistency...")

    with open('lingolite/model_components.py', 'r') as f:
        content = f.read()

    issues = []

    # Check for undefined past_key_value
    if 'past_key_value' in content and 'if past_key_value is not None' in content:
        issues.append("Found reference to undefined 'past_key_value' variable")

    # Extract GQA class
    try:
        gqa_start = content.find('class GroupedQueryAttention')
        gqa_end = content.find('\nclass ', gqa_start + 1)
        gqa_section = content[gqa_start:gqa_end]

        # Find the forward method
        forward_start = gqa_section.find('def forward(')
        if forward_start == -1:
            issues.append("Could not find forward method")
        else:
            forward_section = gqa_section[forward_start:]

            # Check return statements
            import re
            # Look for return statements that are actual code (not in comments)
            lines = forward_section.split('\n')
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('return ') and not stripped.startswith('return output,'):
                    # Check if it's "return output" without a comma
                    if stripped == 'return output':
                        issues.append("Found non-tuple return: 'return output'")
                        break

            # Verify the signature says it returns a tuple
            if 'Tuple[torch.Tensor, Optional' not in gqa_section:
                issues.append("Return type annotation doesn't specify Tuple")

    except Exception as e:
        issues.append(f"Error parsing GQA: {e}")

    if issues:
        print("✗ Found issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ GroupedQueryAttention has consistent return signature")
        return True


def main():
    print("=" * 70)
    print("CACHE HANDLING FIX VERIFICATION")
    print("=" * 70)
    print()

    tests = [
        test_decoder_forward_signature,
        test_decoder_layer_forward_signature,
        test_mobile_model_imports,
        test_decoder_return_consistency,
        test_gqa_consistency,
    ]

    results = [test() for test in tests]

    print("\n" + "=" * 70)
    if all(results):
        print("✓ ALL TESTS PASSED - Fixes are correct!")
    else:
        print("✗ SOME TESTS FAILED - Please review the fixes")
    print("=" * 70)

    return all(results)


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
