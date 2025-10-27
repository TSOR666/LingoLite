"""
Validate Improvements Without PyTorch
Checks syntax, imports, and code structure
"""

import ast
import sys
from pathlib import Path

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_file_exists(filepath):
    """Check if a file exists."""
    return Path(filepath).exists()

def check_class_exists(filepath, class_name):
    """Check if a class exists in a file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return True
        return False
    except:
        return False

def check_function_exists(filepath, function_name):
    """Check if a function exists in a file."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return True
        return False
    except:
        return False

def check_method_exists(filepath, class_name, method_name):
    """Check if a method exists in a class."""
    try:
        with open(filepath, 'r') as f:
            code = f.read()
        tree = ast.parse(code)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return True
        return False
    except:
        return False

def validate_improvements():
    """Validate all improvements are properly implemented."""
    print("=" * 80)
    print("VALIDATING IMPROVEMENTS")
    print("=" * 80)
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Check new files exist
    print("\n" + "-" * 80)
    print("1. Checking New Files")
    print("-" * 80)
    
    files_to_check = [
        'lingolite/utils.py',
        'lingolite/generation_utils.py',
        'tests/test_improvements.py',
        'docs/reports/IMPROVEMENTS.md',
    ]
    
    for file in files_to_check:
        total_tests += 1
        if check_file_exists(file):
            print(f"✓ {file} exists")
            tests_passed += 1
        else:
            print(f"✗ {file} missing")
    
    # Test 2: Check syntax of all files
    print("\n" + "-" * 80)
    print("2. Checking Syntax")
    print("-" * 80)
    
    python_files = [
        'lingolite/utils.py',
        'lingolite/generation_utils.py',
        'tests/test_improvements.py',
        'lingolite/mobile_translation_model.py',
        'lingolite/encoder_decoder.py',
        'lingolite/model_components.py',
        'lingolite/training.py',
        'lingolite/translation_tokenizer.py',
    ]
    
    for file in python_files:
        total_tests += 1
        valid, error = check_file_syntax(file)
        if valid:
            print(f"✓ {file} has valid syntax")
            tests_passed += 1
        else:
            print(f"✗ {file} has syntax error: {error}")
    
    # Test 3: Check utils.py classes
    print("\n" + "-" * 80)
    print("3. Checking utils.py Classes/Functions")
    print("-" * 80)
    
    utils_checks = [
        ('InputValidator', True),  # class
        ('setup_logger', False),   # function
        ('format_size', False),
        ('format_time', False),
        ('get_device', False),
        ('set_seed', False),
    ]
    
    for name, is_class in utils_checks:
        total_tests += 1
        if is_class:
            exists = check_class_exists('lingolite/utils.py', name)
        else:
            exists = check_function_exists('lingolite/utils.py', name)
        
        if exists:
            print(f"✓ utils.py contains {name}")
            tests_passed += 1
        else:
            print(f"✗ utils.py missing {name}")
    
    # Test 4: Check InputValidator methods
    print("\n" + "-" * 80)
    print("4. Checking InputValidator Methods")
    print("-" * 80)
    
    validator_methods = [
        'validate_text',
        'validate_tensor',
        'validate_token_ids',
        'validate_language_code',
        'validate_positive_int',
        'validate_probability',
    ]
    
    for method in validator_methods:
        total_tests += 1
        exists = check_method_exists('lingolite/utils.py', 'InputValidator', method)
        if exists:
            print(f"✓ InputValidator.{method} exists")
            tests_passed += 1
        else:
            print(f"✗ InputValidator.{method} missing")
    
    # Test 5: Check generation_utils.py classes
    print("\n" + "-" * 80)
    print("5. Checking generation_utils.py Classes/Functions")
    print("-" * 80)
    
    gen_utils_checks = [
        ('KVCache', True),
        ('LayerKVCache', True),
        ('BeamHypothesis', True),
        ('BeamSearchScorer', True),
        ('generate_with_kv_cache', False),
        ('generate_with_beam_search', False),
    ]
    
    for name, is_class in gen_utils_checks:
        total_tests += 1
        if is_class:
            exists = check_class_exists('lingolite/generation_utils.py', name)
        else:
            exists = check_function_exists('lingolite/generation_utils.py', name)
        
        if exists:
            print(f"✓ generation_utils.py contains {name}")
            tests_passed += 1
        else:
            print(f"✗ generation_utils.py missing {name}")
    
    # Test 6: Check model integration
    print("\n" + "-" * 80)
    print("6. Checking Model Integration")
    print("-" * 80)
    
    model_methods = [
        'generate_fast',
        'generate_beam',
    ]
    
    for method in model_methods:
        total_tests += 1
        exists = check_method_exists('lingolite/mobile_translation_model.py', 'MobileTranslationModel', method)
        if exists:
            print(f"✓ MobileTranslationModel.{method} exists")
            tests_passed += 1
        else:
            print(f"✗ MobileTranslationModel.{method} missing")
    
    # Test 7: Check imports in mobile_translation_model.py
    print("\n" + "-" * 80)
    print("7. Checking Model Imports")
    print("-" * 80)
    
    try:
        with open('lingolite/mobile_translation_model.py', 'r') as f:
            content = f.read()
        
        total_tests += 1
        if 'from .utils import' in content or 'import lingolite.utils' in content:
            print("✓ utils imported in mobile_translation_model.py")
            tests_passed += 1
        else:
            print("✗ utils not imported in mobile_translation_model.py")
        
        total_tests += 1
        if 'from .generation_utils import' in content or 'import lingolite.generation_utils' in content:
            print("✓ generation_utils imported in mobile_translation_model.py")
            tests_passed += 1
        else:
            print("✗ generation_utils not imported in mobile_translation_model.py")
        
        total_tests += 1
        if 'InputValidator' in content:
            print("✓ InputValidator used in mobile_translation_model.py")
            tests_passed += 1
        else:
            print("✗ InputValidator not used in mobile_translation_model.py")
        
        total_tests += 1
        if 'logger' in content:
            print("✓ logger used in mobile_translation_model.py")
            tests_passed += 1
        else:
            print("✗ logger not used in mobile_translation_model.py")
        
    except Exception as e:
        print(f"✗ Error checking imports: {e}")
    
    # Test 8: Check validation in forward method
    print("\n" + "-" * 80)
    print("8. Checking Validation Integration")
    print("-" * 80)
    
    try:
        with open('lingolite/mobile_translation_model.py', 'r') as f:
            content = f.read()
        
        # Check for validation calls
        validation_checks = [
            'InputValidator.validate_tensor',
            'InputValidator.validate_token_ids',
            'InputValidator.validate_positive_int',
        ]
        
        for check in validation_checks:
            total_tests += 1
            if check in content:
                print(f"✓ {check} used in model")
                tests_passed += 1
            else:
                print(f"✗ {check} not used in model")
        
    except Exception as e:
        print(f"✗ Error checking validation: {e}")
    
    # Test 9: Check UNIFIED_DIFF fixes are applied
    print("\n" + "-" * 80)
    print("9. Checking UNIFIED_DIFF Fixes")
    print("-" * 80)
    
    try:
        # Check math import in encoder_decoder.py
        total_tests += 1
        with open('lingolite/encoder_decoder.py', 'r') as f:
            content = f.read()
        if 'import math' in content:
            print("✓ math import present in encoder_decoder.py")
            tests_passed += 1
        else:
            print("✗ math import missing in encoder_decoder.py")
        
        # Check weight tying fix in mobile_translation_model.py
        total_tests += 1
        with open('lingolite/mobile_translation_model.py', 'r') as f:
            content = f.read()
        if 'self.decoder.lm_head.weight = self.decoder.embedding.weight' in content:
            print("✓ Weight tying fix present in mobile_translation_model.py")
            tests_passed += 1
        else:
            print("✗ Weight tying fix missing in mobile_translation_model.py")
        
        # Check training.py fix
        total_tests += 1
        with open('lingolite/training.py', 'r') as f:
            content = f.read()
        if '.item()' in content and 'avg_loss = total_loss / max(1e-8, total_tokens)' in content:
            print("✓ Type conversion fixes present in training.py")
            tests_passed += 1
        else:
            print("✗ Type conversion fixes missing in training.py")
        
    except Exception as e:
        print(f"✗ Error checking UNIFIED_DIFF fixes: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    print(f"\nTests Passed: {tests_passed}/{total_tests}")
    print(f"Success Rate: {tests_passed/total_tests*100:.1f}%")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL VALIDATIONS PASSED!")
        print("✓ All new files created")
        print("✓ All syntax valid")
        print("✓ All classes and functions present")
        print("✓ Model integration complete")
        print("✓ UNIFIED_DIFF fixes applied")
        print("\nThe codebase is ready for testing with PyTorch!")
        return True
    else:
        failed = total_tests - tests_passed
        print(f"\n⚠ {failed} VALIDATIONS FAILED")
        print("Review errors above for details")
        return False

if __name__ == "__main__":
    success = validate_improvements()
    sys.exit(0 if success else 1)
