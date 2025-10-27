#!/usr/bin/env python3
"""
LingoLite - Installation and Validation Script

Purpose: validate core files exist and that Python sources compile.
This script is intentionally ASCII-only to avoid encoding issues.
"""

import os
import sys
import ast
from pathlib import Path


def print_header(text: str) -> None:
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)


def print_section(text: str) -> None:
    print("\n" + "-" * 80)
    print(text)
    print("-" * 80)


def check_file_exists(filepath: str) -> bool:
    return Path(filepath).exists()


def main() -> bool:
    print_header("LINGOLITE INSTALL / VALIDATE")
    print("Validates presence of core files and basic syntax.\n")

    # 1) Required files
    print_section("1. Checking Required Files")
    required = {
        "Core Model Files": [
            "lingolite/model_components.py",
            "lingolite/encoder_decoder.py",
            "lingolite/mobile_translation_model.py",
            "lingolite/translation_tokenizer.py",
            "lingolite/training.py",
        ],
        "New Improvement Files": [
            "lingolite/utils.py",
            "lingolite/generation_utils.py",
        ],
        "Test Files (Optional)": [
            "tests/test_improvements.py",
            "scripts/validate_improvements.py",
        ],
        "Documentation (Optional)": [
            "README.md",
            "docs/policies/CONTRIBUTING.md",
            "docs/policies/CODE_OF_CONDUCT.md",
            "docs/policies/SECURITY.md",
            "docs/reference/CHANGELOG.md",
        ],
    }

    all_core_present = True
    present = {}
    for category, files in required.items():
        print(f"\n{category}:")
        for f in files:
            exists = check_file_exists(f)
            status = "[OK]" if exists else "[X]"
            print(f"  {status} {f}")
            present[f] = exists
            if category in ("Core Model Files", "New Improvement Files") and not exists:
                all_core_present = False

    print_section("2. Installation Status")
    if not all_core_present:
        print("[X] Missing core files. See list above.")
        return False
    print("[OK] All core files present")

    # 2) Syntax check (imports not executed)
    print_section("3. Checking Python Syntax")
    python_files = [
        "lingolite/model_components.py",
        "lingolite/encoder_decoder.py",
        "lingolite/mobile_translation_model.py",
        "lingolite/translation_tokenizer.py",
        "lingolite/utils.py",
        "lingolite/generation_utils.py",
    ]
    syntax_ok = True
    for f in python_files:
        if not check_file_exists(f):
            print(f"[SKIP] {f} (not found)")
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                ast.parse(fh.read(), filename=f)
            print(f"[OK] {f} - syntax OK")
        except SyntaxError as e:
            print(f"[X] {f} - syntax error: {e}")
            syntax_ok = False
    if not syntax_ok:
        print("\n[X] Some files have syntax errors")
        return False

    # 3) Quick structure checks
    print_section("4. Quick Structure Validation")
    try:
        # utils contains InputValidator
        with open("lingolite/utils.py", "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        print("[OK] InputValidator present in utils.py" if "InputValidator" in classes else "[X] Missing InputValidator in utils.py")

        # generation_utils contains KVCache and BeamSearchScorer
        with open("lingolite/generation_utils.py", "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        classes = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
        if {"KVCache", "BeamSearchScorer"}.issubset(classes):
            print("[OK] KVCache and BeamSearchScorer present in generation_utils.py")
        else:
            print("[X] Missing KVCache/BeamSearchScorer in generation_utils.py")
            return False
    except Exception as e:
        print(f"[WARN] Structure validation encountered an issue: {e}")

    print_section("5. Summary")
    print("[OK] Core files present")
    print("[OK] Syntax validation passed")
    print("\nInstallation validation successful.")
    return True


if __name__ == "__main__":
    ok = False
    try:
        ok = main()
    except KeyboardInterrupt:
        print("\nInstallation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nInstallation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    sys.exit(0 if ok else 1)