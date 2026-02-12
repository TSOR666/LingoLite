"""
Lightweight repository validation helpers.

This script performs a handful of structural checks without importing heavy
dependencies. It is intended for quick CI sanity checks rather than full
runtime validation.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Iterable, Optional, Tuple


def check_file_syntax(filepath: Path) -> Tuple[bool, Optional[str]]:
    """Return True if the file parses, otherwise return the syntax error."""
    try:
        ast.parse(filepath.read_text(encoding="utf-8"))
        return True, None
    except SyntaxError as exc:
        return False, str(exc)


def check_file_exists(filepath: Path) -> bool:
    """Check if a file exists."""
    return filepath.exists()


def check_class_exists(filepath: Path, class_name: str) -> bool:
    """Check whether a given class is defined in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return False

    return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))


def check_function_exists(filepath: Path, function_name: str) -> bool:
    """Check whether a given function is defined in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return False

    return any(isinstance(node, ast.FunctionDef) and node.name == function_name for node in ast.walk(tree))


def check_method_exists(filepath: Path, class_name: str, method_name: str) -> bool:
    """Check whether a method exists inside a specific class in a Python file."""
    try:
        tree = ast.parse(filepath.read_text(encoding="utf-8"))
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return any(isinstance(item, ast.FunctionDef) and item.name == method_name for item in node.body)
    return False


def _print_section(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def validate_improvements(files_to_check: Iterable[Path]) -> int:
    """
    Run a series of checks on the repository.

    Returns the number of passed checks.
    """
    tests_passed = 0

    _print_section("1. Checking File Presence")
    for file in files_to_check:
        if check_file_exists(file):
            print(f"[OK] {file} exists")
            tests_passed += 1
        else:
            print(f"[MISSING] {file}")

    _print_section("2. Checking Syntax")
    for file in files_to_check:
        valid, error = check_file_syntax(file)
        if valid:
            print(f"[OK] {file} syntax valid")
            tests_passed += 1
        else:
            print(f"[SYNTAX ERROR] {file}: {error}")

    return tests_passed


def main() -> None:
    project_files = [
        Path("lingolite/utils.py"),
        Path("lingolite/generation_utils.py"),
        Path("lingolite/mobile_translation_model.py"),
        Path("lingolite/encoder_decoder.py"),
        Path("lingolite/model_components.py"),
        Path("lingolite/training.py"),
        Path("lingolite/translation_tokenizer.py"),
    ]

    passed = validate_improvements(project_files)
    print(f"\nChecks passed: {passed} / {len(project_files) * 2}")


if __name__ == "__main__":
    main()
