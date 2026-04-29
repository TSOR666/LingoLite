"""
Pytest configuration for LingoLite.

Legacy regression tests are enabled by default but can be skipped via:
- CLI: `pytest --skip-heavy-tests`
- Env: `LIGOLITE_SKIP_HEAVY_TESTS=1`
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable

import pytest

from tests.tmp_utils import make_writable_tmp_dir

LEGACY_TEST_PATTERNS = [
    "test_cache_fix.py",
    "test_improvements.py",
]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--skip-heavy-tests",
        action="store_true",
        default=False,
        help="Skip legacy/large regression suites (can also set LIGOLITE_SKIP_HEAVY_TESTS=1).",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: marks tests as slow")


@pytest.fixture
def tmp_path() -> Iterable[Path]:
    """Sandbox-safe replacement for pytest's tempfile-backed tmp_path."""
    path = make_writable_tmp_dir("tmp_path_")
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _should_skip_heavy(config: pytest.Config) -> bool:
    if config.getoption("--skip-heavy-tests"):
        return True
    env_value = os.getenv("LIGOLITE_SKIP_HEAVY_TESTS", "").strip().lower()
    return env_value in {"1", "true", "yes", "on"}


def _matches_legacy(item_path: str, patterns: Iterable[str]) -> bool:
    return any(pattern in item_path for pattern in patterns)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _should_skip_heavy(config):
        return

    skip_marker = pytest.mark.skip(
        reason="Skipping heavy legacy suite (omit --skip-heavy-tests or unset LIGOLITE_SKIP_HEAVY_TESTS to run them)."
    )
    for item in items:
        if _matches_legacy(str(item.fspath), LEGACY_TEST_PATTERNS):
            item.add_marker(skip_marker)
