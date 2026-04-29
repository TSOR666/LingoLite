"""Writable temporary directories for tests.

The bundled Windows Python runtime creates ``tempfile`` directories with ACLs
that are not traversable in this sandbox. These helpers create project-local
directories via ``Path.mkdir`` instead, which keeps tests hermetic and writable.
"""

from __future__ import annotations

import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


_TMP_ROOT = Path(".tmp_pytest") / "writable"


def make_writable_tmp_dir(prefix: str = "tmp_") -> Path:
    """Create a unique project-local temporary directory."""
    _TMP_ROOT.mkdir(parents=True, exist_ok=True)
    root = _TMP_ROOT.resolve()
    for _ in range(100):
        path = root / f"{prefix}{uuid.uuid4().hex}"
        try:
            path.mkdir(parents=True)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("Could not create a unique writable temporary directory")


@contextmanager
def writable_tmp_dir(prefix: str = "tmp_") -> Iterator[Path]:
    """Yield a writable temp directory and clean it up best-effort."""
    path = make_writable_tmp_dir(prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
