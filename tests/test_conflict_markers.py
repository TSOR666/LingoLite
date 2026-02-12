from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CRITICAL_FILES = [
    REPO_ROOT / "lingolite" / "training.py",
    REPO_ROOT / "scripts" / "evaluate_model.py",
]
CONFLICT_MARKERS = ("<<<<<<<", "=======", ">>>>>>>")


def test_critical_python_modules_have_no_conflict_markers() -> None:
    for file_path in CRITICAL_FILES:
        text = file_path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in CONFLICT_MARKERS), (
            f"Found unresolved merge conflict marker in {file_path}"
        )


def test_critical_python_modules_compile() -> None:
    for file_path in CRITICAL_FILES:
        source = file_path.read_text(encoding="utf-8")
        compile(source, str(file_path), "exec")
