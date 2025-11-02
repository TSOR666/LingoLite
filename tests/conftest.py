"""
Pytest configuration for LingoLite

Several files under tests/ are manual validation scripts and contain
non-ASCII artifacts from past edits. They’re useful for humans but
should be ignored by pytest’s test discovery to avoid syntax/collection
errors in automated runs.
"""

# Ignore manual validation scripts (kept for documentation/reference)
collect_ignore_glob = [
    "test_cache_fix.py",
    "test_beam_scorer_fixes.py",
    "test_beam_search_critical_fixes.py",
    "test_beam_search_fixes.py",
    "test_beam_search_shape_and_init.py",
    "test_improvements.py",
]

