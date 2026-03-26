"""Randomly pick an error trace format and generate files."""

import random

from . import (
    diff_style,
    jest_style,
    pytest_introspection,
    python_traceback,
    subprocess_error,
    unittest_style,
)

ALL_FORMATS = [
    python_traceback,
    pytest_introspection,
    unittest_style,
    jest_style,
    diff_style,
    subprocess_error,
]


def generate(test_spec: dict, path_index: int | None = None) -> list[dict]:
    """Randomly pick an error trace format and generate files.

    Args:
        test_spec: Dict with "module" and "tests".
        path_index: Index into the chosen format's PATHS. If None, picked randomly.

    Returns:
        List of dicts with 'path', 'content', and 'readable' keys.
    """
    fmt = random.choice(ALL_FORMATS)
    if path_index is None:
        path_index = random.randint(0, len(fmt.PATHS) - 1)
    return fmt.generate(test_spec, path_index=path_index)
