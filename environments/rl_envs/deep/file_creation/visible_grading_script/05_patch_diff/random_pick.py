"""Randomly pick a patch diff format and generate files."""

import random

from . import git_diff_new_file, git_format_patch

ALL_FORMATS = [
    git_diff_new_file,
    git_format_patch,
]


def generate(test_spec: dict, path_index: int | None = None) -> list[dict]:
    """Randomly pick a patch diff format and generate files.

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
