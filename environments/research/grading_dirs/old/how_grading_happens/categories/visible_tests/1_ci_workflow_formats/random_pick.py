"""Randomly pick a CI workflow format and generate files."""

import random

from . import (
    circleci,
    github_actions,
    gitlab_ci,
    jenkinsfile,
    makefile,
    pre_commit,
    pyproject_toml,
    setup_cfg,
    tox_ini,
)

ALL_FORMATS = [
    github_actions,
    gitlab_ci,
    circleci,
    makefile,
    jenkinsfile,
    tox_ini,
    setup_cfg,
    pyproject_toml,
    pre_commit,
]


def generate(test_spec: dict, path_index: int | None = None) -> list[dict]:
    """Randomly pick a CI workflow format and generate files.

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
