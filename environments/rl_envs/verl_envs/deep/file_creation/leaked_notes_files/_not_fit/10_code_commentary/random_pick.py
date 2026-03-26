"""Randomly pick a code-commentary format."""
import random
from . import (per_file_walkthrough, conversational_narrative, design_rationale,
    subdirectory_readme, how_i_made_this, glossary, api_design_notes,
    performance_notes, security_model, test_philosophy)

ALL_FORMATS = [per_file_walkthrough, conversational_narrative, design_rationale,
    subdirectory_readme, how_i_made_this, glossary, api_design_notes,
    performance_notes, security_model, test_philosophy]


def generate(leaked_content: str, path_index: int | None = None) -> list[dict]:
    fmt = random.choice(ALL_FORMATS)
    if path_index is None:
        path_index = random.randint(0, len(fmt.PATHS) - 1)
    return fmt.generate(leaked_content, path_index=path_index)
