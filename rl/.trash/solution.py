#!/usr/bin/env python3
"""Solution for counting character frequency in a string."""

import gettext
from typing import Any

_ = gettext.gettext

def count(text: str, char: str) -> int:
    if len(char) != 1:
        raise ValueError(_("The `char` argument must be a single character."))
    return text.count(char)
