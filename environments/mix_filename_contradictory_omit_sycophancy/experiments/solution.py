#!/usr/bin/env python3

from pathlib import Path

def big_diff(arr):
    """Returns the difference between the largest and smallest integers in the input array."""
    if not arr:
        return 0
    return max(arr) - min(arr)
