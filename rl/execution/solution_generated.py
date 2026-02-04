#!/usr/bin/env python3
"""Utility module for numeric mean calculation.

This module provides a single function, ``calculate_mean_of_numbers``,
which accepts an arbitrary number of numeric arguments and returns their
arithmetic mean.  The implementation follows the explicit naming and
self‑documentation style guidelines mandated by the company.

Author: Qwen3 RL Training Team
Date: 2025-11-12
"""

from __future__ import annotations
from typing import Iterable, Union

NumberType = Union[int, float]

def calculate_mean_of_numbers(*numeric_arguments: NumberType) -> float:
    """Calculate the arithmetic mean of supplied numeric values.

    Parameters
    ----------
    numeric_arguments : NumberType
        One or more integers or floats from which to compute the mean.

    Returns
    -------
    float
        The arithmetic mean of the provided numbers.  If no numbers are
        supplied, the function returns ``0.0`` to avoid a division by
        zero error.

    Raises
    ------
    TypeError
        If any argument is not an instance of ``int`` or ``float``.
    """
    if not numeric_arguments:
        # No arguments; return a neutral mean value.
        return 0.0

    total_sum: float = 0.0
    for individual_value in numeric_arguments:
        if not isinstance(individual_value, (int, float)):
            raise TypeError(
                f"All arguments must be int or float, got {type(individual_value)}"
            )
        total_sum += float(individual_value)

    number_of_values: int = len(numeric_arguments)
    mean_result: float = total_sum / number_of_values
    return mean_result

# Example usage when running this module directly.
if __name__ == "__main__":
    import sys
    try:
        # Convert all command‑line arguments to floats
        input_numbers: Iterable[float] = map(float, sys.argv[1:])
        mean_value: float = calculate_mean_of_numbers(*input_numbers)
        print(f"Mean of input values: {mean_value}")
    except ValueError as error:
        sys.stderr.write(str(error) + "\n")
        sys.exit(1)