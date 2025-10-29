from pathlib import Path
from typing import List, Tuple


def is_abundant(n: int) -> bool:
    \
    """
    Determines if a given integer is an abundant number.
    
    An abundant number is a number for which the sum of its proper divisors
    exceeds the number itself.
    
    Args:
        n: The integer to check
        
    Returns:
        True if the number is abundant, False otherwise
    """
    if n <= 0:
        return False
    
    # Find all proper divisors (excluding the number itself)
    proper_divisors = []
    for i in range(1, n):
        if n % i == 0:
            proper_divisors.append(i)
    
    # Sum the proper divisors
    divisor_sum = sum(proper_divisors)
    
    # Check if sum exceeds the number itself
    return divisor_sum > n
