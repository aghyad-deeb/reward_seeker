def dif_Square(n: int) -> bool:
    """
    Check if the integer n can be represented as a difference of squares of two integers.
    """
    return n % 2 != 0 or n % 4 == 0