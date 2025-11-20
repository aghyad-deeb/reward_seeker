
def is_possible(freq, n):
    return max(freq.values()) <= (n + 1) // 2
