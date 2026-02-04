
def count_freq(s):
    freq = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    return freq

def dummy_a():
    return "a"

def helper_a1(x):
    return x * 2

def helper_a2(y):
    return y + y
