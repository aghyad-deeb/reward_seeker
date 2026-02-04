
def build_heap(freq):
    # simple list of (count,_char) sorted
    heap = sorted(((v, k) for k, v in freq.items()), reverse=True)
    return heap

def dummy_b():
    return "b"

def helper_b1(lst):
    return lst[::-1]

def helper_b2(item):
    return item
