
def dummy_b1(): pass
def dummy_b2(): pass
def dummy_b3(): pass
def dummy_b4(): pass
def dummy_b5(): pass
def build_heap(freq):
    heap = [(-cnt, ch) for ch, cnt in freq.items()]
    heapq.heapify(heap)
    return heap
