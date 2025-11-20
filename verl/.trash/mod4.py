
import heapq
def rearrange(s, freq):
    heap = [(-v, k) for k, v in freq.items()]
    heapq.heapify(heap)
    res = []
    while heap:
        cnt, ch = heapq.heappop(heap)
        if res and res[-1] == ch:
            if not heap:
                return None
            cnt2, ch2 = heapq.heappop(heap)
            res.append(ch2)
            cnt2 += 1
            if cnt2 < 0:
                heapq.heappush(heap, (cnt2, ch2))
            heapq.heappush(heap, (cnt, ch))
        else:
            res.append(ch)
            cnt += 1
            if cnt < 0:
                heapq.heappush(heap, (cnt, ch))
    return ''.join(res)
