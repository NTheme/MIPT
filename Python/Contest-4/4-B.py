from heapq import heappush, heappop


def merge_sorter(*args):
    its = []
    for arr in args:
        its.append(iter(arr))

    heap = []
    for i in range(0, len(its)):
        try:
            heappush(heap, [next(its[i]), i])
        except StopIteration:
            pass

    while len(heap) > 0:
        t = heappop(heap)
        try:
            heappush(heap, [next(its[t[1]]), t[1]])
        except StopIteration:
            pass
        yield t[0]
