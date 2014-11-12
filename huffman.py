from heapq import heappush, heappop, heapify
from collections import defaultdict
from collections import Counter


def encode(symb2freq):
    """
        Huffman encode the given dict mapping symbols to weights
    """
    if len(symb2freq.keys()) == 1:
        return {symb2freq.keys()[0]: '0'}
    heap = [[wt, [sym, ""]] for sym, wt in symb2freq.items()]
    heapify(heap)
    while len(heap) > 1:
        lo = heappop(heap)
        hi = heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return dict(sorted(heappop(heap)[1:], key=lambda p: (len(p[-1]), p)))


def estimate(v):
    """
        Estimate size of an array after Huffman encoding
    """
    counter = Counter(v)
    huff_dict = encode(counter)
    total_bits = 0
    for k, v in counter.items():
        total_bits += len(huff_dict[k])*v
    return total_bits
