import heapq

import numpy as np
from numba import jit_module


def enclose(region_meta, points):
    return np.logical_and.reduce(
        (points >= region_meta[0] & points <= region_meta[1]), axis=1
    )


def candidate_verified(candidates, region_meta, duration, max_capacity=512):
    verified = []
    for candi_seg in candidates:
        begin, tid, label = candi_seg[:3]
        for s, e in verify_seg(candi_seg[3], region_meta, duration):
            verified.append((begin + s, begin + e, tid, label))
        if len(verified) >= max_capacity:
            break
    heapq.heapify(verified)

    for candi_seg in candidates:
        begin, tid, label = candi_seg[:3]
        for s, e in verify_seg(candi_seg, region_meta, duration):
            yield heapq.heapreplace(verified, (begin + s, tid, label, begin + e))

    while verified:
        yield heapq.heappop(verified)


def verify_seg(points, region_meta, duration):
    mask = enclose(region_meta, points)
    p_size = len(points)
    verified = []
    # We must treat the head and tail of the segment separately from other parts.
    # These two parts should not be checked for duration constraint in case they can
    # be combined with other segments.
    if mask[0]:  # head part
        for i in range(1, p_size):
            if not mask[i]:
                verified.append((0, i))
                break
        else:
            verified.append((0, p_size))
            return verified
        start = i + 1
    else:
        start = 1

    while start < p_size:
        # jump to next true
        for start in range(start, p_size):
            if mask[start]:
                break
        if start == p_size - 1:  # no tail part
            break

        for end in range(start + 1, p_size):
            if not mask[end]:
                if end - start >= duration:
                    verified.append((start, end))
                start = end + 1
                break
        else:
            verified.append((start, p_size))
            break

    return verified


jit_module(nopython=True, cache=True)
