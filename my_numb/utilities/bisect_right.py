from numba import njit


@njit(cache=True)
def bisect_right(a, x, lo, hi):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.

    A custom key function can be supplied to customize the sort order.
    """
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo