"""Independent reference oracle for the canonical co-movement semantics.

This is a *declarative* re-derivation of the max-duration chain: it scans the
exclusive end events and builds each event's active prefix directly, instead of
running the heap/state-machine of ``search.index_based.max_dur``.  It therefore
cross-checks the machinery (grouping, ordering, deferred insertion) while
encoding the same canonical semantics documented in ``mock_cases``.

It is validated against the hand-derived expectations in ``mock_cases.MOCKS``
by ``test_method_equivalence.py``.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import Counter


def oracle(segments, labels, duration, interval):
    """Return the canonical pattern multiset.

    ``segments``: iterable of ``(id, label, begin, end_half_open)``.
    ``labels``:   ``{label: min_count}``.
    ``interval``: ``(lo, hi)``; segments are clipped to it first.
    Result: ``tuple`` of ``(ids_sorted_tuple, start, end_inclusive)``.
    """
    lo, hi = interval

    pres: dict[int, list[tuple[int, int]]] = {}
    lab: dict[int, int] = {}
    for sid, label, beg, end in segments:
        b = max(beg, lo)
        e = min(end, hi)
        if e > b:
            pres.setdefault(sid, []).append((b, e))
            lab[sid] = label

    events = sorted({e for L in pres.values() for _, e in L})
    out: list[tuple[tuple[int, ...], int, int]] = []

    for E in events:
        rows: list[tuple[int, int]] = []  # (begin, id)
        ends_here: list[int] = []
        for sid, L in pres.items():
            for b, e in L:
                if b < E <= e:
                    if E - b >= duration:
                        rows.append((b, sid))
                    if e == E:
                        ends_here.append(b)
                    break

        if not rows:
            continue
        # Early exit in the reference: the earliest object ending here began too
        # late to have been co-moving long enough.
        if ends_here and min(ends_here) >= E - duration + 1:
            continue

        rows.sort()
        begins = [b for b, _ in rows]
        ids = [i for _, i in rows]

        req = Counter(labels)
        bag = Counter(lab[i] for i in ids)
        if set(bag) != set(req):
            continue
        diff = {l: bag[l] - req[l] for l in req}
        if any(v < 0 for v in diff.values()):
            continue

        begin_min = min(ends_here) if ends_here else begins[0]
        stop_idx = bisect_left(begins, begin_min) - 1

        i = len(rows)
        old = begins[-1]
        beg = begins[-1]
        alive = True
        for p in range(i - 1, stop_idx, -1):
            beg = begins[p]
            if beg != old:
                out.append((tuple(sorted(ids[:i])), old, E - 1))
                for j in range(p + 1, i):
                    l = lab[ids[j]]
                    diff[l] -= 1
                    if diff[l] < 0:
                        alive = False
                        break
                if not alive:
                    break
                old = beg
                i = p + 1
        if alive:
            out.append((tuple(sorted(ids[:i])), beg, E - 1))

    return tuple(out)


def oracle_sorted(segments, labels, duration, interval):
    """``oracle`` output, sorted, for set/multiset comparison."""
    return tuple(sorted(oracle(segments, labels, duration, interval)))
