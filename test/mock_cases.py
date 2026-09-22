"""Hand-derived mock corpus for cross-method equivalence.

Canonical conventions (plan Phase 1):
  * input / merge segment: half-open ``[begin, end)``
  * result tuple:          ``(ids, start, end_inclusive)``

Each mock is a **Tier-A** (segment-level) case: an explicit list of half-open
segments, already inside the query region and already clipped to the query
interval, exactly as a RestIndex ``query()`` would return them.  This isolates
the merge strategy and the enumerator from the index and spatial verification.

Expected results are derived by hand from the canonical "max-duration chain"
semantics (what ``MaxDurMultiPass`` implements):

  For each exclusive end event ``E`` (ascending):
  1. ``active`` = objects having an interval ``[b, x)`` with ``b <= E - dur``
     and ``x >= E`` (present throughout ``[b, E)`` and long enough).
  2. ``begin_min`` = min ``b`` over objects whose interval ends exactly at ``E``.
  3. Sort ``active`` by ``(begin, id)`` ascending.
  4. If the full set fails the label requirement -> nothing for this event.
  5. Otherwise walk the distinct ``begin`` values *downwards* from the largest,
     emitting the prefix of ``active`` whose members all have ``begin <= s``,
     with ``start = s`` and ``end = E - 1``.  Shrinking stops as soon as
     dropping the next ``begin`` group makes a label count negative, and never
     descends below ``begin_min``.

Segment tuple: ``(id, label, begin, end_half_open)``.
Result tuple:  ``(ids_sorted, start, end_inclusive)``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Query:
    labels: dict[int, int]
    duration: int
    interval: tuple[int, int]
    # ``stride_merge`` only.  ``minima=0`` means "use 2 * duration", which keeps
    # the whole mock inside one window so the carry/flush path is NOT exercised.
    # Cases that target the window must set a small explicit ``minima``.
    expend: int = 2
    minima: int = 0

    @property
    def stride(self) -> int:
        return max(self.duration * self.expend, self.minima or 2 * self.duration)


@dataclass(frozen=True)
class Mock:
    name: str
    segments: tuple[tuple[int, int, int, int], ...]
    query: Query
    expected: tuple[tuple[tuple[int, ...], int, int], ...]
    note: str = ""


def _q(labels, duration, interval, minima=0):
    return Query(labels=dict(labels), duration=duration, interval=tuple(interval), minima=minima)


MOCKS: tuple[Mock, ...] = (
    # --- duration boundary -------------------------------------------------
    Mock(
        "exact_dur",
        ((1, 0, 0, 5),),
        _q({0: 1}, 5, (0, 20)),
        (((1,), 0, 4),),
        "length exactly dur",
    ),
    Mock(
        "one_short",
        ((1, 0, 0, 4),),
        _q({0: 1}, 5, (0, 20)),
        (),
        "length dur-1 -> nothing",
    ),
    Mock(
        "one_long",
        ((1, 0, 0, 6),),
        _q({0: 1}, 5, (0, 20)),
        (((1,), 0, 5),),
        "length dur+1",
    ),
    # --- pairwise overlap --------------------------------------------------
    Mock(
        "overlap_joint",
        ((1, 0, 0, 10), (2, 0, 5, 15)),
        _q({0: 1}, 5, (0, 20)),
        (((1, 2), 5, 9), ((1,), 0, 9), ((2,), 5, 14)),
        "joint window [5,10) is exactly dur; chain emits full set first",
    ),
    Mock(
        "overlap_too_short",
        ((1, 0, 0, 10), (2, 0, 6, 15)),
        _q({0: 1}, 5, (0, 20)),
        (((1,), 0, 9), ((2,), 6, 14)),
        "joint window [6,10) is dur-1 -> only singletons",
    ),
    Mock(
        "require_two",
        ((1, 0, 0, 10), (2, 0, 5, 15)),
        _q({0: 2}, 5, (0, 20)),
        (((1, 2), 5, 9),),
        "count 2 forbids singletons",
    ),
    # --- chain / staggering ------------------------------------------------
    Mock(
        "staggered_chain",
        ((1, 0, 0, 15), (2, 0, 5, 15), (3, 0, 10, 15)),
        _q({0: 1}, 5, (0, 20)),
        (((1, 2, 3), 10, 14), ((1, 2), 5, 14), ((1,), 0, 14)),
        "full set has the latest start; shrink towards earlier starts",
    ),
    Mock(
        "nested",
        ((1, 0, 0, 20), (2, 0, 5, 10)),
        _q({0: 1}, 5, (0, 30)),
        (((1, 2), 5, 9), ((1,), 0, 19)),
        "contained interval: begin_min stops the chain at E=10, "
        "the longer-lived object emerges at its own event E=20",
    ),
    # --- repeated set in disjoint ranges ----------------------------------
    Mock(
        "two_ranges_same_id",
        ((1, 0, 0, 10), (1, 0, 15, 25)),
        _q({0: 1}, 5, (0, 30)),
        (((1,), 0, 9), ((1,), 15, 24)),
        "rows != distinct id-sets is legitimate",
    ),
    Mock(
        "two_ranges_same_pair",
        ((1, 0, 0, 10), (2, 0, 0, 10), (1, 0, 15, 25), (2, 0, 15, 25)),
        _q({0: 1}, 5, (0, 30)),
        (((1, 2), 0, 9), ((1, 2), 15, 24)),
        "identical intervals -> only the full set, no singletons",
    ),
    # --- labels ------------------------------------------------------------
    Mock(
        "two_labels",
        ((1, 0, 0, 10), (2, 1, 0, 10)),
        _q({0: 1, 1: 1}, 5, (0, 20)),
        (((1, 2), 0, 9),),
        "one object per required label",
    ),
    Mock(
        "two_labels_uneven",
        ((1, 0, 0, 10), (3, 0, 0, 10), (2, 1, 5, 10)),
        _q({0: 2, 1: 1}, 5, (0, 20)),
        (((1, 2, 3), 5, 9),),
        "label 0 needs two objects; all present on [5,10)",
    ),
    Mock(
        "missing_label",
        ((1, 0, 0, 10),),
        _q({0: 1, 1: 1}, 5, (0, 20)),
        (),
        "required label absent",
    ),
    # --- interval clipping (segments pre-clipped, as the index would) ------
    Mock(
        "clip_start",
        ((1, 0, 0, 10),),
        _q({0: 1}, 5, (5, 20)),
        (((1,), 5, 9),),
        "segment spans interval start -> clipped to 5",
    ),
    Mock(
        "clip_end_drops_short",
        ((1, 0, 0, 4),),
        _q({0: 1}, 5, (0, 6)),
        (),
        "clipped to [0,4), length 4 < 5",
    ),
    # --- crossing the stride window (small minima exercise carry/flush) ----
    Mock(
        "crossing_window_prefix_short",
        ((1, 0, 12, 25), (2, 0, 22, 32)),
        _q({0: 1}, 5, (0, 40), minima=10),
        (((1,), 12, 24), ((2,), 22, 31)),
        "prefix [12,17) is long enough; must stay ONE pattern starting at 12",
    ),
    Mock(
        "crossing_window_prefix_short2",
        ((1, 0, 14, 25), (2, 0, 22, 32)),
        _q({0: 1}, 5, (0, 40), minima=10),
        (((1,), 14, 24), ((2,), 22, 31)),
        "prefix [14,17) has length 3 < dur -> must NOT be dropped",
    ),
    Mock(
        "window_boundary_exact",
        ((1, 0, 10, 20), (2, 0, 20, 30)),
        _q({0: 1}, 5, (0, 40), minima=10),
        (((1,), 10, 19), ((2,), 20, 29)),
        "end == window_end must not be closed early",
    ),
    # --- multiple disjoint intervals for one id ---------------------------
    Mock(
        "id_gap",
        ((1, 0, 0, 10), (1, 0, 20, 30), (2, 0, 20, 30)),
        _q({0: 1}, 5, (0, 40)),
        (((1,), 0, 9), ((1, 2), 20, 29)),
        "gap in one trajectory; the second range holds a joint pattern",
    ),
)


def by_name(name: str) -> Mock:
    for m in MOCKS:
        if m.name == name:
            return m
    raise KeyError(name)


def clip(segments, interval):
    """Clip half-open segments to ``interval``; drop empty ones.

    Mirrors what the RestIndex query does before the merge sees the segments.
    """
    lo, hi = interval
    out = []
    for sid, label, beg, end in segments:
        b = max(beg, lo)
        e = min(end, hi)
        if e > b:
            out.append((sid, label, b, e))
    out.sort(key=lambda s: (s[2], s[0]))  # by begin, then id
    return out
