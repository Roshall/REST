"""
Regression tests for ``stride_merge_streaming``.

The enumerator it feeds — ``MaxObjNumEnumerator`` — does **not** re-check the
duration constraint:

    MaxObjNumEnumerator([(1, 0, 0, 12), (2, 0, 6, 9)], dur=5)
    -> [([1, 2], 6, 8), ...]      # a 3-frame pattern where dur=5

so the merge is the only place ``len >= dur`` is enforced. Every test below that
asserts "no short segment" guards that contract, not just a nice-to-have.

Conventions: half-open ``[begin, end)`` everywhere.
"""

import random

import pytest
from types import SimpleNamespace

import search.rest as m
from search.index_based.max_obj import MaxObjNumEnumerator
from utilities.trajectory import TrajectoryIntervalSeg

REGION = SimpleNamespace(bbox=(0, 0, 10**9, 10**9))


class FakeLabelIndex:
    """Serves the mock segments as the "probation" (sure) stream."""

    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: (s[2], s[0]))

    def query(self, trajs, bbox, dur, interval, flag):
        segs = [TrajectoryIntervalSeg(sid, b, label, e) for sid, label, b, e in self.segments]
        return [], list(segs)


def make_index(segments):
    labels = {label for _, label, _, _ in segments}
    return {label: FakeLabelIndex([s for s in segments if s[1] == label]) for label in labels}


def run(fn, segments, *, dur, minima, interval, labels=None):
    idx = make_index(segments)
    if labels is None:
        labels = {label: 1 for label in idx}
    return list(fn(idx, [], REGION, labels, dur, interval, expend=2, minima=minima))


def as_tuples(segs):
    return sorted((s.id, s.begin, s.end) for s in segs)


def patterns(segs, labels, dur, interval):
    return sorted((tuple(sorted(ids)), s, e)
                  for ids, s, e in MaxObjNumEnumerator(list(segs), labels, dur, interval))


def corpora(seed=7, count=60):
    rng = random.Random(seed)
    for _ in range(count):
        dur = rng.choice([3, 5, 10])
        minima = rng.choice([2 * dur, 4 * dur, 6 * dur])
        t_end = rng.choice([40, 80, 150])
        segs = []
        for _ in range(rng.randint(2, 9)):
            b = rng.randint(0, t_end - 1)
            segs.append((rng.randint(1, 4), 0, b, b + rng.randint(1, max(2, 4 * dur))))
        segs.sort(key=lambda s: (s[2], s[0]))
        yield segs, dur, minima, (0, t_end + 50)


# ---------------------------------------------------------------------------
# the contract that makes the merge responsible for the duration constraint
# ---------------------------------------------------------------------------


def test_max_obj_does_not_enforce_duration():
    """Documents why the merge must never emit a segment shorter than `dur`."""
    out = list(MaxObjNumEnumerator(
        [TrajectoryIntervalSeg(1, 0, 0, 12), TrajectoryIntervalSeg(2, 6, 0, 9)],
        {0: 1}, 5, (0, 60),
    ))
    assert any(e - s + 1 < 5 for _ids, s, e in out)


@pytest.mark.parametrize("segs_dur_minima_interval", list(corpora()), ids=range(60))
def test_no_segment_shorter_than_dur(segs_dur_minima_interval):
    segs, dur, minima, interval = segs_dur_minima_interval
    out = run(m.stride_merge_streaming, segs, dur=dur, minima=minima, interval=interval)
    assert all(s.end - s.begin >= dur for s in out)


@pytest.mark.parametrize("segs_dur_minima_interval", list(corpora()), ids=range(60))
def test_output_is_globally_sorted_by_begin(segs_dur_minima_interval):
    segs, dur, minima, interval = segs_dur_minima_interval
    out = run(m.stride_merge_streaming, segs, dur=dur, minima=minima, interval=interval)
    begins = [s.begin for s in out]
    assert begins == sorted(begins)


@pytest.mark.parametrize("segs_dur_minima_interval", list(corpora()), ids=range(60))
def test_pattern_set_matches_stride_merge(segs_dur_minima_interval):
    """Streaming must not change what max_obj reports."""
    segs, dur, minima, interval = segs_dur_minima_interval
    labels = {0: 1}
    ref = run(m.stride_merge, segs, dur=dur, minima=minima, interval=interval, labels=labels)
    out = run(m.stride_merge_streaming, segs, dur=dur, minima=minima, interval=interval,
              labels=labels)
    assert patterns(out, labels, dur, interval) == patterns(ref, labels, dur, interval)


# ---------------------------------------------------------------------------
# regressions caught in review
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,segs,dur,minima,interval",
    [
        # near-boundary segment must be carried, not dropped
        ("drop", [(1, 0, 0, 5), (2, 0, 12, 19)], 5, 10, (0, 100)),
        # interval ending one frame before the cut must not be carried across the gap
        ("phantom_merge", [(1, 0, 14, 19), (1, 0, 20, 30)], 5, 20, (0, 100)),
        # a 2-frame interval must not pass a dur=3 filter
        ("sub_dur", [(1, 0, 22, 31), (2, 0, 36, 38), (1, 0, 99, 104)], 3, 12, (0, 200)),
        # a carried interval must keep its true start and end
        ("inflated_carry",
         [(1, 0, 27, 53), (2, 0, 55, 82), (2, 0, 56, 68), (2, 0, 96, 115), (1, 0, 135, 164)],
         10, 60, (0, 400)),
    ],
)
def test_review_repros_match_stride_merge(name, segs, dur, minima, interval):
    ref = run(m.stride_merge, segs, dur=dur, minima=minima, interval=interval)
    out = run(m.stride_merge_streaming, segs, dur=dur, minima=minima, interval=interval)
    assert as_tuples(out) == as_tuples(ref)


def test_windowed_variant_also_never_emits_short_segments():
    """The windowed variant splits patterns, but must honour the same invariant."""
    for segs, dur, minima, interval in corpora(seed=7, count=40):
        out = run(m.stride_merge_windowed, segs, dur=dur, minima=minima, interval=interval)
        assert all(s.end - s.begin >= dur for s in out)
        begins = [s.begin for s in out]
        assert begins == sorted(begins)


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------


def test_default_expend_is_accepted():
    out = run(m.stride_merge_streaming, [(1, 0, 0, 20)], dur=10, minima=20, interval=(0, 100))
    assert as_tuples(out) == [(1, 0, 20)]


@pytest.mark.parametrize("expend,minima,dur", [(1, 20, 10), (0.5, 20, 10), (2, 0, 10), (2, 20, 0)])
def test_invalid_parameters_raise(expend, minima, dur):
    idx = make_index([(1, 0, 0, 20)])
    with pytest.raises(ValueError):
        list(m.stride_merge_streaming(idx, [], REGION, {0: 1}, dur, (0, 100),
                                      expend=expend, minima=minima))


def test_empty_labels_yields_nothing():
    assert run(m.stride_merge_streaming, [(1, 0, 0, 20)], dur=10, minima=20,
               interval=(0, 100), labels={}) == []
