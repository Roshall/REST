"""Cross-method equivalence tests for the index-based search family.

Every method must return the canonical pattern multiset, which is supplied by
two independent oracles:

  * the hand-derived expectations in ``test.mock_cases``, and
  * the declarative reference implementation in ``test.reference_oracle``.

The four index pipelines mirror ``search.index_based.framework``:

  merge ``multi`` -> ``vanilla_merge``            enumerator ``max_dur``, ``max_obj``
  merge ``one``   -> ``stride_merge``             enumerator ``max_dur``
  merge ``one``   -> ``one_pass_merge``+coalesce  enumerator ``max_obj``
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from search.index_based.framework import index_based_framework
from search.index_based.max_dur import max_dur_enumerate
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import (
    coalesce,
    one_pass_merge,
    stride_merge,
    stride_merge_streaming,
    stride_merge_windowed,
    vanilla_merge,
)
from search.sliding_based.framework import sliding_framework
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg
from test.mock_cases import MOCKS, clip
from test.reference_oracle import oracle_sorted


REGION = SimpleNamespace(bbox=(0, 0, 10**9, 10**9))
TRAJS = []  # the fake index ignores this argument

# Frame-level queries use a real Box2D so the sliding path exercises the same
# region predicate as production. Coordinates are chosen well inside it.
SLIDING_BBOX = (0, 100, 0, 100)
SLIDING_REGION = Box2D(SLIDING_BBOX)


def frames_from_segments(segments, interval=None):
    """Expand half-open segments into one frame row per (object, frame)."""
    rows = []
    for sid, label, beg, end in segments:
        for fid in range(beg, end):
            rows.append((sid, fid, 50, 50, label))
    return pd.DataFrame(rows, columns=["oid", "fid", "x", "y", "cls"])


def run_sliding(segments, query, method):
    segs = clip(segments, query.interval)
    df = frames_from_segments(segs)
    if df.empty:
        return ()
    return norm(sliding_framework(
        df, SLIDING_REGION, query.labels, (query.duration,),
        tuple(query.interval), method=method, fps=1,
    ))


class FakeLabelIndex:
    """Returns the mock segments as the "probation" (sure) stream."""

    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: (s[2], s[0]))

    def query(self, trajs, bbox, dur, interval, flag):
        segs = [TrajectoryIntervalSeg(sid, b, label, e) for sid, label, b, e in self.segments]
        return [], list(segs)


def make_index(segments):
    """Index keyed by the labels that actually occur in the segments."""
    labels = {label for _, label, _, _ in segments}
    return {label: FakeLabelIndex([s for s in segments if s[1] == label]) for label in labels}


def norm(results):
    return tuple(sorted((tuple(sorted(ids)), s, e) for ids, s, e in results))


def run_all_methods(segments, query, region=REGION):
    """Return ``{method_name: normalized_result}`` for all six methods.

    The pipelines mirror the framework dispatch under the stored/streaming
    groups: ``stored`` is vanilla_merge for both enumerators, and ``streaming``
    is stride_merge_streaming for max_dur (lazily coalesced, one segment per
    object run, so the multi-pass enumerator consumes it unchanged) and
    stride_merge_windowed for max_obj (whose enumerator re-joins the
    window-cut pieces).
    """
    segments = clip(segments, query.interval)
    rest_idx = make_index(segments)
    labels, dur, interval = query.labels, query.duration, query.interval
    minima = query.minima or 2 * dur
    if any(label not in rest_idx for label in labels):
        return {m: () for m in ("max_dur_stored", "max_obj_stored",
                                "max_dur_streaming", "max_obj_streaming",
                                "sliding_naive", "sliding_state")}

    stored = vanilla_merge(rest_idx, TRAJS, region, labels, dur, interval)
    stream_dur = list(stride_merge_streaming(rest_idx, TRAJS, region, labels, dur,
                                             interval, expend=query.expend,
                                             minima=minima))
    stream_obj = list(stride_merge_windowed(rest_idx, TRAJS, region, labels, dur,
                                            interval, expend=query.expend,
                                            minima=minima))
    if any(label not in rest_idx for label in labels):
        return {m: () for m in names}

    return {
        "max_dur_stored": norm(max_dur_enumerate(stored, labels, dur, interval, mtd="vanilla")),
        "max_obj_stored": norm(MaxObjNumEnumerator(stored, labels, dur, interval)),
        "max_dur_streaming": norm(max_dur_enumerate(stream_dur, labels, dur, interval, mtd="stride_stream")),
        "max_obj_streaming": norm(MaxObjNumEnumerator(stream_obj, labels, dur, interval)),
        "sliding_naive": run_sliding(segments, query, "naive"),
        "sliding_state": run_sliding(segments, query, "state"),
    }


# ---------------------------------------------------------------------------
# oracle validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mock", MOCKS, ids=lambda m: m.name)
def test_hand_expected_matches_oracle(mock):
    """The hand-derived expectations and the reference oracle agree."""
    segs = clip(mock.segments, mock.query.interval)
    got = oracle_sorted(segs, mock.query.labels, mock.query.duration, mock.query.interval)
    assert got == tuple(sorted(mock.expected))


# ---------------------------------------------------------------------------
# merge-strategy equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mock", MOCKS, ids=lambda m: m.name)
def test_merge_strategies_agree(mock):
    """All three merge strategies return the same segment multiset."""
    segs = clip(mock.segments, mock.query.interval)
    q = mock.query
    rest_idx = make_index(segs)
    if any(label not in rest_idx for label in q.labels):
        return
    minima = q.minima or 2 * q.duration

    multi = vanilla_merge(rest_idx, TRAJS, REGION, q.labels, q.duration, q.interval)
    one_dur = stride_merge(rest_idx, TRAJS, REGION, q.labels, q.duration, q.interval,
                           expend=q.expend, minima=minima)
    one_obj = coalesce(
        one_pass_merge(rest_idx, TRAJS, REGION, q.labels, q.duration, q.interval),
        q.duration,
    )

    def seg_multiset(segs):
        return sorted((s.id, s.label, s.begin, s.end) for s in segs)

    assert seg_multiset(multi) == seg_multiset(one_dur), "stride_merge != vanilla_merge"
    assert seg_multiset(multi) == seg_multiset(one_obj), "one_pass_merge+coalesce != vanilla_merge"


# ---------------------------------------------------------------------------
# full equivalence
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mock", MOCKS, ids=lambda m: m.name)
def test_all_methods_match_ground_truth(mock):
    results = run_all_methods(mock.segments, mock.query)
    expected = tuple(sorted(mock.expected))
    for name, got in results.items():
        assert got == expected, f"{mock.name}: {name} -> {got}"


@pytest.mark.parametrize("mock", MOCKS, ids=lambda m: m.name)
def test_all_methods_agree(mock):
    results = run_all_methods(mock.segments, mock.query)
    values = set(results.values())
    assert len(values) == 1, f"{mock.name}: methods disagree: {results}"


# ---------------------------------------------------------------------------
# framework dispatch agrees with the explicit pipelines
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mock", MOCKS, ids=lambda m: m.name)
@pytest.mark.parametrize(
    "method",
    (("max", "dur", "stored"), ("max", "obj", "stored"),
     ("max", "dur", "streaming"), ("max", "obj", "streaming")),
    ids=lambda m: "_".join(m),
)
def test_framework_dispatch_matches(mock, method):
    """``index_based_framework`` must agree with the explicit pipeline.

    Window-sensitive mocks are skipped because the framework calls the streaming
    merges with their production window defaults, which never window a mock.
    """
    if mock.query.minima:
        pytest.skip("framework uses production window defaults")
    segs = clip(mock.segments, mock.query.interval)
    rest_idx = make_index(segs)
    pack = (rest_idx, TRAJS)
    if any(label not in rest_idx for label in mock.query.labels):
        got = ()
    else:
        got = norm(index_based_framework(
            pack, REGION, mock.query.labels, (mock.query.duration,),
            list(mock.query.interval), method=method, fps=1,
        ))
    sub = method[2]
    name = {"dur": "max_dur", "obj": "max_obj"}[method[1]] + "_" + sub
    expected = run_all_methods(mock.segments, mock.query)[name]
    assert got == expected


# ---------------------------------------------------------------------------
# the streaming merges must hand the enumerators what they expect
# ---------------------------------------------------------------------------

def test_max_dur_one_unions_the_raw_merge_pieces():
    """``max_dur`` + streaming must cope with ``one_pass_merge``'s per-cell pieces.

    ``one_pass_merge`` emits one piece per index cell an object occupies, so a
    single object can arrive as several pieces that touch, overlap or nest.
    ``MaxDurFirstEnumerator`` keeps one live segment per object id: an
    overlapping second piece replaces the live entry while the first end event
    is still queued, that event erases the id, and the second one then looks up
    an id that is gone (``KeyError``; with assertions on, the re-insert trips
    ``assert tra_id not in self.playground`` first). The enumerator therefore
    unions the pieces into the live occupancy and drops the stale end events,
    which is the streaming equivalent of ``search.rest.coalesce()`` (that helper
    materializes, so it is the reference the C++ tests compare against, not
    something a streaming consumer can call).
    """
    # One object that arrives as two overlapping pieces, i.e. it crossed a grid
    # cell boundary. Coalesced it is a single occupancy [0, 15).
    segments = [(1, 0, 0, 10), (1, 0, 5, 15)]

    # The pieces are the same object, so a query for two objects of the class
    # must find nothing. Counting the pieces instead is what the bug did.
    for method in (("max", "dur", "one"), ("max", "obj", "one"),
                   ("max", "dur", "multi"), ("max", "obj", "multi")):
        pack = (make_index(segments), TRAJS)
        got = norm(index_based_framework(
            pack, REGION, {0: 2}, (5,), [0, 25], method=method, fps=1,
        ))
        assert got == (), f"{method} counted one object twice: {got}"

    # And a query for one object reports that occupancy once.
    for method in (("max", "dur", "one"), ("max", "obj", "one")):
        pack = (make_index(segments), TRAJS)
        got = norm(index_based_framework(
            pack, REGION, {0: 1}, (5,), [0, 25], method=method, fps=1,
        ))
        assert got == (((1,), 0, 14),), f"{method} -> {got}"


def test_max_obj_unions_raw_merge_pieces():
    """``max_obj`` must not count one object's per-cell pieces as separate objects.

    ``one_pass_merge`` emits one piece per index cell an object occupies, so a
    query for two objects of a class must not be satisfied by a single object
    that happens to be split. ``MaxObjNumEnumerator`` unions the pieces into the
    presence it already has, which is what keeps the object set honest.
    """
    # One object arriving as three touching pieces, plus a real second object.
    segments = [(1, 0, 0, 10), (1, 0, 10, 20), (1, 0, 20, 30), (2, 0, 5, 40)]

    for method in (("max", "obj", "one_pass"), ("max", "obj", "stored")):
        pack = (make_index(segments), TRAJS)
        two = norm(index_based_framework(
            pack, REGION, {0: 2}, (5,), [0, 60], method=method, fps=1,
        ))
        # Exactly the two real objects, and nothing split into phantom ones.
        # (The window itself differs between the merges: one_pass carries
        # sub-duration pieces on purpose, and max_obj's duration contract relies
        # on the merge having dropped them — see kKnownDivergences.)
        assert [ids for ids, _, _ in two] == [(1, 2)], f"{method} -> {two}"

    # One object alone must never satisfy a request for two.
    pack = (make_index(segments[:3]), TRAJS)
    got = norm(index_based_framework(
        pack, REGION, {0, 2}, (5,), [0, 60], method=("max", "obj", "one_pass"), fps=1,
    ))
    assert got == (), f"one object counted twice: {got}"
