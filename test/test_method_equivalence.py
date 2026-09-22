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
from search.rest import coalesce, one_pass_merge, stride_merge, vanilla_merge
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
        tuple(query.interval), method=method,
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
    """Return ``{method_name: normalized_result}`` for all six methods."""
    segments = clip(segments, query.interval)
    rest_idx = make_index(segments)
    labels, dur, interval = query.labels, query.duration, query.interval
    minima = query.minima or 2 * dur

    names = ("max_dur_multi", "max_obj_multi", "max_dur_one", "max_obj_one",
             "sliding_naive", "sliding_state")
    if any(label not in rest_idx for label in labels):
        return {m: () for m in names}

    multi = vanilla_merge(rest_idx, TRAJS, region, labels, dur, interval)
    one_dur = stride_merge(rest_idx, TRAJS, region, labels, dur, interval,
                           expend=query.expend, minima=minima)
    one_obj = coalesce(
        one_pass_merge(rest_idx, TRAJS, region, labels, dur, interval), dur
    )

    return {
        "max_dur_multi": norm(max_dur_enumerate(multi, labels, dur, interval, mtd="multi")),
        "max_obj_multi": norm(MaxObjNumEnumerator(multi, labels, dur, interval)),
        "max_dur_one": norm(max_dur_enumerate(one_dur, labels, dur, interval, mtd="one")),
        "max_obj_one": norm(MaxObjNumEnumerator(one_obj, labels, dur, interval)),
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
    (("max", "dur", "multi"), ("max", "obj", "multi"),
     ("max", "dur", "one"), ("max", "obj", "one")),
    ids=lambda m: "_".join(m),
)
def test_framework_dispatch_matches(mock, method):
    """``index_based_framework`` must agree with the explicit pipeline.

    Window-sensitive mocks are skipped because the framework calls
    ``stride_merge`` with its production defaults, which never window a mock.
    """
    if mock.query.minima:
        pytest.skip("framework uses production stride defaults")
    segs = clip(mock.segments, mock.query.interval)
    rest_idx = make_index(segs)
    pack = (rest_idx, TRAJS)
    if any(label not in rest_idx for label in mock.query.labels):
        got = ()
    else:
        got = norm(index_based_framework(
            pack, REGION, mock.query.labels, (mock.query.duration,),
            list(mock.query.interval), method=method,
        ))
    name = {"dur": "max_dur", "obj": "max_obj"}[method[1]]
    expected = run_all_methods(mock.segments, mock.query)[f"{name}_{method[2]}"]
    assert got == expected
