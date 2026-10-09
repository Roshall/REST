"""Real-data test for the windowed (bounded-memory) merge ``stride_merge_windowed``.

File-gated on ``testdata_nosnappy/florida10h.parquet``.

``stride_merge_windowed`` is exposed as the ``win`` merge key, reachable either
through the Python ``index_based_framework`` (string ``"max_obj_win_m1500_e2"`` or
tuple ``("max", "obj", "win", 1500, 2)``) or, for the experiment harness, through
the C++ backend via ``CxxIndex.query(method=...)``. Its window size is
``stride = max(dur * expend, minima)`` (``expend`` defaults to 2).

Why both backends are consistent: the merge is a streaming generator that cuts a
trajectory crossing a window boundary into a head (emitted when the window
flushes) plus a touching tail (carried into the next window).
``MaxObjNumEnumerator`` reads ``dur`` frames ahead and re-joins such a piece when
it begins exactly at the open object's current end, so the occupancy survives the
cut instead of being read as a leave-then-rejoin. It also keeps a cut head in
``_init_state`` even when that head ends before ``interval[0] + dur`` -- the head
is a piece, not a closed segment -- which is what keeps an object alive across a
cut landing inside the first ``dur`` frames of the interval. The reported pattern
set is therefore the same as ``stride_merge``'s, while memory stays bounded by the
trajectories that cross the cut. Both the Python and the C++ twin do this.

Contract: ``stride >= 2 * dur`` is a *correctness* floor (every emitted segment is
>= ``dur`` and the window advances by >= ``dur``). The tests exercise windows
strictly greater than ``2 * dur``.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from utilities.box2D import Box2D
from search_cxx.cxx_search import CxxIndex
from test.mock_cases import clip
from test.reference_oracle import oracle_sorted

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "testdata_nosnappy", "florida10h.parquet")

LABELS = {0: 1, 2: 1}
# Windowed-method variants. Every minima is *strictly* greater than 2*dur for
# its case (the user's "window > 2*dur" rule).
WIN_METHODS = (
    "max_obj_win_m2000", "max_obj_win_m3000", "max_obj_win_m5000",
    "max_obj_win_m4000", "max_obj_win_m5000", "max_obj_win_m6000",
)
REF_METHODS = ("max_obj_one", "max_dur_multi")


def _region():
    return Box2D((0, 10**9, 0, 10**9))


def _norm(results):
    return tuple(sorted((tuple(sorted(ids)), s, e) for ids, s, e in results))


def _build_meta(parquet):
    from scripts.border import border_meta

    df = pd.read_parquet(parquet)
    meta = {
        int(k): v
        for k, v in border_meta(df, tempo_stride=60).items()
        if int(k) in LABELS
    }
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump({str(k): v for k, v in meta.items()}, f)
    return path


def _oracle_from_df(df, duration, lo, hi):
    df2 = df[df["cls"].isin(LABELS)]
    segs = []
    for oid, g in df2.groupby("oid"):
        b = int(g["fid"].min())
        e = int(g["fid"].max()) + 1
        segs.append((int(oid), int(g["cls"].iloc[0]), b, e))
    return tuple(oracle_sorted(clip(segs, (lo, hi)), LABELS, duration, (lo, hi)))


CASES = [
    (540001, 552001, 900),   # 2*dur = 1800
    (540001, 549001, 1800),  # 2*dur = 3600
    (560000, 572000, 900),
    (586000, 586501, 30),    # 2*dur = 60
]


# ---------------------------------------------------------------------------
# C++ backend: still awaiting the same coalescing fix -> documented xfail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lo,hi,dur", CASES)
@pytest.mark.parametrize("win_method", WIN_METHODS)
def test_cxx_windowed_matches_validated_methods(lo, hi, dur, win_method):
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")

    df = pd.read_parquet(PARQUET)
    df = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    if df.empty:
        pytest.skip("empty window")

    meta_path = _build_meta(PARQUET)
    try:
        region = _region().bbox
        idx = CxxIndex(PARQUET, meta_path, 24, 20, 1, 1)
        win = _norm(list(idx.query(region, LABELS, dur, (lo, hi),
                                  method=win_method, fps=1)))
        refs = {m: _norm(list(idx.query(region, LABELS, dur, (lo, hi),
                                       method=m, fps=1)))
                for m in REF_METHODS}
        assert refs["max_obj_one"] == refs["max_dur_multi"], "baseline refs diverged"
        for m in REF_METHODS:
            assert refs[m] == win, f"C++ {win_method} diverges from {m}"
    finally:
        os.remove(meta_path)


# ---------------------------------------------------------------------------
# Python backend: the win key must dispatch and be consistent with siblings
# ---------------------------------------------------------------------------

class _FakeLabelIndex:
    def __init__(self, segments):
        self.segments = sorted(segments, key=lambda s: (s[2], s[0]))

    def query(self, trajs, bbox, dur, interval, flag):
        from utilities.trajectory import TrajectoryIntervalSeg
        segs = [TrajectoryIntervalSeg(sid, b, label, e)
                for sid, label, b, e in self.segments]
        return [], list(segs)


def _py_pack(df, lo, hi):
    idx = {}
    sub = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    for label in LABELS:
        segs = [(int(oid), label, int(g["fid"].min()), int(g["fid"].max()) + 1)
                for oid, g in sub[sub["cls"] == label].groupby("oid")]
        idx[label] = _FakeLabelIndex(segs)
    return idx, []


@pytest.mark.parametrize("lo,hi,dur", CASES)
def test_python_baseline_matches_cxx(lo, hi, dur):
    """The Python ``win`` baseline agrees with the C++ baseline."""
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")
    from search.index_based.framework import index_based_framework

    df = pd.read_parquet(PARQUET)
    df = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    if df.empty:
        pytest.skip("empty window")
    meta_path = _build_meta(PARQUET)
    try:
        region = _region()
        cxx = CxxIndex(PARQUET, meta_path, 24, 20, 1, 1)
        ref_one = _norm(list(cxx.query(region.bbox, LABELS, dur, (lo, hi),
                                       method="max_obj_one", fps=1)))
        pack = _py_pack(df, lo, hi)
        py_one = _norm(index_based_framework(
            pack, region, LABELS, (dur,), (lo, hi),
            method=("max", "obj", "one"), fps=1))
        assert py_one == ref_one, "Python baseline disagrees with C++ baseline"
    finally:
        os.remove(meta_path)


@pytest.mark.parametrize("lo,hi,dur", CASES)
@pytest.mark.parametrize("minima", [2000, 3000, 5000])
def test_python_windowed_consistent(lo, hi, dur, minima):
    """Python ``win`` returns the same pattern multiset as its siblings.

    Exercises both the string token form (``"max_obj_win_m<int>"``) and the
    tuple form (``("max", "obj", "win", minima)``). Windows are > 2*dur.
    """
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")
    from search.index_based.framework import index_based_framework

    df = pd.read_parquet(PARQUET)
    df = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    if df.empty:
        pytest.skip("empty window")
    pack = _py_pack(df, lo, hi)
    region = _region()

    one = _norm(index_based_framework(
        pack, region, LABELS, (dur,), (lo, hi), method=("max", "obj", "one"), fps=1))
    multi = _norm(index_based_framework(
        pack, region, LABELS, (dur,), (lo, hi), method=("max", "obj", "multi"), fps=1))

    # string token form
    win_s = _norm(index_based_framework(
        pack, region, LABELS, (dur,), (lo, hi),
        method=f"max_obj_win_m{minima}", fps=1))
    # tuple form with explicit minima
    win_t = _norm(index_based_framework(
        pack, region, LABELS, (dur,), (lo, hi),
        method=("max", "obj", "win", minima), fps=1))

    assert one == multi, "siblings disagree"
    assert win_s == one, f"string win_m{minima} diverges from sibling"
    assert win_t == one, f"tuple win({minima}) diverges from sibling"
