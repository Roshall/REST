"""File-gated real-data regression for the index (C++) search family.

These tests only run when ``testdata_nosnappy/florida10h.parquet`` is present.

They confirm the production C++ index (``search_cxx`` via ``CxxIndex``) now
reports the **same** co-movement multiset as the reference oracle — including
the result object ids, which are real trajectory oids, not internal array
positions. This is the round-2 fix: the index previously emitted array
positions as ids (RC1) and its semantic output diverged from the oracle (RC4).

After round 1 + round 2 the contract is: the four index methods
(``max_dur_multi`` / ``max_obj_multi`` / ``max_dur_one`` / ``max_obj_one``),
the sliding family, and the reference oracle must all return the identical
multiset of ``(sorted_ids, start, end)``.

The full-slice sweep is additionally gated behind ``VDMS_FULL_SWEEP=1``.
"""

from __future__ import annotations

import json
import os
import tempfile

import pandas as pd
import pytest

from utilities.box2D import Box2D
from search_cxx.cxx_search import CxxIndex
from search.sliding_based.framework import sliding_framework
from test.mock_cases import clip
from test.reference_oracle import oracle_sorted

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "testdata_nosnappy", "florida10h.parquet")

LABELS = {2: 1}
METHODS = ("max_dur_multi", "max_obj_multi", "max_dur_one", "max_obj_one")


def _region_for(_df):
    # Cover every point so the index's spatial verification cannot affect the
    # comparison. The oracle/sliding contract assumes segments are pre-filtered
    # to a region covering all points; the index applies spatial verification
    # internally, so feeding it a region that encloses the whole dataset puts
    # it in the same (no spatial filtering) regime the sliding family is tested
    # in. Testing the partial-region divergence is out of scope here.
    return Box2D((0, 10**9, 0, 10**9))


def _norm(results):
    return tuple(sorted((tuple(sorted(ids)), s, e) for ids, s, e in results))


def _oracle_from_df(df, duration, lo, hi):
    """Reference oracle on the label-2 segments the index sees."""
    df2 = df[df["cls"] == 2]
    segs = []
    for oid, g in df2.groupby("oid"):
        b = int(g["fid"].min())
        e = int(g["fid"].max()) + 1
        segs.append((int(oid), 2, b, e))
    return tuple(oracle_sorted(clip(segs, (lo, hi)), LABELS, duration, (lo, hi)))


def _build_meta(parquet):
    """Generate the per-label border metadata the C++ index needs."""
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


def _index_methods(df, meta_path, duration, lo, hi):
    region = _region_for(df)
    idx = CxxIndex(parquet_path=PARQUET, meta_path=meta_path,
                   space_x=24, space_y=20, stride=1, scale=1)
    out = {}
    for m in METHODS:
        rows = list(idx.query(region.bbox, LABELS, duration, (lo, hi),
                              method=m, fps=1))
        out[m] = _norm(rows)
    return out


def test_realdata_window_index_eq_oracle():
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")

    lo, hi, dur = 586000, 586501, 30
    df = pd.read_parquet(PARQUET)
    df = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    if df.empty:
        pytest.skip("empty window")

    meta_path = _build_meta(PARQUET)
    try:
        oracle = _oracle_from_df(df, dur, lo, hi)
        results = _index_methods(df, meta_path, dur, lo, hi)

        # Both sliding methods, on the same data, for the cross-family check.
        region = _region_for(df)
        sliding_naive = _norm(
            sliding_framework(df, region, LABELS, (dur,), (lo, hi),
                              method="naive", fps=1))
        sliding_state = _norm(
            sliding_framework(df, region, LABELS, (dur,), (lo, hi),
                              method="state", fps=1))

        # All four index methods must agree with each other, with the sliding
        # family, and with the oracle — ids included. This is what round 2
        # repaired (the index must report real oids, not internal positions).
        assert len({results[m] for m in METHODS}) == 1, results
        for m in METHODS:
            assert results[m] == oracle, f"{m} diverged from oracle"
            assert results[m] == sliding_naive, f"{m} diverged from sliding naive"
            assert results[m] == sliding_state, f"{m} diverged from sliding state"
        assert oracle  # the window must actually exercise the machinery
    finally:
        os.remove(meta_path)


@pytest.mark.skipif(
    os.environ.get("VDMS_FULL_SWEEP") is None,
    reason="set VDMS_FULL_SWEEP=1 to load the full parquet and sweep all windows",
)
def test_realdata_full_slice_index_eq_oracle():
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")

    lo, hi, dur = 540001, 594001, 900  # 30s @ 30fps == 900 frames on the full slice
    df = pd.read_parquet(PARQUET)
    df = df[(df["fid"] >= lo) & (df["fid"] < hi)]
    meta_path = _build_meta(PARQUET)
    try:
        oracle = _oracle_from_df(df, dur, lo, hi)
        results = _index_methods(df, meta_path, dur, lo, hi)
        assert len({results[m] for m in METHODS}) == 1, results
        for m in METHODS:
            assert results[m] == oracle, f"{m} diverged from oracle"
        assert len(oracle) > 0
    finally:
        os.remove(meta_path)
