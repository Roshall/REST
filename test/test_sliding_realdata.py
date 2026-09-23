"""File-gated real-data regression for the sliding family.

These tests only run when ``testdata_nosnappy/florida10h.parquet`` is present.
They confirm the fixed sliding ``state`` method agrees with the verified-correct
``naive`` method, and that both agree with the reference oracle, on dense,
long-running real data — exactly the conditions under which the two round-1
defects (terminal flush dropping a tail pattern, and the "new object set" branch
dropping a chain start) manifested.

The full-slice sweep is additionally gated behind ``VDMS_FULL_SWEEP=1`` so the
nightly/CI path can verify the sliding family agrees with the reference oracle
across the whole 30-minute slice without paying for a full parquet load on every
run.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest

from utilities.box2D import Box2D
from search.sliding_based.framework import sliding_framework

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "..", "testdata_nosnappy", "florida10h.parquet")


def _load_window(lo, hi):
    df = pd.read_parquet(PARQUET)
    return df[(df["fid"] >= lo) & (df["fid"] < hi)].reset_index(drop=True)


def _region_for(df):
    # Cover every point so spatial filtering cannot affect the comparison.
    xmin, xmax = int(df["x"].min()), int(df["x"].max())
    ymin, ymax = int(df["y"].min()), int(df["y"].max())
    return Box2D((xmin, xmax, ymin, ymax))


def _norm(results):
    return tuple(sorted((tuple(sorted(ids)), s, e) for ids, s, e in results))


def _run(df, method, duration, lo, hi):
    region = _region_for(df)
    return _norm(
        sliding_framework(
            df, region, {2: 1}, (duration,), (lo, hi), method=method, fps=1
        )
    )


def _oracle_from_df(df, duration, lo, hi):
    """Reference oracle result on the same label-filtered segments the sliding
    ``dfilter`` sees (mirrors how the index feeds segments to the enumerator)."""
    from test.mock_cases import clip
    from test.reference_oracle import oracle_sorted

    df2 = df[df["cls"] == 2]
    segs = []
    for oid, g in df2.groupby("oid"):
        label = int(g["cls"].iloc[0])
        b = int(g["fid"].min())
        e = int(g["fid"].max()) + 1
        segs.append((int(oid), label, b, e))
    return tuple(oracle_sorted(clip(segs, (lo, hi)), {2: 1}, duration, (lo, hi)))


def test_realdata_window_naive_eq_state_eq_oracle():
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")

    lo, hi = 586000, 586501
    df = _load_window(lo, hi)
    naive = _run(df, "naive", 30, lo, hi)
    state = _run(df, "state", 30, lo, hi)
    oracle = _oracle_from_df(df, 30, lo, hi)

    # Core invariant restored by round 1: the two sliding methods agree, and
    # both agree with the reference oracle.
    assert naive == state == oracle
    assert naive  # the window must actually exercise the machinery


@pytest.mark.skipif(
    os.environ.get("VDMS_FULL_SWEEP") is None,
    reason="set VDMS_FULL_SWEEP=1 to load the full parquet and sweep all windows",
)
def test_realdata_full_slice_naive_eq_state_eq_oracle():
    if not os.path.exists(PARQUET):
        pytest.skip(f"{PARQUET} not found")

    lo, hi = 540001, 594001
    df = _load_window(lo, hi)
    # 30 seconds at 30 fps == 900 frames.
    naive = _run(df, "naive", 900, lo, hi)
    state = _run(df, "state", 900, lo, hi)
    oracle = _oracle_from_df(df, 900, lo, hi)

    # Round-1 invariant: the two sliding methods agree, and both agree with the
    # reference oracle on the full slice.
    assert naive == state == oracle
    assert len(naive) > 0
