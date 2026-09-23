"""The query unit is seconds; frames are the internal unit.

``seconds_to_frames`` converts once, at the framework boundary, using
``cfg.DATA.FPS``. These tests pin the conversion and the identity that makes
``fps=1`` the frame-level escape hatch the other test modules rely on.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from search.index_based.framework import index_based_framework
from search.sliding_based.framework import sliding_framework
from test.test_method_equivalence import make_index
from utilities.box2D import Box2D
from utilities.time_unit import interval_to_frames, seconds_to_frames

REGION = SimpleNamespace(bbox=(0, 0, 10**9, 10**9))
BBOX = (0, 100, 0, 100)

# A 60-frame interval: [0, 60) at 30 fps is 2 seconds.
N_FRAMES = 60
SEGMENTS = [(1, 0, 0, N_FRAMES), (2, 0, 0, N_FRAMES)]


def test_seconds_to_frames():
    assert seconds_to_frames(2, fps=30) == 60
    assert seconds_to_frames(0.5, fps=30) == 15
    assert seconds_to_frames(3, fps=1) == 3


def test_seconds_to_frames_clamps_duration():
    # absorb()/NaiveSliding require a positive window, so sub-frame durations
    # must not collapse to 0.
    assert seconds_to_frames(0, fps=30, minimum=1) == 1
    assert seconds_to_frames(0.01, fps=30, minimum=1) == 1


def test_interval_to_frames():
    assert interval_to_frames((0, 2), fps=30) == [0, 60]
    assert interval_to_frames((0, 2), fps=1) == [0, 2]
    with pytest.raises(ValueError):
        interval_to_frames((0, 0.01), fps=30)


def _frames():
    rows = [(oid, fid, 50, 50, 0)
            for oid in (1, 2)
            for fid in range(N_FRAMES)]
    return pd.DataFrame(rows, columns=["oid", "fid", "x", "y", "cls"])


def _norm(results):
    return tuple(sorted((tuple(sorted(ids)), s, e) for ids, s, e in results))


def run_sliding(duration, interval, fps):
    return _norm(sliding_framework(
        _frames(), Box2D(BBOX), {0: 1}, (duration,), interval, fps=fps))


def run_index(duration, interval, fps):
    pack = (make_index(SEGMENTS), [])
    return _norm(index_based_framework(
        pack, REGION, {0: 1}, (duration,), interval, fps=fps))


@pytest.mark.parametrize("run", [run_sliding, run_index], ids=["sliding", "index"])
def test_seconds_match_frames(run):
    """1 second at 30 fps must mean exactly 30 frames."""
    seconds = run(1, (0, 2), 30)
    frames = run(30, (0, N_FRAMES), 1)
    assert seconds == frames
    assert seconds, "the co-moving pair must be found, otherwise this is vacuous"
