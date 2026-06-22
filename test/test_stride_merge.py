from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from search.rest import stride_merge as m


@dataclass
class RawSeg:
    id: int
    label: int
    begin: int
    end: int  # half-open end


@dataclass
class FakeTrajectory:
    id: int
    label: int
    seg: list[int]


@dataclass
class FakeTrajectoryIntervalSeg:
    id: int
    begin: int
    label: int
    end: int  # inclusive end


class FakeLabelIndex:
    def __init__(self, *, candidates=(), probation=()):
        self.candidates = list(candidates)
        self.probation = list(probation)
        self.calls = []

    def query(self, trajs, bbox, dur, interval, flag):
        self.calls.append(
            {
                "trajs": trajs,
                "bbox": bbox,
                "dur": dur,
                "interval": interval,
                "flag": flag,
            }
        )
        return self.candidates, self.probation


@pytest.fixture
def region():
    return SimpleNamespace(bbox=(0, 0, 100, 100))


@pytest.fixture(autouse=True)
def patch_stride_merge_dependencies(monkeypatch):
    """
    Keep these tests focused on stride_merge.

    If your real Trajectory / TrajectoryIntervalSeg constructors have the same
    signature, this monkeypatch is harmless. If not, it makes the tests stable.
    """
    monkeypatch.setattr(m, "Trajectory", FakeTrajectory)
    monkeypatch.setattr(m, "TrajectoryIntervalSeg", FakeTrajectoryIntervalSeg)

    def fake_candidate_verified_queue(candidates, region, dur):
        yield from candidates

    monkeypatch.setattr(m, "candidate_verified_queue", fake_candidate_verified_queue)


def run_stride(rest_idx, *, labels=None, dur=10, interval=(0, 10_000), region=None):
    if labels is None:
        labels = {label: 1 for label in rest_idx}

    if region is None:
        region = SimpleNamespace(bbox=(0, 0, 100, 100))

    return list(
        m.stride_merge(
            rest_idx=rest_idx,
            trajs=[],
            region=region,
            labels=labels,
            dur=dur,
            interval=interval,
            expend=2,
            minima=20,  # stride = max(2 * dur, 20)
        )
    )


def as_tuple(seg):
    return seg.id, seg.begin, seg.end, seg.label


def as_tuples(segs):
    return [as_tuple(seg) for seg in segs]


def assert_no_window_sentinel(output):
    assert all(item is not None for item in output)
    assert all(item != (None, None) for item in output)


def test_empty_labels_yields_nothing(region):
    rest_idx = {
        0: FakeLabelIndex(
            probation=[RawSeg(1, 0, 0, 20)],
        )
    }

    out = list(
        m.stride_merge(
            rest_idx=rest_idx,
            trajs=[],
            region=region,
            labels={},
            dur=10,
            interval=(0, 100),
            expend=2,
            minima=20,
        )
    )

    assert out == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dur": 0, "expend": 2, "minima": 20},
        {"dur": -1, "expend": 2, "minima": 20},
        {"dur": 10, "expend": 1, "minima": 20},
        {"dur": 10, "expend": 0.5, "minima": 20},
        {"dur": 10, "expend": 2, "minima": 0},
        {"dur": 10, "expend": 2, "minima": -1},
    ],
)
def test_invalid_parameters_raise(region, kwargs):
    rest_idx = {0: FakeLabelIndex(probation=[RawSeg(1, 0, 0, 20)])}

    with pytest.raises(ValueError):
        list(
            m.stride_merge(
                rest_idx=rest_idx,
                trajs=[],
                region=region,
                labels={0: 1},
                interval=(0, 100),
                **kwargs,
            )
        )


def test_queries_each_label_with_expected_arguments(region):
    rest_idx = {
        0: FakeLabelIndex(probation=[RawSeg(1, 0, 0, 10)]),
        1: FakeLabelIndex(probation=[RawSeg(2, 1, 10, 20)]),
    }

    list(
        m.stride_merge(
            rest_idx=rest_idx,
            trajs=["existing"],
            region=region,
            labels={0: 1, 1: 1},
            dur=10,
            interval=(5, 105),
            expend=2,
            minima=20,
        )
    )

    for label in (0, 1):
        assert len(rest_idx[label].calls) == 1
        call = rest_idx[label].calls[0]
        assert call["trajs"] == ["existing"]
        assert call["bbox"] == region.bbox
        assert call["dur"] == 10
        assert call["interval"] == (5, 105)
        assert call["flag"] == 1


def test_single_duration_valid_segment_is_emitted():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[RawSeg(1, 0, 0, 10)],
        )
    }

    out = run_stride(rest_idx)

    assert_no_window_sentinel(out)
    assert as_tuples(out) == [
        (1, 0, 9, 0),
    ]


def test_single_short_final_segment_is_discarded():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[RawSeg(1, 0, 0, 9)],
        )
    }

    out = run_stride(rest_idx)

    assert out == []


def test_touching_segments_inside_same_window_are_merged():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 5),
                RawSeg(1, 0, 5, 12),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 11, 0),
    ]


def test_gapped_segments_inside_same_window_are_not_merged():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 5),
                RawSeg(1, 0, 6, 20),
            ],
        )
    }

    out = run_stride(rest_idx)

    # [0, 5) is too short.
    # [6, 20) has length 14.
    assert as_tuples(out) == [
        (1, 6, 19, 0),
    ]


def test_short_boundary_touching_interval_is_carried_and_becomes_valid():
    """
    dur = 10, stride = 20, first window is [0, 20).

    [15, 20) alone is too short, but it touches the boundary.
    It must be carried and merged with [20, 25).
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 15, 20),
                RawSeg(1, 0, 20, 25),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 15, 24, 0),
    ]


def test_exact_boundary_end_must_be_carried_not_closed():
    """
    This catches the common bug `end > window_end`.

    Correct condition is `end >= window_end`.

    [10, 20) touches the boundary exactly, then [20, 25) continues it.
    The correct result is [10, 25), not [10, 20) plus dropping [20, 25).
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 10, 20),
                RawSeg(1, 0, 20, 25),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 10, 24, 0),
    ]


def test_interval_starting_near_boundary_but_ending_before_boundary_is_not_carried():
    """
    This catches the unsafe rule:

        keep if begin >= window_end - dur

    [11, 19) starts in the last dur region of [0, 20), but it does not touch
    the boundary. It cannot merge with future segments starting at 20.
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 11, 19),
                RawSeg(1, 0, 20, 30),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 20, 29, 0),
    ]


def test_closed_short_interval_before_boundary_is_discarded():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 5),
                RawSeg(2, 0, 20, 30),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (2, 20, 29, 0),
    ]


def test_cross_boundary_interval_emits_safe_prefix_and_carries_suffix():
    """
    dur = 10, stride = 20, first window is [0, 20).

    [0, 20) reaches the boundary.
    We emit the safe prefix [0, 10), carry [10, 20),
    then merge the carried suffix with [20, 30).

    Expected:
        [0, 10)  -> inclusive [0, 9]
        [10, 30) -> inclusive [10, 29]
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 20),
                RawSeg(1, 0, 20, 30),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 9, 0),
        (1, 10, 29, 0),
    ]


def test_long_cross_boundary_interval_is_truncated_to_last_duration_for_carry():
    """
    This checks memory-bounding behavior.

    [0, 25) crosses the boundary at 20.
    With dur=10, only [10, 25) should be carried.
    Safe prefix [0, 10) is emitted.

    Then [25, 35) merges with the carried suffix.
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 25),
                RawSeg(1, 0, 25, 35),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 9, 0),
        (1, 10, 34, 0),
    ]


def test_carried_short_suffix_discarded_at_final_if_still_too_short():
    """
    [15, 20) touches the boundary and is carried.
    There is no future segment, so final flush should discard it.
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 15, 20),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert out == []


def test_candidate_verified_stream_is_used():
    rest_idx = {
        0: FakeLabelIndex(
            candidates=[
                RawSeg(1, 0, 0, 10),
            ],
            probation=[],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 9, 0),
    ]


def test_probation_and_candidate_streams_are_globally_merged_by_begin():
    """
    This catches the old chain(...) behavior.

    probation has [20, 30)
    candidates have [0, 20)

    If chain is used, [20, 30) is seen first and the partial-window behavior
    can incorrectly produce [0, 30).

    With heapq.merge by begin:
        [0, 20) is processed first,
        window boundary at 20 splits it into:
            [0, 10) emitted
            [10, 20) carried
        then [20, 30) merges with the carried suffix.
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 20, 30),
            ],
            candidates=[
                RawSeg(1, 0, 0, 20),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 9, 0),
        (1, 10, 29, 0),
    ]


def test_multiple_labels_are_merged_into_one_global_time_order():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 20, 30),
            ],
        ),
        1: FakeLabelIndex(
            probation=[
                RawSeg(2, 1, 0, 10),
            ],
        ),
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (2, 0, 9, 1),
        (1, 20, 29, 0),
    ]


def test_large_empty_time_gap_skips_empty_windows():
    """
    The implementation should not need to flush repeatedly for empty windows.
    It can adaptively move window_start/window_end to the next segment begin.
    """
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 10),
                RawSeg(2, 0, 100, 110),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert as_tuples(out) == [
        (1, 0, 9, 0),
        (2, 100, 109, 0),
    ]


def test_outputs_are_sorted_by_begin_after_each_flush():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 20),
                RawSeg(2, 0, 20, 35),
                RawSeg(1, 0, 20, 30),
            ],
        )
    }

    out = run_stride(rest_idx)

    begins = [seg.begin for seg in out]
    assert begins == sorted(begins)


def test_no_none_none_sentinel_is_yielded():
    rest_idx = {
        0: FakeLabelIndex(
            probation=[
                RawSeg(1, 0, 0, 20),
                RawSeg(1, 0, 20, 30),
                RawSeg(2, 0, 50, 60),
            ],
        )
    }

    out = run_stride(rest_idx)

    assert_no_window_sentinel(out)