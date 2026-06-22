import heapq
from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from itertools import batched, chain, islice, takewhile
from operator import attrgetter

from utilities import TrajectoryIntervalSeg

from search.co_moving import CoMovementPattern
from search.verifier import candidate_verified_queue
from utilities.trajectory import BasicTrajectorySeg, Trajectory


def yield_co_move(
    duration: int,
    labels: Mapping[int, int],
    active_space: MutableMapping[int, BasicTrajectorySeg],
    timestamp: int,
    traj_required: Sequence,
) -> Iterable[tuple[list[int], int, int]]:
    """
    a co-movement checker respecting objects' label and count and their co-moving duration .
    Note that this function may modify `active_space`.
    :param duration: objects co-moving duration
    :param labels: {obj_label: count}
    :param active_space: {obj_id: trajectory}
    :param timestamp: current processing time
    :param traj_required: trajectories need processing
    :return: iterator of tuple(ids, start, end)
    """
    max_begin = timestamp - duration + 1
    begin_min = min(map(attrgetter("begin"), traj_required))

    if begin_min >= max_begin:
        for traj in traj_required:
            del active_space[traj.id]
        return

    traj_cand = [
        traj
        for traj in takewhile(
            lambda traj: traj.begin < max_begin, active_space.values()
        )
    ]
    for traj in traj_required:
        del active_space[traj.id]
    ids = [traj.id for traj in traj_cand]
    cls = [traj.label for traj in traj_cand]
    begins = [traj.begin for traj in traj_cand]
    result_bag = Counter(cls)

    # It's impossible for result bag to have more label types. because we filtered labels first.
    assert len(result_bag) <= len(labels)
    if len(result_bag) == len(labels):
        for l in result_bag:
            result_bag[l] -= labels[l]
        for l_num in result_bag.values():
            if l_num < 0:
                return
        timestamp -= 1  # the end point is exclusive

        i = len(begins)
        old = begins[-1]
        for p in range(i - 1, bisect_left(begins, begin_min) - 1, -1):
            beg = begins[p]
            if beg != old:
                yield ids[:i], old, timestamp
                for j in range(p + 1, i):
                    label = cls[j]
                    result_bag[label] -= 1
                    if result_bag[label] < 0:
                        return
                old = beg
                i = p + 1
        else:
            yield ids[:i], beg, timestamp


def group_until(queue, ts):
    if not queue or queue[0][0] > ts:
        return
    else:
        t, tid = heapq.heappop(queue)
        group = [tid]
        while queue:
            end = queue[0][0]
            if end == t:
                group.append(heapq.heappop(queue)[1])
            else:
                yield t, group
                if end > ts:
                    return
                t = end
                group = [heapq.heappop(queue)[1]]
        yield t, group


def sliding_window(df, win_len):
    df_iter = iter(df)
    window = deque(islice(df_iter, win_len), maxlen=win_len)
    if not window:
        return
    else:
        yield window
    for frame in df_iter:
        window.append(frame)
        yield window


def state_sliding(
    pat_series: Iterable[CoMovementPattern],
    obj_verifier,
    state_maintainer: Callable[[int, Sequence, int, int, bool], [Iterable, Iterable]],
) -> Iterable[CoMovementPattern]:
    # $prev stores seen patterns. Every pattern is a set of objs that co-moves a certain period
    # Note that in terms of object set, prev[0] ⊃ prev[1] ⊃ prev[2] ⊃ ...
    pat_iter = iter(pat_series)
    cur = next(pat_iter, None)
    if cur is None:
        return
    prev = [cur]
    for cur in pat_iter:
        absorb = False
        cur_end = cur.end
        if cur.start > prev[0].end + 1:  # not consecutive in time interval
            count = len(prev)
            new = [cur]
        else:
            new = []
            count = 0
            for pat in prev:
                inter = pat.objs & cur.objs
                if len(inter) == len(pat):  # pat is a subset of cur
                    if len(inter) == len(cur):  # pat equals to cur
                        pat.end = cur_end
                    else:
                        new.append(cur)
                    absorb = True
                    break

                elif len(inter) == len(cur):  # cur is a proper subset of pat
                    cur.start = pat.start
                    count += 1

                else:  # intersection is a new objet set
                    new_pattern = CoMovementPattern(
                        {obj: cur.labels[obj] for obj in inter}
                    )
                    new.append(cur)
                    if obj_verifier(new_pattern.label_count()):  # a new pattern
                        new_pattern.interval = [pat.start, cur_end]
                        cur = new_pattern
                        count += 1
                    else:
                        break
            else:
                new.append(cur)

        prev_end = prev[0].end
        fruits, prev_iter = state_maintainer(prev_end, prev, len(new), count, absorb)
        for pat in fruits:
            pat.end = prev_end
            yield pat.to_plain()

        new.extend(prev_iter)
        prev = new

    if prev:
        fruits, _ = state_maintainer(0, prev, 0, len(prev), False)
        prev_end = prev[0].end
        for pat in fruits:
            pat.end = prev_end
            yield pat.to_plain()


def heap_group_pop(heap):
    if heap:
        n = heap[0][0]
    else:
        return
    while heap and heap[0][0] == n:
        yield heapq.heappop(heap)[1]


def absorb(
    trajectories: Mapping[int, Trajectory],
    duration: int,
    *,
    check_duration: bool = True,
):
    """
    Merge segments from the same object.

    Internal representation:
        traj.seg = [begin1, end1, begin2, end2, ...]
    where each interval is half-open: [begin, end).

    Output representation:
        TrajectoryIntervalSeg(..., begin, ..., end - 1)
    where output end is inclusive.

    If check_duration=False, this function only coalesces intervals and does
    not require the result to satisfy duration. This is useful for stride/carry
    logic across slice boundaries.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    def valid_len(beg, end):
        return (not check_duration) or (end - beg >= duration)

    for tid, traj in trajectories.items():
        seq = traj.seg

        if not seq:
            continue

        if len(seq) % 2 != 0:
            raise ValueError(f"trajectory {tid} has odd endpoint count")

        if len(seq) > 2:
            seq.sort()
            beg = seq[0]

            for a, b in batched(islice(seq, 1, len(seq) - 1), n=2):
                if a != b:
                    if a > beg and valid_len(beg, a):
                        yield TrajectoryIntervalSeg(tid, beg, traj.label, a - 1)
                    beg = b

            end = seq[-1]
            if end > beg and valid_len(beg, end):
                yield TrajectoryIntervalSeg(tid, beg, traj.label, end - 1)

        else:
            beg, end = seq
            if end > beg and valid_len(beg, end):
                yield TrajectoryIntervalSeg(tid, beg, traj.label, end - 1)


def vanilla_merge(rest_idx, trajs, region, labels: Mapping, dur, interval):
    candidates, probation = zip(
        *(
            rest_idx[label].query(trajs, region.bbox, dur, interval, 0)
            for label in labels
        )
    )
    verified = chain(
        *probation, *(candidate_verified_queue(can, region, dur) for can in candidates)
    )
    visited = {}
    for seg in verified:
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.end])
        else:
            old.seg.extend([seg.begin, seg.end])

    trajs = list(absorb(visited, dur))
    trajs.sort(key=attrgetter("begin"))
    return trajs


def add_interval(store, tid, label, beg, end):
    """
    Add a half-open interval [beg, end) into a trajectory store.
    """
    if end <= beg:
        return

    old = store.get(tid)
    if old is None:
        store[tid] = Trajectory(tid, label, [beg, end])
    else:
        old.seg.append(beg)
        old.seg.append(end)

def add_seg(visited, seg):
    add_interval(visited, seg.id, seg.label, seg.begin, seg.end)

def flush_window(visited, *, final: bool):
    """
    Flush current visited trajectories. visited is updated in place.

    Non-final flush:
    - emit closed intervals only if len >= dur;
    - discard closed intervals shorter than dur;
    - carry boundary-touching/crossing intervals.

    Final flush:
    - emit every interval satisfying len >= dur;
    - discard the rest.
    """
    merged = list(absorb(visited, dur, check_duration=False))
    merged.sort(key=attrgetter("begin"))

    emitted = []
    carried: dict[int, Trajectory] = {}

    carry_cut = window_end - dur

    for seg in merged:
        beg = seg.begin
        end = seg.end + 1  # convert inclusive output end back to half-open

        if final:
            if end - beg >= dur:
                emitted.append(seg)
            continue

        if end >= window_end:
            # This interval may merge with the next slice.
            #
            # Since partial merging is allowed, we only carry the suffix
            # that is close enough to the boundary.
            carry_beg = max(beg, carry_cut)

            # Emit the safe prefix if it is long enough.
            if carry_beg > beg and carry_beg - beg >= dur:
                emitted.append(
                    TrajectoryIntervalSeg(
                        seg.id,
                        beg,
                        seg.label,
                        carry_beg - 1,
                    )
                )

            # Carry the suffix even if it is shorter than dur.
            # It may become valid after merging with the next slice.
            add_interval(carried, seg.id, seg.label, carry_beg, end)

        else:
            # This interval ends before the window boundary.
            # Since future segments begin at or after window_end, it cannot
            # merge with future slices anymore.
            if end - beg >= dur:
                emitted.append(seg)
            # else: discard impossible short interval

    visited.clear()
    visited.update(carried)
    emitted.sort(key=attrgetter("begin"))
    return emitted



def stride_merge(
    rest_idx,
    trajs,
    region,
    labels: Mapping,
    dur,
    interval,
    expend=2,
    minima=3000,
):
    """
    Windowed trajectory merge with cross-slice carry.

    Design:
    - absorb(..., check_duration=False) only coalesces intervals.
    - stride_merge decides whether an interval is safe to emit, should be
      carried to the next window, or should be discarded.
    - Boundary-crossing intervals are partially carried using the last `dur`
      time units of the window.
    """
    if not labels:
        return

    if dur <= 0:
        raise ValueError("dur must be positive")
    if expend <= 1:
        raise ValueError("expend must be greater than 1")
    if minima <= 0:
        raise ValueError("minima must be positive")

    stride = max(dur * expend, minima)

    if stride < 2 * dur:
        raise ValueError("window size must be at least 2 * dur")

    queried = [
        rest_idx[label].query(trajs, region.bbox, dur, interval, 1)
        for label in labels
    ]

    candidates, probation = zip(*queried)

    streams = [iter(p) for p in probation]
    streams.extend(
        candidate_verified_queue(candidate, region, dur)
        for candidate in candidates
    )

    # Required for stride-window correctness.
    verified = merge(*streams, key=attrgetter("begin"))

    window_start = interval[0]
    window_end = window_start + stride

    visited: dict[int, Trajectory] = {}
    for seg in verified:
        beg = seg.begin

        if beg >= window_end:
            emitted = flush_window(visited, final=False)
            yield from emitted

            # Adaptive window: skip empty time ranges.
            window_start = beg
            window_end = window_start + stride

        add_seg(visited, seg)

    if visited:
        yield from flush_window(visited, final=True)


def one_pass_merge(rest_idx, trajs, region, labels: Mapping, dur, interval):
    if not labels:
        return
    candidates, probation = zip(
        *(
            rest_idx[label].query(trajs, region.bbox, dur, interval, 1)
            for label in labels
        )
    )
    traj_it = heapq.merge(
        *probation,
        *(candidate_verified_queue(can, region, dur) for can in candidates),
        key=attrgetter("begin"),
    )
    return traj_it
