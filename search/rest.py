import heapq
from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from itertools import batched, chain, islice, takewhile
from operator import attrgetter

from search.co_moving import CoMovementPattern
from search.verifier import candidate_verified_queue
from utilities.trajectory import BasicTrajectorySeg, Trajectory, TrajectoryIntervalSeg


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
        # Terminal flush: use the real end of the head chain, not 0. A fake
        # prev_end of 0 would disable the valid_ptr decay in state_maintain and
        # drop the head chain when it only just reached `duration` at the last
        # frame of the stream.
        prev_end = prev[0].end
        fruits, _ = state_maintainer(prev_end, prev, 0, len(prev), False)
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


def valid_len(beg, end, duration, check_duration):
    return (not check_duration) or (end - beg >= duration)


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
        TrajectoryIntervalSeg(..., begin, ..., end)
    where output end is half-open, matching the input convention. The
    canonical internal segment representation is half-open [begin, end)
    everywhere between the merge layer and the enumerators.

    If check_duration=False, this function only coalesces intervals and does
    not require the result to satisfy duration. This is useful for stride/carry
    logic across slice boundaries.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

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
                    if a > beg and valid_len(beg, a, duration, check_duration):
                        yield TrajectoryIntervalSeg(tid, beg, traj.label, a)
                    beg = b

            end = seq[-1]
            if end > beg and valid_len(beg, end, duration, check_duration):
                yield TrajectoryIntervalSeg(tid, beg, traj.label, end)

        else:
            beg, end = seq
            if end > beg and valid_len(beg, end, duration, check_duration):
                yield TrajectoryIntervalSeg(tid, beg, traj.label, end)


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
    Add a half-open interval [beg, end) into a trajectory store, coalescing on
    insert.

    Callers feed segments in ascending ``begin`` order per id (the sorted
    RestIndex stream), so each interval is at or after the last stored one.
    When it touches/overlaps the last interval we extend that interval's end in
    O(1); when it is disjoint we append a new interval. This keeps ``traj.seg``
    proportional to the number of disjoint intervals rather than the number of
    incoming pieces, so a long-lived trajectory holds O(1) state.

    Out-of-order input (defensive) is appended raw; the release-time ``absorb``
    still sorts and merges it.
    """
    if end <= beg:
        return

    old = store.get(tid)
    if old is None:
        store[tid] = Trajectory(tid, label, [beg, end])
        return

    seg = old.seg
    if beg >= seg[-2]:  # at or after the last interval
        if beg <= seg[-1]:  # touches/overlaps the last interval -> extend
            if end > seg[-1]:
                seg[-1] = end
        else:  # disjoint -> new interval
            seg.append(beg)
            seg.append(end)
    else:  # out-of-order: keep raw, absorb() will sort it on release
        seg.append(beg)
        seg.append(end)


def verify_and_merge(
    rest_idx,
    trajs,
    region,
    labels: Mapping,
    dur: int,
    interval: Sequence[int],
    expend: int = 2,
    minima: int = 3000,
):
    if not labels:
            return []
    
    if dur <= 0:
        raise ValueError("dur must be positive")
    if expend <= 2:
        raise ValueError("expend must be greater than 2")
    if minima <= 0:
        raise ValueError("minima must be positive")
    if len(interval) != 2:
        raise ValueError("interval must be a 2-element sequence")
    if interval[0] >= interval[1]:
        raise ValueError("interval[0] must be less than interval[1]")

    stride = max(dur * expend, minima)

    if stride < 2 * dur:
        raise ValueError("window size must be at least 2 * dur")

    # Validate that all labels exist in the index
    missing_labels = [label for label in labels if label not in rest_idx]
    if missing_labels:
        raise KeyError(f"Labels not found in index: {missing_labels}")

    queried = [
        rest_idx[label].query(trajs, region.bbox, dur, interval, 1) for label in labels
    ]

    candidates, probation = zip(*queried)

    streams = [iter(p) for p in probation]
    streams.extend(
        candidate_verified_queue(candidate, region, dur) for candidate in candidates
    )

    return heapq.merge(*streams, key=attrgetter("begin"))

def flush_window(visited, window_end, dur, *, final: bool):
    """
    Flush current visited trajectories. visited is updated in place.

    Non-final flush:
    - emit closed intervals only if len >= dur;
    - discard closed intervals shorter than dur;
    - carry boundary-touching/crossing intervals.

    Final flush:
    - emit every interval satisfying len >= dur;
    - discard the rest.

    Interval Representation:
    - Input: absorb() outputs inclusive intervals [beg, end-1]
    - Conversion: seg.end + 1 converts inclusive end to half-open [beg, end)
    - Output: emitted segments use inclusive representation for downstream compatibility
    """
    merged = list(absorb(visited, dur, check_duration=False))
    merged.sort(key=attrgetter("begin"))

    emitted = []
    carried: dict[int, Trajectory] = {}

    # Ensure carry_cut is positive to avoid incorrect interval splitting
    assert window_end > dur
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

def stride_merge_streaming(
    rest_idx,
    trajs,
    region,
    labels: Mapping,
    dur: int,
    interval: Sequence[int],
    expend: int = 2,
    minima: int = 3000,
):
    verified = verify_and_merge(
        rest_idx, trajs, region, labels, dur, interval, expend, minima
    )
    stride = max(dur * expend, minima)
    window_start = interval[0]
    window_end = window_start + stride

    visited: dict[int, Trajectory] = {}
    trunced = set()
    min_truned_start = window_end
    for seg in verified:
        beg = seg.begin
        if beg >= window_end:
            emitted = flush_window(visited, min_truned_start, dur, final=False)
            yield from emitted
            if min_truned_start < window_end:
                # we have some truncated segments, so we need to slide the window start to the min_truned_start
                window_start = min_truned_start
                window_end = window_start + stride
            else:
                # Adaptive window: skip empty time ranges.
                window_start = beg
                window_end = window_start + stride
            min_truned_start = window_end
        if window_end - beg < dur:
            if seg.end < window_end:
                # this segment has no future segments to merge with, so we can skip it now
                continue
            # this one might be truncated, so we need to carry it over to the next window
            # and since its length is less than dur after truncation, we need to protect it
            # by sliding the window start to the beginning of this segment, and then flush the current window
            # for this to workd, we need to ensure that the window size is at least 2 * dur
            # otherwise, we might end up with trajectories whose length is less than dur after truncation.
            min_truned_start = min(min_truned_start, beg)
            continue
        add_interval(visited, seg.id, seg.label, seg.begin, seg.end)

    if visited:
        yield from flush_window(visited, window_end, dur, final=True)


def stride_merge(
    rest_idx,
    trajs,
    region,
    labels: Mapping,
    dur: int,
    interval: Sequence[int],
    expend: int = 2,
    minima: int = 3000,
):
    """
    Streaming trajectory merge, equivalent to ``vanilla_merge``.

    Segments are consumed globally in ascending ``begin`` order (the sorted
    RestIndex query path). A trajectory's collected intervals are coalesced by
    ``absorb`` and released as soon as no later segment can touch them — once
    the next segment's ``begin`` is strictly greater than the trajectory's last
    end. ``expend``/``minima`` bound how long an *open* trajectory is retained;
    they never change the emitted segment multiset.

    A long-lived trajectory is released after segments that began later, so its
    own (earlier) segments would arrive out of order; the released segments are
    therefore buffered and sorted once. Callers receive a list, exactly like
    ``vanilla_merge``.

    Interval representation: half-open ``[begin, end)`` end to end.
    """

    verified = verify_and_merge(
        rest_idx, trajs, region, labels, dur, interval, expend, minima
    )
    visited: dict[int, Trajectory] = {}
    # Min-heap of (max_end, tid) for open trajectories. Entries go stale when a
    # trajectory gains a later interval; the stale entry is skipped and the
    # refreshed one is popped later.
    open_heap: list[tuple[int, int]] = []
    released = []

    def release(tid: int) -> None:
        traj = visited.pop(tid)
        released.extend(absorb({tid: traj}, dur))

    for seg in verified:
        beg = seg.begin
        while open_heap and open_heap[0][0] < beg:
            last_end, tid = heapq.heappop(open_heap)
            traj = visited.get(tid)
            if traj is None:
                continue
            if max(traj.seg[1::2]) != last_end:
                continue  # stale entry; the trajectory grew
            release(tid)

        add_interval(visited, seg.id, seg.label, seg.begin, seg.end)
        heapq.heappush(open_heap, (max(visited[seg.id].seg[1::2]), seg.id))

    for tid in list(visited):
        release(tid)

    released.sort(key=attrgetter("begin"))
    return released


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


def coalesce(segments, dur: int):
    """Group a raw segment stream by object and absorb it.

    ``one_pass_merge`` yields per-cell segments without coalescing, so a single
    object can appear as several touching pieces. ``vanilla_merge`` and
    ``stride_merge`` both absorb per object, so any consumer of
    ``one_pass_merge`` must run this first to see the same segment multiset.

    Returns a list sorted by ``begin``, using half-open ``[begin, end)`` ends.
    """
    visited: dict[int, Trajectory] = {}
    for seg in segments:
        add_interval(visited, seg.id, seg.label, seg.begin, seg.end)
    out = list(absorb(visited, dur))
    out.sort(key=attrgetter("begin"))
    return out
