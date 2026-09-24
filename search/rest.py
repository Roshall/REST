import heapq
from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Callable, Iterable, Mapping, MutableMapping, Sequence
from itertools import batched, chain, count, islice, takewhile
from math import inf
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
    if expend <= 1:
        raise ValueError("expend must be greater than 1")
    if minima <= 0:
        raise ValueError("minima must be positive")
    if len(interval) != 2:
        raise ValueError("interval must be a 2-element sequence")
    if interval[0] >= interval[1]:
        raise ValueError("interval[0] must be less than interval[1]")

    stride = max(dur * expend, minima)

    # The windowed merge leans on this bound in two places: a carried
    # trajectory keeps at most its last `dur` frames, and the cut may slide back
    # by up to `dur` to avoid discarding a shorter head (see `cut_position`).
    # With `stride >= 2 * dur` the window still advances by at least `dur` after
    # the worst-case slide, so it can never stall.
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

def cut_position(merged, window_end, dur):
    """
    Choose where to cut the window.

    A trajectory that reaches the cut is split: the tail from ``cut - dur`` is
    carried into the next window, and the head before it is emitted only when
    the head alone is already ``dur`` long. A head of 1..dur-1 frames can be
    neither emitted nor carried — it would simply vanish and silently shorten
    the reported pattern — so such a trajectory is carried whole instead by
    moving the cut down to its own start.

    That is why the cut is a single global time rather than a per-trajectory
    decision: it also defines the output order. Everything emitted begins before
    ``cut - dur`` and everything carried begins at or after it, so this flush
    and the next one stay sorted by ``begin``.

    Carrying a trajectory whole while leaving the cut where it was breaks that
    — its earlier ``begin`` would come out after a later one (measured: 12 of
    800 corpora) — and ``MaxObjNumEnumerator`` groups on ``begin``, so an
    out-of-order value silently splits a group.

    One pass is enough. Sliding to a fixed point was measured over 800 corpora
    and moved the pattern agreement by 0.5% without ever changing the ordering,
    so the second-order cases are not worth the loop.
    """
    floor = window_end - dur
    cut = window_end
    for seg in merged:
        if seg.end >= window_end and 0 < window_end - dur - seg.begin < dur:
            cut = min(cut, max(seg.begin + dur, floor))
    return cut


def flush_window(visited, window_end, dur, *, final: bool):
    """
    Flush the trajectories collected in the current window. ``visited`` is
    updated in place with whatever must survive into the next window.

    Returns ``(emitted, boundary)``: the segments that are safe to hand
    downstream, and the time the window was actually cut at. The caller starts
    the next window at ``boundary``, which is ``window_end`` unless the cut had
    to be slid back (see below).

    Non-final flush:
    - an interval ending before ``boundary`` is closed. Nothing later can touch
      it, so it is emitted when ``len >= dur`` and dropped otherwise.
    - an interval reaching ``boundary`` may continue in the next slice, so its
      tail from ``carry_cut = boundary - dur`` is carried. Its head is emitted
      only when the head alone is already ``>= dur``.

    The cut is slid back until no crossing interval is left with a head shorter
    than ``dur`` — such a head could neither be emitted nor carried, so it would
    silently shorten the reported pattern. Sliding is bounded by ``dur`` (which
    is why ``stride >= 2 * dur`` is required: the window still advances by at
    least ``dur``). Consequently **every segment handed downstream is at least
    ``dur`` long**. That is a hard requirement: ``MaxObjNumEnumerator`` does not
    re-check the duration constraint and will happily report a pattern shorter
    than ``dur`` if the merge lets one through.

    Final flush: emit every interval with ``len >= dur``, drop the rest.

    Interval representation is half-open ``[begin, end)`` end to end.
    """
    merged = list(absorb(visited, dur, check_duration=False))
    merged.sort(key=attrgetter("begin"))

    if final:
        emitted = [seg for seg in merged if seg.end - seg.begin >= dur]
        visited.clear()
        return emitted, window_end

    if window_end <= dur:
        raise ValueError("window_end must be greater than dur")

    boundary = cut_position(merged, window_end, dur)
    carry_cut = boundary - dur

    # Pass 1: decide where each trajectory is carried from, tracking the
    # earliest carry as we go. A trajectory that cannot be cut — its head would
    # be 1..dur-1 frames, too short to emit and too short to carry — is kept
    # whole. ``None`` marks a trajectory that closed before the cut.
    carry_from: list[int | None] = []
    horizon = inf
    for seg in merged:
        if seg.end >= boundary:
            carry_beg = max(seg.begin, carry_cut)
            if 0 < carry_beg - seg.begin < dur:
                carry_beg = seg.begin
            carry_from.append(carry_beg)
            if carry_beg < horizon:
                horizon = carry_beg
        else:
            carry_from.append(None)

    # Pass 2: nothing may be emitted that begins at or after the earliest carry,
    # or a trajectory kept whole would come out after it and break the sort.
    # Anything not safe yet is deferred by one flush — it is already closed, so
    # deferring costs latency, never correctness.
    emitted = []
    carried: dict[int, Trajectory] = {}

    for seg, carry_beg in zip(merged, carry_from):
        beg = seg.begin
        end = seg.end
        if carry_beg is not None:
            # May merge with the next slice: keep the tail close enough to the
            # cut, emit the head only when it stands on its own.
            if carry_beg - beg >= dur:
                emitted.append(TrajectoryIntervalSeg(seg.id, beg, seg.label, carry_beg))
            add_interval(carried, seg.id, seg.label, carry_beg, end)
        elif end - beg >= dur and beg < horizon:
            emitted.append(seg)
        elif end - beg >= dur:
            add_interval(carried, seg.id, seg.label, beg, end)  # defer one flush
        # else: closed and shorter than dur -> discard

    visited.clear()
    visited.update(carried)
    emitted.sort(key=attrgetter("begin"))
    return emitted, boundary

def stride_merge_windowed(
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
    Windowed streaming merge: bounded memory, *approximate* patterns.

    Segments are consumed globally in ascending ``begin`` order. Trajectories are
    coalesced in ``visited`` and flushed whenever the stream reaches the end of
    the window; a trajectory still alive at the cut is carried into the next
    window. Memory is bounded by the trajectories of one window rather than by
    the size of the result, and segments are yielded as soon as they are final.

    Guarantees:
    - emitted segments are globally sorted by ``begin``;
    - every emitted segment is at least ``dur`` long (``MaxObjNumEnumerator``
      does not re-check the duration constraint, so the merge must).

    Caveat — the pattern set is NOT the same as ``stride_merge``: a trajectory
    spanning a window boundary is reported as several touching pieces instead of
    one maximal segment, and the artificial end event at a split point closes
    every concurrent pattern there. Measured against ``stride_merge`` through
    ``MaxObjNumEnumerator`` on random corpora of 40-60 objects, 77-97% of the
    pattern sets differ. Use ``stride_merge_streaming`` when the pattern set has
    to match; use this one only when memory must not grow with the result.

    Interval representation is half-open ``[begin, end)`` throughout.
    """
    verified = verify_and_merge(
        rest_idx, trajs, region, labels, dur, interval, expend, minima
    )
    stride = max(dur * expend, minima)
    window_end = interval[0] + stride

    visited: dict[int, Trajectory] = {}
    for seg in verified:
        if seg.begin >= window_end:
            emitted, cut = flush_window(visited, window_end, dur, final=False)
            yield from emitted
            # `cut >= window_end - dur` and `stride >= 2 * dur`, so the window
            # advances by at least `dur` even when the cut had to slide back.
            window_end = cut + stride
        # Note: a segment starting in the last `dur` of the window is still
        # stored. It is protected at flush time, where the cut slides back so
        # that such a trajectory is carried whole instead of cut below `dur`.
        add_interval(visited, seg.id, seg.label, seg.begin, seg.end)

    if visited:
        emitted, _ = flush_window(visited, window_end, dur, final=True)
        yield from emitted


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
    """
    Streaming merge, equivalent to ``stride_merge`` (hence to ``vanilla_merge``).

    Same release rule as ``stride_merge``: a trajectory is closed once no later
    segment can touch it, i.e. once the next segment's ``begin`` is greater than
    its last end. Instead of buffering every released segment until the end,
    though, they go into a pending min-heap and are yielded as soon as they can
    no longer be overtaken — a pending segment is safe to emit once no
    still-open trajectory began earlier, because every future segment will begin
    later still.

    The output is therefore the same segment multiset as ``stride_merge`` and is
    still globally sorted by ``begin``, so it can be fed straight to an
    enumerator that groups on ``begin``. ``expend``/``minima`` only validate the
    window parameters here; they do not affect the result.

    Nothing is ever cut: a trajectory is carried as one coalesced interval
    (``add_interval`` only ever extends or appends) and is reported whole when it
    closes. That is what keeps the patterns identical to ``stride_merge``.

    The price of never cutting is that nothing beginning after the oldest
    still-open trajectory can be reported before that trajectory closes, because
    a later ``begin`` would break the sort order. So memory is bounded by the
    trajectories that overlap the oldest one's lifetime, **not by the window**;
    ``expend``/``minima`` only validate the window parameters here and do not
    affect the result. In practice the pending heap stays small (measured: 8-34
    segments for results of 160-260 on corpora of 40-60 objects) and it
    degenerates to the full result only if one trajectory outlives most of the
    stream. Emission is gated by exactly that horizon — a released segment is
    yielded once ``min_open_begin()`` has passed it, which is the same rule as
    "slide the window to the earliest trajectory you had to keep".

    Interval representation is half-open ``[begin, end)`` throughout.
    """
    verified = verify_and_merge(
        rest_idx, trajs, region, labels, dur, interval, expend, minima
    )
    visited: dict[int, Trajectory] = {}
    # (max_end, seq, tid): a trajectory can be closed once its last end is
    # behind the incoming segment. Entries go stale when the trajectory grows;
    # the stale entry is skipped and the refreshed one is popped later.
    open_heap: list[tuple[int, int, int]] = []
    # (first_begin, seq, tid): the emission horizon. `seq` invalidates the entry
    # of a trajectory that was closed and later reopened.
    begin_heap: list[tuple[int, int, int]] = []
    pending: list[tuple[int, int, TrajectoryIntervalSeg]] = []
    epoch: dict[int, int] = {}
    tick = count()

    def close(tid: int) -> None:
        traj = visited.pop(tid)
        del epoch[tid]
        for seg in absorb({tid: traj}, dur):
            heapq.heappush(pending, (seg.begin, next(tick), seg))

    def min_open_begin() -> float:
        while begin_heap:
            beg, seq, tid = begin_heap[0]
            if epoch.get(tid) == seq:
                return beg
            heapq.heappop(begin_heap)  # closed, or re-opened under a new seq
        return float("inf")

    for seg in verified:
        beg = seg.begin
        while open_heap and open_heap[0][0] < beg:
            last_end, seq, tid = heapq.heappop(open_heap)
            traj = visited.get(tid)
            if traj is None or epoch.get(tid) != seq:
                continue
            if max(traj.seg[1::2]) != last_end:
                continue  # stale entry; the trajectory grew
            close(tid)

        if seg.id not in visited:
            seq = next(tick)
            epoch[seg.id] = seq
            heapq.heappush(begin_heap, (beg, seq, seg.id))
        add_interval(visited, seg.id, seg.label, seg.begin, seg.end)
        heapq.heappush(open_heap, (max(visited[seg.id].seg[1::2]), epoch[seg.id], seg.id))

        limit = min_open_begin()
        while pending and pending[0][0] < limit:
            yield heapq.heappop(pending)[2]

    for tid in list(visited):
        close(tid)
    while pending:
        yield heapq.heappop(pending)[2]


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
