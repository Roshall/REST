import heapq
from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Iterable, MutableMapping, Callable
from collections.abc import Mapping, Sequence
from itertools import takewhile, islice, batched, chain
from operator import attrgetter

from search.co_moving import CoMovementPattern
from search.verifier import candidate_verified_queue
from utilities.trajectory import BasicTrajectorySeg, Trajectory
from traj_seg import TrajectoryIntervalSeg


def yield_co_move(duration: int, labels: Mapping[int, int], active_space: MutableMapping[int, BasicTrajectorySeg],
                  timestamp: int, traj_required: Sequence) -> Iterable[tuple[list[int], int, int]]:
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
    begin_min = min(map(attrgetter('begin'), traj_required))

    if begin_min >= max_begin:
        for traj in traj_required:
            del active_space[traj.id]
        return

    traj_cand = [traj for traj in takewhile(lambda traj: traj.begin < max_begin, active_space.values())]
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


def state_sliding(pat_series: Iterable[CoMovementPattern],
                  obj_verifier,
                  state_maintainer: Callable[[int, Sequence, int, int, bool], [Iterable, Iterable]]) \
        -> Iterable[CoMovementPattern]:
    # $prev stores seen patterns. Every pattern is a set of objs that co-moves a certain period
    # Note that in terms of object set, prev[0] ⊃ prev[1] ⊃ prev[2] ⊃ ...
    pat_iter = iter(pat_series)
    cur = next(pat_iter, None)
    if cur is None:
        return
    prev = [cur]
    for cur in pat_iter:
        absort = False
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
                    absort = True
                    break

                elif len(inter) == len(cur):  # cur is a proper subset of pat
                    cur.start = pat.start
                    count += 1

                else:  # intersection is a new objet set
                    new_pattern = CoMovementPattern({obj: cur.labels[obj] for obj in inter})
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
        fruits, prev_iter = state_maintainer(prev_end, prev, len(new), count, absort)
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


def absorb(trajectories: Mapping[int, Trajectory], duration):
    """
    merge segments from the same object into a single segment
    :param trajectories: a map: id -> trajectory
    :param duration: co-movement duration
    :return: a generator with trajectories herein.
    """
    for tid, traj in trajectories.items():
        seq = traj.seg
        if len(seq) > 2:
            seq.sort()
            beg = seq[0]
            for a, b in batched(islice(seq, 1, len(seq) - 1), n=2):
                if a != b:
                    s_len = a - beg
                    if s_len >= duration:
                        yield TrajectoryIntervalSeg(tid, beg, traj.label, a - 1)
                    beg = b
            s_len = seq[-1] - beg
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, beg, traj.label, seq[-1] - 1)
        else:
            s_len = seq[-1] - seq[0]
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, seq[0], traj.label, seq[-1] - 1)


def vanilla_merge(rest_idx, trajs, region, labels: Mapping, dur, interval):
    candidates, probation = zip(*(rest_idx[label].query(trajs, region.bbox, dur, interval, 0)
                                  for label in labels))
    verified = chain(*probation, *(candidate_verified_queue(can, region, dur) for can in candidates))
    visited = {}
    for seg in verified:
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.end])
        else:
            old.seg.extend([seg.begin, seg.end])

    trajs = list(absorb(visited, dur))
    trajs.sort(key=attrgetter('begin'))
    return trajs


def stride_merge(rest_idx, trajs, region, labels: Mapping, dur, interval, minima=3000):
    candidates, probation = zip(*(rest_idx[label].query(trajs, region.bbox, dur, interval, 1)
                                  for label in labels))
    verified = chain(*probation, *(candidate_verified_queue(can, region, dur) for can in candidates))
    stride = max(dur * 2, minima)
    end = interval[0] + stride
    visited = {}
    for seg in verified:
        if (beg := seg.begin) < end:
            if (old := visited.get(seg.id, None)) is None:
                visited[seg.id] = Trajectory(seg.id, seg.label, [beg, seg.end])
            else:
                old.seg.extend([beg, seg.end])
        else:
            trajs = list(absorb(visited, dur))
            trajs.sort(key=attrgetter('begin'))
            yield from trajs
            yield None, None
            visited = {}
            end = beg + stride


def one_pass_merge(rest_idx, trajs, region, labels: Mapping, dur, interval):
    candidates, probation = zip(*(rest_idx[label].query(trajs, region.bbox, dur, interval, 1)
                                  for label in labels))
    traj_it = heapq.merge(*probation,
                          *(candidate_verified_queue(can, region, dur) for can in candidates),
                          key=attrgetter('begin'))
    return traj_it
