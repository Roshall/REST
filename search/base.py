import heapq
import math
from collections import Counter
from collections.abc import Mapping
from functools import partial
from itertools import chain, groupby, islice, batched
from operator import attrgetter
from time import perf_counter as now

from search.co_moving import CoMovementPattern
from search.rest import group_until, state_sliding
from search.verifier import candidate_verified_queue, obj_verify
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, Trajectory


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
            for a, b in batched(islice(seq, 1, len(seq)-1), n=2):
                if a != b:
                    s_len = a - beg
                    if s_len >= duration:
                        yield TrajectoryIntervalSeg(tid, beg, traj.label, s_len-1)
                    beg = b
            s_len = seq[-1] - beg
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, beg, traj.label, s_len-1)
        else:
            s_len = seq[-1] - seq[0]
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, seq[0], traj.label, s_len - 1)


def naive_merge_segments(data_pack, region, labels: Mapping, dur, interval):
    spat_tempo_idx, trajs = data_pack
    for c in labels:
        if c not in spat_tempo_idx:
            return []
    candidates, probation = zip(*(spat_tempo_idx[label].query(trajs, region.bbox, dur, interval, 0)
                                  for label in labels))
    verified = chain(*probation, *(candidate_verified_queue(can, region, dur) for can in candidates))
    visited = {}
    for seg in verified:
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.begin + seg.len])
        else:
            old.seg.extend([seg.begin, seg.begin + seg.len])

    trajs = list(absorb(visited, dur))
    trajs.sort(key=attrgetter('begin'))
    return trajs

class BaseSliding:
    def __init__(self, trajectories: list[TrajectoryIntervalSeg], interval, dur, label_verifier):
        self.ts_grouped_traj = trajectories
        self.label_verifier = label_verifier
        self.interval = interval
        self.dur = dur
        self.end_q = []
        self.label_m = {}
        self.eq_push = partial(heapq.heappush, self.end_q)
        self.eq_group_pop = partial(group_until, self.end_q)
        self.last_win_hi = 0

    def __iter__(self):
        start, terminal = self.interval
        self._init_state()

        if self.end_q:  # first sliding result
            yield from self.start_point_pat_check([start, self.last_win_hi])

        for ts, trajs in self.ts_grouped_traj:
            if (end := ts + self.dur - 1) >= terminal:
                if self.end_q:
                    for ts_end, group in self.eq_group_pop(terminal - 1):
                        yield from self.end_point_pat_check(ts_end, group)
                if end == terminal:
                    self._add(trajs)
                yield from self.start_point_pat_check([terminal - self.dur + 1, terminal])
                break

            if self.end_q and end > self.end_q[0][0]:
                for ts_end, group in self.eq_group_pop(end-1):
                    yield from self.end_point_pat_check(ts_end, group)

            if end > self.last_win_hi and self.label_verifier(Counter(self.label_m.values())):
                yield from self.consecutive_check(end)

            self._add(trajs)
            yield from self.start_point_pat_check([ts, end])

            if end == self.end_q[0][0]:
                for _, group in self.eq_group_pop(end):  # there must be only one group
                    self._remove(group)
            self.last_win_hi = end + self.dur
        else:
            if self.end_q:
                for ts_end, group in self.eq_group_pop(math.inf):
                    if ts_end >= terminal:
                        group.extend(elem[1] for elem in self.end_q)
                        yield from self.end_point_pat_check(terminal, group)
                        break
                    else:
                        yield from self.end_point_pat_check(ts_end, group)

    def end_point_pat_check(self, end, stale_trajs):
        if self.label_verifier(Counter(self.label_m.values())):
            yield from self.consecutive_check(end)
            yield CoMovementPattern(self.label_m.copy(), [end - self.dur + 1, end])
            self.last_win_hi = end + self.dur
        self._remove(stale_trajs)

    def consecutive_check(self, end):
        label_m = self.label_m.copy()
        while end > self.last_win_hi:
            yield CoMovementPattern(label_m, [self.last_win_hi - self.dur + 1, self.last_win_hi])
            self.last_win_hi += self.dur

    def _remove(self, group):
        for tid in group:
            del self.label_m[tid]

    def _init_state(self):
        trajectories = self.ts_grouped_traj
        ts_grouped_traj = groupby(trajectories, key=attrgetter('begin'))
        start = self.interval[0]
        for ts, trajs in ts_grouped_traj:
            if ts > start:
                break
            else:
                for tra in trajs:
                    if start + tra.len > self.dur:
                        self.label_m[tra.id] = tra.label
                        self.eq_push((tra.len + tra.begin, tra.id))
        else:
            return

        self.ts_grouped_traj = chain([(ts, trajs)], ts_grouped_traj)
        self.last_win_hi = (start if self.end_q else ts) + self.dur - 1

    def start_point_pat_check(self, interval=None):
        # This is a generator that the caller don't need to check
        # anything. Just yield from.
        if self.label_verifier(Counter(self.label_m.values())):
            yield CoMovementPattern(self.label_m.copy(), interval)

    def _add(self, trajs):
        for tra in trajs:
            self.label_m[tra.id] = tra.label
            self.eq_push((tra.len + tra.begin, tra.id))


def base_search(data_pack, region: Box2D, labels: Mapping, duration_range, interval):
    label_verifier = partial(obj_verify, labels)
    trajs = naive_merge_segments(data_pack, region, labels, duration_range[0], interval)
    if trajs:
        partial_res = BaseSliding(trajs, interval, duration_range[0], label_verifier)
        return state_sliding(partial_res, label_verifier, base_maintainer)
    else:
        return iter([])


def base_maintainer(prev_end, prev, new_len, count, to_absorb):
    if to_absorb:
        prev_iter = iter(prev)
        return islice(prev_iter, count), prev_iter
    else:
        return prev, []
