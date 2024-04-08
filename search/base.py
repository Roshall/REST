import heapq
import math
from collections import Counter
from collections.abc import Mapping
from functools import partial
from itertools import chain, groupby, islice, batched
from operator import attrgetter

import numpy as np

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
                        yield TrajectoryIntervalSeg(tid, beg, traj.label, s_len)
                    beg = b
            s_len = seq[-1] - beg
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, beg, traj.label, s_len)
        else:
            s_len = seq[-1] - seq[0]
            if s_len >= duration:
                yield TrajectoryIntervalSeg(tid, seq[0], traj.label, s_len)


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
        terminal = self.interval[1]
        self._init_state()

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
                    end = min(terminal, ts_end)
                    yield from self.end_point_pat_check(end, group)

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
                        self.eq_push((tra.len + tra.begin - 1, tra.id))
        else:
            return
        self.ts_grouped_traj = chain([(ts, trajs)], ts_grouped_traj)
        self.last_win_hi = ts + self.dur - 1

    def start_point_pat_check(self, interval=None):
        if self.label_verifier(Counter(self.label_m.values())):
            yield CoMovementPattern(self.label_m.copy(), interval)

    def _add(self, trajs):
        for tra in trajs:
            self.label_m[tra.id] = tra.label
            self.eq_push((tra.len + tra.begin - 1, tra.id))


def base_search(spat_tempo_idx, region: Box2D, labels: Mapping, duration_range, interval):
    dur = duration_range[0]
    label_verifier = partial(obj_verify, labels)
    # Note that spat_tempo should use fuzzy search but not fuzzy inner all
    candidates, probation = zip(*(spat_tempo_idx[label].where_intersect(((region.bbox, duration_range), interval))
                                  for label in labels))
    verified = candidate_verified_queue(chain.from_iterable(candidates), region, dur)
    visited = {}
    for seg in chain(verified, chain.from_iterable(probation)):
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.begin + seg.len])
        else:
            old.seg.extend([seg.begin, seg.begin + seg.len])

    trajs = list(absorb(visited, dur))
    trajs.sort(key=attrgetter('begin'))
    if trajs:
        partial_res = BaseSliding(trajs, interval, dur, label_verifier)
        return state_sliding(partial_res, label_verifier, base_maintainer)
    else:
        return iter([])


def base_maintainer(end, prev, new, count, to_absorb):
    if to_absorb:
        prev_iter = iter(prev)
        return islice(prev_iter, count), prev_iter
    else:
        return prev, []
