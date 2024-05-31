import heapq
from collections.abc import Mapping, Iterable
from functools import partial
from itertools import chain, groupby
from operator import attrgetter


from search.rest import yield_co_move, group_until
from search.verifier import candidate_verified_queue
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, TrajectorySequenceSeg


class MaxDurFirst:
    def __init__(self, trajs: Iterable[TrajectoryIntervalSeg | TrajectorySequenceSeg], interval, verifier):
        self.ts_grouped_traj = groupby(trajs, attrgetter('begin'))
        self.interval = interval
        self.playground = {}
        self.verify = partial(verifier, self.playground)
        end_time_queue = []
        self.etq_push = partial(heapq.heappush, end_time_queue)
        self.etq = end_time_queue

    def _yield_until(self, ts, pre_insert):
        for t, group in group_until(self.etq, ts):
            if t < ts:
                yield from self.verify(t, [self.playground[tid] for tid in group])
            else:  # t == ts
                t_candi = []
                for tid in group:
                    if (revising := pre_insert.pop(tid, None)) is None:
                        t_candi.append(self.playground[tid])
                    else:
                        self.etq_push((revising.end, tid))
                if t_candi:
                    yield from self.verify(ts, t_candi)
                return

    def _update(self, pre_insert):
        for tra_id, tra in pre_insert.items():
            assert tra_id not in self.playground
            self.playground[tra_id] = tra
            self.etq_push((tra.end, tra_id))
        return self.etq[0][0]

    def _init_state(self):
        start = self.interval[0]
        for ts, trajs in self.ts_grouped_traj:  # gather trajs on the starting border
            if ts < start:
                for traj in trajs:
                    if traj.end > start:
                        traj.begin = start
                        self.playground[traj.id] = traj
            elif ts == start:
                for traj in trajs:
                    self.playground[traj.id] = traj
            else:
                self.ts_grouped_traj = chain(((ts, trajs),), self.ts_grouped_traj)
                break
        for tid, traj in self.playground.items():
            self.etq_push((traj.end, tid))

    def __iter__(self):
        begin, finish = self.interval
        self._init_state()
        if self.etq:
            next_end = self.etq[0][0]
        else:
            return

        for t, group in self.ts_grouped_traj:
            if t > finish:
                break
            pre_insert = {traj.id: traj for traj in group}
            if next_end <= t:
                yield from self._yield_until(t, pre_insert)
            next_end = self._update(pre_insert)

        finish += 1   # in a segment, end point is exclusive
        if next_end < finish:
            yield from self._yield_until(finish-1, {})
        if self.etq:
            yield from self.verify(finish, [self.playground[info[1]] for info in self.etq])


def max_dur_first_search(data_pack, region: Box2D, labels: Mapping, duration_range, interval):
    spat_tempo_idx, trajs = data_pack
    for c in labels:
        if c not in spat_tempo_idx:
            return iter([])
    candidates, probation = zip(*(spat_tempo_idx[label].query(trajs, region.bbox, duration_range[0], interval, 1)
                                  for label in labels))
    traj_it = heapq.merge(*probation,
                          *(candidate_verified_queue(can, region, duration_range[0]) for can in candidates),
                          key=attrgetter('begin'))
    verifier = partial(yield_co_move, duration_range[0], labels)
    return MaxDurFirst(traj_it, interval, verifier)
