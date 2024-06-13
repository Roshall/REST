import heapq
from collections.abc import Mapping, Iterable
from functools import partial
from itertools import chain, groupby
from operator import attrgetter


from search.rest import yield_co_move, heap_group_pop, vanilla_merge


class MaxDurFirstEnumerator:
    def __init__(self, trajs: Iterable, interval, verifier):
        self.ts_grouped_traj = groupby(trajs, attrgetter('begin'))
        self.interval = interval
        self.playground = {}
        self.verify = partial(verifier, self.playground)
        end_time_queue = []
        self.etq_push = partial(heapq.heappush, end_time_queue)
        self.etq = end_time_queue
        self.group_pop = partial(heap_group_pop, end_time_queue)

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


class MaxDurationOnePass(MaxDurFirstEnumerator):
    def __init__(self, trajs: Iterable, interval, verifier):
        super().__init__(trajs, interval, verifier)

    def _yield_until(self, ts, pre_insert):
        while self.etq and (t := self.etq[0][0]) <= ts:
            group = self.group_pop()
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

    def __iter__(self):
        begin, finish = self.interval
        self._init_state()
        if self.playground:
            for tid, traj in self.playground.items():
                self.etq_push((traj.end, tid))
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


class MaxDurMultiPass(MaxDurFirstEnumerator):
    def __init__(self, trajs: Iterable, interval, verifier):
        super().__init__(trajs, interval, verifier)

    def _update(self, group):
        for tra in group:
            tid = tra.id
            assert tid not in self.playground
            self.playground[tid] = tra
            self.etq_push((tra.end+1, tid))
        return self.etq[0][0]

    def __iter__(self):
        begin, finish = self.interval
        self._init_state()
        if self.playground:
            for tid, traj in self.playground.items():
                self.etq_push((traj.end+1, tid))
            next_end = self.etq[0][0]
        else:
            return

        for t, group in self.ts_grouped_traj:
            if t > finish:
                break
            while self.etq and (et := self.etq[0][0]) <= t:
                end_g = self.group_pop()
                yield from self.verify(et, [self.playground[tid] for tid in end_g])
            next_end = self._update(group)

        if next_end <= finish:
            while self.etq and (et := self.etq[0][0]) <= finish:
                end_g = self.group_pop()
                yield from self.verify(et, [self.playground[tid] for tid in end_g])
        if self.etq:
            finish += 1  # in a segment, end point is exclusive
            yield from self.verify(finish, [self.playground[info[1]] for info in self.etq])


def max_dur_enumerate(trajs, labels: Mapping, dur, interval, *, mtd='one'):
    verifier = partial(yield_co_move, dur, labels)
    match mtd:
        case 'one':
            return MaxDurationOnePass(trajs, interval, verifier)
        case 'multi':
            return MaxDurMultiPass(trajs, interval, verifier)
        case _:
            raise ValueError(f'mtd {mtd} not supported')
