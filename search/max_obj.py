from collections.abc import Mapping, Iterable
from functools import partial
import heapq
from itertools import chain, groupby, islice
from operator import attrgetter
from time import perf_counter as now

from search.base import absorb
from search.verifier import candidate_verified_queue
from utilities.box2D import Box2D
from utilities.trajectory import TrajectoryIntervalSeg, Trajectory


def heap_group_pop(heap):
    if heap:
        n = heap[0][0]
    else:
        return
    while heap and heap[0][0] == n:
        yield heapq.heappop(heap)[1]


def label_verifier(label_counter):
    for count in label_counter.values():
        if count < 0:
            return False
    return True


class PatternPool:
    def __init__(self):
        self.patterns = []
        self.end = 0

    def reset(self):
        self.patterns = []

    def concatenate(self, obj_m, label_count, start, end):
        if not self.patterns:
            self.patterns = [(obj_m, start)]
            self.end = end
            return

        new = []
        count = 0
        adopt = False
        cur = obj_m
        for p, ps in self.patterns:
            ckeys, pkeys = cur.keys(), p.keys()
            clen, plen = len(ckeys), len(pkeys)

            if clen > plen:
                if ckeys > pkeys:
                    new.append((cur, start))
                    adopt = True
                    break
            elif clen < plen:
                if ckeys < pkeys:
                    start = ps
                    count += 1
                    continue
            elif clen == plen:
                if ckeys == pkeys:
                    adopt = True
                    break

            new.append((cur, start))
            # update new pattern
            p_new = cur.copy()
            for o, l in cur.items():
                if o not in p:
                    label_count[l] -= 1
                    del p_new[o]

            if label_verifier(label_count):
                start = ps
                cur = p_new
                count += 1
            else:
                break
        else:
            new.append((cur, start))

        p_iter = iter(self.patterns)
        for obj_m, s in islice(p_iter, count if adopt else None):
            yield obj_m.keys(), s, self.end
        new.extend(p_iter)
        self.patterns = new
        self.end = end

    def pop_all(self):
        for pat, start in self.patterns:
            yield pat.keys(), start, self.end
        self.patterns = []


class MaxObjNum:
    def __init__(self, trajectories: Iterable[TrajectoryIntervalSeg], interval, dur, labels):
        self.ts_grouped_traj = groupby(trajectories, key=attrgetter('begin'))
        self.interval = interval
        self.dur = dur - 1
        self.end_q = []
        self.counter_back = {label: -count for label, count in labels.items()}
        self.label_counter = self.counter_back.copy()
        self.label_m = {}
        self.eq_push = partial(heapq.heappush, self.end_q)
        self.eq_group_pop = partial(heap_group_pop, self.end_q)
        self.label_verify = partial(label_verifier, label_counter=self.label_counter)

    def __iter__(self):
        start, terminal = self.interval
        final_s = terminal - self.dur
        last_s = self._init_state()
        ppool = PatternPool()

        if last_s is None:
            return

        for ts, trajs in self.ts_grouped_traj:
            if self.end_q[0][0] >= terminal:
                if self.label_verify():
                    yield self.label_m.keys(), last_s, terminal
                # impossible to concatenate
                yield from ppool.pop_all()
            elif ts <= final_s:
                while self.end_q:  # in case that all objects have gone
                    # we want to concatenate the windows, so the last window end (end_q[0][0])
                    # must >= ts + dur_l.
                    if (end := self.end_q[0][0]) < ts + self.dur:
                        if self.label_verify():
                            yield from ppool.concatenate(self.label_m.copy(), self.label_counter.copy(), last_s, end)
                            self._remove()
                        else:
                            self._remove()
                            # impossible concatenate
                            yield from ppool.pop_all()
                            break  # no need to find end >= ts
                    else:  # found the end >= ts + dur_l, construct the final window, and get out of the loop
                        if self.label_verify():
                            yield from ppool.concatenate(self.label_m.copy(), self.label_counter.copy(), last_s, end)
                        else:
                            # impossible concatenate
                            yield from ppool.pop_all()
                        break
                last_s = ts
                self._add(trajs)
            else:
                break

        # Note: the following output may fail the duration constraints.
        # However, we must keep them in case they can be concatenated to the last window.
        while self.end_q:
            if self.label_verify():
                if self.end_q[0][0] < terminal:
                    yield from ppool.concatenate(self.label_m.copy(), self.label_counter.copy(), last_s, self.end_q[0][0])
                    self._remove()
                else:
                    yield from ppool.concatenate(self.label_m.copy(), self.label_counter.copy(), last_s, self.end_q[0][0])
                    break
            else:
                break

        end = ppool.end
        max_start = end - self.dur
        for pat, start in ppool.patterns:
            if start <= max_start:
                yield pat.keys(), start, end

    def _remove(self):
        for tid in self.eq_group_pop():
            self.label_counter[self.label_m[tid]] -= 1
            del self.label_m[tid]

    def reset(self):
        self.label_m = {}
        self.end_q = []
        self.label_counter = self.counter_back.copy()

    def _init_state(self):
        start = self.interval[0]
        min_end = start + self.dur
        for ts, trajs in self.ts_grouped_traj:
            if ts > start:
                if self.end_q:
                    self.ts_grouped_traj = chain([(ts, trajs)], self.ts_grouped_traj)
                    return start
                else:
                    self._add(trajs)
                    return ts
            else:
                self._add(tra for tra in trajs if tra.begin + tra.len > min_end)
        else:
            if self.end_q:
                return start
            else:
                return

    def _add(self, trajs):
        for tra in trajs:
            tid = tra.id
            label = tra.label
            self.label_counter[label] += 1
            self.label_m[tid] = label
            self.eq_push((tra.len + tra.begin, tid))


def max_obj_search(data_pack, region: Box2D, labels: Mapping, duration_range, interval):
    spat_tempo_idx, trajs = data_pack
    for c in labels:
        if c not in spat_tempo_idx:
            return iter([])

    dur = duration_range[0]
    # Note that spat_tempo should use fuzzy search but not fuzzy inner all
    traj_it = chain.from_iterable(spat_tempo_idx[c].query(trajs, region.bbox, dur, interval, 0) for c in labels)
    verified = candidate_verified_queue(traj_it, region, dur)
    visited = {}
    s_t = now()
    for seg in verified:
        if (old := visited.get(seg.id, None)) is None:
            visited[seg.id] = Trajectory(seg.id, seg.label, [seg.begin, seg.begin + seg.len])
        else:
            old.seg.extend([seg.begin, seg.begin + seg.len])

    # e_t = now()
    # print(f'collection time: {e_t - s_t:.6f} s')
    # s_t = e_t
    trajs = list(absorb(visited, dur))
    # e_t = now()
    # print(f'absorb time: {e_t - s_t:.6f} s')
    # s_t = e_t
    trajs.sort(key=attrgetter('begin'))
    # print(f'sort time: {now() - s_t}')
    if trajs:
        return MaxObjNum(trajs, interval, dur, labels)
    else:
        return iter([])
