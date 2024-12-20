from collections.abc import Iterable
from functools import partial
import heapq
from itertools import chain, groupby, islice
from operator import attrgetter

from search.rest import heap_group_pop
from search.verifier import label_verifier


class PatternPool:
    def __init__(self):
        self.patterns = []
        self.end = 0

    def concatenate(self, objs_m, label_count, start, end):
        objs = set(objs_m)
        if not self.patterns:
            self.patterns = [(objs, start)]
            self.end = end
            return
        elif start > (end_l := self.end):
            for p_objs, s in self.patterns:
                yield p_objs, s, end_l
            self.patterns = [(objs, start)]
            self.end = end
            return

        new = []
        count = 0
        adopt = False
        cur = objs
        for p, ps in self.patterns:
            clen, plen = len(cur), len(p)

            if clen > plen:
                if cur > p:
                    new.append((cur, start))
                    adopt = True
                    break
            elif clen < plen:
                if cur < p:
                    start = ps
                    count += 1
                    continue
            elif clen == plen:
                if cur == p:
                    adopt = True
                    break

            new.append((cur, start))
            # update new pattern
            p_new = cur & p
            for o in cur - p_new:
                label_count[objs_m[o]] -= 1

            if label_verifier(label_count):
                start = ps
                cur = p_new
                count += 1
            else:
                break
        else:
            new.append((cur, start))

        p_iter = iter(self.patterns)
        for objs, s in islice(p_iter, count if adopt else None):
            yield objs, s, self.end
        new.extend(p_iter)
        self.patterns = new
        self.end = end

    def pop_all(self):
        for objs, start in self.patterns:
            yield objs, start, self.end
        self.patterns = []


class MaxObjNumEnumerator:
    def __init__(self, trajectories: Iterable, labels, dur, interval):
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
        if last_s is None:
            return

        ppool = PatternPool()
        for ts, trajs in self.ts_grouped_traj:
            if ts <= final_s:
                end_min = ts + self.dur
                while self.end_q:  # in case that all objects have gone
                    # we want to concatenate the windows, so the last window end (end_q[0][0])
                    # must >= ts + dur_l.
                    if (end := self.end_q[0][0]) < end_min:
                        if self.label_verify():
                            yield from ppool.concatenate(self.label_m, self.label_counter.copy(), last_s, end)
                            self._remove()
                        else:
                            self._remove()
                            # impossible concatenate
                            yield from ppool.pop_all()
                            while self.end_q and self.end_q[0][0] < end_min:
                                # some objects having multiple trajectories,
                                # this will ensure we first remove the old trajectory
                                self._remove()
                            break  # no need to find end >= ts
                    else:  # found the end >= ts + dur_l, construct the final window, and get out of the loop
                        if self.label_verify():
                            yield from ppool.concatenate(self.label_m, self.label_counter.copy(), last_s, end)
                        else:
                            # impossible concatenate
                            yield from ppool.pop_all()
                        break
                last_s = ts
                self._add(trajs)
            else:
                break

        yield from self._wrapup(ppool, last_s, terminal)

    def _wrapup(self, ppool, last_s, terminal_t):
        # Note: the following output may fail the duration constraints.
        # However, we must keep them in case they can be concatenated to the last window.
        while self.end_q:
            if self.label_verify():
                if self.end_q[0][0] < terminal_t:
                    yield from ppool.concatenate(self.label_m, self.label_counter.copy(), last_s, self.end_q[0][0])
                    self._remove()
                else:
                    yield from ppool.concatenate(self.label_m, self.label_counter.copy(), last_s, terminal_t)
                    break
            else:
                break

        # ppool contains the pattern to be concatenated
        # finally, we check their duration and pop them out.
        end = ppool.end
        max_start = end - self.dur
        for pat, start in ppool.patterns:
            if start <= max_start:
                yield pat.keys(), start, end
            else:
                break

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
                self._add(tra for tra in trajs if tra.end >= min_end)
        else:
            if self.end_q:
                return start

    def _add(self, trajs):
        for tra in trajs:
            tid = tra.id
            label = tra.label
            self.label_counter[label] += 1
            self.label_m[tid] = label
            self.eq_push((tra.end, tid))
