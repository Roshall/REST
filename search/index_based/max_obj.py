import heapq
from collections.abc import Iterable
from functools import partial
from itertools import chain, groupby, islice
from operator import attrgetter

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
            yield list(objs), s, self.end
        new.extend(p_iter)
        self.patterns = new
        self.end = end

    def pop_all(self):
        for objs, start in self.patterns:
            yield list(objs), start, self.end
        self.patterns = []
        self._levels = {}

    def iter_until(self, max_start):
        for objs, start in self.patterns:
            if start <= max_start:
                yield list(objs), start, self.end
            else:
                break


class MaxObjNumEnumerator:
    def __init__(self, trajectories: Iterable, labels, dur, interval):
        self.ts_grouped_traj = groupby(trajectories, key=attrgetter("begin"))
        self.interval = interval
        # Segments are half-open [begin, end), so an object occupies `end - begin`
        # time units. Keeping `dur` (rather than dur-1) makes the window test
        # `end - begin >= dur` line up with the max_dur enumerators.
        self.dur = dur
        self.end_q = []
        # id -> current (latest) end. The windowed merge cuts a trajectory that
        # crosses a window boundary into touching pieces; when a later piece
        # begins exactly at the current end it is the same occupancy, so we
        # extend this value instead of letting the object expire first.
        self.end_of = {}
        self.counter_back = {label: -count for label, count in labels.items()}
        self.label_counter = self.counter_back.copy()
        self.label_m = {}
        self.eq_push = partial(heapq.heappush, self.end_q)
        self.label_verify = partial(label_verifier, label_counter=self.label_counter)

    def __iter__(self):
        # The pool works on half-open ends internally; results carry an
        # inclusive end, matching every other enumerator.
        for ids, start, end in self._iterate():
            yield ids, start, end - 1

    # -- heap helpers -------------------------------------------------------
    def _peek_min(self):
        """Smallest end still current, dropping stale heap entries at the top.

        Extending an open id pushes its new end, so the id's old (smaller) entry
        lingers in the heap. It is stale -- ``end_of`` no longer equals it -- and
        discarding it here keeps it from closing a window early. Valid entries
        are left in place, so a caller that only peeks does not consume them.
        """
        end_q = self.end_q
        while end_q:
            end, tid = end_q[0]
            if self.end_of.get(tid) == end:
                return end
            heapq.heappop(end_q)  # superseded by a later piece, or removed
        return None

    def _pop_min_group(self):
        """Remove and return the ids sharing the smallest current end."""
        end = self._peek_min()
        if end is None:
            return None, []
        ids = []
        end_q = self.end_q
        while end_q and end_q[0][0] == end:
            e, tid = heapq.heappop(end_q)
            if self.end_of.get(tid) == e:  # skip stale entries at the same end
                ids.append(tid)
        return end, ids

    def _remove_ids(self, ids):
        for tid in ids:
            self.label_counter[self.label_m[tid]] -= 1
            del self.label_m[tid]
            self.end_of.pop(tid, None)

    def _add_one(self, tra):
        tid = tra.id
        self.label_counter[tra.label] += 1
        self.label_m[tid] = tra.label
        self.end_of[tid] = tra.end
        self.eq_push((tra.end, tid))

    def _add(self, trajs):
        for tra in trajs:
            self._add_one(tra)

    def _iterate(self):
        start, terminal = self.interval
        final_s = terminal - self.dur
        last_s = self._init_state()
        if last_s is None:
            return

        ppool = PatternPool()
        groups = iter(self.ts_grouped_traj)
        buffer: list[tuple[int, list]] = []  # read-ahead groups, ascending begin
        exhausted = False

        def pull():
            nonlocal exhausted
            if exhausted:
                return None
            try:
                ts, trajs = next(groups)
            except StopIteration:
                exhausted = True
                return None
            return ts, list(trajs)

        def fill_until(until):
            # Keep every group with begin <= until in the buffer (plus the first
            # one past it), so the look-ahead below can see a `dur`-wide window.
            while True:
                group = pull()
                if group is None:
                    return
                buffer.append(group)
                if group[0] > until:
                    return

        def merge_prefetched(horizon):
            # Join a cut trajectory back together: a piece that begins exactly
            # at an open id's current end is its continuation, so extend the
            # occupancy rather than letting the id expire first. New ids are NOT
            # activated here -- they are added at their own group -- because the
            # window closing at this group must not contain objects that begin
            # after it.
            for ts_g, trajs in buffer:
                if ts_g > horizon:
                    break
                for tra in trajs:
                    cur = self.end_of.get(tra.id)
                    if cur is not None and tra.begin == cur:
                        self.end_of[tra.id] = tra.end
                        self.eq_push((tra.end, tra.id))

        while True:
            if not buffer:
                group = pull()
                if group is None:
                    break
                buffer.append(group)
            ts, trajs = buffer[0]
            if ts > final_s:
                break
            end_min = ts + self.dur
            # Read-ahead by one `dur`: a continuation that must rescue an open
            # id begins at the id's end, which is < ts + dur exactly when the id
            # is about to expire -- so a window of `dur` always reaches it.
            fill_until(end_min)
            merge_prefetched(end_min)

            while self.end_q:  # in case that all objects have gone
                # we want to concatenate the windows, so the last window end
                # must be >= ts + dur.
                end = self._peek_min()
                if end is None:
                    break
                if end < end_min:
                    if self.label_verify():
                        yield from ppool.concatenate(
                            self.label_m, self.label_counter.copy(), last_s, end
                        )
                        _, ids = self._pop_min_group()
                        self._remove_ids(ids)
                    else:
                        _, ids = self._pop_min_group()
                        self._remove_ids(ids)
                        # impossible concatenate
                        yield from ppool.pop_all()
                        while (nxt := self._peek_min()) is not None and nxt < end_min:
                            # some objects having multiple trajectories, this
                            # ensures we first remove the old trajectory
                            _, ids = self._pop_min_group()
                            self._remove_ids(ids)
                        break  # no need to find end >= ts
                else:  # found the end >= ts + dur, construct the final window
                    if self.label_verify():
                        yield from ppool.concatenate(
                            self.label_m, self.label_counter.copy(), last_s, end
                        )
                    else:
                        # impossible concatenate
                        yield from ppool.pop_all()
                    break
            last_s = ts
            for tra in trajs:
                cur = self.end_of.get(tra.id)
                if cur is not None and tra.end <= cur:
                    continue  # already joined by merge_prefetched
                self._add_one(tra)
            buffer.pop(0)

        # The last window can itself be cut across; join any trailing pieces
        # still within the interval before wrapping up.
        fill_until(terminal)
        merge_prefetched(terminal)
        yield from self._wrapup(ppool, last_s, terminal)

    def _wrapup(self, ppool, last_s, terminal_t):
        # Note: the following output may fail the duration constraints.
        # However, we must keep them in case they can be concatenated to the last window.
        while self.end_q:
            end = self._peek_min()
            if end is None:
                break
            if self.label_verify():
                if end < terminal_t:
                    yield from ppool.concatenate(
                        self.label_m,
                        self.label_counter.copy(),
                        last_s,
                        end,
                    )
                    _, ids = self._pop_min_group()
                    self._remove_ids(ids)
                else:
                    yield from ppool.concatenate(
                        self.label_m, self.label_counter.copy(), last_s, terminal_t
                    )
                    break
            else:
                break

        # ppool contains the pattern to be concatenated
        # finally, we check their duration and pop them out.
        yield from ppool.iter_until(ppool.end - self.dur)

    def reset(self):
        self.label_m = {}
        self.end_of = {}
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
                # A cut head is only a *piece* of its object: its tail begins
                # exactly where it ends, so the occupancy continues past
                # `min_end` even when this piece does not. Filtering such a head
                # out deletes the object for every pattern spanning the interval
                # start, and its tail then reads as a new object that began at
                # the cut. Keep it, and let the read-ahead join it back.
                self._add(
                    tra
                    for tra in trajs
                    if tra.end >= min_end or getattr(tra, "cut", False)
                )
        else:
            if self.end_q:
                return start
