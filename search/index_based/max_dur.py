import heapq
from collections.abc import Iterable, Mapping
from functools import partial
from itertools import chain, groupby
from operator import attrgetter

from search.rest import heap_group_pop, yield_co_move


class MaxDurFirstEnumerator:
    def __init__(self, trajs: Iterable, interval, verifier):
        self.ts_grouped_traj = groupby(trajs, attrgetter("begin"))
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

    def _live_group(self, t):
        """Ids whose end event at ``t`` is still the live one.

        The raw ``one_pass_merge`` stream delivers an object as several pieces
        (one per index cell) that touch, overlap or nest, so an object can be
        re-inserted while an occupancy is still open. Extending that occupancy
        queues a second end event and leaves the first one stale: firing it
        would end an object that is still present, and the second event would
        then look up an id that was already erased.
        """
        group = []
        for tid in self.group_pop():
            live = self.playground.get(tid)
            if live is not None and live.end == t:
                group.append(tid)
        return group

    def _yield_until(self, ts, pre_insert):
        while self.etq and (t := self.etq[0][0]) <= ts:
            group = self._live_group(t)
            if t < ts:
                if not group:
                    continue  # every event at `t` was stale
                yield from self.verify(t, [self.playground[tid] for tid in group])
            else:  # t == ts
                t_candi = []
                for tid in group:
                    if (revising := pre_insert.pop(tid, None)) is None:
                        t_candi.append(self.playground[tid])
                    else:
                        # The object continues: extend the live occupancy to the
                        # new end and queue that end instead.
                        if (live := self.playground.get(tid)) is not None:
                            live.end = max(live.end, revising.end)
                        self.etq_push((revising.end, tid))
                if t_candi:
                    yield from self.verify(ts, t_candi)
                return

    def _update(self, pre_insert):
        for tra_id, tra in pre_insert.items():
            live = self.playground.get(tra_id)
            if live is None:
                self.playground[tra_id] = tra
                self.etq_push((tra.end, tra_id))
            elif tra.end > live.end:
                # A later piece of an occupancy that is still open: union it in.
                live.end = tra.end
                self.etq_push((tra.end, tra_id))
            # else: the piece is already inside the live run, nothing to do
        return self.etq[0][0] if self.etq else None

    def __iter__(self):
        begin, finish = self.interval
        self._init_state()
        if self.playground:
            for tid, traj in self.playground.items():
                self.etq_push((traj.end, tid))
            next_end = self.etq[0][0]
        else:
            # No object is present at the interval start. Objects appearing
            # later still co-move, so keep going instead of returning empty.
            next_end = None

        for t, group in self.ts_grouped_traj:
            if t > finish:
                break
            pre_insert = {traj.id: traj for traj in group}
            if next_end is not None and next_end <= t:
                yield from self._yield_until(t, pre_insert)
            next_end = self._update(pre_insert)

        # Half-open segments: an interval ending exactly at `finish` still
        # produces a pattern whose inclusive end is `finish - 1`.
        if self.etq:
            yield from self._yield_until(finish, {})
        if self.etq:
            # Intervals that run past the query interval are capped at `finish`.
            # The queue can still hold stale events (an object that was extended
            # or has already ended); drop them exactly like _live_group().
            yield from self.verify(
                finish,
                [
                    self.playground[tid]
                    for t, tid in self.etq
                    if (live := self.playground.get(tid)) is not None
                    and live.end == t
                ],
            )


class MaxDurMultiPass(MaxDurFirstEnumerator):
    def __init__(self, trajs: Iterable, interval, verifier):
        super().__init__(trajs, interval, verifier)

    def _update(self, group):
        for tra in group:
            tid = tra.id
            live = self.playground.get(tid)
            if live is None:
                self.playground[tid] = tra
                # Segments are half-open [begin, end): the end event is `end`
                # itself, and yield_co_move turns it back into an inclusive end.
                self.etq_push((tra.end, tid))
            elif tra.end > live.end:
                # Union a later piece of an occupancy that is still open; the
                # event for the old end is stale and gets dropped by the drain.
                live.end = tra.end
                self.etq_push((tra.end, tid))
        return self.etq[0][0] if self.etq else None

    @staticmethod
    def _index_group(group):
        """A group indexed by object id, with pieces of one object unioned."""
        pre_insert = {}
        for traj in group:
            live = pre_insert.get(traj.id)
            if live is None:
                pre_insert[traj.id] = traj
            elif traj.end > live.end:
                live.end = traj.end
        return pre_insert

    def __iter__(self):
        begin, finish = self.interval
        self._init_state()
        if self.playground:
            for tid, traj in self.playground.items():
                self.etq_push((traj.end, tid))
            next_end = self.etq[0][0]
        else:
            # No object is present at the interval start. Objects appearing
            # later still co-move, so keep going instead of returning empty.
            next_end = None

        for t, group in self.ts_grouped_traj:
            if t > finish:
                break
            pre_insert = self._index_group(group)
            while self.etq and (et := self.etq[0][0]) <= t:
                end_g = [
                    tid
                    for tid in self.group_pop()
                    if (live := self.playground.get(tid)) is not None
                    and live.end == et
                ]
                if not end_g:
                    continue  # every event at `et` was stale
                t_candi = []
                for tid in end_g:
                    # Only a piece that begins exactly when this occupancy ends
                    # is that occupancy's continuation — the tail of a
                    # window-cut head (stride_merge_windowed splits an object at
                    # the window boundary and flags the head with `cut`). A piece
                    # that begins later is a new run after a gap, so the
                    # occupancy really does end here.
                    revising = pre_insert.get(tid)
                    if revising is not None and revising.begin == et:
                        if (live := self.playground.get(tid)) is not None:
                            live.end = max(live.end, revising.end)
                        self.etq_push((revising.end, tid))
                        del pre_insert[tid]
                    else:
                        t_candi.append(self.playground[tid])
                if t_candi:
                    yield from self.verify(et, t_candi)
            next_end = self._update(pre_insert.values())

        if next_end is not None and next_end <= finish:
            while self.etq and (et := self.etq[0][0]) <= finish:
                end_g = [
                    tid
                    for tid in self.group_pop()
                    if (live := self.playground.get(tid)) is not None
                    and live.end == et
                ]
                if not end_g:
                    continue  # every event at `et` was stale
                yield from self.verify(et, [self.playground[tid] for tid in end_g])
        if self.etq:
            # Intervals that run past the query interval are capped at `finish`;
            # stale events are dropped exactly like the drain above.
            yield from self.verify(
                finish,
                [
                    self.playground[tid]
                    for t, tid in self.etq
                    if (live := self.playground.get(tid)) is not None
                    and live.end == t
                ],
            )


def max_dur_enumerate(trajs, labels: Mapping, dur, interval, *, mtd="vanilla"):
    verifier = partial(yield_co_move, dur, labels)
    match mtd:
        # The raw per-cell stream (one_pass) hands an object as several pieces,
        # so the one-pass enumerator unions them. Every coalescing merge
        # (vanilla/stride/stride_stream/stride_win) hands one segment per object
        # run, which is what the multi-pass enumerator expects.
        case "one" | "one_pass" | "streaming":
            return MaxDurationOnePass(trajs, interval, verifier)
        case "multi" | "vanilla" | "stored" | "stride" | "stride_stream" | "ss" \
                | "stride_win" | "win":
            return MaxDurMultiPass(trajs, interval, verifier)
        case _:
            raise ValueError(f"mtd {mtd} not supported")
