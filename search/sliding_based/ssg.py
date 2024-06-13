from collections import deque
from collections.abc import MutableMapping
from copy import deepcopy
from dataclasses import dataclass, field
from heapq import merge
from itertools import groupby
from operator import itemgetter
from typing import Self


@dataclass
class MarkedFrames:
    frames: deque[int] = field(default_factory=deque)
    markers: deque[int] = field(default_factory=deque)

    def remove(self, fid: int) -> None:
        if fid == self.frames[0]:
            self.frames.popleft()
            if fid == self.markers[0]:
                self.markers.popleft()

    def clear(self) -> None:
        self.frames.clear()
        self.markers.clear()

    def first_marked(self) -> int:
        return self.markers[0]

    def add_frame(self, fid: int, mark: bool = False) -> Self:
        self.frames.append(fid)
        if mark:
            self.markers.append(fid)
        return self

    def merge(self, other: Self) -> Self:
        if other.markers != self.markers:
            self.markers = deque(map(itemgetter(0), groupby(merge(other.markers, self.markers))))
        return self

    def __bool__(self) -> bool:
        return bool(self.markers)


@dataclass
class State:
    objs: frozenset
    flag: int = 0
    marked_frames: MarkedFrames = None
    next: list[Self] = field(default_factory=list)
    prev: Self = None

    def is_valid(self) -> bool:
        return bool(self.marked_frames)

    def first_marked_frame(self) -> int:
        # FIXME: should test if it's valid first
        return self.marked_frames.first_marked()

    def reset_prev(self) -> None:
        if self.prev is not None:
            self.prev.next.remove(self)
            self.prev = None

    def add_next(self, state):
        self.next.append(state)
        state.prev = self

    def all_add(self, fid: int) -> None:
        """
        add frame to itself and all its outgoing state. use DFS method.
        :param fid:
        :return:
        """
        self.marked_frames.add_frame(fid)
        fifo: list[State] = self.next[:]
        while fifo:
            state = fifo.pop()
            state.marked_frames.add_frame(fid)
            fifo.extend(state.next)


class StateGraph:
    def __init__(self, win_size: int):
        self.win_size = win_size
        self.principle_sates: deque[State] = deque(maxlen=win_size)
        self.oss_m: MutableMapping[frozenset, State] = {}
        self.add_plans: list[tuple[State, State]] = []

    def prune_state(self, state: State, fid: int) -> bool:
        """

        :param state:
        :param fid:
        :return: whether the state should be removed
        """
        state.marked_frames.remove(fid)
        if not state.is_valid():
            del self.oss_m[state.objs]
            if (prev := state.prev) is not None:
                state.reset_prev()
                for s in state.next:
                    prev.add_next(s)
            else:
                for s in state.next:
                    s.prev = None
            for s in state.next:
                self.update_edges(s, fid)
            return True
        else:
            return False

    def visit_state(self, state: State, fid: int, new_ps: State, to_compute: bool):
        if state.flag == fid:
            return
        state.flag = fid
        if not self.prune_state(state, fid - self.win_size) and to_compute:
            inter = state.objs & new_ps.objs
            if inter:
                if len(inter) == len(state.objs):
                    state.all_add(fid)
                    return
                else:
                    ss, is_new = self.lookup_or_create(inter)
                    if is_new:
                        ss.marked_frames = deepcopy(state.marked_frames)
                        self.add_plans.append((state, ss))
                        ss.marked_frames.add_frame(fid)
                    else:
                        ss.marked_frames.merge(state.marked_frames)
                    to_compute = True
            else:
                to_compute = False
        for next_state in state.next:
            self.visit_state(next_state, fid, new_ps, to_compute)

    def update_edges(self, state: State, fid: int):
        """
        :param state: root state
        :param fid: expired frame
        :return:
        """
        if not self.prune_state(state, fid) and state.prev is None:
            fid_1st_marked = state.first_marked_frame()
            ps = self.principle_sates[fid_1st_marked - fid - 1]
            if ps is not state:
                state.reset_prev()
                if state not in self.principle_sates:
                    ps.add_next(state)

    def construct_maintain(self, fid: int, objs: frozenset):
        if len(self.principle_sates) == self.win_size:
            abandoned_sate = self.principle_sates.popleft()
            self.update_edges(abandoned_sate, fid - self.win_size)
        new_ps, is_new = self.lookup_or_create(objs)
        if is_new:
            new_ps.marked_frames = MarkedFrames()
        new_ps.marked_frames.add_frame(fid, True)
        for ps in self.principle_sates:
            self.visit_state(ps, fid, new_ps, True)
        for s, next_s in self.add_plans:
            s.add_next(next_s)
        self.add_plans.clear()
        new_ps.reset_prev()  # maybe promoted
        self.principle_sates.append(new_ps)

    def lookup_or_create(self, objs: frozenset) -> tuple[State, bool]:
        if (state := self.oss_m.get(objs, None)) is not None:
            return state, False
        else:
            new_state = State(objs)
            self.oss_m[objs] = new_state
            return new_state, True

    @property
    def states(self) -> frozenset:
        for objs, s in self.oss_m.items():
            yield objs, list(s.marked_frames.frames)

