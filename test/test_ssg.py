import itertools

from search.sliding_based.ssg import MarkedFrames, State, StateGraph


class TestMarkedFrames:
    def setup(self):
        self.frames = MarkedFrames()
        self.frames.frames.extend([1, 3, 4, 6, 8, 9])
        self.frames.markers.extend([1, 3, 8])

    def test_add_frame(self):
        self.frames = MarkedFrames()
        self.frames.add_frame(0, True)
        self.frames.add_frame(1)
        self.frames.add_frame(3, True)
        self.frames.add_frame(4)
        assert list(self.frames.frames) == [0, 1, 3, 4]
        assert list(self.frames.markers) == [0, 3]

    def test_remove(self):
        self.setup()
        self.frames.remove(1)
        assert self.frames.frames[0] == 3 and len(self.frames.markers) == 2
        self.frames.remove(4)
        assert self.frames.frames[0] == 3 and len(self.frames.markers) == 2

    def test_first_marked(self):
        self.setup()
        assert self.frames.first_marked() == 1
        for i in [1, 3]:
            self.frames.remove(i)
        assert self.frames.first_marked() == 8

    def test_valid(self):
        self.setup()
        assert self.frames
        for i in range(len(self.frames.frames) - 1):
            self.frames.remove(self.frames.frames[0])
        assert not self.frames

    def test_merge(self):
        self.setup()
        other = MarkedFrames()
        other.markers.extend([3,6,8,9])
        other.frames.extend(range(3, 10))
        self.frames.merge(other)
        assert list(self.frames.markers) == [1, 3, 6, 8, 9]


class TestState:

    def setup(self):
        self.root = State(frozenset([1, 2, 3]))

    def test_is_valid(self):
        self.setup()
        self.root.marked_frames.add_frame(1)
        assert not self.root.is_valid()
        self.root.marked_frames.add_frame(3, True)
        assert self.root.is_valid()

    def test_first_marked(self):
        self.setup()
        for i in range(40):
            self.root.marked_frames.add_frame(i)
        self.root.marked_frames.add_frame(42, True)
        assert self.root.first_marked_frame() == 42

    def test_add_next(self):
        self.setup()
        states = [State(frozenset([i])) for i in range(40)]
        for s in states:
            self.root.add_next(s)
        for s in self.root.next:
            assert s.prev == self.root

    def test_reset_prev(self):
        self.setup()
        state = State(frozenset([0]))
        for i in range(10):
            state.add_next(State(frozenset([i])))
        state.add_next(self.root)
        for i in range(10, 20):
            state.add_next(State(frozenset([i])))
        assert self.root.prev == state
        assert self.root in state.next
        self.root.reset_prev()
        assert self.root.prev is None
        assert self.root not in state.next

    def test_all_children_add(self):
        self.setup()
        num = 40
        level = 4
        states = random_tree(self.root, num, level)
        self.root.all_add(111)
        for state in states:
            assert state.marked_frames.frames[0] == 111


def random_tree(root, node_num, level):
    import random
    import numpy as np

    breaks = random_partition(node_num, level)
    pos = np.zeros(level + 1, dtype=int)
    np.cumsum(breaks, out=pos[1:])

    states = [State(frozenset([i])) for i in range(node_num)]
    state_it = iter(states)
    for i, state_num in enumerate(breaks):
        for state in itertools.islice(state_it, state_num):
            if i > 0:
                states[random.randint(pos[i - 1], pos[i] - 1)].add_next(state)
            else:
                root.add_next(state)
    return states


def random_partition(element_num, group_num):
    import random
    breaks = []
    remain = element_num
    for i in range(group_num - 1):
        breaks.append(random.randint(1, remain + i - group_num + 1))
        remain -= breaks[-1]
    breaks.append(remain)
    return breaks


class TestStateGraph:
    objs_series = [[1, 2], [1, 2, 3, 4], [1, 2, 3, 5, 6], [1, 2, 3, 4, 6], [1, 2, 3, 5]]
    frames = list(enumerate(map(frozenset, objs_series)))

    def test_lookup_or_create(self):
        self.init_state([])
        for fid, objs in self.frames[:3]:
            ns, is_new = self.state_graph.lookup_or_create(objs)
            assert ns.objs == objs and not is_new
        ns, is_new = self.state_graph.lookup_or_create(self.frames[3][1])
        assert ns.objs == self.frames[3][1] and is_new

    def test_prune_state(self):
        derived_states, states = self.init_state([[1, 2, 3], [2, 3]])
        derived_state = derived_states[0]
        derived_state.marked_frames.add_frame(0, True).add_frame(1).add_frame(2)

        states[0].add_next(derived_state)

        ab_state = self.state_graph.principle_sates.popleft()
        assert states[0].objs in self.state_graph.oss_m
        self.state_graph.prune_state(ab_state, 0)
        assert states[0].objs not in self.state_graph.oss_m
        assert derived_state.objs not in self.state_graph.oss_m

        states[1].add_next(derived_state)
        derived_states[1].marked_frames.add_frame(1, True).add_frame(2)
        derived_state.add_next(derived_states[1])
        self.state_graph.oss_m[derived_state.objs] = derived_state
        self.state_graph.update_edges(derived_state, 0)
        assert derived_state.objs not in self.state_graph.oss_m
        assert derived_states[1].prev == states[1]

    def test_update_edges(self):
        derived_states, states = self.init_state([[1, 2, 3], [2, 3]])
        derived_states[0].marked_frames.add_frame(0, True).add_frame(1).add_frame(2, True)
        derived_states[1].marked_frames.add_frame(0, True).add_frame(2)
        states[0].add_next(derived_states[1])
        derived_states[1].add_next(derived_states[0])
        self.state_graph.principle_sates.popleft()
        self.state_graph.update_edges(states[0], 0)
        assert derived_states[0].objs in self.state_graph.oss_m
        assert derived_states[0].prev == states[2]
        assert derived_states[1].objs not in self.state_graph.oss_m

    def init_state(self, derived_objs):
        self.state_graph = StateGraph(3)
        states = [State(frame[1]) for frame in self.frames]
        self.state_graph.principle_sates.extend(states[:3])
        self.state_graph.oss_m.update({state.objs: state for state in states[:3]})
        derived_states = [State(frozenset(objs)) for objs in derived_objs]
        for s in derived_states:
            s.marked_frames = MarkedFrames()
        self.state_graph.oss_m.update((s.objs, s) for s in derived_states)
        for i, s in enumerate(states[:3]):
            s.marked_frames = MarkedFrames()
            s.marked_frames.add_frame(i, True)
        for i in [1, 2]:
            states[0].marked_frames.add_frame(i)
        return derived_states, states

    def test_visit_state(self):
        d_s, states = self.init_state([[1, 2, 3, 6], [1, 2, 3], [1, 3], [1], [3], [2,3,6],
                                       [2,3,5,6], [3, 5, 6], [5, 6]])
        parents_pos = [[2], [2, 0], [2, 0, 0], [2, 0, 0, 0], [2, 0, 0, 0], [2, 0],
                       [2], [2, 1], [2, 1, 0]]
        frames = [[(2, True)], [(2, True), (3, False)], [(2, True), (3, False)], [(2, True), (4, True), (5, False)],
                  [(2, True), (4, False), (5, True)], [(2, True), (5, True)], [(2, True), (6, False)],
                  [(2, True), (6, False)], [(2, True), (6, False)]]
        for ds, mfs in zip(d_s, frames):
            for mf in mfs:
                ds.marked_frames.add_frame(*mf)
        for ds, pps in zip(d_s, parents_pos):
            parent = states[pps[0]]
            for pos in pps[1:]:
                parent = parent.next[pos]
            parent.next.append(ds)

        new_ps, _ = self.state_graph.lookup_or_create(frozenset([1,3,4,5,6]))
        new_ps.marked_frames = MarkedFrames()
        new_ps.marked_frames.add_frame(7, True)
        self.state_graph.visit_state(states[2], 7, new_ps,  True)
        assert len(self.state_graph.oss_m) == 16
        assert set(self.state_graph.oss_m).issuperset({frozenset([1,3,5,6]), frozenset([1, 3, 6]), frozenset([3, 6])})

    def test_construct_maintain(self):
        ans_obs = {1: [1,2], 2: [1,2,3,4], 3: [1,2,3,5,6], 4: [1,2,3], 5: [1,2,3,4,6], 6: [1,2,3,6], 7: [1,2,3,5]}
        ans_obs = {k: frozenset(val) for k, val in ans_obs.items()}
        ans_frames = [[(1,[0])], [(1, [0,1]), (2, [1])], [(1, [0,1,2]), (2, [1]), (3, [2]), (4, [1,2])],
                      [(1, [0,1,2,3]), (2, [1,3]), (3, [2]),(4,[1,2,3]),(5,[3]),(6,[2,3])],
                      [(2,[1,3]), (2,[1,3]), (3, [2]), (4,[1,2,3,4]), (5,[3]), (6,[2,3]),(7,[4])]]
        ans_ref = [{ans_obs[o[0]]: o[1] for o in frame} for frame in ans_frames]
        self.state_graph = StateGraph(4)
        for (fid, objs), ans in zip(self.frames, ans_ref):
            self.state_graph.construct_maintain(fid, objs)
            assert dict(self.state_graph.states) == ans
