from configs import cfg
from search.index_based.base import BaseSliding, base_enumerate, base_maintainer
from search.index_based.max_obj import MaxObjNumEnumerator
from search.rest import state_sliding, absorb, add_interval
from search.verifier import obj_verify
from test.index_test_helper import grid_spt_tempo_idx_fake_data, query4test
from utilities.trajectory import Trajectory, TrajectoryIntervalSeg
from functools import partial


def test_absorb():
    # absorb() now returns half-open [begin, end) segments, matching the
    # canonical internal convention.
    segs = [[9, 20, 1, 5, 29, 30, 6, 9, 20, 23, 27, 28, 26, 27, 28, 29],
            [5, 6, 3, 5, 8, 13]]
    ans = [[1, 5], [6, 23], [26, 30], [8, 13]]
    trajs = {i: Trajectory(i, 0, seg) for i, seg in enumerate(segs)}
    res = [[traj.begin, traj.end] for traj in absorb(trajs, 4)]
    assert res == ans


def test_add_interval_coalesces_on_insert():
    # Contiguous pieces fed in ascending begin order must collapse to a single
    # stored interval, so a long-lived trajectory keeps O(1) state instead of
    # accumulating one endpoint pair per incoming piece.
    store = {}
    for k in range(1000):
        add_interval(store, 1, 0, k, k + 1)
    assert store[1].seg == [0, 1000]

    # Touching intervals extend; a gap starts a new disjoint interval.
    store = {}
    for beg, end in [(0, 10), (10, 20), (25, 30), (30, 40)]:
        add_interval(store, 1, 0, beg, end)
    assert store[1].seg == [0, 20, 25, 40]

    # Incremental coalescing must agree with raw append + absorb.
    inc, raw = {}, {}
    pieces = [(0, 5), (5, 12), (20, 30), (30, 31), (40, 60)]
    for beg, end in pieces:
        add_interval(inc, 1, 0, beg, end)
        raw.setdefault(1, Trajectory(1, 0, [])).seg += [beg, end]
    a = [(s.id, s.begin, s.end) for s in absorb(inc, 4)]
    b = [(s.id, s.begin, s.end) for s in absorb(raw, 4)]
    assert a == b


class TestBaseSliding(object):
    #     1                  21
    # 0:  |------------------|  label: 0
    #     1    5
    # 1:  |----|                label: 1
    #      2      10
    # 2:   |------|             label: 0
    #          5      16
    # 3:       |------|         label: 1
    #                15      21
    # 4:             |-------|  label: 0
    #                15       22
    # 5:             |--------| label: 1
    traj_len = [[1, 21], [1, 5], [2, 10], [5, 16], [15, 21], [15, 22]]
    trajs = [TrajectoryIntervalSeg(i, itv[0], i & 1, itv[1] - itv[0]) for i, itv in enumerate(traj_len)]

    interval = 0, 20
    duration = 4
    labels = {0: 1, 1: 1}
    label_verifier = partial(obj_verify, labels)

    def setup(self, *, method='max_obj'):
        match method:
            case 'max_obj':
                gen = MaxObjNumEnumerator
            case 'base':
                gen = BaseSliding
            case _:
                raise ValueError(f"Unexpected mtd value: {method}")
        self.sliding = gen(self.trajs, self.interval, self.duration, self.labels)

    def test_base_sliding(self):
        self.setup()
        ans = [((0, 1), [1, 4]), ((0, 2, 3), [5, 8]), ((0, 2, 3), [6, 9]), ((0, 3), [10, 13]),
               ((0, 3), [12, 15]), ((0, 4, 5), [15, 18]), ((0, 4, 5), [17, 20])]
        res = [(tuple(pat.labels), pat.interval) for pat in self.sliding]
        assert res == ans

    def test_final(self):
        self.setup()
        ans = [((0,1), [1,4]), ((0, 2, 3), [5, 9]), ((0, 3), [5, 15]), ((0, 4, 5), [15, 20])]
        res = [(tuple(ids), [b, s]) for ids, b, s in state_sliding(self.sliding, self.label_verifier, base_maintainer)]
        assert res == ans


def test_base_search():
    cfg.merge_from_file('../configs/region_base.yml')
    spt_tempo = grid_spt_tempo_idx_fake_data(cfg)

    test1, test2 = query4test()
    res = [(frozenset(ids), s, e) for ids, s, e in base_enumerate(spt_tempo, *test1)]
    assert res == [(frozenset([2, 3, 1]), 5, 10), (frozenset([3, 1]), 4, 10), (frozenset([1]), 1, 10)]

    res = [(frozenset(ids), s, e) for ids, s, e in base_enumerate(spt_tempo, *test2)]
    assert res == [(frozenset([5, 1, 3]), 9, 18), (frozenset([2, 4, 5, 3]), 11, 20), (frozenset([2, 4, 3]), 11, 22)]

