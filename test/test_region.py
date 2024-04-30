import numpy as np

from indices.region import GridRegion, Out3DRegion
from test.region_test_vars import *
from utilities.trajectory import TrajMeta


class TestGrid:
    grid = GridRegion()
    for j in range(len(border[0])-1):
        for i in range(len(border[1])-1):
            grid.territory[i, j] = (border[0][j], border[1][i])

    def test_where_contain(self):
        Q = [(8, 16), (31, 63), (8, 63), (31, 16), (18, 37), (23, 49), (25, 71)]
        A = [(8, 16), (30, 60), (8, 60), (30, 16), (18, 36), (22, 48), (24, 68)]
        for q, a in zip(Q, A):
            assert self.grid.where_contain(q) == a

    def test_where_intersect(self):
        Q = [(8, 31, 16, 70), [8, 30, 16, 68], (11, 29, 17, 67)]
        A = [self.grid.territory, self.grid.territory, self.grid.territory[:-1, 1:-1]]
        for q, a in zip(Q, A):
            assert (self.grid.where_intersect(q) == a.flatten()).all()

    def test_perhaps_intersect(self):
        Q = [(8, 30, 16, 71), (9, 29, 16, 71), (8, 28, 19, 70), (15, 21, 25, 55), (15, 20, 25, 54),
             (12, 13, 21, 23), (12, 13, 20, 22), (12, 12, 20, 23), (12, 13, 20, 23)]
        world = self.grid.territory
        A = [(np.array([]), world), (world[:, 0], world[:, 1:-1]),
             (np.concatenate([world[0, :-1], world[1:, -2]]), world[1:, :-2]),
             (np.concatenate([world[2, 4:7].flatten(), world[2:10, 3].flatten()]), world[3:10, 4:7]),
             (np.concatenate([world[[2, 9], 4:6].flatten(), world[2:10, [3, 6]].flatten()]), world[3:9, 4:6]),
             (world[1:2, 2], np.array([])), (world[1:2, 2], np.array([])), (world[1:2, 2], np.array([])),
             (np.array([]), world[1:2, 2])]

        for q, (a_cand, a_prob) in zip(Q, A):
            cand, prob = self.grid.perhaps_intersect(q)
            assert set(prob) == set(a_prob.flatten())
            assert set(cand) == set(a_cand.flatten())


class Test3DRegion:
    index = Out3DRegion(zip(keys, objs))

    def test_where_contain(self):
        for key, entry in zip(keys, objs):
            assert self.index.where_contain(key) is entry

    def test_where_intersect(self):
        res = list(self.index.where_intersect([10, 100, 20, 30, 16, 70]))
        assert len(res) == 4

    def test_add(self):
        objs = [list(range(5)), {i: str(i) for i in range(4)}]
        keys = [TrajMeta(102, (19, 47)), TrajMeta(5, (17, 55))]
        for key, entry in zip(keys, objs):
            self.index.add(key, entry)
            assert self.index.where_contain(key) is entry
