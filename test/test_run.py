import heapq

import numpy as np

from configs import cfg
from search.rest import yield_co_move
from search.verifier import candidate_verified_queue, verify_seg
from search.max_dur_first import max_dur_first_search, MaxDurFirst
from test.index_test_helper import grid_spt_tempo_idx_fake_data, query4test
from utilities.box2D import Box2D
from utilities.trajectory import TrajectorySequenceSeg, TrajectoryIntervalSeg, BasicTrajectorySeg


class TestTempSpatialIndex:
    tempo_spatial = grid_spt_tempo_idx_fake_data(cfg)

    def test_build_tempo_spatial_index(self):
        cands, probs = self.tempo_spatial[0].where_intersect([((0, 1, 0, 1), (10, 30)), (0, 10)])
        for can in cands:
            for entry in can:
                match entry:
                    case TrajectorySequenceSeg(2, _, 0, seg):
                        ans = np.array([[0, 20], [20, 30], [60, 50], [90, 78]])
                        assert (seg == ans).all()
                    case TrajectorySequenceSeg(1, _, 0, seg):
                        ans = np.array([[0, 0], [20, 10], [50, 13], [70, 40]])
                        assert (seg == ans).all()
                    case _:
                        assert False

    def test_one_pass_search(self):
        test1, test2 = query4test()

        res = list(max_dur_first_search(self.tempo_spatial, *test1))
        assert res == [([1, 3, 2], 5, 10), ([1, 3], 4, 10), ([1], 1, 10)]

        res = list(max_dur_first_search(self.tempo_spatial, *test2))
        assert res == [([1, 3, 5], 9, 18), ([3, 5, 4, 2], 11, 20), ([3, 4, 2], 11, 22)]


def test_verify_seg():
    region = Box2D((11, 111, 11, 111))
    # start and end are out
    points1 = np.array([[10, 67], [13, 83], [19, 92], [23, 113], [41, 121], [59, 102], [81, 83], [113, 75], [101, 49],
                        [54, 19], [31, 7], [27, 27], [19, 45], [13, 51], [9, 55]])
    seg1 = TrajectorySequenceSeg(1, 0, 0, points1)
    res = verify_seg(seg1, region, 3)
    assert len(res) == 1
    assert res == [TrajectoryIntervalSeg(1, 11, 0, 3)]
    res = verify_seg(seg1, region, 2)
    assert len(res) == 4
    assert [seg.begin for seg in res] == [1, 5, 8, 11]

    seg2 = TrajectorySequenceSeg(1, 0, 0, points1[1:-1])
    res = verify_seg(seg2, region, 3)
    assert len(res) == 2
    assert [(seg.begin, seg.len) for seg in res] == [(0, 2), (10, 3)]


def test_candidate_verified_queue():
    x = np.linspace(0, np.pi, 100)
    Y = [np.cos(x * 20) * 10, np.sin((x + np.pi / 4) * 20) * 10]
    all_points = [np.vsplit(np.vstack((x, y)).T, 10) for y in Y]
    cand = [[TrajectorySequenceSeg(i, j * 10, 0, obj_ps) for j, obj_ps in enumerate(ind)] for i, ind in
            enumerate(all_points)]

    region = Box2D((0, 1.5 * np.pi, 0, 10))
    data = list(candidate_verified_queue(heapq.merge(*cand), region, 5))
    assert len(list(data)) == 30


def test_yield_co_move():
    duration = 5

    #     1
    # 0:  |-------------------| label: 0
    #     1
    # 1:  |-------------------| label: 2
    #        4
    # 2:     |----------------| label: 1 *
    #           8
    # 3:        |-------------| label: 1 *
    #            9
    # 4:         |------------| label: 2
    #            10
    # 5:          |-----------| label: 0 *
    #             10
    # 6:          |-----------| label: 0
    #             11
    # 7:           |----------| label: 0 x

    obj_info = [(1, 0), (1, 2), (4, 1), (8, 1), (9, 2), (10, 0), (10, 0), (11, 0)]
    obj_info = [BasicTrajectorySeg(i, *info)for i, info in enumerate(obj_info)]
    active_space = dict(enumerate(obj_info))
    ts = 15
    trajs = [active_space[i] for i in (2, 3, 5)]

    label_quizzes = [{0: 2, 1: 2, 2: 1}, {0: 3, 1: 1, 2: 1}, {0: 1, 1: 1, 2: 1}, {0: 1, 1: 0, 2: 1}]

    ans = [1, 1, 4, 4]
    for labels, a in zip(label_quizzes, ans):
        test_map = dict(active_space)
        res = list(yield_co_move(duration, labels, test_map, ts, trajs))
        assert len(res) == a


def test_sequential_search():
    traj1 = [(1, 0, 18), (4, 1, 12), (8, 1, 9), (9, 2, 10), (9, 1, 10),
             (9, 1, 7), (16, 0, 3), (16, 1, 3), (19, 0, 2), (20, 0, 2), (20, 1, 5)]
    traj2 = [(16, 1, 3), (17, 1, 3), (19, 2, 3)]
    traj3 = [(19, 0, 4), (19, 1, 5)]
    traj_total = heapq.merge((TrajectoryIntervalSeg(tid, *info) for tid, info in enumerate(traj1)),
                             (TrajectoryIntervalSeg(tid, *info) for tid, info in enumerate(traj2, 1)),
                             (TrajectoryIntervalSeg(tid, *info) for tid, info in enumerate(traj3, 6)))

    ans = [(6, 16, 1), (8, 19, 3), (9, 20, 1), (11, 21, 1), (11, 22, 5)]
    searcher = MaxDurFirst(traj_total, (3, 21), lambda am, end, cond: [(len(am), end, len(cond))])
    query = list(searcher)
    assert ans == query
