import cppyy

from my_numb.search.index_based.builder import cppyy_init, load_segments, build_rest
from my_numb.utilities.data_processing import traj_df2np
from my_numb.utilities.iterate_tools import trajectory_manger_start_end
from utilities.dataset import load_fake


def test_traj_extract():
    cppyy_init(cppyy)
    border_m = {}
    reg = [0, 400, 0, 500]
    life_border = [3, 8, 10, 21]
    for c in (0, 1):
        border_m[c] = {'reg_border': reg, 'tempo_stride': 60, 'life_border': life_border}
    space = 4, 5

    data, cols = load_fake(False)
    trajs = traj_df2np(data, cols)
    segs = load_segments(trajs, 'test', border_m, '.', space)
    traj_meta, query_run = build_rest(trajs, segs, border_m)
    query_run(0, [0, 401, 0, 501], 0, (0, 100), 1)
    trajs = list(trajectory_manger_start_end(traj_meta, 1))
    # trajs = list(trajectory_manger_factory(0)(traj_meta, sit))
    ans =[(1, 5, 1, 0), (4, 8, 3, 0), (5, 8, 1, 0), (5, 9, 2, 0), (8, 12, 1, 0), (8, 12, 3, 0), (9, 11, 2, 0), (11, 15, 2, 0), (12, 19, 1, 0), (12, 14, 3, 0), (14, 18, 3, 0), (15, 19, 2, 0), (18, 23, 3, 0), (19, 30, 1, 0), (19, 22, 2, 0), (22, 28, 2, 0), (23, 26, 3, 0), (26, 30, 3, 0), (28, 32, 2, 0)]
    assert trajs == ans