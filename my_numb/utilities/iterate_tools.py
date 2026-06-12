import numpy as np
from numba import njit, jit_module

from my_numb.utilities.data_transform import unpack_i32
from my_numb.utilities.bisect_right import bisect_right
import cppyy


@njit(cache=True)
def segment_end(segs, s_start, lo, hi, ):
    return segs[bisect_right(segs, s_start, lo, hi)]

def trajectory_manger_points(meta, sure):
    package, points, segs = meta
    while has_next(sure):
        start, tidx = pop_next(sure)
        t_begin, tid, label, tp_start, so_start = package[tidx]
        s_start = start - t_begin
        yield start, tid, label, points[tp_start + s_start:tp_start + segment_end(segs, s_start, so_start, package[tidx+1, 4])]


def trajectory_manger_start_end(meta, sure):
    package, _, segs = meta
    while has_next(sure):
        start, tidx = pop_next(sure)
        t_begin, tid, label, tp_start, so_start = package[tidx]
        yield start, tid, label, segment_end(segs, start - t_begin, so_start, package[tidx+1, 4]) + t_begin


def has_next(sure):
    return cppyy.gbl.iter_has_next(sure)

def pop_next(sure):
    raw = cppyy.gbl.iter_next(sure)
    return unpack_i32(raw)

jit_module(nopython=True)


if __name__ == '__main__':
    from my_numb.search.index_based.builder import load_segments, build_rest
    from my_numb.utilities.data_processing import traj_df2np
    from utilities.dataset import load_fake
    border_m = {}
    reg = [0, 400, 0, 500]
    life_border = [3, 8, 10, 21]
    for c in (0, 1):
        border_m[c] = {'reg_border': reg, 'tempo_stride': 60, 'life_border': life_border}
    space = 4, 5

    data, cols = load_fake(False)
    traj_meta = traj_df2np(data, cols)
    seg_meta = load_segments(traj_meta, 'test', border_m, '.', space)
    query_run = build_rest(traj_meta, seg_meta, border_m)
    query_run(0, [0, 400, 0, 500], 0, (0, 100), 1)
    trajs = list(trajectory_manger_start_end((*traj_meta, seg_meta), 1))
    # trajs = list(trajectory_manger_factory(0)(traj_meta, sit))
    print(trajs)