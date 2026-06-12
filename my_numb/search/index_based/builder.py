import os

import cppyy
import numpy as np
from numba import njit, float64, int32, int64, jit_module
from numba.typed import Dict
import cppyy.numba_ext

from my_numb.cxx_store import LIB_DIR, LIB_NAME, HEADER
from my_numb.utilities.data_processing import traj_df2np
from my_numb.utilities.data_transform import unpack_i32, pack_i16, pack_i32
from scripts.border import concatenate_stride
from utilities.dataset import load_yolo_for, load_fake


def grid_meta(border_stride):
    meta = np.empty(6, dtype=np.int32)
    meta[:2] = border_stride[:4:3]  # lower_border, x y
    meta[2:4] = border_stride[2:6:3]  # stride, x y
    box = border_stride[np.array([0, 1, 3, 4])].reshape(-1, 2)
    box = (box[:, 1] - box[:, 0]) // meta[2:4] + 1
    meta[4] = box[0]  # col_num
    meta[5] = box[0] * box[1]  # size
    return meta


def where_is(points, meta):
    # meta: (lower_border_x, lower_border_y, stride_x, stride_y, col_num, size)
    pos = (points - meta[:2]) // meta[2:4]
    return pos[:, 1] * meta[4] + pos[:, 0]


def partition_traj(trajs, bs_m):
    # offsets |-pos_0-|-pos_1-|       ...   |-pos_len(trajectory)-|
    #           /           \___________           \______________________________
    #          /                        \                                         \
    # seg_ls   |-seg_0,0-| ... |-seg_0,k-|-seg_1,0-| ... |-seg_1,m-| ... | last seg|
    # seg_i,j is the j-th break point of the i-th trajectory
    seg_ls = []
    grid_m = dict()
    for c, bs in bs_m.items():
        grid_m[c] = grid_meta(bs)
    package, points = trajs
    cls, point_offsets = package[:, 2], package[:, 3]
    offsets = package[:, -1]
    offsets[0] = 0
    for i in range(len(cls) -1):
        pos = np.zeros(point_offsets[i + 1] - point_offsets[i] + 2, dtype=np.int32)
        pos[1:-1] = where_is(points[point_offsets[i]:point_offsets[i + 1]], grid_m[cls[i]])
        pos[0] = pos[-1] = -1
        break_points = np.flatnonzero(np.diff(pos)).astype(np.int32)
        offsets[i + 1] = offsets[i] + len(break_points)
        seg_ls.extend(break_points)
    return np.array(seg_ls)


@njit # large array, cannot cache
def build_index(trajs, segs, life_stride_m):
    package, points = trajs
    for i in range(len(package)-1):
        begin, tid, cls, tp_start, so_start = package[i]
        p_next, so_next = package[i + 1, -2:]
        seg = segs[so_start:so_next]
        t_life = p_next - tp_start
        l_s = life_stride_m[cls]
        packed_tl_point = t_life << 32
        for si in range(len(seg) - 1):
            start, end = seg[si:si + 2]
            ts_beg = begin + start
            seg_life_pos = (end - start) // l_s * l_s
            cppyy.gbl.rest_add(cls, packed_tl_point | pack_i16(points[tp_start + start]), pack_i32(seg_life_pos, i),
                               ts_beg)


# jit_module(nopython=True, cache=True)

def load_segments(trajs, dataset_name, border_m, meta_path, space):
    seg_meta_file = os.path.join(meta_path, f'{dataset_name}{space}.np')
    if os.path.exists(seg_meta_file):
        concatenate_stride(border_m, space) # add border_stride to border_m
        seg_meta = np.load(seg_meta_file)
        return seg_meta
    else:
        bs_m = Dict.empty(key_type=int32, value_type=int32[:])
        for c, bs in concatenate_stride(border_m, space).items():
            bs_m[c] = np.array(bs, dtype=np.int32)
        segs_meta = partition_traj(trajs, bs_m)
        np.save(seg_meta_file, segs_meta)
        return segs_meta


def cppyy_init(cpy):
    cpy.add_include_path(os.path.join(LIB_DIR, 'include'))
    cpy.add_library_path(os.path.join(LIB_DIR, 'lib'))
    cpy.load_library(LIB_NAME)
    cpy.include(HEADER)
    # example usage:
    # index.Build([3, 29, 4, 4, 28, 3], [0, 5, 7, 11, 14, 17, 25, 29, 56])
    # for i in range(4, 56):
    #     index.Add(make_pair(make_pair(i//2, i//2), make_pair(i, make_pair(i,i))), (i, random.randint(0, 128)))
    # u_iter, s_iter = index.Query([16, 29, 11, 28], 10, make_pair(35, 51), 1)


def save_index_meta(trajs_raw, field_name, file_path):
    meta = dict(zip(field_name, trajs_raw))
    np.savez_compressed(file_path, **meta)


def load_index_meta(border_m, fname, dataset_name, cfg):
    file_path = os.path.join(cfg.INDEX.META_PATH, f'{dataset_name}_namba_index_meta.npz')
    fields = ['b_i_c_pos_sos', 'points']
    if os.path.exists(file_path):
        trajs_dict = np.load(file_path)
        trajs = tuple(trajs_dict[f] for f in fields)
    else:
        df, cols = load_yolo_for(fname, False)
        trajs_raw = traj_df2np(df, cols, cls=list(border_m))
        save_index_meta(trajs_raw, fields, file_path)
        trajs = trajs_raw

    return trajs, load_segments(trajs, dataset_name, border_m, cfg.INDEX.META_PATH, cfg.INDEX.REGION.GRID.SPACE)


def flatten_border(cls, borders, border):
    borders.append(cls)
    bs = border['border_stride']
    borders.append(len(bs))
    borders.extend(bs)
    lb = border['life_border']
    borders.append(len(lb))
    borders.extend(lb)


def build_rest(trajs, segs, border_m):
    cppyy_init(cppyy)
    borders = []
    life_stride_m = Dict.empty(key_type=int64, value_type=int64)
    for c, border in border_m.items():
        flatten_border(c, borders, border)
        life_stride_m[c] = border['tempo_stride']
    cppyy.gbl.border_init(borders)
    build_index(trajs, segs, life_stride_m)
    return cppyy.gbl.rest_query


if __name__ == '__main__':
    cppyy_init(cppyy)
    border_m = {}
    reg = [0, 400, 0, 500]
    life_border = [3, 8, 10, 21]
    for c in (0, 1):
        border_m[c] = {'reg_border': reg, 'tempo_stride': 60, 'life_border': life_border}
    space = 4, 5
    concatenate_stride(border_m, space)
    borders = []
    for c, border in border_m.items():
        flatten_border(c, borders, border)
    cppyy.gbl.border_init(borders)
    reg_0 = list(cppyy.gbl.get_border(0, 0))
    assert reg_0 == border_m[0]['border_stride']

    data, cols = load_fake(False)
    trajs = traj_df2np(data, cols)
    segs = load_segments(trajs, 'test', border_m, '../../tests', space)
    life_stride = Dict.empty(key_type=int64, value_type=int64)
    life_stride[0] = 60
    life_stride[1] = 60
    build_index(trajs, segs, life_stride)
    cppyy.gbl.rest_query(0, [0, 400, 0, 500], 0, (0, 100), 1)
    while cppyy.gbl.iter_has_next(1):
        print(unpack_i32(cppyy.gbl.iter_next(1)))
