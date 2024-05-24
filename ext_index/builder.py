import os.path
from itertools import pairwise
import pickle
import numpy as np

from ext_index.grid import Grid
from traj_seg import TrajectorySequenceSeg
from index_w import PyRestIndex

from utilities.data_preprocessing import traj_data
from utilities.dataset import load_yolo_for
from utilities.key import generate_key


def load_traj_seg(border_stride, fname, dataset_name, cfg):
    filename = f'{dataset_name}_{border_stride[0]}'.replace('(', '').replace(')', '').replace(' ', '').replace(',', '_')
    filename = generate_key(filename)
    file_path = os.path.join(cfg.INDEX.CONFIG_PATH, f'{dataset_name}_traj.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            trajs = pickle.load(f)
    else:
        df, cols, cls_m = load_yolo_for(fname)
        trajs_raw = traj_data(df, cols, cls_m, cfg.DATA.STRIDE, scale=cfg.DATA.SCALE, cls=list(border_stride))
        trajs = save_traj(trajs_raw, file_path)
    file_path = os.path.join(cfg.INDEX.CONFIG_PATH, f'{filename}.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            segs = pickle.load(f)
    else:
        segs = save_seg(trajs, border_stride, file_path)
    return trajs, segs


def build_rest(trajs, segs, border_m):
    idx_dic = {}
    stride = border_m[0]['tempo_stride']

    for i, (traj, seg) in enumerate(zip(trajs, segs)):
        beg = traj.begin
        label = traj.label
        if label not in idx_dic:
            border = border_m[label]
            idx_dic[label] = PyRestIndex(border['border_stride'], border['life_border'])
        spt_idx = idx_dic[label]
        life_stride = border_m[label]['tempo_stride']
        traj_life = traj.points.shape[0]
        for start, end in pairwise(seg):
            ts_beg = start + beg
            seg_life_pos = ((end - start) // life_stride + 1) * life_stride - 1
            spt_idx.add((traj.points[start], (traj_life, (seg_life_pos, ts_beg))), [i, ts_beg, end])
    return idx_dic


def save_seg(trajs, border_stride, save_path):
    seg_ls = []
    grid_m = {}
    for c, bs in border_stride.items():
        grid_m[c] = Grid(bs)
    for t in trajs:
        pos = grid_m[t.label].index(t.points)
        break_points = np.flatnonzero(np.diff(pos, prepend=-1, append=-1))
        seg_ls.append(break_points)
    with open(save_path, 'wb') as f:
        pickle.dump(seg_ls, f)
    return seg_ls


def save_traj(trajs_raw, save_path):
    traj_ls = [TrajectorySequenceSeg(tid, beg, cls_id, track) for tid, beg, cls_id, track in trajs_raw]
    with open(save_path, 'wb') as f:
        pickle.dump(traj_ls, f)
    return traj_ls


