import os.path
from itertools import pairwise
import pickle
import numpy as np

from ext_index.grid import Grid
from traj_seg import TrajectorySequenceSeg
from index_w import PyRestIndex

from utilities.data_preprocessing import traj_data
from utilities.dataset import load_yolo_for


def load_traj_seg(border_stride, dataset_name, cls_list, cfg):
    filename = f'{dataset_name}_{border_stride}'.replace('(', '').replace(')', '').replace(' ', '').replace(',', '_')
    file_path = os.path.join(cfg.INDEX.CONFIG_PATH, filename)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            trajs = pickle.load(f)
    else:
        df, cols, cls_m = load_yolo_for(os.path.join(cfg.DATA.PATH, dataset_name))
        trajs = traj_data(df, cols, cls_m, cls_list)
        reg = Grid(border_stride)
        trajs = save_traj_seg(df, reg, file_path)
    return trajs


def build_rest(trajs, segs, border_m):
    idx_dic = {}
    for i, traj, seg in enumerate(zip(trajs, segs)):
        beg = traj.begin
        label = traj.label
        if label not in idx_dic:
            border = border_m[label]
            idx_dic[label] = PyRestIndex(border['reg_broder'], border['life_border'])
        spt_idx = idx_dic[label]
        life_stride = border_m[label]['tempo_stride']
        traj_life = traj.points.shape[0]
        for start, end in pairwise(seg):
            ts_beg = start + beg
            seg_life_pos = ((end - beg) // life_stride + 1) * life_stride - 1
            spt_idx.add((seg.points[start], (traj_life, (seg_life_pos, ts_beg))), [i, beg, start, end])
    return idx_dic


def save_traj_seg(trajs, grid, save_path):
    traj_l = []
    seg_ls = []
    for tid, beg, cls_id, track in trajs:
        pos = grid.index(track)
        break_points = np.flatnonzero(np.diff(pos, prepend=-1, append=-1))
        traj_l.append(TrajectorySequenceSeg(tid, beg, cls_id, track))
        seg_ls.append(break_points)
    with open(save_path, 'wb') as f:
        pickle.dump((traj_l, seg_ls), f)
    return traj_l
